from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .screwdriver_turning_env_cfg import AllegroScrewdriverTurningEnvCfg


FINGER_JOINT_NAMES: dict[str, tuple[str, str, str, str]] = {
    "index": (
        "allegro_hand_hitosashi_finger_finger_joint_0",
        "allegro_hand_hitosashi_finger_finger_joint_1",
        "allegro_hand_hitosashi_finger_finger_joint_2",
        "allegro_hand_hitosashi_finger_finger_joint_3",
    ),
    "middle": (
        "allegro_hand_naka_finger_finger_joint_4",
        "allegro_hand_naka_finger_finger_joint_5",
        "allegro_hand_naka_finger_finger_joint_6",
        "allegro_hand_naka_finger_finger_joint_7",
    ),
    "ring": (
        "allegro_hand_kusuri_finger_finger_joint_8",
        "allegro_hand_kusuri_finger_finger_joint_9",
        "allegro_hand_kusuri_finger_finger_joint_10",
        "allegro_hand_kusuri_finger_finger_joint_11",
    ),
    "thumb": (
        "allegro_hand_oya_finger_joint_12",
        "allegro_hand_oya_finger_joint_13",
        "allegro_hand_oya_finger_joint_14",
        "allegro_hand_oya_finger_joint_15",
    ),
}

SCREWDRIVER_EULER_JOINT_NAMES = (
    "table_screwdriver_joint_1",
    "table_screwdriver_joint_2",
    "table_screwdriver_joint_3",
)
SCREWDRIVER_CAP_JOINT_NAME = "screwdriver_body_cap_joint"


class AllegroScrewdriverTurningEnv(DirectRLEnv):
    """Isaac Lab DirectRLEnv port of the legacy Allegro screwdriver turning task."""

    cfg: AllegroScrewdriverTurningEnvCfg

    def __init__(self, cfg: AllegroScrewdriverTurningEnvCfg, render_mode: str | None = None, **kwargs: Any):
        super().__init__(cfg, render_mode, **kwargs)

        self.fingers = tuple(self.cfg.fingers)
        self.num_fingers = len(self.fingers)
        self.obj_dof = 3

        self._finger_joint_ids_by_name = self._resolve_finger_joints()
        self._finger_joint_ids = [
            joint_id for finger in self.fingers for joint_id in self._finger_joint_ids_by_name[finger]
        ]
        self.num_finger_dofs = len(self._finger_joint_ids)
        self._all_finger_joint_ids = [
            joint_id for joint_ids in self._finger_joint_ids_by_name.values() for joint_id in joint_ids
        ]

        self._screwdriver_euler_joint_ids = self._find_ordered_joints(
            self.screwdriver, SCREWDRIVER_EULER_JOINT_NAMES
        )
        self._screwdriver_cap_joint_id = self._find_ordered_joints(
            self.screwdriver, (SCREWDRIVER_CAP_JOINT_NAME,)
        )[0]
        self._screwdriver_z_joint_id = self._screwdriver_euler_joint_ids[2]

        self._default_finger_pos = self._make_default_finger_pos(self.fingers)
        self._all_pregrasp_pos_by_finger = {
            finger: torch.tensor(self.cfg.pregrasp_positions[finger], dtype=torch.float32, device=self.device)
            for finger in self._finger_joint_ids_by_name
        }
        finger_limits = self.allegro.data.soft_joint_pos_limits[:, self._finger_joint_ids]
        margin = max(0.0, float(getattr(self.cfg, "joint_target_margin", 0.0)))
        self._finger_joint_lower = finger_limits[..., 0] + margin
        self._finger_joint_upper = finger_limits[..., 1] - margin
        invalid_margin = self._finger_joint_lower > self._finger_joint_upper
        if torch.any(invalid_margin):
            hard_limits = self.allegro.data.soft_joint_pos_limits[:, self._finger_joint_ids]
            limit_mid = 0.5 * (hard_limits[..., 0] + hard_limits[..., 1])
            self._finger_joint_lower = torch.where(invalid_margin, limit_mid, self._finger_joint_lower)
            self._finger_joint_upper = torch.where(invalid_margin, limit_mid, self._finger_joint_upper)
        self._goal_euler = torch.tensor(self.cfg.goal_euler_xyz, dtype=torch.float32, device=self.device).unsqueeze(0)

        self._target_actions = self._default_finger_pos.clone()
        self._start_joint_pos = self._default_finger_pos.clone()
        self._apply_step_count = 0
        self._ramp_steps = max(1.0, 0.75 * float(self.cfg.decimation))

        self._validate_spaces()

        # ---- RMA: proprioceptive history buffer ----
        # Stores the last N frames of [joint_positions, joint_targets]
        # Shape: (num_envs, prop_hist_len, history_obs_dim)
        self._prop_hist_len = self.cfg.prop_hist_len
        self._history_obs_dim = self.cfg.history_obs_dim
        self._proprio_hist_buf = torch.zeros(
            (self.num_envs, self._prop_hist_len, self._history_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        # Current joint targets (tracked for history buffer)
        self._cur_targets = self._default_finger_pos.clone()
        # Flag: whether to provide privileged observations
        self._asymmetric_obs = self.cfg.asymmetric_obs

    def _setup_scene(self):
        self.allegro = Articulation(self.cfg.robot_cfg)
        self.screwdriver = Articulation(self.cfg.screwdriver_cfg)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=self.cfg.friction_coefficient,
                    dynamic_friction=self.cfg.friction_coefficient,
                )
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["allegro"] = self.allegro
        self.scene.articulations["screwdriver"] = self.screwdriver

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        action_clip = float(getattr(self.cfg, "action_clip", 0.0))
        if action_clip > 0.0:
            actions = torch.clamp(actions, -action_clip, action_clip)
        self.actions = actions.clone()
        target_actions = actions.clone()
        if self.cfg.action_offset:
            target_actions = target_actions + self._default_finger_pos
        target_actions = self._clamp_finger_targets(target_actions)

        self._target_actions = target_actions
        self._cur_targets = target_actions.clone()

        # Update proprioceptive history buffer for RMA
        self._update_proprio_hist()

        if self.cfg.gradual_control:
            self._start_joint_pos = self.allegro.data.joint_pos[:, self._finger_joint_ids].clone()
            self._apply_step_count = 0

    def _apply_action(self) -> None:
        if self.cfg.gradual_control:
            if self._apply_step_count < self._ramp_steps:
                alpha = float(self._apply_step_count + 1) / self._ramp_steps
                target = alpha * (self._target_actions - self._start_joint_pos) + self._start_joint_pos
            else:
                target = self._target_actions
            self._apply_step_count += 1
        else:
            target = self._target_actions
        target = self._clamp_finger_targets(target)

        self.allegro.set_joint_position_target(target, joint_ids=self._finger_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        finger_q = self.allegro.data.joint_pos[:, self._finger_joint_ids]
        screwdriver_euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_joint_ids]
        obs = torch.cat((finger_q, screwdriver_euler), dim=-1)

        result = {"policy": obs}

        if self._asymmetric_obs:
            result["critic"] = self._compute_privileged_info()
            result["proprio_hist"] = self._proprio_hist_buf.clone()

        return result

    def _get_rewards(self) -> torch.Tensor:
        obj_orientation = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_joint_ids]
        action_cost = self.cfg.reward_action_weight * torch.sum(self.actions**2, dim=-1)
        goal_cost = self.cfg.reward_goal_weight * torch.sum((obj_orientation - self._goal_euler) ** 2, dim=-1)
        upright_cost = self.cfg.reward_upright_weight * torch.sum(obj_orientation[:, :-1] ** 2, dim=-1)
        cost = action_cost + goal_cost + upright_cost
        goal_error = obj_orientation - self._goal_euler
        self.extras["eval_screwdriver_euler"] = obj_orientation.detach().clone()
        self.extras["eval_screwdriver_goal_error"] = goal_error.detach().clone()
        self.extras["eval_screwdriver_upright_norm"] = torch.linalg.norm(obj_orientation[:, :-1], dim=-1).detach()
        self.extras["eval_action_cost"] = action_cost.detach()
        self.extras["eval_goal_cost"] = goal_cost.detach()
        self.extras["eval_upright_cost"] = upright_cost.detach()
        return -torch.nan_to_num(cost, nan=1.0e6)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.allegro._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        robot_root_state = self.allegro.data.default_root_state[env_ids].clone()
        robot_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.allegro.write_root_pose_to_sim(robot_root_state[:, :7], env_ids=env_ids)
        self.allegro.write_root_velocity_to_sim(robot_root_state[:, 7:], env_ids=env_ids)

        robot_joint_pos = self.allegro.data.default_joint_pos[env_ids].clone()
        robot_joint_vel = torch.zeros_like(self.allegro.data.default_joint_vel[env_ids])
        for finger, joint_ids in self._finger_joint_ids_by_name.items():
            robot_joint_pos[:, joint_ids] = self._all_pregrasp_pos_by_finger[finger]
        self.allegro.set_joint_position_target(robot_joint_pos, env_ids=env_ids)
        self.allegro.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=env_ids)

        screwdriver_root_state = self.screwdriver.data.default_root_state[env_ids].clone()
        screwdriver_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.screwdriver.write_root_pose_to_sim(screwdriver_root_state[:, :7], env_ids=env_ids)
        self.screwdriver.write_root_velocity_to_sim(screwdriver_root_state[:, 7:], env_ids=env_ids)

        screwdriver_joint_pos = torch.zeros_like(self.screwdriver.data.default_joint_pos[env_ids])
        screwdriver_joint_vel = torch.zeros_like(self.screwdriver.data.default_joint_vel[env_ids])
        if self.cfg.randomize_obj_start:
            screwdriver_joint_pos[:, self._screwdriver_z_joint_id] = 2.0 * math.pi * (
                torch.rand(len(env_ids), device=self.device) - 0.5
            )
        self.screwdriver.write_joint_state_to_sim(screwdriver_joint_pos, screwdriver_joint_vel, env_ids=env_ids)

        self._target_actions[env_ids] = robot_joint_pos[:, self._finger_joint_ids]
        self._start_joint_pos[env_ids] = robot_joint_pos[:, self._finger_joint_ids]
        self._cur_targets[env_ids] = robot_joint_pos[:, self._finger_joint_ids]

        # Initialize proprioceptive history buffer for reset envs
        if self._asymmetric_obs:
            finger_q = robot_joint_pos[:, self._finger_joint_ids]
            init_frame = self._make_proprio_frame(finger_q, self._cur_targets[env_ids])
            # Fill entire history with the initial state
            self._proprio_hist_buf[env_ids] = init_frame.unsqueeze(1).repeat(
                1, self._prop_hist_len, 1
            )

        self._settle_contacts()

    def _resolve_finger_joints(self) -> dict[str, list[int]]:
        unknown_fingers = set(self.fingers).difference(FINGER_JOINT_NAMES)
        if unknown_fingers:
            raise ValueError(f"Unknown Allegro finger names: {sorted(unknown_fingers)}")

        joint_ids_by_name = {}
        for finger, joint_names in FINGER_JOINT_NAMES.items():
            joint_ids_by_name[finger] = self._find_ordered_joints(self.allegro, joint_names)
        return joint_ids_by_name

    def _clamp_finger_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if not getattr(self.cfg, "clamp_joint_targets", False):
            return targets
        return torch.clamp(targets, self._finger_joint_lower, self._finger_joint_upper)

    def _find_ordered_joints(self, articulation: Articulation, joint_names: Sequence[str]) -> list[int]:
        patterns = [f"^{re.escape(joint_name)}$" for joint_name in joint_names]
        joint_ids, found_names = articulation.find_joints(patterns, preserve_order=True)
        if len(joint_ids) != len(joint_names):
            raise RuntimeError(
                f"Could not resolve joints {tuple(joint_names)} on {articulation.cfg.prim_path}. "
                f"Found {tuple(found_names)}."
            )
        return joint_ids

    def _make_default_finger_pos(self, fingers: Sequence[str]) -> torch.Tensor:
        default_pos = [value for finger in fingers for value in self.cfg.pregrasp_positions[finger]]
        return torch.tensor(default_pos, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

    def _validate_spaces(self) -> None:
        expected_obs_dim = self.num_finger_dofs + self.obj_dof
        action_shape = getattr(self.single_action_space, "shape", None)
        obs_space = self.single_observation_space
        if hasattr(obs_space, "spaces"):  # gym.spaces.Dict
            obs_shape = obs_space["policy"].shape
        else:
            obs_shape = obs_space.shape
        if action_shape != (self.num_finger_dofs,):
            raise ValueError(
                f"action_space shape {action_shape} does not match configured fingers {self.fingers}; "
                f"expected {(self.num_finger_dofs,)}."
            )
        if obs_shape != (expected_obs_dim,):
            raise ValueError(
                f"observation_space shape {obs_shape} does not match configured fingers {self.fingers}; "
                f"expected {(expected_obs_dim,)}."
            )

    def _compute_privileged_info(self) -> torch.Tensor:
        """Compute privileged (asymmetric) observations for teacher-student training.

        Privileged info includes object state that would not be available
        on a real robot, such as ground-truth pose, velocity, and dynamics
        parameters. This is used by the teacher policy (Stage 1) and as
        ground truth for the adaptation module (Stage 2).

        Returns:
            Tensor of shape (num_envs, privileged_obs_dim) with privileged info.
        """
        # Screwdriver euler angles (3)
        screwdriver_euler = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_joint_ids]

        # Screwdriver angular velocity (3)
        screwdriver_angvel = self.screwdriver.data.joint_vel[:, self._screwdriver_euler_joint_ids]

        # Screwdriver root position relative to hand (3)
        screwdriver_pos = self.screwdriver.data.root_pos_w - self.allegro.data.root_pos_w

        # Screwdriver root orientation as quaternion (wxyz) (4)
        screwdriver_quat = self.screwdriver.data.root_quat_w

        # Friction coefficient (scalar, expanded to 1 dim)
        friction = torch.full(
            (self.num_envs, 1),
            self.cfg.friction_coefficient,
            dtype=torch.float32,
            device=self.device,
        )

        priv_info = torch.cat(
            [
                screwdriver_euler,   # 3
                screwdriver_angvel,  # 3
                screwdriver_pos,     # 3
                screwdriver_quat,    # 4
                friction,            # 1
            ],
            dim=-1,
        )
        return priv_info

    def _update_proprio_hist(self) -> None:
        """Update the proprioceptive history buffer with the latest frame.

        Each frame contains: [finger_joint_positions, current_joint_targets]
        This sliding window provides the temporal context the adaptation
        module uses to infer environment properties.
        """
        finger_q = self.allegro.data.joint_pos[:, self._finger_joint_ids]
        frame = self._make_proprio_frame(finger_q, self._cur_targets)
        # Roll buffer and insert newest frame at the end
        self._proprio_hist_buf = torch.roll(self._proprio_hist_buf, shifts=-1, dims=1)
        self._proprio_hist_buf[:, -1, :] = frame

    def _make_proprio_frame(self, finger_q: torch.Tensor, joint_targets: torch.Tensor) -> torch.Tensor:
        """Build one proprioceptive history frame and pad/truncate to cfg size."""
        frame = torch.cat([finger_q, joint_targets], dim=-1)
        frame_dim = frame.shape[-1]
        if frame_dim == self._history_obs_dim:
            return frame
        if frame_dim > self._history_obs_dim:
            return frame[:, : self._history_obs_dim]
        pad = torch.zeros(
            (frame.shape[0], self._history_obs_dim - frame_dim),
            dtype=frame.dtype,
            device=frame.device,
        )
        return torch.cat([frame, pad], dim=-1)

    def _settle_contacts(self) -> None:
        if self.cfg.reset_contact_steps <= 0:
            return
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        for _ in range(self.cfg.reset_contact_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
