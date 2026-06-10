"""Continuous screwdriver turning reward utilities and Allegro task."""

import math
import re
from collections.abc import Sequence
from typing import Any

import torch

from MFR_benchmark.isaac_lab_tasks.screwdriver_turning.screwdriver_turning_env import (
    AllegroScrewdriverTurningEnv,
)

from .screwdriver_continuous_turning_env_cfg import AllegroScrewdriverContinuousTurningEnvCfg


ALLEGRO_FINGERTIP_BODY_NAMES: dict[str, str] = {
    "index": "hitosashi_ee",
    "middle": "naka_ee",
    "ring": "kusuri_ee",
    "thumb": "oya_ee",
}
SCREWDRIVER_NEAR_BODY_NAMES = ("screwdriver_stick", "screwdriver_body", "screwdriver_cap")


class ContinuousTurningRewardMixin:
    """Reusable signed-progress reward for mounted screwdriver turning tasks."""

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs: Any):
        self._prev_z = None
        self._prev_tilt_xy = None
        self._total_turn = None
        self._net_turn = None
        self._prev_actions = None
        self._prev_milestone_count = None
        self._policy_dt = float(cfg.decimation) * float(cfg.sim.dt)
        super().__init__(cfg, render_mode, **kwargs)

        self._policy_dt = float(self.cfg.decimation) * float(self.cfg.sim.dt)
        self._prev_z = self.screwdriver.data.joint_pos[:, self._screwdriver_z_joint_id].detach().clone()
        self._prev_tilt_xy = self.screwdriver.data.joint_pos[
            :, self._screwdriver_euler_joint_ids[:2]
        ].detach().clone()
        self._total_turn = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._net_turn = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_actions = torch.zeros(
            (self.num_envs, self.num_finger_dofs), dtype=torch.float32, device=self.device
        )
        self._prev_milestone_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        obj_orientation = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_joint_ids]
        z_curr = obj_orientation[:, 2]

        if self._prev_z is None:
            self._prev_z = z_curr.detach().clone()
        if self._prev_tilt_xy is None:
            self._prev_tilt_xy = obj_orientation[:, :2].detach().clone()
        if self._total_turn is None:
            self._total_turn = torch.zeros_like(z_curr)
        if self._prev_actions is None:
            self._prev_actions = torch.zeros_like(self.actions)
        if self._net_turn is None:
            self._net_turn = torch.zeros_like(z_curr)
        if self._prev_milestone_count is None:
            self._prev_milestone_count = torch.zeros_like(z_curr)

        raw_delta_z = self.cfg.turn_direction * (z_curr - self._prev_z)
        # Robust to revolute-coordinate wrapping while still working for unbounded z.
        delta_z = torch.atan2(torch.sin(raw_delta_z), torch.cos(raw_delta_z))
        self._prev_z = z_curr.detach().clone()

        turn_velocity = delta_z / self._policy_dt
        forward_velocity = torch.clamp(turn_velocity, min=0.0, max=self.cfg.turn_velocity_clip)
        reverse_velocity = torch.clamp(-turn_velocity, min=0.0, max=self.cfg.turn_velocity_clip)

        raw_turn_reward = self.cfg.reward_turn_weight * forward_velocity
        turn_gate = self._compute_turn_reward_gate()
        turn_reward = raw_turn_reward * turn_gate
        # Gate the reverse penalty with the same contact/motion gate as the turn
        # reward. Otherwise passive off-contact rebound (and the back-off needed
        # to regrasp) is punished while forward progress is gated off, which makes
        # "don't move the screwdriver" the safest policy and blocks finger gaiting.
        reverse_cost = self.cfg.reward_reverse_weight * reverse_velocity * turn_gate

        forward_delta = torch.clamp(delta_z, min=0.0)
        self._total_turn += forward_delta.detach()
        self._net_turn += delta_z.detach()
        milestone_reward = self._compute_milestone_reward(gate=self._compute_milestone_reward_gate())

        tilt_xy = obj_orientation[:, :2]
        tilt_velocity = (tilt_xy - self._prev_tilt_xy) / self._policy_dt
        self._prev_tilt_xy = tilt_xy.detach().clone()

        upright_cost = self.cfg.reward_upright_weight * torch.sum(obj_orientation[:, :-1] ** 2, dim=-1)
        tilt_velocity_cost = getattr(self.cfg, "reward_tilt_velocity_weight", 0.0) * torch.linalg.vector_norm(
            tilt_velocity, ord=1, dim=-1
        )
        action_cost = self.cfg.reward_action_weight * self._action_regularization(self.actions)
        action_rate_cost = self.cfg.reward_action_rate_weight * torch.mean(
            (self.actions - self._prev_actions) ** 2, dim=-1
        )
        self._prev_actions = self.actions.detach().clone()

        finger_q = self.allegro.data.joint_pos[:, self._finger_joint_ids]
        finger_vel = self.allegro.data.joint_vel[:, self._finger_joint_ids]
        finger_pose_cost = getattr(self.cfg, "reward_finger_pose_weight", 0.0) * torch.sum(
            (finger_q - self._default_finger_pos) ** 2, dim=-1
        )
        finger_velocity_cost = getattr(self.cfg, "reward_finger_velocity_weight", 0.0) * torch.mean(
            finger_vel**2, dim=-1
        )

        aux_reward, aux_cost, aux_extras = self._compute_continuous_auxiliary_terms()
        reward = (
            turn_reward
            + milestone_reward
            + aux_reward
            - reverse_cost
            - upright_cost
            - tilt_velocity_cost
            - action_cost
            - action_rate_cost
            - finger_pose_cost
            - finger_velocity_cost
            - aux_cost
        )

        legacy_goal_error = obj_orientation - self._goal_euler
        self.extras["eval_screwdriver_euler"] = obj_orientation.detach().clone()
        self.extras["eval_screwdriver_goal_error"] = legacy_goal_error.detach().clone()
        self.extras["eval_screwdriver_upright_norm"] = torch.linalg.norm(obj_orientation[:, :-1], dim=-1).detach()
        self.extras["eval_screwdriver_tilt_velocity"] = torch.linalg.vector_norm(
            tilt_velocity, ord=1, dim=-1
        ).detach()
        self.extras["eval_turn_delta"] = delta_z.detach()
        self.extras["eval_raw_turn_delta"] = raw_delta_z.detach()
        self.extras["eval_turn_velocity"] = turn_velocity.detach()
        self.extras["eval_forward_turn_velocity"] = forward_velocity.detach()
        self.extras["eval_reverse_turn_velocity"] = reverse_velocity.detach()
        self.extras["eval_total_turn_rad"] = self._total_turn.detach().clone()
        self.extras["eval_total_turns"] = (self._total_turn / (2.0 * math.pi)).detach().clone()
        self.extras["eval_net_turn_rad"] = self._net_turn.detach().clone()
        self.extras["eval_net_turns"] = (self._net_turn / (2.0 * math.pi)).detach().clone()
        self.extras["eval_turn_gate"] = turn_gate.detach()
        self.extras["eval_raw_turn_reward"] = raw_turn_reward.detach()
        self.extras["eval_turn_reward"] = turn_reward.detach()
        self.extras["eval_milestone_reward"] = milestone_reward.detach()
        self.extras["eval_action_cost"] = action_cost.detach()
        self.extras["eval_action_rate_cost"] = action_rate_cost.detach()
        self.extras["eval_reverse_cost"] = reverse_cost.detach()
        self.extras["eval_upright_cost"] = upright_cost.detach()
        self.extras["eval_tilt_velocity_cost"] = tilt_velocity_cost.detach()
        self.extras["eval_finger_pose_cost"] = finger_pose_cost.detach()
        self.extras["eval_finger_velocity_cost"] = finger_velocity_cost.detach()
        self.extras["eval_aux_reward"] = aux_reward.detach()
        self.extras["eval_aux_cost"] = aux_cost.detach()
        self.extras["eval_goal_cost"] = torch.zeros_like(action_cost).detach()
        for key, value in aux_extras.items():
            self.extras[key] = value.detach() if isinstance(value, torch.Tensor) else value

        return torch.nan_to_num(reward, nan=-1.0e6)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.upright_termination_threshold > 0.0:
            obj_orientation = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_joint_ids]
            upright_norm = torch.linalg.norm(obj_orientation[:, :-1], dim=-1)
            terminated = upright_norm > self.cfg.upright_termination_threshold
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["eval_upright_terminated"] = terminated.detach()
        return terminated, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.allegro._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        if self._prev_z is not None:
            self._prev_z[env_ids] = self.screwdriver.data.joint_pos[
                env_ids, self._screwdriver_z_joint_id
            ].detach().clone()
        if self._prev_tilt_xy is not None:
            self._prev_tilt_xy[env_ids] = self.screwdriver.data.joint_pos[env_ids][
                :, self._screwdriver_euler_joint_ids[:2]
            ].detach().clone()
        if self._total_turn is not None:
            self._total_turn[env_ids] = 0.0
        if self._prev_actions is not None:
            self._prev_actions[env_ids] = 0.0
        if self._net_turn is not None:
            self._net_turn[env_ids] = 0.0
        if self._prev_milestone_count is not None:
            self._prev_milestone_count[env_ids] = 0.0

    def _compute_milestone_reward(self, gate: torch.Tensor | None = None) -> torch.Tensor:
        if self.cfg.milestone_angle <= 0.0 or self.cfg.milestone_bonus <= 0.0:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        positive_net_turn = torch.clamp(self._net_turn, min=0.0)
        milestone_count = torch.floor(positive_net_turn / self.cfg.milestone_angle)
        new_milestones = torch.clamp(milestone_count - self._prev_milestone_count, min=0.0)
        self._prev_milestone_count = torch.maximum(self._prev_milestone_count, milestone_count.detach())
        milestone_reward = self.cfg.milestone_bonus * new_milestones
        if gate is not None:
            milestone_reward = milestone_reward * gate
        return milestone_reward

    def _action_regularization(self, actions: torch.Tensor) -> torch.Tensor:
        if getattr(self.cfg, "use_mean_action_penalty", False):
            return torch.mean(actions**2, dim=-1)
        return torch.sum(actions**2, dim=-1)

    def _compute_turn_reward_gate(self) -> torch.Tensor:
        return torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

    def _compute_milestone_reward_gate(self) -> torch.Tensor | None:
        return None

    def _compute_continuous_auxiliary_terms(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        return zeros, zeros, {}


class AllegroScrewdriverContinuousTurningEnv(ContinuousTurningRewardMixin, AllegroScrewdriverTurningEnv):
    """Continuous-turning task with signed progress reward.

    This task keeps the original MFR screwdriver setup and observations, but
    replaces the finite orientation goal reward with HORA-style incremental
    negative-z turning progress. It is intended for many-turn behavior rather
    than stopping at 90 or 360 degrees.
    """

    cfg: AllegroScrewdriverContinuousTurningEnvCfg

    def __init__(
        self, cfg: AllegroScrewdriverContinuousTurningEnvCfg, render_mode: str | None = None, **kwargs: Any
    ):
        self._fingertip_body_ids: list[int] = []
        self._screwdriver_near_body_ids: list[int] = []
        self._thumb_tip_index: int | None = None
        self._non_thumb_tip_indices: list[int] = []
        super().__init__(cfg, render_mode, **kwargs)
        self._fingertip_body_ids = self._resolve_fingertip_bodies()
        self._screwdriver_near_body_ids = self._resolve_screwdriver_near_bodies()
        self._thumb_tip_index = self.fingers.index("thumb") if "thumb" in self.fingers else None
        self._non_thumb_tip_indices = [idx for idx, finger in enumerate(self.fingers) if finger != "thumb"]

    def _resolve_fingertip_bodies(self) -> list[int]:
        body_ids = []
        for finger in self.fingers:
            if finger not in ALLEGRO_FINGERTIP_BODY_NAMES:
                raise ValueError(f"No fingertip body name configured for Allegro finger {finger!r}.")
            body_name = ALLEGRO_FINGERTIP_BODY_NAMES[finger]
            pattern = f"^{re.escape(body_name)}$"
            ids, found_names = self.allegro.find_bodies([pattern], preserve_order=True)
            if len(ids) != 1:
                raise RuntimeError(
                    f"Could not resolve fingertip body {body_name!r} on {self.allegro.cfg.prim_path}. "
                    f"Found {tuple(found_names)}."
                )
            body_ids.append(ids[0])
        return body_ids

    def _resolve_screwdriver_near_bodies(self) -> list[int]:
        body_ids = []
        for body_name in SCREWDRIVER_NEAR_BODY_NAMES:
            pattern = f"^{re.escape(body_name)}$"
            ids, found_names = self.screwdriver.find_bodies([pattern], preserve_order=True)
            if len(ids) != 1:
                raise RuntimeError(
                    f"Could not resolve screwdriver near-contact body {body_name!r} on "
                    f"{self.screwdriver.cfg.prim_path}. Found {tuple(found_names)}."
                )
            body_ids.append(ids[0])
        return body_ids

    def _compute_fingertip_screwdriver_distances(self) -> torch.Tensor:
        if not self._fingertip_body_ids or not self._screwdriver_near_body_ids:
            return torch.empty((self.num_envs, 0), dtype=torch.float32, device=self.device)
        fingertip_pos = self.allegro.data.body_state_w[:, self._fingertip_body_ids, :3]
        screwdriver_pos = self.screwdriver.data.body_state_w[:, self._screwdriver_near_body_ids, :3]
        tip_to_screwdriver = fingertip_pos.unsqueeze(2) - screwdriver_pos.unsqueeze(1)
        return torch.linalg.norm(tip_to_screwdriver, dim=-1).min(dim=-1).values

    def _compute_fingertip_speeds(self) -> torch.Tensor:
        if not self._fingertip_body_ids:
            return torch.empty((self.num_envs, 0), dtype=torch.float32, device=self.device)
        fingertip_vel = self.allegro.data.body_state_w[:, self._fingertip_body_ids, 7:10]
        return torch.linalg.norm(fingertip_vel, dim=-1)

    def _compute_contact_proxy(self, distance_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
        tip_dist = self._compute_fingertip_screwdriver_distances()
        if tip_dist.shape[1] == 0:
            empty_mask = torch.zeros_like(tip_dist, dtype=torch.bool)
            counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            return empty_mask, counts
        contact_mask = tip_dist <= distance_threshold
        contact_count = torch.sum(contact_mask, dim=-1)
        return contact_mask, contact_count

    def _compute_turn_reward_gate(self) -> torch.Tensor:
        threshold = float(getattr(self.cfg, "turn_reward_contact_distance", 0.0))
        if threshold <= 0.0:
            return torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        contact_mask, contact_count = self._compute_contact_proxy(threshold)
        min_contacts = max(1, int(getattr(self.cfg, "turn_reward_min_contact_fingers", 1)))
        contact_gate = (contact_count >= min_contacts).to(dtype=torch.float32)

        fingertip_speeds = self._compute_fingertip_speeds()
        if fingertip_speeds.shape[1] == 0:
            motion_gate = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        else:
            contact_weights = contact_mask.to(dtype=torch.float32)
            active_count = torch.clamp(contact_weights.sum(dim=-1), min=1.0)
            contact_speed = torch.sum(fingertip_speeds * contact_weights, dim=-1) / active_count
            min_speed = float(getattr(self.cfg, "turn_reward_min_fingertip_speed", 0.0))
            full_speed = max(
                float(getattr(self.cfg, "turn_reward_full_fingertip_speed", 0.0)), min_speed + 1.0e-6
            )
            motion_gate = torch.clamp((contact_speed - min_speed) / (full_speed - min_speed), 0.0, 1.0)
            self.extras["eval_turn_contact_fingertip_speed"] = contact_speed.detach()

        gate = contact_gate * motion_gate
        self.extras["eval_turn_contact_count"] = contact_count.detach()
        self.extras["eval_turn_contact_gate"] = contact_gate.detach()
        self.extras["eval_turn_motion_gate"] = motion_gate.detach()
        return gate

    def _compute_milestone_reward_gate(self) -> torch.Tensor | None:
        return self._compute_turn_reward_gate()

    def _compute_continuous_auxiliary_terms(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        tip_dist = self._compute_fingertip_screwdriver_distances()
        if tip_dist.shape[1] == 0:
            return zeros, zeros, {
                "eval_near_reward": zeros,
                "eval_near_score": zeros,
                "eval_mean_fingertip_dist": zeros,
            }

        near = torch.exp(-tip_dist / max(float(self.cfg.near_reward_std), 1.0e-6))
        near_score = self._compute_near_score(near)
        near_reward = self.cfg.near_reward_weight * near_score
        return near_reward, zeros, {
            "eval_near_reward": near_reward,
            "eval_near_score": near_score,
            "eval_mean_fingertip_dist": torch.mean(tip_dist, dim=-1),
            "eval_min_fingertip_dist": torch.min(tip_dist, dim=-1).values,
        }

    def _compute_near_score(self, near: torch.Tensor) -> torch.Tensor:
        if near.shape[1] == 0:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        non_thumb_score = None
        if self._non_thumb_tip_indices:
            non_thumb_near = near[:, self._non_thumb_tip_indices]
            k = max(1, min(int(self.cfg.near_reward_top_k), non_thumb_near.shape[1]))
            non_thumb_score = torch.topk(non_thumb_near, k=k, dim=-1).values.mean(dim=-1)

        thumb_score = None
        if self._thumb_tip_index is not None:
            thumb_score = near[:, self._thumb_tip_index]

        if thumb_score is not None and non_thumb_score is not None:
            return 0.5 * (thumb_score + non_thumb_score)
        if thumb_score is not None:
            return thumb_score
        if non_thumb_score is not None:
            return non_thumb_score
        return torch.mean(near, dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, timed_out = super()._get_dones()
        threshold = float(getattr(self.cfg, "lost_contact_termination_distance", 0.0))
        if threshold <= 0.0:
            return terminated, timed_out

        contact_mask, contact_count = self._compute_contact_proxy(threshold)
        if contact_mask.shape[1] == 0:
            return terminated, timed_out

        min_fingers = max(1, int(getattr(self.cfg, "lost_contact_min_fingers", 1)))
        grace_steps = max(0, int(getattr(self.cfg, "lost_contact_grace_steps", 0)))
        lost_contact = (contact_count < min_fingers) & (self.episode_length_buf >= grace_steps)
        self.extras["eval_contact_count"] = contact_count.detach()
        self.extras["eval_lost_contact_terminated"] = lost_contact.detach()
        return terminated | lost_contact, timed_out
