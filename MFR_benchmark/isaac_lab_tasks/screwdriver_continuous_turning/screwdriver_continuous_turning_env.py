"""Continuous Allegro screwdriver turning environment."""

import math
from collections.abc import Sequence
from typing import Any

import torch

from MFR_benchmark.isaac_lab_tasks.screwdriver_turning.screwdriver_turning_env import (
    AllegroScrewdriverTurningEnv,
)

from .screwdriver_continuous_turning_env_cfg import AllegroScrewdriverContinuousTurningEnvCfg


class AllegroScrewdriverContinuousTurningEnv(AllegroScrewdriverTurningEnv):
    """Continuous-turning task with signed progress reward.

    This task keeps the original MFR screwdriver setup and observations, but
    replaces the finite orientation goal reward with HORA-style incremental
    negative-z turning progress. It is intended for many-turn behavior rather
    than stopping at 90 or 360 degrees.
    """

    cfg: AllegroScrewdriverContinuousTurningEnvCfg

    def __init__(self, cfg: AllegroScrewdriverContinuousTurningEnvCfg, render_mode: str | None = None, **kwargs: Any):
        self._prev_z = None
        self._total_turn = None
        self._net_turn = None
        self._prev_actions = None
        self._prev_milestone_count = None
        self._policy_dt = float(cfg.decimation) * float(cfg.sim.dt)
        super().__init__(cfg, render_mode, **kwargs)

        self._policy_dt = float(self.cfg.decimation) * float(self.cfg.sim.dt)
        self._prev_z = self.screwdriver.data.joint_pos[:, self._screwdriver_z_joint_id].detach().clone()
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

        turn_reward = self.cfg.reward_turn_weight * forward_velocity
        reverse_cost = self.cfg.reward_reverse_weight * reverse_velocity

        forward_delta = torch.clamp(delta_z, min=0.0)
        self._total_turn += forward_delta.detach()
        self._net_turn += delta_z.detach()
        milestone_reward = self._compute_milestone_reward()

        upright_cost = self.cfg.reward_upright_weight * torch.sum(obj_orientation[:, :-1] ** 2, dim=-1)
        action_cost = self.cfg.reward_action_weight * torch.sum(self.actions**2, dim=-1)
        action_rate_cost = self.cfg.reward_action_rate_weight * torch.mean(
            (self.actions - self._prev_actions) ** 2, dim=-1
        )
        self._prev_actions = self.actions.detach().clone()

        reward = turn_reward + milestone_reward - reverse_cost - upright_cost - action_cost - action_rate_cost

        legacy_goal_error = obj_orientation - self._goal_euler
        self.extras["eval_screwdriver_euler"] = obj_orientation.detach().clone()
        self.extras["eval_screwdriver_goal_error"] = legacy_goal_error.detach().clone()
        self.extras["eval_screwdriver_upright_norm"] = torch.linalg.norm(obj_orientation[:, :-1], dim=-1).detach()
        self.extras["eval_turn_delta"] = delta_z.detach()
        self.extras["eval_turn_velocity"] = turn_velocity.detach()
        self.extras["eval_forward_turn_velocity"] = forward_velocity.detach()
        self.extras["eval_reverse_turn_velocity"] = reverse_velocity.detach()
        self.extras["eval_total_turn_rad"] = self._total_turn.detach().clone()
        self.extras["eval_total_turns"] = (self._total_turn / (2.0 * math.pi)).detach().clone()
        self.extras["eval_net_turn_rad"] = self._net_turn.detach().clone()
        self.extras["eval_net_turns"] = (self._net_turn / (2.0 * math.pi)).detach().clone()
        self.extras["eval_turn_reward"] = turn_reward.detach()
        self.extras["eval_milestone_reward"] = milestone_reward.detach()
        self.extras["eval_action_cost"] = action_cost.detach()
        self.extras["eval_action_rate_cost"] = action_rate_cost.detach()
        self.extras["eval_reverse_cost"] = reverse_cost.detach()
        self.extras["eval_upright_cost"] = upright_cost.detach()
        self.extras["eval_goal_cost"] = torch.zeros_like(action_cost).detach()

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
            self._prev_z[env_ids] = self.screwdriver.data.joint_pos[env_ids, self._screwdriver_z_joint_id].detach().clone()
        if self._total_turn is not None:
            self._total_turn[env_ids] = 0.0
        if self._prev_actions is not None:
            self._prev_actions[env_ids] = 0.0
        if self._net_turn is not None:
            self._net_turn[env_ids] = 0.0
        if self._prev_milestone_count is not None:
            self._prev_milestone_count[env_ids] = 0.0

    def _compute_milestone_reward(self) -> torch.Tensor:
        if self.cfg.milestone_angle <= 0.0 or self.cfg.milestone_bonus <= 0.0:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        positive_net_turn = torch.clamp(self._net_turn, min=0.0)
        milestone_count = torch.floor(positive_net_turn / self.cfg.milestone_angle)
        new_milestones = torch.clamp(milestone_count - self._prev_milestone_count, min=0.0)
        self._prev_milestone_count = torch.maximum(self._prev_milestone_count, milestone_count.detach())
        return self.cfg.milestone_bonus * new_milestones
