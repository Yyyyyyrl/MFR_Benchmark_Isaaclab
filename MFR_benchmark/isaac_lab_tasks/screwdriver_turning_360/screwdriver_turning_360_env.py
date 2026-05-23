"""Allegro screwdriver 360-degree turning environment."""

from collections.abc import Sequence
from typing import Any

import torch

from MFR_benchmark.isaac_lab_tasks.screwdriver_turning.screwdriver_turning_env import (
    AllegroScrewdriverTurningEnv,
)

from .screwdriver_turning_360_env_cfg import AllegroScrewdriverTurning360EnvCfg


class AllegroScrewdriverTurning360Env(AllegroScrewdriverTurningEnv):
    """360-degree variant with directional progress reward.

    Adds a delta-progress term to the reward that gives immediate positive
    feedback for each incremental rotation toward the goal (-2*pi). This
    prevents the policy from committing to the wrong turning direction, which
    the pure squared-error reward cannot disambiguate from a stationary start.
    """

    cfg: AllegroScrewdriverTurning360EnvCfg

    def __init__(self, cfg: AllegroScrewdriverTurning360EnvCfg, render_mode: str | None = None, **kwargs: Any):
        self._prev_z = None
        super().__init__(cfg, render_mode, **kwargs)
        self._prev_z = self.screwdriver.data.joint_pos[:, self._screwdriver_z_joint_id].clone()

    def _get_rewards(self) -> torch.Tensor:
        obj_orientation = self.screwdriver.data.joint_pos[:, self._screwdriver_euler_joint_ids]
        z_curr = obj_orientation[:, 2]

        delta_z = self._prev_z - z_curr  # positive when turning toward -2*pi
        progress_reward = self.cfg.reward_delta_weight * torch.clamp(delta_z, min=0.0)
        self._prev_z = z_curr.clone()

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
        self.extras["eval_delta_progress"] = delta_z.detach()

        return progress_reward - torch.nan_to_num(cost, nan=1.0e6)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.allegro._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        if self._prev_z is not None:
            self._prev_z[env_ids] = self.screwdriver.data.joint_pos[env_ids, self._screwdriver_z_joint_id].clone()
