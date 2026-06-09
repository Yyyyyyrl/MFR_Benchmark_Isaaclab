"""Continuous screwdriver turning task for the Linker Hand L20."""

from __future__ import annotations

import re
from typing import Any

import torch

from MFR_benchmark.isaac_lab_tasks.screwdriver_continuous_turning.screwdriver_continuous_turning_env import (
    ContinuousTurningRewardMixin,
)
from MFR_benchmark.isaac_lab_tasks.screwdriver_turning_linker_hand.screwdriver_turning_linker_hand_env import (
    LinkerHandScrewdriverTurningEnv,
)

from .screwdriver_continuous_turning_linker_hand_env_cfg import (
    LinkerHandScrewdriverContinuousTurningEnvCfg,
)


LINKER_INDEPENDENT_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "index": ("index_mcp_roll", "index_mcp_pitch", "index_pip"),
    "middle": ("middle_mcp_roll", "middle_mcp_pitch", "middle_pip"),
    "ring": ("ring_mcp_roll", "ring_mcp_pitch", "ring_pip"),
    "pinky": ("pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip"),
    "thumb": ("thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp"),
}

LINKER_FINGERTIP_BODY_NAMES: dict[str, str] = {
    "index": "index_distal",
    "middle": "middle_distal",
    "ring": "ring_distal",
    "pinky": "pinky_distal",
    "thumb": "thumb_distal",
}
SCREWDRIVER_BODY_PATTERN = ".*screwdriver_body$"
SCREWDRIVER_NEAR_BODY_NAMES = ("screwdriver_stick", "screwdriver_body", "screwdriver_cap")


class LinkerHandScrewdriverContinuousTurningEnv(ContinuousTurningRewardMixin, LinkerHandScrewdriverTurningEnv):
    """Linker Hand continuous-turn task with dense fingertip proximity shaping."""

    cfg: LinkerHandScrewdriverContinuousTurningEnvCfg

    def __init__(
        self, cfg: LinkerHandScrewdriverContinuousTurningEnvCfg, render_mode: str | None = None, **kwargs: Any
    ):
        self._fingertip_body_ids: list[int] = []
        self._screwdriver_body_id: int | None = None
        self._screwdriver_near_body_ids: list[int] = []
        self._thumb_tip_index: int | None = None
        self._non_thumb_tip_indices: list[int] = []
        super().__init__(cfg, render_mode, **kwargs)
        self._fingertip_body_ids = self._resolve_fingertip_bodies()
        self._screwdriver_body_id = self._resolve_screwdriver_body()
        self._screwdriver_near_body_ids = self._resolve_screwdriver_near_bodies()
        self._thumb_tip_index = self.fingers.index("thumb") if "thumb" in self.fingers else None
        self._non_thumb_tip_indices = [idx for idx, finger in enumerate(self.fingers) if finger != "thumb"]

    def _resolve_finger_joints(self) -> dict[str, list[int]]:
        unknown_fingers = set(self.fingers).difference(LINKER_INDEPENDENT_JOINT_NAMES)
        if unknown_fingers:
            raise ValueError(f"Unknown Linker Hand finger names: {sorted(unknown_fingers)}")

        joint_ids_by_name = {}
        for finger, joint_names in LINKER_INDEPENDENT_JOINT_NAMES.items():
            joint_ids_by_name[finger] = self._find_ordered_joints(self.allegro, joint_names)
        return joint_ids_by_name

    def _resolve_fingertip_bodies(self) -> list[int]:
        body_ids = []
        for finger in self.fingers:
            if finger not in LINKER_FINGERTIP_BODY_NAMES:
                raise ValueError(f"No fingertip body name configured for Linker finger {finger!r}.")
            body_name = LINKER_FINGERTIP_BODY_NAMES[finger]
            pattern = f"^{re.escape(body_name)}$"
            ids, found_names = self.allegro.find_bodies([pattern], preserve_order=True)
            if len(ids) != 1:
                raise RuntimeError(
                    f"Could not resolve fingertip body {body_name!r} on {self.allegro.cfg.prim_path}. "
                    f"Found {tuple(found_names)}."
                )
            body_ids.append(ids[0])
        return body_ids

    def _resolve_screwdriver_body(self) -> int:
        ids, found_names = self.screwdriver.find_bodies([SCREWDRIVER_BODY_PATTERN], preserve_order=True)
        if len(ids) != 1:
            raise RuntimeError(
                f"Could not resolve screwdriver body on {self.screwdriver.cfg.prim_path}. "
                f"Pattern: {SCREWDRIVER_BODY_PATTERN}. Found {tuple(found_names)}."
            )
        return ids[0]

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
            full_speed = max(float(getattr(self.cfg, "turn_reward_full_fingertip_speed", 0.0)), min_speed + 1.0e-6)
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
        if self.cfg.near_reward_weight <= 0.0:
            return zeros, zeros, {
                "eval_near_reward": zeros,
                "eval_near_score": zeros,
                "eval_mean_fingertip_dist": zeros,
            }

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
