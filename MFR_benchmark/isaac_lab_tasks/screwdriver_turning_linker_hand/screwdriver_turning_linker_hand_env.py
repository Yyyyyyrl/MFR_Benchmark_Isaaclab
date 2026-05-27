from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.assets import Articulation

from ..screwdriver_turning.screwdriver_turning_env import (
    AllegroScrewdriverTurningEnv,
    SCREWDRIVER_CAP_JOINT_NAME,
    SCREWDRIVER_EULER_JOINT_NAMES,
)

from .screwdriver_turning_linker_hand_env_cfg import AllegroScrewdriverTurningLinkerHandEnvCfg


LINKER_HAND_JOINT_NAMES: dict[str, tuple[str, str, str, str]] = {
    "index": (
        "index_mcp_roll",
        "index_mcp_pitch",
        "index_pip",
        "index_dip",
    ),
    "middle": (
        "middle_mcp_roll",
        "middle_mcp_pitch",
        "middle_pip",
        "middle_dip",
    ),
    "ring": (
        "ring_mcp_roll",
        "ring_mcp_pitch",
        "ring_pip",
        "ring_dip",
    ),
    "pinky": (
        "pinky_mcp_roll",
        "pinky_mcp_pitch",
        "pinky_pip",
        "pinky_dip",
    ),
    "thumb": (
        "thumb_cmc_yaw",
        "thumb_cmc_roll",
        "thumb_cmc_pitch",
        "thumb_mcp",
    ),
}
"""Joint names for the Linker Hand L20 left hand.

Each finger has 4 joints: mcp_roll (X-axis, abduction), mcp_pitch (Y-axis, flexion),
pip (Y-axis, flexion), and dip (Y-axis, mimic of pip). The thumb has 5 joints total;
the 4 included here are the independently-actuated ones (cmc_yaw, cmc_roll, cmc_pitch,
mcp) — thumb_ip is a mimic of thumb_mcp and follows automatically.
"""


class LinkerHandScrewdriverTurningEnv(AllegroScrewdriverTurningEnv):
    """Isaac Lab DirectRLEnv for the Linker Hand L20 screwdriver turning task.

    Inherits all logic from the Allegro screwdriver turning env, overriding only
    the finger joint name resolution to use Linker Hand joint naming conventions.
    """

    cfg: AllegroScrewdriverTurningLinkerHandEnvCfg

    def __init__(
        self, cfg: AllegroScrewdriverTurningLinkerHandEnvCfg, render_mode: str | None = None, **kwargs: Any
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Rebuild pregrasp positions for all Linker Hand fingers (base class only
        # knows about Allegro fingers: index, middle, ring, thumb).
        linker_fingers = ("index", "middle", "ring", "pinky", "thumb")
        self._all_pregrasp_pos_by_finger = {
            finger: torch.tensor(self.cfg.pregrasp_positions[finger], dtype=torch.float32, device=self.device)
            for finger in linker_fingers
        }

    def _resolve_finger_joints(self) -> dict[str, list[int]]:
        unknown_fingers = set(self.fingers).difference(LINKER_HAND_JOINT_NAMES)
        if unknown_fingers:
            raise ValueError(f"Unknown Linker Hand finger names: {sorted(unknown_fingers)}")

        joint_ids_by_name = {}
        for finger, joint_names in LINKER_HAND_JOINT_NAMES.items():
            joint_ids_by_name[finger] = self._find_ordered_joints(self.allegro, joint_names)
        return joint_ids_by_name

    def _find_ordered_joints(self, articulation: Articulation, joint_names: Sequence[str]) -> list[int]:
        import re

        patterns = [f"^{re.escape(joint_name)}$" for joint_name in joint_names]
        joint_ids, found_names = articulation.find_joints(patterns, preserve_order=True)
        if len(joint_ids) != len(joint_names):
            raise RuntimeError(
                f"Could not resolve joints {tuple(joint_names)} on {articulation.cfg.prim_path}. "
                f"Found {tuple(found_names)}."
            )
        return joint_ids
