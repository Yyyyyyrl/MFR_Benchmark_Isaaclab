"""Configuration for the MFR Allegro screwdriver 360-degree turning task."""

import math

from MFR_benchmark.isaac_lab_tasks.screwdriver_turning.screwdriver_turning_env_cfg import (
    AllegroScrewdriverTurningEnvCfg,
)
from isaaclab.utils import configclass


@configclass
class AllegroScrewdriverTurning360EnvCfg(AllegroScrewdriverTurningEnvCfg):
    """360-degree variant of the screwdriver turning task.

    Inherits all robot, screwdriver, simulation, and actuator settings from the
    base task. Overrides only the task-level parameters needed for a full rotation.
    """

    episode_length_s: float = 36.0  # 3x original — more time for a full rotation
    goal_euler_xyz: tuple[float, float, float] = (0.0, 0.0, -2.0 * math.pi)
