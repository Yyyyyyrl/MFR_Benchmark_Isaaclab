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

    episode_length_s: float = 36.0
    goal_euler_xyz: tuple[float, float, float] = (0.0, 0.0, -2.0 * math.pi)

    # Lower goal weight — the squared error is 16x larger than the 90 task
    # because the target is 4x farther away (2*pi vs pi/2).
    reward_goal_weight: float = 50.0

    # Directional progress reward: reward each radian of rotation toward the goal.
    # This prevents the policy from learning the wrong turning direction.
    reward_delta_weight: float = 50.0
