"""Allegro screwdriver 360-degree turning environment."""

from MFR_benchmark.isaac_lab_tasks.screwdriver_turning.screwdriver_turning_env import (
    AllegroScrewdriverTurningEnv,
)

from .screwdriver_turning_360_env_cfg import AllegroScrewdriverTurning360EnvCfg


class AllegroScrewdriverTurning360Env(AllegroScrewdriverTurningEnv):
    """360-degree variant — inherits all logic unchanged.

    The only difference from the base task is the config: goal_euler_xyz is set
    to -2*pi (~-6.283 rad) instead of -pi/2 (~-1.571 rad), and episode_length_s
    is longer. Since the Z-rotation joint is continuous (no limits, no wrapping),
    the existing squared-error reward naturally drives the policy toward exactly
    one full rotation.
    """

    cfg: AllegroScrewdriverTurning360EnvCfg
