"""Configuration for continuous Allegro screwdriver turning."""

import math

from isaaclab.utils import configclass

from MFR_benchmark.isaac_lab_tasks.screwdriver_turning.screwdriver_turning_env_cfg import (
    AllegroScrewdriverTurningEnvCfg,
)


@configclass
class AllegroScrewdriverContinuousTurningEnvCfg(AllegroScrewdriverTurningEnvCfg):
    """Continuous-turning variant of the legacy MFR screwdriver task.

    The original task's fixed goal is retained only for evaluation logging.
    Training reward is based on signed negative-z rotation progress.
    """

    episode_length_s: float = 60.0

    # Legacy 90-degree goal kept for metrics, not used by the reward.
    goal_euler_xyz: tuple[float, float, float] = (0.0, 0.0, -1.5707)
    reward_goal_weight: float = 0.0

    # HORA-style directional turning objective. -1.0 means negative-z progress.
    turn_direction: float = -1.0
    reward_turn_weight: float = 200.0
    turn_velocity_clip: float = 1.0
    reward_reverse_weight: float = 250.0

    # MFR-style stability, softened for initial continuous-turn exploration.
    reward_upright_weight: float = 200.0
    upright_termination_threshold: float = 1.0

    # Regularization. Action cost uses sum(action**2); action-rate uses mean.
    reward_action_weight: float = 0.25
    reward_action_rate_weight: float = 0.1

    # Small sparse helper for long-horizon progress logging/training.
    milestone_angle: float = 0.5 * math.pi
    milestone_bonus: float = 0.25
