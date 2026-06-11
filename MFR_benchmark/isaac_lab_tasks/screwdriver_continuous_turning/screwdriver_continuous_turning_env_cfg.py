"""Configuration for continuous Allegro screwdriver turning."""

import math

import gymnasium as gym
import numpy as np
from isaaclab.utils import configclass

from MFR_benchmark.isaac_lab_tasks.screwdriver_turning.screwdriver_turning_env_cfg import (
    AllegroScrewdriverTurningEnvCfg,
)


@configclass
class AllegroScrewdriverContinuousTurningEnvCfg(AllegroScrewdriverTurningEnvCfg):
    """Continuous-turning variant of the legacy MFR screwdriver task.

    The original task's fixed goal is retained only for evaluation logging.
    Training reward is based on HORA-style signed negative-z rotation progress.
    """

    episode_length_s: float = 60.0

    # HORA-style delta actions: target[t] = target[t-1] + 0.05 * action.
    # Action=0 holds current finger position; no retreat to pregrasp on idle.
    action_delta: bool = True
    # obs = [finger_q(12), cur_targets(12), screwdriver_euler(3)] = 27
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)

    # Legacy 90-degree goal kept for metrics, not used by the reward.
    goal_euler_xyz: tuple[float, float, float] = (0.0, 0.0, -1.5707)
    reward_goal_weight: float = 0.0

    # HORA-style directional turning objective. -1.0 means negative-z progress.
    reward_turn_weight: float = 200.0
    turn_direction: float = -1.0
    turn_velocity_clip: float = 1.0
    # Reverse penalty kept at/below the forward turn weight (mild bias); the
    # reverse cost is gated on contact in the reward, so a large value is not
    # needed and otherwise makes freezing the dominant strategy.
    reward_reverse_weight: float = 220.0

    # Mounted-screwdriver analogue of HORA's object linear-motion penalty.
    # The task should spin about z while keeping x/y tilt quiet.
    reward_upright_weight: float = 200.0
    reward_tilt_velocity_weight: float = 5.0
    upright_termination_threshold: float = 1.0

    # Multiplicative uprightness gate on the turn/milestone reward:
    # gate = exp(-(tilt_norm / std)**2). Tilting directly shrinks the dominant
    # reward term instead of racing it via an additive penalty, which a strong
    # turn reward always wins (tilt-and-scrape exploit). <= 0 disables.
    turn_upright_gate_std: float = 0.25

    # Measure spin as the screwdriver-stick quaternion delta projected on the
    # shaft axis (HORA-style). The raw Euler-z gimbal coordinate also counts
    # precession of a tilted shaft, rewarding wobble instead of true spin.
    use_shaft_spin_measure: bool = True

    # Contact proxy as fingertip distance to the handle axis segment instead of
    # body-origin distances. Thresholds become physically meaningful: handle
    # radius is 0.02 m, so tip-origin axis distance at pad contact is ~0.03 m.
    use_axis_contact_proxy: bool = True

    # HORA-like policy regularization. Action cost uses sum(action**2) for the
    # 12-DOF Allegro setup; action-rate and finger velocity are mean penalties.
    reward_action_weight: float = 0.25
    reward_action_rate_weight: float = 0.1
    reward_finger_pose_weight: float = 0.02
    reward_finger_velocity_weight: float = 0.001
    use_mean_action_penalty: bool = False

    # Small sparse helper for long-horizon progress logging/training.
    milestone_angle: float = 0.5 * math.pi
    milestone_bonus: float = 0.25

    # Dense discovery shaping using body positions only, not contact sensors.
    near_reward_weight: float = 0.2
    near_reward_std: float = 0.03
    near_reward_top_k: int = 2

    # Flat per-step bonus * (fingertips within turn_reward_contact_distance / num
    # fingers). Makes "hold the handle" the safe attractor when tilt costs and
    # termination would otherwise push the policy to disengage entirely.
    contact_bonus_weight: float = 0.0

    # Gate spin rewards so a flicked screwdriver cannot coast for reward after
    # the fingertips leave or stop moving. Set distance <= 0 to disable.
    turn_reward_contact_distance: float = 0.075
    turn_reward_min_contact_fingers: int = 2
    turn_reward_min_fingertip_speed: float = 0.003
    turn_reward_full_fingertip_speed: float = 0.015

    # Optional contact-proxy termination. Kept off by default for the MFR
    # pregrasp; curriculum can enable it after stable turning emerges.
    lost_contact_termination_distance: float = 0.0
    lost_contact_min_fingers: int = 1
    lost_contact_grace_steps: int = 2
