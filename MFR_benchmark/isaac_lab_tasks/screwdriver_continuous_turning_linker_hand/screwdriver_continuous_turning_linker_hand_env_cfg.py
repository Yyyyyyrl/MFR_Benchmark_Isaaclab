"""Configuration for continuous Linker Hand screwdriver turning."""

import copy
import math
from dataclasses import field

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from MFR_benchmark.isaac_lab_tasks.screwdriver_turning_linker_hand.screwdriver_turning_linker_hand_env_cfg import (
    AllegroScrewdriverTurningLinkerHandEnvCfg,
)


def _make_continuous_linker_robot_cfg() -> ArticulationCfg:
    robot_cfg = copy.deepcopy(AllegroScrewdriverTurningLinkerHandEnvCfg().robot_cfg)
    robot_cfg.init_state.pos = (0.13,-0.045, 1.36)
    robot_cfg.init_state.rot = (0.5, -0.5, -0.5, 0.5)  # 180 deg around X, then 90 deg around Z
    robot_cfg.init_state.joint_pos.update(
        {
            "index_mcp_roll": 0.0,
            "index_mcp_pitch": 0.55,
            "index_pip": 0.9,
            "index_dip": 0.8,

            "middle_mcp_roll": 0.0,
            "middle_mcp_pitch": 0.55,
            "middle_pip": 0.9,
            "middle_dip": 0.8,

            "ring_mcp_roll": 0.0,
            "ring_mcp_pitch": 0.55,
            "ring_pip": 0.9,
            "ring_dip": 0.8,

            "pinky_mcp_roll": 0.0,
            "pinky_mcp_pitch": 0.55,
            "pinky_pip": 0.9,
            "pinky_dip": 0.8,

            "thumb_cmc_yaw": 0.24,
            "thumb_cmc_roll": 0.6,
            "thumb_cmc_pitch": 0.62,
            "thumb_mcp": 0.65,
            "thumb_ip": 0.58,
        }
    )
    return robot_cfg


def _make_continuous_linker_screwdriver_cfg() -> ArticulationCfg:
    screwdriver_cfg = copy.deepcopy(AllegroScrewdriverTurningLinkerHandEnvCfg().screwdriver_cfg)
    # The fixed-goal MFR task tolerates a lightly damped z joint, but the
    # continuous-turn objective otherwise rewards flick-and-coast behavior.
    tilt = screwdriver_cfg.actuators["tilt"]
    tilt.stiffness = 20.0
    tilt.damping = 2.0

    # Model the spin resistance as mostly Coulomb (thread/driving) friction with
    # a small viscous component, rather than relying on viscous damping. Combined
    # with the now-explicit screwdriver inertia (see URDF), this gives a
    # deterministic spin that decays quickly on release instead of free-coasting.
    # friction/dynamic_friction/viscous_friction are real ActuatorBaseCfg fields
    # in this Isaac Lab version (verified) and are consumed by the solver.
    rotation = screwdriver_cfg.actuators["rotation"]
    rotation.stiffness = 0.0
    rotation.damping = 0.01
    rotation.friction = 0.05
    rotation.dynamic_friction = 0.04
    rotation.viscous_friction = 0.01

    cap = screwdriver_cfg.actuators["cap"]
    cap.stiffness = 50.0
    cap.damping = 1.0
    cap.friction = 0.0
    cap.dynamic_friction = 0.0
    cap.viscous_friction = 0.0
    return screwdriver_cfg


def _make_continuous_linker_sim_cfg() -> SimulationCfg:
    sim_cfg = copy.deepcopy(AllegroScrewdriverTurningLinkerHandEnvCfg().sim)
    sim_cfg.physx.gpu_max_rigid_patch_count = 2**22
    # The base sim bakes render_interval to the base decimation (60). Re-sync it
    # to the 20 Hz control cadence used by this task so rendered demos are smooth.
    sim_cfg.render_interval = 3
    return sim_cfg


@configclass
class LinkerHandScrewdriverContinuousTurningEnvCfg(AllegroScrewdriverTurningLinkerHandEnvCfg):
    """Continuous-turning variant of the Linker Hand screwdriver task.

    This uses the full five-finger independent Linker action set: three
    non-mimic joints for each non-thumb finger plus four thumb joints.
    Contact sensors are intentionally left for a later phase.
    """

    # obs = [finger_q(16), cur_targets(16), screwdriver_euler(3)] = 35
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)
    sim: SimulationCfg = _make_continuous_linker_sim_cfg()
    # 20 Hz control (override the inherited 1 Hz / decimation=60). Continuous
    # in-hand turning needs reactive finger motion; physics stays at 60 Hz.
    # NOTE: gamma in scripts/train_rma.py must move with this (0.9995 @ 20 Hz)
    # to preserve the effective horizon, and _make_continuous_linker_sim_cfg()
    # re-syncs render_interval so demos render at the control cadence.
    decimation = 3
    # HORA-style delta actions: target[t] = target[t-1] + action_delta_scale * action.
    # Action=0 holds current finger position; no retreat to pregrasp on idle.
    # 0.025 rad/step at 20 Hz = 0.5 rad/s max — matches Allegro's 0.05 rad/step at 10 Hz.
    action_delta: bool = True
    action_delta_scale: float = 0.025
    action_clip: float = 1.0
    episode_length_s: float = 60.0
    # The side-grasp reset is already close to the handle. Extra contact-settle
    # steps can preload the passive screwdriver tilt joints, so keep reset vertical.
    reset_contact_steps: int = 0

    fingers: tuple[str, ...] = ("index", "middle", "ring", "pinky", "thumb")
    pregrasp_positions: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "index": (0.0, 0.55, 0.9),
            "middle": (0.0, 0.55, 0.9),
            "ring": (0.0, 0.55, 0.9),
            "pinky": (0.0, 0.55, 0.9),
            "thumb": (0.24, 0.6, 0.62, 0.65),
        }
    )

    # The base Linker task was ported with the palm above the screwdriver. For
    # continuous turning, reset directly into a five-finger grasp around the
    # screwdriver body/cap so early training receives contact-rich rollouts.
    robot_cfg: ArticulationCfg = _make_continuous_linker_robot_cfg()
    screwdriver_cfg: ArticulationCfg = _make_continuous_linker_screwdriver_cfg()

    # RMA settings. The fixed-goal Linker config does not currently define these,
    # but the inherited environment allocates history buffers when constructed.
    asymmetric_obs: bool = False
    privileged_obs_dim: int = 14
    prop_hist_len: int = 30
    history_obs_dim: int = 32

    # Legacy 90-degree goal kept for metrics, not used by the continuous reward.
    goal_euler_xyz: tuple[float, float, float] = (0.0, 0.0, -1.5707)
    reward_goal_weight: float = 0.0

    # HORA-style directional turning objective. -1.0 means negative-z progress.
    turn_direction: float = -1.0
    reward_turn_weight: float = 200.0
    turn_velocity_clip: float = 1.0
    # Reverse penalty kept at/below the forward turn weight (mild bias). With the
    # reverse cost now gated on contact, a strong reverse penalty is no longer
    # needed and previously made freezing the dominant strategy.
    reward_reverse_weight: float = 220.0

    # Softened stability for continuous-turn exploration. Curriculum can tighten.
    reward_upright_weight: float = 200.0
    upright_termination_threshold: float = 1.0

    # Regularization. Linker defaults to mean action penalty so the weight is
    # stable across the 16 controlled joints.
    reward_action_weight: float = 0.25
    reward_action_rate_weight: float = 0.1
    use_mean_action_penalty: bool = True

    # Small sparse helper for long-horizon progress logging/training.
    milestone_angle: float = 0.5 * math.pi
    milestone_bonus: float = 0.25

    # Dense discovery shaping. This uses body positions only; no contact sensors.
    near_reward_weight: float = 0.2
    near_reward_std: float = 0.03
    near_reward_top_k: int = 3

    # Gate turn reward so a flicked screwdriver cannot score while coasting past
    # a stationary hand. Distances are measured to the nearest screwdriver link.
    turn_reward_contact_distance: float = 0.075
    turn_reward_min_contact_fingers: int = 3
    turn_reward_min_fingertip_speed: float = 0.003
    turn_reward_full_fingertip_speed: float = 0.015

    # Terminate rollouts that have fully lost the screwdriver. Without contact
    # sensors this uses fingertip distance to the nearest screwdriver link as a
    # conservative contact proxy.
    lost_contact_termination_distance: float = 0.085
    lost_contact_min_fingers: int = 1
    lost_contact_grace_steps: int = 2
