from __future__ import annotations

from dataclasses import field
from pathlib import Path

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass


ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets"


@configclass
class AllegroScrewdriverManipulationEnvCfg(DirectRLEnvCfg):
    """Configuration for the MFR Allegro screwdriver manipulation (6D) DirectRLEnv."""

    # env
    decimation = 60
    episode_length_s = 20.0
    action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(12,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=1.5, replicate_physics=True)

    # task behavior
    fingers: tuple[str, ...] = ("index", "middle", "thumb")
    friction_coefficient: float = 1.0
    gradual_control: bool = True
    action_offset: bool = True
    randomize_obj_start: bool = False
    reset_contact_steps: int = 32

    # goal: pick up the screwdriver and hold it upright at the target pose
    goal_pos_xyz: tuple[float, float, float] = (0.0, 0.0, 0.35)
    goal_ori_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # reward weights
    reward_action_weight: float = 1.0
    reward_position_weight: float = 10.0
    reward_orientation_weight: float = 20.0
    reward_upright_weight: float = 1000.0
    drop_threshold: float = -0.25
    reward_drop_weight: float = 1000.0
    termination_height: float = -0.30
    reward_fingertip_distance_weight: float = 300.0

    pregrasp_positions: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "index": (0.0, 0.5, 0.7, 0.7),
            "middle": (0.0, 0.5, 0.7, 0.7),
            "ring": (0.0, 0.5, 0.7, 0.7),
            "thumb": (1.3, 0.3, 0.2, 1.1),
        }
    )

    # robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Allegro",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "xela_models/allegro_hand_right_isaaclab.urdf"),
            fix_base=True,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=True,
            make_instanceable=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.01, -0.028, 0.29),
            rot=(0.5, -0.5, 0.5, 0.5),
            joint_pos={
                "allegro_hand_hitosashi_finger_finger_joint_0": 0.0,
                "allegro_hand_hitosashi_finger_finger_joint_1": 0.5,
                "allegro_hand_hitosashi_finger_finger_joint_2": 0.7,
                "allegro_hand_hitosashi_finger_finger_joint_3": 0.7,
                "allegro_hand_naka_finger_finger_joint_4": 0.0,
                "allegro_hand_naka_finger_finger_joint_5": 0.5,
                "allegro_hand_naka_finger_finger_joint_6": 0.7,
                "allegro_hand_naka_finger_finger_joint_7": 0.7,
                "allegro_hand_kusuri_finger_finger_joint_8": 0.0,
                "allegro_hand_kusuri_finger_finger_joint_9": 0.5,
                "allegro_hand_kusuri_finger_finger_joint_10": 0.7,
                "allegro_hand_kusuri_finger_finger_joint_11": 0.7,
                "allegro_hand_oya_finger_joint_12": 1.3,
                "allegro_hand_oya_finger_joint_13": 0.3,
                "allegro_hand_oya_finger_joint_14": 0.2,
                "allegro_hand_oya_finger_joint_15": 1.1,
            },
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=6.0,
                damping=1.0,
                armature=0.001,
            )
        },
    )

    # 6D screwdriver object: 3 prismatic (pos) + 3 revolute (ori) + 1 revolute (cap)
    screwdriver_6d_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Screwdriver",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "screwdriver/screwdriver_6d_isaaclab.urdf"),
            fix_base=True,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=False,
            make_instanceable=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.205),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "table_screwdriver_joint_1": 0.0,
                "table_screwdriver_joint_2": 0.0,
                "table_screwdriver_joint_3": 0.0,
                "table_screwdriver_joint_4": 0.0,
                "table_screwdriver_joint_5": -1.57,
                "table_screwdriver_joint_6": 0.0,
                "screwdriver_body_cap_joint": 0.0,
            },
        ),
        actuators={
            "position": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_1", "table_screwdriver_joint_2", "table_screwdriver_joint_3"],
                stiffness=0.0,
                damping=0.1,
            ),
            "orientation": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_4", "table_screwdriver_joint_5", "table_screwdriver_joint_6"],
                stiffness=0.0,
                damping=0.001,
            ),
            "cap": ImplicitActuatorCfg(
                joint_names_expr=["screwdriver_body_cap_joint"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
