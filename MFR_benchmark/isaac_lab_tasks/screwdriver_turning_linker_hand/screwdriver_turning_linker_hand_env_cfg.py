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
class AllegroScrewdriverTurningLinkerHandEnvCfg(DirectRLEnvCfg):
    """Configuration for the MFR Linker Hand L20 screwdriver turning DirectRLEnv."""

    # env
    decimation = 60
    episode_length_s = 12.0
    action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(12,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
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
    gradual_control: bool = False
    action_offset: bool = True
    randomize_obj_start: bool = False
    reset_contact_steps: int = 32
    goal_euler_xyz: tuple[float, float, float] = (0.0, 0.0, 1.5707)
    reward_action_weight: float = 1.0
    reward_goal_weight: float = 20.0 
    reward_upright_weight: float = 10000.0 # encourage upright orientation to prevent flipping the screwdriver around and losing contact
    pregrasp_positions: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "index": (0.0, 0.35, 0.45, 0.35),
            "middle": (0.0, 0.35, 0.45, 0.35),
            "ring": (0.0, 0.35, 0.45, 0.35),
            "pinky": (0.0, 0.35, 0.45, 0.35),
            "thumb": (0.35, 0.3, 0.18, 0.25),
        }
    )

    # robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/LinkerHand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "linker_hand_l20" / "linkerhand_l20_left.urdf"),
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
            pos=(0.0, 0.03, 1.33),
            rot=(0.7071, 0.0, 0.7071, 0.0),
            joint_pos={
                # index finger
                "index_mcp_roll": 0.0,
                "index_mcp_pitch": 0.35,
                "index_pip": 0.45,
                "index_dip": 0.35,
                # middle finger
                "middle_mcp_roll": 0.0,
                "middle_mcp_pitch": 0.35,
                "middle_pip": 0.45,
                "middle_dip": 0.35,
                # ring finger (unused in default 3-finger config, but needed for articulation init)
                "ring_mcp_roll": 0.0,
                "ring_mcp_pitch": 0.35,
                "ring_pip": 0.45,
                "ring_dip": 0.35,
                # pinky finger (unused in default 3-finger config, but needed for articulation init)
                "pinky_mcp_roll": 0.0,
                "pinky_mcp_pitch": 0.35,
                "pinky_pip": 0.45,
                "pinky_dip": 0.35,
                # thumb (4 actuated + 1 mimic)
                "thumb_cmc_yaw": 0.35,
                "thumb_cmc_roll": 0.3,
                "thumb_cmc_pitch": 0.18,
                "thumb_mcp": 0.25,
                "thumb_ip": 0.25,
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

    # screwdriver object
    screwdriver_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Screwdriver",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET_ROOT / "screwdriver" / "screwdriver_isaaclab.urdf"),
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
            pos=(0.0, 0.0, 1.205),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "tilt": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_1", "table_screwdriver_joint_2"],
                stiffness=0.0,
                damping=0.0001,
            ),
            "rotation": ImplicitActuatorCfg(
                joint_names_expr=["table_screwdriver_joint_3"],
                stiffness=0.0,
                damping=0.05,
            ),
            "cap": ImplicitActuatorCfg(
                joint_names_expr=["screwdriver_body_cap_joint"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
