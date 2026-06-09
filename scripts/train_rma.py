#!/usr/bin/env python3
"""RMA 2-Stage Training Script for MFR_benchmark (Isaac Lab).

Implements the Rapid Motor Adaptation teacher-student training pipeline.

Usage:
    # Stage 1: Teacher policy with privileged info
    python scripts/train_rma.py --task Isaac-Allegro-Screwdriver-Turning-Direct-v0 \
        --stage 1 --num_envs 8192 --headless

    # Stage 2: Student adaptation module
    python scripts/train_rma.py --task Isaac-Allegro-Screwdriver-Turning-Direct-v0 \
        --stage 2 --num_envs 8192 --headless \
        --checkpoint outputs/rma/stage1/best.pth

    # Continuous turning teacher policy with HORA-style curriculum
    python scripts/train_rma.py --task Isaac-Allegro-Screwdriver-Continuous-Turning-Direct-v0 \
        --stage 1 --num_envs 8192 --headless --continuous_curriculum

Requirements:
    conda activate env_isaaclab
    export PYTHONPATH=/home/user/MFR_benchmark:$PYTHONPATH
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ---- CLI arguments ----
parser = argparse.ArgumentParser(description="RMA 2-Stage Training for MFR Benchmark")

# Environment args
parser.add_argument("--task", type=str, default="Isaac-Allegro-Screwdriver-Turning-Direct-v0",
                    help="Gymnasium task ID")
parser.add_argument("--num_envs", type=int, default=8192,
                    help="Number of parallel environments")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="Override env episode length in seconds/policy steps")

# Continuous-turn reward override args. These are no-ops for tasks without the field.
parser.add_argument("--continuous_phase1", action="store_true",
                    help="Use a softer continuous-turn startup curriculum")
parser.add_argument("--continuous_curriculum", action="store_true",
                    help="Automatically tighten continuous-turn reward/termination settings during Stage 1")
parser.add_argument("--turn_direction", type=float, default=None,
                    help="Override continuous-turn direction: -1 for negative z, +1 for positive z")
parser.add_argument("--reward_turn_weight", type=float, default=None,
                    help="Override continuous-turn forward velocity reward weight")
parser.add_argument("--turn_velocity_clip", type=float, default=None,
                    help="Override continuous-turn velocity clip in rad/s")
parser.add_argument("--reward_reverse_weight", type=float, default=None,
                    help="Override continuous-turn reverse velocity penalty weight")
parser.add_argument("--reward_upright_weight", type=float, default=None,
                    help="Override screwdriver upright penalty weight")
parser.add_argument("--upright_termination_threshold", type=float, default=None,
                    help="Override upright termination threshold; <=0 disables it")
parser.add_argument("--reward_action_weight", type=float, default=None,
                    help="Override action magnitude penalty weight")
parser.add_argument("--reward_action_rate_weight", type=float, default=None,
                    help="Override action-rate penalty weight")
parser.add_argument("--reward_tilt_velocity_weight", type=float, default=None,
                    help="Override x/y screwdriver tilt velocity penalty weight")
parser.add_argument("--reward_finger_pose_weight", type=float, default=None,
                    help="Override finger deviation-from-pregrasp penalty weight")
parser.add_argument("--reward_finger_velocity_weight", type=float, default=None,
                    help="Override finger joint velocity penalty weight")
parser.add_argument("--milestone_angle", type=float, default=None,
                    help="Override milestone angle in radians")
parser.add_argument("--milestone_bonus", type=float, default=None,
                    help="Override milestone bonus")
parser.add_argument("--near_reward_weight", type=float, default=None,
                    help="Override dense fingertip near-contact reward weight")
parser.add_argument("--near_reward_std", type=float, default=None,
                    help="Override dense fingertip near-contact reward distance scale in meters")
parser.add_argument("--near_reward_top_k", type=int, default=None,
                    help="Override number of non-thumb fingertips used by near-contact shaping")
parser.add_argument("--turn_reward_contact_distance", type=float, default=None,
                    help="Override fingertip-distance contact proxy for turn reward gating; <=0 disables")
parser.add_argument("--turn_reward_min_contact_fingers", type=int, default=None,
                    help="Override minimum fingertip contact-proxy count for turn reward gating")
parser.add_argument("--turn_reward_min_fingertip_speed", type=float, default=None,
                    help="Override fingertip speed where motion gate starts opening")
parser.add_argument("--turn_reward_full_fingertip_speed", type=float, default=None,
                    help="Override fingertip speed where motion gate reaches 1")
parser.add_argument("--lost_contact_termination_distance", type=float, default=None,
                    help="Override contact-proxy termination distance; <=0 disables")
parser.add_argument("--lost_contact_min_fingers", type=int, default=None,
                    help="Override minimum contact-proxy fingers before lost-contact termination")
parser.add_argument("--lost_contact_grace_steps", type=int, default=None,
                    help="Override policy steps before lost-contact termination can trigger")
parser.add_argument("--use_mean_action_penalty", action="store_true",
                    help="Use mean(action**2) instead of sum(action**2) when the env supports it")

# RMA-specific args
parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                    help="Training stage: 1=teacher PPO, 2=student adaptation")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint for restoring (Stage 2 requires Stage 1 ckpt)")
parser.add_argument("--output", type=str, default="outputs/rma",
                    help="Output directory for checkpoints and logs")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_steps", type=int, default=1500000000,
                    help="Maximum agent steps")
parser.add_argument("--horizon_length", type=int, default=8,
                    help="PPO horizon length (steps per rollout)")
parser.add_argument("--minibatch_size", type=int, default=32768,
                    help="PPO minibatch size")
parser.add_argument("--learning_rate", type=float, default=5e-3,
                    help="PPO learning rate (Stage 1 only)")
parser.add_argument("--prop_hist_len", type=int, default=30,
                    help="Proprioceptive history length (timesteps)")
parser.add_argument("--privileged_obs_dim", type=int, default=None,
                    help="Dimension of privileged observations. Defaults to the task config.")
parser.add_argument("--history_obs_dim", type=int, default=None,
                    help="Features per history timestep. Defaults to the task config.")

# Append AppLauncher CLI args (provides --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args_cli = parser.parse_args()

# Launch the Omniverse Kit app (bootstraps pxr, USD bindings, etc.)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the training logic follows."""

import os
import sys
import time
import importlib
import torch

import gymnasium as gym

# Register MFR tasks
import MFR_benchmark.isaac_lab_tasks  # noqa: F401

from MFR_benchmark.rma import PPO, ProprioAdapt


def _resolve_entry_point(entry_point_str: str):
    """Resolve a 'module.path:ClassName' string to the actual class."""
    module_path, class_name = entry_point_str.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _set_cfg_if_present(env_cfg, name: str, value):
    if value is None or not hasattr(env_cfg, name):
        return False
    setattr(env_cfg, name, value)
    return True


def _apply_env_overrides(env_cfg):
    """Apply CLI overrides that are useful for curriculum tuning."""
    overrides = {}

    if args_cli.continuous_phase1 and not args_cli.continuous_curriculum:
        phase1_defaults = {
            "reward_turn_weight": 1500.0,
            "reward_reverse_weight": 2000.0,
            "turn_velocity_clip": 1.0,
            "reward_upright_weight": 5.0,
            "upright_termination_threshold": 0.0,
            "reward_action_weight": 0.05,
            "reward_action_rate_weight": 0.0,
            "reward_tilt_velocity_weight": 0.0,
            "reward_finger_pose_weight": 0.0,
            "reward_finger_velocity_weight": 0.0,
            "milestone_bonus": 0.0,
            "near_reward_weight": 0.25,
            "turn_reward_contact_distance": 0.0,
            "lost_contact_termination_distance": 0.0,
        }
        for name, value in phase1_defaults.items():
            if hasattr(env_cfg, name):
                setattr(env_cfg, name, value)
                overrides[name] = value

    for name in (
        "episode_length_s",
        "turn_direction",
        "reward_turn_weight",
        "turn_velocity_clip",
        "reward_reverse_weight",
        "reward_upright_weight",
        "upright_termination_threshold",
        "reward_action_weight",
        "reward_action_rate_weight",
        "reward_tilt_velocity_weight",
        "reward_finger_pose_weight",
        "reward_finger_velocity_weight",
        "milestone_angle",
        "milestone_bonus",
        "near_reward_weight",
        "near_reward_std",
        "near_reward_top_k",
        "turn_reward_contact_distance",
        "turn_reward_min_contact_fingers",
        "turn_reward_min_fingertip_speed",
        "turn_reward_full_fingertip_speed",
        "lost_contact_termination_distance",
        "lost_contact_min_fingers",
        "lost_contact_grace_steps",
    ):
        value = getattr(args_cli, name)
        if _set_cfg_if_present(env_cfg, name, value):
            overrides[name] = value

    if args_cli.use_mean_action_penalty and hasattr(env_cfg, "use_mean_action_penalty"):
        env_cfg.use_mean_action_penalty = True
        overrides["use_mean_action_penalty"] = True

    return overrides


def _build_continuous_curriculum_config():
    if not args_cli.continuous_curriculum:
        return {"enabled": False}

    return {
        "enabled": True,
        "reset_best_on_phase_change": True,
        "save_on_phase_change": True,
        "phases": [
            {
                "name": "phase1_hora_spin_discovery",
                "overrides": {
                    # Debug bootstrap: make any useful contact/rotation show up clearly.
                    "reward_turn_weight": 1000.0,
                    "reward_reverse_weight": 1000,
                    "turn_velocity_clip": 1.0,
                    "reward_upright_weight": 2.0,
                    "reward_tilt_velocity_weight": 0.0,
                    "upright_termination_threshold": 0.0,
                    "reward_action_weight": 0.02,
                    "reward_action_rate_weight": 0.0,
                    "reward_finger_pose_weight": 0.0,
                    "reward_finger_velocity_weight": 0.0,
                    "milestone_bonus": 0.0,
                    "near_reward_weight": 2.0,
                    "near_reward_std": 0.08,
                    "near_reward_top_k": 3,
                    "turn_reward_contact_distance": 0.15,
                    "turn_reward_min_contact_fingers": 1,
                    "turn_reward_min_fingertip_speed": 0.0,
                    "turn_reward_full_fingertip_speed": 0.003,
                    "lost_contact_termination_distance": 0.0,
                },
                "advance": {
                    # Shorten for debugging; restore to 8_000_000 after confirming spin.
                    "min_phase_steps": 8_000_000,
                    "min_episode_length": 45.0,
                    "min_net_turns": 0.10,
                    "min_fwd_minus_rev": 0.04,
                    "max_upright": 2.5,
                    "max_mean_fingertip_dist": 0.16,
                },
            },
            {
                "name": "phase2_contacted_direction",
                "overrides": {
                    "reward_turn_weight": 1000.0,
                    "reward_reverse_weight": 1200.0,
                    "turn_velocity_clip": 1.0,
                    "reward_upright_weight": 25.0,
                    "reward_tilt_velocity_weight": 1.0,
                    "upright_termination_threshold": 0.0,
                    "reward_action_weight": 0.10,
                    "reward_action_rate_weight": 0.02,
                    "reward_finger_pose_weight": 0.005,
                    "reward_finger_velocity_weight": 0.0005,
                    "milestone_bonus": 0.05,
                    "near_reward_weight": 0.15,
                    "turn_reward_contact_distance": 0.12,
                    "turn_reward_min_contact_fingers": 1,
                    "turn_reward_min_fingertip_speed": 0.0,
                    "turn_reward_full_fingertip_speed": 0.006,
                    "lost_contact_termination_distance": 0.0,
                },
                "advance": {
                    "min_phase_steps": 8_000_000,
                    "min_episode_length": 45.0,
                    "min_net_turns": 0.15,
                    "min_fwd_minus_rev": 0.06,
                    "max_upright": 1.20,
                    "max_mean_fingertip_dist": 0.10,
                },
            },
            {
                "name": "phase3_stable_contact_turning",
                "overrides": {
                    "reward_turn_weight": 500.0,
                    "reward_reverse_weight": 700.0,
                    "turn_velocity_clip": 0.75,
                    "reward_upright_weight": 150.0,
                    "reward_tilt_velocity_weight": 3.0,
                    "upright_termination_threshold": 1.0,
                    "reward_action_weight": 0.15,
                    "reward_action_rate_weight": 0.05,
                    "reward_finger_pose_weight": 0.01,
                    "reward_finger_velocity_weight": 0.001,
                    "milestone_bonus": 0.10,
                    "near_reward_weight": 0.05,
                    "turn_reward_contact_distance": 0.08,
                    "turn_reward_min_contact_fingers": 2,
                    "turn_reward_min_fingertip_speed": 0.003,
                    "turn_reward_full_fingertip_speed": 0.015,
                    "lost_contact_termination_distance": 0.0,
                },
                "advance": {
                    "min_phase_steps": 8_000_000,
                    "min_episode_length": 45.0,
                    "min_net_turns": 0.10,
                    "min_fwd_minus_rev": 0.04,
                    "max_upright": 0.70,
                    "max_mean_fingertip_dist": 0.085,
                },
            },
            {
                "name": "phase4_strict_continuous_turning",
                "overrides": {
                    "reward_turn_weight": 200.0,
                    "reward_reverse_weight": 250.0,
                    "turn_velocity_clip": 0.5,
                    "reward_upright_weight": 1000.0,
                    "reward_tilt_velocity_weight": 5.0,
                    "upright_termination_threshold": 0.5,
                    "reward_action_weight": 0.25,
                    "reward_action_rate_weight": 0.1,
                    "reward_finger_pose_weight": 0.02,
                    "reward_finger_velocity_weight": 0.001,
                    "milestone_bonus": 0.25,
                    "near_reward_weight": 0.0,
                    "turn_reward_contact_distance": 0.075,
                    "turn_reward_min_contact_fingers": 2,
                    "turn_reward_min_fingertip_speed": 0.003,
                    "turn_reward_full_fingertip_speed": 0.015,
                    "lost_contact_termination_distance": 0.10,
                    "lost_contact_min_fingers": 1,
                    "lost_contact_grace_steps": 5,
                },
            },
        ],
    }


def main():
    """Create environment and run RMA training."""
    device = args_cli.device if args_cli.device is not None else "cuda:0"

    print("=" * 60)
    print(f"RMA Stage {args_cli.stage} Training")
    print(f"  Task: {args_cli.task}")
    print(f"  Device: {device}")
    print("=" * 60)

    # Get the environment configuration class from the gym registry
    env_spec = gym.spec(args_cli.task)
    env_cfg_cls = _resolve_entry_point(env_spec.kwargs["env_cfg_entry_point"])
    env_cfg = env_cfg_cls()

    # Override config with CLI args
    env_cfg.scene.num_envs = args_cli.num_envs
    env_overrides = _apply_env_overrides(env_cfg)

    # Enable asymmetric (privileged) observations for RMA.
    privileged_obs_dim = (
        args_cli.privileged_obs_dim if args_cli.privileged_obs_dim is not None else env_cfg.privileged_obs_dim
    )
    history_obs_dim = args_cli.history_obs_dim if args_cli.history_obs_dim is not None else env_cfg.history_obs_dim
    env_cfg.asymmetric_obs = True
    env_cfg.prop_hist_len = args_cli.prop_hist_len
    env_cfg.privileged_obs_dim = privileged_obs_dim
    env_cfg.history_obs_dim = history_obs_dim

    # Create the environment via gym with the config object
    env = gym.make(args_cli.task, cfg=env_cfg)
    # Use the raw (unwrapped) env for training — the Gymnasium OrderEnforcing
    # wrapper doesn't delegate custom attributes like num_finger_dofs.
    env = env.unwrapped

    num_finger_dofs = env.num_finger_dofs
    obs_space = env.single_observation_space
    obs_shape = obs_space.shape if not hasattr(obs_space, 'spaces') else obs_space['policy'].shape
    print(f"  Num envs: {env.num_envs}")
    print(f"  Finger DOFs: {num_finger_dofs}")
    print(f"  Obs dim (policy): {obs_shape}")
    print(f"  Action dim: {env.single_action_space.shape}")
    print(f"  Privileged obs dim: {privileged_obs_dim}")
    print(f"  History length: {args_cli.prop_hist_len}")
    print(f"  History obs dim: {history_obs_dim}")
    if env_overrides:
        print("  Env overrides:")
        for name, value in sorted(env_overrides.items()):
            print(f"    {name}: {value}")
    if args_cli.continuous_curriculum:
        print("  Continuous curriculum: enabled")

    # Training configuration
    network_config = {
        "mlp": {"units": [512, 256, 128]},
        "priv_mlp": {"units": [256, 128, 8]},
    }

    ppo_config = {
        "learning_rate": args_cli.learning_rate,
        "horizon_length": args_cli.horizon_length,
        "minibatch_size": args_cli.minibatch_size,
        "mini_epochs": 5,
        "e_clip": 0.2,
        "gamma": 0.99,
        "tau": 0.95,
        "entropy_coef": 0.0,
        "critic_coef": 4.0,
        "bounds_loss_coef": 0.0001,
        "kl_threshold": 0.02,
        "max_agent_steps": args_cli.max_steps,
        "save_frequency": 200,
        "save_best_after": 100,
        "truncate_grads": True,
        "grad_norm": 1.0,
        "normalize_input": True,
        "normalize_value": True,
        "normalize_advantage": True,
        "value_bootstrap": True,
        "curriculum": _build_continuous_curriculum_config(),
    }

    task_prefix = args_cli.task.replace("-", "_").replace("Isaac_", "")
    output_dir = os.path.join(args_cli.output, f"{task_prefix}_stage{args_cli.stage}")

    if args_cli.stage == 1:
        print("=" * 60)
        print("Stage 1: Teacher Policy (PPO with Privileged Information)")
        print("=" * 60)

        trainer = PPO(
            env=env,
            output_dir=output_dir,
            device=device,
            network_config=network_config,
            ppo_config=ppo_config,
            priv_info=True,
            proprio_adapt=False,
            priv_info_dim=privileged_obs_dim,
            adapt_obs_dim=history_obs_dim,
            adapt_history_len=args_cli.prop_hist_len,
        )

        if args_cli.checkpoint:
            print(f"Restoring from checkpoint: {args_cli.checkpoint}")
            trainer.restore_train(args_cli.checkpoint)

        trainer.train()

    elif args_cli.stage == 2:
        print("=" * 60)
        print("Stage 2: Student Adaptation Module (MSE Distillation)")
        print("=" * 60)

        if not args_cli.checkpoint:
            print("ERROR: Stage 2 requires --checkpoint (Stage 1 checkpoint path)")
            env.close()
            simulation_app.close()
            sys.exit(1)

        trainer = ProprioAdapt(
            env=env,
            output_dir=output_dir,
            device=device,
            network_config=network_config,
            ppo_config=ppo_config,
            priv_info_dim=privileged_obs_dim,
            proprio_hist_len=args_cli.prop_hist_len,
            adapt_obs_dim=history_obs_dim,
        )

        print(f"Loading Stage 1 checkpoint: {args_cli.checkpoint}")
        trainer.restore_train(args_cli.checkpoint)

        trainer.train()

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
    simulation_app.close()