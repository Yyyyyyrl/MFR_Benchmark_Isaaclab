#!/usr/bin/env python3
"""Evaluation script for RMA-trained policies on MFR_benchmark.

Evaluates Stage 1 (teacher, with privileged info) or Stage 2 (student,
without privileged info) checkpoints and computes success metrics.

Usage:
    # Evaluate Stage 1 teacher
    python scripts/eval_rma.py \
        --task Isaac-Allegro-Screwdriver-Turning-Direct-v0 \
        --stage 1 --checkpoint outputs/stage1/best.pth \
        --num_envs 256 --num_episodes 1000 --headless

    # Evaluate Stage 2 student (no privileged info)
    python scripts/eval_rma.py \
        --task Isaac-Allegro-Screwdriver-Turning-Direct-v0 \
        --stage 2 --checkpoint outputs/stage2/best.pth \
        --num_envs 256 --num_episodes 1000 --headless

Requirements:
    conda activate env_isaaclab
    export PYTHONPATH=/home/user/MFR_benchmark:$PYTHONPATH
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ---- CLI arguments ----
parser = argparse.ArgumentParser(description="Evaluate RMA-trained policies")

# Environment args
parser.add_argument("--task", type=str, default="Isaac-Allegro-Screwdriver-Turning-Direct-v0",
                    help="Gymnasium task ID")
parser.add_argument("--num_envs", type=int, default=256,
                    help="Number of parallel environments")

# Evaluation args
parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                    help="Which stage checkpoint to evaluate")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint (.pth file)")
parser.add_argument("--num_episodes", type=int, default=1000,
                    help="Number of episodes to evaluate")
parser.add_argument("--success_threshold", type=float, default=0.1,
                    help="Goal error threshold (rad) for success")
parser.add_argument("--prop_hist_len", type=int, default=30,
                    help="Proprioceptive history length")
parser.add_argument("--privileged_obs_dim", type=int, default=14,
                    help="Dimension of privileged observations")
parser.add_argument("--history_obs_dim", type=int, default=24,
                    help="Features per history timestep")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Launch the Omniverse Kit app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the evaluation logic follows."""

import torch
import gymnasium as gym

import MFR_benchmark.isaac_lab_tasks  # noqa: F401

from MFR_benchmark.rma.models import ActorCritic
from MFR_benchmark.rma.running_mean_std import RunningMeanStd


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"

    print("=" * 60)
    print(f"RMA Stage {args_cli.stage} Evaluation")
    print(f"  Task: {args_cli.task}")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Episodes: {args_cli.num_episodes}")
    print("=" * 60)

    # Create environment
    env = gym.make(
        args_cli.task,
        num_envs=args_cli.num_envs,
        headless=args_cli.headless,
    )

    # Enable asymmetric observations
    env.cfg.asymmetric_obs = True
    env.cfg.prop_hist_len = args_cli.prop_hist_len
    env.cfg.privileged_obs_dim = args_cli.privileged_obs_dim
    env.cfg.history_obs_dim = args_cli.history_obs_dim
    env._asymmetric_obs = True
    env._prop_hist_len = args_cli.prop_hist_len
    env._history_obs_dim = args_cli.history_obs_dim
    env._proprio_hist_buf = torch.zeros(
        (env.num_envs, env._prop_hist_len, env._history_obs_dim),
        dtype=torch.float32,
        device=env.device,
    )

    # Build model
    obs_space = env.single_observation_space
    if hasattr(obs_space, "spaces"):
        obs_shape = obs_space["policy"].shape
    else:
        obs_shape = obs_space.shape

    action_space = env.single_action_space
    use_adapt = (args_cli.stage == 2)

    net_config = {
        "actor_units": [512, 256, 128],
        "priv_mlp_units": [256, 128, 8],
        "actions_num": action_space.shape[0],
        "input_shape": obs_shape,
        "priv_info": True,
        "proprio_adapt": use_adapt,
        "priv_info_dim": args_cli.privileged_obs_dim,
        "adapt_obs_dim": args_cli.history_obs_dim,
        "adapt_history_len": args_cli.prop_hist_len,
    }

    model = ActorCritic(net_config)
    model.to(device)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)

    # Normalization
    running_mean_std = RunningMeanStd(obs_shape).to(device)
    running_mean_std.eval()
    if "running_mean_std" in checkpoint:
        running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    sa_mean_std = None
    if use_adapt and "sa_mean_std" in checkpoint:
        sa_mean_std = RunningMeanStd(
            (args_cli.prop_hist_len, args_cli.history_obs_dim)
        ).to(device)
        sa_mean_std.eval()
        sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])

    # Evaluation loop
    obs_dict = env.reset()
    if isinstance(obs_dict, tuple):
        obs_dict = obs_dict[0]

    episode_count = 0
    success_count = 0
    total_reward = 0.0
    episode_rewards = torch.zeros(args_cli.num_envs, device=device)
    goal_euler = torch.tensor(
        env.cfg.goal_euler_xyz, dtype=torch.float32, device=device
    )

    print(f"Evaluating for {args_cli.num_episodes} episodes...")

    while episode_count < args_cli.num_episodes:
        obs = obs_dict["policy"]
        input_dict = {"obs": running_mean_std(obs)}

        if args_cli.stage == 1:
            # Teacher: use privileged info
            input_dict["priv_info"] = obs_dict.get("critic", None)
        elif args_cli.stage == 2 and sa_mean_std is not None:
            # Student: use proprioceptive history only
            input_dict["proprio_hist"] = sa_mean_std(
                obs_dict["proprio_hist"].detach()
            )

        mu = model.act_inference(input_dict)
        mu = torch.clamp(mu, -1.0, 1.0)

        result = env.step(mu)
        obs_dict, rewards, terminated, timed_out, extras = result
        dones = terminated | timed_out

        episode_rewards += rewards

        # Check success for done environments
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_indices:
            if episode_count >= args_cli.num_episodes:
                break
            goal_error = extras.get("eval_screwdriver_goal_error", None)
            if goal_error is not None:
                error_norm = torch.norm(goal_error[idx]).item()
                if error_norm < args_cli.success_threshold:
                    success_count += 1
            total_reward += episode_rewards[idx].item()
            episode_count += 1

        # Reset tracking for non-done envs
        not_dones = 1.0 - dones.float()
        episode_rewards = episode_rewards * not_dones

        if episode_count % 100 == 0 and episode_count > 0:
            success_rate = success_count / max(episode_count, 1) * 100
            avg_reward = total_reward / max(episode_count, 1)
            print(
                f"  Episodes: {episode_count}/{args_cli.num_episodes} | "
                f"Success: {success_rate:.1f}% | "
                f"Avg Reward: {avg_reward:.4f}"
            )

    success_rate = success_count / max(episode_count, 1) * 100
    avg_reward = total_reward / max(episode_count, 1)

    print("=" * 60)
    print(f"Evaluation Complete ({args_cli.num_episodes} episodes)")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Average Reward: {avg_reward:.4f}")
    print(f"  Threshold: {args_cli.success_threshold} rad")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
