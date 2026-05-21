from __future__ import annotations

"""Finite RL-Games checkpoint evaluator for the Isaac Lab screwdriver task."""

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate an RL-Games checkpoint on the screwdriver turning task.")
parser.add_argument("--task", type=str, default="Isaac-Allegro-Screwdriver-Turning-Direct-v0", help="Gym task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to an RL-Games .pth checkpoint.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of vectorized environments.")
parser.add_argument("--num_episodes", type=int, default=128, help="Number of completed episodes to collect.")
parser.add_argument("--max_steps", type=int, default=None, help="Optional hard cap on policy steps.")
parser.add_argument("--seed", type=int, default=None, help="Evaluation seed. Use -1 for a random seed.")
parser.add_argument("--stochastic", action="store_true", help="Sample stochastic actions instead of deterministic ones.")
parser.add_argument("--print_every", type=int, default=25, help="Progress print interval in policy steps. Use 0 to disable.")
parser.add_argument("--csv", type=str, default=None, help="Optional per-episode CSV output path.")
parser.add_argument(
    "--success_thresholds",
    type=float,
    nargs="*",
    default=(0.25, 0.5, 1.0),
    help="Absolute final z-angle error thresholds, in radians, used for success-rate summaries.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle stepping to policy dt.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="RL agent config registry key.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import MFR_benchmark.isaac_lab_tasks  # noqa: F401


def _as_float_list(values: list[float]) -> list[float]:
    return [float(value) for value in values]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def _format_summary(name: str, values: list[float], unit: str = "") -> str:
    stats = _summary(values)
    suffix = f" {unit}" if unit else ""
    return (
        f"{name}: mean={stats['mean']:.6g}{suffix}, std={stats['std']:.6g}{suffix}, "
        f"min={stats['min']:.6g}{suffix}, max={stats['max']:.6g}{suffix}"
    )


def _write_csv(path: str, rows: list[dict[str, float]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "env_id",
        "return",
        "length",
        "final_x",
        "final_y",
        "final_z",
        "goal_error_x",
        "goal_error_y",
        "goal_error_z",
        "abs_goal_error_z",
        "upright_norm",
        "action_cost",
        "goal_cost",
        "upright_cost",
    ]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    print(f"[INFO]: Evaluating checkpoint: {resume_path}")
    print(f"[INFO]: num_envs={env.unwrapped.num_envs}, target_episodes={args_cli.num_episodes}")
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    episode_returns = torch.zeros(env.unwrapped.num_envs, dtype=torch.float32, device=env.unwrapped.device)
    episode_lengths = torch.zeros(env.unwrapped.num_envs, dtype=torch.int32, device=env.unwrapped.device)
    rows: list[dict[str, float]] = []
    completed = 0
    steps = 0
    start_time = time.time()

    while simulation_app.is_running() and completed < args_cli.num_episodes:
        if args_cli.max_steps is not None and steps >= args_cli.max_steps:
            break

        step_start = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=not args_cli.stochastic)
            obs, rewards, dones, extras = env.step(actions)

        rewards_sim = rewards.to(device=episode_returns.device)
        dones_sim = dones.to(device=episode_returns.device)
        episode_returns += rewards_sim
        episode_lengths += 1
        steps += 1

        done_ids = dones_sim.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            euler = extras["eval_screwdriver_euler"].to(device=episode_returns.device)
            goal_error = extras["eval_screwdriver_goal_error"].to(device=episode_returns.device)
            upright_norm = extras["eval_screwdriver_upright_norm"].to(device=episode_returns.device)
            action_cost = extras["eval_action_cost"].to(device=episode_returns.device)
            goal_cost = extras["eval_goal_cost"].to(device=episode_returns.device)
            upright_cost = extras["eval_upright_cost"].to(device=episode_returns.device)

            for env_id_tensor in done_ids:
                env_id = int(env_id_tensor.item())
                row = {
                    "episode": completed,
                    "env_id": env_id,
                    "return": float(episode_returns[env_id].item()),
                    "length": int(episode_lengths[env_id].item()),
                    "final_x": float(euler[env_id, 0].item()),
                    "final_y": float(euler[env_id, 1].item()),
                    "final_z": float(euler[env_id, 2].item()),
                    "goal_error_x": float(goal_error[env_id, 0].item()),
                    "goal_error_y": float(goal_error[env_id, 1].item()),
                    "goal_error_z": float(goal_error[env_id, 2].item()),
                    "abs_goal_error_z": float(torch.abs(goal_error[env_id, 2]).item()),
                    "upright_norm": float(upright_norm[env_id].item()),
                    "action_cost": float(action_cost[env_id].item()),
                    "goal_cost": float(goal_cost[env_id].item()),
                    "upright_cost": float(upright_cost[env_id].item()),
                }
                rows.append(row)
                completed += 1
                if completed >= args_cli.num_episodes:
                    break

            episode_returns[done_ids] = 0.0
            episode_lengths[done_ids] = 0

            if agent.is_rnn and agent.states is not None:
                done_ids_rl = done_ids.to(device=agent.states[0].device)
                for state in agent.states:
                    state[:, done_ids_rl, :] = 0.0

        if args_cli.print_every > 0 and steps % args_cli.print_every == 0:
            elapsed = max(time.time() - start_time, 1.0e-6)
            print(f"[EVAL]: steps={steps}, completed={completed}/{args_cli.num_episodes}, fps={steps / elapsed:.2f}")

        sleep_time = env.unwrapped.step_dt - (time.time() - step_start)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

    if not rows:
        print("[EVAL]: No completed episodes were collected.")
        return

    rows = rows[: args_cli.num_episodes]
    elapsed = time.time() - start_time
    returns = _as_float_list([row["return"] for row in rows])
    lengths = _as_float_list([row["length"] for row in rows])
    final_z = _as_float_list([row["final_z"] for row in rows])
    abs_z_error = _as_float_list([row["abs_goal_error_z"] for row in rows])
    upright_norm = _as_float_list([row["upright_norm"] for row in rows])
    goal_cost = _as_float_list([row["goal_cost"] for row in rows])
    upright_cost = _as_float_list([row["upright_cost"] for row in rows])

    print("\n[EVAL SUMMARY]")
    print(f"checkpoint: {resume_path}")
    print(f"episodes: {len(rows)}")
    print(f"policy_steps: {steps}")
    print(f"elapsed_s: {elapsed:.3f}")
    print(_format_summary("return", returns))
    print(_format_summary("episode_length", lengths, "steps"))
    print(_format_summary("final_z_angle", final_z, "rad"))
    print(_format_summary("abs_final_z_goal_error", abs_z_error, "rad"))
    print(_format_summary("upright_norm_xy", upright_norm, "rad"))
    print(_format_summary("final_goal_cost", goal_cost))
    print(_format_summary("final_upright_cost", upright_cost))
    for threshold in args_cli.success_thresholds:
        successes = sum(error <= threshold for error in abs_z_error)
        print(f"success@{threshold:g}rad: {successes / len(rows):.3f} ({successes}/{len(rows)})")

    if args_cli.csv:
        _write_csv(args_cli.csv, rows)
        print(f"csv: {args_cli.csv}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
