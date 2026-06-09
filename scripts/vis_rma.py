#!/usr/bin/env python3
"""Visualize a trained RMA policy in the Isaac Lab GUI viewer."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize RMA-trained policy")
parser.add_argument("--task", type=str, default="Isaac-Allegro-Screwdriver-Turning-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--decimation", type=int, default=60,
                    help="Sim steps per RL step (lower = faster, default 60)")
parser.add_argument("--playback_fps", type=float, default=60.0,
                    help="Viewer playback FPS in simulated time. Use 30 or 60; <=0 renders once per policy step.")
parser.add_argument("--action_clip", type=float, default=1.0,
                    help="Clamp deterministic policy actions before stepping the env. RMA training uses 1.0.")
parser.add_argument("--enable_self_collisions", action="store_true",
                    help="Enable Allegro self-collisions for visualization/debugging. May be slower or harsher than training.")
parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                    help="1=teacher (uses priv info), 2=student (uses adaptation)")
parser.add_argument("--prop_hist_len", type=int, default=None)
parser.add_argument("--privileged_obs_dim", type=int, default=14)
parser.add_argument("--history_obs_dim", type=int, default=None)
parser.add_argument("--reward_mode", type=str, default=None,
                    choices=["legacy_90deg", "continuous_turn"],
                    help="Task reward mode. Default uses the task config.")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="Episode length in policy steps/seconds. Defaults to 60 for continuous_turn, otherwise task config.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force GUI mode for visualization
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import importlib
import torch
import gymnasium as gym
import MFR_benchmark.isaac_lab_tasks  # noqa

from MFR_benchmark.rma.models import ActorCritic
from MFR_benchmark.rma.running_mean_std import RunningMeanStd


def _resolve_entry_point(entry_point_str):
    module_path, class_name = entry_point_str.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    print(f"Loading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location="cuda:0")

    # Detect model params from checkpoint
    model_state = checkpoint["model"]
    saved_network_config = checkpoint.get("network_config")
    # Try to determine priv_latent_dim from env_mlp weights
    # env_mlp.mlp.4.weight shape is (latent_dim, 128)
    priv_latent_dim = None
    actor_units = None
    if saved_network_config is not None:
        actor_units = saved_network_config["mlp"]["units"]
        priv_latent_dim = saved_network_config["priv_mlp"]["units"][-1]
    for key, tensor in model_state.items():
        if "env_mlp.mlp.4.weight" in key:
            priv_latent_dim = tensor.shape[0]
        if "actor_mlp.mlp.0.weight" in key:
            actor_out = tensor.shape[0]
        if "actor_mlp.mlp.2.weight" in key:
            actor_h1 = tensor.shape[0]
        if "actor_mlp.mlp.4.weight" in key:
            actor_h2 = tensor.shape[0]

    # Fallback detection: count Linear layers in actor_mlp
    actor_layers = sorted([
        int(k.split(".")[2]) for k in model_state
        if k.startswith("actor_mlp.mlp.") and k.endswith(".weight")
    ])
    if actor_layers:
        if actor_units is None:
            actor_units = [model_state[f"actor_mlp.mlp.{i}.weight"].shape[0] for i in actor_layers]
        # input dim from first layer weight
        mlp_input = model_state[f"actor_mlp.mlp.0.weight"].shape[1]

    checkpoint_action_dim = None
    if "sigma" in model_state:
        checkpoint_action_dim = model_state["sigma"].shape[0]
    elif "mu.bias" in model_state:
        checkpoint_action_dim = model_state["mu.bias"].shape[0]

    if priv_latent_dim is None:
        priv_latent_dim = 16  # default
    if actor_units is None:
        actor_units = [1024, 512, 256, 128]

    obs_dim = mlp_input - priv_latent_dim if priv_latent_dim else 15

    print(f"  Detected priv_latent_dim={priv_latent_dim}")
    print(f"  Detected actor_units={actor_units}")
    print(f"  Detected obs_dim={obs_dim}")
    if checkpoint_action_dim is not None:
        print(f"  Detected action_dim={checkpoint_action_dim}")

    # Create environment with GUI
    env_spec = gym.spec(args_cli.task)
    env_cfg_cls = _resolve_entry_point(env_spec.kwargs["env_cfg_entry_point"])
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.decimation = args_cli.decimation
    if args_cli.playback_fps > 0:
        env_cfg.sim.render_interval = max(1, round(1.0 / (args_cli.playback_fps * env_cfg.sim.dt)))
    else:
        env_cfg.sim.render_interval = args_cli.decimation
    # Move camera closer to the hand
    env_cfg.viewer.eye = (0.3, 0.3, 1.5)
    env_cfg.viewer.lookat = (0.0, -0.05, 1.25)
    if args_cli.reward_mode is not None:
        env_cfg.reward_mode = args_cli.reward_mode
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    elif args_cli.reward_mode == "continuous_turn":
        env_cfg.episode_length_s = 60.0
    env_cfg.asymmetric_obs = True
    if hasattr(env_cfg, "action_clip"):
        env_cfg.action_clip = args_cli.action_clip
    if args_cli.enable_self_collisions:
        env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = True
    prop_hist_len = args_cli.prop_hist_len if args_cli.prop_hist_len is not None else env_cfg.prop_hist_len
    history_obs_dim = args_cli.history_obs_dim if args_cli.history_obs_dim is not None else env_cfg.history_obs_dim
    env_cfg.prop_hist_len = prop_hist_len
    env_cfg.privileged_obs_dim = args_cli.privileged_obs_dim
    env_cfg.history_obs_dim = history_obs_dim

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = env.unwrapped
    action_clip = float(args_cli.action_clip)
    env_action_dim = env.single_action_space.shape[0]
    if checkpoint_action_dim is not None and checkpoint_action_dim != env_action_dim:
        raise RuntimeError(
            f"Checkpoint action_dim={checkpoint_action_dim} does not match task '{args_cli.task}' "
            f"action_dim={env_action_dim}. A 16-action checkpoint is a LinkerHand policy; use "
            f"--task Isaac-LinkerHand-Screwdriver-Continuous-Turning-Direct-v0. A 12-action "
            f"checkpoint is an Allegro policy; use --task Isaac-Allegro-Screwdriver-Continuous-Turning-Direct-v0."
        )

    # Build model to match checkpoint architecture
    use_adapt = (args_cli.stage == 2)
    net_config = {
        "actor_units": actor_units,
        "priv_mlp_units": [256, 128, priv_latent_dim],
        "actions_num": env_action_dim,
        "input_shape": (obs_dim,),
        "priv_info": True,
        "proprio_adapt": use_adapt,
        "priv_info_dim": args_cli.privileged_obs_dim,
        "adapt_obs_dim": history_obs_dim,
        "adapt_history_len": prop_hist_len,
    }

    model = ActorCritic(net_config)
    model.to("cuda:0")
    model.eval()
    model.load_state_dict(checkpoint["model"], strict=False)

    running_mean_std = RunningMeanStd((obs_dim,)).to("cuda:0")
    running_mean_std.eval()
    if "running_mean_std" in checkpoint:
        running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    sa_mean_std = None
    if use_adapt and "sa_mean_std" in checkpoint:
        sa_mean_std = RunningMeanStd((prop_hist_len, history_obs_dim)).to("cuda:0")
        sa_mean_std.eval()
        sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])

    print("Starting visualization. Close the viewer window to exit.")

    obs_dict = env.reset()
    if isinstance(obs_dict, tuple):
        obs_dict = obs_dict[0]

    step = 0
    while simulation_app.is_running():
        obs = obs_dict["policy"]
        input_dict = {"obs": running_mean_std(obs)}

        if args_cli.stage == 1:
            input_dict["priv_info"] = obs_dict.get("critic", None)
        elif args_cli.stage == 2 and sa_mean_std is not None:
            input_dict["proprio_hist"] = sa_mean_std(obs_dict["proprio_hist"].detach())

        mu = model.act_inference(input_dict)
        if action_clip > 0.0:
            mu = torch.clamp(mu, -action_clip, action_clip)

        result = env.step(mu)
        obs_dict, rewards, terminated, timed_out, extras = result

        if step % 60 == 0:
            reward_mean = rewards.mean().item()
            euler = extras.get("eval_screwdriver_euler")
            if euler is not None:
                fwd_turns = extras.get("eval_total_turns")
                net_turns = extras.get("eval_net_turns")
                turn_velocity = extras.get("eval_turn_velocity")
                metric_text = ""
                if fwd_turns is not None:
                    metric_text += f"  fwd_turns={fwd_turns[0].item():6.2f}"
                if net_turns is not None:
                    metric_text += f"  net_turns={net_turns[0].item():6.2f}"
                if turn_velocity is not None:
                    metric_text += f"  turn_vel={turn_velocity[0].item():6.2f}"
                print(f"  step={step:4d}  reward={reward_mean:8.2f}{metric_text}  "
                      f"euler=[{euler[0,0].item():6.2f}, {euler[0,1].item():6.2f}, {euler[0,2].item():6.2f}]", end="\r")

        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
