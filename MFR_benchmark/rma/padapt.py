# Ported from allegro_inhand_rotation/hora/algo/padapt/padapt.py
# Original RMA implementation by Haozhi Qi (2022), MIT License
# Adapted for MFR_benchmark Isaac Lab integration.
#
# Key changes from original:
#   - Uses DirectRLEnv interface instead of VecTask
#   - Observation dict format: {"policy": obs, "critic": priv, "proprio_hist": hist}
#   - Done signal: terminated | timed_out
#   - Configurable history obs dim for the adaptation module

import os
import time
import torch

from .models import ActorCritic
from .running_mean_std import RunningMeanStd


class AverageScalarMeter:
    """Tracks running average of scalar values over a fixed window."""

    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0.0

    def update(self, values):
        if values.numel() == 0:
            return
        new_mean = torch.mean(values.float()).cpu().item()
        size = min(values.numel(), self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def get_mean(self):
        return self.mean


class ProprioAdapt:
    """Stage 2 trainer: Adaptation Module via Teacher-Student Distillation.

    Loads a frozen Stage 1 (teacher) policy and trains ONLY the adaptation
    module (ProprioAdaptTConv) to predict the privileged latent from
    proprioceptive history using MSE loss.

    The actor, critic, and privileged encoder are frozen. The adaptation
    module learns to infer environment physics from recent joint history,
    enabling deployment without privileged information.

    Args:
        env: Isaac Lab DirectRLEnv instance.
        output_dir: Directory for checkpoints and tensorboard logs.
        device: Torch device string.
        network_config: Dict with keys: mlp.units, priv_mlp.units.
        ppo_config: Dict with training parameters.
        priv_info_dim: Dimension of privileged information.
        proprio_hist_len: Number of timesteps in proprioceptive history.
        adapt_obs_dim: Features per history timestep.
    """

    def __init__(
        self,
        env,
        output_dir: str,
        device: str = "cuda:0",
        network_config: dict = None,
        ppo_config: dict = None,
        priv_info_dim: int = 0,
        proprio_hist_len: int = 30,
        adapt_obs_dim: int = 24,
    ):
        self.device = device
        self.env = env

        if network_config is None:
            network_config = {
                "mlp": {"units": [512, 256, 128]},
                "priv_mlp": {"units": [256, 128, 8]},
            }
        if ppo_config is None:
            ppo_config = {}

        self.network_config = network_config
        self.ppo_config = ppo_config

        # ---- Environment info ----
        self.num_actors = env.num_envs
        obs_space = env.single_observation_space
        if hasattr(obs_space, "spaces"):
            self.obs_shape = obs_space["policy"].shape
        else:
            self.obs_shape = obs_space.shape
        action_space = env.single_action_space
        self.actions_num = action_space.shape[0]
        self.priv_info_dim = priv_info_dim
        self.proprio_hist_len = proprio_hist_len

        # ---- Model (with adaptation module enabled) ----
        net_config = {
            "actor_units": self.network_config["mlp"]["units"],
            "priv_mlp_units": self.network_config["priv_mlp"]["units"],
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "priv_info": True,
            "proprio_adapt": True,
            "priv_info_dim": self.priv_info_dim,
            "adapt_obs_dim": adapt_obs_dim,
            "adapt_history_len": proprio_hist_len,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()

        # ---- Normalization ----
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        # Separate normalization for proprioceptive history
        self.sa_mean_std = RunningMeanStd(
            (self.proprio_hist_len, adapt_obs_dim)
        ).to(self.device)

        # ---- Output ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, "stage2_nn")
        self.tb_dir = os.path.join(self.output_dir, "stage2_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        try:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(self.tb_dir)
        except ImportError:
            self.writer = None

        self.direct_info = {}

        # ---- Optimizer: only adaptation module parameters ----
        adapt_params = []
        for name, p in self.model.named_parameters():
            if "adapt_tconv" in name:
                adapt_params.append(p)
            else:
                p.requires_grad = False
        self.optim = torch.optim.Adam(adapt_params, lr=3e-4)

        # ---- Counters ----
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -10000.0
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config.get("max_agent_steps", 1500000000)
        self.epoch_num = 0

        # ---- Checkpointing ----
        self.save_freq = self.ppo_config.get("save_frequency", 500)
        self.save_best_after = self.ppo_config.get("save_best_after", 0)

        # ---- Episode tracking ----
        batch_size = self.num_actors
        self.step_reward = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.step_length = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )

    def _prepare_obs(self, obs_dict):
        """Extract observation components from DirectRLEnv output dict."""
        return {
            "obs": obs_dict["policy"],
            "priv_info": obs_dict.get("critic", None),
            "proprio_hist": obs_dict.get("proprio_hist", None),
        }

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()

    def test(self):
        """Stage 2 evaluation: NO privileged information, adaptation module only."""
        self.set_eval()
        obs_dict = self.env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        while True:
            obs = self._prepare_obs(obs_dict)
            input_dict = {
                "obs": self.running_mean_std(obs["obs"]),
                "proprio_hist": self.sa_mean_std(obs["proprio_hist"].detach()),
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            result = self.env.step(mu)
            if isinstance(result, tuple) and len(result) == 5:
                obs_dict, _r, _t, _to, _e = result
            else:
                obs_dict = result

    def train(self):
        """Stage 2 training loop: train adaptation module via MSE distillation."""
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        self.agent_steps += self.batch_size

        while self.agent_steps <= self.max_agent_steps:
            obs = self._prepare_obs(obs_dict)

            # Build input dict with normalized observations
            input_dict = {
                "obs": self.running_mean_std(obs["obs"]).detach(),
                "priv_info": obs["priv_info"],
                "proprio_hist": self.sa_mean_std(obs["proprio_hist"].detach()),
            }

            # Forward pass: get predicted latent (e) and ground truth latent (e_gt)
            mu, _, _, e, e_gt = self.model._actor_critic(input_dict)

            # MSE loss between predicted and ground truth latent
            loss = ((e - e_gt.detach()) ** 2).mean()

            # Update adaptation module only
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Step environment with the student policy's actions
            mu = mu.detach()
            mu = torch.clamp(mu, -1.0, 1.0)
            result = self.env.step(mu)
            obs_dict, rewards, terminated, timed_out, _extras = result
            dones = terminated | timed_out

            self.agent_steps += self.batch_size

            # ---- Track statistics ----
            self.step_reward += rewards
            self.step_length += 1
            done_indices = dones.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - dones.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self._log_tensorboard()

            self.epoch_num += 1

            # ---- Checkpointing ----
            mean_rewards = self.mean_eps_reward.get_mean()
            checkpoint_name = (
                f"ep_{self.epoch_num}_step_{int(self.agent_steps / 1e6):04}M"
                f"_reward_{mean_rewards:.2f}"
            )

            if self.save_freq > 0 and self.epoch_num % self.save_freq == 0:
                self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, "last"))

            if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                print(f"save current best reward: {mean_rewards:.2f}")
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, "best"))

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | "
                f"FPS: {all_fps:.1f} | Last FPS: {last_fps:.1f} | "
                f"Current Best: {self.best_rewards:.2f}"
            )
            print(f"\r{info_string}", end="")

        print("\nmax steps achieved")

    def _log_tensorboard(self):
        if self.writer is None:
            return
        self.writer.add_scalar(
            "episode_rewards/step",
            self.mean_eps_reward.get_mean(),
            self.agent_steps,
        )
        self.writer.add_scalar(
            "episode_lengths/step",
            self.mean_eps_length.get_mean(),
            self.agent_steps,
        )
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"{k}/frame", v, self.agent_steps)

    def restore_train(self, fn):
        """Load Stage 1 checkpoint, freezing all but adaptation module."""
        checkpoint = torch.load(fn, map_location=self.device)
        print("Loading Stage 1 checkpoint with strict=False (adapt_tconv is new)")
        self.model.load_state_dict(checkpoint["model"], strict=False)
        if "running_mean_std" in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def restore_test(self, fn):
        """Restore full Stage 2 checkpoint for evaluation."""
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        self.model.load_state_dict(checkpoint["model"])
        if "sa_mean_std" in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])

    def save(self, name):
        """Save Stage 2 checkpoint."""
        weights = {"model": self.model.state_dict()}
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights["sa_mean_std"] = self.sa_mean_std.state_dict()
        torch.save(weights, f"{name}.pth")
