# Ported from allegro_inhand_rotation/hora/algo/ppo/ppo.py
# Originally based on RLGames (Denys88, MIT License)
# Adapted for MFR_benchmark Isaac Lab integration.
#
# Key changes from original:
#   - Uses DirectRLEnv interface instead of VecTask
#   - Observation dict format: {"policy": obs, "critic": priv, "proprio_hist": hist}
#   - Done signal: terminated | timed_out (instead of single dones tensor)
#   - Config via dataclass/dict instead of Hydra OmegaConf
#   - No wandb dependency (uses tensorboard only)

import os
import time
import torch
import torch.nn as nn

from .experience import ExperienceBuffer
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

    def clear(self):
        self.current_size = 0
        self.mean = 0.0

    def get_mean(self):
        return self.mean


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    """Compute KL divergence between two Gaussian policies."""
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)
    return kl.mean()


class AdaptiveScheduler:
    """Adaptive learning rate scheduler based on KL divergence.

    Increases LR when KL is low (policy changed too little),
    decreases LR when KL is high (policy changed too much).
    """

    def __init__(self, kl_threshold=0.008):
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


class PPO:
    """PPO trainer for Stage 1 (Teacher Policy with privileged information).

    Trains an ActorCritic network using PPO-Clip on an Isaac Lab DirectRLEnv.
    The teacher policy has access to privileged environment information
    (object dynamics, poses, contacts) concatenated with proprioceptive observations.

    Args:
        env: Isaac Lab DirectRLEnv instance.
        output_dir: Directory for checkpoints and tensorboard logs.
        device: Torch device string (e.g., 'cuda:0').
        network_config: Dict with keys: mlp.units, priv_mlp.units.
        ppo_config: Dict with PPO hyperparameters.
        priv_info: Whether to use privileged information encoder.
        proprio_adapt: Whether to create adaptation module (False for Stage 1).
        priv_info_dim: Dimension of privileged information.
        adapt_obs_dim: Features per history timestep for adaptation module.
        adapt_history_len: Number of history timesteps for adaptation module.
    """

    def __init__(
        self,
        env,
        output_dir: str,
        device: str = "cuda:0",
        network_config: dict = None,
        ppo_config: dict = None,
        priv_info: bool = True,
        proprio_adapt: bool = False,
        priv_info_dim: int = 0,
        adapt_obs_dim: int = 24,
        adapt_history_len: int = 30,
    ):
        self.device = device
        self.env = env

        # ---- Default configs ----
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
        action_space = env.single_action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.tensor(
            action_space.low, dtype=torch.float32, device=self.device
        )
        self.actions_high = torch.tensor(
            action_space.high, dtype=torch.float32, device=self.device
        )
        # Handle both Dict and flat Box observation spaces
        obs_space = env.single_observation_space
        if hasattr(obs_space, "spaces"):  # gym.spaces.Dict
            self.obs_shape = obs_space["policy"].shape
        else:
            self.obs_shape = obs_space.shape

        # ---- Privileged info ----
        self.priv_info_dim = priv_info_dim
        self.priv_info = priv_info
        self.proprio_adapt = proprio_adapt

        # ---- Model ----
        net_config = {
            "actor_units": self.network_config["mlp"]["units"],
            "priv_mlp_units": self.network_config["priv_mlp"]["units"],
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "priv_info": self.priv_info,
            "proprio_adapt": self.proprio_adapt,
            "priv_info_dim": self.priv_info_dim,
            "adapt_obs_dim": adapt_obs_dim,
            "adapt_history_len": adapt_history_len,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)

        # ---- Output ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, "stage1_nn")
        self.tb_dir = os.path.join(self.output_dir, "stage1_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        try:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(self.tb_dir)
        except ImportError:
            self.writer = None

        # ---- Optimizer ----
        self.last_lr = float(self.ppo_config.get("learning_rate", 5e-3))
        self.weight_decay = float(self.ppo_config.get("weight_decay", 0.0))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.last_lr, weight_decay=self.weight_decay
        )

        # ---- PPO hyperparams ----
        self.e_clip = self.ppo_config.get("e_clip", 0.2)
        self.clip_value = self.ppo_config.get("clip_value", True)
        self.entropy_coef = self.ppo_config.get("entropy_coef", 0.0)
        self.critic_coef = self.ppo_config.get("critic_coef", 4.0)
        self.bounds_loss_coef = self.ppo_config.get("bounds_loss_coef", 0.0001)
        self.gamma = self.ppo_config.get("gamma", 0.99)
        self.tau = self.ppo_config.get("tau", 0.95)
        self.truncate_grads = self.ppo_config.get("truncate_grads", True)
        self.grad_norm = self.ppo_config.get("grad_norm", 1.0)
        self.value_bootstrap = self.ppo_config.get("value_bootstrap", True)
        self.normalize_advantage = self.ppo_config.get("normalize_advantage", True)
        self.normalize_input = self.ppo_config.get("normalize_input", True)
        self.normalize_value = self.ppo_config.get("normalize_value", True)

        # ---- Collection params ----
        self.horizon_length = self.ppo_config.get("horizon_length", 8)
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config.get("minibatch_size", 32768)
        self.mini_epochs_num = self.ppo_config.get("mini_epochs", 5)

        # ---- Scheduler ----
        self.kl_threshold = self.ppo_config.get("kl_threshold", 0.02)
        self.scheduler = AdaptiveScheduler(self.kl_threshold)

        # ---- Checkpointing ----
        self.save_freq = self.ppo_config.get("save_frequency", 200)
        self.save_best_after = self.ppo_config.get("save_best_after", 100)

        # ---- Counters ----
        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = 0
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config.get("max_agent_steps", 1500000000)
        self.best_rewards = -10000.0
        self.env_metric_keys = tuple(
            self.ppo_config.get(
                "env_metric_keys",
                (
                    "eval_total_turns",
                    "eval_net_turns",
                    "eval_turn_velocity",
                    "eval_forward_turn_velocity",
                    "eval_reverse_turn_velocity",
                    "eval_screwdriver_upright_norm",
                    "eval_turn_reward",
                    "eval_reverse_cost",
                    "eval_upright_cost",
                    "eval_action_cost",
                    "eval_action_rate_cost",
                    "eval_goal_cost",
                ),
            )
        )
        self.env_metric_aliases = {
            "eval_total_turns": "FwdTurns",
            "eval_net_turns": "NetTurns",
            "eval_turn_velocity": "TurnVel",
            "eval_forward_turn_velocity": "FwdVel",
            "eval_reverse_turn_velocity": "RevVel",
            "eval_screwdriver_upright_norm": "Upright",
            "eval_turn_reward": "TurnRew",
            "eval_reverse_cost": "RevCost",
            "eval_upright_cost": "UprightCost",
            "eval_action_cost": "ActionCost",
            "eval_action_rate_cost": "ActionRate",
            "eval_goal_cost": "GoalCost",
        }
        self.last_env_metrics = {}
        self.curriculum_config = self.ppo_config.get("curriculum", {})
        self.curriculum_enabled = False
        self.curriculum_phases = []
        self.curriculum_phase_idx = -1
        self.curriculum_phase_name = "none"
        self.curriculum_phase_start_steps = 0
        self.curriculum_last_wait_reason = ""
        self._init_curriculum()

        # ---- Experience buffer ----
        self.storage = ExperienceBuffer(
            self.num_actors,
            self.horizon_length,
            self.batch_size,
            self.minibatch_size,
            self.obs_shape[0],
            self.actions_num,
            self.priv_info_dim,
            self.device,
        )

        # ---- Episode tracking ----
        batch_size = self.num_actors
        self.current_rewards = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones(
            (batch_size,), dtype=torch.uint8, device=self.device
        )

        # ---- Timing ----
        self.data_collect_time = 0.0
        self.rl_train_time = 0.0

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        """Run policy to collect actions during rollout."""
        processed_obs = self.running_mean_std(obs_dict["obs"])
        input_dict = {
            "obs": processed_obs,
            "priv_info": obs_dict.get("priv_info", None),
        }
        res_dict = self.model.act(input_dict)
        res_dict["values"] = self.value_mean_std(res_dict["values"], True)
        return res_dict

    def _prepare_obs(self, obs_dict):
        """Extract observation components from DirectRLEnv output dict.

        The environment returns {"policy": obs, "critic": priv, "proprio_hist": hist}.
        We remap this to the format expected by the RMA model.
        """
        result = {
            "obs": obs_dict["policy"],
            "priv_info": obs_dict.get("critic", None),
            "proprio_hist": obs_dict.get("proprio_hist", None),
        }
        return result

    def _accumulate_env_metrics(self, extras, metric_sums, metric_counts):
        if not isinstance(extras, dict):
            return
        for key in self.env_metric_keys:
            value = extras.get(key, None)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                scalar = torch.nan_to_num(value.float()).mean().detach().cpu().item()
            else:
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
            metric_sums[key] = metric_sums.get(key, 0.0) + scalar
            metric_counts[key] = metric_counts.get(key, 0) + 1

    def _format_env_metrics(self):
        if not self.last_env_metrics:
            return ""
        parts = []
        for key in self.env_metric_keys:
            if key not in self.last_env_metrics:
                continue
            alias = self.env_metric_aliases.get(key, key.replace("eval_", ""))
            parts.append(f"{alias}: {self.last_env_metrics[key]:.3f}")
        if not parts:
            return ""
        return " | " + " | ".join(parts)

    def _init_curriculum(self):
        self.curriculum_enabled = bool(self.curriculum_config.get("enabled", False))
        self.curriculum_phases = list(self.curriculum_config.get("phases", ()))
        if not self.curriculum_enabled:
            return
        if not self.curriculum_phases:
            print("Continuous curriculum requested, but no phases were provided.")
            self.curriculum_enabled = False
            return
        self._set_curriculum_phase(0, initial=True)

    def _set_curriculum_phase(self, phase_idx, initial=False):
        phase = self.curriculum_phases[phase_idx]
        name = phase.get("name", f"phase{phase_idx + 1}")
        overrides = phase.get("overrides", {})
        applied = {}
        cfg = getattr(self.env, "cfg", None)
        if cfg is not None:
            for key, value in overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
                    applied[key] = value

        self.curriculum_phase_idx = phase_idx
        self.curriculum_phase_name = name
        self.curriculum_phase_start_steps = self.agent_steps
        self.curriculum_last_wait_reason = ""

        prefix = "Initial curriculum phase" if initial else "Curriculum phase"
        if applied:
            applied_text = ", ".join(f"{key}={value}" for key, value in sorted(applied.items()))
            print(f"{prefix}: {phase_idx + 1}/{len(self.curriculum_phases)} {name} ({applied_text})")
        else:
            print(f"{prefix}: {phase_idx + 1}/{len(self.curriculum_phases)} {name}")

        if self.writer:
            self.writer.add_scalar("curriculum/phase_index", phase_idx, self.agent_steps)

    def _format_curriculum(self):
        if not self.curriculum_enabled:
            return ""
        return f"Curr: {self.curriculum_phase_name} | "

    def _format_curriculum_gate(self):
        if not self.curriculum_enabled:
            return ""
        if self.curriculum_phase_idx >= len(self.curriculum_phases) - 1:
            return ""
        if not self.curriculum_last_wait_reason:
            return ""
        return f" | Gate: {self.curriculum_last_wait_reason}"

    def _metric_or_none(self, key):
        value = self.last_env_metrics.get(key, None)
        if value is None:
            return None
        return float(value)

    def _curriculum_can_advance(self, mean_lengths):
        if not self.curriculum_enabled:
            return False, "disabled"
        if self.curriculum_phase_idx < 0:
            return False, "not initialized"
        if self.curriculum_phase_idx >= len(self.curriculum_phases) - 1:
            return False, "final phase"

        phase = self.curriculum_phases[self.curriculum_phase_idx]
        advance = phase.get("advance", {})
        if not advance:
            return False, "no advance rule"

        phase_steps = self.agent_steps - self.curriculum_phase_start_steps
        min_phase_steps = float(advance.get("min_phase_steps", 0.0))
        if phase_steps < min_phase_steps:
            return False, f"phase_steps {phase_steps:.0f} < {min_phase_steps:.0f}"

        min_length = advance.get("min_episode_length", None)
        if min_length is not None and mean_lengths < float(min_length):
            return False, f"Len {mean_lengths:.1f} < {float(min_length):.1f}"

        checks = (
            ("eval_total_turns", "min_total_turns", ">="),
            ("eval_net_turns", "min_net_turns", ">="),
            ("eval_forward_turn_velocity", "min_forward_velocity", ">="),
            ("eval_screwdriver_upright_norm", "max_upright", "<="),
        )
        for metric_key, rule_key, op in checks:
            target = advance.get(rule_key, None)
            if target is None:
                continue
            value = self._metric_or_none(metric_key)
            if value is None:
                return False, f"missing {metric_key}"
            target = float(target)
            if op == ">=" and value < target:
                return False, f"{metric_key} {value:.3f} < {target:.3f}"
            if op == "<=" and value > target:
                return False, f"{metric_key} {value:.3f} > {target:.3f}"

        min_fwd_minus_rev = advance.get("min_fwd_minus_rev", None)
        if min_fwd_minus_rev is not None:
            fwd = self._metric_or_none("eval_forward_turn_velocity")
            rev = self._metric_or_none("eval_reverse_turn_velocity")
            if fwd is None or rev is None:
                return False, "missing forward/reverse velocity"
            margin = fwd - rev
            if margin < float(min_fwd_minus_rev):
                return False, f"FwdVel-RevVel {margin:.3f} < {float(min_fwd_minus_rev):.3f}"

        return True, "ready"

    def _maybe_update_curriculum(self, mean_lengths):
        can_advance, reason = self._curriculum_can_advance(mean_lengths)
        self.curriculum_last_wait_reason = reason
        if not can_advance:
            return None

        old_idx = self.curriculum_phase_idx
        old_name = self.curriculum_phase_name
        if self.curriculum_config.get("save_on_phase_change", True):
            safe_name = old_name.replace("/", "_").replace(" ", "_")
            self.save(os.path.join(self.nn_dir, f"curriculum_exit_{old_idx + 1}_{safe_name}"))

        self._set_curriculum_phase(old_idx + 1)
        if self.curriculum_config.get("reset_best_on_phase_change", True):
            self.best_rewards = -10000.0

        return (
            f"Curriculum advanced: {old_name} -> {self.curriculum_phase_name} "
            f"at {int(self.agent_steps // 1e6):04}M steps"
        )

    def train(self):
        """Main training loop for Stage 1."""
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        # env.reset() returns (obs_dict, extras) tuple in newer Isaac Lab
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        self.obs = self._prepare_obs(obs_dict)
        self.agent_steps = self.batch_size

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            self.storage.data_dict = None

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            curriculum_event = self._maybe_update_curriculum(mean_lengths)
            if curriculum_event:
                print(curriculum_event)
            metric_string = self._format_env_metrics()
            curriculum_string = self._format_curriculum()
            gate_string = self._format_curriculum_gate()
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | "
                f"FPS: {all_fps:.1f} | Last FPS: {last_fps:.1f} | "
                f"Collect: {self.data_collect_time / 60:.1f} min | "
                f"Train: {self.rl_train_time / 60:.1f} min | "
                f"{curriculum_string}"
                f"Reward: {mean_rewards:.2f} | Len: {mean_lengths:.1f} | "
                f"Best: {self.best_rewards:.2f}{metric_string}{gate_string}"
            )
            print(info_string)

            self._write_stats(a_losses, c_losses, b_losses, entropies, kls)

            if self.writer:
                self.writer.add_scalar(
                    "episode_rewards/step", mean_rewards, self.agent_steps
                )
                self.writer.add_scalar(
                    "episode_lengths/step", mean_lengths, self.agent_steps
                )
                for key, value in self.last_env_metrics.items():
                    self.writer.add_scalar(f"env/{key}", value, self.agent_steps)
                if self.curriculum_enabled:
                    self.writer.add_scalar(
                        "curriculum/phase_index",
                        self.curriculum_phase_idx,
                        self.agent_steps,
                    )

            checkpoint_name = (
                f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M"
                f"_reward_{mean_rewards:.2f}"
            )

            if self.save_freq > 0 and self.epoch_num % self.save_freq == 0:
                self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, "last"))

            if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                print(f"save current best reward: {mean_rewards:.2f}")
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, "best"))

        print("max steps achieved")

    def save(self, name):
        """Save model checkpoint."""
        weights = {"model": self.model.state_dict()}
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights["value_mean_std"] = self.value_mean_std.state_dict()
        torch.save(weights, f"{name}.pth")

    def restore_train(self, fn):
        """Restore model from checkpoint (for resuming or Stage 2 init)."""
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        if "running_mean_std" in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if "value_mean_std" in checkpoint:
            self.value_mean_std.load_state_dict(checkpoint["value_mean_std"])

    def restore_test(self, fn):
        """Restore model for testing/evaluation."""
        checkpoint = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def test(self):
        """Run evaluation with privileged information (Stage 1 evaluation)."""
        self.set_eval()
        obs_dict = self.env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        while True:
            obs = self._prepare_obs(obs_dict)
            input_dict = {
                "obs": self.running_mean_std(obs["obs"]),
                "priv_info": obs["priv_info"],
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            result = self.env.step(mu)
            if isinstance(result, tuple) and len(result) == 5:
                obs_dict, _rewards, _terminated, _timed_out, _extras = result
            else:
                obs_dict = result

    def train_epoch(self):
        """One epoch: collect rollout, then PPO updates."""
        # Collect rollout
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += time.time() - _t

        # PPO updates
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []

        for _ in range(self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                (
                    value_preds,
                    old_action_log_probs,
                    advantage,
                    old_mu,
                    old_sigma,
                    returns,
                    actions,
                    obs,
                    priv_info,
                ) = self.storage[i]

                obs = self.running_mean_std(obs)
                batch_dict = {
                    "prev_actions": actions,
                    "obs": obs,
                    "priv_info": priv_info,
                }
                res_dict = self.model(batch_dict)

                action_log_probs = res_dict["prev_neglogp"]
                values = res_dict["values"]
                entropy = res_dict["entropy"]
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]

                # --- PPO-Clip actor loss ---
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(
                    ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
                )
                a_loss = torch.max(-surr1, -surr2)

                # --- Clipped critic loss ---
                value_pred_clipped = value_preds + (values - value_preds).clamp(
                    -self.e_clip, self.e_clip
                )
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)

                # --- Bounds loss ---
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0

                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss)
                    for loss in [a_loss, c_loss, entropy, b_loss]
                ]

                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm
                    )
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(
                        mu.detach(), sigma.detach(), old_mu, old_sigma
                    )

                a_losses.append(a_loss)
                c_losses.append(c_loss)
                entropies.append(entropy)
                if self.bounds_loss_coef > 0:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())
                ep_kls.append(kl_dist)

            av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.last_lr
            kls.append(av_kls)

        self.rl_train_time += time.time() - _t
        return a_losses, c_losses, b_losses, entropies, kls

    def play_steps(self):
        """Collect rollout data by interacting with the environment."""
        metric_sums = {}
        metric_counts = {}
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)

            # Store transition data
            self.storage.update_data("obses", n, self.obs["obs"])
            self.storage.update_data("priv_info", n, self.obs["priv_info"])
            for k in ["actions", "neglogpacs", "values", "mus", "sigmas"]:
                self.storage.update_data(k, n, res_dict[k])

            # Step the environment
            actions = torch.clamp(res_dict["actions"], -1.0, 1.0)
            result = self.env.step(actions)
            # DirectRLEnv.step returns (obs_dict, rewards, terminated, timed_out, extras)
            obs_dict, rewards, terminated, timed_out, extras = result
            self._accumulate_env_metrics(extras, metric_sums, metric_counts)

            # Compose done signal
            dones = terminated | timed_out
            rewards = rewards.unsqueeze(1)

            self.storage.update_data("dones", n, dones.to(torch.uint8))
            shaped_rewards = 0.01 * rewards.clone()

            # Value bootstrap for timeout terminations
            if self.value_bootstrap:
                # timed_out = True means the episode ended due to timeout (not failure)
                shaped_rewards += (
                    self.gamma
                    * res_dict["values"]
                    * timed_out.unsqueeze(1).float()
                )
            self.storage.update_data("rewards", n, shaped_rewards)

            # Track episode stats
            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            # Prepare next observation
            self.obs = self._prepare_obs(obs_dict)

        if metric_sums:
            self.last_env_metrics = {
                key: metric_sums[key] / max(metric_counts[key], 1)
                for key in metric_sums
            }

        # Final value for GAE
        res_dict = self.model_act(self.obs)
        last_values = res_dict["values"]

        self.agent_steps += self.batch_size
        self.storage.computer_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        # Normalize values and returns
        returns = self.storage.data_dict["returns"]
        values = self.storage.data_dict["values"]
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict["values"] = values
        self.storage.data_dict["returns"] = returns

    def _write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        if self.writer is None:
            return
        self.writer.add_scalar(
            "performance/RLTrainFPS",
            self.agent_steps / max(self.rl_train_time, 1e-6),
            self.agent_steps,
        )
        self.writer.add_scalar(
            "performance/EnvStepFPS",
            self.agent_steps / max(self.data_collect_time, 1e-6),
            self.agent_steps,
        )
        if a_losses:
            self.writer.add_scalar(
                "losses/actor_loss",
                torch.mean(torch.stack(a_losses)).item(),
                self.agent_steps,
            )
        if c_losses:
            self.writer.add_scalar(
                "losses/critic_loss",
                torch.mean(torch.stack(c_losses)).item(),
                self.agent_steps,
            )
        if b_losses:
            self.writer.add_scalar(
                "losses/bounds_loss",
                torch.mean(torch.stack(b_losses)).item(),
                self.agent_steps,
            )
        if entropies:
            self.writer.add_scalar(
                "losses/entropy",
                torch.mean(torch.stack(entropies)).item(),
                self.agent_steps,
            )
        if kls:
            self.writer.add_scalar(
                "info/kl", torch.mean(torch.stack(kls)).item(), self.agent_steps
            )
        self.writer.add_scalar("info/last_lr", self.last_lr, self.agent_steps)
