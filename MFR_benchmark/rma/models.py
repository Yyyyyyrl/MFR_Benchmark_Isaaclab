# Ported from allegro_inhand_rotation/hora/algo/models/models.py
# Original RMA implementation by Haozhi Qi (2022), MIT License
# Adapted for MFR_benchmark Isaac Lab integration.
#
# Key changes from original:
#   - ProprioAdaptTConv is configurable (obs_dim, hidden_dim, latent_dim, history_len)
#   - ActorCritic accepts observation dim explicitly (not hardcoded to 96)
#   - Privileged info dim is configurable per task
#   - No dependency on IsaacGym — pure PyTorch module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with ELU activations.

    Args:
        units: List of output sizes for each hidden layer.
        input_size: Dimension of the input.
    """

    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ProprioAdaptTConv(nn.Module):
    """Proprioceptive History Encoder (Adaptation Module).

    Processes a window of proprioceptive history through per-frame MLP
    followed by temporal 1D convolutions to produce an environment
    latent vector. This is the "student" module trained in Stage 2.

    Architecture:
        Input: (N, history_len, obs_dim)
          -> per-frame MLP: (obs_dim -> hidden_dim -> hidden_dim)
          -> permute to (N, hidden_dim, history_len)
          -> 3-layer Conv1D (kernel 9/5/5, stride 2/1/1)
          -> flatten -> Linear -> latent_dim

    The convolution stack is designed so that the temporal dimension
    reduces to exactly 3 at the output, giving hidden_dim * 3 features
    before the final projection.

    Args:
        obs_dim: Features per proprioceptive timestep (default 32).
        hidden_dim: Internal channel dimension (default 32).
        latent_dim: Output environment latent dimension (default 8).
        history_len: Number of timesteps in history window (default 30).
        conv_kernels: Tuple of (kernel_size, stride) for each conv layer.
    """

    def __init__(
        self,
        obs_dim: int = 32,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        history_len: int = 30,
        conv_kernels: tuple = None,
    ):
        super(ProprioAdaptTConv, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.history_len = history_len

        # Per-frame feature extraction
        self.channel_transform = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Temporal aggregation via 1D convolutions
        if conv_kernels is None:
            # Default: designed for history_len=30, produces temporal dim = 3
            # 30 -> Conv1d(k9,s2) -> 11 -> Conv1d(k5,s1) -> 7 -> Conv1d(k5,s1) -> 3
            conv_kernels = (
                (9, 2),
                (5, 1),
                (5, 1),
            )

        conv_layers = []
        in_channels = hidden_dim
        for kernel_size, stride in conv_kernels:
            conv_layers.append(
                nn.Conv1d(in_channels, hidden_dim, kernel_size, stride=stride)
            )
            conv_layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_dim

        self.temporal_aggregation = nn.Sequential(*conv_layers)

        # Compute output temporal dimension after all convs
        temp_len = history_len
        for kernel_size, stride in conv_kernels:
            temp_len = (temp_len - kernel_size) // stride + 1
        self._conv_output_len = temp_len

        # Final projection to latent
        self.low_dim_proj = nn.Linear(hidden_dim * self._conv_output_len, latent_dim)

    def forward(self, x):
        # x: (N, history_len, obs_dim)
        x = self.channel_transform(x)  # (N, history_len, hidden_dim)
        x = x.permute((0, 2, 1))  # (N, hidden_dim, history_len)
        x = self.temporal_aggregation(x)  # (N, hidden_dim, conv_output_len)
        x = self.low_dim_proj(x.flatten(1))  # (N, latent_dim)
        return x


class ActorCritic(nn.Module):
    """Actor-Critic network with optional privileged encoder and adaptation module.

    Supports three modes:
      - Standard (priv_info=False, proprio_adapt=False):
          obs -> actor_mlp -> mu, value
      - Stage 1 / Teacher (priv_info=True, proprio_adapt=False):
          priv_info -> env_mlp -> tanh -> concat(obs, latent) -> actor_mlp -> mu, value
      - Stage 2 / Student (priv_info=True, proprio_adapt=True):
          proprio_hist -> adapt_tconv -> tanh -> concat(obs, latent) -> actor_mlp -> mu, value
          (priv_info -> env_mlp -> tanh -> provides ground truth for MSE loss)

    Args:
        actions_num: Number of action dimensions.
        input_shape: Tuple of observation dimensions, e.g. (obs_dim,).
        actor_units: List of hidden sizes for the actor-critic MLP.
        priv_mlp_units: List of hidden sizes for the privileged encoder MLP.
        priv_info: Whether to use privileged information.
        proprio_adapt: Whether to use the proprioceptive adaptation module.
        priv_info_dim: Dimension of privileged information vector.
        adapt_obs_dim: Features per timestep for the adaptation module.
        adapt_hidden_dim: Hidden dimension for the adaptation module.
        adapt_latent_dim: Output latent dimension for the adaptation module.
        adapt_history_len: Number of history timesteps for the adaptation module.
        adapt_conv_kernels: Conv kernel config for the adaptation module.
    """

    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop("actions_num")
        input_shape = kwargs.pop("input_shape")
        self.units = kwargs.pop("actor_units")
        self.priv_mlp = kwargs.pop("priv_mlp_units")
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]
        self.priv_info = kwargs["priv_info"]
        self.priv_info_stage2 = kwargs["proprio_adapt"]

        # --- Adaptation module config (configurable for different tasks) ---
        adapt_obs_dim = kwargs.get("adapt_obs_dim", 32)
        adapt_hidden_dim = kwargs.get("adapt_hidden_dim", 32)
        adapt_latent_dim = kwargs.get("adapt_latent_dim", 8)
        adapt_history_len = kwargs.get("adapt_history_len", 30)
        adapt_conv_kernels = kwargs.get("adapt_conv_kernels", None)

        # --- Encoder Definition ---
        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1]
            # 1. Privileged Information Encoder
            #    Encodes ground-truth environment physics into a compact latent.
            self.env_mlp = MLP(
                units=self.priv_mlp, input_size=kwargs["priv_info_dim"]
            )

            if self.priv_info_stage2:
                # 2. Proprioceptive History Encoder (Adaptation Module)
                #    Infers environment properties from proprioceptive history alone.
                self.adapt_tconv = ProprioAdaptTConv(
                    obs_dim=adapt_obs_dim,
                    hidden_dim=adapt_hidden_dim,
                    latent_dim=adapt_latent_dim,
                    history_len=adapt_history_len,
                    conv_kernels=adapt_conv_kernels,
                )

        # 3. Policy State Encoder (shared actor-critic backbone)
        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
            requires_grad=True,
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        """Sample actions for rollout collection (with exploration noise)."""
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            "neglogpacs": -distr.log_prob(selected_action).sum(1),
            "values": value,
            "actions": selected_action,
            "mus": mu,
            "sigmas": sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        """Return deterministic actions for evaluation/deployment."""
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        """Core forward pass supporting all three modes."""
        obs = obs_dict["obs"]
        extrin, extrin_gt = None, None

        if self.priv_info:
            if self.priv_info_stage2:
                # Stage 2: Student mode
                # Adaptation module infers environment latent from history
                extrin = self.adapt_tconv(obs_dict["proprio_hist"])

                # Privileged encoder provides ground truth (only during training)
                extrin_gt = (
                    self.env_mlp(obs_dict["priv_info"])
                    if "priv_info" in obs_dict
                    else extrin
                )
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)

                obs = torch.cat([obs, extrin], dim=-1)
            else:
                # Stage 1: Teacher mode
                # Privileged encoder directly encodes environment physics
                extrin = self.env_mlp(obs_dict["priv_info"])
                extrin = torch.tanh(extrin)

                obs = torch.cat([obs, extrin], dim=-1)

        # Shared actor-critic backbone
        x = self.actor_mlp(obs)

        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        """Full forward pass returning all outputs for PPO training."""
        prev_actions = input_dict.get("prev_actions", None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            "prev_neglogp": torch.squeeze(prev_neglogp),
            "values": value,
            "entropy": entropy,
            "mus": mu,
            "sigmas": sigma,
            "extrin": extrin,
            "extrin_gt": extrin_gt,
        }
        return result
