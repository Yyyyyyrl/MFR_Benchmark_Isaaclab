# Ported from allegro_inhand_rotation/hora/algo/models/running_mean_std.py
# Originally based on IsaacGymEnvs (NVIDIA, BSD 3-Clause)
# Adapted for MFR_benchmark Isaac Lab integration.

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Running mean and standard deviation normalization.

    Tracks running statistics of input tensors and normalizes them
    to zero mean and unit variance. Supports per-channel normalization
    for multi-dimensional inputs.

    Args:
        insize: Input size — either an int (flat) or tuple (multi-dim).
        epsilon: Small constant for numerical stability.
        per_channel: If True, normalize each channel independently.
        norm_only: If True, only divide by std (no mean subtraction).
    """

    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        self.insize = insize
        self.epsilon = epsilon
        self.norm_only = norm_only
        self.per_channel = per_channel

        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            elif len(self.insize) == 2:
                self.axis = [0, 2]
            elif len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        if self.training:
            mean = input.mean(self.axis)
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = (
                self._update_mean_var_count_from_moments(
                    self.running_mean,
                    self.running_var,
                    self.count,
                    mean,
                    var,
                    input.size()[0],
                )
            )

        # Reshape statistics to match input dimensions
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view(
                    [1, self.insize[0], 1, 1]
                ).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.insize[0], 1, 1]
                ).expand_as(input)
            elif len(self.insize) == 2:
                current_mean = self.running_mean.view(
                    [1, self.insize[0], 1]
                ).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.insize[0], 1]
                ).expand_as(input)
            elif len(self.insize) == 1:
                current_mean = self.running_mean.view(
                    [1, self.insize[0]]
                ).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.insize[0]]
                ).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = (
                torch.sqrt(current_var.float() + self.epsilon) * y
                + current_mean.float()
            )
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(
                    current_var.float() + self.epsilon
                )
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y
