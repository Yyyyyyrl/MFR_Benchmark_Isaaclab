# MFR_benchmark RMA (Rapid Motor Adaptation) Module
# Ported from allegro_inhand_rotation/hora/algo/ for Isaac Lab integration.
#
# This module implements the 2-stage teacher-student training pipeline:
#   Stage 1: Teacher policy trained with privileged information via PPO
#   Stage 2: Student adaptation module trained via MSE distillation
#
# Original RMA paper: "In-Hand Object Rotation via Rapid Motor Adaptation"
#   https://arxiv.org/abs/2210.04887
# Original implementation by Haozhi Qi (2022)
# Adapted for MFR_benchmark Isaac Lab port (2025)

from .models import ActorCritic, MLP, ProprioAdaptTConv
from .running_mean_std import RunningMeanStd
from .experience import ExperienceBuffer
from .ppo import PPO
from .padapt import ProprioAdapt
