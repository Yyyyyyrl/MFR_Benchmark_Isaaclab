"""Allegro screwdriver 360-degree turning task for Isaac Lab."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Allegro-Screwdriver-Turning-360-v0",
    entry_point=(
        "MFR_benchmark.isaac_lab_tasks.screwdriver_turning_360.screwdriver_turning_360_env:"
        "AllegroScrewdriverTurning360Env"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "MFR_benchmark.isaac_lab_tasks.screwdriver_turning_360.screwdriver_turning_360_env_cfg:"
            "AllegroScrewdriverTurning360EnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
