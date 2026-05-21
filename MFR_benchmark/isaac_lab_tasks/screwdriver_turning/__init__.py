"""Allegro screwdriver turning task for Isaac Lab."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Allegro-Screwdriver-Turning-Direct-v0",
    entry_point=(
        "MFR_benchmark.isaac_lab_tasks.screwdriver_turning.screwdriver_turning_env:"
        "AllegroScrewdriverTurningEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "MFR_benchmark.isaac_lab_tasks.screwdriver_turning.screwdriver_turning_env_cfg:"
            "AllegroScrewdriverTurningEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
