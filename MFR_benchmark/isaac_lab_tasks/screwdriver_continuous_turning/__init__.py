"""Allegro screwdriver continuous turning task for Isaac Lab."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Allegro-Screwdriver-Continuous-Turning-Direct-v0",
    entry_point=(
        "MFR_benchmark.isaac_lab_tasks.screwdriver_continuous_turning."
        "screwdriver_continuous_turning_env:AllegroScrewdriverContinuousTurningEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "MFR_benchmark.isaac_lab_tasks.screwdriver_continuous_turning."
            "screwdriver_continuous_turning_env_cfg:AllegroScrewdriverContinuousTurningEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
