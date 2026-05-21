"""Allegro screwdriver manipulation (6D) task for Isaac Lab."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Allegro-Screwdriver-Manipulation-Direct-v0",
    entry_point=(
        "MFR_benchmark.isaac_lab_tasks.screwdriver_manipulation.screwdriver_manipulation_env:"
        "AllegroScrewdriverManipulationEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "MFR_benchmark.isaac_lab_tasks.screwdriver_manipulation.screwdriver_manipulation_env_cfg:"
            "AllegroScrewdriverManipulationEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
