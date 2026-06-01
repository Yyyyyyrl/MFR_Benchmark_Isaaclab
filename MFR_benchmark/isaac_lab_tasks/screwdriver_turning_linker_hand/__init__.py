"""Linker Hand L20 screwdriver turning task for Isaac Lab."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-LinkerHand-Screwdriver-Turning-Direct-v0",
    entry_point=(
        "MFR_benchmark.isaac_lab_tasks.screwdriver_turning_linker_hand.screwdriver_turning_linker_hand_env:"
        "LinkerHandScrewdriverTurningEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "MFR_benchmark.isaac_lab_tasks.screwdriver_turning_linker_hand."
            "screwdriver_turning_linker_hand_env_cfg:"
            "AllegroScrewdriverTurningLinkerHandEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
