# MFR Benchmark — Isaac Lab Port

This is a fork of [UM-ARM-Lab/MFR_benchmark](https://github.com/UM-ARM-Lab/MFR_benchmark), a Multi-Finger Dexterous Manipulation benchmark using the Allegro robotic hand with NVIDIA Isaac Gym.

Original codebase adapted from [UM-ARM-Lab/isaacgym-arm-envs](https://github.com/UM-ARM-Lab/isaacgym-arm-envs).

## What's ported

**Screwdriver Turning** task ported from Isaac Gym (`gymapi`/`gymtorch`) to **Isaac Lab** (`DirectRLEnv`).

| | Isaac Gym (original) | Isaac Lab (ported) |
|---|---|---|
| Entry point | `example.py` | `MFR_benchmark/isaac_lab_tasks/` |
| Env class | `AllegroScrewdriverTurningEnv` + `AllegroScrewdriverRLWrapper` | `AllegroScrewdriverTurningEnv` (single `DirectRLEnv`) |
| Config | scattered across env + wrapper | `AllegroScrewdriverTurningEnvCfg` (`@configclass`) |
| Registered task | — | `Isaac-Allegro-Screwdriver-Turning-Direct-v0` |

## Install

```bash
pip install -e .
```

Requires Isaac Lab (tested with 0.54.3, Isaac Sim 5.1.0, Python 3.11).

## Usage

Activate the Isaac Lab conda environment, then:

```bash
export PYTHONPATH=/path/to/MFR_benchmark:$PYTHONPATH
python /path/to/IsaacLab/scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Allegro-Screwdriver-Turning-Direct-v0 --headless --num_envs 512
```

See `docs/screwdriver_turning_isaaclab_training.md` for full training commands and configuration options.

## Original tasks (Isaac Gym, unported)

- Valve turning
- Cuboid insertion / alignment
- Object reorientation
- Screwdriver manipulation (6D)

These remain in `MFR_benchmark/tasks/` and `MFR_benchmark/wrapper/`, untouched.
