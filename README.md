# MFR Benchmark — Isaac Lab Port

This is a fork of [UM-ARM-Lab/MFR_benchmark](https://github.com/UM-ARM-Lab/MFR_benchmark), a Multi-Finger Dexterous Manipulation benchmark using the Allegro robotic hand with NVIDIA Isaac Gym.

Original codebase adapted from [UM-ARM-Lab/isaacgym-arm-envs](https://github.com/UM-ARM-Lab/isaacgym-arm-envs).

## What's ported

Three tasks have been ported from the original Isaac Gym (`gymapi`/`gymtorch`) implementation to **Isaac Lab** (`DirectRLEnv`):

| Task | Gym ID | Status |
|---|---|---|
| Screwdriver Turning (90°) | `Isaac-Allegro-Screwdriver-Turning-Direct-v0` | Stable baseline |
| Screwdriver Turning (360°) | `Isaac-Allegro-Screwdriver-Turning-360-v0` | Implemented |
| Screwdriver Manipulation (6D) | `Isaac-Allegro-Screwdriver-Manipulation-Direct-v0` | Draft |

All tasks share the same architecture:

| | Isaac Gym (original) | Isaac Lab (ported) |
|---|---|---|
| Entry point | `example.py` | `MFR_benchmark/isaac_lab_tasks/` |
| Config | scattered across env + wrapper | `@configclass` with Hydra overrides |
| Registered | — | Gymnasium `gym.make()` |

### Screwdriver Turning (90°)

The baseline task. The Allegro hand (3 fingers: index, middle, thumb) must turn a screwdriver ~90° around its Z-axis while keeping it upright. Episode length: 12 s. Goal: `goal_euler_xyz = (0, 0, -1.5707)`.

Source: `MFR_benchmark/isaac_lab_tasks/screwdriver_turning/`

### Screwdriver Turning (360°)

Variant requiring one full rotation (~360°). Inherits from the 90° task — only the config is overridden (`episode_length_s = 36.0`, `goal_euler_xyz = (0, 0, -2π)`). The Z-rotation joint is continuous (no limits), so the existing squared-error reward works without modification.

Source: `MFR_benchmark/isaac_lab_tasks/screwdriver_turning_360/`

### Screwdriver Manipulation (6D)

Draft port of the 6-DOF screwdriver manipulation task. Controls all four fingers (16 DOF) to pick up and reorient a free-floating screwdriver. Observation includes fingertip poses and object state (19-dim). Supports 6-DOF position+orientation control.

Source: `MFR_benchmark/isaac_lab_tasks/screwdriver_manipulation/`

## Install

```bash
pip install -e .
```

Requires Isaac Lab (tested with 0.54.3, Isaac Sim 5.1.0, Python 3.11).

## Usage

Activate the Isaac Lab conda environment, then:

```bash
export PYTHONPATH=/path/to/MFR_benchmark:$PYTHONPATH

# 90° turning (baseline)
python /path/to/IsaacLab/scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Allegro-Screwdriver-Turning-Direct-v0 --headless --num_envs 512

# 360° turning
python /path/to/IsaacLab/scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Allegro-Screwdriver-Turning-360-v0 --headless --num_envs 512

# 6D manipulation (draft)
python /path/to/IsaacLab/scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Allegro-Screwdriver-Manipulation-Direct-v0 --headless --num_envs 512
```

Override config fields with Hydra CLI: `env.<field>=<value>`, `agent.params.<path>=<value>`.

See `docs/screwdriver_turning_isaaclab_training.md` for full training commands and configuration options.

## Original tasks (Isaac Gym, unported)

These remain in `MFR_benchmark/tasks/` and `MFR_benchmark/wrapper/`, untouched:

- Valve turning
- Cuboid insertion / alignment
- Object reorientation
