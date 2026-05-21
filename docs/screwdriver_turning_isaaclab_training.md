# Isaac Lab Screwdriver Turning Training Guide

This document covers the Isaac Lab port of the MFR screwdriver turning task:

```text
Isaac-Allegro-Screwdriver-Turning-Direct-v0
```

The task package lives at:

```text
MFR_benchmark/isaac_lab_tasks/screwdriver_turning/
```

The main files are:

- `screwdriver_turning_env.py`: task runtime logic.
- `screwdriver_turning_env_cfg.py`: simulation, asset, task, action, observation, and actuator configuration.
- `agents/rl_games_ppo_cfg.yaml`: RL-Games PPO configuration.
- `scripts/eval_screwdriver_checkpoint.py`: finite checkpoint evaluator.

## Running Commands

Always import the custom task package before invoking Isaac Lab scripts:

```bash
cd /home/user/MFR_benchmark
export PYTHONPATH=/home/user/MFR_benchmark:$PYTHONPATH
```

Short training smoke test:

```bash
python -c "import MFR_benchmark.isaac_lab_tasks, runpy; runpy.run_path('/home/user/IsaacLab/scripts/reinforcement_learning/rl_games/train.py', run_name='__main__')" \
  --task Isaac-Allegro-Screwdriver-Turning-Direct-v0 \
  --headless \
  --num_envs 16 \
  --max_iterations 5 \
  agent.params.config.minibatch_size=256
```

Normal training:

```bash
python -c "import MFR_benchmark.isaac_lab_tasks, runpy; runpy.run_path('/home/user/IsaacLab/scripts/reinforcement_learning/rl_games/train.py', run_name='__main__')" \
  --task Isaac-Allegro-Screwdriver-Turning-Direct-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 5000
```

Finite checkpoint evaluation:

```bash
python scripts/eval_screwdriver_checkpoint.py \
  --task Isaac-Allegro-Screwdriver-Turning-Direct-v0 \
  --checkpoint logs/rl_games/allegro_screwdriver_turning/<run>/nn/allegro_screwdriver_turning.pth \
  --num_envs 32 \
  --num_episodes 256 \
  --headless
```

## Environment Semantics

The environment is a DirectRLEnv. One policy step is one high-level hand command, and the physics simulator runs many smaller steps per policy step.

Important timing defaults:

- `sim.dt = 1 / 60`: physics step is 16.67 ms.
- `decimation = 60`: one policy action is held/interpolated over 60 physics steps.
- `step_dt = sim.dt * decimation = 1.0 s`: one RL step is one simulated second.
- `episode_length_s = 12.0`: an episode lasts 12 policy steps.

The default task is intentionally close to the legacy IsaacGym wrapper:

- controlled fingers: index, middle, thumb
- action dimension: 12
- observation dimension: 15
- action interpretation: desired joint positions, optionally offset from pregrasp
- reward: negative action, goal, and upright costs

## `screwdriver_turning_env_cfg.py`

### Action Space

```python
action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(12,), dtype=np.float32)
```

The action vector contains 4 commands for each controlled finger. With the default fingers:

```text
index  joints: action[0:4]
middle joints: action[4:8]
thumb  joints: action[8:12]
```

If `action_offset = True`, the policy action is added to the pregrasp pose. This matches the legacy wrapper behavior.

Practical values:

- `low=-2.0, high=2.0`: broad exploration, current default.
- `low=-1.0, high=1.0`: safer if policies frequently drive the hand into bad contacts.
- `low=-0.5, high=0.5`: conservative fine-tuning around pregrasp.

If you change `fingers`, update both `action_space` and `observation_space`:

```text
action_dim = 4 * len(fingers)
obs_dim = action_dim + 3
```

### Observation Space

```python
observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
```

The observation is:

```text
[selected finger joint positions, screwdriver_x, screwdriver_y, screwdriver_z]
```

For default fingers:

```text
12 Allegro joint positions + 3 screwdriver Euler-joint coordinates = 15
```

The observation does not include velocities, tactile signals, object pose as quaternion, previous action, or contact data. Adding those may improve learning, but changes the policy input dimension and requires updating the config and checkpoint compatibility.

### State Space

```python
state_space = 0
```

This means no privileged critic state is used. Isaac Lab's Hydra bridge expects `0`, not `None`, for no asymmetric state.

### Simulation Settings

```python
sim.dt = 1.0 / 60.0
sim.gravity = (0.0, 0.0, -9.81)
sim.physics_material.static_friction = 1.0
sim.physics_material.dynamic_friction = 1.0
```

`dt` controls physics stability and speed. Smaller `dt` is usually more stable and slower. Larger `dt` is faster and less stable.

Recommended values:

- `1/60`: current legacy-compatible default.
- `1/120`: more stable contact behavior, slower.
- `1/240`: useful for debugging contact issues, much slower.

`decimation` should be considered together with `dt`:

```text
policy_dt = dt * decimation
```

Examples:

- `dt=1/60`, `decimation=60`: policy step is 1.0 s.
- `dt=1/120`, `decimation=60`: policy step is 0.5 s.
- `dt=1/120`, `decimation=120`: policy step remains 1.0 s.

Changing `policy_dt` changes the control problem. If you reduce it, the policy gets more frequent control and episodes contain more steps.

### PhysX Solver

```python
solver_type = 1
position iterations = 8
velocity iterations = 0
```

Position iterations matter for contact and joint constraint quality. Higher values improve stability but reduce throughput.

Useful ranges:

- position iterations `4`: faster, less robust.
- position iterations `8`: current default.
- position iterations `12` or `16`: try when the screwdriver penetrates, jitters, or escapes.
- velocity iterations `0`: fast and common for this style of task.
- velocity iterations `1` or `2`: try if contact impulses look unstable.

### Scene Settings

```python
scene.num_envs = 512
scene.env_spacing = 1.5
scene.replicate_physics = True
```

`num_envs` is overridden by `--num_envs` in train/play/eval scripts.

Guidance:

- `1`: visual inspection.
- `16`: smoke tests.
- `128`: debugging training dynamics.
- `512`: default training.
- `1024+`: only if GPU memory and solver throughput are healthy.

Use a larger `env_spacing` if envs visually or physically overlap. `1.5` should be enough for this hand/object setup.

### Task Behavior Fields

```python
fingers = ("index", "middle", "thumb")
friction_coefficient = 1.0
gradual_control = True
action_offset = True
randomize_obj_start = False
reset_contact_steps = 32
goal_euler_xyz = (0.0, 0.0, -1.5707)
```

`fingers` chooses controlled fingers. Valid values are `index`, `middle`, `ring`, `thumb`. Changing this changes action/observation dimensions.

`friction_coefficient` affects the ground plane material. For the screwdriver-hand contact, the URDF-imported body materials may also matter. Still, this can affect stability through table/ground interaction.

Examples:

- `0.5`: more slippery.
- `1.0`: default.
- `2.0`: higher friction, can improve grasping but may create sticky contacts.

`gradual_control` interpolates each new target during the first 75% of the decimation window. This matches legacy behavior and reduces abrupt target jumps.

Examples:

- `True`: smoother, usually better for contact.
- `False`: sharper control, can learn faster in simple tasks but may destabilize contact.

`action_offset` adds the pregrasp pose to actions.

Examples:

- `True`: action means "delta from pregrasp"; easier exploration.
- `False`: action means absolute joint target; may require narrower action bounds or different initialization.

`randomize_obj_start` randomizes the screwdriver z rotation at reset.

Examples:

- `False`: easier fixed-start learning.
- `True`: better robustness, harder training. Use after the policy can solve fixed-start.

`reset_contact_steps` lets the scene settle after reset.

Examples:

- `0`: fastest; good for smoke tests.
- `8`: light settling.
- `32`: current default.
- `60`: more stable reset contacts, slower training.

`goal_euler_xyz` is the screwdriver target orientation. The z target `-1.5707` is approximately -90 degrees.

### Pregrasp Positions

```python
pregrasp_positions = {
    "index": (0.1, 0.6, 0.6, 0.6),
    "middle": (-0.1, 0.5, 0.9, 0.9),
    "ring": (0.0, 0.5, 0.65, 0.65),
    "thumb": (1.2, 0.3, 0.3, 1.2),
}
```

These are the reset poses and, with `action_offset=True`, the action origin. A poor pregrasp can make the task much harder than any PPO setting.

Tuning approach:

1. Run `zero_agent.py` visually with `--num_envs 1`.
2. Check whether the fingers start near the screwdriver without explosive contacts.
3. Adjust one finger at a time.
4. Keep values inside reasonable Allegro joint ranges.

### Robot Actuator

```python
ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    stiffness=6.0,
    damping=1.0,
    armature=0.001,
)
```

The Allegro joints are position-controlled through implicit actuators.

`stiffness` controls how strongly joints track targets.

- `3.0`: softer, less aggressive, may slip.
- `6.0`: current default.
- `10.0`: stronger tracking, may increase contact instability.

`damping` controls velocity damping.

- `0.5`: more responsive.
- `1.0`: current default.
- `2.0`: more damped and stable, slower motion.

`armature` adds inertial smoothing at the joint.

- `0.0`: physically lighter, potentially less stable.
- `0.001`: current default.
- `0.01`: more smoothing, may slow fine motion.

### Screwdriver Actuators

The screwdriver is imported as an articulation with passive joints:

```python
tilt joints: damping=0.0001
rotation joint: damping=0.05
cap joint: damping=0.0
```

The policy does not command these joints directly. They move through contact with the hand.

`rotation` damping affects the main turning axis:

- `0.01`: easier to turn, may spin too freely.
- `0.05`: current default.
- `0.1`: harder to turn, more damping.

`tilt` damping affects x/y upright stability. The reward already strongly penalizes x/y tilt.

## `screwdriver_turning_env.py`

### Joint Discovery

The environment resolves joint IDs by exact joint names from the local URDF. This is safer than relying on joint order.

If the URDF changes, the first failure will usually be a clear joint-resolution error.

### Action Processing

The action path is:

1. `_pre_physics_step(actions)` receives policy action.
2. If `action_offset=True`, add pregrasp joint positions.
3. If `gradual_control=True`, remember current joint positions for interpolation.
4. `_apply_action()` sends joint targets every physics step.

The interpolation ramp length is:

```python
0.75 * decimation
```

With `decimation=60`, the target ramps over 45 physics steps, matching the legacy task.

### Observations

```python
finger_q = allegro joint positions for selected fingers
screwdriver_euler = screwdriver joints 1, 2, 3
obs = concat(finger_q, screwdriver_euler)
```

The final three observation entries are:

```text
x tilt, y tilt, z turn angle
```

### Reward

```python
action_cost = sum(action ** 2)
goal_cost = sum(20.0 * (obj_orientation - goal) ** 2)
upright_cost = 10000.0 * sum(obj_orientation[:, :-1] ** 2)
reward = -(action_cost + goal_cost + upright_cost)
```

Meaning:

- Action cost discourages large commands.
- Goal cost encourages all three screwdriver Euler joints toward the target.
- Upright cost heavily penalizes x/y tilt, so the policy should turn around z without tipping.

The upright penalty is very strong. If learning gets stuck with very negative returns dominated by upright cost, inspect x/y tilt in evaluation CSVs.

Possible tuning:

- Reduce upright coefficient from `10000` to `1000` if the task is too conservative.
- Increase goal coefficient from `20` to `40` if the policy stabilizes but does not turn.
- Increase action penalty only if actions saturate and contacts are unstable.

These reward constants are currently in code, not config. If you plan to tune them often, move them into `AllegroScrewdriverTurningEnvCfg`.

### Termination

The task has no success termination. Episodes end only by timeout:

```python
timed_out = episode_length_buf >= max_episode_length - 1
```

This is useful for PPO because every episode has the same horizon. It also means a good policy keeps acting for the full episode rather than stopping once the target angle is reached.

## RL-Games PPO Config

File:

```text
MFR_benchmark/isaac_lab_tasks/screwdriver_turning/agents/rl_games_ppo_cfg.yaml
```

### Seed

```yaml
seed: 42
```

Controls random seeds in training. Use multiple seeds to compare real improvements.

Examples:

```bash
--seed 0
--seed 1
--seed 2
```

### Observation and Action Clipping

```yaml
clip_observations: 5.0
clip_actions: 2.0
```

`clip_observations` clips values before they reach the policy. Since observations are joint angles and screwdriver angles, `5.0` radians is broad.

`clip_actions` clips policy outputs. It should match the action space range unless you intentionally want tighter runtime clipping.

Examples:

- `clip_actions=2.0`: current default.
- `clip_actions=1.0`: safer fine-tuning.
- `clip_observations=10.0`: use if randomized starts produce larger angle values.

### Algorithm and Model

```yaml
algo.name: a2c_continuous
model.name: continuous_a2c_logstd
ppo: True
```

RL-Games uses the `a2c_continuous` implementation for PPO-style continuous control when `ppo=True`.

### Network

```yaml
units: [1024, 512, 256, 128]
activation: elu
fixed_sigma: True
sigma_init.val: 0
```

The actor and critic share an MLP because `separate=False`.

Network size guidance:

- `[256, 256]`: faster, often enough for small observations.
- `[512, 256, 128]`: middle ground.
- `[1024, 512, 256, 128]`: current default, more capacity.

`fixed_sigma=True` means exploration std is represented as a learned global parameter rather than a state-dependent network output. This is common and stable.

`sigma_init.val=0` means initial log std around 0, so initial std is about 1. For this action scale, that is exploratory.

If actions are too noisy early:

```bash
agent.params.network.space.continuous.sigma_init.val=-1.0
```

If exploration is too weak:

```bash
agent.params.network.space.continuous.sigma_init.val=0.5
```

### Normalization

```yaml
normalize_input: True
normalize_value: True
normalize_advantage: True
```

Input normalization tracks running observation statistics. Keep this enabled unless debugging raw observation scaling.

Value normalization is helpful because rewards can be large negative numbers from upright penalties.

Advantage normalization is usually beneficial for PPO stability.

### Discounting and GAE

```yaml
gamma: 0.99
tau: 0.95
```

`gamma` discounts future rewards. With this task's policy step of 1 second and 12-step episodes, `0.99` is reasonable.

Examples:

- `gamma=0.95`: more myopic, may focus on immediate contact stability.
- `gamma=0.99`: default.
- `gamma=0.995`: more long-horizon, can be useful if episodes get longer.

`tau` is GAE lambda:

- `0.9`: lower variance, more bias.
- `0.95`: default.
- `0.98`: higher variance, less bias.

### Learning Rate

```yaml
learning_rate: 5e-4
lr_schedule: adaptive
kl_threshold: 0.016
```

The adaptive schedule adjusts learning rate based on KL divergence.

Examples:

- `1e-4`: safer fine-tuning.
- `3e-4`: stable default candidate.
- `5e-4`: current default.
- `1e-3`: aggressive; can diverge.

If policy loss/returns are unstable, try:

```bash
agent.params.config.learning_rate=3e-4
agent.params.config.kl_threshold=0.01
```

### PPO Epochs, Horizon, and Minibatches

```yaml
horizon_length: 16
minibatch_size: 8192
mini_epochs: 5
```

One PPO batch contains:

```text
batch_size = num_envs * horizon_length
```

With `num_envs=512` and `horizon_length=16`, batch size is `8192`, matching the default minibatch.

Recommended relationships:

- `minibatch_size <= num_envs * horizon_length`
- Prefer minibatch sizes that divide the batch size.
- Larger horizon improves advantage estimates but uses more memory.

Examples:

```bash
# 128 envs
--num_envs 128 agent.params.config.horizon_length=32 agent.params.config.minibatch_size=4096

# 512 envs
--num_envs 512 agent.params.config.horizon_length=16 agent.params.config.minibatch_size=8192

# 1024 envs
--num_envs 1024 agent.params.config.horizon_length=16 agent.params.config.minibatch_size=16384
```

`mini_epochs` controls how many optimization passes are made over each rollout batch.

- `3`: faster, less overfitting.
- `5`: default.
- `8`: stronger optimization, can overfit stale data.

### PPO Clip and Value Clip

```yaml
e_clip: 0.2
clip_value: True
```

`e_clip` is PPO's policy ratio clipping range.

- `0.1`: conservative updates.
- `0.2`: default.
- `0.3`: larger updates, more risk.

Keep `clip_value=True` for stability.

### Entropy

```yaml
entropy_coef: 0.0
```

Entropy bonus encourages exploration.

For this task, if the policy collapses to tiny actions, try:

```bash
agent.params.config.entropy_coef=0.001
```

If the hand remains noisy and never stabilizes contact, keep `0.0` or reduce initial sigma.

### Critic Coefficient

```yaml
critic_coef: 4
```

Weights value loss in the PPO objective. Large negative rewards can make value learning important.

Examples:

- `2`: less critic emphasis.
- `4`: default.
- `6`: more critic emphasis.

### Gradient Controls

```yaml
grad_norm: 1.0
truncate_grads: True
```

Gradient clipping helps avoid large updates. Keep enabled.

If training has NaNs or spikes:

```bash
agent.params.config.grad_norm=0.5
agent.params.config.learning_rate=1e-4
```

### Bounds Loss

```yaml
bounds_loss_coef: 0.0001
```

Penalizes actions that push outside valid action bounds.

Examples:

- `0.0`: no bounds penalty.
- `0.0001`: default.
- `0.001`: stronger action-bound discipline.

### Reward Shaper

```yaml
reward_shaper.scale_value: 0.01
```

RL-Games multiplies rewards by this value before learning. The raw environment reward is large and negative, so this prevents value targets from being too large.

Examples:

- `0.001`: if value loss is huge or unstable.
- `0.01`: default.
- `0.05`: stronger learning signal, more risk.

## Command-Line Override Patterns

Hydra overrides are appended after normal CLI args.

Named experiment:

```bash
agent.params.config.full_experiment_name=screwdriver_lr3e4_h32
```

Lower learning rate:

```bash
agent.params.config.learning_rate=3e-4
```

Object start randomization:

```bash
env.randomize_obj_start=true
```

Fewer reset settling steps:

```bash
env.reset_contact_steps=8
```

Different target:

```bash
env.goal_euler_xyz='[0.0,0.0,-3.14159]'
```

Softer hand:

```bash
env.robot_cfg.actuators.fingers.stiffness=3.0 env.robot_cfg.actuators.fingers.damping=1.5
```

Harder screwdriver rotation:

```bash
env.screwdriver_cfg.actuators.rotation.damping=0.1
```

## Tuning Recipes

### Stable Baseline

Use when debugging the port or checking that learning starts.

```bash
--num_envs 128 \
agent.params.config.horizon_length=16 \
agent.params.config.minibatch_size=2048 \
agent.params.config.learning_rate=3e-4 \
env.randomize_obj_start=false
```

### Higher Throughput

Use when the baseline is stable and GPU memory is available.

```bash
--num_envs 1024 \
agent.params.config.horizon_length=16 \
agent.params.config.minibatch_size=16384
```

### More Robust Policy

Use after fixed-start training works.

```bash
env.randomize_obj_start=true \
agent.params.config.learning_rate=3e-4 \
agent.params.network.space.continuous.sigma_init.val=-0.5
```

### Contact-Stability Debugging

Use if the hand or screwdriver jitters badly.

```bash
--num_envs 64 \
env.reset_contact_steps=60 \
env.robot_cfg.actuators.fingers.stiffness=4.0 \
env.robot_cfg.actuators.fingers.damping=2.0 \
env.screwdriver_cfg.actuators.rotation.damping=0.1
```

### Faster Iteration

Use for quick reward/observation experiments.

```bash
--num_envs 32 \
--max_iterations 50 \
agent.params.config.horizon_length=8 \
agent.params.config.minibatch_size=256 \
env.reset_contact_steps=0
```

## Evaluating Checkpoints

The stock Isaac Lab `play.py` runs forever unless recording video. The custom evaluator runs finite episodes and reports metrics.

Basic:

```bash
python scripts/eval_screwdriver_checkpoint.py \
  --checkpoint logs/rl_games/allegro_screwdriver_turning/<run>/nn/allegro_screwdriver_turning.pth \
  --num_envs 32 \
  --num_episodes 256 \
  --headless
```

CSV output:

```bash
python scripts/eval_screwdriver_checkpoint.py \
  --checkpoint logs/rl_games/allegro_screwdriver_turning/<run>/nn/allegro_screwdriver_turning.pth \
  --num_envs 32 \
  --num_episodes 256 \
  --headless \
  --csv logs/eval/screwdriver_eval.csv
```

Metrics:

- `return`: raw environment return accumulated over an episode.
- `final_z_angle`: screwdriver z joint at episode end.
- `abs_final_z_goal_error`: absolute final z error relative to `goal_euler_xyz[2]`.
- `upright_norm_xy`: norm of screwdriver x/y tilt at episode end.
- `final_goal_cost`: final-step goal cost.
- `final_upright_cost`: final-step upright penalty.
- `success@Xrad`: fraction of episodes with final absolute z error at or below X radians.

Interpretation:

- Good policies should reduce `abs_final_z_goal_error`.
- Good policies should keep `upright_norm_xy` near zero.
- If return improves but z error does not, the policy may be learning to avoid upright penalties without turning.
- If z error improves but upright norm is high, reduce task aggressiveness or increase upright penalty.

## Common Issues

### "Environment TASK doesn't exist"

You passed the literal string `TASK`. Use:

```bash
--task Isaac-Allegro-Screwdriver-Turning-Direct-v0
```

or define and quote a shell variable:

```bash
export TASK=Isaac-Allegro-Screwdriver-Turning-Direct-v0
--task "$TASK"
```

### "Unsupported space (None)"

`state_space` must be `0`, not `None`, in this Isaac Lab version.

### `play.py` Looks Stuck After Loading Checkpoint

This is normal. Stock `play.py` enters an infinite rollout loop and does not print progress. Use the custom evaluator for finite runs.

### USD Visual Reference Warnings

The URDF importer may print unresolved visual reference warnings. These are noisy but non-fatal if the env starts and steps.

Real failures usually contain:

```text
Traceback
Used null prim
Could not resolve joints
```

### Poor GPU Utilization

Increase `--num_envs` or `horizon_length`. For training, start with `512` envs. For visual or checkpoint debugging, use `1` to `32`.

### GPU Fully Loaded During Evaluation

This is expected when running headless without `--real-time`. The simulator steps as fast as possible.

Use:

```bash
--num_envs 1 --real-time
```

for slower visual playback.
