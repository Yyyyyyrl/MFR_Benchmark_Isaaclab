"""Diagnose Linker Hand screwdriver task reward breakdown."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import MFR_benchmark.isaac_lab_tasks  # noqa: F401
from MFR_benchmark.isaac_lab_tasks.screwdriver_turning_linker_hand.screwdriver_turning_linker_hand_env_cfg import (
    AllegroScrewdriverTurningLinkerHandEnvCfg,
)
from MFR_benchmark.isaac_lab_tasks.screwdriver_turning_linker_hand.screwdriver_turning_linker_hand_env import (
    LinkerHandScrewdriverTurningEnv,
)

cfg = AllegroScrewdriverTurningLinkerHandEnvCfg()
cfg.scene.num_envs = 4
env = LinkerHandScrewdriverTurningEnv(cfg, render_mode=None)
env.reset()

finger_ids = env._finger_joint_ids
all_ids = env._all_finger_joint_ids
screwdriver_euler_ids = env._screwdriver_euler_joint_ids

print("=" * 60)
print("LINKER HAND DIAGNOSTIC")
print("=" * 60)

# Initial joint state
all_q = env.allegro.data.joint_pos[0]
print(f"\nAll {len(all_q)} joint positions (env 0):")
for i, val in enumerate(all_q.tolist()):
    name = env.allegro.joint_names[i] if hasattr(env.allegro, "joint_names") else f"joint_{i}"
    print(f"  [{i:2d}] {val:+.4f}")

print(f"\nControlled joint IDs ({len(finger_ids)}): {finger_ids}")
print(f"All finger joint IDs ({len(all_ids)}): {all_ids[:8]}...")

# Screwdriver state
sd_q = env.screwdriver.data.joint_pos[:, screwdriver_euler_ids]
print(f"\nScrewdriver euler (4 envs):")
for i in range(4):
    print(f"  env {i}: {[f'{x:+.4f}' for x in sd_q[i].tolist()]}")

# Hand root pose
root_pos = env.allegro.data.root_pos_w
root_rot = env.allegro.data.root_quat_w
print(f"\nHand root position (4 envs):")
for i in range(4):
    print(f"  env {i}: pos={[f'{x:.4f}' for x in root_pos[i].tolist()]}")

# Get ALL body positions for the hand (link origins in world)
print(f"\nHand link world positions (env 0):")
body_pos = env.allegro.data.body_pos_w[0]  # [num_bodies, 3]
body_names = env.allegro.body_names
for j, name in enumerate(body_names):
    if any(k in name for k in ["metacarpal", "proximal", "middle", "distal", "base", "palm", "thumb"]):
        print(f"  {name}: pos={[f'{x:.4f}' for x in body_pos[j].tolist()]}")

# Also print fingertip positions explicitly
print(f"\nFingertip world positions (env 0):")
for name in ["index_distal", "middle_distal", "thumb_distal", "ring_distal", "pinky_distal"]:
    if name in body_names:
        idx = body_names.index(name)
        print(f"  {name}: {[f'{x:.4f}' for x in body_pos[idx].tolist()]}")
    else:
        # Try partial match
        for bn in body_names:
            if name in bn:
                idx = body_names.index(bn)
                print(f"  {bn}: {[f'{x:.4f}' for x in body_pos[idx].tolist()]}")

# Screwdriver link positions
print(f"\nScrewdriver link world positions (env 0):")
sd_body_pos = env.screwdriver.data.body_pos_w[0]
sd_body_names = env.screwdriver.body_names
for j, name in enumerate(sd_body_names):
    print(f"  {name}: pos={[f'{x:.4f}' for x in sd_body_pos[j].tolist()]}")

# Screwdriver root pose
sd_root = env.screwdriver.data.root_pos_w
print(f"\nScrewdriver root position (4 envs):")
for i in range(4):
    print(f"  env {i}: pos={[f'{x:.4f}' for x in sd_root[i].tolist()]}")

# Distance from hand base to screwdriver
dist = torch.linalg.norm(root_pos - sd_root, dim=-1)
print(f"\nHand-to-screwdriver distance: {[f'{x:.4f}' for x in dist.tolist()]} m")

# Step for 1 second with zero actions and see what happens to the screwdriver
print("\n--- Zero-action hold test (60 steps) ---")
actions = torch.zeros(4, len(finger_ids), device=env.device)
for step in range(60):
    env._pre_physics_step(actions)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)

sd_q_after = env.screwdriver.data.joint_pos[:, screwdriver_euler_ids]
print(f"Screwdriver euler after hold:")
for i in range(4):
    print(f"  env {i}: {[f'{x:+.4f}' for x in sd_q_after[i].tolist()]}")
print("(should be near zero if hand is not touching screwdriver)")

# Reset and try with random actions
env.reset()
print("\n--- Random action test (60 steps) ---")
actions = torch.randn(4, len(finger_ids), device=env.device) * 0.5
for step in range(60):
    env._pre_physics_step(actions)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)

sd_q_rand = env.screwdriver.data.joint_pos[:, screwdriver_euler_ids]
goal = env._goal_euler[0]
goal_err = sd_q_rand - goal
print(f"Screwdriver euler after random actions:")
for i in range(4):
    print(f"  env {i}: {[f'{x:+.4f}' for x in sd_q_rand[i].tolist()]}")
print(f"Goal error:")
for i in range(4):
    print(f"  env {i}: {[f'{x:+.4f}' for x in goal_err[i].tolist()]}")

# Reward breakdown
action_cost = 1.0 * torch.sum(actions**2, dim=-1)
goal_cost = env.cfg.reward_goal_weight * torch.sum(goal_err**2, dim=-1)
upright_cost = env.cfg.reward_upright_weight * torch.sum(sd_q_rand[:, :-1]**2, dim=-1)
total_cost = action_cost + goal_cost + upright_cost
print(f"\nReward breakdown (per env):")
for i in range(4):
    print(f"  env {i}: action={action_cost[i]:.1f} goal={goal_cost[i]:.1f} upright={upright_cost[i]:.1f} total={total_cost[i]:.1f} reward={-total_cost[i]:.1f}")

# Check if pinky/ring joints are moving (they shouldn't be since we only control index, middle, thumb)
finger_q_all = env.allegro.data.joint_pos[:, all_ids]
print(f"\nAll finger joints shape: {finger_q_all.shape}")
print(f"Ring+pinky should be at pregrasp. First env finger positions:")
for i in range(len(all_ids)):
    print(f"  joint {all_ids[i]:2d} = {finger_q_all[0, i]:+.4f}")

env.close()
print("\nDone.")
