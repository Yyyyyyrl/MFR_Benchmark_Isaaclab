"""Set and render a side-grasp initial pose for LinkerHand screwdriver turning.

The script instantiates the existing LinkerHand L20 continuous screwdriver task,
searches the live USD stage for the actual LinkerHand and Screwdriver prims, then
sets a vertical screwdriver pose followed by a left/rear-left side grasp.

Run from the repository root with the Isaac Lab environment active, for example:

    python scripts/set_linkerhand_screwdriver_initial_pose.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any


TASK_ID = "Isaac-LinkerHand-Screwdriver-Continuous-Turning-Direct-v0"

SCREWDRIVER_ROOT_POS = (0.0, 0.0, 1.205)
SCREWDRIVER_ROOT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
SCREWDRIVER_HANDLE_CENTER_LOCAL = (0.0, 0.0, 0.150)

# Hand root is placed beside the handle, not above it.  The XY offset is a
# left/rear-left approach, and the Z offset puts the palm/finger bases at handle
# height after the hand's local finger direction is tilted upward.
HAND_ROOT_OFFSET_FROM_HANDLE_CENTER = (-0.135, -0.040, -0.130)
HAND_YAW_DEG = 10.0
HAND_PALM_DOWN_TILT_DEG = 38.0
HAND_SPREAD_ROLL_DEG = -4.0

# Finger targets are intentionally not a full fist.  Thumb, middle, and ring form
# the main three-point support; index and pinky stay lighter for stabilization.
FINGER_JOINT_TARGETS: dict[str, dict[str, float]] = {
    "thumb": {
        "thumb_cmc_yaw": 1.18,
        "thumb_cmc_roll": 0.12,
        "thumb_cmc_pitch": 0.50,
        "thumb_mcp": 0.50,
        "thumb_ip": 0.25,
    },
    "index": {
        "index_mcp_roll": 0.16,
        "index_mcp_pitch": 0.36,
        "index_pip": 0.25,
        "index_dip": 0.20,
    },
    "middle": {
        "middle_mcp_roll": -0.16,
        "middle_mcp_pitch": 0.54,
        "middle_pip": 0.82,
        "middle_dip": 0.65,
    },
    "ring": {
        "ring_mcp_roll": -0.16,
        "ring_mcp_pitch": 0.52,
        "ring_pip": 0.76,
        "ring_dip": 0.58,
    },
    "pinky": {
        "pinky_mcp_roll": -0.16,
        "pinky_mcp_pitch": 0.42,
        "pinky_pip": 0.58,
        "pinky_dip": 0.45,
    },
}

FINGERTIP_BODY_BY_FINGER = {
    "thumb": "thumb_distal",
    "index": "index_distal",
    "middle": "middle_distal",
    "ring": "ring_distal",
    "pinky": "pinky_distal",
}
SCREWDRIVER_NEAR_BODIES = ("screwdriver_stick", "screwdriver_body", "screwdriver_cap")


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID, help="Gym task ID to instantiate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/initial_pose_renders"),
        help="Directory for the rendered PNG and JSON summary.",
    )
    parser.add_argument("--image-name", default="linkerhand_side_grasp_initial.png")
    parser.add_argument("--summary-name", default="linkerhand_side_grasp_summary.json")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=(0.34, 0.22, 1.52))
    parser.add_argument("--camera-lookat", type=float, nargs=3, default=(0.0, -0.02, 1.32))
    parser.add_argument("--render-warmup", type=int, default=24)
    parser.add_argument(
        "--hand-root-offset",
        type=float,
        nargs=3,
        default=HAND_ROOT_OFFSET_FROM_HANDLE_CENTER,
        metavar=("X", "Y", "Z"),
        help="Hand root offset from the screwdriver handle center.",
    )
    parser.add_argument("--hand-yaw-deg", type=float, default=HAND_YAW_DEG)
    parser.add_argument("--hand-palm-down-tilt-deg", type=float, default=HAND_PALM_DOWN_TILT_DEG)
    parser.add_argument(
        "--hand-spread-roll-deg",
        type=float,
        default=HAND_SPREAD_ROLL_DEG,
        help="Roll the hand around its local palm-normal axis. Positive values rotate the index-to-pinky row downward.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help="Optional physics settle steps after writing the pose. Defaults to 0 to avoid tilting the passive screwdriver.",
    )
    parser.add_argument("--no-render", action="store_true", help="Only set and summarize the pose.")
    parser.add_argument("--print-stage-paths", action="store_true", help="Print all matching stage paths.")
    parser.add_argument("--show", action="store_true", help="Launch a visible Isaac Sim window instead of headless mode.")
    return parser


def _resolve_entry_point(entry_point: str) -> Any:
    module_name, attr_name = entry_point.split(":", 1)
    return getattr(importlib.import_module(module_name), attr_name)


def _quat_multiply(q0: tuple[float, float, float, float], q1: tuple[float, float, float, float]) -> tuple[float, ...]:
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return (
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    )


def _quat_from_axis_angle(axis: tuple[float, float, float], angle_rad: float) -> tuple[float, ...]:
    half = 0.5 * angle_rad
    scale = math.sin(half)
    return (math.cos(half), axis[0] * scale, axis[1] * scale, axis[2] * scale)


def _normalize_quat(quat: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in quat))
    return tuple(value / norm for value in quat)


def _hand_root_quat_wxyz(yaw_deg: float, palm_down_tilt_deg: float, spread_roll_deg: float) -> tuple[float, ...]:
    yaw = _quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(yaw_deg))
    palm_down_tilt = _quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(palm_down_tilt_deg))
    spread_roll = _quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(spread_roll_deg))
    return _normalize_quat(_quat_multiply(_quat_multiply(yaw, palm_down_tilt), spread_roll))


def _tensor_list(tensor: Any) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _flatten_joint_targets() -> dict[str, float]:
    targets: dict[str, float] = {}
    for finger_targets in FINGER_JOINT_TARGETS.values():
        targets.update(finger_targets)
    return targets


def _find_stage_prims() -> dict[str, Any]:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    matching_paths = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "LinkerHand" in path or "Screwdriver" in path:
            matching_paths.append({"path": path, "type": prim.GetTypeName(), "name": prim.GetName()})

    def select_root(suffix: str) -> str | None:
        candidates = [entry["path"] for entry in matching_paths if entry["path"].endswith(f"/{suffix}")]
        env0_candidates = [path for path in candidates if "/World/envs/env_0/" in path]
        if env0_candidates:
            return env0_candidates[0]
        return candidates[0] if candidates else None

    return {
        "linkerhand_root": select_root("LinkerHand"),
        "screwdriver_root": select_root("Screwdriver"),
        "matching_paths": matching_paths,
    }


def _joint_id_by_name(articulation: Any) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(articulation.joint_names)}


def _body_id_by_name(articulation: Any) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(articulation.body_names)}


def _require_joint_names(env: Any, joint_targets: dict[str, float]) -> None:
    missing = sorted(set(joint_targets).difference(env.allegro.joint_names))
    if missing:
        raise RuntimeError(
            "Could not map all LinkerHand finger joints. "
            f"Missing: {missing}. Available joints: {env.allegro.joint_names}"
        )


def _apply_pose(env: Any, torch: Any, args: argparse.Namespace) -> dict[str, Any]:
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    env_origin = env.scene.env_origins[env_ids]
    joint_targets = _flatten_joint_targets()
    _require_joint_names(env, joint_targets)

    # 1. Screwdriver pose: keep the mounted screwdriver vertical with the shaft
    # aligned to world Z and all screwdriver Euler joints reset to zero.
    screwdriver_root_pose = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    screwdriver_root_pose[:, :3] = env_origin + torch.tensor(SCREWDRIVER_ROOT_POS, dtype=torch.float32, device=env.device)
    screwdriver_root_pose[:, 3:7] = torch.tensor(SCREWDRIVER_ROOT_QUAT_WXYZ, dtype=torch.float32, device=env.device)
    env.screwdriver.write_root_pose_to_sim(screwdriver_root_pose, env_ids=env_ids)
    env.screwdriver.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=env.device), env_ids=env_ids)

    screwdriver_joint_pos = torch.zeros_like(env.screwdriver.data.default_joint_pos[env_ids])
    screwdriver_joint_vel = torch.zeros_like(env.screwdriver.data.default_joint_vel[env_ids])
    env.screwdriver.write_joint_state_to_sim(screwdriver_joint_pos, screwdriver_joint_vel, env_ids=env_ids)

    # 2. Hand base/wrist pose: compute a side approach from the screwdriver
    # handle center.  The hand's local +X palm normal points toward the handle
    # with a downward tilt, while the forearm stays lateral to the screwdriver.
    handle_center = _vec_add(SCREWDRIVER_ROOT_POS, SCREWDRIVER_HANDLE_CENTER_LOCAL)
    hand_root_pos = _vec_add(handle_center, tuple(args.hand_root_offset))
    hand_root_quat = _hand_root_quat_wxyz(args.hand_yaw_deg, args.hand_palm_down_tilt_deg, args.hand_spread_roll_deg)

    hand_root_pose = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    hand_root_pose[:, :3] = env_origin + torch.tensor(hand_root_pos, dtype=torch.float32, device=env.device)
    hand_root_pose[:, 3:7] = torch.tensor(hand_root_quat, dtype=torch.float32, device=env.device)
    env.allegro.write_root_pose_to_sim(hand_root_pose, env_ids=env_ids)
    env.allegro.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=env.device), env_ids=env_ids)

    # 3. Finger joint pose: set all discovered LinkerHand finger joints by name.
    # The env controls 16 independent joints; DIP/IP joints are also initialized
    # here when present so the full articulation starts in the intended grasp.
    hand_joint_pos = env.allegro.data.default_joint_pos[env_ids].clone()
    hand_joint_vel = torch.zeros_like(env.allegro.data.default_joint_vel[env_ids])
    joint_ids = _joint_id_by_name(env.allegro)
    for joint_name, value in joint_targets.items():
        hand_joint_pos[:, joint_ids[joint_name]] = float(value)

    env.allegro.set_joint_position_target(hand_joint_pos, env_ids=env_ids)
    env.allegro.write_joint_state_to_sim(hand_joint_pos, hand_joint_vel, env_ids=env_ids)

    if hasattr(env, "_finger_joint_ids"):
        controlled = hand_joint_pos[:, env._finger_joint_ids]
        env._target_actions[env_ids] = controlled
        env._start_joint_pos[env_ids] = controlled
        env._cur_targets[env_ids] = controlled

    if getattr(env, "_asymmetric_obs", False):
        finger_q = hand_joint_pos[:, env._finger_joint_ids]
        init_frame = env._make_proprio_frame(finger_q, env._cur_targets[env_ids])
        env._proprio_hist_buf[env_ids] = init_frame.unsqueeze(1).repeat(1, env._prop_hist_len, 1)

    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)

    return {
        "screwdriver_root_pos": list(SCREWDRIVER_ROOT_POS),
        "screwdriver_root_quat_wxyz": list(SCREWDRIVER_ROOT_QUAT_WXYZ),
        "screwdriver_handle_center": list(handle_center),
        "hand_root_pos": list(hand_root_pos),
        "hand_root_quat_wxyz": list(hand_root_quat),
        "hand_root_offset_from_handle_center": list(args.hand_root_offset),
        "hand_yaw_deg": float(args.hand_yaw_deg),
        "hand_palm_down_tilt_deg": float(args.hand_palm_down_tilt_deg),
        "hand_spread_roll_deg": float(args.hand_spread_roll_deg),
        "finger_joint_targets": FINGER_JOINT_TARGETS,
    }


def _settle(env: Any, steps: int) -> None:
    for _ in range(max(0, steps)):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)


def _nearest_screwdriver_body_summary(env: Any, torch: Any) -> list[dict[str, Any]]:
    hand_bodies = _body_id_by_name(env.allegro)
    screwdriver_bodies = _body_id_by_name(env.screwdriver)
    near_body_ids = [screwdriver_bodies[name] for name in SCREWDRIVER_NEAR_BODIES if name in screwdriver_bodies]
    near_body_pos = env.screwdriver.data.body_pos_w[0, near_body_ids, :3]
    handle_center = env.screwdriver.data.root_pos_w[0] + torch.tensor(
        SCREWDRIVER_HANDLE_CENTER_LOCAL, dtype=torch.float32, device=env.device
    )

    summary = []
    for finger, body_name in FINGERTIP_BODY_BY_FINGER.items():
        body_id = hand_bodies[body_name]
        tip_pos = env.allegro.data.body_pos_w[0, body_id, :3]
        distances = torch.linalg.norm(tip_pos.unsqueeze(0) - near_body_pos, dim=-1)
        nearest_distance, nearest_index = torch.min(distances, dim=-1)
        tip_offset = tip_pos - handle_center
        radial_xy = torch.linalg.norm(tip_offset[:2])
        summary.append(
            {
                "finger": finger,
                "fingertip_body": body_name,
                "fingertip_pos_w": _tensor_list(tip_pos),
                "nearest_screwdriver_body": env.screwdriver.body_names[near_body_ids[int(nearest_index)]],
                "nearest_screwdriver_distance_m": float(nearest_distance.detach().cpu()),
                "handle_center_offset_xyz": _tensor_list(tip_offset),
                "handle_center_radial_xy_m": float(radial_xy.detach().cpu()),
            }
        )
    return summary


def _render_frame(env: Any, np: Any, max_warmup_frames: int) -> Any:
    frame = None
    for _ in range(max(1, max_warmup_frames)):
        frame = env.render()
        if frame is not None and frame.size > 0 and np.any(frame):
            return frame
    if frame is None:
        raise RuntimeError("Environment returned no RGB frame.")
    return frame


def _run(args: argparse.Namespace, hydra_args: list[str]) -> dict[str, Any]:
    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]] + hydra_args
    if not args.show:
        args.headless = True
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from PIL import Image

        import MFR_benchmark.isaac_lab_tasks  # noqa: F401

        spec = gym.spec(args.task)
        cfg_cls = _resolve_entry_point(spec.kwargs["env_cfg_entry_point"])
        cfg = cfg_cls()
        cfg.scene.num_envs = 1
        cfg.viewer.resolution = (args.width, args.height)
        cfg.viewer.eye = tuple(args.camera_eye)
        cfg.viewer.lookat = tuple(args.camera_lookat)
        cfg.viewer.origin_type = "world"
        cfg.num_rerenders_on_reset = max(getattr(cfg, "num_rerenders_on_reset", 0), 2)

        env = gym.make(args.task, cfg=cfg, render_mode="rgb_array").unwrapped
        try:
            env.reset()
            stage_info = _find_stage_prims()
            applied_pose = _apply_pose(env, torch, args)
            _settle(env, args.settle_steps)

            image_path = None
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if not args.no_render:
                image_path = args.output_dir / args.image_name
                frame = _render_frame(env, np, args.render_warmup)
                Image.fromarray(frame).save(image_path)

            summary = {
                "task": args.task,
                "image": str(image_path.resolve()) if image_path is not None else None,
                "stage": {
                    "linkerhand_root_prim": stage_info["linkerhand_root"],
                    "screwdriver_root_prim": stage_info["screwdriver_root"],
                    "matching_paths": stage_info["matching_paths"] if args.print_stage_paths else None,
                },
                "articulations": {
                    "linkerhand_cfg_prim_path": env.allegro.cfg.prim_path,
                    "screwdriver_cfg_prim_path": env.screwdriver.cfg.prim_path,
                    "linkerhand_joint_names": list(env.allegro.joint_names),
                    "screwdriver_joint_names": list(env.screwdriver.joint_names),
                    "linkerhand_body_names": list(env.allegro.body_names),
                    "screwdriver_body_names": list(env.screwdriver.body_names),
                },
                "applied_pose": applied_pose,
                "sim_state_after_apply": {
                    "hand_root_pos_w": _tensor_list(env.allegro.data.root_pos_w[0]),
                    "hand_root_quat_wxyz": _tensor_list(env.allegro.data.root_quat_w[0]),
                    "screwdriver_root_pos_w": _tensor_list(env.screwdriver.data.root_pos_w[0]),
                    "screwdriver_root_quat_wxyz": _tensor_list(env.screwdriver.data.root_quat_w[0]),
                    "screwdriver_joint_pos": dict(
                        zip(env.screwdriver.joint_names, _tensor_list(env.screwdriver.data.joint_pos[0]))
                    ),
                    "finger_joint_pos": {
                        name: float(env.allegro.data.joint_pos[0, joint_id].detach().cpu())
                        for name, joint_id in _joint_id_by_name(env.allegro).items()
                        if name in _flatten_joint_targets()
                    },
                    "fingertip_summary": _nearest_screwdriver_body_summary(env, torch),
                },
            }

            summary_path = args.output_dir / args.summary_name
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2))
            print(f"Wrote {summary_path.resolve()}")
            if image_path is not None:
                print(f"Wrote {image_path.resolve()}")
            return summary
        finally:
            env.close()
    finally:
        simulation_app.close()


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = _base_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    _run(args, hydra_args)


if __name__ == "__main__":
    main()
