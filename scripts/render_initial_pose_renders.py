"""Render and summarize the initial screwdriver continuous-turning hand poses.

Outputs, by default:

* outputs/initial_pose_renders/allegro_continuous_initial.png
* outputs/initial_pose_renders/linkerhand_continuous_initial.png
* outputs/initial_pose_renders/initial_pose_summary.json

Run from the repository root with the Isaac Lab environment active, for example:

    python scripts/render_initial_pose_renders.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_TASKS: tuple[str, ...] = (
    "Isaac-Allegro-Screwdriver-Continuous-Turning-Direct-v0",
    "Isaac-LinkerHand-Screwdriver-Continuous-Turning-Direct-v0",
)

IMAGE_NAMES: dict[str, str] = {
    "Isaac-Allegro-Screwdriver-Continuous-Turning-Direct-v0": "allegro_continuous_initial.png",
    "Isaac-LinkerHand-Screwdriver-Continuous-Turning-Direct-v0": "linkerhand_continuous_initial.png",
}


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/initial_pose_renders"),
        help="Directory for the two PNG renders and initial_pose_summary.json.",
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="Gym task IDs to render.")
    parser.add_argument("--width", type=int, default=1280, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=960, help="Render height in pixels.")
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=(0.3, 0.3, 1.5),
        metavar=("X", "Y", "Z"),
        help="World-frame viewer camera eye.",
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=(0.0, -0.05, 1.25),
        metavar=("X", "Y", "Z"),
        help="World-frame viewer camera target.",
    )
    parser.add_argument("--render-warmup", type=int, default=24, help="Maximum render warmup frames.")
    parser.add_argument("--_worker-task", dest="worker_task", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_summary-path", dest="summary_path", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def _resolve_entry_point(entry_point: str) -> Any:
    module_name, attr_name = entry_point.split(":", 1)
    return getattr(importlib.import_module(module_name), attr_name)


def _tensor_list(tensor: Any) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _task_slug(task: str) -> str:
    return task.replace("Isaac-", "").replace("-Direct-v0", "").replace("-", "_")


def _nearest_screwdriver_bodies(env: Any) -> tuple[list[str], list[float]]:
    fingertip_pos = env.allegro.data.body_pos_w[0, env._fingertip_body_ids, :3]
    screwdriver_pos = env.screwdriver.data.body_pos_w[0, env._screwdriver_near_body_ids, :3]
    distances = torch.linalg.norm(fingertip_pos.unsqueeze(1) - screwdriver_pos.unsqueeze(0), dim=-1)
    nearest_distances, nearest_indices = torch.min(distances, dim=-1)
    nearest_names = [env.screwdriver.body_names[env._screwdriver_near_body_ids[int(index)]] for index in nearest_indices]
    return nearest_names, _tensor_list(nearest_distances)


def _summarize_env(env: Any, task: str, image_path: Path) -> dict[str, Any]:
    nearest_names, nearest_distances = _nearest_screwdriver_bodies(env)
    return {
        "task": task,
        "image": str(image_path.resolve()),
        "hand_root_pos_w": _tensor_list(env.allegro.data.root_pos_w[0]),
        "hand_root_quat_wxyz": _tensor_list(env.allegro.data.root_quat_w[0]),
        "screwdriver_root_pos_w": _tensor_list(env.screwdriver.data.root_pos_w[0]),
        "screwdriver_root_quat_wxyz": _tensor_list(env.screwdriver.data.root_quat_w[0]),
        "controlled_fingers": list(env.fingers),
        "controlled_finger_joint_names": [env.allegro.joint_names[joint_id] for joint_id in env._finger_joint_ids],
        "controlled_finger_joint_pos": _tensor_list(env.allegro.data.joint_pos[0, env._finger_joint_ids]),
        "screwdriver_euler_joint_names": [
            env.screwdriver.joint_names[joint_id] for joint_id in env._screwdriver_euler_joint_ids
        ],
        "screwdriver_euler_joint_pos": _tensor_list(
            env.screwdriver.data.joint_pos[0, env._screwdriver_euler_joint_ids]
        ),
        "fingertip_body_names": [env.allegro.body_names[body_id] for body_id in env._fingertip_body_ids],
        "fingertip_pos_w": [
            _tensor_list(env.allegro.data.body_pos_w[0, body_id, :3]) for body_id in env._fingertip_body_ids
        ],
        "fingertip_nearest_screwdriver_body": nearest_names,
        "fingertip_nearest_screwdriver_distance_m": nearest_distances,
    }


def _format_xyz(values: list[float]) -> str:
    return f"x={values[0]:.4f}, y={values[1]:.4f}, z={values[2]:.4f}"


def _coordinate_overlay_lines(summary: dict[str, Any]) -> list[str]:
    task = summary["task"]
    hand_root_pos = summary["hand_root_pos_w"]
    screwdriver_root_pos = summary["screwdriver_root_pos_w"]

    lines = [
        "World-frame XYZ coordinates",
        f"task: {_task_slug(task)}",
        f"hand root: {_format_xyz(hand_root_pos)}",
        f"screwdriver root: {_format_xyz(screwdriver_root_pos)}",
    ]
    for name, pos in zip(summary["fingertip_body_names"], summary["fingertip_pos_w"]):
        lines.append(f"{name}: {_format_xyz(pos)}")
    return lines


def _text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _save_annotated_image(
    frame: Any,
    image_path: Path,
    summary: dict[str, Any],
    image_cls: Any,
    image_draw: Any,
    image_font: Any,
) -> None:
    image = image_cls.fromarray(frame).convert("RGBA")
    overlay = image_cls.new("RGBA", image.size, (0, 0, 0, 0))
    draw = image_draw.Draw(overlay)
    font = image_font.load_default()

    lines = _coordinate_overlay_lines(summary)
    line_sizes = [_text_size(draw, line, font) for line in lines]
    line_spacing = 4
    padding = 12
    margin = 12
    panel_width = max(width for width, _ in line_sizes) + 2 * padding
    panel_height = sum(height for _, height in line_sizes) + line_spacing * (len(lines) - 1) + 2 * padding
    panel_width = min(panel_width, max(1, image.width - 2 * margin))
    panel_height = min(panel_height, max(1, image.height - 2 * margin))

    draw.rectangle(
        (margin, margin, margin + panel_width, margin + panel_height),
        fill=(0, 0, 0, 185),
    )

    x = margin + padding
    y = margin + padding
    max_text_width = panel_width - 2 * padding
    max_text_height = panel_height - 2 * padding
    for index, (line, (_, line_height)) in enumerate(zip(lines, line_sizes)):
        if y + line_height > margin + padding + max_text_height:
            break
        fill = (255, 232, 128, 255) if index == 0 else (255, 255, 255, 255)
        clipped_line = line
        while _text_size(draw, clipped_line, font)[0] > max_text_width and len(clipped_line) > 4:
            clipped_line = clipped_line[:-4] + "..."
        draw.text((x, y), clipped_line, font=font, fill=fill)
        y += line_height + line_spacing

    image_cls.alpha_composite(image, overlay).convert("RGB").save(image_path)


def _render_frame(env: Any, max_warmup_frames: int) -> Any:
    frame = None
    for _ in range(max(1, max_warmup_frames)):
        frame = env.render()
        if frame is not None and frame.size > 0 and np.any(frame):
            return frame
    if frame is None:
        raise RuntimeError("Environment returned no RGB frame.")
    return frame


def _run_worker(args: argparse.Namespace, hydra_args: list[str]) -> dict[str, Any]:
    # Imports that require an active Isaac Sim app stay inside the worker path.
    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]] + hydra_args
    args.headless = True
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        global np, torch
        import gymnasium as gym
        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFont

        import MFR_benchmark.isaac_lab_tasks  # noqa: F401

        task = args.worker_task
        if task is None:
            raise RuntimeError("Worker mode requires --_worker-task.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        image_name = IMAGE_NAMES.get(task, f"{_task_slug(task).lower()}_initial.png")
        image_path = args.output_dir / image_name

        spec = gym.spec(task)
        cfg_cls = _resolve_entry_point(spec.kwargs["env_cfg_entry_point"])
        cfg = cfg_cls()
        cfg.scene.num_envs = 1
        cfg.viewer.resolution = (args.width, args.height)
        cfg.viewer.eye = tuple(args.camera_eye)
        cfg.viewer.lookat = tuple(args.camera_lookat)
        cfg.viewer.origin_type = "world"
        cfg.num_rerenders_on_reset = max(getattr(cfg, "num_rerenders_on_reset", 0), 2)

        env = gym.make(task, cfg=cfg, render_mode="rgb_array").unwrapped
        try:
            env.reset()
            summary = _summarize_env(env, task, image_path)
            frame = _render_frame(env, args.render_warmup)
            _save_annotated_image(frame, image_path, summary, Image, ImageDraw, ImageFont)
            if args.summary_path is not None:
                args.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2))
            return summary
        finally:
            env.close()
    finally:
        simulation_app.close()


def _run_parent(args: argparse.Namespace, forwarded_args: list[str]) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for task in args.tasks:
        summary_path = args.output_dir / f".{_task_slug(task)}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(args.output_dir),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--camera-eye",
            *(str(value) for value in args.camera_eye),
            "--camera-lookat",
            *(str(value) for value in args.camera_lookat),
            "--render-warmup",
            str(args.render_warmup),
            "--_worker-task",
            task,
            "--_summary-path",
            str(summary_path),
            "--headless",
            "--enable_cameras",
            *forwarded_args,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not summary_path.exists():
            print(f"\n--- Worker stdout for {task} ---\n{result.stdout}", file=sys.stderr)
            print(f"--- Worker stderr for {task} ---\n{result.stderr}", file=sys.stderr)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, command, output=result.stdout, stderr=result.stderr
                )
            raise RuntimeError(
                f"Worker exited 0 for {task} but did not write {summary_path}. "
                "See stderr above for the likely exception swallowed by simulation_app.close()."
            )
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        summary_path.unlink()

    summary_file = args.output_dir / "initial_pose_summary.json"
    summary_file.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_file.resolve()}")
    return summaries


def main() -> None:
    parser = _base_parser()
    args, forwarded_args = parser.parse_known_args()
    if args.worker_task is not None:
        from isaaclab.app import AppLauncher

        worker_parser = _base_parser()
        AppLauncher.add_app_launcher_args(worker_parser)
        worker_args, hydra_args = worker_parser.parse_known_args()
        _run_worker(worker_args, hydra_args)
    else:
        _run_parent(args, forwarded_args)


if __name__ == "__main__":
    main()
