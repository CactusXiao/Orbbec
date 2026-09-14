from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from task_backend.optimized_pose_source import load_archive_frame

import cv2
import numpy as np
import torch

from common.config import load_config
from common.episode import first_camera_extrinsics, frame_path, load_extrinsics, load_intrinsics, normalize_view_name
from common.mano import build_mano_layers, load_pose, mano_outputs_from_pose, project_points, transform_points
from visualizer.config import parse_visual_config
from visualizer.draw import draw_joints


def _load_shape_scale(subject_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    shape_path = subject_dir / "shape.npy"
    if not shape_path.is_file():
        raise FileNotFoundError(f"shape.npy not found: {shape_path}")
    shape = np.load(shape_path).astype(np.float32).reshape(1, 10).repeat(2, axis=0)
    scale_path = subject_dir / "scale.npy"
    scale_value = 1.0 if not scale_path.is_file() else float(np.load(scale_path).reshape(-1)[0])
    return shape, np.asarray([[scale_value], [scale_value]], dtype=np.float32)


def _project_frame_points(pose: np.ndarray, betas: np.ndarray, scales: np.ndarray, layers: dict[int, object], T_world_to_camera: torch.Tensor, K: torch.Tensor) -> np.ndarray:
    outputs = mano_outputs_from_pose(pose, betas, scales, layers)
    points = np.full((2, 21, 3), -1.0, dtype=np.float32)
    for hand in (0, 1):
        joints_camera = transform_points(outputs[hand]["joints"][0], T_world_to_camera)
        uv, valid = project_points(joints_camera, K)
        points[hand, valid.numpy(), :2] = uv[valid].numpy()
        points[hand, valid.numpy(), 2] = 1.0
    return points


def run(config: dict) -> None:
    visual_config = parse_visual_config(config, ("pose_dir", "camera_stream", "mano_dir"))
    spec = visual_config.spec
    views = visual_config.views
    frames = visual_config.frames
    intrinsics = load_intrinsics(spec.episode_dir, views, config["camera_stream"])
    reference_view = normalize_view_name(config.get("camera_reference_view", 0))
    extrinsic_views = list(dict.fromkeys([reference_view, *views]))
    extrinsics = first_camera_extrinsics(load_extrinsics(spec.episode_dir, extrinsic_views), views, reference_view)
    betas, scales = _load_shape_scale(spec.subject_dir)
    layers = build_mano_layers(config["mano_dir"])
    out_root = visual_config.output_dir
    for frame in frames:
        pose_dir = spec.episode_dir / config["pose_dir"]
        pose = (load_archive_frame(pose_dir, frame) if (pose_dir / "poses.npz").is_file()
                else load_pose(pose_dir / f"{frame:05d}.npy"))
        for view in views:
            image = cv2.imread(str(frame_path(spec.episode_dir, view, "RGB", frame, config["image_ext"])))
            if image is None:
                raise RuntimeError(f"failed to read image for view={view} frame={frame}")
            points = _project_frame_points(
                pose,
                betas,
                scales,
                layers,
                torch.tensor(extrinsics[view], dtype=torch.float32),
                torch.tensor(intrinsics[view], dtype=torch.float32),
            )
            for hand in (0, 1):
                draw_joints(image, points[hand], hand, 0.5)
            out_dir = out_root / "pose_2d" / view
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{frame:05d}.jpg"), image)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
