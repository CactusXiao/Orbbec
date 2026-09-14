from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from task_backend.optimized_pose_source import load_archive_frame

import cv2
import numpy as np
import torch

from common.episode import load_json
from common.mano import (
    build_mano_layers,
    load_pose,
    mano_faces,
    mano_outputs_from_pose,
    transform_points,
)
from visualizer.draw import draw_joints
from visualizer.pose_mesh import build_scene_meshes, render_scene_meshes_on_image


def load_shape_scale(subject_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the shared MANO shape and optional calibrated subject scale."""
    shape_path = subject_dir / "shape.npy"
    if not shape_path.is_file():
        raise FileNotFoundError(f"shape.npy not found: {shape_path}")
    shape = np.load(shape_path).astype(np.float32).reshape(1, 10).repeat(2, axis=0)

    scale_path = subject_dir / "scale.npy"
    scale_value = 1.0 if not scale_path.is_file() else float(
        np.load(scale_path).reshape(-1)[0]
    )
    scales = np.asarray([[scale_value], [scale_value]], dtype=np.float32)
    return shape, scales


def intrinsic_matrix(entry: dict) -> np.ndarray:
    """Convert an intrinsic metadata entry into a pinhole camera matrix."""
    matrix = np.eye(3, dtype=np.float32)
    matrix[0, 0] = float(entry["fx"])
    matrix[1, 1] = float(entry["fy"])
    matrix[0, 2] = float(entry["cx"])
    matrix[1, 2] = float(entry["cy"])
    return matrix


def load_ego_camera(
    episode_dir: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load the original ego fisheye calibration used by the raw RGB frames."""
    camera_path = episode_dir / "camera_params.json"
    camera_data = load_json(camera_path)
    try:
        rgb = camera_data["ego"]["RGB"]
        intrinsic = rgb["intrinsic"]
        distortion = rgb["distortion"]
    except KeyError as error:
        raise KeyError(f"missing ego calibration field {error} in {camera_path}") from error

    if distortion.get("modelName") != "opencv_fisheye":
        raise ValueError(
            f"ego distortion model must be opencv_fisheye: {camera_path}"
        )

    matrix = intrinsic_matrix(intrinsic)
    coefficients = np.asarray(
        [distortion[f"k{index}"] for index in range(1, 5)], dtype=np.float32
    ).reshape(4, 1)
    image_size = (int(intrinsic["width"]), int(intrinsic["height"]))
    return matrix, coefficients, image_size


def build_fisheye_render_map(
    image_size: tuple[int, int],
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw fisheye pixels to the pinhole mesh layer without changing the RGB image."""
    width, height = image_size
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    fisheye_pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
    pinhole_pixels = cv2.fisheye.undistortPoints(
        fisheye_pixels, intrinsic, distortion, P=intrinsic
    )
    pinhole_pixels = pinhole_pixels.reshape(height, width, 2)
    return pinhole_pixels[..., 0], pinhole_pixels[..., 1]


def load_ego_extrinsics(episode_dir: Path) -> dict[int, np.ndarray]:
    """Load per-frame camera0-to-ego transforms from ego_extrinsic.json."""
    extrinsic_path = episode_dir / "ego_extrinsic.json"
    metadata = load_json(extrinsic_path)
    transforms: dict[int, np.ndarray] = {}
    for frame_text, value in metadata.items():
        if not str(frame_text).isdigit():
            raise ValueError(f"invalid ego extrinsic frame key: {frame_text}")
        frame = int(frame_text)
        if frame in transforms:
            raise ValueError(f"duplicate ego extrinsic frame: {frame}")
        transform = np.asarray(value, dtype=np.float32)
        if transform.shape != (4, 4):
            raise ValueError(
                f"ego extrinsic must have shape (4, 4), got {transform.shape} "
                f"for frame {frame}"
            )
        transforms[frame] = transform
    if not transforms:
        raise ValueError(f"no ego extrinsics found in {extrinsic_path}")
    return transforms


def select_frames(
    transforms: dict[int, np.ndarray], start: int | None, end: int | None
) -> list[int]:
    """Select available extrinsic frames inside the inclusive requested range."""
    if start is not None and start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if end is not None and end < 0:
        raise ValueError(f"end must be non-negative, got {end}")
    if start is not None and end is not None and start > end:
        raise ValueError(f"start must not exceed end, got start={start}, end={end}")

    frames = [
        frame
        for frame in sorted(transforms)
        if (start is None or frame >= start) and (end is None or frame <= end)
    ]
    if not frames:
        raise ValueError(f"no ego extrinsic frames found in range [{start}, {end}]")
    return frames


def project_frame_joints(
    outputs: dict[int, dict[str, torch.Tensor]],
    transform: torch.Tensor,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Transform MANO joints and project them with the original ego fisheye model."""
    points = np.full((2, 21, 3), -1.0, dtype=np.float32)
    rotation = np.zeros((3, 1), dtype=np.float32)
    translation = np.zeros((3, 1), dtype=np.float32)
    for hand in (0, 1):
        joints_camera = (
            transform_points(outputs[hand]["joints"][0], transform)
            .detach()
            .cpu()
            .numpy()
        )
        valid = np.isfinite(joints_camera).all(axis=1) & (joints_camera[:, 2] > 0.0)
        if not valid.any():
            continue
        projected, _ = cv2.fisheye.projectPoints(
            joints_camera[valid].reshape(-1, 1, 3),
            rotation,
            translation,
            intrinsic,
            distortion,
        )
        points[hand, valid, :2] = projected.reshape(-1, 2)
        points[hand, valid, 2] = 1.0
    return points


def render_kp2d(
    image: np.ndarray,
    outputs: dict[int, dict[str, torch.Tensor]],
    transform: torch.Tensor,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Draw both fisheye-projected MANO skeletons on an unmodified ego RGB frame."""
    points = project_frame_joints(outputs, transform, intrinsic, distortion)
    rendered = image.copy()
    for hand in (0, 1):
        draw_joints(rendered, points[hand], hand, 0.5)
    return rendered


def run(args: argparse.Namespace) -> None:
    """Load one episode and save ego overlays for its calibrated frames."""
    episode_value = Path(args.episode)
    if episode_value.is_absolute():
        raise ValueError("--episode must be relative to the subject directory")

    data_root = Path(args.data_root)
    subject_dir = data_root / args.subject
    episode_dir = subject_dir / episode_value
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode directory not found: {episode_dir}")

    pose_dir = episode_dir / "optimized_pose"
    rgb_dir = episode_dir / "ego" / "RGB"
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"optimized_pose directory not found: {pose_dir}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"ego RGB directory not found: {rgb_dir}")

    transforms = load_ego_extrinsics(episode_dir)
    frames = select_frames(transforms, args.start, args.end)
    intrinsic, distortion, image_size = load_ego_camera(episode_dir)

    betas, scales = load_shape_scale(subject_dir)
    mano_dir = Path(__file__).resolve().parents[1] / "ckpt" / "mano"
    layers = build_mano_layers(mano_dir)
    faces = {hand: mano_faces(layers[hand]) for hand in (0, 1)}
    intrinsic_tensor = torch.tensor(intrinsic, dtype=torch.float32)
    fisheye_map = build_fisheye_render_map(image_size, intrinsic, distortion)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        pose = (load_archive_frame(pose_dir, frame) if (pose_dir / "poses.npz").is_file()
                else load_pose(pose_dir / f"{frame:05d}.npy"))
        image_path = rgb_dir / f"{frame:05d}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read ego RGB frame: {image_path}")
        if (image.shape[1], image.shape[0]) != image_size:
            raise ValueError(
                f"ego RGB size mismatch for frame {frame}: "
                f"expected {image_size}, got {(image.shape[1], image.shape[0])}"
            )

        outputs = mano_outputs_from_pose(pose, betas, scales, layers)
        transform = torch.tensor(transforms[frame], dtype=torch.float32)
        if args.type == "mesh":
            scene_meshes = build_scene_meshes(outputs, faces, transform)
            rendered = render_scene_meshes_on_image(
                image, scene_meshes, intrinsic_tensor, fisheye_map=fisheye_map
            )
        else:
            rendered = render_kp2d(image, outputs, transform, intrinsic, distortion)

        output_path = output_dir / f"{frame:05d}.jpg"
        if not cv2.imwrite(str(output_path), rendered):
            raise RuntimeError(f"failed to write visualization: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse the shell entrypoint arguments."""
    parser = argparse.ArgumentParser(
        description="Overlay optimized MANO poses on calibrated ego RGB frames."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--episode",
        required=True,
        help="Episode path relative to the subject, for example hand_shape_calibration/episode_1.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=None, help="Inclusive start frame.")
    parser.add_argument("--end", type=int, default=None, help="Inclusive end frame.")
    parser.add_argument("--type", choices=("mesh", "kp2d"), default="mesh")
    return parser.parse_args()


def main() -> None:
    """Run the ego pose visualizer from the command line."""
    run(parse_args())


if __name__ == "__main__":
    main()
