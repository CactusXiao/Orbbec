from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import smplx


# chumpy is only needed while unpickling the legacy MANO model files. Patch
# APIs removed by modern Python/NumPy before smplx imports a MANO pickle.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
for _name, _value in {
    "bool": np.bool_,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
TIP_VERTEX_IDS = (745, 317, 444, 556, 673)
MANO_TO_HAND21 = (
    0, 13, 14, 15, 16,
    1, 2, 3, 17,
    4, 5, 6, 18,
    10, 11, 12, 19,
    7, 8, 9, 20,
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class SequentialEgoRGBSource:
    """Read increasing ego RGB frames from numbered images or a video file."""

    def __init__(self, rgb_dir: Path):
        self.rgb_dir = rgb_dir
        self.current_index = -1
        self._last_frame: np.ndarray | None = None
        self._capture: cv2.VideoCapture | None = None
        video_path = rgb_dir / "rgb.h265"
        if video_path.is_file():
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"failed to open ego RGB video: {video_path}")
            self.kind = "video"
            self.source_path = video_path
            self._capture = capture
            self._images: dict[int, Path] = {}
            return

        self._images = {
            int(path.stem): path
            for path in rgb_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.stem.isdigit()
        }
        if self._images:
            self.kind = "image_sequence"
            self.source_path = rgb_dir
            return

        raise FileNotFoundError(
            f"no standardized ego RGB source found: expected {video_path} "
            f"or numbered images in {rgb_dir}"
        )

    def read(self, frame_index: int) -> np.ndarray | None:
        """Read one frame while preserving efficient sequential video decoding."""
        target = int(frame_index)
        if target < self.current_index:
            raise ValueError(
                f"ego RGB frame access must be increasing: requested {target}, "
                f"current {self.current_index}"
            )
        if self.kind == "image_sequence":
            self.current_index = target
            image_path = self._images.get(target)
            return cv2.imread(str(image_path)) if image_path is not None else None

        if self._capture is None:
            raise RuntimeError("ego RGB video source is closed")
        if target == self.current_index:
            return None if self._last_frame is None else self._last_frame.copy()
        frame: np.ndarray | None = None
        while self.current_index < target:
            ok, frame = self._capture.read()
            self.current_index += 1
            if not ok or frame is None:
                return None
        self._last_frame = frame
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._last_frame = None


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    import json

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_pose(path: Path) -> np.ndarray:
    """Load one pair of 16-joint 6D MANO poses plus translations."""
    if not path.is_file():
        raise FileNotFoundError(f"optimized pose not found: {path}")
    pose = np.load(path).astype(np.float32)
    if pose.shape != (2, 99):
        raise ValueError(f"optimized pose must have shape (2, 99), got {pose.shape}: {path}")
    return pose


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert Zhou et al. 6D rotations to orthonormal 3x3 matrices."""
    value = np.asarray(rotation_6d, dtype=np.float64).reshape(-1, 6)
    first = value[:, :3]
    second = value[:, 3:]
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    # The stored representation follows PyTorch3D/Zhou et al.: the two
    # 3-vectors are the first two rows of the rotation matrix.
    return np.stack((first, second, third), axis=-2).reshape(-1, 3, 3)


def matrix_to_axis_angle(rotations: np.ndarray) -> np.ndarray:
    """Convert a batch of rotation matrices to OpenCV axis-angle vectors."""
    return np.stack(
        [cv2.Rodrigues(rotation)[0].reshape(3) for rotation in rotations], axis=0
    ).astype(np.float32)


def resolve_mano_dir(explicit_path: str | None) -> Path:
    """Find the MANO_LEFT.pkl and MANO_RIGHT.pkl model pair."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    script_root = Path(__file__).resolve().parents[1]
    candidates.extend((script_root / "ckpt" / "mano", script_root / "orbbec" / "mano"))
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "MANO_LEFT.pkl").is_file() and (resolved / "MANO_RIGHT.pkl").is_file():
            return resolved
    raise FileNotFoundError(
        "MANO models were not found; pass --mano-dir containing MANO_LEFT.pkl and MANO_RIGHT.pkl"
    )


def build_mano_layers(mano_dir: Path) -> dict[int, smplx.MANO]:
    """Build left/right non-PCA MANO layers from the local model files."""
    return {
        0: smplx.MANO(
            str(mano_dir / "MANO_LEFT.pkl"),
            is_rhand=False,
            use_pca=False,
            flat_hand_mean=True,
        ),
        1: smplx.MANO(
            str(mano_dir / "MANO_RIGHT.pkl"),
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=True,
        ),
    }


def mano_faces(layer: smplx.MANO) -> np.ndarray:
    return np.asarray(layer.faces, dtype=np.int32)


def mano_outputs_from_pose(
    pose: np.ndarray,
    betas: np.ndarray,
    scales: np.ndarray,
    layers: dict[int, smplx.MANO],
) -> dict[int, dict[str, torch.Tensor]]:
    """Evaluate both MANO hands in reference-camera coordinates."""
    outputs: dict[int, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for hand in (0, 1):
            rotations = rotation_6d_to_matrix(pose[hand, :96]).reshape(16, 3, 3)
            axis_angles = matrix_to_axis_angle(rotations)
            layer_output = layers[hand](
                global_orient=torch.from_numpy(axis_angles[:1]),
                hand_pose=torch.from_numpy(axis_angles[1:].reshape(1, 45)),
                betas=torch.from_numpy(betas[hand : hand + 1]),
            )
            scale = float(scales[hand, 0])
            translation = torch.from_numpy(pose[hand, 96:99]).reshape(1, 1, 3)
            vertices = layer_output.vertices * scale + translation
            base_joints = layer_output.joints[:, :16] * scale + translation
            tips = vertices[:, TIP_VERTEX_IDS]
            joints_21 = torch.cat((base_joints, tips), dim=1)[:, MANO_TO_HAND21]
            outputs[hand] = {"vertices": vertices, "joints": joints_21}
    return outputs


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply p_target = T_target_from_source * p_source to Nx3 points."""
    return points @ transform[:3, :3].T + transform[:3, 3]


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
    camera_params: str | Path | None = None,
    fisheye_calibration: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load the original ego fisheye calibration used by the raw RGB frames."""
    if fisheye_calibration is not None:
        calibration_path = Path(fisheye_calibration).expanduser().resolve()
        if not calibration_path.is_file():
            raise FileNotFoundError(f"fisheye calibration not found: {calibration_path}")
        with np.load(calibration_path) as calibration:
            required = {"K", "D", "image_size"}
            missing = sorted(required - set(calibration.files))
            if missing:
                raise KeyError(
                    f"fisheye calibration missing {', '.join(missing)}: {calibration_path}"
                )
            matrix = np.asarray(calibration["K"], dtype=np.float32).reshape(3, 3)
            coefficients = np.asarray(calibration["D"], dtype=np.float32).reshape(4, 1)
            size_values = np.asarray(calibration["image_size"]).reshape(-1)
            if size_values.size != 2:
                raise ValueError(
                    f"fisheye image_size must contain width and height: {calibration_path}"
                )
            image_size = (int(size_values[0]), int(size_values[1]))
        return matrix, coefficients, image_size

    camera_path = (
        Path(camera_params).expanduser().resolve()
        if camera_params is not None
        else episode_dir / "camera_params.json"
    )
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


def load_reference_to_ego_frames(episode_dir: Path) -> dict[int, int]:
    """Load the synchronized reference-frame to raw PICO-frame mapping."""
    timestamps_path = episode_dir / "timestamps.csv"
    if not timestamps_path.is_file():
        raise FileNotFoundError(f"timestamps.csv not found: {timestamps_path}")

    mapping: dict[int, int] = {}
    with timestamps_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            reference_text = row.get("frame_index", "").strip()
            ego_text = row.get("ego_frame_index", "").strip()
            if not reference_text or not ego_text:
                continue
            reference_frame = int(reference_text)
            ego_frame = int(ego_text)
            if reference_frame in mapping and mapping[reference_frame] != ego_frame:
                raise ValueError(
                    f"duplicate ego mapping for reference frame {reference_frame}: "
                    f"{mapping[reference_frame]} and {ego_frame}"
                )
            mapping[reference_frame] = ego_frame
    if not mapping:
        raise ValueError(f"no aligned ego frames found in {timestamps_path}")
    return mapping


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


def load_ego_extrinsics(
    episode_dir: Path, explicit_path: str | None = None
) -> tuple[dict[int, np.ndarray], Path]:
    """Load per-frame camera0-to-ego transforms from JSON."""
    extrinsic_path = (
        Path(explicit_path).expanduser().resolve()
        if explicit_path
        else episode_dir / "ego_extrinsic.json"
    )
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
    return transforms, extrinsic_path


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


def draw_joints(
    image: np.ndarray,
    points: np.ndarray,
    hand: int,
    confidence_threshold: float,
) -> None:
    """Draw one 21-joint hand skeleton in-place."""
    color = (70, 170, 255) if hand == 0 else (255, 145, 70)
    valid = (
        np.isfinite(points[:, :2]).all(axis=1)
        & (points[:, 2] >= float(confidence_threshold))
    )
    for start, end in HAND_CONNECTIONS:
        if valid[start] and valid[end]:
            cv2.line(
                image,
                tuple(np.rint(points[start, :2]).astype(int)),
                tuple(np.rint(points[end, :2]).astype(int)),
                color,
                4,
                cv2.LINE_AA,
            )
    for index, point in enumerate(points):
        if not valid[index]:
            continue
        center = tuple(np.rint(point[:2]).astype(int))
        cv2.circle(image, center, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, center, 4, color, -1, cv2.LINE_AA)


def project_fisheye_points(
    points_camera: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-camera 3D vertices to raw fisheye pixels."""
    points_camera = np.asarray(points_camera, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1e-4)
    pixels = np.full((len(points_camera), 2), np.nan, dtype=np.float32)
    if valid.any():
        projected, _ = cv2.fisheye.projectPoints(
            points_camera[valid].reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            intrinsic,
            distortion,
        )
        pixels[valid] = projected.reshape(-1, 2)
    return pixels, valid


def render_mesh(
    image: np.ndarray,
    outputs: dict[int, dict[str, torch.Tensor]],
    faces: dict[int, np.ndarray],
    transform: torch.Tensor,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Render both MANO meshes with a lightweight painter-style rasterizer."""
    triangles: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    base_colors = {0: np.array([70, 170, 255]), 1: np.array([255, 145, 70])}
    height, width = image.shape[:2]

    for hand in (0, 1):
        vertices_camera = (
            transform_points(outputs[hand]["vertices"][0], transform)
            .detach()
            .cpu()
            .numpy()
        )
        pixels, valid_vertices = project_fisheye_points(
            vertices_camera, intrinsic, distortion
        )
        hand_faces = faces[hand]
        valid_faces = valid_vertices[hand_faces].all(axis=1)
        for face in hand_faces[valid_faces]:
            polygon = pixels[face]
            if (
                np.max(polygon[:, 0]) < 0
                or np.max(polygon[:, 1]) < 0
                or np.min(polygon[:, 0]) >= width
                or np.min(polygon[:, 1]) >= height
            ):
                continue
            vertices = vertices_camera[face]
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            norm = float(np.linalg.norm(normal))
            facing = abs(float(normal[2])) / norm if norm > 1e-9 else 0.0
            shade = 0.48 + 0.52 * facing
            color = tuple(int(value) for value in np.clip(base_colors[hand] * shade, 0, 255))
            triangles.append(
                (float(np.mean(vertices[:, 2])), np.rint(polygon).astype(np.int32), color)
            )

    overlay = image.copy()
    for _, polygon, color in sorted(triangles, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(overlay, polygon, color, cv2.LINE_AA)
        cv2.polylines(overlay, [polygon], True, (45, 45, 45), 1, cv2.LINE_AA)
    rendered = cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0)
    return render_kp2d(rendered, outputs, transform, intrinsic, distortion)


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

    transforms, extrinsic_path = load_ego_extrinsics(episode_dir, args.extrinsics)
    frames = select_frames(transforms, args.start, args.end)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    intrinsic, distortion, image_size = load_ego_camera(
        episode_dir,
        args.camera_params,
        args.fisheye_calibration,
    )
    reference_to_ego = load_reference_to_ego_frames(episode_dir)
    missing_mappings = [frame for frame in frames if frame not in reference_to_ego]
    frames = [frame for frame in frames if frame in reference_to_ego]
    if not frames:
        raise ValueError("none of the selected extrinsic frames has an aligned PICO frame")
    if missing_mappings:
        print(f"[ego_pose] skipped_unaligned_reference_frames={len(missing_mappings)}")

    shape_dir = Path(args.shape_dir).expanduser().resolve() if args.shape_dir else subject_dir
    betas, scales = load_shape_scale(shape_dir)
    mano_dir = resolve_mano_dir(args.mano_dir)
    layers = build_mano_layers(mano_dir)
    faces = {hand: mano_faces(layers[hand]) for hand in (0, 1)}
    rgb_source = SequentialEgoRGBSource(rgb_dir)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = Path(args.output_video).expanduser().resolve() if args.output_video else None
    writer: cv2.VideoWriter | None = None
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps),
            image_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open output video: {output_video}")

    print(f"[ego_pose] episode={episode_dir.resolve()}")
    print(f"[ego_pose] extrinsics={extrinsic_path}")
    print("[ego_pose] ego_frame_mapping=timestamps.csv:frame_index->ego_frame_index")
    if args.fisheye_calibration:
        print(
            "[ego_pose] fisheye_calibration="
            f"{Path(args.fisheye_calibration).expanduser().resolve()}"
        )
    print(f"[ego_pose] mano_dir={mano_dir}")
    print(f"[ego_pose] rgb_source={rgb_source.source_path.resolve()} ({rgb_source.kind})")
    print(f"[ego_pose] type={args.type} frames={len(frames)} range={frames[0]}..{frames[-1]}")
    extrinsic_label = (
        extrinsic_path.parent.name
        if extrinsic_path.parent != episode_dir
        else extrinsic_path.stem
    )
    try:
        for output_index, frame in enumerate(frames, start=1):
            pose = load_pose(pose_dir / f"{frame:05d}.npy")
            ego_frame = reference_to_ego[frame]
            image = rgb_source.read(ego_frame)
            if image is None:
                raise RuntimeError(
                    f"failed to read ego RGB frame {ego_frame} from {rgb_source.source_path}"
                )
            if (image.shape[1], image.shape[0]) != image_size:
                raise ValueError(
                    f"ego RGB size mismatch for reference frame {frame}, ego frame {ego_frame}: "
                    f"expected {image_size}, got {(image.shape[1], image.shape[0])}"
                )

            outputs = mano_outputs_from_pose(pose, betas, scales, layers)
            transform = torch.tensor(transforms[frame], dtype=torch.float32)
            if not torch.isfinite(transform).all():
                raise ValueError(f"non-finite ego extrinsic for frame {frame}: {extrinsic_path}")
            if args.type == "mesh":
                rendered = render_mesh(
                    image, outputs, faces, transform, intrinsic, distortion
                )
            else:
                rendered = render_kp2d(
                    image, outputs, transform, intrinsic, distortion
                )

            cv2.putText(
                rendered,
                f"ref={frame:05d}  ego={ego_frame:05d}  extrinsics={extrinsic_label}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                rendered,
                f"ref={frame:05d}  ego={ego_frame:05d}  extrinsics={extrinsic_label}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            if not args.video_only:
                output_path = output_dir / f"{frame:05d}.jpg"
                if not cv2.imwrite(str(output_path), rendered):
                    raise RuntimeError(f"failed to write visualization: {output_path}")
            if writer is not None:
                writer.write(rendered)
            if output_index == 1 or output_index % 25 == 0 or output_index == len(frames):
                print(
                    f"[ego_pose] rendered={output_index}/{len(frames)} "
                    f"reference={frame:05d} ego={ego_frame:05d}"
                )
    finally:
        rgb_source.close()
        if writer is not None:
            writer.release()

    print(f"[ego_pose] output_dir={output_dir}")
    if output_video is not None:
        print(f"[ego_pose] output_video={output_video}")


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
    parser.add_argument(
        "--extrinsics",
        default=None,
        help="Per-frame camera0-to-ego JSON; defaults to <episode>/ego_extrinsic.json.",
    )
    parser.add_argument(
        "--mano-dir",
        default=None,
        help="Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl.",
    )
    parser.add_argument(
        "--shape-dir",
        default=None,
        help="Directory containing shape.npy and optional scale.npy; defaults to the subject directory.",
    )
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument(
        "--camera-params",
        default=None,
        help="Camera parameter JSON containing ego.RGB; defaults to <episode>/camera_params.json.",
    )
    calibration_group.add_argument(
        "--fisheye-calibration",
        default=None,
        help="OpenCV fisheye NPZ containing K, D and image_size for the raw ego RGB frames.",
    )
    parser.add_argument("--output-video", default=None)
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Write only --output-video and skip individual JPEG frames.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start", type=int, default=None, help="Inclusive start frame.")
    parser.add_argument("--end", type=int, default=None, help="Inclusive end frame.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--type", choices=("mesh", "kp2d"), default="mesh")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be >= 1")
    if args.video_only and not args.output_video:
        parser.error("--video-only requires --output-video")
    return args


def main() -> None:
    """Run the ego pose visualizer from the command line."""
    run(parse_args())


if __name__ == "__main__":
    main()
