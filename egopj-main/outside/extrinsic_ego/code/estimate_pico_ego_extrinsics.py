#!/usr/bin/env python3
"""
Estimate PICO ego RGB camera extrinsics against fixed third-person cameras.

The output transform follows the same convention as the existing extrinsics.json:

    p_ego = T_ego_from_reference * p_reference

The default reference camera is camera 00.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EPISODE_DIR = (
    SCRIPT_DIR.parent / "test_sample_final" / "hand_shape_calibration" / "episode_1"
)
DEFAULT_FISHEYE_CALIBRATION = SCRIPT_DIR.parent.parent / "camera_info" / "fisheye_calibration_result.npz"

APRILTAG_FAMILY = "tag36h11"
APRILTAG_SIZE_M = 0.096
REFERENCE_FRAME_LIMIT = 10
MIN_REFERENCE_TAGS = 2
MIN_REFERENCE_INLIER_OBSERVATIONS = 3
REFERENCE_TAG_REPROJ_RMSE_LIMIT_PX = 2.0
MIN_EGO_TAGS = 2
MIN_EGO_INLIER_CORNERS = 8
EGO_RANSAC_REPROJ_ERROR_PX = 2.0
EGO_PNP_ITERATIONS = 200
EGO_REPROJ_RMSE_LIMIT_PX = 4.0
SMOOTHING_WINDOW_LENGTH = 11
SMOOTHING_POLYORDER = 3
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

APRILTAG_DICT_NAMES = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}


@dataclass
class CameraCalibration:
    camera_id: str
    K: np.ndarray
    dist: np.ndarray
    T_camera_from_world: np.ndarray
    T_camera_from_reference: np.ndarray | None = None
    T_reference_from_camera: np.ndarray | None = None
    video_path: Path | None = None


@dataclass
class TimestampRow:
    row_index: int
    frame_index: str
    frame_number: int
    ref_timestamp_us: str
    ego_frame_index: str
    ego_frame_number: int | None
    ego_timestamp_us: str
    raw: dict[str, str]


@dataclass
class TagFusionResult:
    tag_id: int
    T_reference_from_tag: np.ndarray
    corners_reference: np.ndarray
    detection_count: int
    candidate_count: int
    inlier_count: int
    reprojection_rmse_px: float


@dataclass
class EgoImageModel:
    enabled: bool
    source: str
    K_raw: np.ndarray
    D_fisheye: np.ndarray | None
    K_pnp: np.ndarray
    dist_pnp: np.ndarray
    image_size: tuple[int, int]
    map1: np.ndarray | None = None
    map2: np.ndarray | None = None


@dataclass
class FrameEstimate:
    row: TimestampRow
    detected_tag_ids: list[int] = field(default_factory=list)
    used_tag_ids: list[int] = field(default_factory=list)
    rmse_px: float | None = None
    pose_direct: np.ndarray | None = None
    pose_final: np.ndarray | None = None
    status_initial: str = ""
    status_final: str = ""
    source: str = "nan"
    detection_space: str = ""
    prev_direct_frame_index: str = ""
    next_direct_frame_index: str = ""
    interp_alpha: float | None = None
    interp_gap_frames: int | None = None
    pose_smoothed: np.ndarray | None = None
    smoothing_translation_delta_m: float = 0.0
    smoothing_rotation_delta_deg: float = 0.0
    smoothing_status: str = "disabled"


class SequentialVideoFrameSource:
    """Read increasing frames from raw video or a synchronized image directory."""

    def __init__(self, source_path: Path, label: str, *, force_temp_remux: bool = False):
        self.source_path = Path(source_path)
        self.label = label
        self.force_temp_remux = force_temp_remux
        self.active_path = self.source_path
        self.used_temp_remux = False
        self.current_index = -1
        self.is_image_sequence = False
        self._image_paths: dict[int, Path] = {}
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._cap: cv2.VideoCapture | None = None
        self._open()

    def _open(self) -> None:
        if self.source_path.is_dir():
            self._image_paths = {
                int(path.stem): path
                for path in self.source_path.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.stem.isdigit()
            }
            if not self._image_paths:
                raise FileNotFoundError(
                    f"No numbered images found for {self.label}: {self.source_path}"
                )
            self.is_image_sequence = True
            self.active_path = self.source_path
            self.current_index = -1
            return

        if not self.source_path.is_file():
            raise FileNotFoundError(f"Missing frame source for {self.label}: {self.source_path}")

        if self.force_temp_remux:
            self._remux_to_temp()
        elif not self._probe_video(self.source_path):
            self._remux_to_temp()

        self._cap = cv2.VideoCapture(str(self.active_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video for {self.label}: {self.active_path}")
        self.current_index = -1

    @staticmethod
    def _probe_video(path: Path) -> bool:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return False
            ok, frame = cap.read()
            return bool(ok and frame is not None)
        finally:
            cap.release()

    def _remux_to_temp(self) -> None:
        ffmpeg = _find_ffmpeg()
        self._temp_dir = tempfile.TemporaryDirectory(prefix=f"pico_ego_{self.label}_")
        output_path = Path(self._temp_dir.name) / f"{self.source_path.stem}.mp4"
        commands = [
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "hevc", "-i", str(self.source_path), "-c", "copy", str(output_path)],
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(self.source_path), "-c", "copy", str(output_path)],
        ]
        last_error = ""
        for command in commands:
            proc = subprocess.run(command, capture_output=True, text=True)
            if proc.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
                self.active_path = output_path
                self.used_temp_remux = True
                return
            last_error = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Failed to remux {self.source_path} for {self.label}: {last_error}")

    def read(self, frame_index: int) -> np.ndarray | None:
        target = int(frame_index)
        if target < self.current_index:
            raise ValueError(
                f"{self.label} frame access must be increasing: requested {target}, "
                f"current {self.current_index}"
            )
        if self.is_image_sequence:
            self.current_index = target
            image_path = self._image_paths.get(target)
            return cv2.imread(str(image_path)) if image_path is not None else None

        if self._cap is None:
            raise RuntimeError("Video source is closed")
        frame = None
        while self.current_index < target:
            ok, frame = self._cap.read()
            self.current_index += 1
            if not ok or frame is None:
                return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


def _find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("ffmpeg was not found on PATH and imageio_ffmpeg is unavailable") from exc


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _format_camera_id(camera_id: str | int) -> str:
    text = str(camera_id).strip()
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def _camera_sort_key(camera_id: str) -> tuple[int, int | str]:
    return (0, int(camera_id)) if camera_id.isdigit() else (1, camera_id)


def _parse_camera_id_list(raw: str) -> list[str]:
    return [_format_camera_id(item) for item in raw.split(",") if item.strip()]


def _parse_tag_id_list(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    values = {int(item.strip()) for item in raw.split(",") if item.strip()}
    return values or None


def _resolve_json_entry(data: dict[str, Any], camera_id: str) -> dict[str, Any]:
    candidates = [camera_id, _format_camera_id(camera_id)]
    if str(camera_id).isdigit():
        candidates.extend([str(int(camera_id)), f"{int(camera_id):02d}"])
    for key in dict.fromkeys(candidates):
        entry = data.get(key)
        if isinstance(entry, dict):
            return entry
    raise KeyError(f"Missing JSON entry for camera {camera_id}")


def _extract_intrinsics(entry: dict[str, Any], camera_id: str) -> tuple[np.ndarray, np.ndarray, str]:
    rgb = entry.get("RGB")
    if not isinstance(rgb, dict):
        raise KeyError(f"Camera {camera_id} missing RGB block")
    intrinsic = rgb.get("intrinsic")
    distortion = rgb.get("distortion", {})
    if not isinstance(intrinsic, dict):
        raise KeyError(f"Camera {camera_id} missing RGB intrinsic")
    if not isinstance(distortion, dict):
        distortion = {}

    K = np.array(
        [
            [float(intrinsic["fx"]), 0.0, float(intrinsic["cx"])],
            [0.0, float(intrinsic["fy"]), float(intrinsic["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    model = str(distortion.get("modelName", distortion.get("model", "opencv"))).lower()
    if "fisheye" in model:
        dist = np.array(
            [
                float(distortion.get("k1", 0.0)),
                float(distortion.get("k2", 0.0)),
                float(distortion.get("k3", 0.0)),
                float(distortion.get("k4", 0.0)),
            ],
            dtype=np.float64,
        ).reshape(4, 1)
    else:
        dist = np.array(
            [
                float(distortion.get("k1", 0.0)),
                float(distortion.get("k2", 0.0)),
                float(distortion.get("p1", 0.0)),
                float(distortion.get("p2", 0.0)),
                float(distortion.get("k3", 0.0)),
                float(distortion.get("k4", 0.0)),
                float(distortion.get("k5", 0.0)),
                float(distortion.get("k6", 0.0)),
            ],
            dtype=np.float64,
        )
    return K, dist, model


def _extract_image_size(entry: dict[str, Any], camera_id: str) -> tuple[int, int]:
    rgb = entry.get("RGB")
    if not isinstance(rgb, dict):
        raise KeyError(f"Camera {camera_id} missing RGB block")
    return int(rgb["width"]), int(rgb["height"])


def _extract_extrinsic_camera_from_world(entry: dict[str, Any], camera_id: str) -> np.ndarray:
    if "rotation" not in entry or "translation" not in entry:
        raise KeyError(f"Camera {camera_id} missing rotation/translation")
    R = np.asarray(entry["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(entry["translation"], dtype=np.float64).reshape(3)
    if not _is_valid_rotation(R):
        raise ValueError(f"Camera {camera_id} has invalid rotation matrix")
    return _make_transform(R, t)


def _camera_video_path(episode_root: Path, camera_id: str) -> Path:
    """Resolve the standardized RGB source for one camera."""
    image_dir = episode_root / camera_id / "RGB"
    video_path = image_dir / "rgb.h265"
    if video_path.is_file():
        return video_path
    if image_dir.is_dir() and any(
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem.isdigit()
        for path in image_dir.iterdir()
    ):
        return image_dir

    raise FileNotFoundError(
        f"Missing standardized RGB source for camera {camera_id}: expected "
        f"{video_path} or numbered images in {image_dir}"
    )


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def _invert_transform(T_ab: np.ndarray) -> np.ndarray:
    T_ab = np.asarray(T_ab, dtype=np.float64).reshape(4, 4)
    R_ab = T_ab[:3, :3]
    t_ab = T_ab[:3, 3]
    T_ba = np.eye(4, dtype=np.float64)
    T_ba[:3, :3] = R_ab.T
    T_ba[:3, 3] = -R_ab.T @ t_ab
    return T_ba


def _compose(*transforms: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    for T in transforms:
        out = out @ np.asarray(T, dtype=np.float64).reshape(4, 4)
    return out


def _is_valid_rotation(R: np.ndarray, atol: float = 1e-4) -> bool:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return np.allclose(R.T @ R, np.eye(3), atol=atol) and abs(float(np.linalg.det(R)) - 1.0) < atol


def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def _matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
    return rvec.reshape(3, 1)


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp_quaternion(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = math.acos(dot)
    theta = theta_0 * alpha
    sin_theta_0 = math.sin(theta_0)
    out = (
        math.sin(theta_0 - theta) / sin_theta_0 * q0
        + math.sin(theta) / sin_theta_0 * q1
    )
    return out / np.linalg.norm(out)


def _interpolate_transform(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(max(0.0, min(1.0, alpha)))
    return _interpolate_transform_unclamped(T0, T1, alpha)


def _interpolate_transform_unclamped(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    t = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    q = _slerp_quaternion(_matrix_to_quaternion(T0[:3, :3]), _matrix_to_quaternion(T1[:3, :3]), alpha)
    return _make_transform(_quaternion_to_matrix(q), t)


def _create_aruco_detector(tag_family: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV has no aruco module. Install opencv-contrib-python.")
    dict_name = APRILTAG_DICT_NAMES.get(tag_family.lower())
    if dict_name is None or not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unsupported AprilTag family: {tag_family}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return dictionary, params


def _detect_apriltags(image: np.ndarray | None, detector) -> tuple[list[tuple[int, np.ndarray]], str]:
    if image is None:
        return [], "image_missing"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if isinstance(detector, tuple):
        dictionary, params = detector
        corners_list, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    else:
        corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return [], "tags_not_found"
    detections: list[tuple[int, np.ndarray]] = []
    for corners, tag_id_array in zip(corners_list, ids):
        tag_id = int(np.asarray(tag_id_array).reshape(-1)[0])
        detections.append((tag_id, np.asarray(corners, dtype=np.float32).reshape(4, 2)))
    return detections, ""


def _build_tag_local_corners(size_m: float) -> np.ndarray:
    half = 0.5 * float(size_m)
    return np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float32,
    )


def _tag_pose_candidates_in_reference(
    camera: CameraCalibration,
    corners_px: np.ndarray,
    local_tag_corners: np.ndarray,
) -> list[np.ndarray]:
    if camera.T_reference_from_camera is None:
        return []
    image_points = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
    candidates: list[np.ndarray] = []
    try:
        result = cv2.solvePnPGeneric(
            local_tag_corners,
            image_points,
            camera.K,
            camera.dist,
            flags=cv2.SOLVEPNP_IPPE,
        )
        rvecs = result[1] if len(result) > 1 else ()
        tvecs = result[2] if len(result) > 2 else ()
        for rvec, tvec in zip(rvecs, tvecs):
            rotation = _rodrigues_to_matrix(rvec)
            translation = np.asarray(tvec, dtype=np.float64).reshape(3)
            points_camera = local_tag_corners @ rotation.T + translation
            if np.all(points_camera[:, 2] > 0.0):
                T_camera_from_tag = _make_transform(rotation, translation)
                candidates.append(_compose(camera.T_reference_from_camera, T_camera_from_tag))
    except cv2.error:
        pass
    return candidates


def _tag_pose_to_parameters(T_reference_from_tag: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            _matrix_to_rodrigues(T_reference_from_tag[:3, :3]).reshape(3),
            np.asarray(T_reference_from_tag[:3, 3], dtype=np.float64).reshape(3),
        ]
    )


def _tag_pose_from_parameters(parameters: np.ndarray) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=np.float64).reshape(6)
    return _make_transform(_rodrigues_to_matrix(parameters[:3]), parameters[3:])


def _tag_reprojection_residuals(
    parameters: np.ndarray,
    observations: list[tuple[CameraCalibration, np.ndarray]],
    local_tag_corners: np.ndarray,
) -> np.ndarray:
    T_reference_from_tag = _tag_pose_from_parameters(parameters)
    residuals: list[np.ndarray] = []
    for camera, corners_px in observations:
        if camera.T_camera_from_reference is None:
            continue
        T_camera_from_tag = _compose(camera.T_camera_from_reference, T_reference_from_tag)
        projected, _ = cv2.projectPoints(
            local_tag_corners,
            _matrix_to_rodrigues(T_camera_from_tag[:3, :3]),
            T_camera_from_tag[:3, 3].reshape(3, 1),
            camera.K,
            camera.dist,
        )
        residuals.append(projected.reshape(4, 2) - np.asarray(corners_px, dtype=np.float64).reshape(4, 2))
    return np.concatenate(residuals, axis=0).reshape(-1)


def _tag_reprojection_rmse(residuals: np.ndarray) -> float:
    residuals_2d = np.asarray(residuals, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals_2d * residuals_2d, axis=1))))


def _camera_balanced_point_weights(
    observations: list[tuple[CameraCalibration, np.ndarray]],
    enabled: bool,
) -> np.ndarray:
    """Return one weight per observed corner, optionally balancing cameras."""
    if not observations:
        return np.empty((0,), dtype=np.float64)
    if not enabled:
        return np.ones((len(observations) * 4,), dtype=np.float64)

    counts = Counter(camera.camera_id for camera, _ in observations)
    camera_count = len(counts)
    observation_count = len(observations)
    observation_weights = [
        observation_count / (camera_count * counts[camera.camera_id])
        for camera, _ in observations
    ]
    return np.repeat(np.asarray(observation_weights, dtype=np.float64), 4)


def _robust_reprojection_cost(
    residuals: np.ndarray,
    point_weights: np.ndarray,
    huber_delta_px: float | None,
) -> float:
    residuals_2d = np.asarray(residuals, dtype=np.float64).reshape(-1, 2)
    squared_norms = np.sum(residuals_2d * residuals_2d, axis=1)
    if huber_delta_px is None:
        losses = squared_norms
    else:
        norms = np.sqrt(squared_norms)
        losses = np.where(
            norms <= huber_delta_px,
            squared_norms,
            2.0 * huber_delta_px * norms - huber_delta_px * huber_delta_px,
        )
    return float(np.sum(point_weights * losses))


def _irls_coordinate_weights(
    residuals: np.ndarray,
    point_weights: np.ndarray,
    huber_delta_px: float | None,
) -> np.ndarray:
    residuals_2d = np.asarray(residuals, dtype=np.float64).reshape(-1, 2)
    robust_weights = np.ones((residuals_2d.shape[0],), dtype=np.float64)
    if huber_delta_px is not None:
        norms = np.linalg.norm(residuals_2d, axis=1)
        large = norms > huber_delta_px
        robust_weights[large] = huber_delta_px / np.maximum(norms[large], 1e-12)
    return np.repeat(np.sqrt(point_weights * robust_weights), 2)


def _optimize_tag_pose_multiview(
    initial_pose: np.ndarray,
    observations: list[tuple[CameraCalibration, np.ndarray]],
    local_tag_corners: np.ndarray,
    *,
    huber_delta_px: float | None = None,
    balance_cameras: bool = False,
) -> tuple[np.ndarray, float]:
    parameters = _tag_pose_to_parameters(initial_pose)
    residuals = _tag_reprojection_residuals(parameters, observations, local_tag_corners)
    point_weights = _camera_balanced_point_weights(observations, balance_cameras)
    cost = _robust_reprojection_cost(residuals, point_weights, huber_delta_px)
    damping = 1e-3

    for _ in range(60):
        coordinate_weights = _irls_coordinate_weights(
            residuals, point_weights, huber_delta_px
        )
        jacobian = np.empty((residuals.size, 6), dtype=np.float64)
        for column in range(6):
            step = 1e-6 if column < 3 else 1e-5
            shifted = parameters.copy()
            shifted[column] += step
            shifted_residuals = _tag_reprojection_residuals(shifted, observations, local_tag_corners)
            jacobian[:, column] = (
                (shifted_residuals - residuals) / step
            ) * coordinate_weights

        weighted_residuals = residuals * coordinate_weights
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ weighted_residuals
        try:
            delta = np.linalg.solve(normal + damping * np.eye(6), -gradient)
        except np.linalg.LinAlgError:
            break

        candidate_parameters = parameters + delta
        candidate_residuals = _tag_reprojection_residuals(candidate_parameters, observations, local_tag_corners)
        candidate_cost = _robust_reprojection_cost(
            candidate_residuals, point_weights, huber_delta_px
        )
        if candidate_cost < cost:
            parameters = candidate_parameters
            residuals = candidate_residuals
            cost = candidate_cost
            damping = max(1e-9, damping * 0.3)
            if np.linalg.norm(delta) < 1e-8:
                break
        else:
            damping = min(1e9, damping * 10.0)

    return _tag_pose_from_parameters(parameters), _tag_reprojection_rmse(residuals)


def _observation_reprojection_rmse(
    pose: np.ndarray,
    observations: list[tuple[CameraCalibration, np.ndarray]],
    local_tag_corners: np.ndarray,
) -> np.ndarray:
    residuals = _tag_reprojection_residuals(
        _tag_pose_to_parameters(pose), observations, local_tag_corners
    ).reshape(len(observations), 4, 2)
    return np.sqrt(np.mean(np.sum(residuals * residuals, axis=2), axis=1))


def _select_reference_frame_numbers(
    rows: list[TimestampRow],
    count: int,
    sampling: str,
) -> list[int]:
    """Select increasing reference frames from the aligned episode rows."""
    available = [row.frame_number for row in rows]
    if count >= len(available):
        return available
    if sampling == "first":
        return available[:count]
    positions = np.linspace(0, len(available) - 1, num=count)
    indices = np.rint(positions).astype(int)
    return [available[index] for index in indices]


def _solve_ego_pose(
    detections: list[tuple[int, np.ndarray]],
    K: np.ndarray,
    dist: np.ndarray,
    reference_tag_corners: dict[int, np.ndarray],
) -> tuple[np.ndarray | None, float | None, list[int], str]:
    object_pts_list: list[np.ndarray] = []
    image_pts_list: list[np.ndarray] = []
    used_tag_ids: list[int] = []

    for tag_id, corners_px in detections:
        tag_corners = reference_tag_corners.get(tag_id)
        if tag_corners is None:
            continue
        object_pts_list.append(np.asarray(tag_corners, dtype=np.float32).reshape(4, 3))
        image_pts_list.append(np.asarray(corners_px, dtype=np.float32).reshape(4, 2))
        used_tag_ids.append(int(tag_id))

    used_tag_ids = sorted(set(used_tag_ids))
    if len(used_tag_ids) < MIN_EGO_TAGS:
        return None, None, used_tag_ids, "insufficient_tags"

    object_pts = np.concatenate(object_pts_list, axis=0).astype(np.float32)
    image_pts = np.concatenate(image_pts_list, axis=0).astype(np.float32)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_pts,
        image_pts,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=EGO_RANSAC_REPROJ_ERROR_PX,
        iterationsCount=EGO_PNP_ITERATIONS,
        confidence=0.999,
    )
    if not ok:
        return None, None, used_tag_ids, "pnp_failed"

    inlier_indices = np.asarray(inliers).reshape(-1).astype(int) if inliers is not None else np.array([], dtype=int)
    if inlier_indices.size < MIN_EGO_INLIER_CORNERS:
        return None, None, used_tag_ids, "few_inliers"

    if hasattr(cv2, "solvePnPRefineLM") and inlier_indices.size >= 4:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_pts[inlier_indices],
                image_pts[inlier_indices],
                K,
                dist,
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            )
        except cv2.error:
            pass

    projected, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    reproj = np.linalg.norm(projected.reshape(-1, 2) - image_pts, axis=1)
    rmse = float(np.sqrt(np.mean(reproj * reproj)))
    if rmse > EGO_REPROJ_RMSE_LIMIT_PX:
        return None, rmse, used_tag_ids, "high_reproj_error"

    return _make_transform(_rodrigues_to_matrix(rvec), tvec.reshape(3)), rmse, used_tag_ids, "ok"


def _load_timestamp_rows(timestamps_csv: Path, max_rows: int | None) -> list[TimestampRow]:
    """Load reference rows that have an explicit source PICO frame mapping."""
    rows: list[TimestampRow] = []
    with timestamps_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            ego_frame_index = row.get("ego_frame_index", "").strip()
            frame_index = row.get("frame_index", "").strip()
            if not frame_index or not ego_frame_index:
                continue
            rows.append(
                TimestampRow(
                    row_index=row_index,
                    frame_index=frame_index,
                    frame_number=int(frame_index),
                    ref_timestamp_us=row.get("ref_timestamp_us", ""),
                    ego_frame_index=ego_frame_index,
                    ego_frame_number=int(ego_frame_index) if ego_frame_index else None,
                    ego_timestamp_us=row.get("ego_timestamp_us", ""),
                    raw=row,
                )
            )
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _load_static_cameras(
    episode_root: Path,
    camera_params: dict[str, Any],
    extrinsics: dict[str, Any],
    static_camera_ids: list[str],
    reference_camera_id: str,
) -> list[CameraCalibration]:
    ordered_ids = [reference_camera_id] + [cid for cid in static_camera_ids if cid != reference_camera_id]
    cameras: list[CameraCalibration] = []
    for camera_id in ordered_ids:
        params_entry = _resolve_json_entry(camera_params, camera_id)
        extr_entry = _resolve_json_entry(extrinsics, camera_id)
        K, dist, _ = _extract_intrinsics(params_entry, camera_id)
        cameras.append(
            CameraCalibration(
                camera_id=camera_id,
                K=K,
                dist=dist,
                T_camera_from_world=_extract_extrinsic_camera_from_world(extr_entry, camera_id),
                video_path=_camera_video_path(episode_root, camera_id),
            )
        )

    reference = next((camera for camera in cameras if camera.camera_id == reference_camera_id), None)
    if reference is None:
        raise RuntimeError(f"Reference camera {reference_camera_id} is not in static cameras")
    T_world_from_reference = _invert_transform(reference.T_camera_from_world)
    for camera in cameras:
        camera.T_camera_from_reference = _compose(camera.T_camera_from_world, T_world_from_reference)
        camera.T_reference_from_camera = _invert_transform(camera.T_camera_from_reference)
    return cameras


def _load_ego_model(
    episode_root: Path,
    fisheye_calibration_path: Path,
    disable_fisheye_undistort: bool,
) -> tuple[EgoImageModel, Path]:
    ego_params_path = episode_root / "ego" / "camera_params.json"
    ego_params = _load_json(ego_params_path)
    ego_entry = _resolve_json_entry(ego_params, "ego")
    ego_video_path = _camera_video_path(episode_root, "ego")
    K, dist, model = _extract_intrinsics(ego_entry, "ego")
    width, height = _extract_image_size(ego_entry, "ego")
    image_size = (width, height)

    if disable_fisheye_undistort:
        return EgoImageModel(False, "disabled", K, None, K, np.zeros((5, 1), dtype=np.float64), image_size), ego_video_path

    if fisheye_calibration_path.is_file():
        calibration = np.load(str(fisheye_calibration_path))
        calib_size = tuple(int(x) for x in calibration["image_size"])
        if calib_size == image_size:
            K_raw = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
            D = np.asarray(calibration["D"], dtype=np.float64).reshape(4, 1)
            K_pnp = np.asarray(calibration["new_K"], dtype=np.float64).reshape(3, 3)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K_raw,
                D,
                np.eye(3, dtype=np.float64),
                K_pnp,
                image_size,
                cv2.CV_16SC2,
            )
            return (
                EgoImageModel(
                    True,
                    str(fisheye_calibration_path),
                    K_raw,
                    D,
                    K_pnp,
                    np.zeros((5, 1), dtype=np.float64),
                    image_size,
                    map1,
                    map2,
                ),
                ego_video_path,
            )

    if "fisheye" in model:
        K_pnp = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            np.asarray(dist, dtype=np.float64).reshape(4, 1),
            image_size,
            np.eye(3, dtype=np.float64),
            balance=1.0,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K,
            np.asarray(dist, dtype=np.float64).reshape(4, 1),
            np.eye(3, dtype=np.float64),
            K_pnp,
            image_size,
            cv2.CV_16SC2,
        )
        return (
            EgoImageModel(
                True,
                "ego/camera_params.json",
                K,
                np.asarray(dist, dtype=np.float64).reshape(4, 1),
                K_pnp,
                np.zeros((5, 1), dtype=np.float64),
                image_size,
                map1,
                map2,
            ),
            ego_video_path,
        )

    return EgoImageModel(False, "raw_pinhole", K, None, K, np.zeros((8,), dtype=np.float64), image_size), ego_video_path


def _detect_ego_apriltags(
    frame: np.ndarray | None,
    detector,
    ego_model: EgoImageModel,
) -> tuple[list[tuple[int, np.ndarray]], str, str, np.ndarray | None]:
    if frame is None:
        return [], "image_missing", "", None

    if not ego_model.enabled:
        detections, reason = _detect_apriltags(frame, detector)
        return detections, reason, "raw", frame

    assert ego_model.map1 is not None and ego_model.map2 is not None and ego_model.D_fisheye is not None
    undistorted = cv2.remap(frame, ego_model.map1, ego_model.map2, cv2.INTER_LINEAR)
    undistorted_detections, undistorted_reason = _detect_apriltags(undistorted, detector)

    raw_detections, raw_reason = _detect_apriltags(frame, detector)
    raw_projected: list[tuple[int, np.ndarray]] = []
    for tag_id, raw_corners in raw_detections:
        undistorted_points = cv2.fisheye.undistortPoints(
            np.asarray(raw_corners, dtype=np.float64).reshape(-1, 1, 2),
            ego_model.K_raw,
            ego_model.D_fisheye,
            R=np.eye(3, dtype=np.float64),
            P=ego_model.K_pnp,
        ).reshape(4, 2)
        raw_projected.append((tag_id, undistorted_points.astype(np.float32)))

    if len(raw_projected) > len(undistorted_detections):
        return raw_projected, raw_reason, "raw_fisheye_corners_to_undistorted", undistorted
    return undistorted_detections, undistorted_reason, "undistorted_image", undistorted


def _build_reference_from_static_video_frames(
    static_cameras: list[CameraCalibration],
    reference_frame_numbers: list[int],
    detector,
    tag_size_m: float,
    force_temp_remux: bool,
    reference_camera_id: str,
    reference_map_mode: str,
    robust_huber_delta_px: float | None,
    balance_cameras: bool,
    observation_outlier_px: float | None,
    allowed_tag_ids: set[int] | None,
) -> tuple[dict[int, np.ndarray], dict[int, TagFusionResult], set[int], dict[str, Any]]:
    sources: dict[str, SequentialVideoFrameSource] = {}
    try:
        for camera in static_cameras:
            if camera.video_path is None:
                raise RuntimeError(f"Camera {camera.camera_id} has no video path")
            sources[camera.camera_id] = SequentialVideoFrameSource(
                camera.video_path,
                f"static_{camera.camera_id}",
                force_temp_remux=force_temp_remux,
            )

        local_tag_corners = _build_tag_local_corners(tag_size_m)
        observations_by_tag: dict[int, list[tuple[CameraCalibration, np.ndarray]]] = {}
        candidates_by_tag: dict[int, list[tuple[str, np.ndarray]]] = {}
        detected_ids: set[int] = set()
        detection_counts: Counter[int] = Counter()

        for frame_number in reference_frame_numbers:
            for camera in static_cameras:
                frame = sources[camera.camera_id].read(frame_number)
                detections, _ = _detect_apriltags(frame, detector)
                for tag_id, corners_px in detections:
                    if allowed_tag_ids is not None and tag_id not in allowed_tag_ids:
                        continue
                    detected_ids.add(tag_id)
                    detection_counts[tag_id] += 1
                    observations_by_tag.setdefault(tag_id, []).append((camera, np.asarray(corners_px, dtype=np.float64)))
                    candidates_by_tag.setdefault(tag_id, []).extend(
                        (camera.camera_id, pose)
                        for pose in _tag_pose_candidates_in_reference(
                            camera, corners_px, local_tag_corners
                        )
                    )

        reference_tag_corners: dict[int, np.ndarray] = {}
        fused_tags: dict[int, TagFusionResult] = {}
        for tag_id in sorted(observations_by_tag):
            all_observations = observations_by_tag[tag_id]
            all_candidates = candidates_by_tag.get(tag_id, [])
            if reference_map_mode == "reference":
                observations = [
                    item
                    for item in all_observations
                    if item[0].camera_id == reference_camera_id
                ]
                candidates = [
                    pose
                    for camera_id, pose in all_candidates
                    if camera_id == reference_camera_id
                ]
            else:
                observations = all_observations
                candidates = [pose for _, pose in all_candidates]
            if len(observations) < MIN_REFERENCE_INLIER_OBSERVATIONS or not candidates:
                continue
            initial_pose = min(
                candidates,
                key=lambda pose: float(
                    np.mean(
                        _tag_reprojection_residuals(
                            _tag_pose_to_parameters(pose),
                            observations,
                            local_tag_corners,
                        )
                        ** 2
                    )
                ),
            )
            optimized_pose, _ = _optimize_tag_pose_multiview(
                initial_pose,
                observations,
                local_tag_corners,
                huber_delta_px=robust_huber_delta_px,
                balance_cameras=balance_cameras,
            )
            inlier_observations = observations
            if observation_outlier_px is not None:
                observation_rmse = _observation_reprojection_rmse(
                    optimized_pose, observations, local_tag_corners
                )
                inlier_observations = [
                    observation
                    for observation, error in zip(observations, observation_rmse)
                    if error <= observation_outlier_px
                ]
                if len(inlier_observations) < MIN_REFERENCE_INLIER_OBSERVATIONS:
                    continue
                optimized_pose, _ = _optimize_tag_pose_multiview(
                    optimized_pose,
                    inlier_observations,
                    local_tag_corners,
                    huber_delta_px=robust_huber_delta_px,
                    balance_cameras=balance_cameras,
                )
            residuals = _tag_reprojection_residuals(
                _tag_pose_to_parameters(optimized_pose),
                inlier_observations,
                local_tag_corners,
            )
            rmse = _tag_reprojection_rmse(residuals)
            if rmse > REFERENCE_TAG_REPROJ_RMSE_LIMIT_PX:
                continue
            corners_reference = (
                local_tag_corners @ optimized_pose[:3, :3].T + optimized_pose[:3, 3].reshape(1, 3)
            ).astype(np.float32)
            fused_tags[tag_id] = TagFusionResult(
                tag_id=tag_id,
                T_reference_from_tag=optimized_pose,
                corners_reference=corners_reference,
                detection_count=detection_counts[tag_id],
                candidate_count=len(candidates),
                inlier_count=len(inlier_observations),
                reprojection_rmse_px=rmse,
            )
            reference_tag_corners[tag_id] = corners_reference

        video_stats = {
            camera_id: {
                "source_path": str(source.source_path),
                "active_path": str(source.active_path),
                "source_kind": "image_sequence" if source.is_image_sequence else "video",
                "used_temp_remux": source.used_temp_remux,
            }
            for camera_id, source in sources.items()
        }
        return reference_tag_corners, fused_tags, detected_ids, video_stats
    finally:
        for source in sources.values():
            source.close()


def _run_direct_ego_pass(
    rows: list[TimestampRow],
    ego_video_path: Path,
    ego_model: EgoImageModel,
    reference_tag_corners: dict[int, np.ndarray],
    detector,
    output_dir: Path,
    write_debug_images: bool,
    debug_image_limit: int,
    force_temp_remux: bool,
) -> tuple[list[FrameEstimate], dict[str, Any]]:
    estimates: list[FrameEstimate] = []
    debug_dir = output_dir / "debug_ego_detections"
    debug_count = 0
    source = SequentialVideoFrameSource(ego_video_path, "ego", force_temp_remux=force_temp_remux)
    try:
        for row in rows:
            if row.ego_frame_number is None:
                raise RuntimeError(f"Missing ego_frame_index for reference frame {row.frame_index}")
            source_frame_number = row.ego_frame_number
            source_frame_index = row.ego_frame_index
            frame = source.read(source_frame_number)
            estimate = FrameEstimate(row=row)
            detections, reason, detection_space, debug_image = _detect_ego_apriltags(frame, detector, ego_model)
            estimate.detection_space = detection_space
            estimate.detected_tag_ids = sorted({tag_id for tag_id, _ in detections})
            if not detections:
                estimate.status_initial = "no_detection" if reason == "tags_not_found" else reason
                estimates.append(estimate)
                continue

            pose, rmse, used_tag_ids, status = _solve_ego_pose(
                detections,
                ego_model.K_pnp,
                ego_model.dist_pnp,
                reference_tag_corners,
            )
            estimate.used_tag_ids = used_tag_ids
            estimate.rmse_px = rmse
            estimate.pose_direct = pose
            estimate.status_initial = status
            estimates.append(estimate)

            if write_debug_images and debug_image is not None and debug_count < debug_image_limit:
                debug_dir.mkdir(parents=True, exist_ok=True)
                drawn = debug_image.copy()
                if detections:
                    cv2.aruco.drawDetectedMarkers(
                        drawn,
                        [corners.reshape(1, 4, 2).astype(np.float32) for _, corners in detections],
                        np.asarray([[tag_id] for tag_id, _ in detections], dtype=np.int32),
                    )
                cv2.imwrite(
                    str(debug_dir / f"ego_{source_frame_index}_{estimate.status_initial}.jpg"),
                    drawn,
                )
                debug_count += 1

        video_stats = {
            "source_path": str(source.source_path),
            "active_path": str(source.active_path),
            "source_kind": "image_sequence" if source.is_image_sequence else "video",
            "frame_mapping": "timestamps.csv:frame_index->ego_frame_index",
            "used_temp_remux": source.used_temp_remux,
        }
        return estimates, video_stats
    finally:
        source.close()


def _apply_interpolation(estimates: list[FrameEstimate], unbounded_gap_mode: str) -> dict[str, int]:
    for estimate in estimates:
        if estimate.pose_direct is not None:
            estimate.pose_final = np.array(estimate.pose_direct, copy=True)
            estimate.status_final = "direct"
            estimate.source = "direct"
        else:
            estimate.pose_final = None
            estimate.status_final = estimate.status_initial or "not_solved"
            estimate.source = "nan"

    valid_indices = [idx for idx, estimate in enumerate(estimates) if estimate.pose_final is not None]
    if not valid_indices:
        for estimate in estimates:
            estimate.status_final = "nan_no_direct_anchor"
            estimate.source = "nan"
        return {"direct": 0, "interpolated": 0, "extrapolated": 0, "nan": len(estimates)}

    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    if unbounded_gap_mode == "extrapolate" and len(valid_indices) >= 2:
        first_next = valid_indices[1]
        first_frame = estimates[first_valid].row.frame_number
        denom = float(estimates[first_next].row.frame_number - first_frame)
        for idx in range(0, first_valid):
            alpha = (estimates[idx].row.frame_number - first_frame) / denom
            estimates[idx].pose_final = _interpolate_transform_unclamped(
                estimates[first_valid].pose_final,
                estimates[first_next].pose_final,
                alpha,
            )
            estimates[idx].status_final = "extrapolated_start"
            estimates[idx].source = "extrapolated"
            estimates[idx].prev_direct_frame_index = estimates[first_valid].row.frame_index
            estimates[idx].next_direct_frame_index = estimates[first_next].row.frame_index
            estimates[idx].interp_alpha = alpha
            estimates[idx].interp_gap_frames = first_frame - estimates[idx].row.frame_number

        last_prev = valid_indices[-2]
        last_prev_frame = estimates[last_prev].row.frame_number
        last_frame = estimates[last_valid].row.frame_number
        denom = float(last_frame - last_prev_frame)
        trailing_len = estimates[-1].row.frame_number - last_frame
        for idx in range(last_valid + 1, len(estimates)):
            alpha = (estimates[idx].row.frame_number - last_prev_frame) / denom
            estimates[idx].pose_final = _interpolate_transform_unclamped(
                estimates[last_prev].pose_final,
                estimates[last_valid].pose_final,
                alpha,
            )
            estimates[idx].status_final = "extrapolated_end"
            estimates[idx].source = "extrapolated"
            estimates[idx].prev_direct_frame_index = estimates[last_prev].row.frame_index
            estimates[idx].next_direct_frame_index = estimates[last_valid].row.frame_index
            estimates[idx].interp_alpha = alpha
            estimates[idx].interp_gap_frames = trailing_len
    elif unbounded_gap_mode == "extrapolate" and len(valid_indices) == 1:
        for idx in range(0, first_valid):
            estimates[idx].pose_final = np.array(estimates[first_valid].pose_final, copy=True)
            estimates[idx].status_final = "edge_filled_start_single_anchor"
            estimates[idx].source = "extrapolated"
            estimates[idx].next_direct_frame_index = estimates[first_valid].row.frame_index
            estimates[idx].interp_gap_frames = first_valid
        trailing_len = len(estimates) - last_valid - 1
        for idx in range(last_valid + 1, len(estimates)):
            estimates[idx].pose_final = np.array(estimates[last_valid].pose_final, copy=True)
            estimates[idx].status_final = "edge_filled_end_single_anchor"
            estimates[idx].source = "extrapolated"
            estimates[idx].prev_direct_frame_index = estimates[last_valid].row.frame_index
            estimates[idx].interp_gap_frames = trailing_len
    else:
        for idx in range(0, first_valid):
            estimates[idx].status_final = "nan_unbracketed_start"
            estimates[idx].source = "nan"
        for idx in range(last_valid + 1, len(estimates)):
            estimates[idx].status_final = "nan_unbracketed_end"
            estimates[idx].source = "nan"

    for left, right in zip(valid_indices, valid_indices[1:]):
        left_frame = estimates[left].row.frame_number
        right_frame = estimates[right].row.frame_number
        gap_len = right_frame - left_frame - 1
        if gap_len <= 0:
            continue
        left_pose = estimates[left].pose_final
        right_pose = estimates[right].pose_final
        if left_pose is None or right_pose is None:
            continue
        for idx in range(left + 1, right):
            alpha = (estimates[idx].row.frame_number - left_frame) / float(
                right_frame - left_frame
            )
            estimates[idx].pose_final = _interpolate_transform(left_pose, right_pose, alpha)
            estimates[idx].status_final = "interpolated"
            estimates[idx].source = "interpolated"
            estimates[idx].prev_direct_frame_index = estimates[left].row.frame_index
            estimates[idx].next_direct_frame_index = estimates[right].row.frame_index
            estimates[idx].interp_alpha = alpha
            estimates[idx].interp_gap_frames = gap_len

    source_counts = Counter(estimate.source for estimate in estimates)
    return {
        "direct": int(source_counts.get("direct", 0)),
        "interpolated": int(source_counts.get("interpolated", 0)),
        "extrapolated": int(source_counts.get("extrapolated", 0)),
        "nan": int(source_counts.get("nan", 0)),
    }


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _require_savgol_filter() -> Any:
    try:
        from scipy.signal import savgol_filter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required for Savitzky-Golay trajectory smoothing. "
            "Install scipy before using --smooth-trajectory."
        ) from exc
    return savgol_filter


def _matrix_to_rotation_6d(rotation: np.ndarray) -> np.ndarray:
    """Match the reference implementation's first-two-rows 6D representation."""
    rotation = np.asarray(rotation, dtype=np.float64)
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)


def _rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Project a 6D rotation representation back onto SO(3)."""
    rotation_6d = np.asarray(rotation_6d, dtype=np.float64)
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def _smooth_pose_stack(
    poses: np.ndarray,
    savgol_filter: Any,
    *,
    window_size: int,
    polyorder: int,
) -> np.ndarray:
    """Apply the reference Savitzky-Golay logic to camera rotations and translations."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    if poses.shape[0] < window_size:
        return poses.copy()

    smoothed = poses.copy()
    rotation_filtered = savgol_filter(
        poses[:, :3, :3],
        window_length=window_size,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )
    rotation_6d = _matrix_to_rotation_6d(rotation_filtered)
    smoothed[:, :3, :3] = _rotation_6d_to_matrix(rotation_6d)
    smoothed[:, :3, 3] = savgol_filter(
        poses[:, :3, 3],
        window_length=window_size,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )
    smoothed[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return smoothed


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _trajectory_smoothness_metrics(
    estimates: list[FrameEstimate],
    pose_attribute: str,
) -> dict[str, dict[str, float | int | None]]:
    translation_second_differences_cm: list[float] = []
    rotation_increment_differences_deg: list[float] = []
    for index in range(1, len(estimates) - 1):
        poses = [getattr(estimates[j], pose_attribute) for j in (index - 1, index, index + 1)]
        if any(pose is None for pose in poses):
            continue
        world_poses = [_invert_transform(pose) for pose in poses]
        centers = [pose[:3, 3] for pose in world_poses]
        second_difference = centers[2] - 2.0 * centers[1] + centers[0]
        translation_second_differences_cm.append(float(np.linalg.norm(second_difference) * 100.0))

        first_increment = world_poses[0][:3, :3].T @ world_poses[1][:3, :3]
        second_increment = world_poses[1][:3, :3].T @ world_poses[2][:3, :3]
        increment_change = first_increment.T @ second_increment
        rotation_increment_differences_deg.append(_rotation_angle_deg(increment_change))

    return {
        "translation_second_difference_cm": _numeric_summary(translation_second_differences_cm),
        "rotation_increment_difference_deg": _numeric_summary(rotation_increment_differences_deg),
    }


def _apply_savgol_smoothing(
    estimates: list[FrameEstimate],
    *,
    window_size: int,
    polyorder: int,
) -> dict[str, Any]:
    """Smooth contiguous camera-pose runs using the supplied reference algorithm."""
    savgol_filter = _require_savgol_filter()
    raw_smoothness = _trajectory_smoothness_metrics(estimates, "pose_final")
    for estimate in estimates:
        estimate.pose_smoothed = None if estimate.pose_final is None else np.array(estimate.pose_final, copy=True)
        estimate.smoothing_translation_delta_m = 0.0
        estimate.smoothing_rotation_delta_deg = 0.0
        estimate.smoothing_status = "unavailable" if estimate.pose_final is None else "copied_short_segment"

    segment_count = 0
    index = 0
    while index < len(estimates):
        if estimates[index].pose_final is None:
            index += 1
            continue
        start = index
        while index < len(estimates) and estimates[index].pose_final is not None:
            index += 1
        stop = index
        segment_count += 1
        if stop - start < window_size:
            continue

        raw_poses = np.stack(
            [np.asarray(estimates[position].pose_final, dtype=np.float64) for position in range(start, stop)],
            axis=0,
        )
        smoothed_poses = _smooth_pose_stack(
            raw_poses,
            savgol_filter,
            window_size=window_size,
            polyorder=polyorder,
        )
        for offset, position in enumerate(range(start, stop)):
            estimate = estimates[position]
            raw_pose = raw_poses[offset]
            smoothed_pose = smoothed_poses[offset]
            estimate.pose_smoothed = smoothed_pose
            estimate.smoothing_translation_delta_m = float(
                np.linalg.norm(smoothed_pose[:3, 3] - raw_pose[:3, 3])
            )
            estimate.smoothing_rotation_delta_deg = _rotation_angle_deg(
                raw_pose[:3, :3].T @ smoothed_pose[:3, :3]
            )
            estimate.smoothing_status = "smoothed"

    status_counts = Counter(estimate.smoothing_status for estimate in estimates)
    return {
        "enabled": True,
        "method": "savitzky_golay_rotation6d_translation",
        "window_size": int(window_size),
        "polyorder": int(polyorder),
        "segment_count": int(segment_count),
        "status_counts": dict(sorted(status_counts.items())),
        "translation_correction_m": _numeric_summary(
            [estimate.smoothing_translation_delta_m for estimate in estimates if estimate.pose_smoothed is not None]
        ),
        "rotation_correction_deg": _numeric_summary(
            [estimate.smoothing_rotation_delta_deg for estimate in estimates if estimate.pose_smoothed is not None]
        ),
        "raw_trajectory_smoothness": raw_smoothness,
        "smoothed_trajectory_smoothness": _trajectory_smoothness_metrics(estimates, "pose_smoothed"),
    }


def _matrix_or_nan(pose: np.ndarray | None) -> np.ndarray:
    if pose is None:
        return np.full((4, 4), float("nan"), dtype=np.float64)
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def _float_or_empty(value: float | None) -> str | float:
    return "" if value is None else float(value)


def _ids_to_text(values: list[int]) -> str:
    return ";".join(str(value) for value in values)


def _write_direct_csv(path: Path, estimates: list[FrameEstimate]) -> None:
    matrix_fields = [f"m{r}{c}" for r in range(4) for c in range(4)]
    fields = [
        "row_index",
        "frame_index",
        "ref_timestamp_us",
        "ego_frame_index",
        "ego_timestamp_us",
        "status_initial",
        "detection_space",
        "detected_tag_ids",
        "used_tag_ids",
        "rmse_px",
        *matrix_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for estimate in estimates:
            matrix = _matrix_or_nan(estimate.pose_direct).reshape(-1)
            row = {
                "row_index": estimate.row.row_index,
                "frame_index": estimate.row.frame_index,
                "ref_timestamp_us": estimate.row.ref_timestamp_us,
                "ego_frame_index": estimate.row.ego_frame_index,
                "ego_timestamp_us": estimate.row.ego_timestamp_us,
                "status_initial": estimate.status_initial,
                "detection_space": estimate.detection_space,
                "detected_tag_ids": _ids_to_text(estimate.detected_tag_ids),
                "used_tag_ids": _ids_to_text(estimate.used_tag_ids),
                "rmse_px": _float_or_empty(estimate.rmse_px),
            }
            row.update({field: float(value) for field, value in zip(matrix_fields, matrix)})
            writer.writerow(row)


def _write_final_csv(path: Path, estimates: list[FrameEstimate]) -> None:
    matrix_fields = [f"m{r}{c}" for r in range(4) for c in range(4)]
    fields = [
        "row_index",
        "frame_index",
        "ref_timestamp_us",
        "ego_frame_index",
        "ego_timestamp_us",
        "status_initial",
        "status_final",
        "source",
        "detection_space",
        "detected_tag_ids",
        "used_tag_ids",
        "rmse_px",
        "prev_direct_frame_index",
        "next_direct_frame_index",
        "interp_alpha",
        "interp_gap_frames",
        *matrix_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for estimate in estimates:
            matrix = _matrix_or_nan(estimate.pose_final).reshape(-1)
            row = {
                "row_index": estimate.row.row_index,
                "frame_index": estimate.row.frame_index,
                "ref_timestamp_us": estimate.row.ref_timestamp_us,
                "ego_frame_index": estimate.row.ego_frame_index,
                "ego_timestamp_us": estimate.row.ego_timestamp_us,
                "status_initial": estimate.status_initial,
                "status_final": estimate.status_final,
                "source": estimate.source,
                "detection_space": estimate.detection_space,
                "detected_tag_ids": _ids_to_text(estimate.detected_tag_ids),
                "used_tag_ids": _ids_to_text(estimate.used_tag_ids),
                "rmse_px": _float_or_empty(estimate.rmse_px),
                "prev_direct_frame_index": estimate.prev_direct_frame_index,
                "next_direct_frame_index": estimate.next_direct_frame_index,
                "interp_alpha": _float_or_empty(estimate.interp_alpha),
                "interp_gap_frames": "" if estimate.interp_gap_frames is None else estimate.interp_gap_frames,
            }
            row.update({field: float(value) for field, value in zip(matrix_fields, matrix)})
            writer.writerow(row)


def _write_smoothed_csv(path: Path, estimates: list[FrameEstimate]) -> None:
    matrix_fields = [f"m{r}{c}" for r in range(4) for c in range(4)]
    fields = [
        "row_index",
        "frame_index",
        "ref_timestamp_us",
        "ego_frame_index",
        "ego_timestamp_us",
        "source",
        "used_tag_ids",
        "raw_rmse_px",
        "smoothing_status",
        "smoothing_translation_delta_m",
        "smoothing_rotation_delta_deg",
        *matrix_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for estimate in estimates:
            matrix = _matrix_or_nan(estimate.pose_smoothed).reshape(-1)
            row = {
                "row_index": estimate.row.row_index,
                "frame_index": estimate.row.frame_index,
                "ref_timestamp_us": estimate.row.ref_timestamp_us,
                "ego_frame_index": estimate.row.ego_frame_index,
                "ego_timestamp_us": estimate.row.ego_timestamp_us,
                "source": estimate.source,
                "used_tag_ids": _ids_to_text(estimate.used_tag_ids),
                "raw_rmse_px": _float_or_empty(estimate.rmse_px),
                "smoothing_status": estimate.smoothing_status,
                "smoothing_translation_delta_m": float(estimate.smoothing_translation_delta_m),
                "smoothing_rotation_delta_deg": float(estimate.smoothing_rotation_delta_deg),
            }
            row.update({field: float(value) for field, value in zip(matrix_fields, matrix)})
            writer.writerow(row)


def _transform_reference_points_to_ego(T_ego_from_reference: np.ndarray, points_reference: np.ndarray) -> np.ndarray:
    points_reference = np.asarray(points_reference, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate([points_reference, np.ones((points_reference.shape[0], 1), dtype=np.float64)], axis=1)
    points_ego = (np.asarray(T_ego_from_reference, dtype=np.float64).reshape(4, 4) @ homogeneous.T).T
    return points_ego[:, :3]


def _write_used_apriltag_corners_ego_csv(
    path: Path,
    estimates: list[FrameEstimate],
    reference_tag_corners: dict[int, np.ndarray],
) -> int:
    # CSV 输出结构：
    # - 每一行表示一个“用于直接 PnP 定位的 AprilTag 角点”的 3D 坐标。
    # - 只输出 pose_direct 成功的帧；插值、外推、NaN 帧不会输出。
    # - tag_id 用于区分 AprilTag；corner_index 为同一 tag 的 0..3 角点编号。
    # - corner_index 的顺序沿用建图时的 tag 局部角点顺序，但后续可视化点云时可以忽略顺序，
    #   直接把同一帧/同一 tag 的 4 个点作为 tag 四角显示。
    # - corner_*_ego_m 是米制 3D 坐标，参考系与 ego RGB 相机坐标系一致。
    fields = [
        "row_index",
        "frame_index",
        "ref_timestamp_us",
        "ego_frame_index",
        "ego_timestamp_us",
        "tag_id",
        "corner_index",
        "corner_x_ego_m",
        "corner_y_ego_m",
        "corner_z_ego_m",
        "rmse_px",
        "status_initial",
        "detection_space",
    ]
    count = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for estimate in estimates:
            if estimate.pose_direct is None or estimate.status_initial != "ok":
                continue
            for tag_id in estimate.used_tag_ids:
                corners_reference = reference_tag_corners.get(tag_id)
                if corners_reference is None:
                    continue
                corners_ego = _transform_reference_points_to_ego(estimate.pose_direct, corners_reference)
                for corner_index, corner_ego in enumerate(corners_ego):
                    writer.writerow(
                        {
                            "row_index": estimate.row.row_index,
                            "frame_index": estimate.row.frame_index,
                            "ref_timestamp_us": estimate.row.ref_timestamp_us,
                            "ego_frame_index": estimate.row.ego_frame_index,
                            "ego_timestamp_us": estimate.row.ego_timestamp_us,
                            "tag_id": int(tag_id),
                            "corner_index": int(corner_index),
                            "corner_x_ego_m": float(corner_ego[0]),
                            "corner_y_ego_m": float(corner_ego[1]),
                            "corner_z_ego_m": float(corner_ego[2]),
                            "rmse_px": _float_or_empty(estimate.rmse_px),
                            "status_initial": estimate.status_initial,
                            "detection_space": estimate.detection_space,
                        }
                    )
                    count += 1
    return count


def _estimate_to_json(estimate: FrameEstimate) -> dict[str, Any]:
    return {
        "row_index": estimate.row.row_index,
        "frame_index": estimate.row.frame_index,
        "ref_timestamp_us": estimate.row.ref_timestamp_us,
        "ego_frame_index": estimate.row.ego_frame_index,
        "ego_timestamp_us": estimate.row.ego_timestamp_us,
        "status_initial": estimate.status_initial,
        "status_final": estimate.status_final,
        "source": estimate.source,
        "detection_space": estimate.detection_space,
        "detected_tag_ids": estimate.detected_tag_ids,
        "used_tag_ids": estimate.used_tag_ids,
        "rmse_px": estimate.rmse_px,
        "prev_direct_frame_index": estimate.prev_direct_frame_index,
        "next_direct_frame_index": estimate.next_direct_frame_index,
        "interp_alpha": estimate.interp_alpha,
        "interp_gap_frames": estimate.interp_gap_frames,
        "T_ego_from_reference": _matrix_or_nan(estimate.pose_final).tolist(),
    }


def _smoothed_estimate_to_json(estimate: FrameEstimate) -> dict[str, Any]:
    return {
        "row_index": estimate.row.row_index,
        "frame_index": estimate.row.frame_index,
        "ref_timestamp_us": estimate.row.ref_timestamp_us,
        "ego_frame_index": estimate.row.ego_frame_index,
        "ego_timestamp_us": estimate.row.ego_timestamp_us,
        "source": estimate.source,
        "used_tag_ids": estimate.used_tag_ids,
        "raw_rmse_px": estimate.rmse_px,
        "smoothing_status": estimate.smoothing_status,
        "smoothing_translation_delta_m": estimate.smoothing_translation_delta_m,
        "smoothing_rotation_delta_deg": estimate.smoothing_rotation_delta_deg,
        "T_ego_from_reference": _matrix_or_nan(estimate.pose_smoothed).tolist(),
    }


def _serialize_pose_dict(
    estimates: list[FrameEstimate],
    key_field: str,
    pose_attribute: str = "pose_final",
) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = {}
    for estimate in estimates:
        if key_field == "ego_frame_index":
            key = estimate.row.ego_frame_index
            if not key:
                continue
        elif key_field == "frame_index":
            key = estimate.row.frame_index
        else:
            raise ValueError(f"Unsupported pose dict key field: {key_field}")
        result[key] = _matrix_or_nan(getattr(estimate, pose_attribute)).tolist()
    return result


def _write_outputs(
    output_dir: Path,
    estimates: list[FrameEstimate],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_direct_csv(output_dir / "ego_extrinsics_direct_pass.csv", estimates)
    _write_final_csv(output_dir / "ego_extrinsics_aligned.csv", estimates)
    aligned_json = {
        "coordinate_convention": "p_ego = T_ego_from_reference * p_reference",
        "frames": [_estimate_to_json(estimate) for estimate in estimates],
    }
    with (output_dir / "ego_extrinsics_aligned.json").open("w", encoding="utf-8") as f:
        json.dump(aligned_json, f, indent=2, allow_nan=True)
    with (output_dir / "ego_extrinsics_pose_dict.json").open("w", encoding="utf-8") as f:
        json.dump(_serialize_pose_dict(estimates, "frame_index"), f, indent=2, allow_nan=True)
    with (output_dir / "ego_extrinsics_pose_dict_by_ego_frame.json").open("w", encoding="utf-8") as f:
        json.dump(_serialize_pose_dict(estimates, "ego_frame_index"), f, indent=2, allow_nan=True)
    if summary.get("trajectory_smoothing", {}).get("enabled"):
        _write_smoothed_csv(output_dir / "ego_extrinsics_smoothed.csv", estimates)
        smoothed_json = {
            "coordinate_convention": "p_ego = T_ego_from_reference * p_reference",
            "frames": [_smoothed_estimate_to_json(estimate) for estimate in estimates],
        }
        with (output_dir / "ego_extrinsics_smoothed.json").open("w", encoding="utf-8") as f:
            json.dump(smoothed_json, f, indent=2, allow_nan=True)
        with (output_dir / "ego_extrinsics_pose_dict_smoothed.json").open("w", encoding="utf-8") as f:
            json.dump(
                _serialize_pose_dict(estimates, "frame_index", "pose_smoothed"),
                f,
                indent=2,
                allow_nan=True,
            )
        with (output_dir / "ego_extrinsics_pose_dict_smoothed_by_ego_frame.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                _serialize_pose_dict(estimates, "ego_frame_index", "pose_smoothed"),
                f,
                indent=2,
                allow_nan=True,
            )
    with (output_dir / "ego_extrinsics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)


def _run_optional_visualization(
    args: argparse.Namespace,
    episode_root: Path,
    output_dir: Path,
) -> Path | None:
    if not args.write_visualization_video:
        return None

    script_path = Path(__file__).resolve().with_name("visualize_pico_ego_extrinsics.py")
    if not script_path.is_file():
        raise FileNotFoundError(f"Visualization script not found: {script_path}")

    output_video = (
        Path(args.visualization_video).expanduser().resolve()
        if args.visualization_video
        else output_dir / "ego_trajectory_visualization.mp4"
    )
    command = [
        sys.executable,
        str(script_path),
        "--episode-dir",
        str(episode_root),
        "--ego-csv",
        str(output_dir / "ego_extrinsics_aligned.csv"),
        "--extrinsics-json",
        str(episode_root / "extrinsics.json"),
        "--output-video",
        str(output_video),
        "--fps",
        str(args.visualization_fps),
        "--stride",
        str(args.visualization_stride),
    ]
    if args.visualization_match_static_layout:
        command.append("--match-static-layout")

    print("[pico_ego_extrinsics] visualization_command=" + " ".join(command))
    proc = subprocess.run(command, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Visualization failed with exit code {proc.returncode}")
    return output_video


def estimate_pico_ego_extrinsics(args: argparse.Namespace) -> dict[str, Any]:
    episode_root = Path(args.episode_dir).expanduser().resolve()
    if not episode_root.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {episode_root}")

    timestamps_csv = Path(args.timestamps_csv).expanduser().resolve() if args.timestamps_csv else episode_root / "timestamps.csv"
    if not timestamps_csv.is_file():
        raise FileNotFoundError(f"Missing timestamps.csv: {timestamps_csv}")

    camera_params = _load_json(episode_root / "camera_params.json")
    extrinsics = _load_json(episode_root / "extrinsics.json")
    reference_camera_id = _format_camera_id(args.reference_camera_id)
    static_camera_ids = _parse_camera_id_list(args.static_camera_ids)
    reference_tag_ids = _parse_tag_id_list(args.reference_tag_ids)
    if reference_camera_id not in static_camera_ids:
        static_camera_ids.insert(0, reference_camera_id)

    static_cameras = _load_static_cameras(
        episode_root,
        camera_params,
        extrinsics,
        static_camera_ids,
        reference_camera_id,
    )
    ego_model, ego_video_path = _load_ego_model(
        episode_root,
        Path(args.fisheye_calibration).expanduser().resolve(),
        args.disable_fisheye_undistort,
    )
    rows = _load_timestamp_rows(timestamps_csv, args.max_rows)
    if not rows:
        raise RuntimeError(
            f"No rows with both frame_index and ego_frame_index in {timestamps_csv}"
        )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else episode_root / "ego_extrinsics_pico"
    detector = _create_aruco_detector(args.tag_family)
    reference_frame_numbers = _select_reference_frame_numbers(
        rows,
        args.reference_frame_count,
        args.reference_frame_sampling,
    )

    print(f"[pico_ego_extrinsics] episode={episode_root}")
    print(f"[pico_ego_extrinsics] output_dir={output_dir}")
    print(f"[pico_ego_extrinsics] timestamp_rows={len(rows)}")
    print(f"[pico_ego_extrinsics] reference_camera={reference_camera_id}")
    print(f"[pico_ego_extrinsics] static_cameras={','.join(camera.camera_id for camera in static_cameras)}")
    print(f"[pico_ego_extrinsics] reference_frames={','.join(str(v) for v in reference_frame_numbers)}")
    print(f"[pico_ego_extrinsics] ego_fisheye_model={ego_model.source}")
    print("[pico_ego_extrinsics] ego_frame_mapping=timestamps.csv:frame_index->ego_frame_index")
    print(f"[pico_ego_extrinsics] reference_map_mode={args.reference_map_mode}")

    reference_tag_corners, fused_tags, detected_reference_ids, static_video_stats = _build_reference_from_static_video_frames(
        static_cameras,
        reference_frame_numbers,
        detector,
        args.tag_size_m,
        args.force_temp_remux,
        reference_camera_id,
        args.reference_map_mode,
        args.reference_robust_huber_delta_px,
        args.reference_balance_cameras,
        args.reference_observation_outlier_px,
        reference_tag_ids,
    )
    if len(reference_tag_corners) < MIN_REFERENCE_TAGS:
        raise RuntimeError(
            f"Fixed reference tag count is too small: need at least {MIN_REFERENCE_TAGS}, "
            f"got {len(reference_tag_corners)}."
        )

    print(
        "[pico_ego_extrinsics] reference_detected_tag_ids="
        + (",".join(str(tag_id) for tag_id in sorted(detected_reference_ids)) if detected_reference_ids else "(none)")
    )
    print(
        "[pico_ego_extrinsics] reference_kept_tag_ids="
        + (",".join(str(tag_id) for tag_id in sorted(fused_tags)) if fused_tags else "(none)")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    estimates, ego_video_stats = _run_direct_ego_pass(
        rows,
        ego_video_path,
        ego_model,
        reference_tag_corners,
        detector,
        output_dir,
        args.write_debug_images,
        args.debug_image_limit,
        args.force_temp_remux,
    )
    interpolation_counts = _apply_interpolation(estimates, args.unbounded_gap_mode)
    if args.smooth_trajectory:
        smoothing_summary = _apply_savgol_smoothing(
            estimates,
            window_size=args.smoothing_window,
            polyorder=args.smoothing_polyorder,
        )
    else:
        smoothing_summary = {"enabled": False}

    used_apriltag_corners_output: Path | None = None
    used_apriltag_corners_row_count = 0
    if args.write_used_apriltag_corners:
        used_apriltag_corners_output = (
            Path(args.used_apriltag_corners_csv).expanduser().resolve()
            if args.used_apriltag_corners_csv
            else output_dir / "ego_used_apriltag_corners_ego.csv"
        )
        used_apriltag_corners_output.parent.mkdir(parents=True, exist_ok=True)
        used_apriltag_corners_row_count = _write_used_apriltag_corners_ego_csv(
            used_apriltag_corners_output,
            estimates,
            reference_tag_corners,
        )

    initial_status_counts = Counter(estimate.status_initial or "unknown" for estimate in estimates)
    final_status_counts = Counter(estimate.status_final or "unknown" for estimate in estimates)
    detected_ego_tag_ids = sorted({tag_id for estimate in estimates for tag_id in estimate.detected_tag_ids})
    used_ego_tag_ids = sorted({tag_id for estimate in estimates for tag_id in estimate.used_tag_ids})
    rmse_values = [estimate.rmse_px for estimate in estimates if estimate.rmse_px is not None]

    summary = {
        "episode_dir": str(episode_root),
        "timestamps_csv": str(timestamps_csv),
        "output_dir": str(output_dir),
        "coordinate_convention": "p_ego = T_ego_from_reference * p_reference",
        "reference_camera_id": reference_camera_id,
        "reference_map_mode": args.reference_map_mode,
        "reference_frame_sampling": args.reference_frame_sampling,
        "reference_robust_huber_delta_px": args.reference_robust_huber_delta_px,
        "reference_balance_cameras": args.reference_balance_cameras,
        "reference_observation_outlier_px": args.reference_observation_outlier_px,
        "reference_tag_ids": sorted(reference_tag_ids) if reference_tag_ids is not None else None,
        "static_camera_ids": [camera.camera_id for camera in static_cameras],
        "tag_family": args.tag_family,
        "tag_size_m": args.tag_size_m,
        "unbounded_gap_mode": args.unbounded_gap_mode,
        "timestamp_row_count": len(rows),
        "aligned_row_count": sum(bool(row.ego_frame_index) for row in rows),
        "ego_frame_index_space": "ego",
        "ego_frame_mapping": "timestamps.csv:frame_index->ego_frame_index",
        "reference_frame_numbers": reference_frame_numbers,
        "reference_detected_tag_ids": sorted(detected_reference_ids),
        "reference_kept_tag_ids": sorted(fused_tags),
        "reference_tag_count": len(reference_tag_corners),
        "reference_tags": {
            str(tag_id): {
                "detection_count": tag.detection_count,
                "candidate_count": tag.candidate_count,
                "inlier_count": tag.inlier_count,
                "reprojection_rmse_px": tag.reprojection_rmse_px,
            }
            for tag_id, tag in fused_tags.items()
        },
        "ego_fisheye": {
            "enabled": ego_model.enabled,
            "source": ego_model.source,
            "image_size": list(ego_model.image_size),
            "K_pnp": ego_model.K_pnp.tolist(),
        },
        "ego_detected_tag_ids": detected_ego_tag_ids,
        "ego_used_tag_ids": used_ego_tag_ids,
        "initial_status_counts": dict(sorted(initial_status_counts.items())),
        "final_status_counts": dict(sorted(final_status_counts.items())),
        "source_counts": interpolation_counts,
        "trajectory_smoothing": smoothing_summary,
        "rmse_px": {
            "count": len(rmse_values),
            "min": min(rmse_values) if rmse_values else None,
            "median": float(np.median(rmse_values)) if rmse_values else None,
            "max": max(rmse_values) if rmse_values else None,
        },
        "video_sources": {
            "static": static_video_stats,
            "ego": ego_video_stats,
        },
        "used_apriltag_corners_ego": {
            "enabled": bool(args.write_used_apriltag_corners),
            "output_csv": str(used_apriltag_corners_output) if used_apriltag_corners_output else None,
            "row_count": used_apriltag_corners_row_count,
            "coordinate_convention": "corner coordinates are in the ego RGB camera coordinate frame, meters",
        },
        "output_files": {
            "aligned_csv": str(output_dir / "ego_extrinsics_aligned.csv"),
            "aligned_json": str(output_dir / "ego_extrinsics_aligned.json"),
            "direct_pass_csv": str(output_dir / "ego_extrinsics_direct_pass.csv"),
            "pose_dict_by_frame_index": str(output_dir / "ego_extrinsics_pose_dict.json"),
            "pose_dict_by_ego_frame_index": str(output_dir / "ego_extrinsics_pose_dict_by_ego_frame.json"),
            "summary": str(output_dir / "ego_extrinsics_summary.json"),
        },
    }
    if args.smooth_trajectory:
        summary["output_files"].update(
            {
                "smoothed_csv": str(output_dir / "ego_extrinsics_smoothed.csv"),
                "smoothed_json": str(output_dir / "ego_extrinsics_smoothed.json"),
                "smoothed_pose_dict_by_frame_index": str(
                    output_dir / "ego_extrinsics_pose_dict_smoothed.json"
                ),
                "smoothed_pose_dict_by_ego_frame_index": str(
                    output_dir / "ego_extrinsics_pose_dict_smoothed_by_ego_frame.json"
                ),
            }
        )
    if used_apriltag_corners_output is not None:
        summary["output_files"]["used_apriltag_corners_ego_csv"] = str(used_apriltag_corners_output)
    _write_outputs(output_dir, estimates, summary)
    visualization_video = _run_optional_visualization(args, episode_root, output_dir)
    if visualization_video is not None:
        summary["output_files"]["visualization_video"] = str(visualization_video)
        with (output_dir / "ego_extrinsics_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=True)

    print(f"[pico_ego_extrinsics] direct_frames={interpolation_counts['direct']}")
    print(f"[pico_ego_extrinsics] interpolated_frames={interpolation_counts['interpolated']}")
    print(f"[pico_ego_extrinsics] extrapolated_frames={interpolation_counts['extrapolated']}")
    print(f"[pico_ego_extrinsics] nan_frames={interpolation_counts['nan']}")
    print(f"[pico_ego_extrinsics] output_csv={output_dir / 'ego_extrinsics_aligned.csv'}")
    print(f"[pico_ego_extrinsics] output_json={output_dir / 'ego_extrinsics_aligned.json'}")
    print(f"[pico_ego_extrinsics] output_pose_dict={output_dir / 'ego_extrinsics_pose_dict.json'}")
    if args.smooth_trajectory:
        print(
            "[pico_ego_extrinsics] smoothing_status_counts="
            + json.dumps(smoothing_summary.get("status_counts", {}), sort_keys=True)
        )
        print(
            "[pico_ego_extrinsics] output_smoothed_pose_dict="
            f"{output_dir / 'ego_extrinsics_pose_dict_smoothed.json'}"
        )
    if used_apriltag_corners_output is not None:
        print(
            f"[pico_ego_extrinsics] output_used_apriltag_corners={used_apriltag_corners_output} "
            f"rows={used_apriltag_corners_row_count}"
        )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate PICO ego RGB camera extrinsics aligned to third-person camera 00."
    )
    parser.add_argument("--episode-dir", "--path", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--timestamps-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-camera-id", type=str, default="00")
    parser.add_argument("--static-camera-ids", type=str, default="00,01,02,03,04,05")
    parser.add_argument(
        "--reference-map-mode",
        choices=("reference", "multiview"),
        default="reference",
        help=(
            "Build tag coordinates from the reference camera only (default), or "
            "jointly optimize every configured static camera."
        ),
    )
    parser.add_argument("--tag-family", type=str, default=APRILTAG_FAMILY)
    parser.add_argument("--tag-size-m", type=float, default=APRILTAG_SIZE_M)
    parser.add_argument(
        "--reference-tag-ids",
        type=str,
        default=None,
        help="Optional comma-separated allowlist of fixed AprilTag IDs used to build the map.",
    )
    parser.add_argument("--reference-frame-count", type=int, default=REFERENCE_FRAME_LIMIT)
    parser.add_argument(
        "--reference-frame-sampling",
        choices=("first", "uniform"),
        default="first",
        help="Use the first N aligned frames or sample N frames uniformly across the episode.",
    )
    parser.add_argument(
        "--reference-robust-huber-delta-px",
        type=float,
        default=None,
        help="Optional Huber transition in pixels for robust multiview tag-map optimization.",
    )
    parser.add_argument(
        "--reference-balance-cameras",
        action="store_true",
        help="Give each camera equal total weight for every tag regardless of observation count.",
    )
    parser.add_argument(
        "--reference-observation-outlier-px",
        type=float,
        default=None,
        help="Reject a camera/frame tag observation when its four-corner RMSE exceeds this value.",
    )
    parser.add_argument("--fisheye-calibration", type=Path, default=DEFAULT_FISHEYE_CALIBRATION)
    parser.add_argument("--disable-fisheye-undistort", action="store_true")
    parser.add_argument("--force-temp-remux", action="store_true")
    parser.add_argument(
        "--unbounded-gap-mode",
        choices=("nan", "extrapolate"),
        default="nan",
        help="How to handle leading/trailing frames that cannot be bracketed by direct poses.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--write-debug-images", action="store_true")
    parser.add_argument("--debug-image-limit", type=int, default=100)
    parser.add_argument(
        "--write-used-apriltag-corners",
        action="store_true",
        help="Write 3D ego-frame coordinates for AprilTag corners used by successful direct ego PnP frames.",
    )
    parser.add_argument(
        "--used-apriltag-corners-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for --write-used-apriltag-corners.",
    )
    parser.add_argument(
        "--smooth-trajectory",
        action="store_true",
        help=(
            "Write a separate Savitzky-Golay-smoothed ego trajectory. "
            "Raw/direct outputs are never overwritten."
        ),
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=SMOOTHING_WINDOW_LENGTH,
        help="Odd Savitzky-Golay window length (default: 11).",
    )
    parser.add_argument(
        "--smoothing-polyorder",
        type=int,
        default=SMOOTHING_POLYORDER,
        help="Savitzky-Golay polynomial order (default: 3).",
    )
    parser.add_argument("--write-visualization-video", action="store_true", help="Generate a visualization MP4 after saving extrinsics.")
    parser.add_argument("--visualization-video", type=Path, default=None, help="Optional visualization MP4 output path.")
    parser.add_argument("--visualization-fps", type=int, default=30)
    parser.add_argument("--visualization-stride", type=int, default=1)
    parser.add_argument("--visualization-match-static-layout", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.reference_frame_count < 1:
        parser.error("--reference-frame-count must be >= 1")
    if (
        args.reference_robust_huber_delta_px is not None
        and args.reference_robust_huber_delta_px <= 0.0
    ):
        parser.error("--reference-robust-huber-delta-px must be > 0")
    if (
        args.reference_observation_outlier_px is not None
        and args.reference_observation_outlier_px <= 0.0
    ):
        parser.error("--reference-observation-outlier-px must be > 0")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be >= 1")
    if args.smoothing_window < 3 or args.smoothing_window % 2 == 0:
        parser.error("--smoothing-window must be an odd integer >= 3")
    if args.smoothing_polyorder < 0 or args.smoothing_polyorder >= args.smoothing_window:
        parser.error("--smoothing-polyorder must be >= 0 and smaller than --smoothing-window")
    if args.visualization_fps < 1:
        parser.error("--visualization-fps must be >= 1")
    if args.visualization_stride < 1:
        parser.error("--visualization-stride must be >= 1")
    estimate_pico_ego_extrinsics(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
