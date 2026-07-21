#!/usr/bin/env python3
"""Live ego AprilTag overlay worker for the interaction view.

The C++ interaction UI stays deliberately small: it sends the current calibrated
Orbbec RGB views plus the latest ego video frame index, and this helper returns
world-frame line segments for the AprilTags seen by the ego camera.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - depends on deployment env
    cv2 = None
    CV2_ERROR = str(exc)
else:
    CV2_ERROR = ""


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import estimate_pico_ego_extrinsics as est
except Exception as exc:  # pragma: no cover - reported to the C++ UI
    est = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


DEFAULT_TAG_FAMILY = "tag36h11"
DEFAULT_TAG_SIZE_M = 0.096
MAX_RANSAC_REPROJ_ERROR_PX = 3.0

TAG_COLORS_BGR = [
    (40, 80, 255),
    (255, 180, 40),
    (80, 220, 80),
    (40, 220, 255),
    (255, 80, 220),
    (220, 220, 60),
    (255, 130, 130),
    (120, 255, 220),
]


@dataclass
class EgoImageModel:
    source: str
    K_raw: np.ndarray
    D_fisheye: np.ndarray | None
    K_pnp: np.ndarray
    dist_pnp: np.ndarray
    image_size: tuple[int, int]
    map1: np.ndarray | None = None
    map2: np.ndarray | None = None

    @property
    def fisheye_enabled(self) -> bool:
        return self.map1 is not None and self.map2 is not None and self.D_fisheye is not None


class LiveVideoFrameSource:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.current_index = -1

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.path = None
        self.current_index = -1

    def _open(self, path: Path) -> bool:
        if cv2 is None:
            return False
        self.close()
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        self.current_index = -1
        return bool(self.cap is not None and self.cap.isOpened())

    def read(self, path: Path, frame_index: int) -> tuple[np.ndarray | None, str]:
        if frame_index < 0:
            return None, "ego_frame_index_missing"
        if not path.is_file():
            self.close()
            return None, "ego_video_missing"
        if self.cap is None or self.path != path or frame_index < self.current_index:
            if not self._open(path):
                return None, "ego_video_open_failed"
        assert self.cap is not None
        frame = None
        while self.current_index < frame_index:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None, "ego_video_frame_not_ready"
            self.current_index += 1
        return frame, ""


class WorkerState:
    def __init__(self) -> None:
        self.detectors: dict[str, Any] = {}
        self.ego_models: dict[tuple[str, int], EgoImageModel] = {}
        self.video_source = LiveVideoFrameSource()
        self.last_response_time = 0.0
        self.smoothed_fps = 0.0

    def detector(self, tag_family: str) -> Any:
        key = str(tag_family or DEFAULT_TAG_FAMILY).lower()
        if key not in self.detectors:
            if est is None:
                raise RuntimeError("estimate_pico_ego_extrinsics import failed: " + IMPORT_ERROR)
            self.detectors[key] = est._create_aruco_detector(key)
        return self.detectors[key]

    def ego_model(self, camera_params_path: Path) -> EgoImageModel:
        stat_mtime = int(camera_params_path.stat().st_mtime_ns) if camera_params_path.is_file() else -1
        key = (str(camera_params_path), stat_mtime)
        cached = self.ego_models.get(key)
        if cached is not None:
            return cached
        model = load_ego_image_model(camera_params_path)
        self.ego_models.clear()
        self.ego_models[key] = model
        return model


def read_message() -> tuple[dict[str, Any], bytes] | None:
    header = sys.stdin.buffer.read(12)
    if not header:
        return None
    if len(header) != 12:
        raise RuntimeError("short request header")
    json_size, payload_size = struct.unpack("<IQ", header)
    payload_json = sys.stdin.buffer.read(json_size)
    if len(payload_json) != json_size:
        raise RuntimeError("short request json")
    payload = sys.stdin.buffer.read(payload_size)
    if len(payload) != payload_size:
        raise RuntimeError("short request payload")
    return json.loads(payload_json.decode("utf-8")), payload


def write_message(response: dict[str, Any]) -> None:
    data = json.dumps(response, separators=(",", ":"), allow_nan=False).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<IQ", len(data), 0))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def invert_transform(T_ab: np.ndarray) -> np.ndarray:
    R = np.asarray(T_ab, dtype=np.float64).reshape(4, 4)[:3, :3]
    t = np.asarray(T_ab, dtype=np.float64).reshape(4, 4)[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    return (np.asarray(T, dtype=np.float64).reshape(4, 4) @ homogeneous.T).T[:, :3]


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
    return rvec.reshape(3, 1)


def build_tag_local_corners(size_m: float) -> np.ndarray:
    half = 0.5 * float(size_m)
    return np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float64,
    )


def project_rmse(object_pts: np.ndarray, image_pts: np.ndarray, T_camera_from_object: np.ndarray, K: np.ndarray, dist: np.ndarray) -> float:
    projected, _ = cv2.projectPoints(
        np.asarray(object_pts, dtype=np.float64).reshape(-1, 3),
        matrix_to_rodrigues(T_camera_from_object[:3, :3]),
        T_camera_from_object[:3, 3].reshape(3, 1),
        K,
        dist,
    )
    residual = projected.reshape(-1, 2) - np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def solve_tag_pose(corners_px: np.ndarray, K: np.ndarray, dist: np.ndarray, local_corners: np.ndarray) -> tuple[np.ndarray, float] | None:
    object_pts = np.asarray(local_corners, dtype=np.float32).reshape(4, 3)
    image_pts = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
    candidates: list[tuple[float, np.ndarray]] = []

    if hasattr(cv2, "solvePnPGeneric") and hasattr(cv2, "SOLVEPNP_IPPE"):
        try:
            result = cv2.solvePnPGeneric(object_pts, image_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
            rvecs = result[1] if len(result) > 1 else ()
            tvecs = result[2] if len(result) > 2 else ()
            for rvec, tvec in zip(rvecs, tvecs):
                T = make_transform(rodrigues_to_matrix(rvec), np.asarray(tvec, dtype=np.float64).reshape(3))
                points_camera = transform_points(T, object_pts)
                if np.all(points_camera[:, 2] > 0.0):
                    candidates.append((project_rmse(object_pts, image_pts, T, K, dist), T))
        except cv2.error:
            pass

    if not candidates:
        try:
            ok, rvec, tvec = cv2.solvePnP(object_pts, image_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            ok = False
        if not ok:
            return None
        T = make_transform(rodrigues_to_matrix(rvec), np.asarray(tvec, dtype=np.float64).reshape(3))
        points_camera = transform_points(T, object_pts)
        if not np.all(points_camera[:, 2] > 0.0):
            return None
        candidates.append((project_rmse(object_pts, image_pts, T, K, dist), T))

    rmse, transform = min(candidates, key=lambda item: item[0])
    return transform, rmse


def detect_apriltags(image: np.ndarray | None, detector: Any) -> tuple[list[tuple[int, np.ndarray]], str]:
    if est is not None:
        return est._detect_apriltags(image, detector)
    if image is None:
        return [], "image_missing"
    return [], "estimate_module_missing"


def load_ego_image_model(camera_params_path: Path) -> EgoImageModel:
    if est is None:
        raise RuntimeError("estimate_pico_ego_extrinsics import failed: " + IMPORT_ERROR)
    if not camera_params_path.is_file():
        raise FileNotFoundError(f"ego camera params missing: {camera_params_path}")
    data = est._load_json(camera_params_path)
    entry = est._resolve_json_entry(data, "ego")
    K, dist, model_name = est._extract_intrinsics(entry, "ego")
    width, height = est._extract_image_size(entry, "ego")
    image_size = (int(width), int(height))

    rgb = entry.get("RGB", {})
    undistort = rgb.get("undistort", {}) if isinstance(rgb, dict) else {}
    new_intrinsic = undistort.get("new_intrinsic", {}) if isinstance(undistort, dict) else {}

    if "fisheye" in str(model_name).lower():
        D = np.asarray(dist, dtype=np.float64).reshape(4, 1)
        if isinstance(new_intrinsic, dict) and all(k in new_intrinsic for k in ("fx", "fy", "cx", "cy")):
            K_pnp = np.array(
                [
                    [float(new_intrinsic["fx"]), 0.0, float(new_intrinsic["cx"])],
                    [0.0, float(new_intrinsic["fy"]), float(new_intrinsic["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            source = "ego/camera_params.json undistort"
        else:
            K_pnp = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K,
                D,
                image_size,
                np.eye(3, dtype=np.float64),
                balance=1.0,
            )
            source = "fisheye_estimate_new_K"
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K,
            D,
            np.eye(3, dtype=np.float64),
            K_pnp,
            image_size,
            cv2.CV_16SC2,
        )
        return EgoImageModel(source, K, D, K_pnp, np.zeros((5, 1), dtype=np.float64), image_size, map1, map2)

    return EgoImageModel("raw_pinhole", K, None, K, np.asarray(dist, dtype=np.float64).reshape(-1, 1), image_size)


def detect_ego_apriltags(frame: np.ndarray | None, detector: Any, model: EgoImageModel) -> tuple[list[tuple[int, np.ndarray]], str, str]:
    if frame is None:
        return [], "image_missing", ""
    if not model.fisheye_enabled:
        detections, reason = detect_apriltags(frame, detector)
        return detections, reason, "raw"

    assert model.map1 is not None and model.map2 is not None and model.D_fisheye is not None
    undistorted = cv2.remap(frame, model.map1, model.map2, cv2.INTER_LINEAR)
    undistorted_detections, undistorted_reason = detect_apriltags(undistorted, detector)

    raw_detections, raw_reason = detect_apriltags(frame, detector)
    raw_projected: list[tuple[int, np.ndarray]] = []
    for tag_id, raw_corners in raw_detections:
        undistorted_points = cv2.fisheye.undistortPoints(
            np.asarray(raw_corners, dtype=np.float64).reshape(-1, 1, 2),
            model.K_raw,
            model.D_fisheye,
            R=np.eye(3, dtype=np.float64),
            P=model.K_pnp,
        ).reshape(4, 2)
        raw_projected.append((int(tag_id), undistorted_points.astype(np.float32)))

    if len(raw_projected) > len(undistorted_detections):
        return raw_projected, raw_reason, "raw_fisheye_corners_to_undistorted"
    return undistorted_detections, undistorted_reason, "undistorted_image"


def parse_camera_image(camera: dict[str, Any], payload: bytes) -> np.ndarray | None:
    width = int(camera.get("rgb_width", 0))
    height = int(camera.get("rgb_height", 0))
    offset = int(camera.get("rgb_offset", 0))
    size = int(camera.get("rgb_size", 0))
    expected = width * height * 3
    if width <= 0 or height <= 0 or size < expected or offset < 0 or offset + expected > len(payload):
        return None
    arr = np.frombuffer(payload, dtype=np.uint8, count=expected, offset=offset)
    return arr.reshape(height, width, 3).copy()


def camera_matrix(camera: dict[str, Any]) -> np.ndarray:
    intrinsic = camera.get("intrinsic", {})
    sx = float(camera.get("rgb_scale_x", 1.0)) or 1.0
    sy = float(camera.get("rgb_scale_y", 1.0)) or 1.0
    return np.array(
        [
            [float(intrinsic.get("fx", 0.0)) / sx, 0.0, float(intrinsic.get("cx", 0.0)) / sx],
            [0.0, float(intrinsic.get("fy", 0.0)) / sy, float(intrinsic.get("cy", 0.0)) / sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def camera_distortion(camera: dict[str, Any]) -> np.ndarray:
    distortion = camera.get("distortion", {})
    return np.array(
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
    ).reshape(-1, 1)


def camera_world_from_camera(camera: dict[str, Any]) -> np.ndarray:
    R = np.asarray(camera.get("Rwc", np.eye(3).reshape(-1).tolist()), dtype=np.float64).reshape(3, 3)
    t = np.asarray(camera.get("twc", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return make_transform(R, t)


def build_reference_tag_corners(
    request: dict[str, Any],
    payload: bytes,
    detector: Any,
    local_corners: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    observations: dict[int, list[np.ndarray]] = {}
    camera_count = 0
    detection_count = 0
    solved_count = 0

    for camera in request.get("cameras", []):
        image = parse_camera_image(camera, payload)
        if image is None:
            continue
        camera_count += 1
        K = camera_matrix(camera)
        if not (K[0, 0] > 0.0 and K[1, 1] > 0.0):
            continue
        dist = camera_distortion(camera)
        T_world_from_camera = camera_world_from_camera(camera)
        detections, _reason = detect_apriltags(image, detector)
        detection_count += len(detections)
        for tag_id, corners_px in detections:
            solved = solve_tag_pose(corners_px, K, dist, local_corners)
            if solved is None:
                continue
            T_camera_from_tag, _rmse = solved
            T_world_from_tag = T_world_from_camera @ T_camera_from_tag
            observations.setdefault(int(tag_id), []).append(transform_points(T_world_from_tag, local_corners))
            solved_count += 1

    reference: dict[int, np.ndarray] = {}
    for tag_id, corner_sets in observations.items():
        if not corner_sets:
            continue
        reference[tag_id] = np.mean(np.stack(corner_sets, axis=0), axis=0).astype(np.float64)

    stats = {
        "reference_camera_count": camera_count,
        "reference_detection_count": detection_count,
        "reference_solved_observations": solved_count,
        "reference_tag_count": len(reference),
    }
    return reference, stats


def solve_ego_from_world(
    detections: list[tuple[int, np.ndarray]],
    reference_corners: dict[int, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray | None, float | None, list[int], str]:
    object_pts: list[np.ndarray] = []
    image_pts: list[np.ndarray] = []
    used_ids: list[int] = []
    for tag_id, corners_px in detections:
        corners_world = reference_corners.get(int(tag_id))
        if corners_world is None:
            continue
        object_pts.append(np.asarray(corners_world, dtype=np.float32).reshape(4, 3))
        image_pts.append(np.asarray(corners_px, dtype=np.float32).reshape(4, 2))
        used_ids.append(int(tag_id))

    if not object_pts:
        return None, None, [], "no_shared_tags"

    obj = np.concatenate(object_pts, axis=0)
    img = np.concatenate(image_pts, axis=0)
    used_ids = sorted(set(used_ids))

    try:
        if len(used_ids) == 1 and obj.shape[0] == 4 and hasattr(cv2, "solvePnPGeneric") and hasattr(cv2, "SOLVEPNP_IPPE"):
            result = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
            rvecs = result[1] if len(result) > 1 else ()
            tvecs = result[2] if len(result) > 2 else ()
            candidates: list[tuple[float, np.ndarray]] = []
            for rvec_candidate, tvec_candidate in zip(rvecs, tvecs):
                T_candidate = make_transform(rodrigues_to_matrix(rvec_candidate), np.asarray(tvec_candidate, dtype=np.float64).reshape(3))
                points_ego = transform_points(T_candidate, obj)
                if np.all(points_ego[:, 2] > 0.0):
                    candidates.append((project_rmse(obj, img, T_candidate, K, dist), T_candidate))
            if not candidates:
                return None, None, used_ids, "ego_pnp_failed"
            rmse, T_ego_from_world = min(candidates, key=lambda item: item[0])
            return T_ego_from_world, rmse, used_ids, "ok"
        if len(used_ids) >= 2 and obj.shape[0] >= 8:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                img,
                K,
                dist,
                iterationsCount=120,
                reprojectionError=MAX_RANSAC_REPROJ_ERROR_PX,
                confidence=0.995,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok or inliers is None or len(inliers) < 4:
                return None, None, used_ids, "ego_pnp_failed"
        else:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None, None, used_ids, "ego_pnp_failed"
    except cv2.error:
        return None, None, used_ids, "ego_pnp_error"

    T_ego_from_world = make_transform(rodrigues_to_matrix(rvec), np.asarray(tvec, dtype=np.float64).reshape(3))
    rmse = project_rmse(obj, img, T_ego_from_world, K, dist)
    if not math.isfinite(rmse):
        return None, None, used_ids, "ego_pnp_nan"
    return T_ego_from_world, rmse, used_ids, "ok"


def color_for_tag(tag_id: int) -> tuple[int, int, int]:
    return TAG_COLORS_BGR[int(tag_id) % len(TAG_COLORS_BGR)]


def append_line(lines: list[list[float]], p0: np.ndarray, p1: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    lines.append(
        [
            float(p0[0]),
            float(p0[1]),
            float(p0[2]),
            float(p1[0]),
            float(p1[1]),
            float(p1[2]),
            float(color_bgr[0]),
            float(color_bgr[1]),
            float(color_bgr[2]),
        ]
    )


def append_tag_geometry(lines: list[list[float]], T_world_from_tag: np.ndarray, local_corners: np.ndarray, tag_id: int) -> None:
    color = color_for_tag(tag_id)
    corners = transform_points(T_world_from_tag, local_corners)
    for i in range(4):
        append_line(lines, corners[i], corners[(i + 1) % 4], color)
    axis_size = max(0.025, float(np.linalg.norm(local_corners[1] - local_corners[0])) * 0.45)
    origin = transform_points(T_world_from_tag, np.array([[0.0, 0.0, 0.0]], dtype=np.float64))[0]
    axes = transform_points(
        T_world_from_tag,
        np.array([[axis_size, 0.0, 0.0], [0.0, axis_size, 0.0], [0.0, 0.0, axis_size]], dtype=np.float64),
    )
    append_line(lines, origin, axes[0], (40, 40, 255))
    append_line(lines, origin, axes[1], (40, 220, 40))
    append_line(lines, origin, axes[2], (255, 80, 40))


def append_camera_geometry(lines: list[list[float]], T_world_from_ego: np.ndarray, K: np.ndarray, image_size: tuple[int, int]) -> None:
    width, height = image_size
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    if not (fx > 0.0 and fy > 0.0):
        return
    depth_m = 0.22
    pixels = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)
    corners_ego = np.column_stack(
        [
            (pixels[:, 0] - cx) / fx * depth_m,
            (pixels[:, 1] - cy) / fy * depth_m,
            np.full(4, depth_m, dtype=np.float64),
        ]
    )
    origin = transform_points(T_world_from_ego, np.array([[0.0, 0.0, 0.0]], dtype=np.float64))[0]
    corners = transform_points(T_world_from_ego, corners_ego)
    for corner in corners:
        append_line(lines, origin, corner, (210, 210, 210))
    for i in range(4):
        append_line(lines, corners[i], corners[(i + 1) % 4], (210, 210, 210))


def handle_frame_request(state: WorkerState, request: dict[str, Any], payload: bytes) -> dict[str, Any]:
    started = time.monotonic()
    frame_id = int(request.get("frame_id", 0))
    if cv2 is None:
        return {
            "ok": False,
            "frame_id": frame_id,
            "status": "OpenCV unavailable in Python: " + CV2_ERROR,
            "lines": [],
        }
    tag_family = str(request.get("tag_family", DEFAULT_TAG_FAMILY) or DEFAULT_TAG_FAMILY)
    tag_size_m = float(request.get("tag_size_m", DEFAULT_TAG_SIZE_M) or DEFAULT_TAG_SIZE_M)
    detector = state.detector(tag_family)
    local_corners = build_tag_local_corners(tag_size_m)

    reference_corners, ref_stats = build_reference_tag_corners(request, payload, detector, local_corners)
    if not reference_corners:
        return {
            "ok": False,
            "frame_id": frame_id,
            "status": "waiting for Orbbec AprilTags",
            "lines": [],
            **ref_stats,
        }

    video_path = Path(str(request.get("ego_video_path", ""))).expanduser()
    camera_params_path = Path(str(request.get("ego_camera_params_path", ""))).expanduser()
    ego_frame_index = int(request.get("ego_video_frame_index", -1))

    try:
        ego_model = state.ego_model(camera_params_path)
    except Exception as exc:
        return {
            "ok": False,
            "frame_id": frame_id,
            "status": "ego camera params unavailable: " + str(exc),
            "lines": [],
            **ref_stats,
        }

    ego_frame, video_reason = state.video_source.read(video_path, ego_frame_index)
    if ego_frame is None:
        return {
            "ok": False,
            "frame_id": frame_id,
            "ego_video_frame_index": ego_frame_index,
            "status": video_reason,
            "lines": [],
            **ref_stats,
        }

    detections, detection_reason, detection_space = detect_ego_apriltags(ego_frame, detector, ego_model)
    detected_ids = sorted({int(tag_id) for tag_id, _ in detections})
    if not detections:
        return {
            "ok": False,
            "frame_id": frame_id,
            "ego_video_frame_index": ego_frame_index,
            "status": "ego no AprilTags" if not detection_reason else "ego " + detection_reason,
            "detected_tag_ids": detected_ids,
            "lines": [],
            **ref_stats,
        }

    T_ego_from_world, rmse, used_ids, status = solve_ego_from_world(detections, reference_corners, ego_model.K_pnp, ego_model.dist_pnp)
    if T_ego_from_world is None:
        return {
            "ok": False,
            "frame_id": frame_id,
            "ego_video_frame_index": ego_frame_index,
            "status": status,
            "detected_tag_ids": detected_ids,
            "used_tag_ids": used_ids,
            "lines": [],
            **ref_stats,
        }

    T_world_from_ego = invert_transform(T_ego_from_world)
    lines: list[list[float]] = []
    append_camera_geometry(lines, T_world_from_ego, ego_model.K_pnp, ego_model.image_size)

    solved_ids: list[int] = []
    per_tag_rmses: list[float] = []
    for tag_id, corners_px in detections:
        solved = solve_tag_pose(corners_px, ego_model.K_pnp, ego_model.dist_pnp, local_corners)
        if solved is None:
            continue
        T_ego_from_tag, tag_rmse = solved
        T_world_from_tag = T_world_from_ego @ T_ego_from_tag
        append_tag_geometry(lines, T_world_from_tag, local_corners, int(tag_id))
        solved_ids.append(int(tag_id))
        per_tag_rmses.append(float(tag_rmse))

    now = time.monotonic()
    if state.last_response_time > 0.0:
        dt = now - state.last_response_time
        if dt > 1e-6:
            inst = 1.0 / dt
            state.smoothed_fps = 0.8 * state.smoothed_fps + 0.2 * inst if state.smoothed_fps > 0.0 else inst
    state.last_response_time = now
    elapsed_ms = (now - started) * 1000.0
    status_text = f"ego tags {len(solved_ids)}/{len(detected_ids)} ref {len(reference_corners)} rmse {rmse:.2f}px"

    return {
        "ok": True,
        "frame_id": frame_id,
        "ego_video_frame_index": ego_frame_index,
        "status": status_text,
        "detected_tag_ids": detected_ids,
        "used_tag_ids": used_ids,
        "solved_tag_ids": sorted(set(solved_ids)),
        "tag_count": len(set(solved_ids)),
        "reference_tag_count": len(reference_corners),
        "rmse_px": float(rmse) if rmse is not None else None,
        "median_tag_rmse_px": float(np.median(per_tag_rmses)) if per_tag_rmses else None,
        "detection_space": detection_space,
        "ego_fisheye_source": ego_model.source,
        "worker_fps": state.smoothed_fps,
        "elapsed_ms": elapsed_ms,
        "lines": lines,
        **ref_stats,
    }


def main() -> int:
    state = WorkerState()
    while True:
        try:
            msg = read_message()
            if msg is None:
                return 0
            request, payload = msg
            if request.get("type") != "frame":
                write_message({"ok": False, "status": "unsupported_request", "lines": []})
                continue
            response = handle_frame_request(state, request, payload)
            write_message(response)
        except Exception as exc:
            write_message({"ok": False, "status": "worker_error: " + str(exc), "lines": []})


if __name__ == "__main__":
    raise SystemExit(main())
