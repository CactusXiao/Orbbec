#!/usr/bin/env python3
"""
Calibrate an ego camera against a fixed AprilTag reference built from static cameras.

Expected episode layout:

    episode_dir/
      camera_params.json
      extrinsics.json
      00/RGB/00000.jpg
      01/RGB/00000.jpg
      ...
      08/RGB/00000.jpg

The script:
1. Uses the first 10 common frames from static third-person cameras to build a fixed
   AprilTag map in the reference camera coordinate system.
2. Solves the ego camera extrinsic for every ego frame against that fixed tag map.
3. Applies temporal cleanup with outlier replacement and local interpolation.
4. Writes per-frame ego extrinsics to:

       <episode_dir>/ego_extrinsics_camXX.json

The output matrices follow the same convention as extrinsics.json:

    p_camera = T_camera_from_reference * p_reference

where the reference frame is the first static camera in the episode.

Usage:

    python3 ego_apriltag_calib.py --path /path/to/episode --ego-idx 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


APRILTAG_FAMILY = "tag36h11"
APRILTAG_SIZE_M = 0.096
REFERENCE_FRAME_LIMIT = 10
MIN_REFERENCE_COMMON_FRAMES = 3
MIN_REFERENCE_TAGS = 2
MIN_REFERENCE_INLIER_OBSERVATIONS = 3
MIN_EGO_TAGS = 2
MIN_EGO_INLIER_CORNERS = 8
SINGLE_TAG_REPROJ_LIMIT_PX = 4.0
EGO_RANSAC_REPROJ_ERROR_PX = 2.0
EGO_PNP_ITERATIONS = 200
EGO_REPROJ_RMSE_LIMIT_PX = 4.0
FUSION_TRANS_THRESH_M = 0.08
FUSION_ROT_THRESH_DEG = 8.0
OUTLIER_TRANS_THRESH_M = 0.08
OUTLIER_ROT_THRESH_DEG = 8.0
MAX_INTERP_GAP = 5
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

APRILTAG_DICT_NAMES: dict[str, str] = {
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
    T_camera_from_world: np.ndarray | None
    T_camera_from_reference: np.ndarray | None = None
    T_reference_from_camera: np.ndarray | None = None
    rgb_dir: Path | None = None


@dataclass
class TagObservation:
    tag_id: int
    T_reference_from_tag: np.ndarray
    rmse: float
    camera_id: str
    frame_index: str


@dataclass
class TagFusionResult:
    tag_id: int
    T_reference_from_tag: np.ndarray
    corners_reference: np.ndarray
    detection_count: int
    candidate_count: int
    inlier_count: int


@dataclass
class FrameEstimate:
    frame_index: str
    detected_tag_ids: list[int] = field(default_factory=list)
    used_tag_ids: list[int] = field(default_factory=list)
    rmse: float | None = None
    pose_raw: np.ndarray | None = None
    pose_final: np.ndarray | None = None
    status_initial: str = ""
    status_final: str = ""


def _format_camera_id(camera_id: int | str) -> str:
    if isinstance(camera_id, int):
        return f"{camera_id:02d}"
    text = str(camera_id).strip()
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def _camera_sort_key(camera_id: str) -> tuple[int, int | str]:
    if camera_id.isdigit():
        return (0, int(camera_id))
    return (1, camera_id)


def _candidate_camera_ids(camera_id: int | str) -> list[str]:
    text = str(camera_id).strip()
    candidates: list[str] = []
    for value in (text, _format_camera_id(text)):
        if value and value not in candidates:
            candidates.append(value)
    if text.isdigit():
        for value in (str(int(text)), f"{int(text):02d}"):
            if value not in candidates:
                candidates.append(value)
    return candidates


def _resolve_json_entry(data: dict[str, Any], camera_id: str, *, strict: bool) -> dict[str, Any] | None:
    for key in _candidate_camera_ids(camera_id):
        if key in data:
            entry = data[key]
            if isinstance(entry, dict):
                return entry
    if strict:
        raise KeyError(f"Camera {camera_id} not found in json data")
    return None


def _find_camera_rgb_dir(root: Path, camera_id: str) -> Path | None:
    for key in _candidate_camera_ids(camera_id):
        rgb_dir = root / key / "RGB"
        if rgb_dir.is_dir():
            return rgb_dir
    return None


def _list_frame_paths(rgb_dir: Path) -> dict[str, Path]:
    frames: dict[str, Path] = {}
    for ext in IMAGE_EXTENSIONS:
        for path in rgb_dir.glob(f"*{ext}"):
            if not path.is_file():
                continue
            frames.setdefault(path.stem, path)
    return frames


def _frame_sort_key(frame_index: str) -> tuple[int, int | str]:
    if frame_index.isdigit():
        return (0, int(frame_index))
    return (1, frame_index)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _extract_intrinsics(entry: dict[str, Any], camera_id: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = entry.get("RGB")
    if not isinstance(rgb, dict):
        raise KeyError(f"Camera {camera_id} missing RGB block in camera_params.json")

    intrinsic = rgb.get("intrinsic")
    distortion = rgb.get("distortion")
    if not isinstance(intrinsic, dict) or not isinstance(distortion, dict):
        raise KeyError(f"Camera {camera_id} missing RGB intrinsic/distortion")

    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
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
    return K, dist


def _extract_extrinsic_camera_from_world(entry: dict[str, Any], camera_id: str) -> np.ndarray:
    if "rotation" not in entry or "translation" not in entry:
        raise KeyError(f"Camera {camera_id} missing rotation/translation in extrinsics.json")
    R = np.asarray(entry["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(entry["translation"], dtype=np.float64).reshape(3)
    if not _is_valid_rotation(R):
        raise ValueError(f"Camera {camera_id} has invalid rotation matrix in extrinsics.json")
    return _make_transform(R, t)


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
    det = np.linalg.det(R)
    return np.allclose(R.T @ R, np.eye(3), atol=atol) and abs(det - 1.0) < atol


def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def _matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
    return rvec.reshape(3, 1)


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = np.asarray(R_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    cos_theta = max(-1.0, min(1.0, (float(np.trace(R_rel)) - 1.0) / 2.0))
    return math.degrees(math.acos(cos_theta))


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


def _average_quaternions(quaternions: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not quaternions:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    anchor = quaternions[0]
    accum = np.zeros(4, dtype=np.float64)
    for quat, weight in zip(quaternions, weights):
        q = np.asarray(quat, dtype=np.float64).reshape(4)
        if np.dot(anchor, q) < 0.0:
            q = -q
        accum += float(weight) * q
    norm = np.linalg.norm(accum)
    if norm < 1e-12:
        return anchor / np.linalg.norm(anchor)
    return accum / norm


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
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    out = s0 * q0 + s1 * q1
    return out / np.linalg.norm(out)


def _interpolate_transform(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(max(0.0, min(1.0, alpha)))
    t0 = np.asarray(T0[:3, 3], dtype=np.float64)
    t1 = np.asarray(T1[:3, 3], dtype=np.float64)
    q0 = _matrix_to_quaternion(T0[:3, :3])
    q1 = _matrix_to_quaternion(T1[:3, :3])
    t = (1.0 - alpha) * t0 + alpha * t1
    q = _slerp_quaternion(q0, q1, alpha)
    return _make_transform(_quaternion_to_matrix(q), t)


def _pose_difference(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    trans = float(np.linalg.norm(np.asarray(T_a[:3, 3]) - np.asarray(T_b[:3, 3])))
    rot = _rotation_angle_deg(T_a[:3, :3], T_b[:3, :3])
    return trans, rot


def _create_aruco_detector(tag_family: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV build has no aruco module. Install opencv-contrib-python.")
    family_key = str(tag_family).lower()
    dict_name = APRILTAG_DICT_NAMES.get(family_key)
    if dict_name is None or not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unsupported AprilTag family: {tag_family}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return (dictionary, params)


def _detect_apriltags(gray: np.ndarray, detector) -> tuple[list[Any], Any]:
    if isinstance(detector, tuple):
        dictionary, params = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        return corners, ids
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def _detect_apriltag_markers(image: np.ndarray, detector) -> tuple[list[tuple[int, np.ndarray]], str]:
    if image is None:
        return [], "image_missing"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_list, ids = _detect_apriltags(gray, detector)
    if ids is None or len(ids) == 0:
        return [], "tags_not_found"
    detections: list[tuple[int, np.ndarray]] = []
    for corners, tag_id_array in zip(corners_list, ids):
        detections.append((int(tag_id_array[0]), np.asarray(corners, dtype=np.float32).reshape(4, 2)))
    return detections, ""


def _build_tag_local_corners(size_m: float) -> np.ndarray:
    half = 0.5 * float(size_m)
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float32,
    )


def _solve_single_tag_pnp(corners_px: np.ndarray, K: np.ndarray, dist: np.ndarray) -> tuple[bool, np.ndarray | None, float]:
    object_pts = _build_tag_local_corners(APRILTAG_SIZE_M)
    image_pts = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_pts,
        image_pts,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        return False, None, float("inf")
    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    reproj = np.linalg.norm(proj.reshape(-1, 2) - image_pts, axis=1)
    rmse = float(np.sqrt(np.mean(reproj * reproj)))
    if rmse > SINGLE_TAG_REPROJ_LIMIT_PX:
        return False, None, rmse
    T_camera_from_tag = _make_transform(_rodrigues_to_matrix(rvec), tvec.reshape(3))
    return True, T_camera_from_tag, rmse


def _fuse_tag_observations(
    tag_id: int,
    observations: list[TagObservation],
    detection_count: int,
    local_tag_corners: np.ndarray,
) -> TagFusionResult | None:
    if len(observations) < MIN_REFERENCE_INLIER_OBSERVATIONS:
        return None

    best_inliers: list[int] = []
    for idx_hypo, hypo in enumerate(observations):
        current: list[int] = []
        t_h = hypo.T_reference_from_tag[:3, 3]
        R_h = hypo.T_reference_from_tag[:3, :3]
        for idx_obs, obs in enumerate(observations):
            trans_err = float(np.linalg.norm(obs.T_reference_from_tag[:3, 3] - t_h))
            rot_err = _rotation_angle_deg(R_h, obs.T_reference_from_tag[:3, :3])
            if trans_err <= FUSION_TRANS_THRESH_M and rot_err <= FUSION_ROT_THRESH_DEG:
                current.append(idx_obs)
        if len(current) > len(best_inliers):
            best_inliers = current
        elif len(current) == len(best_inliers) and current:
            current_rmse = sum(observations[i].rmse for i in current)
            best_rmse = sum(observations[i].rmse for i in best_inliers) if best_inliers else float("inf")
            if current_rmse < best_rmse:
                best_inliers = current

    if len(best_inliers) < MIN_REFERENCE_INLIER_OBSERVATIONS:
        return None

    inlier_obs = [observations[i] for i in best_inliers]
    weights = np.array([1.0 / max(obs.rmse, 1e-6) for obs in inlier_obs], dtype=np.float64)
    weights /= np.sum(weights)

    translations = np.array([obs.T_reference_from_tag[:3, 3] for obs in inlier_obs], dtype=np.float64)
    translation = np.sum(translations * weights[:, None], axis=0)
    quaternions = [_matrix_to_quaternion(obs.T_reference_from_tag[:3, :3]) for obs in inlier_obs]
    rotation = _quaternion_to_matrix(_average_quaternions(quaternions, weights))
    T_reference_from_tag = _make_transform(rotation, translation)
    corners_reference = (local_tag_corners @ rotation.T + translation.reshape(1, 3)).astype(np.float32)
    return TagFusionResult(
        tag_id=tag_id,
        T_reference_from_tag=T_reference_from_tag,
        corners_reference=corners_reference,
        detection_count=detection_count,
        candidate_count=len(observations),
        inlier_count=len(inlier_obs),
    )


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

    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    reproj = np.linalg.norm(proj.reshape(-1, 2) - image_pts, axis=1)
    rmse = float(np.sqrt(np.mean(reproj * reproj)))
    if rmse > EGO_REPROJ_RMSE_LIMIT_PX:
        return None, rmse, used_tag_ids, "high_reproj_error"

    T_ego_from_reference = _make_transform(_rodrigues_to_matrix(rvec), tvec.reshape(3))
    return T_ego_from_reference, rmse, used_tag_ids, "ok"


def _read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _build_reference_from_static_frames(
    static_cameras: list[CameraCalibration],
    static_frame_paths: dict[str, dict[str, Path]],
    reference_frames: list[str],
    detector,
) -> tuple[dict[int, np.ndarray], dict[int, TagFusionResult], set[int]]:
    local_tag_corners = _build_tag_local_corners(APRILTAG_SIZE_M)
    detection_counts: dict[int, int] = {}
    observations_by_tag: dict[int, list[TagObservation]] = {}
    detected_ids: set[int] = set()

    for frame_index in reference_frames:
        for camera in static_cameras:
            image_path = static_frame_paths[camera.camera_id][frame_index]
            image = _read_image(image_path)
            detections, _ = _detect_apriltag_markers(image, detector)
            for tag_id, corners_px in detections:
                detected_ids.add(tag_id)
                detection_counts[tag_id] = detection_counts.get(tag_id, 0) + 1
                ok, T_camera_from_tag, rmse = _solve_single_tag_pnp(corners_px, camera.K, camera.dist)
                if not ok or T_camera_from_tag is None or camera.T_reference_from_camera is None:
                    continue
                T_reference_from_tag = _compose(camera.T_reference_from_camera, T_camera_from_tag)
                observations_by_tag.setdefault(tag_id, []).append(
                    TagObservation(
                        tag_id=tag_id,
                        T_reference_from_tag=T_reference_from_tag,
                        rmse=rmse,
                        camera_id=camera.camera_id,
                        frame_index=frame_index,
                    )
                )

    fused_tags: dict[int, TagFusionResult] = {}
    reference_tag_corners: dict[int, np.ndarray] = {}
    for tag_id in sorted(observations_by_tag):
        fused = _fuse_tag_observations(
            tag_id=tag_id,
            observations=observations_by_tag[tag_id],
            detection_count=detection_counts.get(tag_id, 0),
            local_tag_corners=local_tag_corners,
        )
        if fused is None:
            continue
        fused_tags[tag_id] = fused
        reference_tag_corners[tag_id] = fused.corners_reference

    return reference_tag_corners, fused_tags, detected_ids


def _apply_temporal_postprocess(estimates: list[FrameEstimate]) -> dict[str, int]:
    stats = {
        "outlier_replaced": 0,
        "interpolated_gap": 0,
        "long_gap_interpolated": 0,
        "edge_filled": 0,
    }

    for estimate in estimates:
        estimate.pose_final = None if estimate.pose_raw is None else np.array(estimate.pose_raw, copy=True)
        estimate.status_final = estimate.status_initial

    for idx in range(1, len(estimates) - 1):
        prev_est = estimates[idx - 1]
        cur_est = estimates[idx]
        next_est = estimates[idx + 1]
        if prev_est.pose_raw is None or cur_est.pose_raw is None or next_est.pose_raw is None:
            continue
        predicted = _interpolate_transform(prev_est.pose_raw, next_est.pose_raw, 0.5)
        trans_err, rot_err = _pose_difference(cur_est.pose_raw, predicted)
        if trans_err > OUTLIER_TRANS_THRESH_M or rot_err > OUTLIER_ROT_THRESH_DEG:
            cur_est.pose_final = predicted
            cur_est.status_final = "outlier_replaced"
            stats["outlier_replaced"] += 1

    valid_indices = [idx for idx, estimate in enumerate(estimates) if estimate.pose_final is not None]
    if not valid_indices:
        _print_failure_debug(estimates, "no_valid_ego_pose")
        raise RuntimeError("No valid ego pose could be solved; cannot build output trajectory.")

    gap_start = 0
    while gap_start < len(estimates):
        if estimates[gap_start].pose_final is not None:
            gap_start += 1
            continue
        gap_end = gap_start
        while gap_end < len(estimates) and estimates[gap_end].pose_final is None:
            gap_end += 1

        prev_idx = gap_start - 1 if gap_start > 0 else None
        next_idx = gap_end if gap_end < len(estimates) else None
        has_prev = prev_idx is not None and estimates[prev_idx].pose_final is not None
        has_next = next_idx is not None and estimates[next_idx].pose_final is not None
        gap_len = gap_end - gap_start

        if has_prev and has_next:
            for offset, idx in enumerate(range(gap_start, gap_end), start=1):
                alpha = offset / float(gap_len + 1)
                estimates[idx].pose_final = _interpolate_transform(
                    estimates[prev_idx].pose_final,
                    estimates[next_idx].pose_final,
                    alpha,
                )
                if gap_len <= MAX_INTERP_GAP:
                    estimates[idx].status_final = "interpolated"
                    stats["interpolated_gap"] += 1
                else:
                    estimates[idx].status_final = "interpolated_long_gap"
                    stats["long_gap_interpolated"] += 1
        elif has_prev:
            for idx in range(gap_start, gap_end):
                estimates[idx].pose_final = np.array(estimates[prev_idx].pose_final, copy=True)
                estimates[idx].status_final = "edge_filled_backward"
                stats["edge_filled"] += 1
        elif has_next:
            for idx in range(gap_start, gap_end):
                estimates[idx].pose_final = np.array(estimates[next_idx].pose_final, copy=True)
                estimates[idx].status_final = "edge_filled_forward"
                stats["edge_filled"] += 1
        else:
            _print_failure_debug(estimates, "temporal_fill_failed")
            raise RuntimeError("Could not fill missing ego poses: no valid neighboring poses found.")

        gap_start = gap_end

    return stats


def _serialize_pose_dict(estimates: list[FrameEstimate]) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = {}
    for estimate in estimates:
        if estimate.pose_final is None:
            raise RuntimeError(f"Frame {estimate.frame_index} has no final pose after temporal postprocess")
        result[estimate.frame_index] = np.asarray(estimate.pose_final, dtype=np.float64).tolist()
    return result


def _print_failure_debug(estimates: list[FrameEstimate], message: str, *, max_examples: int = 20) -> None:
    total_frames = len(estimates)
    initial_success_frames = sum(int(estimate.pose_raw is not None) for estimate in estimates)
    final_success_frames = sum(int(estimate.pose_final is not None) for estimate in estimates)
    status_counts = Counter(estimate.status_initial or "unknown" for estimate in estimates)

    print(f"[ego_apriltag_calib] debug_failure={message}", file=sys.stderr)
    print(f"[ego_apriltag_calib] debug_total_frames={total_frames}", file=sys.stderr)
    print(f"[ego_apriltag_calib] debug_initial_success_frames={initial_success_frames}", file=sys.stderr)
    print(f"[ego_apriltag_calib] debug_final_success_frames={final_success_frames}", file=sys.stderr)
    print(
        "[ego_apriltag_calib] debug_status_counts="
        + ", ".join(f"{key}:{status_counts[key]}" for key in sorted(status_counts)),
        file=sys.stderr,
    )

    failed_estimates = [estimate for estimate in estimates if estimate.pose_raw is None]
    if not failed_estimates:
        failed_estimates = [estimate for estimate in estimates if estimate.pose_final is None]

    for estimate in failed_estimates[:max_examples]:
        rmse_text = "None" if estimate.rmse is None else f"{estimate.rmse:.4f}"
        print(
            "[ego_apriltag_calib] debug_frame "
            f"frame={estimate.frame_index} "
            f"status_initial={estimate.status_initial or 'unknown'} "
            f"status_final={estimate.status_final or 'unknown'} "
            f"detected={estimate.detected_tag_ids} "
            f"used={estimate.used_tag_ids} "
            f"rmse={rmse_text}",
            file=sys.stderr,
        )


def calibrate_ego_apriltag(path: str | Path, ego_idx: int | str = 8) -> dict[str, list[list[float]]]:
    episode_root = Path(path).expanduser().resolve()
    if not episode_root.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {episode_root}")

    camera_params_path = episode_root / "camera_params.json"
    extrinsics_path = episode_root / "extrinsics.json"
    if not camera_params_path.is_file():
        raise FileNotFoundError(f"Missing camera_params.json under {episode_root}")
    if not extrinsics_path.is_file():
        raise FileNotFoundError(f"Missing extrinsics.json under {episode_root}")

    ego_id = _format_camera_id(ego_idx)
    ego_rgb_dir = _find_camera_rgb_dir(episode_root, ego_id)
    if ego_rgb_dir is None:
        raise FileNotFoundError(f"Ego camera RGB directory not found for camera {ego_id} under {episode_root}")

    camera_params = _load_json(camera_params_path)
    extrinsics = _load_json(extrinsics_path)

    ego_camera_entry = _resolve_json_entry(camera_params, ego_id, strict=True)
    ego_K, ego_dist = _extract_intrinsics(ego_camera_entry, ego_id)
    ego_frames = _list_frame_paths(ego_rgb_dir)
    if not ego_frames:
        raise RuntimeError(f"No ego RGB frames found under {ego_rgb_dir}")

    static_cameras: list[CameraCalibration] = []
    static_frame_paths: dict[str, dict[str, Path]] = {}
    for child in sorted(episode_root.iterdir(), key=lambda p: _camera_sort_key(p.name)):
        if not child.is_dir() or child.name.lower() == "fisheye":
            continue
        rgb_dir = child / "RGB"
        if not rgb_dir.is_dir():
            continue
        camera_id = child.name
        if _format_camera_id(camera_id) == ego_id:
            continue
        try:
            cam_param_entry = _resolve_json_entry(camera_params, camera_id, strict=True)
            extr_entry = _resolve_json_entry(extrinsics, camera_id, strict=True)
        except KeyError:
            continue
        frame_paths = _list_frame_paths(rgb_dir)
        if not frame_paths:
            continue
        K, dist = _extract_intrinsics(cam_param_entry, camera_id)
        T_camera_from_world = _extract_extrinsic_camera_from_world(extr_entry, camera_id)
        calibration = CameraCalibration(
            camera_id=camera_id,
            K=K,
            dist=dist,
            T_camera_from_world=T_camera_from_world,
            rgb_dir=rgb_dir,
        )
        static_cameras.append(calibration)
        static_frame_paths[camera_id] = frame_paths

    static_cameras.sort(key=lambda cam: _camera_sort_key(cam.camera_id))
    if not static_cameras:
        raise RuntimeError("No static third-person cameras with intrinsics, extrinsics, and RGB frames were found.")

    reference_camera = static_cameras[0]
    if reference_camera.T_camera_from_world is None:
        raise RuntimeError(f"Reference camera {reference_camera.camera_id} has no valid extrinsic.")
    # extrinsics.json stores T_camera_from_world, while the reference frame is the first static camera.
    T_world_from_reference = _invert_transform(reference_camera.T_camera_from_world)
    for camera in static_cameras:
        if camera.T_camera_from_world is None:
            raise RuntimeError(f"Static camera {camera.camera_id} has no valid extrinsic.")
        camera.T_camera_from_reference = _compose(camera.T_camera_from_world, T_world_from_reference)
        camera.T_reference_from_camera = _invert_transform(camera.T_camera_from_reference)

    common_frames = set(static_frame_paths[static_cameras[0].camera_id].keys())
    for camera in static_cameras[1:]:
        common_frames &= set(static_frame_paths[camera.camera_id].keys())
    common_reference_frames = sorted(common_frames, key=_frame_sort_key)[:REFERENCE_FRAME_LIMIT]
    if len(common_reference_frames) < MIN_REFERENCE_COMMON_FRAMES:
        raise RuntimeError(
            f"Need at least {MIN_REFERENCE_COMMON_FRAMES} common static-camera frames to build reference, "
            f"found {len(common_reference_frames)}."
        )

    detector = _create_aruco_detector(APRILTAG_FAMILY)
    reference_tag_corners, fused_tags, detected_reference_ids = _build_reference_from_static_frames(
        static_cameras=static_cameras,
        static_frame_paths=static_frame_paths,
        reference_frames=common_reference_frames,
        detector=detector,
    )
    if len(reference_tag_corners) < MIN_REFERENCE_TAGS:
        raise RuntimeError(
            f"Fixed reference tag count is too small: need at least {MIN_REFERENCE_TAGS}, got {len(reference_tag_corners)}."
        )

    print(f"[ego_apriltag_calib] episode={episode_root}")
    print(f"[ego_apriltag_calib] reference_camera={reference_camera.camera_id}")
    print(f"[ego_apriltag_calib] reference_frames={','.join(common_reference_frames)}")
    print(
        "[ego_apriltag_calib] reference_detected_tag_ids="
        + (",".join(str(tag_id) for tag_id in sorted(detected_reference_ids)) if detected_reference_ids else "(none)")
    )
    print(f"[ego_apriltag_calib] reference_detected_tag_count={len(detected_reference_ids)}")
    kept_reference_ids = sorted(fused_tags)
    print(
        "[ego_apriltag_calib] reference_kept_tag_ids="
        + (",".join(str(tag_id) for tag_id in kept_reference_ids) if kept_reference_ids else "(none)")
    )
    for tag_id in kept_reference_ids:
        fused = fused_tags[tag_id]
        print(
            f"[ego_apriltag_calib] reference_tag id={tag_id} "
            f"detections={fused.detection_count} candidates={fused.candidate_count} inliers={fused.inlier_count}"
        )
    print(f"[ego_apriltag_calib] reference_tag_count={len(reference_tag_corners)}")

    ego_frame_items = sorted(ego_frames.items(), key=lambda item: _frame_sort_key(item[0]))
    estimates: list[FrameEstimate] = []
    ego_detected_any = 0
    ego_no_detection = 0
    ego_direct_success = 0
    distinct_ego_tag_ids: set[int] = set()

    for frame_index, image_path in ego_frame_items:
        image = _read_image(image_path)
        detections, reason = _detect_apriltag_markers(image, detector)
        detected_ids = sorted({tag_id for tag_id, _ in detections})
        distinct_ego_tag_ids.update(detected_ids)
        estimate = FrameEstimate(frame_index=frame_index, detected_tag_ids=detected_ids)

        if not detections:
            estimate.status_initial = "no_detection" if reason == "tags_not_found" else reason
            estimate.status_final = estimate.status_initial
            ego_no_detection += 1
            estimates.append(estimate)
            continue

        ego_detected_any += 1
        pose, rmse, used_tag_ids, status = _solve_ego_pose(
            detections=detections,
            K=ego_K,
            dist=ego_dist,
            reference_tag_corners=reference_tag_corners,
        )
        estimate.used_tag_ids = used_tag_ids
        estimate.rmse = rmse
        estimate.pose_raw = pose
        estimate.status_initial = status
        estimate.status_final = status
        if pose is not None:
            ego_direct_success += 1
        estimates.append(estimate)

    post_stats = _apply_temporal_postprocess(estimates)
    result = _serialize_pose_dict(estimates)

    output_path = episode_root / f"ego_extrinsics_cam{ego_id}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[ego_apriltag_calib] ego_total_frames={len(estimates)}")
    print(f"[ego_apriltag_calib] ego_detected_any_tag_frames={ego_detected_any}")
    print(f"[ego_apriltag_calib] ego_no_detection_frames={ego_no_detection}")
    print(
        "[ego_apriltag_calib] ego_detected_tag_ids="
        + (",".join(str(tag_id) for tag_id in sorted(distinct_ego_tag_ids)) if distinct_ego_tag_ids else "(none)")
    )
    print(f"[ego_apriltag_calib] ego_detected_tag_count={len(distinct_ego_tag_ids)}")
    print(f"[ego_apriltag_calib] ego_direct_pnp_success_frames={ego_direct_success}")
    print(f"[ego_apriltag_calib] ego_outlier_replaced_frames={post_stats['outlier_replaced']}")
    print(f"[ego_apriltag_calib] ego_interpolated_gap_frames={post_stats['interpolated_gap']}")
    print(f"[ego_apriltag_calib] ego_long_gap_interpolated_frames={post_stats['long_gap_interpolated']}")
    print(f"[ego_apriltag_calib] ego_edge_filled_frames={post_stats['edge_filled']}")
    print(f"[ego_apriltag_calib] output_json={output_path}")

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate ego camera extrinsics from a fixed AprilTag reference")
    parser.add_argument("--path", type=Path, required=True, help="Episode directory path")
    parser.add_argument("--ego-idx", type=str, default="8", help="Ego camera index, default: 8")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        calibrate_ego_apriltag(path=args.path, ego_idx=args.ego_idx)
    except Exception as exc:
        print(f"[ego_apriltag_calib] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
