from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .chessboard import PnPResult
from .se3 import make_transform, rodrigues_to_matrix


@dataclass
class AprilTagLayout:
    tag_corners_m: dict[int, np.ndarray]


_APRILTAG_DICT: dict[str, str] = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}


def _rpy_deg_to_matrix(rpy_deg: list[float]) -> np.ndarray:
    roll, pitch, yaw = [math.radians(float(x)) for x in rpy_deg]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _build_corners_from_pose(translation_m: list[float], rpy_deg: list[float], size_m: float) -> np.ndarray:
    half = 0.5 * float(size_m)
    corners_tag = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )
    R = _rpy_deg_to_matrix(rpy_deg)
    t = np.asarray(translation_m, dtype=np.float64).reshape(1, 3)
    return (corners_tag @ R.T + t).astype(np.float32)


def build_tag_local_corners(size_m: float) -> np.ndarray:
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


def load_apriltag_layout(layout_path: Path, default_tag_size_m: float) -> AprilTagLayout:
    if not layout_path.exists():
        raise FileNotFoundError(f"AprilTag layout not found: {layout_path}")

    with layout_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    tags = data.get("tags", [])
    if not isinstance(tags, list):
        raise ValueError("AprilTag layout must contain `tags` list")

    tag_corners_m: dict[int, np.ndarray] = {}
    for item in tags:
        if not isinstance(item, dict):
            continue
        if "id" not in item:
            continue

        tag_id = int(item["id"])
        if "corners_m" in item:
            corners = np.asarray(item["corners_m"], dtype=np.float32).reshape(4, 3)
            tag_corners_m[tag_id] = corners
            continue

        pose = item.get("pose", {})
        if not isinstance(pose, dict):
            continue
        translation_m = pose.get("translation_m")
        rpy_deg = pose.get("rpy_deg")
        if translation_m is None or rpy_deg is None:
            continue
        size_m = float(item.get("size_m", default_tag_size_m))
        tag_corners_m[tag_id] = _build_corners_from_pose(translation_m, rpy_deg, size_m)

    if not tag_corners_m:
        raise ValueError("No valid AprilTag definitions were parsed from layout file")
    return AprilTagLayout(tag_corners_m=tag_corners_m)


def _create_aruco_detector(tag_family: str):
    if not hasattr(cv2, "aruco"):
        return None, "opencv_has_no_aruco"

    family_key = str(tag_family).lower()
    dict_name = _APRILTAG_DICT.get(family_key)
    if dict_name is None or not hasattr(cv2.aruco, dict_name):
        return None, f"unsupported_apriltag_family:{tag_family}"
    dict_id = getattr(cv2.aruco, dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return detector, ""
    return (dictionary, params), ""


def _detect_apriltags(gray: np.ndarray, detector):
    if detector is None:
        return [], None

    if isinstance(detector, tuple):
        dictionary, params = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        return corners, ids

    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def detect_apriltag_markers(image: np.ndarray, tag_family: str) -> tuple[list[tuple[int, np.ndarray]], str]:
    if image is None:
        return [], "image_missing"
    detector, detector_err = _create_aruco_detector(tag_family)
    if detector is None:
        return [], detector_err

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_list, ids = _detect_apriltags(gray, detector)
    if ids is None or len(ids) == 0:
        return [], "tags_not_found"

    out: list[tuple[int, np.ndarray]] = []
    for corner, tag_id_arr in zip(corners_list, ids):
        tag_id = int(tag_id_arr[0])
        img_xy = np.asarray(corner, dtype=np.float32).reshape(4, 2)
        out.append((tag_id, img_xy))
    return out, ""


def solve_single_tag_pnp(
    corners_px: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    tag_size_m: float,
    reproj_error_px: float,
    pnp_iterations: int,
    min_inliers: int,
) -> PnPResult:
    object_pts = build_tag_local_corners(tag_size_m)
    image_pts = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)

    # Single tag only has 4 points; solvePnPRansac cannot reject outliers with so few points.
    # Use deterministic solvePnP instead.
    # NOTE: SOLVEPNP_IPPE_SQUARE is designed for a C++ overload that returns two solutions via
    # OutputArrayOfArrays.  The Python cv2.solvePnP binding only accepts a single rvec/tvec
    # output and silently returns a near-zero tvec, causing projectPoints to produce thousands
    # of pixels of reprojection error.  Use SOLVEPNP_IPPE instead: it is the general planar
    # solver, works correctly with the single-solution Python binding, and is equally appropriate
    # for a square tag (square is just a special case of a planar target).
    ok, rvec, tvec = cv2.solvePnP(
        object_pts,
        image_pts,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        return PnPResult(False, None, 0, float("inf"), 4, reason="pnp_failed")

    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    reproj = np.linalg.norm(proj.reshape(-1, 2) - image_pts.reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(reproj * reproj)))

    if rmse > float(reproj_error_px) * 2:
        return PnPResult(False, None, 4, rmse, 4, reason="high_reproj_error")

    R = rodrigues_to_matrix(rvec)
    T_c_tag = make_transform(R, tvec.reshape(3))
    return PnPResult(True, T_c_tag, 4, rmse, 4, reason="")


def solve_camera_pose_from_tag_map(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    tag_family: str,
    world_tag_corners: dict[int, np.ndarray],
    reproj_error_px: float,
    pnp_iterations: int,
    min_inliers: int,
    min_tags: int,
) -> PnPResult:
    detections, reason = detect_apriltag_markers(image, tag_family)
    if not detections:
        return PnPResult(False, None, 0, float("inf"), 0, reason=reason)

    object_pts_list: list[np.ndarray] = []
    image_pts_list: list[np.ndarray] = []
    used_tags = 0
    for tag_id, corners_px in detections:
        obj = world_tag_corners.get(tag_id)
        if obj is None:
            continue
        object_pts_list.append(np.asarray(obj, dtype=np.float32).reshape(4, 3))
        image_pts_list.append(np.asarray(corners_px, dtype=np.float32).reshape(4, 2))
        used_tags += 1

    if used_tags < int(min_tags):
        return PnPResult(False, None, 0, float("inf"), used_tags * 4, reason="few_tags")
    if not object_pts_list:
        return PnPResult(False, None, 0, float("inf"), 0, reason="tags_not_in_map")

    object_pts = np.concatenate(object_pts_list, axis=0).astype(np.float32)
    image_pts = np.concatenate(image_pts_list, axis=0).astype(np.float32)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_pts,
        image_pts,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=float(reproj_error_px),
        iterationsCount=int(pnp_iterations),
        confidence=0.999,
    )
    if not ok:
        return PnPResult(False, None, 0, float("inf"), len(image_pts), reason="pnp_failed")

    inlier_count = int(len(inliers)) if inliers is not None else 0
    if inlier_count < int(min_inliers):
        return PnPResult(False, None, inlier_count, float("inf"), len(image_pts), reason="few_inliers")

    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    reproj = np.linalg.norm(proj.reshape(-1, 2) - image_pts.reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(reproj * reproj)))

    R = rodrigues_to_matrix(rvec)
    T_c_w = make_transform(R, tvec.reshape(3))
    return PnPResult(True, T_c_w, inlier_count, rmse, len(image_pts), reason="")


def solve_apriltag_pnp(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    layout: AprilTagLayout,
    tag_family: str,
    reproj_error_px: float,
    pnp_iterations: int,
    min_inliers: int,
    min_tags: int,
) -> PnPResult:
    if image is None:
        return PnPResult(False, None, 0, float("inf"), 0, reason="image_missing")

    detector, detector_err = _create_aruco_detector(tag_family)
    if detector is None:
        return PnPResult(False, None, 0, float("inf"), 0, reason=detector_err)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_list, ids = _detect_apriltags(gray, detector)
    if ids is None or len(ids) == 0:
        return PnPResult(False, None, 0, float("inf"), 0, reason="tags_not_found")

    object_pts_list: list[np.ndarray] = []
    image_pts_list: list[np.ndarray] = []
    used_tags = 0
    for corner, tag_id_arr in zip(corners_list, ids):
        tag_id = int(tag_id_arr[0])
        tag_obj = layout.tag_corners_m.get(tag_id)
        if tag_obj is None:
            continue
        img_xy = np.asarray(corner, dtype=np.float32).reshape(4, 2)
        object_pts_list.append(tag_obj)
        image_pts_list.append(img_xy)
        used_tags += 1

    if used_tags < int(min_tags):
        return PnPResult(False, None, 0, float("inf"), used_tags * 4, reason="few_tags")
    if not object_pts_list:
        return PnPResult(False, None, 0, float("inf"), 0, reason="tags_not_in_layout")

    object_pts = np.concatenate(object_pts_list, axis=0).astype(np.float32)
    image_pts = np.concatenate(image_pts_list, axis=0).astype(np.float32)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_pts,
        image_pts,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=float(reproj_error_px),
        iterationsCount=int(pnp_iterations),
        confidence=0.999,
    )
    if not ok:
        return PnPResult(False, None, 0, float("inf"), len(image_pts), reason="pnp_failed")

    inlier_count = int(len(inliers)) if inliers is not None else 0
    if inlier_count < int(min_inliers):
        return PnPResult(False, None, inlier_count, float("inf"), len(image_pts), reason="few_inliers")

    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    reproj = np.linalg.norm(proj.reshape(-1, 2) - image_pts.reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(reproj * reproj)))

    R = rodrigues_to_matrix(rvec)
    T_c_b = make_transform(R, tvec.reshape(3))
    return PnPResult(True, T_c_b, inlier_count, rmse, len(image_pts), reason="")
