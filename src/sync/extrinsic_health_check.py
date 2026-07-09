#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime dependency check
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None


APRILTAG_DICT_NAMES = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = np.asarray(R_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    cos_theta = max(-1.0, min(1.0, (float(np.trace(R_rel)) - 1.0) / 2.0))
    return math.degrees(math.acos(cos_theta))


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
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


def average_poses(poses: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    t = np.sum(np.array([p[:3, 3] for p in poses]) * weights[:, None], axis=0)
    q0 = matrix_to_quaternion(poses[0][:3, :3])
    q_sum = np.zeros(4, dtype=np.float64)
    for pose, weight in zip(poses, weights):
        q = matrix_to_quaternion(pose[:3, :3])
        if float(np.dot(q0, q)) < 0.0:
            q = -q
        q_sum += float(weight) * q
    q_norm = np.linalg.norm(q_sum)
    q = q_sum / q_norm if q_norm > 1e-12 else q0
    return make_transform(quaternion_to_matrix(q), t)


def local_tag_corners(tag_size_m: float) -> np.ndarray:
    h = 0.5 * float(tag_size_m)
    return np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]], dtype=np.float32)


def create_detector(tag_family: str):
    if cv2 is None:
        raise RuntimeError(f"cv2 import failed: {CV2_IMPORT_ERROR}")
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV has no aruco module. Install opencv-contrib-python.")
    dict_name = APRILTAG_DICT_NAMES.get(str(tag_family).lower())
    if dict_name is None or not hasattr(cv2.aruco, dict_name):
        raise RuntimeError(f"unsupported AprilTag family: {tag_family}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return dictionary, params


def detect_tags(image: np.ndarray, detector) -> list[tuple[int, np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if isinstance(detector, tuple):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, detector[0], parameters=detector[1])
    else:
        corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    tag_ids = np.asarray(ids).reshape(-1)
    return [(int(tag_id), np.asarray(corner, dtype=np.float32).reshape(4, 2)) for corner, tag_id in zip(corners, tag_ids)]


def load_camera_data(snapshot_dir: Path, manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    camera_params = json.loads((snapshot_dir / manifest.get("camera_params_json", "camera_params.json")).read_text(encoding="utf-8"))
    extrinsics = json.loads((snapshot_dir / manifest.get("extrinsics_json", "extrinsics.json")).read_text(encoding="utf-8"))
    cameras: dict[str, dict[str, Any]] = {}
    world_from_camera: dict[str, np.ndarray] = {}
    for cam_id, entry in camera_params.items():
        if not isinstance(entry, dict) or cam_id == "viewer":
            continue
        rgb = entry.get("RGB")
        extr = extrinsics.get(cam_id)
        if not isinstance(rgb, dict) or not isinstance(extr, dict):
            continue
        intr = rgb.get("intrinsic", {})
        dist = rgb.get("distortion", {})
        K = np.array([[float(intr["fx"]), 0.0, float(intr["cx"])], [0.0, float(intr["fy"]), float(intr["cy"])], [0.0, 0.0, 1.0]], dtype=np.float64)
        d = np.array(
            [
                float(dist.get("k1", 0.0)),
                float(dist.get("k2", 0.0)),
                float(dist.get("p1", 0.0)),
                float(dist.get("p2", 0.0)),
                float(dist.get("k3", 0.0)),
                float(dist.get("k4", 0.0)),
                float(dist.get("k5", 0.0)),
                float(dist.get("k6", 0.0)),
            ],
            dtype=np.float64,
        )
        R = np.asarray(extr["rotation"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(extr["translation"], dtype=np.float64).reshape(3)
        T_camera_from_world = make_transform(R, t)
        cameras[cam_id] = {"K": K, "dist": d}
        world_from_camera[cam_id] = invert_transform(T_camera_from_world)
    return cameras, world_from_camera


def cfg_int(cfg: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(cfg.get(key, default))
    except Exception:
        value = default
    return max(lo, min(hi, value))


def cfg_float(cfg: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(cfg.get(key, default))
    except Exception:
        value = default
    return max(lo, min(hi, value))


def load_aligned_depth(snapshot_dir: Path, cam: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    depth_rel = cam.get("depth") or cam.get("depth_image")
    if not depth_rel:
        return None
    depth = cv2.imread(str(snapshot_dir / str(depth_rel)), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        return None
    if depth.dtype != np.uint16:
        if not np.issubdtype(depth.dtype, np.integer):
            return None
        depth = depth.astype(np.uint16)
    try:
        scale_mm = float(cam.get("depth_value_scale_mm", 1.0))
    except Exception:
        scale_mm = 1.0
    if not (scale_mm > 0.0):
        return None
    return depth, scale_mm * 0.001


def depth_window_median_m(depth16: np.ndarray, u: float, v: float, radius_px: int, scale_m: float, z_min_m: float, z_max_m: float) -> float | None:
    x = int(round(float(u)))
    y = int(round(float(v)))
    if x < 0 or y < 0 or x >= depth16.shape[1] or y >= depth16.shape[0]:
        return None
    r = max(0, int(radius_px))
    x0 = max(0, x - r)
    x1 = min(depth16.shape[1], x + r + 1)
    y0 = max(0, y - r)
    y1 = min(depth16.shape[0], y + r + 1)
    values = depth16[y0:y1, x0:x1].reshape(-1).astype(np.float64)
    values = values[values > 0.0] * float(scale_m)
    values = values[np.isfinite(values)]
    values = values[(values >= z_min_m) & (values <= z_max_m)]
    if values.size == 0:
        return None
    return float(np.median(values))


def sample_tag_depth_points(
    corners_px: np.ndarray,
    depth16: np.ndarray,
    depth_scale_m: float,
    K: np.ndarray,
    dist: np.ndarray,
    tag_size_m: float,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    grid_n = cfg_int(cfg, "depthSampleGridSize", 11, 3, 31)
    inset = cfg_float(cfg, "depthSampleInsetFrac", 0.18, 0.0, 0.45)
    radius_px = cfg_int(cfg, "depthSampleWindowRadiusPx", 2, 0, 8)
    z_min_m = cfg_float(cfg, "depthMinM", 0.15, 0.01, 20.0)
    z_max_m = cfg_float(cfg, "depthMaxM", 5.0, z_min_m + 0.01, 20.0)

    src_uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_uv, corners_px.astype(np.float32))
    pixels: list[list[float]] = []
    local_xy: list[list[float]] = []
    zs: list[float] = []
    for vv in np.linspace(inset, 1.0 - inset, grid_n):
        for uu in np.linspace(inset, 1.0 - inset, grid_n):
            hp = H @ np.array([uu, vv, 1.0], dtype=np.float64)
            if abs(float(hp[2])) < 1e-9:
                continue
            u = float(hp[0] / hp[2])
            v = float(hp[1] / hp[2])
            z_m = depth_window_median_m(depth16, u, v, radius_px, depth_scale_m, z_min_m, z_max_m)
            if z_m is None:
                continue
            pixels.append([u, v])
            local_xy.append([(float(uu) - 0.5) * tag_size_m, (float(vv) - 0.5) * tag_size_m])
            zs.append(z_m)

    min_samples = cfg_int(cfg, "depthMinValidSamples", 24, 3, grid_n * grid_n)
    if len(zs) < min_samples:
        return None

    pixel_arr = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(pixel_arr, K, dist).reshape(-1, 2)
    z_arr = np.asarray(zs, dtype=np.float64)
    points_camera = np.column_stack([rays[:, 0] * z_arr, rays[:, 1] * z_arr, z_arr])
    return np.asarray(local_xy, dtype=np.float64), points_camera, rays


def fit_plane_inliers(points: np.ndarray, cfg: dict[str, Any]) -> np.ndarray | None:
    n = int(points.shape[0])
    min_inliers = cfg_int(cfg, "depthMinPlaneInliers", 18, 3, n)
    min_ratio = cfg_float(cfg, "depthMinPlaneInlierRatio", 0.45, 0.05, 1.0)
    thresh_m = cfg_float(cfg, "depthPlaneInlierThreshM", 0.008, 0.001, 0.05)
    iters = cfg_int(cfg, "depthPlaneRansacIters", 96, 1, 1000)
    if n < min_inliers:
        return None

    rng = np.random.default_rng(20240719)
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float] = (-1, float("inf"))
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        distances = np.abs((points - p0) @ normal)
        mask = distances <= thresh_m
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        med = float(np.median(distances[mask]))
        score = (count, -med)
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None or int(np.count_nonzero(best_mask)) < min_inliers:
        return None
    if float(np.count_nonzero(best_mask)) / float(n) < min_ratio:
        return None

    for _ in range(2):
        inlier_points = points[best_mask]
        center = np.mean(inlier_points, axis=0)
        _, _, vt = np.linalg.svd(inlier_points - center, full_matrices=False)
        normal = vt[-1]
        distances = np.abs((points - center) @ normal)
        inlier_distances = distances[best_mask]
        med = float(np.median(inlier_distances))
        mad = float(np.median(np.abs(inlier_distances - med)))
        robust_thresh = max(thresh_m, med + 3.0 * 1.4826 * mad)
        best_mask = distances <= robust_thresh
        if int(np.count_nonzero(best_mask)) < min_inliers:
            return None
    return best_mask


def fit_plane_from_inliers(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if points.shape[0] < 3:
        return None
    center = np.mean(points, axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    normal = vt[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    return center, normal / norm


def intersect_rays_with_plane(rays_xy: np.ndarray, plane_center: np.ndarray, plane_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = np.column_stack([rays_xy[:, 0], rays_xy[:, 1], np.ones(rays_xy.shape[0], dtype=np.float64)])
    denom = directions @ plane_normal
    numer = float(plane_center @ plane_normal)
    valid = np.abs(denom) > 1e-9
    lambdas = np.full(rays_xy.shape[0], np.nan, dtype=np.float64)
    lambdas[valid] = numer / denom[valid]
    valid &= np.isfinite(lambdas) & (lambdas > 0.0)
    points = directions * lambdas[:, None]
    return points, valid


def rigid_transform_from_points(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if source.shape[0] < 3 or target.shape[0] != source.shape[0]:
        return None
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    H = source_zero.T @ target_zero
    try:
        U, _, vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    R = vt.T @ U.T
    if float(np.linalg.det(R)) < 0.0:
        vt[-1, :] *= -1.0
        R = vt.T @ U.T
    t = target_center - R @ source_center
    return R, t


def fit_depth_tag_pose(local_xy: np.ndarray, points_camera: np.ndarray, rays_xy: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    plane_mask = fit_plane_inliers(points_camera, cfg)
    if plane_mask is None:
        return None

    min_inliers = cfg_int(cfg, "depthMinPoseInliers", 18, 3, int(points_camera.shape[0]))
    thresh_m = cfg_float(cfg, "depthPoseInlierThreshM", 0.012, 0.001, 0.08)

    plane = fit_plane_from_inliers(points_camera[plane_mask])
    if plane is None:
        return None
    plane_center, plane_normal = plane
    target_points, ray_valid = intersect_rays_with_plane(rays_xy, plane_center, plane_normal)

    source_points = np.column_stack([local_xy[:, 0], local_xy[:, 1], np.zeros(local_xy.shape[0], dtype=np.float64)])
    mask = plane_mask & ray_valid
    residuals = None
    fitted = None
    for _ in range(3):
        if int(np.count_nonzero(mask)) < min_inliers:
            return None
        fitted = rigid_transform_from_points(source_points[mask], target_points[mask])
        if fitted is None:
            return None
        R, t = fitted
        pred_all = (R @ source_points.T).T + t
        residuals = np.linalg.norm(pred_all - target_points, axis=1)
        inlier_residuals = residuals[mask]
        med = float(np.median(inlier_residuals))
        mad = float(np.median(np.abs(inlier_residuals - med)))
        robust_thresh = max(thresh_m, med + 3.0 * 1.4826 * mad)
        mask = plane_mask & ray_valid & (residuals <= robust_thresh)

    if fitted is None or residuals is None or int(np.count_nonzero(mask)) < min_inliers:
        return None
    fitted = rigid_transform_from_points(source_points[mask], target_points[mask])
    if fitted is None:
        return None
    R, t = fitted
    pred_all = (R @ source_points.T).T + t
    residuals = np.linalg.norm(pred_all - target_points, axis=1)
    T = make_transform(R, t)
    fit_rmse_m = float(np.sqrt(np.mean(residuals[mask] * residuals[mask]))) if np.count_nonzero(mask) else float("inf")
    return T, fit_rmse_m


def estimate_single_tag_from_depth(
    corners_px: np.ndarray,
    depth16: np.ndarray,
    depth_scale_m: float,
    K: np.ndarray,
    dist: np.ndarray,
    object_pts: np.ndarray,
    cfg: dict[str, Any],
):
    sampled = sample_tag_depth_points(corners_px, depth16, depth_scale_m, K, dist, float(cfg["tagSizeM"]), cfg)
    if sampled is None:
        return None
    local_xy, points_camera, rays_xy = sampled
    fitted = fit_depth_tag_pose(local_xy, points_camera, rays_xy, cfg)
    if fitted is None:
        return None
    T_camera_from_tag, _fit_rmse_m = fitted
    rvec, _ = cv2.Rodrigues(T_camera_from_tag[:3, :3])
    tvec = T_camera_from_tag[:3, 3].reshape(3, 1)
    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - corners_px.reshape(-1, 2), axis=1)
    reproj_rmse = float(np.sqrt(np.mean(err * err)))
    if reproj_rmse > float(cfg["singleTagReprojLimitPx"]):
        return None
    return T_camera_from_tag, reproj_rmse


def fuse_tag_observations(observations: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any] | None:
    if len(observations) < int(cfg["minSharedCamerasPerTag"]):
        return None
    best: list[int] = []
    for i, hypo in enumerate(observations):
        current = []
        for j, obs in enumerate(observations):
            trans = float(np.linalg.norm(obs["T_world_from_tag"][:3, 3] - hypo["T_world_from_tag"][:3, 3]))
            rot = rotation_angle_deg(obs["T_world_from_tag"][:3, :3], hypo["T_world_from_tag"][:3, :3])
            if trans <= float(cfg["fusionTransThreshM"]) and rot <= float(cfg["fusionRotThreshDeg"]):
                current.append(j)
        if len(current) > len(best):
            best = current
        elif len(current) == len(best) and current:
            if sum(observations[x]["rmse"] for x in current) < sum(observations[x]["rmse"] for x in best):
                best = current
    if len(best) < int(cfg["minTagInlierObservations"]):
        return None
    inliers = [observations[i] for i in best]
    weights = np.array([1.0 / max(float(obs["rmse"]), 1e-6) for obs in inliers], dtype=np.float64)
    fused = average_poses([obs["T_world_from_tag"] for obs in inliers], weights)
    return {"pose": fused, "inlier_cameras": sorted(obs["camera_id"] for obs in inliers), "candidate_count": len(observations)}


def project_residual_px(T_camera_from_world: np.ndarray, T_world_from_tag: np.ndarray, K: np.ndarray, dist: np.ndarray, object_pts: np.ndarray, image_pts: np.ndarray) -> float:
    T_camera_from_tag = T_camera_from_world @ T_world_from_tag
    rvec, _ = cv2.Rodrigues(T_camera_from_tag[:3, :3])
    tvec = T_camera_from_tag[:3, 3].reshape(3, 1)
    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - image_pts.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(err * err)))


def median(values: list[float]) -> float:
    if not values:
        return float("inf")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def status_counts(statuses: list[str]) -> dict[str, int]:
    return {key: sum(1 for status in statuses if status == key) for key in ("pass", "warn", "fail", "inconclusive")}


def evaluate_sample(snapshot_dir: Path, sample: dict[str, Any], cameras: dict[str, dict[str, Any]], world_from_camera: dict[str, np.ndarray], detector, cfg: dict[str, Any]) -> dict[str, Any]:
    object_pts = local_tag_corners(float(cfg["tagSizeM"]))
    observations_by_tag: dict[int, list[dict[str, Any]]] = {}
    detected_by_camera: dict[str, list[int]] = {}
    expected_camera_ids: list[str] = []

    for cam in sample.get("cameras", []):
        cam_id = str(cam.get("id", ""))
        if cam_id not in cameras or cam_id not in world_from_camera:
            continue
        expected_camera_ids.append(cam_id)
        image = cv2.imread(str(snapshot_dir / cam.get("image", "")), cv2.IMREAD_COLOR)
        if image is None:
            continue
        detections = detect_tags(image, detector)
        detected_by_camera[cam_id] = sorted({tag_id for tag_id, _ in detections})
        depth_loaded = load_aligned_depth(snapshot_dir, cam)
        if depth_loaded is None:
            continue
        depth16, depth_scale_m = depth_loaded
        if depth16.shape[:2] != image.shape[:2]:
            continue
        for tag_id, corners_px in detections:
            solved = estimate_single_tag_from_depth(corners_px, depth16, depth_scale_m, cameras[cam_id]["K"], cameras[cam_id]["dist"], object_pts, cfg)
            if solved is None:
                continue
            T_camera_from_tag, rmse = solved
            observations_by_tag.setdefault(tag_id, []).append(
                {
                    "camera_id": cam_id,
                    "corners_px": corners_px,
                    "T_camera_from_tag": T_camera_from_tag,
                    "T_world_from_tag": world_from_camera[cam_id] @ T_camera_from_tag,
                    "rmse": rmse,
                }
            )

    fused_by_tag: dict[int, dict[str, Any]] = {}
    for tag_id, observations in observations_by_tag.items():
        fused = fuse_tag_observations(observations, cfg)
        if fused is not None:
            fused_by_tag[tag_id] = fused

    residuals: dict[str, dict[str, list[float]]] = {}
    for tag_id, fused in fused_by_tag.items():
        for obs in observations_by_tag.get(tag_id, []):
            cam_id = obs["camera_id"]
            T_fused = fused["pose"]
            trans = float(np.linalg.norm(obs["T_world_from_tag"][:3, 3] - T_fused[:3, 3]))
            rot = rotation_angle_deg(obs["T_world_from_tag"][:3, :3], T_fused[:3, :3])
            T_camera_from_world = invert_transform(world_from_camera[cam_id])
            reproj = project_residual_px(T_camera_from_world, T_fused, cameras[cam_id]["K"], cameras[cam_id]["dist"], object_pts, obs["corners_px"])
            r = residuals.setdefault(cam_id, {"trans_m": [], "rot_deg": [], "reproj_px": [], "tags": []})
            r["trans_m"].append(trans)
            r["rot_deg"].append(rot)
            r["reproj_px"].append(reproj)
            r["tags"].append(float(tag_id))

    camera_results: dict[str, Any] = {}
    pass_cameras: list[str] = []
    fail_cameras: list[str] = []
    warn_cameras: list[str] = []
    checked_cameras: list[str] = []
    for cam_id, r in residuals.items():
        tag_count = len(set(int(x) for x in r["tags"]))
        if tag_count < int(cfg["minTagsPerCamera"]):
            continue
        trans = median(r["trans_m"])
        rot = median(r["rot_deg"])
        reproj = median(r["reproj_px"])
        status = "pass"
        if trans > float(cfg["failTransThreshM"]) or rot > float(cfg["failRotThreshDeg"]) or reproj > float(cfg["failReprojThreshPx"]):
            status = "fail"
            fail_cameras.append(cam_id)
        elif trans > float(cfg["warnTransThreshM"]) or rot > float(cfg["warnRotThreshDeg"]) or reproj > float(cfg["warnReprojThreshPx"]):
            status = "warn"
            warn_cameras.append(cam_id)
        else:
            pass_cameras.append(cam_id)
        checked_cameras.append(cam_id)
        camera_results[cam_id] = {
            "status": status,
            "tag_count": tag_count,
            "median_trans_m": trans,
            "median_rot_deg": rot,
            "median_reproj_px": reproj,
            "max_trans_m": max(r["trans_m"]) if r["trans_m"] else 0.0,
            "max_rot_deg": max(r["rot_deg"]) if r["rot_deg"] else 0.0,
            "max_reproj_px": max(r["reproj_px"]) if r["reproj_px"] else 0.0,
        }

    missing_checked_cameras = sorted(set(expected_camera_ids) - set(checked_cameras))
    if fail_cameras:
        status = "fail"
        reason = "camera_residual_exceeded"
    elif bool(cfg.get("requireAllCameras", True)) and missing_checked_cameras:
        status = "inconclusive"
        reason = "missing_checked_cameras"
    elif len(checked_cameras) >= int(cfg["minCheckedCameras"]) and len(fused_by_tag) > 0:
        status = "warn" if warn_cameras else "pass"
        reason = "ok"
    else:
        status = "inconclusive"
        reason = "insufficient_shared_tag_observations"

    camera_statuses = {cam_id: "pass" for cam_id in pass_cameras}
    camera_statuses.update({cam_id: "warn" for cam_id in warn_cameras})
    camera_statuses.update({cam_id: "fail" for cam_id in fail_cameras})
    camera_statuses.update({cam_id: "inconclusive" for cam_id in missing_checked_cameras})
    camera_counts = status_counts(list(camera_statuses.values()))
    camera_counts["total"] = len(camera_statuses)

    return {
        "sample_index": sample.get("index"),
        "status": status,
        "reason": reason,
        "expected_cameras": sorted(expected_camera_ids),
        "checked_cameras": sorted(checked_cameras),
        "missing_checked_cameras": missing_checked_cameras,
        "pass_cameras": sorted(pass_cameras),
        "fail_cameras": sorted(fail_cameras),
        "warn_cameras": sorted(warn_cameras),
        "inconclusive_cameras": missing_checked_cameras,
        "camera_counts": camera_counts,
        "camera_statuses": dict(sorted(camera_statuses.items())),
        "fused_tag_count": len(fused_by_tag),
        "detected_by_camera": detected_by_camera,
        "cameras": camera_results,
    }


def sample_expected_cameras(sample: dict[str, Any]) -> set[str]:
    expected = {str(cam) for cam in sample.get("expected_cameras", [])}
    expected.update(str(cam) for cam in sample.get("checked_cameras", []))
    expected.update(str(cam) for cam in sample.get("missing_checked_cameras", []))
    expected.update(str(cam) for cam in sample.get("cameras", {}).keys())
    return expected


def aggregate_camera_statuses(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    all_cameras = sorted({cam for sample in samples for cam in sample_expected_cameras(sample)})
    by_camera: dict[str, dict[str, Any]] = {}
    for cam_id in all_cameras:
        per_sample: list[str] = []
        for sample in samples:
            expected = sample_expected_cameras(sample)
            if cam_id not in expected:
                continue
            camera_entry = sample.get("cameras", {}).get(cam_id)
            if isinstance(camera_entry, dict) and camera_entry.get("status") in ("pass", "warn", "fail"):
                per_sample.append(str(camera_entry["status"]))
            else:
                per_sample.append("inconclusive")
        counts = status_counts(per_sample)
        pass_like = counts["pass"] + counts["warn"]
        if counts["fail"] >= int(cfg["minFailingSnapshots"]):
            status = "fail"
        elif pass_like >= int(cfg["minPassingSnapshots"]):
            status = "warn" if counts["fail"] > 0 or counts["warn"] > 0 else "pass"
        else:
            status = "inconclusive"
        by_camera[cam_id] = {"status": status, "sample_counts": counts, "samples_seen": len(per_sample)}

    camera_statuses = {cam_id: item["status"] for cam_id, item in by_camera.items()}
    camera_counts = status_counts(list(camera_statuses.values()))
    camera_counts["total"] = len(all_cameras)
    return {
        "camera_statuses": dict(sorted(camera_statuses.items())),
        "camera_sample_counts": by_camera,
        "camera_counts": camera_counts,
        "pass_cameras": [cam for cam in all_cameras if camera_statuses.get(cam) == "pass"],
        "warn_cameras": [cam for cam in all_cameras if camera_statuses.get(cam) == "warn"],
        "fail_cameras": [cam for cam in all_cameras if camera_statuses.get(cam) == "fail"],
        "inconclusive_cameras": [cam for cam in all_cameras if camera_statuses.get(cam) == "inconclusive"],
    }


def summarize(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    sample_counts = status_counts([str(s["status"]) for s in samples])
    camera_summary = aggregate_camera_statuses(samples, cfg)
    pass_like = sample_counts["pass"] + sample_counts["warn"]
    if sample_counts["fail"] >= int(cfg["minFailingSnapshots"]):
        status = "fail"
        reason = "repeated_camera_residual_failures"
    elif pass_like >= int(cfg["minPassingSnapshots"]):
        status = "warn" if sample_counts["warn"] > 0 else "pass"
        reason = "ok"
    else:
        status = "inconclusive"
        reason = "not_enough_passing_snapshots"

    camera_counts = camera_summary["camera_counts"]
    parts = [
        f"status={status}",
        f"sample_pass={sample_counts['pass']}",
        f"sample_warn={sample_counts['warn']}",
        f"sample_fail={sample_counts['fail']}",
        f"sample_inconclusive={sample_counts['inconclusive']}",
        f"camera_total={camera_counts['total']}",
        f"camera_pass={camera_counts['pass']}",
        f"camera_warn={camera_counts['warn']}",
        f"camera_fail={camera_counts['fail']}",
        f"camera_inconclusive={camera_counts['inconclusive']}",
    ]
    fail_cameras = camera_summary["fail_cameras"]
    warn_cameras = camera_summary["warn_cameras"]
    pass_cameras = camera_summary["pass_cameras"]
    inconclusive_cameras = camera_summary["inconclusive_cameras"]
    if pass_cameras:
        parts.append("pass_cameras=" + ",".join(pass_cameras))
    if fail_cameras:
        parts.append("fail_cameras=" + ",".join(fail_cameras))
    if warn_cameras:
        parts.append("warn_cameras=" + ",".join(warn_cameras))
    if inconclusive_cameras:
        parts.append("inconclusive_cameras=" + ",".join(inconclusive_cameras))
    return {
        "status": status,
        "reason": reason,
        "sample_counts": sample_counts,
        "camera_counts": camera_counts,
        "counts": camera_counts,
        "pass_cameras": pass_cameras,
        "fail_cameras": fail_cameras,
        "warn_cameras": warn_cameras,
        "inconclusive_cameras": inconclusive_cameras,
        "missing_cameras": inconclusive_cameras,
        "camera_statuses": camera_summary["camera_statuses"],
        "camera_sample_counts": camera_summary["camera_sample_counts"],
        "summary_line": " ".join(parts),
        "samples": samples,
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def run(snapshot_dir: Path, config_json: Path, result_json: Path) -> int:
    try:
        cfg = json.loads(config_json.read_text(encoding="utf-8"))
        manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
        detector = create_detector(str(cfg.get("tagFamily", "tag36h11")))
        cameras, world_from_camera = load_camera_data(snapshot_dir, manifest)
        samples = [
            evaluate_sample(snapshot_dir, sample, cameras, world_from_camera, detector, cfg)
            for sample in manifest.get("samples", [])
        ]
        result = summarize(samples, cfg)
    except Exception as exc:
        result = {"status": "error", "reason": str(exc), "traceback": traceback.format_exc(), "summary_line": f"status=error reason={exc}"}
        write_result(result_json, result)
        print(result["summary_line"])
        return 1

    write_result(result_json, result)
    print(result["summary_line"])
    if result["status"] in ("pass", "warn"):
        return 0
    if result["status"] == "inconclusive":
        return 2
    if result["status"] == "fail":
        return 3
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check multi-camera extrinsic consistency from AprilTag snapshots.")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args.snapshot_dir.resolve(), args.config_json.resolve(), args.result_json.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
