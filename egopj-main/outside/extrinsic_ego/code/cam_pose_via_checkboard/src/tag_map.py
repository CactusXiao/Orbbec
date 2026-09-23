from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .apriltag import build_tag_local_corners, detect_apriltag_markers, solve_single_tag_pnp
from .calib_io import CameraCalibration
from .fusion import BoardCandidate, fuse_board_pose
from .se3 import compose


@dataclass
class TagMap:
    family: str
    tag_size_m: float
    world_tag_corners: dict[int, np.ndarray]
    observation_counts: dict[int, int]
    reprojection_rmse_px: dict[int, float]
    calibration_signature: str = ""


def calibration_signature(calibrations: dict[str, CameraCalibration], camera_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for camera_id in sorted(camera_ids):
        cam = calibrations[camera_id]
        digest.update(camera_id.encode("utf-8"))
        digest.update(cam.camera_model.encode("utf-8"))
        for array in (cam.K, cam.dist, cam.T_w_c):
            digest.update(np.asarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def save_tag_map(path: Path, tag_map: TagMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "pose_convention": "T_world_from_camera",
        "family": tag_map.family,
        "tag_size_m": tag_map.tag_size_m,
        "fixed_calibration_signature": tag_map.calibration_signature,
        "tags": {
            str(tag_id): {
                "corners_world_m": corners.tolist(),
                "observation_count": tag_map.observation_counts.get(tag_id, 0),
                "reprojection_rmse_px": tag_map.reprojection_rmse_px.get(tag_id),
            }
            for tag_id, corners in sorted(tag_map.world_tag_corners.items())
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_tag_map(path: Path, expected_family: str, expected_tag_size_m: float) -> TagMap:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    family = str(payload.get("family", ""))
    size = float(payload.get("tag_size_m", 0.0))
    if family != expected_family:
        raise ValueError(f"Tag map family {family!r} does not match configured {expected_family!r}")
    if not np.isclose(size, expected_tag_size_m, rtol=0.0, atol=1e-9):
        raise ValueError(f"Tag map size {size} does not match configured {expected_tag_size_m}")
    corners: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    rmses: dict[int, float] = {}
    for key, item in payload.get("tags", {}).items():
        tag_id = int(key)
        corners[tag_id] = np.asarray(item["corners_world_m"], dtype=np.float32).reshape(4, 3)
        counts[tag_id] = int(item.get("observation_count", 0))
        if item.get("reprojection_rmse_px") is not None:
            rmses[tag_id] = float(item["reprojection_rmse_px"])
    if not corners:
        raise ValueError(f"Tag map contains no tags: {path}")
    return TagMap(family, size, corners, counts, rmses, str(payload.get("fixed_calibration_signature", "")))


def build_tag_map(
    dataset_root: Path,
    fixed_camera_ids: Iterable[str],
    calibrations: dict[str, CameraCalibration],
    frame_indices: Iterable[str],
    family: str,
    tag_size_m: float,
    reproj_error_px: float,
    pnp_iterations: int,
    min_inliers: int,
    trans_thresh_m: float,
    rot_thresh_deg: float,
) -> TagMap:
    fixed_camera_ids = tuple(fixed_camera_ids)
    candidates_by_tag: dict[int, list[BoardCandidate]] = {}
    rmse_by_tag: dict[int, list[float]] = {}
    for frame in frame_indices:
        for cam_id in fixed_camera_ids:
            cam = calibrations.get(cam_id)
            if cam is None:
                continue
            image = None
            for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
                image_path = dataset_root / cam_id / "RGB" / f"{frame}{suffix}"
                if image_path.exists():
                    image = cv2.imread(str(image_path))
                    break
            detections, _ = detect_apriltag_markers(image, family)
            for tag_id, corners_px in detections:
                result = solve_single_tag_pnp(
                    corners_px,
                    cam.K,
                    cam.dist,
                    tag_size_m,
                    reproj_error_px,
                    pnp_iterations,
                    min_inliers,
                    camera_model=cam.camera_model,
                )
                if not result.success or result.T_c_b is None:
                    continue
                candidates_by_tag.setdefault(tag_id, []).append(
                    BoardCandidate(f"{cam_id}:{frame}", compose(cam.T_w_c, result.T_c_b), result.reproj_rmse, result.inliers)
                )
                rmse_by_tag.setdefault(tag_id, []).append(result.reproj_rmse)

    local_corners = build_tag_local_corners(tag_size_m)
    world_corners: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    rmses: dict[int, float] = {}
    for tag_id, candidates in candidates_by_tag.items():
        fused = fuse_board_pose(candidates, trans_thresh_m, rot_thresh_deg)
        if not fused.success or fused.T_w_b is None or len(fused.inlier_camera_ids) < 2:
            continue
        R = fused.T_w_b[:3, :3]
        t = fused.T_w_b[:3, 3]
        world_corners[tag_id] = (local_corners @ R.T + t.reshape(1, 3)).astype(np.float32)
        counts[tag_id] = len(fused.inlier_camera_ids)
        rmses[tag_id] = float(np.median(rmse_by_tag[tag_id]))
    if not world_corners:
        raise RuntimeError("Could not build a tag map from the fixed-camera observations")
    return TagMap(
        family,
        tag_size_m,
        world_corners,
        counts,
        rmses,
        calibration_signature(calibrations, fixed_camera_ids),
    )
