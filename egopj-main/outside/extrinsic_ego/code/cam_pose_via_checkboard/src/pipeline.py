from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .apriltag import (
    detect_apriltag_markers,
    solve_camera_pose_from_tag_map,
)
from .calib_io import CameraCalibration
from .chessboard import solve_board_pnp
from .config import RuntimeConfig
from .fusion import BoardCandidate, fuse_board_pose
from .se3 import compose, invert_transform
from .tag_map import build_tag_map, calibration_signature, load_tag_map, save_tag_map
from .trajectory import postprocess_trajectory


def _discover_frames(dataset_root: Path, camera_id: str) -> set[str]:
    rgb_dir = dataset_root / camera_id / "RGB"
    if not rgb_dir.exists():
        return set()
    return {
        p.stem
        for p in rgb_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    }


def _collect_frame_indices(dataset_root: Path, cfg: RuntimeConfig) -> List[str]:
    target_frames = _discover_frames(dataset_root, cfg.target_camera_id)
    fixed_sets = [_discover_frames(dataset_root, cid) for cid in cfg.fixed_camera_ids]

    if cfg.frame_policy == "target_primary":
        return sorted(target_frames)

    if not fixed_sets:
        return sorted(target_frames)

    inter = set(target_frames)
    for s in fixed_sets:
        inter &= s
    return sorted(inter)


def _read_image(dataset_root: Path, cam_id: str, frame_index: str):
    rgb_dir = dataset_root / cam_id / "RGB"
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        path = rgb_dir / f"{frame_index}{suffix}"
        if path.exists():
            return cv2.imread(str(path))
    return None


def run_pipeline(dataset_root: Path, cfg: RuntimeConfig, calibrations: Dict[str, CameraCalibration]) -> List[dict]:
    logger = logging.getLogger("pipeline")
    frames = _collect_frame_indices(dataset_root, cfg)
    logger.info("Discovered %d frame indices", len(frames))

    tag_map = None
    if cfg.target_type == "apriltag":
        tag_map_path = Path(cfg.tag_map_filename)
        if not tag_map_path.is_absolute():
            tag_map_path = dataset_root / tag_map_path
        if tag_map_path.exists() and not cfg.rebuild_tag_map:
            tag_map = load_tag_map(tag_map_path, cfg.apriltag_family, cfg.apriltag_default_size_m)
            current_signature = calibration_signature(calibrations, cfg.fixed_camera_ids)
            if tag_map.calibration_signature and tag_map.calibration_signature != current_signature:
                raise ValueError(
                    "Fixed-camera calibration differs from the cached tag map; set rebuild_tag_map=true after verifying the setup"
                )
            logger.info("Loaded persistent tag map with %d tags: %s", len(tag_map.world_tag_corners), tag_map_path)
        else:
            fixed_frames: set[str] = set()
            for cam_id in cfg.fixed_camera_ids:
                fixed_frames |= _discover_frames(dataset_root, cam_id)
            sampled_frames = sorted(fixed_frames)[: max(int(cfg.tag_map_max_frames), 1)]
            tag_map = build_tag_map(
                dataset_root=dataset_root,
                fixed_camera_ids=cfg.fixed_camera_ids,
                calibrations=calibrations,
                frame_indices=sampled_frames,
                family=cfg.apriltag_family,
                tag_size_m=cfg.apriltag_default_size_m,
                reproj_error_px=cfg.pnp_reproj_error_px,
                pnp_iterations=cfg.pnp_iterations,
                min_inliers=cfg.apriltag_min_inliers,
                trans_thresh_m=cfg.fusion_ransac_trans_thresh_m,
                rot_thresh_deg=cfg.fusion_ransac_rot_thresh_deg,
            )
            save_tag_map(tag_map_path, tag_map)
            logger.info("Built and saved persistent tag map with %d tags: %s", len(tag_map.world_tag_corners), tag_map_path)

    def solve_target_pose(image, cam: CameraCalibration):
        return solve_board_pnp(
            image=image,
            K=cam.K,
            dist=cam.dist,
            cols=cfg.board_cols,
            rows=cfg.board_rows,
            square_size_m=cfg.square_size_m,
            reproj_error_px=cfg.pnp_reproj_error_px,
            pnp_iterations=cfg.pnp_iterations,
            min_inliers=cfg.min_inliers,
        )

    rows: List[dict] = []
    for frame_idx, frame in enumerate(frames):
        if frame_idx % 10 == 0:
            logger.info("Processing frame %d / %d (%s)", frame_idx + 1, len(frames), frame)
        row = {
            "frame_index": frame,
            "success": False,
            "reason": "",
            "visible_fixed": [],
            "used_fixed": [],
            "inlier_fixed": [],
            "target_inliers": 0,
            "target_rmse": float("inf"),
            "T_w_c07": None,
            "T_world_from_ego_raw": None,
            "T_world_from_ego": None,
            "status": "",
            "confidence": 0.0,
            "detected_tag_ids": [],
            "used_tag_ids": [],
        }

        if cfg.target_type == "apriltag":
            assert tag_map is not None
            # Fixed images validate that the persistent map remains observable. They are
            # deliberately not used to rebuild the world map on every frame.
            for cam_id in cfg.fixed_camera_ids:
                image = _read_image(dataset_root, cam_id, frame)
                detections, _ = detect_apriltag_markers(image, cfg.apriltag_family)
                if any(tag_id in tag_map.world_tag_corners for tag_id, _ in detections):
                    row["visible_fixed"].append(cam_id)

            target_cam = calibrations[cfg.target_camera_id]
            image_target = _read_image(dataset_root, cfg.target_camera_id, frame)
            detections, _ = detect_apriltag_markers(image_target, cfg.apriltag_family)
            row["detected_tag_ids"] = sorted({tag_id for tag_id, _ in detections})
            row["used_tag_ids"] = sorted(tag_id for tag_id in row["detected_tag_ids"] if tag_id in tag_map.world_tag_corners)
            target_res = solve_camera_pose_from_tag_map(
                image=image_target,
                K=target_cam.K,
                dist=target_cam.dist,
                tag_family=cfg.apriltag_family,
                world_tag_corners=tag_map.world_tag_corners,
                reproj_error_px=cfg.pnp_reproj_error_px,
                pnp_iterations=cfg.pnp_iterations,
                min_inliers=cfg.apriltag_min_inliers,
                min_tags=cfg.apriltag_min_tags,
                camera_model=target_cam.camera_model,
            )
            row["target_inliers"] = target_res.inliers
            row["target_rmse"] = target_res.reproj_rmse
            if target_res.success and target_res.T_c_b is not None and target_res.reproj_rmse <= cfg.max_target_reproj_rmse_px:
                pose = invert_transform(target_res.T_c_b)
                row["success"] = True
                row["reason"] = ""
                row["status"] = "measured"
                row["T_world_from_ego_raw"] = pose
            else:
                row["reason"] = f"target_failed:{target_res.reason or 'high_reproj_error'}"
                row["status"] = row["reason"]
            rows.append(row)
            continue

        candidates: List[BoardCandidate] = []
        for cam_id in cfg.fixed_camera_ids:
            cam = calibrations.get(cam_id)
            if cam is None:
                continue
            image = _read_image(dataset_root, cam_id, frame)
            res = solve_target_pose(image, cam)
            if not res.success or res.T_c_b is None:
                continue
            row["visible_fixed"].append(cam_id)
            T_w_b = compose(cam.T_w_c, res.T_c_b)
            candidates.append(BoardCandidate(cam_id, T_w_b, res.reproj_rmse, res.inliers))

        if len(candidates) < cfg.min_fixed_observations:
            row["reason"] = "insufficient_fixed_observations"
            rows.append(row)
            continue

        fused = fuse_board_pose(
            candidates,
            trans_thresh_m=cfg.fusion_ransac_trans_thresh_m,
            rot_thresh_deg=cfg.fusion_ransac_rot_thresh_deg,
        )
        row["used_fixed"] = fused.used_camera_ids
        row["inlier_fixed"] = fused.inlier_camera_ids
        if not fused.success or fused.T_w_b is None:
            row["reason"] = f"fusion_failed:{fused.reason}"
            rows.append(row)
            continue

        target_cam = calibrations[cfg.target_camera_id]
        image_target = _read_image(dataset_root, cfg.target_camera_id, frame)
        target_res = solve_target_pose(image_target, target_cam)
        if not target_res.success or target_res.T_c_b is None:
            row["reason"] = f"target_failed:{target_res.reason}"
            rows.append(row)
            continue

        T_w_c07 = compose(fused.T_w_b, invert_transform(target_res.T_c_b))
        row["success"] = True
        row["reason"] = ""
        row["target_inliers"] = target_res.inliers
        row["target_rmse"] = target_res.reproj_rmse
        row["T_w_c07"] = T_w_c07
        rows.append(row)

    if cfg.target_type == "apriltag":
        rows = postprocess_trajectory(
            rows,
            max_interp_gap=cfg.max_interp_gap,
            smoothing_radius=cfg.smoothing_radius,
            trans_outlier_m=cfg.temporal_trans_outlier_m,
            rot_outlier_deg=cfg.temporal_rot_outlier_deg,
        )
    logger.info("Pipeline finished: success=%d / total=%d", sum(int(r["success"]) for r in rows), len(rows))
    return rows
