from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .se3 import make_transform, rotation_angle_deg


@dataclass
class BoardCandidate:
    camera_id: str
    T_w_b: np.ndarray
    rmse: float
    inliers: int


@dataclass
class FusionResult:
    success: bool
    T_w_b: np.ndarray | None
    used_camera_ids: list[str]
    inlier_camera_ids: list[str]
    reason: str = ""


def _weighted_average_rotation(rotations: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    M = np.zeros((3, 3), dtype=np.float64)
    for R, w in zip(rotations, weights):
        M += w * R
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _weights_from_rmse(candidates: List[BoardCandidate]) -> np.ndarray:
    rmse = np.array([max(c.rmse, 1e-6) for c in candidates], dtype=np.float64)
    w = 1.0 / rmse
    w /= np.sum(w)
    return w


def fuse_board_pose(
    candidates: List[BoardCandidate],
    trans_thresh_m: float,
    rot_thresh_deg: float,
) -> FusionResult:
    if len(candidates) == 0:
        return FusionResult(False, None, [], [], reason="no_candidates")
    if len(candidates) == 1:
        single = candidates[0]
        return FusionResult(True, single.T_w_b, [single.camera_id], [single.camera_id])

    best_inlier_idx: list[int] = []
    for i, hypo in enumerate(candidates):
        inliers = []
        t_h = hypo.T_w_b[:3, 3]
        R_h = hypo.T_w_b[:3, :3]
        for j, c in enumerate(candidates):
            t = c.T_w_b[:3, 3]
            R = c.T_w_b[:3, :3]
            trans_err = float(np.linalg.norm(t - t_h))
            rot_err = rotation_angle_deg(R_h, R)
            if trans_err <= trans_thresh_m and rot_err <= rot_thresh_deg:
                inliers.append(j)
        if len(inliers) > len(best_inlier_idx):
            best_inlier_idx = inliers

    if len(best_inlier_idx) == 0:
        return FusionResult(False, None, [c.camera_id for c in candidates], [], reason="ransac_failed")

    inlier_candidates = [candidates[i] for i in best_inlier_idx]
    weights = _weights_from_rmse(inlier_candidates)
    t_stack = np.array([c.T_w_b[:3, 3] for c in inlier_candidates], dtype=np.float64)
    t_fused = np.sum(t_stack * weights[:, None], axis=0)
    R_fused = _weighted_average_rotation([c.T_w_b[:3, :3] for c in inlier_candidates], weights)

    T_w_b = make_transform(R_fused, t_fused)
    return FusionResult(
        True,
        T_w_b,
        used_camera_ids=[c.camera_id for c in candidates],
        inlier_camera_ids=[c.camera_id for c in inlier_candidates],
        reason="",
    )
