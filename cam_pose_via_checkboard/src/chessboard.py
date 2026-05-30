from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .se3 import make_transform, rodrigues_to_matrix


@dataclass
class PnPResult:
    success: bool
    T_c_b: np.ndarray | None
    inliers: int
    reproj_rmse: float
    corner_count: int
    reason: str = ""


def build_board_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    grid[:, 0] = xs.reshape(-1) * square_size_m
    grid[:, 1] = ys.reshape(-1) * square_size_m
    return grid


def detect_chessboard_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags_sb)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found or corners is None:
        return False, None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
    return True, corners_refined


def solve_board_pnp(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    cols: int,
    rows: int,
    square_size_m: float,
    reproj_error_px: float,
    pnp_iterations: int,
    min_inliers: int,
) -> PnPResult:
    if image is None:
        return PnPResult(False, None, 0, float("inf"), 0, reason="image_missing")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_size = (cols, rows)
    found, corners = detect_chessboard_corners(gray, pattern_size)
    if not found or corners is None:
        return PnPResult(False, None, 0, float("inf"), 0, reason="corners_not_found")

    object_pts = build_board_points(cols, rows, square_size_m)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_pts,
        corners,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=float(reproj_error_px),
        iterationsCount=int(pnp_iterations),
        confidence=0.999,
    )
    if not ok:
        return PnPResult(False, None, 0, float("inf"), len(corners), reason="pnp_failed")

    inlier_count = int(len(inliers)) if inliers is not None else 0
    if inlier_count < int(min_inliers):
        return PnPResult(False, None, inlier_count, float("inf"), len(corners), reason="few_inliers")

    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    reproj = np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(reproj * reproj)))

    R = rodrigues_to_matrix(rvec)
    T_c_b = make_transform(R, tvec.reshape(3))
    return PnPResult(True, T_c_b, inlier_count, rmse, len(corners), reason="")
