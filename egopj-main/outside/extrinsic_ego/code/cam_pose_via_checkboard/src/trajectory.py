from __future__ import annotations

import math

import numpy as np

from .se3 import make_transform, rotation_angle_deg


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    q = np.empty(4, dtype=np.float64)  # w, x, y, z
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q[:] = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[:] = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[:] = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[:] = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return q / np.linalg.norm(q)


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    t = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    q0, q1 = _matrix_to_quaternion(T0[:3, :3]), _matrix_to_quaternion(T1[:3, :3])
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
    else:
        theta = math.acos(dot)
        q = (math.sin((1 - alpha) * theta) * q0 + math.sin(alpha * theta) * q1) / math.sin(theta)
    return make_transform(_quaternion_to_matrix(q), t)


def _pose_weight(row: dict) -> float:
    rmse = float(row.get("target_rmse", 4.0))
    if not np.isfinite(rmse):
        return 0.25
    rmse = max(rmse, 0.25)
    inliers = max(int(row.get("target_inliers", 4)), 1)
    return min(float(inliers), 32.0) / rmse


def _smooth_pose_window(poses: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = weights / np.sum(weights)
    translations = np.stack([T[:3, 3] for T in poses])
    translation = np.sum(translations * weights[:, None], axis=0)
    quaternions = [_matrix_to_quaternion(T[:3, :3]) for T in poses]
    reference = quaternions[0]
    A = np.zeros((4, 4), dtype=np.float64)
    for q, weight in zip(quaternions, weights):
        if np.dot(q, reference) < 0:
            q = -q
        A += weight * np.outer(q, q)
    _, vectors = np.linalg.eigh(A)
    return make_transform(_quaternion_to_matrix(vectors[:, -1]), translation)


def postprocess_trajectory(
    rows: list[dict],
    max_interp_gap: int,
    smoothing_radius: int,
    trans_outlier_m: float,
    rot_outlier_deg: float,
) -> list[dict]:
    raw = [r.get("T_world_from_ego_raw") for r in rows]
    final = [None if T is None else np.array(T, dtype=np.float64, copy=True) for T in raw]

    # Replace only isolated, geometrically implausible raw measurements.
    for i in range(1, len(rows) - 1):
        if raw[i - 1] is None or raw[i] is None or raw[i + 1] is None:
            continue
        predicted = interpolate_pose(raw[i - 1], raw[i + 1], 0.5)
        dt = float(np.linalg.norm(raw[i][:3, 3] - predicted[:3, 3]))
        dr = rotation_angle_deg(raw[i][:3, :3], predicted[:3, :3])
        if dt > trans_outlier_m or dr > rot_outlier_deg:
            final[i] = None
            rows[i]["status"] = "outlier_rejected"

    i = 0
    while i < len(rows):
        if final[i] is not None:
            i += 1
            continue
        start = i
        while i < len(rows) and final[i] is None:
            i += 1
        end = i
        gap = end - start
        if start > 0 and end < len(rows) and final[start - 1] is not None and final[end] is not None and gap <= max_interp_gap:
            for offset, idx in enumerate(range(start, end), start=1):
                final[idx] = interpolate_pose(final[start - 1], final[end], offset / float(gap + 1))
                rows[idx]["status"] = "interpolated_short_gap"
        else:
            status = "invalid_long_gap" if start > 0 and end < len(rows) else "invalid_edge_gap"
            for idx in range(start, end):
                rows[idx]["status"] = status

    smoothed = list(final)
    radius = max(int(smoothing_radius), 0)
    if radius:
        for i, pose in enumerate(final):
            if pose is None:
                continue
            indices = [j for j in range(max(0, i - radius), min(len(rows), i + radius + 1)) if final[j] is not None]
            poses = [final[j] for j in indices]
            weights = np.array([_pose_weight(rows[j]) for j in indices], dtype=np.float64)
            smoothed[i] = _smooth_pose_window(poses, weights)

    for row, pose in zip(rows, smoothed):
        row["T_world_from_ego"] = pose
        row["T_w_c07"] = pose  # Backward-compatible output key.
        row["valid"] = pose is not None
        row["success"] = pose is not None
        if pose is not None and not row.get("status"):
            row["status"] = "measured_smoothed"
        rmse = float(row.get("target_rmse", float("inf")))
        row["confidence"] = 0.0 if pose is None or not np.isfinite(rmse) else float(np.clip(1.0 / max(rmse, 1.0), 0.0, 1.0))
    return rows
