from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import estimate_pico_ego_extrinsics as estimator


def _make_estimate(index: int, pose: np.ndarray | None, *, source: str = "direct"):
    row = estimator.TimestampRow(
        row_index=index,
        frame_index=f"{index:05d}",
        frame_number=index,
        ref_timestamp_us=str(1_000_000 + index * 33_333),
        ego_frame_index=str(index),
        ego_frame_number=index,
        ego_timestamp_us=str(1_000_000 + index * 33_333),
        raw={},
    )
    return estimator.FrameEstimate(
        row=row,
        pose_direct=None if source == "interpolated" else pose,
        pose_final=pose,
        status_initial="ok",
        status_final=source,
        source=source,
    )


def _make_jittery_estimates(frame_count: int = 31):
    estimates = []
    for index in range(frame_count):
        jitter = 0.003 if index % 2 == 0 else -0.003
        angle = math.radians(0.15 * index + (0.35 if index % 2 == 0 else -0.35))
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        world_from_ego = estimator._make_transform(
            rotation,
            np.array([0.002 * index + jitter, 0.0, 0.0], dtype=np.float64),
        )
        source = "interpolated" if index == frame_count // 2 else "direct"
        estimates.append(_make_estimate(index, estimator._invert_transform(world_from_ego), source=source))
    return estimates


def test_savgol_smoothing_reduces_motion_jitter_and_smooths_interpolated_frames():
    estimates = _make_jittery_estimates()
    raw_interpolated_pose = estimates[len(estimates) // 2].pose_final.copy()
    summary = estimator._apply_savgol_smoothing(estimates, window_size=11, polyorder=3)

    raw = summary["raw_trajectory_smoothness"]
    smoothed = summary["smoothed_trajectory_smoothness"]
    assert smoothed["translation_second_difference_cm"]["median"] < raw["translation_second_difference_cm"]["median"] * 0.25
    assert smoothed["rotation_increment_difference_deg"]["median"] < raw["rotation_increment_difference_deg"]["median"] * 0.25
    assert estimates[len(estimates) // 2].smoothing_status == "smoothed"
    assert not np.allclose(estimates[len(estimates) // 2].pose_smoothed, raw_interpolated_pose)


def test_savgol_rotations_are_valid_and_raw_poses_remain_unchanged():
    estimates = _make_jittery_estimates()
    raw_poses = [estimate.pose_final.copy() for estimate in estimates]
    estimator._apply_savgol_smoothing(estimates, window_size=11, polyorder=3)

    for estimate, raw_pose in zip(estimates, raw_poses):
        assert np.array_equal(estimate.pose_final, raw_pose)
        rotation = estimate.pose_smoothed[:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)


def test_short_and_missing_pose_segments_are_copied_without_filtering():
    estimates = _make_jittery_estimates(frame_count=9)
    estimates[4].pose_final = None
    estimates[4].pose_direct = None
    estimator._apply_savgol_smoothing(estimates, window_size=11, polyorder=3)

    assert estimates[4].pose_smoothed is None
    assert estimates[4].smoothing_status == "unavailable"
    for index, estimate in enumerate(estimates):
        if index == 4:
            continue
        assert np.array_equal(estimate.pose_smoothed, estimate.pose_final)
        assert estimate.smoothing_status == "copied_short_segment"
