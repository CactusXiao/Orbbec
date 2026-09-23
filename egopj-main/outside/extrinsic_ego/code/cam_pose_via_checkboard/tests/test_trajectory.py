import unittest

import cv2
import numpy as np

from src.se3 import make_transform
from src.trajectory import interpolate_pose, postprocess_trajectory


def pose(x: float, yaw_deg: float = 0.0) -> np.ndarray:
    rvec = np.array([0.0, 0.0, np.deg2rad(yaw_deg)])
    R, _ = cv2.Rodrigues(rvec)
    return make_transform(R, np.array([x, 0.0, 0.0]))


def row(index: int, T):
    return {
        "frame_index": str(index),
        "T_world_from_ego_raw": T,
        "target_rmse": 1.0 if T is not None else float("inf"),
        "target_inliers": 12 if T is not None else 0,
        "success": T is not None,
        "status": "measured" if T is not None else "target_failed:tags_not_found",
    }


class TrajectoryTest(unittest.TestCase):
    def test_interpolate_pose_uses_slerp(self):
        mid = interpolate_pose(pose(0.0, 170.0), pose(2.0, -170.0), 0.5)
        self.assertAlmostEqual(mid[0, 3], 1.0)
        direction = mid[:3, :3] @ np.array([1.0, 0.0, 0.0])
        self.assertLess(direction[0], -0.99)

    def test_short_gap_is_filled_but_long_and_edge_gaps_remain_invalid(self):
        rows = [row(0, None), row(1, pose(1)), row(2, None), row(3, pose(3)), row(4, None), row(5, None), row(6, pose(6))]
        result = postprocess_trajectory(rows, max_interp_gap=1, smoothing_radius=0, trans_outlier_m=10, rot_outlier_deg=180)
        self.assertFalse(result[0]["valid"])
        self.assertTrue(result[2]["valid"])
        self.assertEqual(result[2]["status"], "interpolated_short_gap")
        self.assertFalse(result[4]["valid"])
        self.assertFalse(result[5]["valid"])
        self.assertEqual(result[4]["status"], "invalid_long_gap")

    def test_isolated_outlier_is_rejected_and_replaced_as_short_gap(self):
        rows = [row(0, pose(0)), row(1, pose(100)), row(2, pose(2))]
        result = postprocess_trajectory(rows, max_interp_gap=1, smoothing_radius=0, trans_outlier_m=0.5, rot_outlier_deg=10)
        self.assertTrue(result[1]["valid"])
        self.assertEqual(result[1]["status"], "interpolated_short_gap")
        self.assertAlmostEqual(result[1]["T_world_from_ego"][0, 3], 1.0)


if __name__ == "__main__":
    unittest.main()
