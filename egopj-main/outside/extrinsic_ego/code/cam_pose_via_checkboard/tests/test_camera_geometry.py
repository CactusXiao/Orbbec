import unittest

import cv2
import numpy as np
from pathlib import Path

from src.camera_geometry import prepare_pnp_image_points, project_points, reprojection_rmse
from src.fisheye_calibration import _board_points, _calibrate


class CameraGeometryTest(unittest.TestCase):
    def test_fisheye_sparse_rectification_recovers_pose(self):
        K = np.array([[420.0, 0.0, 640.0], [0.0, 418.0, 360.0], [0.0, 0.0, 1.0]])
        D = np.array([-0.04, 0.006, -0.001, 0.0002])
        object_points = np.array(
            [
                [-0.4, -0.3, 0.0],
                [0.4, -0.3, 0.0],
                [0.4, 0.3, 0.0],
                [-0.4, 0.3, 0.0],
                [0.0, 0.0, 0.2],
                [0.15, -0.1, 0.35],
            ],
            dtype=np.float64,
        )
        rvec = np.array([0.25, -0.35, 0.1])
        tvec = np.array([0.5, 0.1, 1.8])
        pixels = project_points(object_points, rvec, tvec, K, D, "fisheye")
        rectified, pnp_K, pnp_dist = prepare_pnp_image_points(pixels, K, D, "fisheye")
        ok, estimated_rvec, estimated_tvec = cv2.solvePnP(
            object_points.astype(np.float32), rectified, pnp_K, pnp_dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        self.assertTrue(ok)
        self.assertLess(
            reprojection_rmse(object_points, pixels, estimated_rvec, estimated_tvec, K, D, "fisheye"), 1e-3
        )

    def test_fisheye_requires_four_coefficients(self):
        with self.assertRaises(ValueError):
            prepare_pnp_image_points(np.zeros((4, 2)), np.eye(3), np.zeros(5), "fisheye")

    def test_fisheye_calibration_accepts_opencv_compatible_point_layout(self):
        object_points = _board_points(9, 6, 0.0255)
        K = np.array([[430.0, 0.0, 640.0], [0.0, 428.0, 360.0], [0.0, 0.0, 1.0]])
        D = np.array([-0.04, 0.005, -0.001, 0.0001])
        observations = []
        for i in range(20):
            rvec = np.array([-0.3 + 0.03 * i, -0.2 + 0.1 * (i % 5), 0.05])
            tvec = np.array([-0.1 + 0.05 * (i % 5), -0.1 + 0.04 * (i // 5), 0.7 + 0.05 * (i % 3)])
            pixels, _ = cv2.fisheye.projectPoints(object_points.reshape(-1, 1, 3), rvec, tvec, K, D)
            observations.append((Path(str(i)), object_points.copy(), pixels.reshape(1, -1, 2)))
        rms, estimated_K, estimated_D = _calibrate(observations, (1280, 720))
        self.assertLess(rms, 1e-5)
        np.testing.assert_allclose(estimated_K, K, atol=1e-3)
        np.testing.assert_allclose(estimated_D, D, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
