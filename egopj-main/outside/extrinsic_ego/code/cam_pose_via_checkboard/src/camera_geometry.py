from __future__ import annotations

import cv2
import numpy as np


PINHOLE = "pinhole"
FISHEYE = "fisheye"


def validate_camera_model(camera_model: str) -> str:
    model = str(camera_model).strip().lower()
    if model not in {PINHOLE, FISHEYE}:
        raise ValueError(f"Unsupported camera_model={camera_model!r}; expected pinhole or fisheye")
    return model


def prepare_pnp_image_points(
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    camera_model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return image points, K and distortion in a pinhole plane suitable for solvePnP.

    For fisheye cameras only sparse detected points are rectified. Keeping P=K means
    RANSAC thresholds remain expressed in pixels without remapping the whole image.
    """
    model = validate_camera_model(camera_model)
    points = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    K64 = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if model == FISHEYE:
        D = np.asarray(dist, dtype=np.float64).reshape(-1)
        if D.size != 4:
            raise ValueError(f"Fisheye distortion must contain exactly 4 coefficients, got {D.size}")
        rectified = cv2.fisheye.undistortPoints(points, K64, D.reshape(4, 1), P=K64)
        return rectified.reshape(-1, 2).astype(np.float32), K64, np.zeros(4, dtype=np.float64)
    return points.reshape(-1, 2).astype(np.float32), K64, np.asarray(dist, dtype=np.float64).reshape(-1)


def project_points(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    camera_model: str,
) -> np.ndarray:
    model = validate_camera_model(camera_model)
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    rv = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tv = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    K64 = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if model == FISHEYE:
        D = np.asarray(dist, dtype=np.float64).reshape(-1)
        if D.size != 4:
            raise ValueError(f"Fisheye distortion must contain exactly 4 coefficients, got {D.size}")
        projected, _ = cv2.fisheye.projectPoints(obj, rv, tv, K64, D.reshape(4, 1))
    else:
        projected, _ = cv2.projectPoints(obj, rv, tv, K64, np.asarray(dist, dtype=np.float64))
    return projected.reshape(-1, 2)


def reprojection_rmse(
    object_points: np.ndarray,
    observed_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    camera_model: str,
) -> float:
    projected = project_points(object_points, rvec, tvec, K, dist, camera_model)
    observed = np.asarray(observed_points, dtype=np.float64).reshape(-1, 2)
    errors = np.linalg.norm(projected - observed, axis=1)
    return float(np.sqrt(np.mean(errors * errors)))
