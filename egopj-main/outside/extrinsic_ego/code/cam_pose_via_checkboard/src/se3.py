from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from R(3x3) and t(3,)."""
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


def invert_transform(T_ab: np.ndarray) -> np.ndarray:
    """Compute inverse transform T_ba from T_ab."""
    T_ab = np.asarray(T_ab, dtype=np.float64).reshape(4, 4)
    R_ab = T_ab[:3, :3]
    t_ab = T_ab[:3, 3]
    T_ba = np.eye(4, dtype=np.float64)
    T_ba[:3, :3] = R_ab.T
    T_ba[:3, 3] = -R_ab.T @ t_ab
    return T_ba


def compose(*Ts: Iterable[np.ndarray]) -> np.ndarray:
    """Compose multiple 4x4 transforms in left-to-right order."""
    out = np.eye(4, dtype=np.float64)
    for T in Ts:
        out = out @ np.asarray(T, dtype=np.float64).reshape(4, 4)
    return out


def is_valid_rotation(R: np.ndarray, atol: float = 1e-4) -> bool:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    should_be_identity = R.T @ R
    det = np.linalg.det(R)
    return np.allclose(should_be_identity, np.eye(3), atol=atol) and abs(det - 1.0) < atol


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angular difference in degrees between two rotation matrices."""
    R_rel = np.asarray(R_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R_rel)
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cos_theta))


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
    return rvec.reshape(3)
