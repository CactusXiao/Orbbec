from __future__ import annotations

import cv2
import numpy as np

# smplx MANO order: wrist, index, middle, pinky, ring, thumb, then fingertips.
HAND_SKELETON = (
    (0, 1), (1, 2), (2, 3), (3, 17),
    (0, 4), (4, 5), (5, 6), (6, 18),
    (0, 7), (7, 8), (8, 9), (9, 20),
    (0, 10), (10, 11), (11, 12), (12, 19),
    (0, 13), (13, 14), (14, 15), (15, 16),
)
EDGE_COLORS = ((0,180,255),)*4 + ((80,220,80),)*4 + ((255,180,40),)*4 + ((220,90,220),)*4 + ((80,120,255),)*4
POINT_COLORS = ((255,255,255), (30,30,255))
BBOX_COLORS = ((0,255,255), (255,120,0))


def draw_joints(image: np.ndarray, joints: np.ndarray, hand: int, score_threshold: float = 0.0) -> None:
    if joints.shape[0] != 21:
        raise ValueError(f"joints must have 21 rows, got {joints.shape}")
    height, width = image.shape[:2]
    coords = joints[:, :2]
    scores = joints[:, 2] if joints.shape[1] >= 3 else np.ones(21, dtype=np.float32)
    drawable = np.isfinite(coords).all(axis=1)
    visible = scores >= float(score_threshold)
    for edge_index, (a, b) in enumerate(HAND_SKELETON):
        if drawable[a] and drawable[b] and visible[a] and visible[b]:
            # Clip projected bones to the image instead of dropping off-image joints.
            p1 = tuple(np.round(coords[a]).astype(np.int32).tolist())
            p2 = tuple(np.round(coords[b]).astype(np.int32).tolist())
            ok, clipped_p1, clipped_p2 = cv2.clipLine((0, 0, width, height), p1, p2)
            if ok:
                cv2.line(image, clipped_p1, clipped_p2, EDGE_COLORS[edge_index], 1, cv2.LINE_AA)
    for index, point in enumerate(coords):
        if drawable[index] and visible[index]:
            # OpenCV clips circles that intersect the image boundary.
            center = tuple(np.round(point).astype(np.int32).tolist())
            cv2.circle(image, center, 1, POINT_COLORS[hand], -1, cv2.LINE_AA)


def draw_bbox(image: np.ndarray, bbox: np.ndarray, hand: int) -> None:
    if bbox.shape[0] < 5 or bbox[4] <= 0:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    if x2 > x1 and y2 > y1:
        cv2.rectangle(image, (x1, y1), (x2, y2), BBOX_COLORS[hand], 1, cv2.LINE_AA)
