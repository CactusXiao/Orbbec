from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


def hand_focus_region(
    points: Sequence[Sequence[Point]],
    visible: Sequence[Sequence[bool]],
    *,
    image_size: Tuple[int, int],
    padding_ratio: float = 0.30,
    min_size_pixels: float = 96.0,
) -> Optional[Box]:
    """Return a padded image-space box around all visible projected hand joints.

    Only finite joints that land inside the image participate in the crop.  The
    returned region stays inside the image without changing its requested size;
    the canvas expands it to its own aspect ratio when rendering.
    """
    image_width, image_height = (int(image_size[0]), int(image_size[1]))
    if image_width <= 0 or image_height <= 0:
        return None

    valid_points: List[Point] = []
    for hand_index, hand_points in enumerate(points):
        hand_visible = visible[hand_index] if hand_index < len(visible) else ()
        for joint_index, point in enumerate(hand_points):
            if joint_index >= len(hand_visible) or not hand_visible[joint_index]:
                continue
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if 0.0 <= x < image_width and 0.0 <= y < image_height:
                valid_points.append((x, y))

    if not valid_points:
        return None

    xs = [point[0] for point in valid_points]
    ys = [point[1] for point in valid_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    padding_ratio = max(0.0, float(padding_ratio))
    min_size_pixels = max(1.0, float(min_size_pixels))
    width = max(min_size_pixels, (x_max - x_min) * (1.0 + 2.0 * padding_ratio))
    height = max(min_size_pixels, (y_max - y_min) * (1.0 + 2.0 * padding_ratio))
    x1, x2 = _bounded_axis(center_x, width, float(image_width))
    y1, y2 = _bounded_axis(center_y, height, float(image_height))
    return (x1, y1, x2, y2)


def _bounded_axis(center: float, size: float, limit: float) -> Tuple[float, float]:
    size = min(limit, max(1.0, size))
    start = min(max(0.0, center - size / 2.0), limit - size)
    return (start, start + size)


def expand_region_to_aspect(region: Box, image_size: Tuple[int, int], target_aspect: float) -> Box:
    """Expand an image-space region to a display aspect ratio, keeping it in bounds."""
    iw, ih = float(image_size[0]), float(image_size[1])
    x1, y1, x2, y2 = region
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    target_aspect = max(1e-6, float(target_aspect))

    crop_width = max(width, height * target_aspect)
    crop_height = crop_width / target_aspect
    if crop_height > ih:
        crop_height = ih
        crop_width = crop_height * target_aspect
    if crop_width > iw:
        crop_width = iw
        crop_height = crop_width / target_aspect

    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    crop_x1 = _position_crop_axis(center_x, crop_width, iw, x1, x2)
    crop_y1 = _position_crop_axis(center_y, crop_height, ih, y1, y2)
    return (crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height)


def _position_crop_axis(center: float, crop_size: float, limit: float, focus_start: float, focus_end: float) -> float:
    lower = max(0.0, focus_end - crop_size)
    upper = min(focus_start, limit - crop_size)
    centered = min(max(0.0, center - crop_size / 2.0), max(0.0, limit - crop_size))
    if lower <= upper:
        return min(max(centered, lower), upper)
    return centered
