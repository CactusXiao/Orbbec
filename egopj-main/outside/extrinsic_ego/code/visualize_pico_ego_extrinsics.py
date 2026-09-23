#!/usr/bin/env python3
"""
Visualize static third-person cameras and the PICO ego trajectory in 3D.

Static cameras are black points. Ego points are red for direct PnP estimates and
blue for non-direct estimates such as interpolation. Unavailable edge poses are
shown as green points in the upper-right corner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EPISODE_DIR = (
    SCRIPT_DIR.parent / "test_sample_final" / "hand_shape_calibration" / "episode_1"
)


@dataclass
class StaticCameraPoint:
    camera_id: str
    position: np.ndarray


@dataclass
class EgoPoint:
    frame_index: str
    ego_frame_index: str
    status_final: str
    source: str
    position: np.ndarray | None


@dataclass
class RenderCamera:
    eye: np.ndarray
    target: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    focal_px: float


@dataclass
class InteractiveCameraState:
    camera: RenderCamera
    dragging: bool = False
    last_x: int = 0
    last_y: int = 0


@dataclass
class ViewDefinition:
    ego_center: np.ndarray
    static_center: np.ndarray
    direction_ego_to_static: np.ndarray
    eye: np.ndarray
    target: np.ndarray


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def _invert_transform(T_ab: np.ndarray) -> np.ndarray:
    T_ab = np.asarray(T_ab, dtype=np.float64).reshape(4, 4)
    R_ab = T_ab[:3, :3]
    t_ab = T_ab[:3, 3]
    T_ba = np.eye(4, dtype=np.float64)
    T_ba[:3, :3] = R_ab.T
    T_ba[:3, 3] = -R_ab.T @ t_ab
    return T_ba


def _compose(*transforms: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    for T in transforms:
        out = out @ np.asarray(T, dtype=np.float64).reshape(4, 4)
    return out


def _camera_sort_key(camera_id: str) -> tuple[int, int | str]:
    return (0, int(camera_id)) if camera_id.isdigit() else (1, camera_id)


def _load_static_camera_points(extrinsics_path: Path, reference_camera_id: str) -> list[StaticCameraPoint]:
    extrinsics = _load_json(extrinsics_path)
    if reference_camera_id not in extrinsics:
        raise KeyError(f"Reference camera {reference_camera_id} not found in {extrinsics_path}")

    reference_entry = extrinsics[reference_camera_id]
    T_reference_camera_from_world = _make_transform(
        np.asarray(reference_entry["rotation"], dtype=np.float64),
        np.asarray(reference_entry["translation"], dtype=np.float64),
    )
    T_world_from_reference = _invert_transform(T_reference_camera_from_world)

    points: list[StaticCameraPoint] = []
    for camera_id in sorted(extrinsics, key=_camera_sort_key):
        entry = extrinsics[camera_id]
        if not isinstance(entry, dict) or "rotation" not in entry or "translation" not in entry:
            continue
        T_camera_from_world = _make_transform(
            np.asarray(entry["rotation"], dtype=np.float64),
            np.asarray(entry["translation"], dtype=np.float64),
        )
        T_camera_from_reference = _compose(T_camera_from_world, T_world_from_reference)
        T_reference_from_camera = _invert_transform(T_camera_from_reference)
        points.append(StaticCameraPoint(camera_id=camera_id, position=T_reference_from_camera[:3, 3]))
    return points


def _parse_matrix_from_row(row: dict[str, str]) -> np.ndarray | None:
    values: list[float] = []
    for r in range(4):
        for c in range(4):
            raw = row.get(f"m{r}{c}", "")
            try:
                value = float(raw)
            except ValueError:
                return None
            values.append(value)
    T = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(T)):
        return None
    return T


def _load_ego_points(ego_csv_path: Path) -> tuple[list[EgoPoint], dict[str, int]]:
    points: list[EgoPoint] = []
    finite_count = 0
    nan_count = 0
    invalid_count = 0
    with ego_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            point = EgoPoint(
                frame_index=row.get("frame_index", ""),
                ego_frame_index=row.get("ego_frame_index", ""),
                status_final=row.get("status_final", ""),
                source=row.get("source", ""),
                position=None,
            )
            T_ego_from_reference = _parse_matrix_from_row(row)
            if T_ego_from_reference is None:
                nan_count += 1
                points.append(point)
                continue
            try:
                T_reference_from_ego = _invert_transform(T_ego_from_reference)
            except Exception:
                invalid_count += 1
                points.append(point)
                continue
            point.position = T_reference_from_ego[:3, 3]
            points.append(point)
            finite_count += 1
    stats = {
        "total_rows": len(points),
        "finite": finite_count,
        "nan": nan_count,
        "invalid": invalid_count,
    }
    return points, stats


def _finite_ego_positions(ego_points: list[EgoPoint]) -> np.ndarray:
    positions = [point.position for point in ego_points if point.position is not None]
    if not positions:
        raise RuntimeError("No finite ego poses are available for 3D camera setup.")
    return np.asarray(positions, dtype=np.float64)


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return vector / norm


def _rotate_vector(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis, np.array([0.0, 0.0, 1.0]))
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    cos_a = math.cos(float(angle_rad))
    sin_a = math.sin(float(angle_rad))
    return vector * cos_a + np.cross(axis, vector) * sin_a + axis * float(np.dot(axis, vector)) * (1.0 - cos_a)


def _reorthonormalize_camera(camera: RenderCamera, eye: np.ndarray, up_hint: np.ndarray) -> RenderCamera:
    forward = _normalize(camera.target - eye, camera.forward)
    right = _normalize(np.cross(forward, up_hint), camera.right)
    up = _normalize(np.cross(right, forward), camera.up)
    return RenderCamera(
        eye=eye,
        target=camera.target,
        right=right,
        up=up,
        forward=forward,
        focal_px=camera.focal_px,
    )


def _orbit_camera(camera: RenderCamera, dx: int, dy: int, sensitivity: float = 0.006) -> RenderCamera:
    offset = camera.eye - camera.target
    if float(np.linalg.norm(offset)) < 1e-6:
        return camera

    yaw = -float(dx) * sensitivity
    pitch = -float(dy) * sensitivity

    offset = _rotate_vector(offset, camera.up, yaw)
    right = _rotate_vector(camera.right, camera.up, yaw)
    up = camera.up

    offset = _rotate_vector(offset, right, pitch)
    up = _rotate_vector(up, right, pitch)
    eye = camera.target + offset
    return _reorthonormalize_camera(camera, eye, up)


def _zoom_camera(camera: RenderCamera, scale: float) -> RenderCamera:
    offset = camera.eye - camera.target
    distance = max(0.05, float(np.linalg.norm(offset)) * float(scale))
    eye = camera.target + _normalize(offset, np.array([0.0, 0.0, -1.0])) * distance
    return _reorthonormalize_camera(camera, eye, camera.up)


def _build_view_definition(
    static_points: list[StaticCameraPoint],
    ego_points: list[EgoPoint],
    behind_distance_m: float,
) -> ViewDefinition:
    static_positions = np.asarray([point.position for point in static_points], dtype=np.float64)
    ego_positions = _finite_ego_positions(ego_points)

    static_center = np.mean(static_positions, axis=0)
    ego_center = np.mean(ego_positions, axis=0)
    direction = _normalize(static_center - ego_center, np.array([0.0, 0.0, 1.0]))
    eye = ego_center - behind_distance_m * direction
    return ViewDefinition(
        ego_center=ego_center,
        static_center=static_center,
        direction_ego_to_static=direction,
        eye=eye,
        target=static_center,
    )


def _build_render_camera(
    view: ViewDefinition,
    static_points: list[StaticCameraPoint],
    width: int,
    height: int,
    fov_deg: float,
    match_static_layout: bool,
) -> RenderCamera:
    forward = _normalize(view.target - view.eye, np.array([0.0, 0.0, 1.0]))

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = _normalize(np.cross(forward, world_up), np.array([1.0, 0.0, 0.0]))
    up = _normalize(np.cross(right, forward), np.array([0.0, 0.0, 1.0]))

    focal_px = 0.5 * min(width, height) / math.tan(math.radians(fov_deg) * 0.5)
    camera = RenderCamera(
        eye=view.eye,
        target=view.target,
        right=right,
        up=up,
        forward=forward,
        focal_px=focal_px,
    )
    if match_static_layout:
        return _roll_camera_to_match_real_static_layout(camera, static_points)
    return camera


def _roll_camera_to_match_real_static_layout(
    camera: RenderCamera,
    static_points: list[StaticCameraPoint],
) -> RenderCamera:
    desired_screen = {
        "00": np.array([1.0, 1.0], dtype=np.float64),
        "01": np.array([1.0, -1.0], dtype=np.float64),
        "02": np.array([0.75, 0.0], dtype=np.float64),
        "03": np.array([-0.75, 0.0], dtype=np.float64),
        "04": np.array([-1.0, -1.0], dtype=np.float64),
        "05": np.array([-1.0, 1.0], dtype=np.float64),
    }
    usable = [point for point in static_points if point.camera_id in desired_screen]
    if len(usable) < 2:
        return camera

    best_score = -float("inf")
    best_right = camera.right
    best_up = camera.up
    for angle in np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False):
        cos_a = math.cos(float(angle))
        sin_a = math.sin(float(angle))
        right = cos_a * camera.right + sin_a * camera.up
        up = -sin_a * camera.right + cos_a * camera.up
        score = 0.0
        for point in usable:
            rel = point.position - camera.eye
            x = float(np.dot(rel, right))
            y = float(np.dot(rel, up))
            norm = max(math.sqrt(x * x + y * y), 1e-9)
            xy = np.array([x / norm, y / norm], dtype=np.float64)
            target = desired_screen[point.camera_id]
            target_norm = max(float(np.linalg.norm(target)), 1e-9)
            score += float(np.dot(xy, target / target_norm))
            if abs(float(target[1])) < 1e-9:
                score -= 0.2 * abs(xy[1])
        if score > best_score:
            best_score = score
            best_right = right
            best_up = up

    return RenderCamera(
        eye=camera.eye,
        target=camera.target,
        right=_normalize(best_right, camera.right),
        up=_normalize(best_up, camera.up),
        forward=camera.forward,
        focal_px=camera.focal_px,
    )


def _project_points(points: np.ndarray, camera: RenderCamera, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rel = np.asarray(points, dtype=np.float64).reshape(-1, 3) - camera.eye.reshape(1, 3)
    x = rel @ camera.right
    y = rel @ camera.up
    z = rel @ camera.forward
    visible = z > 1e-4
    uv = np.zeros((rel.shape[0], 2), dtype=np.float64)
    uv[:, 0] = width * 0.5 + camera.focal_px * x / np.maximum(z, 1e-4)
    uv[:, 1] = height * 0.5 - camera.focal_px * y / np.maximum(z, 1e-4)
    return uv, visible


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def _draw_line_3d(
    image: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    camera: RenderCamera,
    color: tuple[int, int, int],
    width_px: int,
) -> None:
    uv, visible = _project_points(np.asarray([p0, p1]), camera, image.shape[1], image.shape[0])
    if bool(visible[0]) and bool(visible[1]):
        cv2.line(
            image,
            tuple(np.round(uv[0]).astype(int)),
            tuple(np.round(uv[1]).astype(int)),
            color,
            width_px,
            cv2.LINE_AA,
        )


def _draw_point_3d(
    image: np.ndarray,
    point: np.ndarray,
    camera: RenderCamera,
    color: tuple[int, int, int],
    radius: int,
    label: str | None = None,
) -> None:
    uv, visible = _project_points(np.asarray([point]), camera, image.shape[1], image.shape[0])
    if not bool(visible[0]):
        return
    x, y = np.round(uv[0]).astype(int)
    if x < -100 or y < -100 or x > image.shape[1] + 100 or y > image.shape[0] + 100:
        return
    cv2.circle(image, (x, y), radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
    if label:
        _draw_text(image, label, (x + radius + 5, y - radius - 4), 0.45)


def _draw_unavailable_marker(image: np.ndarray, point: EgoPoint, radius: int) -> None:
    center = (image.shape[1] - 72, 72)
    cv2.circle(image, center, radius + 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius + 2, (40, 170, 40), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius + 2, (15, 100, 15), 2, cv2.LINE_AA)
    text = f"unavailable pose: {point.status_final or point.source or 'nan'}"
    text_origin = (max(24, center[0] - 430), center[1] + 6)
    _draw_text(
        image,
        text,
        text_origin,
        0.5,
    )


def _mouse_callback(event: int, x: int, y: int, flags: int, state: InteractiveCameraState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        state.dragging = True
        state.last_x = int(x)
        state.last_y = int(y)
        return
    if event == cv2.EVENT_LBUTTONUP:
        state.dragging = False
        return
    if event == cv2.EVENT_MOUSEMOVE and state.dragging:
        dx = int(x) - state.last_x
        dy = int(y) - state.last_y
        state.camera = _orbit_camera(state.camera, dx, dy)
        state.last_x = int(x)
        state.last_y = int(y)
        return
    if event == cv2.EVENT_MOUSEWHEEL:
        state.camera = _zoom_camera(state.camera, 0.9 if flags > 0 else 1.1)


def _draw_axes(
    image: np.ndarray,
    all_positions: np.ndarray,
    camera: RenderCamera,
) -> None:
    center = np.mean(all_positions, axis=0)
    span = np.ptp(all_positions, axis=0)
    axis_len = max(0.1, float(np.max(span)) * 0.18)
    _draw_line_3d(image, center, center + np.array([axis_len, 0.0, 0.0]), camera, (40, 40, 220), 2)
    _draw_line_3d(image, center, center + np.array([0.0, axis_len, 0.0]), camera, (40, 180, 40), 2)
    _draw_line_3d(image, center, center + np.array([0.0, 0.0, axis_len]), camera, (220, 80, 40), 2)
    _draw_point_3d(image, center + np.array([axis_len, 0.0, 0.0]), camera, (40, 40, 220), 3, "X")
    _draw_point_3d(image, center + np.array([0.0, axis_len, 0.0]), camera, (40, 180, 40), 3, "Y")
    _draw_point_3d(image, center + np.array([0.0, 0.0, axis_len]), camera, (220, 80, 40), 3, "Z")


def _nice_grid_step(span: float, target_lines: int = 8) -> float:
    raw = max(float(span) / max(1, target_lines), 1e-4)
    exponent = math.floor(math.log10(raw))
    base = 10.0**exponent
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        step = multiplier * base
        if step >= raw:
            return step
    return 10.0 * base


def _grid_ticks(low: float, high: float, step: float) -> np.ndarray:
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    return np.arange(start, end + step * 0.5, step, dtype=np.float64)


def _draw_spatial_grid(
    image: np.ndarray,
    all_positions: np.ndarray,
    camera: RenderCamera,
    grid_step_m: float | None,
) -> None:
    bounds_min = np.min(all_positions, axis=0)
    bounds_max = np.max(all_positions, axis=0)
    span = np.maximum(bounds_max - bounds_min, 0.1)
    pad = max(0.1, float(np.max(span)) * 0.12)
    low = bounds_min - pad
    high = bounds_max + pad

    step = float(grid_step_m) if grid_step_m is not None and grid_step_m > 0 else _nice_grid_step(float(np.max(high - low)))
    xs = _grid_ticks(low[0], high[0], step)
    ys = _grid_ticks(low[1], high[1], step)
    zs = _grid_ticks(low[2], high[2], step)

    grid_color = (218, 218, 218)
    box_color = (185, 185, 185)

    z_floor = low[2]
    y_back = low[1]
    x_side = low[0]

    for x in xs:
        _draw_line_3d(image, np.array([x, low[1], z_floor]), np.array([x, high[1], z_floor]), camera, grid_color, 1)
        _draw_line_3d(image, np.array([x, y_back, low[2]]), np.array([x, y_back, high[2]]), camera, grid_color, 1)
    for y in ys:
        _draw_line_3d(image, np.array([low[0], y, z_floor]), np.array([high[0], y, z_floor]), camera, grid_color, 1)
        _draw_line_3d(image, np.array([x_side, y, low[2]]), np.array([x_side, y, high[2]]), camera, grid_color, 1)
    for z in zs:
        _draw_line_3d(image, np.array([low[0], y_back, z]), np.array([high[0], y_back, z]), camera, grid_color, 1)
        _draw_line_3d(image, np.array([x_side, low[1], z]), np.array([x_side, high[1], z]), camera, grid_color, 1)

    corners = [
        np.array([low[0], low[1], low[2]]),
        np.array([high[0], low[1], low[2]]),
        np.array([high[0], high[1], low[2]]),
        np.array([low[0], high[1], low[2]]),
        np.array([low[0], low[1], high[2]]),
        np.array([high[0], low[1], high[2]]),
        np.array([high[0], high[1], high[2]]),
        np.array([low[0], high[1], high[2]]),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        _draw_line_3d(image, corners[a], corners[b], camera, box_color, 1)


def _render_frame(
    static_points: list[StaticCameraPoint],
    ego_points: list[EgoPoint],
    scene_ego_positions: np.ndarray,
    camera: RenderCamera,
    frame_idx: int,
    width: int,
    height: int,
    trajectory_mode: str,
    point_radius: int,
    show_grid: bool,
    grid_step_m: float | None,
) -> np.ndarray:
    image = np.full((height, width, 3), (248, 248, 248), dtype=np.uint8)
    static_positions = np.asarray([point.position for point in static_points], dtype=np.float64)
    all_positions = np.vstack([static_positions, scene_ego_positions])

    if show_grid:
        _draw_spatial_grid(image, all_positions, camera, grid_step_m)
    _draw_axes(image, all_positions, camera)

    if trajectory_mode == "all":
        ego_subset = ego_points
    else:
        ego_subset = ego_points[: frame_idx + 1]

    if len(ego_subset) >= 2:
        for prev, cur in zip(ego_subset, ego_subset[1:]):
            if prev.position is None or cur.position is None:
                continue
            color = (0, 0, 220) if cur.source == "direct" else (220, 70, 30)
            _draw_line_3d(image, prev.position, cur.position, camera, color, 1)

    for point in ego_subset:
        if point.position is None:
            continue
        color = (0, 0, 230) if point.source == "direct" else (230, 90, 30)
        _draw_point_3d(image, point.position, camera, color, max(2, point_radius - 1))

    if ego_points:
        current = ego_points[min(frame_idx, len(ego_points) - 1)]
        if current.position is None:
            _draw_unavailable_marker(image, current, point_radius + 6)
        else:
            current_color = (0, 0, 255) if current.source == "direct" else (255, 80, 20)
            _draw_point_3d(image, current.position, camera, current_color, point_radius + 3)

    for point in static_points:
        radius = point_radius + 4 if point.camera_id == "00" else point_radius + 2
        _draw_point_3d(image, point.position, camera, (0, 0, 0), radius, point.camera_id)

    direct_count = sum(1 for point in ego_subset if point.source == "direct" and point.position is not None)
    nan_count = sum(1 for point in ego_subset if point.position is None)
    nondirect_count = len(ego_subset) - direct_count - nan_count
    current = ego_points[min(frame_idx, len(ego_points) - 1)]
    _draw_text(image, "Black: third-person cameras", (24, 34))
    _draw_text(image, "Red: ego direct PnP   Blue: ego non-direct/interpolated   Green upper-right: unavailable edge pose", (24, 62), 0.5)
    _draw_text(image, "3D grid: reference coordinates; saved view uses ego-center -> static-center ray", (24, 90), 0.5)
    _draw_text(
        image,
        f"frame_index={current.frame_index} ego_frame_index={current.ego_frame_index} "
        f"source={current.source} status={current.status_final}",
        (24, height - 52),
    )
    _draw_text(
        image,
        f"shown rows={len(ego_subset)} direct={direct_count} non_direct={nondirect_count} unavailable={nan_count}",
        (24, height - 24),
    )
    return image


def _select_video_points(ego_points: list[EgoPoint], stride: int, max_frames: int | None) -> list[EgoPoint]:
    selected = ego_points[:: max(1, stride)]
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def visualize(args: argparse.Namespace) -> None:
    episode_dir = Path(args.episode_dir).expanduser().resolve()
    extrinsics_path = Path(args.extrinsics_json).expanduser().resolve() if args.extrinsics_json else episode_dir / "extrinsics.json"
    ego_csv_path = Path(args.ego_csv).expanduser().resolve() if args.ego_csv else episode_dir / "ego_extrinsics_pico" / "ego_extrinsics_aligned.csv"

    static_points = _load_static_camera_points(extrinsics_path, args.reference_camera_id)
    ego_points_all, ego_stats = _load_ego_points(ego_csv_path)
    scene_ego_positions = _finite_ego_positions(ego_points_all)
    ego_points = _select_video_points(ego_points_all, args.stride, args.max_frames)
    if not static_points:
        raise RuntimeError(f"No static camera points loaded from {extrinsics_path}")
    if not ego_points:
        raise RuntimeError(f"No ego rows loaded from {ego_csv_path}")

    output_video = Path(args.output_video).expanduser().resolve() if args.output_video else None
    if output_video is None and not args.show:
        output_video = ego_csv_path.parent / "ego_trajectory_visualization.mp4"

    view = _build_view_definition(
        static_points,
        ego_points_all,
        args.behind_distance_m,
    )
    camera = _build_render_camera(
        view,
        static_points,
        args.width,
        args.height,
        args.fov_deg,
        args.match_static_layout,
    )

    print(f"[ego_visualize] extrinsics={extrinsics_path}")
    print(f"[ego_visualize] ego_csv={ego_csv_path}")
    print(f"[ego_visualize] static_camera_count={len(static_points)}")
    print(
        f"[ego_visualize] ego_rows_total={ego_stats['total_rows']} "
        f"finite={ego_stats['finite']} unavailable_nan={ego_stats['nan']} invalid={ego_stats['invalid']}"
    )
    print(f"[ego_visualize] video_frames_to_render={len(ego_points)} stride={args.stride}")
    print(f"[ego_visualize] ego_position_center={view.ego_center.tolist()}")
    print(f"[ego_visualize] static_camera_center={view.static_center.tolist()}")
    print(f"[ego_visualize] direction_ego_center_to_static_center={view.direction_ego_to_static.tolist()}")
    print(f"[ego_visualize] behind_distance_m={args.behind_distance_m}")
    print(f"[ego_visualize] virtual_eye={camera.eye.tolist()}")
    print(f"[ego_visualize] virtual_target={camera.target.tolist()}")
    print(f"[ego_visualize] match_static_layout={args.match_static_layout}")
    for static_point in static_points:
        static_uv, static_visible = _project_points(
            np.asarray([static_point.position]),
            camera,
            args.width,
            args.height,
        )
        print(
            f"[ego_visualize] camera_{static_point.camera_id}_screen_xy="
            f"{static_uv[0].round(2).tolist()} visible={bool(static_visible[0])}"
        )

    writer: cv2.VideoWriter | None = None
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, float(args.fps), (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_video}")

    window_name = "PICO Ego Extrinsics"
    interactive_state = InteractiveCameraState(camera=camera)
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)
        cv2.setMouseCallback(window_name, _mouse_callback, interactive_state)
        print("[ego_visualize] interactive_controls=left-drag orbit, mouse-wheel zoom, space pause, +/- zoom, q/Esc quit")

    try:
        stop_requested = False
        paused = False
        for idx in range(len(ego_points)):
            wrote_video_frame = False
            while True:
                active_camera = interactive_state.camera if args.show else camera
                image = _render_frame(
                    static_points,
                    ego_points,
                    scene_ego_positions,
                    active_camera,
                    idx,
                    args.width,
                    args.height,
                    args.trajectory_mode,
                    args.point_radius,
                    not args.no_grid,
                    args.grid_step_m,
                )
                if writer is not None and not wrote_video_frame:
                    writer.write(image)
                    wrote_video_frame = True
                if not args.show:
                    break
                cv2.imshow(window_name, image)
                key = cv2.waitKey(max(1, int(1000 / max(1, args.fps)))) & 0xFF
                if key in (27, ord("q")):
                    stop_requested = True
                    break
                if key == ord(" "):
                    paused = not paused
                elif key in (ord("+"), ord("=")):
                    interactive_state.camera = _zoom_camera(interactive_state.camera, 0.9)
                elif key in (ord("-"), ord("_")):
                    interactive_state.camera = _zoom_camera(interactive_state.camera, 1.1)
                if not paused:
                    break
            if stop_requested:
                break
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if output_video is not None:
        print(f"[ego_visualize] output_video={output_video}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize PICO ego extrinsics in a 3D reference coordinate system.")
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--extrinsics-json", type=Path, default=None)
    parser.add_argument("--ego-csv", type=Path, default=None)
    parser.add_argument("--reference-camera-id", type=str, default="00")
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="Show an OpenCV playback window.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth aligned row, including unavailable poses.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--trajectory-mode", choices=("progressive", "all"), default="progressive")
    parser.add_argument("--behind-distance-m", type=float, default=0.5)
    parser.add_argument("--fov-deg", type=float, default=65.0)
    parser.add_argument("--point-radius", type=int, default=5)
    parser.add_argument("--no-grid", action="store_true", help="Disable the 3D reference grid.")
    parser.add_argument("--grid-step-m", type=float, default=None, help="Optional fixed 3D grid spacing in meters.")
    parser.add_argument(
        "--match-static-layout",
        action="store_true",
        help="Roll the view to match the expected 00-05 screen layout. Disabled by default for exact center-ray video view.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.width < 320 or args.height < 240:
        parser.error("--width/--height are too small")
    if args.fps < 1:
        parser.error("--fps must be >= 1")
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be >= 1")
    visualize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
