#!/usr/bin/env python3
"""Project captured PICO eye-tracking rays to fisheye camera UV coordinates.

This is the session-oriented version of the reference project's
outside/project_gaze_uv.py and
outside/gaze_direction_projection/project_gaze_direction_only.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import decode_h265_to_jpg as decoder


cv2 = None
np = None
EPSILON = 1e-9

SCRIPT_DIR = Path(__file__).resolve().parent
OUTSIDE_ROOT = SCRIPT_DIR.parent
DEFAULT_CALIBRATION = OUTSIDE_ROOT / "camera_info" / "fisheye_calibration_result.npz"


@dataclass
class ProjectionConfig:
    name: str
    mode: str
    assumed_depth_m: Optional[float]


@dataclass
class ProjectionResult:
    projected: bool
    inside: bool
    pixel_x: float = math.nan
    pixel_y: float = math.nan
    uv_u: float = math.nan
    uv_v_top: float = math.nan
    uv_v_bottom: float = math.nan
    point_camera_x: float = math.nan
    point_camera_y: float = math.nan
    point_camera_z: float = math.nan
    failure_reason: str = ""


def require_cv2_numpy(load_cv2: bool = True):
    global cv2, np
    if np is not None and (cv2 is not None or not load_cv2):
        return cv2, np
    try:
        import numpy as _np  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "project_gaze_uv.py requires numpy. "
            "Install the server environment first.\n"
            f"Import error: {exc}"
        ) from exc
    _cv2 = None
    if load_cv2:
        try:
            import cv2 as _cv2  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "project_gaze_uv.py visualization requires opencv-python/opencv-python-headless. "
                "Install the server environment first, or use --no-visualization for CSV-only UV output.\n"
                f"Import error: {exc}"
            ) from exc
    cv2 = _cv2
    np = _np
    return cv2, np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project PICO gaze rays to undistorted center-cropped camera UVs."
    )
    parser.add_argument("--session-dir", type=Path, default=None, help="Session directory. Defaults to latest session.")
    parser.add_argument("--metadata", type=Path, default=None, help="metadata.csv path.")
    parser.add_argument("--camera", type=Path, default=None, help="camera.json path.")
    parser.add_argument("--decoded-dir", type=Path, default=None, help="Decoded JPG directory.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION, help="OpenCV fisheye .npz file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--projection-mode",
        choices=["direction", "fixed-depth", "both"],
        default="direction",
        help="direction ignores parallax; fixed-depth intersects the eye ray with depth planes.",
    )
    parser.add_argument("--depths", default="0.6,0.8,1.0", help="Comma-separated fixed depths in meters.")
    parser.add_argument(
        "--spaces",
        default="cropped",
        help="Comma-separated visualization spaces: raw, undistorted, cropped. Default: cropped.",
    )
    parser.add_argument(
        "--crop-size",
        default="1280x960",
        help="Center crop size after fisheye undistortion, formatted WIDTHxHEIGHT. Default: 1280x960.",
    )
    parser.add_argument("--marker-diameter", type=int, default=50, help="Gaze marker diameter in pixels.")
    parser.add_argument("--marker-alpha", type=int, default=200, help="Marker alpha in [0,255].")
    parser.add_argument("--video-fps", type=float, default=30.0, help="Visualization MP4 FPS.")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for quick checks; 0 means all.")
    parser.add_argument("--include-invalid", action="store_true", help="Try projecting rows even when gaze_valid is false.")
    parser.add_argument("--no-visualization", action="store_true", help="Write UV CSV and summary only.")
    parser.add_argument("--no-video", action="store_true", help="Do not create visualization MP4 files.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(row: dict[str, str], name: str, default: float = math.nan) -> float:
    value = row.get(name, "")
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(row: dict[str, str], name: str, default: int = 0) -> int:
    value = row.get(name, "")
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def vec3(row: dict[str, str], prefix: str):
    return np.array(
        [
            parse_float(row, f"{prefix}_x"),
            parse_float(row, f"{prefix}_y"),
            parse_float(row, f"{prefix}_z"),
        ],
        dtype=np.float64,
    )


def quat(row: dict[str, str], prefix: str):
    return normalize_quat(
        np.array(
            [
                parse_float(row, f"{prefix}_x"),
                parse_float(row, f"{prefix}_y"),
                parse_float(row, f"{prefix}_z"),
                parse_float(row, f"{prefix}_w"),
            ],
            dtype=np.float64,
        )
    )


def normalize_vec(values):
    norm = float(np.linalg.norm(values))
    if norm <= EPSILON:
        return values * 0.0
    return values / norm


def normalize_quat(q):
    norm = float(np.linalg.norm(q))
    if norm <= EPSILON:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def is_finite_array(values) -> bool:
    return bool(np.all(np.isfinite(values)))


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return normalize_quat(
        np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float64,
        )
    )


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def rotate_vec(q, v):
    q = normalize_quat(q)
    q_vec = q[:3]
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2.0 * (q[3] * uv + uuv)


def pico_right_handed_pose_to_unity(position, rotation):
    unity_pos = np.array([position[0], position[1], -position[2]], dtype=np.float64)
    unity_rot = normalize_quat(
        np.array([rotation[0], rotation[1], -rotation[2], -rotation[3]], dtype=np.float64)
    )
    return unity_pos, unity_rot


def rgb_local_pose_from_camera_json(camera: dict):
    ext = camera["extrinsics_head_to_rgb_camera"]
    position_rh = np.array([ext["x"], ext["y"], ext["z"]], dtype=np.float64)
    rotation_rh = normalize_quat(
        np.array([ext["rx"], ext["ry"], ext["rz"], ext["rw"]], dtype=np.float64)
    )
    rotation_x_180 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    rgb_rotation_rh = quat_multiply(rotation_rh, rotation_x_180)
    return pico_right_handed_pose_to_unity(position_rh, rgb_rotation_rh)


def compose_pose(parent_pos, parent_rot, local_pos, local_rot):
    world_pos = parent_pos + rotate_vec(parent_rot, local_pos)
    world_rot = quat_multiply(parent_rot, local_rot)
    return world_pos, world_rot


def world_direction_to_camera(gaze_world, camera_world_rot):
    inv_camera_rot = quat_conjugate(camera_world_rot)
    return normalize_vec(rotate_vec(inv_camera_rot, gaze_world))


def world_point_to_camera(point_world, camera_world_pos, camera_world_rot):
    inv_camera_rot = quat_conjugate(camera_world_rot)
    return rotate_vec(inv_camera_rot, point_world - camera_world_pos)


def load_calibration(path: Path):
    data = np.load(path)
    k = np.asarray(data["K"], dtype=np.float64)
    d = np.asarray(data["D"], dtype=np.float64).reshape(-1)
    new_k = np.asarray(data["new_K"], dtype=np.float64)
    if d.shape[0] != 4:
        raise ValueError(f"Expected 4 fisheye coefficients, got {d.shape[0]}")
    if "image_size" in data:
        raw_size = np.asarray(data["image_size"], dtype=np.int64).reshape(-1)
        image_size = (int(raw_size[0]), int(raw_size[1]))
    else:
        image_size = (0, 0)
    return k, d, new_k, image_size


def build_undistort_maps(k, d, new_k, image_size: tuple[int, int]):
    width, height = image_size
    return cv2.fisheye.initUndistortRectifyMap(
        k,
        d.reshape(4, 1),
        np.eye(3, dtype=np.float64),
        new_k,
        (width, height),
        cv2.CV_16SC2,
    )


def project_fisheye(vector_camera_unity, k, d):
    x_cv = float(vector_camera_unity[0])
    y_cv = float(-vector_camera_unity[1])
    z_cv = float(vector_camera_unity[2])
    if z_cv <= EPSILON:
        raise ValueError("behind_camera")
    x = x_cv / z_cv
    y = y_cv / z_cv
    radius = math.hypot(x, y)
    if radius <= EPSILON:
        xd = 0.0
        yd = 0.0
    else:
        theta = math.atan(radius)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        theta_d = theta * (1.0 + d[0] * theta2 + d[1] * theta4 + d[2] * theta6 + d[3] * theta8)
        scale = theta_d / radius
        xd = x * scale
        yd = y * scale
    pixel_x = float(k[0, 0] * xd + k[0, 1] * yd + k[0, 2])
    pixel_y = float(k[1, 0] * xd + k[1, 1] * yd + k[1, 2])
    return pixel_x, pixel_y


def project_pinhole(vector_camera_unity, k):
    z = float(vector_camera_unity[2])
    if z <= EPSILON:
        raise ValueError("behind_camera")
    pixel_x = float(k[0, 0] * (vector_camera_unity[0] / z) + k[0, 2])
    pixel_y = float(k[1, 1] * (-vector_camera_unity[1] / z) + k[1, 2])
    return pixel_x, pixel_y


def make_projection_result(pixel_x: float, pixel_y: float, width: int, height: int, point) -> ProjectionResult:
    inside = 0.0 <= pixel_x < width and 0.0 <= pixel_y < height
    return ProjectionResult(
        projected=True,
        inside=inside,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        uv_u=pixel_x / width,
        uv_v_top=pixel_y / height,
        uv_v_bottom=1.0 - (pixel_y / height),
        point_camera_x=float(point[0]),
        point_camera_y=float(point[1]),
        point_camera_z=float(point[2]),
        failure_reason="" if inside else "outside_image",
    )


def make_cropped_projection_result(
    undistorted: ProjectionResult,
    crop_rect: tuple[int, int, int, int],
) -> ProjectionResult:
    crop_left, crop_top, crop_width, crop_height = crop_rect
    if not undistorted.projected:
        return ProjectionResult(False, False, failure_reason=undistorted.failure_reason)

    pixel_x = undistorted.pixel_x - crop_left
    pixel_y = undistorted.pixel_y - crop_top
    inside = 0.0 <= pixel_x < crop_width and 0.0 <= pixel_y < crop_height
    return ProjectionResult(
        projected=True,
        inside=inside,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        uv_u=pixel_x / crop_width,
        uv_v_top=pixel_y / crop_height,
        uv_v_bottom=1.0 - (pixel_y / crop_height),
        point_camera_x=undistorted.point_camera_x,
        point_camera_y=undistorted.point_camera_y,
        point_camera_z=undistorted.point_camera_z,
        failure_reason="" if inside else "outside_center_crop",
    )


def select_fixed_depth_point(origin_camera, direction_camera, depth_m: float):
    if depth_m <= EPSILON:
        return origin_camera * math.nan, "invalid_assumed_depth"
    if abs(float(direction_camera[2])) <= EPSILON:
        return origin_camera * math.nan, "parallel_to_depth_plane"
    t = (depth_m - float(origin_camera[2])) / float(direction_camera[2])
    if t <= 0.0:
        return origin_camera * math.nan, "intersection_behind_eye_ray"
    return origin_camera + t * direction_camera, ""


def result_fields(prefix: str, result: ProjectionResult, outside_text_for_outside: bool = False) -> dict[str, object]:
    if outside_text_for_outside and result.projected and not result.inside:
        coordinate_value: object = "outside"
    else:
        coordinate_value = None

    return {
        f"{prefix}_projected": result.projected,
        f"{prefix}_inside_image": result.inside,
        f"{prefix}_pixel_x": coordinate_value if coordinate_value is not None else format_float(result.pixel_x),
        f"{prefix}_pixel_y": coordinate_value if coordinate_value is not None else format_float(result.pixel_y),
        f"{prefix}_uv_image_u": coordinate_value if coordinate_value is not None else format_float(result.uv_u),
        f"{prefix}_uv_image_v_top": coordinate_value if coordinate_value is not None else format_float(result.uv_v_top),
        f"{prefix}_uv_unity_v_bottom": coordinate_value
        if coordinate_value is not None
        else format_float(result.uv_v_bottom),
        f"{prefix}_failure_reason": result.failure_reason,
    }


def format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.10g}"


def project_row(
    row: dict[str, str],
    config: ProjectionConfig,
    rgb_local_pos,
    rgb_local_rot,
    k,
    d,
    new_k,
    crop_rect: tuple[int, int, int, int],
    include_invalid: bool,
) -> tuple[ProjectionResult, ProjectionResult, ProjectionResult, dict[str, object]]:
    width = parse_int(row, "width")
    height = parse_int(row, "height")
    diagnostics = {
        "gaze_direction_camera_x": math.nan,
        "gaze_direction_camera_y": math.nan,
        "gaze_direction_camera_z": math.nan,
        "eye_origin_camera_x": math.nan,
        "eye_origin_camera_y": math.nan,
        "eye_origin_camera_z": math.nan,
        "camera_pos_world_x": math.nan,
        "camera_pos_world_y": math.nan,
        "camera_pos_world_z": math.nan,
        "camera_rot_world_x": math.nan,
        "camera_rot_world_y": math.nan,
        "camera_rot_world_z": math.nan,
        "camera_rot_world_w": math.nan,
    }
    if width <= 0 or height <= 0:
        result = ProjectionResult(False, False, failure_reason="invalid_image_size")
        return result, result, result, diagnostics
    if not include_invalid and not parse_bool(row.get("gaze_valid")):
        result = ProjectionResult(False, False, failure_reason="gaze_invalid")
        return result, result, result, diagnostics
    if not parse_bool(row.get("xr_head_valid")):
        result = ProjectionResult(False, False, failure_reason="xr_head_invalid")
        return result, result, result, diagnostics

    gaze_world = normalize_vec(vec3(row, "gaze_world_direction"))
    xr_head_pos = vec3(row, "xr_head_pos")
    xr_head_rot = quat(row, "xr_head_rot")
    eye_world = vec3(row, "eye_pose_position_unity")
    if not is_finite_array(gaze_world) or not is_finite_array(xr_head_pos) or not is_finite_array(xr_head_rot):
        result = ProjectionResult(False, False, failure_reason="non_finite_input_pose")
        return result, result, result, diagnostics
    if float(np.linalg.norm(gaze_world)) <= EPSILON:
        result = ProjectionResult(False, False, failure_reason="zero_gaze_direction")
        return result, result, result, diagnostics

    rgb_pos, rgb_rot = compose_pose(xr_head_pos, xr_head_rot, rgb_local_pos, rgb_local_rot)
    direction_camera = world_direction_to_camera(gaze_world, rgb_rot)
    eye_origin_camera = (
        world_point_to_camera(eye_world, rgb_pos, rgb_rot)
        if is_finite_array(eye_world)
        else np.array([math.nan, math.nan, math.nan], dtype=np.float64)
    )

    diagnostics.update(
        {
            "gaze_direction_camera_x": float(direction_camera[0]),
            "gaze_direction_camera_y": float(direction_camera[1]),
            "gaze_direction_camera_z": float(direction_camera[2]),
            "eye_origin_camera_x": float(eye_origin_camera[0]),
            "eye_origin_camera_y": float(eye_origin_camera[1]),
            "eye_origin_camera_z": float(eye_origin_camera[2]),
            "camera_pos_world_x": float(rgb_pos[0]),
            "camera_pos_world_y": float(rgb_pos[1]),
            "camera_pos_world_z": float(rgb_pos[2]),
            "camera_rot_world_x": float(rgb_rot[0]),
            "camera_rot_world_y": float(rgb_rot[1]),
            "camera_rot_world_z": float(rgb_rot[2]),
            "camera_rot_world_w": float(rgb_rot[3]),
        }
    )

    if config.mode == "direction":
        projection_vector = direction_camera
    elif config.mode == "fixed-depth":
        if not is_finite_array(eye_origin_camera):
            result = ProjectionResult(False, False, failure_reason="non_finite_eye_origin")
            return result, result, result, diagnostics
        projection_vector, failure = select_fixed_depth_point(
            eye_origin_camera, direction_camera, float(config.assumed_depth_m or 0.0)
        )
        if failure:
            result = ProjectionResult(False, False, failure_reason=failure)
            return result, result, result, diagnostics
    else:
        result = ProjectionResult(False, False, failure_reason="unknown_projection_mode")
        return result, result, result, diagnostics

    try:
        raw_x, raw_y = project_fisheye(projection_vector, k, d)
        undistorted_x, undistorted_y = project_pinhole(projection_vector, new_k)
    except ValueError as exc:
        result = ProjectionResult(False, False, failure_reason=str(exc))
        return result, result, result, diagnostics

    raw = make_projection_result(raw_x, raw_y, width, height, projection_vector)
    undistorted = make_projection_result(undistorted_x, undistorted_y, width, height, projection_vector)
    cropped = make_cropped_projection_result(undistorted, crop_rect)
    return raw, undistorted, cropped, diagnostics


def ensure_decoded_image_index(
    session_dir: Path,
    metadata_path: Path,
    decoded_dir: Path,
) -> Path:
    image_paths = sorted(decoded_dir.glob("frame_*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No decoded frame_*.jpg files found under {decoded_dir}")

    metadata_rows = decoder.read_metadata_rows(metadata_path)
    video_path = session_dir / "video.h265"
    network_log_path = decoder.resolve_network_log_path(session_dir, video_path)
    encoded_frame_indices = decoder.read_encoded_frame_indices(network_log_path)
    log_alignment_error = decoder.network_log_alignment_error(
        network_log_path, encoded_frame_indices
    )
    if log_alignment_error:
        raise RuntimeError(f"Frame alignment validation failed: {log_alignment_error}")
    client_count_error = decoder.client_encoder_count_alignment_error(
        session_dir,
        len(encoded_frame_indices) if encoded_frame_indices is not None else None,
    )
    if client_count_error:
        raise RuntimeError(f"Frame alignment validation failed: {client_count_error}")
    alignment_error = decoder.decoded_frame_alignment_error(
        len(image_paths),
        len(metadata_rows),
        encoded_frame_indices,
        metadata_available=True,
    )
    if alignment_error:
        raise RuntimeError(f"Frame alignment validation failed: {alignment_error}")
    metadata_alignment_error = decoder.metadata_frame_alignment_error(
        metadata_rows, encoded_frame_indices
    )
    if metadata_alignment_error:
        raise RuntimeError(f"Frame alignment validation failed: {metadata_alignment_error}")
    return decoder.write_index_csv(
        decoded_dir,
        image_paths,
        metadata_rows,
        encoded_frame_indices=encoded_frame_indices,
    )


def build_decoded_image_index(decoded_dir: Path):
    index_path = decoded_dir / "decoded_index.csv"
    by_frame: dict[int, Path] = {}
    by_tuple: dict[tuple[int, int], Path] = {}
    by_decoded: dict[int, Path] = {}
    exact_frame_index_alignment = False
    if not index_path.is_file():
        return by_frame, by_tuple, by_decoded, exact_frame_index_alignment
    for row in load_csv_rows(index_path):
        image_name = row.get("image_file") or row.get("jpg_file") or ""
        if not image_name:
            continue
        path = decoded_dir / image_name
        frame_index = parse_int(row, "frame_index", -1)
        frame_data_index = parse_int(row, "frame_data_index", -1)
        decoded_index = parse_int(row, "decoded_frame_number", -1)
        if row.get("alignment_source") == "network_log.frame_index":
            exact_frame_index_alignment = True
        if frame_index >= 0:
            by_frame.setdefault(frame_index, path)
        if frame_index >= 0 and frame_data_index >= 0:
            by_tuple[(frame_index, frame_data_index)] = path
        if decoded_index >= 0:
            by_decoded[decoded_index] = path
    return by_frame, by_tuple, by_decoded, exact_frame_index_alignment


def resolve_image_path(row: dict[str, str], row_index: int, decoded_dir: Path, index):
    by_frame, by_tuple, by_decoded, exact_frame_index_alignment = index
    frame_index = parse_int(row, "frame_index", -1)
    frame_data_index = parse_int(row, "frame_data_index", -1)
    if (frame_index, frame_data_index) in by_tuple:
        return by_tuple[(frame_index, frame_data_index)]
    if frame_index in by_frame:
        return by_frame[frame_index]
    if exact_frame_index_alignment:
        return None
    if row_index in by_decoded:
        return by_decoded[row_index]
    return decoded_dir / f"frame_{row_index:06d}.jpg"


def visualize_marker(
    input_path: Path,
    output_path: Path,
    projection: ProjectionResult,
    marker_diameter: int,
    marker_alpha: int,
    undistort_maps=None,
    crop_rect: Optional[tuple[int, int, int, int]] = None,
) -> bool:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    if undistort_maps is not None:
        map1, map2 = undistort_maps
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if crop_rect is not None:
        crop_left, crop_top, crop_width, crop_height = crop_rect
        image = image[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width].copy()
    if projection.inside:
        alpha = max(0.0, min(1.0, marker_alpha / 255.0))
        overlay = image.copy()
        center = (int(round(projection.pixel_x)), int(round(projection.pixel_y)))
        radius = max(1, int(round(marker_diameter / 2.0)))
        cv2.circle(overlay, center, radius, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)
        cv2.circle(image, center, radius, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def write_video_from_frames(visualized_dir: Path, records: Sequence[dict[str, object]], output_path: Path, fps: float):
    if fps <= 0:
        return 0, "invalid_video_fps"
    first_image = None
    first_path = None
    paths = [visualized_dir / str(record["image_file"]) for record in records]
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            first_image = image
            first_path = path
            break
    if first_image is None:
        return 0, "no_readable_visualized_frames"
    height, width = first_image.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return 0, "video_writer_open_failed"
    written = 0
    try:
        for path in paths:
            image = first_image if path == first_path else cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
            written += 1
    finally:
        writer.release()
    return written, "" if written else "no_video_frames_written"


def parse_depths(text: str) -> list[float]:
    values = []
    for token in [item.strip() for item in text.split(",") if item.strip()]:
        depth = float(token)
        if depth <= EPSILON:
            raise ValueError(f"depth must be positive: {depth}")
        values.append(depth)
    return values


def parse_spaces(text: str) -> list[str]:
    spaces = []
    for token in [item.strip().lower() for item in text.split(",") if item.strip()]:
        if token not in {"raw", "undistorted", "cropped"}:
            raise ValueError(f"unknown visualization space: {token}")
        if token not in spaces:
            spaces.append(token)
    return spaces


def parse_crop_size(text: str) -> tuple[int, int]:
    normalized = text.lower().replace(",", "x").replace("*", "x")
    parts = [part.strip() for part in normalized.split("x") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--crop-size must look like 1280x960")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("--crop-size values must be positive")
    return width, height


def center_crop_rect(image_size: tuple[int, int], crop_size: tuple[int, int]) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    crop_width, crop_height = crop_size
    if crop_width > image_width or crop_height > image_height:
        raise ValueError(
            f"crop size {crop_width}x{crop_height} is larger than image size {image_width}x{image_height}"
        )
    crop_left = (image_width - crop_width) // 2
    crop_top = (image_height - crop_height) // 2
    return crop_left, crop_top, crop_width, crop_height


def safe_depth_label(depth: float) -> str:
    return f"{depth:g}".replace(".", "p")


def projection_configs(mode: str, depths_text: str) -> list[ProjectionConfig]:
    configs: list[ProjectionConfig] = []
    if mode in {"direction", "both"}:
        configs.append(ProjectionConfig("direction", "direction", None))
    if mode in {"fixed-depth", "both"}:
        for depth in parse_depths(depths_text):
            configs.append(ProjectionConfig(f"fixed_depth_{safe_depth_label(depth)}m", "fixed-depth", depth))
    return configs


def process_config(
    rows: Sequence[dict[str, str]],
    config: ProjectionConfig,
    rgb_local_pos,
    rgb_local_rot,
    k,
    d,
    new_k,
    decoded_dir: Path,
    output_dir: Path,
    spaces: Sequence[str],
    undistort_maps,
    crop_rect: tuple[int, int, int, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    config_dir = output_dir / config.name
    raw_dir = config_dir / "visualized_raw"
    undistorted_dir = config_dir / "visualized_undistorted"
    cropped_dir = config_dir / "visualized_cropped"
    if not args.no_visualization:
        for directory in (raw_dir, undistorted_dir, cropped_dir):
            if directory.exists():
                shutil.rmtree(directory)
        for stale_video in (
            config_dir / "gaze_video_raw.mp4",
            config_dir / "gaze_video_undistorted.mp4",
            config_dir / "gaze_video_cropped.mp4",
        ):
            if stale_video.exists():
                stale_video.unlink()
    index = build_decoded_image_index(decoded_dir)
    records: list[dict[str, object]] = []
    valid_gaze_count = 0
    raw_projected_count = 0
    raw_inside_count = 0
    undistorted_projected_count = 0
    undistorted_inside_count = 0
    cropped_projected_count = 0
    cropped_inside_count = 0
    missing_image_count = 0
    visualized_counts = {"raw": 0, "undistorted": 0, "cropped": 0}

    limit = args.max_rows if args.max_rows and args.max_rows > 0 else len(rows)
    for row_index, row in enumerate(rows[:limit]):
        if parse_bool(row.get("gaze_valid")):
            valid_gaze_count += 1
        raw_projection, undistorted_projection, cropped_projection, diagnostics = project_row(
            row, config, rgb_local_pos, rgb_local_rot, k, d, new_k, crop_rect, args.include_invalid
        )
        raw_projected_count += 1 if raw_projection.projected else 0
        raw_inside_count += 1 if raw_projection.inside else 0
        undistorted_projected_count += 1 if undistorted_projection.projected else 0
        undistorted_inside_count += 1 if undistorted_projection.inside else 0
        cropped_projected_count += 1 if cropped_projection.projected else 0
        cropped_inside_count += 1 if cropped_projection.inside else 0

        image_path = resolve_image_path(row, row_index, decoded_dir, index)
        image_file = image_path.name if image_path is not None else ""
        if not args.no_visualization:
            if image_path is not None and image_path.exists():
                if "raw" in spaces and visualize_marker(
                    image_path, raw_dir / image_file, raw_projection, args.marker_diameter, args.marker_alpha
                ):
                    visualized_counts["raw"] += 1
                if "undistorted" in spaces and visualize_marker(
                    image_path,
                    undistorted_dir / image_file,
                    undistorted_projection,
                    args.marker_diameter,
                    args.marker_alpha,
                    undistort_maps,
                ):
                    visualized_counts["undistorted"] += 1
                if "cropped" in spaces and visualize_marker(
                    image_path,
                    cropped_dir / image_file,
                    cropped_projection,
                    args.marker_diameter,
                    args.marker_alpha,
                    undistort_maps,
                    crop_rect,
                ):
                    visualized_counts["cropped"] += 1
            else:
                missing_image_count += 1

        record = {
            "frame_index": parse_int(row, "frame_index", -1),
            "image_file": image_file,
            "gaze_valid": parse_bool(row.get("gaze_valid")),
            "xr_head_valid": parse_bool(row.get("xr_head_valid")),
            "projection_model": config.mode,
            "assumed_depth_m": "" if config.assumed_depth_m is None else format_float(config.assumed_depth_m),
            "uses_eye_origin": config.mode == "fixed-depth",
            "parallax_ignored": config.mode == "direction",
            **result_fields("raw", raw_projection),
            **result_fields("undistorted", undistorted_projection),
            **result_fields("cropped", cropped_projection, outside_text_for_outside=True),
            "crop_left": crop_rect[0],
            "crop_top": crop_rect[1],
            "crop_width": crop_rect[2],
            "crop_height": crop_rect[3],
            "point_camera_x": format_float(undistorted_projection.point_camera_x),
            "point_camera_y": format_float(undistorted_projection.point_camera_y),
            "point_camera_z": format_float(undistorted_projection.point_camera_z),
            **{key: format_float(value) for key, value in diagnostics.items()},
            "ref_timestamp_us": row.get("ref_timestamp_us", ""),
            "gaze_source": row.get("gaze_source", ""),
            "gaze_failure_reason": row.get("gaze_failure_reason", ""),
        }
        records.append(record)

    write_csv(config_dir / "gaze_uv.csv", records)

    videos = {}
    if not args.no_visualization and not args.no_video:
        if "raw" in spaces:
            path = config_dir / "gaze_video_raw.mp4"
            count, error = write_video_from_frames(raw_dir, records, path, args.video_fps)
            videos["raw"] = {"path": str(path), "frame_count": count, "error": error}
        if "undistorted" in spaces:
            path = config_dir / "gaze_video_undistorted.mp4"
            count, error = write_video_from_frames(undistorted_dir, records, path, args.video_fps)
            videos["undistorted"] = {"path": str(path), "frame_count": count, "error": error}
        if "cropped" in spaces:
            path = config_dir / "gaze_video_cropped.mp4"
            count, error = write_video_from_frames(cropped_dir, records, path, args.video_fps)
            videos["cropped"] = {"path": str(path), "frame_count": count, "error": error}

    summary = {
        "projection_name": config.name,
        "projection_model": config.mode,
        "assumed_depth_m": config.assumed_depth_m,
        "row_count": len(records),
        "valid_gaze_count": valid_gaze_count,
        "raw_projected_count": raw_projected_count,
        "raw_inside_count": raw_inside_count,
        "undistorted_projected_count": undistorted_projected_count,
        "undistorted_inside_count": undistorted_inside_count,
        "cropped_projected_count": cropped_projected_count,
        "cropped_inside_count": cropped_inside_count,
        "cropped_outside_count": cropped_projected_count - cropped_inside_count,
        "center_crop": {
            "left": crop_rect[0],
            "top": crop_rect[1],
            "width": crop_rect[2],
            "height": crop_rect[3],
        },
        "missing_image_count": missing_image_count,
        "visualized_image_counts": visualized_counts,
        "csv": str(config_dir / "gaze_uv.csv"),
        "visualized_raw_dir": str(raw_dir) if "raw" in spaces else "",
        "visualized_undistorted_dir": str(undistorted_dir) if "undistorted" in spaces else "",
        "visualized_cropped_dir": str(cropped_dir) if "cropped" in spaces else "",
        "videos": videos,
    }
    (config_dir / "projection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records, summary


def find_latest_session(sessions_dir: Path) -> Path:
    candidates = [path for path in sessions_dir.iterdir() if path.is_dir() and (path / "metadata.csv").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No session with metadata.csv found under {sessions_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_paths(args: argparse.Namespace):
    session_dir = args.session_dir or find_latest_session(SCRIPT_DIR / "sessions")
    session_dir = session_dir.resolve()
    metadata_path = (args.metadata or session_dir / "metadata.csv").resolve()
    camera_path = (args.camera or session_dir / "camera.json").resolve()
    decoded_dir = (args.decoded_dir or session_dir / "decoded_jpg").resolve()
    output_dir = (args.output_dir or session_dir / "gaze_projection").resolve()
    calibration_path = args.calibration.resolve()
    return session_dir, metadata_path, camera_path, decoded_dir, output_dir, calibration_path


def main() -> int:
    args = parse_args()
    require_cv2_numpy(load_cv2=not args.no_visualization)
    if args.marker_diameter <= 0:
        raise SystemExit("--marker-diameter must be positive")
    if not 0 <= args.marker_alpha <= 255:
        raise SystemExit("--marker-alpha must be in [0,255]")
    if args.video_fps <= 0:
        raise SystemExit("--video-fps must be positive")

    session_dir, metadata_path, camera_path, decoded_dir, output_dir, calibration_path = resolve_paths(args)
    for path in (metadata_path, camera_path, calibration_path):
        if not path.exists():
            raise SystemExit(f"required file not found: {path}")
    if not args.no_visualization and not decoded_dir.is_dir():
        raise SystemExit(f"decoded frame directory not found: {decoded_dir}. Run decode_camera_data.py first.")
    if not args.no_visualization:
        try:
            ensure_decoded_image_index(session_dir, metadata_path, decoded_dir)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

    rows = load_csv_rows(metadata_path)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit(f"No metadata rows found in {metadata_path}")
    camera = read_json(camera_path)
    k, d, new_k, calibration_image_size = load_calibration(calibration_path)
    first_width = parse_int(rows[0], "width")
    first_height = parse_int(rows[0], "height")
    if calibration_image_size != (0, 0) and calibration_image_size != (first_width, first_height):
        raise SystemExit(
            "Calibration image_size does not match metadata frame size: "
            f"calib={calibration_image_size}, metadata={(first_width, first_height)}"
        )

    spaces = [] if args.no_visualization else parse_spaces(args.spaces)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_local_pos, rgb_local_rot = rgb_local_pose_from_camera_json(camera)
    undistort_maps = None
    crop_size = parse_crop_size(args.crop_size)
    crop_rect = center_crop_rect((first_width, first_height), crop_size)

    if not args.no_visualization and any(space in spaces for space in ("undistorted", "cropped")):
        undistort_maps = build_undistort_maps(k, d, new_k, (first_width, first_height))

    summaries = []
    for config in projection_configs(args.projection_mode, args.depths):
        _records, summary = process_config(
            rows=rows,
            config=config,
            rgb_local_pos=rgb_local_pos,
            rgb_local_rot=rgb_local_rot,
            k=k,
            d=d,
            new_k=new_k,
            decoded_dir=decoded_dir,
            output_dir=output_dir,
            spaces=spaces,
            undistort_maps=undistort_maps,
            crop_rect=crop_rect,
            args=args,
        )
        summaries.append(summary)

    full_summary = {
        "session_dir": str(session_dir),
        "metadata_csv": str(metadata_path),
        "camera_json": str(camera_path),
        "decoded_dir": str(decoded_dir),
        "calibration_npz": str(calibration_path),
        "output_dir": str(output_dir),
        "raw_image_size": {"width": first_width, "height": first_height},
        "calibration_model": "opencv_fisheye",
        "fisheye_K": k.tolist(),
        "fisheye_D": d.tolist(),
        "undistorted_new_K": new_k.tolist(),
        "center_crop": {
            "left": crop_rect[0],
            "top": crop_rect[1],
            "width": crop_rect[2],
            "height": crop_rect[3],
        },
        "projection_runs": summaries,
        "tracking_alignment_model": "xr_head_plus_rgb_extrinsic",
        "rgb_local_pose_from_camera_json": {
            "position": rgb_local_pos.tolist(),
            "rotation": rgb_local_rot.tolist(),
        },
    }
    summary_path = output_dir / "projection_summary.json"
    summary_path.write_text(json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[project_gaze_uv] output: {output_dir}")
    print(f"[project_gaze_uv] summary: {summary_path}")
    for summary in summaries:
        print(
            "[project_gaze_uv] {projection_name}: cropped_inside={cropped_inside_count} "
            "undistorted_inside={undistorted_inside_count}".format(**summary)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
