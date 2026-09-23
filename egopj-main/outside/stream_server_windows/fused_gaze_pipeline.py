#!/usr/bin/env python3
"""One-command PICO session post-processing for training-ready gaze data.

Inputs are a captured PICO streaming session and an existing fisheye
calibration file. Camera calibration itself intentionally stays outside this
pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import decode_h265_to_jpg as decoder
import project_gaze_uv as gaze


SCRIPT_DIR = Path(__file__).resolve().parent
OUTSIDE_ROOT = SCRIPT_DIR.parent
DEFAULT_CALIBRATION = OUTSIDE_ROOT / "camera_info" / "fisheye_calibration_result.npz"
DEFAULT_CROP_SIZE = "1280x960"
DEFAULT_VIDEO_FPS = 30.0
DEFAULT_CRF = 18


@dataclass
class PipelineConfig:
    name: str
    depth_m: float
    output_dir: Path
    video_path: Path
    csv_path: Path
    visualized: bool


class H265Mp4Writer:
    def __init__(self, ffmpeg: str, output_path: Path, fps: float, crf: int):
        self.ffmpeg = ffmpeg
        self.output_path = output_path
        self.fps = fps
        self.crf = crf
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.width = 0
        self.height = 0
        self.frame_count = 0

    def write(self, image) -> None:
        if self.process is None:
            self._open(image)
        if image.shape[1] != self.width or image.shape[0] != self.height:
            raise ValueError(
                f"video frame size changed from {self.width}x{self.height} "
                f"to {image.shape[1]}x{image.shape[0]}"
            )
        assert self.process is not None
        assert self.process.stdin is not None
        self.process.stdin.write(image.tobytes())
        self.frame_count += 1

    def close(self) -> None:
        if self.process is None:
            return
        assert self.process.stdin is not None
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        return_code = self.process.wait()
        self.process = None
        if return_code != 0:
            raise RuntimeError(
                "ffmpeg H.265 MP4 encode failed.\n"
                f"Output: {self.output_path}\n"
                f"STDERR:\n{stderr}"
            )

    def _open(self, image) -> None:
        if self.fps <= 0:
            raise ValueError("--video-fps must be positive")
        self.height, self.width = image.shape[:2]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode one PICO streaming session, undistort and center-crop the fisheye "
            "camera frames, project fixed-depth gaze UVs, and write H.265 MP4 outputs."
        )
    )
    parser.add_argument("--session-dir", type=Path, required=True, help="Session directory containing video.h265.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION, help="OpenCV fisheye .npz file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to session-dir/fused_gaze.")
    parser.add_argument("--depth", type=float, default=None, help="Single fixed gaze depth in meters.")
    parser.add_argument("--depths", default="", help="Debug mode only: comma-separated fixed depths in meters.")
    parser.add_argument("--crop-size", default=DEFAULT_CROP_SIZE, help="Center crop size, for example 1280x960.")
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS, help="Output MP4 frame rate.")
    parser.add_argument("--h265-crf", type=int, default=DEFAULT_CRF, help="libx265 CRF. Lower is higher quality.")
    parser.add_argument("--ffmpeg", default="", help="ffmpeg path. Defaults to PATH ffmpeg, then imageio-ffmpeg.")
    parser.add_argument("--debug", action="store_true", help="Enable multi-depth visualized debug outputs.")
    parser.add_argument(
        "--debug-output-raw",
        action="store_true",
        help="Debug mode: keep raw decoded JPG frames without undistortion or crop.",
    )
    parser.add_argument(
        "--debug-output-undistorted",
        action="store_true",
        help="Debug mode: save full-size undistorted JPG frames before the center crop.",
    )
    parser.add_argument("--marker-diameter", type=int, default=20, help="Debug gaze marker diameter in pixels.")
    parser.add_argument("--marker-alpha", type=int, default=200, help="Debug gaze marker alpha in [0,255].")
    parser.add_argument("--include-invalid", action="store_true", help="Try projecting rows even when gaze_valid is false.")
    parser.add_argument("--decoded-dir", type=Path, default=None, help="Optional existing decoded JPG directory to reuse.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional quick-check row limit; 0 means all rows.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary decoded frames after non-debug runs.")
    return parser.parse_args()


def write_csv(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_depth_arguments(args: argparse.Namespace) -> list[float]:
    if args.debug:
        if args.depths.strip():
            depths = gaze.parse_depths(args.depths)
        elif args.depth is not None:
            depths = [args.depth]
        else:
            raise SystemExit("Debug mode requires --depths 0.6,1.0 or --depth 1.0.")
        return depths

    if args.depth is None:
        raise SystemExit("Non-debug mode requires exactly one --depth value, for example --depth 1.0.")
    if args.depths.strip():
        raise SystemExit("Non-debug mode accepts only one depth. Use --depth, or add --debug for --depths.")
    if args.depth <= 0:
        raise SystemExit("--depth must be positive")
    return [args.depth]


def safe_output_root(args: argparse.Namespace, session_dir: Path) -> Path:
    output_dir = (args.output_dir or session_dir / "fused_gaze").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_session_decoded_index(
    session_dir: Path,
    decoded_dir: Path,
) -> None:
    gaze.ensure_decoded_image_index(
        session_dir,
        session_dir / "metadata.csv",
        decoded_dir,
    )


def decode_session_to_images(
    args: argparse.Namespace,
    session_dir: Path,
    output_root: Path,
    ffmpeg: str,
) -> tuple[Path, bool]:
    video_path = session_dir / "video.h265"
    if args.decoded_dir is not None:
        decoded_dir = args.decoded_dir.resolve()
        if not decoded_dir.is_dir():
            raise FileNotFoundError(f"decoded JPG directory not found: {decoded_dir}")
        write_session_decoded_index(session_dir, decoded_dir)
        return decoded_dir, False

    if not video_path.is_file():
        raise FileNotFoundError(f"video.h265 not found: {video_path}")

    if args.debug and args.debug_output_raw:
        decoded_dir = output_root / "debug_intermediates" / "raw_decoded"
    else:
        decoded_dir = output_root / "_tmp_decoded_jpg"

    if decoded_dir.exists():
        shutil.rmtree(decoded_dir)
    decoded_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fused_gaze_pipeline] decoding camera stream -> {decoded_dir}")
    decoder.run_ffmpeg_decode_jpg(
        ffmpeg=ffmpeg,
        video_path=video_path,
        output_dir=decoded_dir,
        jpg_quality=2,
    )

    write_session_decoded_index(session_dir, decoded_dir)
    return decoded_dir, not (args.debug and args.debug_output_raw)


def load_session_inputs(session_dir: Path) -> tuple[list[dict[str, str]], dict]:
    metadata_path = session_dir / "metadata.csv"
    camera_path = session_dir / "camera.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")
    if not camera_path.is_file():
        raise FileNotFoundError(f"camera.json not found: {camera_path}")
    rows = gaze.load_csv_rows(metadata_path)
    if not rows:
        raise ValueError(f"metadata.csv has no rows: {metadata_path}")
    return rows, gaze.read_json(camera_path)


def first_decoded_frame(decoded_dir: Path):
    cv2, _np = gaze.require_cv2_numpy(load_cv2=True)
    for image_path in sorted(decoded_dir.glob("frame_*.jpg")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image_path, image
    raise FileNotFoundError(f"No readable frame_*.jpg files found under {decoded_dir}")


def ensure_row_image_size(row: dict[str, str], width: int, height: int) -> dict[str, str]:
    fixed = dict(row)
    if gaze.parse_int(fixed, "width", 0) <= 0:
        fixed["width"] = str(width)
    if gaze.parse_int(fixed, "height", 0) <= 0:
        fixed["height"] = str(height)
    return fixed


def final_coordinate_fields(projection: gaze.ProjectionResult) -> tuple[str, object, object, object, object, object]:
    if projection.projected and projection.inside:
        return (
            "inside",
            gaze.format_float(projection.pixel_x),
            gaze.format_float(projection.pixel_y),
            gaze.format_float(projection.uv_u),
            gaze.format_float(projection.uv_v_top),
            gaze.format_float(projection.uv_v_bottom),
        )
    if projection.projected and not projection.inside:
        return "outside", "outside", "outside", "outside", "outside", "outside"
    return projection.failure_reason or "projection_failed", "", "", "", "", ""


def draw_gaze_marker(image, projection: gaze.ProjectionResult, diameter: int, alpha_255: int):
    cv2, _np = gaze.require_cv2_numpy(load_cv2=True)
    if not projection.inside:
        return image
    alpha = max(0.0, min(1.0, alpha_255 / 255.0))
    radius = max(1, int(round(diameter / 2.0)))
    center = (int(round(projection.pixel_x)), int(round(projection.pixel_y)))
    overlay = image.copy()
    cv2.circle(overlay, center, radius, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)
    cv2.circle(image, center, radius, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    return image


def save_debug_undistorted_frames(
    rows: Sequence[dict[str, str]],
    decoded_dir: Path,
    output_root: Path,
    undistort_maps,
    max_rows: int,
    fallback_width: int,
    fallback_height: int,
) -> dict[str, object]:
    cv2, _np = gaze.require_cv2_numpy(load_cv2=True)
    output_dir = output_root / "debug_intermediates" / "undistorted_full"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = gaze.build_decoded_image_index(decoded_dir)
    limit = max_rows if max_rows > 0 else len(rows)
    written = 0
    missing = 0
    for row_index, raw_row in enumerate(rows[:limit]):
        row = ensure_row_image_size(raw_row, fallback_width, fallback_height)
        image_path = gaze.resolve_image_path(row, row_index, decoded_dir, index)
        if image_path is None:
            missing += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            missing += 1
            continue
        map1, map2 = undistort_maps
        undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if cv2.imwrite(str(output_dir / image_path.name), undistorted):
            written += 1
    return {"path": str(output_dir), "written_count": written, "missing_count": missing}


def make_pipeline_configs(depths: Sequence[float], output_root: Path, debug: bool) -> list[PipelineConfig]:
    configs: list[PipelineConfig] = []
    for depth in depths:
        name = f"fixed_depth_{gaze.safe_depth_label(depth)}m"
        output_dir = output_root / name
        video_name = "cropped_undistorted_gaze_h265.mp4" if debug else "cropped_undistorted_h265.mp4"
        configs.append(
            PipelineConfig(
                name=name,
                depth_m=depth,
                output_dir=output_dir,
                video_path=output_dir / video_name,
                csv_path=output_dir / "gaze_uv.csv",
                visualized=debug,
            )
        )
    return configs


def process_depth(
    config: PipelineConfig,
    args: argparse.Namespace,
    rows: Sequence[dict[str, str]],
    camera: dict,
    decoded_dir: Path,
    ffmpeg: str,
    k,
    d,
    new_k,
    undistort_maps,
    crop_rect: tuple[int, int, int, int],
    fallback_width: int,
    fallback_height: int,
) -> dict[str, object]:
    cv2, _np = gaze.require_cv2_numpy(load_cv2=True)
    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rgb_local_pos, rgb_local_rot = gaze.rgb_local_pose_from_camera_json(camera)
    projection_config = gaze.ProjectionConfig(config.name, "fixed-depth", config.depth_m)
    image_index = gaze.build_decoded_image_index(decoded_dir)
    writer = H265Mp4Writer(ffmpeg, config.video_path, args.video_fps, args.h265_crf)

    records: list[dict[str, object]] = []
    row_limit = args.max_rows if args.max_rows > 0 else len(rows)
    missing_images = 0
    inside_count = 0
    outside_count = 0
    failed_count = 0

    try:
        for metadata_row_index, raw_row in enumerate(rows[:row_limit]):
            row = ensure_row_image_size(raw_row, fallback_width, fallback_height)
            image_path = gaze.resolve_image_path(row, metadata_row_index, decoded_dir, image_index)
            if image_path is None:
                missing_images += 1
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                missing_images += 1
                continue

            raw_projection, undistorted_projection, cropped_projection, diagnostics = gaze.project_row(
                row,
                projection_config,
                rgb_local_pos,
                rgb_local_rot,
                k,
                d,
                new_k,
                crop_rect,
                args.include_invalid,
            )

            map1, map2 = undistort_maps
            undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            crop_left, crop_top, crop_width, crop_height = crop_rect
            cropped = undistorted[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width].copy()
            if config.visualized:
                cropped = draw_gaze_marker(cropped, cropped_projection, args.marker_diameter, args.marker_alpha)

            mp4_frame_index = writer.frame_count
            writer.write(cropped)

            status, pixel_x, pixel_y, uv_u, uv_v_top, uv_v_bottom = final_coordinate_fields(cropped_projection)
            if status == "inside":
                inside_count += 1
            elif status == "outside":
                outside_count += 1
            else:
                failed_count += 1

            records.append(
                {
                    "mp4_frame_index": mp4_frame_index,
                    "mp4_timestamp_s": f"{mp4_frame_index / args.video_fps:.10g}",
                    "image_file": image_path.name,
                    "metadata_row_index": metadata_row_index,
                    "source_frame_index": gaze.parse_int(row, "frame_index", -1),
                    "source_ref_timestamp_us": row.get("ref_timestamp_us", ""),
                    "source_pico_frame_timestamp_ns": row.get("frame_timestamp_ns", ""),
                    "gaze_valid": gaze.parse_bool(row.get("gaze_valid")),
                    "assumed_depth_m": gaze.format_float(config.depth_m),
                    "gaze_status": status,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "uv_image_u": uv_u,
                    "uv_image_v_top": uv_v_top,
                    "uv_unity_v_bottom": uv_v_bottom,
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                    "crop_left": crop_left,
                    "crop_top": crop_top,
                    "projection_failure_reason": cropped_projection.failure_reason,
                    "point_camera_x": gaze.format_float(undistorted_projection.point_camera_x),
                    "point_camera_y": gaze.format_float(undistorted_projection.point_camera_y),
                    "point_camera_z": gaze.format_float(undistorted_projection.point_camera_z),
                    **{key: gaze.format_float(value) for key, value in diagnostics.items()},
                }
            )
    finally:
        writer.close()

    write_csv(config.csv_path, records)
    summary = {
        "projection_name": config.name,
        "assumed_depth_m": config.depth_m,
        "debug_visualized": config.visualized,
        "row_count": len(records),
        "inside_count": inside_count,
        "outside_count": outside_count,
        "projection_failed_count": failed_count,
        "missing_image_count": missing_images,
        "video": {
            "path": str(config.video_path),
            "codec": "h265",
            "encoder": "libx265",
            "fps": args.video_fps,
            "crf": args.h265_crf,
            "frame_count": writer.frame_count,
            "width": crop_rect[2],
            "height": crop_rect[3],
        },
        "csv": str(config.csv_path),
    }
    write_json(config.output_dir / "process_summary.json", summary)
    return summary


def build_top_level_summary(
    args: argparse.Namespace,
    session_dir: Path,
    output_root: Path,
    decoded_dir: Path,
    cleanup_decoded: bool,
    depths: Sequence[float],
    crop_rect: tuple[int, int, int, int],
    depth_summaries: Sequence[dict[str, object]],
    debug_outputs: dict[str, object],
) -> dict[str, object]:
    return {
        "session_dir": str(session_dir),
        "calibration": str(args.calibration.resolve()),
        "output_dir": str(output_root),
        "debug": args.debug,
        "depths_m": list(depths),
        "crop_size": {"width": crop_rect[2], "height": crop_rect[3]},
        "crop_rect": {"left": crop_rect[0], "top": crop_rect[1], "width": crop_rect[2], "height": crop_rect[3]},
        "video_fps": args.video_fps,
        "h265_crf": args.h265_crf,
        "decoded_dir": str(decoded_dir),
        "decoded_dir_is_temporary": cleanup_decoded,
        "debug_outputs": debug_outputs,
        "runs": list(depth_summaries),
    }


def main() -> int:
    args = parse_args()
    if args.marker_diameter <= 0:
        raise SystemExit("--marker-diameter must be positive")
    if args.video_fps <= 0:
        raise SystemExit("--video-fps must be positive")
    if args.h265_crf < 0:
        raise SystemExit("--h265-crf must be non-negative")

    gaze.require_cv2_numpy(load_cv2=True)
    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        raise SystemExit(f"Session directory not found: {session_dir}")
    args.calibration = args.calibration.resolve()
    if not args.calibration.is_file():
        raise SystemExit(f"Calibration file not found: {args.calibration}")

    depths = parse_depth_arguments(args)
    output_root = safe_output_root(args, session_dir)
    ffmpeg = decoder.resolve_ffmpeg_executable(args.ffmpeg)
    rows, camera = load_session_inputs(session_dir)

    decoded_dir, cleanup_decoded = decode_session_to_images(args, session_dir, output_root, ffmpeg)
    _first_path, first_image = first_decoded_frame(decoded_dir)
    first_height, first_width = first_image.shape[:2]

    k, d, new_k, calibration_image_size = gaze.load_calibration(args.calibration)
    if calibration_image_size != (0, 0) and calibration_image_size != (first_width, first_height):
        print(
            "[fused_gaze_pipeline] WARNING: calibration image_size "
            f"{calibration_image_size[0]}x{calibration_image_size[1]} does not match decoded frames "
            f"{first_width}x{first_height}",
            file=sys.stderr,
        )
    crop_size = gaze.parse_crop_size(args.crop_size)
    crop_rect = gaze.center_crop_rect((first_width, first_height), crop_size)
    undistort_maps = gaze.build_undistort_maps(k, d, new_k, (first_width, first_height))

    debug_outputs: dict[str, object] = {}
    if args.debug and args.debug_output_raw:
        debug_outputs["raw_decoded"] = str(decoded_dir)
    if args.debug and args.debug_output_undistorted:
        debug_outputs["undistorted_full"] = save_debug_undistorted_frames(
            rows,
            decoded_dir,
            output_root,
            undistort_maps,
            args.max_rows,
            first_width,
            first_height,
        )

    depth_summaries = []
    for config in make_pipeline_configs(depths, output_root, args.debug):
        print(f"[fused_gaze_pipeline] processing {config.name} -> {config.output_dir}")
        depth_summaries.append(
            process_depth(
                config=config,
                args=args,
                rows=rows,
                camera=camera,
                decoded_dir=decoded_dir,
                ffmpeg=ffmpeg,
                k=k,
                d=d,
                new_k=new_k,
                undistort_maps=undistort_maps,
                crop_rect=crop_rect,
                fallback_width=first_width,
                fallback_height=first_height,
            )
        )

    summary = build_top_level_summary(
        args,
        session_dir,
        output_root,
        decoded_dir,
        cleanup_decoded,
        depths,
        crop_rect,
        depth_summaries,
        debug_outputs,
    )
    write_json(output_root / "fused_gaze_summary.json", summary)

    if cleanup_decoded and not args.keep_temp:
        shutil.rmtree(decoded_dir, ignore_errors=True)

    print(f"[fused_gaze_pipeline] output: {output_root}")
    for run in depth_summaries:
        print(f"[fused_gaze_pipeline] video: {run['video']['path']}")
        print(f"[fused_gaze_pipeline] csv: {run['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
