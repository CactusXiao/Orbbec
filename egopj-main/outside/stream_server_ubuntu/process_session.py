#!/usr/bin/env python3
"""Run camera decode and gaze UV projection for one PICO streaming session."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTSIDE_ROOT = SCRIPT_DIR.parent
DEFAULT_CALIBRATION = OUTSIDE_ROOT / "camera_info" / "fisheye_calibration_result.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode video and project gaze UVs for one captured session.")
    parser.add_argument("--session-dir", type=Path, required=True, help="Session directory containing video.h265.")
    parser.add_argument("--skip-decode", action="store_true", help="Reuse an existing decoded_jpg directory.")
    parser.add_argument("--skip-gaze", action="store_true", help="Skip gaze UV projection and visualization.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION, help="OpenCV fisheye .npz file.")
    parser.add_argument(
        "--projection-mode",
        choices=["direction", "fixed-depth", "both"],
        default="direction",
        help="Gaze projection model for project_gaze_uv.py.",
    )
    parser.add_argument(
        "--depths",
        default="0.6,0.8,1.0",
        help="Comma-separated depths in meters for fixed-depth projection.",
    )
    parser.add_argument("--crop-size", default="1280x960", help="Center crop size after undistortion.")
    parser.add_argument("--video-fps", type=float, default=30.0, help="FPS for generated MP4 outputs.")
    parser.add_argument("--video-codec", choices=["copy", "h264"], default="copy", help="Camera decode MP4 codec.")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit gaze rows for quick checks; 0 means all.")
    parser.add_argument("--include-invalid", action="store_true", help="Try projecting rows even when gaze_valid is false.")
    parser.add_argument("--no-video", action="store_true", help="Do not create MP4 outputs.")
    parser.add_argument("--no-visualization", action="store_true", help="Write UV CSV only.")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[process_session] " + " ".join(command))
    subprocess.check_call(command)


def main() -> int:
    args = parse_args()
    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        print(f"[process_session] ERROR: session directory not found: {session_dir}", file=sys.stderr)
        return 1

    if not args.skip_decode:
        decode_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "decode_camera_data.py"),
            "--session-dir",
            str(session_dir),
            "--video-fps",
            str(args.video_fps),
            "--video-codec",
            args.video_codec,
        ]
        if args.no_video:
            decode_cmd.append("--no-video")
        run(decode_cmd)

    if not args.skip_gaze:
        gaze_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "project_gaze_uv.py"),
            "--session-dir",
            str(session_dir),
            "--calibration",
            str(args.calibration),
            "--projection-mode",
            args.projection_mode,
            "--depths",
            args.depths,
            "--crop-size",
            args.crop_size,
            "--video-fps",
            str(args.video_fps),
        ]
        if args.max_rows > 0:
            gaze_cmd.extend(["--max-rows", str(args.max_rows)])
        if args.include_invalid:
            gaze_cmd.append("--include-invalid")
        if args.no_video:
            gaze_cmd.append("--no-video")
        if args.no_visualization:
            gaze_cmd.append("--no-visualization")
        run(gaze_cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
