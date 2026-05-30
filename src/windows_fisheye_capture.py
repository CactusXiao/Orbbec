#!/usr/bin/env python3
"""
Windows 单相机实时预览脚本，不保存数据。

依赖：
    pip install opencv-python

用法：
    python src\windows_fisheye_capture.py --list
    python src\windows_fisheye_capture.py --index 0
    python src\windows_fisheye_capture.py --index 0 --backend msmf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 60
DEFAULT_PROBE_MAX_INDEX = 10
BLACK_MEAN_THRESHOLD = 5.0
BLACK_STD_THRESHOLD = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Windows 单相机实时预览")
    parser.add_argument("--list", action="store_true", help="探测并打印可用相机索引")
    parser.add_argument("--index", type=int, default=None, help="要预览的相机索引")
    parser.add_argument("--backend", choices=["dshow", "msmf", "auto"], default="dshow",
                        help="视频后端，默认 dshow")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="请求宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="请求高度")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="请求帧率")
    parser.add_argument("--probe-max-index", type=int, default=DEFAULT_PROBE_MAX_INDEX,
                        help="--list 时探测范围，默认 0..9")
    parser.add_argument("--convert-rgb", action="store_true",
                        help="显式启用 OpenCV RGB 转换")
    parser.add_argument("--fourcc", choices=["auto", "mjpg", "yuy2", "raw"], default="auto",
                        help="请求的视频编码格式，默认 auto")
    return parser.parse_args()


def backend_flag(name: str) -> int:
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def fourcc_to_text(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "0x00000000"
    chars = [chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
    text = "".join(ch if 32 <= ord(ch) <= 126 else "." for ch in chars)
    return f"{text} (0x{code:08x})"


def requested_fourcc_code(name: str) -> Optional[int]:
    if name == "mjpg":
        return cv2.VideoWriter.fourcc("M", "J", "P", "G")
    if name == "yuy2":
        return cv2.VideoWriter.fourcc("Y", "U", "Y", "2")
    return None


def powershell_list_cameras() -> List[str]:
    if sys.platform != "win32":
        return []

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.PNPClass -eq 'Camera' -or $_.Service -eq 'usbvideo' } | "
            "Select-Object -ExpandProperty Name"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


@dataclass
class StreamProfile:
    backend: str
    fourcc: str
    width: int
    height: int
    fps: int


def build_profiles(args: argparse.Namespace) -> List[StreamProfile]:
    if args.fourcc != "auto":
        return [StreamProfile(args.backend, args.fourcc, args.width, args.height, args.fps)]

    profiles: List[StreamProfile] = [
        StreamProfile(args.backend, "mjpg", args.width, args.height, args.fps),
        StreamProfile(args.backend, "yuy2", args.width, args.height, min(args.fps, 30)),
        StreamProfile(args.backend, "raw", args.width, args.height, min(args.fps, 30)),
    ]

    alt_backend = "msmf" if args.backend == "dshow" else "dshow"
    if args.backend != "auto":
        profiles.extend([
            StreamProfile(alt_backend, "mjpg", args.width, args.height, min(args.fps, 30)),
            StreamProfile(alt_backend, "yuy2", args.width, args.height, min(args.fps, 30)),
        ])
    return profiles


def open_camera(index: int,
                backend_name: str,
                width: int,
                height: int,
                fps: int,
                convert_rgb: bool,
                fourcc_name: str = "mjpg") -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, backend_flag(backend_name))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera index {index} with backend {backend_name}")

    if convert_rgb:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    requested_fourcc = requested_fourcc_code(fourcc_name)
    if requested_fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, requested_fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def frame_mean_std(frame) -> tuple[float, float]:
    mean_scalar, std_scalar = cv2.meanStdDev(frame)
    mean_value = float(mean_scalar.mean())
    std_value = float(std_scalar.mean())
    return mean_value, std_value


def describe_camera(cap: cv2.VideoCapture, requested: StreamProfile, index: int) -> None:
    backend_name = "unknown"
    if hasattr(cap, "getBackendName"):
        try:
            backend_name = cap.getBackendName()
        except Exception:
            backend_name = "unknown"

    print("camera info:")
    print(f"  index           : {index}")
    print(f"  requested       : backend={requested.backend} fourcc={requested.fourcc} "
          f"{requested.width}x{requested.height} fps={requested.fps}")
    print(f"  backend actual  : {backend_name}")
    print(f"  size actual     : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  fps actual      : {cap.get(cv2.CAP_PROP_FPS):.2f}")
    print(f"  fourcc actual   : {fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))}")
    print(f"  convert rgb     : {cap.get(cv2.CAP_PROP_CONVERT_RGB):.0f}")


def looks_black(frame) -> bool:
    mean_value, std_value = frame_mean_std(frame)
    return mean_value < BLACK_MEAN_THRESHOLD and std_value < BLACK_STD_THRESHOLD


def try_open_profile(index: int, profile: StreamProfile, convert_rgb: bool) -> tuple[cv2.VideoCapture, bool]:
    cap = open_camera(
        index=index,
        backend_name=profile.backend,
        width=profile.width,
        height=profile.height,
        fps=profile.fps,
        convert_rgb=convert_rgb,
        fourcc_name=profile.fourcc,
    )

    good_frames = 0
    black_frames = 0
    for _ in range(30):
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            time.sleep(0.02)
            continue
        if looks_black(frame):
            black_frames += 1
        else:
            good_frames += 1
        if good_frames >= 3:
            return cap, True

    print(f"profile test: backend={profile.backend} fourcc={profile.fourcc} "
          f"good_frames={good_frames} black_frames={black_frames}")
    if black_frames > 0 and good_frames == 0:
        return cap, False
    return cap, good_frames > 0


def select_profile(args: argparse.Namespace) -> tuple[cv2.VideoCapture, StreamProfile]:
    last_error: Optional[Exception] = None
    for profile in build_profiles(args):
        print(f"trying profile: backend={profile.backend} fourcc={profile.fourcc} "
              f"{profile.width}x{profile.height}@{profile.fps}")
        try:
            cap, usable = try_open_profile(args.index, profile, args.convert_rgb)
        except Exception as exc:
            last_error = exc
            print(f"  open failed: {exc}")
            continue

        if usable:
            return cap, profile

        print("  profile opened but frames look black/unusable, trying next profile")
        cap.release()

    if last_error is not None:
        raise RuntimeError(f"no usable profile found, last error: {last_error}")
    raise RuntimeError("no usable profile found")


def probe_mode(args: argparse.Namespace) -> int:
    print("=== Windows PnP camera names ===")
    names = powershell_list_cameras()
    if names:
        for idx, name in enumerate(names):
            print(f"[{idx}] {name}")
    else:
        print("PnP camera names unavailable or not running on Windows.")

    print("\n=== OpenCV probe results ===")
    for index in range(args.probe_max_index):
        try:
            cap = open_camera(index, args.backend, args.width, args.height, args.fps, args.convert_rgb, args.fourcc)
        except Exception:
            print(f"index={index:<2d} opened=no")
            continue

        try:
            ok, frame = cap.read()
            mean_value = -1.0
            std_value = -1.0
            if ok and frame is not None and frame.size != 0:
                mean_value, std_value = frame_mean_std(frame)
            print(
                f"index={index:<2d} opened=yes "
                f"first_frame={'yes' if ok and frame is not None else 'no '} "
                f"size={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                f"fps={cap.get(cv2.CAP_PROP_FPS):.2f} "
                f"fourcc={fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))} "
                f"mean={mean_value:.2f} std={std_value:.2f}"
            )
        finally:
            cap.release()
    return 0


def preview_mode(args: argparse.Namespace) -> int:
    if args.index is None:
        raise SystemExit("preview mode requires --index")

    cap, profile = select_profile(args)
    window_name = f"camera_{args.index}"
    frame_count = 0
    last_report = time.time()
    last_black_report = 0.0

    print(f"preview index : {args.index}")
    describe_camera(cap, profile, args.index)
    print("press q or ESC to exit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                print("frame read failed")
                time.sleep(0.01)
                continue

            frame_count += 1
            now = time.time()
            if now - last_report >= 1.0:
                actual_fps = frame_count / (now - last_report)
                mean_value, std_value = frame_mean_std(frame)
                print(f"streaming ok, approx_fps={actual_fps:.2f}, mean={mean_value:.2f}, std={std_value:.2f}",
                      flush=True)
                frame_count = 0
                last_report = now

            if looks_black(frame) and now - last_black_report >= 1.0:
                print("warning: frame looks almost black; try --backend msmf or --fourcc yuy2", flush=True)
                last_black_report = now

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    if args.list:
        return probe_mode(args)
    return preview_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
