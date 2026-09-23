#!/usr/bin/env python3
"""Calibrate a PICO fisheye/VST camera with OpenCV's fisheye model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def require_cv2_numpy():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "calibrate_fisheye_camera.py requires numpy and opencv-python/opencv-python-headless. "
            "Install the server environment first.\n"
            f"Import error: {exc}"
        ) from exc
    return cv2, np


def parse_pattern(value: str) -> tuple[int, int]:
    text = value.lower().replace(",", "x").replace("*", "x")
    parts = [part.strip() for part in text.split("x") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pattern must look like 11x8")
    cols, rows = int(parts[0]), int(parts[1])
    if cols <= 0 or rows <= 0:
        raise argparse.ArgumentTypeError("pattern values must be positive")
    return cols, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate an OpenCV fisheye camera from checkerboard images.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing checkerboard images.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to outside/camera_info.")
    parser.add_argument("--pattern", type=parse_pattern, default=parse_pattern("11x8"), help="Inner corners, e.g. 11x8.")
    parser.add_argument("--square-size", type=float, default=0.03, help="Checkerboard square size in meters.")
    parser.add_argument("--balance", type=float, default=1.0, help="Undistort balance, 0 cropped to 1 max FOV.")
    parser.add_argument("--fov-scale", type=float, default=1.0, help="OpenCV fisheye FOV scale.")
    parser.add_argument("--max-mean-error", type=float, default=0.0, help="Optional per-image error filter in pixels.")
    parser.add_argument("--skip-check-cond", action="store_true", help="Retry without CALIB_CHECK_COND from the start.")
    parser.add_argument("--glob", default="*.jpg;*.jpeg;*.png;*.bmp", help="Semicolon-separated image globs.")
    return parser.parse_args()


def list_images(image_dir: Path, glob_text: str) -> list[Path]:
    images: list[Path] = []
    for pattern in [item.strip() for item in glob_text.split(";") if item.strip()]:
        images.extend(image_dir.glob(pattern))
    return sorted(set(path for path in images if path.is_file()))


def build_object_points(np, pattern_size: tuple[int, int], square_size: float):
    cols, rows = pattern_size
    objp = np.zeros((1, cols * rows, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def find_corners(cv2, np, gray, pattern_size: tuple[int, int]):
    try:
        flags_sb = 0
        flags_sb |= getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0)
        flags_sb |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        flags_sb |= getattr(cv2, "CALIB_CB_ACCURACY", 0)
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags_sb)
        if ok and corners is not None:
            return True, corners.astype(np.float64).reshape(-1, 1, 2), "findChessboardCornersSB"
    except Exception:
        pass

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok or corners is None:
        return False, None, "failed"

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners.astype(np.float64).reshape(-1, 1, 2), "findChessboardCorners"


def calibrate(cv2, np, objpoints, imgpoints, image_size: tuple[int, int], use_check_cond: bool):
    k = np.zeros((3, 3), dtype=np.float64)
    d = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    if use_check_cond:
        flags |= cv2.fisheye.CALIB_CHECK_COND
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, image_size, k, d, rvecs, tvecs, flags=flags, criteria=criteria
    )
    return float(rms), k, d, rvecs, tvecs, int(flags)


def reprojection_errors(cv2, np, objpoints, imgpoints, rvecs, tvecs, k, d):
    per_image = []
    total_sq = 0.0
    total_points = 0
    for objp, observed, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, k, d)
        diff = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        sq = float(np.sum(diff ** 2))
        count = int(diff.shape[0])
        per_image.append(math.sqrt(sq / count))
        total_sq += sq
        total_points += count
    return math.sqrt(total_sq / total_points), per_image


def write_yaml(path: Path, k, d, new_k, image_size: tuple[int, int], rms: float) -> None:
    content = [
        "calibration_model: opencv_fisheye",
        f"image_width: {image_size[0]}",
        f"image_height: {image_size[1]}",
        f"rms_reprojection_error: {rms:.10g}",
        "K:",
    ]
    content.extend("  - [" + ", ".join(f"{float(v):.12g}" for v in row) + "]" for row in k.tolist())
    content.append("D: [" + ", ".join(f"{float(v):.12g}" for v in d.reshape(-1).tolist()) + "]")
    content.append("new_K:")
    content.extend("  - [" + ", ".join(f"{float(v):.12g}" for v in row) + "]" for row in new_k.tolist())
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cv2, np = require_cv2_numpy()
    if args.square_size <= 0:
        raise SystemExit("--square-size must be positive")

    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"image directory not found: {image_dir}")
    output_dir = (args.output_dir or Path(__file__).resolve().parent.parent / "camera_info").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    corners_dir = output_dir / "corners_debug"
    undistorted_dir = output_dir / "undistorted"
    corners_dir.mkdir(parents=True, exist_ok=True)
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir, args.glob)
    if not images:
        raise SystemExit(f"no calibration images found under {image_dir}")

    obj_template = build_object_points(np, args.pattern, args.square_size)
    objpoints = []
    imgpoints = []
    used_images = []
    rejected = []
    image_size = None

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"image": str(image_path), "reason": "unreadable"})
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (int(gray.shape[1]), int(gray.shape[0]))
        if image_size is None:
            image_size = size
        elif size != image_size:
            rejected.append({"image": str(image_path), "reason": f"size_mismatch_{size}"})
            continue
        ok, corners, method = find_corners(cv2, np, gray, args.pattern)
        if not ok or corners is None:
            rejected.append({"image": str(image_path), "reason": "checkerboard_not_found"})
            continue
        debug = image.copy()
        cv2.drawChessboardCorners(debug, args.pattern, corners, ok)
        cv2.imwrite(str(corners_dir / image_path.name), debug)
        objpoints.append(obj_template.copy())
        imgpoints.append(corners)
        used_images.append({"image": str(image_path), "detector": method})

    if image_size is None or len(objpoints) < 3:
        raise SystemExit(f"need at least 3 valid checkerboard images, got {len(objpoints)}")

    try:
        rms, k, d, rvecs, tvecs, flags = calibrate(cv2, np, objpoints, imgpoints, image_size, not args.skip_check_cond)
    except cv2.error:
        rms, k, d, rvecs, tvecs, flags = calibrate(cv2, np, objpoints, imgpoints, image_size, False)

    overall_error, per_image_errors = reprojection_errors(cv2, np, objpoints, imgpoints, rvecs, tvecs, k, d)

    if args.max_mean_error and args.max_mean_error > 0:
        kept = [i for i, error in enumerate(per_image_errors) if error <= args.max_mean_error]
        if len(kept) >= 3 and len(kept) < len(objpoints):
            objpoints = [objpoints[i] for i in kept]
            imgpoints = [imgpoints[i] for i in kept]
            used_images = [used_images[i] for i in kept]
            rms, k, d, rvecs, tvecs, flags = calibrate(cv2, np, objpoints, imgpoints, image_size, False)
            overall_error, per_image_errors = reprojection_errors(cv2, np, objpoints, imgpoints, rvecs, tvecs, k, d)

    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        k, d, image_size, np.eye(3, dtype=np.float64), balance=args.balance, fov_scale=args.fov_scale
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        k, d, np.eye(3, dtype=np.float64), new_k, image_size, cv2.CV_16SC2
    )
    for image_path in images[: min(len(images), 20)]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None and (image.shape[1], image.shape[0]) == image_size:
            cv2.imwrite(str(undistorted_dir / image_path.name), cv2.remap(image, map1, map2, cv2.INTER_LINEAR))

    np.savez(
        output_dir / "fisheye_calibration_result.npz",
        K=k,
        D=d.reshape(4),
        new_K=new_k,
        image_size=np.array(image_size, dtype=np.int64),
        rms=np.array([rms], dtype=np.float64),
        reprojection_error=np.array([overall_error], dtype=np.float64),
    )
    write_yaml(output_dir / "fisheye_calibration_result.yaml", k, d, new_k, image_size, rms)

    summary = {
        "calibration_model": "opencv_fisheye",
        "image_dir": str(image_dir),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "pattern": {"cols": args.pattern[0], "rows": args.pattern[1]},
        "square_size_m": args.square_size,
        "used_image_count": len(used_images),
        "rejected_image_count": len(rejected),
        "rms": rms,
        "overall_reprojection_error_px": overall_error,
        "per_image_reprojection_error_px": per_image_errors,
        "balance": args.balance,
        "fov_scale": args.fov_scale,
        "flags": flags,
        "K": k.tolist(),
        "D": d.reshape(4).tolist(),
        "new_K": new_k.tolist(),
        "used_images": used_images,
        "rejected_images": rejected,
    }
    (output_dir / "fisheye_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[calibrate_fisheye_camera] output: {output_dir}")
    print(f"[calibrate_fisheye_camera] valid images: {len(used_images)}")
    print(f"[calibrate_fisheye_camera] rms: {rms:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
