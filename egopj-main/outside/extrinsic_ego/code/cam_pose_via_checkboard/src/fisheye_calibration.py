from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .camera_geometry import prepare_pnp_image_points, reprojection_rmse


def _board_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    # 1xN multi-channel layout is accepted by both OpenCV 4 and OpenCV 5.
    points = np.zeros((1, cols * rows, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[0, :, :2] = grid * float(square_size_m)
    return points


def _detect(images: list[Path], cols: int, rows: int, square_size_m: float):
    object_template = _board_points(cols, rows, square_size_m)
    observations = []
    image_size = None
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(f"All calibration images must have the same resolution; {path} is {size}, expected {image_size}")
        found, corners = cv2.findChessboardCornersSB(
            image,
            (cols, rows),
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if found and corners is not None:
            observations.append((path, object_template.copy(), corners.reshape(1, -1, 2).astype(np.float64)))
    if image_size is None:
        raise RuntimeError("No readable calibration images")
    return observations, image_size


def _calibrate(observations, image_size):
    K = np.array(
        [[image_size[0] / 2.0, 0.0, image_size[0] / 2.0], [0.0, image_size[0] / 2.0, image_size[1] / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D = np.zeros((4, 1), dtype=np.float64)
    def calibration_flag(name: str) -> int:
        return int(getattr(cv2.fisheye, name, getattr(cv2, name)))

    flags = (
        calibration_flag("CALIB_RECOMPUTE_EXTRINSIC")
        | calibration_flag("CALIB_CHECK_COND")
        | calibration_flag("CALIB_FIX_SKEW")
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        [item[1] for item in observations],
        [item[2] for item in observations],
        image_size,
        K,
        D,
        flags=flags,
        criteria=criteria,
    )
    return float(rms), K, D.reshape(4)


def _validation_errors(observations, K, D):
    errors = []
    for _, object_points, corners in observations:
        pnp_points, pnp_K, pnp_dist = prepare_pnp_image_points(corners, K, D, "fisheye")
        ok, rvec, tvec = cv2.solvePnP(
            object_points.reshape(-1, 3).astype(np.float32),
            pnp_points,
            pnp_K,
            pnp_dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            errors.append(reprojection_rmse(object_points, corners, rvec, tvec, K, D, "fisheye"))
    return errors


def calibrate_fisheye(
    image_paths: list[Path],
    camera_id: str,
    cols: int,
    rows: int,
    square_size_m: float,
    validation_p95_limit_px: float = 2.0,
) -> dict:
    observations, image_size = _detect(image_paths, cols, rows, square_size_m)
    if len(observations) < 15:
        raise RuntimeError(f"Need at least 15 usable calibration views, got {len(observations)}")
    holdout = observations[::5]
    train = [item for idx, item in enumerate(observations) if idx % 5 != 0]
    _, K_train, D_train = _calibrate(train, image_size)
    holdout_errors = _validation_errors(holdout, K_train, D_train)
    if not holdout_errors:
        raise RuntimeError("Could not estimate validation poses")
    p95 = float(np.percentile(holdout_errors, 95))
    if p95 > validation_p95_limit_px:
        raise RuntimeError(
            f"Fisheye calibration rejected: held-out reprojection P95={p95:.3f}px exceeds {validation_p95_limit_px:.3f}px"
        )
    rms, K, D = _calibrate(observations, image_size)
    return {
        camera_id: {
            "camera_model": "fisheye",
            "RGB": {
                "intrinsic": {
                    "fx": float(K[0, 0]),
                    "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                    "width": image_size[0],
                    "height": image_size[1],
                },
                "distortion": {f"k{i + 1}": float(D[i]) for i in range(4)},
            },
            "calibration_quality": {
                "usable_views": len(observations),
                "calibration_rms_px": rms,
                "heldout_median_px": float(np.median(holdout_errors)),
                "heldout_p95_px": p95,
            },
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate an OpenCV fisheye camera from chessboard images")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing calibration images")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--cols", type=int, required=True, help="Inner chessboard corners per row")
    parser.add_argument("--rows", type=int, required=True, help="Inner chessboard corners per column")
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-p95-limit-px", type=float, default=2.0)
    args = parser.parse_args()
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    result = calibrate_fisheye(
        images,
        args.camera_id,
        args.cols,
        args.rows,
        args.square_size_m,
        args.validation_p95_limit_px,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote fisheye calibration: {args.output}")


if __name__ == "__main__":
    main()
