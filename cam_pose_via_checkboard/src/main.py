from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .calib_io import load_calibrations
from .config import load_config
from .exporters import write_diagnostics_csv, write_frame_matrices, write_plots, write_trajectory_flat
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate camera-07 T_w_c from chessboard/AprilTag observations")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Dataset root path")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Config yaml path")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory (overrides config output_subdir)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    camera_ids = set(cfg.fixed_camera_ids)
    camera_ids.add(cfg.target_camera_id)

    calib_json, calibrations = load_calibrations(
        dataset_root=args.dataset_root,
        camera_ids=sorted(camera_ids),
        required_extrinsics_ids=cfg.fixed_camera_ids,
        calibration_filename=cfg.calibration_filename,
        extrinsics_filename=cfg.extrinsics_filename,
        use_mm_to_m_auto_scale=cfg.use_mm_to_m_auto_scale,
        fixed_extrinsics_are_twc=cfg.fixed_extrinsics_are_twc,
    )
    logging.getLogger("main").info("Using calibration file: %s", calib_json)

    rows = run_pipeline(args.dataset_root, cfg, calibrations)

    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        output_root = Path(cfg.output_subdir)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root

    write_trajectory_flat(output_root / "trajectory_tw_c07.txt", rows)
    write_frame_matrices(output_root / "matrices", rows)
    write_diagnostics_csv(output_root / "diagnostics.csv", rows)
    write_plots(output_root / "plots", rows)

    total = len(rows)
    ok = sum(int(r["success"]) for r in rows)
    logging.getLogger("main").info("Done. success=%d total=%d output=%s", ok, total, output_root)


if __name__ == "__main__":
    main()
