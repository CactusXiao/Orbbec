from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class RuntimeConfig:
    target_type: str = "chessboard"  # chessboard | apriltag
    board_cols: int = 9
    board_rows: int = 6
    square_size_m: float = 0.0255
    apriltag_family: str = "tag36h11"  # tag16h5 | tag25h9 | tag36h10 | tag36h11
    apriltag_min_tags: int = 1
    apriltag_min_inliers: int = 4
    apriltag_default_size_m: float = 0.10
    fixed_camera_ids: tuple[str, ...] = ("00", "01", "02", "03", "04", "05")
    target_camera_id: str = "07"
    frame_policy: str = "intersection"  # intersection | target_primary
    use_mm_to_m_auto_scale: bool = True
    fixed_extrinsics_are_twc: bool = True
    pnp_reproj_error_px: float = 2.0
    pnp_iterations: int = 200
    min_inliers: int = 12
    min_fixed_observations: int = 2
    fusion_ransac_trans_thresh_m: float = 0.08
    fusion_ransac_rot_thresh_deg: float = 8.0
    output_subdir: str = "outputs"
    calibration_filename: str = "camera_params.json"
    extrinsics_filename: str = "extrinsics.json"


def _merge_default(default: RuntimeConfig, data: Dict[str, Any]) -> RuntimeConfig:
    values = default.__dict__.copy()
    for key, value in data.items():
        if key not in values:
            continue
        values[key] = value
    if isinstance(values["fixed_camera_ids"], list):
        values["fixed_camera_ids"] = tuple(values["fixed_camera_ids"])
    return RuntimeConfig(**values)


def load_config(config_path: Path) -> RuntimeConfig:
    if not config_path.exists():
        return RuntimeConfig()
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must be a mapping")
    return _merge_default(RuntimeConfig(), data)
