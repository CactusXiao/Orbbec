from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from .se3 import invert_transform, is_valid_rotation, make_transform


@dataclass
class CameraCalibration:
    camera_id: str
    K: np.ndarray
    dist: np.ndarray
    T_w_c: np.ndarray
    image_size: tuple[int, int]


def _find_calib_json(dataset_root: Path, preferred_name: str) -> Path:
    candidate = dataset_root / preferred_name
    if candidate.exists():
        return candidate

    matches = []
    matches.extend(sorted(dataset_root.glob("*camera*params*.json")))
    matches.extend(sorted(dataset_root.glob("*camera*parms*.json")))
    # Remove obvious extrinsic files if the wildcard accidentally matches them.
    matches = [p for p in matches if "extrinsic" not in p.name.lower()]
    # De-duplicate while preserving order.
    seen = set()
    dedup = []
    for p in matches:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            dedup.append(p)
    matches = dedup

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = "\n".join(str(p) for p in matches)
        raise FileNotFoundError(f"Multiple candidate calibration json files found:\n{names}")
    raise FileNotFoundError(f"No calibration json found under {dataset_root}")


def _find_extrinsics_json(dataset_root: Path, preferred_name: str) -> Path | None:
    candidate = dataset_root / preferred_name
    if candidate.exists():
        return candidate

    # Also try common singular/plural names directly.
    common = [dataset_root / "extrinsic.json", dataset_root / "extrinsics.json"]
    for p in common:
        if p.exists():
            return p

    matches = sorted(dataset_root.glob("*extrinsic*.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = "\n".join(str(p) for p in matches)
        raise FileNotFoundError(f"Multiple candidate extrinsics json files found:\n{names}")
    return None


def _extract_intrinsics(entry: dict, camera_id: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    try:
        intr = entry["RGB"]["intrinsic"]
        dist = entry["RGB"]["distortion"]
    except KeyError as exc:
        raise KeyError(f"Missing RGB intrinsic/distortion in camera {camera_id}") from exc

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    width = int(intr.get("width", entry["RGB"].get("width", 0)))
    height = int(intr.get("height", entry["RGB"].get("height", 0)))

    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist_vec = np.array(
        [
            float(dist.get("k1", 0.0)),
            float(dist.get("k2", 0.0)),
            float(dist.get("p1", 0.0)),
            float(dist.get("p2", 0.0)),
            float(dist.get("k3", 0.0)),
        ],
        dtype=np.float64,
    )
    return K, dist_vec, (width, height)


def _extract_twc(
    entry: dict,
    camera_id: str,
    use_mm_to_m_auto_scale: bool,
    extrinsics_are_twc: bool,
    extr_entry: dict | None = None,
) -> np.ndarray:
    source = entry
    if "rotation" not in source or "translation" not in source:
        if extr_entry is not None and "rotation" in extr_entry and "translation" in extr_entry:
            source = extr_entry
        else:
            raise KeyError(
                f"Missing rotation/translation for camera {camera_id}. "
                "Please provide extrinsic.json/extrinsics.json with per-camera rotation/translation."
            )

    R = np.array(source["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.array(source["translation"], dtype=np.float64).reshape(3)

    if use_mm_to_m_auto_scale and np.linalg.norm(t) > 10.0:
        t = t / 1000.0

    if not is_valid_rotation(R):
        raise ValueError(f"Invalid rotation matrix in camera {camera_id}")

    T = make_transform(R, t)
    if extrinsics_are_twc:
        return T
    return invert_transform(T)


def _resolve_camera_entry(data: dict | None, cam_id: str, strict: bool = True) -> dict | None:
    """Resolve camera key robustly for ids like 07/7."""
    if data is None:
        if strict:
            raise KeyError(f"Camera id {cam_id} not found in calibration json")
        return None

    if cam_id in data:
        return data[cam_id]

    try:
        cam_num = int(cam_id)
    except ValueError:
        cam_num = None

    candidates = []
    if cam_num is not None:
        candidates.extend([str(cam_num), f"{cam_num:02d}"])

    for key in candidates:
        if key in data:
            return data[key]

    if strict:
        raise KeyError(f"Camera id {cam_id} not found in calibration json")
    return None


def load_calibrations(
    dataset_root: Path,
    camera_ids: Iterable[str],
    required_extrinsics_ids: Iterable[str],
    calibration_filename: str,
    extrinsics_filename: str,
    use_mm_to_m_auto_scale: bool,
    fixed_extrinsics_are_twc: bool,
) -> tuple[Path, Dict[str, CameraCalibration]]:
    calib_json_path = _find_calib_json(dataset_root, calibration_filename)
    with calib_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    extr_json_path = _find_extrinsics_json(dataset_root, extrinsics_filename)
    extr_data = None
    if extr_json_path is not None:
        with extr_json_path.open("r", encoding="utf-8") as f:
            extr_data = json.load(f)

    out: Dict[str, CameraCalibration] = {}
    required_extrinsics_ids = set(required_extrinsics_ids)
    for cam_id in camera_ids:
        entry = _resolve_camera_entry(data, cam_id)
        extr_entry = _resolve_camera_entry(extr_data, cam_id, strict=False)

        entry_has_rt = "rotation" in entry and "translation" in entry
        extr_has_rt = extr_entry is not None and "rotation" in extr_entry and "translation" in extr_entry

        if cam_id in required_extrinsics_ids and not entry_has_rt and not extr_has_rt:
            raise KeyError(
                f"Camera id {cam_id} is required in extrinsics but not found. "
                "Please ensure extrinsic.json/extrinsics.json contains this camera."
            )

        K, dist, size = _extract_intrinsics(entry, cam_id)
        if entry_has_rt or extr_has_rt:
            T_w_c = _extract_twc(
                entry,
                cam_id,
                use_mm_to_m_auto_scale,
                fixed_extrinsics_are_twc,
                extr_entry=extr_entry,
            )
        else:
            # For target camera without known world extrinsics, use identity placeholder.
            T_w_c = np.eye(4, dtype=np.float64)

        out[cam_id] = CameraCalibration(camera_id=cam_id, K=K, dist=dist, T_w_c=T_w_c, image_size=size)
    return calib_json_path, out
