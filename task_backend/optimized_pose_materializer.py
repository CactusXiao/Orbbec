#!/usr/bin/env python3
"""Materialize Publisher optimized poses into the episode MANO 3D artifact.

This module is intentionally executed with ORBBEC_MANO_PYTHON.  The collection
backend itself stays standard-library-only while this process imports numpy,
torch and the optimizer toolkit used by the original MANO conversion scripts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 1
CONVERTER_NAME = "optimized_pose_to_mano_v1"
EXPECTED_POSE_SHAPE = (2, 99)
EXPECTED_FRAME_JOINTS_SHAPE = (2, 21, 3)


class MaterializationError(RuntimeError):
    """A deterministic input/configuration error that should fail the job."""


class MaterializationNotReady(RuntimeError):
    """A result that is labeled but not yet visible on the NAS mount."""


def _load_mano_module(toolkit_root: Path) -> Any:
    toolkit_root = toolkit_root.expanduser().resolve()
    if not toolkit_root.is_dir():
        raise MaterializationError(f"MANO toolkit root is not a directory: {toolkit_root}")
    toolkit_text = str(toolkit_root)
    if toolkit_text not in sys.path:
        sys.path.insert(0, toolkit_text)

    mano_source = Path(__file__).resolve().parents[1] / "mano" / "mano(1).py"
    if not mano_source.is_file():
        raise MaterializationError(f"shared MANO conversion module not found: {mano_source}")
    spec = importlib.util.spec_from_file_location("orbbec_shared_mano", mano_source)
    if spec is None or spec.loader is None:
        raise MaterializationError(f"cannot load shared MANO conversion module: {mano_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_shape_and_scale(episode_dir: Path, default_shape_path: Path | None) -> Tuple[np.ndarray, np.ndarray, Path, float]:
    subject_dir = episode_dir.parents[1]
    subject_shape = subject_dir / "shape.npy"
    if subject_shape.is_file():
        shape_path = subject_shape
    elif default_shape_path is not None and default_shape_path.is_file():
        shape_path = default_shape_path
    else:
        expected = str(subject_shape)
        fallback = str(default_shape_path) if default_shape_path is not None else "not configured"
        raise MaterializationError(f"shape.npy not found: subject={expected}; default={fallback}")

    try:
        one_shape = np.load(shape_path, allow_pickle=False).astype(np.float32).reshape(10)
    except (OSError, EOFError) as exc:
        raise MaterializationNotReady(f"cannot read MANO shape file yet {shape_path}: {exc}") from exc
    except Exception as exc:
        raise MaterializationError(f"invalid MANO shape file {shape_path}: {exc}") from exc
    if not np.all(np.isfinite(one_shape)):
        raise MaterializationError(f"MANO shape contains non-finite values: {shape_path}")
    betas = one_shape.reshape(1, 10).repeat(2, axis=0)

    scale_path = subject_dir / "scale.npy"
    scale_value = 1.0
    if scale_path.is_file():
        try:
            scale_value = float(np.load(scale_path, allow_pickle=False).reshape(-1)[0])
        except (OSError, EOFError) as exc:
            raise MaterializationNotReady(f"cannot read MANO scale file yet {scale_path}: {exc}") from exc
        except Exception as exc:
            raise MaterializationError(f"invalid MANO scale file {scale_path}: {exc}") from exc
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise MaterializationError(f"MANO scale must be finite and positive, got {scale_value}: {scale_path}")
    scales = np.asarray([[scale_value], [scale_value]], dtype=np.float32)
    return betas, scales, shape_path.resolve(), scale_value


def discover_optimized_pose_files(optimized_pose_dir: Path) -> List[Tuple[int, Path]]:
    if not optimized_pose_dir.is_dir():
        raise MaterializationNotReady(f"optimized_pose directory is not visible yet: {optimized_pose_dir}")
    frames: List[Tuple[int, Path]] = []
    seen = set()
    try:
        for path in optimized_pose_dir.glob("*.npy"):
            if not path.stem.isdigit():
                continue
            frame = int(path.stem)
            if frame in seen:
                raise MaterializationError(f"duplicate optimized pose frame number {frame}: {optimized_pose_dir}")
            seen.add(frame)
            frames.append((frame, path))
    except OSError as exc:
        raise MaterializationNotReady(f"cannot list optimized_pose yet {optimized_pose_dir}: {exc}") from exc
    frames.sort(key=lambda item: item[0])
    if not frames:
        raise MaterializationNotReady(f"optimized_pose has no numeric .npy frames yet: {optimized_pose_dir}")
    return frames


def _validate_existing_artifact(
    output_dir: Path,
    *,
    generation: int,
    result_manifest_sha256: str,
) -> Dict[str, Any] | None:
    meta_path = output_dir / "mano_episode.json"
    joints_path = output_dir / "joints_3d.npy"
    if not meta_path.is_file() or not joints_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source = meta.get("source") if isinstance(meta, dict) else None
        if not isinstance(source, dict):
            return None
        if int(source.get("generation") or 0) != generation:
            return None
        if str(source.get("result_manifest_sha256") or "") != result_manifest_sha256:
            return None
        frames = [int(value) for value in meta.get("frames", [])]
        joints = np.load(joints_path, allow_pickle=False)
    except Exception:
        return None
    expected_shape = (len(frames), *EXPECTED_FRAME_JOINTS_SHAPE)
    if joints.dtype != np.float32 or joints.shape != expected_shape or not np.all(np.isfinite(joints)):
        return None
    return {
        "reused": True,
        "frames": frames,
        "cameras": [str(value) for value in meta.get("cameras", [])],
        "optimized_pose_shape": list(EXPECTED_POSE_SHAPE),
        "joints_3d_shape": list(joints.shape),
        "generation": generation,
        "result_manifest_sha256": result_manifest_sha256,
        "shape_source": str(meta.get("shape_source") or ""),
        "scale": float(meta.get("scale") or 1.0),
    }


def _atomic_write_npy(path: Path, value: np.ndarray) -> None:
    tmp_path = path.parent / f".{path.name}.tmp"
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    tmp_path = path.parent / f".{path.name}.tmp"
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def materialize(
    *,
    episode_dir: Path,
    toolkit_root: Path,
    mano_model_dir: Path,
    default_shape_path: Path | None,
    generation: int,
    result_manifest_sha256: str,
    cameras: Sequence[str],
) -> Dict[str, Any]:
    episode_dir = episode_dir.expanduser().resolve()
    if not episode_dir.is_dir():
        raise MaterializationNotReady(f"episode directory is not visible on NAS: {episode_dir}")
    if generation <= 0:
        raise MaterializationError(f"Publisher generation must be positive, got {generation}")
    if len(result_manifest_sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in result_manifest_sha256):
        raise MaterializationError("Publisher result_manifest_sha256 must be a 64-character hex digest")

    output_dir = episode_dir / "mano" / "episode"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MaterializationNotReady(f"cannot create MANO output directory yet {output_dir}: {exc}") from exc
    existing = _validate_existing_artifact(
        output_dir,
        generation=generation,
        result_manifest_sha256=result_manifest_sha256,
    )
    if existing is not None:
        return existing

    frame_files = discover_optimized_pose_files(episode_dir / "optimized_pose")
    betas, scales, shape_source, scale_value = _load_shape_and_scale(episode_dir, default_shape_path)
    mano = _load_mano_module(toolkit_root)
    mano_model_dir = mano_model_dir.expanduser().resolve()
    if not mano_model_dir.is_dir():
        raise MaterializationError(f"MANO model directory is not a directory: {mano_model_dir}")
    layers = mano.build_mano_layers(mano_model_dir)

    joints_by_frame: List[np.ndarray] = []
    frame_numbers: List[int] = []
    for frame, pose_path in frame_files:
        try:
            pose = mano.load_pose(pose_path)
            if tuple(pose.shape) != EXPECTED_POSE_SHAPE:
                raise ValueError(f"normalized pose has shape {pose.shape}")
            outputs = mano.mano_outputs_from_pose(pose, betas, scales, layers)
            frame_joints = np.stack(
                [outputs[hand]["joints"][0].detach().cpu().numpy() for hand in (0, 1)],
                axis=0,
            ).astype(np.float32, copy=False)
        except (OSError, EOFError) as exc:
            raise MaterializationNotReady(f"cannot read optimized pose yet {pose_path}: {exc}") from exc
        except Exception as exc:
            raise MaterializationError(f"failed to convert optimized pose {pose_path}: {exc}") from exc
        if frame_joints.shape != EXPECTED_FRAME_JOINTS_SHAPE:
            raise MaterializationError(f"MANO joints for frame {frame} have shape {frame_joints.shape}")
        if not np.all(np.isfinite(frame_joints)):
            raise MaterializationError(f"MANO joints for frame {frame} contain non-finite values")
        frame_numbers.append(frame)
        joints_by_frame.append(frame_joints)

    joints_3d = np.stack(joints_by_frame, axis=0).astype(np.float32, copy=False)
    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "orbbec_mano_3d_episode",
        "frames": frame_numbers,
        "cameras": [str(camera) for camera in cameras],
        "joints_3d_file": "joints_3d.npy",
        "coordinate_system": "episode_world",
        "hand_order": list(mano.MANO_HAND_ORDER),
        "joint_order": list(mano.SMPLX_MANO_JOINT_NAMES),
        "source": {
            "kind": "optimized_pose",
            "shape": list(EXPECTED_POSE_SHAPE),
            "generation": generation,
            "result_manifest_sha256": result_manifest_sha256,
        },
        "shape_source": str(shape_source),
        "scale": scale_value,
        "converter": CONVERTER_NAME,
    }

    joints_path = output_dir / "joints_3d.npy"
    meta_path = output_dir / "mano_episode.json"
    try:
        _atomic_write_npy(joints_path, joints_3d)
        _atomic_write_json(meta_path, metadata)  # Completion marker is always written last.
    except OSError as exc:
        raise MaterializationNotReady(f"cannot atomically write MANO artifact yet {output_dir}: {exc}") from exc

    # Reload both files after the atomic replacements.  A successful return is
    # therefore strong enough for the Bridge to complete the backend job.
    verified = _validate_existing_artifact(
        output_dir,
        generation=generation,
        result_manifest_sha256=result_manifest_sha256,
    )
    if verified is None:
        raise MaterializationNotReady(f"written MANO artifact is not readable yet: {output_dir}")
    verified["reused"] = False
    return verified


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Publisher optimized_pose to an Orbbec MANO episode artifact")
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--mano-model-dir", type=Path, required=True)
    parser.add_argument("--default-shape-path", type=Path)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--result-manifest-sha256", required=True)
    parser.add_argument("--cameras-json", default="[]")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cameras_value = json.loads(args.cameras_json)
        if not isinstance(cameras_value, list):
            raise MaterializationError("cameras-json must decode to a list")
        result = materialize(
            episode_dir=args.episode_dir,
            toolkit_root=args.toolkit_root,
            mano_model_dir=args.mano_model_dir,
            default_shape_path=args.default_shape_path,
            generation=args.generation,
            result_manifest_sha256=args.result_manifest_sha256,
            cameras=[str(value) for value in cameras_value],
        )
    except MaterializationNotReady as exc:
        print(json.dumps({"ok": False, "retryable": True, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 75
    except Exception as exc:
        print(json.dumps({"ok": False, "retryable": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
