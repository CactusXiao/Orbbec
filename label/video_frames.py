from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote

try:
    from .storage import CorrectionTask, find_frame_path
except Exception:
    from storage import CorrectionTask, find_frame_path


def ensure_decoded_rgb_frames(
    task: CorrectionTask,
    payload: Optional[Mapping[str, Any]] = None,
    *,
    cache_root: Optional[Path] = None,
    ffmpeg_executable: str = "ffmpeg",
) -> CorrectionTask:
    """Decode the RGB H265 episode videos needed by a backend label segment."""
    payload = payload or {}
    episode_dir = task.episode_dir()
    cameras_to_decode = [
        cam for cam in task.cameras
        if any(find_frame_path(episode_dir, cam, frame, task.rgb_path_template) is None for frame in task.frames)
    ]
    if not cameras_to_decode:
        return task

    cache_base = _cache_base_dir(cache_root)
    episode_cache = cache_base / _episode_cache_key(episode_dir, payload)
    decoded_any = False
    failures: List[str] = []
    for cam in cameras_to_decode:
        try:
            video_path, timestamp_path = _locate_rgb_video(episode_dir, cam, payload)
            frame_map = _load_frame_map(timestamp_path)
            _decode_camera_frames(
                video_path=video_path,
                timestamp_path=timestamp_path,
                frame_map=frame_map,
                camera=cam,
                frames=task.frames,
                out_dir=episode_cache / cam,
                ffmpeg_executable=ffmpeg_executable,
            )
            decoded_any = True
        except Exception as exc:
            failures.append(f"{cam}: {exc}")

    if failures:
        joined = "; ".join(failures)
        raise ValueError(f"Failed to decode RGB frames from episode H265 video: {joined}")
    if not decoded_any:
        return task

    template = str((episode_cache / "{camera}" / "{frame:05d}.png").resolve())
    return replace(task, rgb_path_template=template)


def _cache_base_dir(cache_root: Optional[Path]) -> Path:
    if cache_root is not None:
        base = Path(cache_root).expanduser()
    else:
        base = Path.home() / ".cache" / "orbbec_label" / "rgb_frames"
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def _episode_cache_key(episode_dir: Path, payload: Mapping[str, Any]) -> str:
    parts = [
        str(episode_dir.expanduser().resolve()),
        str(payload.get("episode_id") or ""),
        str(payload.get("job_id") or ""),
        str(payload.get("episode_uri") or ""),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _locate_rgb_video(
    episode_dir: Path,
    camera: str,
    payload: Mapping[str, Any],
) -> Tuple[Path, Optional[Path]]:
    from_params = _video_from_camera_params(episode_dir, camera)
    if from_params is None:
        raise FileNotFoundError(f"RGB storageFile missing from {episode_dir / 'camera_params.json'} for camera {camera}")
    return from_params

def _video_from_camera_params(episode_dir: Path, camera: str) -> Optional[Tuple[Path, Optional[Path]]]:
    cam_obj = _camera_params_for(episode_dir, camera)
    if not isinstance(cam_obj, Mapping):
        return None
    rgb_obj = cam_obj.get("RGB") or cam_obj.get("rgb")
    if not isinstance(rgb_obj, Mapping):
        return None
    storage_file = str(rgb_obj.get("storageFile") or "").strip()
    if not storage_file:
        return None
    video_path = _resolve_storage_file(episode_dir, camera, "RGB", storage_file)
    if not video_path.exists():
        raise FileNotFoundError(f"RGB storageFile from camera_params.json not found: {video_path}")
    timestamp_file = str(rgb_obj.get("timestampFile") or "").strip()
    if not timestamp_file:
        raise FileNotFoundError(f"RGB timestampFile missing from camera_params.json for camera {camera}")
    timestamp_path = _resolve_storage_file(episode_dir, camera, "RGB", timestamp_file)
    if not timestamp_path.exists():
        raise FileNotFoundError(f"RGB timestampFile from camera_params.json not found: {timestamp_path}")
    return video_path.resolve(), timestamp_path.resolve()


def _camera_params_for(episode_dir: Path, camera: str) -> Optional[Mapping[str, Any]]:
    path = episode_dir / "camera_params.json"
    if not path.exists() or not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, Mapping):
        cam_obj = obj.get(camera)
        if isinstance(cam_obj, Mapping):
            return cam_obj
    return None


def _resolve_storage_file(episode_dir: Path, camera: str, stream: str, storage_file: str) -> Path:
    raw = unquote(str(storage_file or "").strip())
    if not raw:
        raise ValueError("storageFile must be a non-empty file name")
    p = Path(raw)
    if p.is_absolute():
        raise ValueError(f"storageFile must be relative to the NAS episode root: {storage_file}")
    return (episode_dir / camera / stream / p).resolve()


def _load_frame_map(timestamp_path: Optional[Path]) -> Dict[int, int]:
    if timestamp_path is None or not timestamp_path.exists() or not timestamp_path.is_file():
        return {}
    out: Dict[int, int] = {}
    try:
        with timestamp_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    video_idx = int(str(row.get("video_frame_index") or "").strip())
                    frame_idx = int(str(row.get("frame_index") or "").strip())
                except (TypeError, ValueError):
                    continue
                out[frame_idx] = video_idx
    except Exception:
        return {}
    return out


def _decode_camera_frames(
    *,
    video_path: Path,
    timestamp_path: Optional[Path],
    frame_map: Mapping[int, int],
    camera: str,
    frames: Sequence[int],
    out_dir: Path,
    ffmpeg_executable: str = "ffmpeg",
) -> None:
    requested = sorted({int(frame) for frame in frames if not isinstance(frame, bool)})
    if not requested:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: List[Tuple[int, int]] = []
    for frame in requested:
        target = out_dir / f"{frame:05d}.png"
        if target.exists() and target.is_file():
            continue
        if frame_map:
            if frame not in frame_map:
                raise ValueError(f"frame {frame} is absent from {timestamp_path}")
            video_idx = int(frame_map[frame])
        else:
            video_idx = int(frame)
        missing.append((frame, video_idx))
    if not missing:
        return

    missing.sort(key=lambda item: item[1])
    with tempfile.TemporaryDirectory(prefix=f"decode_{camera}_", dir=str(out_dir)) as tmp_name:
        tmp = Path(tmp_name)
        _run_ffmpeg_select(
            video_path,
            [video_idx for _, video_idx in missing],
            tmp / "%06d.png",
            ffmpeg_executable=ffmpeg_executable,
        )
        outputs = sorted(tmp.glob("*.png"))
        if len(outputs) != len(missing):
            raise RuntimeError(
                f"ffmpeg decoded {len(outputs)} frame(s), expected {len(missing)} from {video_path}"
            )
        for output_path, (frame, _video_idx) in zip(outputs, missing):
            target = out_dir / f"{frame:05d}.png"
            tmp_target = out_dir / f".{target.name}.{os.getpid()}.tmp"
            shutil.move(str(output_path), str(tmp_target))
            os.replace(tmp_target, target)


def _run_ffmpeg_select(
    video_path: Path,
    video_indices: Sequence[int],
    output_pattern: Path,
    *,
    ffmpeg_executable: str = "ffmpeg",
) -> None:
    ffmpeg = str(ffmpeg_executable or "ffmpeg").strip() or "ffmpeg"
    selects = "+".join(f"eq(n\\,{int(idx)})" for idx in video_indices)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select={selects}",
        "-vsync",
        "0",
        str(output_pattern),
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg not found: {ffmpeg}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed for {video_path}: {detail}")
