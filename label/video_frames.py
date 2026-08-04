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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote

try:
    from .storage import CorrectionTask, find_frame_path
except Exception:
    from storage import CorrectionTask, find_frame_path


_RGB_VIDEO_CANDIDATES = ("rgb.h265", "rgb.hevc", "rgb.mp4", "rgb.mkv", "rgb.mov")
_VIDEO_SUFFIXES = (".h265", ".hevc", ".mp4", ".mkv", ".mov")


def ensure_decoded_rgb_frames(
    task: CorrectionTask,
    payload: Optional[Mapping[str, Any]] = None,
    *,
    cache_root: Optional[Path] = None,
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
        raw = os.environ.get("ORBBEC_LABEL_FRAME_CACHE_DIR", "").strip()
        base = Path(raw).expanduser() if raw else Path.home() / ".cache" / "orbbec_label" / "rgb_frames"
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def _episode_cache_key(episode_dir: Path, payload: Mapping[str, Any]) -> str:
    parts = [
        str(episode_dir.expanduser().resolve()),
        str(payload.get("episode_id") or ""),
        str(payload.get("segment_id") or payload.get("job_id") or ""),
        str(payload.get("data_uri") or payload.get("episode_base_uri") or ""),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _locate_rgb_video(
    episode_dir: Path,
    camera: str,
    payload: Mapping[str, Any],
) -> Tuple[Path, Optional[Path]]:
    from_payload = _video_from_payload(episode_dir, camera, payload)
    if from_payload is not None:
        return from_payload

    from_params = _video_from_camera_params(episode_dir, camera)
    if from_params is not None:
        return from_params

    rgb_dir = episode_dir / camera / "RGB"
    for name in _RGB_VIDEO_CANDIDATES:
        candidate = rgb_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate.resolve(), _timestamp_sidecar(candidate)
    if rgb_dir.exists() and rgb_dir.is_dir():
        videos = sorted(p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES)
        if videos:
            return videos[0].resolve(), _timestamp_sidecar(videos[0])
    raise FileNotFoundError(f"RGB H265 video not found under {rgb_dir}")


def _video_from_payload(
    episode_dir: Path,
    camera: str,
    payload: Mapping[str, Any],
) -> Optional[Tuple[Path, Optional[Path]]]:
    media = payload.get("episode_media")
    if not isinstance(media, Mapping):
        return None
    cameras = media.get("cameras")
    if not isinstance(cameras, Mapping):
        return None
    cam_obj = cameras.get(camera)
    if not isinstance(cam_obj, Mapping):
        return None
    rgb_obj = cam_obj.get("rgb") or cam_obj.get("RGB")
    if not isinstance(rgb_obj, Mapping):
        return None

    video_path = _path_from_media_obj(episode_dir, rgb_obj, ("path", "local_path", "resolved_path"))
    if video_path is None:
        storage_file = str(rgb_obj.get("storage_file") or rgb_obj.get("storageFile") or "").strip()
        if storage_file:
            video_path = _resolve_storage_file(episode_dir, camera, "RGB", storage_file)
    if video_path is None or not video_path.exists():
        return None

    timestamp_path = _path_from_media_obj(episode_dir, rgb_obj, ("timestamp_path", "timestamp_local_path", "timestamp_resolved_path"))
    if timestamp_path is None:
        timestamp_file = str(rgb_obj.get("timestamp_file") or rgb_obj.get("timestampFile") or "").strip()
        if timestamp_file:
            timestamp_path = _resolve_storage_file(episode_dir, camera, "RGB", timestamp_file)
    if timestamp_path is None:
        timestamp_path = _timestamp_sidecar(video_path)
    return video_path.resolve(), timestamp_path.resolve() if timestamp_path and timestamp_path.exists() else timestamp_path


def _path_from_media_obj(
    episode_dir: Path,
    obj: Mapping[str, Any],
    keys: Iterable[str],
) -> Optional[Path]:
    for key in keys:
        raw = str(obj.get(key) or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    uri = str(obj.get("uri") or "").strip()
    base_uri = str(obj.get("episode_uri") or obj.get("episode_base_uri") or "").strip().rstrip("/")
    if uri and base_uri and (uri == base_uri or uri.startswith(base_uri + "/")):
        rel = unquote(uri[len(base_uri):].lstrip("/"))
        return (episode_dir / rel).resolve()
    return None


def _video_from_camera_params(episode_dir: Path, camera: str) -> Optional[Tuple[Path, Optional[Path]]]:
    cam_obj = _camera_params_for(episode_dir, camera)
    if not isinstance(cam_obj, Mapping):
        return None
    rgb_obj = cam_obj.get("RGB") or cam_obj.get("rgb")
    if not isinstance(rgb_obj, Mapping):
        return None
    storage_file = str(rgb_obj.get("storageFile") or rgb_obj.get("storage_file") or "").strip()
    if not storage_file:
        return None
    video_path = _resolve_storage_file(episode_dir, camera, "RGB", storage_file)
    if not video_path.exists():
        return None
    timestamp_file = str(rgb_obj.get("timestampFile") or rgb_obj.get("timestamp_file") or "").strip()
    timestamp_path = _resolve_storage_file(episode_dir, camera, "RGB", timestamp_file) if timestamp_file else _timestamp_sidecar(video_path)
    return video_path.resolve(), timestamp_path.resolve() if timestamp_path and timestamp_path.exists() else timestamp_path


def _camera_params_for(episode_dir: Path, camera: str) -> Optional[Mapping[str, Any]]:
    for path in (episode_dir / "camera_params.json", episode_dir / camera / "camera_params.json"):
        if not path.exists() or not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, Mapping):
            cam_obj = obj.get(camera)
            if isinstance(cam_obj, Mapping):
                return cam_obj
            if isinstance(obj.get("RGB"), Mapping) or isinstance(obj.get("rgb"), Mapping):
                return obj
    return None


def _resolve_storage_file(episode_dir: Path, camera: str, stream: str, storage_file: str) -> Path:
    raw = unquote(str(storage_file or "").strip())
    if not raw:
        return episode_dir / camera / stream
    p = Path(raw)
    if p.is_absolute():
        return p.expanduser().resolve()
    candidates = [
        episode_dir / camera / stream / p,
        episode_dir / camera / p,
        episode_dir / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _timestamp_sidecar(video_path: Path) -> Optional[Path]:
    candidate = Path(str(video_path) + ".timestamps.csv")
    return candidate if candidate.exists() else candidate


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
        _run_ffmpeg_select(video_path, [video_idx for _, video_idx in missing], tmp / "%06d.png")
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


def _run_ffmpeg_select(video_path: Path, video_indices: Sequence[int], output_pattern: Path) -> None:
    ffmpeg = os.environ.get("ORBBEC_LABEL_FFMPEG", "ffmpeg")
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
        raise RuntimeError("ffmpeg not found; set ORBBEC_LABEL_FFMPEG or install ffmpeg") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed for {video_path}: {detail}")
