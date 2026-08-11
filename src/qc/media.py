from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

try:
    from label.storage import CorrectionTask, correction_task_from_backend_payload, find_frame_path
    from label.video_frames import _decode_camera_frames, _load_frame_map, _locate_rgb_video
except Exception:
    from ...label.storage import CorrectionTask, correction_task_from_backend_payload, find_frame_path  # type: ignore
    from ...label.video_frames import _decode_camera_frames, _load_frame_map, _locate_rgb_video  # type: ignore


ProgressCallback = Callable[[str, Dict[str, Any]], None]


@dataclass(frozen=True)
class QcEpisodeMedia:
    task: CorrectionTask
    cache_dir: Path

    @property
    def episode_dir(self) -> Path:
        return self.task.episode_dir()

    @property
    def mano_dir(self) -> Path:
        return self.episode_dir / self.task.mano_episode_dir

    def frame_path(self, camera: str, frame_idx: int) -> Optional[Path]:
        cached = self.cache_dir / str(camera) / f"{int(frame_idx):05d}.png"
        if cached.exists() and cached.is_file():
            return cached
        return find_frame_path(self.episode_dir, camera, int(frame_idx), self.task.rgb_path_template)


def media_from_payload(payload: Mapping[str, Any], mounts: Mapping[str, str]) -> QcEpisodeMedia:
    task = correction_task_from_backend_payload(dict(payload), mounts=dict(mounts or {}))
    episode_id = str(payload.get("episode_id") or task.episode or task.episode_dir().name)
    return QcEpisodeMedia(task=task, cache_dir=Path(episode_id))


def prepare_qc_media(
    payload: Mapping[str, Any],
    *,
    mounts: Mapping[str, str],
    tmp_dir: Path,
    on_progress: Optional[ProgressCallback] = None,
) -> QcEpisodeMedia:
    task = correction_task_from_backend_payload(dict(payload), mounts=dict(mounts or {}))
    episode_id = str(payload.get("episode_id") or task.episode or task.episode_dir().name)
    cache_dir = Path(tmp_dir).expanduser().resolve() / episode_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    total = len(task.frames)
    for camera in task.cameras:
        _emit(on_progress, camera, status="pending", decoded=_count_cached(cache_dir, camera, task.frames), total=total, error="")

    for camera in task.cameras:
        decoded = _count_cached(cache_dir, camera, task.frames)
        if decoded >= total and total > 0:
            _emit(on_progress, camera, status="done", decoded=decoded, total=total, error="")
            continue
        try:
            video_path, timestamp_path = _locate_rgb_video(task.episode_dir(), camera, payload)
            frame_map = _load_frame_map(timestamp_path)
            _emit(on_progress, camera, status="decoding", decoded=decoded, total=total, error="", video=str(video_path))
            _decode_camera_frames(
                video_path=video_path,
                timestamp_path=timestamp_path,
                frame_map=frame_map,
                camera=camera,
                frames=task.frames,
                out_dir=cache_dir / camera,
            )
            decoded = _count_cached(cache_dir, camera, task.frames)
            _emit(on_progress, camera, status="done", decoded=decoded, total=total, error="")
        except Exception as exc:
            has_original_frames = all(find_frame_path(task.episode_dir(), camera, frame, task.rgb_path_template) is not None for frame in task.frames)
            if has_original_frames:
                _emit(on_progress, camera, status="done", decoded=total, total=total, error="", note="using existing RGB frames")
                continue
            _emit(on_progress, camera, status="failed", decoded=decoded, total=total, error=str(exc))
            raise
    return QcEpisodeMedia(task=task, cache_dir=cache_dir)


def cleanup_qc_cache(cache_dir: Path) -> None:
    try:
        path = Path(cache_dir)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
    except OSError:
        pass


def _count_cached(cache_dir: Path, camera: str, frames: List[int]) -> int:
    cam_dir = Path(cache_dir) / str(camera)
    if not cam_dir.exists() or not cam_dir.is_dir():
        return 0
    return sum(1 for frame in frames if (cam_dir / f"{int(frame):05d}.png").exists())


def _emit(callback: Optional[ProgressCallback], camera: str, **fields: Any) -> None:
    if callback is None:
        return
    payload = {"camera": camera}
    payload.update(fields)
    callback(camera, payload)

