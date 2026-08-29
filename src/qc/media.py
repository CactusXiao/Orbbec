from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

try:
    from label.mano_view import require_episode_calibration, require_mano_episode_artifact
    from label.storage import CorrectionTask, correction_task_from_backend_payload, find_frame_path
    from label.video_frames import _decode_camera_frames, _load_frame_map, _locate_rgb_video
except Exception:
    from ...label.mano_view import require_episode_calibration, require_mano_episode_artifact  # type: ignore
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
        rendered = self.cache_dir / "mesh" / str(camera) / f"{int(frame_idx):05d}.jpg"
        if rendered.exists() and rendered.is_file():
            return rendered
        cached = self.cache_dir / str(camera) / f"{int(frame_idx):05d}.png"
        if cached.exists() and cached.is_file():
            return cached
        return find_frame_path(self.episode_dir, camera, int(frame_idx), self.task.rgb_path_template)


@dataclass(frozen=True)
class MeshRendererSettings:
    python_executable: str
    mano_toolkit_root: Path
    mano_model_dir: Path
    render_factor: float = 1.0
    workers: int = 8


def media_from_payload(payload: Mapping[str, Any], mounts: Mapping[str, str]) -> QcEpisodeMedia:
    task = correction_task_from_backend_payload(dict(payload), mounts=dict(mounts or {}))
    _validate_qc_episode(task)
    episode_id = str(payload.get("episode_id") or task.episode or task.episode_dir().name)
    return QcEpisodeMedia(task=task, cache_dir=Path(episode_id))


def prepare_qc_media(
    payload: Mapping[str, Any],
    *,
    mounts: Mapping[str, str],
    tmp_dir: Path,
    on_progress: Optional[ProgressCallback] = None,
    mesh_renderer: Optional[MeshRendererSettings] = None,
) -> QcEpisodeMedia:
    task = correction_task_from_backend_payload(dict(payload), mounts=dict(mounts or {}))
    _validate_qc_episode(task)
    episode_id = str(payload.get("episode_id") or task.episode or task.episode_dir().name)
    cache_dir = Path(tmp_dir).expanduser().resolve() / episode_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    total = len(task.frames)
    for camera in task.cameras:
        _emit(on_progress, camera, status="pending", decoded=_count_cached(cache_dir, camera, task.frames), total=total, error="")

    def prepare_camera(camera: str) -> None:
        decoded = _count_cached(cache_dir, camera, task.frames)
        if decoded >= total and total > 0:
            _emit(on_progress, camera, status="done", decoded=decoded, total=total, error="")
            return
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
                return
            _emit(on_progress, camera, status="failed", decoded=decoded, total=total, error=str(exc))
            raise

    cameras = [str(camera) for camera in task.cameras]
    with ThreadPoolExecutor(max_workers=max(1, min(6, len(cameras)))) as executor:
        futures = [executor.submit(prepare_camera, camera) for camera in cameras]
        for future in as_completed(futures):
            future.result()
    if mesh_renderer is not None:
        _prepare_mesh_frames(
            task=task,
            cache_dir=cache_dir,
            settings=mesh_renderer,
            on_progress=on_progress,
        )
    return QcEpisodeMedia(task=task, cache_dir=cache_dir)


def _validate_qc_episode(task: CorrectionTask) -> None:
    episode_dir = task.episode_dir()
    require_episode_calibration(episode_dir)
    require_mano_episode_artifact(episode_dir / task.mano_episode_dir)


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


def _prepare_mesh_frames(
    *,
    task: CorrectionTask,
    cache_dir: Path,
    settings: MeshRendererSettings,
    on_progress: Optional[ProgressCallback],
) -> None:
    cameras = [str(camera) for camera in task.cameras[:6]]
    frames = [int(frame) for frame in task.frames]
    output_dir = cache_dir / "mesh"
    total = len(frames)
    if total <= 0:
        raise ValueError("QC episode has no frames to render")
    if all((output_dir / camera / f"{frame:05d}.jpg").is_file() for camera in cameras for frame in frames):
        for camera in cameras:
            _emit(on_progress, camera, status="mesh_done", decoded=total, rendered=total, total=total, error="")
        return

    request = {
        "episode_dir": str(task.episode_dir()),
        "rgb_cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "cameras": cameras,
        "frames": frames,
        "mano_toolkit_root": str(settings.mano_toolkit_root),
        "mano_model_dir": str(settings.mano_model_dir),
        "render_factor": float(settings.render_factor),
        "workers": max(1, int(settings.workers)),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, request_name = tempfile.mkstemp(prefix="mesh_render_", suffix=".json", dir=str(cache_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(request, handle, ensure_ascii=False)
        command = [str(settings.python_executable), "-m", "src.qc.mesh_renderer", "--request", request_name]
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"mesh renderer Python not found: {settings.python_executable}") from exc
        output_lines: List[str] = []
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            output_lines.append(line)
            if len(output_lines) > 20:
                output_lines.pop(0)
            try:
                status = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(status, dict):
                continue
            camera = str(status.get("camera") or "")
            if camera:
                rendered = int(status.get("rendered") or 0)
                _emit(
                    on_progress,
                    camera,
                    status=str(status.get("status") or "mesh_rendering"),
                    decoded=rendered,
                    rendered=rendered,
                    total=int(status.get("total") or total),
                    error="",
                )
        return_code = process.wait()
        if return_code != 0:
            detail = "\n".join(output_lines[-8:]) or f"exit code {return_code}"
            raise RuntimeError(f"MANO mesh rendering failed: {detail}")
    finally:
        try:
            Path(request_name).unlink()
        except OSError:
            pass

    missing = [
        f"{camera}/{frame:05d}.jpg"
        for camera in cameras
        for frame in frames
        if not (output_dir / camera / f"{frame:05d}.jpg").is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"mesh renderer completed with {len(missing)} missing frame(s): {preview}")
