from __future__ import annotations

import csv
import json
import os
import signal
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

try:
    from label.mano_view import require_episode_calibration, require_mano_episode_artifact
    from label.storage import CorrectionTask, correction_task_from_backend_payload, find_frame_path
    from label.video_frames import _decode_camera_frames, _decode_camera_frames_streaming, _load_frame_map, _locate_rgb_video
except Exception:
    from ...label.mano_view import require_episode_calibration, require_mano_episode_artifact  # type: ignore
    from ...label.storage import CorrectionTask, correction_task_from_backend_payload, find_frame_path  # type: ignore
    from ...label.video_frames import _decode_camera_frames, _decode_camera_frames_streaming, _load_frame_map, _locate_rgb_video  # type: ignore


ProgressCallback = Callable[[str, Dict[str, Any]], None]
QC_RGB_CAMERAS = ("00", "02", "03", "05")
EGO_CAMERA = "ego"


@dataclass(frozen=True)
class QcEpisodeMedia:
    task: CorrectionTask
    cache_dir: Path
    requires_mesh: bool = False
    view_cameras: tuple[str, ...] = ()
    preparation: Optional["_QcMediaPreparation"] = field(default=None, compare=False, repr=False)

    @property
    def episode_dir(self) -> Path:
        return self.task.episode_dir()

    @property
    def mano_dir(self) -> Path:
        return self.episode_dir / self.task.mano_episode_dir

    @property
    def display_cameras(self) -> List[str]:
        return list(self.view_cameras or tuple(str(camera) for camera in self.task.cameras[:6]))

    def frame_path(self, camera: str, frame_idx: int) -> Optional[Path]:
        if str(camera) == EGO_CAMERA:
            preview = self.cache_dir / "mesh_preview" / EGO_CAMERA / f"{int(frame_idx):05d}.jpg"
            if preview.exists() and preview.is_file():
                return preview
            if self.requires_mesh:
                return None
        rendered = self.cache_dir / "mesh" / str(camera) / f"{int(frame_idx):05d}.jpg"
        if rendered.exists() and rendered.is_file():
            return rendered
        if self.requires_mesh:
            return None
        cached_jpg = self.cache_dir / str(camera) / f"{int(frame_idx):05d}.jpg"
        if cached_jpg.exists() and cached_jpg.is_file():
            return cached_jpg
        cached = self.cache_dir / str(camera) / f"{int(frame_idx):05d}.png"
        if cached.exists() and cached.is_file():
            return cached
        return find_frame_path(self.episode_dir, camera, int(frame_idx), self.task.rgb_path_template)

    def frame_ready(self, frame_idx: int) -> bool:
        return all(self.frame_path(camera, int(frame_idx)) is not None for camera in self.display_cameras)

    @property
    def preparation_error(self) -> Optional[str]:
        return self.preparation.error if self.preparation is not None else None

    @property
    def preparation_done(self) -> bool:
        return self.preparation is None or self.preparation.done

    def close(self, *, cleanup: bool = False) -> None:
        if self.preparation is not None:
            self.preparation.stop()
        if cleanup:
            cleanup_qc_cache(self.cache_dir)


@dataclass(frozen=True)
class MeshRendererSettings:
    python_executable: str
    mano_toolkit_root: Path
    mano_model_dir: Path
    render_factor: float = 0.5
    workers: int = 6
    prefer_integrated_gpu: bool = True
    prebuffer_frames: int = 30


class _QcMediaPreparation:
    def __init__(
        self,
        *,
        task: CorrectionTask,
        payload: Mapping[str, Any],
        cache_dir: Path,
        settings: MeshRendererSettings,
        on_progress: Optional[ProgressCallback],
        cameras: List[str],
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.task = task
        self.payload = payload
        self.cache_dir = cache_dir
        self.settings = settings
        self.on_progress = on_progress
        self.cameras = list(cameras)
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.done_event = threading.Event()
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self._renderer_process: Optional[subprocess.Popen[str]] = None
        self._thread = threading.Thread(target=self._run, name="qc-media-preparation", daemon=True)

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    @property
    def done(self) -> bool:
        return self.done_event.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._terminate_renderer()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                self._terminate_renderer(force=True)
                self._thread.join()

    def register_renderer(self, process: Optional[subprocess.Popen[str]]) -> None:
        with self._lock:
            self._renderer_process = process
        if process is not None and self.stop_event.is_set():
            self._terminate_process(process)

    def _terminate_renderer(self, *, force: bool = False) -> None:
        with self._lock:
            process = self._renderer_process
        if process is not None:
            self._terminate_process(process, force=force)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str], *, force: bool = False) -> None:
        if os.name == "nt" and process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                process.kill()
            else:
                process.terminate()
        except (OSError, ProcessLookupError):
            pass

    def _set_error(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = str(exc)

    def _prepare_camera(self, camera: str) -> None:
        total = len(self.task.frames)
        decoded = _count_cached(self.cache_dir, camera, self.task.frames)
        if decoded >= total and total > 0:
            _emit(self.on_progress, camera, status="done", decoded=decoded, total=total, error="")
            return
        try:
            video_path, timestamp_path = _locate_rgb_video(self.task.episode_dir(), camera, self.payload)
            frame_map = _load_frame_map(timestamp_path)
            _emit(
                self.on_progress,
                camera,
                status="decoding",
                decoded=decoded,
                total=total,
                error="",
                video=str(video_path),
            )
            progress_step = max(1, total // 100)

            def frame_decoded(_frame: int, published: int) -> None:
                current = min(total, decoded + published)
                if current == total or current % progress_step == 0:
                    _emit(self.on_progress, camera, status="decoding", decoded=current, total=total, error="")

            _decode_camera_frames_streaming(
                video_path=video_path,
                timestamp_path=timestamp_path,
                frame_map=frame_map,
                camera=camera,
                frames=self.task.frames,
                out_dir=self.cache_dir / camera,
                image_extension="jpg",
                jpeg_quality=2,
                ffmpeg_threads=4,
                stop_event=self.stop_event,
                on_frame=frame_decoded,
            )
            decoded = _count_cached(self.cache_dir, camera, self.task.frames)
            _emit(self.on_progress, camera, status="done", decoded=decoded, total=total, error="")
        except InterruptedError:
            raise
        except Exception as exc:
            has_original_frames = all(
                find_frame_path(self.task.episode_dir(), camera, frame, self.task.rgb_path_template) is not None
                for frame in self.task.frames
            )
            if has_original_frames:
                _emit(
                    self.on_progress,
                    camera,
                    status="done",
                    decoded=total,
                    total=total,
                    error="",
                    note="using existing RGB frames",
                )
                return
            _emit(self.on_progress, camera, status="failed", decoded=decoded, total=total, error=str(exc))
            raise

    def _prepare_ego_camera(self) -> None:
        camera = EGO_CAMERA
        total = len(self.task.frames)
        decoded = _count_cached(self.cache_dir, camera, self.task.frames)
        if decoded >= total and total > 0:
            _emit(self.on_progress, camera, status="done", decoded=decoded, total=total, error="")
            return
        try:
            mapping = _load_reference_to_ego_frames(self.task.episode_dir())
            requested = [int(frame) for frame in self.task.frames]
            missing = [frame for frame in requested if frame not in mapping]
            if missing:
                preview = ", ".join(str(frame) for frame in missing[:5])
                raise ValueError(f"Pico timestamp mapping missing {len(missing)} reference frame(s): {preview}")
            video_path = _locate_ego_rgb_video(self.task.episode_dir())
            reverse: Dict[int, List[int]] = {}
            for reference_frame in requested:
                reverse.setdefault(mapping[reference_frame], []).append(reference_frame)
            raw_dir = self.cache_dir / "ego_raw"
            output_dir = self.cache_dir / camera
            output_dir.mkdir(parents=True, exist_ok=True)
            published_references = decoded

            def publish(ego_frame: int, _published: int) -> None:
                nonlocal published_references
                source = raw_dir / f"{int(ego_frame):05d}.jpg"
                if not source.is_file():
                    return
                for reference_frame in reverse.get(int(ego_frame), []):
                    target = output_dir / f"{reference_frame:05d}.jpg"
                    if target.is_file():
                        continue
                    temporary = output_dir / f".{target.name}.{os.getpid()}.tmp"
                    try:
                        os.link(source, temporary)
                    except OSError:
                        shutil.copyfile(source, temporary)
                    os.replace(temporary, target)
                    published_references += 1
                try:
                    source.unlink()
                except OSError:
                    pass
                current = published_references
                if current == total or current % max(1, total // 100) == 0:
                    _emit(self.on_progress, camera, status="decoding", decoded=current, total=total, error="")

            _emit(self.on_progress, camera, status="decoding", decoded=decoded, total=total, error="", video=str(video_path))
            pending_ego_frames = [
                ego_frame
                for ego_frame, reference_frames in sorted(reverse.items())
                if any(
                    not (output_dir / f"{reference_frame:05d}.jpg").is_file()
                    for reference_frame in reference_frames
                )
            ]
            _decode_camera_frames_streaming(
                video_path=video_path,
                timestamp_path=None,
                frame_map={},
                camera=camera,
                frames=pending_ego_frames,
                out_dir=raw_dir,
                image_extension="jpg",
                jpeg_quality=2,
                ffmpeg_threads=4,
                stop_event=self.stop_event,
                on_frame=publish,
            )
            # Cached raw frames do not trigger the streaming callback on a resumed run.
            for ego_frame in sorted(reverse):
                publish(ego_frame, 0)
            decoded = _count_cached(self.cache_dir, camera, requested)
            if decoded != total:
                raise RuntimeError(f"decoded {decoded}/{total} synchronized Pico RGB frames")
            _emit(self.on_progress, camera, status="done", decoded=decoded, total=total, error="")
        except InterruptedError:
            raise
        except Exception as exc:
            _emit(self.on_progress, camera, status="failed", decoded=decoded, total=total, error=str(exc))
            raise

    def _run(self) -> None:
        cameras = list(self.cameras)
        executor = ThreadPoolExecutor(max_workers=max(2, min(7, len(cameras) + 1)))
        try:
            futures = []
            for camera in cameras:
                if camera == EGO_CAMERA:
                    futures.append(executor.submit(self._prepare_ego_camera))
                else:
                    futures.append(executor.submit(self._prepare_camera, camera))
            futures.append(
                executor.submit(
                    _prepare_mesh_frames,
                    task=self.task,
                    cache_dir=self.cache_dir,
                    settings=self.settings,
                    on_progress=self.on_progress,
                    stop_event=self.stop_event,
                    process_callback=self.register_renderer,
                    cameras=cameras,
                )
            )
            for future in as_completed(futures):
                try:
                    future.result()
                except InterruptedError as exc:
                    if not self.stop_event.is_set():
                        self._set_error(exc)
                    self.stop_event.set()
                    self._terminate_renderer()
                    break
                except Exception as exc:
                    self._set_error(exc)
                    self.stop_event.set()
                    self._terminate_renderer()
                    break
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            self.register_renderer(None)
            self.done_event.set()


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
    stop_event: Optional[threading.Event] = None,
    on_media_created: Optional[Callable[[QcEpisodeMedia], None]] = None,
) -> QcEpisodeMedia:
    task = correction_task_from_backend_payload(dict(payload), mounts=dict(mounts or {}))
    _validate_qc_episode(task)
    episode_id = str(payload.get("episode_id") or task.episode or task.episode_dir().name)
    cache_dir = Path(tmp_dir).expanduser().resolve() / episode_id
    total = len(task.frames)
    if total <= 0:
        raise ValueError("QC episode has no frames to prepare")
    view_cameras = _qc_view_cameras(task, include_ego=mesh_renderer is not None)
    if mesh_renderer is not None:
        _validate_ego_assets(task.episode_dir())
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("QC preparation stopped")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for camera in view_cameras:
        _emit(on_progress, camera, status="pending", decoded=_count_cached(cache_dir, camera, task.frames), total=total, error="")

    if mesh_renderer is not None:
        cameras = list(view_cameras)
        media = QcEpisodeMedia(task=task, cache_dir=cache_dir, requires_mesh=True, view_cameras=tuple(cameras))
        if all(
            (
                cache_dir
                / ("mesh_preview" if camera == EGO_CAMERA else "mesh")
                / camera
                / f"{int(frame):05d}.jpg"
            ).is_file()
            for camera in cameras
            for frame in task.frames
        ):
            return media
        preparation = _QcMediaPreparation(
            task=task,
            payload=dict(payload),
            cache_dir=cache_dir,
            settings=mesh_renderer,
            on_progress=on_progress,
            cameras=cameras,
            stop_event=stop_event,
        )
        media = QcEpisodeMedia(
            task=task,
            cache_dir=cache_dir,
            requires_mesh=True,
            view_cameras=tuple(cameras),
            preparation=preparation,
        )
        if on_media_created is not None:
            on_media_created(media)
        preparation.start()
        prebuffer_count = min(total, max(1, int(mesh_renderer.prebuffer_frames)))
        prebuffer_frames = [int(frame) for frame in task.frames[:prebuffer_count]]
        try:
            while not all(media.frame_ready(frame) for frame in prebuffer_frames):
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError("QC preparation stopped")
                error = preparation.error
                if error:
                    raise RuntimeError(error)
                if preparation.done:
                    raise RuntimeError("mesh preparation stopped before the playback buffer was ready")
                time.sleep(0.02)
        except Exception:
            media.close(cleanup=True)
            raise
        return media

    def prepare_camera(camera: str) -> None:
        decoded = _count_cached(cache_dir, camera, task.frames)
        if decoded >= total and total > 0:
            _emit(on_progress, camera, status="done", decoded=decoded, total=total, error="")
            return
        try:
            video_path, timestamp_path = _locate_rgb_video(task.episode_dir(), camera, payload)
            frame_map = _load_frame_map(timestamp_path)
            _emit(on_progress, camera, status="decoding", decoded=decoded, total=total, error="", video=str(video_path))
            decode = _decode_camera_frames if stop_event is None else _decode_camera_frames_streaming
            cancellation = {} if stop_event is None else {"stop_event": stop_event}
            decode(
                **cancellation,
                video_path=video_path,
                timestamp_path=timestamp_path,
                frame_map=frame_map,
                camera=camera,
                frames=task.frames,
                out_dir=cache_dir / camera,
                image_extension="jpg",
                jpeg_quality=2,
                ffmpeg_threads=4,
            )
            decoded = _count_cached(cache_dir, camera, task.frames)
            _emit(on_progress, camera, status="done", decoded=decoded, total=total, error="")
        except InterruptedError:
            raise
        except Exception as exc:
            has_original_frames = all(find_frame_path(task.episode_dir(), camera, frame, task.rgb_path_template) is not None for frame in task.frames)
            if has_original_frames:
                _emit(on_progress, camera, status="done", decoded=total, total=total, error="", note="using existing RGB frames")
                return
            _emit(on_progress, camera, status="failed", decoded=decoded, total=total, error=str(exc))
            raise

    cameras = [str(camera) for camera in task.cameras]
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(6, len(cameras)))) as executor:
            futures = [executor.submit(prepare_camera, camera) for camera in cameras]
            for future in as_completed(futures):
                future.result()
    except Exception:
        cleanup_qc_cache(cache_dir)
        raise
    return QcEpisodeMedia(task=task, cache_dir=cache_dir)


def _validate_qc_episode(task: CorrectionTask) -> None:
    episode_dir = task.episode_dir()
    require_episode_calibration(episode_dir)
    require_mano_episode_artifact(episode_dir / task.mano_episode_dir)


def _qc_view_cameras(task: CorrectionTask, *, include_ego: bool) -> List[str]:
    available = [str(camera) for camera in task.cameras]
    selected = [camera for camera in QC_RGB_CAMERAS if camera in available]
    if not selected:
        selected = available[:4]
    if include_ego:
        selected.append(EGO_CAMERA)
    return selected


def _ego_camera_params_path(episode_dir: Path) -> Path:
    path = Path(episode_dir) / "ego" / "camera_params.json"
    if not path.is_file():
        raise FileNotFoundError(f"Pico camera calibration not found: {path}")
    return path


def _ego_pose_path(episode_dir: Path) -> Path:
    for name in ("ego_pose.json", "ego_extrinsic.json"):
        path = Path(episode_dir) / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Pico per-frame extrinsics not found: {Path(episode_dir) / 'ego_pose.json'}")


def _ego_timestamp_path(episode_dir: Path) -> Path:
    candidates = (
        Path(episode_dir) / "timestamps.csv",
        Path(episode_dir) / "ego" / "RGB" / "rgb.h265.timestamps.csv",
        Path(episode_dir) / "ego" / "timestamps.csv",
    )
    existing: List[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        existing.append(path)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                fields = set(csv.DictReader(handle).fieldnames or [])
            if {"frame_index", "ego_frame_index"}.issubset(fields):
                return path
        except OSError:
            continue
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise ValueError(f"Pico timestamps are missing frame_index/ego_frame_index columns: {joined}")
    raise FileNotFoundError(f"Pico synchronized timestamps not found under {Path(episode_dir) / 'ego'}")


def _load_reference_to_ego_frames(episode_dir: Path) -> Dict[int, int]:
    path = _ego_timestamp_path(episode_dir)
    mapping: Dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            reference_text = str(row.get("frame_index") or "").strip()
            ego_text = str(row.get("ego_frame_index") or "").strip()
            if not reference_text or not ego_text:
                continue
            reference_frame = int(reference_text)
            ego_frame = int(ego_text)
            previous = mapping.get(reference_frame)
            if previous is not None and previous != ego_frame:
                raise ValueError(
                    f"conflicting Pico frame mapping for reference frame {reference_frame}: "
                    f"{previous} and {ego_frame}"
                )
            mapping[reference_frame] = ego_frame
    if not mapping:
        raise ValueError(f"no frame_index -> ego_frame_index mappings found: {path}")
    return mapping


def _locate_ego_rgb_video(episode_dir: Path) -> Path:
    params_path = _ego_camera_params_path(episode_dir)
    data = json.loads(params_path.read_text(encoding="utf-8"))
    try:
        storage_file = str(data["ego"]["RGB"]["storageFile"]).strip()
    except (KeyError, TypeError) as exc:
        raise KeyError(f"missing ego.RGB.storageFile in {params_path}") from exc
    relative = Path(storage_file)
    if not storage_file or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid Pico RGB storageFile in {params_path}: {storage_file}")
    candidates = (
        Path(episode_dir) / "ego" / relative,
        Path(episode_dir) / "ego" / "RGB" / relative.name,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Pico RGB video not found for storageFile={storage_file}: {params_path}")


def _validate_ego_assets(episode_dir: Path) -> None:
    _ego_camera_params_path(episode_dir)
    _ego_pose_path(episode_dir)
    _ego_timestamp_path(episode_dir)
    _locate_ego_rgb_video(episode_dir)


def cleanup_qc_cache(cache_dir: Path) -> None:
    path = Path(cache_dir)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def _count_cached(cache_dir: Path, camera: str, frames: List[int]) -> int:
    cam_dir = Path(cache_dir) / str(camera)
    if not cam_dir.exists() or not cam_dir.is_dir():
        return 0
    return sum(
        1
        for frame in frames
        if (cam_dir / f"{int(frame):05d}.jpg").is_file() or (cam_dir / f"{int(frame):05d}.png").is_file()
    )


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
    stop_event: Optional[threading.Event] = None,
    process_callback: Optional[Callable[[Optional[subprocess.Popen[str]]], None]] = None,
    cameras: Optional[List[str]] = None,
) -> None:
    cameras = list(cameras or [str(camera) for camera in task.cameras[:6]])
    frames = [int(frame) for frame in task.frames]
    output_dir = cache_dir / "mesh"
    preview_output_dir = cache_dir / "mesh_preview"
    total = len(frames)
    if total <= 0:
        raise ValueError("QC episode has no frames to render")
    def rendered_path(camera: str, frame: int) -> Path:
        root = preview_output_dir if camera == EGO_CAMERA else output_dir
        return root / camera / f"{frame:05d}.jpg"

    if all(rendered_path(camera, frame).is_file() for camera in cameras for frame in frames):
        for camera in cameras:
            _emit(on_progress, camera, status="mesh_done", decoded=total, rendered=total, total=total, error="")
        return

    request = {
        "episode_dir": str(task.episode_dir()),
        "rgb_cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "preview_output_dir": str(preview_output_dir),
        "ego_preview_max_width": 960,
        "cameras": cameras,
        "frames": frames,
        "mano_toolkit_root": str(settings.mano_toolkit_root),
        "mano_model_dir": str(settings.mano_model_dir),
        "render_factor": float(settings.render_factor),
        "workers": max(1, int(settings.workers)),
        "prefer_integrated_gpu": bool(settings.prefer_integrated_gpu),
        "source_wait_seconds": 300.0,
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
                start_new_session=os.name != "nt",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"mesh renderer Python not found: {settings.python_executable}") from exc
        if process_callback is not None:
            process_callback(process)
        output_lines: List[str] = []
        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError("mesh preparation stopped")
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
                        error=str(status.get("error") or ""),
                    )
            return_code = process.wait()
        finally:
            if process.poll() is None or (stop_event is not None and stop_event.is_set()):
                _QcMediaPreparation._terminate_process(process)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    _QcMediaPreparation._terminate_process(process, force=True)
                    process.wait()
                # The parent may already be gone while a worker still owns a
                # cache file. Kill the entire group before releasing its handle.
                _QcMediaPreparation._terminate_process(process, force=True)
            if process.stdout is not None:
                process.stdout.close()
            if process_callback is not None:
                process_callback(None)
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("mesh preparation stopped")
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
        if not rendered_path(camera, frame).is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"mesh renderer completed with {len(missing)} missing frame(s): {preview}")
