"""Browser viewer sessions for one captured episode.

The desktop capture viewer writes ``decoded_pic`` into the episode directory.  The
web backend must never mutate NAS data, so this module keeps the same preparation
contract in an isolated temporary session directory and removes it when the tab
closes (with an expiry sweep as a crash-safe fallback).
"""

from __future__ import annotations

import atexit
import csv
import json
import math
import mimetypes
import os
import signal
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


VIDEO_SUFFIXES = {".h265", ".hevc", ".mkv", ".mp4", ".mov", ".avi"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIEWER_MODES = ("rgb", "pico", "pointcloud", "manomesh", "picohand")


class ViewerError(Exception):
    pass


def _json_file(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _frame_number(path: Path) -> Optional[int]:
    try:
        return int(path.stem)
    except ValueError:
        return None


def _images(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=lambda p: (_frame_number(p) is None, _frame_number(p) or 0, p.name))


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error, UnicodeError):
        return []


def _ego_metadata_rows(path: Path) -> List[Dict[str, str]]:
    """Parse Pico metadata with the same legacy column repair as viewer.cpp."""
    if not path.is_file():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if not header:
                return []
            gaze_index = header.index("gaze_valid") if "gaze_valid" in header else -1
            rows: List[Dict[str, str]] = []
            bool_values = {"0", "1", "true", "false", "yes", "no"}
            for columns in reader:
                if len(columns) == len(header) + 1 and gaze_index >= 0 and gaze_index + 1 < len(columns):
                    left = columns[gaze_index].strip().lower()
                    right = columns[gaze_index + 1].strip().lower()
                    if left in bool_values and right in bool_values:
                        del columns[gaze_index + 1]
                rows.append(dict(zip(header, columns)))
            return rows
    except (OSError, csv.Error, UnicodeError):
        return []


def _integer(value: Any, default: int = -1) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "valid"}


def _resolve_storage_file(episode_dir: Path, camera: str, stream: str, value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    choices = [episode_dir / camera / stream / candidate, episode_dir / candidate]
    for choice in choices:
        try:
            resolved = choice.resolve()
        except OSError:
            continue
        if (resolved == episode_dir or episode_dir in resolved.parents) and resolved.is_file():
            return resolved
    return None


def _video_frame_map(timestamp_file: Optional[Path]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if timestamp_file is None:
        return mapping
    for row_index, row in enumerate(_csv_rows(timestamp_file)):
        aligned = _integer(row.get("frame_index"), row_index)
        video = _integer(row.get("video_frame_index"), _integer(row.get("ego_frame_index"), row_index))
        if aligned >= 0 and video >= 0:
            mapping[aligned] = video
    return mapping


def _metadata_frame_map(timestamp_file: Optional[Path]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if timestamp_file is None:
        return mapping
    for row_index, row in enumerate(_csv_rows(timestamp_file)):
        aligned = _integer(row.get("frame_index"), row_index)
        source = _integer(row.get("ego_source_frame_index"), _integer(row.get("ego_frame_index"), row_index))
        if aligned >= 0 and source >= 0:
            mapping[aligned] = source
    return mapping


@dataclass
class ViewerSource:
    source_id: str
    label: str
    kind: str
    camera: str
    rgb_dir: Optional[Path] = None
    rgb_video: Optional[Path] = None
    rgb_timestamps: Optional[Path] = None
    rgb_frame_map: Dict[int, int] = field(default_factory=dict)
    rgb_metadata_map: Dict[int, int] = field(default_factory=dict)
    depth_dir: Optional[Path] = None
    depth_video: Optional[Path] = None
    depth_timestamps: Optional[Path] = None
    depth_frame_map: Dict[int, int] = field(default_factory=dict)
    decoded_rgb: Optional[Path] = None
    decoded_depth: Optional[Path] = None


@dataclass
class ModeState:
    status: str = "idle"
    progress: int = 0
    total: int = 0
    message: str = ""
    error: str = ""
    playable: bool = False
    complete: bool = False
    available_frames: List[int] = field(default_factory=list)

    def payload(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "message": self.message,
            "error": self.error,
            "playable": self.playable,
            "complete": self.complete,
            "available_frames": list(self.available_frames),
        }


@dataclass
class ViewerSession:
    session_id: str
    episode_id: str
    episode_dir: Path
    temp_dir: Path
    ffmpeg: str
    created_at: float = field(default_factory=time.time)
    touched_at: float = field(default_factory=time.time)
    state: str = "preparing"
    error: str = ""
    frame_indices: List[int] = field(default_factory=list)
    fps: float = 30.0
    sources: List[ViewerSource] = field(default_factory=list)
    decode_progress: int = 0
    decode_total: int = 0
    decode_jobs: List[Dict[str, Any]] = field(default_factory=list)
    mano_context: Dict[str, Any] = field(default_factory=dict)
    modes: Dict[str, ModeState] = field(default_factory=lambda: {name: ModeState() for name in VIEWER_MODES})
    closed: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    processes: List[subprocess.Popen[bytes]] = field(default_factory=list)
    process_groups: set = field(default_factory=set)

    def touch(self) -> None:
        self.touched_at = time.time()

    def payload(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "episode_id": self.episode_id,
                "state": self.state,
                "error": self.error,
                "decode": {
                    "completed": self.decode_progress,
                    "total": self.decode_total,
                    "jobs": list(self.decode_jobs),
                },
                "frame_count": len(self.frame_indices),
                "frames": list(self.frame_indices),
                "fps": self.fps,
                "sources": [
                    {"id": s.source_id, "label": s.label, "kind": s.kind, "camera": s.camera}
                    for s in self.sources
                ],
                "mano": dict(self.mano_context),
                "modes": {name: state.payload() for name, state in self.modes.items()},
            }


class ViewerSessionManager:
    """Owns decode workers and the disposable files for browser tabs."""

    def __init__(
        self,
        *,
        temp_root: Optional[Path] = None,
        ffmpeg: str = "ffmpeg",
        max_decode_workers: int = 8,
        session_ttl_seconds: int = 120,
        mano_toolkit_root: Optional[Path] = None,
        mano_model_dir: Optional[Path] = None,
        mesh_python: str = "python3",
        mesh_prebuffer_frames: int = 30,
    ):
        requested_root = Path(temp_root).expanduser() if temp_root else Path(tempfile.gettempdir()) / "orbbec_web_viewer"
        self.temp_root = requested_root.resolve()
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = ffmpeg
        self.max_decode_workers = max(1, min(16, int(max_decode_workers)))
        self.session_ttl_seconds = max(60, int(session_ttl_seconds))
        self.mano_toolkit_root = Path(mano_toolkit_root).expanduser().resolve() if mano_toolkit_root else None
        self.mano_model_dir = Path(mano_model_dir).expanduser().resolve() if mano_model_dir else None
        self.mesh_python = mesh_python
        self.mesh_prebuffer_frames = max(1, min(300, int(mesh_prebuffer_frames)))
        self._sessions: Dict[str, ViewerSession] = {}
        self._lock = threading.RLock()
        self._coordinator = ThreadPoolExecutor(max_workers=4, thread_name_prefix="viewer-session")
        self._shutdown = threading.Event()
        self._janitor = threading.Thread(target=self._janitor_loop, name="viewer-session-cleaner", daemon=True)
        self._janitor.start()
        atexit.register(self.shutdown)

    def create(self, episode_id: str, episode_dir: Path) -> ViewerSession:
        self.sweep_expired()
        episode_dir = Path(episode_dir).expanduser().resolve()
        if not episode_dir.is_dir():
            raise ViewerError(f"Episode 数据目录不存在：{episode_dir}")
        session_id = uuid.uuid4().hex
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{session_id}_", dir=str(self.temp_root))).resolve()
        session = ViewerSession(
            session_id=session_id,
            episode_id=str(episode_id),
            episode_dir=episode_dir,
            temp_dir=temp_dir,
            ffmpeg=self.ffmpeg,
        )
        with self._lock:
            self._sessions[session_id] = session
        self._coordinator.submit(self._prepare_initial, session)
        return session

    def get(self, session_id: str) -> ViewerSession:
        with self._lock:
            session = self._sessions.get(str(session_id))
        if session is None or session.closed:
            raise ViewerError("Viewer 会话不存在或已结束")
        session.touch()
        return session

    def close(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(str(session_id), None)
        if session is None:
            return False
        with session.lock:
            session.closed = True
            session.cancel.set()
            processes = list(session.processes)
            process_groups = set(session.process_groups)
        for process in processes:
            try:
                if os.name != "nt" and process.pid in process_groups:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except OSError:
                pass
        shutil.rmtree(session.temp_dir, ignore_errors=True)
        return True

    def close_all(self) -> None:
        with self._lock:
            ids = list(self._sessions)
        for session_id in ids:
            self.close(session_id)

    def shutdown(self) -> None:
        self._shutdown.set()
        self.close_all()
        self._coordinator.shutdown(wait=False, cancel_futures=True)

    def _janitor_loop(self) -> None:
        interval = max(10.0, min(30.0, self.session_ttl_seconds / 2.0))
        while not self._shutdown.wait(interval):
            self.sweep_expired()

    def sweep_expired(self) -> None:
        cutoff = time.time() - self.session_ttl_seconds
        with self._lock:
            expired = [session_id for session_id, session in self._sessions.items() if session.touched_at < cutoff]
        for session_id in expired:
            self.close(session_id)

    def prepare_mode(self, session_id: str, mode: str) -> ViewerSession:
        session = self.get(session_id)
        mode = str(mode).strip().lower()
        if mode not in VIEWER_MODES:
            raise ViewerError(f"未知查看模态：{mode}")
        with session.lock:
            if session.state != "ready":
                raise ViewerError("基础视频仍在解码，请等待全部解码完成")
            state = session.modes[mode]
            if state.status in {"preparing", "ready"}:
                return session
            state.status = "preparing"
            state.message = "正在准备数据"
            state.error = ""
        self._coordinator.submit(self._prepare_mode, session, mode)
        return session

    def media_path(self, session_id: str, mode: str, source_id: str, frame: int) -> Tuple[Path, str]:
        session = self.get(session_id)
        if mode not in VIEWER_MODES:
            raise ViewerError("未知查看模态")
        with session.lock:
            if session.state != "ready" or session.modes[mode].status != "ready":
                raise ViewerError("该查看模态尚未准备完成")
        frame = int(frame)
        if mode == "rgb":
            source = next((s for s in session.sources if s.source_id == source_id and s.kind == "multiview"), None)
            if source is None:
                raise ViewerError("RGB 相机不存在")
            path = self._rgb_frame_path(source, frame)
        elif mode == "pico":
            path = session.temp_dir / "modes" / "pico" / f"{frame:05d}.jpg"
        elif mode == "pointcloud":
            path = session.temp_dir / "modes" / "pointcloud" / f"{frame:05d}.bin"
        elif mode == "manomesh":
            camera = str(source_id)
            path = session.temp_dir / "modes" / "manomesh" / camera / f"{frame:05d}.jpg"
        else:
            if str(source_id) != "ego":
                raise ViewerError("Pico 手部 Pose 数据源不存在")
            path = session.temp_dir / "modes" / "picohand" / "ego" / f"{frame:05d}.jpg"
        if not path.is_file():
            raise ViewerError(f"帧不存在：{frame}")
        mime = mimetypes.guess_type(path.name)[0] or ("application/json" if path.suffix == ".json" else "application/octet-stream")
        return path, mime

    def _prepare_initial(self, session: ViewerSession) -> None:
        try:
            sources, frames, fps = self._discover_episode(session.episode_dir, session.temp_dir)
            mano_context = self._mano_context(session.episode_dir, frames)
            with session.lock:
                session.sources = sources
                session.frame_indices = frames
                session.fps = fps
                session.mano_context = mano_context
            jobs = self._decode_jobs(session)
            with session.lock:
                session.decode_total = len(jobs)
                session.decode_jobs = [
                    {"name": label, "status": "pending", "error": ""} for label, _video, _output, _depth in jobs
                ]
            if jobs:
                with ThreadPoolExecutor(
                    max_workers=min(self.max_decode_workers, len(jobs)), thread_name_prefix="viewer-decode"
                ) as executor:
                    pending = {
                        executor.submit(self._decode_video, session, video, output, depth): index
                        for index, (_label, video, output, depth) in enumerate(jobs)
                    }
                    for future in as_completed(pending):
                        index = pending[future]
                        try:
                            future.result()
                            job_status, error = "done", ""
                        except Exception as exc:
                            job_status, error = "failed", str(exc)
                        with session.lock:
                            session.decode_progress += 1
                            session.decode_jobs[index]["status"] = job_status
                            session.decode_jobs[index]["error"] = error
                        if error:
                            raise ViewerError(error)
            if session.cancel.is_set():
                return
            rgb_sources = [s for s in sources if s.kind == "multiview" and self._source_has_rgb(s, frames)]
            if not rgb_sources:
                raise ViewerError("Episode 中未找到可显示的 RGB 数据")
            with session.lock:
                session.sources = [s for s in sources if s.kind != "multiview"] + rgb_sources[:6]
                session.modes["rgb"] = ModeState(
                    "ready",
                    len(frames),
                    len(frames),
                    "六路 RGB 已就绪",
                    playable=True,
                    complete=True,
                    available_frames=list(frames),
                )
                session.state = "ready"
        except Exception as exc:
            with session.lock:
                if not session.closed:
                    session.state = "failed"
                    session.error = str(exc)

    @staticmethod
    def _mano_context(episode_dir: Path, frames: Sequence[int]) -> Dict[str, Any]:
        report_path = episode_dir / "qc" / "qc_report.json"
        report = _json_file(report_path)
        qc_completed = bool(report) and str(report.get("kind") or "") == "orbbec_qc_report"
        raw_segments = report.get("segments") if isinstance(report.get("segments"), list) else []
        segments: List[Dict[str, int]] = []
        for value in raw_segments:
            if not isinstance(value, Mapping):
                continue
            start = _integer(value.get("start_frame"), -1)
            end = _integer(value.get("end_frame"), -1)
            if start < 0 or end < start:
                continue
            segments.append({"start_frame": start, "end_frame": end})
        segments.sort(key=lambda item: (item["start_frame"], item["end_frame"]))

        frame_set = {int(frame) for frame in frames}
        bad_frames = sorted(
            frame
            for segment in segments
            for frame in range(segment["start_frame"], segment["end_frame"] + 1)
            if frame in frame_set
        )
        manual_mtime: Dict[int, float] = {}
        manual_root = episode_dir / "manual_2d"
        if manual_root.is_dir():
            for path in manual_root.rglob("*.npy"):
                frame = _frame_number(path)
                if frame is None or frame not in frame_set:
                    continue
                try:
                    manual_mtime[frame] = max(manual_mtime.get(frame, 0.0), path.stat().st_mtime)
                except OSError:
                    continue
        pose_mtime: Dict[int, float] = {}
        pose_root = episode_dir / "optimized_pose"
        if pose_root.is_dir():
            for path in pose_root.glob("*.npy"):
                frame = _frame_number(path)
                if frame is None or frame not in frame_set:
                    continue
                try:
                    pose_mtime[frame] = path.stat().st_mtime
                except OSError:
                    continue
        corrected = bool(bad_frames) and all(
            frame in manual_mtime
            and frame in pose_mtime
            and pose_mtime[frame] > manual_mtime[frame]
            for frame in bad_frames
        )
        return {
            "qc_completed": qc_completed,
            "qc_passed": bool(report.get("passed")) if qc_completed else False,
            "bad_ranges": segments,
            "source": "corrected_3d" if corrected else "auto_label",
            "source_label": "纠偏后 3D 优化" if corrected else "初始 AutoLabel 结果",
        }

    def _discover_episode(self, episode_dir: Path, temp_dir: Path) -> Tuple[List[ViewerSource], List[int], float]:
        params = _json_file(episode_dir / "camera_params.json")
        sources: List[ViewerSource] = []
        for camera_dir in sorted((p for p in episode_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: p.name):
            camera = camera_dir.name
            cam_params = params.get(camera) if isinstance(params.get(camera), dict) else {}
            rgb_params = (cam_params.get("RGB") or cam_params.get("rgb") or {}) if isinstance(cam_params, dict) else {}
            depth_params = (cam_params.get("Depth") or cam_params.get("depth") or {}) if isinstance(cam_params, dict) else {}
            rgb_dir = camera_dir / "RGB"
            rgb_video = _resolve_storage_file(episode_dir, camera, "RGB", rgb_params.get("storageFile"))
            if rgb_video is None:
                rgb_video = next((p for p in sorted(rgb_dir.glob("*")) if p.suffix.lower() in VIDEO_SUFFIXES), None)
            timestamp = _resolve_storage_file(episode_dir, camera, "RGB", rgb_params.get("timestampFile"))
            if timestamp is None:
                timestamp = next(iter(sorted(rgb_dir.glob("*.timestamps.csv"))), None)
            depth_dir = camera_dir / "Depth"
            depth_video = _resolve_storage_file(episode_dir, camera, "Depth", depth_params.get("storageFile"))
            if depth_video is None:
                depth_video = next((p for p in sorted(depth_dir.glob("*")) if p.suffix.lower() in VIDEO_SUFFIXES), None)
            depth_timestamp = _resolve_storage_file(episode_dir, camera, "Depth", depth_params.get("timestampFile"))
            if depth_timestamp is None:
                depth_timestamp = next(iter(sorted(depth_dir.glob("*.timestamps.csv"))), None)
            sources.append(
                ViewerSource(
                    source_id=f"mv:{camera}",
                    label=camera,
                    kind="multiview",
                    camera=camera,
                    rgb_dir=rgb_dir if rgb_dir.is_dir() else None,
                    rgb_video=rgb_video,
                    rgb_timestamps=timestamp,
                    rgb_frame_map=_video_frame_map(timestamp),
                    rgb_metadata_map=_metadata_frame_map(timestamp),
                    depth_dir=depth_dir if depth_dir.is_dir() else None,
                    depth_video=depth_video,
                    depth_timestamps=depth_timestamp,
                    depth_frame_map=_video_frame_map(depth_timestamp),
                    decoded_rgb=temp_dir / "decoded" / camera / "RGB" if rgb_video else None,
                    decoded_depth=temp_dir / "decoded" / camera / "Depth" if depth_video else None,
                )
            )

        ego_dir = episode_dir / "ego"
        if ego_dir.is_dir():
            ego_params = _json_file(ego_dir / "camera_params.json")
            ego_obj = ego_params.get("ego") if isinstance(ego_params.get("ego"), dict) else ego_params
            rgb_obj = (ego_obj.get("RGB") or ego_obj.get("rgb") or {}) if isinstance(ego_obj, dict) else {}
            rgb_dir = ego_dir / "RGB"
            video = _resolve_storage_file(episode_dir, "ego", "RGB", rgb_obj.get("storageFile"))
            if video is None:
                video = next((p for p in sorted(rgb_dir.glob("*")) if p.suffix.lower() in VIDEO_SUFFIXES), None)
            timestamp = _resolve_storage_file(episode_dir, "ego", "RGB", rgb_obj.get("timestampFile"))
            if timestamp is None:
                timestamp = next(iter(sorted(rgb_dir.glob("*.timestamps.csv"))), None)
            sources.append(
                ViewerSource(
                    source_id="ego:ego",
                    label="Pico + 眼动",
                    kind="ego",
                    camera="ego",
                    rgb_dir=rgb_dir if rgb_dir.is_dir() else None,
                    rgb_video=video,
                    rgb_timestamps=timestamp,
                    rgb_frame_map=_video_frame_map(timestamp),
                    rgb_metadata_map=_metadata_frame_map(timestamp),
                    decoded_rgb=temp_dir / "decoded" / "ego" / "RGB" if video else None,
                )
            )

        timestamp_rows = _csv_rows(episode_dir / "timestamps.csv")
        frames = []
        for index, row in enumerate(timestamp_rows):
            value = _integer(row.get("frame_index"), index)
            if value >= 0:
                frames.append(value)
        if not frames:
            counts: List[List[int]] = []
            for source in sources:
                if source.kind != "multiview" or source.rgb_dir is None:
                    continue
                numbered = [value for value in (_frame_number(path) for path in _images(source.rgb_dir)) if value is not None]
                if numbered:
                    counts.append(numbered)
                elif source.rgb_frame_map:
                    counts.append(sorted(source.rgb_frame_map))
            frames = min(counts, key=len) if counts else []
        if not frames:
            # ffmpeg output is numbered from zero.  ffprobe is deliberately done only
            # when timestamp metadata is absent.
            video = next((s.rgb_video for s in sources if s.kind == "multiview" and s.rgb_video), None)
            count = self._probe_frame_count(video) if video else 0
            frames = list(range(count))
        fps = 30.0
        for camera in params.values():
            if isinstance(camera, dict):
                rgb = camera.get("RGB") or camera.get("rgb")
                if isinstance(rgb, dict) and _float(rgb.get("fps"), 0.0) > 0:
                    fps = _float(rgb.get("fps"), 30.0)
                    break
        return sources, sorted(set(frames)), max(1.0, min(60.0, fps))

    def _decode_jobs(self, session: ViewerSession) -> List[Tuple[str, Path, Path, bool]]:
        rgb_jobs: List[Tuple[str, Path, Path, bool]] = []
        depth_jobs: List[Tuple[str, Path, Path, bool]] = []
        seen: set = set()
        for source in session.sources:
            if source.rgb_video and source.decoded_rgb:
                key = str(source.rgb_video)
                if key not in seen:
                    rgb_jobs.append((f"{source.label} RGB", source.rgb_video, source.decoded_rgb, False))
                    seen.add(key)
            if source.depth_video and source.decoded_depth:
                key = str(source.depth_video)
                if key not in seen:
                    depth_jobs.append((f"{source.label} Depth", source.depth_video, source.decoded_depth, True))
                    seen.add(key)
        # Decode any additional recorded containers too.  This mirrors the native
        # viewer's up-front decoded_pic preparation and prevents a later modality
        # from unexpectedly blocking on H265/MKV decoding.
        for video in sorted(p for p in session.episode_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES):
            key = str(video.resolve())
            if key in seen:
                continue
            rel = video.relative_to(session.episode_dir)
            depth = "depth" in {part.lower() for part in rel.parts} or "depth" in video.stem.lower()
            job = (str(rel), video, session.temp_dir / "decoded_extra" / rel.parent / video.stem, depth)
            (depth_jobs if depth else rgb_jobs).append(job)
            seen.add(key)
        # Match QC's per-camera scheduling: start every RGB camera decoder first.
        # The shared pool then admits Depth jobs as RGB workers become available,
        # instead of the previous RGB/Depth interleaving that only started about
        # half of the RGB cameras at once.
        return rgb_jobs + depth_jobs

    def _decode_video(self, session: ViewerSession, video: Path, output: Path, depth: bool) -> None:
        if session.cancel.is_set():
            raise ViewerError("解码已取消")
        output.mkdir(parents=True, exist_ok=True)
        pattern = output / ("%05d.png" if depth else "%05d.jpg")
        command = [session.ffmpeg, "-hide_banner", "-v", "error", "-threads", "0", "-y", "-i", str(video)]
        if depth:
            command.extend(["-pix_fmt", "gray16be"])
        else:
            command.extend(["-q:v", "2"])
        command.extend(["-start_number", "0", str(pattern)])
        try:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise ViewerError(f"找不到 ffmpeg：{session.ffmpeg}") from exc
        with session.lock:
            session.processes.append(process)
        try:
            while process.poll() is None:
                if session.cancel.wait(0.1):
                    process.terminate()
                    raise ViewerError("解码已取消")
            stderr = (process.stderr.read() if process.stderr else b"").decode("utf-8", errors="replace").strip()
            if process.returncode != 0:
                raise ViewerError(f"{video.name} 解码失败：{stderr[-800:] or 'ffmpeg error'}")
            if not any(output.iterdir()):
                raise ViewerError(f"{video.name} 未解出任何帧")
        finally:
            with session.lock:
                if process in session.processes:
                    session.processes.remove(process)

    def _source_has_rgb(self, source: ViewerSource, frames: Sequence[int]) -> bool:
        if not frames:
            return False
        return self._rgb_frame_path(source, frames[0], required=False) is not None

    def _rgb_frame_path(self, source: ViewerSource, frame: int, *, required: bool = True) -> Optional[Path]:
        video_frame = source.rgb_frame_map.get(frame, frame)
        candidates: List[Path] = []
        if source.decoded_rgb:
            candidates.extend(source.decoded_rgb / f"{video_frame:05d}{suffix}" for suffix in (".jpg", ".png"))
        if source.rgb_dir:
            for value in (frame, video_frame):
                candidates.extend(source.rgb_dir / f"{value:05d}{suffix}" for suffix in (".jpg", ".jpeg", ".png"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        if required:
            raise ViewerError(f"{source.label} 缺少第 {frame} 帧")
        return None

    def _depth_frame_path(self, source: ViewerSource, frame: int, *, required: bool = True) -> Optional[Path]:
        video_frame = source.depth_frame_map.get(frame, frame)
        candidates: List[Path] = []
        if source.decoded_depth:
            candidates.append(source.decoded_depth / f"{video_frame:05d}.png")
        if source.depth_dir:
            for value in (frame, video_frame):
                candidates.append(source.depth_dir / f"{value:05d}.png")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        if required:
            raise ViewerError(f"{source.label} 缺少第 {frame} 帧 Depth")
        return None

    def _probe_frame_count(self, video: Path) -> int:
        ffprobe = "ffprobe" if Path(self.ffmpeg).name == "ffmpeg" else str(Path(self.ffmpeg).with_name("ffprobe"))
        command = [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "json",
            str(video),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            stream = (json.loads(result.stdout).get("streams") or [{}])[0]
            return max(0, _integer(stream.get("nb_read_frames"), _integer(stream.get("nb_frames"), 0)))
        except Exception:
            return 0

    def _prepare_mode(self, session: ViewerSession, mode: str) -> None:
        state = session.modes[mode]
        try:
            if mode == "rgb":
                pass
            elif mode == "pico":
                self._prepare_pico(session, state)
            elif mode == "pointcloud":
                self._prepare_pointcloud(session, state)
            elif mode == "manomesh":
                self._prepare_manomesh(session, state)
            elif mode == "picohand":
                self._prepare_picohand(session, state)
            if session.cancel.is_set():
                return
            with session.lock:
                state.status = "ready"
                state.progress = state.total
                state.message = "准备完成"
                state.playable = True
                state.complete = True
        except Exception as exc:
            with session.lock:
                state.status = "failed"
                state.error = str(exc)
                state.message = "准备失败"
                state.complete = True

    def _prepare_pico(self, session: ViewerSession, state: ModeState) -> None:
        source = next((s for s in session.sources if s.kind == "ego"), None)
        if source is None or not self._source_has_rgb(source, session.frame_indices):
            raise ViewerError("该 Episode 没有 Pico RGB 数据")
        out_dir = session.temp_dir / "modes" / "pico"
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata = _ego_metadata_rows(session.episode_dir / "ego" / "metadata.csv")
        by_frame = {_integer(row.get("frame_index"), index): row for index, row in enumerate(metadata)}
        ego_params = self._ego_rgb_params(session.episode_dir / "ego")
        local_pose = self._ego_rgb_local_pose(session.episode_dir / "ego" / "camera.json")
        state.total = len(session.frame_indices)

        def render(frame: int) -> None:
            source_path = self._rgb_frame_path(source, frame)
            destination = out_dir / f"{frame:05d}.jpg"
            metadata_frame = source.rgb_metadata_map.get(frame, source.rgb_frame_map.get(frame, frame))
            self._render_pico_frame(source_path, destination, by_frame.get(metadata_frame, {}), ego_params, local_pose)

        with ThreadPoolExecutor(max_workers=min(self.max_decode_workers, max(1, len(session.frame_indices)))) as executor:
            futures = [executor.submit(render, frame) for frame in session.frame_indices]
            for future in as_completed(futures):
                future.result()
                with session.lock:
                    state.progress += 1
                    state.message = f"正在叠加眼动 {state.progress}/{state.total}"

    @staticmethod
    def _ego_rgb_params(ego_dir: Path) -> Dict[str, Any]:
        root = _json_file(ego_dir / "camera_params.json")
        for candidate in (root.get("ego"), root.get("pico"), root):
            if not isinstance(candidate, dict):
                continue
            rgb = candidate.get("RGB") or candidate.get("rgb")
            if isinstance(rgb, dict):
                return rgb
        return {}

    @staticmethod
    def _ego_undistort_intrinsic(rgb_params: Mapping[str, Any]) -> Mapping[str, Any]:
        direct = rgb_params.get("undistortIntrinsic") or rgb_params.get("undistort_intrinsic")
        if isinstance(direct, Mapping):
            return direct
        undistort = rgb_params.get("undistort")
        if isinstance(undistort, Mapping):
            nested = undistort.get("new_intrinsic")
            if isinstance(nested, Mapping):
                return nested
        return {}

    @classmethod
    def _ego_rgb_local_pose(cls, camera_json: Path) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        ext = _json_file(camera_json).get("extrinsics_head_to_rgb_camera")
        if not isinstance(ext, dict):
            return None
        pos_rh = (_float(ext.get("x")), _float(ext.get("y")), _float(ext.get("z")))
        rot_rh = (_float(ext.get("rx")), _float(ext.get("ry")), _float(ext.get("rz")), _float(ext.get("rw"), 1.0))
        # Native viewer: right-handed pose * RotateX(180), then convert to Unity.
        rotated = cls._quat_mul(rot_rh, (1.0, 0.0, 0.0, 0.0))
        return (pos_rh[0], pos_rh[1], -pos_rh[2]), cls._quat_norm((rotated[0], rotated[1], -rotated[2], -rotated[3]))

    @staticmethod
    def _quat_norm(q: Sequence[float]) -> Tuple[float, float, float, float]:
        n = math.sqrt(sum(float(value) ** 2 for value in q))
        if not math.isfinite(n) or n <= 1e-12:
            return (0.0, 0.0, 0.0, 1.0)
        return tuple(float(value) / n for value in q)  # type: ignore[return-value]

    @classmethod
    def _quat_mul(cls, a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float, float]:
        ax, ay, az, aw = cls._quat_norm(a)
        bx, by, bz, bw = cls._quat_norm(b)
        return cls._quat_norm((
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ))

    @classmethod
    def _quat_rotate(cls, q: Sequence[float], vector: Sequence[float]) -> Tuple[float, float, float]:
        x, y, z, w = cls._quat_norm(q)
        vx, vy, vz = (float(value) for value in vector)
        uv = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
        uuv = (y * uv[2] - z * uv[1], z * uv[0] - x * uv[2], x * uv[1] - y * uv[0])
        return tuple(v + 2.0 * w * u + 2.0 * uu for v, u, uu in zip((vx, vy, vz), uv, uuv))  # type: ignore[return-value]

    @classmethod
    def _project_gaze(
        cls,
        row: Mapping[str, Any],
        rgb_params: Mapping[str, Any],
        local_pose: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]],
        image_size: Tuple[int, int],
    ) -> Optional[Tuple[float, float]]:
        if not row or not _bool(row.get("gaze_valid")) or not _bool(row.get("xr_head_valid")) or local_pose is None:
            return None
        gaze = tuple(_float(row.get(f"gaze_world_direction_{axis}"), math.nan) for axis in "xyz")
        eye = tuple(_float(row.get(f"eye_pose_position_unity_{axis}"), math.nan) for axis in "xyz")
        head_pos = tuple(_float(row.get(f"xr_head_pos_{axis}"), math.nan) for axis in "xyz")
        head_rot = tuple(_float(row.get(f"xr_head_rot_{axis}"), 1.0 if axis == "w" else 0.0) for axis in "xyzw")
        if not all(math.isfinite(value) for value in gaze + eye + head_pos + head_rot):
            return None
        gaze_n = math.sqrt(sum(value * value for value in gaze))
        if gaze_n <= 1e-12:
            return None
        gaze = tuple(value / gaze_n for value in gaze)
        local_pos, local_rot = local_pose
        rotated_local = cls._quat_rotate(head_rot, local_pos)
        rgb_pos = tuple(head_pos[i] + rotated_local[i] for i in range(3))
        rgb_rot = cls._quat_mul(head_rot, local_rot)
        inverse = (-rgb_rot[0], -rgb_rot[1], -rgb_rot[2], rgb_rot[3])
        direction = cls._quat_rotate(inverse, gaze)
        origin = cls._quat_rotate(inverse, tuple(eye[i] - rgb_pos[i] for i in range(3)))
        if abs(direction[2]) <= 1e-9:
            return None
        t = (1.0 - origin[2]) / direction[2]
        if not math.isfinite(t) or t <= 0:
            return None
        point = tuple(origin[i] + direction[i] * t for i in range(3))
        if point[2] <= 1e-9:
            return None
        und = cls._ego_undistort_intrinsic(rgb_params)
        fx, fy, cx, cy = (_float(und.get(key)) for key in ("fx", "fy", "cx", "cy"))
        if fx <= 0 or fy <= 0:
            return None
        source_w = _integer(row.get("width"), _integer(rgb_params.get("width"), image_size[0]))
        source_h = _integer(row.get("height"), _integer(rgb_params.get("height"), image_size[1]))
        crop_w, crop_h = min(source_w, 1280), min(source_h, 960)
        crop_x, crop_y = (source_w - crop_w) / 2.0, (source_h - crop_h) / 2.0
        px = fx * (point[0] / point[2]) + cx - crop_x
        py = fy * (-point[1] / point[2]) + cy - crop_y
        return px * image_size[0] / max(1, crop_w), py * image_size[1] / max(1, crop_h)

    @classmethod
    def _render_pico_frame(
        cls,
        source: Path,
        destination: Path,
        row: Mapping[str, Any],
        rgb_params: Mapping[str, Any],
        local_pose: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]],
    ) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise ViewerError("Pico 眼动叠加需要 OpenCV 与 NumPy") from exc
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ViewerError(f"无法读取 Pico 帧：{source.name}")
        intr = rgb_params.get("intrinsic") or {}
        und = cls._ego_undistort_intrinsic(rgb_params)
        dist = rgb_params.get("distortion") or {}
        if isinstance(intr, dict) and isinstance(und, dict) and _float(intr.get("fx")) > 0 and _float(und.get("fx")) > 0:
            k = np.asarray([[intr.get("fx"), 0, intr.get("cx")], [0, intr.get("fy"), intr.get("cy")], [0, 0, 1]], dtype=np.float64)
            new_k = np.asarray([[und.get("fx"), 0, und.get("cx")], [0, und.get("fy"), und.get("cy")], [0, 0, 1]], dtype=np.float64)
            d = np.asarray([[_float(dist.get(f"k{i}"))] for i in range(1, 5)], dtype=np.float64)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), new_k, (image.shape[1], image.shape[0]), cv2.CV_16SC2)
            image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            crop_w, crop_h = min(image.shape[1], 1280), min(image.shape[0], 960)
            x0, y0 = (image.shape[1] - crop_w) // 2, (image.shape[0] - crop_h) // 2
            image = image[y0:y0 + crop_h, x0:x0 + crop_w].copy()
        pixel = cls._project_gaze(row, rgb_params, local_pose, (image.shape[1], image.shape[0]))
        if pixel is not None:
            inside = 0 <= pixel[0] < image.shape[1] and 0 <= pixel[1] < image.shape[0]
            x = max(0, min(image.shape[1] - 1, int(round(pixel[0]))))
            y = max(0, min(image.shape[0] - 1, int(round(pixel[1]))))
            color = (80, 240, 120) if inside else (0, 190, 255)
            radius = max(8, image.shape[1] // 120)
            cv2.circle(image, (x, y), radius, color, 2, cv2.LINE_AA)
            cv2.line(image, (max(0, x - 16), y), (min(image.shape[1] - 1, x + 16), y), color, 2, cv2.LINE_AA)
            cv2.line(image, (x, max(0, y - 16)), (x, min(image.shape[0] - 1, y + 16)), color, 2, cv2.LINE_AA)
        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise ViewerError(f"无法写入 Pico 帧：{destination.name}")

    def _prepare_pointcloud(self, session: ViewerSession, state: ModeState) -> None:
        """Reconstruct and preload rendered cloud canvases like the native viewer."""
        frames = list(session.frame_indices)
        if not frames:
            raise ViewerError("Episode 没有可重建的帧")
        sources = [source for source in session.sources if source.kind == "multiview"][:6]
        if not sources or any(self._depth_frame_path(source, frames[0], required=False) is None for source in sources):
            raise ViewerError("彩色融合点云需要六路已解码的 RGB 与 16 位 Depth")
        if not (session.episode_dir / "camera_params.json").is_file() or not (session.episode_dir / "extrinsics.json").is_file():
            raise ViewerError("彩色融合点云缺少 camera_params.json 或 extrinsics.json")
        state.total = len(frames)
        prebuffer_count = min(len(frames), 60)
        state.message = f"正在准备点云播放缓冲 0/{prebuffer_count}"
        out_dir = session.temp_dir / "modes" / "pointcloud"
        out_dir.mkdir(parents=True, exist_ok=True)

        def render(frame: int) -> int:
            if session.cancel.is_set():
                raise ViewerError("点云准备已取消")
            self._write_pointcloud_frame(session, frame, out_dir / f"{frame:05d}.bin")
            return frame

        # The native viewer renders cloud canvases on a preload worker and only
        # swaps already-rendered frames during playback.  Do the same here instead
        # of sending 180k JSON points to the browser on every tick.
        workers = min(self.max_decode_workers, 6, max(1, len(frames)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="viewer-cloud") as executor:
            for completed, frame in enumerate(executor.map(render, frames), start=1):
                with session.lock:
                    state.available_frames.append(frame)
                    state.progress = completed
                    if not state.playable and completed >= prebuffer_count:
                        state.playable = True
                        state.status = "ready"
                    state.message = (
                        f"可播放，后台继续准备 {completed}/{state.total}"
                        if state.playable
                        else f"正在准备点云播放缓冲 {completed}/{prebuffer_count}"
                    )

    def _write_pointcloud_frame(self, session: ViewerSession, frame: int, destination: Path) -> None:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise ViewerError("彩色融合点云需要 OpenCV 与 NumPy") from exc
        points = self._build_rgbd_color_cloud(session, int(frame), max_points=180_000)
        records = np.empty(len(points), dtype=np.dtype([("xyz", "<f4", (3,)), ("rgba", "u1", (4,))]))
        records["xyz"] = points[:, :3]
        records["rgba"][:, :3] = np.clip(np.rint(points[:, 3:6]), 0, 255).astype(np.uint8)
        records["rgba"][:, 3] = 255
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_bytes(struct.pack("<II", int(frame), len(records)) + records.tobytes())
        os.replace(temporary, destination)

    @staticmethod
    def _intrinsic(stream: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        intrinsic = stream.get("intrinsic") if isinstance(stream, Mapping) else None
        if not isinstance(intrinsic, Mapping):
            return None
        fx = _float(intrinsic.get("fx"))
        fy = _float(intrinsic.get("fy"))
        if fx <= 0.0 or fy <= 0.0:
            return None
        return fx, fy, _float(intrinsic.get("cx")), _float(intrinsic.get("cy"))

    def _build_rgbd_color_cloud(self, session: ViewerSession, frame: int, *, max_points: int) -> Any:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise ViewerError("彩色融合点云需要 OpenCV 与 NumPy") from exc

        params = _json_file(session.episode_dir / "camera_params.json")
        extrinsics = _json_file(session.episode_dir / "extrinsics.json")
        clouds = []
        step = 2
        max_depth_m = 6.0
        for source in [item for item in session.sources if item.kind == "multiview"][:6]:
            rgb_path = self._rgb_frame_path(source, frame)
            depth_path = self._depth_frame_path(source, frame)
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
                raise ViewerError(f"相机 {source.camera} 的 RGB/Depth 解码格式无效")

            camera_params = params.get(source.camera)
            camera_extrinsic = extrinsics.get(source.camera)
            if not isinstance(camera_params, Mapping) or not isinstance(camera_extrinsic, Mapping):
                raise ViewerError(f"相机 {source.camera} 缺少内参或外参")
            rgb_params = camera_params.get("RGB") or camera_params.get("rgb") or {}
            depth_params = camera_params.get("Depth") or camera_params.get("depth") or {}
            aligned = depth.shape[:2] == rgb.shape[:2]
            intrinsic = self._intrinsic(rgb_params if aligned else depth_params)
            if intrinsic is None:
                raise ViewerError(f"相机 {source.camera} 的点云内参无效")
            fx, fy, cx, cy = intrinsic

            # extrinsics.json stores world -> camera (Rcw, tcw).  The native
            # viewer explicitly inverts it before fusing camera points:
            # Rwc = Rcw^T, twc = -(Rwc * tcw).
            rotation_cw = np.asarray(camera_extrinsic.get("rotation"), dtype=np.float32)
            translation_cw = np.asarray(camera_extrinsic.get("translation"), dtype=np.float32).reshape(-1)
            if rotation_cw.shape != (3, 3) or translation_cw.size != 3:
                raise ViewerError(f"相机 {source.camera} 的世界外参无效")
            rotation_wc = rotation_cw.T
            translation_wc = -(rotation_wc @ translation_cw)

            yy, xx = np.mgrid[0 : depth.shape[0] : step, 0 : depth.shape[1] : step]
            depth_mm = depth[::step, ::step].astype(np.float32, copy=False)
            z = depth_mm * 0.001
            valid = (z >= 0.2) & (z <= max_depth_m)
            if not np.any(valid):
                continue
            xs = xx[valid].astype(np.float32, copy=False)
            ys = yy[valid].astype(np.float32, copy=False)
            zs = z[valid]
            camera_points = np.column_stack(((xs - cx) * zs / fx, (ys - cy) * zs / fy, zs))
            world = camera_points @ rotation_wc.T + translation_wc.reshape(1, 3)

            if aligned:
                bgr = rgb[::step, ::step][valid]
                colors = bgr[:, ::-1]
            else:
                colors = self._map_depth_colors(
                    cv2,
                    np,
                    xs,
                    ys,
                    depth_mm[valid],
                    rgb,
                    camera_params,
                )
            finite = np.isfinite(world).all(axis=1)
            if not np.any(finite):
                continue
            clouds.append(np.column_stack((world[finite], colors[finite].astype(np.float32))))

        if not clouds:
            raise ViewerError(f"第 {frame} 帧没有有效 RGB-D 点")
        fused = np.concatenate(clouds, axis=0)
        if fused.shape[0] > max_points:
            keep = np.linspace(0, fused.shape[0] - 1, max_points, dtype=np.int64)
            fused = fused[keep]
        return fused.astype(np.float32, copy=False)

    @staticmethod
    def _map_depth_colors(cv2: Any, np: Any, xs: Any, ys: Any, depth_mm: Any, rgb: Any, camera_params: Mapping[str, Any]) -> Any:
        mapping = camera_params.get("rgb_to_depth") or {}
        if not isinstance(mapping, Mapping):
            return np.full((len(xs), 3), 210, dtype=np.uint8)
        depth_intrinsic = mapping.get("depth_intrinsic") or {}
        rgb_intrinsic = mapping.get("rgb_intrinsic") or {}
        d2c = mapping.get("d2c_extrinsic") or {}
        try:
            kd = np.array(
                [[depth_intrinsic["fx"], 0.0, depth_intrinsic["cx"]], [0.0, depth_intrinsic["fy"], depth_intrinsic["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            kr = np.array(
                [[rgb_intrinsic["fx"], 0.0, rgb_intrinsic["cx"]], [0.0, rgb_intrinsic["fy"], rgb_intrinsic["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            dd = mapping.get("depth_distortion") or {}
            rd = mapping.get("rgb_distortion") or {}
            depth_dist = np.array([dd.get("k1", 0), dd.get("k2", 0), dd.get("p1", 0), dd.get("p2", 0), dd.get("k3", 0), dd.get("k4", 0), dd.get("k5", 0), dd.get("k6", 0)], dtype=np.float64)
            rgb_dist = np.array([rd.get("k1", 0), rd.get("k2", 0), rd.get("p1", 0), rd.get("p2", 0), rd.get("k3", 0), rd.get("k4", 0), rd.get("k5", 0), rd.get("k6", 0)], dtype=np.float64)
            normalized = cv2.undistortPoints(np.column_stack((xs, ys)).reshape(-1, 1, 2), kd, depth_dist).reshape(-1, 2)
            points_mm = np.column_stack((normalized[:, 0] * depth_mm, normalized[:, 1] * depth_mm, depth_mm))
            rotation = np.asarray(d2c.get("rotation"), dtype=np.float64)
            translation = np.asarray(d2c.get("translation"), dtype=np.float64).reshape(1, 3)
            color_points = points_mm @ rotation.T + translation
            projected, _ = cv2.projectPoints(color_points, np.zeros(3), np.zeros(3), kr, rgb_dist)
            uv = np.rint(projected.reshape(-1, 2)).astype(np.int32)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < rgb.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < rgb.shape[0])
            colors = np.full((len(xs), 3), 210, dtype=np.uint8)
            colors[inside] = rgb[uv[inside, 1], uv[inside, 0]][:, ::-1]
            return colors
        except (KeyError, TypeError, ValueError, cv2.error):
            return np.full((len(xs), 3), 210, dtype=np.uint8)

    @staticmethod
    def _read_color_ply(path: Path, max_points: int) -> List[List[float]]:
        with path.open("rb") as handle:
            header_lines: List[str] = []
            while True:
                line = handle.readline()
                if not line:
                    raise ViewerError(f"PLY 头损坏：{path.name}")
                decoded = line.decode("ascii", errors="replace").strip()
                header_lines.append(decoded)
                if decoded == "end_header":
                    break
            fmt = next((line.split()[1] for line in header_lines if line.startswith("format ")), "")
            count = 0
            properties: List[Tuple[str, str]] = []
            in_vertex = False
            for line in header_lines:
                parts = line.split()
                if parts[:2] == ["element", "vertex"] and len(parts) >= 3:
                    count, in_vertex = int(parts[2]), True
                elif parts[:1] == ["element"]:
                    in_vertex = False
                elif in_vertex and parts[:1] == ["property"] and len(parts) == 3:
                    properties.append((parts[1], parts[2]))
            if count <= 0:
                return []
            stride = max(1, int(math.ceil(count / max(1, max_points))))
            names = [name for _kind, name in properties]
            wanted = {name: index for index, name in enumerate(names)}

            def compact(values: Sequence[float]) -> List[float]:
                x = float(values[wanted.get("x", 0)])
                y = float(values[wanted.get("y", 1)])
                z = float(values[wanted.get("z", 2)])
                r = float(values[wanted.get("red", wanted.get("r", -1))]) if ("red" in wanted or "r" in wanted) else 210.0
                g = float(values[wanted.get("green", wanted.get("g", -1))]) if ("green" in wanted or "g" in wanted) else 210.0
                b = float(values[wanted.get("blue", wanted.get("b", -1))]) if ("blue" in wanted or "b" in wanted) else 210.0
                return [round(x, 5), round(y, 5), round(z, 5), int(r), int(g), int(b)]

            points: List[List[float]] = []
            if fmt == "ascii":
                for index in range(count):
                    values = handle.readline().split()
                    if index % stride == 0 and len(values) >= len(properties):
                        points.append(compact([float(value) for value in values]))
                return points
            if fmt != "binary_little_endian":
                raise ViewerError(f"不支持的 PLY 格式：{fmt}")
            formats = {
                "char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h", "int16": "h",
                "ushort": "H", "uint16": "H", "int": "i", "int32": "i", "uint": "I", "uint32": "I",
                "float": "f", "float32": "f", "double": "d", "float64": "d",
            }
            try:
                record = struct.Struct("<" + "".join(formats[kind] for kind, _name in properties))
            except KeyError as exc:
                raise ViewerError(f"PLY 属性类型不支持：{exc}") from exc
            for index in range(count):
                raw = handle.read(record.size)
                if len(raw) != record.size:
                    break
                if index % stride == 0:
                    points.append(compact(record.unpack(raw)))
            return points

    def _prepare_manomesh(self, session: ViewerSession, state: ModeState) -> None:
        if not self.mano_toolkit_root or not self.mano_model_dir:
            raise ViewerError("后端未配置 MANO toolkit/model 路径")
        cameras = [s.camera for s in session.sources if s.kind == "multiview"][:6]
        frames = list(session.frame_indices)
        if not cameras or not frames:
            raise ViewerError("MANO mesh 缺少相机或帧")
        if not (session.episode_dir / "optimized_pose").is_dir():
            raise ViewerError("Episode 缺少 optimized_pose")
        state.total = len(frames)
        output_dir = session.temp_dir / "modes" / "manomesh"
        rgb_cache_dir = session.temp_dir / "mesh_rgb"
        for camera in cameras:
            source = next(item for item in session.sources if item.kind == "multiview" and item.camera == camera)
            camera_cache = rgb_cache_dir / camera
            camera_cache.mkdir(parents=True, exist_ok=True)
            for frame in frames:
                rgb_path = self._rgb_frame_path(source, frame)
                target = camera_cache / f"{frame:05d}{rgb_path.suffix.lower()}"
                if not target.exists():
                    try:
                        target.symlink_to(rgb_path)
                    except OSError:
                        shutil.copyfile(rgb_path, target)
        self._run_mesh_renderer(
            session,
            state,
            label="MANO mesh",
            cameras=cameras,
            frames=frames,
            rgb_cache_dir=rgb_cache_dir,
            output_dir=output_dir,
            available_dir=output_dir,
            request_name="manomesh_request.json",
        )

    def _prepare_picohand(self, session: ViewerSession, state: ModeState) -> None:
        if not self.mano_toolkit_root or not self.mano_model_dir:
            raise ViewerError("后端未配置 MANO toolkit/model 路径")
        source = next((item for item in session.sources if item.kind == "ego"), None)
        frames = list(session.frame_indices)
        if source is None or not frames or not self._source_has_rgb(source, frames):
            raise ViewerError("该 Episode 没有完整的 Pico RGB 数据")
        if not (session.episode_dir / "optimized_pose").is_dir():
            raise ViewerError("Episode 缺少 optimized_pose")
        if not any((session.episode_dir / name).is_file() for name in ("ego_pose.json", "ego_extrinsic.json")):
            raise ViewerError("Episode 缺少 ego_pose.json/ego_extrinsic.json，无法投影 Pico 手部 Pose")

        state.total = len(frames)
        rgb_cache_dir = session.temp_dir / "picohand_rgb"
        camera_cache = rgb_cache_dir / "ego"
        camera_cache.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            rgb_path = self._rgb_frame_path(source, frame)
            target = camera_cache / f"{frame:05d}{rgb_path.suffix.lower()}"
            if not target.exists():
                try:
                    target.symlink_to(rgb_path)
                except OSError:
                    shutil.copyfile(rgb_path, target)

        preview_dir = session.temp_dir / "modes" / "picohand"
        self._run_mesh_renderer(
            session,
            state,
            label="Pico 手部 Pose",
            cameras=["ego"],
            frames=frames,
            rgb_cache_dir=rgb_cache_dir,
            output_dir=session.temp_dir / "modes" / "picohand_full",
            available_dir=preview_dir,
            request_name="picohand_request.json",
            preview_output_dir=preview_dir,
        )

    def _run_mesh_renderer(
        self,
        session: ViewerSession,
        state: ModeState,
        *,
        label: str,
        cameras: Sequence[str],
        frames: Sequence[int],
        rgb_cache_dir: Path,
        output_dir: Path,
        available_dir: Path,
        request_name: str,
        preview_output_dir: Optional[Path] = None,
    ) -> None:
        request = {
            "episode_dir": str(session.episode_dir),
            "rgb_cache_dir": str(rgb_cache_dir),
            "output_dir": str(output_dir),
            "cameras": list(cameras),
            "frames": list(frames),
            "mano_toolkit_root": str(self.mano_toolkit_root),
            "mano_model_dir": str(self.mano_model_dir),
            "render_factor": 0.5,
            "workers": min(6, max(1, os.cpu_count() or 1)),
            "prefer_integrated_gpu": False,
            # Same producer/consumer contract as QC: the renderer may start before
            # every source frame is published and waits for the decoder output.
            "source_wait_seconds": 300.0,
        }
        if preview_output_dir is not None:
            request["preview_output_dir"] = str(preview_output_dir)
            request["ego_preview_max_width"] = 960
        request_path = session.temp_dir / request_name
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        project_root = Path(__file__).resolve().parents[1]
        process = subprocess.Popen(
            [self.mesh_python, "-m", "src.qc.mesh_renderer", "--request", str(request_path)],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        with session.lock:
            session.processes.append(process)
            if os.name != "nt":
                session.process_groups.add(process.pid)
        try:
            prebuffer_count = min(len(frames), self.mesh_prebuffer_frames)
            prebuffer = set(frames[:prebuffer_count])
            while process.poll() is None:
                if session.cancel.wait(0.25):
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    raise ViewerError(f"{label} 准备已取消")
                available = [
                    frame
                    for frame in frames
                    if all((available_dir / camera / f"{frame:05d}.jpg").is_file() for camera in cameras)
                ]
                with session.lock:
                    state.available_frames = available
                    state.progress = len(available)
                    if not state.playable and prebuffer.issubset(available):
                        state.playable = True
                        state.status = "ready"
                    state.message = (
                        f"可播放，后台继续渲染 {len(available)}/{state.total}"
                        if state.playable
                        else f"正在准备播放缓冲 {len(available)}/{prebuffer_count}"
                    )
            stderr = (process.stderr.read() if process.stderr else b"").decode("utf-8", errors="replace").strip()
            if process.returncode != 0:
                raise ViewerError(f"{label} 渲染失败：{stderr[-1200:] or 'renderer error'}")
            available = [
                frame
                for frame in frames
                if all((available_dir / camera / f"{frame:05d}.jpg").is_file() for camera in cameras)
            ]
            if len(available) != len(frames):
                raise ViewerError(f"{label} 渲染结束但缺少 {len(frames) - len(available)} 个同步帧")
            with session.lock:
                state.available_frames = available
                state.progress = len(available)
                state.playable = True
        finally:
            with session.lock:
                if process in session.processes:
                    session.processes.remove(process)
                session.process_groups.discard(process.pid)


def render_viewer_page(episode_id: str) -> str:
    """Self-contained viewer UI; no CDN is required on capture/NAS networks."""
    encoded_id = (
        json.dumps(str(episode_id), ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Episode Viewer</title><style>
:root{{--bg:#0b1015;--panel:#121a22;--line:#263541;--text:#eef5f8;--muted:#8fa1ad;--accent:#31b58b;--danger:#ff6b78}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}}
.app{{display:grid;grid-template-columns:236px 1fr;height:100vh}}aside{{background:var(--panel);border-right:1px solid var(--line);padding:22px 14px;display:flex;flex-direction:column;gap:8px}}
h1{{font-size:18px;margin:0 8px 18px}}.sub{{color:var(--muted);font:12px ui-monospace;margin:0 8px 14px;overflow:hidden;text-overflow:ellipsis}}
.mode{{appearance:none;text-align:left;border:1px solid transparent;border-radius:8px;background:transparent;color:var(--muted);padding:12px;cursor:pointer;font-size:14px}}
.mode:hover:not(:disabled){{background:#17232c;color:var(--text)}}.mode.active{{background:#173a34;border-color:#245d50;color:#bff4e3}}.mode:disabled{{opacity:.35;cursor:not-allowed}}
.spacer{{flex:1}}#sessionState{{color:var(--muted);font-size:12px;line-height:1.5;padding:10px}}main{{min-width:0;min-height:0;height:100%;overflow:hidden;display:grid;grid-template-rows:58px minmax(0,1fr) 74px}}
header{{display:flex;align-items:center;gap:14px;padding:0 20px;border-bottom:1px solid var(--line);background:#0e151b}}header strong{{font-size:15px}}#modeStatus{{color:var(--muted);font-size:13px}}#manoSource{{display:none;margin-left:auto;padding:6px 10px;border:1px solid #405563;border-radius:999px;background:#17232c;color:#cbd9df;font-size:12px}}#manoSource.corrected{{border-color:#28745e;background:#173a34;color:#bff4e3}}
#stage{{position:relative;min-height:0;padding:14px;background:#080c10;overflow:hidden}}#grid{{height:100%;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:repeat(2,minmax(0,1fr));gap:8px}}
.tile{{position:relative;min-width:0;min-height:0;background:#101820;border:1px solid #1e2a34;border-radius:7px;overflow:hidden;display:flex;align-items:center;justify-content:center}}.tile img.frame{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:0;visibility:hidden}}.tile img.frame.active{{opacity:1;visibility:visible}}.tile label{{position:absolute;z-index:2;left:8px;top:7px;background:#0009;padding:3px 7px;border-radius:4px;font:12px ui-monospace}}
#single{{position:relative;width:100%;height:100%;display:none;align-items:center;justify-content:center}}#single img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:0;visibility:hidden}}#single img.active{{opacity:1;visibility:visible}}#cloud{{width:100%;height:100%;display:none;cursor:grab}}#cloud.panning{{cursor:move}}#cloud:active{{cursor:grabbing}}#cloudTools{{position:absolute;right:26px;top:50%;transform:translateY(-50%);display:none;z-index:3;flex-direction:column;gap:8px}}#cloudTools button{{width:42px;height:42px;border:1px solid #45606e;border-radius:8px;background:#15232cdd;color:#fff;font-size:22px;cursor:pointer}}#cloudHint{{position:absolute;left:26px;top:24px;display:none;z-index:3;background:#0009;padding:6px 10px;border-radius:5px;color:#d9e5eb;font-size:12px}}
#loading{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#080c10e8;z-index:4}}.card{{width:min(520px,80%);text-align:center}}.track{{height:7px;background:#202b34;border-radius:10px;overflow:hidden;margin:18px 0}}#bar{{height:100%;width:0;background:var(--accent);transition:width .2s}}#loadError{{color:var(--danger);white-space:pre-wrap}}
footer{{border-top:1px solid var(--line);display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;padding:12px 20px;background:#0e151b}}button.control{{border:1px solid var(--line);background:#17212a;color:var(--text);padding:8px 14px;border-radius:6px;cursor:pointer}}#timelineWrap{{position:relative;height:30px;display:flex;align-items:center}}#timeline{{position:relative;z-index:2;width:100%;accent-color:var(--accent);margin:0}}#badRanges{{position:absolute;z-index:3;pointer-events:none;left:0;right:0;top:1px;height:6px;display:none}}#badRanges span{{position:absolute;height:100%;min-width:3px;border-radius:3px;background:#ff5365;box-shadow:0 0 5px #ff536599}}#counter{{font:12px ui-monospace;color:var(--muted);min-width:110px;text-align:right}}
@media(max-width:850px){{.app{{grid-template-columns:190px 1fr}}#grid{{grid-template-columns:repeat(2,1fr);grid-template-rows:repeat(3,1fr)}}}}
</style></head><body><div class="app"><aside><h1>Episode Viewer</h1><div class="sub" id="episodeLabel"></div>
<button class="mode active" data-mode="rgb" disabled>6 路 RGB</button><button class="mode" data-mode="pico" disabled>Pico + 眼动</button>
<button class="mode" data-mode="pointcloud" disabled>彩色融合点云</button><button class="mode" data-mode="manomesh" disabled>MANO mesh 视频</button>
<button class="mode" data-mode="picohand" disabled>Pico 手部 Pose</button>
<div class="spacer"></div><div id="sessionState">正在创建临时会话…</div></aside><main><header><strong id="title">6 路 RGB</strong><span id="modeStatus"></span><span id="manoSource"></span></header>
<div id="stage"><div id="grid"></div><div id="single"><img class="active" alt=""><img alt=""></div><canvas id="cloud"></canvas><div id="cloudHint">左键拖动旋转 · 右键拖动平移</div><div id="cloudTools"><button id="zoomIn" title="放大">＋</button><button id="zoomOut" title="缩小">－</button><button id="resetView" title="重置视角" style="font-size:13px">重置</button></div><div id="loading"><div class="card"><div id="loadText">正在扫描 Episode 视频…</div><div class="track"><div id="bar"></div></div><div id="loadError"></div></div></div></div>
<footer><button class="control" id="play">播放</button><div id="timelineWrap"><div id="badRanges"></div><input id="timeline" type="range" min="0" max="0" value="0"></div><span id="counter">0 / 0</span></footer></main></div>
<script>
const episodeId={encoded_id};let sessionId=null,info=null,mode='rgb',framePos=0,playing=false,timer=null,closed=false,gridRenderToken=0,lastGridKey='',pendingGridKey='',gridSignature='',imageCacheEpoch=0,prefetchDesired=-1,prefetchRunning=false;const frameCache=new Map();
const $=s=>document.querySelector(s), buttons=[...document.querySelectorAll('.mode')];$('#episodeLabel').textContent=episodeId;
async function api(url,options={{}}){{const r=await fetch(url,options);const data=await r.json().catch(()=>({{}}));if(!r.ok)throw new Error(data.error||r.statusText);return data}}
function showLoading(text,pct=0,error=''){{$('#loading').style.display='flex';$('#loadText').textContent=text;$('#bar').style.width=`${{Math.max(0,Math.min(100,pct))}}%`;$('#loadError').textContent=error}}
function hideLoading(){{$('#loading').style.display='none'}}
async function start(){{try{{const data=await api(`/api/v1/viewer/episodes/${{encodeURIComponent(episodeId)}}/sessions`,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:'{{}}'}});sessionId=data.session_id;poll()}}catch(e){{showLoading('无法启动 Viewer',0,e.message)}}}}
async function poll(){{if(!sessionId||closed)return;try{{info=await api(`/api/v1/viewer/sessions/${{sessionId}}`);renderState();const active=info.modes[mode];if(info.state==='preparing'||active&&(!active.complete&&(active.status==='preparing'||active.playable)))setTimeout(poll,250)}}catch(e){{showLoading('Viewer 会话错误',0,e.message)}}}}
function renderState(){{const d=info.decode||{{completed:0,total:0}};$('#sessionState').textContent=`临时解码：${{d.completed}} / ${{d.total}}\n离开页面后自动清理`;if(info.state==='failed')return showLoading('基础视频解码失败',0,info.error);if(info.state!=='ready')return showLoading(`正在并发解码全部视频 ${{d.completed}} / ${{d.total}}`,d.total?100*d.completed/d.total:5);
buttons.forEach(b=>b.disabled=false);const m=info.modes[mode];if(m.status==='failed')return showLoading('模态准备失败',0,m.error);if((m.status==='preparing'&&!m.playable)||(m.status==='idle'&&mode!=='rgb')){{const pct=m.total?100*m.progress/m.total:8;showLoading(m.message||'正在准备模态数据',pct);return}}hideLoading();$('#modeStatus').textContent=m.complete?'':m.message;setupFrames();renderFrame()}}
function setupFrames(){{const frames=info.frames||[];$('#timeline').max=Math.max(0,frames.length-1);framePos=Math.min(framePos,Math.max(0,frames.length-1));$('#timeline').value=framePos;$('#counter').textContent=frames.length?`${{frames[framePos]}} · ${{framePos+1}} / ${{frames.length}}`:'0 / 0';renderManoMeta()}}
function renderManoMeta(){{const meta=info&&info.mano||{{}},badge=$('#manoSource'),ranges=$('#badRanges'),show=mode==='manomesh'||mode==='picohand';badge.style.display=show?'inline-block':'none';if(show){{badge.textContent=meta.source_label||'初始 AutoLabel 结果';badge.classList.toggle('corrected',meta.source==='corrected_3d')}}const values=info&&info.frames||[],bad=meta.qc_completed&&Array.isArray(meta.bad_ranges)?meta.bad_ranges:[];ranges.style.display=show&&bad.length?'block':'none';const signature=show?JSON.stringify([mode,values.length,bad]):'';if(ranges.dataset.signature===signature)return;ranges.dataset.signature=signature;ranges.innerHTML='';const denominator=Math.max(1,values.length-1);for(const segment of bad){{let first=values.findIndex(frame=>frame>=segment.start_frame),last=-1;for(let i=values.length-1;i>=0;i--)if(values[i]<=segment.end_frame){{last=i;break}}if(first<0||last<first)continue;const mark=document.createElement('span');mark.style.left=`${{100*first/denominator}}%`;mark.style.width=`${{Math.max(.35,100*(last-first)/denominator)}}%`;mark.title=`QC 坏帧 ${{segment.start_frame}}–${{segment.end_frame}}`;ranges.appendChild(mark)}}}}
async function choose(next){{if(!info||info.state!=='ready')return;mode=next;playing=false;clearTimeout(timer);gridRenderToken++;imageCacheEpoch++;frameCache.clear();prefetchDesired=-1;lastGridKey='';pendingGridKey='';$('#play').textContent='播放';buttons.forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));$('#title').textContent={{rgb:'6 路 RGB',pico:'Pico + 眼动',pointcloud:'彩色融合点云',manomesh:'MANO mesh 视频',picohand:'Pico 手部 Pose'}}[mode];const m=info.modes[mode];if(m.status==='idle'){{await api(`/api/v1/viewer/sessions/${{sessionId}}/modes/${{mode}}`,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:'{{}}'}});poll()}}else renderState()}}
buttons.forEach(b=>b.onclick=()=>choose(b.dataset.mode));
function mediaUrl(source,frame){{return `/api/v1/viewer/sessions/${{sessionId}}/media/${{mode}}/${{encodeURIComponent(source)}}/${{frame}}`}}
function frameAvailable(frame){{if(mode!=='manomesh'&&mode!=='picohand'&&mode!=='pointcloud')return true;const m=info&&info.modes&&info.modes[mode];return !!m&&(m.complete||Array.isArray(m.available_frames)&&m.available_frames.includes(frame))}}
function ensureSixGrid(sources){{const signature=sources.map(s=>s.id).join('|');if(signature===gridSignature)return;gridSignature=signature;lastGridKey='';pendingGridKey='';$('#grid').innerHTML=sources.map(s=>`<div class="tile"><img class="frame active" alt=""><img class="frame" alt=""><label>${{s.label}}</label></div>`).join('')}}
function preloadImage(url){{return new Promise((resolve,reject)=>{{const image=new Image();image.onload=async()=>{{try{{if(image.decode)await image.decode()}}catch(_e){{}}resolve(image)}};image.onerror=()=>reject(new Error('画面加载失败'));image.src=url}})}}
function cachedImage(url){{let promise=frameCache.get(url);if(!promise){{promise=preloadImage(url).catch(error=>{{frameCache.delete(url);throw error}});frameCache.set(url,promise)}}return promise}}
function cachedCloud(url){{let promise=frameCache.get(url);if(!promise){{promise=fetch(url).then(r=>{{if(!r.ok)throw new Error('点云帧加载失败');return r.arrayBuffer()}}).catch(error=>{{frameCache.delete(url);throw error}});frameCache.set(url,promise)}}return promise}}
function frameUrlsAt(position){{const frames=info.frames||[],frame=frames[(position+frames.length)%frames.length];if(mode==='rgb'||mode==='manomesh'){{const sources=info.sources.filter(s=>s.kind==='multiview').slice(0,6);return sources.map(s=>mediaUrl(mode==='rgb'?s.id:s.camera,frame))}}return [mediaUrl(mode==='pointcloud'?'cloud':'ego',frame)]}}
async function requestFramePrefetch(){{if(!['rgb','pico','manomesh','picohand','pointcloud'].includes(mode)||!info||!info.frames.length)return;prefetchDesired=framePos;if(prefetchRunning)return;prefetchRunning=true;const epoch=imageCacheEpoch;try{{while(epoch===imageCacheEpoch){{const start=prefetchDesired;for(let i=1;i<=24&&epoch===imageCacheEpoch;i++){{const position=(start+i)%info.frames.length,frame=info.frames[position];if(!frameAvailable(frame))break;try{{await Promise.all(frameUrlsAt(position).map(mode==='pointcloud'?cachedCloud:cachedImage))}}catch(_e){{break}}}}const keep=new Set();for(let i=-2;i<=24;i++)frameUrlsAt((prefetchDesired+i+info.frames.length)%info.frames.length).forEach(url=>keep.add(url));for(const url of frameCache.keys())if(!keep.has(url))frameCache.delete(url);if(start===prefetchDesired)break}}}}finally{{prefetchRunning=false}}}}
async function renderSix(frame){{const sources=info.sources.filter(s=>s.kind==='multiview').slice(0,6);ensureSixGrid(sources);const key=`${{mode}}:${{frame}}`;if(key===lastGridKey||key===pendingGridKey){{requestFramePrefetch();return true}}pendingGridKey=key;const token=++gridRenderToken,urls=sources.map(s=>mediaUrl(mode==='rgb'?s.id:s.camera,frame));try{{await Promise.all(urls.map(cachedImage));if(token!==gridRenderToken)return false;const tiles=[...document.querySelectorAll('#grid .tile')],backs=tiles.map(tile=>tile.querySelector('img.frame:not(.active)'));backs.forEach((img,i)=>img.src=urls[i]);await Promise.all(backs.map(async img=>{{try{{if(img.decode)await img.decode()}}catch(_e){{}}}}));if(token!==gridRenderToken)return false;tiles.forEach((tile,i)=>{{tile.querySelectorAll('img.frame').forEach(img=>img.classList.remove('active'));backs[i].classList.add('active')}});lastGridKey=key;requestFramePrefetch();return true}}finally{{if(pendingGridKey===key)pendingGridKey=''}}}}
async function renderSingle(frame,source){{const key=`${{mode}}:${{frame}}`;if(key===lastGridKey||key===pendingGridKey){{requestFramePrefetch();return true}}pendingGridKey=key;const token=++gridRenderToken,url=mediaUrl(source,frame);try{{await cachedImage(url);if(token!==gridRenderToken)return false;const images=[...document.querySelectorAll('#single img')],back=images.find(img=>!img.classList.contains('active'));back.src=url;try{{if(back.decode)await back.decode()}}catch(_e){{}}if(token!==gridRenderToken)return false;images.forEach(img=>img.classList.remove('active'));back.classList.add('active');lastGridKey=key;requestFramePrefetch();return true}}finally{{if(pendingGridKey===key)pendingGridKey=''}}}}
async function renderFrame(){{if(!info||!info.frames.length)return false;const frame=info.frames[framePos];setupFrames();$('#grid').style.display=mode==='rgb'||mode==='manomesh'?'grid':'none';$('#single').style.display=mode==='pico'||mode==='picohand'?'flex':'none';$('#cloud').style.display=mode==='pointcloud'?'block':'none';$('#cloudTools').style.display=mode==='pointcloud'?'flex':'none';$('#cloudHint').style.display=mode==='pointcloud'?'block':'none';
if(!frameAvailable(frame)){{$('#modeStatus').textContent='后台渲染中，正在缓冲该同步帧…';return false}}
if(mode==='rgb'||mode==='manomesh'){{try{{return await renderSix(frame)}}catch(e){{$('#modeStatus').textContent=e.message;return false}}}}
else if(mode==='pico'||mode==='picohand'){{try{{return await renderSingle(frame,'ego')}}catch(e){{$('#modeStatus').textContent=e.message;return false}}}}else{{try{{return await renderCloudFrame(frame)}}catch(e){{$('#modeStatus').textContent=e.message;return false}}}}}}
$('#timeline').oninput=e=>{{framePos=Number(e.target.value);renderFrame()}};
function schedulePlay(delay){{clearTimeout(timer);if(playing)timer=setTimeout(playTick,delay)}}
async function playTick(){{if(!playing||!info||!info.frames.length)return;const started=performance.now(),previous=framePos,next=(framePos+1)%info.frames.length,frame=info.frames[next];if(!frameAvailable(frame)){{$('#modeStatus').textContent='后台渲染中，正在缓冲下一同步帧…';schedulePlay(30);return}}framePos=next;const shown=await renderFrame();if(!shown){{framePos=previous;setupFrames();schedulePlay(30);return}}schedulePlay(Math.max(0,1000/(info.fps||30)-(performance.now()-started)))}}
$('#play').onclick=()=>{{playing=!playing;$('#play').textContent=playing?'暂停':'播放';clearTimeout(timer);if(playing)schedulePlay(0)}};
const canvas=$('#cloud'),gl=canvas.getContext('webgl',{{antialias:false,alpha:false}});let cloudProgram=null,cloudBuffer=null,cloudCount=0,yaw=0,pitch=0,distance=1.5,target=[0,0,1],dragMode='',lastX=0,lastY=0;
function makeShader(type,source){{const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(shader));return shader}}
function initCloudGL(){{if(!gl)throw new Error('浏览器不支持 WebGL 点云显示');if(cloudProgram)return;const vs=makeShader(gl.VERTEX_SHADER,`attribute vec3 aPosition;attribute vec4 aColor;uniform vec3 uRight,uUp,uForward,uCamera;uniform vec2 uViewport;uniform float uPointSize;varying vec4 vColor;void main(){{vec3 v=aPosition-uCamera;float x=dot(v,uRight),y=dot(v,uUp),z=dot(v,uForward);float n=.05,f=20.;gl_Position=vec4(1800.*x/uViewport.x,1800.*y/uViewport.y,((f+n)/(f-n))*z-(2.*f*n/(f-n)),z);gl_PointSize=uPointSize;vColor=aColor;}}`),fs=makeShader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec4 vColor;void main(){{gl_FragColor=vColor;}}`);cloudProgram=gl.createProgram();gl.attachShader(cloudProgram,vs);gl.attachShader(cloudProgram,fs);gl.linkProgram(cloudProgram);if(!gl.getProgramParameter(cloudProgram,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(cloudProgram));cloudBuffer=gl.createBuffer();gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LESS);gl.clearColor(0,0,0,1)}}
function cloudBasis(){{const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),forward=[sy*cp,-sp,cy*cp],right=[-forward[2],0,forward[0]],rn=Math.hypot(...right);right[0]/=rn;right[2]/=rn;const up=[forward[1]*right[2]-forward[2]*right[1],forward[2]*right[0]-forward[0]*right[2],forward[0]*right[1]-forward[1]*right[0]],camera=target.map((v,i)=>v-forward[i]*distance);return{{right,up,forward,camera}}}}
function drawCloudGL(){{if(!gl||!cloudProgram||!cloudCount)return;const rect=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1,w=Math.max(1,Math.round(rect.width*d)),h=Math.max(1,Math.round(rect.height*d));if(canvas.width!==w||canvas.height!==h){{canvas.width=w;canvas.height=h}}gl.viewport(0,0,w,h);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(cloudProgram);gl.bindBuffer(gl.ARRAY_BUFFER,cloudBuffer);const pos=gl.getAttribLocation(cloudProgram,'aPosition'),color=gl.getAttribLocation(cloudProgram,'aColor');gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,3,gl.FLOAT,false,16,0);gl.enableVertexAttribArray(color);gl.vertexAttribPointer(color,4,gl.UNSIGNED_BYTE,true,16,12);const b=cloudBasis();gl.uniform3fv(gl.getUniformLocation(cloudProgram,'uRight'),b.right);gl.uniform3fv(gl.getUniformLocation(cloudProgram,'uUp'),b.up);gl.uniform3fv(gl.getUniformLocation(cloudProgram,'uForward'),b.forward);gl.uniform3fv(gl.getUniformLocation(cloudProgram,'uCamera'),b.camera);gl.uniform2f(gl.getUniformLocation(cloudProgram,'uViewport'),rect.width,rect.height);gl.uniform1f(gl.getUniformLocation(cloudProgram,'uPointSize'),Math.max(1,d));gl.drawArrays(gl.POINTS,0,cloudCount)}}
async function renderCloudFrame(frame){{initCloudGL();const data=await cachedCloud(mediaUrl('cloud',frame)),view=new DataView(data);if(data.byteLength<8)throw new Error('点云帧数据损坏');const count=view.getUint32(4,true);if(data.byteLength<8+count*16)throw new Error('点云帧长度错误');gl.bindBuffer(gl.ARRAY_BUFFER,cloudBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Uint8Array(data,8,count*16),gl.DYNAMIC_DRAW);cloudCount=count;drawCloudGL();requestFramePrefetch();return true}}
canvas.oncontextmenu=e=>e.preventDefault();canvas.onpointerdown=e=>{{if(e.button!==0&&e.button!==2)return;e.preventDefault();dragMode=e.button===0?'rotate':'pan';canvas.classList.toggle('panning',dragMode==='pan');lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)}};canvas.onpointermove=e=>{{if(!dragMode)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;if(dragMode==='rotate'){{yaw+=dx*.005;pitch=Math.max(-1.55,Math.min(1.55,pitch+dy*.005))}}else{{const b=cloudBasis(),scale=distance*.001;for(let i=0;i<3;i++)target[i]+=-b.right[i]*dx*scale+b.up[i]*dy*scale}}drawCloudGL()}};function stopDrag(){{dragMode='';canvas.classList.remove('panning')}}canvas.onpointerup=stopDrag;canvas.onpointercancel=stopDrag;
function zoomCloud(factor){{distance=Math.max(.2,Math.min(20,distance*factor));drawCloudGL()}}$('#zoomIn').onclick=()=>zoomCloud(.9);$('#zoomOut').onclick=()=>zoomCloud(1.1);$('#resetView').onclick=()=>{{yaw=0;pitch=0;distance=1.5;target=[0,0,1];drawCloudGL()}};canvas.onwheel=e=>{{e.preventDefault();zoomCloud(e.deltaY<0?.9:1.1)}};window.addEventListener('keydown',e=>{{if(mode==='pointcloud'&&e.ctrlKey&&(e.key==='+'||e.key==='='||e.key==='-')){{e.preventDefault();zoomCloud(e.key==='-'?1.1:.9)}}}});window.addEventListener('resize',()=>{{if(mode==='pointcloud')drawCloudGL()}});
async function closeSession(){{if(closed||!sessionId)return;closed=true;const url=`/api/v1/viewer/sessions/${{sessionId}}`;if(navigator.sendBeacon)navigator.sendBeacon(url+'/close');else fetch(url,{{method:'DELETE',keepalive:true}})}}
window.addEventListener('pagehide',closeSession);window.addEventListener('beforeunload',closeSession);window.addEventListener('unload',closeSession);
setInterval(()=>{{if(sessionId&&!closed)fetch(`/api/v1/viewer/sessions/${{sessionId}}`,{{cache:'no-store'}}).catch(()=>{{}})}},20000);
start();
</script></body></html>"""
