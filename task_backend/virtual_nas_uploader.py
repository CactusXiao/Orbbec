from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from .job_service import JobService
    from .storage_resolver import path_from_local_uri, uri_join
except ImportError:  # pragma: no cover - script execution fallback
    from job_service import JobService  # type: ignore
    from storage_resolver import path_from_local_uri, uri_join  # type: ignore


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clean_path_part(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    return text or fallback


def format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def iter_files(root: Path) -> Iterable[Tuple[Path, Path]]:
    if root.is_file():
        yield root, Path(root.name)
        return
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            src = base / name
            if not src.is_file():
                continue
            yield src, src.relative_to(root)


def path_totals(root: Path) -> Tuple[int, int]:
    files = 0
    bytes_total = 0
    for src, _ in iter_files(root):
        try:
            stat = src.stat()
        except OSError:
            continue
        files += 1
        bytes_total += int(stat.st_size)
    return files, bytes_total


@dataclass
class VirtualNasUploadConfig:
    enabled: bool = True
    root: Path = Path("./task_backend_state/virtual_nas")
    uri_prefix: str = "nas://orbbec-virtual"
    worker_id: str = ""
    poll_interval_seconds: float = 1.0
    lease_seconds: int = 300
    progress_interval_seconds: float = 0.5
    chunk_bytes: int = 1024 * 1024


class VirtualNasUploader:
    def __init__(self, service: JobService, config: VirtualNasUploadConfig):
        self.service = service
        self.config = config
        self.config.root = self.config.root.expanduser().resolve()
        self.config.uri_prefix = self.config.uri_prefix.rstrip("/")
        self.worker_id = self.config.worker_id or f"virtual_nas_uploader_{os.getpid()}"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.config.enabled or self._thread is not None:
            return
        self.config.root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="virtual-nas-uploader", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                did_work = self.process_one()
            except Exception as exc:  # pragma: no cover - defensive worker loop
                print(f"[virtual-nas] worker error: {format_error(exc)}", flush=True)
                did_work = False
            if not did_work:
                self._stop.wait(max(0.1, float(self.config.poll_interval_seconds)))

    def process_one(self) -> bool:
        try:
            leased = self.service.lease_job(
                {
                    "type": "upload",
                    "worker_id": self.worker_id,
                    "lease_seconds": self.config.lease_seconds,
                }
            )
        except Exception as exc:
            message = str(exc)
            if "no queued upload job" in message:
                return False
            raise

        job = leased.get("job") or {}
        episode = leased.get("episode") or {}
        payload = dict(leased.get("payload") or {})
        job_id = str(job.get("job_id") or "")
        episode_id = str(payload.get("episode_id") or episode.get("episode_id") or "")
        try:
            self._upload_leased_job(job_id, episode_id, payload, episode)
        except Exception as exc:
            error = format_error(exc)
            result = {
                "ok": False,
                "phase": "failed",
                "error": error,
                "worker_id": self.worker_id,
                "failed_at": now_iso(),
            }
            try:
                self.service.fail_job(job_id, {"error": error, "result": result})
                if episode_id:
                    self.service.store.update_episode_status(episode_id, "captured", {"upload_error": error})
            except Exception as fail_exc:  # pragma: no cover - preserve root failure in logs
                print(f"[virtual-nas] fail upload job {job_id}: {format_error(fail_exc)}", flush=True)
            return True
        return True

    def _upload_leased_job(self, job_id: str, episode_id: str, payload: Dict[str, Any], episode: Dict[str, Any]) -> None:
        if not job_id:
            raise RuntimeError("leased upload job is missing job_id")
        if not episode_id:
            raise RuntimeError("leased upload job is missing episode_id")
        source = self._source_path(payload, episode)
        if source is None or not source.exists():
            raise FileNotFoundError(f"local capture path not found: {source or ''}")
        source = source.resolve()
        if not (source.is_dir() or source.is_file()):
            raise RuntimeError(f"local capture path is not a file or directory: {source}")

        subject = clean_path_part(payload.get("subject_id") or episode.get("subject_id"), "subject")
        task = clean_path_part(payload.get("task_name") or episode.get("task_name"), "task")
        episode_part = clean_path_part(episode_id, "episode")
        nas_uri = uri_join(self.config.uri_prefix, subject, task, episode_part)
        dest = self._local_path_for_uri(nas_uri)
        tmp = self.config.root / ".upload_tmp" / f"{clean_path_part(job_id, 'upload')}.{uuid.uuid4().hex}"

        files_total, total_bytes = path_totals(source)
        progress: Dict[str, Any] = {
            "ok": False,
            "phase": "scanning",
            "local_path": str(source),
            "nas_uri": nas_uri,
            "copied_bytes": 0,
            "total_bytes": total_bytes,
            "files_done": 0,
            "files_total": files_total,
            "percent": 0.0 if total_bytes else 100.0,
            "worker_id": self.worker_id,
            "started_at": now_iso(),
        }
        self._progress(job_id, progress)

        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

        copied_bytes = 0
        files_done = 0
        last_progress = 0.0
        for src, rel in iter_files(source):
            if self._stop.is_set():
                raise RuntimeError("upload stopped before completion")
            out = tmp / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            file_size = int(src.stat().st_size)
            with src.open("rb") as rf, out.open("wb") as wf:
                while True:
                    if self._stop.is_set():
                        raise RuntimeError("upload stopped before completion")
                    chunk = rf.read(max(64 * 1024, int(self.config.chunk_bytes)))
                    if not chunk:
                        break
                    wf.write(chunk)
                    copied_bytes += len(chunk)
                    now = time.monotonic()
                    if now - last_progress >= float(self.config.progress_interval_seconds):
                        progress.update(
                            {
                                "phase": "copying",
                                "copied_bytes": copied_bytes,
                                "files_done": files_done,
                                "percent": self._percent(copied_bytes, total_bytes),
                            }
                        )
                        self._progress(job_id, progress)
                        last_progress = now
            try:
                shutil.copystat(src, out)
            except OSError:
                pass
            files_done += 1
            if file_size == 0:
                copied_bytes += 0
            progress.update(
                {
                    "phase": "copying",
                    "copied_bytes": copied_bytes,
                    "files_done": files_done,
                    "percent": self._percent(copied_bytes, total_bytes),
                }
            )
            self._progress(job_id, progress)

        dest_files, dest_bytes = path_totals(tmp)
        if dest_files != files_total or dest_bytes != total_bytes:
            raise RuntimeError(
                f"upload verification failed: source files/bytes={files_total}/{total_bytes}, "
                f"copied files/bytes={dest_files}/{dest_bytes}"
            )

        manifest = {
            "job_id": job_id,
            "episode_id": episode_id,
            "source_path": str(source),
            "nas_uri": nas_uri,
            "files_total": files_total,
            "total_bytes": total_bytes,
            "completed_at": now_iso(),
            "virtual_nas": True,
        }
        (tmp / ".orbbec_upload_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        progress.update(
            {
                "phase": "finalizing",
                "copied_bytes": total_bytes,
                "files_done": files_total,
                "percent": 100.0,
            }
        )
        self._progress(job_id, progress)

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        os.replace(tmp, dest)
        result = {
            **progress,
            "ok": True,
            "phase": "complete",
            "local_path": str(source),
            "nas_uri": nas_uri,
            "nas_local_path": str(dest),
            "copied_bytes": total_bytes,
            "total_bytes": total_bytes,
            "files_done": files_total,
            "files_total": files_total,
            "percent": 100.0,
            "completed_at": now_iso(),
        }
        self.service.complete_job(
            job_id,
            {
                "result": result,
                "artifacts": [
                    {
                        "kind": "nas_episode",
                        "uri": nas_uri,
                        "metadata": {
                            "worker_id": self.worker_id,
                            "local_path": str(source),
                            "nas_local_path": str(dest),
                            "files_total": files_total,
                            "total_bytes": total_bytes,
                        },
                    }
                ],
            },
        )

    def _progress(self, job_id: str, progress: Dict[str, Any]) -> None:
        self.service.heartbeat_job(
            job_id,
            {
                "worker_id": self.worker_id,
                "lease_seconds": self.config.lease_seconds,
                "status": "running",
            },
        )
        self.service.store.update_job_progress(job_id=job_id, progress=progress, status="running")

    def _source_path(self, payload: Dict[str, Any], episode: Dict[str, Any]) -> Optional[Path]:
        local_path = str(
            payload.get("local_capture_path")
            or episode.get("local_capture_path")
            or payload.get("local_path")
            or ""
        ).strip()
        if local_path:
            return Path(local_path).expanduser()
        data_uri = str(payload.get("data_uri") or episode.get("data_uri") or "")
        if data_uri.startswith("local://"):
            return Path(path_from_local_uri(data_uri)).expanduser()
        return None

    def _local_path_for_uri(self, uri: str) -> Path:
        if not uri.startswith(self.config.uri_prefix):
            raise ValueError(f"URI does not belong to virtual NAS: {uri}")
        suffix = uri[len(self.config.uri_prefix):].strip("/")
        return (self.config.root / suffix).resolve()

    @staticmethod
    def _percent(copied_bytes: int, total_bytes: int) -> float:
        if total_bytes <= 0:
            return 100.0
        return round(min(100.0, max(0.0, copied_bytes * 100.0 / total_bytes)), 2)
