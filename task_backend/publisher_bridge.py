from __future__ import annotations

import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

try:
    from .job_service import JobService
    from .storage_resolver import uri_join
    from .workflow_models import WorkflowError
except ImportError:  # pragma: no cover - script execution fallback
    from job_service import JobService  # type: ignore
    from storage_resolver import uri_join  # type: ignore
    from workflow_models import WorkflowError  # type: ignore


PUBLISHER_WAITING_STATES = {
    "pending",
    "uploading",
    "ready",
    "cleaning",
    "cleaned",
    "result_downloading",
}
PUBLISHER_RESULT_STATES = {"labeled", "quality_checked"}


def _log(message: str) -> None:
    print(f"[publisher-bridge] {message}", file=sys.stderr, flush=True)


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class BridgeShutdown(RuntimeError):
    pass


class PublisherCommandError(RuntimeError):
    pass


class MaterializerError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class PublisherBridgeConfig:
    enabled: bool = False
    max_inflight: int = 4
    poll_seconds: float = 20.0
    lease_seconds: int = 300
    heartbeat_seconds: float = 60.0
    command_timeout_seconds: float = 120.0
    publisher_ssh_host: str = "synology"
    publisher_publish_command: str = "/usr/local/sbin/nas-uploader-publish"
    publisher_status_command: str = "/usr/local/sbin/nas-uploader-status"
    mano_python: Path = Path("python3")
    mano_toolkit_root: Path = Path(".")
    mano_model_dir: Path = Path(".")
    mano_default_shape_path: Optional[Path] = None

    def validate(self) -> None:
        if self.max_inflight <= 0:
            raise ValueError("ORBBEC_PUBLISHER_BRIDGE_MAX_INFLIGHT must be greater than 0")
        if self.poll_seconds <= 0:
            raise ValueError("ORBBEC_PUBLISHER_BRIDGE_POLL_SECONDS must be greater than 0")
        if self.lease_seconds <= 0:
            raise ValueError("ORBBEC_PUBLISHER_BRIDGE_LEASE_SECONDS must be greater than 0")
        if self.heartbeat_seconds <= 0:
            raise ValueError("ORBBEC_PUBLISHER_BRIDGE_HEARTBEAT_SECONDS must be greater than 0")
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("Publisher Bridge heartbeat interval must be shorter than its lease")
        if self.command_timeout_seconds <= 0:
            raise ValueError("Publisher Bridge command timeout must be greater than 0")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(self.publisher_ssh_host)) or str(self.publisher_ssh_host).startswith("-"):
            raise ValueError("Publisher Bridge SSH host is required")
        for name, value in (
            ("publish command", self.publisher_publish_command),
            ("status command", self.publisher_status_command),
        ):
            if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(value)):
                raise ValueError(f"Publisher Bridge {name} must be an absolute NAS command path")

        # Keep the venv launcher path itself.  Resolving its symlink to
        # /usr/bin/python would lose pyvenv.cfg discovery and all MANO deps.
        self.mano_python = Path(os.path.abspath(str(self.mano_python.expanduser())))
        self.mano_toolkit_root = self.mano_toolkit_root.expanduser().resolve()
        self.mano_model_dir = self.mano_model_dir.expanduser().resolve()
        if not self.mano_python.is_file() or not os.access(self.mano_python, os.X_OK):
            raise ValueError(f"ORBBEC_MANO_PYTHON is not executable: {self.mano_python}")
        if not self.mano_toolkit_root.is_dir():
            raise ValueError(f"ORBBEC_MANO_TOOLKIT_ROOT is not a directory: {self.mano_toolkit_root}")
        if not self.mano_model_dir.is_dir():
            raise ValueError(f"ORBBEC_MANO_MODEL_DIR is not a directory: {self.mano_model_dir}")
        if self.mano_default_shape_path is not None:
            self.mano_default_shape_path = self.mano_default_shape_path.expanduser().resolve()
            if not self.mano_default_shape_path.is_file():
                raise ValueError(
                    f"ORBBEC_MANO_DEFAULT_SHAPE_PATH is not a file: {self.mano_default_shape_path}"
                )


@dataclass
class ManualPublisherBridgeConfig(PublisherBridgeConfig):
    completion_poll_count: int = 3
    wait_for_nas_qc_sync: bool = False

    def validate(self) -> None:
        super().validate()
        if self.completion_poll_count <= 0:
            raise ValueError("ORBBEC_MANUAL_PUBLISHER_BRIDGE_COMPLETION_POLLS must be greater than 0")


class PublisherClient:
    def __init__(self, config: PublisherBridgeConfig, stop_event: threading.Event):
        self.config = config
        self.stop_event = stop_event

    def status(self, episode_id: str) -> Dict[str, Any]:
        output = self._run_remote(self.config.publisher_status_command, episode_id)
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            raise PublisherCommandError(f"Publisher status returned invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise PublisherCommandError("Publisher status JSON must be an object")
        returned_id = str(value.get("episode_id") or "")
        if returned_id and returned_id != episode_id:
            raise PublisherCommandError(
                f"Publisher status episode mismatch: requested={episode_id}, returned={returned_id}"
            )
        return value

    def publish(self, episode_id: str) -> None:
        self._run_remote(self.config.publisher_publish_command, episode_id)

    def publish_manual(self, episode_id: str) -> None:
        self._run_remote(self.config.publisher_publish_command, "--manual-2d", episode_id)

    def _run_remote(self, command: str, *arguments: str) -> str:
        remote_command = "sudo -- " + " ".join(
            [shlex.quote(command), *(shlex.quote(str(argument)) for argument in arguments)]
        )
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            self.config.publisher_ssh_host,
            remote_command,
        ]
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        started = time.monotonic()
        while process.poll() is None:
            if self.stop_event.wait(0.2):
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise BridgeShutdown("backend is stopping")
            if time.monotonic() - started >= self.config.command_timeout_seconds:
                process.kill()
                process.wait()
                raise PublisherCommandError(f"Publisher command timed out after {self.config.command_timeout_seconds:g}s")
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            detail = " ".join((stderr or stdout or "").strip().split())
            if len(detail) > 500:
                detail = detail[:497] + "..."
            raise PublisherCommandError(
                f"Publisher command failed with exit {process.returncode}: {detail or command}"
            )
        return stdout.strip()


class OptimizedPoseMaterializer:
    def __init__(self, config: PublisherBridgeConfig, stop_event: threading.Event):
        self.config = config
        self.stop_event = stop_event

    def run(
        self,
        *,
        episode_dir: Path,
        generation: int,
        result_manifest_sha256: str,
        cameras: Sequence[str],
    ) -> Dict[str, Any]:
        script_path = Path(__file__).resolve().with_name("optimized_pose_materializer.py")
        argv = [
            str(self.config.mano_python),
            str(script_path),
            "--episode-dir",
            str(episode_dir),
            "--toolkit-root",
            str(self.config.mano_toolkit_root),
            "--mano-model-dir",
            str(self.config.mano_model_dir),
            "--generation",
            str(generation),
            "--result-manifest-sha256",
            result_manifest_sha256,
            "--cameras-json",
            json.dumps([str(camera) for camera in cameras], ensure_ascii=False),
        ]
        if self.config.mano_default_shape_path is not None:
            argv.extend(["--default-shape-path", str(self.config.mano_default_shape_path)])

        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while process.poll() is None:
            if self.stop_event.wait(0.2):
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise BridgeShutdown("backend is stopping")
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raw = (stderr or stdout or "").strip()
            message = raw
            try:
                error_payload = json.loads(raw.splitlines()[-1])
                if isinstance(error_payload, dict):
                    message = str(error_payload.get("error") or raw)
            except (IndexError, json.JSONDecodeError):
                pass
            raise MaterializerError(message or "MANO materializer failed", retryable=process.returncode == 75)
        try:
            value = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise MaterializerError(f"MANO materializer returned invalid JSON: {exc}", retryable=False) from exc
        if not isinstance(value, dict) or not value.get("ok"):
            raise MaterializerError("MANO materializer did not report success", retryable=False)
        return value


class _LeaseHeartbeat:
    def __init__(
        self,
        service: JobService,
        *,
        job_id: str,
        worker_id: str,
        lease_seconds: int,
        interval_seconds: float,
        bridge_stop: threading.Event,
    ):
        self.service = service
        self.job_id = job_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds
        self.bridge_stop = bridge_stop
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"publisher-heartbeat-{job_id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, min(5.0, self.interval_seconds + 1.0)))

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            if self.bridge_stop.is_set():
                return
            try:
                self.service.heartbeat_job(
                    self.job_id,
                    {
                        "worker_id": self.worker_id,
                        "lease_seconds": self.lease_seconds,
                        "status": "running",
                    },
                )
            except Exception as exc:  # A later heartbeat can recover a transient SQLite error.
                _log(f"heartbeat failed job={self.job_id}: {_format_error(exc)}")


class PublisherBridge:
    def __init__(
        self,
        service: JobService,
        config: PublisherBridgeConfig,
        *,
        publisher_client: Optional[Any] = None,
        materializer: Optional[Any] = None,
        hostname: str = "",
    ):
        self.service = service
        self.config = config
        self.hostname = hostname or socket.gethostname()
        self.stop_event = threading.Event()
        self.publisher = publisher_client or PublisherClient(config, self.stop_event)
        self.materializer = materializer or OptimizedPoseMaterializer(config, self.stop_event)
        self.threads: List[threading.Thread] = []

    def start(self) -> None:
        if not self.config.enabled or self.threads:
            return
        self.config.validate()
        self.stop_event.clear()
        for slot_index in range(1, self.config.max_inflight + 1):
            thread = threading.Thread(
                target=self._run_slot,
                args=(slot_index,),
                name=f"publisher-bridge-slot-{slot_index:02d}",
                daemon=True,
            )
            self.threads.append(thread)
            thread.start()
        _log(
            f"started slots={self.config.max_inflight} publisher={self.config.publisher_ssh_host} "
            f"poll={self.config.poll_seconds:g}s lease={self.config.lease_seconds}s"
        )

    def stop(self, timeout: float = 15.0) -> None:
        self.stop_event.set()
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self.threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in self.threads if thread.is_alive()]
        if alive:
            _log(f"shutdown timed out while waiting for: {', '.join(alive)}")
        self.threads = []

    def worker_id(self, slot_index: int) -> str:
        return f"publisher_bridge:{self.hostname}:slot-{slot_index:02d}"

    def _run_slot(self, slot_index: int) -> None:
        worker_id = self.worker_id(slot_index)
        while not self.stop_event.is_set():
            try:
                did_work = self.process_once(slot_index)
            except BridgeShutdown:
                break
            except Exception as exc:  # Preserve the slot after an unexpected per-job error.
                _log(f"slot-{slot_index:02d} error: {_format_error(exc)}")
                did_work = False
            if not did_work:
                self.stop_event.wait(self.config.poll_seconds)

    def process_once(self, slot_index: int) -> bool:
        if self.stop_event.is_set():
            return False
        worker_id = self.worker_id(slot_index)
        try:
            leased = self.service.lease_job(
                {
                    "type": "auto_label",
                    "worker_id": worker_id,
                    "lease_seconds": self.config.lease_seconds,
                }
            )
        except WorkflowError as exc:
            if exc.status in {HTTPStatus.NOT_FOUND, HTTPStatus.CONFLICT}:
                return False
            raise

        job = dict(leased.get("job") or {})
        payload = dict(leased.get("payload") or {})
        episode = dict(leased.get("episode") or {})
        job_id = str(job.get("job_id") or "")
        if not job_id:
            raise RuntimeError("leased auto_label job is missing job_id")

        heartbeat = _LeaseHeartbeat(
            self.service,
            job_id=job_id,
            worker_id=worker_id,
            lease_seconds=self.config.lease_seconds,
            interval_seconds=self.config.heartbeat_seconds,
            bridge_stop=self.stop_event,
        )
        heartbeat.start()
        finished = False
        try:
            finished = self._process_leased_job(job_id, worker_id, payload, episode)
            return True
        finally:
            heartbeat.stop()
            if not finished:
                reason = "backend_shutdown" if self.stop_event.is_set() else "publisher_bridge_interrupted"
                try:
                    self.service.release_job(job_id, {"reason": reason, "worker_id": worker_id})
                except Exception as exc:
                    _log(f"release failed job={job_id}: {_format_error(exc)}")

    def _process_leased_job(
        self,
        job_id: str,
        worker_id: str,
        payload: Dict[str, Any],
        episode: Dict[str, Any],
    ) -> bool:
        episode_uri = str(payload.get("episode_uri") or episode.get("episode_uri") or episode.get("data_uri") or "").strip()
        try:
            publisher_episode_id, episode_dir = self._resolve_episode(episode_uri)
        except ValueError as exc:
            error = f"invalid Publisher episode URI: {exc}"
            self.service.fail_job(
                job_id,
                {
                    "error": error,
                    "result": {"ok": False, "worker_id": worker_id, "phase": "episode_validation"},
                },
            )
            _log(f"failed job={job_id}: {error}")
            return True
        cameras = payload.get("cameras") if isinstance(payload.get("cameras"), list) else episode.get("cameras")
        camera_ids = [str(value) for value in (cameras or [])]

        while not self.stop_event.is_set():
            try:
                status = self.publisher.status(publisher_episode_id)
                if not bool(status.get("found")):
                    self.publisher.publish(publisher_episode_id)
                    _log(f"published job={job_id} episode={publisher_episode_id}")
                    self.stop_event.wait(self.config.poll_seconds)
                    continue

                state = str(status.get("state") or "").strip().lower()
                if state in PUBLISHER_WAITING_STATES:
                    self.stop_event.wait(self.config.poll_seconds)
                    continue
                if state not in PUBLISHER_RESULT_STATES:
                    _log(f"job={job_id} waiting on unknown Publisher state={state or '<empty>'}")
                    self.stop_event.wait(self.config.poll_seconds)
                    continue

                generation = int(status.get("generation") or 0)
                result_sha256 = str(status.get("result_manifest_sha256") or "").strip()
                materialized = self.materializer.run(
                    episode_dir=episode_dir,
                    generation=generation,
                    result_manifest_sha256=result_sha256,
                    cameras=camera_ids,
                )
                frames = [int(value) for value in materialized.get("frames", [])]
                result = {
                    "ok": True,
                    "worker_id": worker_id,
                    "generation": generation,
                    "frames": frames,
                    "optimized_pose_shape": [2, 99],
                    "joints_3d_shape": list(materialized.get("joints_3d_shape") or []),
                    "result_manifest_sha256": result_sha256,
                    "materializer_reused": bool(materialized.get("reused")),
                }
                artifacts = [
                    {
                        "kind": "optimized_pose",
                        "uri": uri_join(episode_uri, "optimized_pose"),
                        "metadata": {"generation": generation, "frame_shape": [2, 99]},
                    },
                    {
                        "kind": "mano_episode",
                        "uri": uri_join(episode_uri, "mano", "episode"),
                        "metadata": {"generation": generation, "coordinate_system": "episode_world"},
                    },
                ]
                self.service.complete_job(job_id, {"result": result, "artifacts": artifacts})
                _log(f"completed job={job_id} episode={publisher_episode_id} frames={len(frames)}")
                return True
            except BridgeShutdown:
                break
            except MaterializerError as exc:
                if exc.retryable:
                    _log(f"job={job_id} MANO result not ready: {exc}")
                    self.stop_event.wait(self.config.poll_seconds)
                    continue
                error = f"optimized_pose materialization failed: {exc}"
                self.service.fail_job(
                    job_id,
                    {
                        "error": error,
                        "result": {"ok": False, "worker_id": worker_id, "phase": "mano_materialization"},
                    },
                )
                _log(f"failed job={job_id}: {error}")
                return True
            except (PublisherCommandError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                # Publisher/NAS/SSH failures are explicitly non-terminal.  The
                # independent heartbeat keeps this job leased while we retry.
                _log(f"job={job_id} transient Publisher/NAS error: {_format_error(exc)}")
                self.stop_event.wait(self.config.poll_seconds)
        raise BridgeShutdown("backend is stopping")

    def _resolve_episode(self, episode_uri: str) -> Tuple[str, Path]:
        parsed = urlparse(episode_uri)
        if parsed.scheme != "nas" or not parsed.netloc or parsed.params or parsed.query or parsed.fragment:
            raise ValueError(f"auto_label episode_uri must be a nas:// URI: {episode_uri}")
        raw_suffix = parsed.path.lstrip("/")
        decoded_suffix = unquote(raw_suffix)
        pure = PurePosixPath(decoded_suffix)
        parts = pure.parts
        if len(parts) != 3 or any(part in {"", ".", ".."} or "/" in part or "\\" in part for part in parts):
            raise ValueError(f"auto_label episode_uri must contain exactly subject/task/episode: {episode_uri}")
        publisher_episode_id = "/".join(parts)

        prefix = f"nas://{parsed.netloc}"
        mount_value = self.service.nas_mounts.get(prefix)
        if not mount_value:
            raise ValueError(f"auto_label episode_uri has no configured NAS mount: {episode_uri}")
        mount_root = Path(mount_value).expanduser().resolve()
        episode_dir = mount_root.joinpath(*parts).resolve()
        if episode_dir == mount_root or mount_root not in episode_dir.parents:
            raise ValueError(f"auto_label episode_uri escapes configured NAS mount: {episode_uri}")
        return publisher_episode_id, episode_dir


class ManualPublisherBridge:
    """Bridge one episode-level manual_3d job to one Publisher manual publish."""

    def __init__(
        self,
        service: JobService,
        config: ManualPublisherBridgeConfig,
        *,
        publisher_client: Optional[Any] = None,
        materializer: Optional[Any] = None,
        hostname: str = "",
    ):
        self.service = service
        self.config = config
        self.hostname = hostname or socket.gethostname()
        self.stop_event = threading.Event()
        self.publisher = publisher_client or PublisherClient(config, self.stop_event)
        self.materializer = materializer or OptimizedPoseMaterializer(config, self.stop_event)
        self.threads: List[threading.Thread] = []

    def start(self) -> None:
        if not self.config.enabled or self.threads:
            return
        self.config.validate()
        self.stop_event.clear()
        for slot_index in range(1, self.config.max_inflight + 1):
            thread = threading.Thread(
                target=self._run_slot,
                args=(slot_index,),
                name=f"manual-publisher-bridge-slot-{slot_index:02d}",
                daemon=True,
            )
            self.threads.append(thread)
            thread.start()
        _log(
            f"manual bridge started slots={self.config.max_inflight} "
            f"poll={self.config.poll_seconds:g}s completion_polls={self.config.completion_poll_count}"
        )

    def stop(self, timeout: float = 15.0) -> None:
        self.stop_event.set()
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self.threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in self.threads if thread.is_alive()]
        if alive:
            _log(f"manual bridge shutdown timed out while waiting for: {', '.join(alive)}")
        self.threads = []

    def worker_id(self, slot_index: int) -> str:
        return f"manual_publisher_bridge:{self.hostname}:slot-{slot_index:02d}"

    def _run_slot(self, slot_index: int) -> None:
        while not self.stop_event.is_set():
            try:
                did_work = self.process_once(slot_index)
            except BridgeShutdown:
                break
            except Exception as exc:
                _log(f"manual slot-{slot_index:02d} error: {_format_error(exc)}")
                did_work = False
            if not did_work:
                self.stop_event.wait(self.config.poll_seconds)

    def process_once(self, slot_index: int) -> bool:
        if self.stop_event.is_set():
            return False
        worker_id = self.worker_id(slot_index)
        try:
            leased = self.service.lease_job(
                {
                    "type": "manual_3d",
                    "worker_id": worker_id,
                    "lease_seconds": self.config.lease_seconds,
                }
            )
        except WorkflowError as exc:
            if exc.status in {HTTPStatus.NOT_FOUND, HTTPStatus.CONFLICT}:
                return False
            raise

        job = dict(leased.get("job") or {})
        payload = dict(leased.get("payload") or {})
        episode = dict(leased.get("episode") or {})
        job_id = str(job.get("job_id") or "")
        if not job_id:
            raise RuntimeError("leased manual_3d job is missing job_id")

        heartbeat = _LeaseHeartbeat(
            self.service,
            job_id=job_id,
            worker_id=worker_id,
            lease_seconds=self.config.lease_seconds,
            interval_seconds=self.config.heartbeat_seconds,
            bridge_stop=self.stop_event,
        )
        heartbeat.start()
        finished = False
        try:
            finished = self._process_leased_job(job_id, worker_id, payload, episode)
            return True
        finally:
            heartbeat.stop()
            if not finished:
                reason = "backend_shutdown" if self.stop_event.is_set() else "manual_publisher_bridge_interrupted"
                try:
                    self.service.release_job(job_id, {"reason": reason, "worker_id": worker_id})
                except Exception as exc:
                    _log(f"manual release failed job={job_id}: {_format_error(exc)}")

    def _process_leased_job(
        self,
        job_id: str,
        worker_id: str,
        payload: Dict[str, Any],
        episode: Dict[str, Any],
    ) -> bool:
        episode_uri = str(payload.get("episode_uri") or episode.get("episode_uri") or "").strip()
        try:
            publisher_episode_id, episode_dir = PublisherBridge._resolve_episode(self, episode_uri)
        except ValueError as exc:
            error = f"invalid manual Publisher episode URI: {exc}"
            self.service.fail_job(
                job_id,
                {"error": error, "result": {"ok": False, "worker_id": worker_id, "phase": "episode_validation"}},
            )
            return True

        cameras = payload.get("cameras") if isinstance(payload.get("cameras"), list) else episode.get("cameras")
        camera_ids = [str(value) for value in (cameras or [])]

        while not self.stop_event.is_set():
            current = self.service.store.get_job(job_id) or {}
            progress = dict(current.get("result") or {})
            try:
                if self.config.wait_for_nas_qc_sync:
                    sync_events = [
                        event
                        for event in self.service.store.list_nas_sync_events()
                        if str(event.get("episode_id") or "") == str(current.get("episode_id") or "")
                        and str(event.get("action") or "") == "quality_needs_labeling"
                    ]
                    if not sync_events or str(sync_events[-1].get("status") or "") != "succeeded":
                        self.stop_event.wait(self.config.poll_seconds)
                        continue
                if not progress.get("manual_publish_started_at"):
                    self.publisher.publish_manual(publisher_episode_id)
                    progress = dict(
                        self.service.update_job_progress(
                            job_id,
                            {
                                "manual_publish_started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "manual_publish_episode_id": publisher_episode_id,
                                "completion_poll_count": 0,
                            },
                        ).get("job", {}).get("result", {})
                    )
                    _log(f"manual published job={job_id} episode={publisher_episode_id}")

                poll_count = int(progress.get("completion_poll_count") or 0)
                last_status = progress.get("last_publisher_status")
                while poll_count < self.config.completion_poll_count and not self.stop_event.is_set():
                    status = self.publisher.status(publisher_episode_id)
                    poll_count += 1
                    last_status = status
                    self.service.update_job_progress(
                        job_id,
                        {
                            "completion_poll_count": poll_count,
                            "last_publisher_status": status,
                        },
                    )
                    if poll_count < self.config.completion_poll_count:
                        self.stop_event.wait(self.config.poll_seconds)
                if self.stop_event.is_set():
                    break

                status_obj = dict(last_status) if isinstance(last_status, dict) else {}
                observed_generation = int(status_obj.get("generation") or 0)
                observed_sha256 = str(status_obj.get("result_manifest_sha256") or "").strip()
                generation = max(1, observed_generation)
                result_sha256 = observed_sha256
                if len(result_sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in result_sha256):
                    result_sha256 = "0" * 64
                materialized = self.materializer.run(
                    episode_dir=episode_dir,
                    generation=generation,
                    result_manifest_sha256=result_sha256,
                    cameras=camera_ids,
                )
                frames = [int(value) for value in materialized.get("frames", [])]
                result = {
                    "ok": True,
                    "worker_id": worker_id,
                    "frames": frames,
                    "generation": generation,
                    "result_manifest_sha256": result_sha256,
                    "observed_generation": observed_generation,
                    "observed_result_manifest_sha256": observed_sha256,
                    "joints_3d_shape": list(materialized.get("joints_3d_shape") or []),
                    "materializer_reused": bool(materialized.get("reused")),
                    "completion_policy": f"assumed_complete_after_{self.config.completion_poll_count}_polls",
                    "completion_poll_count": self.config.completion_poll_count,
                    "publisher_status_at_completion": status_obj,
                    "temporary_completion_policy": True,
                }
                artifacts = [
                    {
                        "kind": "optimized_pose",
                        "uri": uri_join(episode_uri, "optimized_pose"),
                        "metadata": {"generation": generation, "source": "manual_3d", "source_job_id": job_id},
                    },
                    {
                        "kind": "mano_episode",
                        "uri": uri_join(episode_uri, "mano", "episode"),
                        "metadata": {
                            "generation": generation,
                            "source": "manual_3d",
                            "source_job_id": job_id,
                            "scope": "episode",
                        },
                    },
                ]
                self.service.complete_job(job_id, {"result": result, "artifacts": artifacts})
                _log(f"manual completed job={job_id} episode={publisher_episode_id} frames={len(frames)}")
                return True
            except MaterializerError as exc:
                if exc.retryable:
                    _log(f"manual job={job_id} 3D files not ready after assumed completion: {exc}")
                    self.stop_event.wait(self.config.poll_seconds)
                    continue
                error = f"manual optimized_pose materialization failed: {exc}"
                self.service.fail_job(
                    job_id,
                    {"error": error, "result": {"ok": False, "worker_id": worker_id, "phase": "mano_materialization"}},
                )
                return True
            except (PublisherCommandError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                _log(f"manual job={job_id} transient Publisher/NAS error: {_format_error(exc)}")
                self.stop_event.wait(self.config.poll_seconds)
        raise BridgeShutdown("backend is stopping")
