from __future__ import annotations

import re
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

try:
    from .workflow_store import WorkflowStore
except ImportError:  # pragma: no cover - script execution fallback
    from workflow_store import WorkflowStore  # type: ignore


class NasStatusSyncError(RuntimeError):
    pass


@dataclass
class NasStatusSyncConfig:
    enabled: bool = False
    poll_seconds: float = 2.0
    command_timeout_seconds: float = 60.0
    retry_base_seconds: int = 5
    retry_max_seconds: int = 300
    ssh_host: str = "synology"
    check_command: str = "/usr/local/sbin/nas-uploader-check"

    def validate(self) -> None:
        if self.poll_seconds <= 0:
            raise ValueError("ORBBEC_NAS_STATUS_SYNC_POLL_SECONDS must be greater than 0")
        if self.command_timeout_seconds <= 0:
            raise ValueError("ORBBEC_NAS_STATUS_SYNC_COMMAND_TIMEOUT_SECONDS must be greater than 0")
        if self.retry_base_seconds <= 0 or self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("NAS status sync retry interval is invalid")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.ssh_host) or self.ssh_host.startswith("-"):
            raise ValueError("NAS status sync SSH host is invalid")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", self.check_command):
            raise ValueError("NAS status sync check command must be an absolute path")


class NasStatusSync:
    def __init__(
        self,
        store: WorkflowStore,
        config: NasStatusSyncConfig,
        *,
        executor: Optional[Callable[[Sequence[str]], None]] = None,
    ):
        self.store = store
        self.config = config
        self.config.validate()
        self.executor = executor
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.config.enabled or self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, name="nas-status-sync", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.config.command_timeout_seconds + 2.0)
            self.thread = None

    def run_once(self) -> bool:
        event = self.store.claim_nas_sync_event()
        if event is None:
            return False
        event_id = str(event.get("event_id") or "")
        try:
            argv = self._command_for_event(event)
            self._execute(argv)
            self.store.complete_nas_sync_event(event_id)
            self._log(
                f"completed event={event_id} action={event.get('action')} "
                f"episode={event.get('nas_episode_id')}"
            )
        except Exception as exc:
            attempt = max(1, int(event.get("attempt") or 1))
            retry_seconds = min(
                self.config.retry_max_seconds,
                self.config.retry_base_seconds * (2 ** min(attempt - 1, 10)),
            )
            error = f"{type(exc).__name__}: {exc}"
            self.store.retry_nas_sync_event(event_id, error=error, retry_seconds=retry_seconds)
            self._log(f"retry event={event_id} in={retry_seconds}s error={error}")
        return True

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                if self.run_once():
                    continue
            except Exception as exc:  # keep transient SQLite/process errors from killing the worker
                self._log(f"worker error: {type(exc).__name__}: {exc}")
            self.stop_event.wait(self.config.poll_seconds)

    def _execute(self, argv: Sequence[str]) -> None:
        if self.executor is not None:
            self.executor(argv)
            return
        try:
            process = subprocess.run(
                list(argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.config.command_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NasStatusSyncError(
                f"command timed out after {self.config.command_timeout_seconds:g}s"
            ) from exc
        if process.returncode != 0:
            detail = " ".join((process.stderr or process.stdout or "").strip().split())
            if len(detail) > 500:
                detail = detail[:497] + "..."
            raise NasStatusSyncError(
                f"command failed with exit {process.returncode}: {detail or argv[-1]}"
            )

    def _command_for_event(self, event: Dict[str, Any]) -> Sequence[str]:
        episode_id = str(event.get("nas_episode_id") or "").strip()
        parts = episode_id.split("/")
        if len(parts) != 3 or any(not part or part in {".", ".."} for part in parts):
            raise NasStatusSyncError(f"invalid NAS episode ID: {episode_id}")
        action = str(event.get("action") or "")
        if action == "quality_take":
            command = [self.config.check_command, "take", episode_id]
        elif action == "quality_passed":
            command = [self.config.check_command, "result", "passed", episode_id]
        elif action == "quality_needs_labeling":
            command = [self.config.check_command, "result", "needs-labeling", episode_id]
        else:
            raise NasStatusSyncError(f"unsupported NAS sync action: {action}")
        remote_command = "sudo -- " + " ".join(shlex.quote(part) for part in command)
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            self.config.ssh_host,
            remote_command,
        ]

    @staticmethod
    def _log(message: str) -> None:
        print(f"[nas-status-sync] {message}", file=sys.stderr, flush=True)
