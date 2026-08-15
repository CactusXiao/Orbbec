#!/usr/bin/env python3
"""Small HTTP task backend for Orbbec collection.

The service owns task definitions, episode reservations, and confirmed
progress.  It intentionally uses only the Python standard library so it can run
on a capture host without extra packages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import html
import json
import math
import os
import re
import secrets
import socket
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

try:
    from .job_service import JobService
    from .nas_uploader import NasUploadConfig, NasUploader
    from .workflow_models import WorkflowError
    from .workflow_store import WorkflowStore
except ImportError:  # pragma: no cover - script execution fallback
    from job_service import JobService  # type: ignore
    from nas_uploader import NasUploadConfig, NasUploader  # type: ignore
    from workflow_models import WorkflowError  # type: ignore
    from workflow_store import WorkflowStore  # type: ignore

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - non-Linux fallback
    fcntl = None


Task = Dict[str, Any]
State = Dict[str, Any]
Registry = Dict[str, Any]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def strip_json_comments(text: str) -> str:
    """Remove // comments outside strings, matching the existing C++ loader."""
    out_lines: List[str] = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        cut_at: Optional[int] = None
        for idx, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string and ch == "/" and idx + 1 < len(line) and line[idx + 1] == "/":
                cut_at = idx
                break
        out_lines.append(line if cut_at is None else line[:cut_at])
    return "\n".join(out_lines)


def load_task_file(path: Path) -> List[Task]:
    with path.open("r", encoding="utf-8") as f:
        raw = strip_json_comments(f.read())
    parsed = json.loads(raw)

    tasks: List[Task] = []

    def normalize_task(name: str, obj: Dict[str, Any]) -> Task:
        total = obj.get("total", obj.get("repeat_times", obj.get("episodes", 1)))
        try:
            total_int = max(1, int(total))
        except (TypeError, ValueError):
            total_int = 1
        return {
            "task_name": str(obj.get("task_name", obj.get("name", name))),
            "description_cn": str(obj.get("description_cn", obj.get("task_description_cn", ""))),
            "description_en": str(obj.get("description_en", obj.get("task_description_en", ""))),
            "total": total_int,
            "raw": obj,
        }

    if isinstance(parsed, dict):
        if isinstance(parsed.get("tasks"), list):
            for idx, item in enumerate(parsed["tasks"]):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("task_name", item.get("name", f"task_{idx}")))
                tasks.append(normalize_task(name, item))
        else:
            for name, item in parsed.items():
                if isinstance(item, dict):
                    tasks.append(normalize_task(str(name), item))
    elif isinstance(parsed, list):
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            name = str(item.get("task_name", item.get("name", f"task_{idx}")))
            tasks.append(normalize_task(name, item))

    tasks = [task for task in tasks if task["task_name"]]
    if not tasks:
        raise ValueError(f"no tasks found in {path}")
    return tasks


def default_task_file(data_root: Path) -> Path:
    candidates = [
        data_root / "task.json",
        data_root / "tasks.json",
        Path.cwd() / "task.json",
        Path.cwd() / "tasks.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def strip_env_comment(value: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(value):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_double:
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            if idx == 0 or value[idx - 1].isspace():
                return value[:idx].rstrip()
    return value.strip()


def unquote_env_value(value: str) -> str:
    value = strip_env_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        inner = value[1:-1]
        if value[0] == '"':
            inner = inner.replace(r"\\", "\\").replace(r"\"", '"')
        return inner
    return value


def load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    result: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                raise ValueError(f"invalid .env line {line_no}: missing '='")
            key, value = line.split("=", 1)
            key = key.strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                raise ValueError(f"invalid .env line {line_no}: invalid key {key!r}")
            result[key] = unquote_env_value(value)
    return result


def env_get(env: Dict[str, str], *keys: str) -> Optional[str]:
    for key in keys:
        value = env.get(key)
        if value is not None and value != "":
            return value
    return None


def env_path(env: Dict[str, str], *keys: str) -> Optional[Path]:
    value = env_get(env, *keys)
    return Path(value) if value is not None else None


def env_int(env: Dict[str, str], default: int, *keys: str) -> int:
    value = env_get(env, *keys)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        joined = ", ".join(keys)
        raise ValueError(f"invalid integer in .env for {joined}: {value!r}") from exc


def env_bool(env: Dict[str, str], default: bool, *keys: str) -> bool:
    value = env_get(env, *keys)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    joined = ", ".join(keys)
    raise ValueError(f"invalid boolean in .env for {joined}: {value!r}")


def env_json_object(env: Dict[str, str], *keys: str) -> Dict[str, Any]:
    value = env_get(env, *keys)
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        joined = ", ".join(keys)
        raise ValueError(f"invalid JSON object in .env for {joined}: {exc}") from exc
    if not isinstance(parsed, dict):
        joined = ", ".join(keys)
        raise ValueError(f"invalid JSON object in .env for {joined}: value must be an object")
    return parsed


def slugify(value: str, fallback: str = "item") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-._")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or fallback


def unique_slug(base: str, existing: Iterable[str], fallback: str = "item") -> str:
    used = set(existing)
    stem = slugify(base, fallback)
    if stem not in used:
        return stem
    idx = 2
    while f"{stem}-{idx}" in used:
        idx += 1
    return f"{stem}-{idx}"


def path_from_user(value: str) -> Path:
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def state_path_from_instance(data_root: Path, instance: Dict[str, Any]) -> Path:
    stored = str(instance.get("state_file", "")).strip()
    if not stored:
        stored = "progress_state.json"
    path = Path(stored).expanduser()
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


class TaskInstanceRegistry:
    """Persistent setup registry for task files and their independent instances."""

    def __init__(self,
                 data_root: Path,
                 seed_task_files: Iterable[Tuple[Path, Optional[Path]]] = ()):
        self.data_root = data_root
        self.registry_file = data_root / "task_backend_registry.json"
        self.lock_file = self.registry_file.with_suffix(self.registry_file.suffix + ".lock")
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.ensure_initialized(seed_task_files)

    @contextmanager
    def locked_registry(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as lock_f:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                registry = self._read_unlocked()
                yield registry
                self._write_unlocked(registry)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def ensure_initialized(self, seed_task_files: Iterable[Tuple[Path, Optional[Path]]]) -> None:
        with self.locked_registry() as registry:
            changed = False
            for task_path, state_file in seed_task_files:
                if not task_path.exists():
                    continue
                _, added = self._add_task_file_unlocked(
                    registry,
                    task_path.resolve(),
                    task_path.name,
                    seed_default=True,
                    default_state_file=state_file.resolve() if state_file else None,
                )
                changed = changed or added
            if changed:
                registry["updated_at"] = now_iso()

    def _read_unlocked(self) -> Registry:
        if not self.registry_file.exists():
            return {"version": 1, "task_files": [], "created_at": now_iso(), "updated_at": now_iso()}
        with self.registry_file.open("r", encoding="utf-8") as f:
            try:
                registry = json.load(f)
            except json.JSONDecodeError as exc:
                raise BackendError(HTTPStatus.INTERNAL_SERVER_ERROR, f"invalid registry file: {exc}") from exc
        if not isinstance(registry, dict):
            raise BackendError(HTTPStatus.INTERNAL_SERVER_ERROR, "registry file root must be an object")
        registry.setdefault("version", 1)
        registry.setdefault("task_files", [])
        registry.setdefault("created_at", now_iso())
        registry.setdefault("updated_at", now_iso())
        if not isinstance(registry["task_files"], list):
            registry["task_files"] = []
        return registry

    def _write_unlocked(self, registry: Registry) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.registry_file.name + ".",
            suffix=".tmp",
            dir=str(self.registry_file.parent),
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(registry, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.registry_file)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def snapshot(self) -> Registry:
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as lock_f:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_SH)
            try:
                registry = self._read_unlocked()
                return json.loads(json.dumps(registry, ensure_ascii=False))
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _add_task_file_unlocked(self,
                                registry: Registry,
                                task_path: Path,
                                label: str,
                                seed_default: bool,
                                default_state_file: Optional[Path] = None) -> Tuple[Dict[str, Any], bool]:
        task_path = task_path.resolve()
        for entry in registry.get("task_files", []):
            if isinstance(entry, dict) and str(entry.get("path", "")) == str(task_path):
                entry.setdefault("instances", [])
                if seed_default and not entry["instances"]:
                    self._add_instance_unlocked(entry, "default", default_state_file)
                return entry, False

        existing_ids = [str(entry.get("id", "")) for entry in registry.get("task_files", []) if isinstance(entry, dict)]
        file_id = unique_slug(task_path.stem, existing_ids, "tasks")
        entry = {
            "id": file_id,
            "label": label.strip() or task_path.name,
            "path": str(task_path),
            "created_at": now_iso(),
            "instances": [],
        }
        if seed_default:
            self._add_instance_unlocked(entry, "default", default_state_file)
        registry.setdefault("task_files", []).append(entry)
        return entry, True

    def _add_instance_unlocked(self,
                               task_file: Dict[str, Any],
                               label: str,
                               state_file: Optional[Path] = None) -> Dict[str, Any]:
        instances = task_file.setdefault("instances", [])
        existing_ids = [str(item.get("id", "")) for item in instances if isinstance(item, dict)]
        instance_id = unique_slug(label, existing_ids, "instance")
        if state_file is None:
            state_file = self.data_root / "instances" / str(task_file["id"]) / instance_id / "progress_state.json"
        instance = {
            "id": instance_id,
            "label": label.strip() or instance_id,
            "state_file": relative_to_root(state_file, self.data_root),
            "created_at": now_iso(),
        }
        instances.append(instance)
        return instance

    def add_task_file(self, task_path: Path, label: str = "") -> Dict[str, Any]:
        if not task_path.exists():
            raise BackendError(HTTPStatus.BAD_REQUEST, f"task file not found: {task_path}")
        if not task_path.is_file():
            raise BackendError(HTTPStatus.BAD_REQUEST, f"task file is not a file: {task_path}")
        load_task_file(task_path)
        with self.locked_registry() as registry:
            entry, added = self._add_task_file_unlocked(
                registry,
                task_path,
                label or task_path.name,
                seed_default=True,
            )
            if added:
                registry["updated_at"] = now_iso()
            return json.loads(json.dumps(entry, ensure_ascii=False))

    def add_instance(self, task_file_id: str, label: str) -> Dict[str, Any]:
        label = label.strip()
        if not label:
            raise BackendError(HTTPStatus.BAD_REQUEST, "instance name is required")
        with self.locked_registry() as registry:
            entry = self._find_task_file_unlocked(registry, task_file_id)
            instance = self._add_instance_unlocked(entry, label)
            registry["updated_at"] = now_iso()
            return json.loads(json.dumps(instance, ensure_ascii=False))

    def _find_task_file_unlocked(self, registry: Registry, task_file_id: str) -> Dict[str, Any]:
        for entry in registry.get("task_files", []):
            if isinstance(entry, dict) and str(entry.get("id", "")) == task_file_id:
                return entry
        raise BackendError(HTTPStatus.NOT_FOUND, f"unknown task file: {task_file_id}")

    def resolve(self, task_file_id: str, instance_id: str) -> Dict[str, Any]:
        registry = self.snapshot()
        entry = self._find_task_file_unlocked(registry, task_file_id)
        for instance in entry.get("instances", []):
            if isinstance(instance, dict) and str(instance.get("id", "")) == instance_id:
                task_path = path_from_user(str(entry.get("path", "")))
                if not task_path.exists():
                    raise BackendError(HTTPStatus.BAD_REQUEST, f"task file not found: {task_path}")
                load_task_file(task_path)
                state_file = state_path_from_instance(self.data_root, instance)
                instance_dir = state_file.parent
                return {
                    "task_file": entry,
                    "instance": instance,
                    "task_path": task_path,
                    "state_file": state_file,
                    "instance_dir": instance_dir,
                }
        raise BackendError(HTTPStatus.NOT_FOUND, f"unknown instance: {instance_id}")


class BackendRuntime:
    def __init__(self, registry: TaskInstanceRegistry, workflow_service: JobService):
        self.registry = registry
        self.accounts = AccountStore(registry.data_root)
        self.workflow_service = workflow_service
        self.lock = threading.RLock()
        self.backend: Optional[TaskBackend] = None
        self.active_selection: Optional[Dict[str, Any]] = None

    def is_started(self) -> bool:
        with self.lock:
            return self.backend is not None

    def start(self, task_file_id: str, instance_id: str) -> Dict[str, Any]:
        with self.lock:
            if self.backend is not None:
                raise BackendError(
                    HTTPStatus.CONFLICT,
                    "task backend is already started; restart the process to choose another instance",
                )
            resolved = self.registry.resolve(task_file_id, instance_id)
            info = {
                "task_file_id": str(resolved["task_file"].get("id", "")),
                "task_file_label": str(resolved["task_file"].get("label", "")),
                "task_file": str(resolved["task_path"]),
                "instance_id": str(resolved["instance"].get("id", "")),
                "instance_label": str(resolved["instance"].get("label", "")),
                "state_file": str(resolved["state_file"]),
                "started_at": now_iso(),
            }
            self.backend = TaskBackend(
                data_root=resolved["instance_dir"],
                task_file=resolved["task_path"],
                state_file=resolved["state_file"],
                runtime_info=info,
                workflow_service=self.workflow_service,
            )
            self.active_selection = info
            return dict(info)


def new_state() -> State:
    return {"version": 1, "subjects": {}}


def ensure_subject(state: State, subject_id: str) -> Dict[str, Any]:
    subjects = state.setdefault("subjects", {})
    subject = subjects.setdefault(subject_id, {})
    subject.setdefault("reservations", {})
    subject.setdefault("idempotency", {})
    return subject


def confirmed_reservations(subject: Dict[str, Any], task_name: str) -> List[Dict[str, Any]]:
    reservations = subject.get("reservations", {})
    result = []
    for item in reservations.values():
        if item.get("task_name") == task_name and item.get("status") == "confirmed":
            result.append(item)
    return result


def task_claim_owner(state: State, task_name: str) -> str:
    subjects = state.get("subjects", {})
    if not isinstance(subjects, dict):
        return ""
    for subject_id in sorted(str(key) for key in subjects.keys()):
        subject = subjects.get(subject_id, {})
        if not isinstance(subject, dict):
            continue
        reservations = subject.get("reservations", {})
        if not isinstance(reservations, dict):
            continue
        for item in reservations.values():
            if not isinstance(item, dict):
                continue
            if item.get("task_name") == task_name and item.get("status") != "released":
                return str(item.get("subject_id", subject_id))
    return ""


def task_progress(subject_id: str, subject: Dict[str, Any], task: Task, state: State) -> Dict[str, Any]:
    completed = len(confirmed_reservations(subject, task["task_name"]))
    total = int(task["total"])
    owner = task_claim_owner(state, str(task["task_name"]))
    return {
        "task_name": task["task_name"],
        "description_cn": task["description_cn"],
        "description_en": task["description_en"],
        "completed": min(completed, total),
        "total": total,
        "claimed_by_subject": owner,
        "claimed_by_other": bool(owner and owner != subject_id),
    }


def progress_payload(subject_id: str, subject: Dict[str, Any], tasks: Iterable[Task], state: State) -> Dict[str, Any]:
    return {"tasks": [task_progress(subject_id, subject, task, state) for task in tasks]}


def next_episode_number(subject: Dict[str, Any], task_name: str) -> int:
    used = set()
    for item in subject.get("reservations", {}).values():
        if item.get("task_name") != task_name:
            continue
        if item.get("status") == "released":
            continue
        try:
            episode = int(item.get("episode_number", 0))
        except (TypeError, ValueError):
            continue
        if episode > 0:
            used.add(episode)
    episode = 1
    while episode in used:
        episode += 1
    return episode


def html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def url_part(value: Any) -> str:
    return quote(str(value), safe="")


def reservation_sort_key(item: Dict[str, Any]) -> Tuple[str, str, int, str]:
    try:
        episode = int(item.get("episode_number", 0))
    except (TypeError, ValueError):
        episode = 0
    return (
        str(item.get("subject_id", "")),
        str(item.get("task_name", "")),
        episode,
        str(item.get("created_at", "")),
    )


def latest_timestamp(items: Iterable[Dict[str, Any]]) -> str:
    latest = ""
    for item in items:
        for key in ("updated_at", "confirmed_at", "released_at", "created_at"):
            value = str(item.get(key, "") or "")
            if value > latest:
                latest = value
    return latest


def count_statuses(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"reserved": 0, "confirmed": 0, "released": 0, "other": 0}
    for item in items:
        status = str(item.get("status", "other"))
        if status in counts:
            counts[status] += 1
        else:
            counts["other"] += 1
    return counts


def parse_nonnegative_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def parse_nonnegative_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    total = int(seconds + 0.5)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def format_bytes(size: Optional[int]) -> str:
    if size is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, size))
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} B"
    return f"{value:.2f} {unit}"


def safe_collection_path(value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text or text == "-":
        return None
    try:
        return path_from_user(text)
    except (OSError, RuntimeError):
        return None


def directory_size_bytes(path: Path) -> Optional[int]:
    try:
        if path.is_file():
            return path.stat().st_size
        if not path.is_dir():
            return None
    except OSError:
        return None

    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def timestamp_csv_stats(episode_dir: Path) -> Dict[str, Optional[float]]:
    path = episode_dir / "timestamps.csv"
    if not path.exists():
        return {"frame_count": None, "duration_seconds": None}
    frame_count = 0
    first_ref_us: Optional[int] = None
    last_ref_us: Optional[int] = None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            ref_idx = header.index("ref_timestamp_us") if "ref_timestamp_us" in header else -1
            for row in reader:
                if not any(cell.strip() for cell in row):
                    continue
                frame_count += 1
                if ref_idx >= 0 and ref_idx < len(row):
                    try:
                        ref_us = int(row[ref_idx])
                    except (TypeError, ValueError):
                        continue
                    if first_ref_us is None:
                        first_ref_us = ref_us
                    last_ref_us = ref_us
    except (OSError, csv.Error, UnicodeDecodeError):
        return {"frame_count": None, "duration_seconds": None}
    duration_seconds: Optional[float] = None
    if first_ref_us is not None and last_ref_us is not None and last_ref_us >= first_ref_us:
        duration_seconds = (last_ref_us - first_ref_us) / 1000000.0
    return {"frame_count": frame_count, "duration_seconds": duration_seconds}


def reservation_duration_seconds(item: Dict[str, Any]) -> Optional[float]:
    for key in ("duration_seconds", "capture_duration_seconds", "duration_sec"):
        parsed = parse_nonnegative_float(item.get(key))
        if parsed is not None:
            return parsed
    return None


def reservation_frame_count(item: Dict[str, Any]) -> Optional[int]:
    for key in ("frame_count", "total_frames", "frames"):
        parsed = parse_nonnegative_int(item.get(key))
        if parsed is not None:
            return parsed
    return None


def episode_stats(item: Dict[str, Any]) -> Dict[str, Any]:
    duration_seconds = reservation_duration_seconds(item)
    frame_count = reservation_frame_count(item)
    collection_path = safe_collection_path(item.get("collection_path"))
    storage_bytes: Optional[int] = None
    path_exists = False
    if collection_path is not None:
        try:
            path_exists = collection_path.exists()
        except OSError:
            path_exists = False
        if path_exists:
            storage_bytes = directory_size_bytes(collection_path)
            csv_stats = timestamp_csv_stats(collection_path)
            if frame_count is None:
                frame_value = csv_stats.get("frame_count")
                frame_count = int(frame_value) if frame_value is not None else None
            if duration_seconds is None:
                duration_seconds = csv_stats.get("duration_seconds")
    return {
        "duration_seconds": duration_seconds,
        "duration_label": format_duration(duration_seconds),
        "frame_count": frame_count,
        "frame_count_label": str(frame_count) if frame_count is not None else "-",
        "storage_bytes": storage_bytes,
        "storage_label": format_bytes(storage_bytes),
        "collection_path_exists": path_exists,
    }


def with_episode_stats(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    out["stats"] = episode_stats(out)
    return out


def sum_duration_seconds(items: Iterable[Dict[str, Any]]) -> float:
    total = 0.0
    for item in items:
        value = item.get("stats", {}).get("duration_seconds")
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def sum_storage_bytes(items: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for item in items:
        value = item.get("stats", {}).get("storage_bytes")
        if isinstance(value, int):
            total += value
    return total


def status_class(status: str) -> str:
    status = str(status or "").strip()
    if status in {
        "confirmed",
        "uploaded",
        "auto_labeled",
        "mano_optimized",
        "qc_passed",
        "review_passed",
        "manual_labeled",
        "manual_labeled",
        "mano_succeeded",
        "finalized",
        "succeeded",
        "complete",
        "active",
    }:
        return "ok"
    if status in {
        "reserved",
        "reserved_for_collection",
        "captured",
        "queued",
        "leased",
        "running",
        "auto_labeling",
        "mano_optimizing",
        "qc_running",
        "review_pending",
        "manual_label_pending",
        "manual_labeling",
        "manual_correction_pending",
        "manual_correction_running",
        "pending_manual",
        "segment_mano_optimizing",
        "mano_queued",
        "mano_running",
    }:
        return "warn"
    if status in {"failed", "canceled", "qc_failed", "qc_bad_episode", "expired"}:
        return "bad"
    if status in {"released", "planned", "missing", "not queued", "-"}:
        return "muted"
    return "neutral"


def metadata_pairs(task: Task) -> List[Tuple[str, str]]:
    raw = task.get("raw", {})
    if not isinstance(raw, dict):
        return []
    hidden = {
        "task_name",
        "name",
        "total",
        "repeat_times",
        "episodes",
        "description_cn",
        "description_en",
        "task_description_cn",
        "task_description_en",
    }
    result: List[Tuple[str, str]] = []
    for key in sorted(raw.keys()):
        if key in hidden:
            continue
        value = raw.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        result.append((str(key), rendered))
    return result


class BackendError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class AccountStore:
    """Tiny file-backed account store for the local collection team."""

    HASH_ITERATIONS = 120_000

    def __init__(self, data_root: Path):
        self.accounts_file = data_root / "accounts.json"
        self.lock_file = self.accounts_file.with_suffix(self.accounts_file.suffix + ".lock")
        data_root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked_accounts(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as lock_f:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                accounts = self._read_unlocked()
                yield accounts
                self._write_unlocked(accounts)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.accounts_file.exists():
            return {"version": 1, "users": {}, "created_at": now_iso(), "updated_at": now_iso()}
        with self.accounts_file.open("r", encoding="utf-8") as f:
            try:
                accounts = json.load(f)
            except json.JSONDecodeError as exc:
                raise BackendError(HTTPStatus.INTERNAL_SERVER_ERROR, f"invalid accounts file: {exc}") from exc
        if not isinstance(accounts, dict):
            raise BackendError(HTTPStatus.INTERNAL_SERVER_ERROR, "accounts file root must be an object")
        accounts.setdefault("version", 1)
        accounts.setdefault("users", {})
        accounts.setdefault("created_at", now_iso())
        accounts.setdefault("updated_at", now_iso())
        if not isinstance(accounts["users"], dict):
            accounts["users"] = {}
        return accounts

    def _write_unlocked(self, accounts: Dict[str, Any]) -> None:
        accounts["updated_at"] = now_iso()
        self.accounts_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.accounts_file.name + ".",
            suffix=".tmp",
            dir=str(self.accounts_file.parent),
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(accounts, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.accounts_file)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _username(value: Any) -> str:
        username = str(value or "").strip()
        if not username:
            raise BackendError(HTTPStatus.BAD_REQUEST, "username is required")
        return username

    @staticmethod
    def _password(value: Any) -> str:
        password = str(value or "")
        if password == "":
            raise BackendError(HTTPStatus.BAD_REQUEST, "password is required")
        return password

    @classmethod
    def _hash_password(cls, password: str, salt_hex: str) -> str:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            cls.HASH_ITERATIONS,
        )
        return digest.hex()

    def register(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        username = self._username(payload.get("username"))
        password = self._password(payload.get("password"))
        password_repeat = self._password(payload.get("password_repeat", payload.get("repeat_password")))
        if password != password_repeat:
            raise BackendError(HTTPStatus.BAD_REQUEST, "passwords do not match")

        with self.locked_accounts() as accounts:
            users = accounts.setdefault("users", {})
            if username in users:
                raise BackendError(HTTPStatus.CONFLICT, "username already exists")
            salt = secrets.token_hex(16)
            created_at = now_iso()
            users[username] = {
                "username": username,
                "password_salt": salt,
                "password_hash": self._hash_password(password, salt),
                "hash": "pbkdf2_sha256",
                "iterations": self.HASH_ITERATIONS,
                "created_at": created_at,
                "updated_at": created_at,
            }
            return {"ok": True, "username": username, "created_at": created_at}

    def login(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        username = self._username(payload.get("username"))
        password = self._password(payload.get("password"))
        with self.locked_accounts() as accounts:
            users = accounts.setdefault("users", {})
            user = users.get(username)
            if not isinstance(user, dict):
                raise BackendError(HTTPStatus.UNAUTHORIZED, "invalid username or password")
            salt = str(user.get("password_salt") or "")
            stored_hash = str(user.get("password_hash") or "")
            if not salt or not stored_hash:
                raise BackendError(HTTPStatus.UNAUTHORIZED, "invalid username or password")
            actual_hash = self._hash_password(password, salt)
            if not hmac.compare_digest(actual_hash, stored_hash):
                raise BackendError(HTTPStatus.UNAUTHORIZED, "invalid username or password")
            user["last_login_at"] = now_iso()
            user["updated_at"] = user["last_login_at"]
            return {
                "ok": True,
                "username": username,
                "created_at": str(user.get("created_at") or ""),
                "last_login_at": user["last_login_at"],
            }


class TaskBackend:
    def __init__(self,
                 data_root: Path,
                 task_file: Path,
                 state_file: Optional[Path] = None,
                 runtime_info: Optional[Dict[str, Any]] = None,
                 workflow_service: Optional[JobService] = None):
        self.data_root = data_root
        self.task_file = task_file
        self.state_file = state_file or (data_root / "progress_state.json")
        self.lock_file = self.state_file.with_suffix(self.state_file.suffix + ".lock")
        self.tasks = load_task_file(task_file)
        self.tasks_by_name = {task["task_name"]: task for task in self.tasks}
        self.runtime_info = runtime_info or {}
        self.workflow_service = workflow_service
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _workflow_hook(self, method_name: str, payload: Dict[str, Any]) -> None:
        if self.workflow_service is None:
            return
        try:
            method = getattr(self.workflow_service, method_name)
            method(dict(payload))
        except Exception as exc:
            print(f"[task-backend] workflow hook {method_name} failed: {exc}", file=sys.stderr)

    @contextmanager
    def locked_state(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as lock_f:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_state_unlocked()
                yield state
                self._write_state_unlocked(state)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _read_state_unlocked(self) -> State:
        if not self.state_file.exists():
            return new_state()
        with self.state_file.open("r", encoding="utf-8") as f:
            try:
                state = json.load(f)
            except json.JSONDecodeError as exc:
                raise BackendError(HTTPStatus.INTERNAL_SERVER_ERROR, f"invalid state file: {exc}") from exc
        if not isinstance(state, dict):
            raise BackendError(HTTPStatus.INTERNAL_SERVER_ERROR, "state file root must be an object")
        state.setdefault("version", 1)
        state.setdefault("subjects", {})
        return state

    def _write_state_unlocked(self, state: State) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.state_file.name + ".",
            suffix=".tmp",
            dir=str(self.state_file.parent),
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.state_file)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def state_snapshot(self) -> State:
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as lock_f:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_state_unlocked()
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def reservation_list(self, state: State) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        subjects = state.get("subjects", {})
        if not isinstance(subjects, dict):
            return result
        for subject_id, subject in subjects.items():
            if not isinstance(subject, dict):
                continue
            reservations = subject.get("reservations", {})
            if not isinstance(reservations, dict):
                continue
            for reservation_key, reservation in reservations.items():
                if not isinstance(reservation, dict):
                    continue
                item = dict(reservation)
                item["reservation_id"] = str(item.get("reservation_id", reservation_key))
                item["subject_id"] = str(item.get("subject_id", subject_id))
                result.append(item)
        result.sort(key=reservation_sort_key)
        return result

    def dashboard_model(self) -> Dict[str, Any]:
        state = self.state_snapshot()
        reservations = [with_episode_stats(item) for item in self.reservation_list(state)]
        subjects_obj = state.get("subjects", {})
        subjects = sorted(str(key) for key in subjects_obj.keys()) if isinstance(subjects_obj, dict) else []
        summaries = []
        for task in self.tasks:
            task_name = str(task["task_name"])
            related = [item for item in reservations if item.get("task_name") == task_name]
            counts = count_statuses(related)
            task_subjects = sorted({str(item.get("subject_id", "")) for item in related if item.get("subject_id")})
            confirmed = [item for item in related if item.get("status") == "confirmed"]
            summaries.append(
                {
                    "task": task,
                    "counts": counts,
                    "subjects": task_subjects,
                    "latest_at": latest_timestamp(related),
                    "reservation_count": len(related),
                    "duration_seconds": sum_duration_seconds(confirmed),
                    "storage_bytes": sum_storage_bytes(confirmed),
                }
            )
        confirmed_reservations = [item for item in reservations if item.get("status") == "confirmed"]
        return {
            "tasks": summaries,
            "subjects": subjects,
            "reservation_count": len(reservations),
            "duration_seconds": sum_duration_seconds(confirmed_reservations),
            "storage_bytes": sum_storage_bytes(confirmed_reservations),
            "state_file": str(self.state_file),
            "task_file": str(self.task_file),
            "runtime_info": self.runtime_info,
        }

    def task_detail_model(self, task_name: str) -> Dict[str, Any]:
        task = self.tasks_by_name.get(task_name)
        if task is None:
            raise BackendError(HTTPStatus.NOT_FOUND, f"unknown task: {task_name}")
        state = self.state_snapshot()
        reservations = [
            with_episode_stats(item)
            for item in self.reservation_list(state)
            if item.get("task_name") == task_name
        ]
        for item in reservations:
            item["workflow_summary"] = self._workflow_summary(str(item.get("reservation_id") or ""))
        subjects = sorted({str(item.get("subject_id", "")) for item in reservations if item.get("subject_id")})
        confirmed = [item for item in reservations if item.get("status") == "confirmed"]
        return {
            "task": task,
            "reservations": reservations,
            "subjects": subjects,
            "counts": count_statuses(reservations),
            "latest_at": latest_timestamp(reservations),
            "duration_seconds": sum_duration_seconds(confirmed),
            "storage_bytes": sum_storage_bytes(confirmed),
        }

    def _workflow_summary(self, reservation_id: str) -> Dict[str, Any]:
        if self.workflow_service is None:
            return {"status": "-", "active_job": "", "job_count": 0}
        try:
            workflow = self.workflow_service.upload_status(reservation_id)
        except Exception as exc:
            return {"status": "missing", "active_job": "", "job_count": 0, "error": str(exc)}
        summary = workflow.get("workflow") if isinstance(workflow.get("workflow"), dict) else {}
        active_type = str(summary.get("active_job_type") or "")
        active_status = str(summary.get("active_job_status") or "")
        active_job = (active_type + ":" + active_status).strip(":") if active_type or active_status else ""
        return {
            "status": str(summary.get("status") or "-"),
            "active_job": active_job,
            "active_job_id": str(summary.get("active_job_id") or ""),
            "job_count": int(summary.get("job_count") or 0),
        }

    def episode_detail_model(self, reservation_id: str) -> Dict[str, Any]:
        state = self.state_snapshot()
        for item in self.reservation_list(state):
            if item.get("reservation_id") == reservation_id:
                task = self.tasks_by_name.get(str(item.get("task_name", "")))
                workflow: Dict[str, Any] = {}
                if self.workflow_service is not None:
                    try:
                        workflow = self.workflow_service.upload_status(reservation_id)
                    except Exception as exc:
                        workflow = {"error": str(exc)}
                return {
                    "reservation": with_episode_stats(item),
                    "task": task,
                    "metadata_pairs": metadata_pairs(task) if task is not None else [],
                    "workflow": workflow,
                }
        raise BackendError(HTTPStatus.NOT_FOUND, f"episode not found: {reservation_id}")

    def delete_episode(self, reservation_id: str) -> Dict[str, Any]:
        reservation_id = reservation_id.strip()
        if not reservation_id:
            raise BackendError(HTTPStatus.BAD_REQUEST, "reservation_id is required")
        with self.locked_state() as state:
            subjects = state.get("subjects", {})
            if not isinstance(subjects, dict):
                raise BackendError(HTTPStatus.NOT_FOUND, f"episode not found: {reservation_id}")
            for subject_id, subject in subjects.items():
                if not isinstance(subject, dict):
                    continue
                reservations = subject.get("reservations", {})
                if not isinstance(reservations, dict) or reservation_id not in reservations:
                    continue
                reservation = reservations.pop(reservation_id)
                idempotency = subject.get("idempotency", {})
                if isinstance(idempotency, dict):
                    for key, value in list(idempotency.items()):
                        if value == reservation_id:
                            del idempotency[key]
                return {
                    "deleted": True,
                    "reservation_id": reservation_id,
                    "subject_id": str(subject_id),
                    "task_name": str(reservation.get("task_name", "")) if isinstance(reservation, dict) else "",
                }
        raise BackendError(HTTPStatus.NOT_FOUND, f"episode not found: {reservation_id}")

    def get_tasks(self, subject_id: str) -> Dict[str, Any]:
        with self.locked_state() as state:
            subject = ensure_subject(state, subject_id)
            return progress_payload(subject_id, subject, self.tasks, state)

    def reserve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client_id = str(payload.get("client_id", "")).strip()
        subject_id = str(payload.get("subject_id", "")).strip()
        task_name = str(payload.get("task_name", "")).strip()
        if not client_id or not subject_id or not task_name:
            raise BackendError(HTTPStatus.BAD_REQUEST, "client_id, subject_id, and task_name are required")
        task = self.tasks_by_name.get(task_name)
        if task is None:
            raise BackendError(HTTPStatus.NOT_FOUND, f"unknown task: {task_name}")

        with self.locked_state() as state:
            subject = ensure_subject(state, subject_id)
            owner = task_claim_owner(state, task_name)
            if owner and owner != subject_id:
                raise BackendError(HTTPStatus.CONFLICT, f"task already claimed by subject: {owner}")
            completed = len(confirmed_reservations(subject, task_name))
            if completed >= int(task["total"]):
                raise BackendError(HTTPStatus.CONFLICT, f"task already complete: {task_name}")

            reservation_id = str(uuid.uuid4())
            episode_number = next_episode_number(subject, task_name)
            reservation = {
                "reservation_id": reservation_id,
                "client_id": client_id,
                "subject_id": subject_id,
                "task_name": task_name,
                "episode_number": episode_number,
                "status": "reserved",
                "created_at": now_iso(),
                "updated_at": now_iso(),
            }
            subject["reservations"][reservation_id] = reservation
            self._workflow_hook("record_collection_reservation", reservation)
            return {
                "reservation_id": reservation_id,
                "task_name": task_name,
                "episode_number": episode_number,
            }

    def confirm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        reservation_id = str(payload.get("reservation_id", "")).strip()
        subject_id = str(payload.get("subject_id", "")).strip()
        task_name = str(payload.get("task_name", "")).strip()
        idempotency_key = str(payload.get("idempotency_key", "")).strip()
        collection_path = str(payload.get("collection_path", "")).strip()
        try:
            episode_number = int(payload.get("episode_number"))
        except (TypeError, ValueError):
            raise BackendError(HTTPStatus.BAD_REQUEST, "episode_number must be an integer")
        if not reservation_id or not subject_id or not task_name or not idempotency_key:
            raise BackendError(HTTPStatus.BAD_REQUEST, "reservation_id, subject_id, task_name, and idempotency_key are required")
        if task_name not in self.tasks_by_name:
            raise BackendError(HTTPStatus.NOT_FOUND, f"unknown task: {task_name}")

        with self.locked_state() as state:
            subject = ensure_subject(state, subject_id)
            existing_reservation_id = subject["idempotency"].get(idempotency_key)
            if existing_reservation_id:
                if existing_reservation_id != reservation_id:
                    raise BackendError(HTTPStatus.CONFLICT, "idempotency_key belongs to another reservation")
                reservation = subject["reservations"].get(reservation_id)
                if reservation is None:
                    raise BackendError(HTTPStatus.NOT_FOUND, "reservation not found for idempotency_key")
                return progress_payload(subject_id, subject, self.tasks, state)

            reservation = subject["reservations"].get(reservation_id)
            if reservation is None:
                raise BackendError(HTTPStatus.NOT_FOUND, f"reservation not found: {reservation_id}")
            if reservation.get("subject_id") != subject_id or reservation.get("task_name") != task_name:
                raise BackendError(HTTPStatus.CONFLICT, "reservation does not match subject/task")
            if int(reservation.get("episode_number", -1)) != episode_number:
                raise BackendError(HTTPStatus.CONFLICT, "reservation does not match episode_number")
            if reservation.get("status") == "released":
                raise BackendError(HTTPStatus.CONFLICT, "reservation has been released")
            if reservation.get("status") == "confirmed":
                previous_key = reservation.get("idempotency_key")
                if previous_key and previous_key != idempotency_key:
                    raise BackendError(HTTPStatus.CONFLICT, "reservation already confirmed with another idempotency_key")
                subject["idempotency"][idempotency_key] = reservation_id
                return progress_payload(subject_id, subject, self.tasks, state)

            reservation["status"] = "confirmed"
            reservation["confirmed_at"] = now_iso()
            reservation["updated_at"] = now_iso()
            reservation["idempotency_key"] = idempotency_key
            reservation["collection_path"] = collection_path
            duration_seconds = parse_nonnegative_float(payload.get("duration_seconds", payload.get("duration_sec")))
            if duration_seconds is not None:
                reservation["duration_seconds"] = round(duration_seconds, 3)
            frame_count = parse_nonnegative_int(payload.get("frame_count", payload.get("total_frames")))
            if frame_count is not None:
                reservation["frame_count"] = frame_count
            subject["idempotency"][idempotency_key] = reservation_id
            self._workflow_hook("record_collection_confirm", reservation)
            return progress_payload(subject_id, subject, self.tasks, state)

    def release(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        reservation_id = str(payload.get("reservation_id", "")).strip()
        subject_id = str(payload.get("subject_id", "")).strip()
        task_name = str(payload.get("task_name", "")).strip()
        if not reservation_id or not subject_id:
            raise BackendError(HTTPStatus.BAD_REQUEST, "reservation_id and subject_id are required")

        with self.locked_state() as state:
            subject = ensure_subject(state, subject_id)
            reservation = subject["reservations"].get(reservation_id)
            if reservation is None:
                return {"released": False, "reason": "not_found", **progress_payload(subject_id, subject, self.tasks, state)}
            if task_name and reservation.get("task_name") != task_name:
                raise BackendError(HTTPStatus.CONFLICT, "reservation does not match task")
            if reservation.get("status") == "reserved":
                reservation["status"] = "released"
                reservation["released_at"] = now_iso()
                reservation["updated_at"] = now_iso()
                released = True
                self._workflow_hook("record_collection_release", reservation)
            else:
                released = False
            return {"released": released, **progress_payload(subject_id, subject, self.tasks, state)}


def render_layout(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-2: #eef2f6;
      --text: #17202a;
      --muted: #647181;
      --line: #d7dde5;
      --accent: #176b87;
      --accent-2: #1f8a70;
      --warn: #a86400;
      --bad: #9d2f3f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    header {{
      background: #0f1c24;
      color: #f8fbfd;
      padding: 18px 28px;
      border-bottom: 4px solid var(--accent-2);
    }}
    header h1 {{ margin: 0; font-size: 24px; font-weight: 650; }}
    main {{ padding: 24px 28px 40px; max-width: 1440px; margin: 0 auto; }}
    .crumbs {{ margin-bottom: 16px; color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px 16px;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .metric .value {{ margin-top: 6px; font-size: 24px; font-weight: 680; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-top: 16px;
      overflow: hidden;
    }}
    section h2 {{
      margin: 0;
      padding: 13px 16px;
      font-size: 16px;
      background: var(--panel-2);
      border-bottom: 1px solid var(--line);
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; font-size: 12px; text-transform: uppercase; letter-spacing: .035em; }}
    tr:last-child td {{ border-bottom: none; }}
    .num {{ text-align: right; white-space: nowrap; }}
    .muted {{ color: var(--muted); }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 650;
      border: 1px solid transparent;
      white-space: nowrap;
    }}
    .badge.ok {{ color: #0f5f46; background: #e5f5ed; border-color: #b7e0cc; }}
    .badge.warn {{ color: var(--warn); background: #fff3dc; border-color: #f0d09b; }}
    .badge.bad {{ color: var(--bad); background: #fff0f2; border-color: #e4a6af; }}
    .badge.muted {{ color: #596675; background: #eef1f4; border-color: #d4dae1; }}
    .badge.neutral {{ color: #535f6d; background: #eef2f7; border-color: #d5dde8; }}
    .desc {{ max-width: 760px; white-space: pre-wrap; }}
    .kv {{ display: grid; grid-template-columns: minmax(160px, 240px) 1fr; }}
    .kv div {{ padding: 10px 12px; border-bottom: 1px solid var(--line); }}
    .kv div:nth-child(odd) {{ color: var(--muted); background: #fbfcfd; }}
    .empty {{ padding: 16px; color: var(--muted); }}
    .notice {{ margin-bottom: 14px; padding: 12px 14px; border-radius: 6px; border: 1px solid var(--line); background: #ffffff; }}
    .notice.warn {{ border-color: #f0d09b; background: #fff8ea; color: var(--warn); }}
    .notice.bad {{ border-color: #e4a6af; background: #fff0f2; color: var(--bad); }}
    form {{ margin: 0; }}
    .form-grid {{ display: grid; grid-template-columns: minmax(180px, 260px) minmax(260px, 1fr); gap: 12px 16px; padding: 16px; align-items: center; }}
    label {{ color: var(--muted); font-weight: 650; }}
    input[type="text"], select {{
      width: 100%;
      min-height: 38px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }}
    input[type="radio"] {{ margin-right: 8px; }}
    button {{
      border: 1px solid var(--accent);
      background: var(--accent);
      color: #fff;
      padding: 9px 14px;
      border-radius: 5px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }}
    button.secondary {{ background: #fff; color: var(--accent); }}
    button.danger {{ background: var(--bad); border-color: var(--bad); }}
    a.button {{
      display: inline-flex;
      align-items: center;
      min-height: 38px;
      border: 1px solid var(--accent);
      background: #fff;
      color: var(--accent);
      padding: 8px 13px;
      border-radius: 5px;
      font-weight: 650;
      text-decoration: none;
    }}
    a.button:hover {{ text-decoration: none; background: #eef8fb; }}
    .actions {{ padding: 0 16px 16px; display: flex; gap: 10px; flex-wrap: wrap; }}
    .top-actions {{ padding-top: 16px; }}
    @media (max-width: 760px) {{
      main {{ padding: 18px 14px 28px; }}
      header {{ padding: 16px; }}
      .wide {{ overflow-x: auto; }}
      .kv {{ grid-template-columns: 1fr; }}
      .form-grid {{ grid-template-columns: 1fr; }}
      .kv div:nth-child(odd) {{ padding-bottom: 2px; }}
      .kv div:nth-child(even) {{ padding-top: 2px; }}
    }}
  </style>
</head>
<body>
  <header><h1>{html_escape(title)}</h1></header>
  <main>{body}</main>
</body>
</html>"""


def render_metric(label: str, value: Any, note: str = "") -> str:
    note_html = f"<div class=\"muted\">{html_escape(note)}</div>" if note else ""
    return (
        "<div class=\"metric\">"
        f"<div class=\"label\">{html_escape(label)}</div>"
        f"<div class=\"value\">{html_escape(value)}</div>"
        f"{note_html}</div>"
    )


def render_status_badge(status: str) -> str:
    return f"<span class=\"badge {status_class(status)}\">{html_escape(status or 'unknown')}</span>"


WORKFLOW_STAGE_LABELS = [
    ("auto_label", "自动标注"),
    ("mano_opt", "MANO 优化"),
    ("qc", "质检结果"),
    ("manual_segment", "人工纠偏"),
]


def render_hidden_inputs(fields: Dict[str, Any]) -> str:
    return "".join(
        f"<input type=\"hidden\" name=\"{html_escape(key)}\" value=\"{html_escape(value)}\">"
        for key, value in fields.items()
        if value is not None
    )


def render_push_auto_label_form(action: str, fields: Dict[str, Any], label: str = "一键推送标注") -> str:
    return (
        f"<form method=\"post\" action=\"{html_escape(action)}\">"
        + render_hidden_inputs(fields)
        + f"<button type=\"submit\">{html_escape(label)}</button>"
        + "</form>"
    )


def render_workflow_stage_shortcuts(include_push_all: bool = False) -> str:
    links = [
        f"<a class=\"button\" href=\"/workflow/stages/{url_part(job_type)}\">{html_escape(label)}：{html_escape(job_type)}</a>"
        for job_type, label in WORKFLOW_STAGE_LABELS
    ]
    if include_push_all:
        links.append(render_push_auto_label_form("/workflow/episodes/push-auto-label", {"scope": "all"}))
    return (
        "<section><h2>Workflow Stages</h2>"
        "<div class=\"actions top-actions\">"
        + "".join(links)
        + "</div></section>"
    )


def render_setup_page(registry: Registry, data_root: Path, message: str = "", error: str = "") -> str:
    task_files = [entry for entry in registry.get("task_files", []) if isinstance(entry, dict)]
    notices = ""
    if message:
        notices += f"<div class=\"notice\">{html_escape(message)}</div>"
    if error:
        notices += f"<div class=\"notice bad\">{html_escape(error)}</div>"

    file_rows = []
    start_rows = []
    task_options = []
    for entry in task_files:
        file_id = str(entry.get("id", ""))
        label = str(entry.get("label", file_id))
        path = str(entry.get("path", ""))
        task_count: Any = "-"
        validation = "OK"
        try:
            task_count = len(load_task_file(path_from_user(path)))
        except Exception as exc:
            validation = str(exc)
        instances = [item for item in entry.get("instances", []) if isinstance(item, dict)]
        task_options.append(f"<option value=\"{html_escape(file_id)}\">{html_escape(label)} - {html_escape(path)}</option>")
        file_rows.append(
            "<tr>"
            f"<td>{html_escape(label)}<div class=\"muted mono\">{html_escape(file_id)}</div></td>"
            f"<td class=\"mono\">{html_escape(path)}</td>"
            f"<td class=\"num\">{html_escape(task_count)}</td>"
            f"<td class=\"num\">{html_escape(len(instances))}</td>"
            f"<td>{html_escape(validation)}</td>"
            "</tr>"
        )
        for instance in instances:
            instance_id = str(instance.get("id", ""))
            instance_label = str(instance.get("label", instance_id))
            state_file = state_path_from_instance(data_root, instance)
            value = f"{file_id}::{instance_id}"
            start_rows.append(
                "<tr>"
                f"<td><label><input type=\"radio\" name=\"selection\" value=\"{html_escape(value)}\" required>"
                f"{html_escape(label)}</label><div class=\"muted mono\">{html_escape(path)}</div></td>"
                f"<td>{html_escape(instance_label)}<div class=\"muted mono\">{html_escape(instance_id)}</div></td>"
                f"<td class=\"mono\">{html_escape(state_file)}</td>"
                "</tr>"
            )

    if not file_rows:
        file_rows.append("<tr><td colspan=\"5\" class=\"empty\">No task files registered yet. Add a tasks.json below.</td></tr>")
    if not start_rows:
        start_rows.append("<tr><td colspan=\"3\" class=\"empty\">No instances available yet.</td></tr>")
    if not task_options:
        task_options.append("<option value=\"\">No task files registered</option>")

    start_disabled = "" if task_files else " disabled"
    body = (
        "<div class=\"crumbs\">Task backend / Setup</div>"
        + notices
        + "<div class=\"notice warn\">Select a task file and one isolated instance before starting the API. "
        "After the backend starts, this process is locked to that instance; restart the process to change it.</div>"
        "<section><h2>Registered Task Files</h2><div class=\"wide\"><table>"
        "<thead><tr><th>Name</th><th>Path</th><th class=\"num\">Tasks</th><th class=\"num\">Instances</th><th>Status</th></tr></thead><tbody>"
        + "\n".join(file_rows)
        + "</tbody></table></div></section>"
        "<section><h2>Start Existing Instance</h2>"
        "<form method=\"post\" action=\"/setup/start\"><div class=\"wide\"><table>"
        "<thead><tr><th>Task file</th><th>Instance</th><th>State file</th></tr></thead><tbody>"
        + "\n".join(start_rows)
        + "</tbody></table></div><div class=\"actions\">"
        f"<button type=\"submit\"{start_disabled}>Start Selected Instance</button>"
        "</div></form></section>"
        "<section><h2>Create Instance And Start</h2>"
        "<form method=\"post\" action=\"/setup/start\">"
        "<input type=\"hidden\" name=\"mode\" value=\"new_instance\">"
        "<div class=\"form-grid\">"
        "<label for=\"task_file_id\">Task file</label>"
        f"<select id=\"task_file_id\" name=\"task_file_id\">{''.join(task_options)}</select>"
        "<label for=\"instance_label\">New instance name</label>"
        "<input id=\"instance_label\" name=\"instance_label\" type=\"text\" placeholder=\"debug-run-1\" required>"
        "</div><div class=\"actions\">"
        f"<button type=\"submit\"{start_disabled}>Create And Start</button>"
        "</div></form></section>"
        "<section><h2>Add Task File</h2>"
        "<form method=\"post\" action=\"/setup/task-files\">"
        "<div class=\"form-grid\">"
        "<label for=\"task_path\">tasks.json path</label>"
        "<input id=\"task_path\" name=\"task_path\" type=\"text\" placeholder=\"./tasks.json\" required>"
        "<label for=\"task_label\">Display name</label>"
        "<input id=\"task_label\" name=\"task_label\" type=\"text\" placeholder=\"optional\">"
        "</div><div class=\"actions\">"
        "<button type=\"submit\" class=\"secondary\">Add Task File</button>"
        "</div></form></section>"
    )
    return render_layout("Task Backend Setup", body)


def render_dashboard(model: Dict[str, Any]) -> str:
    task_rows = []
    totals = {"reserved": 0, "confirmed": 0, "released": 0, "other": 0}
    for summary in model["tasks"]:
        task = summary["task"]
        counts = summary["counts"]
        for key in totals:
            totals[key] += int(counts.get(key, 0))
        task_name = str(task["task_name"])
        subjects = summary["subjects"]
        task_rows.append(
            "<tr>"
            f"<td><a href=\"/tasks/{url_part(task_name)}\">{html_escape(task_name)}</a></td>"
            f"<td class=\"num\">{html_escape(task.get('total', ''))}</td>"
            f"<td class=\"num\">{html_escape(counts.get('confirmed', 0))}</td>"
            f"<td class=\"num\">{html_escape(counts.get('reserved', 0))}</td>"
            f"<td class=\"num\">{html_escape(counts.get('released', 0))}</td>"
            f"<td class=\"num\">{html_escape(format_duration(summary.get('duration_seconds')))}</td>"
            f"<td class=\"num\">{html_escape(format_bytes(summary.get('storage_bytes')))}</td>"
            f"<td>{html_escape(', '.join(subjects) if subjects else '-')}</td>"
            f"<td class=\"muted mono\">{html_escape(summary.get('latest_at') or '-')}</td>"
            "</tr>"
        )
    table_body = "\n".join(task_rows) if task_rows else "<tr><td colspan=\"9\" class=\"empty\">No tasks loaded.</td></tr>"
    subject_note = ", ".join(model["subjects"]) if model["subjects"] else "No subjects yet"
    runtime_info = model.get("runtime_info") or {}
    instance_label = runtime_info.get("instance_label") or "-"
    task_file_label = runtime_info.get("task_file_label") or "-"
    body = (
        f"<div class=\"crumbs\">Task backend / Overview / {html_escape(instance_label)}</div>"
        "<div class=\"notice warn\">This backend process is locked to the selected task file and instance. "
        "Restart the process to choose another instance.</div>"
        + render_workflow_stage_shortcuts(include_push_all=True)
        + "<div class=\"summary\">"
        + render_metric("Tasks", len(model["tasks"]), "from task file")
        + render_metric("Subjects", len(model["subjects"]), subject_note)
        + render_metric("Episodes", model["reservation_count"], "all reservations")
        + render_metric("Confirmed", totals["confirmed"], "completed episodes")
        + render_metric("Total Duration", format_duration(model.get("duration_seconds")), "confirmed episodes")
        + render_metric("Storage", format_bytes(model.get("storage_bytes")), "confirmed collection paths")
        + "</div>"
        "<section><h2>Task Summary</h2><div class=\"wide\"><table>"
        "<thead><tr>"
        "<th>Task</th><th class=\"num\">Required / Subject</th><th class=\"num\">Confirmed</th>"
        "<th class=\"num\">Reserved</th><th class=\"num\">Released</th><th class=\"num\">Duration</th>"
        "<th class=\"num\">Storage</th><th>Subjects</th><th>Latest Update</th>"
        "</tr></thead><tbody>"
        + table_body
        + "</tbody></table></div></section>"
        "<section><h2>Backend Files</h2><div class=\"kv\">"
        f"<div>Task file selection</div><div>{html_escape(task_file_label)}</div>"
        f"<div>Instance</div><div>{html_escape(instance_label)}</div>"
        f"<div>Task file</div><div class=\"mono\">{html_escape(model['task_file'])}</div>"
        f"<div>State file</div><div class=\"mono\">{html_escape(model['state_file'])}</div>"
        f"<div>Started at</div><div class=\"mono\">{html_escape(runtime_info.get('started_at') or '-')}</div>"
        "</div></section>"
    )
    return render_layout("Task Dashboard", body)


def render_task_detail(model: Dict[str, Any]) -> str:
    task = model["task"]
    task_name = str(task["task_name"])
    counts = model["counts"]
    rows = []
    for item in model["reservations"]:
        reservation_id = str(item.get("reservation_id", ""))
        status = str(item.get("status", "unknown"))
        stats = item.get("stats", {})
        workflow_summary = item.get("workflow_summary") if isinstance(item.get("workflow_summary"), dict) else {}
        workflow_status = str(workflow_summary.get("status") or "-")
        active_job = str(workflow_summary.get("active_job") or "")
        workflow_note = f"<div class=\"muted mono\">{html_escape(active_job)}</div>" if active_job else ""
        rows.append(
            "<tr>"
            f"<td><a class=\"mono\" href=\"/episodes/{url_part(reservation_id)}\">{html_escape(reservation_id[:8])}</a></td>"
            f"<td>{html_escape(item.get('subject_id', ''))}</td>"
            f"<td class=\"num\">{html_escape(item.get('episode_number', ''))}</td>"
            f"<td>{render_status_badge(status)}</td>"
            f"<td>{render_status_badge(workflow_status)}{workflow_note}</td>"
            f"<td class=\"num\">{html_escape(stats.get('duration_label', '-'))}</td>"
            f"<td class=\"num\">{html_escape(stats.get('frame_count_label', '-'))}</td>"
            f"<td class=\"num\">{html_escape(stats.get('storage_label', '-'))}</td>"
            f"<td class=\"mono\">{html_escape(item.get('client_id', '-'))}</td>"
            f"<td class=\"muted mono\">{html_escape(item.get('updated_at') or item.get('created_at') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(item.get('collection_path') or '-')}</td>"
            "</tr>"
        )
    episode_rows = "\n".join(rows) if rows else "<tr><td colspan=\"11\" class=\"empty\">No episodes reserved yet.</td></tr>"
    meta_rows = []
    for key, value in metadata_pairs(task):
        meta_rows.append(f"<div>{html_escape(key)}</div><div>{html_escape(value)}</div>")
    if not meta_rows:
        meta_rows.append("<div>Extra metadata</div><div class=\"muted\">No extra task metadata.</div>")
    description = task.get("description_cn") or task.get("description_en") or ""
    body = (
        f"<div class=\"crumbs\"><a href=\"/\">Task backend</a> / {html_escape(task_name)}</div>"
        "<section><h2>Workflow Action</h2><div class=\"actions top-actions\">"
        + render_push_auto_label_form(f"/tasks/{url_part(task_name)}/push-auto-label", {"task_name": task_name})
        + "</div></section>"
        + "<div class=\"summary\">"
        + render_metric("Required / Subject", task.get("total", ""), "configured episodes")
        + render_metric("Confirmed", counts.get("confirmed", 0))
        + render_metric("Reserved", counts.get("reserved", 0))
        + render_metric("Released", counts.get("released", 0))
        + render_metric("Total Duration", format_duration(model.get("duration_seconds")), "confirmed episodes")
        + render_metric("Storage", format_bytes(model.get("storage_bytes")), "confirmed collection paths")
        + "</div>"
        "<section><h2>Task Description</h2>"
        f"<div class=\"empty desc\">{html_escape(description or 'No description.')}</div></section>"
        "<section><h2>Task Metadata</h2><div class=\"kv\">"
        + "".join(meta_rows)
        + f"<div>Subjects</div><div>{html_escape(', '.join(model['subjects']) if model['subjects'] else '-')}</div>"
        + f"<div>Latest update</div><div class=\"mono\">{html_escape(model.get('latest_at') or '-')}</div>"
        + "</div></section>"
        "<section><h2>Episodes</h2><div class=\"wide\"><table>"
        "<thead><tr><th>Episode</th><th>Subject</th><th class=\"num\">No.</th><th>Status</th><th>Workflow</th>"
        "<th class=\"num\">Duration</th><th class=\"num\">Frames</th><th class=\"num\">Storage</th>"
        "<th>Client</th><th>Updated</th><th>Local Path</th></tr></thead><tbody>"
        + episode_rows
        + "</tbody></table></div></section>"
    )
    return render_layout(task_name, body)


def stage_scope_label(item: Dict[str, Any]) -> str:
    scope = str(item.get("scope") or "").strip()
    if scope:
        return scope
    return "-"


def stage_frames_label(item: Dict[str, Any]) -> str:
    frames = item.get("frames_count")
    return str(frames) if frames is not None else "-"


def stage_episode_cell(item: Dict[str, Any]) -> str:
    episode_id = str(item.get("episode_id") or "")
    if not episode_id:
        return "-"
    label = episode_id[:8]
    url = str(item.get("episode_url") or f"/episodes/{url_part(episode_id)}")
    return f"<a class=\"mono\" href=\"{html_escape(url)}\">{html_escape(label)}</a>"


def stage_subject_cell(item: Dict[str, Any]) -> str:
    episode_index = item.get("episode_index")
    episode_label = "-" if episode_index is None else str(episode_index)
    return (
        f"{html_escape(item.get('subject_id') or '-')}"
        f"<div class=\"muted mono\">{html_escape(item.get('task_name') or '-')} / {html_escape(episode_label)}</div>"
    )


def render_workflow_stage_page(stage: Dict[str, Any]) -> str:
    job_type = str(stage.get("job_type") or "")
    control = stage.get("control") if isinstance(stage.get("control"), dict) else {}
    stats = stage.get("stats") if isinstance(stage.get("stats"), dict) else {}
    lease_label = "开放" if control.get("lease_enabled") else "暂停"
    enable_disabled = " disabled" if control.get("lease_enabled") else ""
    disable_disabled = "" if control.get("lease_enabled") else " disabled"

    active_rows = []
    for item in stage.get("active") or []:
        if not isinstance(item, dict):
            continue
        active_rows.append(
            "<tr>"
            f"<td class=\"mono\">{html_escape(item.get('job_id') or '-')}</td>"
            f"<td>{stage_episode_cell(item)}</td>"
            f"<td>{stage_subject_cell(item)}</td>"
            f"<td class=\"mono\">{html_escape(item.get('lease_owner') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(item.get('lease_until') or '-')}</td>"
            f"<td>{render_status_badge('expired' if item.get('lease_expired') else 'active')}</td>"
            f"<td class=\"mono\">{html_escape(item.get('updated_at') or '-')}</td>"
            f"<td class=\"num\">{html_escape(item.get('attempt') or 0)}</td>"
            f"<td class=\"num\">{html_escape(stage_scope_label(item))}</td>"
            f"<td class=\"num\">{html_escape(stage_frames_label(item))}</td>"
            "</tr>"
        )
    active_html = "\n".join(active_rows) if active_rows else "<tr><td colspan=\"10\" class=\"empty\">No active leases.</td></tr>"

    queued_rows = []
    for item in stage.get("queued") or []:
        if not isinstance(item, dict):
            continue
        wait = item.get("waiting_seconds")
        queued_rows.append(
            "<tr>"
            f"<td class=\"mono\">{html_escape(item.get('job_id') or '-')}</td>"
            f"<td>{stage_episode_cell(item)}</td>"
            f"<td>{stage_subject_cell(item)}</td>"
            f"<td class=\"mono\">{html_escape(item.get('created_at') or '-')}</td>"
            f"<td class=\"num\">{html_escape(format_duration(float(wait)) if wait is not None else '-')}</td>"
            f"<td class=\"num\">{html_escape(stage_scope_label(item))}</td>"
            f"<td class=\"num\">{html_escape(stage_frames_label(item))}</td>"
            "</tr>"
        )
    queued_html = "\n".join(queued_rows) if queued_rows else "<tr><td colspan=\"7\" class=\"empty\">No queued jobs.</td></tr>"

    completed_rows = []
    for item in stage.get("completed") or []:
        if not isinstance(item, dict):
            continue
        completed_rows.append(
            "<tr>"
            f"<td class=\"mono\">{html_escape(item.get('job_id') or '-')}</td>"
            f"<td>{stage_episode_cell(item)}</td>"
            f"<td>{render_status_badge(str(item.get('status') or '-'))}</td>"
            f"<td class=\"mono\">{html_escape(item.get('updated_at') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(item.get('result_summary') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(item.get('error_summary') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(item.get('artifact_summary') or '-')}</td>"
            "</tr>"
        )
    completed_html = "\n".join(completed_rows) if completed_rows else "<tr><td colspan=\"7\" class=\"empty\">No completed jobs.</td></tr>"

    body = (
        f"<div class=\"crumbs\"><a href=\"/\">Task backend</a> / Workflow / {html_escape(job_type)}</div>"
        "<div class=\"summary\">"
        + render_metric("租借状态", lease_label, str(control.get("updated_at") or ""))
        + render_metric("Queued", stats.get("queued", 0))
        + render_metric("Leased / Running", stats.get("leased_running", 0))
        + render_metric("Succeeded", stats.get("succeeded", 0))
        + render_metric("Failed", stats.get("failed", 0))
        + "</div>"
        "<section><h2>Stage Control</h2><div class=\"actions top-actions\">"
        f"<form method=\"post\" action=\"/workflow/stages/{url_part(job_type)}/enable\">"
        f"<button type=\"submit\"{enable_disabled}>开启</button></form>"
        f"<form method=\"post\" action=\"/workflow/stages/{url_part(job_type)}/disable\">"
        f"<button type=\"submit\" class=\"secondary\"{disable_disabled}>暂停</button></form>"
        f"<span class=\"muted mono\">updated_by={html_escape(control.get('updated_by') or '-')} note={html_escape(control.get('note') or '-')}</span>"
        "</div></section>"
        "<section><h2>Active Leases</h2><div class=\"wide\"><table>"
        "<thead><tr><th>Job</th><th>Episode</th><th>Subject / Task / Episode</th><th>Lease Owner</th>"
        "<th>Lease Until</th><th>Expired</th><th>Updated</th><th class=\"num\">Attempt</th>"
        "<th class=\"num\">Scope</th><th class=\"num\">Frames</th></tr></thead><tbody>"
        + active_html
        + "</tbody></table></div></section>"
        "<section><h2>Queued</h2><div class=\"wide\"><table>"
        "<thead><tr><th>Job</th><th>Episode</th><th>Subject / Task / Episode</th><th>Created</th>"
        "<th class=\"num\">Waiting</th><th class=\"num\">Scope</th><th class=\"num\">Frames</th></tr></thead><tbody>"
        + queued_html
        + "</tbody></table></div></section>"
        "<section><h2>Completed</h2><div class=\"wide\"><table>"
        "<thead><tr><th>Job</th><th>Episode</th><th>Status</th><th>Updated</th><th>Result</th><th>Error</th><th>Artifacts</th></tr></thead><tbody>"
        + completed_html
        + "</tbody></table></div></section>"
    )
    return render_layout("Workflow " + job_type, body)


def render_episode_detail(model: Dict[str, Any]) -> str:
    item = model["reservation"]
    task = model["task"]
    task_name = str(item.get("task_name", ""))
    reservation_id = str(item.get("reservation_id", ""))
    status = str(item.get("status", "unknown"))
    stats = item.get("stats", {})
    workflow = model.get("workflow") or {}
    workflow_episode = workflow.get("episode") if isinstance(workflow.get("episode"), dict) else {}
    upload = workflow.get("upload") if isinstance(workflow.get("upload"), dict) else {}
    upload_available = bool(upload.get("available")) if isinstance(upload, dict) else False
    upload_status = str(upload.get("status") or ("queued" if upload_available else "not queued")) if isinstance(upload, dict) else "not queued"
    upload_phase = str(upload.get("phase") or "-") if isinstance(upload, dict) else "-"
    upload_percent = upload.get("percent", 0) if isinstance(upload, dict) else 0
    try:
        upload_percent_label = f"{float(upload_percent):.1f}%"
    except (TypeError, ValueError):
        upload_percent_label = "-"
    upload_copied = int(upload.get("copied_bytes") or 0) if isinstance(upload, dict) else 0
    upload_total = int(upload.get("total_bytes") or 0) if isinstance(upload, dict) else 0
    upload_files_done = int(upload.get("files_done") or 0) if isinstance(upload, dict) else 0
    upload_files_total = int(upload.get("files_total") or 0) if isinstance(upload, dict) else 0
    upload_nas_uri = str(upload.get("nas_uri") or "") if isinstance(upload, dict) else ""
    upload_error = str(upload.get("error") or workflow.get("error") or "") if isinstance(upload, dict) else str(workflow.get("error") or "")
    collection_path = str(item.get("collection_path") or upload.get("collection_path") or "")
    workflow_info = workflow.get("workflow") if isinstance(workflow.get("workflow"), dict) else {}
    workflow_status = str(workflow_info.get("status") or "-")
    active_job_type = str(workflow_info.get("active_job_type") or "")
    active_job_status = str(workflow_info.get("active_job_status") or "")
    active_job_id = str(workflow_info.get("active_job_id") or "")
    active_job = (active_job_type + ":" + active_job_status).strip(":") if active_job_type or active_job_status else "-"
    workflow_job_count = int(workflow_info.get("job_count") or 0)
    jobs = workflow.get("jobs") if isinstance(workflow.get("jobs"), list) else []
    workflow_artifacts = workflow.get("workflow_artifacts") if isinstance(workflow.get("workflow_artifacts"), list) else []
    segments = workflow.get("segments") if isinstance(workflow.get("segments"), list) else []

    def latest_artifact(kinds: set) -> Dict[str, Any]:
        for artifact in reversed(workflow_artifacts):
            if isinstance(artifact, dict) and str(artifact.get("kind") or "") in kinds:
                return artifact
        return {}

    def jobs_of_type(job_type: str) -> List[Dict[str, Any]]:
        return [job for job in jobs if isinstance(job, dict) and str(job.get("type") or "") == job_type]

    def latest_job(job_type: str) -> Dict[str, Any]:
        typed = jobs_of_type(job_type)
        return typed[-1] if typed else {}

    def compact_value(value: Any, fallback: str = "-") -> str:
        text = str(value if value is not None else "").strip()
        return text or fallback

    def uri_cell(value: Any) -> str:
        text = compact_value(value)
        return f"<span class=\"mono\">{html_escape(text)}</span>"

    def flow_row(step: str, stage_status: str, updated_at: Any, detail: Any, artifact: Any = "") -> str:
        artifact_html = uri_cell(artifact) if compact_value(artifact) != "-" else "<span class=\"muted\">-</span>"
        return (
            "<tr>"
            f"<td>{html_escape(step)}</td>"
            f"<td>{render_status_badge(stage_status)}</td>"
            f"<td class=\"mono\">{html_escape(compact_value(updated_at))}</td>"
            f"<td>{html_escape(compact_value(detail))}</td>"
            f"<td>{artifact_html}</td>"
            "</tr>"
        )

    def flow_status_from_job(job_type: str, fallback: str = "not queued") -> str:
        job = latest_job(job_type)
        return str(job.get("status") or fallback) if job else fallback

    auto_job = latest_job("auto_label")
    mano_job = latest_job("mano_opt")
    qc_job = latest_job("qc")
    auto_artifact = latest_artifact({"pred_2d", "auto_2d"})
    mano_episode_artifact = latest_artifact({"mano_episode"})
    qc_report_artifact = latest_artifact({"qc_report"})
    final_manifest_uri = upload_nas_uri.rstrip("/") + "/workflow/final_3d_sources.json" if upload_nas_uri else ""

    segment_counts: Dict[str, int] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_status = str(segment.get("status") or "unknown")
        segment_counts[segment_status] = segment_counts.get(segment_status, 0) + 1
    segment_total = sum(segment_counts.values())
    segment_ready = segment_counts.get("mano_succeeded", 0)
    segment_pending = segment_counts.get("pending_manual", 0) + segment_counts.get("manual_labeling", 0) + segment_counts.get("mano_queued", 0) + segment_counts.get("mano_running", 0)

    if status == "released":
        capture_status = "released"
    elif status == "confirmed":
        capture_status = "confirmed"
    elif status == "reserved":
        capture_status = "reserved"
    else:
        capture_status = status

    if segment_total:
        correction_detail = f"{segment_ready}/{segment_total} segments ready"
        correction_status = "finalized" if segment_ready == segment_total and workflow_status == "finalized" else workflow_status
    elif workflow_status in {"qc_failed", "manual_correction_pending", "manual_correction_running", "segment_mano_optimizing"}:
        correction_detail = "waiting for failed segments"
        correction_status = workflow_status
    else:
        correction_detail = "not needed"
        correction_status = "not queued"

    job_rows = []
    failed_job_alerts = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        scope_label = str(job.get("scope") or "").strip()
        if not scope_label:
            scope_label = "-"
        frames = job.get("frames")
        frames_label = str(frames) if frames is not None else "-"
        error = str(job.get("error") or "")
        if str(job.get("status") or "") == "failed":
            failed_job_alerts.append(
                f"{job.get('job_id') or '-'} / {job.get('type') or '-'} / {error or job.get('error_summary') or '-'}"
            )
        job_rows.append(
            "<tr>"
            f"<td class=\"mono\">{html_escape(job.get('job_id') or '-')}</td>"
            f"<td>{html_escape(job.get('type') or '-')}</td>"
            f"<td>{render_status_badge(str(job.get('status') or '-'))}</td>"
            f"<td class=\"num\">{html_escape(scope_label)}</td>"
            f"<td class=\"num\">{html_escape(frames_label)}</td>"
            f"<td class=\"mono\">{html_escape(job.get('lease_owner') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(job.get('updated_at') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(error or '-')}</td>"
            "</tr>"
        )
    workflow_jobs_html = "\n".join(job_rows) if job_rows else "<tr><td colspan=\"8\" class=\"empty\">No workflow jobs yet.</td></tr>"
    artifact_rows = []
    for artifact in workflow_artifacts:
        if not isinstance(artifact, dict):
            continue
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        source = str(metadata.get("source_job_id") or metadata.get("source") or metadata.get("segment_id") or "")
        artifact_rows.append(
            "<tr>"
            f"<td>{html_escape(artifact.get('kind') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(artifact.get('uri') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(source or '-')}</td>"
            f"<td class=\"mono\">{html_escape(artifact.get('created_at') or '-')}</td>"
            "</tr>"
        )
    artifacts_html = "\n".join(artifact_rows) if artifact_rows else "<tr><td colspan=\"4\" class=\"empty\">No workflow artifacts yet.</td></tr>"

    segment_rows = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        frame_range = f"{segment.get('start_frame', '-')}-{segment.get('end_frame', '-')}"
        metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
        reason = str(metadata.get("reason") or metadata.get("label") or "")
        error = str(segment.get("error") or "")
        segment_rows.append(
            "<tr>"
            f"<td class=\"mono\">{html_escape(segment.get('segment_id') or '-')}</td>"
            f"<td class=\"num\">{html_escape(frame_range)}</td>"
            f"<td>{render_status_badge(str(segment.get('status') or '-'))}</td>"
            f"<td>{html_escape(reason or '-')}</td>"
            f"<td class=\"mono\">{html_escape(segment.get('lease_owner') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(segment.get('manual_2d_uri') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(segment.get('mano_patch_uri') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(error or '-')}</td>"
            "</tr>"
        )
    segments_html = "\n".join(segment_rows) if segment_rows else "<tr><td colspan=\"8\" class=\"empty\">QC passed or no failed segments have been created.</td></tr>"

    failed_alert_html = ""
    if failed_job_alerts:
        failed_alert_html = (
            "<div class=\"notice bad\">Worker failure: "
            + html_escape(" | ".join(failed_job_alerts[:5]))
            + (" ..." if len(failed_job_alerts) > 5 else "")
            + "</div>"
        )

    flow_rows = [
        flow_row(
            "1. 采集预约 / 确认",
            capture_status,
            item.get("confirmed_at") or item.get("updated_at") or item.get("created_at"),
            f"{item.get('subject_id', '-')} / episode {item.get('episode_number', '-')}",
            collection_path,
        ),
        flow_row(
            "2. 上传到 NAS",
            upload_status,
            upload.get("updated_at") if isinstance(upload, dict) else "",
            f"{upload_phase} {upload_percent_label} | {upload_files_done}/{upload_files_total} files",
            upload_nas_uri,
        ),
        flow_row(
            "3. 自动 2D 标注",
            flow_status_from_job("auto_label"),
            auto_job.get("updated_at") if auto_job else "",
            auto_job.get("job_id") if auto_job else "waiting for push",
            auto_artifact.get("uri") if auto_artifact else "",
        ),
        flow_row(
            "4. Episode MANO 优化",
            flow_status_from_job("mano_opt"),
            mano_job.get("updated_at") if mano_job else "",
            mano_job.get("job_id") if mano_job else "waiting for auto label",
            mano_episode_artifact.get("uri") if mano_episode_artifact else "",
        ),
        flow_row(
            "5. QC 质检",
            flow_status_from_job("qc"),
            qc_job.get("updated_at") if qc_job else "",
            qc_job.get("job_id") if qc_job else "waiting for MANO",
            qc_report_artifact.get("uri") if qc_report_artifact else "",
        ),
        flow_row(
            "6. 人工纠偏 / 最终 3D",
            correction_status,
            workflow_info.get("updated_at") or workflow_episode.get("updated_at") or item.get("updated_at"),
            correction_detail,
            final_manifest_uri,
        ),
    ]

    capture_fields = [
        ("任务", task_name),
        ("采集对象", item.get("subject_id", "")),
        ("Episode 序号", item.get("episode_number", "")),
        ("采集状态", status),
        ("采集时长", stats.get("duration_label", "-")),
        ("帧数", stats.get("frame_count_label", "-")),
        ("本地大小", stats.get("storage_label", "-")),
        ("本地路径", collection_path or "-"),
        ("创建时间", item.get("created_at") or "-"),
        ("确认时间", item.get("confirmed_at") or "-"),
    ]
    capture_html = "".join(
        f"<div>{html_escape(label)}</div><div class=\"mono\">{html_escape(value)}</div>"
        for label, value in capture_fields
    )

    storage_fields = [
        ("上传状态", upload_status),
        ("上传阶段", upload_phase),
        ("上传进度", upload_percent_label),
        ("上传文件", f"{upload_files_done} / {upload_files_total}"),
        ("上传字节", f"{format_bytes(upload_copied)} / {format_bytes(upload_total)}"),
        ("NAS Episode URI", upload_nas_uri or "-"),
        ("Workflow Episode URI", workflow_episode.get("episode_uri") or "-"),
        ("上传错误", upload_error or "-"),
    ]
    storage_html = "".join(
        f"<div>{html_escape(label)}</div><div class=\"mono\">{html_escape(value)}</div>"
        for label, value in storage_fields
    )

    workflow_fields = [
        ("当前状态", workflow_status),
        ("当前活跃任务", active_job),
        ("活跃任务 ID", active_job_id or "-"),
        ("Workflow Job 数", workflow_job_count),
        ("失败分段数", segment_total),
        ("已完成纠偏分段", segment_ready),
        ("待处理纠偏分段", segment_pending),
        ("更新时间", workflow_info.get("updated_at") or workflow_episode.get("updated_at") or "-"),
    ]
    workflow_html = "".join(
        f"<div>{html_escape(label)}</div><div class=\"mono\">{html_escape(value)}</div>"
        for label, value in workflow_fields
    )

    meta_html = "".join(
        f"<div>{html_escape(key)}</div><div>{html_escape(value)}</div>"
        for key, value in model["metadata_pairs"]
    )
    if not meta_html:
        meta_html = "<div>Task metadata</div><div class=\"muted\">No extra task metadata.</div>"

    trace_fields = [
        ("Reservation ID", reservation_id),
        ("Workflow Episode ID", workflow_episode.get("episode_id") or "-"),
        ("Client ID", item.get("client_id") or "-"),
        ("Idempotency Key", item.get("idempotency_key") or "-"),
        ("Reservation Updated", item.get("updated_at") or "-"),
        ("Workflow Updated", workflow_episode.get("updated_at") or "-"),
        ("Released At", item.get("released_at") or "-"),
    ]
    trace_html = "".join(
        f"<div>{html_escape(label)}</div><div class=\"mono\">{html_escape(value)}</div>"
        for label, value in trace_fields
    )

    body = (
        f"<div class=\"crumbs\"><a href=\"/\">Task backend</a> / "
        f"<a href=\"/tasks/{url_part(task_name)}\">{html_escape(task_name)}</a> / "
        f"{html_escape(reservation_id[:8])}</div>"
        "<div class=\"summary\">"
        + render_metric("Episode", item.get("episode_number", "-"), str(item.get("subject_id", "")))
        + render_metric("采集状态", status)
        + render_metric("Duration", stats.get("duration_label", "-"))
        + render_metric("Frames", stats.get("frame_count_label", "-"))
        + render_metric("NAS Upload", upload_status, upload_percent_label if upload_available else "waiting")
        + render_metric("Workflow", workflow_status, active_job if active_job != "-" else "idle")
        + "</div>"
        + failed_alert_html
        + "<section><h2>当前流程</h2><div class=\"actions top-actions\">"
        + render_push_auto_label_form(f"/episodes/{url_part(reservation_id)}/push-auto-label", {"episode_id": reservation_id}, "推送自动标注")
        + "</div><div class=\"wide\"><table>"
        "<thead><tr><th>步骤</th><th>状态</th><th>更新时间</th><th>说明</th><th>关键路径 / 产物</th></tr></thead><tbody>"
        + "\n".join(flow_rows)
        + "</tbody></table></div></section>"
        "<section><h2>采集信息</h2><div class=\"kv\">"
        + capture_html
        + "</div></section>"
        "<section><h2>上传与存储</h2><div class=\"kv\">"
        + storage_html
        + "</div></section>"
        "<section><h2>自动处理 / 质检</h2><div class=\"kv\">"
        + workflow_html
        + "</div></section>"
        "<section><h2>流程产物</h2><div class=\"wide\"><table>"
        "<thead><tr><th>类型</th><th>URI</th><th>来源</th><th>创建时间</th></tr></thead><tbody>"
        + artifacts_html
        + "</tbody></table></div></section>"
        "<section><h2>QC 失败分段 / 人工纠偏</h2><div class=\"wide\"><table>"
        "<thead><tr><th>Segment</th><th class=\"num\">Frames</th><th>Status</th><th>原因</th><th>Lease Owner</th>"
        "<th>Manual 2D</th><th>MANO Patch</th><th>Error</th></tr></thead><tbody>"
        + segments_html
        + "</tbody></table></div></section>"
        "<section><h2>Workflow Jobs</h2><div class=\"wide\"><table>"
        "<thead><tr><th>Job</th><th>Type</th><th>Status</th><th class=\"num\">Scope</th>"
        "<th class=\"num\">Frames</th><th>Owner</th><th>Updated</th><th>Error</th></tr></thead><tbody>"
        + workflow_jobs_html
        + "</tbody></table></div></section>"
        "<section><h2>任务定义快照</h2><div class=\"kv\">"
        + meta_html
        + "</div></section>"
        "<section><h2>后端记录</h2><div class=\"kv\">"
        + trace_html
        + "</div></section>"
        "<section><h2>管理操作</h2>"
        "<div class=\"empty\">删除只会移除后端进度记录，不会删除本地采集文件或 NAS 文件。</div>"
        f"<form method=\"post\" action=\"/episodes/{url_part(reservation_id)}/delete\" "
        "onsubmit=\"return confirm('Delete this episode from backend only? Local and NAS files will not be removed.');\">"
        "<div class=\"actions\"><button type=\"submit\" class=\"danger\">Delete Episode From Backend</button></div></form>"
        "</section>"
    )
    return render_layout("Episode " + reservation_id[:8], body)


def render_error_page(status: HTTPStatus, message: str) -> str:
    body = (
        "<div class=\"crumbs\"><a href=\"/\">Task backend</a> / Error</div>"
        "<section><h2>Error</h2>"
        f"<div class=\"empty\">{html_escape(int(status))} {html_escape(status.phrase)}: {html_escape(message)}</div>"
        "</section>"
    )
    return render_layout("Task Backend Error", body)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "OrbbecTaskBackend/1.0"

    @property
    def runtime(self) -> BackendRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    @property
    def backend(self) -> TaskBackend:
        with self.runtime.lock:
            backend = self.runtime.backend
        if backend is None:
            raise BackendError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "task backend instance is not started; open the setup page and choose a task file instance",
            )
        return backend

    @property
    def workflow(self) -> JobService:
        return self.runtime.workflow_service

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (now_iso(), fmt % args))

    def _json_response(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, status: HTTPStatus, html_body: str) -> None:
        body = html_body.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(int(HTTPStatus.SEE_OTHER))
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise BackendError(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}") from exc
        if not isinstance(parsed, dict):
            raise BackendError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
        return parsed

    def _read_form(self) -> Dict[str, str]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise BackendError(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
        raw = self.rfile.read(max(0, length))
        try:
            parsed = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        except UnicodeDecodeError as exc:
            raise BackendError(HTTPStatus.BAD_REQUEST, f"invalid form body: {exc}") from exc
        return {key: values[0].strip() if values else "" for key, values in parsed.items()}

    def _setup_page(self, message: str = "", error: str = "") -> str:
        return render_setup_page(self.runtime.registry.snapshot(), self.runtime.registry.data_root, message, error)

    @staticmethod
    def _path_job_action(path: str, prefix: str) -> Optional[Tuple[str, str]]:
        if not path.startswith(prefix + "/"):
            return None
        rest = path[len(prefix) + 1:].strip("/")
        parts = rest.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        return unquote(parts[0]), parts[1]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        is_api = parsed.path.startswith("/api/")
        try:
            workflow_stage_api_prefix = "/api/v1/workflow/stages/"
            if parsed.path.startswith(workflow_stage_api_prefix):
                job_type = unquote(parsed.path[len(workflow_stage_api_prefix):].strip("/")).strip()
                if not job_type or "/" in job_type:
                    raise BackendError(HTTPStatus.NOT_FOUND, "workflow stage not found")
                self._json_response(HTTPStatus.OK, self.workflow.workflow_stage(job_type))
                return

            if parsed.path.startswith("/api/v1/jobs/"):
                job_id = unquote(parsed.path[len("/api/v1/jobs/"):].strip("/")).strip()
                if not job_id or "/" in job_id:
                    raise BackendError(HTTPStatus.NOT_FOUND, "job not found")
                self._json_response(HTTPStatus.OK, self.workflow.get_job(job_id))
                return

            if parsed.path.startswith("/api/v1/label/jobs/"):
                job_id = unquote(parsed.path[len("/api/v1/label/jobs/"):].strip("/")).strip()
                if not job_id or "/" in job_id:
                    raise BackendError(HTTPStatus.NOT_FOUND, "label job not found")
                self._json_response(HTTPStatus.OK, self.workflow.get_job(job_id))
                return

            if parsed.path == "/api/v1/label/tasks":
                self._json_response(HTTPStatus.OK, self.workflow.label_tasks())
                return

            label_task_prefix = "/api/v1/label/tasks/"
            if parsed.path.startswith(label_task_prefix) and parsed.path.endswith("/episodes"):
                raw_name = parsed.path[len(label_task_prefix):-len("/episodes")]
                task_name = unquote(raw_name.strip("/")).strip()
                if not task_name:
                    raise BackendError(HTTPStatus.NOT_FOUND, "label task not found")
                self._json_response(HTTPStatus.OK, self.workflow.label_task_episodes(task_name))
                return

            collection_upload_prefix = "/api/v1/collection/episodes/"
            upload_prefix = "/api/v1/episodes/"
            if parsed.path.startswith(collection_upload_prefix) and parsed.path.endswith("/upload"):
                episode_id = unquote(parsed.path[len(collection_upload_prefix):-len("/upload")].strip("/")).strip()
                if not episode_id:
                    raise BackendError(HTTPStatus.NOT_FOUND, "episode not found")
                self._json_response(HTTPStatus.OK, self.workflow.upload_status(episode_id))
                return
            if parsed.path.startswith(upload_prefix) and parsed.path.endswith("/upload"):
                episode_id = unquote(parsed.path[len(upload_prefix):-len("/upload")].strip("/")).strip()
                if not episode_id:
                    raise BackendError(HTTPStatus.NOT_FOUND, "episode not found")
                self._json_response(HTTPStatus.OK, self.workflow.upload_status(episode_id))
                return

            if not self.runtime.is_started():
                if parsed.path in ("", "/", "/setup"):
                    self._html_response(HTTPStatus.OK, self._setup_page())
                    return
                if is_api:
                    raise BackendError(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "task backend instance is not started; choose a task file and instance on the setup page",
                    )
                if parsed.path.startswith("/workflow/stages/"):
                    job_type = unquote(parsed.path[len("/workflow/stages/"):].strip("/")).strip()
                    if not job_type or "/" in job_type:
                        raise BackendError(HTTPStatus.NOT_FOUND, "workflow stage not found")
                    self._html_response(HTTPStatus.OK, render_workflow_stage_page(self.workflow.workflow_stage(job_type)))
                    return
                raise BackendError(HTTPStatus.NOT_FOUND, "not found")

            if parsed.path == "/setup":
                raise BackendError(
                    HTTPStatus.CONFLICT,
                    "task backend is already started; restart the process to choose another instance",
                )
            elif parsed.path in ("/api/v1/tasks", "/api/v1/collection/tasks"):
                query = parse_qs(parsed.query)
                subject_id = (query.get("subject_id") or [""])[0].strip()
                if not subject_id:
                    raise BackendError(HTTPStatus.BAD_REQUEST, "subject_id is required")
                self._json_response(HTTPStatus.OK, self.backend.get_tasks(subject_id))
            elif parsed.path in ("", "/", "/tasks"):
                self._html_response(HTTPStatus.OK, render_dashboard(self.backend.dashboard_model()))
            elif parsed.path.startswith("/workflow/stages/"):
                job_type = unquote(parsed.path[len("/workflow/stages/"):].strip("/")).strip()
                if not job_type or "/" in job_type:
                    raise BackendError(HTTPStatus.NOT_FOUND, "workflow stage not found")
                self._html_response(HTTPStatus.OK, render_workflow_stage_page(self.workflow.workflow_stage(job_type)))
            elif parsed.path.startswith("/tasks/"):
                task_name = unquote(parsed.path[len("/tasks/"):]).strip()
                if not task_name:
                    raise BackendError(HTTPStatus.NOT_FOUND, "task not found")
                self._html_response(HTTPStatus.OK, render_task_detail(self.backend.task_detail_model(task_name)))
            elif parsed.path.startswith("/episodes/"):
                reservation_id = unquote(parsed.path[len("/episodes/"):]).strip()
                if not reservation_id:
                    raise BackendError(HTTPStatus.NOT_FOUND, "episode not found")
                self._html_response(HTTPStatus.OK, render_episode_detail(self.backend.episode_detail_model(reservation_id)))
            else:
                raise BackendError(HTTPStatus.NOT_FOUND, "not found")
        except BackendError as exc:
            if is_api:
                self._json_response(exc.status, {"error": exc.message})
            else:
                self._html_response(exc.status, render_error_page(exc.status, exc.message))
        except WorkflowError as exc:
            if is_api:
                self._json_response(exc.status, {"error": exc.message})
            else:
                self._html_response(exc.status, render_error_page(exc.status, exc.message))
        except Exception as exc:  # pragma: no cover - defensive server boundary
            if is_api:
                self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            else:
                self._html_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    render_error_page(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc)),
                )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        is_episode_html_post = parsed.path.startswith("/episodes/")
        is_workflow_html_post = not parsed.path.startswith("/api/") and (
            parsed.path.startswith("/workflow/") or parsed.path.startswith("/tasks/")
        )
        try:
            if parsed.path == "/api/v1/auth/register":
                self._json_response(HTTPStatus.OK, self.runtime.accounts.register(self._read_json()))
                return

            if parsed.path == "/api/v1/auth/login":
                self._json_response(HTTPStatus.OK, self.runtime.accounts.login(self._read_json()))
                return

            workflow_stage_api_prefix = "/api/v1/workflow/stages/"
            if parsed.path.startswith(workflow_stage_api_prefix):
                rest = parsed.path[len(workflow_stage_api_prefix):].strip("/")
                parts = rest.split("/")
                if len(parts) != 2 or parts[1] not in {"enable", "disable"}:
                    raise BackendError(HTTPStatus.NOT_FOUND, "workflow stage action not found")
                body = self._read_json()
                enabled = parts[1] == "enable"
                self._json_response(HTTPStatus.OK, self.workflow.set_stage_leasing(unquote(parts[0]), enabled, body))
                return

            if parsed.path == "/api/v1/workflow/episodes/push-auto-label":
                self._json_response(HTTPStatus.OK, self.workflow.push_auto_label(self._read_json()))
                return

            if parsed.path.startswith("/workflow/stages/"):
                rest = parsed.path[len("/workflow/stages/"):].strip("/")
                parts = rest.split("/")
                if len(parts) != 2 or parts[1] not in {"enable", "disable"}:
                    raise BackendError(HTTPStatus.NOT_FOUND, "workflow stage action not found")
                form = self._read_form()
                enabled = parts[1] == "enable"
                job_type = unquote(parts[0])
                self.workflow.set_stage_leasing(job_type, enabled, form)
                self._redirect(f"/workflow/stages/{url_part(job_type)}")
                return

            if parsed.path == "/workflow/episodes/push-auto-label":
                form = self._read_form()
                self.workflow.push_auto_label(form)
                self._redirect("/")
                return

            if parsed.path.startswith("/tasks/") and parsed.path.endswith("/push-auto-label"):
                raw_name = parsed.path[len("/tasks/"):-len("/push-auto-label")]
                task_name = unquote(raw_name.strip("/")).strip()
                if not task_name:
                    raise BackendError(HTTPStatus.NOT_FOUND, "task not found")
                form = self._read_form()
                body = {**form, "task_name": task_name}
                self.workflow.push_auto_label(body)
                self._redirect(f"/tasks/{url_part(task_name)}")
                return

            if parsed.path.startswith("/episodes/") and parsed.path.endswith("/push-auto-label"):
                raw_id = parsed.path[len("/episodes/"):-len("/push-auto-label")]
                episode_id = unquote(raw_id.strip("/")).strip()
                if not episode_id:
                    raise BackendError(HTTPStatus.NOT_FOUND, "episode not found")
                form = self._read_form()
                body = {**form, "episode_id": episode_id}
                self.workflow.push_auto_label(body)
                self._redirect(f"/episodes/{url_part(episode_id)}")
                return

            if parsed.path.startswith("/setup/"):
                if self.runtime.is_started():
                    raise BackendError(
                        HTTPStatus.CONFLICT,
                        "task backend is already started; restart the process to choose another instance",
                    )
                form = self._read_form()
                if parsed.path == "/setup/task-files":
                    task_path = path_from_user(form.get("task_path", ""))
                    label = form.get("task_label", "")
                    entry = self.runtime.registry.add_task_file(task_path, label)
                    message = "Added task file: " + str(entry.get("label", task_path.name))
                    self._html_response(HTTPStatus.OK, self._setup_page(message=message))
                    return
                if parsed.path == "/setup/start":
                    if form.get("mode") == "new_instance":
                        task_file_id = form.get("task_file_id", "")
                        instance = self.runtime.registry.add_instance(task_file_id, form.get("instance_label", ""))
                        instance_id = str(instance.get("id", ""))
                    else:
                        selection = form.get("selection", "")
                        if "::" not in selection:
                            raise BackendError(HTTPStatus.BAD_REQUEST, "select an existing task file instance")
                        task_file_id, instance_id = selection.split("::", 1)
                    self.runtime.start(task_file_id, instance_id)
                    self._redirect("/")
                    return
                raise BackendError(HTTPStatus.NOT_FOUND, "not found")

            if parsed.path.startswith("/episodes/") and parsed.path.endswith("/delete"):
                raw_id = parsed.path[len("/episodes/"):-len("/delete")]
                reservation_id = unquote(raw_id.strip("/")).strip()
                result = self.backend.delete_episode(reservation_id)
                task_name = str(result.get("task_name", ""))
                self._redirect(f"/tasks/{url_part(task_name)}" if task_name else "/")
                return

            body = self._read_json()
            if parsed.path == "/api/v1/dev/label/jobs":
                self._json_response(HTTPStatus.OK, self.workflow.create_manual_label_job(body))
            elif parsed.path == "/api/v1/dev/jobs":
                self._json_response(HTTPStatus.OK, self.workflow.create_dev_job(body))
            elif parsed.path == "/api/v1/jobs/lease":
                self._json_response(HTTPStatus.OK, self.workflow.lease_job(body))
            elif parsed.path == "/api/v1/label/segments/lease":
                self._json_response(HTTPStatus.OK, self.workflow.lease_label_segment(body))
            elif parsed.path == "/api/v1/label/jobs/lease":
                self._json_response(HTTPStatus.OK, self.workflow.lease_label_segment(body))
            else:
                segment_action = self._path_job_action(parsed.path, "/api/v1/label/segments")
                label_action = self._path_job_action(parsed.path, "/api/v1/label/jobs")
                job_action = self._path_job_action(parsed.path, "/api/v1/jobs")
                if segment_action is not None:
                    segment_id, action = segment_action
                    if action == "heartbeat":
                        self._json_response(HTTPStatus.OK, self.workflow.heartbeat_label_segment(segment_id, body))
                    elif action == "complete":
                        self._json_response(HTTPStatus.OK, self.workflow.complete_label_segment(segment_id, body))
                    elif action == "release":
                        self._json_response(HTTPStatus.OK, self.workflow.release_label_segment(segment_id, body))
                    elif action == "fail":
                        self._json_response(HTTPStatus.OK, self.workflow.fail_label_segment(segment_id, body))
                    else:
                        raise BackendError(HTTPStatus.NOT_FOUND, "not found")
                elif label_action is not None:
                    job_id, action = label_action
                    if action == "heartbeat":
                        self._json_response(HTTPStatus.OK, self.workflow.heartbeat_label_segment(job_id, body))
                    elif action == "complete":
                        self._json_response(HTTPStatus.OK, self.workflow.complete_label_segment(job_id, body))
                    elif action == "release":
                        self._json_response(HTTPStatus.OK, self.workflow.release_label_segment(job_id, body))
                    elif action == "fail":
                        self._json_response(HTTPStatus.OK, self.workflow.fail_label_segment(job_id, body))
                    else:
                        raise BackendError(HTTPStatus.NOT_FOUND, "not found")
                elif job_action is not None:
                    job_id, action = job_action
                    if action == "heartbeat":
                        self._json_response(HTTPStatus.OK, self.workflow.heartbeat_job(job_id, body))
                    elif action == "complete":
                        self._json_response(HTTPStatus.OK, self.workflow.complete_job(job_id, body))
                    elif action == "fail":
                        self._json_response(HTTPStatus.OK, self.workflow.fail_job(job_id, body))
                    elif action == "release":
                        self._json_response(HTTPStatus.OK, self.workflow.release_job(job_id, body))
                    else:
                        raise BackendError(HTTPStatus.NOT_FOUND, "not found")
                elif parsed.path in ("/api/v1/episodes/reserve", "/api/v1/collection/episodes/reserve"):
                    self._json_response(HTTPStatus.OK, self.backend.reserve(body))
                elif parsed.path in ("/api/v1/episodes/confirm", "/api/v1/collection/episodes/confirm"):
                    self._json_response(HTTPStatus.OK, self.backend.confirm(body))
                elif parsed.path in ("/api/v1/episodes/release", "/api/v1/collection/episodes/release"):
                    self._json_response(HTTPStatus.OK, self.backend.release(body))
                else:
                    raise BackendError(HTTPStatus.NOT_FOUND, "not found")
        except BackendError as exc:
            if parsed.path.startswith("/setup/"):
                self._html_response(exc.status, self._setup_page(error=exc.message))
            elif is_episode_html_post or is_workflow_html_post:
                self._html_response(exc.status, render_error_page(exc.status, exc.message))
            else:
                self._json_response(exc.status, {"error": exc.message})
        except WorkflowError as exc:
            if is_episode_html_post or is_workflow_html_post:
                self._html_response(exc.status, render_error_page(exc.status, exc.message))
            else:
                self._json_response(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive server boundary
            if parsed.path.startswith("/setup/"):
                self._html_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    self._setup_page(error=str(exc)),
                )
            elif is_episode_html_post or is_workflow_html_post:
                self._html_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    render_error_page(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc)),
                )
            else:
                self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


class TaskHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int], handler_cls: Any, runtime: BackendRuntime):
        self.runtime = runtime
        super().__init__(server_address, handler_cls)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    env_args, _ = env_parser.parse_known_args(argv)
    env_file = env_args.env_file.expanduser()
    if not env_file.is_absolute():
        env_file = Path.cwd() / env_file
    try:
        env_values = load_env_file(env_file.resolve())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    parser = argparse.ArgumentParser(description="Run the Orbbec collection task backend")
    parser.add_argument("--env-file", type=Path, default=env_file, help="path to backend .env config, default: ./.env")
    parser.add_argument("--host", help="bind host; overrides ORBBEC_TASK_BACKEND_HOST")
    parser.add_argument("--port", type=int, help="bind port; overrides ORBBEC_TASK_BACKEND_PORT")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="backend-owned setup and instance state directory; overrides ORBBEC_TASK_BACKEND_DATA_ROOT",
    )
    parser.add_argument("--save-root", type=Path, help="deprecated alias for --data-root")
    parser.add_argument("--task-file", type=Path, help="seed task.json/tasks.json file; overrides ORBBEC_TASK_BACKEND_TASK_FILE")
    parser.add_argument("--state-file", type=Path, help="legacy state file; overrides ORBBEC_TASK_BACKEND_STATE_FILE")
    args = parser.parse_args(argv)
    args.env_file = env_file.resolve()
    args.env_values = env_values
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    env = args.env_values
    host = args.host or env_get(env, "ORBBEC_TASK_BACKEND_HOST", "TASK_BACKEND_HOST") or "127.0.0.1"
    port = args.port if args.port is not None else env_int(env, 8765, "ORBBEC_TASK_BACKEND_PORT", "TASK_BACKEND_PORT")
    data_root_env = env_path(env, "ORBBEC_TASK_BACKEND_DATA_ROOT", "TASK_BACKEND_DATA_ROOT")
    save_root_env = env_path(env, "ORBBEC_TASK_BACKEND_SAVE_ROOT", "TASK_BACKEND_SAVE_ROOT")
    data_root_arg = args.data_root if args.data_root is not None else (data_root_env or args.save_root or save_root_env or Path("./task_backend_state"))
    if args.data_root is None and (args.save_root is not None or (data_root_env is None and save_root_env is not None)):
        print("[task-backend] warning: --save-root is deprecated; use --data-root", file=sys.stderr)
    data_root = data_root_arg.expanduser().resolve()
    seed_task_files: List[Tuple[Path, Optional[Path]]] = []
    state_file_arg = args.state_file or env_path(env, "ORBBEC_TASK_BACKEND_STATE_FILE", "TASK_BACKEND_STATE_FILE")
    task_file_arg = args.task_file or env_path(env, "ORBBEC_TASK_BACKEND_TASK_FILE", "TASK_BACKEND_TASK_FILE")
    seed_state_file = state_file_arg.expanduser().resolve() if state_file_arg else None
    if task_file_arg:
        task_file = task_file_arg.expanduser().resolve()
        if task_file.exists():
            seed_task_files.append((task_file, seed_state_file))
        else:
            print(f"[task-backend] warning: seed task file not found: {task_file}", file=sys.stderr)
    else:
        task_file = default_task_file(data_root).resolve()
        if task_file.exists():
            seed_task_files.append((task_file, seed_state_file))

    workflow_db_env = env_path(env, "ORBBEC_WORKFLOW_DB", "TASK_BACKEND_WORKFLOW_DB")
    workflow_db = (workflow_db_env or (data_root / "workflow.sqlite3")).expanduser().resolve()
    nas_enabled = env_bool(
        env,
        True,
        "ORBBEC_NAS_ENABLED",
        "TASK_BACKEND_NAS_ENABLED",
    )
    nas_root_env = env_path(
        env,
        "ORBBEC_NAS_ROOT",
        "TASK_BACKEND_NAS_ROOT",
    )
    nas_root = (nas_root_env or Path("/mnt/nas")).expanduser().resolve()
    nas_uri_prefix = (
        env_get(
            env,
            "ORBBEC_NAS_URI_PREFIX",
            "TASK_BACKEND_NAS_URI_PREFIX",
        )
        or "nas://ego"
    )
    nas_mounts = env_json_object(env, "ORBBEC_NAS_MOUNTS_JSON")
    nas_mounts.setdefault(nas_uri_prefix.rstrip("/"), str(nas_root))
    auto_label_after_upload = env_bool(
        env,
        False,
        "ORBBEC_AUTO_LABEL_AFTER_UPLOAD",
        "TASK_BACKEND_AUTO_LABEL_AFTER_UPLOAD",
    )
    workflow_store = WorkflowStore(workflow_db)
    workflow_service = JobService(
        workflow_store,
        auto_label_after_upload=auto_label_after_upload,
        nas_mounts=nas_mounts,
    )
    nas_uploader = NasUploader(
        workflow_service,
        NasUploadConfig(
            enabled=nas_enabled,
            root=nas_root,
            uri_prefix=nas_uri_prefix,
        ),
    )
    registry = TaskInstanceRegistry(data_root=data_root, seed_task_files=seed_task_files)
    runtime = BackendRuntime(registry, workflow_service)
    host_info = socket.getfqdn(host) if host not in ("", "0.0.0.0", "::") else host
    if args.env_file.exists():
        print(f"[task-backend] env_file={args.env_file}", file=sys.stderr)
    print(f"[task-backend] setup_registry={registry.registry_file}", file=sys.stderr)
    print(f"[task-backend] workflow_db={workflow_db}", file=sys.stderr)
    print(f"[task-backend] auto_label_after_upload={'enabled' if auto_label_after_upload else 'disabled'} auto_label_scope=episode", file=sys.stderr)
    print(f"[task-backend] nas={'enabled' if nas_enabled else 'disabled'} root={nas_root} uri={nas_uri_prefix}", file=sys.stderr)
    print(f"[task-backend] nas_mounts={nas_mounts}", file=sys.stderr)
    print(f"[task-backend] data_root={data_root}", file=sys.stderr)
    print(f"[task-backend] listening http://{host}:{port} ({host_info})", file=sys.stderr)
    print("[task-backend] open the web setup page and start one task-file instance", file=sys.stderr)

    server = TaskHTTPServer((host, port), RequestHandler, runtime)
    try:
        nas_uploader.start()
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[task-backend] stopping", file=sys.stderr)
    finally:
        nas_uploader.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
