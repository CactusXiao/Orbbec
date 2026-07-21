#!/usr/bin/env python3
"""Small HTTP task backend for Orbbec collection.

The service owns task definitions, episode reservations, and confirmed
progress.  It intentionally uses only the Python standard library so it can run
on a capture host without extra packages.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
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
    def __init__(self, registry: TaskInstanceRegistry):
        self.registry = registry
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


def task_progress(subject: Dict[str, Any], task: Task) -> Dict[str, Any]:
    completed = len(confirmed_reservations(subject, task["task_name"]))
    total = int(task["total"])
    return {
        "task_name": task["task_name"],
        "description_cn": task["description_cn"],
        "description_en": task["description_en"],
        "completed": min(completed, total),
        "total": total,
    }


def progress_payload(subject: Dict[str, Any], tasks: Iterable[Task]) -> Dict[str, Any]:
    return {"tasks": [task_progress(subject, task) for task in tasks]}


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


def status_class(status: str) -> str:
    if status == "confirmed":
        return "ok"
    if status == "reserved":
        return "warn"
    if status == "released":
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


class TaskBackend:
    def __init__(self,
                 data_root: Path,
                 task_file: Path,
                 state_file: Optional[Path] = None,
                 runtime_info: Optional[Dict[str, Any]] = None):
        self.data_root = data_root
        self.task_file = task_file
        self.state_file = state_file or (data_root / "progress_state.json")
        self.lock_file = self.state_file.with_suffix(self.state_file.suffix + ".lock")
        self.tasks = load_task_file(task_file)
        self.tasks_by_name = {task["task_name"]: task for task in self.tasks}
        self.runtime_info = runtime_info or {}
        self.data_root.mkdir(parents=True, exist_ok=True)

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
        reservations = self.reservation_list(state)
        subjects_obj = state.get("subjects", {})
        subjects = sorted(str(key) for key in subjects_obj.keys()) if isinstance(subjects_obj, dict) else []
        summaries = []
        for task in self.tasks:
            task_name = str(task["task_name"])
            related = [item for item in reservations if item.get("task_name") == task_name]
            counts = count_statuses(related)
            task_subjects = sorted({str(item.get("subject_id", "")) for item in related if item.get("subject_id")})
            summaries.append(
                {
                    "task": task,
                    "counts": counts,
                    "subjects": task_subjects,
                    "latest_at": latest_timestamp(related),
                    "reservation_count": len(related),
                }
            )
        return {
            "tasks": summaries,
            "subjects": subjects,
            "reservation_count": len(reservations),
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
            item
            for item in self.reservation_list(state)
            if item.get("task_name") == task_name
        ]
        subjects = sorted({str(item.get("subject_id", "")) for item in reservations if item.get("subject_id")})
        return {
            "task": task,
            "reservations": reservations,
            "subjects": subjects,
            "counts": count_statuses(reservations),
            "latest_at": latest_timestamp(reservations),
        }

    def episode_detail_model(self, reservation_id: str) -> Dict[str, Any]:
        state = self.state_snapshot()
        for item in self.reservation_list(state):
            if item.get("reservation_id") == reservation_id:
                task = self.tasks_by_name.get(str(item.get("task_name", "")))
                return {
                    "reservation": item,
                    "task": task,
                    "metadata_pairs": metadata_pairs(task) if task is not None else [],
                }
        raise BackendError(HTTPStatus.NOT_FOUND, f"episode not found: {reservation_id}")

    def get_tasks(self, subject_id: str) -> Dict[str, Any]:
        with self.locked_state() as state:
            subject = ensure_subject(state, subject_id)
            return progress_payload(subject, self.tasks)

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
            completed = len(confirmed_reservations(subject, task_name))
            if completed >= int(task["total"]):
                raise BackendError(HTTPStatus.CONFLICT, f"task already complete: {task_name}")

            reservation_id = str(uuid.uuid4())
            episode_number = next_episode_number(subject, task_name)
            subject["reservations"][reservation_id] = {
                "reservation_id": reservation_id,
                "client_id": client_id,
                "subject_id": subject_id,
                "task_name": task_name,
                "episode_number": episode_number,
                "status": "reserved",
                "created_at": now_iso(),
                "updated_at": now_iso(),
            }
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
        local_path = str(payload.get("local_path", "")).strip()
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
                return progress_payload(subject, self.tasks)

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
                return progress_payload(subject, self.tasks)

            reservation["status"] = "confirmed"
            reservation["confirmed_at"] = now_iso()
            reservation["updated_at"] = now_iso()
            reservation["idempotency_key"] = idempotency_key
            reservation["local_path"] = local_path
            subject["idempotency"][idempotency_key] = reservation_id
            return progress_payload(subject, self.tasks)

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
                return {"released": False, "reason": "not_found", **progress_payload(subject, self.tasks)}
            if task_name and reservation.get("task_name") != task_name:
                raise BackendError(HTTPStatus.CONFLICT, "reservation does not match task")
            if reservation.get("status") == "reserved":
                reservation["status"] = "released"
                reservation["released_at"] = now_iso()
                reservation["updated_at"] = now_iso()
                released = True
            else:
                released = False
            return {"released": released, **progress_payload(subject, self.tasks)}


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
    .actions {{ padding: 0 16px 16px; display: flex; gap: 10px; flex-wrap: wrap; }}
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
            f"<td>{html_escape(', '.join(subjects) if subjects else '-')}</td>"
            f"<td class=\"muted mono\">{html_escape(summary.get('latest_at') or '-')}</td>"
            "</tr>"
        )
    table_body = "\n".join(task_rows) if task_rows else "<tr><td colspan=\"7\" class=\"empty\">No tasks loaded.</td></tr>"
    subject_note = ", ".join(model["subjects"]) if model["subjects"] else "No subjects yet"
    runtime_info = model.get("runtime_info") or {}
    instance_label = runtime_info.get("instance_label") or "-"
    task_file_label = runtime_info.get("task_file_label") or "-"
    body = (
        f"<div class=\"crumbs\">Task backend / Overview / {html_escape(instance_label)}</div>"
        "<div class=\"notice warn\">This backend process is locked to the selected task file and instance. "
        "Restart the process to choose another instance.</div>"
        "<div class=\"summary\">"
        + render_metric("Tasks", len(model["tasks"]), "from task file")
        + render_metric("Subjects", len(model["subjects"]), subject_note)
        + render_metric("Episodes", model["reservation_count"], "all reservations")
        + render_metric("Confirmed", totals["confirmed"], "completed episodes")
        + "</div>"
        "<section><h2>Task Summary</h2><div class=\"wide\"><table>"
        "<thead><tr>"
        "<th>Task</th><th class=\"num\">Required / Subject</th><th class=\"num\">Confirmed</th>"
        "<th class=\"num\">Reserved</th><th class=\"num\">Released</th><th>Subjects</th><th>Latest Update</th>"
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
        rows.append(
            "<tr>"
            f"<td><a class=\"mono\" href=\"/episodes/{url_part(reservation_id)}\">{html_escape(reservation_id[:8])}</a></td>"
            f"<td>{html_escape(item.get('subject_id', ''))}</td>"
            f"<td class=\"num\">{html_escape(item.get('episode_number', ''))}</td>"
            f"<td>{render_status_badge(status)}</td>"
            f"<td class=\"mono\">{html_escape(item.get('client_id', '-'))}</td>"
            f"<td class=\"muted mono\">{html_escape(item.get('updated_at') or item.get('created_at') or '-')}</td>"
            f"<td class=\"mono\">{html_escape(item.get('local_path') or '-')}</td>"
            "</tr>"
        )
    episode_rows = "\n".join(rows) if rows else "<tr><td colspan=\"7\" class=\"empty\">No episodes reserved yet.</td></tr>"
    meta_rows = []
    for key, value in metadata_pairs(task):
        meta_rows.append(f"<div>{html_escape(key)}</div><div>{html_escape(value)}</div>")
    if not meta_rows:
        meta_rows.append("<div>Extra metadata</div><div class=\"muted\">No extra task metadata.</div>")
    description = task.get("description_cn") or task.get("description_en") or ""
    body = (
        f"<div class=\"crumbs\"><a href=\"/\">Task backend</a> / {html_escape(task_name)}</div>"
        "<div class=\"summary\">"
        + render_metric("Required / Subject", task.get("total", ""), "configured episodes")
        + render_metric("Confirmed", counts.get("confirmed", 0))
        + render_metric("Reserved", counts.get("reserved", 0))
        + render_metric("Released", counts.get("released", 0))
        + "</div>"
        "<section><h2>Task Description</h2>"
        f"<div class=\"empty desc\">{html_escape(description or 'No description.')}</div></section>"
        "<section><h2>Task Metadata</h2><div class=\"kv\">"
        + "".join(meta_rows)
        + f"<div>Subjects</div><div>{html_escape(', '.join(model['subjects']) if model['subjects'] else '-')}</div>"
        + f"<div>Latest update</div><div class=\"mono\">{html_escape(model.get('latest_at') or '-')}</div>"
        + "</div></section>"
        "<section><h2>Episodes</h2><div class=\"wide\"><table>"
        "<thead><tr><th>Episode</th><th>Subject</th><th class=\"num\">No.</th><th>Status</th>"
        "<th>Client</th><th>Updated</th><th>Local Path</th></tr></thead><tbody>"
        + episode_rows
        + "</tbody></table></div></section>"
    )
    return render_layout(task_name, body)


def render_episode_detail(model: Dict[str, Any]) -> str:
    item = model["reservation"]
    task = model["task"]
    task_name = str(item.get("task_name", ""))
    reservation_id = str(item.get("reservation_id", ""))
    status = str(item.get("status", "unknown"))
    fields = [
        ("Reservation ID", reservation_id),
        ("Task", task_name),
        ("Subject", item.get("subject_id", "")),
        ("Episode number", item.get("episode_number", "")),
        ("Status", status),
        ("Client ID", item.get("client_id", "")),
        ("Local capture path", item.get("local_path") or "-"),
        ("Idempotency key", item.get("idempotency_key") or "-"),
        ("Created at", item.get("created_at") or "-"),
        ("Updated at", item.get("updated_at") or "-"),
        ("Confirmed at", item.get("confirmed_at") or "-"),
        ("Released at", item.get("released_at") or "-"),
    ]
    field_html = "".join(
        f"<div>{html_escape(label)}</div><div class=\"mono\">{html_escape(value)}</div>"
        for label, value in fields
    )
    meta_html = "".join(
        f"<div>{html_escape(key)}</div><div>{html_escape(value)}</div>"
        for key, value in model["metadata_pairs"]
    )
    if not meta_html:
        meta_html = "<div>Task metadata</div><div class=\"muted\">No extra task metadata.</div>"
    storage_note = (
        "Capture data is still stored on the capture host. NAS-backed file indexing "
        "and quality results can be attached here later."
    )
    body = (
        f"<div class=\"crumbs\"><a href=\"/\">Task backend</a> / "
        f"<a href=\"/tasks/{url_part(task_name)}\">{html_escape(task_name)}</a> / "
        f"{html_escape(reservation_id[:8])}</div>"
        "<div class=\"summary\">"
        + render_metric("Status", status)
        + render_metric("Subject", item.get("subject_id", ""))
        + render_metric("Episode", item.get("episode_number", ""))
        + render_metric("QA", "Pending", "quality module not connected")
        + "</div>"
        "<section><h2>Episode Details</h2><div class=\"kv\">"
        + field_html
        + "</div></section>"
        "<section><h2>Storage And Quality</h2><div class=\"kv\">"
        f"<div>Storage status</div><div>{html_escape('Local path recorded' if item.get('local_path') else 'Waiting for confirm')}</div>"
        f"<div>Backend file access</div><div class=\"muted\">{html_escape(storage_note)}</div>"
        f"<div>Quality status</div><div class=\"muted\">Not integrated yet.</div>"
        + "</div></section>"
        "<section><h2>Task Metadata Snapshot</h2><div class=\"kv\">"
        + meta_html
        + "</div></section>"
        "<section><h2>Raw Reservation JSON</h2>"
        f"<div class=\"empty\"><pre class=\"mono\">{html_escape(json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True))}</pre></div>"
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

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        is_api = parsed.path.startswith("/api/")
        try:
            if not self.runtime.is_started():
                if parsed.path in ("", "/", "/setup"):
                    self._html_response(HTTPStatus.OK, self._setup_page())
                    return
                if is_api:
                    raise BackendError(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "task backend instance is not started; choose a task file and instance on the setup page",
                    )
                raise BackendError(HTTPStatus.NOT_FOUND, "not found")

            if parsed.path == "/setup":
                raise BackendError(
                    HTTPStatus.CONFLICT,
                    "task backend is already started; restart the process to choose another instance",
                )
            elif parsed.path == "/api/v1/tasks":
                query = parse_qs(parsed.query)
                subject_id = (query.get("subject_id") or [""])[0].strip()
                if not subject_id:
                    raise BackendError(HTTPStatus.BAD_REQUEST, "subject_id is required")
                self._json_response(HTTPStatus.OK, self.backend.get_tasks(subject_id))
            elif parsed.path in ("", "/", "/tasks"):
                self._html_response(HTTPStatus.OK, render_dashboard(self.backend.dashboard_model()))
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
        try:
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

            body = self._read_json()
            if parsed.path == "/api/v1/episodes/reserve":
                self._json_response(HTTPStatus.OK, self.backend.reserve(body))
            elif parsed.path == "/api/v1/episodes/confirm":
                self._json_response(HTTPStatus.OK, self.backend.confirm(body))
            elif parsed.path == "/api/v1/episodes/release":
                self._json_response(HTTPStatus.OK, self.backend.release(body))
            else:
                raise BackendError(HTTPStatus.NOT_FOUND, "not found")
        except BackendError as exc:
            if parsed.path.startswith("/setup/"):
                self._html_response(exc.status, self._setup_page(error=exc.message))
            else:
                self._json_response(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive server boundary
            if parsed.path.startswith("/setup/"):
                self._html_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    self._setup_page(error=str(exc)),
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
    parser = argparse.ArgumentParser(description="Run the Orbbec collection task backend")
    parser.add_argument("--host", default="127.0.0.1", help="bind host, default: 127.0.0.1")
    parser.add_argument("--port", default=8765, type=int, help="bind port, default: 8765")
    parser.add_argument("--data-root", type=Path, help="backend-owned setup and instance state directory")
    parser.add_argument("--save-root", type=Path, help="deprecated alias for --data-root")
    parser.add_argument("--task-file", type=Path, help="seed task.json/tasks.json file for the setup page")
    parser.add_argument("--state-file", type=Path, help="legacy state file used for the seeded default instance")
    args = parser.parse_args(argv)
    if args.data_root is None and args.save_root is None:
        parser.error("--data-root is required (or legacy --save-root)")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    data_root_arg = args.data_root if args.data_root is not None else args.save_root
    if args.data_root is None and args.save_root is not None:
        print("[task-backend] warning: --save-root is deprecated; use --data-root", file=sys.stderr)
    data_root = data_root_arg.expanduser().resolve()
    seed_task_files: List[Tuple[Path, Optional[Path]]] = []
    seed_state_file = args.state_file.expanduser().resolve() if args.state_file else None
    if args.task_file:
        task_file = args.task_file.expanduser().resolve()
        if task_file.exists():
            seed_task_files.append((task_file, seed_state_file))
        else:
            print(f"[task-backend] warning: seed task file not found: {task_file}", file=sys.stderr)
    else:
        task_file = default_task_file(data_root).resolve()
        if task_file.exists():
            seed_task_files.append((task_file, seed_state_file))

    registry = TaskInstanceRegistry(data_root=data_root, seed_task_files=seed_task_files)
    runtime = BackendRuntime(registry)
    host_info = socket.getfqdn(args.host) if args.host not in ("", "0.0.0.0", "::") else args.host
    print(f"[task-backend] setup_registry={registry.registry_file}", file=sys.stderr)
    print(f"[task-backend] data_root={data_root}", file=sys.stderr)
    print(f"[task-backend] listening http://{args.host}:{args.port} ({host_info})", file=sys.stderr)
    print("[task-backend] open the web setup page and start one task-file instance", file=sys.stderr)

    server = TaskHTTPServer((args.host, args.port), RequestHandler, runtime)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[task-backend] stopping", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
