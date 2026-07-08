#!/usr/bin/env python3
"""Small HTTP task backend for Orbbec collection.

The service owns task definitions, episode reservations, and confirmed
progress.  It intentionally uses only the Python standard library so it can run
on a capture host without extra packages.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - non-Linux fallback
    fcntl = None


Task = Dict[str, Any]
State = Dict[str, Any]


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


def default_task_file(save_root: Path) -> Path:
    candidates = [
        save_root / "task.json",
        save_root / "tasks.json",
        Path.cwd() / "task.json",
        Path.cwd() / "tasks.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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


class BackendError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class TaskBackend:
    def __init__(self, save_root: Path, task_file: Path, state_file: Optional[Path] = None):
        self.save_root = save_root
        self.task_file = task_file
        self.state_file = state_file or (save_root / "progress_state.json")
        self.lock_file = self.state_file.with_suffix(self.state_file.suffix + ".lock")
        self.tasks = load_task_file(task_file)
        self.tasks_by_name = {task["task_name"]: task for task in self.tasks}
        self.save_root.mkdir(parents=True, exist_ok=True)

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


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "OrbbecTaskBackend/1.0"

    @property
    def backend(self) -> TaskBackend:
        return self.server.backend  # type: ignore[attr-defined]

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

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/v1/tasks":
                query = parse_qs(parsed.query)
                subject_id = (query.get("subject_id") or [""])[0].strip()
                if not subject_id:
                    raise BackendError(HTTPStatus.BAD_REQUEST, "subject_id is required")
                self._json_response(HTTPStatus.OK, self.backend.get_tasks(subject_id))
            else:
                raise BackendError(HTTPStatus.NOT_FOUND, "not found")
        except BackendError as exc:
            self._json_response(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
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
            self._json_response(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


class TaskHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int], handler_cls: Any, backend: TaskBackend):
        super().__init__(server_address, handler_cls)
        self.backend = backend


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Orbbec collection task backend")
    parser.add_argument("--host", default="127.0.0.1", help="bind host, default: 127.0.0.1")
    parser.add_argument("--port", default=8765, type=int, help="bind port, default: 8765")
    parser.add_argument("--save-root", required=True, type=Path, help="collection save root; progress_state.json is stored here")
    parser.add_argument("--task-file", type=Path, help="task.json/tasks.json file; defaults to save-root/task.json, save-root/tasks.json, then cwd/tasks.json")
    parser.add_argument("--state-file", type=Path, help="override progress_state.json path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    save_root = args.save_root.expanduser().resolve()
    task_file = (args.task_file.expanduser().resolve() if args.task_file else default_task_file(save_root).resolve())
    if not task_file.exists():
        print(f"task file not found: {task_file}", file=sys.stderr)
        return 2

    state_file = args.state_file.expanduser().resolve() if args.state_file else None
    backend = TaskBackend(save_root=save_root, task_file=task_file, state_file=state_file)
    host_info = socket.getfqdn(args.host) if args.host not in ("", "0.0.0.0", "::") else args.host
    print(f"[task-backend] tasks={len(backend.tasks)} task_file={task_file}", file=sys.stderr)
    print(f"[task-backend] state_file={backend.state_file}", file=sys.stderr)
    print(f"[task-backend] listening http://{args.host}:{args.port} ({host_info})", file=sys.stderr)

    server = TaskHTTPServer((args.host, args.port), RequestHandler, backend)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[task-backend] stopping", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
