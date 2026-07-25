from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from .workflow_models import (
        TERMINAL_JOB_STATUSES,
        WorkflowError,
        require_episode_status,
        require_job_status,
        require_job_type,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from workflow_models import (  # type: ignore
        TERMINAL_JOB_STATUSES,
        WorkflowError,
        require_episode_status,
        require_job_status,
        require_job_type,
    )


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_dumps(value: Any) -> str:
    if value is None:
        value = {}
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any, fallback: Any) -> Any:
    if value is None or value == "":
        return fallback
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return fallback


def _future_iso(seconds: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + max(1, int(seconds))))


def _new_id(prefix: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in prefix.strip().lower())
    clean = clean.strip("_") or "item"
    return f"{clean}_{uuid.uuid4().hex[:12]}"


class WorkflowStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path.expanduser().resolve()
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    episode_index INTEGER,
                    status TEXT NOT NULL,
                    data_uri TEXT,
                    local_capture_path TEXT,
                    frame_count INTEGER,
                    cameras_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    episode_id TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    lease_owner TEXT,
                    lease_until TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_subject_task
                    ON episodes(subject_id, task_name);
                CREATE INDEX IF NOT EXISTS idx_artifacts_episode_kind
                    ON artifacts(episode_id, kind);
                CREATE INDEX IF NOT EXISTS idx_jobs_type_status_created
                    ON jobs(type, status, created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_episode
                    ON jobs(episode_id);
                PRAGMA user_version = 1;
                """
            )

    def create_or_update_episode(
        self,
        *,
        episode_id: Optional[str] = None,
        subject_id: str,
        task_name: str,
        episode_index: Optional[int] = None,
        status: str = "planned",
        data_uri: str = "",
        local_capture_path: str = "",
        frame_count: Optional[int] = None,
        cameras: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        status = require_episode_status(status)
        episode_id = str(episode_id or "").strip() or _new_id("episode")
        subject_id = str(subject_id or "").strip()
        task_name = str(task_name or "").strip()
        if not subject_id or not task_name:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "subject_id and task_name are required")

        with self.connect() as conn:
            existing = self._get_episode_unlocked(conn, episode_id)
            now = now_iso()
            values = {
                "episode_id": episode_id,
                "subject_id": subject_id,
                "task_name": task_name,
                "episode_index": episode_index,
                "status": status,
                "data_uri": data_uri,
                "local_capture_path": local_capture_path,
                "frame_count": frame_count,
                "cameras": cameras or [],
                "metadata": metadata or {},
                "created_at": now,
                "updated_at": now,
            }
            if existing is not None:
                values.update(
                    {
                        "subject_id": subject_id or existing["subject_id"],
                        "task_name": task_name or existing["task_name"],
                        "episode_index": episode_index if episode_index is not None else existing.get("episode_index"),
                        "status": status or existing["status"],
                        "data_uri": data_uri or existing.get("data_uri", ""),
                        "local_capture_path": local_capture_path or existing.get("local_capture_path", ""),
                        "frame_count": frame_count if frame_count is not None else existing.get("frame_count"),
                        "cameras": cameras if cameras is not None else existing.get("cameras", []),
                        "metadata": {**existing.get("metadata", {}), **(metadata or {})},
                        "created_at": existing["created_at"],
                        "updated_at": now,
                    }
                )
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, subject_id, task_name, episode_index, status,
                    data_uri, local_capture_path, frame_count, cameras_json,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["episode_id"],
                    values["subject_id"],
                    values["task_name"],
                    values["episode_index"],
                    values["status"],
                    values["data_uri"],
                    values["local_capture_path"],
                    values["frame_count"],
                    _json_dumps(values["cameras"]),
                    _json_dumps(values["metadata"]),
                    values["created_at"],
                    values["updated_at"],
                ),
            )
            return self._get_episode_unlocked(conn, episode_id) or values

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            return self._get_episode_unlocked(conn, episode_id)

    def update_episode_status(self, episode_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        status = require_episode_status(status)
        with self.connect() as conn:
            episode = self._get_episode_unlocked(conn, episode_id)
            if episode is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
            merged_metadata = dict(episode.get("metadata", {}))
            if metadata:
                merged_metadata.update(metadata)
            conn.execute(
                "UPDATE episodes SET status = ?, metadata_json = ?, updated_at = ? WHERE episode_id = ?",
                (status, _json_dumps(merged_metadata), now_iso(), episode_id),
            )
            return self._get_episode_unlocked(conn, episode_id) or episode

    def update_episode_storage(
        self,
        episode_id: str,
        *,
        data_uri: str = "",
        local_capture_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        episode_id = str(episode_id or "").strip()
        if not episode_id:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "episode_id is required")
        with self.connect() as conn:
            episode = self._get_episode_unlocked(conn, episode_id)
            if episode is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
            merged_metadata = dict(episode.get("metadata", {}))
            if metadata:
                merged_metadata.update(metadata)
            conn.execute(
                """
                UPDATE episodes
                SET data_uri = ?, local_capture_path = ?, metadata_json = ?, updated_at = ?
                WHERE episode_id = ?
                """,
                (
                    str(data_uri or episode.get("data_uri") or ""),
                    str(local_capture_path or episode.get("local_capture_path") or ""),
                    _json_dumps(merged_metadata),
                    now_iso(),
                    episode_id,
                ),
            )
            return self._get_episode_unlocked(conn, episode_id) or episode

    def register_artifact(
        self,
        *,
        episode_id: str,
        kind: str,
        uri: str,
        metadata: Optional[Dict[str, Any]] = None,
        artifact_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        episode_id = str(episode_id or "").strip()
        kind = str(kind or "").strip()
        uri = str(uri or "").strip()
        if not episode_id or not kind or not uri:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "episode_id, kind, and uri are required")
        with self.connect() as conn:
            if self._get_episode_unlocked(conn, episode_id) is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
            artifact_id = str(artifact_id or "").strip() or _new_id(kind)
            created_at = now_iso()
            conn.execute(
                """
                INSERT INTO artifacts (artifact_id, episode_id, kind, uri, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (artifact_id, episode_id, kind, uri, _json_dumps(metadata or {}), created_at),
            )
            return {
                "artifact_id": artifact_id,
                "episode_id": episode_id,
                "kind": kind,
                "uri": uri,
                "metadata": metadata or {},
                "created_at": created_at,
            }

    def artifacts_for_episode(self, episode_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE episode_id = ? ORDER BY created_at, artifact_id",
                (episode_id,),
            ).fetchall()
            return [self._row_to_artifact(row) for row in rows]

    def create_job(
        self,
        *,
        job_type: str,
        episode_id: Optional[str],
        payload: Dict[str, Any],
        status: str = "queued",
        result: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        job_type = require_job_type(job_type)
        status = require_job_status(status)
        job_id = str(job_id or "").strip() or _new_id(job_type)
        episode_id = str(episode_id or "").strip() or None
        with self.connect() as conn:
            if episode_id and self._get_episode_unlocked(conn, episode_id) is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
            now = now_iso()
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, type, status, episode_id, payload_json, result_json,
                    lease_owner, lease_until, attempt, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, 0, ?, ?)
                """,
                (job_id, job_type, status, episode_id, _json_dumps(payload), _json_dumps(result or {}), now, now),
            )
            job = self._get_job_unlocked(conn, job_id)
            if job is None:
                raise WorkflowError(HTTPStatus.INTERNAL_SERVER_ERROR, f"created job not found: {job_id}")
            return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            return self._get_job_unlocked(conn, job_id)

    def jobs_for_episode(self, episode_id: str, job_type: str = "") -> List[Dict[str, Any]]:
        episode_id = str(episode_id or "").strip()
        if not episode_id:
            return []
        params: Tuple[Any, ...]
        query = "SELECT * FROM jobs WHERE episode_id = ?"
        params = (episode_id,)
        if job_type:
            job_type = require_job_type(job_type)
            query += " AND type = ?"
            params = (episode_id, job_type)
        query += " ORDER BY created_at, job_id"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_job(row) for row in rows]

    def lease_job(self, *, job_type: str, lease_owner: str, lease_seconds: int = 300) -> Optional[Dict[str, Any]]:
        job_type = require_job_type(job_type)
        lease_owner = str(lease_owner or "").strip()
        if not lease_owner:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "lease_owner is required")
        lease_seconds = max(1, int(lease_seconds or 300))
        now = now_iso()
        lease_until = _future_iso(lease_seconds)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE type = ? AND status NOT IN ('succeeded', 'failed', 'canceled')
                ORDER BY created_at, job_id
                """,
                (job_type,),
            ).fetchall()
            selected: Optional[sqlite3.Row] = None
            for row in rows:
                status = str(row["status"])
                row_lease_until = str(row["lease_until"] or "")
                if status == "queued" or not row_lease_until or row_lease_until <= now:
                    selected = row
                    break
            if selected is None:
                return None
            conn.execute(
                """
                UPDATE jobs
                SET status = 'leased', lease_owner = ?, lease_until = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (lease_owner, lease_until, now, selected["job_id"]),
            )
            return self._get_job_unlocked(conn, str(selected["job_id"]))

    def heartbeat_job(
        self,
        *,
        job_id: str,
        lease_owner: str = "",
        lease_seconds: int = 300,
        status: str = "",
    ) -> Dict[str, Any]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = self._get_job_unlocked(conn, job_id)
            if job is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"job not found: {job_id}")
            if str(job["status"]) in TERMINAL_JOB_STATUSES:
                return job
            if lease_owner and job.get("lease_owner") and job.get("lease_owner") != lease_owner:
                raise WorkflowError(HTTPStatus.CONFLICT, "job is leased by another owner")
            next_status = str(job["status"])
            if status:
                status = require_job_status(status)
                if status not in {"leased", "running"}:
                    raise WorkflowError(HTTPStatus.BAD_REQUEST, "heartbeat status must be leased or running")
                next_status = status
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, lease_until = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (next_status, _future_iso(max(1, int(lease_seconds or 300))), now_iso(), job_id),
            )
            return self._get_job_unlocked(conn, job_id) or job

    def update_job_progress(
        self,
        *,
        job_id: str,
        progress: Dict[str, Any],
        status: str = "running",
    ) -> Dict[str, Any]:
        status = require_job_status(status)
        if status in TERMINAL_JOB_STATUSES or status not in {"queued", "leased", "running"}:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "progress status must be queued, leased, or running")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = self._get_job_unlocked(conn, job_id)
            if job is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"job not found: {job_id}")
            if str(job["status"]) in TERMINAL_JOB_STATUSES:
                return job
            merged = dict(job.get("result") or {})
            merged.update(progress or {})
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, result_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, _json_dumps(merged), now_iso(), job_id),
            )
            return self._get_job_unlocked(conn, job_id) or job

    def complete_job(self, *, job_id: str, result: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = self._get_job_unlocked(conn, job_id)
            if job is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"job not found: {job_id}")
            if job["status"] == "succeeded":
                return job, False
            if job["status"] in {"failed", "canceled"}:
                raise WorkflowError(HTTPStatus.CONFLICT, f"job is already {job['status']}")
            conn.execute(
                """
                UPDATE jobs
                SET status = 'succeeded', result_json = ?, lease_until = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (_json_dumps(result), now_iso(), job_id),
            )
            updated = self._get_job_unlocked(conn, job_id)
            if updated is None:
                raise WorkflowError(HTTPStatus.INTERNAL_SERVER_ERROR, f"completed job not found: {job_id}")
            return updated, True

    def fail_job(self, *, job_id: str, error: str, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = self._get_job_unlocked(conn, job_id)
            if job is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"job not found: {job_id}")
            if job["status"] == "succeeded":
                raise WorkflowError(HTTPStatus.CONFLICT, "succeeded job cannot be failed")
            if job["status"] == "canceled":
                raise WorkflowError(HTTPStatus.CONFLICT, "canceled job cannot be failed")
            payload = dict(result or {})
            payload["error"] = str(error or "job failed")
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', result_json = ?, lease_until = NULL,
                    attempt = attempt + 1, updated_at = ?
                WHERE job_id = ?
                """,
                (_json_dumps(payload), now_iso(), job_id),
            )
            updated = self._get_job_unlocked(conn, job_id)
            if updated is None:
                raise WorkflowError(HTTPStatus.INTERNAL_SERVER_ERROR, f"failed job not found: {job_id}")
            return updated

    def release_job(self, *, job_id: str, reason: str = "") -> Tuple[Dict[str, Any], bool]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = self._get_job_unlocked(conn, job_id)
            if job is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"job not found: {job_id}")
            if job["status"] in TERMINAL_JOB_STATUSES:
                return job, False
            if job["status"] == "queued" and not job.get("lease_owner"):
                return job, False
            result = dict(job.get("result") or {})
            if reason:
                result["last_release_reason"] = reason
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued', lease_owner = NULL, lease_until = NULL,
                    result_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (_json_dumps(result), now_iso(), job_id),
            )
            updated = self._get_job_unlocked(conn, job_id)
            if updated is None:
                raise WorkflowError(HTTPStatus.INTERNAL_SERVER_ERROR, f"released job not found: {job_id}")
            return updated, True

    def _get_episode_unlocked(self, conn: sqlite3.Connection, episode_id: str) -> Optional[Dict[str, Any]]:
        row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
        return self._row_to_episode(row) if row is not None else None

    def _get_job_unlocked(self, conn: sqlite3.Connection, job_id: str) -> Optional[Dict[str, Any]]:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row is not None else None

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "episode_id": row["episode_id"],
            "subject_id": row["subject_id"],
            "task_name": row["task_name"],
            "episode_index": row["episode_index"],
            "status": row["status"],
            "data_uri": row["data_uri"] or "",
            "local_capture_path": row["local_capture_path"] or "",
            "frame_count": row["frame_count"],
            "cameras": _json_loads(row["cameras_json"], []),
            "metadata": _json_loads(row["metadata_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _row_to_artifact(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "artifact_id": row["artifact_id"],
            "episode_id": row["episode_id"],
            "kind": row["kind"],
            "uri": row["uri"],
            "metadata": _json_loads(row["metadata_json"], {}),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "type": row["type"],
            "status": row["status"],
            "episode_id": row["episode_id"] or "",
            "payload": _json_loads(row["payload_json"], {}),
            "result": _json_loads(row["result_json"], {}),
            "lease_owner": row["lease_owner"] or "",
            "lease_until": row["lease_until"] or "",
            "attempt": int(row["attempt"] or 0),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
