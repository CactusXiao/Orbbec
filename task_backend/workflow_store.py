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
        CONTROLLED_STAGE_JOB_TYPES,
        TERMINAL_JOB_STATUSES,
        WorkflowError,
        require_episode_status,
        require_job_status,
        require_segment_status,
        require_stage_job_type,
        require_job_type,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from workflow_models import (  # type: ignore
        CONTROLLED_STAGE_JOB_TYPES,
        TERMINAL_JOB_STATUSES,
        WorkflowError,
        require_episode_status,
        require_job_status,
        require_segment_status,
        require_stage_job_type,
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


DEFAULT_OPEN_STAGE_JOB_TYPES = set(CONTROLLED_STAGE_JOB_TYPES)


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
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    episode_index INTEGER,
                    storage_name TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    episode_uri TEXT,
                    collection_path TEXT,
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

                CREATE TABLE IF NOT EXISTS segments (
                    segment_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    manual_job_id TEXT NOT NULL DEFAULT '',
                    mano_job_id TEXT NOT NULL DEFAULT '',
                    manual_2d_uri TEXT NOT NULL DEFAULT '',
                    mano_patch_uri TEXT NOT NULL DEFAULT '',
                    lease_owner TEXT NOT NULL DEFAULT '',
                    lease_until TEXT NOT NULL DEFAULT '',
                    cleanup_manifest_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    manual_completed_at TEXT NOT NULL DEFAULT '',
                    mano_completed_at TEXT NOT NULL DEFAULT '',
                    failed_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );
                CREATE INDEX IF NOT EXISTS idx_segments_episode_status
                    ON segments(episode_id, status, start_frame);
                CREATE INDEX IF NOT EXISTS idx_segments_status_created
                    ON segments(status, created_at);

                CREATE TABLE IF NOT EXISTS workflow_stage_controls (
                    job_type TEXT PRIMARY KEY,
                    lease_enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS nas_sync_outbox (
                    event_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    nas_episode_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT NOT NULL,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );
                CREATE INDEX IF NOT EXISTS idx_nas_sync_outbox_due
                    ON nas_sync_outbox(status, next_attempt_at, created_at);
                PRAGMA user_version = 5;
                """
            )
            # A process may stop after claiming an event but before recording its
            # result. Replaying the fixed NAS commands is safer than stranding the
            # lifecycle transition forever.
            conn.execute(
                "UPDATE nas_sync_outbox SET status = 'pending', updated_at = ? WHERE status = 'running'",
                (now_iso(),),
            )
            self._ensure_episode_storage_columns(conn)
            conn.execute("DROP INDEX IF EXISTS idx_episodes_storage_name")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_storage_location "
                "ON episodes(subject_id, task_name, storage_name) WHERE storage_name <> ''"
            )
            timestamp = now_iso()
            for job_type in sorted(CONTROLLED_STAGE_JOB_TYPES):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO workflow_stage_controls (
                        job_type, lease_enabled, updated_at, updated_by, note
                    ) VALUES (?, ?, ?, 'system', ?)
                    """,
                    (
                        job_type,
                        1 if job_type in DEFAULT_OPEN_STAGE_JOB_TYPES else 0,
                        timestamp,
                        "default open" if job_type in DEFAULT_OPEN_STAGE_JOB_TYPES else "default paused",
                    ),
                )
            default_open_types = sorted(DEFAULT_OPEN_STAGE_JOB_TYPES)
            conn.execute(
                f"""
                UPDATE workflow_stage_controls
                SET lease_enabled = 1,
                    updated_at = ?,
                    note = 'default open'
                WHERE job_type IN ({", ".join("?" for _ in default_open_types)})
                  AND lease_enabled = 0
                  AND updated_by = 'system'
                  AND note = 'default paused'
                """,
                (timestamp, *default_open_types),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [str(row[1]) for row in rows]

    def _ensure_episode_storage_columns(self, conn: sqlite3.Connection) -> None:
        columns = set(self._table_columns(conn, "episodes"))
        if "episode_uri" not in columns:
            conn.execute("ALTER TABLE episodes ADD COLUMN episode_uri TEXT")
            columns.add("episode_uri")
        if "collection_path" not in columns:
            conn.execute("ALTER TABLE episodes ADD COLUMN collection_path TEXT")
            columns.add("collection_path")
        if "storage_name" not in columns:
            conn.execute("ALTER TABLE episodes ADD COLUMN storage_name TEXT NOT NULL DEFAULT ''")

    def enqueue_nas_sync_event(
        self,
        *,
        event_id: str,
        episode_id: str,
        nas_episode_id: str,
        action: str,
    ) -> Dict[str, Any]:
        event_id = str(event_id or "").strip()
        episode_id = str(episode_id or "").strip()
        nas_episode_id = str(nas_episode_id or "").strip()
        action = str(action or "").strip()
        if not event_id or not episode_id or not nas_episode_id or not action:
            raise WorkflowError(
                HTTPStatus.BAD_REQUEST,
                "event_id, episode_id, nas_episode_id, and action are required",
            )
        timestamp = now_iso()
        with self.connect() as conn:
            if self._get_episode_unlocked(conn, episode_id) is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
            conn.execute(
                """
                INSERT OR IGNORE INTO nas_sync_outbox (
                    event_id, episode_id, nas_episode_id, action, status,
                    attempt, next_attempt_at, last_error, created_at, updated_at,
                    completed_at
                ) VALUES (?, ?, ?, ?, 'pending', 0, ?, '', ?, ?, '')
                """,
                (event_id, episode_id, nas_episode_id, action, timestamp, timestamp, timestamp),
            )
            row = conn.execute(
                "SELECT * FROM nas_sync_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise WorkflowError(HTTPStatus.INTERNAL_SERVER_ERROR, f"NAS sync event not found: {event_id}")
            existing = self._row_to_nas_sync_event(row)
            if existing["episode_id"] != episode_id or existing["action"] != action:
                raise WorkflowError(HTTPStatus.CONFLICT, f"NAS sync event identity conflict: {event_id}")
            return existing

    def claim_nas_sync_event(self) -> Optional[Dict[str, Any]]:
        timestamp = now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT current.* FROM nas_sync_outbox AS current
                WHERE current.status = 'pending' AND current.next_attempt_at <= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM nas_sync_outbox AS earlier
                      WHERE earlier.episode_id = current.episode_id
                        AND earlier.rowid < current.rowid
                        AND earlier.status <> 'succeeded'
                  )
                ORDER BY current.created_at, current.rowid
                LIMIT 1
                """,
                (timestamp,),
            ).fetchone()
            if row is None:
                return None
            event_id = str(row["event_id"])
            conn.execute(
                """
                UPDATE nas_sync_outbox
                SET status = 'running', attempt = attempt + 1, updated_at = ?
                WHERE event_id = ?
                """,
                (timestamp, event_id),
            )
            claimed = conn.execute(
                "SELECT * FROM nas_sync_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            return self._row_to_nas_sync_event(claimed) if claimed is not None else None

    def complete_nas_sync_event(self, event_id: str) -> Dict[str, Any]:
        timestamp = now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE nas_sync_outbox
                SET status = 'succeeded', last_error = '', completed_at = ?, updated_at = ?
                WHERE event_id = ?
                """,
                (timestamp, timestamp, str(event_id or "").strip()),
            )
            if cursor.rowcount != 1:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"NAS sync event not found: {event_id}")
            row = conn.execute(
                "SELECT * FROM nas_sync_outbox WHERE event_id = ?",
                (str(event_id or "").strip(),),
            ).fetchone()
            return self._row_to_nas_sync_event(row)

    def retry_nas_sync_event(self, event_id: str, *, error: str, retry_seconds: int) -> Dict[str, Any]:
        timestamp = now_iso()
        next_attempt_at = _future_iso(max(1, int(retry_seconds or 1)))
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE nas_sync_outbox
                SET status = 'pending', next_attempt_at = ?, last_error = ?, updated_at = ?
                WHERE event_id = ?
                """,
                (next_attempt_at, str(error or ""), timestamp, str(event_id or "").strip()),
            )
            if cursor.rowcount != 1:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"NAS sync event not found: {event_id}")
            row = conn.execute(
                "SELECT * FROM nas_sync_outbox WHERE event_id = ?",
                (str(event_id or "").strip(),),
            ).fetchone()
            return self._row_to_nas_sync_event(row)

    def list_nas_sync_events(self, status: str = "") -> List[Dict[str, Any]]:
        clean_status = str(status or "").strip()
        query = "SELECT * FROM nas_sync_outbox"
        params: Tuple[Any, ...] = ()
        if clean_status:
            query += " WHERE status = ?"
            params = (clean_status,)
        query += " ORDER BY created_at, rowid"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_nas_sync_event(row) for row in rows]

    @staticmethod
    def _row_to_nas_sync_event(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "event_id": str(row["event_id"]),
            "episode_id": str(row["episode_id"]),
            "nas_episode_id": str(row["nas_episode_id"]),
            "action": str(row["action"]),
            "status": str(row["status"]),
            "attempt": int(row["attempt"]),
            "next_attempt_at": str(row["next_attempt_at"]),
            "last_error": str(row["last_error"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "completed_at": str(row["completed_at"]),
        }

    def create_or_update_episode(
        self,
        *,
        episode_id: Optional[str] = None,
        subject_id: str,
        task_name: str,
        episode_index: Optional[int] = None,
        storage_name: str = "",
        status: str = "planned",
        episode_uri: str = "",
        collection_path: str = "",
        frame_count: Optional[int] = None,
        cameras: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        status = require_episode_status(status)
        episode_id = str(episode_id or "").strip() or _new_id("episode")
        subject_id = str(subject_id or "").strip()
        task_name = str(task_name or "").strip()
        storage_name = str(storage_name or "").strip()
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
                "storage_name": storage_name,
                "status": status,
                "episode_uri": episode_uri,
                "collection_path": collection_path,
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
                        "storage_name": storage_name or existing.get("storage_name", ""),
                        "status": status or existing["status"],
                        "episode_uri": episode_uri or existing.get("episode_uri", ""),
                        "collection_path": collection_path or existing.get("collection_path", ""),
                        "frame_count": frame_count if frame_count is not None else existing.get("frame_count"),
                        "cameras": cameras if cameras is not None else existing.get("cameras", []),
                        "metadata": {**existing.get("metadata", {}), **(metadata or {})},
                        "created_at": existing["created_at"],
                        "updated_at": now,
                    }
                )
            conn.execute(
                """
                INSERT INTO episodes (
                    episode_id, subject_id, task_name, episode_index, storage_name, status,
                    episode_uri, collection_path, frame_count, cameras_json,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    subject_id = excluded.subject_id,
                    task_name = excluded.task_name,
                    episode_index = excluded.episode_index,
                    storage_name = excluded.storage_name,
                    status = excluded.status,
                    episode_uri = excluded.episode_uri,
                    collection_path = excluded.collection_path,
                    frame_count = excluded.frame_count,
                    cameras_json = excluded.cameras_json,
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    values["episode_id"],
                    values["subject_id"],
                    values["task_name"],
                    values["episode_index"],
                    values["storage_name"],
                    values["status"],
                    values["episode_uri"],
                    values["collection_path"],
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

    def list_episodes(
        self,
        *,
        subject_id: str = "",
        task_name: str = "",
        statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        subject_id = str(subject_id or "").strip()
        task_name = str(task_name or "").strip()
        if subject_id:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        if task_name:
            clauses.append("task_name = ?")
            params.append(task_name)
        if statuses:
            clean_statuses = [require_episode_status(status) for status in statuses]
            placeholders = ", ".join("?" for _ in clean_statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(clean_statuses)
        query = "SELECT * FROM episodes"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY subject_id, task_name, episode_index, created_at, episode_id"
        with self.connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_episode(row) for row in rows]

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
        episode_uri: str = "",
        collection_path: str = "",
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
                SET episode_uri = ?, collection_path = ?, metadata_json = ?, updated_at = ?
                WHERE episode_id = ?
                """,
                (
                    str(episode_uri or episode.get("episode_uri") or ""),
                    str(collection_path or episode.get("collection_path") or ""),
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

    def jobs_by_type(self, job_type: str) -> List[Dict[str, Any]]:
        job_type = require_job_type(job_type)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE type = ? ORDER BY created_at, job_id",
                (job_type,),
            ).fetchall()
            return [self._row_to_job(row) for row in rows]

    def get_stage_control(self, job_type: str) -> Dict[str, Any]:
        job_type = require_stage_job_type(job_type)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM workflow_stage_controls WHERE job_type = ?",
                (job_type,),
            ).fetchone()
            if row is None:
                timestamp = now_iso()
                default_open = job_type in DEFAULT_OPEN_STAGE_JOB_TYPES
                conn.execute(
                    """
                    INSERT INTO workflow_stage_controls (
                        job_type, lease_enabled, updated_at, updated_by, note
                    ) VALUES (?, ?, ?, 'system', ?)
                    """,
                    (job_type, 1 if default_open else 0, timestamp, "default open" if default_open else "default paused"),
                )
                row = conn.execute(
                    "SELECT * FROM workflow_stage_controls WHERE job_type = ?",
                    (job_type,),
                ).fetchone()
            if row is None:
                raise WorkflowError(HTTPStatus.INTERNAL_SERVER_ERROR, f"stage control not found: {job_type}")
            return self._row_to_stage_control(row)

    def set_stage_control(
        self,
        *,
        job_type: str,
        lease_enabled: bool,
        updated_by: str = "",
        note: str = "",
    ) -> Dict[str, Any]:
        job_type = require_stage_job_type(job_type)
        updated_by = str(updated_by or "").strip() or "api"
        note = str(note or "").strip()
        with self.connect() as conn:
            timestamp = now_iso()
            conn.execute(
                """
                INSERT INTO workflow_stage_controls (
                    job_type, lease_enabled, updated_at, updated_by, note
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_type) DO UPDATE SET
                    lease_enabled = excluded.lease_enabled,
                    updated_at = excluded.updated_at,
                    updated_by = excluded.updated_by,
                    note = excluded.note
                """,
                (job_type, 1 if lease_enabled else 0, timestamp, updated_by, note),
            )
            row = conn.execute(
                "SELECT * FROM workflow_stage_controls WHERE job_type = ?",
                (job_type,),
            ).fetchone()
            if row is None:
                raise WorkflowError(HTTPStatus.INTERNAL_SERVER_ERROR, f"stage control not found: {job_type}")
            return self._row_to_stage_control(row)

    def lease_enabled_for_job_type(self, job_type: str) -> bool:
        job_type = require_job_type(job_type)
        if job_type not in CONTROLLED_STAGE_JOB_TYPES:
            return True
        return bool(self.get_stage_control(job_type).get("lease_enabled"))

    def lease_job(
        self,
        *,
        job_type: str,
        lease_owner: str,
        lease_seconds: int = 300,
        task_name: str = "",
        subject_id: str = "",
        episode_id: str = "",
        job_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        job_type = require_job_type(job_type)
        lease_owner = str(lease_owner or "").strip()
        if not lease_owner:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "lease_owner is required")
        task_name = str(task_name or "").strip()
        subject_id = str(subject_id or "").strip()
        episode_id = str(episode_id or "").strip()
        job_id = str(job_id or "").strip()
        lease_seconds = max(1, int(lease_seconds or 300))
        now = now_iso()
        lease_until = _future_iso(lease_seconds)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            params: List[Any] = [job_type]
            query = """
                SELECT jobs.* FROM jobs
                LEFT JOIN episodes ON episodes.episode_id = jobs.episode_id
                WHERE jobs.type = ? AND jobs.status NOT IN ('succeeded', 'failed', 'canceled')
            """
            if task_name:
                query += " AND episodes.task_name = ?"
                params.append(task_name)
            if subject_id:
                query += " AND episodes.subject_id = ?"
                params.append(subject_id)
            if episode_id:
                query += " AND jobs.episode_id = ?"
                params.append(episode_id)
            if job_id:
                query += " AND jobs.job_id = ?"
                params.append(job_id)
            query += " ORDER BY jobs.created_at, jobs.job_id"
            rows = conn.execute(query, tuple(params)).fetchall()
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

    def merge_job_result(self, *, job_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """Persist non-terminal worker progress without changing the lease."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = self._get_job_unlocked(conn, job_id)
            if job is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"job not found: {job_id}")
            merged = dict(job.get("result") or {})
            merged.update(dict(values or {}))
            conn.execute(
                "UPDATE jobs SET result_json = ?, updated_at = ? WHERE job_id = ?",
                (_json_dumps(merged), now_iso(), job_id),
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

    def create_segment(
        self,
        *,
        segment_id: Optional[str] = None,
        episode_id: str,
        start_frame: int,
        end_frame: int,
        status: str = "pending_manual",
        manual_job_id: str = "",
        mano_job_id: str = "",
        manual_2d_uri: str = "",
        mano_patch_uri: str = "",
        cleanup_manifest: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        status = require_segment_status(status)
        episode_id = str(episode_id or "").strip()
        if not episode_id:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "episode_id is required")
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if end_frame < start_frame:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "segment end_frame must be >= start_frame")
        segment_id = str(segment_id or "").strip() or _new_id("segment")
        with self.connect() as conn:
            if self._get_episode_unlocked(conn, episode_id) is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
            existing = self._get_segment_unlocked(conn, segment_id)
            if existing is not None:
                return existing
            now = now_iso()
            conn.execute(
                """
                INSERT INTO segments (
                    segment_id, episode_id, start_frame, end_frame, status,
                    manual_job_id, mano_job_id, manual_2d_uri, mano_patch_uri,
                    lease_owner, lease_until, cleanup_manifest_json, metadata_json,
                    error, created_at, updated_at, manual_completed_at, mano_completed_at, failed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?, ?, '', ?, ?, '', '', '')
                """,
                (
                    segment_id,
                    episode_id,
                    start_frame,
                    end_frame,
                    status,
                    str(manual_job_id or ""),
                    str(mano_job_id or ""),
                    str(manual_2d_uri or ""),
                    str(mano_patch_uri or ""),
                    _json_dumps(cleanup_manifest or {}),
                    _json_dumps(metadata or {}),
                    now,
                    now,
                ),
            )
            segment = self._get_segment_unlocked(conn, segment_id)
            if segment is None:
                raise WorkflowError(HTTPStatus.INTERNAL_SERVER_ERROR, f"created segment not found: {segment_id}")
            return segment

    def get_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            return self._get_segment_unlocked(conn, segment_id)

    def segments_for_episode(self, episode_id: str, statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        episode_id = str(episode_id or "").strip()
        if not episode_id:
            return []
        clauses = ["episode_id = ?"]
        params: List[Any] = [episode_id]
        if statuses:
            clean_statuses = [require_segment_status(status) for status in statuses]
            placeholders = ", ".join("?" for _ in clean_statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(clean_statuses)
        query = "SELECT * FROM segments WHERE " + " AND ".join(clauses)
        query += " ORDER BY start_frame, end_frame, segment_id"
        with self.connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_segment(row) for row in rows]

    def list_segments(
        self,
        *,
        task_name: str = "",
        subject_id: str = "",
        episode_id: str = "",
        statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        task_name = str(task_name or "").strip()
        subject_id = str(subject_id or "").strip()
        episode_id = str(episode_id or "").strip()
        if task_name:
            clauses.append("episodes.task_name = ?")
            params.append(task_name)
        if subject_id:
            clauses.append("episodes.subject_id = ?")
            params.append(subject_id)
        if episode_id:
            clauses.append("segments.episode_id = ?")
            params.append(episode_id)
        if statuses:
            clean_statuses = [require_segment_status(status) for status in statuses]
            placeholders = ", ".join("?" for _ in clean_statuses)
            clauses.append(f"segments.status IN ({placeholders})")
            params.extend(clean_statuses)
        query = """
            SELECT segments.* FROM segments
            JOIN episodes ON episodes.episode_id = segments.episode_id
        """
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += """
            ORDER BY episodes.task_name, episodes.subject_id,
                     episodes.episode_index IS NULL, episodes.episode_index,
                     episodes.created_at, episodes.episode_id,
                     segments.start_frame, segments.end_frame, segments.segment_id
        """
        with self.connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_segment(row) for row in rows]

    def set_episode_segments_manual_labeling(self, *, episode_id: str, manual_job_id: str) -> List[Dict[str, Any]]:
        timestamp = now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE segments
                SET status = 'manual_labeling', manual_job_id = ?, error = '', updated_at = ?
                WHERE episode_id = ? AND status IN ('pending_manual', 'manual_labeling')
                """,
                (str(manual_job_id or ""), timestamp, str(episode_id or "")),
            )
        return self.segments_for_episode(episode_id)

    def reset_episode_segments_manual(self, *, episode_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE segments
                SET status = 'pending_manual', error = '', updated_at = ?
                WHERE episode_id = ? AND status = 'manual_labeling'
                """,
                (now_iso(), str(episode_id or "")),
            )
        return self.segments_for_episode(episode_id)

    def complete_episode_segments_manual(
        self,
        *,
        episode_id: str,
        manual_job_id: str,
        manual_2d_uri: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        timestamp = now_iso()
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM segments WHERE episode_id = ? ORDER BY start_frame, end_frame, segment_id",
                (str(episode_id or ""),),
            ).fetchall()
            if not rows:
                raise WorkflowError(HTTPStatus.CONFLICT, f"episode has no manual segments: {episode_id}")
            for row in rows:
                segment = self._row_to_segment(row)
                merged_metadata = dict(segment.get("metadata") or {})
                merged_metadata.update(dict(metadata or {}))
                conn.execute(
                    """
                    UPDATE segments
                    SET status = 'manual_labeled', manual_job_id = ?, manual_2d_uri = ?,
                        metadata_json = ?, error = '', manual_completed_at = ?, updated_at = ?
                    WHERE segment_id = ?
                    """,
                    (
                        str(manual_job_id or ""),
                        str(manual_2d_uri or ""),
                        _json_dumps(merged_metadata),
                        timestamp,
                        timestamp,
                        str(segment.get("segment_id") or ""),
                    ),
                )
        return self.segments_for_episode(episode_id)

    def set_episode_segments_mano_state(
        self,
        *,
        episode_id: str,
        status: str,
        mano_job_id: str,
        error: str = "",
    ) -> List[Dict[str, Any]]:
        if status not in {"mano_queued", "mano_running", "mano_succeeded", "failed"}:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, f"unsupported episode segment 3D status: {status}")
        timestamp = now_iso()
        mano_completed_at = timestamp if status == "mano_succeeded" else ""
        failed_at = timestamp if status == "failed" else ""
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE segments
                SET status = ?, mano_job_id = ?, mano_patch_uri = '', error = ?,
                    mano_completed_at = ?, failed_at = ?, updated_at = ?
                WHERE episode_id = ?
                """,
                (
                    status,
                    str(mano_job_id or ""),
                    str(error or ""),
                    mano_completed_at,
                    failed_at,
                    timestamp,
                    str(episode_id or ""),
                ),
            )
        return self.segments_for_episode(episode_id)

    def _get_episode_unlocked(self, conn: sqlite3.Connection, episode_id: str) -> Optional[Dict[str, Any]]:
        row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
        return self._row_to_episode(row) if row is not None else None

    def _get_job_unlocked(self, conn: sqlite3.Connection, job_id: str) -> Optional[Dict[str, Any]]:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row is not None else None

    def _get_segment_unlocked(self, conn: sqlite3.Connection, segment_id: str) -> Optional[Dict[str, Any]]:
        row = conn.execute("SELECT * FROM segments WHERE segment_id = ?", (segment_id,)).fetchone()
        return self._row_to_segment(row) if row is not None else None

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "episode_id": row["episode_id"],
            "subject_id": row["subject_id"],
            "task_name": row["task_name"],
            "episode_index": row["episode_index"],
            "storage_name": row["storage_name"] or "",
            "status": row["status"],
            "episode_uri": row["episode_uri"] or "",
            "collection_path": row["collection_path"] or "",
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

    @staticmethod
    def _row_to_segment(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "segment_id": row["segment_id"],
            "episode_id": row["episode_id"],
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "status": row["status"],
            "manual_job_id": row["manual_job_id"] or "",
            "mano_job_id": row["mano_job_id"] or "",
            "manual_2d_uri": row["manual_2d_uri"] or "",
            "mano_patch_uri": row["mano_patch_uri"] or "",
            "lease_owner": row["lease_owner"] or "",
            "lease_until": row["lease_until"] or "",
            "cleanup_manifest": _json_loads(row["cleanup_manifest_json"], {}),
            "metadata": _json_loads(row["metadata_json"], {}),
            "error": row["error"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "manual_completed_at": row["manual_completed_at"] or "",
            "mano_completed_at": row["mano_completed_at"] or "",
            "failed_at": row["failed_at"] or "",
        }

    @staticmethod
    def _row_to_stage_control(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "job_type": row["job_type"],
            "lease_enabled": bool(row["lease_enabled"]),
            "updated_at": row["updated_at"],
            "updated_by": row["updated_by"] or "",
            "note": row["note"] or "",
        }
