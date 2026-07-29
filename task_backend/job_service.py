from __future__ import annotations

import calendar
import json
import re
import time
import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote, unquote, urlparse

try:
    from .storage_resolver import local_uri_from_path, path_from_local_uri, uri_join
    from .workflow_models import TERMINAL_JOB_STATUSES, WorkflowError, json_object, require_job_type, require_stage_job_type
    from .workflow_store import WorkflowStore
except ImportError:  # pragma: no cover - script execution fallback
    from storage_resolver import local_uri_from_path, path_from_local_uri, uri_join  # type: ignore
    from workflow_models import TERMINAL_JOB_STATUSES, WorkflowError, json_object, require_job_type, require_stage_job_type  # type: ignore
    from workflow_store import WorkflowStore  # type: ignore


_FRAME_RE = re.compile(r"^(\d+)\.[^.]+$")
AUTO_LABEL_PUSHABLE_STATUSES = {"uploaded"}
STAGE_ARTIFACT_KINDS = {
    "auto_label": {"pred_2d"},
    "qc": {"qc_report"},
    "manual_label": {"corrected_2d"},
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_iso_seconds(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError, OverflowError):
        return None


def _elapsed_seconds_since(value: Any) -> Optional[int]:
    parsed = _parse_iso_seconds(value)
    if parsed is None:
        return None
    return max(0, int(time.time() - parsed))


def _truthy_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "passed", "pass"}:
            return True
        if normalized in {"0", "false", "no", "off", "failed", "fail"}:
            return False
    return bool(value)


def _short_summary(value: Any, limit: int = 220) -> str:
    if value is None or value == "" or value == {} or value == []:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _new_id(prefix: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(prefix or "item").strip().lower()).strip("_")
    return f"{clean or 'item'}_{uuid.uuid4().hex[:12]}"


def _stable_id_part(value: Any, fallback: str = "item") -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return clean or fallback


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _as_int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _normalize_uri_mounts(value: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for prefix, root in (value or {}).items():
        clean_prefix = str(prefix or "").strip().rstrip("/")
        clean_root = str(root or "").strip()
        if clean_prefix and clean_root:
            out[clean_prefix] = clean_root
    return out


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact_job(job: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(job.get("payload") or {})
    result = dict(job.get("result") or {})
    out = {
        "job_id": str(job.get("job_id") or ""),
        "type": str(job.get("type") or ""),
        "status": str(job.get("status") or ""),
        "lease_owner": str(job.get("lease_owner") or ""),
        "lease_until": str(job.get("lease_until") or ""),
        "attempt": _optional_int(job.get("attempt")) or 0,
        "created_at": str(job.get("created_at") or ""),
        "updated_at": str(job.get("updated_at") or ""),
        "batch_index": _optional_int(payload.get("batch_index")),
        "batch_count": _optional_int(payload.get("batch_count")),
        "frames": len(payload.get("frames") or []) if isinstance(payload.get("frames"), list) else None,
    }
    if result.get("error"):
        out["error"] = str(result.get("error") or "")
    return out


def _discover_cameras(episode_dir: Optional[Path]) -> List[str]:
    if episode_dir is None or not episode_dir.exists():
        return []
    cameras = []
    for child in episode_dir.iterdir():
        if child.is_dir() and (child / "RGB").is_dir():
            cameras.append(child.name)
    return sorted(cameras)


def _discover_frames(episode_dir: Optional[Path], cameras: List[str]) -> List[int]:
    if episode_dir is None or not cameras:
        return []
    rgb_dir = episode_dir / cameras[0] / "RGB"
    if not rgb_dir.is_dir():
        return []
    frames = set()
    for child in rgb_dir.iterdir():
        if not child.is_file():
            continue
        match = _FRAME_RE.match(child.name)
        if match:
            frames.add(int(match.group(1)))
    return sorted(frames)


def _local_episode_path(data_uri: str, local_path: str) -> Optional[Path]:
    if local_path:
        return Path(local_path).expanduser().resolve()
    if data_uri.startswith("local://"):
        return Path(path_from_local_uri(data_uri)).expanduser().resolve()
    return None


def _infer_subject_task_episode(path: Optional[Path]) -> Tuple[str, str, str]:
    if path is None:
        return "", "", ""
    parts = path.parts
    if len(parts) < 3:
        return "", "", path.name
    return parts[-3], parts[-2], parts[-1]


class JobService:
    def __init__(
        self,
        store: WorkflowStore,
        *,
        auto_label_after_upload: bool = False,
        auto_label_batch_size: int = 200,
        uri_mounts: Optional[Mapping[str, Any]] = None,
    ):
        self.store = store
        self.auto_label_after_upload = bool(auto_label_after_upload)
        self.auto_label_batch_size = max(1, int(auto_label_batch_size or 200))
        self.uri_mounts = _normalize_uri_mounts(uri_mounts)

    def create_manual_label_job(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload_in = dict(body or {})
        local_path = str(
            payload_in.get("local_episode_path")
            or payload_in.get("local_capture_path")
            or payload_in.get("local_path")
            or ""
        ).strip()
        data_uri = str(payload_in.get("data_uri") or "").strip()
        if not data_uri and local_path:
            data_uri = local_uri_from_path(local_path)
        if not data_uri:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "data_uri or local_path is required")

        episode_path = _local_episode_path(data_uri, local_path)
        inferred_subject, inferred_task, inferred_episode = _infer_subject_task_episode(episode_path)
        subject_id = str(payload_in.get("subject_id") or inferred_subject or "dev_subject").strip()
        task_name = str(payload_in.get("task_name") or inferred_task or "manual_label").strip()
        episode_id = str(payload_in.get("episode_id") or inferred_episode or _new_id("episode")).strip()
        episode_index = _optional_int(payload_in.get("episode_index"))

        cameras = _as_str_list(payload_in.get("cameras")) or _discover_cameras(episode_path)
        frames = _as_int_list(payload_in.get("frames")) or _discover_frames(episode_path, cameras)
        frame_count = _optional_int(payload_in.get("frame_count")) or len(frames) or None
        priority = _optional_int(payload_in.get("priority"))
        if priority is None:
            priority = 50

        metadata = json_object(payload_in.get("metadata"), "metadata")
        metadata.setdefault("created_by", "dev_label_job")
        episode = self.store.create_or_update_episode(
            episode_id=episode_id,
            subject_id=subject_id,
            task_name=task_name,
            episode_index=episode_index,
            status="manual_label_pending",
            data_uri=data_uri,
            local_capture_path=str(episode_path) if episode_path is not None else local_path,
            frame_count=frame_count,
            cameras=cameras,
            metadata=metadata,
        )

        job_id = str(payload_in.get("job_id") or _new_id("manual_label")).strip()
        job_payload = {
            "job_id": job_id,
            "episode_id": episode["episode_id"],
            "subject_id": episode["subject_id"],
            "task_name": episode["task_name"],
            "data_uri": episode["data_uri"],
            "cameras": cameras,
            "frames": frames,
            "rgb_path_template": str(payload_in.get("rgb_path_template") or "{camera}/RGB/{frame:05d}.png"),
            "prediction_dir": str(payload_in.get("prediction_dir") or "pred_2d"),
            "correction_dir": str(payload_in.get("correction_dir") or "corrected_2d"),
            "reason": str(payload_in.get("reason") or "dev_created"),
            "priority": priority,
        }
        extra_payload = json_object(payload_in.get("payload"), "payload")
        job_payload.update(extra_payload)
        job = self._create_job_once(
            job_id=job_id,
            job_type="manual_label",
            episode_id=episode["episode_id"],
            payload=job_payload,
        )
        return self.enrich_job(job)

    def create_dev_job(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload_in = dict(body or {})
        job_type = require_job_type(str(payload_in.get("type") or payload_in.get("job_type") or ""))
        if job_type == "manual_label":
            return self.create_manual_label_job(payload_in)

        episode_obj = json_object(payload_in.get("episode"), "episode")
        episode_id = str(payload_in.get("episode_id") or episode_obj.get("episode_id") or "").strip()
        if not episode_id:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "episode_id is required for dev jobs")
        if self.store.get_episode(episode_id) is None:
            self.store.create_or_update_episode(
                episode_id=episode_id,
                subject_id=str(episode_obj.get("subject_id") or payload_in.get("subject_id") or "dev_subject"),
                task_name=str(episode_obj.get("task_name") or payload_in.get("task_name") or "dev_task"),
                episode_index=_optional_int(episode_obj.get("episode_index")),
                status=str(episode_obj.get("status") or "planned"),
                data_uri=str(episode_obj.get("data_uri") or payload_in.get("data_uri") or ""),
                local_capture_path=str(episode_obj.get("local_capture_path") or ""),
                frame_count=_optional_int(episode_obj.get("frame_count")),
                cameras=_as_str_list(episode_obj.get("cameras")),
                metadata=json_object(episode_obj.get("metadata"), "episode.metadata"),
            )
        job_id = str(payload_in.get("job_id") or _new_id(job_type)).strip()
        job_payload = json_object(payload_in.get("payload"), "payload")
        if not job_payload:
            job_payload = {k: v for k, v in payload_in.items() if k not in {"episode", "payload"}}
        job_payload.setdefault("job_id", job_id)
        job_payload.setdefault("episode_id", episode_id)
        job = self._create_job_once(job_id=job_id, job_type=job_type, episode_id=episode_id, payload=job_payload)
        return self.enrich_job(job)

    def lease_job(self, body: Dict[str, Any], *, forced_type: str = "") -> Dict[str, Any]:
        job_type = require_job_type(forced_type or str(body.get("type") or body.get("job_type") or ""))
        if not self.store.lease_enabled_for_job_type(job_type):
            raise WorkflowError(HTTPStatus.CONFLICT, f"leasing disabled for job type: {job_type}")
        owner = str(body.get("lease_owner") or body.get("operator_id") or body.get("worker_id") or "").strip()
        lease_seconds = _optional_int(body.get("lease_seconds")) or 300
        task_name = str(body.get("task_name") or body.get("task") or "").strip()
        subject_id = str(body.get("subject_id") or body.get("subject") or "").strip()
        job = self.store.lease_job(
            job_type=job_type,
            lease_owner=owner,
            lease_seconds=lease_seconds,
            task_name=task_name,
            subject_id=subject_id,
        )
        if job is None:
            if task_name:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"no queued {job_type} job is available for task: {task_name}")
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"no queued {job_type} job is available")
        self._mark_episode_for_leased_job(job)
        return self.enrich_job(job)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        job = self.store.get_job(job_id)
        if job is None:
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"job not found: {job_id}")
        return self.enrich_job(job)

    def upload_status(self, episode_id: str) -> Dict[str, Any]:
        episode_id = str(episode_id or "").strip()
        if not episode_id:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "episode_id is required")
        episode = self.store.get_episode(episode_id)
        if episode is None:
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
        all_jobs = self.store.jobs_for_episode(episode_id)
        upload_jobs = [job for job in all_jobs if job.get("type") == "upload"]
        upload_job = upload_jobs[-1] if upload_jobs else None
        artifacts = self.store.artifacts_for_episode(episode_id)
        upload_artifacts = [item for item in artifacts if item.get("kind") == "nas_episode"]
        result = dict(upload_job.get("result") or {}) if upload_job else {}
        status = str(upload_job.get("status") or "missing") if upload_job else "missing"
        total_bytes = _optional_int(result.get("total_bytes")) or 0
        copied_bytes = _optional_int(result.get("copied_bytes")) or 0
        percent = result.get("percent")
        try:
            percent_value = float(percent)
        except (TypeError, ValueError):
            percent_value = 100.0 if status == "succeeded" and total_bytes == 0 else 0.0
            if total_bytes > 0:
                percent_value = min(100.0, max(0.0, copied_bytes * 100.0 / total_bytes))
        nas_uri = str(result.get("nas_uri") or "")
        if not nas_uri and upload_artifacts:
            nas_uri = str(upload_artifacts[-1].get("uri") or "")
        if not nas_uri and str(episode.get("data_uri") or "").startswith("nas://"):
            nas_uri = str(episode.get("data_uri") or "")
        workflow_status = str(episode.get("status") or "planned")
        active_jobs = [
            job for job in all_jobs
            if str(job.get("status") or "") not in {"succeeded", "failed", "canceled"}
        ]
        active_job = active_jobs[-1] if active_jobs else None
        job_counts: Dict[str, int] = {}
        for job in all_jobs:
            key = str(job.get("type") or "job") + ":" + str(job.get("status") or "unknown")
            job_counts[key] = job_counts.get(key, 0) + 1
        return {
            "episode": episode,
            "workflow": {
                "status": workflow_status,
                "active_job_id": str(active_job.get("job_id") or "") if active_job else "",
                "active_job_type": str(active_job.get("type") or "") if active_job else "",
                "active_job_status": str(active_job.get("status") or "") if active_job else "",
                "job_count": len(all_jobs),
                "job_counts": job_counts,
                "updated_at": str(episode.get("updated_at") or ""),
            },
            "upload": {
                "available": upload_job is not None,
                "job_id": str(upload_job.get("job_id") or "") if upload_job else "",
                "status": status,
                "phase": str(result.get("phase") or ("complete" if status == "succeeded" else status)),
                "percent": round(percent_value, 2),
                "copied_bytes": copied_bytes,
                "total_bytes": total_bytes,
                "files_done": _optional_int(result.get("files_done")) or 0,
                "files_total": _optional_int(result.get("files_total")) or 0,
                "nas_uri": nas_uri,
                "local_path": str(result.get("local_path") or episode.get("local_capture_path") or ""),
                "error": str(result.get("error") or ""),
                "updated_at": str(upload_job.get("updated_at") or "") if upload_job else "",
                "result": result,
            },
            "artifacts": upload_artifacts,
            "jobs": [_compact_job(job) for job in all_jobs],
        }

    def workflow_stage(self, job_type: str) -> Dict[str, Any]:
        job_type = require_stage_job_type(job_type)
        control = self.store.get_stage_control(job_type)
        jobs = self.store.jobs_by_type(job_type)
        now = now_iso()
        counts = {
            "queued": 0,
            "leased_running": 0,
            "succeeded": 0,
            "failed": 0,
            "canceled": 0,
            "total": len(jobs),
        }
        active: List[Dict[str, Any]] = []
        queued: List[Dict[str, Any]] = []
        completed: List[Dict[str, Any]] = []
        for job in jobs:
            status = str(job.get("status") or "")
            if status == "queued":
                counts["queued"] += 1
            elif status in {"leased", "running"}:
                counts["leased_running"] += 1
            elif status == "succeeded":
                counts["succeeded"] += 1
            elif status == "failed":
                counts["failed"] += 1
            elif status == "canceled":
                counts["canceled"] += 1

            item = self._stage_job_item(job, now)
            if status == "queued":
                queued.append(item)
            elif status in {"leased", "running"}:
                active.append(item)
            elif status in TERMINAL_JOB_STATUSES:
                completed.append(item)

        completed.sort(key=lambda item: (str(item.get("updated_at") or ""), str(item.get("job_id") or "")), reverse=True)
        return {
            "job_type": job_type,
            "control": control,
            "lease_status": "开放" if control.get("lease_enabled") else "暂停",
            "stats": counts,
            "active": active,
            "queued": queued,
            "completed": completed,
            "generated_at": now,
        }

    def set_stage_leasing(self, job_type: str, enabled: bool, body: Dict[str, Any]) -> Dict[str, Any]:
        control = self.store.set_stage_control(
            job_type=require_stage_job_type(job_type),
            lease_enabled=enabled,
            updated_by=str(body.get("updated_by") or body.get("operator_id") or body.get("user") or "api"),
            note=str(body.get("note") or ""),
        )
        return {"control": control, "stage": self.workflow_stage(job_type)}

    def push_auto_label(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(body or {})
        episode_id = str(payload.get("episode_id") or payload.get("episode") or "").strip()
        task_name = str(payload.get("task_name") or payload.get("task") or "").strip()
        subject_id = str(payload.get("subject_id") or payload.get("subject") or "").strip()
        scope = str(payload.get("scope") or "").strip().lower()
        pushed_by = str(payload.get("pushed_by") or payload.get("operator_id") or payload.get("user") or "api").strip() or "api"

        if episode_id:
            episode = self.store.get_episode(episode_id)
            if episode is None:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
            episodes = [episode]
            resolved_scope = "episode"
        elif task_name:
            episodes = self.store.list_episodes(subject_id=subject_id, task_name=task_name)
            resolved_scope = "task"
        elif scope in {"", "all", "eligible"} or bool(payload.get("all")):
            episodes = self.store.list_episodes(subject_id=subject_id)
            resolved_scope = "all"
        else:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "episode_id, task_name, or scope=all is required")

        outcomes = [self._push_auto_label_for_episode(episode, pushed_by=pushed_by) for episode in episodes]
        created_jobs = [job for item in outcomes for job in item.get("jobs", [])]
        pushed_episodes = [item for item in outcomes if item.get("pushed")]
        skipped = [item for item in outcomes if not item.get("pushed")]
        return {
            "scope": resolved_scope,
            "episode_id": episode_id,
            "subject_id": subject_id,
            "task_name": task_name,
            "checked": len(outcomes),
            "pushed": len(pushed_episodes),
            "created_jobs": len(created_jobs),
            "skipped": len(skipped),
            "jobs": created_jobs,
            "episodes": outcomes,
        }

    def heartbeat_job(self, job_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        owner = str(body.get("lease_owner") or body.get("operator_id") or body.get("worker_id") or "").strip()
        lease_seconds = _optional_int(body.get("lease_seconds")) or 300
        status = str(body.get("status") or "").strip()
        job = self.store.heartbeat_job(
            job_id=job_id,
            lease_owner=owner,
            lease_seconds=lease_seconds,
            status=status,
        )
        return self.enrich_job(job)

    def complete_job(self, job_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        before = self.store.get_job(job_id)
        if before is None:
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"job not found: {job_id}")
        result = json_object(body.get("result"), "result")
        if not result:
            result = {k: v for k, v in body.items() if k not in {"artifacts", "artifact"}}
        job, changed = self.store.complete_job(job_id=job_id, result=result)
        if changed:
            self._after_job_complete(job, result, self._artifacts_from_body(body))
        enriched = self.enrich_job(job)
        enriched["completed"] = True
        enriched["changed"] = changed
        return enriched

    def fail_job(self, job_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        error = str(body.get("error") or body.get("message") or "job failed")
        result = json_object(body.get("result"), "result")
        job = self.store.fail_job(job_id=job_id, error=error, result=result)
        return self.enrich_job(job)

    def release_job(self, job_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        reason = str(body.get("reason") or "")
        job, released = self.store.release_job(job_id=job_id, reason=reason)
        if released and job.get("episode_id") and job.get("type") == "manual_label":
            self.store.update_episode_status(job["episode_id"], "manual_label_pending")
        enriched = self.enrich_job(job)
        enriched["released"] = released
        return enriched

    def record_collection_reservation(self, reservation: Dict[str, Any]) -> None:
        self.store.create_or_update_episode(
            episode_id=str(reservation.get("reservation_id") or ""),
            subject_id=str(reservation.get("subject_id") or ""),
            task_name=str(reservation.get("task_name") or ""),
            episode_index=_optional_int(reservation.get("episode_number")),
            status="reserved_for_collection",
            metadata={
                "reservation_id": reservation.get("reservation_id"),
                "client_id": reservation.get("client_id"),
                "source": "collection_api",
            },
        )

    def record_collection_confirm(self, reservation: Dict[str, Any]) -> None:
        episode_id = str(reservation.get("reservation_id") or "")
        local_path = str(reservation.get("local_path") or "")
        data_uri = local_uri_from_path(local_path) if local_path else ""
        frame_count = _optional_int(reservation.get("frame_count"))
        self.store.create_or_update_episode(
            episode_id=episode_id,
            subject_id=str(reservation.get("subject_id") or ""),
            task_name=str(reservation.get("task_name") or ""),
            episode_index=_optional_int(reservation.get("episode_number")),
            status="captured",
            data_uri=data_uri,
            local_capture_path=local_path,
            frame_count=frame_count,
            metadata={
                "reservation_id": reservation.get("reservation_id"),
                "client_id": reservation.get("client_id"),
                "idempotency_key": reservation.get("idempotency_key"),
                "source": "collection_api",
            },
        )
        job_id = f"upload_{episode_id.replace('-', '_')}"
        payload = {
            "job_id": job_id,
            "episode_id": episode_id,
            "data_uri": data_uri,
            "local_capture_path": local_path,
            "reason": "collection_confirmed",
        }
        self._create_job_once(job_id=job_id, job_type="upload", episode_id=episode_id, payload=payload)

    def record_collection_release(self, reservation: Dict[str, Any]) -> None:
        episode_id = str(reservation.get("reservation_id") or "")
        if not episode_id:
            return
        episode = self.store.get_episode(episode_id)
        if episode is not None:
            self.store.update_episode_status(episode_id, "planned", {"released_from_collection": True})

    def enrich_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        episode = self.store.get_episode(str(job.get("episode_id") or "")) if job.get("episode_id") else None
        artifacts = self.store.artifacts_for_episode(job["episode_id"]) if job.get("episode_id") else []
        payload = dict(job.get("payload") or {})
        payload.setdefault("job_id", job.get("job_id"))
        if episode is not None:
            payload.setdefault("episode_id", episode.get("episode_id"))
            payload.setdefault("subject_id", episode.get("subject_id"))
            payload.setdefault("task_name", episode.get("task_name"))
            payload.setdefault("data_uri", episode.get("data_uri"))
            payload.setdefault("cameras", episode.get("cameras") or [])
            payload.setdefault("local_capture_path", episode.get("local_capture_path") or "")
        resolved_data_path = self.resolve_data_path(
            str(payload.get("data_uri") or ""),
            str(payload.get("local_capture_path") or (episode or {}).get("local_capture_path") or ""),
        )
        if resolved_data_path:
            payload.setdefault("resolved_data_path", resolved_data_path)
        return {
            "job": job,
            "episode": episode,
            "artifacts": artifacts,
            "payload": payload,
        }

    def resolve_data_path(self, data_uri: str, local_capture_path: str = "") -> str:
        local_capture_path = str(local_capture_path or "").strip()
        if local_capture_path:
            return str(Path(local_capture_path).expanduser().resolve())
        value = str(data_uri or "").strip()
        if not value:
            return ""
        parsed = urlparse(value)
        if not parsed.scheme:
            return str(Path(value).expanduser().resolve())
        if parsed.scheme == "local":
            return str(Path(path_from_local_uri(value)).expanduser().resolve())
        best_prefix = ""
        best_root = ""
        for prefix, root in self.uri_mounts.items():
            if value == prefix or value.startswith(prefix + "/"):
                if len(prefix) > len(best_prefix):
                    best_prefix = prefix
                    best_root = root
        if not best_prefix:
            return ""
        suffix = value[len(best_prefix):].lstrip("/")
        return str((Path(best_root).expanduser() / unquote(suffix)).resolve())

    def _stage_job_item(self, job: Dict[str, Any], now: str) -> Dict[str, Any]:
        episode_id = str(job.get("episode_id") or "")
        episode = self.store.get_episode(episode_id) if episode_id else None
        artifacts = self.store.artifacts_for_episode(episode_id) if episode_id else []
        payload = dict(job.get("payload") or {})
        result = dict(job.get("result") or {})
        job_type = str(job.get("type") or "")
        relevant_artifacts = self._relevant_artifacts_for_job(job_type, artifacts)
        batch_index = _optional_int(payload.get("batch_index"))
        batch_count = _optional_int(payload.get("batch_count"))
        frames = payload.get("frames")
        frames_count = len(frames) if isinstance(frames, list) else None
        lease_until = str(job.get("lease_until") or "")
        lease_expired = bool(lease_until and lease_until <= now and str(job.get("status") or "") not in TERMINAL_JOB_STATUSES)
        subject_id = str((episode or {}).get("subject_id") or payload.get("subject_id") or "")
        task_name = str((episode or {}).get("task_name") or payload.get("task_name") or "")
        episode_index = (episode or {}).get("episode_index")
        if episode_index is None:
            episode_index = _optional_int(payload.get("episode_index"))
        return {
            "job_id": str(job.get("job_id") or ""),
            "type": job_type,
            "status": str(job.get("status") or ""),
            "episode_id": episode_id,
            "episode_url": f"/episodes/{quote(episode_id, safe='')}" if episode_id else "",
            "subject_id": subject_id,
            "task_name": task_name,
            "episode_index": episode_index,
            "episode_status": str((episode or {}).get("status") or ""),
            "lease_owner": str(job.get("lease_owner") or ""),
            "lease_until": lease_until,
            "lease_expired": lease_expired,
            "created_at": str(job.get("created_at") or ""),
            "updated_at": str(job.get("updated_at") or ""),
            "waiting_seconds": _elapsed_seconds_since(job.get("created_at")) if str(job.get("status") or "") == "queued" else None,
            "attempt": _optional_int(job.get("attempt")) or 0,
            "batch_index": batch_index,
            "batch_count": batch_count,
            "frames_count": frames_count,
            "frames": frames if isinstance(frames, list) else [],
            "result": result,
            "result_summary": _short_summary(result),
            "error": str(result.get("error") or ""),
            "error_summary": _short_summary(result.get("error")),
            "artifacts": relevant_artifacts,
            "artifact_summary": self._artifact_summary(relevant_artifacts),
        }

    @staticmethod
    def _relevant_artifacts_for_job(job_type: str, artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kinds = STAGE_ARTIFACT_KINDS.get(job_type, set())
        if not kinds:
            return artifacts
        return [artifact for artifact in artifacts if str(artifact.get("kind") or "") in kinds]

    @staticmethod
    def _artifact_summary(artifacts: List[Dict[str, Any]]) -> str:
        if not artifacts:
            return ""
        parts = []
        for artifact in artifacts[:5]:
            parts.append(f"{artifact.get('kind')}: {artifact.get('uri')}")
        if len(artifacts) > 5:
            parts.append(f"+{len(artifacts) - 5} more")
        return _short_summary("; ".join(parts))

    def _create_job_once(self, *, job_id: str, job_type: str, episode_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        existing = self.store.get_job(job_id)
        if existing is not None:
            return existing
        return self.store.create_job(job_id=job_id, job_type=job_type, episode_id=episode_id, payload=payload)

    def _mark_episode_for_leased_job(self, job: Dict[str, Any]) -> None:
        episode_id = str(job.get("episode_id") or "")
        if not episode_id:
            return
        status_by_type = {
            "auto_label": "auto_labeling",
            "qc": "qc_running",
            "manual_label": "manual_labeling",
        }
        status = status_by_type.get(str(job.get("type") or ""))
        if status:
            self.store.update_episode_status(episode_id, status, {"active_job_id": job.get("job_id")})

    def _after_job_complete(self, job: Dict[str, Any], result: Dict[str, Any], artifacts: List[Dict[str, Any]]) -> None:
        episode_id = str(job.get("episode_id") or "")
        if not episode_id:
            return
        job_type = str(job.get("type") or "")
        payload = dict(job.get("payload") or {})
        for artifact in artifacts:
            self._register_artifact_from_payload(episode_id, artifact)

        if job_type == "upload":
            nas_uri = str(result.get("nas_uri") or result.get("data_uri") or "")
            for artifact in artifacts:
                if str(artifact.get("kind") or "") == "nas_episode" and str(artifact.get("uri") or ""):
                    nas_uri = str(artifact.get("uri"))
                    break
            if nas_uri:
                episode = self.store.get_episode(episode_id)
                self.store.update_episode_storage(
                    episode_id,
                    data_uri=nas_uri,
                    local_capture_path=str((episode or {}).get("local_capture_path") or payload.get("local_capture_path") or ""),
                    metadata={
                        "nas_uri": nas_uri,
                        "upload_job_id": job.get("job_id"),
                        "uploaded_at": result.get("completed_at") or result.get("finished_at") or "",
                    },
                )
            self.store.update_episode_status(episode_id, "uploaded")
            if self.auto_label_after_upload:
                self._create_auto_label_jobs_from_upload(episode_id, payload, result, nas_uri)
        elif job_type == "auto_label":
            if not any(str(artifact.get("kind") or "") == "pred_2d" for artifact in artifacts) and payload.get("data_uri"):
                self.store.register_artifact(
                    episode_id=episode_id,
                    kind="pred_2d",
                    uri=uri_join(str(payload.get("data_uri")), str(payload.get("prediction_dir") or "pred_2d")),
                    metadata={"source_job_id": job.get("job_id"), "compatibility": "kp2d may map to pred_2d"},
                )
            if self._all_episode_jobs_succeeded(episode_id, "auto_label"):
                self.store.update_episode_status(episode_id, "auto_labeled")
                self._create_qc_job_from_existing_episode(episode_id, result)
        elif job_type == "qc":
            if not any(str(artifact.get("kind") or "") == "qc_report" for artifact in artifacts):
                self._register_default_qc_report(episode_id, job, result)
            passed = _truthy_bool(result.get("passed") if "passed" in result else result.get("qc_passed"))
            self.store.update_episode_status(episode_id, "qc_passed" if passed else "qc_failed")
            if not passed:
                self._create_manual_label_from_existing_episode(episode_id, result)
        elif job_type == "review":
            self.store.update_episode_status(episode_id, "review_passed")
        elif job_type == "manual_label":
            if not any(str(artifact.get("kind") or "") == "corrected_2d" for artifact in artifacts) and payload.get("data_uri"):
                self.store.register_artifact(
                    episode_id=episode_id,
                    kind="corrected_2d",
                    uri=uri_join(str(payload.get("data_uri")), str(payload.get("correction_dir") or "corrected_2d")),
                    metadata={"source_job_id": job.get("job_id")},
                )
            self.store.update_episode_status(episode_id, "manual_labeled")

    def _create_auto_label_jobs_from_upload(
        self,
        episode_id: str,
        upload_payload: Dict[str, Any],
        upload_result: Dict[str, Any],
        nas_uri: str,
    ) -> None:
        episode = self.store.get_episode(episode_id)
        if episode is None:
            return
        created = self._create_auto_label_jobs_for_episode(
            episode,
            data_uri=str(nas_uri or upload_result.get("nas_uri") or episode.get("data_uri") or ""),
            local_path=str(
                upload_result.get("local_path")
                or upload_payload.get("local_capture_path")
                or episode.get("local_capture_path")
                or ""
            ),
            source_payload=upload_payload,
            reason="upload_succeeded",
        )
        if created:
            self.store.update_episode_status(
                episode_id,
                "uploaded",
                {
                    "auto_label_after_upload": True,
                    "auto_label_batch_count": len(created),
                    "auto_label_batch_size": self.auto_label_batch_size,
                },
            )

    def _push_auto_label_for_episode(self, episode: Dict[str, Any], *, pushed_by: str) -> Dict[str, Any]:
        episode_id = str(episode.get("episode_id") or "")
        status = str(episode.get("status") or "")
        data_uri = self._auto_label_data_uri(episode)
        base = {
            "episode_id": episode_id,
            "subject_id": str(episode.get("subject_id") or ""),
            "task_name": str(episode.get("task_name") or ""),
            "episode_index": episode.get("episode_index"),
            "status": status,
            "data_uri": data_uri,
            "pushed": False,
            "jobs": [],
        }
        if status not in AUTO_LABEL_PUSHABLE_STATUSES:
            return {**base, "reason": f"episode status is not pushable: {status or 'unknown'}"}
        if not data_uri:
            return {**base, "reason": "episode has no nas_uri or data_uri"}
        existing = self.store.jobs_for_episode(episode_id, "auto_label")
        if existing:
            return {
                **base,
                "reason": "auto_label job already exists",
                "existing_jobs": [_compact_job(job) for job in existing],
            }

        jobs = self._create_auto_label_jobs_for_episode(
            episode,
            data_uri=data_uri,
            local_path=str(episode.get("local_capture_path") or ""),
            source_payload={},
            reason="manual_push",
            pushed_by=pushed_by,
        )
        if jobs:
            self.store.update_episode_status(
                episode_id,
                "uploaded",
                {
                    "auto_label_pushed_at": now_iso(),
                    "auto_label_pushed_by": pushed_by,
                    "auto_label_batch_count": len(jobs),
                    "auto_label_batch_size": self.auto_label_batch_size,
                },
            )
        return {
            **base,
            "pushed": bool(jobs),
            "reason": "created" if jobs else "no jobs created",
            "jobs": [_compact_job(job) for job in jobs],
        }

    @staticmethod
    def _auto_label_data_uri(episode: Dict[str, Any]) -> str:
        metadata = episode.get("metadata") if isinstance(episode.get("metadata"), dict) else {}
        nas_uri = str(metadata.get("nas_uri") or "").strip()
        return nas_uri or str(episode.get("data_uri") or "").strip()

    def _create_auto_label_jobs_for_episode(
        self,
        episode: Dict[str, Any],
        *,
        data_uri: str,
        local_path: str,
        source_payload: Dict[str, Any],
        reason: str,
        pushed_by: str = "",
    ) -> List[Dict[str, Any]]:
        episode_id = str(episode.get("episode_id") or "")
        data_uri = str(data_uri or "").strip()
        if not episode_id or not data_uri:
            return []
        source_payload = dict(source_payload or {})
        episode_path = _local_episode_path(str(episode.get("data_uri") or ""), local_path)
        cameras = (
            _as_str_list(source_payload.get("cameras"))
            or _as_str_list(episode.get("cameras"))
            or _discover_cameras(episode_path)
        )
        frames = _as_int_list(source_payload.get("frames")) or _discover_frames(episode_path, cameras)
        if not frames:
            frame_count = _optional_int(episode.get("frame_count")) or _optional_int(source_payload.get("frame_count"))
            if frame_count:
                frames = list(range(frame_count))

        batches = self._frame_batches(frames, self.auto_label_batch_size)
        if not batches:
            batches = [[]]
        batch_count = len(batches)
        base_id = _stable_id_part(episode_id, "episode")
        created: List[Dict[str, Any]] = []
        for index, batch_frames in enumerate(batches, 1):
            job_id = f"auto_label_{base_id}_b{index:04d}"
            payload = {
                "job_id": job_id,
                "episode_id": episode_id,
                "subject_id": episode.get("subject_id"),
                "task_name": episode.get("task_name"),
                "data_uri": data_uri,
                "local_capture_path": local_path,
                "cameras": cameras,
                "frames": batch_frames,
                "batch_index": index,
                "batch_count": batch_count,
                "batch_size": self.auto_label_batch_size,
                "rgb_path_template": str(source_payload.get("rgb_path_template") or "{camera}/RGB/{frame:05d}.png"),
                "prediction_dir": str(source_payload.get("prediction_dir") or "pred_2d"),
                "correction_dir": str(source_payload.get("correction_dir") or "corrected_2d"),
                "reason": reason,
            }
            if pushed_by:
                payload["pushed_by"] = pushed_by
            created.append(self._create_job_once(job_id=job_id, job_type="auto_label", episode_id=episode_id, payload=payload))
        return created

    @staticmethod
    def _frame_batches(frames: Sequence[int], batch_size: int) -> List[List[int]]:
        clean_frames = [int(frame) for frame in frames if not isinstance(frame, bool)]
        if not clean_frames:
            return []
        size = max(1, int(batch_size or 1))
        return [clean_frames[i : i + size] for i in range(0, len(clean_frames), size)]

    def _all_episode_jobs_succeeded(self, episode_id: str, job_type: str) -> bool:
        jobs = self.store.jobs_for_episode(episode_id, job_type)
        return bool(jobs) and all(str(job.get("status") or "") == "succeeded" for job in jobs)

    def _episode_frames_from_jobs(self, episode_id: str, job_type: str) -> List[int]:
        frames: List[int] = []
        seen = set()
        for job in self.store.jobs_for_episode(episode_id, job_type):
            for frame in _as_int_list((job.get("payload") or {}).get("frames")):
                if frame in seen:
                    continue
                seen.add(frame)
                frames.append(frame)
        return frames

    def _create_qc_job_from_existing_episode(self, episode_id: str, result: Dict[str, Any]) -> None:
        if self.store.jobs_for_episode(episode_id, "qc"):
            return
        episode = self.store.get_episode(episode_id)
        if episode is None:
            return
        frames = self._episode_frames_from_jobs(episode_id, "auto_label")
        if not frames:
            frame_count = _optional_int(episode.get("frame_count"))
            if frame_count:
                frames = list(range(frame_count))
        pred_artifacts = self._relevant_artifacts_for_job("auto_label", self.store.artifacts_for_episode(episode_id))
        base_id = _stable_id_part(episode_id, "episode")
        job_id = str(result.get("qc_job_id") or f"qc_{base_id}")
        payload = {
            "job_id": job_id,
            "episode_id": episode["episode_id"],
            "subject_id": episode["subject_id"],
            "task_name": episode["task_name"],
            "data_uri": episode["data_uri"],
            "cameras": episode.get("cameras") or [],
            "frames": frames,
            "pred_artifacts": pred_artifacts,
            "pred_uri": str(pred_artifacts[-1].get("uri") or "") if pred_artifacts else "",
            "reason": "auto_label_succeeded",
        }
        self._create_job_once(job_id=job_id, job_type="qc", episode_id=episode_id, payload=payload)

    def _register_default_qc_report(self, episode_id: str, job: Dict[str, Any], result: Dict[str, Any]) -> None:
        episode = self.store.get_episode(episode_id)
        payload = dict(job.get("payload") or {})
        report_uri = str(result.get("qc_report_uri") or result.get("report_uri") or "").strip()
        data_uri = str((episode or {}).get("data_uri") or payload.get("data_uri") or "").strip()
        if not report_uri and data_uri:
            report_uri = uri_join(data_uri, "qc_report.json")
        if not report_uri:
            return
        self.store.register_artifact(
            episode_id=episode_id,
            kind="qc_report",
            uri=report_uri,
            metadata={
                "source_job_id": job.get("job_id"),
                "passed": _truthy_bool(result.get("passed") if "passed" in result else result.get("qc_passed")),
            },
        )

    def _create_manual_label_from_existing_episode(self, episode_id: str, result: Dict[str, Any]) -> None:
        if self.store.jobs_for_episode(episode_id, "manual_label"):
            return
        episode = self.store.get_episode(episode_id)
        if episode is None:
            return
        frames = result.get("frames") if isinstance(result.get("frames"), list) else self._episode_frames_from_jobs(episode_id, "auto_label")
        if not frames:
            frame_count = _optional_int(episode.get("frame_count"))
            if frame_count:
                frames = list(range(frame_count))
        base_id = _stable_id_part(episode_id, "episode")
        job_id = str(result.get("manual_label_job_id") or f"manual_label_{base_id}")
        payload = {
            "job_id": job_id,
            "episode_id": episode["episode_id"],
            "subject_id": episode["subject_id"],
            "task_name": episode["task_name"],
            "data_uri": episode["data_uri"],
            "cameras": episode.get("cameras") or [],
            "frames": frames,
            "rgb_path_template": "{camera}/RGB/{frame:05d}.png",
            "prediction_dir": "pred_2d",
            "correction_dir": "corrected_2d",
            "reason": "qc_failed",
            "priority": _optional_int(result.get("priority")) or 50,
        }
        self._create_job_once(job_id=job_id, job_type="manual_label", episode_id=episode_id, payload=payload)

    def _register_artifact_from_payload(self, episode_id: str, artifact: Dict[str, Any]) -> None:
        kind = str(artifact.get("kind") or "").strip()
        uri = str(artifact.get("uri") or "").strip()
        if not kind or not uri:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "artifact kind and uri are required")
        metadata = json_object(artifact.get("metadata"), "artifact.metadata")
        self.store.register_artifact(
            episode_id=episode_id,
            kind=kind,
            uri=uri,
            metadata=metadata,
            artifact_id=str(artifact.get("artifact_id") or "").strip() or None,
        )

    @staticmethod
    def _artifacts_from_body(body: Dict[str, Any]) -> List[Dict[str, Any]]:
        artifacts = body.get("artifacts")
        if isinstance(artifacts, list):
            return [dict(item) for item in artifacts if isinstance(item, dict)]
        artifact = body.get("artifact")
        if isinstance(artifact, dict):
            return [dict(artifact)]
        return []
