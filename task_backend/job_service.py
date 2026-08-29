from __future__ import annotations

import ast
import calendar
import json
import re
import shutil
import struct
import time
import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote, unquote, urlparse

try:
    from .storage_resolver import uri_join
    from .workflow_models import TERMINAL_JOB_STATUSES, WorkflowError, json_object, require_job_type, require_stage_job_type
    from .workflow_store import WorkflowStore
except ImportError:  # pragma: no cover - script execution fallback
    from storage_resolver import uri_join  # type: ignore
    from workflow_models import TERMINAL_JOB_STATUSES, WorkflowError, json_object, require_job_type, require_stage_job_type  # type: ignore
    from workflow_store import WorkflowStore  # type: ignore


_FRAME_RE = re.compile(r"^(\d+)\.[^.]+$")
_NON_LABEL_CAMERA_TOKENS = ("ego", "pico", "fisheye")
AUTO_LABEL_PUSHABLE_STATUSES = {"uploaded"}
STAGE_ARTIFACT_KINDS = {
    "auto_label": {"pred_2d", "auto_2d", "optimized_pose", "mano_episode"},
    "qc": {"qc_report"},
    "manual_label": {"manual_2d", "corrected_2d"},
    "manual_3d": {"optimized_pose", "mano_episode"},
}
FINAL_3D_SOURCES_REL_PATH = "workflow/final_3d_sources.json"
NAS_SYNC_QUALITY_TAKE = "quality_take"
NAS_SYNC_QUALITY_PASSED = "quality_passed"
NAS_SYNC_QUALITY_NEEDS_LABELING = "quality_needs_labeling"


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


def _human_operator_id(*sources: Any) -> str:
    for source in sources:
        if isinstance(source, Mapping):
            for key in ("operator_id", "operator", "completed_by", "confirmed_by", "username", "user"):
                value = str(source.get(key) or "").strip()
                if value:
                    return value
        else:
            value = str(source or "").strip()
            if value:
                return value
    return ""


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


def _is_label_camera_id(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return not any(token in lowered for token in _NON_LABEL_CAMERA_TOKENS)


def _label_camera_ids(value: Any) -> List[str]:
    return [camera for camera in _as_str_list(value) if _is_label_camera_id(camera)]


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


def _normalize_nas_mounts(value: Optional[Mapping[str, Any]]) -> Dict[str, str]:
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


def _is_joint3d_materialization_failure(job: Dict[str, Any]) -> bool:
    result = dict(job.get("result") or {})
    return (
        str(job.get("type") or "") == "auto_label"
        and str(job.get("status") or "") == "failed"
        and str(result.get("phase") or "") == "mano_materialization"
        and str(result.get("worker_id") or "").startswith("publisher_bridge:")
    )


def _compact_job(job: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(job.get("payload") or {})
    result = dict(job.get("result") or {})
    operator_id = _human_operator_id(result)
    out = {
        "job_id": str(job.get("job_id") or ""),
        "type": str(job.get("type") or ""),
        "status": str(job.get("status") or ""),
        "lease_owner": str(job.get("lease_owner") or ""),
        "lease_until": str(job.get("lease_until") or ""),
        "attempt": _optional_int(job.get("attempt")) or 0,
        "created_at": str(job.get("created_at") or ""),
        "updated_at": str(job.get("updated_at") or ""),
        "scope": str(payload.get("scope") or payload.get("label_scope") or payload.get("mano_scope") or ""),
        "frames": len(payload.get("frames") or []) if isinstance(payload.get("frames"), list) else None,
    }
    if operator_id:
        out["operator_id"] = operator_id
    if result.get("error"):
        out["error"] = str(result.get("error") or "")
    if result.get("phase"):
        out["failure_phase"] = str(result.get("phase") or "")
    if _is_joint3d_materialization_failure(job):
        out["can_retry_joint3d"] = True
    return out


def _discover_cameras(episode_dir: Optional[Path]) -> List[str]:
    if episode_dir is None or not episode_dir.exists():
        return []
    cameras = []
    for child in episode_dir.iterdir():
        if child.is_dir() and (child / "RGB").is_dir() and _is_label_camera_id(child.name):
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


def _infer_subject_task_episode(path: Optional[Path]) -> Tuple[str, str, str]:
    if path is None:
        return "", "", ""
    parts = path.parts
    if len(parts) < 3:
        return "", "", path.name
    return parts[-3], parts[-2], parts[-1]


def _episode_storage_name(subject_id: Any, task_name: Any, episode_index: Any) -> str:
    del subject_id, task_name
    index = _optional_int(episode_index)
    if index is None or index <= 0:
        return ""
    return f"episode{index}"


class JobService:
    def __init__(
        self,
        store: WorkflowStore,
        *,
        auto_label_after_upload: bool = True,
        nas_mounts: Optional[Mapping[str, Any]] = None,
    ):
        self.store = store
        self.auto_label_after_upload = True
        self.nas_mounts = _normalize_nas_mounts(nas_mounts)

    def create_manual_label_job(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload_in = dict(body or {})
        collection_path = str(payload_in.get("collection_path") or "").strip()
        episode_uri = str(payload_in.get("episode_uri") or "").strip()
        if not episode_uri:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "episode_uri is required")

        nas_root_dir = self.nas_root_dir_from_uri(episode_uri)
        inferred_subject, inferred_task, inferred_episode = _infer_subject_task_episode(nas_root_dir)
        subject_id = str(payload_in.get("subject_id") or inferred_subject or "dev_subject").strip()
        task_name = str(payload_in.get("task_name") or inferred_task or "manual_label").strip()
        episode_id = str(payload_in.get("episode_id") or inferred_episode or _new_id("episode")).strip()
        episode_index = _optional_int(payload_in.get("episode_index"))

        cameras = _label_camera_ids(payload_in.get("cameras")) or _discover_cameras(nas_root_dir)
        frames = _as_int_list(payload_in.get("frames")) or _discover_frames(nas_root_dir, cameras)
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
            status="manual_correction_pending",
            episode_uri=episode_uri,
            collection_path=collection_path,
            frame_count=frame_count,
            cameras=cameras,
            metadata=metadata,
        )

        start_frame = _optional_int(payload_in.get("start_frame"))
        end_frame = _optional_int(payload_in.get("end_frame"))
        if start_frame is None or end_frame is None:
            if frames:
                start_frame = min(frames)
                end_frame = max(frames)
            else:
                start_frame = 0
                end_frame = 0
        segment_id = str(payload_in.get("segment_id") or payload_in.get("job_id") or _new_id("segment")).strip()
        segment_metadata = {
            "created_by": "dev_label_segment",
            "reason": str(payload_in.get("reason") or "dev_created"),
            "priority": priority,
        }
        extra_payload = json_object(payload_in.get("payload"), "payload")
        segment_metadata.update(extra_payload)
        self.store.create_segment(
            segment_id=segment_id,
            episode_id=episode["episode_id"],
            start_frame=start_frame,
            end_frame=end_frame,
            status="pending_manual",
            metadata=segment_metadata,
        )
        job = self._create_manual_label_episode_job(episode["episode_id"], reason="dev_created")
        return self.enrich_manual_label_job(job)

    def create_dev_job(self, body: Dict[str, Any]) -> Dict[str, Any]:
        payload_in = dict(body or {})
        job_type = require_job_type(str(payload_in.get("type") or payload_in.get("job_type") or ""))
        if job_type == "manual_label":
            return self.create_manual_label_job(payload_in)
        job_payload = json_object(payload_in.get("payload"), "payload")
        if not job_payload:
            job_payload = {k: v for k, v in payload_in.items() if k not in {"episode", "payload"}}

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
                episode_uri=str(episode_obj.get("episode_uri") or payload_in.get("episode_uri") or ""),
                collection_path=str(episode_obj.get("collection_path") or ""),
                frame_count=_optional_int(episode_obj.get("frame_count")),
                cameras=_label_camera_ids(episode_obj.get("cameras")),
                metadata=json_object(episode_obj.get("metadata"), "episode.metadata"),
            )
        job_id = str(payload_in.get("job_id") or _new_id(job_type)).strip()
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
        episode_id = str(body.get("episode_id") or body.get("episode") or "").strip()
        job_id = str(body.get("job_id") or "").strip()
        job = self.store.lease_job(
            job_type=job_type,
            lease_owner=owner,
            lease_seconds=lease_seconds,
            task_name=task_name,
            subject_id=subject_id,
            episode_id=episode_id,
            job_id=job_id,
        )
        if job is None:
            if job_id:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"no queued {job_type} job is available for job: {job_id}")
            if episode_id:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"no queued {job_type} job is available for episode: {episode_id}")
            if task_name:
                raise WorkflowError(HTTPStatus.NOT_FOUND, f"no queued {job_type} job is available for task: {task_name}")
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"no queued {job_type} job is available")
        self._mark_episode_for_leased_job(job)
        if str(job.get("type") or "") == "qc":
            self._enqueue_nas_sync(
                episode_id=str(job.get("episode_id") or ""),
                action=NAS_SYNC_QUALITY_TAKE,
                event_id=f"nas_quality_take:{job.get('episode_id')}",
            )
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
        if not nas_uri and str(episode.get("episode_uri") or "").startswith("nas://"):
            nas_uri = str(episode.get("episode_uri") or "")
        if upload_job is None and nas_uri:
            status = "succeeded"
            result = {
                "phase": "capture_uploaded",
                "percent": 100.0,
                "nas_uri": nas_uri,
                "collection_path": str(episode.get("collection_path") or ""),
                "direct_from_capture": True,
            }
            percent_value = 100.0
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
                "available": upload_job is not None or bool(nas_uri),
                "job_id": str(upload_job.get("job_id") or "") if upload_job else "",
                "status": status,
                "phase": str(result.get("phase") or ("complete" if status == "succeeded" else status)),
                "percent": round(percent_value, 2),
                "copied_bytes": copied_bytes,
                "total_bytes": total_bytes,
                "files_done": _optional_int(result.get("files_done")) or 0,
                "files_total": _optional_int(result.get("files_total")) or 0,
                "nas_uri": nas_uri,
                "collection_path": str(result.get("collection_path") or episode.get("collection_path") or ""),
                "error": str(result.get("error") or ""),
                "updated_at": str(upload_job.get("updated_at") or "") if upload_job else "",
                "result": result,
            },
            "artifacts": upload_artifacts,
            "workflow_artifacts": artifacts,
            "segments": [
                self._segment_episode_status_view(segment)
                for segment in self.store.segments_for_episode(episode_id)
            ],
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

    def joint3d_materialization_retry_candidate(self, episode_id: str) -> Dict[str, Any]:
        """Return the failed Publisher job eligible for a direct joint3d retry."""
        episode_id = str(episode_id or "").strip()
        if not episode_id:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "episode_id is required")
        episode = self.store.get_episode(episode_id)
        if episode is None:
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
        jobs = self.store.jobs_for_episode(episode_id, "auto_label")
        if not jobs:
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"auto_label job not found: {episode_id}")
        job = jobs[-1]
        if not _is_joint3d_materialization_failure(job):
            raise WorkflowError(
                HTTPStatus.CONFLICT,
                "latest auto_label job is not a retryable (2,99) -> joint3d conversion failure",
            )
        return self.enrich_job(job)

    def start_joint3d_materialization_retry(
        self,
        episode_id: str,
        body: Dict[str, Any],
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> Dict[str, Any]:
        """Mark a failed conversion running for one direct, non-queued retry."""
        candidate = self.joint3d_materialization_retry_candidate(episode_id)
        job = dict(candidate.get("job") or {})
        requested_by = str(
            body.get("requested_by") or body.get("operator_id") or body.get("user") or "episode_page"
        ).strip() or "episode_page"
        started = self.store.start_failed_job_retry(
            job_id=str(job.get("job_id") or ""),
            lease_owner=worker_id,
            lease_seconds=lease_seconds,
            requested_by=requested_by,
            reason="retry_joint3d_materialization",
        )
        self.store.update_episode_status(
            episode_id,
            "auto_labeling",
            {
                "joint3d_retry_job_id": started.get("job_id"),
                "joint3d_retry_requested_at": started.get("updated_at"),
                "joint3d_retry_requested_by": requested_by,
            },
        )
        return self.enrich_job(started)

    def label_tasks(self) -> Dict[str, Any]:
        jobs = [job for job in self.store.jobs_by_type("manual_label") if self._job_is_available_or_active(job)]
        groups: Dict[str, Dict[str, Any]] = {}
        subject_sets: Dict[str, set] = {}
        episode_sets: Dict[str, set] = {}
        for job in jobs:
            episode = self.store.get_episode(str(job.get("episode_id") or "")) or {}
            segments = self.store.segments_for_episode(str(job.get("episode_id") or ""))
            task_name = str(episode.get("task_name") or "Unspecified")
            group = groups.setdefault(
                task_name,
                {
                    "task_name": task_name,
                    "jobs": 0,
                    "segments": 0,
                    "pending_episodes": 0,
                    "leased_episodes": 0,
                    "frames": 0,
                    "oldest_created_at": "",
                },
            )
            subjects = subject_sets.setdefault(task_name, set())
            episodes = episode_sets.setdefault(task_name, set())
            group["jobs"] += 1
            group["segments"] += len(segments)
            if job.get("status") == "queued":
                group["pending_episodes"] += 1
            else:
                group["leased_episodes"] += 1
            group["frames"] += sum(self._segment_frame_count(segment) for segment in segments)
            created_at = str(job.get("created_at") or "")
            if created_at and (not group["oldest_created_at"] or created_at < group["oldest_created_at"]):
                group["oldest_created_at"] = created_at
            if episode.get("subject_id"):
                subjects.add(str(episode.get("subject_id")))
            if job.get("episode_id"):
                episodes.add(str(job.get("episode_id")))
        out = []
        for task_name, group in groups.items():
            subjects = sorted(subject_sets.get(task_name, set()))
            episodes = sorted(episode_sets.get(task_name, set()))
            group["subjects"] = subjects
            group["subject_summary"] = ", ".join(subjects)
            group["episodes"] = len(episodes)
            out.append(group)
        out.sort(key=lambda item: (str(item.get("task_name") or ""), str(item.get("oldest_created_at") or "")))
        return {"tasks": out, "generated_at": now_iso()}

    def label_task_episodes(self, task_name: str) -> Dict[str, Any]:
        task_name = str(task_name or "").strip()
        if not task_name:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "task_name is required")
        episodes: List[Dict[str, Any]] = []
        for job in self.store.jobs_by_type("manual_label"):
            if not self._job_is_available_or_active(job):
                continue
            episode_id = str(job.get("episode_id") or "")
            episode = self.store.get_episode(episode_id) or {}
            if str(episode.get("task_name") or "") != task_name:
                continue
            segments = self.store.segments_for_episode(episode_id)
            starts = [_optional_int(segment.get("start_frame")) for segment in segments]
            starts = [value for value in starts if value is not None]
            episodes.append(
                {
                    "episode_id": episode_id,
                    "job_id": str(job.get("job_id") or ""),
                    "task_name": task_name,
                    "subject_id": str(episode.get("subject_id") or ""),
                    "episode_index": episode.get("episode_index"),
                    "episode_status": str(episode.get("status") or ""),
                    "episode_uri": str(episode.get("episode_uri") or ""),
                    "job_status": str(job.get("status") or ""),
                    "segments": len(segments),
                    "frames": sum(self._segment_frame_count(segment) for segment in segments),
                    "first_start_frame": min(starts) if starts else None,
                    "oldest_created_at": str(job.get("created_at") or ""),
                }
            )
        episodes.sort(
            key=lambda item: (
                str(item.get("subject_id") or ""),
                item.get("episode_index") is None,
                item.get("episode_index") if item.get("episode_index") is not None else 0,
                str(item.get("oldest_created_at") or ""),
                str(item.get("episode_id") or ""),
            )
        )
        return {"task_name": task_name, "episodes": episodes, "generated_at": now_iso()}

    def lease_label_episode(self, body: Dict[str, Any]) -> Dict[str, Any]:
        leased = self.lease_job(body, forced_type="manual_label")
        job = dict(leased.get("job") or {})
        episode_id = str(job.get("episode_id") or "")
        self.store.set_episode_segments_manual_labeling(
            episode_id=episode_id,
            manual_job_id=str(job.get("job_id") or ""),
        )
        self.store.update_episode_status(
            episode_id,
            "manual_labeling",
            {"active_job_id": job.get("job_id"), "manual_label_scope": "episode"},
        )
        return self.enrich_manual_label_job(self.store.get_job(str(job.get("job_id") or "")) or job)

    def get_label_episode(self, episode_id: str) -> Dict[str, Any]:
        return self.enrich_manual_label_job(self._manual_label_job_for_episode(episode_id))

    def heartbeat_label_episode(self, episode_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        job = self._manual_label_job_for_episode(episode_id)
        updated = self.heartbeat_job(str(job.get("job_id") or ""), body)
        return self.enrich_manual_label_job(dict(updated.get("job") or job))

    def release_label_episode(self, episode_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        job = self._manual_label_job_for_episode(episode_id)
        released = self.release_job(str(job.get("job_id") or ""), body)
        if released.get("released"):
            self.store.reset_episode_segments_manual(episode_id=episode_id)
            self.store.update_episode_status(episode_id, "manual_correction_pending")
        response = self.enrich_manual_label_job(self.store.get_job(str(job.get("job_id") or "")) or job)
        response["released"] = bool(released.get("released"))
        return response

    def complete_label_episode(self, episode_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        job = self._manual_label_job_for_episode(episode_id)
        return self.complete_job(str(job.get("job_id") or ""), body)

    def fail_label_episode(self, episode_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        job = self._manual_label_job_for_episode(episode_id)
        return self.fail_job(str(job.get("job_id") or ""), body)

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

    def update_job_progress(self, job_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        return self.enrich_job(self.store.merge_job_result(job_id=job_id, values=values))

    def complete_job(self, job_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        before = self.store.get_job(job_id)
        if before is None:
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"job not found: {job_id}")
        result = json_object(body.get("result"), "result")
        if not result:
            result = {k: v for k, v in body.items() if k not in {"artifacts", "artifact"}}
        artifacts = self._artifacts_from_body(body)
        if str(before.get("type") or "") == "auto_label" and self._is_publisher_bridge_job(before, result):
            self._validate_publisher_bridge_completion(before, result, artifacts)
        operator_id = _human_operator_id(body, result)
        if operator_id:
            result["operator_id"] = operator_id
        job, changed = self.store.complete_job(job_id=job_id, result=result)
        # Publisher Bridge completion is also a repair operation.  If the job
        # row reached succeeded but a prior process interruption happened while
        # registering artifacts or queuing QC, an idempotent retry finishes the
        # downstream state transition.
        if changed or self._is_publisher_bridge_job(job, result):
            self._after_job_complete(job, result, artifacts)
        if str(job.get("type") or "") == "qc":
            # A repeated completion is also a repair path for NAS synchronization.
            # The QC job row may already be succeeded even if an earlier process
            # stopped before the outbox event was inserted.
            self._ensure_nas_qc_completion_sync(job, dict(job.get("result") or result))
        enriched = self.enrich_job(job)
        enriched["completed"] = True
        enriched["changed"] = changed
        return enriched

    def fail_job(self, job_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        error = str(body.get("error") or body.get("message") or "job failed")
        result = json_object(body.get("result"), "result")
        job = self.store.fail_job(job_id=job_id, error=error, result=result)
        cleanup_manifest = json_object(body.get("cleanup_manifest") or result.get("cleanup_manifest"), "cleanup_manifest")
        self._cleanup_manifest_outputs(cleanup_manifest)
        self._after_job_fail(job, error, result)
        return self.enrich_job(job)

    def release_job(self, job_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        reason = str(body.get("reason") or "")
        job, released = self.store.release_job(job_id=job_id, reason=reason)
        enriched = self.enrich_job(job)
        enriched["released"] = released
        return enriched

    def record_collection_reservation(self, reservation: Dict[str, Any]) -> None:
        operator_id = _human_operator_id(reservation, reservation.get("subject_id"))
        storage_name = str(reservation.get("storage_name") or "").strip() or _episode_storage_name(
            reservation.get("subject_id"), reservation.get("task_name"), reservation.get("episode_number")
        )
        self.store.create_or_update_episode(
            episode_id=str(reservation.get("reservation_id") or ""),
            subject_id=str(reservation.get("subject_id") or ""),
            task_name=str(reservation.get("task_name") or ""),
            episode_index=_optional_int(reservation.get("episode_number")),
            storage_name=storage_name,
            status="reserved_for_collection",
            metadata={
                "reservation_id": reservation.get("reservation_id"),
                "storage_name": storage_name,
                "client_id": reservation.get("client_id"),
                "collection_reserved_by": operator_id,
                "source": "collection_api",
            },
        )

    def record_collection_confirm(self, reservation: Dict[str, Any]) -> None:
        episode_id = str(reservation.get("reservation_id") or "")
        collection_path = str(reservation.get("collection_path") or "")
        episode_uri = str(reservation.get("episode_uri") or reservation.get("nas_uri") or "").strip()
        frame_count = _optional_int(reservation.get("frame_count"))
        operator_id = _human_operator_id(reservation, reservation.get("subject_id"))
        storage_name = str(reservation.get("storage_name") or "").strip() or _episode_storage_name(
            reservation.get("subject_id"), reservation.get("task_name"), reservation.get("episode_number")
        )
        self.store.create_or_update_episode(
            episode_id=episode_id,
            subject_id=str(reservation.get("subject_id") or ""),
            task_name=str(reservation.get("task_name") or ""),
            episode_index=_optional_int(reservation.get("episode_number")),
            storage_name=storage_name,
            status="uploaded" if episode_uri else "captured",
            episode_uri=episode_uri,
            collection_path=collection_path,
            frame_count=frame_count,
            metadata={
                "reservation_id": reservation.get("reservation_id"),
                "storage_name": storage_name,
                "client_id": reservation.get("client_id"),
                "idempotency_key": reservation.get("idempotency_key"),
                "nas_uri": episode_uri,
                "collection_operator_id": operator_id,
                "collection_confirmed_by": operator_id,
                "last_human_operator_id": operator_id,
                "source": "collection_api",
                "upload_mode": "capture_side" if episode_uri else "backend_upload",
            },
        )
        if episode_uri:
            existing_artifacts = self.store.artifacts_for_episode(episode_id)
            if not any(str(item.get("kind") or "") == "nas_episode" and str(item.get("uri") or "") == episode_uri for item in existing_artifacts):
                self.store.register_artifact(
                    episode_id=episode_id,
                    kind="nas_episode",
                    uri=episode_uri,
                    metadata={
                        "source": "capture_side_uploader",
                        "collection_path": collection_path,
                        "confirmed_at": now_iso(),
                    },
                )
            self._create_auto_label_jobs_from_upload(
                episode_id,
                {
                    "episode_id": episode_id,
                    "collection_path": collection_path,
                    "episode_uri": episode_uri,
                    "reason": "collection_uploaded_by_capture",
                },
                {
                    "ok": True,
                    "phase": "capture_uploaded",
                    "percent": 100.0,
                    "nas_uri": episode_uri,
                    "completed_at": now_iso(),
                    "direct_from_capture": True,
                },
                episode_uri,
            )
            return
        job_id = f"upload_{episode_id.replace('-', '_')}"
        payload = {
            "job_id": job_id,
            "episode_id": episode_id,
            "collection_path": collection_path,
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
            payload.setdefault("episode_index", episode.get("episode_index"))
            payload.setdefault("storage_name", episode.get("storage_name"))
            payload.setdefault("episode_uri", episode.get("episode_uri"))
            payload.setdefault("cameras", _label_camera_ids(episode.get("cameras") or []))
        self._strip_downstream_path_fields(payload)
        if episode is not None:
            episode_id = str(episode.get("episode_id") or job.get("episode_id") or "")
            cameras = _label_camera_ids(payload.get("cameras")) or self._cameras_for_episode_context(episode_id, episode, payload)
            if cameras:
                payload["cameras"] = cameras
            frames = _as_int_list(payload.get("frames")) or self._frames_for_episode_context(
                episode_id,
                episode,
                cameras,
                payload,
            )
            if frames:
                payload["frames"] = frames
        return {
            "job": job,
            "episode": episode,
            "artifacts": artifacts,
            "payload": payload,
        }

    def enrich_manual_label_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        response = self.enrich_job(job)
        episode_id = str(job.get("episode_id") or "")
        segments = self.store.segments_for_episode(episode_id)
        payload = dict(response.get("payload") or {})
        payload["scope"] = "episode"
        payload["label_scope"] = "episode"
        payload["segments"] = [
            {
                "segment_id": str(segment.get("segment_id") or ""),
                "start_frame": int(segment.get("start_frame") or 0),
                "end_frame": int(segment.get("end_frame") or 0),
                "status": str(segment.get("status") or ""),
                "reason": str((segment.get("metadata") or {}).get("reason") or ""),
            }
            for segment in segments
        ]
        payload["frames"] = sorted(
            {
                frame
                for segment in segments
                for frame in range(int(segment.get("start_frame") or 0), int(segment.get("end_frame") or 0) + 1)
            }
        )
        payload["manual_2d_output_uri"] = uri_join(
            str(payload.get("episode_uri") or ""),
            "manual_2d",
            "segments",
            str(job.get("job_id") or ""),
        )
        response["segments"] = segments
        response["payload"] = payload
        return response

    @staticmethod
    def _job_is_available_or_active(job: Dict[str, Any]) -> bool:
        status = str(job.get("status") or "")
        if status == "queued":
            return True
        lease_until = str(job.get("lease_until") or "")
        return status in {"leased", "running"} and (not lease_until or lease_until <= now_iso())

    def _manual_label_job_for_episode(self, episode_id: str) -> Dict[str, Any]:
        episode_id = str(episode_id or "").strip()
        jobs = self.store.jobs_for_episode(episode_id, "manual_label")
        if not jobs:
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"manual label episode not found: {episode_id}")
        active = [job for job in jobs if str(job.get("status") or "") not in TERMINAL_JOB_STATUSES]
        return (active or jobs)[-1]

    @staticmethod
    def _segment_frame_count(segment: Dict[str, Any]) -> int:
        start_frame = _optional_int(segment.get("start_frame")) or 0
        end_frame = _optional_int(segment.get("end_frame")) or start_frame
        return max(0, end_frame - start_frame + 1)

    @staticmethod
    def _segment_episode_status_view(segment: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(segment)
        status = str(segment.get("status") or "")
        if status == "pending_manual":
            manual_state = "待人工标注"
            optimization_state = "等待人工标注"
        elif status == "manual_labeling":
            manual_state = "人工标注中"
            optimization_state = "等待人工标注"
        elif status == "manual_labeled":
            manual_state = "人工标注完成"
            optimization_state = "待3D优化"
        elif status == "mano_queued":
            manual_state = "人工标注完成"
            optimization_state = "待3D优化"
        elif status == "mano_running":
            manual_state = "人工标注完成"
            optimization_state = "3D优化中"
        elif status == "mano_succeeded":
            manual_state = "人工标注完成"
            optimization_state = "3D优化完成"
        else:
            manual_state = "失败"
            optimization_state = "失败"
        out["manual_state"] = manual_state
        out["optimization_3d_state"] = optimization_state
        return out

    def nas_root_dir_from_uri(self, episode_uri: str) -> Optional[Path]:
        value = str(episode_uri or "").strip()
        if not value:
            return None
        best_prefix = ""
        best_root = ""
        for prefix, root in self.nas_mounts.items():
            if value == prefix or value.startswith(prefix + "/"):
                if len(prefix) > len(best_prefix):
                    best_prefix = prefix
                    best_root = root
        if not best_prefix:
            return None
        suffix = value[len(best_prefix):].lstrip("/")
        return (Path(best_root).expanduser() / unquote(suffix)).resolve()

    @staticmethod
    def _strip_downstream_path_fields(payload: Dict[str, Any]) -> None:
        payload.pop("collection_path", None)

    def _cameras_for_episode_context(
        self,
        episode_id: str,
        episode: Dict[str, Any],
        *payloads: Dict[str, Any],
    ) -> List[str]:
        for payload in payloads:
            cameras = _label_camera_ids((payload or {}).get("cameras"))
            if cameras:
                return cameras
        cameras = _label_camera_ids(episode.get("cameras"))
        if cameras:
            return cameras
        for job_type in ("auto_label", "qc", "manual_label", "manual_3d", "upload"):
            for job in self.store.jobs_for_episode(episode_id, job_type):
                cameras = _label_camera_ids((job.get("payload") or {}).get("cameras"))
                if cameras:
                    return cameras
        nas_root_dir = self.nas_root_dir_from_uri(str(episode.get("episode_uri") or ""))
        return _discover_cameras(nas_root_dir)

    def _frames_for_episode_context(
        self,
        episode_id: str,
        episode: Dict[str, Any],
        cameras: List[str],
        *payloads: Dict[str, Any],
    ) -> List[int]:
        for payload in payloads:
            frames = _as_int_list((payload or {}).get("frames"))
            if frames:
                return frames
        for job_type in ("manual_3d", "manual_label", "qc", "auto_label", "upload"):
            for job in self.store.jobs_for_episode(episode_id, job_type):
                frames = _as_int_list((job.get("payload") or {}).get("frames"))
                if frames:
                    return frames
        nas_root_dir = self.nas_root_dir_from_uri(str(episode.get("episode_uri") or ""))
        frames = _discover_frames(nas_root_dir, cameras)
        if frames:
            return frames
        frame_count = _optional_int(episode.get("frame_count"))
        if frame_count:
            return list(range(frame_count))
        return []

    def _stage_job_item(self, job: Dict[str, Any], now: str) -> Dict[str, Any]:
        episode_id = str(job.get("episode_id") or "")
        episode = self.store.get_episode(episode_id) if episode_id else None
        artifacts = self.store.artifacts_for_episode(episode_id) if episode_id else []
        payload = dict(job.get("payload") or {})
        result = dict(job.get("result") or {})
        operator_id = _human_operator_id(result)
        job_type = str(job.get("type") or "")
        relevant_artifacts = self._relevant_artifacts_for_job(job_type, artifacts)
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
            "operator_id": operator_id,
            "lease_until": lease_until,
            "lease_expired": lease_expired,
            "created_at": str(job.get("created_at") or ""),
            "updated_at": str(job.get("updated_at") or ""),
            "waiting_seconds": _elapsed_seconds_since(job.get("created_at")) if str(job.get("status") or "") == "queued" else None,
            "attempt": _optional_int(job.get("attempt")) or 0,
            "scope": str(payload.get("scope") or payload.get("label_scope") or payload.get("mano_scope") or ""),
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

    @staticmethod
    def _artifact_summary_text(parts: List[str]) -> str:
        if not parts:
            return ""
        return _short_summary("; ".join(parts[:5]) + (f"; +{len(parts) - 5} more" if len(parts) > 5 else ""))

    def _create_job_once(self, *, job_id: str, job_type: str, episode_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        existing = self.store.get_job(job_id)
        if existing is not None:
            return existing
        return self.store.create_job(job_id=job_id, job_type=job_type, episode_id=episode_id, payload=payload)

    def _mark_episode_for_leased_job(self, job: Dict[str, Any]) -> None:
        episode_id = str(job.get("episode_id") or "")
        if not episode_id:
            return
        if str(job.get("type") or "") == "manual_3d":
            self.store.set_episode_segments_mano_state(
                episode_id=episode_id,
                status="mano_running",
                mano_job_id=str(job.get("job_id") or ""),
            )
            self.store.update_episode_status(
                episode_id,
                "manual_3d_optimizing",
                {"active_job_id": job.get("job_id"), "manual_3d_scope": "episode"},
            )
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
        operator_id = _human_operator_id(result)
        for artifact in artifacts:
            self._register_artifact_from_payload(episode_id, artifact)

        if job_type == "upload":
            nas_uri = str(result.get("nas_uri") or result.get("episode_uri") or "")
            for artifact in artifacts:
                if str(artifact.get("kind") or "") == "nas_episode" and str(artifact.get("uri") or ""):
                    nas_uri = str(artifact.get("uri"))
                    break
            if nas_uri:
                episode = self.store.get_episode(episode_id)
                self.store.update_episode_storage(
                    episode_id,
                    episode_uri=nas_uri,
                    collection_path=str((episode or {}).get("collection_path") or payload.get("collection_path") or ""),
                    metadata={
                        "nas_uri": nas_uri,
                        "upload_job_id": job.get("job_id"),
                        "uploaded_at": result.get("completed_at") or result.get("finished_at") or "",
                    },
                )
            self.store.update_episode_status(episode_id, "uploaded")
            self._create_auto_label_jobs_from_upload(episode_id, payload, result, nas_uri)
        elif job_type == "auto_label":
            publisher_bridge_job = self._is_publisher_bridge_job(job, result)
            if (
                not publisher_bridge_job
                and not any(str(artifact.get("kind") or "") in {"pred_2d", "auto_2d"} for artifact in artifacts)
                and payload.get("episode_uri")
            ):
                self.store.register_artifact(
                    episode_id=episode_id,
                    kind="pred_2d",
                    uri=uri_join(str(payload.get("episode_uri")), "pred_2d"),
                    metadata={"source_job_id": job.get("job_id")},
                )
            if not publisher_bridge_job and not any(str(artifact.get("kind") or "") == "mano_episode" for artifact in artifacts):
                self._register_default_mano_episode(episode_id, job, result)
            if self._all_episode_jobs_succeeded(episode_id, "auto_label"):
                self.store.update_episode_status(episode_id, "mano_optimized")
                self._refresh_final_3d_sources_manifest(episode_id, "auto_label_episode_3d_completed")
                self._create_qc_job_from_existing_episode(episode_id, result, reason="auto_label_episode_3d_succeeded")
        elif job_type == "qc":
            qc_operator_metadata = (
                {"qc_operator_id": operator_id, "last_human_operator_id": operator_id}
                if operator_id
                else {}
            )
            if not any(str(artifact.get("kind") or "") == "qc_report" for artifact in artifacts):
                self._register_default_qc_report(episode_id, job, result)
            result_type = str(result.get("result_type") or result.get("qc_result_type") or "").strip().lower()
            bad_episode = (
                result_type in {"bad_episode", "abnormal_episode", "episode_abnormal", "episode_exception"}
                or _truthy_bool(result.get("bad_episode"))
                or _truthy_bool(result.get("episode_abnormal"))
                or _truthy_bool(result.get("abnormal_episode"))
            )
            if bad_episode:
                self.store.update_episode_status(
                    episode_id,
                    "qc_bad_episode",
                    {
                        "qc_status": "bad_episode",
                        "bad_episode_job_id": job.get("job_id"),
                        "bad_episode_reason": str(result.get("reason") or result.get("abnormal_reason") or "bad_episode"),
                        "nas_sync_status": "unsupported_bad_episode",
                        **qc_operator_metadata,
                    },
                )
                return
            passed = _truthy_bool(result.get("passed") if "passed" in result else result.get("qc_passed"))
            if passed:
                self.store.update_episode_status(episode_id, "finalized", {"qc_status": "passed", **qc_operator_metadata})
                self._refresh_final_3d_sources_manifest(episode_id, "qc_passed")
                self._ensure_nas_qc_completion_sync(job, result)
            else:
                self.store.update_episode_status(episode_id, "qc_failed", {"qc_status": "failed", **qc_operator_metadata})
                created = self._create_segments_from_qc_failure(episode_id, job, result)
                manual_job = self._create_manual_label_episode_job(
                    episode_id,
                    reason="qc_failed_needs_manual_label",
                )
                self.store.update_episode_status(
                    episode_id,
                    "manual_correction_pending",
                    {
                        "qc_status": "failed",
                        "failed_segment_count": len(created),
                        "manual_label_job_id": manual_job.get("job_id"),
                        "manual_label_scope": "episode",
                        **qc_operator_metadata,
                    },
                )
                self._refresh_final_3d_sources_manifest(episode_id, "qc_failed")
                self._ensure_nas_qc_completion_sync(job, result)
        elif job_type == "review":
            self.store.update_episode_status(episode_id, "review_passed")
        elif job_type == "manual_label":
            manual_uri = uri_join(
                str(payload.get("episode_uri") or ""),
                "manual_2d",
                "segments",
                str(job.get("job_id") or ""),
            )
            if not any(str(artifact.get("kind") or "") in {"manual_2d", "corrected_2d"} for artifact in artifacts) and manual_uri:
                self.store.register_artifact(
                    episode_id=episode_id,
                    kind="manual_2d",
                    uri=manual_uri,
                    metadata={"source_job_id": job.get("job_id"), "scope": "episode"},
                )
            manual_metadata = (
                {"manual_label_operator_id": operator_id, "last_human_operator_id": operator_id}
                if operator_id
                else {}
            )
            self.store.complete_episode_segments_manual(
                episode_id=episode_id,
                manual_job_id=str(job.get("job_id") or ""),
                manual_2d_uri=manual_uri,
                metadata={"operator_id": operator_id, "source": "manual_label_episode"},
            )
            self.store.update_episode_status(
                episode_id,
                "manual_labeled",
                {**manual_metadata, "manual_label_scope": "episode"},
            )
            self._create_manual_3d_episode_job(episode_id, source_job_id=str(job.get("job_id") or ""))
            self._refresh_final_3d_sources_manifest(episode_id, "manual_label_episode_completed")
        elif job_type == "manual_3d":
            self.store.set_episode_segments_mano_state(
                episode_id=episode_id,
                status="mano_succeeded",
                mano_job_id=str(job.get("job_id") or ""),
            )
            self.store.update_episode_status(
                episode_id,
                "finalized",
                {
                    "manual_3d_job_id": job.get("job_id"),
                    "manual_3d_scope": "episode",
                    "manual_3d_completion_policy": result.get("completion_policy") or "publisher_status",
                },
            )
            self._refresh_final_3d_sources_manifest(episode_id, "manual_3d_episode_completed")

    def _after_job_fail(self, job: Dict[str, Any], error: str, result: Dict[str, Any]) -> None:
        episode_id = str(job.get("episode_id") or "")
        job_type = str(job.get("type") or "")
        payload = dict(job.get("payload") or {})
        if job_type == "manual_label" and episode_id:
            self.store.reset_episode_segments_manual(episode_id=episode_id)
            self.store.update_episode_status(
                episode_id,
                "manual_label_failed",
                {"failed_job_id": job.get("job_id"), "worker_error": error},
            )
            return
        if job_type == "manual_3d" and episode_id:
            self.store.set_episode_segments_mano_state(
                episode_id=episode_id,
                status="failed",
                mano_job_id=str(job.get("job_id") or ""),
                error=error,
            )
            self.store.update_episode_status(
                episode_id,
                "manual_3d_failed",
                {"failed_job_id": job.get("job_id"), "worker_error": error},
            )
            return
        if episode_id:
            metadata = {
                "failed_job_id": job.get("job_id"),
                "failed_job_type": job_type,
                "worker_error": error,
                "cleanup_manifest": result.get("cleanup_manifest") or {},
            }
            current = self.store.get_episode(episode_id)
            if current is not None:
                self.store.update_episode_status(episode_id, str(current.get("status") or "uploaded"), metadata)

    def _cleanup_manifest_outputs(self, manifest: Dict[str, Any]) -> None:
        if not isinstance(manifest, dict) or not manifest:
            return
        candidates: List[str] = []
        for key in ("tmp_uri", "tmp_dir", "attempt_uri", "attempt_dir"):
            value = str(manifest.get(key) or "").strip()
            if value:
                candidates.append(value)
        for key in ("paths", "uris", "outputs", "attempt_outputs", "cleanup_paths"):
            value = manifest.get(key)
            if isinstance(value, list):
                candidates.extend(str(item or "").strip() for item in value if str(item or "").strip())
        for value in candidates:
            path = self._cleanup_path_from_uri(value)
            if path is None:
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except FileNotFoundError:
                continue

    def _cleanup_path_from_uri(self, value: str) -> Optional[Path]:
        value = str(value or "").strip()
        if not value:
            return None
        parsed = urlparse(value)
        if not parsed.scheme:
            return Path(value).expanduser().resolve()
        resolved = self.nas_root_dir_from_uri(value)
        if resolved:
            return resolved
        return None

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
            episode_uri=str(nas_uri or upload_result.get("nas_uri") or episode.get("episode_uri") or ""),
            source_payload=upload_payload,
            reason="upload_succeeded",
        )
        if created:
            self.store.update_episode_status(
                episode_id,
                "uploaded",
                {
                    "auto_label_after_upload": True,
                    "auto_label_job_count": len(created),
                    "auto_label_scope": "episode",
                    "episode_3d_merged_with_auto_label": True,
                },
            )

    def _push_auto_label_for_episode(self, episode: Dict[str, Any], *, pushed_by: str) -> Dict[str, Any]:
        episode_id = str(episode.get("episode_id") or "")
        status = str(episode.get("status") or "")
        episode_uri = self._auto_label_episode_uri(episode)
        base = {
            "episode_id": episode_id,
            "subject_id": str(episode.get("subject_id") or ""),
            "task_name": str(episode.get("task_name") or ""),
            "episode_index": episode.get("episode_index"),
            "status": status,
            "episode_uri": episode_uri,
            "pushed": False,
            "jobs": [],
        }
        if status not in AUTO_LABEL_PUSHABLE_STATUSES:
            return {**base, "reason": f"episode status is not pushable: {status or 'unknown'}"}
        if not episode_uri:
            return {**base, "reason": "episode has no nas_uri or episode_uri"}
        existing = self.store.jobs_for_episode(episode_id, "auto_label")
        if existing:
            return {
                **base,
                "reason": "auto_label job already exists",
                "existing_jobs": [_compact_job(job) for job in existing],
            }

        jobs = self._create_auto_label_jobs_for_episode(
            episode,
            episode_uri=episode_uri,
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
                    "auto_label_job_count": len(jobs),
                    "auto_label_scope": "episode",
                    "episode_3d_merged_with_auto_label": True,
                },
            )
        return {
            **base,
            "pushed": bool(jobs),
            "reason": "created" if jobs else "no jobs created",
            "jobs": [_compact_job(job) for job in jobs],
        }

    @staticmethod
    def _auto_label_episode_uri(episode: Dict[str, Any]) -> str:
        metadata = episode.get("metadata") if isinstance(episode.get("metadata"), dict) else {}
        nas_uri = str(metadata.get("nas_uri") or "").strip()
        return nas_uri or str(episode.get("episode_uri") or "").strip()

    def _create_auto_label_jobs_for_episode(
        self,
        episode: Dict[str, Any],
        *,
        episode_uri: str,
        source_payload: Dict[str, Any],
        reason: str,
        pushed_by: str = "",
    ) -> List[Dict[str, Any]]:
        episode_id = str(episode.get("episode_id") or "")
        episode_uri = str(episode_uri or "").strip()
        if not episode_id or not episode_uri:
            return []
        source_payload = dict(source_payload or {})
        nas_root_dir = self.nas_root_dir_from_uri(episode_uri)
        cameras = (
            _label_camera_ids(source_payload.get("cameras"))
            or _label_camera_ids(episode.get("cameras"))
            or _discover_cameras(nas_root_dir)
        )
        frames = _as_int_list(source_payload.get("frames")) or _discover_frames(nas_root_dir, cameras)
        if not frames:
            frame_count = _optional_int(episode.get("frame_count")) or _optional_int(source_payload.get("frame_count"))
            if frame_count:
                frames = list(range(frame_count))

        base_id = _stable_id_part(episode_id, "episode")
        job_id = f"auto_label_{base_id}_episode"
        payload = {
            "job_id": job_id,
            "episode_id": episode_id,
            "subject_id": episode.get("subject_id"),
            "task_name": episode.get("task_name"),
            "episode_uri": episode_uri,
            "cameras": cameras,
            "frames": frames,
            "scope": "episode",
            "label_scope": "episode",
            "mano_scope": "episode",
            "produces": ["pred_2d", "mano_episode"],
            "reason": reason,
        }
        if pushed_by:
            payload["pushed_by"] = pushed_by
        return [self._create_job_once(job_id=job_id, job_type="auto_label", episode_id=episode_id, payload=payload)]

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

    def _create_qc_job_from_existing_episode(self, episode_id: str, result: Dict[str, Any], *, reason: str = "mano_episode_succeeded") -> None:
        if self.store.jobs_for_episode(episode_id, "qc"):
            return
        episode = self.store.get_episode(episode_id)
        if episode is None:
            return
        cameras = self._cameras_for_episode_context(episode_id, episode, result)
        frames = _as_int_list(result.get("frames")) or self._frames_for_episode_context(
            episode_id,
            episode,
            cameras,
            result,
        )
        base_id = _stable_id_part(episode_id, "episode")
        job_id = str(result.get("qc_job_id") or f"qc_{base_id}")
        payload = {
            "job_id": job_id,
            "episode_id": episode["episode_id"],
            "subject_id": episode["subject_id"],
            "task_name": episode["task_name"],
            "episode_uri": episode["episode_uri"],
            "cameras": cameras,
            "frames": frames,
            "reason": reason,
        }
        self._create_job_once(job_id=job_id, job_type="qc", episode_id=episode_id, payload=payload)

    def _register_default_mano_episode(self, episode_id: str, job: Dict[str, Any], result: Dict[str, Any]) -> None:
        episode = self.store.get_episode(episode_id)
        payload = dict(job.get("payload") or {})
        episode_uri = str((episode or {}).get("episode_uri") or payload.get("episode_uri") or "").strip()
        uri = uri_join(episode_uri, "mano", "episode") if episode_uri else ""
        if not uri:
            return
        self.store.register_artifact(
            episode_id=episode_id,
            kind="mano_episode",
            uri=uri,
            metadata={"source_job_id": job.get("job_id"), "scope": "episode"},
        )

    def _register_default_qc_report(self, episode_id: str, job: Dict[str, Any], result: Dict[str, Any]) -> None:
        episode = self.store.get_episode(episode_id)
        payload = dict(job.get("payload") or {})
        episode_uri = str((episode or {}).get("episode_uri") or payload.get("episode_uri") or "").strip()
        report_uri = uri_join(episode_uri, "qc", "qc_report.json") if episode_uri else ""
        if not report_uri:
            return
        self.store.register_artifact(
            episode_id=episode_id,
            kind="qc_report",
            uri=report_uri,
            metadata={
                "source_job_id": job.get("job_id"),
                "passed": _truthy_bool(result.get("passed") if "passed" in result else result.get("qc_passed")),
                "operator_id": _human_operator_id(result),
            },
        )

    def _create_segments_from_qc_failure(
        self,
        episode_id: str,
        job: Dict[str, Any],
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        episode = self.store.get_episode(episode_id)
        if episode is None:
            return []
        payload = dict(job.get("payload") or {})
        cameras = self._cameras_for_episode_context(episode_id, episode, result, payload)
        frames = _as_int_list(result.get("frames")) or self._frames_for_episode_context(
            episode_id,
            episode,
            cameras,
            result,
            payload,
        )
        intervals = self._qc_failure_segments(result, frames)
        base_id = _stable_id_part(episode_id, "episode")
        created: List[Dict[str, Any]] = []
        for index, (start_frame, end_frame, metadata) in enumerate(intervals, 1):
            segment_id = str(metadata.get("segment_id") or metadata.get("id") or "").strip()
            if not segment_id:
                segment_id = f"segment_{base_id}_{int(start_frame):05d}_{int(end_frame):05d}_{index:03d}"
            segment = self.store.create_segment(
                segment_id=segment_id,
                episode_id=episode_id,
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                status="pending_manual",
                metadata={
                    "source_qc_job_id": job.get("job_id"),
                    "reason": metadata.get("reason") or metadata.get("label") or "qc_failed",
                    "score": metadata.get("score"),
                },
            )
            created.append(segment)
        return created

    def _create_manual_label_episode_job(self, episode_id: str, *, reason: str) -> Dict[str, Any]:
        existing = self.store.jobs_for_episode(episode_id, "manual_label")
        if existing:
            return existing[-1]
        episode = self.store.get_episode(episode_id)
        if episode is None:
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
        segments = self.store.segments_for_episode(episode_id)
        if not segments:
            raise WorkflowError(HTTPStatus.CONFLICT, f"episode has no failed segments: {episode_id}")
        frames = sorted(
            {
                frame
                for segment in segments
                for frame in range(int(segment.get("start_frame") or 0), int(segment.get("end_frame") or 0) + 1)
            }
        )
        job_id = f"manual_label_{_stable_id_part(episode_id, 'episode')}"
        payload = {
            "job_id": job_id,
            "episode_id": episode_id,
            "subject_id": episode.get("subject_id"),
            "task_name": episode.get("task_name"),
            "episode_uri": episode.get("episode_uri"),
            "cameras": self._cameras_for_episode_context(episode_id, episode, {}),
            "frames": frames,
            "segment_ids": [str(segment.get("segment_id") or "") for segment in segments],
            "scope": "episode",
            "label_scope": "episode",
            "reason": reason,
        }
        return self._create_job_once(job_id=job_id, job_type="manual_label", episode_id=episode_id, payload=payload)

    def _create_manual_3d_episode_job(self, episode_id: str, *, source_job_id: str) -> Dict[str, Any]:
        existing = self.store.jobs_for_episode(episode_id, "manual_3d")
        if existing:
            return existing[-1]
        episode = self.store.get_episode(episode_id)
        if episode is None:
            raise WorkflowError(HTTPStatus.NOT_FOUND, f"episode not found: {episode_id}")
        segments = self.store.segments_for_episode(episode_id)
        if not segments or any(not str(segment.get("manual_completed_at") or "") for segment in segments):
            raise WorkflowError(HTTPStatus.CONFLICT, f"episode manual labeling is incomplete: {episode_id}")
        frames = self._frames_for_episode_context(
            episode_id,
            episode,
            self._cameras_for_episode_context(episode_id, episode, {}),
            {},
        )
        job_id = f"manual_3d_{_stable_id_part(episode_id, 'episode')}"
        payload = {
            "job_id": job_id,
            "episode_id": episode_id,
            "subject_id": episode.get("subject_id"),
            "task_name": episode.get("task_name"),
            "episode_uri": episode.get("episode_uri"),
            "cameras": self._cameras_for_episode_context(episode_id, episode, {}),
            "frames": frames,
            "segment_ids": [str(segment.get("segment_id") or "") for segment in segments],
            "scope": "episode",
            "mano_scope": "episode",
            "source_manual_label_job_id": source_job_id,
            "reason": "manual_2d_episode_completed",
        }
        job = self._create_job_once(job_id=job_id, job_type="manual_3d", episode_id=episode_id, payload=payload)
        self.store.set_episode_segments_mano_state(
            episode_id=episode_id,
            status="mano_queued",
            mano_job_id=job_id,
        )
        self.store.update_episode_status(
            episode_id,
            "manual_3d_pending",
            {"manual_3d_job_id": job_id, "manual_3d_scope": "episode"},
        )
        return job

    def _qc_failure_segments(self, result: Dict[str, Any], frames: Sequence[int]) -> List[Tuple[int, int, Dict[str, Any]]]:
        raw = None
        for key in ("segments", "failed_segments", "qc_failed_segments", "failure_segments"):
            if isinstance(result.get(key), list):
                raw = result.get(key)
                break
        intervals: List[Tuple[int, int, Dict[str, Any]]] = []
        if isinstance(raw, list):
            for item in raw:
                metadata: Dict[str, Any] = {}
                start_value: Any = None
                end_value: Any = None
                if isinstance(item, dict):
                    metadata = dict(item)
                    start_value = item.get("start_frame", item.get("start", item.get("first_frame")))
                    end_value = item.get("end_frame", item.get("end", item.get("last_frame", start_value)))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    start_value, end_value = item[0], item[1]
                if isinstance(start_value, bool) or isinstance(end_value, bool):
                    continue
                try:
                    start_frame = int(start_value)
                    end_frame = int(end_value)
                except (TypeError, ValueError):
                    continue
                if end_frame < start_frame:
                    start_frame, end_frame = end_frame, start_frame
                intervals.append((start_frame, end_frame, metadata))
        if intervals:
            intervals.sort(key=lambda item: (item[0], item[1]))
            return intervals
        clean_frames = [int(frame) for frame in frames if not isinstance(frame, bool)]
        if clean_frames:
            return [(min(clean_frames), max(clean_frames), {"reason": "qc_failed_without_segments"})]
        return [(0, 0, {"reason": "qc_failed_without_segments"})]

    def _refresh_final_3d_sources_manifest(self, episode_id: str, reason: str) -> None:
        episode_id = str(episode_id or "").strip()
        if not episode_id:
            return
        try:
            self._write_final_3d_sources_manifest(episode_id, reason)
        except Exception as exc:  # pragma: no cover - manifest failures should not strand completed jobs
            print(f"[workflow] failed to update final 3d sources manifest for {episode_id}: {exc}", flush=True)

    def _ensure_nas_qc_completion_sync(self, job: Dict[str, Any], result: Dict[str, Any]) -> None:
        episode_id = str(job.get("episode_id") or "").strip()
        if not episode_id:
            return
        # Ensure the NAS sees the takeover before any result, including when a
        # process stopped in the narrow gap after the local lease committed.
        self._enqueue_nas_sync(
            episode_id=episode_id,
            action=NAS_SYNC_QUALITY_TAKE,
            event_id=f"nas_quality_take:{episode_id}",
        )
        result_type = str(result.get("result_type") or result.get("qc_result_type") or "").strip().lower()
        bad_episode = (
            result_type in {"bad_episode", "abnormal_episode", "episode_abnormal", "episode_exception"}
            or _truthy_bool(result.get("bad_episode"))
            or _truthy_bool(result.get("episode_abnormal"))
            or _truthy_bool(result.get("abnormal_episode"))
        )
        if bad_episode:
            return
        passed = _truthy_bool(result.get("passed") if "passed" in result else result.get("qc_passed"))
        self._enqueue_nas_sync(
            episode_id=episode_id,
            action=NAS_SYNC_QUALITY_PASSED if passed else NAS_SYNC_QUALITY_NEEDS_LABELING,
            event_id=f"nas_qc_result:{job.get('job_id')}",
        )

    def _enqueue_nas_sync(self, *, episode_id: str, action: str, event_id: str) -> None:
        episode = self.store.get_episode(episode_id)
        if episode is None:
            return
        nas_episode_id = self._nas_episode_id(episode)
        self.store.enqueue_nas_sync_event(
            event_id=event_id,
            episode_id=episode_id,
            nas_episode_id=nas_episode_id,
            action=action,
        )

    @staticmethod
    def _nas_episode_id(episode: Mapping[str, Any]) -> str:
        episode_uri = str(episode.get("episode_uri") or "").strip()
        if episode_uri:
            parts = [unquote(part) for part in urlparse(episode_uri).path.split("/") if part]
            if len(parts) == 3 and all(part not in {".", ".."} and "/" not in part for part in parts):
                return "/".join(parts)
        parts = [
            str(episode.get("subject_id") or "").strip(),
            str(episode.get("task_name") or "").strip(),
            str(episode.get("storage_name") or "").strip(),
        ]
        if all(parts) and all(part not in {".", ".."} and "/" not in part for part in parts):
            return "/".join(parts)
        raise WorkflowError(
            HTTPStatus.CONFLICT,
            f"episode has no valid three-part NAS path: {episode.get('episode_id')}",
        )

    def _write_final_3d_sources_manifest(self, episode_id: str, reason: str) -> None:
        episode = self.store.get_episode(episode_id)
        if episode is None:
            return
        episode_uri = str(episode.get("episode_uri") or "").strip().rstrip("/")
        if not episode_uri:
            return
        nas_root_dir = self.nas_root_dir_from_uri(episode_uri)
        if not nas_root_dir:
            return

        segments = self.store.segments_for_episode(episode_id)
        metadata = episode.get("metadata") if isinstance(episode.get("metadata"), dict) else {}
        mano_episode_uri = uri_join(episode_uri, "mano", "episode")
        qc_report_uri = uri_join(episode_uri, "qc", "qc_report.json")
        pred_uri = uri_join(episode_uri, "pred_2d")

        manual_segments = []
        for segment in segments:
            segment_id = str(segment.get("segment_id") or "")
            start_frame = _optional_int(segment.get("start_frame")) or 0
            end_frame = _optional_int(segment.get("end_frame")) or start_frame
            status = str(segment.get("status") or "")
            manual_2d_uri = str(segment.get("manual_2d_uri") or "")
            manual_segments.append(
                {
                    "segment_id": segment_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "source": "manual_label_episode",
                    "status": status,
                    "segment_status": status,
                    "manual_2d_uri": manual_2d_uri,
                    "manual_2d_relative_path": self._relative_episode_uri_path(episode_uri, manual_2d_uri),
                }
            )

        manual_3d_jobs = self.store.jobs_for_episode(episode_id, "manual_3d")
        manual_3d_ready = bool(manual_3d_jobs and str(manual_3d_jobs[-1].get("status") or "") == "succeeded")

        manifest = {
            "schema_version": 2,
            "kind": "orbbec_final_3d_sources",
            "episode_id": episode_id,
            "subject_id": str(episode.get("subject_id") or ""),
            "task_name": str(episode.get("task_name") or ""),
            "episode_index": episode.get("episode_index"),
            "episode_status": str(episode.get("status") or ""),
            "episode_uri": episode_uri,
            "manifest_uri": uri_join(episode_uri, FINAL_3D_SOURCES_REL_PATH),
            "updated_at": now_iso(),
            "updated_reason": str(reason or ""),
            "base_2d": {
                "source": "manual_label_episode" if segments else "auto_label",
                "uri": uri_join(episode_uri, "manual_2d") if segments else pred_uri,
                "relative_path": "manual_2d" if segments else self._relative_episode_uri_path(episode_uri, pred_uri),
            },
            "base_3d": {
                "source": "manual_3d_episode" if manual_3d_ready else "auto_label",
                "uri": mano_episode_uri,
                "relative_path": self._relative_episode_uri_path(episode_uri, mano_episode_uri),
                "status": "ready" if mano_episode_uri else "missing",
            },
            "qc": {
                "status": str(metadata.get("qc_status") or ""),
                "report_uri": qc_report_uri,
                "relative_path": self._relative_episode_uri_path(episode_uri, qc_report_uri),
            },
            "manual_segments": manual_segments,
            "reconstruction_rule": {
                "default": "base_3d",
                "scope": "whole_episode",
                "segment_patches": False,
            },
        }

        manifest_path = Path(nas_root_dir).expanduser().resolve() / FINAL_3D_SOURCES_REL_PATH
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(manifest_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _relative_episode_uri_path(episode_uri: str, uri: str) -> str:
        episode_uri = str(episode_uri or "").strip().rstrip("/")
        uri = str(uri or "").strip().rstrip("/")
        if not episode_uri or not uri:
            return ""
        if uri == episode_uri:
            return "."
        prefix = episode_uri + "/"
        if uri.startswith(prefix):
            return uri[len(prefix):]
        return ""

    @staticmethod
    def _is_publisher_bridge_job(job: Dict[str, Any], result: Dict[str, Any]) -> bool:
        worker_id = str(result.get("worker_id") or job.get("lease_owner") or "").strip()
        return worker_id.startswith("publisher_bridge:")

    @staticmethod
    def _npy_header(path: Path) -> Tuple[Tuple[int, ...], str, bool]:
        """Read an NPY shape/dtype using only the backend's standard library."""
        try:
            with path.open("rb") as handle:
                if handle.read(6) != b"\x93NUMPY":
                    raise ValueError("missing NPY magic")
                version = handle.read(2)
                if len(version) != 2:
                    raise ValueError("truncated NPY version")
                if version[0] == 1:
                    raw_length = handle.read(2)
                    if len(raw_length) != 2:
                        raise ValueError("truncated NPY header length")
                    header_length = struct.unpack("<H", raw_length)[0]
                elif version[0] in {2, 3}:
                    raw_length = handle.read(4)
                    if len(raw_length) != 4:
                        raise ValueError("truncated NPY header length")
                    header_length = struct.unpack("<I", raw_length)[0]
                else:
                    raise ValueError(f"unsupported NPY version {version[0]}.{version[1]}")
                raw_header = handle.read(header_length)
            encoding = "utf-8" if version[0] == 3 else "latin1"
            header = ast.literal_eval(raw_header.decode(encoding).strip())
            if not isinstance(header, dict):
                raise ValueError("NPY header is not a mapping")
            shape_value = header.get("shape")
            if not isinstance(shape_value, tuple):
                raise ValueError("NPY shape is not a tuple")
            shape = tuple(int(value) for value in shape_value)
            return shape, str(header.get("descr") or ""), bool(header.get("fortran_order"))
        except (OSError, UnicodeError, ValueError, SyntaxError, struct.error) as exc:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, f"invalid NPY artifact {path}: {exc}") from exc

    def _validate_publisher_bridge_completion(
        self,
        job: Dict[str, Any],
        result: Dict[str, Any],
        artifacts: List[Dict[str, Any]],
    ) -> None:
        kinds = {str(artifact.get("kind") or "").strip() for artifact in artifacts}
        missing = {"optimized_pose", "mano_episode"} - kinds
        if missing:
            raise WorkflowError(
                HTTPStatus.BAD_REQUEST,
                "Publisher Bridge completion is missing artifacts: " + ", ".join(sorted(missing)),
            )

        payload = dict(job.get("payload") or {})
        episode = self.store.get_episode(str(job.get("episode_id") or "")) or {}
        episode_uri = str(episode.get("episode_uri") or payload.get("episode_uri") or "").strip()
        episode_dir = self.nas_root_dir_from_uri(episode_uri)
        if episode_dir is None:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "Publisher Bridge job has no resolvable NAS episode URI")

        optimized_pose_dir = episode_dir / "optimized_pose"
        if not optimized_pose_dir.is_dir():
            raise WorkflowError(HTTPStatus.BAD_REQUEST, f"optimized_pose artifact is missing: {optimized_pose_dir}")
        pose_files: List[Tuple[int, Path]] = []
        seen_frames = set()
        for pose_path in optimized_pose_dir.glob("*.npy"):
            if not pose_path.stem.isdigit():
                continue
            frame = int(pose_path.stem)
            if frame in seen_frames:
                raise WorkflowError(HTTPStatus.BAD_REQUEST, f"duplicate optimized_pose frame: {frame}")
            seen_frames.add(frame)
            shape, descr, fortran_order = self._npy_header(pose_path)
            if shape != (2, 99) or descr not in {"<f4", ">f4", "=f4", "|f4"} or fortran_order:
                raise WorkflowError(
                    HTTPStatus.BAD_REQUEST,
                    f"optimized_pose frame must be C-order float32 (2,99), got shape={shape} dtype={descr}: {pose_path}",
                )
            pose_files.append((frame, pose_path))
        pose_files.sort(key=lambda value: value[0])
        pose_frames = [frame for frame, _ in pose_files]
        if not pose_frames:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, f"optimized_pose has no numeric frames: {optimized_pose_dir}")

        frames = _as_int_list(result.get("frames"))
        if frames != pose_frames:
            raise WorkflowError(
                HTTPStatus.BAD_REQUEST,
                f"Publisher Bridge result frames do not match optimized_pose files: result={frames}, files={pose_frames}",
            )
        generation = _optional_int(result.get("generation")) or 0
        result_sha256 = str(result.get("result_manifest_sha256") or "").strip()
        if generation <= 0 or len(result_sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in result_sha256):
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "Publisher Bridge result generation/hash is invalid")

        mano_dir = episode_dir / "mano" / "episode"
        meta_path = mano_dir / "mano_episode.json"
        joints_path = mano_dir / "joints_3d.npy"
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, f"invalid MANO completion marker {meta_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise WorkflowError(HTTPStatus.BAD_REQUEST, f"MANO completion marker must be an object: {meta_path}")
        source = metadata.get("source")
        if not isinstance(source, dict):
            raise WorkflowError(HTTPStatus.BAD_REQUEST, f"MANO completion marker is missing source metadata: {meta_path}")
        metadata_frames = _as_int_list(metadata.get("frames"))
        expected_metadata = (
            int(metadata.get("schema_version") or 0) == 1
            and str(metadata.get("kind") or "") == "orbbec_mano_3d_episode"
            and str(metadata.get("coordinate_system") or "") == "episode_world"
            and str(metadata.get("joints_3d_file") or "") == "joints_3d.npy"
            and metadata_frames == frames
            and str(source.get("kind") or "") == "optimized_pose"
            and _as_int_list(source.get("shape")) == [2, 99]
            and int(source.get("generation") or 0) == generation
            and str(source.get("result_manifest_sha256") or "") == result_sha256
        )
        if not expected_metadata:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, f"MANO completion marker does not match Publisher result: {meta_path}")
        joints_shape, joints_descr, joints_fortran = self._npy_header(joints_path)
        expected_joints_shape = (len(frames), 2, 21, 3)
        if joints_shape != expected_joints_shape or joints_descr not in {"<f4", ">f4", "=f4", "|f4"} or joints_fortran:
            raise WorkflowError(
                HTTPStatus.BAD_REQUEST,
                f"joints_3d must be C-order float32 {expected_joints_shape}, got shape={joints_shape} dtype={joints_descr}",
            )
        if _as_int_list(result.get("joints_3d_shape")) != list(expected_joints_shape):
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "Publisher Bridge result joints_3d_shape does not match artifact")

    def _register_artifact_from_payload(self, episode_id: str, artifact: Dict[str, Any]) -> None:
        kind = str(artifact.get("kind") or "").strip()
        episode = self.store.get_episode(episode_id) or {}
        episode_uri = str(episode.get("episode_uri") or "").strip()
        if kind == "nas_episode":
            uri = str(artifact.get("uri") or "").strip()
            if not uri.startswith("nas://"):
                uri = ""
        else:
            uri = self._fixed_artifact_uri(episode_uri, kind, dict(artifact.get("metadata") or {}))
        if not kind or not uri:
            raise WorkflowError(HTTPStatus.BAD_REQUEST, "artifact kind and episode_uri are required")
        metadata = json_object(artifact.get("metadata"), "artifact.metadata")
        source_job_id = str(metadata.get("source_job_id") or "").strip()
        for existing in self.store.artifacts_for_episode(episode_id):
            existing_metadata = existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}
            same_source = source_job_id and str(existing_metadata.get("source_job_id") or "") == source_job_id
            legacy_same_uri = not source_job_id and str(existing.get("uri") or "") == uri
            if str(existing.get("kind") or "") == kind and (same_source or legacy_same_uri):
                return
        self.store.register_artifact(
            episode_id=episode_id,
            kind=kind,
            uri=uri,
            metadata=metadata,
            artifact_id=str(artifact.get("artifact_id") or "").strip() or None,
        )

    @staticmethod
    def _fixed_artifact_uri(episode_uri: str, kind: str, metadata: Dict[str, Any]) -> str:
        episode_uri = str(episode_uri or "").strip().rstrip("/")
        kind = str(kind or "").strip()
        if kind == "nas_episode":
            return episode_uri
        if not episode_uri:
            return ""
        if kind in {"pred_2d", "auto_2d"}:
            return uri_join(episode_uri, "pred_2d")
        if kind == "optimized_pose":
            return uri_join(episode_uri, "optimized_pose")
        if kind == "mano_episode":
            return uri_join(episode_uri, "mano", "episode")
        if kind == "qc_report":
            return uri_join(episode_uri, "qc", "qc_report.json")
        if kind in {"manual_2d", "corrected_2d"}:
            return uri_join(episode_uri, "manual_2d")
        return ""

    @staticmethod
    def _artifacts_from_body(body: Dict[str, Any]) -> List[Dict[str, Any]]:
        artifacts = body.get("artifacts")
        if isinstance(artifacts, list):
            return [dict(item) for item in artifacts if isinstance(item, dict)]
        artifact = body.get("artifact")
        if isinstance(artifact, dict):
            return [dict(artifact)]
        return []
