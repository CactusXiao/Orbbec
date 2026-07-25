from __future__ import annotations

import re
import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .storage_resolver import local_uri_from_path, path_from_local_uri, uri_join
    from .workflow_models import WorkflowError, json_object, require_job_type
    from .workflow_store import WorkflowStore
except ImportError:  # pragma: no cover - script execution fallback
    from storage_resolver import local_uri_from_path, path_from_local_uri, uri_join  # type: ignore
    from workflow_models import WorkflowError, json_object, require_job_type  # type: ignore
    from workflow_store import WorkflowStore  # type: ignore


_FRAME_RE = re.compile(r"^(\d+)\.[^.]+$")


def _new_id(prefix: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(prefix or "item").strip().lower()).strip("_")
    return f"{clean or 'item'}_{uuid.uuid4().hex[:12]}"


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


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
    def __init__(self, store: WorkflowStore):
        self.store = store

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
        owner = str(body.get("lease_owner") or body.get("operator_id") or body.get("worker_id") or "").strip()
        lease_seconds = _optional_int(body.get("lease_seconds")) or 300
        job = self.store.lease_job(job_type=job_type, lease_owner=owner, lease_seconds=lease_seconds)
        if job is None:
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
        jobs = self.store.jobs_for_episode(episode_id, "upload")
        upload_job = jobs[-1] if jobs else None
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
        return {
            "episode": episode,
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
        return {
            "job": job,
            "episode": episode,
            "artifacts": artifacts,
            "payload": payload,
        }

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
        elif job_type == "auto_label":
            if not artifacts and payload.get("data_uri"):
                self.store.register_artifact(
                    episode_id=episode_id,
                    kind="pred_2d",
                    uri=uri_join(str(payload.get("data_uri")), str(payload.get("prediction_dir") or "pred_2d")),
                    metadata={"source_job_id": job.get("job_id"), "compatibility": "kp2d may map to pred_2d"},
                )
            self.store.update_episode_status(episode_id, "auto_labeled")
        elif job_type == "qc":
            passed = bool(result.get("passed") or result.get("qc_passed"))
            self.store.update_episode_status(episode_id, "qc_passed" if passed else "qc_failed")
            if not passed and result.get("create_manual_label_job", True):
                self._create_manual_label_from_existing_episode(episode_id, result)
        elif job_type == "review":
            self.store.update_episode_status(episode_id, "review_passed")
        elif job_type == "manual_label":
            if not artifacts and payload.get("data_uri"):
                self.store.register_artifact(
                    episode_id=episode_id,
                    kind="corrected_2d",
                    uri=uri_join(str(payload.get("data_uri")), str(payload.get("correction_dir") or "corrected_2d")),
                    metadata={"source_job_id": job.get("job_id")},
                )
            self.store.update_episode_status(episode_id, "manual_labeled")

    def _create_manual_label_from_existing_episode(self, episode_id: str, result: Dict[str, Any]) -> None:
        episode = self.store.get_episode(episode_id)
        if episode is None:
            return
        job_id = str(result.get("manual_label_job_id") or _new_id("manual_label"))
        payload = {
            "job_id": job_id,
            "episode_id": episode["episode_id"],
            "subject_id": episode["subject_id"],
            "task_name": episode["task_name"],
            "data_uri": episode["data_uri"],
            "cameras": episode.get("cameras") or [],
            "frames": result.get("frames") if isinstance(result.get("frames"), list) else [],
            "rgb_path_template": "{camera}/RGB/{frame:05d}.png",
            "prediction_dir": "pred_2d",
            "correction_dir": "corrected_2d",
            "reason": "qc_stub_failed",
            "priority": _optional_int(result.get("priority")) or 50,
        }
        self.store.update_episode_status(episode_id, "manual_label_pending")
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
