from __future__ import annotations

from http import HTTPStatus
from typing import Any, Dict


EPISODE_STATUSES = {
    "planned",
    "reserved_for_collection",
    "captured",
    "uploaded",
    "auto_labeling",
    "auto_labeled",
    "mano_optimizing",
    "mano_optimized",
    "qc_running",
    "qc_passed",
    "qc_failed",
    "qc_bad_episode",
    "review_pending",
    "review_passed",
    "manual_label_pending",
    "manual_labeling",
    "manual_labeled",
    "manual_correction_pending",
    "manual_correction_running",
    "manual_3d_pending",
    "manual_3d_optimizing",
    "manual_label_failed",
    "manual_3d_failed",
    "finalized",
}

JOB_TYPES = {
    "upload",
    "auto_label",
    "qc",
    "review",
    "manual_label",
    "manual_3d",
}

CONTROLLED_STAGE_JOB_TYPES = {
    "auto_label",
    "qc",
    "manual_label",
    "manual_3d",
}

SEGMENT_STATUSES = {
    "pending_manual",
    "manual_labeling",
    "manual_labeled",
    "mano_queued",
    "mano_running",
    "mano_succeeded",
    "failed",
}

JOB_STATUSES = {
    "queued",
    "leased",
    "running",
    "succeeded",
    "failed",
    "released",
    "canceled",
}

TERMINAL_JOB_STATUSES = {"succeeded", "failed", "canceled"}


class WorkflowError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def require_job_type(value: str) -> str:
    value = str(value or "").strip()
    if value not in JOB_TYPES:
        raise WorkflowError(HTTPStatus.BAD_REQUEST, f"unsupported job type: {value}")
    return value


def require_stage_job_type(value: str) -> str:
    value = str(value or "").strip()
    if value not in CONTROLLED_STAGE_JOB_TYPES:
        raise WorkflowError(HTTPStatus.BAD_REQUEST, f"unsupported workflow stage: {value}")
    return value


def require_job_status(value: str) -> str:
    value = str(value or "").strip()
    if value not in JOB_STATUSES:
        raise WorkflowError(HTTPStatus.BAD_REQUEST, f"unsupported job status: {value}")
    return value


def require_episode_status(value: str) -> str:
    value = str(value or "").strip()
    if value not in EPISODE_STATUSES:
        raise WorkflowError(HTTPStatus.BAD_REQUEST, f"unsupported episode status: {value}")
    return value


def require_segment_status(value: str) -> str:
    value = str(value or "").strip()
    if value not in SEGMENT_STATUSES:
        raise WorkflowError(HTTPStatus.BAD_REQUEST, f"unsupported segment status: {value}")
    return value


def json_object(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise WorkflowError(HTTPStatus.BAD_REQUEST, f"{field_name} must be an object")
    return value


SHAPE_CALIBRATION_TASK = "task_handshapeCalibration"


def is_shape_calibration_episode(episode: Dict[str, Any]) -> bool:
    metadata = episode.get("metadata") or {}
    return (episode.get("task_name") == SHAPE_CALIBRATION_TASK
            or metadata.get("shape_calibration") is True)
