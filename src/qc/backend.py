from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class QcBackendError(Exception):
    pass


def episode_display_id(*sources: Mapping[str, Any]) -> str:
    """Return the human-facing episode number without exposing the backend UUID."""
    for source in sources:
        value = source.get("episode_index")
        if value is not None and str(value).strip():
            return str(value).strip()
    for source in sources:
        storage_name = str(source.get("storage_name") or "").strip()
        if storage_name:
            return storage_name
    return "-"


class QcBackendClient:
    def __init__(self, backend_url: str, *, timeout_seconds: float = 10.0):
        self.backend_url = (backend_url or "http://127.0.0.1:8765").strip().rstrip("/")
        self.timeout_seconds = float(timeout_seconds or 10.0)

    def workflow_stage(self) -> Dict[str, Any]:
        return self._get("/api/v1/workflow/stages/qc")

    def available_qc_items(self) -> List[Dict[str, Any]]:
        stage = self.workflow_stage()
        items: List[Dict[str, Any]] = []
        queued = stage.get("queued")
        if isinstance(queued, list):
            items.extend(dict(item) for item in queued if isinstance(item, dict))
        active = stage.get("active")
        if isinstance(active, list):
            for item in active:
                if isinstance(item, dict) and item.get("lease_expired"):
                    clone = dict(item)
                    clone["status"] = "queued"
                    clone["was_expired_lease"] = True
                    items.append(clone)
        return items

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self._get(f"/api/v1/jobs/{quote(str(job_id or ''), safe='')}")

    def lease_qc_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        task_name: str = "",
        episode_id: str = "",
        job_id: str = "",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": "qc",
            "worker_id": str(worker_id),
            "lease_seconds": int(lease_seconds),
        }
        if task_name:
            payload["task_name"] = task_name
        if episode_id:
            payload["episode_id"] = episode_id
        if job_id:
            payload["job_id"] = job_id
        return self._post("/api/v1/jobs/lease", payload)

    def heartbeat_qc_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
        status: str = "running",
    ) -> Dict[str, Any]:
        return self._post(
            f"/api/v1/jobs/{quote(str(job_id or ''), safe='')}/heartbeat",
            {"worker_id": str(worker_id), "lease_seconds": int(lease_seconds), "status": status},
        )

    def complete_qc_job(
        self,
        job_id: str,
        *,
        result: Dict[str, Any],
        artifacts: Optional[List[Dict[str, Any]]] = None,
        operator_id: str = "",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"result": dict(result or {})}
        if operator_id:
            payload["operator_id"] = str(operator_id)
        if artifacts:
            payload["artifacts"] = artifacts
        return self._post(f"/api/v1/jobs/{quote(str(job_id or ''), safe='')}/complete", payload)

    def release_qc_job(self, job_id: str, *, reason: str = "") -> Dict[str, Any]:
        return self._post(f"/api/v1/jobs/{quote(str(job_id or ''), safe='')}/release", {"reason": reason})

    def _get(self, path: str) -> Dict[str, Any]:
        return self._request("GET", path, None)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", path, payload)

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(self.backend_url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as exc:
            raise QcBackendError(self._http_error_message(exc)) from exc
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise QcBackendError(f"Cannot reach backend at {self.backend_url}: {reason}") from exc
        except TimeoutError as exc:
            raise QcBackendError(f"Backend request timed out: {self.backend_url}") from exc
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise QcBackendError("Backend returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise QcBackendError("Backend JSON response must be an object.")
        if "error" in parsed:
            raise QcBackendError(str(parsed.get("error") or "Backend returned an error."))
        return parsed

    @staticmethod
    def _http_error_message(exc: HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8")
            parsed = json.loads(raw or "{}")
            if isinstance(parsed, dict) and parsed.get("error"):
                return str(parsed["error"])
        except Exception:
            pass
        return f"Backend HTTP {exc.code}: {exc.reason}"


def group_qc_items_by_task(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for item in items:
        task_name = str(item.get("task_name") or "Unspecified").strip() or "Unspecified"
        group = groups.setdefault(
            task_name,
            {
                "task_name": task_name,
                "queued": 0,
                "frames": 0,
                "oldest_created_at": "",
                "episodes": [],
            },
        )
        group["queued"] += 1
        try:
            group["frames"] += max(0, int(item.get("frames_count") or len(item.get("frames") or [])))
        except (TypeError, ValueError):
            pass
        created_at = str(item.get("created_at") or "")
        if created_at and (not group["oldest_created_at"] or created_at < group["oldest_created_at"]):
            group["oldest_created_at"] = created_at
        group["episodes"].append(item)
    return sorted(groups.values(), key=lambda item: (str(item.get("task_name") or ""), str(item.get("oldest_created_at") or "")))
