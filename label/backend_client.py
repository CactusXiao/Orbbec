from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen


class BackendClientError(Exception):
    pass


@dataclass(frozen=True)
class LabelJobSession:
    client: "LabelBackendClient"
    operator_id: str
    job_id: str
    job: Dict[str, Any]
    payload: Dict[str, Any]
    mounts: Dict[str, str]
    episode_id: str = ""


class NasEpisodeResolver:
    def __init__(self, mounts: Optional[Mapping[str, str]] = None):
        self.mounts = {str(k).rstrip("/"): str(v) for k, v in (mounts or {}).items() if str(k).strip()}

    def resolve(self, episode_uri: str) -> Path:
        value = str(episode_uri or "").strip()
        if not value:
            raise ValueError("Episode URI is empty.")

        parsed = urlparse(value)
        if parsed.scheme != "nas":
            raise ValueError(f"Episode URI must be a nas:// URI: {value}")

        base = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        best_prefix = ""
        best_root = ""
        for prefix, root in self.mounts.items():
            if value == prefix or value.startswith(prefix + "/"):
                if len(prefix) > len(best_prefix):
                    best_prefix = prefix
                    best_root = root
            elif base == prefix:
                if len(prefix) > len(best_prefix):
                    best_prefix = prefix
                    best_root = root
        if best_prefix:
            suffix = value[len(best_prefix):].lstrip("/")
            return (Path(best_root).expanduser() / unquote(suffix)).resolve()
        raise ValueError(f"No NAS mount mapping is configured for URI: {value}")


class LabelBackendClient:
    def __init__(self, backend_url: str, *, timeout_seconds: float = 10.0):
        self.backend_url = (backend_url or "http://127.0.0.1:8765").strip().rstrip("/")
        self.timeout_seconds = float(timeout_seconds or 10.0)

    def create_dev_label_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/api/v1/dev/label/jobs", payload)

    def lease_label_episode(
        self,
        operator_id: str,
        *,
        lease_seconds: int = 600,
        task_name: str = "",
        episode_id: str = "",
    ) -> Dict[str, Any]:
        payload = {"operator_id": operator_id, "lease_seconds": int(lease_seconds)}
        task_name = str(task_name or "").strip()
        if task_name:
            payload["task_name"] = task_name
        episode_id = str(episode_id or "").strip()
        if episode_id:
            payload["episode_id"] = episode_id
        return self._post(
            "/api/v1/label/episodes/lease",
            payload,
        )

    def lease_label_job(
        self,
        operator_id: str,
        *,
        lease_seconds: int = 600,
        task_name: str = "",
        episode_id: str = "",
    ) -> Dict[str, Any]:
        return self.lease_label_episode(
            operator_id,
            lease_seconds=lease_seconds,
            task_name=task_name,
            episode_id=episode_id,
        )

    def manual_label_queue(self) -> Dict[str, Any]:
        return self._get("/api/v1/label/tasks")

    def queued_label_tasks(self) -> List[Dict[str, Any]]:
        response = self.manual_label_queue()
        tasks = response.get("tasks")
        if isinstance(tasks, list):
            return [dict(item) for item in tasks if isinstance(item, dict)]
        return []

    def label_task_episodes(self, task_name: str) -> List[Dict[str, Any]]:
        response = self._get(f"/api/v1/label/tasks/{quote(str(task_name or ''), safe='')}/episodes")
        episodes = response.get("episodes")
        if isinstance(episodes, list):
            return [dict(item) for item in episodes if isinstance(item, dict)]
        return []

    def get_label_episode(self, episode_id: str) -> Dict[str, Any]:
        return self._get(f"/api/v1/label/episodes/{episode_id}")

    def heartbeat_label_job(self, job_id: str, operator_id: str, *, lease_seconds: int = 600) -> Dict[str, Any]:
        return self._post(
            f"/api/v1/label/episodes/{job_id}/heartbeat",
            {"operator_id": operator_id, "lease_seconds": int(lease_seconds), "status": "running"},
        )

    def complete_label_job(
        self,
        job_id: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        artifacts: Optional[list] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"result": result or {}}
        if artifacts:
            payload["artifacts"] = artifacts
        return self._post(f"/api/v1/label/episodes/{job_id}/complete", payload)

    def release_label_job(self, job_id: str, *, reason: str = "") -> Dict[str, Any]:
        return self._post(f"/api/v1/label/episodes/{job_id}/release", {"reason": reason})

    def fail_label_job(
        self,
        job_id: str,
        *,
        error: str,
        result: Optional[Dict[str, Any]] = None,
        cleanup_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"error": error, "result": result or {}}
        if cleanup_manifest:
            payload["cleanup_manifest"] = cleanup_manifest
        return self._post(f"/api/v1/label/episodes/{job_id}/fail", payload)

    def _get(self, path: str) -> Dict[str, Any]:
        return self._request("GET", path, None)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", path, payload)

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        url = self.backend_url + path
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as exc:
            raise BackendClientError(self._http_error_message(exc)) from exc
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise BackendClientError(f"Cannot reach backend at {self.backend_url}: {reason}") from exc
        except TimeoutError as exc:
            raise BackendClientError(f"Backend request timed out: {self.backend_url}") from exc

        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise BackendClientError("Backend returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise BackendClientError("Backend JSON response must be an object.")
        if "error" in parsed:
            raise BackendClientError(str(parsed.get("error") or "Backend returned an error."))
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


def parse_mounts_json(text: str) -> Dict[str, str]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Mount mapping must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Mount mapping must be a JSON object.")
    out: Dict[str, str] = {}
    for key, value in parsed.items():
        prefix = str(key or "").strip().rstrip("/")
        root = str(value or "").strip()
        if prefix and root:
            out[prefix] = root
    return out


def grouped_label_tasks(queued_jobs: Any) -> List[Dict[str, Any]]:
    if not isinstance(queued_jobs, list):
        return []
    groups: Dict[str, Dict[str, Any]] = {}
    subject_sets: Dict[str, Set[str]] = {}
    for item in queued_jobs:
        if not isinstance(item, dict):
            continue
        task_name = str(item.get("task_name") or "Unspecified").strip() or "Unspecified"
        group = groups.setdefault(
            task_name,
            {
                "task_name": task_name,
                "queued": 0,
                "frames": 0,
                "oldest_created_at": "",
                "jobs": [],
            },
        )
        subjects = subject_sets.setdefault(task_name, set())
        group["queued"] += 1
        frames_count = item.get("frames_count")
        if not isinstance(frames_count, bool):
            try:
                group["frames"] += max(0, int(frames_count))
            except (TypeError, ValueError):
                pass
        created_at = str(item.get("created_at") or "")
        if created_at and (not group["oldest_created_at"] or created_at < group["oldest_created_at"]):
            group["oldest_created_at"] = created_at
        subject_id = str(item.get("subject_id") or "").strip()
        if subject_id:
            subjects.add(subject_id)
        group["jobs"].append(item)
    out = []
    for task_name, group in groups.items():
        subjects = sorted(subject_sets.get(task_name, set()))
        group["subjects"] = subjects
        group["subject_summary"] = ", ".join(subjects)
        out.append(group)
    return sorted(out, key=lambda item: (str(item.get("task_name") or ""), str(item.get("oldest_created_at") or "")))


def session_from_lease(
    *,
    backend_url: str,
    operator_id: str,
    mounts: Optional[Mapping[str, str]] = None,
    lease_seconds: int = 600,
    timeout_seconds: float = 10.0,
    task_name: str = "",
    episode_id: str = "",
) -> LabelJobSession:
    client = LabelBackendClient(backend_url, timeout_seconds=timeout_seconds)
    response = client.lease_label_episode(
        operator_id,
        lease_seconds=lease_seconds,
        task_name=task_name,
        episode_id=episode_id,
    )
    job = dict(response.get("job") or {})
    payload = dict(response.get("payload") or job.get("payload") or {})
    job_id = str(payload.get("job_id") or job.get("job_id") or "").strip()
    leased_episode_id = str(payload.get("episode_id") or job.get("episode_id") or episode_id or "").strip()
    if not job_id:
        raise BackendClientError("Backend did not return an episode label job id.")
    if not leased_episode_id:
        raise BackendClientError("Backend did not return an episode id.")
    return LabelJobSession(
        client=client,
        operator_id=operator_id,
        job_id=leased_episode_id,
        job=job,
        payload=payload,
        mounts=dict(mounts or {}),
        episode_id=leased_episode_id,
    )
