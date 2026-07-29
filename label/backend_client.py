from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
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


class UriResolver:
    def __init__(self, mounts: Optional[Mapping[str, str]] = None):
        self.mounts = {str(k).rstrip("/"): str(v) for k, v in (mounts or {}).items() if str(k).strip()}

    def resolve(self, uri_or_path: str) -> Path:
        value = str(uri_or_path or "").strip()
        if not value:
            raise ValueError("Task data URI/path is empty.")

        parsed = urlparse(value)
        if not parsed.scheme:
            return Path(value).expanduser().resolve()
        if parsed.scheme == "local":
            return self._resolve_local(parsed)

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
        raise ValueError(f"No local mount mapping is configured for URI: {value}")

    @staticmethod
    def _resolve_local(parsed) -> Path:
        if parsed.netloc and parsed.path:
            raw_path = "/" + parsed.netloc + parsed.path
        elif parsed.netloc:
            raw_path = parsed.netloc
        else:
            raw_path = parsed.path
        return Path(unquote(raw_path)).expanduser().resolve()


class LabelBackendClient:
    def __init__(self, backend_url: str, *, timeout_seconds: float = 10.0):
        self.backend_url = (backend_url or "http://127.0.0.1:8765").strip().rstrip("/")
        self.timeout_seconds = float(timeout_seconds or 10.0)

    def create_dev_label_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/api/v1/dev/label/jobs", payload)

    def lease_label_job(self, operator_id: str, *, lease_seconds: int = 600) -> Dict[str, Any]:
        return self._post(
            "/api/v1/label/jobs/lease",
            {"operator_id": operator_id, "lease_seconds": int(lease_seconds)},
        )

    def get_label_job(self, job_id: str) -> Dict[str, Any]:
        return self._get(f"/api/v1/label/jobs/{job_id}")

    def heartbeat_label_job(self, job_id: str, operator_id: str, *, lease_seconds: int = 600) -> Dict[str, Any]:
        return self._post(
            f"/api/v1/label/jobs/{job_id}/heartbeat",
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
        return self._post(f"/api/v1/label/jobs/{job_id}/complete", payload)

    def release_label_job(self, job_id: str, *, reason: str = "") -> Dict[str, Any]:
        return self._post(f"/api/v1/label/jobs/{job_id}/release", {"reason": reason})

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


def session_from_lease(
    *,
    backend_url: str,
    operator_id: str,
    mounts: Optional[Mapping[str, str]] = None,
    lease_seconds: int = 600,
) -> LabelJobSession:
    client = LabelBackendClient(backend_url)
    response = client.lease_label_job(operator_id, lease_seconds=lease_seconds)
    job = dict(response.get("job") or {})
    payload = dict(response.get("payload") or job.get("payload") or {})
    job_id = str(payload.get("job_id") or job.get("job_id") or "").strip()
    if not job_id:
        raise BackendClientError("Backend did not return a label job id.")
    return LabelJobSession(
        client=client,
        operator_id=operator_id,
        job_id=job_id,
        job=job,
        payload=payload,
        mounts=dict(mounts or {}),
    )
