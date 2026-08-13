from __future__ import annotations

import getpass
import json
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


def strip_env_comment(value: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(value):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_double:
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            if idx == 0 or value[idx - 1].isspace():
                return value[:idx].rstrip()
    return value.strip()


def unquote_env_value(value: str) -> str:
    value = strip_env_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        inner = value[1:-1]
        if value[0] == '"':
            inner = inner.replace(r"\\", "\\").replace(r"\"", '"')
        return inner
    return value


def load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                raise ValueError(f"invalid .env line {line_no}: missing '='")
            key, value = line.split("=", 1)
            key = key.strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                raise ValueError(f"invalid .env line {line_no}: invalid key {key!r}")
            out[key] = unquote_env_value(value)
    return out


def _first(env: Dict[str, str], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
        value = env.get(key)
        if value:
            return value
    return default


def _int(env: Dict[str, str], keys: Iterable[str], default: int) -> int:
    value = _first(env, keys, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        joined = ", ".join(keys)
        raise ValueError(f"invalid integer for {joined}: {value!r}") from exc


def _float(env: Dict[str, str], keys: Iterable[str], default: float) -> float:
    value = _first(env, keys, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        joined = ", ".join(keys)
        raise ValueError(f"invalid number for {joined}: {value!r}") from exc


def _path(env: Dict[str, str], keys: Iterable[str], default: str) -> Path:
    raw = _first(env, keys, default)
    return Path(raw).expanduser().resolve()


def _json_object(env: Dict[str, str], keys: Iterable[str]) -> Dict[str, str]:
    raw = _first(env, keys, "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        joined = ", ".join(keys)
        raise ValueError(f"invalid JSON object for {joined}: {exc}") from exc
    if not isinstance(parsed, dict):
        joined = ", ".join(keys)
        raise ValueError(f"{joined} must be a JSON object")
    out: Dict[str, str] = {}
    for key, value in parsed.items():
        prefix = str(key or "").strip().rstrip("/")
        root = str(value or "").strip()
        if prefix and root:
            out[prefix] = root
    return out


def _default_worker_id() -> str:
    host = socket.gethostname() or "unknown-host"
    user = getpass.getuser() or "user"
    return f"qc_{host}_{user}"


def _default_env_path(cwd: Optional[Path]) -> Optional[Path]:
    explicit = os.environ.get("QC_ENV_FILE") or os.environ.get("ORBBEC_QC_ENV")
    if explicit:
        return Path(explicit).expanduser().resolve()
    base = (cwd or Path.cwd()).expanduser().resolve()
    candidate = base / ".env"
    return candidate if candidate.exists() else None


@dataclass(frozen=True)
class QcConfig:
    backend_url: str
    sample_interval: int
    default_lease_minutes: int
    crash_lease_extension_minutes: int
    tmp_dir: Path
    state_dir: Path
    worker_machine_id: str
    range_merge_gap_frames: int
    request_timeout_seconds: float
    nas_mounts: Dict[str, str]

    @property
    def default_lease_seconds(self) -> int:
        return max(1, int(self.default_lease_minutes) * 60)

    @property
    def crash_lease_extension_seconds(self) -> int:
        return max(1, int(self.crash_lease_extension_minutes) * 60)


def load_qc_config(*, cwd: Optional[Path] = None) -> QcConfig:
    env_path = _default_env_path(cwd)
    file_env = load_env_file(env_path) if env_path is not None else {}
    config = QcConfig(
        backend_url=_first(file_env, ("QC_BACKEND_URL", "ORBBEC_BACKEND_URL", "TASK_BACKEND_URL"), "http://127.0.0.1:8765").rstrip("/"),
        sample_interval=max(1, _int(file_env, ("QC_SAMPLE_INTERVAL",), 10)),
        default_lease_minutes=max(1, _int(file_env, ("QC_DEFAULT_LEASE_MINUTES",), 10)),
        crash_lease_extension_minutes=max(1, _int(file_env, ("QC_CRASH_LEASE_EXTENSION_MINUTES",), 10)),
        tmp_dir=_path(file_env, ("QC_TMP_DIR",), "./tmp"),
        state_dir=_path(file_env, ("QC_STATE_DIR",), "./qc_state"),
        worker_machine_id=_first(file_env, ("QC_WORKER_MACHINE_ID",), _default_worker_id()),
        range_merge_gap_frames=max(0, _int(file_env, ("QC_RANGE_MERGE_GAP_FRAMES",), 5)),
        request_timeout_seconds=max(1.0, _float(file_env, ("QC_BACKEND_TIMEOUT_SECONDS",), 10.0)),
        nas_mounts=_json_object(file_env, ("ORBBEC_NAS_MOUNTS_JSON",)),
    )
    config.tmp_dir.mkdir(parents=True, exist_ok=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    return config
