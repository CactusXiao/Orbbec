from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional

try:
    from .backend_client import parse_mounts_json
except Exception:
    from backend_client import parse_mounts_json


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
        if ch == "#" and not in_single and not in_double and (idx == 0 or value[idx - 1].isspace()):
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


def default_label_env_path() -> Optional[Path]:
    explicit = os.environ.get("ORBBEC_LABEL_ENV") or os.environ.get("ORBBEC_ENV_FILE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    base = Path.cwd().expanduser().resolve()
    for parent in (base, *base.parents):
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    repo_candidate = Path(__file__).resolve().parents[1] / ".env"
    return repo_candidate if repo_candidate.exists() else None


def env_value(file_env: Dict[str, str], key: str) -> str:
    return os.environ.get(key, "").strip() or str(file_env.get(key) or "").strip()


def label_nas_mounts_from_env() -> Dict[str, str]:
    env_path = default_label_env_path()
    file_env = load_env_file(env_path) if env_path is not None else {}
    raw = env_value(file_env, "ORBBEC_NAS_MOUNTS_JSON")
    mounts = parse_mounts_json(raw) if raw else {}
    prefix = env_value(file_env, "ORBBEC_NAS_URI_PREFIX").rstrip("/")
    root = env_value(file_env, "ORBBEC_NAS_ROOT")
    if prefix and root:
        mounts.setdefault(prefix, root)
    return mounts
