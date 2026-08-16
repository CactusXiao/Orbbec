from __future__ import annotations

import getpass
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class LabelConfig:
    backend_url: str = "http://127.0.0.1:8765"
    operator_id: str = "labeler_01"
    frame_cache_dir: Optional[Path] = None
    ffmpeg_executable: str = "ffmpeg"
    lease_seconds: int = 600
    request_timeout_seconds: float = 10.0
    nas_mounts: Dict[str, str] = field(default_factory=dict)


def _string(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer config value: {value!r}") from exc


def _float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid number config value: {value!r}") from exc


def _mounts(value: Any) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("nas_mounts must be a JSON object")
    out: Dict[str, str] = {}
    for key, root in value.items():
        prefix = str(key or "").strip().rstrip("/")
        path = str(root or "").strip()
        if prefix and path:
            out[prefix] = path
    return out


def _load_object(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)
    if not isinstance(parsed, dict):
        raise ValueError(f"Label config must be a JSON object: {path}")
    return dict(parsed)


def _config_base_dir(config_path: Optional[Path]) -> Path:
    if config_path is None:
        return Path.cwd().expanduser().resolve()
    return Path(config_path).expanduser().resolve().parent


def _optional_path(value: Any, base_dir: Path) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_label_config(config_path: Optional[Path] = None) -> LabelConfig:
    data = _load_object(config_path)
    base_dir = _config_base_dir(config_path)
    default_operator = getpass.getuser() or "labeler_01"
    return LabelConfig(
        backend_url=_string(data.get("backend_url"), "http://127.0.0.1:8765").rstrip("/"),
        operator_id=_string(data.get("operator_id"), default_operator),
        frame_cache_dir=_optional_path(data.get("frame_cache_dir"), base_dir),
        ffmpeg_executable=_string(data.get("ffmpeg_executable"), "ffmpeg"),
        lease_seconds=max(1, _int(data.get("lease_seconds"), 600)),
        request_timeout_seconds=max(1.0, _float(data.get("request_timeout_seconds"), 10.0)),
        nas_mounts=_mounts(data.get("nas_mounts")),
    )
