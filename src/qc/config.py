from __future__ import annotations

import getpass
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class QcConfig:
    backend_url: str
    sample_interval: int
    default_lease_minutes: int
    crash_lease_extension_minutes: int
    tmp_dir: Path
    state_dir: Path
    worker_machine_id: str
    operator_id: str
    range_merge_gap_frames: int
    request_timeout_seconds: float
    nas_mounts: Dict[str, str]
    playback_fps: float
    mesh_renderer_python: str
    mano_toolkit_root: Path
    mano_model_dir: Path
    mesh_render_factor: float
    mesh_render_workers: int
    mesh_prefer_integrated_gpu: bool
    mesh_prebuffer_frames: int

    @property
    def default_lease_seconds(self) -> int:
        return max(1, int(self.default_lease_minutes) * 60)

    @property
    def crash_lease_extension_seconds(self) -> int:
        return max(1, int(self.crash_lease_extension_minutes) * 60)


def _default_worker_id() -> str:
    host = socket.gethostname() or "unknown-host"
    user = getpass.getuser() or "user"
    return f"qc_{host}_{user}"


def _read_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)
    if not isinstance(parsed, dict):
        raise ValueError(f"QC config must be a JSON object: {path}")
    return dict(parsed)


def _base_dir(config_path: Optional[Path], cwd: Optional[Path]) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve().parent
    return (cwd or Path.cwd()).expanduser().resolve()


def _string(data: Mapping[str, Any], key: str, default: str = "") -> str:
    value = str(data.get(key) or "").strip()
    return value if value else default


def _int(data: Mapping[str, Any], key: str, default: int) -> int:
    value = data.get(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer for {key}: {value!r}") from exc


def _float(data: Mapping[str, Any], key: str, default: float) -> float:
    value = data.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid number for {key}: {value!r}") from exc


def _bool(data: Mapping[str, Any], key: str, default: bool) -> bool:
    value = data.get(key)
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"invalid boolean for {key}: {value!r}")


def _path(data: Mapping[str, Any], key: str, default: str, base_dir: Path) -> Path:
    raw = str(data.get(key) or default).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _mounts(data: Mapping[str, Any]) -> Dict[str, str]:
    raw = data.get("nas_mounts")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("nas_mounts must be a JSON object")
    out: Dict[str, str] = {}
    for key, root in raw.items():
        prefix = str(key or "").strip().rstrip("/")
        path = str(root or "").strip()
        if prefix and path:
            out[prefix] = path
    return out


def load_qc_config(*, config_path: Optional[Path] = None, cwd: Optional[Path] = None) -> QcConfig:
    data = _read_config(config_path)
    base = _base_dir(config_path, cwd)
    config = QcConfig(
        backend_url=_string(data, "backend_url", "http://127.0.0.1:8765").rstrip("/"),
        sample_interval=max(1, _int(data, "sample_interval", 10)),
        default_lease_minutes=max(1, _int(data, "default_lease_minutes", 10)),
        crash_lease_extension_minutes=max(1, _int(data, "crash_lease_extension_minutes", 10)),
        tmp_dir=_path(data, "tmp_dir", "./tmp", base),
        state_dir=_path(data, "state_dir", "./qc_state", base),
        worker_machine_id=_string(data, "worker_machine_id", _default_worker_id()),
        operator_id=_string(data, "operator_id", getpass.getuser() or "qc_operator"),
        range_merge_gap_frames=max(0, _int(data, "range_merge_gap_frames", 5)),
        request_timeout_seconds=max(1.0, _float(data, "request_timeout_seconds", 10.0)),
        nas_mounts=_mounts(data),
        playback_fps=max(1.0, min(60.0, _float(data, "playback_fps", 30.0))),
        mesh_renderer_python=_string(data, "mesh_renderer_python", "python3"),
        mano_toolkit_root=_path(
            data,
            "mano_toolkit_root",
            "/home/ubuntu/WorkSpace/zhenghao/opt_toolkits",
            base,
        ),
        mano_model_dir=_path(
            data,
            "mano_model_dir",
            "/home/ubuntu/WorkSpace/zhenghao/opt_toolkits/ckpt/mano",
            base,
        ),
        mesh_render_factor=max(0.5, min(4.0, _float(data, "mesh_render_factor", 0.5))),
        mesh_render_workers=max(1, min(32, _int(data, "mesh_render_workers", 6))),
        mesh_prefer_integrated_gpu=_bool(data, "mesh_prefer_integrated_gpu", True),
        mesh_prebuffer_frames=max(1, min(300, _int(data, "mesh_prebuffer_frames", 30))),
    )
    config.tmp_dir.mkdir(parents=True, exist_ok=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    return config
