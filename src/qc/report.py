from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .state_store import now_iso


Range = Tuple[int, int]


def build_qc_result(
    *,
    episode_id: str,
    worker_id: str,
    bad_ranges: List[Range],
    bad_episode: bool = False,
    sample_interval: int = 10,
) -> Dict[str, Any]:
    if bad_episode:
        return {
            "passed": False,
            "qc_passed": False,
            "result_type": "bad_episode",
            "bad_episode": True,
            "reason": "bad_episode",
            "worker_id": worker_id,
            "episode_id": episode_id,
            "sample_interval": int(sample_interval),
            "segments": [],
            "mano_3d_checked": True,
        }
    segments = [{"start_frame": int(start), "end_frame": int(end)} for start, end in bad_ranges]
    passed = len(segments) == 0
    return {
        "passed": passed,
        "qc_passed": passed,
        "result_type": "passed" if passed else "bad_frames",
        "score": 1.0 if passed else 0.0,
        "reason": "qc_passed" if passed else "qc_failed",
        "worker_id": worker_id,
        "episode_id": episode_id,
        "sample_interval": int(sample_interval),
        "segments": segments,
        "mano_3d_checked": True,
    }


def write_qc_report(
    *,
    episode_dir: Path,
    result: Mapping[str, Any],
    bad_ranges: List[Range],
    sample_interval: int,
) -> Path:
    qc_dir = Path(episode_dir) / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    path = qc_dir / "qc_report.json"
    report = {
        "schema_version": 1,
        "kind": "orbbec_qc_report",
        "episode_id": str(result.get("episode_id") or ""),
        "passed": bool(result.get("passed")),
        "result_type": str(result.get("result_type") or ""),
        "score": result.get("score", 0.0),
        "reason": str(result.get("reason") or ""),
        "sample_interval": int(sample_interval),
        "segments": [{"start_frame": int(start), "end_frame": int(end)} for start, end in bad_ranges],
        "metrics": {},
        "worker_id": str(result.get("worker_id") or ""),
        "created_at": now_iso(),
    }
    fd, tmp_name = tempfile.mkstemp(prefix="qc_report_", suffix=".json", dir=str(qc_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass
    return path
