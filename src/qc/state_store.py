from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Range = Tuple[int, int]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def seconds_until(value: str, *, now: Optional[datetime] = None) -> int:
    dt = parse_iso(value)
    if dt is None:
        return 0
    base = now or utc_now()
    return max(0, int((dt - base).total_seconds()))


def format_seconds(total_seconds: int) -> str:
    total = max(0, int(total_seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def normalize_ranges(ranges: Iterable[Iterable[int]], *, max_gap_frames: int = 5) -> List[Range]:
    clean: List[Range] = []
    for item in ranges:
        pair = list(item)
        if len(pair) < 2:
            continue
        start = int(pair[0])
        end = int(pair[1])
        if end < start:
            start, end = end, start
        clean.append((start, end))
    clean.sort(key=lambda x: (x[0], x[1]))
    merged: List[Range] = []
    for start, end in clean:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1
        if start <= prev_end + 1 or gap < max(0, int(max_gap_frames)):
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def first_sample_after(frame: int, *, first_frame: int, last_frame: int, sample_interval: int) -> int:
    base = int(first_frame)
    last = int(last_frame)
    interval = max(1, int(sample_interval))
    offset = max(0, int(frame) - base)
    target = base + ((offset // interval) + 1) * interval
    return last + 1 if target > last else target


def _state_file_name(worker_id: str, job_id: str, episode_id: str) -> str:
    raw = f"{worker_id}|{job_id}|{episode_id}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    safe_episode = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in episode_id)[:80]
    return f"{safe_episode or 'episode'}_{digest}.json"


@dataclass
class QcProgress:
    task_name: str
    episode_id: str
    job_id: str
    worker_machine_id: str
    lease_until: str
    sample_interval: int
    current_frame: int
    frames: List[int] = field(default_factory=list)
    result_type: str = "in_progress"
    bad_frame_ranges: List[Range] = field(default_factory=list)
    ego_bad_frame_ranges: List[Range] = field(default_factory=list)
    checked_sample_frames: List[int] = field(default_factory=list)
    playback_complete: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)
    job: Dict[str, Any] = field(default_factory=dict)
    episode: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: int = 2
    kind: str = "orbbec_qc_local_progress"
    updated_at: str = field(default_factory=now_iso)

    @property
    def first_frame(self) -> int:
        return min(self.frames) if self.frames else 0

    @property
    def last_frame(self) -> int:
        return max(self.frames) if self.frames else 0

    @property
    def lease_seconds_remaining(self) -> int:
        return seconds_until(self.lease_until)

    @property
    def lease_valid(self) -> bool:
        return self.lease_seconds_remaining > 0

    @property
    def is_complete(self) -> bool:
        return self.result_type == "bad_episode" or bool(self.playback_complete)

    def touch(self) -> None:
        self.updated_at = now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "task_name": self.task_name,
            "episode_id": self.episode_id,
            "job_id": self.job_id,
            "worker_machine_id": self.worker_machine_id,
            "lease_until": self.lease_until,
            "sample_interval": int(self.sample_interval),
            "current_frame": int(self.current_frame),
            "frames": [int(frame) for frame in self.frames],
            "result_type": self.result_type,
            "bad_frame_ranges": [[int(a), int(b)] for a, b in self.bad_frame_ranges],
            "ego_bad_frame_ranges": [[int(a), int(b)] for a, b in self.ego_bad_frame_ranges],
            "checked_sample_frames": [int(frame) for frame in self.checked_sample_frames],
            "playback_complete": bool(self.playback_complete),
            "payload": self.payload,
            "job": self.job,
            "episode": self.episode,
            "artifacts": self.artifacts,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "QcProgress":
        ranges = normalize_ranges(obj.get("bad_frame_ranges") or [], max_gap_frames=0)
        ego_ranges = normalize_ranges(obj.get("ego_bad_frame_ranges") or [], max_gap_frames=0)
        frames = []
        for frame in obj.get("frames") or []:
            if isinstance(frame, bool):
                continue
            try:
                frames.append(int(frame))
            except (TypeError, ValueError):
                continue
        checked = []
        for frame in obj.get("checked_sample_frames") or []:
            if isinstance(frame, bool):
                continue
            try:
                checked.append(int(frame))
            except (TypeError, ValueError):
                continue
        current_frame = int(obj.get("current_frame") or 0)
        playback_complete = bool(obj.get("playback_complete", False))
        if "playback_complete" not in obj and frames and current_frame > max(frames):
            playback_complete = True
            current_frame = max(frames)
        return cls(
            task_name=str(obj.get("task_name") or ""),
            episode_id=str(obj.get("episode_id") or ""),
            job_id=str(obj.get("job_id") or ""),
            worker_machine_id=str(obj.get("worker_machine_id") or ""),
            lease_until=str(obj.get("lease_until") or ""),
            sample_interval=max(1, int(obj.get("sample_interval") or 10)),
            current_frame=current_frame,
            frames=sorted(set(frames)),
            result_type=str(obj.get("result_type") or "in_progress"),
            bad_frame_ranges=ranges,
            ego_bad_frame_ranges=ego_ranges,
            checked_sample_frames=sorted(set(checked)),
            playback_complete=playback_complete,
            payload=dict(obj.get("payload") or {}),
            job=dict(obj.get("job") or {}),
            episode=dict(obj.get("episode") or {}),
            artifacts=[dict(item) for item in obj.get("artifacts") or [] if isinstance(item, dict)],
            schema_version=max(2, int(obj.get("schema_version") or 1)),
            kind=str(obj.get("kind") or "orbbec_qc_local_progress"),
            updated_at=str(obj.get("updated_at") or now_iso()),
        )

    @classmethod
    def from_lease_response(cls, response: Dict[str, Any], *, worker_machine_id: str, sample_interval: int) -> "QcProgress":
        job = dict(response.get("job") or {})
        payload = dict(response.get("payload") or job.get("payload") or {})
        episode = dict(response.get("episode") or {})
        frames: List[int] = []
        for item in payload.get("frames") or []:
            if isinstance(item, bool):
                continue
            try:
                frames.append(int(item))
            except (TypeError, ValueError):
                continue
        frames = sorted(set(frames))
        first = frames[0] if frames else 0
        return cls(
            task_name=str(payload.get("task_name") or episode.get("task_name") or ""),
            episode_id=str(payload.get("episode_id") or job.get("episode_id") or episode.get("episode_id") or ""),
            job_id=str(payload.get("job_id") or job.get("job_id") or ""),
            worker_machine_id=worker_machine_id,
            lease_until=str(job.get("lease_until") or ""),
            sample_interval=max(1, int(sample_interval)),
            current_frame=first,
            frames=frames,
            payload=payload,
            job=job,
            episode=episode,
            artifacts=[dict(item) for item in response.get("artifacts") or [] if isinstance(item, dict)],
        )


class QcStateStore:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def list_progress(self, *, worker_machine_id: str = "") -> List[QcProgress]:
        out: List[QcProgress] = []
        for path in sorted(self.state_dir.glob("*.json")):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                progress = QcProgress.from_dict(obj)
            except Exception:
                continue
            if worker_machine_id and progress.worker_machine_id != worker_machine_id:
                continue
            out.append(progress)
        return out

    def save(self, progress: QcProgress) -> Path:
        progress.touch()
        name = _state_file_name(progress.worker_machine_id, progress.job_id, progress.episode_id)
        path = self.state_dir / name
        fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".json", dir=str(self.state_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(progress.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp_name, path)
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
            except Exception:
                pass
        return path

    def delete(self, progress: QcProgress) -> None:
        name = _state_file_name(progress.worker_machine_id, progress.job_id, progress.episode_id)
        path = self.state_dir / name
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
