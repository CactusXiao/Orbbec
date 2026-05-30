from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_FRAME_RE = re.compile(r"^(\d{5})\.[^.]+$")


@dataclass(frozen=True)
class TaskRecord:
    task_name: str
    done_frame: int
    total_frames: int


def session_dir(base_dir: str, subject: str) -> Path:
    return Path(base_dir).expanduser().resolve() / subject


def discover_tasks(sess: Path) -> List[str]:
    if not sess.exists() or not sess.is_dir():
        return []

    ignore_names = {
        "RGB",
        "Depth",
        "IR",
        "PointCloud",
        "record.csv",
        "labels.json",
        "camera_params.json",
        "extrinsics.json",
    }

    tasks: List[str] = []
    for p in sorted(sess.iterdir()):
        if p.name in ignore_names:
            continue
        if p.is_dir():
            tasks.append(p.name)
    return tasks


def list_camera_ids(task_dir: Path) -> List[str]:
    if not task_dir.exists() or not task_dir.is_dir():
        return []

    cams: List[str] = []
    for p in sorted(task_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.isdigit():
            cams.append(p.name)
    return cams


def _rgb_dir(task_dir: Path, cam_id: str) -> Path:
    return task_dir / cam_id / "RGB"


def find_frame_path(task_dir: Path, cam_id: str, frame_idx: int) -> Optional[Path]:
    rgb = _rgb_dir(task_dir, cam_id)
    if not rgb.exists() or not rgb.is_dir():
        return None
    patt = f"{frame_idx:05d}."
    matches = [p for p in rgb.iterdir() if p.is_file() and p.name.startswith(patt)]
    if not matches:
        return None
    matches.sort(key=lambda p: p.name)
    return matches[0]


def _compute_total_frames_from_cam(task_dir: Path, cam_id: str) -> int:
    rgb = _rgb_dir(task_dir, cam_id)
    if not rgb.exists() or not rgb.is_dir():
        return 0
    max_idx = -1
    for p in rgb.iterdir():
        if not p.is_file():
            continue
        m = _FRAME_RE.match(p.name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0


def compute_total_frames(task_dir: Path) -> int:
    cams = list_camera_ids(task_dir)
    if not cams:
        return 0
    return _compute_total_frames_from_cam(task_dir, cams[0])


def record_csv_path(sess: Path) -> Path:
    return sess / "record.csv"


def ensure_record_csv(sess: Path, tasks: List[str]) -> Dict[str, TaskRecord]:
    p = record_csv_path(sess)
    existing = load_record_csv(sess)

    updated: Dict[str, TaskRecord] = {}
    for t in tasks:
        total = compute_total_frames(sess / t)
        prev = existing.get(t)
        done = prev.done_frame if prev else 0
        if done < 0:
            done = 0
        if total >= 0 and done > total:
            done = total
        updated[t] = TaskRecord(task_name=t, done_frame=done, total_frames=total)

    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

    save_record_csv(sess, list(updated.values()))
    return updated


def load_record_csv(sess: Path) -> Dict[str, TaskRecord]:
    p = record_csv_path(sess)
    if not p.exists() or not p.is_file():
        return {}

    out: Dict[str, TaskRecord] = {}
    try:
        with p.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                task = (row[0] or "").strip()
                if not task:
                    continue
                try:
                    done = int(row[1])
                except Exception:
                    done = 0
                try:
                    total = int(row[2])
                except Exception:
                    total = 0
                out[task] = TaskRecord(task_name=task, done_frame=done, total_frames=total)
    except Exception:
        return {}
    return out


def save_record_csv(sess: Path, records: List[TaskRecord]) -> None:
    p = record_csv_path(sess)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = p.parent
    fd, tmp_name = tempfile.mkstemp(prefix="record_", suffix=".csv", dir=str(tmp_dir))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in sorted(records, key=lambda x: x.task_name):
                writer.writerow([r.task_name, int(r.done_frame), int(r.total_frames)])
        os.replace(tmp_name, p)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass


def labels_json_path(sess: Path) -> Path:
    return sess / "labels.json"


def load_labels(sess: Path) -> Dict:
    p = labels_json_path(sess)
    if not p.exists() or not p.is_file():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_labels(sess: Path, obj: Dict) -> None:
    p = labels_json_path(sess)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = p.parent
    fd, tmp_name = tempfile.mkstemp(prefix="labels_", suffix=".json", dir=str(tmp_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, p)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass


def update_labels_for_frame(
    sess: Path,
    task_name: str,
    cam_points: Dict[str, List[Tuple[int, int]]],
    frame_idx: int,
) -> None:
    labels = load_labels(sess)
    task_obj = labels.get(task_name)
    if not isinstance(task_obj, dict):
        task_obj = {}
        labels[task_name] = task_obj

    frame_key = str(int(frame_idx))
    for cam_id, pts in cam_points.items():
        cam_obj = task_obj.get(cam_id)
        if not isinstance(cam_obj, dict):
            cam_obj = {}
            task_obj[cam_id] = cam_obj
        cam_obj[frame_key] = [[int(x), int(y)] for (x, y) in pts]

    save_labels(sess, labels)


def clear_labels_for_frame(sess: Path, task_name: str, cam_ids: List[str], frame_idx: int) -> bool:
    labels = load_labels(sess)
    task_obj = labels.get(task_name)
    if not isinstance(task_obj, dict):
        return False

    frame_key = str(int(frame_idx))
    touched = False
    for cam_id in cam_ids:
        cam_obj = task_obj.get(cam_id)
        if not isinstance(cam_obj, dict):
            continue
        if frame_key not in cam_obj:
            continue
        cam_obj[frame_key] = []
        touched = True

    if touched:
        save_labels(sess, labels)
    return touched
