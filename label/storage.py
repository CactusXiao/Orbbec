from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from .backend_client import UriResolver
except Exception:
    from backend_client import UriResolver


_FRAME_RE = re.compile(r"^(\d{5})\.[^.]+$")
_HAND_COUNT = 2
_JOINT_COUNT = 21


@dataclass(frozen=True)
class CorrectionTask:
    line_no: int
    root: str
    subject: str
    task: str
    episode: str
    cameras: List[str]
    frames: List[int]
    rgb_path_template: str = "{camera}/RGB/{frame:05d}.png"
    prediction_dir: str = "pred_2d"
    correction_dir: str = "corrected_2d"
    mano_projection_dir: str = "mano/episode/projected_2d"
    episode_path: Optional[str] = None

    @property
    def key(self) -> str:
        return str(self.line_no)

    @property
    def display_name(self) -> str:
        return f"{self.line_no} {self.subject}/{self.task}/{self.episode}"

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    def episode_dir(self) -> Path:
        if self.episode_path:
            return Path(self.episode_path).expanduser().resolve()
        return Path(self.root).expanduser().resolve() / self.subject / self.task / self.episode


@dataclass
class CorrectionProgress:
    task_key: str
    done_positions: Set[int] = field(default_factory=set)
    total_frames: int = 0

    @property
    def done_count(self) -> int:
        return len(self.done_positions)


@dataclass
class PredictionBundle:
    mode: str
    episode_dir: Path
    prediction_dir: Path
    pred_dir: Path
    corrected_dir: Path
    samples: Dict[Tuple[str, int], "PredictionSample"]
    pending: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionSample:
    cam_id: str
    frame_idx: int
    source_path: Optional[Path]
    corrected_path: Path


def find_frame_path(
    task_dir: Path,
    cam_id: str,
    frame_idx: int,
    rgb_path_template: str = "{camera}/RGB/{frame:05d}.png",
) -> Optional[Path]:
    template = rgb_path_template or "{camera}/RGB/{frame:05d}.png"
    try:
        relative = template.format(camera=cam_id, frame=int(frame_idx))
    except Exception as exc:
        raise ValueError(f"Invalid rgb_path_template: {template}") from exc
    candidate = task_dir / relative
    if candidate.exists() and candidate.is_file():
        return candidate

    rgb = task_dir / cam_id / "RGB"
    if not rgb.exists() or not rgb.is_dir():
        return None
    patt = f"{frame_idx:05d}."
    matches = [p for p in rgb.iterdir() if p.is_file() and p.name.startswith(patt)]
    if not matches:
        return None
    matches.sort(key=lambda p: p.name)
    return matches[0]


def _as_str(obj: Dict, key: str, line_no: int) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Line {line_no}: `{key}` must be a non-empty string.")
    return value


def _as_str_list(obj: Dict, key: str, line_no: int) -> List[str]:
    value = obj.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Line {line_no}: `{key}` must be a non-empty list.")
    out: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"Line {line_no}: `{key}` must contain non-empty strings.")
        out.append(item)
    return out


def _optional_str_list(obj: Dict, key: str) -> List[str]:
    value = obj.get(key)
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
    return out


def _as_int_list(obj: Dict, key: str, line_no: int) -> List[int]:
    value = obj.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Line {line_no}: `{key}` must be a non-empty list.")
    out: List[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"Line {line_no}: `{key}` must contain integer frame indices.")
        out.append(int(item))
    return out


def _optional_int_list(obj: Dict, key: str) -> List[int]:
    value = obj.get(key)
    if not isinstance(value, list):
        return []
    out: List[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _discover_cameras(episode_dir: Path) -> List[str]:
    if not episode_dir.exists() or not episode_dir.is_dir():
        return []
    cameras: List[str] = []
    for child in episode_dir.iterdir():
        if child.is_dir() and (child / "RGB").is_dir():
            cameras.append(child.name)
    return sorted(cameras)


def _discover_frames(episode_dir: Path, cameras: List[str]) -> List[int]:
    if not cameras:
        return []
    rgb_dir = episode_dir / cameras[0] / "RGB"
    if not rgb_dir.exists() or not rgb_dir.is_dir():
        return []
    frames = set()
    for child in rgb_dir.iterdir():
        if not child.is_file():
            continue
        match = _FRAME_RE.match(child.name)
        if match:
            frames.add(int(match.group(1)))
    return sorted(frames)


def load_correction_tasks(jsonl_path: str) -> List[CorrectionTask]:
    p = Path(jsonl_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise ValueError(f"Tasks JSONL not found: {p}")

    tasks: List[CorrectionTask] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no}: JSONL item must be an object.")
            tasks.append(
                CorrectionTask(
                    line_no=line_no,
                    root=_as_str(obj, "root", line_no),
                    subject=_as_str(obj, "subject", line_no),
                    task=_as_str(obj, "task", line_no),
                    episode=_as_str(obj, "episode", line_no),
                    cameras=_as_str_list(obj, "cameras", line_no),
                    frames=_as_int_list(obj, "frames", line_no),
                )
            )

    if not tasks:
        raise ValueError(f"Tasks JSONL is empty: {p}")
    return tasks


def correction_task_from_backend_payload(
    payload: Dict[str, Any],
    *,
    mounts: Optional[Dict[str, str]] = None,
    line_no: int = 1,
) -> CorrectionTask:
    if not isinstance(payload, dict):
        raise ValueError("Backend label job payload must be an object.")
    resolved_path = str(
        payload.get("resolved_data_path")
        or payload.get("local_episode_path")
        or payload.get("local_capture_path")
        or ""
    ).strip()
    if resolved_path:
        episode_dir = Path(resolved_path).expanduser().resolve()
    else:
        resolver = UriResolver(mounts or {})
        data_uri = str(payload.get("data_uri") or "").strip()
        if not data_uri:
            raise ValueError("Backend label job payload must include `resolved_data_path` or `data_uri`.")
        episode_dir = resolver.resolve(data_uri)
    cameras = _optional_str_list(payload, "cameras") or _discover_cameras(episode_dir)
    frames = _optional_int_list(payload, "frames") or _discover_frames(episode_dir, cameras)
    if not cameras:
        raise ValueError(
            "Backend label job payload must include non-empty `cameras`, "
            "or `resolved_data_path` must point to an episode directory with camera/RGB folders."
        )
    if not frames:
        raise ValueError(
            "Backend label job payload must include non-empty `frames`, "
            "or `resolved_data_path` must point to an episode directory with RGB frame files."
        )
    return CorrectionTask(
        line_no=line_no,
        root=str(episode_dir.parent.parent.parent) if len(episode_dir.parts) >= 3 else str(episode_dir.parent),
        subject=str(payload.get("subject_id") or "unknown_subject"),
        task=str(payload.get("task_name") or "manual_label"),
        episode=str(payload.get("episode_id") or episode_dir.name),
        cameras=cameras,
        frames=frames,
        rgb_path_template=str(payload.get("rgb_path_template") or "{camera}/RGB/{frame:05d}.png"),
        prediction_dir=str(payload.get("prediction_dir") or "pred_2d"),
        correction_dir=str(payload.get("correction_dir") or "corrected_2d"),
        mano_projection_dir=str(payload.get("mano_projection_dir") or "mano/episode/projected_2d"),
        episode_path=str(episode_dir),
    )


def progress_csv_path(jsonl_path: str) -> Path:
    p = Path(jsonl_path).expanduser().resolve()
    return p.with_name(f"{p.stem}.record.csv")


def load_correction_progress(jsonl_path: str, tasks: List[CorrectionTask]) -> Dict[str, CorrectionProgress]:
    p = progress_csv_path(jsonl_path)
    existing: Dict[str, Set[int]] = {}
    if p.exists() and p.is_file():
        try:
            with p.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0] == "task_key":
                        continue
                    key = (row[0] or "").strip()
                    if not key:
                        continue
                    existing[key] = _parse_done_positions(row[1] if len(row) > 1 else "")
        except Exception:
            existing = {}

    out: Dict[str, CorrectionProgress] = {}
    for task in tasks:
        done = {pos for pos in existing.get(task.key, set()) if 0 <= pos < task.total_frames}
        out[task.key] = CorrectionProgress(task_key=task.key, done_positions=done, total_frames=task.total_frames)
    return out


def ensure_correction_progress(jsonl_path: str, tasks: List[CorrectionTask]) -> Dict[str, CorrectionProgress]:
    records = load_correction_progress(jsonl_path, tasks)
    save_correction_progress(jsonl_path, records)
    return records


def _parse_done_positions(text: str) -> Set[int]:
    text = (text or "").strip()
    if not text:
        return set()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return {int(x) for x in obj if isinstance(x, int) and not isinstance(x, bool)}
    except Exception:
        pass
    out: Set[int] = set()
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def save_correction_progress(jsonl_path: str, records: Dict[str, CorrectionProgress]) -> None:
    p = progress_csv_path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{p.stem}_", suffix=".csv", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["task_key", "done_positions", "total_frames"])
            for key in sorted(records, key=lambda x: int(x) if x.isdigit() else x):
                r = records[key]
                writer.writerow([r.task_key, json.dumps(sorted(r.done_positions)), int(r.total_frames)])
        os.replace(tmp_name, p)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass


def load_prediction_bundle(task: CorrectionTask, *, mode: str = "pred") -> PredictionBundle:
    mode = (mode or "pred").strip().lower()
    if mode not in {"mano", "pred", "correct", "last", "scratch"}:
        raise ValueError(f"Unsupported label mode: {mode}")

    episode_dir = task.episode_dir()
    pred_dir = _source_dir_for_mode(task, mode)
    corrected_dir = episode_dir / task.correction_dir
    return PredictionBundle(
        mode=mode,
        episode_dir=episode_dir,
        prediction_dir=episode_dir / task.prediction_dir,
        pred_dir=pred_dir,
        corrected_dir=corrected_dir,
        samples={},
    )


def _source_dir_for_mode(task: CorrectionTask, mode: str) -> Path:
    episode_dir = task.episode_dir()
    if mode in {"correct", "last"}:
        return episode_dir / task.correction_dir
    if mode == "mano":
        return episode_dir / task.mano_projection_dir
    return episode_dir / task.prediction_dir


def _source_frame_idx(mode: str, frame_idx: int) -> int:
    if mode == "last":
        return int(frame_idx) - 1
    return int(frame_idx)


def _corrected_name(pred_dir: Path, cam_id: str, frame_idx: int) -> str:
    pred_path = find_optional_prediction_frame_path(pred_dir, cam_id, frame_idx)
    if pred_path is not None:
        return pred_path.name
    return f"{int(frame_idx):05d}.npy"


def _correction_dtype(pred: np.ndarray) -> np.dtype:
    if pred.dtype.kind == "u":
        return np.dtype("float32")
    return pred.dtype


def _validate_prediction_view(arr: np.ndarray, path: Path) -> None:
    if arr.shape != (_HAND_COUNT, _JOINT_COUNT, 2):
        raise ValueError(f"Prediction frame array must have shape (2,21,2), got {arr.shape}: {path}")


def find_prediction_frame_path(pred_dir: Path, cam_id: str, frame_idx: int) -> Path:
    match = find_optional_prediction_frame_path(pred_dir, cam_id, frame_idx)
    if match is None:
        raise ValueError(f"Prediction npy not found for camera {cam_id}, frame {frame_idx}: {pred_dir / cam_id}")
    return match


def find_optional_prediction_frame_path(pred_dir: Path, cam_id: str, frame_idx: int) -> Optional[Path]:
    frame_idx = int(frame_idx)
    if frame_idx < 0:
        return None
    cam_dir = pred_dir / cam_id
    if not cam_dir.exists() or not cam_dir.is_dir():
        return None

    padded_prefix = f"{frame_idx:05d}."
    matches = [p for p in cam_dir.iterdir() if p.is_file() and p.suffix == ".npy" and p.name.startswith(padded_prefix)]
    if not matches:
        exact = cam_dir / f"{int(frame_idx)}.npy"
        if exact.exists() and exact.is_file():
            matches = [exact]
    if not matches:
        return None
    if len(matches) > 1:
        names = ", ".join(p.name for p in sorted(matches))
        raise ValueError(f"Multiple prediction npy files found for camera {cam_id}, frame {frame_idx}: {names}")
    return matches[0]


def source_frame_path(bundle: PredictionBundle, frame_idx: int, cam_id: str) -> Optional[Path]:
    if bundle.mode == "scratch":
        return None
    return find_optional_prediction_frame_path(bundle.pred_dir, cam_id, _source_frame_idx(bundle.mode, frame_idx))


def load_frame_visibility(base_dir: Path, cam_id: str, frame_idx: int) -> Optional[List[List[bool]]]:
    path = find_optional_prediction_frame_path(base_dir, cam_id, frame_idx)
    if path is None:
        return None
    arr = _load_prediction_view(path)
    _validate_prediction_view(arr, path)
    return _visibility_from_array(np.asarray(arr, dtype=float)).astype(bool).tolist()


def load_frame_points(base_dir: Path, cam_id: str, frame_idx: int) -> Optional[List[List[Tuple[float, float]]]]:
    path = find_optional_prediction_frame_path(base_dir, cam_id, frame_idx)
    if path is None:
        return None
    arr = _load_prediction_view(path)
    _validate_prediction_view(arr, path)
    return _array_to_points(np.asarray(arr, dtype=float))


def _sample_for(bundle: PredictionBundle, frame_idx: int, cam_id: str) -> PredictionSample:
    key = (str(cam_id), int(frame_idx))
    sample = PredictionSample(
        cam_id=str(cam_id),
        frame_idx=int(frame_idx),
        source_path=source_frame_path(bundle, frame_idx, cam_id),
        corrected_path=bundle.corrected_dir / str(cam_id) / _corrected_name(bundle.pred_dir, cam_id, frame_idx),
    )
    bundle.samples[key] = sample
    return sample


def _load_prediction_view(path: Path) -> np.ndarray:
    try:
        return np.load(path)
    except Exception as exc:
        raise ValueError(f"Failed to load prediction npy: {path}") from exc


def _load_corrected_view(path: Path) -> np.ndarray:
    try:
        return np.load(path)
    except Exception as exc:
        raise ValueError(f"Failed to load corrected npy: {path}") from exc


def view_state_from_bundle(
    bundle: PredictionBundle,
    frame_idx: int,
    cam_id: str,
    *,
    default_points: Optional[List[List[Tuple[float, float]]]] = None,
    default_visible: Optional[List[List[bool]]] = None,
) -> Tuple[List[List[Tuple[float, float]]], List[List[bool]]]:
    sample = _sample_for(bundle, frame_idx, cam_id)
    if sample.source_path is not None:
        pred = _load_prediction_view(sample.source_path)
        _validate_prediction_view(pred, sample.source_path)
        points_view = np.asarray(pred, dtype=float)
    else:
        points_view = _points_to_array(default_points) if default_points is not None else _empty_points_array()

    key = (sample.cam_id, sample.frame_idx)
    if key in bundle.pending:
        points_view = np.asarray(bundle.pending[key], dtype=float)
        visible = _visibility_from_array(points_view)
    elif default_visible is not None:
        visible = _visible_to_array(default_visible)
    elif sample.source_path is not None:
        visible = _visibility_from_array(points_view)
    else:
        visible = np.zeros((_HAND_COUNT, _JOINT_COUNT), dtype=bool)

    points = points_view.copy()
    if default_points is not None:
        fallback_view = _points_to_array(default_points)
        missing = _missing_points(points)
        points[missing] = fallback_view[missing]
    return _array_to_points(points), visible.astype(bool).tolist()


def apply_view_state_to_corrected(
    bundle: PredictionBundle,
    frame_idx: int,
    cam_id: str,
    points: List[List[Tuple[float, float]]],
    visible: List[List[bool]],
) -> None:
    sample = _sample_for(bundle, frame_idx, cam_id)

    if sample.corrected_path.exists() and sample.corrected_path.is_file():
        corrected = _load_corrected_view(sample.corrected_path)
        _validate_prediction_view(corrected, sample.corrected_path)
        dtype = corrected.dtype
    else:
        pred_path = find_optional_prediction_frame_path(bundle.prediction_dir, cam_id, frame_idx)
        if pred_path is not None:
            pred = _load_prediction_view(pred_path)
            _validate_prediction_view(pred, pred_path)
            dtype = _correction_dtype(pred)
        elif sample.source_path is not None:
            pred = _load_prediction_view(sample.source_path)
            _validate_prediction_view(pred, sample.source_path)
            dtype = _correction_dtype(pred)
        else:
            dtype = np.dtype("float32")

    pts = np.asarray(points, dtype=dtype)
    vis = np.asarray(visible, dtype=bool)
    if pts.shape != (_HAND_COUNT, _JOINT_COUNT, 2):
        raise ValueError(f"Expected points shape (2,21,2), got {pts.shape}.")
    if vis.shape != (_HAND_COUNT, _JOINT_COUNT):
        raise ValueError(f"Expected visibility shape (2,21), got {vis.shape}.")

    out = pts.copy()
    out[~vis] = -1
    bundle.pending[(sample.cam_id, sample.frame_idx)] = out


def _empty_points_array() -> np.ndarray:
    return np.zeros((_HAND_COUNT, _JOINT_COUNT, 2), dtype=float)


def _points_to_array(points: List[List[Tuple[float, float]]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.shape != (_HAND_COUNT, _JOINT_COUNT, 2):
        raise ValueError(f"Expected default points shape (2,21,2), got {arr.shape}.")
    return arr


def _visible_to_array(visible: List[List[bool]]) -> np.ndarray:
    arr = np.asarray(visible, dtype=bool)
    if arr.shape != (_HAND_COUNT, _JOINT_COUNT):
        raise ValueError(f"Expected default visibility shape (2,21), got {arr.shape}.")
    return arr


def _missing_points(arr: np.ndarray) -> np.ndarray:
    return np.logical_and(arr[:, :, 0] == -1, arr[:, :, 1] == -1)


def _visibility_from_array(arr: np.ndarray) -> np.ndarray:
    return ~_missing_points(arr)


def _array_to_points(arr: np.ndarray) -> List[List[Tuple[float, float]]]:
    return [
        [(float(arr[hand, joint, 0]), float(arr[hand, joint, 1])) for joint in range(_JOINT_COUNT)]
        for hand in range(_HAND_COUNT)
    ]


def save_corrected_array(bundle: PredictionBundle) -> None:
    for key, arr in list(bundle.pending.items()):
        sample = bundle.samples[key]
        p = sample.corrected_path
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f"{p.stem}_", suffix=".npy", dir=str(p.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, arr)
            os.replace(tmp_name, p)
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
            except Exception:
                pass
    bundle.pending.clear()


# Legacy helpers retained for locating frame counts in older datasets.
def list_camera_ids(task_dir: Path) -> List[str]:
    if not task_dir.exists() or not task_dir.is_dir():
        return []
    return sorted(p.name for p in task_dir.iterdir() if p.is_dir() and p.name.isdigit())


def compute_total_frames(task_dir: Path) -> int:
    cams = list_camera_ids(task_dir)
    if not cams:
        return 0
    rgb = task_dir / cams[0] / "RGB"
    if not rgb.exists() or not rgb.is_dir():
        return 0
    max_idx = -1
    for p in rgb.iterdir():
        if not p.is_file():
            continue
        m = _FRAME_RE.match(p.name)
        if not m:
            continue
        max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1 if max_idx >= 0 else 0
