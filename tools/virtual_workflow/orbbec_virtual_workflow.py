#!/usr/bin/env python3
"""Standalone virtual workflow companions for the Orbbec task backend.

The script intentionally talks to the backend only through HTTP.  It does not
import task_backend or label modules, so it can be used as an external test
fixture for the current server and frontends.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import struct
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

try:
    import mediapipe as mp
except ModuleNotFoundError:  # pragma: no cover - optional local demo dependency
    mp = None

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional local demo dependency
    np = None

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - optional local demo dependency
    Image = None


Json = Dict[str, Any]
_FRAME_RE = re.compile(r"^(\d+)\.[^.]+$")
_HAND_COUNT = 2
_JOINT_COUNT = 21
_FALLBACK_IMAGE_SIZE = (640, 480)
_HAND_TEMPLATE = (
    (0.00, 0.00),
    (-0.18, -0.14),
    (-0.34, -0.25),
    (-0.48, -0.34),
    (-0.62, -0.42),
    (-0.16, -0.38),
    (-0.19, -0.61),
    (-0.22, -0.82),
    (-0.24, -1.02),
    (0.00, -0.43),
    (0.00, -0.70),
    (0.00, -0.94),
    (0.00, -1.16),
    (0.16, -0.38),
    (0.21, -0.62),
    (0.26, -0.83),
    (0.31, -1.02),
    (0.30, -0.28),
    (0.43, -0.47),
    (0.54, -0.64),
    (0.64, -0.80),
)
MIN_DETECTION_LANDMARKS = 6
LANDMARK_NORMALIZED_MARGIN = 0.18


class BackendError(RuntimeError):
    pass


class NoJobAvailable(Exception):
    pass


def now_ms() -> int:
    return int(time.time() * 1000)


def clean_id(value: str, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_").lower()
    return text or fallback


def normalize_handedness(label: str) -> str:
    # Match interaction handGT: MediaPipe handedness is mirrored/selfie-oriented.
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def local_uri_from_path(path: Path) -> str:
    return "local://" + str(path.expanduser().resolve())


def path_from_local_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "local":
        raise ValueError(f"not a local URI: {uri}")
    if parsed.netloc and parsed.path:
        raw = "/" + parsed.netloc + parsed.path
    elif parsed.netloc:
        raw = parsed.netloc
    else:
        raw = parsed.path
    return Path(unquote(raw)).expanduser().resolve()


def uri_join(base_uri: str, *parts: str) -> str:
    base = str(base_uri or "").rstrip("/")
    clean = [str(part).strip("/") for part in parts if str(part or "").strip("/")]
    return base if not clean else base + "/" + "/".join(clean)


def print_event(event: str, **payload: Any) -> None:
    out = {"event": event, **payload}
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))
    sys.stdout.flush()


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = (base_url or "http://127.0.0.1:8765").rstrip("/")
        self.timeout = float(timeout)

    def get(self, path: str) -> Json:
        return self._request("GET", path, None)

    def post(self, path: str, body: Json) -> Json:
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: Optional[Json]) -> Json:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as exc:
            message = self._http_error_message(exc)
            if exc.code == 404 and (message.startswith("no queued ") or message.startswith("no pending manual segment")):
                raise NoJobAvailable(message) from exc
            raise BackendError(f"HTTP {exc.code}: {message}") from exc
        except (URLError, TimeoutError) as exc:
            raise BackendError(f"cannot reach backend {self.base_url}: {exc}") from exc
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise BackendError(f"backend returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise BackendError("backend JSON response must be an object")
        if parsed.get("error"):
            raise BackendError(str(parsed["error"]))
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
        return str(exc.reason)

    def create_manual_label_job(self, payload: Json) -> Json:
        return self.post("/api/v1/dev/label/jobs", payload)

    def create_dev_job(self, job_type: str, episode_id: str, payload: Json, episode: Optional[Json] = None) -> Json:
        body: Json = {
            "type": job_type,
            "episode_id": episode_id,
            "job_id": str(payload.get("job_id") or ""),
            "payload": payload,
        }
        if episode:
            body["episode"] = episode
        return self.post("/api/v1/dev/jobs", body)

    def lease_job(self, job_type: str, owner: str, lease_seconds: int = 300) -> Json:
        return self.post(
            "/api/v1/jobs/lease",
            {"type": job_type, "worker_id": owner, "lease_seconds": lease_seconds},
        )

    def heartbeat_job(self, job_id: str, owner: str, lease_seconds: int = 300) -> Json:
        return self.post(
            f"/api/v1/jobs/{quote(job_id, safe='')}/heartbeat",
            {"worker_id": owner, "lease_seconds": lease_seconds, "status": "running"},
        )

    def complete_job(self, job_id: str, result: Json, artifacts: Optional[List[Json]] = None) -> Json:
        body: Json = {"result": result}
        if artifacts:
            body["artifacts"] = artifacts
        return self.post(f"/api/v1/jobs/{quote(job_id, safe='')}/complete", body)

    def lease_label_job(self, operator_id: str, lease_seconds: int = 600) -> Json:
        return self.post(
            "/api/v1/label/segments/lease",
            {"operator_id": operator_id, "lease_seconds": lease_seconds},
        )

    def heartbeat_label_job(self, job_id: str, operator_id: str, lease_seconds: int = 600) -> Json:
        return self.post(
            f"/api/v1/label/segments/{quote(job_id, safe='')}/heartbeat",
            {"operator_id": operator_id, "lease_seconds": lease_seconds, "status": "running"},
        )

    def complete_label_job(self, job_id: str, result: Json, artifacts: Optional[List[Json]] = None) -> Json:
        body: Json = {"result": result}
        if artifacts:
            body["artifacts"] = artifacts
        return self.post(f"/api/v1/label/segments/{quote(job_id, safe='')}/complete", body)


@dataclass
class LabelTask:
    root: Path
    subject: str
    task: str
    episode: str
    cameras: List[str]
    frames: List[int]
    rgb_path_template: str = "{camera}/RGB/{frame:05d}.png"
    prediction_dir: str = "pred_2d"
    correction_dir: str = "corrected_2d"

    @property
    def episode_dir(self) -> Path:
        return self.root.expanduser().resolve() / self.subject / self.task / self.episode

    @property
    def episode_id(self) -> str:
        return clean_id(f"{self.subject}_{self.task}_{self.episode}", "episode")


def load_label_tasks(path: Path, limit: int = 0) -> List[LabelTask]:
    tasks: List[LabelTask] = []
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            text = raw.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: JSONL item must be an object")
            cameras = [str(item) for item in obj.get("cameras") or [] if str(item)]
            frames = [int(item) for item in obj.get("frames") or [] if not isinstance(item, bool)]
            if not cameras or not frames:
                raise ValueError(f"{path}:{line_no}: cameras and frames are required")
            tasks.append(
                LabelTask(
                    root=Path(str(obj.get("root") or ".")),
                    subject=str(obj.get("subject") or "subject"),
                    task=str(obj.get("task") or "task"),
                    episode=str(obj.get("episode") or f"episode_{line_no}"),
                    cameras=cameras,
                    frames=frames,
                    rgb_path_template=str(obj.get("rgb_path_template") or "{camera}/RGB/{frame:05d}.png"),
                    prediction_dir=str(obj.get("prediction_dir") or "pred_2d"),
                    correction_dir=str(obj.get("correction_dir") or "corrected_2d"),
                )
            )
            if limit and len(tasks) >= limit:
                break
    return tasks


def task_with_frames(task: LabelTask, frames: Sequence[int]) -> LabelTask:
    return LabelTask(
        root=task.root,
        subject=task.subject,
        task=task.task,
        episode=task.episode,
        cameras=list(task.cameras),
        frames=[int(frame) for frame in frames],
        rgb_path_template=task.rgb_path_template,
        prediction_dir=task.prediction_dir,
        correction_dir=task.correction_dir,
    )


def split_task_segments(task: LabelTask, frames_per_segment: int = 0) -> List[LabelTask]:
    size = int(frames_per_segment or 0)
    if size <= 0 or len(task.frames) <= size:
        return [task]
    return [task_with_frames(task, task.frames[i : i + size]) for i in range(0, len(task.frames), size)]


def write_placeholder_ppm(path: Path, width: int = 64, height: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    # A deterministic low-contrast checker keeps files tiny enough for tests and
    # still readable by Pillow/OpenCV when the real label UI opens them.
    rows = bytearray()
    for y in range(height):
        for x in range(width):
            v = 80 if ((x // 8) + (y // 8)) % 2 == 0 else 130
            rows.extend((v, v, v))
    path.write_bytes(header + bytes(rows))


def write_placeholder_png(path: Path, width: int = 64, height: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    scanlines = bytearray()
    for y in range(height):
        scanlines.append(0)
        for x in range(width):
            v = 80 if ((x // 8) + (y // 8)) % 2 == 0 else 130
            scanlines.extend((v, v, v))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(scanlines)))
        + chunk(b"IEND", b"")
    )


def write_placeholder_rgb_image(path: Path) -> None:
    if path.suffix.lower() == ".png":
        write_placeholder_png(path)
    else:
        write_placeholder_ppm(path)


def _npy_header(shape: Tuple[int, ...]) -> bytes:
    header = f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}"
    header_bytes = header.encode("latin1")
    pad = 16 - ((10 + len(header_bytes) + 1) % 16)
    return header_bytes + b" " * pad + b"\n"


def write_float32_npy(path: Path, values: Sequence[float], shape: Tuple[int, ...] = (_HAND_COUNT, _JOINT_COUNT, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = 1
    for dim in shape:
        expected *= int(dim)
    clean_values = [float(value) for value in values]
    if len(clean_values) != expected:
        raise ValueError(f"npy value count mismatch: expected {expected}, got {len(clean_values)}")
    full_header = _npy_header(shape)
    data = struct.pack("<" + "f" * expected, *clean_values)
    path.write_bytes(b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(full_header)) + full_header + data)


def _stable_rng(*parts: Any) -> random.Random:
    seed = zlib.crc32("|".join(str(part) for part in parts).encode("utf-8")) & 0xFFFFFFFF
    return random.Random(seed)


def read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"P6") or data.startswith(b"P3"):
        tokens: List[bytes] = []
        token = bytearray()
        in_comment = False
        for byte in data[2:256]:
            if in_comment:
                if byte in b"\r\n":
                    in_comment = False
                continue
            if byte == ord("#"):
                in_comment = True
                continue
            if chr(byte).isspace():
                if token:
                    tokens.append(bytes(token))
                    token.clear()
                    if len(tokens) >= 2:
                        break
                continue
            token.append(byte)
        if len(tokens) >= 2:
            try:
                return int(tokens[0]), int(tokens[1])
            except ValueError:
                return None
    if data.startswith(b"\xff\xd8"):
        idx = 2
        while idx + 9 < len(data):
            if data[idx] != 0xFF:
                idx += 1
                continue
            marker = data[idx + 1]
            idx += 2
            if marker in {0xD8, 0xD9}:
                continue
            if idx + 2 > len(data):
                return None
            length = struct.unpack(">H", data[idx:idx + 2])[0]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF} and length >= 7:
                height, width = struct.unpack(">HH", data[idx + 3:idx + 7])
                return width, height
            idx += max(2, length)
    return None


def find_rgb_frame_path(episode_dir: Path, cam: str, frame: int, rgb_path_template: str = "") -> Optional[Path]:
    candidates: List[Path] = []
    if rgb_path_template:
        try:
            candidates.append(episode_dir / rgb_path_template.format(camera=cam, frame=int(frame)))
        except Exception:
            pass
    for suffix in (".png", ".jpg", ".jpeg", ".ppm"):
        candidates.append(episode_dir / str(cam) / "RGB" / f"{int(frame):05d}{suffix}")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    rgb_dir = episode_dir / str(cam) / "RGB"
    if rgb_dir.exists() and rgb_dir.is_dir():
        prefix = f"{int(frame):05d}."
        for child in sorted(rgb_dir.iterdir()):
            if child.is_file() and child.name.startswith(prefix):
                return child
    return None


def image_size_for_frame(episode_dir: Path, cam: str, frame: int, rgb_path_template: str = "") -> Tuple[int, int]:
    path = find_rgb_frame_path(episode_dir, cam, frame, rgb_path_template)
    if path is not None:
        size = read_image_size(path)
        if size is not None and size[0] > 0 and size[1] > 0:
            return size
    return _FALLBACK_IMAGE_SIZE


def virtual_hand_values(cam: str, frame: int, width: int, height: int, variant: str = "pred") -> List[float]:
    rng = _stable_rng(cam, frame, variant)
    scale = max(18.0, min(float(width), float(height)) * 0.22)
    centers = ((0.40 * width, 0.70 * height), (0.62 * width, 0.70 * height))
    variant_noise = {"pred": 3.5, "mano": 2.0, "manual": 1.0}.get(variant, 2.5)
    values: List[float] = []
    for hand_idx, center in enumerate(centers):
        mirror = -1.0 if hand_idx == 1 else 1.0
        drift_x = ((int(frame) % 17) - 8) * 0.7
        drift_y = ((int(frame) % 11) - 5) * 0.45
        for joint_idx, (tx, ty) in enumerate(_HAND_TEMPLATE):
            x = center[0] + mirror * tx * scale + drift_x + rng.uniform(-variant_noise, variant_noise)
            y = center[1] + ty * scale + drift_y + rng.uniform(-variant_noise, variant_noise)
            # Quantize a little so the fixture looks like low-precision hand GT.
            quant = 4.0 if variant == "pred" else 2.0
            x = round(max(0.0, min(float(width - 1), x)) / quant) * quant
            y = round(max(0.0, min(float(height - 1), y)) / quant) * quant
            values.extend((x, y))
    return values


def write_virtual_hand_npy(
    path: Path,
    *,
    cam: str = "00",
    frame: int = 0,
    width: int = _FALLBACK_IMAGE_SIZE[0],
    height: int = _FALLBACK_IMAGE_SIZE[1],
    variant: str = "pred",
) -> None:
    write_float32_npy(path, virtual_hand_values(cam, frame, width, height, variant))


def write_frame_hand_npy(
    path: Path,
    episode_dir: Path,
    cam: str,
    frame: int,
    *,
    rgb_path_template: str = "",
    variant: str = "pred",
) -> None:
    width, height = image_size_for_frame(episode_dir, cam, frame, rgb_path_template)
    write_virtual_hand_npy(path, cam=cam, frame=frame, width=width, height=height, variant=variant)


def write_placeholder_npy(path: Path) -> None:
    write_virtual_hand_npy(path)


class InteractionHandGtDetector:
    def __init__(self):
        self.available = mp is not None and np is not None and Image is not None
        self._hands = None
        if self.available:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.25,
                min_tracking_confidence=0.25,
            )

    def close(self) -> None:
        hands = self._hands
        self._hands = None
        if hands is not None:
            close = getattr(hands, "close", None)
            if close is not None:
                close()

    def detect_values(
        self,
        episode_dir: Path,
        cam: str,
        frame: int,
        *,
        rgb_path_template: str = "",
    ) -> Optional[List[float]]:
        if self._hands is None or np is None or Image is None:
            return None
        rgb_path = find_rgb_frame_path(episode_dir, cam, int(frame), rgb_path_template)
        if rgb_path is None:
            return None
        try:
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except Exception:
            return None
        if rgb.ndim != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
            return None

        result = self._hands.process(rgb)
        landmarks_list = result.multi_hand_landmarks or []
        handedness_list = result.multi_handedness or []
        if not landmarks_list:
            return None

        width = float(rgb.shape[1])
        height = float(rgb.shape[0])
        values = [-1.0] * (_HAND_COUNT * _JOINT_COUNT * 2)
        used_slots = set()
        detections: List[Tuple[float, int, List[Tuple[float, float]]]] = []
        for idx, hand_landmarks in enumerate(landmarks_list[:_HAND_COUNT]):
            side = ""
            score = 0.0
            if idx < len(handedness_list) and handedness_list[idx].classification:
                cls = handedness_list[idx].classification[0]
                side = normalize_handedness(str(cls.label))
                score = float(cls.score)

            points: List[Tuple[float, float]] = []
            valid_count = 0
            for lm in hand_landmarks.landmark[:_JOINT_COUNT]:
                valid = (
                    (-LANDMARK_NORMALIZED_MARGIN) <= float(lm.x) <= (1.0 + LANDMARK_NORMALIZED_MARGIN)
                    and (-LANDMARK_NORMALIZED_MARGIN) <= float(lm.y) <= (1.0 + LANDMARK_NORMALIZED_MARGIN)
                )
                if valid:
                    valid_count += 1
                    points.append((float(lm.x) * width, float(lm.y) * height))
                else:
                    points.append((-1.0, -1.0))
            if valid_count < MIN_DETECTION_LANDMARKS:
                continue

            preferred_slot = 0 if side == "Left" else 1 if side == "Right" else idx
            detections.append((score, preferred_slot, points))

        for _score, preferred_slot, points in sorted(detections, key=lambda item: item[0], reverse=True):
            slot = preferred_slot
            if slot in used_slots:
                free = [candidate for candidate in range(_HAND_COUNT) if candidate not in used_slots]
                if not free:
                    continue
                slot = free[0]
            used_slots.add(slot)
            base_index = slot * _JOINT_COUNT * 2
            for joint_idx, (x, y) in enumerate(points[:_JOINT_COUNT]):
                values[base_index + joint_idx * 2] = float(x)
                values[base_index + joint_idx * 2 + 1] = float(y)
        return values if used_slots else None


def hand_gt_detector_for_args(args: argparse.Namespace) -> InteractionHandGtDetector:
    detector = getattr(args, "_interaction_hand_gt_detector", None)
    if detector is None:
        detector = InteractionHandGtDetector()
        setattr(args, "_interaction_hand_gt_detector", detector)
        print_event(
            "interaction_handgt",
            available=bool(detector.available),
            mode="mediapipe_lightweight" if detector.available else "fallback_low_precision",
        )
    return detector


def copy_or_write_projected_npy(
    out_path: Path,
    source_path: Optional[Path],
    episode_dir: Path,
    cam: str,
    frame: int,
    *,
    rgb_path_template: str = "",
    variant: str = "mano",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path is not None and source_path.exists() and source_path.is_file():
        shutil.copyfile(source_path, out_path)
        return
    write_frame_hand_npy(out_path, episode_dir, cam, frame, rgb_path_template=rgb_path_template, variant=variant)


def copy_required_projected_npy(out_path: Path, source_path: Path, cam: str, frame: int) -> None:
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"manual_2d npy is required for segment MANO camera={cam} frame={frame}: {source_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, out_path)


class NasSimulator:
    def __init__(self, root: Path, uri_prefix: str = "nas://orbbec-virtual"):
        self.root = root.expanduser().resolve()
        self.uri_prefix = uri_prefix.rstrip("/")
        self.root.mkdir(parents=True, exist_ok=True)

    def local_path_for_uri(self, uri: str) -> Path:
        uri = str(uri or "").rstrip("/")
        if not uri.startswith(self.uri_prefix):
            raise ValueError(f"URI does not belong to this virtual NAS: {uri}")
        suffix = uri[len(self.uri_prefix):].lstrip("/")
        return (self.root / unquote(suffix)).resolve()

    def task_uri(self, subject: str, task: str, episode: str) -> str:
        return uri_join(self.uri_prefix, clean_id(subject), clean_id(task), clean_id(episode))

    def materialize_task(
        self,
        task: LabelTask,
        *,
        copy_source: bool = False,
        max_frames: int = 0,
        materialize_predictions: bool = True,
    ) -> str:
        uri = self.task_uri(task.subject, task.task, task.episode)
        dst = self.local_path_for_uri(uri)
        self._materialize_episode_dir(
            dst,
            subject=task.subject,
            task_name=task.task,
            episode=task.episode,
            cameras=task.cameras,
            frames=task.frames,
            rgb_path_template=task.rgb_path_template,
            prediction_dir=task.prediction_dir,
            source=task.episode_dir if copy_source else None,
            max_frames=max_frames,
            materialize_predictions=materialize_predictions,
        )
        return uri

    def materialize_payload(
        self,
        payload: Json,
        episode: Optional[Json] = None,
        *,
        copy_source: bool = False,
        max_frames: int = 0,
        materialize_predictions: bool = True,
    ) -> str:
        episode = episode or {}
        episode_id = str(payload.get("episode_id") or episode.get("episode_id") or f"episode_{uuid.uuid4().hex[:8]}")
        subject = str(payload.get("subject_id") or episode.get("subject_id") or "virtual_subject")
        task_name = str(payload.get("task_name") or episode.get("task_name") or "virtual_task")
        episode_name = str(payload.get("episode") or episode.get("episode_name") or episode_id)
        cameras = cameras_from_payload(payload, episode, self)
        frames = frames_from_payload(payload, episode, self, cameras)
        source = source_path_from_payload(payload, episode, self) if copy_source else None
        uri = self.task_uri(subject, task_name, episode_name)
        dst = self.local_path_for_uri(uri)
        self._materialize_episode_dir(
            dst,
            subject=subject,
            task_name=task_name,
            episode=episode_name,
            cameras=cameras,
            frames=frames,
            rgb_path_template=str(payload.get("rgb_path_template") or "{camera}/RGB/{frame:05d}.ppm"),
            prediction_dir=str(payload.get("prediction_dir") or "pred_2d"),
            source=source,
            max_frames=max_frames,
            materialize_predictions=materialize_predictions,
        )
        return uri

    def _materialize_episode_dir(
        self,
        dst: Path,
        *,
        subject: str,
        task_name: str,
        episode: str,
        cameras: Sequence[str],
        frames: Sequence[int],
        rgb_path_template: str,
        prediction_dir: str,
        source: Optional[Path],
        max_frames: int,
        materialize_predictions: bool,
    ) -> None:
        if source and source.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(source, dst)
        dst.mkdir(parents=True, exist_ok=True)
        selected_frames = list(frames)
        if max_frames > 0:
            selected_frames = selected_frames[:max_frames]
        for cam in cameras:
            for frame in selected_frames:
                try:
                    rel = rgb_path_template.format(camera=cam, frame=int(frame))
                except Exception:
                    rel = f"{cam}/RGB/{int(frame):05d}.ppm"
                if not Path(rel).suffix:
                    rel += ".ppm"
                rgb_path = dst / rel
                if not rgb_path.exists():
                    write_placeholder_rgb_image(rgb_path)
                if materialize_predictions:
                    pred_path = dst / prediction_dir / cam / f"{int(frame):05d}.npy"
                    if not pred_path.exists():
                        write_frame_hand_npy(pred_path, dst, cam, int(frame), rgb_path_template=rgb_path_template, variant="pred")
        metadata = {
            "virtual_nas": True,
            "subject_id": subject,
            "task_name": task_name,
            "episode": episode,
            "cameras": list(cameras),
            "frames": list(frames),
            "materialized_frames": selected_frames,
            "created_at_ms": now_ms(),
        }
        (dst / "virtual_episode.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def write_prediction_artifact(
        self,
        data_uri: str,
        cameras: Sequence[str],
        frames: Sequence[int],
        prediction_dir: str = "pred_2d",
        rgb_path_template: str = "",
        detector: Optional[InteractionHandGtDetector] = None,
    ) -> str:
        base = self.local_path_for_uri(data_uri)
        for cam in cameras or ["00"]:
            for frame in frames or [0]:
                out = base / prediction_dir / str(cam) / f"{int(frame):05d}.npy"
                values = detector.detect_values(base, str(cam), int(frame), rgb_path_template=rgb_path_template) if detector is not None else None
                if values is not None:
                    write_float32_npy(out, values)
                else:
                    write_frame_hand_npy(out, base, str(cam), int(frame), rgb_path_template=rgb_path_template, variant="pred")
        return uri_join(data_uri, prediction_dir)

    def write_mano_episode_artifact(
        self,
        data_uri: str,
        cameras: Sequence[str],
        frames: Sequence[int],
        prediction_dir: str = "pred_2d",
        rgb_path_template: str = "",
    ) -> str:
        base = self.local_path_for_uri(data_uri)
        out = base / "mano" / "episode"
        out.mkdir(parents=True, exist_ok=True)
        (out / "mano_episode.json").write_text(
            json.dumps({"mock": True, "frames": list(frames), "cameras": list(cameras)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for cam in cameras or ["00"]:
            for frame in frames or [0]:
                source = base / prediction_dir / str(cam) / f"{int(frame):05d}.npy"
                copy_or_write_projected_npy(out / "projected_2d" / str(cam) / f"{int(frame):05d}.npy", source, base, str(cam), int(frame), rgb_path_template=rgb_path_template, variant="mano")
        return uri_join(data_uri, "mano", "episode")

    def write_mano_segment_patch(
        self,
        data_uri: str,
        segment_id: str,
        cameras: Sequence[str],
        frames: Sequence[int],
        manual_2d_dir: str = "",
        manual_2d_uri: str = "",
        rgb_path_template: str = "",
    ) -> str:
        base = self.local_path_for_uri(data_uri)
        out = base / "mano" / "segments" / clean_id(segment_id, "segment")
        out.mkdir(parents=True, exist_ok=True)
        (out / "mano_patch.json").write_text(
            json.dumps({"mock": True, "segment_id": segment_id, "frames": list(frames), "cameras": list(cameras)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        source_dir = self.manual_2d_source_path(data_uri, segment_id, manual_2d_dir, manual_2d_uri)
        for cam in cameras or ["00"]:
            for frame in frames or [0]:
                source = source_dir / str(cam) / f"{int(frame):05d}.npy"
                copy_required_projected_npy(out / "projected_2d" / str(cam) / f"{int(frame):05d}.npy", source, str(cam), int(frame))
        return uri_join(data_uri, "mano", "segments", segment_id)

    def manual_2d_source_path(self, data_uri: str, segment_id: str, manual_2d_dir: str = "", manual_2d_uri: str = "") -> Path:
        data_uri = str(data_uri or "").rstrip("/")
        manual_2d_uri = str(manual_2d_uri or "").rstrip("/")
        if manual_2d_uri:
            if data_uri and manual_2d_uri.startswith(data_uri + "/"):
                return self.local_path_for_uri(data_uri) / manual_2d_uri[len(data_uri):].lstrip("/")
            if manual_2d_uri.startswith(self.uri_prefix):
                return self.local_path_for_uri(manual_2d_uri)
            if manual_2d_uri.startswith("local://"):
                return path_from_local_uri(manual_2d_uri)
        return self.local_path_for_uri(data_uri) / (manual_2d_dir or f"manual_2d/segments/{segment_id}")


def as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def as_int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            pass
    return out


def discover_cameras(episode_dir: Optional[Path]) -> List[str]:
    if episode_dir is None or not episode_dir.exists() or not episode_dir.is_dir():
        return []
    cameras: List[str] = []
    for child in episode_dir.iterdir():
        if child.is_dir() and (child / "RGB").is_dir():
            cameras.append(child.name)
    return sorted(cameras)


def discover_frames(episode_dir: Optional[Path], cameras: Sequence[str]) -> List[int]:
    if episode_dir is None or not cameras:
        return []
    rgb_dir = episode_dir / str(cameras[0]) / "RGB"
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


def source_path_from_payload(
    payload: Json,
    episode: Optional[Json] = None,
    nas: Optional[NasSimulator] = None,
) -> Optional[Path]:
    episode = episode or {}
    raw = str(
        payload.get("resolved_data_path")
        or episode.get("resolved_data_path")
        or payload.get("local_episode_path")
        or episode.get("local_episode_path")
        or payload.get("local_capture_path")
        or episode.get("local_capture_path")
        or payload.get("local_path")
        or ""
    )
    if raw:
        return Path(raw).expanduser().resolve()
    data_uri = str(payload.get("data_uri") or episode.get("data_uri") or "")
    if data_uri.startswith("local://"):
        return path_from_local_uri(data_uri)
    if nas is not None and data_uri.startswith(nas.uri_prefix):
        return nas.local_path_for_uri(data_uri)
    return None


def cameras_from_payload(payload: Json, episode: Optional[Json], nas: NasSimulator) -> List[str]:
    cameras = as_string_list(payload.get("cameras")) or as_string_list((episode or {}).get("cameras"))
    if cameras:
        return cameras
    return discover_cameras(source_path_from_payload(payload, episode, nas)) or ["00"]


def frames_from_payload(payload: Json, episode: Optional[Json], nas: NasSimulator, cameras: Sequence[str]) -> List[int]:
    frames = as_int_list(payload.get("frames")) or discover_frames(source_path_from_payload(payload, episode, nas), cameras)
    if frames:
        return frames
    frame_count = payload.get("frame_count") or (episode or {}).get("frame_count")
    try:
        count = int(frame_count)
    except (TypeError, ValueError):
        count = 0
    return list(range(max(1, count or 1)))


def write_prediction_artifact_for_payload(
    nas: NasSimulator,
    payload: Json,
    episode: Optional[Json],
    cameras: Sequence[str],
    frames: Sequence[int],
    prediction_dir: str,
    detector: Optional[InteractionHandGtDetector] = None,
) -> str:
    data_uri = str(payload.get("data_uri") or "")
    rgb_path_template = str(payload.get("rgb_path_template") or "{camera}/RGB/{frame:05d}.png")
    if data_uri.startswith(nas.uri_prefix):
        return nas.write_prediction_artifact(data_uri, cameras, frames, prediction_dir, rgb_path_template, detector)
    episode_path = source_path_from_payload(payload, episode, nas)
    if episode_path is not None:
        for cam in cameras or ["00"]:
            for frame in frames or [0]:
                out = episode_path / prediction_dir / str(cam) / f"{int(frame):05d}.npy"
                values = detector.detect_values(episode_path, str(cam), int(frame), rgb_path_template=rgb_path_template) if detector is not None else None
                if values is not None:
                    write_float32_npy(out, values)
                else:
                    write_frame_hand_npy(out, episode_path, str(cam), int(frame), rgb_path_template=rgb_path_template, variant="pred")
        return uri_join(data_uri or local_uri_from_path(episode_path), prediction_dir)
    return uri_join(data_uri, prediction_dir)


def write_mano_artifact_for_payload(
    nas: NasSimulator,
    payload: Json,
    episode: Optional[Json],
    cameras: Sequence[str],
    frames: Sequence[int],
) -> Tuple[str, str]:
    data_uri = str(payload.get("data_uri") or "")
    scope = str(payload.get("scope") or payload.get("mano_scope") or "episode")
    segment_id = str(payload.get("segment_id") or "")
    prediction_dir = str(payload.get("prediction_dir") or "pred_2d")
    manual_2d_dir = str(payload.get("manual_2d_dir") or f"manual_2d/segments/{segment_id}")
    manual_2d_uri = str(payload.get("manual_2d_uri") or payload.get("input_2d_uri") or "")
    rgb_path_template = str(payload.get("rgb_path_template") or "{camera}/RGB/{frame:05d}.png")
    if data_uri.startswith(nas.uri_prefix):
        if scope == "segment":
            return "mano_segment_patch", nas.write_mano_segment_patch(data_uri, segment_id, cameras, frames, manual_2d_dir, manual_2d_uri, rgb_path_template)
        return "mano_episode", nas.write_mano_episode_artifact(data_uri, cameras, frames, prediction_dir, rgb_path_template)
    episode_path = source_path_from_payload(payload, episode, nas)
    if episode_path is not None:
        if scope == "segment":
            out = episode_path / "mano" / "segments" / clean_id(segment_id, "segment")
            out.mkdir(parents=True, exist_ok=True)
            (out / "mano_patch.json").write_text(json.dumps({"mock": True, "segment_id": segment_id, "frames": list(frames), "cameras": list(cameras)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            source_dir = manual_2d_source_path_for_payload(nas, data_uri, episode_path, segment_id, manual_2d_dir, manual_2d_uri)
            for cam in cameras or ["00"]:
                for frame in frames or [0]:
                    source = source_dir / str(cam) / f"{int(frame):05d}.npy"
                    copy_required_projected_npy(out / "projected_2d" / str(cam) / f"{int(frame):05d}.npy", source, str(cam), int(frame))
            return "mano_segment_patch", uri_join(data_uri or local_uri_from_path(episode_path), "mano", "segments", segment_id)
        out = episode_path / "mano" / "episode"
        out.mkdir(parents=True, exist_ok=True)
        (out / "mano_episode.json").write_text(json.dumps({"mock": True, "frames": list(frames), "cameras": list(cameras)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for cam in cameras or ["00"]:
            for frame in frames or [0]:
                source = episode_path / prediction_dir / str(cam) / f"{int(frame):05d}.npy"
                copy_or_write_projected_npy(out / "projected_2d" / str(cam) / f"{int(frame):05d}.npy", source, episode_path, str(cam), int(frame), rgb_path_template=rgb_path_template, variant="mano")
        return "mano_episode", uri_join(data_uri or local_uri_from_path(episode_path), "mano", "episode")
    if scope == "segment":
        return "mano_segment_patch", uri_join(data_uri, "mano", "segments", segment_id)
    return "mano_episode", uri_join(data_uri, "mano", "episode")


def manual_2d_source_path_for_payload(
    nas: NasSimulator,
    data_uri: str,
    episode_path: Path,
    segment_id: str,
    manual_2d_dir: str,
    manual_2d_uri: str,
) -> Path:
    data_uri = str(data_uri or "").rstrip("/")
    manual_2d_uri = str(manual_2d_uri or "").rstrip("/")
    if manual_2d_uri:
        if data_uri and manual_2d_uri.startswith(data_uri + "/"):
            return episode_path / manual_2d_uri[len(data_uri):].lstrip("/")
        if manual_2d_uri.startswith(nas.uri_prefix):
            return nas.local_path_for_uri(manual_2d_uri)
        if manual_2d_uri.startswith("local://"):
            return path_from_local_uri(manual_2d_uri)
    return episode_path / (manual_2d_dir or f"manual_2d/segments/{segment_id}")


def write_qc_report_for_payload(nas: NasSimulator, payload: Json, episode: Optional[Json], result: Json) -> str:
    data_uri = str(payload.get("data_uri") or (episode or {}).get("data_uri") or "")
    report = {
        "mock": True,
        "generated_at_ms": now_ms(),
        "episode_id": payload.get("episode_id") or (episode or {}).get("episode_id") or "",
        "passed": bool(result.get("passed")),
        "qc_passed": bool(result.get("qc_passed")),
        "score": result.get("score"),
        "segments": result.get("segments") if isinstance(result.get("segments"), list) else [],
        "reason": str(result.get("reason") or ""),
        "virtual_worker": str(result.get("virtual_worker") or ""),
    }
    if data_uri.startswith(nas.uri_prefix):
        out = nas.local_path_for_uri(data_uri) / "qc" / "qc_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return uri_join(data_uri, "qc", "qc_report.json")
    episode_path = source_path_from_payload(payload, episode, nas)
    if episode_path is not None:
        out = episode_path / "qc" / "qc_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return uri_join(data_uri or local_uri_from_path(episode_path), "qc", "qc_report.json")
    return uri_join(data_uri, "qc", "qc_report.json")


def payload_from_task(task: LabelTask, data_uri: str, job_id: str, reason: str) -> Json:
    return {
        "job_id": job_id,
        "episode_id": task.episode_id,
        "subject_id": task.subject,
        "task_name": task.task,
        "data_uri": data_uri,
        "cameras": task.cameras,
        "frames": task.frames,
        "rgb_path_template": task.rgb_path_template,
        "prediction_dir": task.prediction_dir,
        "correction_dir": task.correction_dir,
        "reason": reason,
        "metadata": {"source": "virtual_workflow", "label_jsonl_episode": task.episode},
    }


def seed_manual_label_jobs(args: argparse.Namespace) -> int:
    client = BackendClient(args.backend_url, timeout=args.timeout)
    nas = NasSimulator(args.nas_root, args.nas_uri_prefix)
    count = 0
    stop = False
    for task in load_label_tasks(args.jsonl, args.limit):
        segments = split_task_segments(task, args.frames_per_segment)
        for segment_index, segment_task in enumerate(segments, 1):
            if args.max_jobs and count >= args.max_jobs:
                stop = True
                break
            if args.use_nas:
                data_uri = nas.materialize_task(segment_task, copy_source=args.copy_source, max_frames=args.max_materialized_frames)
                local_path = ""
            else:
                data_uri = local_uri_from_path(task.episode_dir)
                local_path = str(task.episode_dir)
            suffix = f"_s{segment_index:04d}" if len(segments) > 1 else ""
            job_id = f"{clean_id(args.job_prefix)}_{task.episode_id}{suffix}"
            body = payload_from_task(segment_task, data_uri, job_id, "seeded_from_label_jsonl")
            body["payload"] = {
                "segment_index": segment_index,
                "segment_count": len(segments),
                "frames_per_segment": int(args.frames_per_segment or 0),
            }
            if local_path:
                body["local_path"] = local_path
            result = client.create_manual_label_job(body)
            segment = result.get("segment", {})
            count += 1
            print_event(
                "manual_segment_seeded",
                segment_id=segment.get("segment_id", job_id),
                episode_id=task.episode_id,
                data_uri=data_uri,
                frames=len(segment_task.frames),
                segment_index=segment_index,
                segment_count=len(segments),
                status=segment.get("status"),
            )
        if stop:
            break
    print_event("manual_segment_seed_done", count=count)
    return 0


def seed_captured_upload_jobs(args: argparse.Namespace) -> int:
    client = BackendClient(args.backend_url, timeout=args.timeout)
    count = 0
    for task in load_label_tasks(args.jsonl, args.limit):
        episode = {
            "episode_id": task.episode_id,
            "subject_id": task.subject,
            "task_name": task.task,
            "episode_index": None,
            "status": "captured",
            "data_uri": local_uri_from_path(task.episode_dir),
            "local_capture_path": str(task.episode_dir),
            "frame_count": len(task.frames),
            "cameras": task.cameras,
            "metadata": {"source": "virtual_seed_captured", "label_jsonl_episode": task.episode},
        }
        job_id = f"{clean_id(args.job_prefix)}_{task.episode_id}"
        payload = {
            "job_id": job_id,
            "episode_id": task.episode_id,
            "subject_id": task.subject,
            "task_name": task.task,
            "data_uri": episode["data_uri"],
            "local_capture_path": str(task.episode_dir),
            "cameras": task.cameras,
            "frames": task.frames,
            "rgb_path_template": task.rgb_path_template,
            "prediction_dir": task.prediction_dir,
            "correction_dir": task.correction_dir,
            "reason": "virtual_captured_seed",
        }
        result = client.create_dev_job("upload", task.episode_id, payload, episode=episode)
        count += 1
        print_event(
            "upload_seeded",
            job_id=result.get("job", {}).get("job_id", job_id),
            episode_id=task.episode_id,
            status=result.get("job", {}).get("status"),
        )
    print_event("upload_seed_done", count=count)
    return 0


def enriched_payload(response: Json) -> Json:
    payload = dict(response.get("payload") or {})
    job = response.get("job") or response.get("segment") or {}
    episode = response.get("episode") or {}
    for key in ("job_id", "segment_id", "episode_id"):
        payload.setdefault(key, job.get(key) or episode.get(key))
    for key in (
        "subject_id",
        "task_name",
        "data_uri",
        "episode_base_uri",
        "local_capture_path",
        "resolved_data_path",
        "cameras",
        "frames",
        "start_frame",
        "end_frame",
        "frame_count",
        "rgb_path_template",
        "prediction_dir",
        "correction_dir",
        "manual_2d_output_uri",
        "manual_2d_uri",
        "mano_episode_uri",
    ):
        if payload.get(key) in (None, "", []):
            payload[key] = episode.get(key)
    return payload


def handle_upload_once(client: BackendClient, nas: NasSimulator, args: argparse.Namespace) -> bool:
    owner = args.worker_id or f"virtual_upload_{os.getpid()}"
    try:
        leased = client.lease_job("upload", owner, args.lease_seconds)
    except NoJobAvailable:
        return False
    job = leased.get("job") or {}
    episode = leased.get("episode") or {}
    payload = enriched_payload(leased)
    client.heartbeat_job(str(job["job_id"]), owner, args.lease_seconds)
    nas_uri = nas.materialize_payload(
        payload,
        episode,
        copy_source=args.copy_source,
        max_frames=args.max_materialized_frames,
        materialize_predictions=False,
    )
    cameras = cameras_from_payload(payload, episode, nas)
    frames = frames_from_payload(payload, episode, nas, cameras)
    client.complete_job(
        str(job["job_id"]),
        {
            "ok": True,
            "nas_uri": nas_uri,
            "virtual_worker": owner,
            "copied_from": str(source_path_from_payload(payload, episode, nas) or ""),
        },
        artifacts=[{"kind": "nas_episode", "uri": nas_uri, "metadata": {"worker_id": owner}}],
    )
    print_event(
        "upload_completed",
        job_id=job["job_id"],
        episode_id=payload.get("episode_id") or episode.get("episode_id"),
        nas_uri=nas_uri,
        auto_label="manual_push_required",
        cameras=len(cameras),
        frames=len(frames),
    )
    return True


def handle_auto_label_once(client: BackendClient, nas: NasSimulator, args: argparse.Namespace) -> bool:
    owner = args.worker_id or f"virtual_auto_label_{os.getpid()}"
    try:
        leased = client.lease_job("auto_label", owner, args.lease_seconds)
    except NoJobAvailable:
        return False
    job = leased.get("job") or {}
    episode = leased.get("episode") or {}
    payload = enriched_payload(leased)
    client.heartbeat_job(str(job["job_id"]), owner, args.lease_seconds)
    data_uri = str(payload.get("data_uri") or "")
    cameras = cameras_from_payload(payload, episode, nas)
    frames = frames_from_payload(payload, episode, nas, cameras)
    if args.max_materialized_frames > 0:
        frames = frames[: args.max_materialized_frames]
    prediction_dir = str(payload.get("prediction_dir") or "pred_2d")
    pred_uri = write_prediction_artifact_for_payload(nas, payload, episode, cameras, frames, prediction_dir, hand_gt_detector_for_args(args))
    client.complete_job(
        str(job["job_id"]),
        {"ok": True, "model": "virtual_hand2d", "frames_predicted": frames, "virtual_worker": owner},
        artifacts=[{"kind": "pred_2d", "uri": pred_uri, "metadata": {"worker_id": owner, "mock": True}}],
    )
    print_event("auto_label_completed", job_id=job["job_id"], episode_id=payload.get("episode_id"), pred_uri=pred_uri, next_job="queued_by_backend")
    return True


def handle_mano_opt_once(client: BackendClient, nas: NasSimulator, args: argparse.Namespace) -> bool:
    owner = args.worker_id or f"virtual_mano_opt_{os.getpid()}"
    try:
        leased = client.lease_job("mano_opt", owner, args.lease_seconds)
    except NoJobAvailable:
        return False
    job = leased.get("job") or {}
    episode = leased.get("episode") or {}
    payload = enriched_payload(leased)
    client.heartbeat_job(str(job["job_id"]), owner, args.lease_seconds)
    cameras = cameras_from_payload(payload, episode, nas)
    frames = frames_from_payload(payload, episode, nas, cameras)
    if args.max_materialized_frames > 0:
        frames = frames[: args.max_materialized_frames]
    kind, uri = write_mano_artifact_for_payload(nas, payload, episode, cameras, frames)
    scope = str(payload.get("scope") or payload.get("mano_scope") or "episode")
    client.complete_job(
        str(job["job_id"]),
        {
            "ok": True,
            "scope": scope,
            "segment_id": payload.get("segment_id") or "",
            "frames_optimized": frames,
            "virtual_worker": owner,
            "output_uri": uri,
        },
        artifacts=[{"kind": kind, "uri": uri, "metadata": {"worker_id": owner, "mock": True, "scope": scope}}],
    )
    print_event(
        "mano_opt_completed",
        job_id=job["job_id"],
        episode_id=payload.get("episode_id"),
        segment_id=payload.get("segment_id") or "",
        scope=scope,
        uri=uri,
    )
    return True


def handle_qc_once(client: BackendClient, nas: NasSimulator, args: argparse.Namespace) -> bool:
    owner = args.worker_id or f"virtual_qc_{os.getpid()}"
    try:
        leased = client.lease_job("qc", owner, args.lease_seconds)
    except NoJobAvailable:
        return False
    job = leased.get("job") or {}
    episode = leased.get("episode") or {}
    payload = enriched_payload(leased)
    client.heartbeat_job(str(job["job_id"]), owner, args.lease_seconds)
    passed = random.random() >= float(args.qc_fail_rate)
    frames = frames_from_payload(payload, episode, nas, cameras_from_payload(payload, episode, nas))
    if args.max_materialized_frames > 0:
        frames = frames[: args.max_materialized_frames]
    failed_segments = [] if passed else virtual_failed_segments(frames)
    result = {
        "passed": passed,
        "qc_passed": passed,
        "score": round(random.random(), 4),
        "segments": failed_segments,
        "reason": "virtual_qc_pass" if passed else "virtual_qc_failed_needs_manual_segments",
        "virtual_worker": owner,
    }
    qc_report_uri = write_qc_report_for_payload(nas, payload, episode, result)
    client.complete_job(
        str(job["job_id"]),
        result,
        artifacts=[{"kind": "qc_report", "uri": qc_report_uri, "metadata": {"passed": passed, "worker_id": owner}}],
    )
    if passed:
        print_event("qc_passed", job_id=job["job_id"], episode_id=payload.get("episode_id"))
    else:
        print_event(
            "qc_failed_manual_segments_created",
            job_id=job["job_id"],
            episode_id=payload.get("episode_id"),
            segments=failed_segments,
        )
    return True


def virtual_failed_segments(frames: Sequence[int]) -> List[Json]:
    clean_frames = sorted({int(frame) for frame in frames if not isinstance(frame, bool)})
    if not clean_frames:
        return [{"start_frame": 0, "end_frame": 0, "reason": "virtual_qc_failed"}]
    target_count = min(random.randint(2, 3), max(1, len(clean_frames) // 10)) if len(clean_frames) >= 20 else 1
    segments: List[Json] = []
    used_ranges: List[Tuple[int, int]] = []
    attempts = 0
    while len(segments) < target_count and attempts < 80:
        attempts += 1
        length = min(random.randint(10, 20), len(clean_frames))
        start_idx = random.randint(0, max(0, len(clean_frames) - length))
        end_idx = min(len(clean_frames) - 1, start_idx + length - 1)
        start_frame = int(clean_frames[start_idx])
        end_frame = int(clean_frames[end_idx])
        overlaps = any(not (end_frame < used_start or start_frame > used_end) for used_start, used_end in used_ranges)
        if overlaps and len(clean_frames) >= target_count * 12:
            continue
        used_ranges.append((start_frame, end_frame))
        segments.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "reason": random.choice(["virtual_low_score", "virtual_temporal_jump", "virtual_projection_error"]),
                "score": round(random.uniform(0.05, 0.45), 4),
            }
        )
    segments.sort(key=lambda item: (int(item["start_frame"]), int(item["end_frame"])))
    return segments or [{"start_frame": clean_frames[0], "end_frame": clean_frames[min(len(clean_frames) - 1, 9)], "reason": "virtual_qc_failed"}]


WORKER_HANDLERS = {
    "upload": handle_upload_once,
    "auto-label": handle_auto_label_once,
    "mano-opt": handle_mano_opt_once,
    "qc": handle_qc_once,
}


def parse_workers(raw: str) -> List[str]:
    if raw == "default":
        return ["upload", "auto-label", "mano-opt", "qc"]
    if raw == "all":
        return ["upload", "auto-label", "mano-opt", "qc"]
    workers = [part.strip() for part in raw.split(",") if part.strip()]
    bad = [worker for worker in workers if worker not in WORKER_HANDLERS]
    if bad:
        raise ValueError(f"unknown workers: {', '.join(bad)}")
    return workers


def run_workers(args: argparse.Namespace) -> int:
    client = BackendClient(args.backend_url, timeout=args.timeout)
    nas = NasSimulator(args.nas_root, args.nas_uri_prefix)
    workers = parse_workers(args.workers)
    idle_rounds = 0
    processed = 0
    iteration = 0
    while True:
        iteration += 1
        did_work = False
        for worker in workers:
            handler = WORKER_HANDLERS[worker]
            try:
                did_one = handler(client, nas, args)
            except NoJobAvailable:
                did_one = False
            if did_one:
                did_work = True
                processed += 1
                if args.once:
                    print_event("worker_run_done", processed=processed, reason="once")
                    return 0
        if did_work:
            idle_rounds = 0
        else:
            idle_rounds += 1
            print_event("worker_idle", iteration=iteration, idle_rounds=idle_rounds, workers=workers)
        if args.max_iterations and iteration >= args.max_iterations:
            print_event("worker_run_done", processed=processed, reason="max_iterations")
            return 0
        if args.stop_after_idle_rounds and idle_rounds >= args.stop_after_idle_rounds:
            print_event("worker_run_done", processed=processed, reason="idle")
            return 0
        time.sleep(float(args.idle_sleep))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend-url", default=os.environ.get("ORBBEC_TASK_BACKEND_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--nas-root", type=Path, default=Path("task_backend_state/virtual_nas"))
    parser.add_argument("--nas-uri-prefix", default="nas://orbbec-virtual")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Virtual NAS, auto-label, QC, and label workers for Orbbec backend testing.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    seed_label = sub.add_parser("seed-label", help="Queue manual correction segments from an existing label JSONL file.")
    add_common_args(seed_label)
    seed_label.add_argument("--jsonl", type=Path, default=Path("label/task.jsonl"))
    seed_label.add_argument("--limit", type=int, default=0)
    seed_label.add_argument("--max-jobs", type=int, default=0, help="Stop after seeding this many manual label jobs.")
    seed_label.add_argument("--frames-per-segment", type=int, default=0, help="Split each JSONL task into manual-label segments of N frames. 0 keeps one segment per task.")
    seed_label.add_argument("--job-prefix", default="seeded_manual")
    seed_label.add_argument("--use-nas", action="store_true", help="Materialize tasks under the virtual NAS and use nas:// URIs.")
    seed_label.add_argument("--copy-source", action="store_true", help="Copy real source episode folders when they exist.")
    seed_label.add_argument("--max-materialized-frames", type=int, default=0, help="0 means materialize all frames.")
    seed_label.set_defaults(func=seed_manual_label_jobs)

    seed_upload = sub.add_parser("seed-captured", help="Queue upload jobs from label JSONL as if collection already captured them.")
    add_common_args(seed_upload)
    seed_upload.add_argument("--jsonl", type=Path, default=Path("label/task.jsonl"))
    seed_upload.add_argument("--limit", type=int, default=0)
    seed_upload.add_argument("--job-prefix", default="seeded_upload")
    seed_upload.set_defaults(func=seed_captured_upload_jobs)

    workers = sub.add_parser("run-workers", help="Run virtual upload, auto-label, MANO, and QC workers. Manual labeling stays in the real label frontend.")
    add_common_args(workers)
    workers.add_argument("--workers", default="default", help="default, all, or comma list: upload,auto-label,mano-opt,qc")
    workers.add_argument("--worker-id", default="")
    workers.add_argument("--lease-seconds", type=int, default=300)
    workers.add_argument("--qc-fail-rate", type=float, default=0.5)
    workers.add_argument("--copy-source", action="store_true")
    workers.add_argument("--max-materialized-frames", type=int, default=0)
    workers.add_argument("--once", action="store_true")
    workers.add_argument("--max-iterations", type=int, default=0)
    workers.add_argument("--stop-after-idle-rounds", type=int, default=1)
    workers.add_argument("--idle-sleep", type=float, default=1.0)
    workers.set_defaults(func=run_workers)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (BackendError, ValueError, OSError) as exc:
        print_event("error", message=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
