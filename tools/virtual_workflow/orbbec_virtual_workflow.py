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


Json = Dict[str, Any]


class BackendError(RuntimeError):
    pass


class NoJobAvailable(Exception):
    pass


def now_ms() -> int:
    return int(time.time() * 1000)


def clean_id(value: str, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_").lower()
    return text or fallback


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
            if exc.code == 404 and message.startswith("no queued "):
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
            "/api/v1/label/jobs/lease",
            {"operator_id": operator_id, "lease_seconds": lease_seconds},
        )

    def heartbeat_label_job(self, job_id: str, operator_id: str, lease_seconds: int = 600) -> Json:
        return self.post(
            f"/api/v1/label/jobs/{quote(job_id, safe='')}/heartbeat",
            {"operator_id": operator_id, "lease_seconds": lease_seconds, "status": "running"},
        )

    def complete_label_job(self, job_id: str, result: Json, artifacts: Optional[List[Json]] = None) -> Json:
        body: Json = {"result": result}
        if artifacts:
            body["artifacts"] = artifacts
        return self.post(f"/api/v1/label/jobs/{quote(job_id, safe='')}/complete", body)


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


def split_task_frames(task: LabelTask, frames_per_job: int = 0) -> List[LabelTask]:
    size = int(frames_per_job or 0)
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


def write_placeholder_npy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (2, 21, 2)
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 21, 2), }"
    header_bytes = header.encode("latin1")
    pad = 16 - ((10 + len(header_bytes) + 1) % 16)
    full_header = header_bytes + b" " * pad + b"\n"
    data = struct.pack("<" + "f" * (shape[0] * shape[1] * shape[2]), *([-1.0] * 84))
    path.write_bytes(b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(full_header)) + full_header + data)


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
        )
        return uri

    def materialize_payload(
        self,
        payload: Json,
        episode: Optional[Json] = None,
        *,
        copy_source: bool = False,
        max_frames: int = 0,
    ) -> str:
        episode = episode or {}
        episode_id = str(payload.get("episode_id") or episode.get("episode_id") or f"episode_{uuid.uuid4().hex[:8]}")
        subject = str(payload.get("subject_id") or episode.get("subject_id") or "virtual_subject")
        task_name = str(payload.get("task_name") or episode.get("task_name") or "virtual_task")
        episode_name = str(payload.get("episode") or episode.get("episode_name") or episode_id)
        cameras = as_string_list(payload.get("cameras") or episode.get("cameras")) or ["00"]
        frames = as_int_list(payload.get("frames")) or list(range(max(1, int(episode.get("frame_count") or 1))))
        source = source_path_from_payload(payload, episode) if copy_source else None
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
                pred_path = dst / prediction_dir / cam / f"{int(frame):05d}.npy"
                if not pred_path.exists():
                    write_placeholder_npy(pred_path)
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

    def write_prediction_artifact(self, data_uri: str, cameras: Sequence[str], frames: Sequence[int], prediction_dir: str = "pred_2d") -> str:
        base = self.local_path_for_uri(data_uri)
        for cam in cameras or ["00"]:
            for frame in frames or [0]:
                write_placeholder_npy(base / prediction_dir / str(cam) / f"{int(frame):05d}.npy")
        return uri_join(data_uri, prediction_dir)

    def write_corrected_artifact(self, data_uri: str, cameras: Sequence[str], frames: Sequence[int], correction_dir: str = "corrected_2d") -> str:
        base = self.local_path_for_uri(data_uri)
        for cam in cameras or ["00"]:
            for frame in frames or [0]:
                write_placeholder_npy(base / correction_dir / str(cam) / f"{int(frame):05d}.npy")
        return uri_join(data_uri, correction_dir)


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


def source_path_from_payload(payload: Json, episode: Optional[Json] = None) -> Optional[Path]:
    episode = episode or {}
    raw = str(payload.get("local_capture_path") or episode.get("local_capture_path") or payload.get("local_path") or "")
    if raw:
        return Path(raw).expanduser().resolve()
    data_uri = str(payload.get("data_uri") or episode.get("data_uri") or "")
    if data_uri.startswith("local://"):
        return path_from_local_uri(data_uri)
    return None


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
        batches = split_task_frames(task, args.frames_per_job)
        for batch_index, batch_task in enumerate(batches, 1):
            if args.max_jobs and count >= args.max_jobs:
                stop = True
                break
            if args.use_nas:
                data_uri = nas.materialize_task(batch_task, copy_source=args.copy_source, max_frames=args.max_materialized_frames)
                local_path = ""
            else:
                data_uri = local_uri_from_path(task.episode_dir)
                local_path = str(task.episode_dir)
            suffix = f"_b{batch_index:04d}" if len(batches) > 1 else ""
            job_id = f"{clean_id(args.job_prefix)}_{task.episode_id}{suffix}"
            body = payload_from_task(batch_task, data_uri, job_id, "seeded_from_label_jsonl")
            body["payload"] = {
                "batch_index": batch_index,
                "batch_count": len(batches),
                "frames_per_job": int(args.frames_per_job or 0),
            }
            if local_path:
                body["local_path"] = local_path
            result = client.create_manual_label_job(body)
            count += 1
            print_event(
                "manual_label_seeded",
                job_id=result.get("job", {}).get("job_id", job_id),
                episode_id=task.episode_id,
                data_uri=data_uri,
                frames=len(batch_task.frames),
                batch_index=batch_index,
                batch_count=len(batches),
                status=result.get("job", {}).get("status"),
            )
        if stop:
            break
    print_event("manual_label_seed_done", count=count)
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
    job = response.get("job") or {}
    episode = response.get("episode") or {}
    for key in ("job_id", "episode_id"):
        payload.setdefault(key, job.get(key) or episode.get(key))
    for key in ("subject_id", "task_name", "data_uri", "cameras"):
        payload.setdefault(key, episode.get(key))
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
    nas_uri = nas.materialize_payload(payload, episode, copy_source=args.copy_source, max_frames=args.max_materialized_frames)
    cameras = as_string_list(payload.get("cameras")) or as_string_list(episode.get("cameras")) or ["00"]
    frames = as_int_list(payload.get("frames")) or list(range(max(1, int(episode.get("frame_count") or 1))))
    client.complete_job(
        str(job["job_id"]),
        {
            "ok": True,
            "nas_uri": nas_uri,
            "virtual_worker": owner,
            "copied_from": str(source_path_from_payload(payload, episode) or ""),
        },
        artifacts=[{"kind": "nas_episode", "uri": nas_uri, "metadata": {"worker_id": owner}}],
    )
    print_event(
        "upload_completed",
        job_id=job["job_id"],
        episode_id=payload.get("episode_id") or episode.get("episode_id"),
        nas_uri=nas_uri,
        auto_label="queued_by_backend",
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
    payload = enriched_payload(leased)
    client.heartbeat_job(str(job["job_id"]), owner, args.lease_seconds)
    data_uri = str(payload.get("data_uri") or "")
    cameras = as_string_list(payload.get("cameras")) or ["00"]
    frames = as_int_list(payload.get("frames")) or [0]
    if args.max_materialized_frames > 0:
        frames = frames[: args.max_materialized_frames]
    prediction_dir = str(payload.get("prediction_dir") or "pred_2d")
    pred_uri = nas.write_prediction_artifact(data_uri, cameras, frames, prediction_dir) if data_uri.startswith(nas.uri_prefix) else uri_join(data_uri, prediction_dir)
    client.complete_job(
        str(job["job_id"]),
        {"ok": True, "model": "virtual_hand2d", "frames_predicted": frames, "virtual_worker": owner},
        artifacts=[{"kind": "pred_2d", "uri": pred_uri, "metadata": {"worker_id": owner, "mock": True}}],
    )
    qc_job_id = f"qc_{clean_id(str(payload.get('episode_id')))}"
    qc_payload = dict(payload)
    qc_payload.update({"job_id": qc_job_id, "pred_uri": pred_uri, "reason": "auto_label_completed_by_virtual_worker"})
    client.create_dev_job("qc", str(payload.get("episode_id")), qc_payload)
    print_event("auto_label_completed", job_id=job["job_id"], episode_id=payload.get("episode_id"), pred_uri=pred_uri, next_job=qc_job_id)
    return True


def handle_qc_once(client: BackendClient, nas: NasSimulator, args: argparse.Namespace) -> bool:
    owner = args.worker_id or f"virtual_qc_{os.getpid()}"
    try:
        leased = client.lease_job("qc", owner, args.lease_seconds)
    except NoJobAvailable:
        return False
    job = leased.get("job") or {}
    payload = enriched_payload(leased)
    client.heartbeat_job(str(job["job_id"]), owner, args.lease_seconds)
    passed = random.random() >= float(args.qc_fail_rate)
    qc_report_uri = uri_join(str(payload.get("data_uri") or ""), "virtual_qc_report.json")
    result = {
        "passed": passed,
        "qc_passed": passed,
        "score": round(random.random(), 4),
        "reason": "virtual_qc_pass" if passed else "virtual_qc_failed_needs_manual_label",
        "create_manual_label_job": False,
        "virtual_worker": owner,
    }
    client.complete_job(
        str(job["job_id"]),
        result,
        artifacts=[{"kind": "qc_report", "uri": qc_report_uri, "metadata": {"passed": passed, "worker_id": owner}}],
    )
    if not passed:
        manual_job_id = f"manual_label_qc_{clean_id(str(payload.get('episode_id')))}_{now_ms()}"
        manual_payload = dict(payload)
        manual_payload.update(
            {
                "job_id": manual_job_id,
                "reason": "virtual_qc_failed",
                "metadata": {
                    "source": "virtual_qc",
                    "qc_job_id": job["job_id"],
                    "qc_report_uri": qc_report_uri,
                },
            }
        )
        created = client.create_manual_label_job(manual_payload)
        print_event(
            "qc_failed_manual_label_queued",
            job_id=job["job_id"],
            episode_id=payload.get("episode_id"),
            manual_job_id=created.get("job", {}).get("job_id", manual_job_id),
        )
    else:
        print_event("qc_passed", job_id=job["job_id"], episode_id=payload.get("episode_id"))
    return True


def handle_manual_label_once(client: BackendClient, nas: NasSimulator, args: argparse.Namespace) -> bool:
    operator = args.worker_id or f"virtual_labeler_{os.getpid()}"
    try:
        leased = client.lease_label_job(operator, args.lease_seconds)
    except NoJobAvailable:
        return False
    job = leased.get("job") or {}
    payload = enriched_payload(leased)
    client.heartbeat_label_job(str(job["job_id"]), operator, args.lease_seconds)
    data_uri = str(payload.get("data_uri") or "")
    cameras = as_string_list(payload.get("cameras")) or ["00"]
    frames = as_int_list(payload.get("frames")) or [0]
    if args.max_materialized_frames > 0:
        frames = frames[: args.max_materialized_frames]
    correction_dir = str(payload.get("correction_dir") or "corrected_2d")
    if data_uri.startswith(nas.uri_prefix):
        corrected_uri = nas.write_corrected_artifact(data_uri, cameras, frames, correction_dir)
    else:
        corrected_uri = uri_join(data_uri, correction_dir)
    client.complete_label_job(
        str(job["job_id"]),
        {"operator_id": operator, "frames_completed": frames, "virtual_labeler": True},
        artifacts=[
            {
                "kind": "corrected_2d",
                "uri": corrected_uri,
                "metadata": {"operator_id": operator, "frames": frames, "mock": True},
            }
        ],
    )
    print_event("manual_label_completed", job_id=job["job_id"], episode_id=payload.get("episode_id"), corrected_uri=corrected_uri)
    return True


WORKER_HANDLERS = {
    "upload": handle_upload_once,
    "auto-label": handle_auto_label_once,
    "qc": handle_qc_once,
    "manual-label": handle_manual_label_once,
}


def parse_workers(raw: str) -> List[str]:
    if raw == "default":
        return ["upload", "auto-label", "qc"]
    if raw == "all":
        return ["upload", "auto-label", "qc", "manual-label"]
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
    parser.add_argument("--nas-root", type=Path, default=Path(".virtual_nas"))
    parser.add_argument("--nas-uri-prefix", default="nas://orbbec-virtual")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Virtual NAS, auto-label, QC, and label workers for Orbbec backend testing.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    seed_label = sub.add_parser("seed-label", help="Queue manual_label jobs from an existing label JSONL file.")
    add_common_args(seed_label)
    seed_label.add_argument("--jsonl", type=Path, default=Path("label/task.jsonl"))
    seed_label.add_argument("--limit", type=int, default=0)
    seed_label.add_argument("--max-jobs", type=int, default=0, help="Stop after seeding this many manual label jobs.")
    seed_label.add_argument("--frames-per-job", type=int, default=0, help="Split each JSONL task into batches of N frames. 0 keeps one job per task.")
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

    workers = sub.add_parser("run-workers", help="Run virtual upload, auto-label, QC, and optionally manual-label workers.")
    add_common_args(workers)
    workers.add_argument("--workers", default="default", help="default, all, or comma list: upload,auto-label,qc,manual-label")
    workers.add_argument("--worker-id", default="")
    workers.add_argument("--lease-seconds", type=int, default=300)
    workers.add_argument("--qc-fail-rate", type=float, default=0.35)
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
