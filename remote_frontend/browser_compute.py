"""Bounded async calculations for local browser drafts; never publish artifacts."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import threading
import uuid

import numpy as np
from label.storage import correction_task_from_backend_payload, find_frame_path
from label.video_frames import ensure_decoded_rgb_frames
from src.qc.media import MeshRendererSettings, _prepare_mesh_frames
from .batch import reject


class BrowserCompute:
    def __init__(self, batch, media, config, *, pool=None):
        self.batch, self.media, self.config = batch, media, config
        self.pool = pool or ThreadPoolExecutor(max_workers=2, thread_name_prefix="browser-compute")
        self.lock = threading.Lock()
        self.active = set()

    def directory(self, sid, operation):
        if len(operation) != 32 or any(c not in "0123456789abcdef" for c in operation):
            reject("计算编号无效", 400)
        return self.media.directory(sid) / "operations" / operation

    def start(self, sid, body):
        item = self.batch.session(sid)
        self.batch.check(item)
        if item["role"] != "label" or body.get("action") not in {"track", "skeleton", "mesh"}:
            reject("不支持的计算", 400)
        task = correction_task_from_backend_payload(item["payload"], mounts=self.batch.mounts)
        frame = body.get("frame")
        if type(frame) is not int or frame not in task.frames:
            reject("计算帧不属于该任务", 400)
        action = body["action"]
        samples = body.get("samples", {})
        if action != "mesh":
            if not isinstance(samples, dict) or set(samples) != set(task.cameras):
                reject("计算需要当前帧的全部机位", 400)
            for sample in samples.values():
                points = np.asarray(sample.get("points"), dtype=float)
                visible = np.asarray(sample.get("visible"))
                if (points.shape != (2,21,2) or not np.isfinite(points).all()
                        or visible.shape != (2,21) or not np.isin(visible, [0,1]).all()):
                    reject("关节坐标或可见性无效", 400)
                if any(type(sample.get(k)) is not int or not 1 <= sample[k] <= 8192 for k in ("width", "height")):
                    reject("画面尺寸无效", 400)
        if action == "skeleton":
            counts = sum(np.asarray(s["visible"], dtype=int) for s in samples.values())
            missing = int((counts < 2).sum())
            if missing:
                reject(f"还有 {missing} 个关节的可见机位少于 2，无法重建骨架", 400)
        if action == "track":
            target = body.get("target")
            if type(target) is not int or target not in task.frames or target <= frame:
                reject("跟踪目标必须是该任务后续帧", 400)
            selected = body.get("selected")
            if not isinstance(selected, dict) or not set(selected).issubset(task.cameras):
                reject("跟踪机位无效", 400)
            count = 0
            for camera, pairs in selected.items():
                if not isinstance(pairs, list) or len(pairs) > 42:
                    reject("跟踪关节无效", 400)
                for pair in pairs:
                    if (not isinstance(pair, list) or len(pair) != 2
                            or any(type(v) is not int for v in pair)
                            or not 0 <= pair[0] < 2 or not 0 <= pair[1] < 21):
                        reject("跟踪关节无效", 400)
                    h, j = pair
                    if not samples[camera]["visible"][h][j] or min(samples[camera]["points"][h][j]) < 0:
                        reject("请先将跟踪关节设为可见并放置到画面中", 400)
                    count += 1
            if not count:
                reject("请先选择需要跟踪的关节", 400)
        with self.lock:
            if sid in self.active:
                reject("当前任务已有计算正在进行，请稍候")
            self.active.add(sid)
        operation = uuid.uuid4().hex
        root = self.directory(sid, operation)
        root.mkdir(parents=True)
        # Paths always come from the server-owned task, never the request.
        request = dict(action=action, frame=frame, samples=samples,
                       selected=body.get("selected"), target=body.get("target"))
        self.pool.submit(self._run, sid, root, item, task, request)
        return {"id": operation, "ready": False}

    def status(self, sid, operation):
        root = self.directory(sid, operation)
        if not root.is_dir():
            reject("计算不存在", 404)
        if (root/"error.txt").exists():
            return {"id": operation, "ready": False, "error": (root/"error.txt").read_text()}
        if (root/"ready.json").exists():
            return {"id": operation, "ready": True, **json.loads((root/"ready.json").read_text())}
        return {"id": operation, "ready": False}

    def _run(self, sid, root, item, task, body):
        try:
            cfg = self.config
            if body["action"] in {"track", "mesh"}:
                frames = ([body["frame"], body["target"]] if body["action"] == "track" else [body["frame"]])
                task = ensure_decoded_rgb_frames(replace(task, frames=frames), item["payload"],
                                                cache_root=self.media.directory(sid)/"decode")
            if body["action"] == "mesh":
                for camera in task.cameras:
                    source = find_frame_path(task.episode_dir(), camera, body["frame"], task.rgb_path_template)
                    if source is None:
                        raise ValueError(f"Missing RGB frame: {camera}")
                    folder = root/camera
                    folder.mkdir()
                    (folder/f"{body['frame']:05d}{source.suffix}").symlink_to(source.resolve())
                settings = MeshRendererSettings(python_executable=cfg["mesh_renderer_python"],
                    mano_toolkit_root=Path(cfg["mano_toolkit_root"]), mano_model_dir=Path(cfg["mano_model_dir"]),
                    workers=1, render_factor=float(cfg.get("mesh_render_factor", .5)))
                _prepare_mesh_frames(task=task, cache_dir=root, settings=settings,
                                     on_progress=None, cameras=task.cameras)
                from PIL import Image
                for camera in task.cameras:
                    with Image.open(root/"mesh"/camera/f"{body['frame']:05d}.jpg") as image:
                        image.save(root/f"{camera}.png")
                result = {"cameras": task.cameras}
            else:
                body.update(episode=str(task.episode_dir()), template=task.rgb_path_template)
                (root/"input.json").write_text(json.dumps(body, allow_nan=False))
                python = cfg.get("tracking_python", sys.executable) if body["action"] == "track" else sys.executable
                process = subprocess.run([python, "-m", "remote_frontend.compute_worker", str(root)],
                    capture_output=True, text=True, timeout=300)
                if process.returncode:
                    raise RuntimeError(process.stderr[-1500:] or "计算进程失败")
                result = json.loads((root/"result.json").read_text())
            (root/"ready.tmp").write_text(json.dumps(result, allow_nan=False))
            (root/"ready.tmp").replace(root/"ready.json")
        except Exception as exc:
            (root/"error.txt").write_text(str(exc))
        finally:
            with self.lock:
                self.active.discard(sid)
