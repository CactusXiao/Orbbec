"""Backend-only raw decoding, MANO projection and QC preview preparation."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

from PIL import Image
from label.mano_view import ManoViewRuntime
from label.storage import (correction_task_from_backend_payload, find_frame_path,
                           load_joint_visibility, load_prediction_bundle, source_frame_path, view_state_from_bundle)
from label.video_frames import ensure_decoded_rgb_frames
from src.qc.media import MeshRendererSettings, prepare_qc_media
from .browser_label import browser_task, decode_label, BrowserLabelRuntime
from .layered_preview import encode_layered, raw_frame, content_layout, layout


def qc_encoding_settings(root, config):
    """Pin a session's encoding so resumes never mix different video layouts."""
    marker = root / 'encoding.json'
    if marker.exists():
        return json.loads(marker.read_text())
    if (root / 'ready.json').exists():
        settings = dict(profile='legacy', encoder='libx264', bitrate='6M')
    else:
        settings = dict(profile=config.get('qc_video_profile', 'detail960'),
                        encoder=config.get('qc_video_encoder', 'libx264'),
                        bitrate=config.get('qc_video_maxrate', '7M'))
    layout(['00', 'ego'], settings['profile'])  # Validate before saving.
    if settings['encoder'] not in ('libx264', 'h264_nvenc'):
        raise ValueError('Unsupported QC video encoder')
    temporary = root / 'encoding.tmp'
    temporary.write_text(json.dumps(settings))
    temporary.replace(marker)
    return settings


def label_states(runtime, task, corrected, frame, camera):
    """Match desktop fallback: saved corrections survive missing references."""
    import numpy as np
    hidden = dict(points=np.full((2,21,2), -1.).tolist(), visible=np.zeros((2,21), dtype=bool).tolist())
    references = {"mano": hidden, "mano_visible": hidden, "errors": {}}
    try:
        state = runtime.project_mano_frame(episode_dir=task.episode_dir(),
            mano_dir=task.episode_dir()/task.mano_episode_dir, cam_id=camera, frame_idx=frame)
        if state is None:
            raise ValueError("MANO 投影缺失")
        raw, projected = state
        raw = np.nan_to_num(np.asarray(raw), nan=-1, posinf=-1, neginf=-1).tolist()
        references["mano"] = dict(points=raw, visible=projected)
        mask = load_joint_visibility(task.episode_dir()/"joints_vis", camera, frame)
        if mask is None:
            raise ValueError("原始可见性文件缺失")
        masked = [[bool(projected[h][j] and mask[h][j]) for j in range(21)] for h in range(2)]
        references["mano_visible"] = dict(points=raw, visible=masked)
    except Exception as exc:
        references["errors"]["mano_visible"] = str(exc)
        if references["mano"] is hidden:
            references["errors"]["mano"] = str(exc)
    if camera == "ego" and references["mano"] is not hidden and references["mano_visible"] is hidden:
        # Preserve projected locations when an older episode has no Ego mask.
        # Visibility stays unasserted until the operator checks each joint.
        references["mano_visible"] = dict(points=references["mano"]["points"], visible=hidden["visible"])
    if source_frame_path(corrected, frame, camera) is not None:
        points, visible = view_state_from_bundle(corrected, frame, camera)
    else:
        initial = references["mano_visible"]
        points, visible = initial["points"], initial["visible"]
    return points, visible, references


class BrowserMedia:
    def __init__(self, batch, root, config, *, slots=None):
        self.batch, self.root, self.config = batch, Path(root), config
        self.lock = threading.Lock()
        self.running = set()
        self.progress = {}
        self.stops = {}
        self.threads = {}
        self.content_layouts = {}
        self.slots = slots or threading.Semaphore(2)

    def directory(self, sid):
        item = self.batch.session(sid)  # Only server-issued IDs can select a directory.
        return self.root / sid / "layered-v1" if item["role"] == "qc" else self.root / sid / "label-v2"

    def status(self, sid):
        root = self.directory(sid)
        state = json.loads((root / "ready.json").read_text()) if (root / "ready.json").is_file() else {"ready": False}
        # Existing encoded clips remain valid: amend only their display rectangles.
        if state.get("layout") and not state["layout"].get("content_bounds"):
            if sid not in self.content_layouts:
                cameras = state.get("cameras", [])
                paths = {c: next((root/"raw_frames"/c).glob("*.jpg"), None) for c in cameras}
                if cameras and all(paths.values()):
                    self.content_layouts[sid] = content_layout(cameras, paths)
            if sid in self.content_layouts:
                state["layout"] = self.content_layouts[sid]
        error = root / "error.txt"
        if error.exists():
            state["error"] = error.read_text()
            return state
        if state.get("complete", state.get("ready", False)):
            return state
        with self.lock:
            state["progress"] = dict(self.progress.get(sid, {}))
            if sid not in self.running:
                self.running.add(sid)
                self.stops[sid] = threading.Event()
                self.threads[sid] = threading.Thread(target=self._prepare, args=(sid,), daemon=True)
                self.threads[sid].start()
        return state

    def ensure_ego(self, sid):
        root = self.directory(sid)
        if (root / "ego-ready.json").exists():
            return
        key = "ego:" + sid
        with self.lock:
            if key in self.running:
                return
            self.running.add(key)
            def prepare():
                preview = None
                try:
                    from label.ego_preview import EgoPreview
                    task = correction_task_from_backend_payload(self.batch.session(sid)["payload"], mounts=self.batch.mounts)
                    preview = EgoPreview(task)
                    preview.done_event.wait()
                    target = root / "frames" / "ego"
                    target.mkdir(parents=True, exist_ok=True)
                    for frame in task.frames:
                        path = preview.path(frame)
                        if path:
                            shutil.copyfile(path, target / f"{frame}.jpg")
                    (root / "ego-ready.json").write_text(json.dumps({"error": preview.error}))
                except Exception as exc:
                    (root / "ego-ready.json").write_text(json.dumps({"error": str(exc)}))
                finally:
                    if preview:
                        preview.close()
                    with self.lock:
                        self.running.discard(key)
            threading.Thread(target=prepare, daemon=True).start()

    def cancel(self, sid):
        self.batch.session(sid)
        with self.lock:
            stop = self.stops.get(sid)
            thread = self.threads.get(sid)
            if stop:
                stop.set()
        if thread:
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError("后端仍在停止解码和渲染，请稍后重试返回")

    @staticmethod
    def publish(root, value):
        temporary = root / "ready.tmp"
        temporary.write_text(json.dumps(value))
        temporary.replace(root / "ready.json")

    def retry(self, sid):
        self.batch.check(self.batch.session(sid))
        root = self.directory(sid)
        with self.lock:
            if sid in self.running:
                return {"ok": True}
            (root / "error.txt").unlink(missing_ok=True)
        self.status(sid)
        return {"ok": True}

    def _prepare(self, sid):
        root = self.directory(sid)
        root.mkdir(parents=True, exist_ok=True)
        stop = self.stops[sid]
        def progress(camera, fields):
            with self.lock:
                current = self.progress.setdefault(sid, {})
                current[camera] = {**current.get(camera, {}), **{k: v for k, v in fields.items()
                    if k in {"status", "decoded", "rendered", "total"}}}
        try:
            with self.slots:
                item = self.batch.session(sid)
                payload = item["payload"]
                task = browser_task(payload, mounts=self.batch.mounts, role=item["role"])
                if item["role"] == "label":
                    decoded = decode_label(task, payload, cache_root=root / "decode", stop_event=stop)
                    runtime, samples, sources = BrowserLabelRuntime(), {}, {}
                    corrected = load_prediction_bundle(task, mode="correct")
                    for frame in task.frames:
                        if stop.is_set():
                            raise InterruptedError("任务已暂停")
                        for camera in task.cameras:
                            source = find_frame_path(decoded.episode_dir(), camera, frame, decoded.rgb_path_template)
                            if source is None:
                                raise ValueError(f"Missing RGB frame {camera}/{frame}")
                            target = root / "frames" / camera / f"{frame}.jpg"
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with Image.open(source) as image:
                                width, height = image.size
                                image.convert("RGB").save(target, quality=90)
                            points, visible, references = label_states(runtime, task, corrected, frame, camera)
                            sources[f"{frame}:{camera}"] = references
                            samples[f"{frame}:{camera}"] = dict(points=points, visible=visible, width=width, height=height)
                            progress(camera, dict(rendered=task.frames.index(frame)+1, total=len(task.frames)))
                    (root / "samples.json").write_text(json.dumps(samples, allow_nan=False))
                    (root / "sources.json").write_text(json.dumps(sources, allow_nan=False))
                    ready = dict(ready=True, cameras=task.cameras)
                else:
                    cfg = self.config
                    encoding = qc_encoding_settings(root, cfg)
                    settings = MeshRendererSettings(
                        python_executable=cfg["mesh_renderer_python"],
                        mano_toolkit_root=Path(cfg["mano_toolkit_root"]), mano_model_dir=Path(cfg["mano_model_dir"]),
                        workers=int(cfg.get("mesh_render_workers", 4)), render_factor=float(cfg.get("mesh_render_factor", .5)),
                        prefer_integrated_gpu=bool(cfg.get("mesh_prefer_integrated_gpu", False)))
                    media = prepare_qc_media(payload, mounts=self.batch.mounts, tmp_dir=root/"decode", mesh_renderer=settings,
                                             on_progress=progress, stop_event=stop)
                    try:
                        chunks = []
                        chunk_size = 90  # Three seconds; playback starts before the episode finishes rendering.
                        for start in range(0, len(task.frames), chunk_size):
                            frames = task.frames[start:start+chunk_size]
                            while not all(media.frame_ready(frame) for frame in frames):
                                if stop.is_set():
                                    raise InterruptedError("任务已暂停")
                                if media.preparation_error:
                                    raise RuntimeError(media.preparation_error)
                                if media.preparation_done:
                                    raise RuntimeError("渲染结束但仍缺少画面")
                                time.sleep(.1)
                            for frame in frames:
                                for camera in media.display_cameras:
                                    source = media.cache_dir / "mesh" / camera / f"{frame:05d}.jpg"
                                    target = root / "frames" / camera / f"{frame}.jpg"
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    target.write_bytes(source.read_bytes())
                                    raw_target = root / "raw_frames" / camera / f"{frame}.jpg"
                                    raw_target.parent.mkdir(parents=True, exist_ok=True)
                                    raw_source = raw_frame(media.cache_dir, camera, frame)
                                    if raw_source.suffix == ".jpg":
                                        raw_target.write_bytes(raw_source.read_bytes())
                                    else:
                                        with Image.open(raw_source) as raw_image:
                                            raw_image.convert("RGB").save(raw_target, quality=95)
                            output = root / "chunks" / f"{len(chunks)}.mp4"
                            info = encode_layered(media.cache_dir, output, frames, media.display_cameras, **encoding)
                            chunks.append(dict(index=len(chunks), start=start, count=len(frames), bytes=info["bytes"]))
                            ready = dict(ready=True, complete=False, cameras=media.display_cameras,
                                         chunks=chunks, prepared=start+len(frames), total=len(task.frames), codec=info["codec"], layout=info["layout"])
                            self.publish(root, ready)
                        # Concatenation copies encoded packets; no second render/decode pass.
                        listing = root / "chunks" / "concat.txt"
                        listing.write_text("".join(f"file '{c['index']}.mp4'\n" for c in chunks))
                        result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-f", "concat", "-safe", "1", "-i", str(listing), "-c", "copy", "-movflags", "+faststart",
                            str(root/"preview.mp4")], capture_output=True, text=True, timeout=120)
                        if result.returncode:
                            raise RuntimeError(result.stderr[-1000:])
                        ready["complete"] = True
                    finally:
                        media.close()
                self.publish(root, ready)
        except InterruptedError:
            pass
        except Exception as exc:
            (root / "error.txt").write_text(str(exc))
        finally:
            with self.lock:
                self.running.discard(sid)
