"""Background original MANO previews using the same renderer as QC."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import threading

from src.qc.media import MeshRendererSettings, _prepare_mesh_frames
from .storage import find_frame_path


def qc_renderer_settings() -> MeshRendererSettings:
    # The capture launcher config is also the source of QC's renderer settings.
    config_path = Path(__file__).resolve().parents[1] / "src/sync/config.json"
    data = json.loads(config_path.read_text()) if config_path.is_file() else {}
    qc = data.get("frontends", {}).get("qc", {})
    toolkit = Path(qc.get("manoToolkitRoot") or "/home/ubuntu/WorkSpace/zhenghao/opt_toolkits")
    if not toolkit.is_absolute():
        toolkit = (config_path.parent / toolkit).resolve()
    model = Path(qc.get("manoModelDir") or toolkit / "ckpt/mano")
    if not model.is_absolute():
        model = (config_path.parent / model).resolve()
    return MeshRendererSettings(
        python_executable=qc.get("meshRendererPython") or "python3",
        mano_toolkit_root=toolkit,
        mano_model_dir=model,
        render_factor=float(qc.get("meshRenderFactor", 0.5)),
        workers=max(1, int(qc.get("meshRenderWorkers", 6))),
        prefer_integrated_gpu=bool(qc.get("meshPreferIntegratedGpu", True)),
    )


class OriginalMeshCache:
    def __init__(self, task, *, settings=None, cache_parent=None):
        self.task = task
        self.settings = settings or qc_renderer_settings()
        self.cache_dir = Path(tempfile.mkdtemp(prefix="label_mano_", dir=cache_parent))
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.error = ""
        self._lock = threading.Lock()
        self._process = None
        self._thread = threading.Thread(target=self._run, name="label-mano-cache", daemon=True)
        self._thread.start()

    def path(self, camera, frame):
        path = self.cache_dir / "mesh" / str(camera) / f"{int(frame):05d}.jpg"
        return path if path.is_file() else None

    def _register_process(self, process):
        with self._lock:
            self._process = process
        if process is not None and self.stop_event.is_set():
            self._terminate(process)

    @staticmethod
    def _terminate(process, *, force=False):
        if process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            pass

    def _run(self):
        try:
            # Reuse Label's decoded RGB frames without decoding or copying again.
            for camera in self.task.cameras[:6]:
                folder = self.cache_dir / str(camera)
                folder.mkdir()
                for frame in self.task.frames:
                    if self.stop_event.is_set():
                        return
                    source = find_frame_path(self.task.episode_dir(), camera, frame, self.task.rgb_path_template)
                    if source is None:
                        raise FileNotFoundError(f"RGB frame missing: {camera}/{frame:05d}")
                    (folder / f"{frame:05d}{source.suffix}").symlink_to(source.resolve())
            if not self.stop_event.is_set():
                _prepare_mesh_frames(
                    task=self.task, cache_dir=self.cache_dir, settings=self.settings,
                    on_progress=None, stop_event=self.stop_event,
                    process_callback=self._register_process,
                    cameras=list(self.task.cameras[:6]),
                )
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = str(exc)
        finally:
            self.done_event.set()

    def close(self):
        """Cancel and clean up off the Tk thread, including renderer children."""
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        with self._lock:
            process = self._process
        if process is not None:
            self._terminate(process)

        def reap():
            if not self.done_event.wait(3):
                with self._lock:
                    process = self._process
                if process is not None:
                    self._terminate(process, force=True)
            self._thread.join()
            shutil.rmtree(self.cache_dir, ignore_errors=True)

        threading.Thread(target=reap, name="label-mano-cleanup", daemon=True).start()
