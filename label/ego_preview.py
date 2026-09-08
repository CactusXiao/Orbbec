"""Timestamp-aligned, read-only ego RGB frames for the Label overview."""
from pathlib import Path
import shutil
import tempfile
import threading

from .storage import find_frame_path
from .video_frames import _decode_camera_frames_streaming


class EgoPreview:
    def __init__(self, task):
        self.task = task
        self.cache_dir = Path(tempfile.mkdtemp(prefix="label_ego_"))
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.error = ""
        self._thread = threading.Thread(target=self._run, name="label-ego-preview", daemon=True)
        self._thread.start()

    def path(self, frame):
        path = self.cache_dir / f"{int(frame):05d}.jpg"
        return path if path.is_file() else None

    def _run(self):
        try:
            # Reuse QC's timestamp and storage contracts, without requiring a MANO model.
            from src.qc.media import _load_reference_to_ego_frames, _locate_ego_rgb_video
            episode = self.task.episode_dir()
            mapping = _load_reference_to_ego_frames(episode)
            reverse = {}
            for reference in self.task.frames:
                if self.stop_event.is_set():
                    return
                ego_frame = mapping.get(int(reference))
                if ego_frame is None:
                    continue
                source = find_frame_path(episode, "ego", ego_frame)
                if source is not None:
                    # Keep original bytes; PIL identifies their actual image format.
                    (self.cache_dir / f"{int(reference):05d}.jpg").symlink_to(source.resolve())
                else:
                    reverse.setdefault(ego_frame, []).append(int(reference))
            if reverse:
                raw = self.cache_dir / "raw"

                def publish(ego_frame, _count):
                    source = raw / f"{ego_frame:05d}.jpg"
                    for reference in reverse[ego_frame]:
                        (self.cache_dir / f"{reference:05d}.jpg").symlink_to(source)

                _decode_camera_frames_streaming(
                    video_path=_locate_ego_rgb_video(episode), timestamp_path=None,
                    frame_map={}, camera="ego", frames=sorted(reverse), out_dir=raw,
                    image_extension="jpg", ffmpeg_threads=2, stop_event=self.stop_event,
                    on_frame=publish,
                )
        except InterruptedError:
            pass
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.done_event.set()

    def close(self):
        self.stop_event.set()
        self._thread.join()
        shutil.rmtree(self.cache_dir, ignore_errors=True)
