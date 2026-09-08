from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from label.ego_preview import EgoPreview
from label.storage import CorrectionTask


class EgoPreviewTest(unittest.TestCase):
    def test_timestamp_alignment_duplicates_missing_frames_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = root / "ego/RGB"
            rgb.mkdir(parents=True)
            (root / "timestamps.csv").write_text("frame_index,ego_frame_index\n5,2\n6,2\n7,3\n")
            Image.new("RGB", (8, 8), "red").save(rgb / "00002.png")
            Image.new("RGB", (8, 8), "blue").save(rgb / "00003.png")
            task = SimpleNamespace(episode_dir=lambda: root, frames=[5, 6, 7, 8])
            preview = EgoPreview(task)
            try:
                self.assertTrue(preview.done_event.wait(3))
                self.assertFalse(preview.error)
                self.assertEqual(preview.path(5).resolve(), preview.path(6).resolve())
                self.assertEqual(Image.open(preview.path(7)).getpixel((0, 0)), (0, 0, 255))
                self.assertIsNone(preview.path(8))
            finally:
                preview.close()
            self.assertFalse(preview.cache_dir.exists())

    def test_video_frames_are_decoded_in_ego_order_and_published_as_reference_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = SimpleNamespace(episode_dir=lambda: root, frames=[10, 11, 12])

            def decode(**kwargs):
                self.assertEqual(kwargs["frames"], [2, 3])
                kwargs["out_dir"].mkdir()
                for frame in kwargs["frames"]:
                    Image.new("RGB", (8, 8)).save(kwargs["out_dir"] / f"{frame:05d}.jpg")
                    kwargs["on_frame"](frame, 1)

            with patch("src.qc.media._load_reference_to_ego_frames", return_value={10: 2, 11: 2, 12: 3}), \
                 patch("src.qc.media._locate_ego_rgb_video", return_value=root / "rgb.h265"), \
                 patch("label.ego_preview._decode_camera_frames_streaming", side_effect=decode):
                preview = EgoPreview(task)
                try:
                    self.assertTrue(preview.done_event.wait(3))
                    self.assertFalse(preview.error)
                    self.assertTrue(all(preview.path(frame) for frame in task.frames))
                finally:
                    preview.close()

    def test_old_payload_discovers_06_but_ego_is_never_editable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "06/RGB").mkdir(parents=True)
            task = CorrectionTask(1, tmp, "s", "t", "e", ["00", "01", "ego", "pico"], [0], nas_root_path=tmp)
            self.assertEqual(task.cameras, ["00", "01", "06"])


if __name__ == "__main__":
    unittest.main()
