import gc
from pathlib import Path
import tempfile
import threading
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from label.app import LabelPage
from label.canvas_view import ImageAnnotatorCanvas
from label.env_config import LabelConfig
from label.mesh_cache import OriginalMeshCache
from label.storage import CorrectionTask


class MeshCacheTest(unittest.TestCase):
    def test_background_renderer_reuses_rgb_and_caches_each_camera(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode = root / "subject/task/episode"
            cameras = [f"{i:02d}" for i in range(6)]
            for camera in cameras:
                folder = episode / camera / "RGB"
                folder.mkdir(parents=True)
                Image.new("RGB", (40, 30)).save(folder / "00005.png")
            task = SimpleNamespace(episode_dir=lambda: episode, cameras=cameras, frames=[5],
                                   rgb_path_template="{camera}/RGB/{frame:05d}.png")
            entered, release = threading.Event(), threading.Event()

            def render(**kwargs):
                entered.set()
                release.wait(5)
                for camera in kwargs["cameras"]:
                    rgb = kwargs["cache_dir"] / camera / "00005.png"
                    self.assertTrue(rgb.is_symlink())
                    output = kwargs["cache_dir"] / "mesh" / camera / "00005.jpg"
                    output.parent.mkdir(parents=True)
                    Image.open(rgb).save(output)

            with patch("label.mesh_cache._prepare_mesh_frames", side_effect=render) as renderer:
                cache = OriginalMeshCache(task, settings=object())
                try:
                    self.assertTrue(entered.wait(3))
                    self.assertFalse(cache.done_event.is_set())
                    self.assertIsNone(cache.path("00", 5))
                    release.set()
                    self.assertTrue(cache.done_event.wait(5))
                    self.assertEqual(cache.error, "")
                    for camera in cameras:
                        self.assertTrue(cache.path(camera, 5).is_file())
                    self.assertIsNone(cache.path("00", 6))
                    self.assertEqual(renderer.call_count, 1)
                finally:
                    release.set()
                    cache.close()
                self.assertTrue((episode / "00/RGB/00005.png").exists())


class LabelViewUiTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.addCleanup(self.cleanup_ui)
        self.page = LabelPage(self.root, config=LabelConfig(), on_back=lambda: None)
        self.page.pack(fill="both", expand=True)
        self.root.update()

    def cleanup_ui(self):
        self.page.on_hide()
        self.root.update_idletasks()
        self.root.destroy()
        self.page = None
        self.root = None
        gc.collect()

    def test_original_is_read_only_for_mouse_undo_and_ignore(self):
        p = self.page
        p._mode = "mano"
        p._sync_visualization_canvas_state()
        self.assertTrue(p._canvas._read_only)
        with patch.object(p._canvas, "undo") as undo, patch.object(p._canvas, "ignore_view") as ignore:
            p._undo()
            p._ignore_view()
            undo.assert_not_called()
            ignore.assert_not_called()
        p._mode = "correct"
        p._sync_visualization_canvas_state()
        self.assertFalse(p._canvas._read_only)

    def test_number_keys_switch_all_cameras_and_keep_current_edits(self):
        p = self.page
        p._active_task = object()
        p._camera_ids = [f"{i:02d}" for i in range(6)]
        with patch.object(p, "_cache_current_source_state") as save, patch.object(p, "_refresh_view") as refresh:
            self.root.focus_force()
            for index in range(6):
                self.root.event_generate(f"<KeyPress-{index + 1}>")
                self.root.update()
                self.assertEqual(p._cam_idx, index)
            self.assertEqual(save.call_count, 6)
            self.assertEqual(refresh.call_count, 6)
            entry = ttk.Entry(p)
            event = SimpleNamespace(widget=entry, state=0)
            self.assertIsNone(p._camera_shortcut(event, 0))
            self.assertEqual(p._cam_idx, 5)

    def test_original_show_mano_uses_cache_and_switching_source_clears_preview(self):
        p = self.page
        p._mode = "mano"
        p._active_task = SimpleNamespace(frames=[5])
        p._camera_ids = ["00", "01"]
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "mesh.jpg"
            Image.new("RGB", (100, 80), "red").save(image)
            p._canvas.set_image(image)
            cache = Mock(error="", done_event=threading.Event())
            cache.path.return_value = image
            p._original_mesh_cache = cache
            p._toggle_mano()
            self.assertTrue(p._show_mano)
            self.assertEqual(p._canvas._rendered_path, image)
            self.assertTrue(p._canvas._read_only)
            p._cam_idx = 1
            cache.path.return_value = None
            p._refresh_original_mesh_preview()
            self.assertIsNone(p._canvas._rendered_image)
            self.assertIsNotNone(p._mesh_poll_id)
            p._reset_visualizations()
            self.assertIsNone(p._mesh_poll_id)
            self.assertIsNone(p._canvas._rendered_image)

    def test_modified_view_reuses_original_mano_without_changing_edits(self):
        p = self.page
        p._active_task = SimpleNamespace(frames=[5])
        p._camera_ids = ["00"]
        points = [[(50.0 + j, 20.0) for j in range(21)] for _ in range(2)]
        visible = [[False] * 21 for _ in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "mesh.jpg"
            Image.new("RGB", (100, 80), "red").save(image)
            p._canvas.set_image(image)
            p._canvas.set_hand_state(points, visible)
            cache = Mock(error="", done_event=threading.Event())
            cache.path.return_value = image
            p._original_mesh_cache = cache
            with patch.object(p, "_mano_runtime_instance", side_effect=AssertionError("must not fit edited points")), \
                 patch.object(p, "_incomplete_joint_count", side_effect=AssertionError("no visibility restriction")):
                for mode in ("mano", "correct"):
                    p._mode = mode
                    p._toggle_mano()
                    self.assertEqual(p._canvas._rendered_path, image)
                    self.assertEqual(p._canvas.get_hand_state(), (points, visible))
                    self.assertEqual(p._view_states["00"], (points, visible))
                    p._toggle_mano()
                    self.assertIsNone(p._canvas._rendered_image)
                    self.assertEqual(p._canvas.get_hand_state(), (points, visible))
                    self.assertEqual(p._canvas._read_only, mode == "mano")

    def test_confirm_from_original_and_mano_saves_corrected_state_and_continues(self):
        p = self.page
        original = ([[(10.0, 20.0)] * 21 for _ in range(2)], [[True] * 21 for _ in range(2)])
        initial = ([[(10.0, 20.0)] * 21 for _ in range(2)], [[False] * 21 for _ in range(2)])
        edited = ([[(42.0, 24.0)] * 21 for _ in range(2)], [[False] * 21 for _ in range(2)])
        for mode in ("mano", "correct"):
            for show_mano in (False, True):
                with self.subTest(mode=mode, show_mano=show_mano):
                    p._active_task = SimpleNamespace(key="test", frames=[5, 6], total_frames=2)
                    p._active_key = "test"
                    p._jsonl_path = "unused"
                    p._camera_ids = ["00", "01"]
                    p._cam_idx = p._frame_pos = 0
                    p._mode = mode
                    p._show_mano = show_mano
                    p._progress = {}
                    p._source_state_cache = {("test", 0, "00", "correct"): edited}
                    p._canvas.set_hand_state(*(original if mode == "mano" else edited))
                    p._view_states = {"00": original if mode == "mano" else edited}
                    with patch.object(p, "_save_bundle", return_value=object()), \
                         patch.object(p, "_build_modified_view_state", return_value=initial), \
                         patch.object(p, "_load_current_sample") as next_frame, \
                         patch.object(p, "_update_tree_row"), \
                         patch("label.app.apply_view_state_to_corrected") as save, \
                         patch("label.app.save_corrected_array"), \
                         patch("label.app.save_correction_progress"), \
                         patch("label.app.messagebox.showwarning") as warning:
                        p._confirm()
                        warning.assert_not_called()
                        self.assertEqual(save.call_count, 2)
                        self.assertEqual(save.call_args_list[0].args[3:], edited)
                        self.assertEqual(save.call_args_list[1].args[3:], initial)
                        self.assertEqual(p._progress["test"].done_positions, {0})
                        self.assertEqual(p._frame_pos, 1)
                        self.assertEqual(p._mode, mode)
                        self.assertEqual(p._show_mano, show_mano)
                        next_frame.assert_called_once()

    def test_preview_swap_preserves_coordinates_visibility_and_zoom(self):
        canvas = self.page._canvas
        with tempfile.TemporaryDirectory() as temp:
            raw, mesh = Path(temp) / "raw.png", Path(temp) / "mesh.jpg"
            Image.new("RGB", (100, 80)).save(raw)
            Image.new("RGB", (100, 80), "red").save(mesh)
            canvas.set_image(raw)
            points = [[(float(j), float(j)) for j in range(21)] for _ in range(2)]
            visible = [[j % 2 == 0 for j in range(21)] for _ in range(2)]
            canvas.set_hand_state(points, visible)
            canvas._view.scale = 2.0
            canvas.set_rendered_image(mesh)
            self.assertEqual(canvas.get_hand_state(), (points, visible))
            self.assertEqual(canvas._view.scale, 2.0)
            canvas.set_rendered_image(None)
            self.assertEqual(canvas.get_hand_state(), (points, visible))


if __name__ == "__main__":
    unittest.main()
