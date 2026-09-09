import gc
from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from label.app import LabelPage
from label.env_config import LabelConfig
from label.theme import apply_theme


class LabelOverviewTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        apply_theme(self.root)
        self.root.geometry("1480x920")
        self.temp = tempfile.TemporaryDirectory()
        self.page = LabelPage(self.root, config=LabelConfig(), on_back=Mock())
        self.page.pack(fill="both", expand=True)
        p = self.page
        cameras = [f"{i:02d}" for i in range(7)]
        base = Path(self.temp.name)
        for index, camera in enumerate(cameras):
            (base/camera).mkdir()
            for frame in (5, 12):
                Image.new("RGB", (360, 220), (index * 30, frame * 10, 60)).save(base/camera/f"{frame:05d}.png")
        (base / "ego" / "RGB").mkdir(parents=True)
        for ego_frame in (2, 3):
            Image.new("RGB", (360, 220), (10, 20, 30)).save(base / "ego" / "RGB" / f"{ego_frame:05d}.png")
        (base / "timestamps.csv").write_text("frame_index,ego_frame_index\n5,2\n12,3\n")
        p._active_task = SimpleNamespace(key="overview", frames=[5, 12], total_frames=2,
            cameras=cameras, display_name="Overview test", episode_dir=lambda: base,
            rgb_path_template="{camera}/{frame:05d}.png")
        p._active_key = "overview"
        p._camera_ids = cameras
        p._bundles = {"pred": object()}
        self.patches = [
            patch.object(p, "_build_modified_view_state", side_effect=lambda frame, cam: self.state(int(cam)+frame)),
            patch.object(p, "_build_mano_3d_view_state", side_effect=lambda frame, cam: self.state(90+int(cam), True)),
            patch.object(p, "_view_source_status", return_value="test"),
        ]
        for item in self.patches:
            item.start()
        p._refresh_view()
        self.root.update()

    def tearDown(self):
        self.page.on_hide()
        for item in self.patches:
            item.stop()
        self.root.update_idletasks()
        self.root.destroy()
        self.temp.cleanup()
        self.page = self.root = None
        gc.collect()

    @staticmethod
    def state(value, visible=False):
        return ([[(float(value+j), 40.0+j) for j in range(21)] for _ in range(2)],
                [[visible]*21 for _ in range(2)])

    def shortcut(self, index):
        p = self.page
        return p._camera_shortcut(SimpleNamespace(widget=p._canvas, state=0), index)

    def test_zero_opens_six_readonly_views_and_numbers_restore_edits(self):
        p = self.page
        edited = self.state(42)
        p._canvas.set_hand_state(*edited)
        self.assertEqual(self.shortcut(-1), "break")
        self.root.update()
        self.assertTrue(p._overview)
        self.assertFalse(p._canvas.winfo_ismapped())
        self.assertEqual(list(p._overview_grid.canvases), ["00", "02", "03", "05", "06", "ego"])
        self.assertNotIn("ego", p._camera_ids)
        self.assertNotIn("ego", p._view_states)
        self.assertEqual(p._overview_grid.canvases['00'].get_hand_state(), edited)
        for canvas in p._overview_grid.canvases.values():
            before = canvas.get_hand_state()
            event = SimpleNamespace(x=50, y=50)
            canvas._on_left_down(event)
            canvas._on_left_drag(SimpleNamespace(x=80, y=80))
            canvas._on_left_up(event)
            canvas.undo()
            canvas.ignore_view()
            self.assertTrue(canvas._read_only)
            self.assertEqual(canvas.get_hand_state(), before)
            self.assertGreater(canvas.winfo_width(), 150)
            self.assertGreater(canvas.winfo_height(), 100)
        # A hidden canvas must never overwrite overview state for another frame.
        p._canvas.set_hand_state(*self.state(999))
        p._skip_frame()
        p._back_frame()
        with patch.object(p._canvas, 'undo') as undo, patch.object(p._canvas, 'ignore_view') as ignore:
            p._undo(); p._ignore_view()
            undo.assert_not_called(); ignore.assert_not_called()
        for index in range(7):
            self.shortcut(index)
            self.assertFalse(p._overview)
            self.assertEqual(p._active_cam_id(), f"{index:02d}")
            self.assertFalse(p._canvas._read_only)
        self.shortcut(0)
        self.assertEqual(p._canvas.get_hand_state(), edited)

    def test_original_and_modified_sources_work_in_overview(self):
        p = self.page
        self.shortcut(-1)
        self.assertEqual(p._overview_grid.canvases['02'].get_hand_state(), self.state(7))
        p._toggle_source()
        self.assertEqual(p._overview_grid.canvases['02'].get_hand_state(), self.state(92, True))
        with patch("label.app.load_joint_visibility", return_value=[[False]*21 for _ in range(2)]):
            p._toggle_source()
        self.assertEqual(p._overview_grid.canvases['02'].get_hand_state(), self.state(92, False))
        self.assertTrue(p._overview_grid.canvases['02']._read_only)
        p._toggle_source()
        self.assertEqual(p._overview_grid.canvases['02'].get_hand_state(), self.state(7))

    def test_each_view_zooms_and_pans_independently_and_mano_preserves_transform(self):
        p = self.page
        self.shortcut(-1)
        self.root.update()
        canvases = p._overview_grid.canvases
        camera = canvases['02']
        other = canvases['03']
        other_transform = vars(other._view).copy()
        camera._zoom_at(70, 70, 1.5)
        camera._on_right_down(SimpleNamespace(x=80, y=80))
        camera._on_right_drag(SimpleNamespace(x=100, y=110))
        camera._on_right_up(None)
        transform = vars(camera._view).copy()
        self.assertNotEqual(transform, other_transform)
        self.assertEqual(vars(other._view), other_transform)
        base = Path(self.temp.name)
        cache = Mock()
        cache.error = ''
        cache.done_event.is_set.return_value = False
        cache.path.side_effect = lambda cam, frame: None if cam == '05' else base/cam/f'{frame:05d}.png'
        p._original_mesh_cache = cache
        p._toggle_mano()
        self.assertEqual(vars(camera._view), transform)
        self.assertIsNone(canvases['05']._rendered_image)
        self.assertIsNotNone(p._mesh_poll_id)
        cache.path.side_effect = lambda cam, frame: base/cam/f'{frame:05d}.png'
        p._refresh_original_mesh_preview()
        self.assertIsNone(p._mesh_poll_id)
        self.assertEqual(vars(camera._view), transform)
        for cam, canvas in canvases.items():
            if cam == "ego":
                self.assertIsNone(canvas._rendered_path)
                continue
            self.assertEqual(canvas._rendered_path, base/cam/'00005.png')
            self.assertFalse(canvas._annotation_visible)
        p._skip_frame()
        for cam, canvas in canvases.items():
            if cam == "ego":
                self.assertIsNone(canvas._rendered_path)
                continue
            self.assertEqual(canvas._rendered_path, base/cam/'00012.png')
        p._toggle_mano()
        for canvas in canvases.values():
            self.assertIsNone(canvas._rendered_image)
            self.assertTrue(canvas._read_only)
            self.assertTrue(canvas._annotation_visible)

    def test_confirm_in_overview_saves_edits_and_advances_without_leaving_overview(self):
        p = self.page
        edited = self.state(42)
        p._canvas.set_hand_state(*edited)
        self.shortcut(-1)
        p._canvas.set_hand_state(*self.state(999))
        p._jsonl_path = "unused"
        with patch("label.app.apply_view_state_to_corrected") as apply, \
             patch("label.app.save_corrected_array"), \
             patch("label.app.save_correction_progress"), \
             patch.object(p, "_update_tree_row"):
            p._confirm()
        self.assertEqual(apply.call_count, 7)
        self.assertEqual({call.args[2] for call in apply.call_args_list}, {f"{i:02d}" for i in range(7)})
        self.assertEqual(apply.call_args_list[0].args[3:], edited)
        self.assertEqual(p._frame_pos, 1)
        self.assertTrue(p._overview)
        self.assertEqual(p._progress['overview'].done_positions, {0})
        self.assertTrue(all(canvas._read_only for canvas in p._overview_grid.canvases.values()))

    def test_zero_is_ignored_when_typing(self):
        entry = tk.Entry(self.root)
        event = SimpleNamespace(widget=entry, state=0)
        self.assertIsNone(self.page._camera_shortcut(event, -1))
        self.assertFalse(self.page._overview)


if __name__ == '__main__':
    unittest.main()
