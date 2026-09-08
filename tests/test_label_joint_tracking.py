from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from label.storage import CorrectionProgress
from label.tracking import CoTrackerRuntime
from tests import test_label_overview as overview_tests


class LabelJointTrackingTest(unittest.TestCase):
    setUp = overview_tests.LabelOverviewTest.setUp
    tearDown = overview_tests.LabelOverviewTest.tearDown
    state = staticmethod(overview_tests.LabelOverviewTest.state)

    def double_click(self, hand=0, joint=4):
        canvas = self.page._canvas
        centers, scale = canvas._visibility_schematic_layout()
        x, y = canvas._schematic_point(hand, joint, *centers[hand], scale)
        event = SimpleNamespace(x=x, y=y)
        canvas._on_left_down(event)
        canvas._on_left_up(event)
        canvas._on_left_double(event)
        canvas._on_left_up(event)

    def test_double_click_toggles_tracking_without_changing_visibility_or_undo(self):
        p = self.page
        p._canvas.set_hand_state(*self.state(30, True))
        before = p._canvas.get_hand_state()
        self.double_click()
        self.assertEqual(p._tracked_joints_by_cam['00'], {(0, 4)})
        self.assertEqual(p._canvas.get_hand_state(), before)
        self.assertEqual(p._canvas._history, [])
        self.assertTrue(p._canvas.find_withtag('tracking-0-4'))
        self.double_click()
        self.assertEqual(p._tracked_joints_by_cam['00'], set())
        self.assertEqual(p._canvas.get_hand_state(), before)
        self.assertFalse(p._canvas.find_withtag('tracking-highlight'))

    def test_invisible_point_is_rejected_even_though_first_click_makes_it_visible(self):
        with patch('label.app.messagebox.showwarning') as warning:
            self.double_click()
        warning.assert_called_once()
        self.assertFalse(self.page._canvas.get_hand_state()[1][0][4])
        self.assertFalse(self.page._tracked_joints_by_cam['00'])

    def test_single_click_still_toggles_visibility(self):
        canvas = self.page._canvas
        centers, scale = canvas._visibility_schematic_layout()
        x, y = canvas._schematic_point(0, 4, *centers[0], scale)
        canvas._on_left_down(SimpleNamespace(x=x, y=y))
        canvas._on_left_up(SimpleNamespace(x=x, y=y))
        self.assertTrue(canvas._visible[0][4])
        self.assertEqual(self.page._tracked_joints_by_cam, {})

    def test_real_tk_double_click_preserves_visibility(self):
        p = self.page
        p._canvas.set_hand_state(*self.state(30, True))
        centers, scale = p._canvas._visibility_schematic_layout()
        x, y = p._canvas._schematic_point(0, 4, *centers[0], scale)
        for time in (1000, 1100):
            p._canvas.event_generate('<ButtonPress-1>', x=int(x), y=int(y), time=time)
            p._canvas.event_generate('<ButtonRelease-1>', x=int(x), y=int(y), time=time+30)
        self.root.update()
        self.assertEqual(p._tracked_joints_by_cam['00'], {(0, 4)})
        self.assertTrue(p._canvas._visible[0][4])

    def test_confirmed_destination_is_not_overwritten(self):
        p = self.page
        p._canvas.set_hand_state(*self.state(30, True))
        self.double_click()
        p._progress['overview'] = CorrectionProgress(task_key='overview', total_frames=2, done_positions={1})
        p._tracker = Mock()
        p._skip_frame()
        p._tracker.track_points.assert_not_called()
        self.assertEqual(p._canvas.get_hand_state(), self.state(12))

    def test_confirm_also_tracks_to_next_frame(self):
        p = self.page
        p._jsonl_path = 'unused.jsonl'
        p._canvas.set_hand_state(*self.state(30, True))
        self.double_click()
        p._tracker = Mock()
        p._tracker.track_points.return_value = self.state(200, True)
        with patch('label.app.apply_view_state_to_corrected'), patch('label.app.save_corrected_array'), \
             patch('label.app.save_correction_progress'), patch.object(p, '_update_tree_row'), \
             patch.object(p, '_invalidate_corrected_source_cache'):
            p._confirm()
        p._tracker.track_points.assert_called_once()
        self.assertEqual(p._frame_pos, 1)
        self.assertEqual(p._canvas._points[0][4], (204.0, 44.0))

    def test_highlights_follow_camera_including_overview_and_skeleton_overlay(self):
        p = self.page
        p._canvas.set_hand_state(*self.state(30, True))
        self.double_click()
        p._select_camera(1)
        self.assertFalse(p._canvas.find_withtag('tracking-highlight'))
        p._select_camera(-1)
        self.root.update()
        for camera, canvas in p._overview_grid.canvases.items():
            self.assertEqual(bool(canvas.find_withtag('tracking-highlight')), camera == '00')
        canvas = p._overview_grid.canvases['00']
        canvas.set_skeleton_overlay(*self.state(40, True))
        canvas.set_annotation_visible(False)
        self.assertTrue(canvas.find_withtag('tracking-0-4'))
        p._select_camera(0)
        self.assertEqual(p._canvas._tracked_joints, {(0, 4)})

    def test_forward_tracking_uses_unsaved_point_and_preserves_other_joints_and_cameras(self):
        p = self.page
        initial = self.state(30, True)
        initial[0][0][4] = (123.0, 87.0)
        p._canvas.set_hand_state(*initial)
        self.double_click()
        predicted = self.state(200, True)
        p._tracker = Mock()
        p._tracker.track_points.return_value = predicted
        p._skip_frame()
        call = p._tracker.track_points.call_args.kwargs
        self.assertEqual((call['prev_frame_idx'], call['frame_idx'], call['cam_id']), (5, 12, '00'))
        self.assertEqual(call['points'][0][4], (123.0, 87.0))
        self.assertEqual(sum(sum(hand) for hand in call['visible']), 1)
        expected = self.state(12)
        expected[0][0][4] = predicted[0][0][4]
        expected[1][0][4] = True
        self.assertEqual(p._canvas.get_hand_state(), expected)
        p._select_camera(-1)
        self.assertEqual(p._overview_grid.canvases['00'].get_hand_state(), expected)
        self.assertEqual(p._overview_grid.canvases['01'].get_hand_state(), self.state(13))
        p._back_frame()
        p._skip_frame()
        p._tracker.track_points.assert_called_once()

    def test_tracking_failure_keeps_destination_annotation(self):
        p = self.page
        p._canvas.set_hand_state(*self.state(30, True))
        self.double_click()
        p._tracker = Mock()
        p._tracker.track_points.side_effect = RuntimeError('model unavailable')
        with patch('label.app.messagebox.showwarning') as warning:
            p._skip_frame()
        warning.assert_called_once()
        self.assertEqual(p._canvas.get_hand_state(), self.state(12))

    def test_empty_selection_does_not_load_model(self):
        runtime = CoTrackerRuntime()
        with patch.object(runtime, '_ensure_model') as model:
            points, visible = runtime.track_points(
                episode_dir=self.page._active_task.episode_dir(), cam_id='00',
                prev_frame_idx=5, frame_idx=12, points=self.state(20)[0],
                visible=self.state(20)[1],
            )
        model.assert_not_called()
        self.assertFalse(any(any(hand) for hand in visible))
