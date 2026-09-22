"""Exercise the actual confirmation method without opening a desktop window."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# frontend_runtime's Linux process lock is unrelated to frame confirmation.
# Isolate that import on Windows; never exercise or emulate SingleInstance here.
if importlib.util.find_spec("fcntl") is None:
    sys.modules["fcntl"] = types.ModuleType("fcntl")
    try:
        from label.app import LabelPage
    finally:
        sys.modules.pop("fcntl", None)
else:
    from label.app import LabelPage
from label.storage import CorrectionProgress, CorrectionTask, load_correction_progress, save_correction_progress


class LabelFrameDecisionTest(unittest.TestCase):
    def setUp(self):
        self.original = ([[[10,20]]*21]*2, [[True]*21]*2)
        self.edited = ([[[30,40]]*21]*2, [[False]*21]*2)
        self.client = Mock()
        self.page = SimpleNamespace(
            _active_task=SimpleNamespace(key="1",frames=[20,21],total_frames=2),
            _active_key="1",_jsonl_path="unused",_show_skeleton=False,
            _camera_ids=["00","01"],_frame_pos=0,_progress={},_show_mano=False,
            _tracked_joints_by_cam={},_view_states={},
            _backend_session=SimpleNamespace(client=self.client,episode_id="episode",operator_id="alice"))
        for method in ("_save_bundle","_cache_current_source_state","_invalidate_corrected_source_cache",
                       "_fail_backend_job","_update_tree_row","_update_submit_button",
                       "_reset_visualizations","_update_mano_button","_load_current_sample"):
            setattr(self.page,method,Mock())
        self.page._track_selected_to_frame=Mock(return_value=False)
        self.page._build_visible_mano_view_state=Mock(return_value=self.original)
        self.page._build_initial_view_state=Mock(side_effect=lambda f,c,s:self.original if s=="mano_visible" else self.edited)
        self.apply=self.enterContext(patch("label.app.apply_view_state_to_corrected"))
        self.enterContext(patch("label.app.save_corrected_array"))
        self.enterContext(patch("label.app.save_correction_progress"))
        self.error=self.enterContext(patch("label.app.messagebox.showerror"))
        self.warning=self.enterContext(patch("label.app.messagebox.showwarning"))

    def test_no_error_preserves_original_across_cameras_and_records_backend(self):
        LabelPage._confirm(self.page,no_error=True)
        self.assertEqual(self.apply.call_count,2)
        for call in self.apply.call_args_list:
            self.assertEqual(call.args[3:],self.original)
        self.client.record_label_frames.assert_called_once_with("episode","alice",[20],"no_error")
        self.assertEqual(self.page._progress["1"].no_error_positions,{0})
        self.assertEqual(self.page._progress["1"].done_positions,{0})
        self.page._track_selected_to_frame.assert_not_called()

    def test_normal_confirmation_replaces_no_error_and_saves_edits(self):
        self.page._progress["1"]=CorrectionProgress("1",done_positions={0},no_error_positions={0})
        LabelPage._confirm(self.page)
        self.client.record_label_frames.assert_called_once_with("episode","alice",[20],"corrected")
        self.assertEqual(self.page._progress["1"].no_error_positions,set())
        self.assertEqual(self.apply.call_args.args[3:],self.edited)

    def test_backend_failure_does_not_mark_or_advance_frame(self):
        self.client.record_label_frames.side_effect=RuntimeError("connection failed")
        LabelPage._confirm(self.page,no_error=True)
        self.assertEqual(self.page._frame_pos,0)
        self.assertEqual(self.page._progress,{})
        self.error.assert_called_once()

    def test_missing_original_does_not_record_false_positive(self):
        self.page._build_visible_mano_view_state.return_value=None
        LabelPage._confirm(self.page,no_error=True)
        self.apply.assert_not_called()
        self.client.record_label_frames.assert_not_called()
        self.warning.assert_called_once()

    def test_progress_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/"tasks.jsonl")
            task=CorrectionTask(1,tmp,"S","task","ep",["00"],[20,21])
            record=CorrectionProgress("1",done_positions={0,1},total_frames=2,no_error_positions={1})
            save_correction_progress(path,{"1":record})
            restored=load_correction_progress(path,[task])["1"]
            self.assertEqual(restored.no_error_positions,{1})
            self.assertEqual(restored.done_positions,{0,1})


if __name__=="__main__":
    unittest.main()
