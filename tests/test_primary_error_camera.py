"""QC camera metadata survives persistence and drives one-time Label entry."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from label.app import LabelPage
from label.storage import CorrectionTask, CorrectionProgress, save_correction_progress, load_correction_progress
from src.qc.app import QcPage
from src.qc.report import build_qc_result, write_qc_report, write_ego_pose_qc_report
from src.qc.state_store import QcProgress, normalize_segments
from tests import test_browser_parity


class PrimaryErrorCameraTest(unittest.TestCase):
    def test_selection_required_and_only_selected_canvas_highlighted(self):
        page = object.__new__(QcPage)
        page.progress = SimpleNamespace(bad_frame_ranges=[], ego_bad_frame_ranges=[], bad_frame_segments=[], ego_bad_frame_segments=[])
        page.app = SimpleNamespace(config=SimpleNamespace(range_merge_gap_frames=5))
        page.mode = 'bad_range'
        page.bad_start, page.bad_end = 10, 12
        page.primary_camera = None
        page._canvases = {'00': Mock(), '02': Mock()}
        page._update_controls = Mock()
        page._persist_and_refresh = lambda: page._refresh_primary_camera()
        with patch('src.qc.app.messagebox.showwarning') as warning:
            page.confirm_bad_range('hand_pose')
            warning.assert_called_once()
        self.assertEqual(page.progress.bad_frame_ranges, [])
        page.select_primary_camera('00')
        page.select_primary_camera('02')
        self.assertEqual(page._canvases['02'].configure.call_args.kwargs['highlightbackground'], '#e5484d')
        self.assertNotEqual(page._canvases['00'].configure.call_args.kwargs['highlightbackground'], '#e5484d')
        page.confirm_bad_range('hand_pose')
        self.assertEqual(page.progress.bad_frame_segments, [dict(start_frame=10, end_frame=12, primary_camera='02')])
        self.assertIsNone(page.primary_camera)
        self.assertNotEqual(page._canvases['02'].configure.call_args.kwargs['highlightbackground'], '#e5484d')
        page.select_primary_camera('00')
        self.assertIsNone(page.primary_camera)

    def test_segments_persist_and_reports_keep_distinct_cameras(self):
        segments = normalize_segments([
            dict(start_frame=10, end_frame=12, primary_camera='02'),
            dict(start_frame=13, end_frame=14, primary_camera='03'),
            dict(start_frame=14, end_frame=15, primary_camera='02'),
        ])
        self.assertEqual(segments, [dict(start_frame=10, end_frame=15, primary_camera='02'), dict(start_frame=13, end_frame=14, primary_camera='03')])
        progress = QcProgress(task_name='t', episode_id='e', job_id='j', worker_machine_id='w', lease_until='', sample_interval=10, current_frame=0,
                              bad_frame_ranges=[(10, 15)], bad_frame_segments=segments, ego_bad_frame_segments=[dict(start_frame=1, end_frame=2, primary_camera='ego')])
        restored = QcProgress.from_dict(json.loads(json.dumps(progress.to_dict())))
        self.assertEqual(restored.bad_frame_segments, segments)
        self.assertEqual(restored.ego_bad_frame_segments, progress.ego_bad_frame_segments)
        with tempfile.TemporaryDirectory() as tmp:
            result = build_qc_result(episode_id='e', worker_id='w', bad_ranges=progress.bad_frame_ranges, segments=segments)
            path = write_qc_report(episode_dir=Path(tmp), result=result, bad_ranges=progress.bad_frame_ranges, sample_interval=10)
            self.assertEqual(json.loads(path.read_text())['segments'], segments)
            ego = write_ego_pose_qc_report(episode_dir=Path(tmp), episode_id='e', worker_id='w', operator_id='o', bad_ranges=[(1, 2)], segments=progress.ego_bad_frame_segments)
            self.assertEqual(json.loads(ego.read_text())['segments'][0]['primary_camera'], 'ego')
            bad = build_qc_result(episode_id='e', worker_id='w', bad_ranges=[(10, 15)], segments=segments, bad_episode=True)
            self.assertEqual(bad['segments'], [])

    def test_label_first_entry_middle_revisit_reload_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = CorrectionTask(1, tmp, 's', 't', 'e', ['00', '02', '03'], list(range(40)), segments=[
                dict(segment_id='a', start_frame=10, end_frame=15, primary_camera='02'),
                dict(segment_id='b', start_frame=20, end_frame=22, primary_camera='03'),
                dict(segment_id='old', start_frame=25, end_frame=26),
                dict(segment_id='ego', start_frame=30, end_frame=32, primary_camera='ego'),
            ])
            rec = CorrectionProgress(task_key=task.key, total_frames=40)
            observed = [rec.enter_segment(task, f) for f in [0, 12, 13, 20, 10, 25, 30, 31]]
            self.assertEqual(observed, [None, '02', None, '03', None, None, 'ego', None])
            path = str(Path(tmp) / 'progress.jsonl')
            save_correction_progress(path, {task.key: rec})
            loaded = load_correction_progress(path, [task])[task.key]
            self.assertIsNone(loaded.enter_segment(task, 12))
            browser = test_browser_parity.WorkflowParityTest().js('''import {enterLabelSegment} from './remote_frontend/web/workflow.js';
              let raw='';for await(const c of process.stdin)raw+=c;const manifest=JSON.parse(raw);
              let d={manifest};const out=[0,12,13,20,10,25,30,31].map(f=>enterLabelSegment(d,f));
              d=JSON.parse(JSON.stringify(d));out.push(enterLabelSegment(d,12));console.log(JSON.stringify(out));''',
              {'cameras': task.cameras, 'qc_segments': task.segments})
            self.assertEqual(browser, observed + [None])

    def test_desktop_frame_loading_switches_only_on_first_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = CorrectionTask(1, tmp, 's', 't', 'e', ['00', '02'], [10, 11, 20, 21], segments=[
                dict(segment_id='a', start_frame=10, end_frame=11, primary_camera='02'),
                dict(segment_id='b', start_frame=20, end_frame=21, primary_camera='ego'),
            ])
            page = object.__new__(LabelPage)
            page._active_task, page._active_key = task, task.key
            page._bundles = {'pred': object()}
            page._camera_ids, page._cam_idx = task.cameras, 0
            page._frame_pos, page._progress = 0, {}
            page._jsonl_path = str(Path(tmp) / 'task.jsonl')
            page._refresh_view = Mock()
            page._load_current_sample()
            self.assertEqual(page._cam_idx, 1)
            self.assertFalse(page._overview)
            page._cam_idx = 0  # User chooses another view.
            page._frame_pos = 1
            page._load_current_sample()
            self.assertEqual(page._cam_idx, 0)
            page._frame_pos = 2
            page._load_current_sample()
            self.assertTrue(page._overview)
            self.assertTrue(page._primary_ego_focus)
            page._frame_pos = 0
            page._load_current_sample()
            self.assertTrue(page._primary_ego_focus)  # Revisiting keeps current view.
            page._refresh_view.assert_called()

    def test_browser_requires_camera_and_preserves_camera_during_boundary_changes(self):
        result = test_browser_parity.WorkflowParityTest().js('''import {QcWorkflow} from './remote_frontend/web/workflow.js';
          const r={bad_ranges:[],ego_ranges:[]},q=new QcWorkflow([0,1,2,3],r);
          q.enterBadRange();q.boundary('start');q.step(1);q.boundary('end');
          let required=false;try{q.confirmBadRange('hand_pose');}catch{required=true;}
          q.primaryCamera='02';q.step(1);q.boundary('end');q.confirmBadRange('hand_pose');
          q.enterBadRange();const cleared=q.primaryCamera===null;q.primaryCamera='03';q.boundary('start');q.step(1);q.boundary('end');q.confirmBadRange('hand_pose');
          console.log(JSON.stringify({required,cleared,segments:r.bad_segments}));''', None)
        self.assertEqual(result, dict(required=True, cleared=True, segments=[dict(start_frame=0, end_frame=2, primary_camera='02'), dict(start_frame=2, end_frame=3, primary_camera='03')]))
