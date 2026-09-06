from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.qc.app import QcPage, QcWorkerApp
from src.qc.state_store import QcProgress


class QcPlaybackCompletionTest(unittest.TestCase):
    def progress(self, complete=False):
        return QcProgress(task_name="task", episode_id="episode", job_id="job",
                          worker_machine_id="worker", lease_until="2099-01-01T00:00:00Z",
                          sample_interval=1, current_frame=0, frames=[0, 1, 2],
                          playback_complete=complete)

    def test_finished_playback_remains_complete_after_seeking_and_reloading(self):
        progress = self.progress()
        page = SimpleNamespace(progress=progress, _playing=True, _play_after_id=None,
            _buffering=False, _play_clock_started_at=0.0, _play_clock_start_position=0,
            _ticks_since_save=0, media=None, _busy_text="", mode="playback", _timeline_dragging=False,
            app=SimpleNamespace(config=SimpleNamespace(playback_fps=30), save_current_progress=Mock()),
            _current_position=lambda: progress.frames.index(progress.current_frame),
            pause_playback=Mock(), _refresh=Mock(), _schedule_play_tick=Mock())
        with patch("src.qc.app.time.monotonic", return_value=1.0):
            QcPage._play_tick(page)
        self.assertTrue(progress.playback_complete)
        for position, commit in ((1, False), (0, True)):
            QcPage.seek_position(page, position, commit)
            self.assertTrue(progress.is_complete)
        loaded = QcProgress.from_dict(progress.to_dict())
        self.assertEqual(loaded.current_frame, 0)
        self.assertTrue(loaded.is_complete)

    def test_seeking_to_end_without_playing_does_not_grant_completion(self):
        progress = self.progress()
        page = SimpleNamespace(progress=progress, mode="playback", _busy_text="",
            _timeline_dragging=False, _play_after_id=None, _playing=False,
            app=SimpleNamespace(save_current_progress=Mock()), _refresh=Mock())
        QcPage.seek_position(page, 2)
        self.assertFalse(progress.is_complete)

    def test_submit_button_remains_enabled_during_replay(self):
        progress = self.progress(complete=True)
        states = {}
        page = SimpleNamespace(progress=progress, mode="playback", _busy_text="",
            _playing=True, _buffering=False, _timeline_dragging=False,
            _playback_bar=Mock(), _bad_bar=Mock(), _playback_status=Mock(),
            _set_enabled=lambda button, enabled: states.__setitem__(button, enabled),
            app=SimpleNamespace(config=SimpleNamespace(playback_fps=30)))
        page._bad_bar.winfo_manager.return_value = ""
        for name in ("_prev_ten", "_prev_one", "_next_one", "_next_ten", "_reject_button",
                     "_bad_episode_button", "_play_button", "_submit_button"):
            setattr(page, name, Mock())
        QcPage._update_controls(page)
        self.assertTrue(states[page._submit_button])
        progress.playback_complete = False
        QcPage._update_controls(page)
        self.assertFalse(states[page._submit_button])

    def test_submit_at_beginning_after_playback_reaches_submission_path(self):
        progress = self.progress(complete=True)
        app = SimpleNamespace(_current_progress=progress, _current_media=Mock(),
            config=SimpleNamespace(range_merge_gap_frames=5, worker_machine_id="worker", operator_id="qc"),
            qc_page=Mock(), _run_bg=Mock())
        with patch("src.qc.app.messagebox.showwarning") as warning:
            QcWorkerApp.submit_current_progress(app)
            warning.assert_not_called()
        app._run_bg.assert_called_once()
        app.qc_page.set_busy.assert_called_once_with("正在提交 QC 结果...")


if __name__ == "__main__":
    unittest.main()
