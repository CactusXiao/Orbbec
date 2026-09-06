from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from frontend_runtime import SingleInstance


class SingleInstanceTest(unittest.TestCase):
    def test_same_type_is_blocked_and_other_type_can_run(self):
        with tempfile.TemporaryDirectory() as temp:
            with SingleInstance('label', runtime_dir=temp) as label:
                self.assertTrue(label.acquired)
                command = [sys.executable, '-c',
                    'from frontend_runtime import SingleInstance; import sys\n'
                    'with SingleInstance(sys.argv[1], runtime_dir=sys.argv[2]) as i: print(i.acquired)',
                    'label', temp]
                self.assertEqual(subprocess.check_output(command, text=True).strip(), 'False')
                app = Mock()
                label.attach(app)
                app.after.call_args.args[1]()
                app.lift.assert_called_once()
                command[-2] = 'qc'
                self.assertEqual(subprocess.check_output(command, text=True).strip(), 'True')
            with SingleInstance('label', runtime_dir=temp) as reopened:
                self.assertTrue(reopened.acquired)

    def test_dead_owner_does_not_leave_permanent_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            child = subprocess.Popen([sys.executable, '-u', '-c',
                'from frontend_runtime import SingleInstance; import sys,time\n'
                'with SingleInstance("qc", runtime_dir=sys.argv[1]):\n print("ready"); time.sleep(30)', temp],
                stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'ready')
                child.kill()
                child.wait(5)
                with SingleInstance('qc', runtime_dir=temp) as next_instance:
                    self.assertTrue(next_instance.acquired)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                child.stdout.close()


class LabelCleanupTest(unittest.TestCase):
    def test_stop_waits_for_last_decode_write_before_removing_cache(self):
        import queue
        from label.app import LabelPage
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / 'rgb'
            cache.mkdir()
            stop = threading.Event()
            def decode():
                stop.wait(3)
                (cache / 'last.png').write_bytes(b'image')
            worker = threading.Thread(target=decode)
            worker.start()
            page = SimpleNamespace(_decode_stop=stop, _decode_poll_id=None,
                _stop_original_mesh_cache=Mock(), _decode_thread=worker,
                _decode_results=queue.Queue(), _decode_cache_dir=cache)
            LabelPage._cleanup_decode_cache(page)
            self.assertFalse(worker.is_alive())
            self.assertFalse(cache.exists())
            self.assertIsNone(page._decode_cache_dir)

    def test_release_failure_prevents_window_destroy(self):
        from label.app import LabelToolApp
        app = SimpleNamespace(_label=Mock(), destroy=Mock())
        app._label.release_on_exit.return_value = False
        LabelToolApp._on_exit(app)
        app.destroy.assert_not_called()
        app._label.release_on_exit.return_value = True
        LabelToolApp._on_exit(app)
        app.destroy.assert_called_once()

    def test_renderer_finishes_before_close_removes_cache(self):
        from label.mesh_cache import OriginalMeshCache
        with tempfile.TemporaryDirectory() as temp:
            task = SimpleNamespace(cameras=[], frames=[])
            entered = threading.Event()
            def render(**kwargs):
                entered.set()
                kwargs['stop_event'].wait(3)
                (kwargs['cache_dir'] / 'last.jpg').write_bytes(b'preview')
            with patch('label.mesh_cache._prepare_mesh_frames', side_effect=render):
                cache = OriginalMeshCache(task, settings=object(), cache_parent=temp)
                self.assertTrue(entered.wait(3))
                cache.close()
                cache.close()
                self.assertFalse(cache.cache_dir.exists())
                self.assertFalse(cache._thread.is_alive())


class QcCleanupTest(unittest.TestCase):
    def app(self):
        from src.qc.app import QcWorkerApp
        from src.qc.state_store import QcProgress
        p = QcProgress(task_name='task', episode_id='episode', job_id='job',
            worker_machine_id='worker', lease_until='2099-01-01T00:00:00Z', sample_interval=1,
            current_frame=0, frames=[0,1], playback_complete=True)
        app = SimpleNamespace(_current_progress=p, _current_media=Mock(), _pending_work=0,
            _closing=False, _exit_after_cleanup=False, _media_stop=threading.Event(),
            _cancel_heartbeat=Mock(), qc_page=Mock(), decode_page=Mock(),
            state_store=Mock(), client=Mock(), destroy=Mock(), after=Mock(),
            _last_task_name='task', show_episodes=Mock())
        app._run_bg=lambda work, done, **kw: done(work())
        app._wait_for_cleanup=lambda: QcWorkerApp._wait_for_cleanup(app)
        return app

    def test_exit_and_return_release_and_cleanup_without_renewal(self):
        from src.qc.app import QcWorkerApp
        for exit_after in (True, False):
            app=self.app(); media=app._current_media; progress=app._current_progress
            QcWorkerApp.handle_interrupt(app, exit_after=exit_after)
            media.close.assert_called_once_with(cleanup=True)
            app.client.release_qc_job.assert_called_once_with('job', reason='operator_left_qc')
            app.client.heartbeat_qc_job.assert_not_called()
            self.assertTrue(progress.playback_complete)
            self.assertEqual(progress.lease_until, '')
            self.assertEqual(app.destroy.call_count, int(exit_after))
            self.assertEqual(app.show_episodes.call_count, int(not exit_after))

    def test_waits_for_late_lease_and_releases_it(self):
        from src.qc.app import QcWorkerApp
        app=self.app(); progress=app._current_progress
        app._current_progress=None; app._pending_work=1
        QcWorkerApp.handle_interrupt(app, exit_after=True)
        app.destroy.assert_not_called()
        app.client.release_qc_job.assert_not_called()
        QcWorkerApp._after_progress_ready(app, progress, None)
        self.assertIs(app._current_progress, progress)
        app._pending_work=0
        app.after.call_args.args[1]()
        app.client.release_qc_job.assert_called_once()
        app.destroy.assert_called_once()

    def test_release_failure_cleans_media_and_allows_retry(self):
        from src.qc.app import QcWorkerApp
        app=self.app(); media=app._current_media
        app.client.release_qc_job.side_effect=RuntimeError('offline')
        with patch('src.qc.app.messagebox.showerror') as error:
            QcWorkerApp.handle_interrupt(app, exit_after=True)
        error.assert_called_once()
        app.destroy.assert_not_called()
        self.assertFalse(app._closing)
        self.assertIsNotNone(app._current_progress)
        media.close.assert_called_once_with(cleanup=True)
        app.client.release_qc_job.side_effect=None
        QcWorkerApp.handle_interrupt(app, exit_after=True)
        app.destroy.assert_called_once()

    def test_media_stop_waits_for_decoder_before_cache_removal(self):
        from src.qc.media import _QcMediaPreparation, QcEpisodeMedia
        with tempfile.TemporaryDirectory() as temp:
            cache=Path(temp)/'decoded'; cache.mkdir()
            prep=_QcMediaPreparation(task=Mock(), payload={}, cache_dir=cache,
                settings=Mock(), on_progress=None, cameras=[])
            def write_last_frame():
                prep.stop_event.wait(3)
                (cache/'last.jpg').write_bytes(b'last')
            prep._thread=threading.Thread(target=write_last_frame)
            prep._thread.start()
            media=QcEpisodeMedia(task=Mock(), cache_dir=cache, preparation=prep)
            media.close(cleanup=True)
            self.assertFalse(prep._thread.is_alive())
            self.assertFalse(cache.exists())


class QcWindowLifecycleTest(unittest.TestCase):
    def test_close_during_preparation_waits_and_releases(self):
        import gc
        import tkinter as tk
        from dataclasses import replace
        from src.qc.app import QcWorkerApp
        from src.qc.config import load_qc_config
        from src.qc.media import QcEpisodeMedia
        from src.qc.state_store import QcProgress
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cache = base / 'rgb'
            entered = threading.Event()
            config = replace(load_qc_config(), tmp_dir=base/'cache', state_dir=base/'state')
            client = Mock()
            client.available_qc_items.return_value = []
            progress = QcProgress(task_name='task', episode_id='episode', job_id='job',
                worker_machine_id=config.worker_machine_id, lease_until='2099-01-01T00:00:00Z',
                sample_interval=1, current_frame=0, frames=[0, 1])
            def prepare(*args, **kwargs):
                cache.mkdir()
                media = QcEpisodeMedia(task=Mock(), cache_dir=cache)
                kwargs['on_media_created'](media)
                entered.set()
                kwargs['stop_event'].wait(5)
                (cache/'last.jpg').write_bytes(b'late write')
                return media
            with patch('src.qc.app.QcBackendClient', return_value=client), \
                 patch('src.qc.app.prepare_qc_media', side_effect=prepare), \
                 patch('src.qc.app.messagebox.showerror') as errors:
                try:
                    app = QcWorkerApp(config)
                except tk.TclError as exc:
                    self.skipTest(str(exc))
                callback_errors = []
                def callback_failed(*error):
                    callback_errors.append(error)
                    app.quit()
                app.report_callback_exception = callback_failed
                app.withdraw()
                app._after_progress_ready(progress, None)
                def close_when_started():
                    if entered.is_set():
                        app.request_exit()
                    else:
                        app.after(10, close_when_started)
                app.after(10, close_when_started)
                app.after(5000, app.quit)
                app.mainloop()
                self.assertFalse(callback_errors)
                self.assertIsNone(app._current_progress)
                self.assertFalse(cache.exists())
                self.assertEqual(app._pending_work, 0)
                errors.assert_not_called()
                client.release_qc_job.assert_called_once()
                client.heartbeat_qc_job.assert_not_called()
                app = None
                gc.collect()

    def test_stubborn_renderer_is_killed_before_cache_removal(self):
        from src.qc.media import _QcMediaPreparation, QcEpisodeMedia
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / 'cache'
            cache.mkdir()
            process = subprocess.Popen([sys.executable, '-u', '-c',
                'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready"); time.sleep(30)'],
                start_new_session=True, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), 'ready')
                prep = _QcMediaPreparation(task=Mock(), payload={}, cache_dir=cache,
                    settings=Mock(), on_progress=None, cameras=[])
                prep.register_renderer(process)
                prep._thread = threading.Thread(target=process.wait)
                prep._thread.start()
                QcEpisodeMedia(task=Mock(), cache_dir=cache, preparation=prep).close(cleanup=True)
                self.assertIsNotNone(process.poll())
                self.assertFalse(prep._thread.is_alive())
                self.assertFalse(cache.exists())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()


if __name__ == '__main__':
    unittest.main()
