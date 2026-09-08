from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from label.runtime_env import tracking_python, ensure_tracking_environment


class LabelRuntimeEnvironmentTest(unittest.TestCase):
    def test_keeps_current_python_if_torch_is_available(self):
        with patch.dict('os.environ', {}, clear=True), \
             patch('label.runtime_env.importlib.util.find_spec', return_value=object()), \
             patch('label.runtime_env.subprocess.run') as run:
            self.assertIsNone(tracking_python())
            run.assert_not_called()

    def test_finds_existing_track_environment_when_base_has_no_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / 'envs/track/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.dict('os.environ', {}, clear=True), \
                 patch('label.runtime_env.sys.prefix', tmp), \
                 patch('label.runtime_env.importlib.util.find_spec', return_value=None), \
                 patch('label.runtime_env.subprocess.run') as run:
                run.return_value.returncode = 0
                self.assertEqual(tracking_python(), python)

    def test_restart_preserves_launch_config_arguments(self):
        with patch('label.runtime_env.tracking_python', return_value=Path('/env/track/bin/python')), \
             patch('label.runtime_env.sys.argv', ['main.py', '--config', '/tmp/label.json']), \
             patch('label.runtime_env.os.execv') as restart:
            ensure_tracking_environment()
            self.assertEqual(restart.call_args.args[1][-2:], ['--config', '/tmp/label.json'])
