import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from frontend_runtime import SingleInstance
from remote_frontend.preview_server import byte_range
from remote_frontend.session import Session, browser_url, parser, prepare_config


class RemoteFrontendTest(unittest.TestCase):
    def test_remote_instances_do_not_activate_each_other_or_local_desktop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with SingleInstance("qc", runtime_dir=root / "desktop") as desktop:
                with patch.dict("os.environ", ORBBEC_FRONTEND_RUNTIME_DIR=str(root / "alice")):
                    with SingleInstance("qc") as alice:
                        with patch.dict("os.environ", ORBBEC_FRONTEND_RUNTIME_DIR=str(root / "bob")):
                            with SingleInstance("qc") as bob:
                                self.assertTrue(desktop.acquired and alice.acquired and bob.acquired)
                                self.assertNotEqual(alice.path, bob.path)

    def test_config_moves_compute_and_caches_to_private_worker_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "launch.json"
            original = {"backend_url": "http://127.0.0.1:8765", "operator_id": "desktop",
                        "nas_mounts": {"nas://ego": "nas"}, "mano_model_dir": "models"}
            config.write_text(json.dumps(original))
            alice = prepare_config(config, root / "alice", root / "a-cache", "alice", "qc")
            bob = prepare_config(config, root / "bob", root / "b-cache", "bob", "qc")
            self.assertEqual(alice["nas_mounts"]["nas://ego"], str((root / "nas").resolve()))
            self.assertEqual(alice["mano_model_dir"], str((root / "models").resolve()))
            for key in ("worker_machine_id", "operator_id", "state_dir", "tmp_dir", "frame_cache_dir"):
                self.assertNotEqual(alice[key], bob[key])
            self.assertEqual(json.loads(config.read_text()), original)

    def test_token_is_only_in_fragment_and_gateway_path(self):
        url = urlsplit(browser_url(16091, "test-bearer"))
        self.assertEqual(url.netloc, "127.0.0.1:16091")
        self.assertEqual(url.query, "")
        self.assertEqual(parse_qs(url.fragment)["path"], ["websockify?token=test-bearer"])

    def test_failed_start_does_not_delete_an_existing_connection_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parser().parse_args(["qc", "--name", "existing", "--operator", "worker", "--config", "x",
                "--display", "91", "--web-port", "16091", "--vnc-port", "15991", "--state-dir", tmp])
            session = Session(args)
            session.state.mkdir()
            record = session.state / "connection.json"
            record.write_text("preserve")
            session.close()
            self.assertEqual(record.read_text(), "preserve")

    def test_preview_seek_ranges(self):
        self.assertEqual(byte_range("", 100), (0, 99))
        self.assertEqual(byte_range("bytes=40-", 100), (40, 99))
        self.assertEqual(byte_range("bytes=40-999", 100), (40, 99))
        self.assertEqual(byte_range("bytes=-20", 100), (80, 99))
        for value in ("bytes=100-", "bytes=20-10", "bytes=-0", "bytes=0-1,20-21", "items=0-1"):
            with self.assertRaises(ValueError, msg=value):
                byte_range(value, 100)


if __name__ == "__main__":
    unittest.main()
