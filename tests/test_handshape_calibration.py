from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.publish_handshape import EPISODE, TASK, Publisher, episode_path, publish_capture


class FakePublisher:
    def __init__(self):
        self.found = False
        self.calls = []
        self.fail = False

    def status(self, episode):
        return {"episode_id": episode, "found": self.found, "state": "shape_calibrated" if self.found else ""}

    def publish(self, episode):
        self.calls.append(episode)
        if self.fail:
            raise RuntimeError("--shape-calibration not supported")
        self.found = True


class HandshapePublishTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = episode_path(self.root / "local", "S001")
        self.source.mkdir(parents=True)
        (self.source / "camera_params.json").write_text("new capture")
        self.nas = self.root / "nas"
        self.nas.mkdir()
        self.dest = episode_path(self.nas, "S001")
        self.publisher = FakePublisher()

    def publish(self, token="capture-1"):
        return publish_capture(self.source, self.nas, "S001", token, self.publisher)

    def test_first_capture_publishes_exact_path_and_retry_is_idempotent(self):
        result = self.publish()
        self.assertEqual(result["episode_id"], "S001/task_handshapeCalibration/episode1")
        self.assertTrue(result["submitted"])
        self.assertEqual((self.dest / "camera_params.json").read_text(), "new capture")
        self.assertTrue(self.source.exists())
        self.publish()
        self.assertEqual(len(self.publisher.calls), 1)

    def test_overwrite_unpublished_capture_removes_stale_files(self):
        self.dest.mkdir(parents=True)
        (self.dest / "old_video.h265").write_text("stale")
        self.publish()
        self.assertFalse((self.dest / "old_video.h265").exists())
        self.assertEqual([p.name for p in self.dest.parent.iterdir() if p.is_dir()], [EPISODE])

    def test_failure_keeps_capture_and_can_retry_same_token(self):
        self.publisher.fail = True
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            self.publish()
        self.assertTrue((self.source / "camera_params.json").exists())
        receipt = json.loads((self.dest.parent / ".episode1.publish.json").read_text())
        self.assertFalse(receipt["submitted"])
        self.publisher.fail = False
        self.assertTrue(self.publish()["submitted"])

    def test_legacy_publisher_does_not_silently_accept_recalibration(self):
        self.publish()
        (self.source / "camera_params.json").write_text("second capture")
        with self.assertRaisesRegex(RuntimeError, "source-replacement protocol"):
            self.publish("capture-2")
        self.assertEqual((self.dest / "camera_params.json").read_text(), "new capture")
        self.assertEqual((self.source / "camera_params.json").read_text(), "second capture")

    def test_rejects_symlink_and_path_traversal(self):
        with self.assertRaises(ValueError):
            episode_path(self.nas, "../other")
        (self.source / "outside").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.publish()
        self.assertFalse(self.dest.exists())

    def test_publisher_uses_shape_flag_and_preserves_subject_as_one_argument(self):
        with patch("scripts.publish_handshape.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            Publisher().publish("S001/task_handshapeCalibration/episode1")
            command = run.call_args.args[0]
            self.assertEqual(command[-1], "sudo -- /usr/local/sbin/nas-uploader-publish --shape-calibration S001/task_handshapeCalibration/episode1")
            self.assertEqual(run.call_args.kwargs["timeout"], 120)


class HandshapeCapturePathTest(unittest.TestCase):
    def test_cpp_capture_overwrite_is_limited_to_one_subject_episode(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            program = root / "test.cpp"
            program.write_text(r'''
#include "handshape_calibration.hpp"
#include <cassert>
#include <fstream>
int main(int argc, char **argv) {
    namespace fs = std::filesystem;
    using namespace sync_app::handshape;
    fs::path root = argv[1];
    prepareEpisode(root, "S001", false);
    auto path = episodePath(root, "S001");
    assert(path.filename() == "episode1");
    std::ofstream(path / "old.h265") << "old";
    fs::create_directories(root / "S001" / "normal_task");
    fs::create_directories(root / "S002");
    prepareEpisode(root, "S001", true);
    assert(fs::is_empty(path));
    assert(fs::exists(root / "S001" / "normal_task"));
    assert(fs::exists(root / "S002"));
    bool rejected = false;
    try { prepareEpisode(root, "../S002", true); } catch(...) { rejected = true; }
    assert(rejected);
    fs::remove(path);
    fs::create_directory_symlink(root / "S002", path);
    rejected = false;
    try { prepareEpisode(root, "S001", true); } catch(...) { rejected = true; }
    assert(rejected && fs::exists(root / "S002"));
    fs::remove(path);
    CaptureLock first;
    first.acquire(root, "S001");
    rejected = false;
    try { CaptureLock second; second.acquire(root, "S001"); } catch(...) { rejected = true; }
    assert(rejected);
}
''')
            include = Path(__file__).resolve().parents[1] / "src" / "sync"
            subprocess.run([compiler, "-std=c++17", "-I", str(include), str(program), "-o", str(root / "test")], check=True, capture_output=True)
            subprocess.run([str(root / "test"), str(root / "data")], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
