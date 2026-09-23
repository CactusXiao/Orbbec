from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from task_backend.job_service import JobService
from task_backend.publisher_bridge import (
    PublisherBridge, PublisherBridgeConfig, PublisherClient, PublisherCommandError,
    MaterializerError,
)
from task_backend.workflow_store import WorkflowStore


class BatchPublisher:
    def __init__(self):
        self.states = {}
        self.batches = []
        self.published = []
        self.fail = False

    def statuses(self, ids):
        self.batches.append(list(ids))
        if self.fail:
            raise PublisherCommandError("batch endpoint unavailable")
        return {i: {"episode_id": i, **self.states.get(i, {"found": True, "state": "cleaned"})}
                for i in ids}

    def publish(self, episode_id):
        self.published.append(episode_id)
        self.states[episode_id] = {"found": True, "state": "ready"}


class PublisherMonitorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = WorkflowStore(self.root / "workflow.sqlite3")
        self.service = JobService(self.store, nas_mounts={"nas://ego": str(self.root)})
        self.publisher = BatchPublisher()
        self.bridge = PublisherBridge(self.service, PublisherBridgeConfig(
            enabled=True, max_inflight=2, poll_seconds=0.02,
            heartbeat_seconds=0.05, lease_seconds=10,
            mano_python=Path(sys.executable), mano_toolkit_root=self.root, mano_model_dir=self.root,
        ), publisher_client=self.publisher)

    def tearDown(self):
        self.bridge.stop(timeout=3)
        self.tmp.cleanup()

    def add_episode(self, index, *, shape=False):
        episode_id = f"episode{index:03d}"
        task = "task_handshapeCalibration" if shape else "task"
        nas_id = f"subject01/{task}/{episode_id}"
        (self.root / nas_id).mkdir(parents=True)
        self.store.create_or_update_episode(
            episode_id=episode_id, subject_id="subject01", task_name=task,
            status="uploaded", episode_uri="nas://ego/" + nas_id,
            metadata={"shape_calibration": shape},
        )
        self.store.create_job(job_id="job_" + episode_id, job_type="auto_label", episode_id=episode_id,
                              payload={"episode_uri": "nas://ego/" + nas_id, "scope": "episode"})
        return nas_id

    def until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("monitor did not reach expected state")

    def test_100_waiters_use_batch_and_new_episode_publishes_without_old_results(self):
        for i in range(101):
            nas_id = self.add_episode(i)
            if i == 99:
                self.publisher.states[nas_id] = {"found": False}
        self.bridge.start()
        self.until(lambda: len(self.publisher.published) == 1)
        self.assertEqual(self.publisher.published, ["subject01/task/episode099"])
        self.assertEqual(len(self.publisher.batches[0]), 100)
        self.assertEqual(len(self.bridge._active), 100)
        self.assertEqual(self.store.get_job("job_episode100")["status"], "queued")
        self.assertEqual(len(self.bridge.threads), 2)
        self.assertLessEqual(len(self.bridge._publish_pool._threads), 1)
        self.assertEqual(len(self.bridge._result_pool._threads), 0)
        self.until(lambda: all(self.store.get_job(f"job_episode{i:03d}")["status"] == "running"
                               for i in range(100)))
        self.bridge.stop(timeout=3)
        self.assertFalse(self.bridge.threads)
        self.assertTrue(all(self.store.get_job(f"job_episode{i:03d}")["status"] == "queued"
                            for i in range(101)))

    def test_slow_results_do_not_block_polling_publishing_or_renewal(self):
        gate = threading.Event()
        entered = []

        class SlowMaterializer:
            def run(self, **kwargs):
                entered.append(kwargs)
                gate.wait(4)
                raise MaterializerError("waiting for files", retryable=True)

        self.bridge.materializer = SlowMaterializer()
        for i in range(5):
            nas_id = self.add_episode(i)
            self.publisher.states[nas_id] = {"found": True, "state": "labeled"}
        self.bridge.start()
        try:
            self.until(lambda: len(entered) == 2)
            nas_id = self.add_episode(5)
            self.publisher.states[nas_id] = {"found": False}
            self.until(lambda: nas_id in self.publisher.published)
            self.until(lambda: self.store.get_job("job_episode000")["status"] == "running")
            self.assertEqual(len(entered), 2)
            self.assertGreaterEqual(len(self.publisher.batches), 2)
        finally:
            gate.set()

    def test_failed_batch_never_means_missing_and_recovers(self):
        nas_id = self.add_episode(0)
        self.publisher.states[nas_id] = {"found": False}
        self.publisher.fail = True
        self.bridge.start()
        self.until(lambda: len(self.publisher.batches) >= 2)
        self.assertEqual(self.publisher.published, [])
        self.publisher.fail = False
        self.until(lambda: self.publisher.published == [nas_id])

    def test_shape_missing_is_never_republished_and_completion_frees_capacity(self):
        nas_id = self.add_episode(0, shape=True)
        self.publisher.states[nas_id] = {"found": False}
        self.bridge.config.monitor_capacity = 1
        self.add_episode(1)
        self.bridge.start()
        self.until(lambda: len(self.publisher.batches) >= 2)
        self.assertEqual(self.publisher.published, [])
        result = self.root / "subject01/shape_calibration_result"
        result.mkdir()
        for name in ("shape.npy", "scale.npy", "pose_mesh.png", "pose_2d.png"):
            (result / name).write_bytes(b"result")
        self.publisher.states[nas_id] = {"found": True, "state": "shape_calibrated"}
        self.until(lambda: self.store.get_job("job_episode000")["status"] == "succeeded")
        self.until(lambda: self.store.get_job("job_episode001")["status"] in {"leased", "running"})
        self.assertEqual(self.publisher.published, [])
        self.assertFalse(self.store.jobs_for_episode("episode000", "qc"))

    def test_batch_heartbeat_and_shutdown_respect_ownership(self):
        for i in range(3):
            self.add_episode(i)
        self.bridge.start()
        self.until(lambda: len(self.bridge._active) == 3)
        with self.store.connect() as conn:
            conn.execute("UPDATE jobs SET lease_owner = 'another-worker' WHERE job_id = 'job_episode000'")
            conn.execute("UPDATE jobs SET status = 'canceled' WHERE job_id = 'job_episode001'")
        owned = self.store.heartbeat_jobs(job_ids=[f"job_episode{i:03d}" for i in range(3)],
                                          lease_owner=self.bridge._monitor_owner, lease_seconds=10)
        self.assertEqual(owned, ["job_episode002"])
        self.bridge.stop(timeout=3)
        self.assertEqual(self.store.get_job("job_episode000")["lease_owner"], "another-worker")
        self.assertEqual(self.store.get_job("job_episode001")["status"], "canceled")
        self.assertEqual(self.store.get_job("job_episode002")["status"], "queued")


class BatchClientTest(unittest.TestCase):
    def test_100_statuses_use_one_remote_call_and_validate_response(self):
        client = PublisherClient(PublisherBridgeConfig(), threading.Event())
        ids = [f"subject/task/episode{i}" for i in range(100)]
        payload = {"episodes": [{"episode_id": i, "found": False} for i in ids]}
        with patch.object(client, "_run_remote", return_value=json.dumps(payload)) as remote:
            self.assertEqual(set(client.statuses(ids)), set(ids))
            remote.assert_called_once_with(client.config.publisher_status_command, "--batch", *ids)
        for invalid in ({}, {"episodes": []}, {"episodes": [None] * 100},
                        {"episodes": [{"episode_id": ids[0], "found": False}] * 100}):
            with self.subTest(invalid=str(invalid)[:40]):
                with patch.object(client, "_run_remote", return_value=json.dumps(invalid)):
                    with self.assertRaises(PublisherCommandError):
                        client.statuses(ids)


if __name__ == "__main__":
    unittest.main()
