from __future__ import annotations

import json
import numpy as np
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from label.backend_client import LabelBackendClient, UriResolver, grouped_label_tasks
from label.storage import correction_task_from_backend_payload
from label.tracking import CoTrackerRuntime
from task_backend.job_service import JobService
from task_backend.server import BackendRuntime, RequestHandler, TaskHTTPServer, TaskInstanceRegistry
from task_backend.workflow_store import WorkflowStore


class LabelBackendClientSmokeTest(unittest.TestCase):
    def test_grouped_label_tasks_by_task(self) -> None:
        groups = grouped_label_tasks(
            [
                {"job_id": "job_1", "task_name": "pick_object", "subject_id": "S001", "frames_count": 2, "created_at": "2026-01-01T00:00:01Z"},
                {"job_id": "job_2", "task_name": "place_object", "subject_id": "S002", "frames_count": 3, "created_at": "2026-01-01T00:00:02Z"},
                {"job_id": "job_3", "task_name": "pick_object", "subject_id": "S003", "frames_count": 4, "created_at": "2026-01-01T00:00:00Z"},
            ]
        )
        by_task = {item["task_name"]: item for item in groups}
        self.assertEqual(by_task["pick_object"]["queued"], 2)
        self.assertEqual(by_task["pick_object"]["frames"], 6)
        self.assertEqual(by_task["pick_object"]["subjects"], ["S001", "S003"])
        self.assertEqual(by_task["pick_object"]["oldest_created_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(by_task["place_object"]["queued"], 1)

    def test_dev_create_lease_resolve_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episode_dir = tmp_path / "S001" / "pick_object" / "episode_000456"
            (episode_dir / "camera_01" / "RGB").mkdir(parents=True)

            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store)
            registry = TaskInstanceRegistry(tmp_path / "backend_state", seed_task_files=[])
            runtime = BackendRuntime(registry, service)
            server = TaskHTTPServer(("127.0.0.1", 0), RequestHandler, runtime)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                base_url = f"http://{host}:{port}"
                req = Request(
                    base_url + "/api/v1/workflow/stages/manual_label/enable",
                    data=json.dumps({"updated_by": "smoke"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=5) as resp:
                    enabled = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(enabled["control"]["lease_enabled"])

                client = LabelBackendClient(f"http://{host}:{port}", timeout_seconds=5)
                created = client.create_dev_label_job(
                    {
                        "local_path": str(episode_dir),
                        "subject_id": "S001",
                        "task_name": "pick_object",
                        "episode_id": "episode_000456",
                        "cameras": ["camera_01"],
                        "frames": [120, 121],
                    }
                )
                self.assertEqual(created["job"]["status"], "queued")

                leased = client.lease_label_job("labeler_01", lease_seconds=60)
                job_id = leased["payload"]["job_id"]
                self.assertEqual(leased["job"]["status"], "leased")
                self.assertEqual(UriResolver().resolve(leased["payload"]["data_uri"]), episode_dir.resolve())

                completed = client.complete_label_job(job_id, result={"ok": True})
                self.assertEqual(completed["job"]["status"], "succeeded")

                with urlopen(base_url + "/api/v1/workflow/stages/manual_label", timeout=5) as resp:
                    stage = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(stage["stats"]["succeeded"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_backend_payload_resolved_path_does_not_need_mount_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            payload = {
                "data_uri": "nas://orbbec-test/S001/pick_object/episode_001",
                "resolved_data_path": str(tmp_path / "virtual_nas" / "S001" / "pick_object" / "episode_001"),
                "subject_id": "S001",
                "task_name": "pick_object",
                "episode_id": "episode_001",
                "cameras": ["camera_01"],
                "frames": [1],
            }
            task = correction_task_from_backend_payload(payload)
            self.assertEqual(task.episode_dir(), Path(payload["resolved_data_path"]).resolve())

    def test_backend_payload_can_discover_missing_cameras_and_frames_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episode_dir = tmp_path / "S001" / "pick_object" / "episode_001"
            for camera in ("00", "01"):
                rgb = episode_dir / camera / "RGB"
                rgb.mkdir(parents=True)
                (rgb / "00001.png").write_bytes(b"rgb")
                (rgb / "00002.png").write_bytes(b"rgb")
            task = correction_task_from_backend_payload(
                {
                    "resolved_data_path": str(episode_dir),
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "episode_id": "episode_001",
                }
            )
            self.assertEqual(task.cameras, ["00", "01"])
            self.assertEqual(task.frames, [1, 2])

    def test_backend_payload_missing_context_uses_backend_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "data_uri": "local://" + str(Path(tmp) / "empty_episode"),
                "resolved_data_path": str(Path(tmp) / "empty_episode"),
            }
            with self.assertRaises(ValueError) as exc:
                correction_task_from_backend_payload(payload)
            self.assertIn("Backend label job payload", str(exc.exception))
            self.assertNotIn("Line 1", str(exc.exception))

    def test_tracking_uses_task_correction_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "S001" / "pick_object" / "episode_001"
            corrected = episode_dir / "human_fixed" / "00"
            corrected.mkdir(parents=True)
            arr = np.zeros((2, 21, 2), dtype=np.float32)
            np.save(corrected / "00001.npy", arr)

            loaded = CoTrackerRuntime()._load_previous_annotation(episode_dir, "00", 1, "human_fixed")
            self.assertEqual(loaded.shape, (2, 21, 2))


if __name__ == "__main__":
    unittest.main()
