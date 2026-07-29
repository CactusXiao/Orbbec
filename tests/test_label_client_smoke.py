from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from label.backend_client import LabelBackendClient, UriResolver
from label.storage import correction_task_from_backend_payload
from task_backend.job_service import JobService
from task_backend.server import BackendRuntime, RequestHandler, TaskHTTPServer, TaskInstanceRegistry
from task_backend.workflow_store import WorkflowStore


class LabelBackendClientSmokeTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
