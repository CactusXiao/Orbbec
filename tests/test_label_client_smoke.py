from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from label.backend_client import LabelBackendClient, UriResolver
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
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
