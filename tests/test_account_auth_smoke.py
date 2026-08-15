from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_backend.job_service import JobService
from task_backend.server import BackendRuntime, RequestHandler, TaskHTTPServer, TaskInstanceRegistry
from task_backend.workflow_store import WorkflowStore


class AccountAuthSmokeTest(unittest.TestCase):
    def test_register_and_login_before_task_instance_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry = TaskInstanceRegistry(tmp_path / "backend_state", seed_task_files=[])
            service = JobService(WorkflowStore(tmp_path / "workflow.sqlite3"))
            runtime = BackendRuntime(registry, service)
            server = TaskHTTPServer(("127.0.0.1", 0), RequestHandler, runtime)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                base_url = f"http://{host}:{port}"

                def post(path: str, payload: dict[str, str]) -> dict[str, object]:
                    req = Request(
                        base_url + path,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(req, timeout=5) as resp:
                        return json.loads(resp.read().decode("utf-8"))

                registered = post(
                    "/api/v1/auth/register",
                    {"username": "alice", "password": "pw", "password_repeat": "pw"},
                )
                self.assertEqual(registered["username"], "alice")

                logged_in = post("/api/v1/auth/login", {"username": "alice", "password": "pw"})
                self.assertEqual(logged_in["username"], "alice")
                self.assertIn("last_login_at", logged_in)

                with self.assertRaises(HTTPError) as duplicate:
                    post("/api/v1/auth/register", {"username": "alice", "password": "pw", "password_repeat": "pw"})
                self.assertEqual(duplicate.exception.code, 409)

                with self.assertRaises(HTTPError) as bad_login:
                    post("/api/v1/auth/login", {"username": "alice", "password": "wrong"})
                self.assertEqual(bad_login.exception.code, 401)

                accounts_text = (tmp_path / "backend_state" / "accounts.json").read_text(encoding="utf-8")
                self.assertIn("password_hash", accounts_text)
                self.assertNotIn('"pw"', accounts_text)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
