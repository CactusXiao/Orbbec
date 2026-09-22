import io
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_backend import task_assets
from task_backend.job_service import JobService
from task_backend.server import (BackendRuntime, RequestHandler, TaskBackend, TaskHTTPServer,
                                 TaskInstanceRegistry, load_task_file)
from task_backend.workflow_store import WorkflowStore


class TaskCreationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nas = self.root / "nas"
        self.nas.mkdir()
        self.service = JobService(WorkflowStore(self.root / "workflow.sqlite3"), nas_mounts={"nas://ego": str(self.nas)})
        self.backend = TaskBackend(self.root / "state", workflow_service=self.service)

    def fields(self, name="整理杯子"):
        document = self.root / (name + ".json")
        document.write_text(json.dumps({"task_name": name, "task_description": {
            "step1": "拿起杯子", "step2": "放回桌面"}}, ensure_ascii=False), encoding="utf-8")
        video = self.root / "sample.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 600)
        return {"task_name": name, "total": "3", "task_json": (document, document.name), "demo_video": (video, video.name)}

    def test_empty_add_live_progress_and_restart(self):
        self.assertEqual(self.backend.get_tasks("alice"), {"tasks": []})
        fields = self.fields()
        self.backend.add_task(fields)
        tasks = self.backend.get_tasks("alice")["tasks"]
        self.assertEqual(tasks[0]["description_cn"], "step1: 拿起杯子\nstep2: 放回桌面")
        self.assertEqual(tasks[0]["total"], 3)
        reserved = self.backend.reserve({"client_id": "test", "subject_id": "alice", "task_name": fields["task_name"]})
        self.assertEqual(reserved["episode_number"], 1)
        self.assertIsNotNone(self.service.store.get_episode(reserved["reservation_id"]))
        before = self.backend.state_file.read_bytes()
        self.backend.add_task(self.fields("第二任务"))
        self.assertEqual(before, self.backend.state_file.read_bytes())
        restarted = TaskBackend(self.root / "state", workflow_service=self.service)
        self.assertEqual(len(restarted.dashboard_model()["tasks"]), 2)
        self.assertEqual((self.nas / "tasks" / fields["task_name"] / "demo.mp4").read_bytes(), fields["demo_video"][0].read_bytes())

    def test_catalog_formats_preserved_and_description_fallback(self):
        for raw in ({"old": {"total": 4, "description_cn": "旧描述", "custom": 7}},
                    [{"task_name": "old", "total": 4}],
                    {"version": 9, "tasks": [{"task_name": "old", "total": 4}]}):
            with self.subTest(raw=raw):
                task_assets.atomic_json(self.backend.task_file, raw)
                name = "new_" + str(len(list((self.nas / "tasks").glob("*")))) if (self.nas / "tasks").exists() else "new_0"
                self.backend.add_task(self.fields(name))
                updated = json.loads(self.backend.task_file.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    self.assertEqual(updated[:-1], raw)
                elif "tasks" in raw:
                    self.assertEqual(updated["version"], 9)
                    self.assertEqual(updated["tasks"][:-1], raw["tasks"])
                else:
                    self.assertEqual(updated["old"], raw["old"])
                self.assertEqual(len(load_task_file(self.backend.task_file)), 2)
        task_assets.atomic_json(self.backend.task_file, {"old": {"description_cn": "回退"}})
        directory = self.nas / "tasks" / "old"
        directory.mkdir()
        task_assets.atomic_json(directory / "old.json", {"task_name": "old", "steps": [{"step": 1, "description": "优先描述"}]})
        self.assertIn("优先描述", self.backend.tasks[0]["description_cn"])
        (directory / "old.json").write_text("broken", encoding="utf-8")
        self.assertEqual(self.backend.tasks[0]["description_cn"], "回退")

    def test_failures_leave_catalog_and_nas_unchanged(self):
        fields = self.fields()
        before = self.backend.task_file.read_bytes()
        with patch.object(task_assets, "atomic_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.backend.add_task(fields)
        self.assertEqual(before, self.backend.task_file.read_bytes())
        self.assertEqual(list((self.nas / "tasks").iterdir()), [])
        for name in ("../escape", "CON", "different-name"):
            with self.assertRaises(Exception):
                self.backend.add_task({**fields, "task_name": name})
            self.assertEqual(before, self.backend.task_file.read_bytes())
        self.backend.add_task(fields)
        with self.assertRaises(Exception) as duplicate:
            self.backend.add_task(fields)
        self.assertEqual(duplicate.exception.status, 409)

    def test_concurrent_additions_are_not_lost(self):
        first, second = self.fields("first"), self.fields("second")
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(self.backend.add_task, [first, second]))
        self.assertEqual({task["task_name"] for task in self.backend.tasks}, {"first", "second"})

    def test_empty_catalogs_and_invalid_uploads(self):
        for catalog in ({}, [], {"tasks": []}, {"version": 1, "tasks": []}):
            task_assets.atomic_json(self.backend.task_file, catalog)
            self.assertEqual(load_task_file(self.backend.task_file), [])
        fields = self.fields()
        for changes in ({"total": "0"}, {"total": "1.5"},
                        {"demo_video": (fields["demo_video"][0], "demo.exe")}):
            with self.assertRaises(Exception) as invalid:
                self.backend.add_task({**fields, **changes})
            self.assertEqual(invalid.exception.status, 400)
        task_assets.atomic_json(fields["task_json"][0], {"task_name": fields["task_name"]})
        with self.assertRaises(Exception) as missing_description:
            self.backend.add_task(fields)
        self.assertEqual(missing_description.exception.status, 400)
        fields = self.fields()
        self.backend.nas_root = self.root / "missing-mount"
        with self.assertRaises(Exception) as missing_nas:
            self.backend.add_task(fields)
        self.assertEqual(missing_nas.exception.status, 400)
        self.assertFalse(self.backend.nas_root.exists())
        self.assertEqual(self.backend.tasks, [])

    def multipart(self, fields):
        chunks = []
        for key, value in fields.items():
            header = f'--test-boundary\r\nContent-Disposition: form-data; name="{key}"'
            if isinstance(value, tuple):
                path, filename = value
                header += f'; filename="{filename}"\r\nContent-Type: application/octet-stream'
                value = path.read_bytes()
            else:
                value = value.encode("utf-8")
            chunks.append(header.encode("utf-8") + b"\r\n\r\n" + value + b"\r\n")
        return b"".join(chunks) + b"--test-boundary--"

    def test_http_empty_setup_upload_and_detail(self):
        registry = TaskInstanceRegistry(self.root / "registry")
        self.assertEqual(len(registry.snapshot()["task_files"]), 1)
        runtime = BackendRuntime(registry, self.service)
        server = TaskHTTPServer(("127.0.0.1", 0), RequestHandler, runtime)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(Request(base + "/setup/empty", data=b"instance_label=empty")) as response:
            self.assertIn("增加任务", response.read().decode())
        with urlopen(base + "/manage/tasks/new") as response:
            self.assertIn('name="task_json"', response.read().decode())
        fields = self.fields()
        body = self.multipart(fields)
        request = Request(base + "/api/v1/tasks", data=body, headers={"Content-Type": "multipart/form-data; boundary=test-boundary"})
        with urlopen(request) as response:
            self.assertEqual(response.status, 201)
            location = json.load(response)["url"]
        with urlopen(base + location) as response:
            detail = response.read().decode()
            self.assertIn("拿起杯子", detail)
            self.assertIn("<video", detail)
        with urlopen(base + location.replace("/tasks/", "/task-assets/") + "/demo.mp4") as response:
            self.assertEqual(response.read(), fields["demo_video"][0].read_bytes())
        with self.assertRaises(HTTPError) as duplicate:
            urlopen(request)
        self.assertEqual(duplicate.exception.code, 409)

    def test_multipart_truncation_and_chunk_boundaries(self):
        body = self.multipart(self.fields())
        class ShortReads(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 7))
        directory = self.root / "upload"
        directory.mkdir()
        result = task_assets.read_multipart(ShortReads(body), "multipart/form-data; boundary=test-boundary", len(body), directory)
        self.assertEqual(result["demo_video"][0].read_bytes(), (self.root / "sample.mp4").read_bytes())
        with self.assertRaises(ValueError):
            task_assets.read_multipart(io.BytesIO(body[:-10]), "multipart/form-data; boundary=test-boundary", len(body), directory)


if __name__ == "__main__":
    unittest.main()
