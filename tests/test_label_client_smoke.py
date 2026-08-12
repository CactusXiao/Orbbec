from __future__ import annotations

import json
import numpy as np
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from label.backend_client import LabelBackendClient, UriResolver, grouped_label_tasks
from label.mano_view import ManoViewRuntime
from label.storage import correction_task_from_backend_payload, find_frame_path
from label.tracking import CoTrackerRuntime
from label.video_frames import ensure_decoded_rgb_frames
from task_backend.job_service import JobService
from task_backend.server import BackendRuntime, RequestHandler, TaskHTTPServer, TaskInstanceRegistry
from task_backend.workflow_store import WorkflowStore


class LabelBackendClientSmokeTest(unittest.TestCase):
    def test_mano_source_projects_episode_3d_result(self) -> None:
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("cv2 is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "S001" / "pick_object" / "episode_001"
            mano_dir = episode_dir / "mano" / "episode"
            mano_dir.mkdir(parents=True)
            (episode_dir / "camera_params.json").write_text(
                json.dumps(
                    {
                        "00": {
                            "RGB": {
                                "intrinsic": {"fx": 100.0, "fy": 100.0, "cx": 320.0, "cy": 200.0},
                                "distortion": {},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (episode_dir / "extrinsics.json").write_text(
                json.dumps({"00": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [0, 0, 0]}}),
                encoding="utf-8",
            )
            joints = np.zeros((1, 2, 21, 3), dtype=np.float32)
            joints[:, :, :, 2] = 1.0
            joints[0, 1, 0] = [0.1, 0.2, 1.0]
            np.save(mano_dir / "joints_3d.npy", joints)
            (mano_dir / "mano_episode.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "orbbec_mano_3d_episode",
                        "frames": [5],
                        "joints_3d_file": "joints_3d.npy",
                    }
                ),
                encoding="utf-8",
            )

            projected = ManoViewRuntime().project_mano_frame(
                episode_dir=episode_dir,
                mano_dir=mano_dir,
                cam_id="00",
                frame_idx=5,
            )

            self.assertIsNotNone(projected)
            points, visible = projected  # type: ignore[misc]
            self.assertAlmostEqual(points[0][0][0], 320.0)
            self.assertAlmostEqual(points[0][0][1], 200.0)
            self.assertTrue(visible[0][0])
            self.assertAlmostEqual(points[1][0][0], 330.0)
            self.assertAlmostEqual(points[1][0][1], 220.0)
            self.assertTrue(visible[1][0])

    def test_mano_source_hides_nan_3d_joints(self) -> None:
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("cv2 is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "S001" / "pick_object" / "episode_001"
            mano_dir = episode_dir / "mano" / "episode"
            mano_dir.mkdir(parents=True)
            (episode_dir / "camera_params.json").write_text(
                json.dumps(
                    {
                        "00": {
                            "RGB": {
                                "intrinsic": {"fx": 100.0, "fy": 100.0, "cx": 320.0, "cy": 200.0},
                                "distortion": {},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (episode_dir / "extrinsics.json").write_text(
                json.dumps({"00": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [0, 0, 0]}}),
                encoding="utf-8",
            )
            joints = np.zeros((1, 2, 21, 3), dtype=np.float32)
            joints[:, :, :, 2] = 1.0
            joints[0, 1, 0] = [np.nan, np.nan, np.nan]
            np.save(mano_dir / "joints_3d.npy", joints)
            (mano_dir / "mano_episode.json").write_text(
                json.dumps({"schema_version": 1, "kind": "orbbec_mano_3d_episode", "frames": [5], "joints_3d_file": "joints_3d.npy"}),
                encoding="utf-8",
            )

            projected = ManoViewRuntime().project_mano_frame(
                episode_dir=episode_dir,
                mano_dir=mano_dir,
                cam_id="00",
                frame_idx=5,
            )

            self.assertIsNotNone(projected)
            points, visible = projected  # type: ignore[misc]
            self.assertFalse(visible[1][0])
            self.assertTrue(np.isnan(points[1][0][0]))
            self.assertTrue(np.isnan(points[1][0][1]))

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
                service.store.create_or_update_episode(
                    episode_id="episode_000456",
                    subject_id="S001",
                    task_name="pick_object",
                    episode_index=1,
                    status="manual_correction_pending",
                    data_uri="local://" + str(episode_dir),
                    cameras=["camera_01"],
                    frame_count=2,
                )
                service.store.create_segment(
                    segment_id="segment_000456_120_121",
                    episode_id="episode_000456",
                    start_frame=120,
                    end_frame=121,
                )
                req = Request(
                    base_url + "/api/v1/workflow/stages/manual_segment/enable",
                    data=json.dumps({"updated_by": "smoke"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=5) as resp:
                    enabled = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(enabled["control"]["lease_enabled"])

                client = LabelBackendClient(f"http://{host}:{port}", timeout_seconds=5)
                tasks = client.queued_label_tasks()
                self.assertEqual(tasks[0]["task_name"], "pick_object")
                episodes = client.label_task_episodes("pick_object")
                self.assertEqual(episodes[0]["episode_id"], "episode_000456")

                leased = client.lease_label_segment("labeler_01", lease_seconds=60, task_name="pick_object", episode_id="episode_000456")
                job_id = leased["payload"]["segment_id"]
                self.assertEqual(leased["segment"]["status"], "manual_labeling")
                self.assertEqual(UriResolver().resolve(leased["payload"]["data_uri"]), episode_dir.resolve())
                self.assertEqual(leased["payload"]["frames"], [120, 121])
                self.assertTrue(leased["payload"]["episode_media"]["requires_rgb_video_decode"])
                self.assertIn("camera_01", leased["payload"]["episode_media"]["cameras"])

                completed = client.complete_label_job(
                    job_id,
                    result={"ok": True, "operator_id": "labeler_01"},
                    artifacts=[{"kind": "manual_2d", "uri": "local://" + str(episode_dir / "manual_2d" / "segments" / job_id)}],
                )
                self.assertEqual(completed["segment"]["status"], "mano_queued")

                with urlopen(base_url + "/api/v1/workflow/stages/manual_segment", timeout=5) as resp:
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
                "resolved_data_path": str(tmp_path / "nas" / "S001" / "pick_object" / "episode_001"),
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
            for camera in ("00", "01", "ego", "pico_ego"):
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

    def test_backend_payload_filters_pico_views_from_explicit_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "S001" / "pick_object" / "episode_001"
            payload = {
                "resolved_data_path": str(episode_dir),
                "subject_id": "S001",
                "task_name": "pick_object",
                "episode_id": "episode_001",
                "cameras": ["00", "ego", "pico_ego", "fisheye_0", "camera_01"],
                "frames": [1],
            }

            task = correction_task_from_backend_payload(payload)

            self.assertEqual(task.cameras, ["00", "camera_01"])

    def test_backend_payload_h265_decodes_to_label_cache(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episode_dir = tmp_path / "S001" / "pick_object" / "episode_001"
            rgb_dir = episode_dir / "camera_01" / "RGB"
            rgb_dir.mkdir(parents=True)
            video_path = rgb_dir / "rgb.h265"
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=32x24:rate=1:duration=3",
                    "-frames:v",
                    "3",
                    "-c:v",
                    "libx265",
                    "-x265-params",
                    "log-level=error",
                    str(video_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                self.skipTest(f"ffmpeg cannot create h265 fixture: {result.stderr.strip()}")
            (rgb_dir / "rgb.h265.timestamps.csv").write_text(
                "video_frame_index,frame_index,rgb_timestamp_us\n"
                "0,120,1000\n"
                "1,121,2000\n"
                "2,122,3000\n",
                encoding="utf-8",
            )
            payload = {
                "resolved_data_path": str(episode_dir),
                "data_uri": "local://" + str(episode_dir),
                "subject_id": "S001",
                "task_name": "pick_object",
                "episode_id": "episode_001",
                "segment_id": "segment_001",
                "cameras": ["camera_01"],
                "frames": [120, 122],
            }
            task = correction_task_from_backend_payload(payload)
            decoded = ensure_decoded_rgb_frames(task, payload, cache_root=tmp_path / "cache")
            first = find_frame_path(decoded.episode_dir(), "camera_01", 120, decoded.rgb_path_template)
            last = find_frame_path(decoded.episode_dir(), "camera_01", 122, decoded.rgb_path_template)
            self.assertIsNotNone(first)
            self.assertIsNotNone(last)
            self.assertTrue(first.exists())  # type: ignore[union-attr]
            self.assertTrue(last.exists())  # type: ignore[union-attr]

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
