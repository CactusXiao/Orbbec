from __future__ import annotations

import json
import os
import numpy as np
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from label.app import SOURCE_LABELS, SOURCE_ORDER, LabelPage
from label.backend_client import LabelBackendClient, NasEpisodeResolver, grouped_label_tasks
from label.env_config import load_label_config
from label.mano_view import ManoViewRuntime, describe_mano_projection_issue
from label.storage import (
    PredictionBundle,
    apply_view_state_to_corrected,
    correction_task_from_backend_payload,
    find_frame_path,
    load_joint_visibility,
    save_corrected_array,
    view_state_from_bundle,
)
from label.tracking import CoTrackerRuntime
from label.video_frames import ensure_decoded_rgb_frames
from task_backend.job_service import JobService
from task_backend.server import BackendRuntime, RequestHandler, TaskHTTPServer, TaskInstanceRegistry
from task_backend.workflow_store import WorkflowStore


class LabelBackendClientSmokeTest(unittest.TestCase):
    def test_only_original_and_modified_views_are_exposed(self) -> None:
        self.assertEqual(SOURCE_ORDER, ("mano", "correct"))
        self.assertEqual(SOURCE_LABELS, {"mano": "原始视角", "correct": "修改后视角"})

    def test_modified_view_starts_from_original_when_no_saved_correction_exists(self) -> None:
        original = (
            [[(10.0, 20.0) for _ in range(21)] for _ in range(2)],
            [[True for _ in range(21)] for _ in range(2)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = PredictionBundle(
                mode="correct",
                episode_dir=root,
                prediction_dir=root / "pred_2d",
                pred_dir=root / "manual_2d",
                corrected_dir=root / "manual_2d",
                samples={},
            )

            class PageStub:
                @staticmethod
                def _ensure_bundle(_source):
                    return bundle

                @staticmethod
                def _build_visible_mano_view_state(_frame_idx, _cam_id):
                    return original

                @staticmethod
                def _copy_view_state(state):
                    points, visible = state
                    return (
                        [[(float(x), float(y)) for x, y in hand] for hand in points],
                        [[bool(value) for value in hand] for hand in visible],
                    )

                @staticmethod
                def _hidden_points():
                    return [[(-1.0, -1.0) for _ in range(21)] for _ in range(2)]

                @staticmethod
                def _none_visible():
                    return [[False for _ in range(21)] for _ in range(2)]

            initialized = LabelPage._build_modified_view_state(PageStub(), 5, "00")

        self.assertEqual(initialized, original)
        self.assertIsNot(initialized, original)
        self.assertIsNot(initialized[0], original[0])
        self.assertIsNot(initialized[1], original[1])

    def test_modified_view_reloads_saved_correction_when_returning_to_confirmed_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corrected_dir = root / "manual_2d"
            (corrected_dir / "00").mkdir(parents=True)
            corrected = np.full((2, 21, 2), 25.0, dtype=np.float32)
            corrected[1, 4] = [-1.0, -1.0]
            np.save(corrected_dir / "00" / "00005.npy", corrected)
            bundle = PredictionBundle(
                mode="correct",
                episode_dir=root,
                prediction_dir=root / "pred_2d",
                pred_dir=corrected_dir,
                corrected_dir=corrected_dir,
                samples={},
            )

            class PageStub:
                @staticmethod
                def _ensure_bundle(_source):
                    return bundle

                @staticmethod
                def _build_visible_mano_view_state(_frame_idx, _cam_id):
                    return (
                        [[(100.0 + joint, 200.0 + joint) for joint in range(21)] for _ in range(2)],
                        [[True for _ in range(21)] for _ in range(2)],
                    )

            points, visible = LabelPage._build_modified_view_state(PageStub(), 5, "00")
            self.assertEqual(points[0][0], (25.0, 25.0))
            self.assertTrue(visible[0][0])
            self.assertEqual(points[1][4], (104.0, 204.0))
            self.assertFalse(visible[1][4])

            visible[1][4] = True
            apply_view_state_to_corrected(bundle, 5, "00", points, visible)
            save_corrected_array(bundle)
            reloaded_points, reloaded_visible = view_state_from_bundle(bundle, 5, "00")

        self.assertEqual(reloaded_points[1][4], (104.0, 204.0))
        self.assertTrue(reloaded_visible[1][4])

    def test_modified_initial_view_uses_joints_vis_in_canvas_joint_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp)
            visibility_dir = episode_dir / "joints_vis" / "00"
            visibility_dir.mkdir(parents=True)
            original_visibility = np.zeros((2, 21), dtype=np.float32)
            original_visibility[0, 13] = 1.0  # canonical MANO thumb MCP
            original_visibility[1, 1] = 1.0   # canonical MANO index MCP
            np.save(visibility_dir / "00005.npy", original_visibility)

            projected = (
                [[(10.0 + joint, 20.0 + joint) for joint in range(21)] for _ in range(2)],
                [[True for _ in range(21)] for _ in range(2)],
            )

            class TaskStub:
                key = "task"
                mano_episode_dir = "mano/episode"

                @staticmethod
                def episode_dir():
                    return episode_dir

            class RuntimeStub:
                @staticmethod
                def project_mano_frame(**_kwargs):
                    return projected

            class PageStub:
                _active_task = TaskStub()
                _mano_projection_errors = {}

                @staticmethod
                def _mano_runtime_instance():
                    return RuntimeStub()

            PageStub._build_mano_3d_view_state = LabelPage._build_mano_3d_view_state
            points, visible = LabelPage._build_visible_mano_view_state(PageStub(), 5, "00")

        self.assertEqual(points, projected[0])
        self.assertTrue(visible[0][13])  # canonical MANO thumb MCP
        self.assertTrue(visible[1][1])   # canonical MANO index MCP
        self.assertFalse(visible[0][1])

    def test_joint_visibility_loader_accepts_bool_or_numeric_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            (base_dir / "00").mkdir()
            mask = np.zeros((2, 21, 1), dtype=np.int8)
            mask[0, 3, 0] = 1
            mask[1, 7, 0] = -1
            np.save(base_dir / "00" / "5.npy", mask)

            visible = load_joint_visibility(base_dir, "00", 5)

        self.assertTrue(visible[0][3])
        self.assertFalse(visible[1][7])

    def test_modified_source_caches_canvas_edits_before_switching_views(self) -> None:
        original = (
            [[(10.0, 20.0) for _ in range(21)] for _ in range(2)],
            [[True for _ in range(21)] for _ in range(2)],
        )
        edited = (
            [[(30.0, 40.0) for _ in range(21)] for _ in range(2)],
            [[False for _ in range(21)] for _ in range(2)],
        )

        class CanvasStub:
            def get_hand_state(self):
                return edited

        class PageStub:
            _mode = "correct"
            _view_states = {"00": original}
            _source_state_cache = {}
            _canvas = CanvasStub()

            @staticmethod
            def _active_cam_id():
                return "00"

            def _source_cache_key(self, cam_id, source=None):
                return ("task", 0, cam_id, source or self._mode)

            @staticmethod
            def _copy_view_state(state):
                points, visible = state
                return (
                    [[(float(x), float(y)) for x, y in hand] for hand in points],
                    [[bool(value) for value in hand] for hand in visible],
                )

        page = PageStub()
        LabelPage._cache_current_source_state(page)

        self.assertEqual(page._view_states["00"], edited)
        self.assertEqual(page._source_state_cache[("task", 0, "00", "correct")], edited)

    def test_original_view_is_read_only_and_modified_view_is_editable(self) -> None:
        class CanvasStub:
            def __init__(self):
                self.read_only = None
                self.annotation_visible = None

            def set_read_only(self, value):
                self.read_only = value

            def set_annotation_visible(self, value):
                self.annotation_visible = value

        class PageStub:
            _mode = "mano"
            _canvas = CanvasStub()

            @staticmethod
            def _visualization_active():
                return False

        page = PageStub()
        LabelPage._sync_visualization_canvas_state(page)
        self.assertTrue(page._canvas.read_only)
        self.assertTrue(page._canvas.annotation_visible)

        page._mode = "correct"
        LabelPage._sync_visualization_canvas_state(page)
        self.assertFalse(page._canvas.read_only)
        self.assertTrue(page._canvas.annotation_visible)

    def test_label_app_loads_mount_from_launch_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            config_path = tmp_path / "label_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "backend_url": "http://127.0.0.1:9999",
                        "operator_id": "labeler_a",
                        "frame_cache_dir": "label_cache",
                        "ffmpeg_executable": "/usr/local/bin/ffmpeg",
                        "lease_seconds": 300,
                        "request_timeout_seconds": 3.5,
                        "nas_mounts": {"nas://ego": str(nas_root)},
                    }
                ),
                encoding="utf-8",
            )

            config = load_label_config(config_path)

            self.assertEqual(config.backend_url, "http://127.0.0.1:9999")
            self.assertEqual(config.operator_id, "labeler_a")
            self.assertEqual(config.frame_cache_dir, (tmp_path / "label_cache").resolve())
            self.assertEqual(config.ffmpeg_executable, "/usr/local/bin/ffmpeg")
            self.assertEqual(config.lease_seconds, 300)
            self.assertEqual(config.request_timeout_seconds, 3.5)
            self.assertEqual(config.nas_mounts, {"nas://ego": str(nas_root)})

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

    def test_manual_2d_is_saved_in_canonical_smplx_mano_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = PredictionBundle(
                mode="correct",
                episode_dir=root,
                prediction_dir=root / "pred_2d",
                pred_dir=root / "manual_2d",
                corrected_dir=root / "manual_2d",
                samples={},
            )
            points = [
                [(float(hand * 100 + joint), float(hand * 100 + joint) + 0.5) for joint in range(21)]
                for hand in range(2)
            ]
            visible = [[True] * 21 for _ in range(2)]

            apply_view_state_to_corrected(bundle, 5, "00", points, visible)
            save_corrected_array(bundle)
            stored = np.load(root / "manual_2d" / "00" / "00005.npy")

        canonical = np.asarray(points, dtype=np.float32)
        np.testing.assert_array_equal(stored, canonical)
        self.assertEqual(stored[0, 13, 0], 13.0)  # canonical thumb MCP
        self.assertEqual(stored[0, 1, 0], 1.0)    # canonical index MCP

    def test_mano_source_reloads_when_npy_changes(self) -> None:
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
            joints_path = mano_dir / "joints_3d.npy"
            joints = np.zeros((1, 2, 21, 3), dtype=np.float32)
            joints[:, :, :, 2] = 1.0
            np.save(joints_path, joints)
            (mano_dir / "mano_episode.json").write_text(
                json.dumps({"schema_version": 1, "kind": "orbbec_mano_3d_episode", "frames": [5], "joints_3d_file": "joints_3d.npy"}),
                encoding="utf-8",
            )
            runtime = ManoViewRuntime()

            first = runtime.project_mano_frame(episode_dir=episode_dir, mano_dir=mano_dir, cam_id="00", frame_idx=5)
            self.assertIsNotNone(first)
            points, _visible = first  # type: ignore[misc]
            self.assertAlmostEqual(points[0][0][0], 320.0)

            joints[0, 0, 0] = [0.2, 0.0, 1.0]
            np.save(joints_path, joints)
            stat = joints_path.stat()
            os.utime(joints_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            second = runtime.project_mano_frame(episode_dir=episode_dir, mano_dir=mano_dir, cam_id="00", frame_idx=5)

            self.assertIsNotNone(second)
            points, _visible = second  # type: ignore[misc]
            self.assertAlmostEqual(points[0][0][0], 340.0)

    def test_mano_projection_issue_reports_missing_calibration_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "S001" / "pick_object" / "episode_001"
            mano_dir = episode_dir / "mano" / "episode"
            mano_dir.mkdir(parents=True)
            joints = np.zeros((1, 2, 21, 3), dtype=np.float32)
            joints[:, :, :, 2] = 1.0
            np.save(mano_dir / "joints_3d.npy", joints)
            (mano_dir / "mano_episode.json").write_text(
                json.dumps({"schema_version": 1, "kind": "orbbec_mano_3d_episode", "frames": [5], "joints_3d_file": "joints_3d.npy"}),
                encoding="utf-8",
            )

            message = describe_mano_projection_issue(episode_dir, mano_dir, "00", 5)

            self.assertIn("camera_params.json", message)
            self.assertIn("extrinsics.json", message)
            self.assertIn(str(episode_dir), message)

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
            service = JobService(store, nas_mounts={"nas://orbbec-test": str(tmp_path / "nas")})
            registry = TaskInstanceRegistry(tmp_path / "backend_state", seed_task_files=[])
            runtime = BackendRuntime(registry, service)
            server = TaskHTTPServer(("127.0.0.1", 0), RequestHandler, runtime)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                base_url = f"http://{host}:{port}"
                nas_episode_dir = tmp_path / "nas" / "S001" / "pick_object" / "episode_000456"
                (nas_episode_dir / "camera_01" / "RGB").mkdir(parents=True)
                service.store.create_or_update_episode(
                    episode_id="episode_000456",
                    subject_id="S001",
                    task_name="pick_object",
                    episode_index=1,
                    status="manual_correction_pending",
                    episode_uri="nas://orbbec-test/S001/pick_object/episode_000456",
                    cameras=["camera_01"],
                    frame_count=2,
                )
                service.store.create_segment(
                    segment_id="segment_000456_120_121",
                    episode_id="episode_000456",
                    start_frame=120,
                    end_frame=121,
                )
                service._create_manual_label_episode_job("episode_000456", reason="smoke")
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
                tasks = client.queued_label_tasks()
                self.assertEqual(tasks[0]["task_name"], "pick_object")
                episodes = client.label_task_episodes("pick_object")
                self.assertEqual(episodes[0]["episode_id"], "episode_000456")

                leased = client.lease_label_episode("labeler_01", lease_seconds=60, task_name="pick_object", episode_id="episode_000456")
                job_id = leased["payload"]["episode_id"]
                self.assertEqual(leased["job"]["status"], "leased")
                self.assertEqual(len(leased["segments"]), 1)
                self.assertEqual(NasEpisodeResolver({"nas://orbbec-test": str(tmp_path / "nas")}).resolve(leased["payload"]["episode_uri"]), nas_episode_dir.resolve())
                self.assertEqual(leased["payload"]["frames"], [120, 121])
                self.assertNotIn("episode_media", leased["payload"])
                self.assertNotIn("rgb_path_template", leased["payload"])

                completed = client.complete_label_job(
                    job_id,
                    result={"ok": True, "operator_id": "labeler_01"},
                    artifacts=[{"kind": "manual_2d", "metadata": {"scope": "episode"}}],
                )
                self.assertEqual(completed["job"]["status"], "succeeded")
                self.assertEqual(len(service.store.jobs_for_episode("episode_000456", "manual_3d")), 1)

                with urlopen(base_url + "/api/v1/workflow/stages/manual_label", timeout=5) as resp:
                    stage = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(stage["stats"]["succeeded"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_backend_payload_resolves_nas_episode_uri_from_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            episode_dir = nas_root / "S001" / "pick_object" / "episode_001"
            (episode_dir / "camera_01" / "RGB").mkdir(parents=True)
            (episode_dir / "camera_01" / "RGB" / "00001.png").write_bytes(b"rgb")
            payload = {
                "episode_uri": "nas://ego/S001/pick_object/episode_001",
                "subject_id": "S001",
                "task_name": "pick_object",
                "episode_id": "episode_001",
            }
            task = correction_task_from_backend_payload(payload, mounts={"nas://ego": str(nas_root)})
            self.assertEqual(task.episode_dir(), episode_dir.resolve())
            self.assertEqual(task.cameras, ["camera_01"])
            self.assertEqual(task.frames, [1])

    def test_backend_payload_nas_episode_uri_requires_mount(self) -> None:
        payload = {
            "episode_uri": "nas://ego/S001/pick_object/episode_001",
            "subject_id": "S001",
            "task_name": "pick_object",
            "episode_id": "episode_001",
            "cameras": ["00"],
            "frames": [1],
        }

        with self.assertRaisesRegex(ValueError, "ORBBEC_NAS_URI_PREFIX"):
            correction_task_from_backend_payload(payload)

    def test_backend_payload_ignores_non_contract_path_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            episode_dir = nas_root / "S001" / "pick_object" / "episode_001"
            (episode_dir / "00" / "RGB").mkdir(parents=True)
            (episode_dir / "00" / "RGB" / "00001.png").write_bytes(b"rgb")
            payload = {
                "episode_uri": "nas://ego/S001/pick_object/episode_001",
                "subject_id": "S001",
                "task_name": "pick_object",
                "episode_id": "episode_001",
                "cameras": ["00"],
                "frames": [1],
            }

            task = correction_task_from_backend_payload(payload, mounts={"nas://ego": str(nas_root)})

            self.assertEqual(task.episode_dir(), episode_dir.resolve())

    def test_backend_payload_can_discover_missing_cameras_and_frames_from_nas_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            episode_dir = nas_root / "S001" / "pick_object" / "episode_001"
            for camera in ("00", "01", "ego", "pico_ego"):
                rgb = episode_dir / camera / "RGB"
                rgb.mkdir(parents=True)
                (rgb / "00001.png").write_bytes(b"rgb")
                (rgb / "00002.png").write_bytes(b"rgb")
            task = correction_task_from_backend_payload(
                {
                    "episode_uri": "nas://ego/S001/pick_object/episode_001",
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "episode_id": "episode_001",
                },
                mounts={"nas://ego": str(nas_root)},
            )
            self.assertEqual(task.cameras, ["00", "01"])
            self.assertEqual(task.frames, [1, 2])

    def test_backend_payload_filters_pico_views_from_explicit_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "S001" / "pick_object" / "episode_001"
            payload = {
                "episode_uri": "nas://ego/S001/pick_object/episode_001",
                "subject_id": "S001",
                "task_name": "pick_object",
                "episode_id": "episode_001",
                "cameras": ["00", "ego", "pico_ego", "fisheye_0", "camera_01"],
                "frames": [1],
            }

            task = correction_task_from_backend_payload(payload, mounts={"nas://ego": str(Path(tmp))})

            self.assertEqual(task.cameras, ["00", "camera_01"])

    def test_backend_payload_h265_decodes_to_label_cache(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            episode_dir = nas_root / "S001" / "pick_object" / "episode_001"
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
            (episode_dir / "camera_params.json").write_text(
                json.dumps(
                    {
                        "camera_01": {
                            "RGB": {
                                "storageEncoding": "h265",
                                "storageFile": "rgb.h265",
                                "timestampFile": "rgb.h265.timestamps.csv",
                                "intrinsic": {"fx": 100.0, "fy": 100.0, "cx": 16.0, "cy": 12.0},
                                "distortion": {},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "episode_uri": "nas://ego/S001/pick_object/episode_001",
                "subject_id": "S001",
                "task_name": "pick_object",
                "episode_id": "episode_001",
                "segment_id": "segment_001",
                "cameras": ["camera_01"],
                "frames": [120, 122],
            }
            task = correction_task_from_backend_payload(payload, mounts={"nas://ego": str(nas_root)})
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
                "episode_uri": "nas://ego/empty_episode",
            }
            with self.assertRaises(ValueError) as exc:
                correction_task_from_backend_payload(payload, mounts={"nas://ego": str(Path(tmp))})
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
