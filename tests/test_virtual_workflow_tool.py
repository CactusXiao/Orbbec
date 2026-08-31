from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from label.backend_client import LabelBackendClient
from label.storage import (
    apply_view_state_to_corrected,
    correction_task_from_backend_payload,
    load_frame_visibility,
    load_prediction_bundle,
    save_corrected_array,
    view_state_from_bundle,
)
from task_backend.job_service import FINAL_3D_SOURCES_REL_PATH, JobService
from task_backend.server import BackendRuntime, RequestHandler, TaskHTTPServer, TaskInstanceRegistry
from task_backend.workflow_store import WorkflowStore
from tools.virtual_workflow.orbbec_virtual_workflow import (
    BackendError,
    BackendClient,
    LabelTask,
    NasSimulator,
    annotation_hand_values_to_mano_order,
    handle_auto_label_once,
    handle_qc_once,
    handle_upload_once,
    build_parser,
    find_rgb_frame_path,
    load_env_defaults,
    virtual_hand_values,
    virtual_failed_segments,
    write_float32_npy,
    write_mano_artifact_for_payload,
    write_placeholder_rgb_image,
)


class FakeHandGtDetector:
    available = True

    def detect_values(self, episode_dir: Path, cam: str, frame: int, *, rgb_path_template: str = "") -> list[float]:
        width, height = VirtualWorkflowToolSmokeTest._frame_size(episode_dir, cam, frame, rgb_path_template)
        target = VirtualWorkflowToolSmokeTest._synthetic_3d_hands(frame)
        values: list[float] = []
        for hand in range(2):
            for joint in range(21):
                values.extend(VirtualWorkflowToolSmokeTest._project_test_point(target[hand, joint], cam, width=width, height=height))
        return values

    def close(self) -> None:
        pass


class StrictImageHandGtDetector(FakeHandGtDetector):
    def detect_values(self, episode_dir: Path, cam: str, frame: int, *, rgb_path_template: str = "") -> list[float]:
        path = find_rgb_frame_path(episode_dir, cam, frame, rgb_path_template)
        if path is None:
            raise BackendError(f"strict detector missing decoded RGB frame: camera={cam} frame={frame}")
        return super().detect_values(episode_dir, cam, frame, rgb_path_template=rgb_path_template)


class NoHandDetector(FakeHandGtDetector):
    def detect_values(self, _episode_dir: Path, cam: str, frame: int, *, rgb_path_template: str = "") -> list[float] | None:
        return None


class VirtualWorkflowToolSmokeTest(unittest.TestCase):
    def test_virtual_prediction_persistence_uses_smplx_mano_joint_order(self) -> None:
        annotation = []
        for hand in range(2):
            for joint in range(21):
                annotation.extend((float(hand * 100 + joint), 0.0))

        stored = np.asarray(annotation_hand_values_to_mano_order(annotation), dtype=np.float32).reshape(2, 21, 2)

        self.assertEqual(stored[0, 13, 0], 1.0)  # canonical thumb MCP
        self.assertEqual(stored[0, 1, 0], 5.0)   # canonical index MCP

    def test_run_workers_defaults_load_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "ORBBEC_TASK_BACKEND_URL=http://127.0.0.1:9999",
                        f"ORBBEC_WORKFLOW_NAS_ROOT={tmp_path / 'nas'}",
                        "ORBBEC_WORKFLOW_NAS_URI_PREFIX=nas://orbbec-test",
                        "ORBBEC_VIRTUAL_WORKFLOW_WORKERS=all",
                        "ORBBEC_VIRTUAL_WORKFLOW_QC_FAIL_RATE=1.0",
                        "ORBBEC_VIRTUAL_WORKFLOW_MAX_ITERATIONS=0",
                        "ORBBEC_VIRTUAL_WORKFLOW_STOP_AFTER_IDLE_ROUNDS=0",
                        "ORBBEC_VIRTUAL_WORKFLOW_IDLE_LOG_INTERVAL=30",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                load_env_defaults(env_file)
                args = build_parser().parse_args(["run-workers"])

            self.assertEqual(args.backend_url, "http://127.0.0.1:9999")
            self.assertEqual(args.nas_root, tmp_path / "nas")
            self.assertEqual(args.nas_uri_prefix, "nas://orbbec-test")
            self.assertEqual(args.workers, "all")
            self.assertEqual(args.qc_fail_rate, 1.0)
            self.assertEqual(args.max_iterations, 0)
            self.assertEqual(args.stop_after_idle_rounds, 0)
            self.assertEqual(args.idle_log_interval, 30)

    def test_nas_upload_does_not_precreate_auto_label_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "captures"
            self._write_capture_episode(source / "S001" / "pick_object" / "episode_001", frames=1, cameras=["00"])
            nas = NasSimulator(tmp_path / "nas", "nas://orbbec-test")
            task = LabelTask(
                root=source,
                subject="S001",
                task="pick_object",
                episode="episode_001",
                cameras=["00"],
                frames=[0],
            )

            episode_uri = nas.materialize_task(task, copy_source=True, materialize_predictions=False)
            episode_dir = nas.nas_path_for_uri(episode_uri)

            self.assertTrue((episode_dir / "00" / "RGB" / "00000.png").exists())
            self.assertFalse((episode_dir / "pred_2d").exists())

            pred_uri = nas.write_prediction_artifact(episode_uri, ["00"], [0], detector=FakeHandGtDetector())
            pred = np.load(episode_dir / "pred_2d" / "00" / "00000.npy")
            self.assertEqual(pred_uri, episode_uri + "/pred_2d")
            self.assertEqual(pred.shape, (2, 21, 2))
            self.assertTrue(np.any(pred >= 0))

    def test_auto_label_prediction_requires_handgt_detector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas = NasSimulator(tmp_path / "nas", "nas://orbbec-test")
            task = LabelTask(
                root=tmp_path / "captures",
                subject="S001",
                task="pick_object",
                episode="episode_001",
                cameras=["00"],
                frames=[0],
            )
            episode_uri = nas.materialize_task(task, copy_source=False, materialize_predictions=False)

            with self.assertRaisesRegex(BackendError, "requires interaction handGT detector"):
                nas.write_prediction_artifact(episode_uri, ["00"], [0])

    def test_auto_label_no_hand_writes_invisible_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas = NasSimulator(tmp_path / "nas", "nas://orbbec-test")
            task = LabelTask(
                root=tmp_path / "captures",
                subject="S001",
                task="pick_object",
                episode="episode_001",
                cameras=["00"],
                frames=[0],
            )
            episode_uri = nas.materialize_task(task, copy_source=False, materialize_predictions=False)
            episode_dir = nas.nas_path_for_uri(episode_uri)

            nas.write_prediction_artifact(episode_uri, ["00"], [0], detector=NoHandDetector())

            pred = np.load(episode_dir / "pred_2d" / "00" / "00000.npy")
            visible = load_frame_visibility(episode_dir / "pred_2d", "00", 0)
            self.assertEqual(pred.shape, (2, 21, 2))
            self.assertTrue(np.all(pred == -1))
            self.assertEqual(visible, [[False] * 21, [False] * 21])

    def test_mano_worker_triangulates_2d_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas = NasSimulator(tmp_path / "nas", "nas://orbbec-test")
            task = LabelTask(
                root=tmp_path / "captures",
                subject="S001",
                task="pick_object",
                episode="episode_001",
                cameras=["00", "01"],
                frames=[0],
            )
            episode_uri = nas.materialize_task(task, copy_source=False, materialize_predictions=False)
            episode_dir = nas.nas_path_for_uri(episode_uri)
            self._write_camera_calibration(episode_dir, ["00", "01"], width=640, height=480)
            target = self._synthetic_3d_hands(0)
            for cam in task.cameras:
                pred = np.full((2, 21, 2), -1.0, dtype=np.float32)
                for hand in range(2):
                    for joint in range(21):
                        pred[hand, joint] = self._project_test_point(target[hand, joint], cam)
                write_float32_npy(episode_dir / "pred_2d" / cam / "00000.npy", pred.reshape(-1).tolist())

            uri = nas.write_mano_episode_artifact(episode_uri, task.cameras, task.frames)

            self.assertEqual(uri, episode_uri + "/mano/episode")
            joints = np.load(episode_dir / "mano" / "episode" / "joints_3d.npy")
            manifest = json.loads((episode_dir / "mano" / "episode" / "mano_episode.json").read_text(encoding="utf-8"))
            self.assertEqual(joints.shape, (1, 2, 21, 3))
            self.assertFalse(manifest["mock"])
            self.assertEqual(manifest["model"], "dlt_triangulation_from_2d")
            self.assertTrue(np.allclose(joints[0, 0, 0], target[0, 0], atol=1e-4), joints[0, 0, 0])

    def test_mano_worker_keeps_missing_joints_as_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas = NasSimulator(tmp_path / "nas", "nas://orbbec-test")
            task = LabelTask(
                root=tmp_path / "captures",
                subject="S001",
                task="pick_object",
                episode="episode_001",
                cameras=["00", "01"],
                frames=[0],
            )
            episode_uri = nas.materialize_task(task, copy_source=False, materialize_predictions=False)
            episode_dir = nas.nas_path_for_uri(episode_uri)
            self._write_camera_calibration(episode_dir, ["00", "01"], width=640, height=480)
            target = self._synthetic_3d_hands(0)
            for cam in task.cameras:
                pred = np.full((2, 21, 2), -1.0, dtype=np.float32)
                for joint in range(21):
                    pred[0, joint] = self._project_test_point(target[0, joint], cam)
                write_float32_npy(episode_dir / "pred_2d" / cam / "00000.npy", pred.reshape(-1).tolist())

            nas.write_mano_episode_artifact(episode_uri, task.cameras, task.frames)

            joints = np.load(episode_dir / "mano" / "episode" / "joints_3d.npy")
            manifest = json.loads((episode_dir / "mano" / "episode" / "mano_episode.json").read_text(encoding="utf-8"))
            self.assertTrue(np.all(np.isfinite(joints[0, 0])))
            self.assertTrue(np.all(np.isnan(joints[0, 1])))
            self.assertEqual(manifest["metrics"]["valid_joint_count"], 21)
            self.assertEqual(manifest["metrics"]["missing_joint_count"], 21)

    def test_mano_worker_writes_all_nan_frame_when_no_joints_triangulate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas = NasSimulator(tmp_path / "nas", "nas://orbbec-test")
            task = LabelTask(
                root=tmp_path / "captures",
                subject="S001",
                task="pick_object",
                episode="episode_001",
                cameras=["00", "01"],
                frames=[0],
            )
            episode_uri = nas.materialize_task(task, copy_source=False, materialize_predictions=False)
            episode_dir = nas.nas_path_for_uri(episode_uri)
            self._write_camera_calibration(episode_dir, ["00", "01"], width=640, height=480)
            pred = np.full((2, 21, 2), -1.0, dtype=np.float32)
            for cam in task.cameras:
                write_float32_npy(episode_dir / "pred_2d" / cam / "00000.npy", pred.reshape(-1).tolist())

            nas.write_mano_episode_artifact(episode_uri, task.cameras, task.frames)

            joints = np.load(episode_dir / "mano" / "episode" / "joints_3d.npy")
            manifest = json.loads((episode_dir / "mano" / "episode" / "mano_episode.json").read_text(encoding="utf-8"))
            self.assertTrue(np.all(np.isnan(joints)))
            self.assertEqual(manifest["metrics"]["valid_joint_count"], 0)
            self.assertEqual(manifest["metrics"]["missing_joint_count"], 42)

    def test_mano_worker_requires_episode_uri_to_match_configured_nas_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas = NasSimulator(tmp_path / "nas", "nas://orbbec-test")
            payload = {
                "episode_uri": "nas://ego/S001/pick_object/episode_001",
                "scope": "episode",
            }

            with self.assertRaisesRegex(BackendError, "episode 3d requires episode_uri under configured NAS prefix"):
                write_mano_artifact_for_payload(nas, payload, None, ["00", "01"], [0])

    def test_auto_label_worker_decodes_h265_only_episode(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            nas_prefix = "nas://orbbec-test"
            episode_dir = nas_root / "S001" / "pick_object" / "episode_video"
            self._write_camera_calibration(episode_dir, ["00"], width=32, height=24)
            if not self._write_h265_rgb_video(episode_dir / "00" / "RGB", frames=3, frame_offset=10):
                self.skipTest("ffmpeg cannot create h265 fixture")

            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={nas_prefix: str(nas_root)})
            registry = TaskInstanceRegistry(tmp_path / "backend_state", seed_task_files=[])
            runtime = BackendRuntime(registry, service)
            server = TaskHTTPServer(("127.0.0.1", 0), RequestHandler, runtime)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                client = BackendClient(f"http://{host}:{port}", timeout=5)
                nas = NasSimulator(nas_root, nas_prefix)
                service.store.create_or_update_episode(
                    episode_id="episode_video",
                    subject_id="S001",
                    task_name="pick_object",
                    episode_index=1,
                    status="uploaded",
                    episode_uri=f"{nas_prefix}/S001/pick_object/episode_video",
                    frame_count=3,
                    cameras=["00"],
                    metadata={"nas_uri": f"{nas_prefix}/S001/pick_object/episode_video"},
                )
                service.set_stage_leasing("auto_label", True, {"updated_by": "smoke"})
                pushed = service.push_auto_label({"episode_id": "episode_video", "pushed_by": "smoke"})
                self.assertEqual(pushed["created_jobs"], 1)
                auto_job = store.jobs_for_episode("episode_video", "auto_label")[0]
                self.assertEqual(auto_job["payload"]["frames"], [0, 1, 2])

                args = argparse.Namespace(
                    worker_id="video_smoke",
                    lease_seconds=60,
                    max_materialized_frames=0,
                    _interaction_hand_gt_detector=StrictImageHandGtDetector(),
                )
                self.assertTrue(handle_auto_label_once(client, nas, args))
                self.assertTrue((episode_dir / "pred_2d" / "00" / "00000.npy").exists())
                self.assertTrue((episode_dir / "mano" / "episode" / "joints_3d.npy").exists())
                self.assertFalse((episode_dir / "00" / "RGB" / "00000.png").exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_virtual_failed_segments_use_random_ten_to_twenty_frame_ranges(self) -> None:
        random.seed(123)
        segments = virtual_failed_segments(list(range(100)))

        self.assertGreaterEqual(len(segments), 2)
        self.assertLessEqual(len(segments), 3)
        for segment in segments:
            length = int(segment["end_frame"]) - int(segment["start_frame"]) + 1
            self.assertGreaterEqual(length, 10)
            self.assertLessEqual(length, 20)

    def test_collection_virtual_workers_real_label_storage_full_failure_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            nas_prefix = "nas://orbbec-test"
            capture_dir = tmp_path / "captures" / "S001" / "pick_object" / "episode_full"
            self._write_capture_episode(capture_dir, frames=60, cameras=["00", "01"])

            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={nas_prefix: str(nas_root)})
            registry = TaskInstanceRegistry(tmp_path / "backend_state", seed_task_files=[])
            runtime = BackendRuntime(registry, service)
            server = TaskHTTPServer(("127.0.0.1", 0), RequestHandler, runtime)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                base_url = f"http://{host}:{port}"
                client = BackendClient(base_url, timeout=5)
                label_client = LabelBackendClient(base_url, timeout_seconds=5)
                nas = NasSimulator(nas_root, nas_prefix)
                args = argparse.Namespace(
                    worker_id="smoke",
                    lease_seconds=60,
                    max_materialized_frames=0,
                    copy_source=True,
                    qc_fail_rate=1.0,
                    _interaction_hand_gt_detector=FakeHandGtDetector(),
                )

                service.record_collection_confirm(
                    {
                        "reservation_id": "episode_full",
                        "subject_id": "S001",
                        "task_name": "pick_object",
                        "episode_number": 1,
                        "client_id": "collection_smoke",
                        "idempotency_key": "collection_smoke:episode_full",
                        "collection_path": str(capture_dir),
                        "frame_count": 60,
                    }
                )
                self.assertTrue(handle_upload_once(client, nas, args))
                episode = store.get_episode("episode_full")
                self.assertIsNotNone(episode)
                episode_uri = str((episode or {}).get("episode_uri") or "")
                episode_dir = nas.nas_path_for_uri(episode_uri)
                self.assertEqual((episode or {}).get("status"), "uploaded")
                self.assertFalse((episode_dir / "pred_2d").exists())

                auto_jobs = store.jobs_for_episode("episode_full", "auto_label")
                self.assertEqual(len(auto_jobs), 1)
                self.assertEqual(auto_jobs[0]["payload"]["scope"], "episode")
                self.assertEqual(auto_jobs[0]["payload"]["frames"], list(range(60)))

                self.assertTrue(handle_auto_label_once(client, nas, args))
                self.assertTrue((episode_dir / "pred_2d" / "00" / "00000.npy").exists())
                self.assertTrue((episode_dir / "mano" / "episode" / "joints_3d.npy").exists())
                self.assertFalse((episode_dir / "mano" / "episode" / "projected_2d").exists())
                self.assertFalse(any(job["type"] == "manual_3d" for job in store.jobs_for_episode("episode_full")))

                random.seed(7)
                self.assertTrue(handle_qc_once(client, nas, args))
                qc_report = json.loads((episode_dir / "qc" / "qc_report.json").read_text(encoding="utf-8"))
                self.assertFalse(qc_report["passed"])
                self.assertTrue(qc_report["mano_3d_checked"])
                segments = store.segments_for_episode("episode_full")
                self.assertGreaterEqual(len(segments), 2)
                self.assertLessEqual(len(segments), 3)
                for segment in segments:
                    length = int(segment["end_frame"]) - int(segment["start_frame"]) + 1
                    self.assertGreaterEqual(length, 10)
                    self.assertLessEqual(length, 20)

                leased = label_client.lease_label_episode(
                    "real_label_storage_smoke",
                    lease_seconds=60,
                    task_name="pick_object",
                    episode_id="episode_full",
                )
                self.assertEqual(len(leased["segments"]), len(segments))
                self._complete_episode_with_real_label_storage(label_client, leased, {nas_prefix: str(nas_root)})
                manual_3d = store.jobs_for_episode("episode_full", "manual_3d")[0]
                service.complete_job(
                    manual_3d["job_id"],
                    {
                        "result": {"ok": True, "completion_policy": "virtual_test"},
                        "artifacts": [
                            {"kind": "optimized_pose", "metadata": {"source_job_id": manual_3d["job_id"]}},
                            {"kind": "mano_episode", "metadata": {"source_job_id": manual_3d["job_id"]}},
                        ],
                    },
                )

                final_episode = store.get_episode("episode_full")
                self.assertEqual((final_episode or {}).get("status"), "finalized")

                manifest_path = episode_dir / FINAL_3D_SOURCES_REL_PATH
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["episode_status"], "finalized")
                self.assertEqual(manifest["base_3d"]["source"], "manual_3d_episode")
                self.assertEqual(manifest["qc"]["relative_path"], "qc/qc_report.json")
                self.assertTrue(all(item["status"] == "mano_succeeded" for item in manifest["manual_segments"]))
            finally:
                detector = getattr(args, "_interaction_hand_gt_detector", None)
                if detector is not None:
                    detector.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    @staticmethod
    def _write_capture_episode(episode_dir: Path, *, frames: int, cameras: list[str]) -> None:
        VirtualWorkflowToolSmokeTest._write_camera_calibration(episode_dir, cameras)
        for cam in cameras:
            for frame in range(frames):
                write_placeholder_rgb_image(episode_dir / cam / "RGB" / f"{frame:05d}.png")

    @staticmethod
    def _write_camera_calibration(episode_dir: Path, cameras: list[str], *, width: int = 64, height: int = 48) -> None:
        camera_params = {}
        extrinsics = {}
        for index, cam in enumerate(cameras):
            camera_params[cam] = {
                "RGB": {
                    "intrinsic": {"fx": float(width), "fy": float(width), "cx": float(width) * 0.5, "cy": float(height) * 0.5},
                    "distortion": {},
                }
            }
            extrinsics[cam] = {
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "translation": [float(index) * 0.1, 0.0, 0.0],
            }
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "camera_params.json").write_text(json.dumps(camera_params), encoding="utf-8")
        (episode_dir / "extrinsics.json").write_text(json.dumps(extrinsics), encoding="utf-8")

    @staticmethod
    def _synthetic_3d_hands(frame: int) -> np.ndarray:
        target = np.zeros((2, 21, 3), dtype=np.float32)
        drift_x = (int(frame) % 17 - 8) * 0.001
        drift_y = (int(frame) % 11 - 5) * 0.001
        for hand in range(2):
            for joint in range(21):
                target[hand, joint] = [
                    -0.08 + hand * 0.16 + joint * 0.001 + drift_x,
                    -0.03 + joint * 0.002 + drift_y,
                    1.0 + joint * 0.001,
                ]
        return target

    @staticmethod
    def _project_test_point(point: object, cam: str, *, width: int = 640, height: int = 480) -> list[float]:
        xyz = np.asarray(point, dtype=np.float64)
        tx = 0.0 if str(cam) == "00" else 0.1
        cam_xyz = xyz + np.asarray([tx, 0.0, 0.0], dtype=np.float64)
        return [float(width * cam_xyz[0] / cam_xyz[2] + width * 0.5), float(width * cam_xyz[1] / cam_xyz[2] + height * 0.5)]

    @staticmethod
    def _frame_size(episode_dir: Path, cam: str, frame: int, rgb_path_template: str = "") -> tuple[int, int]:
        path = find_rgb_frame_path(episode_dir, cam, frame, rgb_path_template)
        if path is None:
            return 640, 480
        try:
            from tools.virtual_workflow.orbbec_virtual_workflow import read_image_size

            size = read_image_size(path)
        except Exception:
            size = None
        return size or (640, 480)

    @staticmethod
    def _write_h265_rgb_video(rgb_dir: Path, *, frames: int, frame_offset: int = 0) -> bool:
        rgb_dir.mkdir(parents=True, exist_ok=True)
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
                f"testsrc=size=32x24:rate=1:duration={int(frames)}",
                "-frames:v",
                str(int(frames)),
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
            return False
        rows = ["video_frame_index,frame_index,rgb_timestamp_us"]
        for idx in range(frames):
            rows.append(f"{idx},{idx},{int(frame_offset + idx)}")
        (rgb_dir / "rgb.h265.timestamps.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        episode_dir = rgb_dir.parent.parent
        camera = rgb_dir.parent.name
        params_path = episode_dir / "camera_params.json"
        try:
            params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
        except Exception:
            params = {}
        if not isinstance(params, dict):
            params = {}
        cam_obj = params.get(camera) if isinstance(params.get(camera), dict) else {}
        rgb_obj = cam_obj.get("RGB") if isinstance(cam_obj.get("RGB"), dict) else {}
        rgb_obj.update(
            {
                "storageEncoding": "h265",
                "storageFile": "rgb.h265",
                "timestampFile": "rgb.h265.timestamps.csv",
                "intrinsic": rgb_obj.get("intrinsic") or {"fx": 32.0, "fy": 32.0, "cx": 16.0, "cy": 12.0},
                "distortion": rgb_obj.get("distortion") or {},
            }
        )
        cam_obj["RGB"] = rgb_obj
        params[camera] = cam_obj
        params_path.write_text(json.dumps(params), encoding="utf-8")
        return True

    @staticmethod
    def _complete_episode_with_real_label_storage(label_client: LabelBackendClient, leased: dict, mounts: dict[str, str]) -> None:
        payload = dict(leased.get("payload") or {})
        task = correction_task_from_backend_payload(payload, mounts=mounts)
        bundle = load_prediction_bundle(task, mode="pred")
        for frame in task.frames:
            for cam in task.cameras:
                points, visible = view_state_from_bundle(bundle, int(frame), cam)
                apply_view_state_to_corrected(bundle, int(frame), cam, points, visible)
        save_corrected_array(bundle)

        label_client.complete_label_job(
            str(payload.get("episode_id") or ""),
            result={
                "operator_id": "real_label_storage_smoke",
                "frames_completed": list(task.frames),
            },
            artifacts=[
                {
                    "kind": "manual_2d",
                    "metadata": {
                        "scope": "episode",
                        "episode_id": str(payload.get("episode_id") or ""),
                        "cameras": list(task.cameras),
                        "frames": list(task.frames),
                        "operator_id": "real_label_storage_smoke",
                    },
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
