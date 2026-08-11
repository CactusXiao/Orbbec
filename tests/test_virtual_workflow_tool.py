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
    handle_auto_label_once,
    handle_mano_opt_once,
    handle_qc_once,
    handle_upload_once,
    build_parser,
    find_rgb_frame_path,
    load_env_defaults,
    virtual_hand_values,
    virtual_failed_segments,
    write_placeholder_rgb_image,
)


class FakeHandGtDetector:
    available = True

    def detect_values(self, _episode_dir: Path, cam: str, frame: int, *, rgb_path_template: str = "") -> list[float]:
        return virtual_hand_values(cam, frame, 640, 480, "pred")

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

            data_uri = nas.materialize_task(task, copy_source=True, materialize_predictions=False)
            episode_dir = nas.local_path_for_uri(data_uri)

            self.assertTrue((episode_dir / "00" / "RGB" / "00000.png").exists())
            self.assertFalse((episode_dir / "pred_2d").exists())

            pred_uri = nas.write_prediction_artifact(data_uri, ["00"], [0], detector=FakeHandGtDetector())
            pred = np.load(episode_dir / "pred_2d" / "00" / "00000.npy")
            self.assertEqual(pred_uri, data_uri + "/pred_2d")
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
            data_uri = nas.materialize_task(task, copy_source=False, materialize_predictions=False)

            with self.assertRaisesRegex(BackendError, "requires interaction handGT detector"):
                nas.write_prediction_artifact(data_uri, ["00"], [0])

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
            data_uri = nas.materialize_task(task, copy_source=False, materialize_predictions=False)
            episode_dir = nas.local_path_for_uri(data_uri)

            nas.write_prediction_artifact(data_uri, ["00"], [0], detector=NoHandDetector())

            pred = np.load(episode_dir / "pred_2d" / "00" / "00000.npy")
            visible = load_frame_visibility(episode_dir / "pred_2d", "00", 0)
            self.assertEqual(pred.shape, (2, 21, 2))
            self.assertTrue(np.all(pred == -1))
            self.assertEqual(visible, [[False] * 21, [False] * 21])

    def test_auto_label_worker_decodes_h265_only_episode(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            nas_prefix = "nas://orbbec-test"
            episode_dir = nas_root / "S001" / "pick_object" / "episode_video"
            if not self._write_h265_rgb_video(episode_dir / "00" / "RGB", frames=3, frame_offset=10):
                self.skipTest("ffmpeg cannot create h265 fixture")

            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, uri_mounts={nas_prefix: str(nas_root)})
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
                    data_uri=f"{nas_prefix}/S001/pick_object/episode_video",
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
            service = JobService(store, uri_mounts={nas_prefix: str(nas_root)})
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
                        "local_path": str(capture_dir),
                        "frame_count": 60,
                    }
                )
                self.assertTrue(handle_upload_once(client, nas, args))
                episode = store.get_episode("episode_full")
                self.assertIsNotNone(episode)
                data_uri = str((episode or {}).get("data_uri") or "")
                episode_dir = nas.local_path_for_uri(data_uri)
                self.assertEqual((episode or {}).get("status"), "uploaded")
                self.assertFalse((episode_dir / "pred_2d").exists())

                for stage in ("auto_label", "mano_opt", "qc", "manual_segment"):
                    service.set_stage_leasing(stage, True, {"updated_by": "smoke"})
                pushed = service.push_auto_label({"episode_id": "episode_full", "pushed_by": "smoke"})
                self.assertEqual(pushed["created_jobs"], 1)
                auto_jobs = store.jobs_for_episode("episode_full", "auto_label")
                self.assertEqual(len(auto_jobs), 1)
                self.assertEqual(auto_jobs[0]["payload"]["scope"], "episode")
                self.assertEqual(auto_jobs[0]["payload"]["frames"], list(range(60)))

                self.assertTrue(handle_auto_label_once(client, nas, args))
                self.assertTrue((episode_dir / "pred_2d" / "00" / "00000.npy").exists())
                self.assertTrue(handle_mano_opt_once(client, nas, args))
                self.assertTrue((episode_dir / "mano" / "episode" / "joints_3d.npy").exists())
                self.assertFalse((episode_dir / "mano" / "episode" / "projected_2d").exists())

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

                completed_segments = 0
                while True:
                    try:
                        leased = label_client.lease_label_segment(
                            "real_label_storage_smoke",
                            lease_seconds=60,
                            task_name="pick_object",
                            episode_id="episode_full",
                        )
                    except Exception as exc:
                        if "no pending manual segment" in str(exc):
                            break
                        raise
                    self._complete_segment_with_real_label_storage(label_client, leased)
                    completed_segments += 1
                    self.assertTrue(handle_mano_opt_once(client, nas, args))

                self.assertEqual(completed_segments, len(segments))
                final_episode = store.get_episode("episode_full")
                self.assertEqual((final_episode or {}).get("status"), "finalized")

                manifest_path = episode_dir / FINAL_3D_SOURCES_REL_PATH
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["episode_status"], "finalized")
                self.assertEqual(manifest["ready_override_count"], len(segments))
                self.assertEqual(manifest["qc"]["relative_path"], "qc/qc_report.json")
                for override in manifest["overrides"]:
                    self.assertEqual(override["status"], "ready")
                    self.assertTrue((episode_dir / str(override["manual_2d_relative_path"])).exists())
                    self.assertTrue((episode_dir / str(override["relative_path"])).exists())
            finally:
                detector = getattr(args, "_interaction_hand_gt_detector", None)
                if detector is not None:
                    detector.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    @staticmethod
    def _write_capture_episode(episode_dir: Path, *, frames: int, cameras: list[str]) -> None:
        for cam in cameras:
            for frame in range(frames):
                write_placeholder_rgb_image(episode_dir / cam / "RGB" / f"{frame:05d}.png")

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
        return True

    @staticmethod
    def _complete_segment_with_real_label_storage(label_client: LabelBackendClient, leased: dict) -> None:
        payload = dict(leased.get("payload") or {})
        segment = dict(leased.get("segment") or {})
        task = correction_task_from_backend_payload(payload)
        bundle = load_prediction_bundle(task, mode="pred")
        for frame in task.frames:
            for cam in task.cameras:
                points, visible = view_state_from_bundle(bundle, int(frame), cam)
                apply_view_state_to_corrected(bundle, int(frame), cam, points, visible)
        save_corrected_array(bundle)

        artifact_uri = str(payload.get("manual_2d_output_uri") or payload.get("manual_2d_uri") or "").strip()
        if not artifact_uri:
            data_uri = str(payload.get("data_uri") or "").rstrip("/")
            correction_dir = str(payload.get("correction_dir") or task.correction_dir or "manual_2d").strip("/")
            artifact_uri = f"{data_uri}/{correction_dir}" if data_uri and correction_dir else data_uri
        label_client.complete_label_job(
            str(segment.get("segment_id") or payload.get("segment_id") or ""),
            result={
                "operator_id": "real_label_storage_smoke",
                "frames_completed": list(task.frames),
                "local_progress_cache": "",
            },
            artifacts=[
                {
                    "kind": "manual_2d",
                    "uri": artifact_uri,
                    "metadata": {
                        "segment_id": segment.get("segment_id") or payload.get("segment_id") or "",
                        "cameras": list(task.cameras),
                        "frames": list(task.frames),
                        "operator_id": "real_label_storage_smoke",
                    },
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
