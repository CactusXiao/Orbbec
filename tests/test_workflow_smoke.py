from __future__ import annotations

import json
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path

from task_backend.job_service import FINAL_3D_SOURCES_REL_PATH, JobService
from task_backend.server import TaskBackend, episode_storage_name, next_episode_number, render_episode_detail, render_workflow_stage_page
from task_backend.nas_uploader import NasUploadConfig, NasUploader
from task_backend.workflow_models import WorkflowError
from task_backend.workflow_store import WorkflowStore


class WorkflowStoreSmokeTest(unittest.TestCase):
    def test_episode_storage_name_is_human_readable_and_never_reuses_released_number(self) -> None:
        self.assertEqual(
            episode_storage_name("xiaojiazhou", "task-clean-the-bowl", 12),
            "episode12",
        )
        subject = {
            "reservations": {
                "released-uuid": {
                    "task_name": "task-clean-the-bowl",
                    "episode_number": 1,
                    "status": "released",
                }
            }
        }
        self.assertEqual(next_episode_number(subject, "task-clean-the-bowl"), 2)

    def test_collection_reservation_persists_uuid_to_storage_name_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_file = tmp_path / "tasks.json"
            task_file.write_text(
                json.dumps({"tasks": [{"task_name": "task-clean-the-bowl", "total": 20}]}),
                encoding="utf-8",
            )
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(tmp_path / "nas")})
            backend = TaskBackend(
                tmp_path / "state",
                task_file,
                workflow_service=service,
            )

            reservation = backend.reserve(
                {
                    "client_id": "capture-01",
                    "subject_id": "xiaojiazhou",
                    "task_name": "task-clean-the-bowl",
                }
            )
            self.assertEqual(reservation["storage_name"], "episode1")
            episode_uuid = reservation["reservation_id"]
            mapped = store.get_episode(episode_uuid)
            self.assertEqual(mapped["episode_id"], episode_uuid)  # type: ignore[index]
            self.assertEqual(mapped["storage_name"], reservation["storage_name"])  # type: ignore[index]

            episode_uri = f"nas://ego/xiaojiazhou/task-clean-the-bowl/{reservation['storage_name']}"
            backend.confirm(
                {
                    **reservation,
                    "subject_id": "xiaojiazhou",
                    "collection_path": "/capture/local/episode_1",
                    "episode_uri": episode_uri,
                    "idempotency_key": f"capture-01:{episode_uuid}",
                }
            )
            mapped = store.get_episode(episode_uuid)
            self.assertEqual(mapped["episode_uri"], episode_uri)  # type: ignore[index]

    def test_workflow_stage_leases_are_open_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "workflow.sqlite3"
            store = WorkflowStore(db_path)
            for job_type in ("auto_label", "mano_opt", "qc", "manual_segment"):
                self.assertTrue(store.get_stage_control(job_type)["lease_enabled"])

            store.set_stage_control(job_type="auto_label", lease_enabled=False, updated_by="system", note="default paused")
            store.set_stage_control(job_type="qc", lease_enabled=False, updated_by="api", note="manual pause")

            reopened = WorkflowStore(db_path)
            self.assertTrue(reopened.get_stage_control("auto_label")["lease_enabled"])
            self.assertFalse(reopened.get_stage_control("qc")["lease_enabled"])

    def test_episode_detail_page_is_grouped_by_current_workflow(self) -> None:
        html = render_episode_detail(
            {
                "reservation": {
                    "reservation_id": "episode_001",
                    "task_name": "pick_object",
                    "subject_id": "S001",
                    "episode_number": 1,
                    "status": "confirmed",
                    "collection_path": "/captures/S001/pick_object/episode_1",
                    "created_at": "2026-01-01T00:00:00Z",
                    "confirmed_at": "2026-01-01T00:02:00Z",
                    "updated_at": "2026-01-01T00:02:00Z",
                    "stats": {
                        "duration_label": "10.00 s",
                        "frame_count_label": "300",
                        "storage_label": "1.00 GB",
                    },
                },
                "task": {"task_name": "pick_object", "total": 1, "raw": {}},
                "metadata_pairs": [],
                "workflow": {
                    "episode": {
                        "episode_id": "episode_001",
                        "episode_uri": "nas://ego/S001/pick_object/episode_1",
                        "metadata": {
                            "collection_operator_id": "collector",
                            "qc_operator_id": "qc_user",
                            "manual_correction_operator_id": "labeler",
                        },
                        "updated_at": "2026-01-01T00:10:00Z",
                    },
                    "workflow": {
                        "status": "manual_correction_pending",
                        "active_job_type": "qc",
                        "active_job_status": "succeeded",
                        "job_count": 3,
                        "updated_at": "2026-01-01T00:10:00Z",
                    },
                    "upload": {
                        "available": True,
                        "status": "succeeded",
                        "phase": "complete",
                        "percent": 100,
                        "files_done": 10,
                        "files_total": 10,
                        "copied_bytes": 1024,
                        "total_bytes": 1024,
                        "nas_uri": "nas://ego/S001/pick_object/episode_1",
                    },
                    "workflow_artifacts": [
                        {"kind": "pred_2d", "uri": "nas://ego/S001/pick_object/episode_1/pred_2d", "metadata": {}, "created_at": "t1"},
                        {"kind": "mano_episode", "uri": "nas://ego/S001/pick_object/episode_1/mano/episode", "metadata": {}, "created_at": "t2"},
                        {"kind": "qc_report", "uri": "nas://ego/S001/pick_object/episode_1/qc/qc_report.json", "metadata": {}, "created_at": "t3"},
                    ],
                    "segments": [
                        {
                            "segment_id": "seg_1",
                            "start_frame": 10,
                            "end_frame": 20,
                            "status": "pending_manual",
                            "metadata": {"reason": "qc_failed", "operator_id": "labeler"},
                        }
                    ],
                    "jobs": [
                        {"job_id": "upload_episode_001", "type": "upload", "status": "succeeded", "updated_at": "t0"},
                        {"job_id": "auto_label_episode_001", "type": "auto_label", "status": "succeeded", "updated_at": "t1"},
                        {"job_id": "qc_episode_001", "type": "qc", "status": "succeeded", "operator_id": "qc_user", "updated_at": "t3"},
                    ],
                },
            }
        )
        self.assertIn("当前流程", html)
        self.assertIn("上传与存储", html)
        self.assertIn("QC 失败分段 / 人工纠偏", html)
        self.assertIn("操作员", html)
        self.assertIn("collector", html)
        self.assertIn("qc_user", html)
        self.assertIn("labeler", html)
        self.assertIn("1. 采集预约 / 确认", html)
        self.assertIn("3. 自动标注 + Episode 3D", html)
        self.assertIn("5. 人工纠偏 / Segment 3D / 最终 3D", html)
        self.assertNotIn("Raw Reservation JSON", html)

    def test_nas_uploader_always_queues_auto_label_after_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "captures" / "S001" / "pick_object" / "episode_1"
            (source / "00" / "RGB").mkdir(parents=True)
            (source / "00" / "RGB" / "00001.png").write_bytes(b"rgb")
            (source / "00" / "RGB" / "00002.png").write_bytes(b"rgb2")
            (source / "ego" / "RGB").mkdir(parents=True)
            (source / "ego" / "RGB" / "00001.png").write_bytes(b"pico")
            (source / "timestamps.csv").write_text("ref_timestamp_us\n1\n", encoding="utf-8")

            nas_root = tmp_path / "nas"
            nas_prefix = "nas://ego-test"
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, auto_label_after_upload=False, nas_mounts={nas_prefix: str(nas_root)})
            service.record_collection_confirm(
                {
                    "reservation_id": "reservation_001",
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "episode_number": 1,
                    "client_id": "smoke",
                    "idempotency_key": "smoke:reservation_001",
                    "collection_path": str(source),
                    "frame_count": 2,
                    "operator_id": "collector",
                }
            )
            episode = store.get_episode("reservation_001")
            self.assertEqual(episode["subject_id"], "S001")  # type: ignore[index]
            self.assertEqual(episode["storage_name"], "episode1")  # type: ignore[index]
            self.assertEqual(episode["metadata"]["collection_operator_id"], "collector")  # type: ignore[index]

            uploader = NasUploader(
                service,
                NasUploadConfig(
                    root=nas_root,
                    uri_prefix=nas_prefix,
                    worker_id="smoke_uploader",
                ),
            )
            self.assertTrue(uploader.process_one())

            status_after = service.upload_status("reservation_001")
            self.assertEqual(status_after["upload"]["status"], "succeeded")
            self.assertEqual(status_after["workflow"]["status"], "uploaded")
            self.assertTrue(any(job["type"] == "auto_label" for job in status_after["jobs"]))
            auto_jobs = store.jobs_for_episode("reservation_001", "auto_label")
            self.assertEqual(len(auto_jobs), 1)
            self.assertEqual(auto_jobs[0]["payload"]["scope"], "episode")
            self.assertEqual(auto_jobs[0]["payload"]["frames"], [1, 2])
            self.assertEqual(auto_jobs[0]["payload"]["cameras"], ["00"])
            self.assertIn("/episodes/reservation_001", render_workflow_stage_page(service.workflow_stage("auto_label")))
            nas_episode_dir = nas_root / "S001" / "pick_object" / "episode1"
            self.assertTrue(nas_episode_dir.is_dir())
            manifest = json.loads((nas_episode_dir / ".orbbec_upload_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["episode_uuid"], "reservation_001")
            self.assertEqual(manifest["storage_name"], "episode1")

    def test_capture_side_upload_confirm_skips_backend_upload_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            episode_dir = nas_root / "S001" / "pick_object" / "episode1"
            (episode_dir / "00" / "RGB").mkdir(parents=True)
            (episode_dir / "00" / "RGB" / "00001.png").write_bytes(b"rgb")
            (episode_dir / "00" / "RGB" / "00002.png").write_bytes(b"rgb2")
            (episode_dir / "timestamps.csv").write_text("ref_timestamp_us\n1\n", encoding="utf-8")

            nas_prefix = "nas://ego-test"
            episode_uri = f"{nas_prefix}/S001/pick_object/episode1"
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={nas_prefix: str(nas_root)})
            service.record_collection_confirm(
                {
                    "reservation_id": "reservation_direct",
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "episode_number": 1,
                    "client_id": "capture",
                    "idempotency_key": "capture:reservation_direct",
                    "collection_path": "/data/local/S001/pick_object/episode_1",
                    "episode_uri": episode_uri,
                    "frame_count": 2,
                    "operator_id": "collector",
                }
            )

            status = service.upload_status("reservation_direct")
            self.assertEqual(status["workflow"]["status"], "uploaded")
            self.assertEqual(status["upload"]["status"], "succeeded")
            self.assertEqual(status["upload"]["phase"], "capture_uploaded")
            self.assertEqual(status["upload"]["nas_uri"], episode_uri)
            self.assertEqual(status["episode"]["storage_name"], "episode1")
            self.assertFalse(any(job["type"] == "upload" for job in status["jobs"]))
            self.assertTrue(any(job["type"] == "auto_label" for job in status["jobs"]))
            self.assertTrue(any(item["kind"] == "nas_episode" and item["uri"] == episode_uri for item in status["artifacts"]))

    def test_auto_label_mano_episode_qc_pass_finalizes_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episode_dir = tmp_path / "S001" / "pick_object" / "episode_pass"
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(tmp_path)})
            self._create_uploaded_episode(store, tmp_path, "episode_pass", frames=2)

            service.push_auto_label({"episode_id": "episode_pass", "pushed_by": "smoke"})
            auto_job = store.jobs_for_episode("episode_pass", "auto_label")[0]
            service.complete_job(
                auto_job["job_id"],
                {
                    "result": {"ok": True},
                    "artifacts": [
                        {"kind": "pred_2d", "metadata": {"mock": True}},
                        {"kind": "mano_episode", "metadata": {"mock": True}},
                    ],
                },
            )
            self.assertEqual(store.get_episode("episode_pass")["status"], "mano_optimized")  # type: ignore[index]
            self.assertEqual(store.jobs_for_episode("episode_pass", "mano_opt"), [])

            qc_jobs = store.jobs_for_episode("episode_pass", "qc")
            self.assertEqual(len(qc_jobs), 1)
            self.assertEqual(qc_jobs[0]["payload"]["reason"], "auto_label_episode_3d_succeeded")
            service.complete_job(
                qc_jobs[0]["job_id"],
                {
                    "result": {"passed": True, "score": 0.99, "operator_id": "qc_user"},
                    "artifacts": [{"kind": "qc_report", "metadata": {"passed": True, "operator_id": "qc_user"}}],
                },
            )
            self.assertEqual(store.get_episode("episode_pass")["status"], "finalized")  # type: ignore[index]
            self.assertEqual(store.get_episode("episode_pass")["metadata"]["qc_operator_id"], "qc_user")  # type: ignore[index]
            kinds = [item["kind"] for item in store.artifacts_for_episode("episode_pass")]
            self.assertIn("pred_2d", kinds)
            self.assertIn("mano_episode", kinds)
            self.assertIn("qc_report", kinds)
            manifest = json.loads((episode_dir / FINAL_3D_SOURCES_REL_PATH).read_text(encoding="utf-8"))
            self.assertEqual(manifest["qc"]["status"], "passed")
            self.assertEqual(manifest["base_3d"]["relative_path"], "mano/episode")
            self.assertEqual(manifest["overrides"], [])

    def test_qc_failure_segments_manual_segment_mano_patch_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episode_dir = tmp_path / "S001" / "pick_object" / "episode_fail"
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(tmp_path)})
            self._create_uploaded_episode(store, tmp_path, "episode_fail", frames=30, cameras=["00", "01"])
            self._advance_to_qc_job(service, store, "episode_fail")

            qc_job = store.jobs_for_episode("episode_fail", "qc")[0]
            service.complete_job(
                qc_job["job_id"],
                {
                    "result": {
                        "passed": False,
                        "score": 0.2,
                        "operator_id": "qc_user",
                        "segments": [
                            {"start_frame": 10, "end_frame": 12, "reason": "low_confidence"},
                            {"start_frame": 20, "end_frame": 21, "reason": "temporal_jump"},
                        ],
                    }
                },
            )
            self.assertEqual(store.get_episode("episode_fail")["status"], "manual_correction_pending")  # type: ignore[index]
            self.assertEqual(store.get_episode("episode_fail")["metadata"]["qc_operator_id"], "qc_user")  # type: ignore[index]
            segments = store.segments_for_episode("episode_fail")
            self.assertEqual([(s["start_frame"], s["end_frame"], s["status"]) for s in segments], [(10, 12, "pending_manual"), (20, 21, "pending_manual")])
            manifest = json.loads((episode_dir / FINAL_3D_SOURCES_REL_PATH).read_text(encoding="utf-8"))
            self.assertEqual([(o["start_frame"], o["end_frame"], o["status"]) for o in manifest["overrides"]], [(10, 12, "pending_manual"), (20, 21, "pending_manual")])
            self.assertEqual(service.label_tasks()["tasks"][0]["segments"], 2)
            self.assertEqual(service.label_task_episodes("pick_object")["episodes"][0]["episode_id"], "episode_fail")

            service.set_stage_leasing("manual_segment", True, {"updated_by": "smoke"})
            leased = service.lease_label_segment({"operator_id": "labeler", "task_name": "pick_object", "episode_id": "episode_fail"})
            self.assertEqual(leased["segment"]["start_frame"], 10)
            self.assertEqual(leased["payload"]["frames"], [10, 11, 12])
            self.assertEqual(leased["payload"]["cameras"], ["00", "01"])
            self.assertNotIn("correction_dir", leased["payload"])
            self.assertNotIn("episode_media", leased["payload"])
            self.assertNotIn("rgb_path_template", leased["payload"])

            first_segment_id = leased["segment"]["segment_id"]
            completed = service.complete_label_segment(
                first_segment_id,
                {
                    "result": {"operator_id": "labeler", "frames_completed": [10, 11, 12]},
                    "artifacts": [{"kind": "manual_2d", "metadata": {"segment_id": first_segment_id}}],
                },
            )
            self.assertEqual(completed["segment"]["status"], "mano_queued")
            self.assertEqual(store.get_episode("episode_fail")["metadata"]["manual_correction_operator_id"], "labeler")  # type: ignore[index]
            manifest = json.loads((episode_dir / FINAL_3D_SOURCES_REL_PATH).read_text(encoding="utf-8"))
            self.assertEqual(manifest["overrides"][0]["status"], "mano_queued")
            self.assertEqual(manifest["overrides"][0]["manual_2d_relative_path"], f"manual_2d/segments/{first_segment_id}")
            segment_mano = [job for job in store.jobs_for_episode("episode_fail", "mano_opt") if job["payload"].get("scope") == "segment"]
            self.assertEqual(len(segment_mano), 1)
            self.assertEqual(segment_mano[0]["payload"]["segment_id"], first_segment_id)
            self.assertNotIn("rgb_path_template", segment_mano[0]["payload"])

            service.complete_job(
                segment_mano[0]["job_id"],
                {
                    "result": {"ok": True},
                    "artifacts": [{"kind": "mano_segment_patch", "metadata": {"segment_id": first_segment_id}}],
                },
            )
            self.assertNotEqual(store.get_episode("episode_fail")["status"], "finalized")  # type: ignore[index]
            manifest = json.loads((episode_dir / FINAL_3D_SOURCES_REL_PATH).read_text(encoding="utf-8"))
            self.assertEqual(manifest["ready_override_count"], 1)
            self.assertEqual(manifest["overrides"][0]["status"], "ready")
            self.assertEqual(manifest["overrides"][0]["relative_path"], f"mano/segments/{first_segment_id}")

            second = service.lease_label_segment({"operator_id": "labeler", "task_name": "pick_object", "episode_id": "episode_fail"})
            second_segment_id = second["segment"]["segment_id"]
            self.assertEqual(second["segment"]["start_frame"], 20)
            service.complete_label_segment(
                second_segment_id,
                {
                    "result": {"operator_id": "labeler", "frames_completed": [20, 21]},
                    "artifacts": [{"kind": "manual_2d", "metadata": {"segment_id": second_segment_id}}],
                },
            )
            second_mano = [
                job
                for job in store.jobs_for_episode("episode_fail", "mano_opt")
                if job["payload"].get("scope") == "segment" and job["payload"].get("segment_id") == second_segment_id
            ][0]
            service.complete_job(
                second_mano["job_id"],
                {
                    "result": {"ok": True},
                    "artifacts": [{"kind": "mano_segment_patch", "metadata": {"segment_id": second_segment_id}}],
                },
            )
            self.assertEqual(store.get_episode("episode_fail")["status"], "finalized")  # type: ignore[index]
            manifest = json.loads((episode_dir / FINAL_3D_SOURCES_REL_PATH).read_text(encoding="utf-8"))
            self.assertEqual(manifest["episode_status"], "finalized")
            self.assertEqual(manifest["ready_override_count"], 2)
            self.assertTrue(all(item["status"] == "ready" for item in manifest["overrides"]))

    def test_qc_bad_episode_does_not_create_manual_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(tmp_path)})
            self._create_uploaded_episode(store, tmp_path, "episode_bad", frames=30)
            self._advance_to_qc_job(service, store, "episode_bad")

            qc_job = store.jobs_for_episode("episode_bad", "qc")[0]
            service.complete_job(
                qc_job["job_id"],
                {
                    "result": {
                        "passed": False,
                        "qc_passed": False,
                        "result_type": "bad_episode",
                        "bad_episode": True,
                        "reason": "sensor_data_unusable",
                        "segments": [],
                    }
                },
            )

            self.assertEqual(store.get_episode("episode_bad")["status"], "qc_bad_episode")  # type: ignore[index]
            self.assertEqual(store.segments_for_episode("episode_bad"), [])

    def test_qc_lease_can_target_exact_episode_or_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(tmp_path)})
            self._create_uploaded_episode(store, tmp_path, "episode_a", frames=3, episode_index=1)
            self._create_uploaded_episode(store, tmp_path, "episode_b", frames=3, episode_index=2)
            self._advance_to_qc_job(service, store, "episode_a")
            self._advance_to_qc_job(service, store, "episode_b")
            service.set_stage_leasing("qc", True, {"updated_by": "smoke"})
            target_job = store.jobs_for_episode("episode_b", "qc")[0]

            leased = service.lease_job(
                {
                    "type": "qc",
                    "worker_id": "qc_worker",
                    "lease_seconds": 60,
                    "episode_id": "episode_b",
                    "job_id": target_job["job_id"],
                }
            )

            self.assertEqual(leased["job"]["episode_id"], "episode_b")
            self.assertEqual(leased["job"]["job_id"], target_job["job_id"])
            self.assertEqual(store.jobs_for_episode("episode_a", "qc")[0]["status"], "queued")

    def test_backend_maps_nas_episode_uri_to_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(nas_root)})

            resolved = service.nas_root_dir_from_uri("nas://ego/S001/pick_object/episode_001")

            self.assertEqual(Path(resolved), (nas_root / "S001" / "pick_object" / "episode_001").resolve())

    def test_manual_segment_lease_orders_by_task_episode_and_start_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(tmp_path)})
            self._create_uploaded_episode(store, tmp_path, "episode_late", frames=10, episode_index=2)
            self._create_uploaded_episode(store, tmp_path, "episode_early", frames=10, episode_index=1)
            store.create_segment(segment_id="late_001", episode_id="episode_late", start_frame=1, end_frame=1)
            store.create_segment(segment_id="early_005", episode_id="episode_early", start_frame=5, end_frame=5)
            store.create_segment(segment_id="early_002", episode_id="episode_early", start_frame=2, end_frame=3)

            service.set_stage_leasing("manual_segment", True, {"updated_by": "smoke"})
            first = service.lease_label_segment({"operator_id": "labeler", "task_name": "pick_object"})
            self.assertEqual(first["segment"]["segment_id"], "early_002")
            released = service.release_label_segment("early_002", {"reason": "smoke"})
            self.assertTrue(released["released"])
            second = service.lease_label_segment({"operator_id": "labeler", "task_name": "pick_object", "episode_id": "episode_early"})
            self.assertEqual(second["segment"]["segment_id"], "early_002")

            with self.assertRaises(WorkflowError):
                service.lease_label_segment({"operator_id": "labeler", "task_name": "missing"})

    def test_label_queue_lists_only_currently_leaseable_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(tmp_path)})
            self._create_uploaded_episode(store, tmp_path, "episode_active", frames=10, episode_index=1)
            self._create_uploaded_episode(store, tmp_path, "episode_pending", frames=10, episode_index=2)
            self._create_uploaded_episode(store, tmp_path, "episode_expired", frames=10, episode_index=3)
            store.create_segment(segment_id="active_seg", episode_id="episode_active", start_frame=1, end_frame=1)
            store.create_segment(segment_id="pending_seg", episode_id="episode_pending", start_frame=2, end_frame=2)
            store.create_segment(segment_id="expired_seg", episode_id="episode_expired", start_frame=3, end_frame=3, status="manual_labeling")
            store.lease_segment(lease_owner="active_labeler", lease_seconds=3600, task_name="pick_object", episode_id="episode_active")
            with store.connect() as conn:
                conn.execute(
                    "UPDATE segments SET lease_owner = ?, lease_until = ? WHERE segment_id = ?",
                    ("stale_labeler", "2000-01-01T00:00:00Z", "expired_seg"),
                )

            tasks = service.label_tasks()["tasks"]
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["segments"], 2)
            self.assertEqual(tasks[0]["pending_segments"], 1)
            self.assertEqual(tasks[0]["leased_segments"], 1)

            episodes = {item["episode_id"]: item for item in service.label_task_episodes("pick_object")["episodes"]}
            self.assertNotIn("episode_active", episodes)
            self.assertEqual(episodes["episode_pending"]["segments"], 1)
            self.assertEqual(episodes["episode_expired"]["segments"], 1)
            self.assertEqual(episodes["episode_expired"]["leased_segments"], 1)

            service.set_stage_leasing("manual_segment", True, {"updated_by": "smoke"})
            with self.assertRaises(WorkflowError) as cm:
                service.lease_label_segment({"operator_id": "labeler", "task_name": "pick_object", "episode_id": "episode_active"})
            self.assertEqual(cm.exception.status, HTTPStatus.NOT_FOUND)

            leased = service.lease_label_segment({"operator_id": "labeler", "task_name": "pick_object", "episode_id": "episode_expired"})
            self.assertEqual(leased["segment"]["segment_id"], "expired_seg")

    def test_segment_mano_failure_requeues_only_segment_and_cleans_attempt_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episode_dir = tmp_path / "S001" / "pick_object" / "episode_retry"
            official_pred = episode_dir / "pred_2d" / "00"
            official_pred.mkdir(parents=True)
            (official_pred / "00000.npy").write_bytes(b"official")
            attempt_dir = episode_dir / ".tmp" / "mano_attempt"
            attempt_dir.mkdir(parents=True)
            (attempt_dir / "partial.json").write_text("partial", encoding="utf-8")

            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(tmp_path)})
            self._create_uploaded_episode(store, tmp_path, "episode_retry", frames=1)
            store.register_artifact(episode_id="episode_retry", kind="pred_2d", uri="nas://ego/S001/pick_object/episode_retry/pred_2d")
            store.register_artifact(episode_id="episode_retry", kind="mano_episode", uri="nas://ego/S001/pick_object/episode_retry/mano/episode")
            segment = store.create_segment(segment_id="retry_seg", episode_id="episode_retry", start_frame=0, end_frame=0)
            service.complete_label_segment(
                segment["segment_id"],
                {
                    "result": {"operator_id": "labeler"},
                    "artifacts": [{"kind": "manual_2d", "metadata": {"segment_id": "retry_seg"}}],
                },
            )
            mano_job = [
                job
                for job in store.jobs_for_episode("episode_retry", "mano_opt")
                if job["payload"].get("scope") == "segment"
            ][0]

            service.fail_job(
                mano_job["job_id"],
                {
                    "error": "optimizer crashed",
                    "result": {"cleanup_manifest": {"paths": [str(attempt_dir)]}},
                },
            )
            self.assertFalse(attempt_dir.exists())
            self.assertTrue((official_pred / "00000.npy").exists())
            self.assertTrue(any(item["kind"] == "pred_2d" for item in store.artifacts_for_episode("episode_retry")))
            updated_segment = store.get_segment("retry_seg")
            self.assertIsNotNone(updated_segment)
            self.assertEqual(updated_segment["status"], "mano_queued")  # type: ignore[index]
            segment_mano_jobs = [job for job in store.jobs_for_episode("episode_retry", "mano_opt") if job["payload"].get("scope") == "segment"]
            self.assertEqual(len(segment_mano_jobs), 2)
            self.assertEqual(segment_mano_jobs[0]["status"], "failed")
            self.assertEqual(segment_mano_jobs[1]["status"], "queued")
            self.assertEqual(segment_mano_jobs[1]["payload"]["segment_id"], "retry_seg")

    @staticmethod
    def _create_uploaded_episode(
        store: WorkflowStore,
        tmp_path: Path,
        episode_id: str,
        *,
        frames: int,
        cameras: list[str] | None = None,
        episode_index: int = 1,
        episode_uri: str = "",
    ) -> None:
        store.create_or_update_episode(
            episode_id=episode_id,
            subject_id="S001",
            task_name="pick_object",
            episode_index=episode_index,
            status="uploaded",
            episode_uri=episode_uri or f"nas://ego/S001/pick_object/{episode_id}",
            frame_count=frames,
            cameras=cameras or ["00"],
            metadata={"nas_uri": episode_uri or f"nas://ego/S001/pick_object/{episode_id}"},
        )

    def _advance_to_qc_job(self, service: JobService, store: WorkflowStore, episode_id: str) -> None:
        service.push_auto_label({"episode_id": episode_id, "pushed_by": "smoke"})
        for auto_job in store.jobs_for_episode(episode_id, "auto_label"):
            service.complete_job(auto_job["job_id"], {"result": {"ok": True}})
        self.assertEqual(store.jobs_for_episode(episode_id, "mano_opt"), [])


if __name__ == "__main__":
    unittest.main()
