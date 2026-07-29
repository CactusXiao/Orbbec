from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from task_backend.job_service import JobService
from task_backend.server import render_workflow_stage_page
from task_backend.virtual_nas_uploader import VirtualNasUploadConfig, VirtualNasUploader
from task_backend.workflow_models import WorkflowError
from task_backend.workflow_store import WorkflowStore


class WorkflowStoreSmokeTest(unittest.TestCase):
    def test_manual_label_job_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store)

            episode = store.create_or_update_episode(
                episode_id="episode_000456",
                subject_id="S001",
                task_name="pick_object",
                episode_index=456,
                status="planned",
                data_uri=f"local://{tmp_path / 'S001' / 'pick_object' / 'episode_000456'}",
                cameras=["camera_01", "camera_02"],
                frame_count=4,
                metadata={"smoke": True},
            )
            self.assertEqual(episode["status"], "planned")

            created = service.create_manual_label_job(
                {
                    "episode_id": episode["episode_id"],
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "data_uri": episode["data_uri"],
                    "cameras": ["camera_01", "camera_02"],
                    "frames": [120, 121, 122, 123],
                }
            )
            job_id = created["job"]["job_id"]
            self.assertEqual(created["job"]["type"], "manual_label")
            self.assertEqual(created["job"]["status"], "queued")

            service.set_stage_leasing("manual_label", True, {"updated_by": "smoke"})
            leased = service.lease_job(
                {"operator_id": "labeler_01", "lease_seconds": 60},
                forced_type="manual_label",
            )
            self.assertEqual(leased["job"]["job_id"], job_id)
            self.assertEqual(leased["job"]["status"], "leased")

            with self.assertRaises(WorkflowError):
                service.lease_job(
                    {"operator_id": "labeler_02", "lease_seconds": 60},
                    forced_type="manual_label",
                )

            with sqlite3.connect(store.db_path) as conn:
                conn.execute(
                    "UPDATE jobs SET lease_until = '1970-01-01T00:00:00Z' WHERE job_id = ?",
                    (job_id,),
                )

            leased_again = service.lease_job(
                {"operator_id": "labeler_02", "lease_seconds": 1},
                forced_type="manual_label",
            )
            self.assertEqual(leased_again["job"]["lease_owner"], "labeler_02")

            heartbeat = service.heartbeat_job(
                job_id,
                {"operator_id": "labeler_02", "lease_seconds": 60, "status": "running"},
            )
            self.assertEqual(heartbeat["job"]["status"], "running")
            self.assertGreater(heartbeat["job"]["lease_until"], leased_again["job"]["lease_until"])

            completed = service.complete_job(job_id, {"result": {"ok": True}})
            self.assertTrue(completed["changed"])
            self.assertEqual(completed["job"]["status"], "succeeded")
            self.assertEqual(store.get_episode(episode["episode_id"])["status"], "manual_labeled")  # type: ignore[index]

            completed_again = service.complete_job(job_id, {"result": {"ok": True}})
            self.assertFalse(completed_again["changed"])
            self.assertEqual(completed_again["job"]["status"], "succeeded")

            created_release = service.create_manual_label_job(
                {
                    "episode_id": "episode_release",
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "data_uri": f"local://{tmp_path / 'release'}",
                    "cameras": ["camera_01"],
                    "frames": [1],
                }
            )
            release_job_id = created_release["job"]["job_id"]
            service.lease_job({"operator_id": "labeler_01"}, forced_type="manual_label")
            released = service.release_job(release_job_id, {"reason": "smoke"})
            self.assertTrue(released["released"])
            self.assertEqual(released["job"]["status"], "queued")
            leased_after_release = service.lease_job({"operator_id": "labeler_03"}, forced_type="manual_label")
            self.assertEqual(leased_after_release["job"]["job_id"], release_job_id)

    def test_stage_lease_controls_block_new_leases_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store)
            created = service.create_manual_label_job(
                {
                    "episode_id": "episode_lease_control",
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "data_uri": f"local://{tmp_path / 'episode_lease_control'}",
                    "cameras": ["camera_01"],
                    "frames": [1],
                }
            )
            job_id = created["job"]["job_id"]

            with self.assertRaises(WorkflowError) as disabled:
                service.lease_job({"operator_id": "labeler_01"}, forced_type="manual_label")
            self.assertEqual(disabled.exception.status.value, 409)
            self.assertIn("leasing disabled for job type: manual_label", disabled.exception.message)

            service.set_stage_leasing("manual_label", True, {"updated_by": "smoke"})
            leased = service.lease_job({"operator_id": "labeler_01"}, forced_type="manual_label")
            self.assertEqual(leased["job"]["job_id"], job_id)

            service.set_stage_leasing("manual_label", False, {"updated_by": "smoke"})
            heartbeat = service.heartbeat_job(job_id, {"operator_id": "labeler_01", "status": "running"})
            self.assertEqual(heartbeat["job"]["status"], "running")
            completed = service.complete_job(job_id, {"result": {"ok": True}})
            self.assertEqual(completed["job"]["status"], "succeeded")

            fail_created = service.create_manual_label_job(
                {
                    "episode_id": "episode_fail_after_disable",
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "data_uri": f"local://{tmp_path / 'episode_fail_after_disable'}",
                    "frames": [1],
                }
            )
            fail_job_id = fail_created["job"]["job_id"]
            service.set_stage_leasing("manual_label", True, {"updated_by": "smoke"})
            service.lease_job({"operator_id": "labeler_02"}, forced_type="manual_label")
            service.set_stage_leasing("manual_label", False, {"updated_by": "smoke"})
            failed = service.fail_job(fail_job_id, {"error": "smoke fail"})
            self.assertEqual(failed["job"]["status"], "failed")

            release_created = service.create_manual_label_job(
                {
                    "episode_id": "episode_release_after_disable",
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "data_uri": f"local://{tmp_path / 'episode_release_after_disable'}",
                    "frames": [1],
                }
            )
            release_job_id = release_created["job"]["job_id"]
            service.set_stage_leasing("manual_label", True, {"updated_by": "smoke"})
            service.lease_job({"operator_id": "labeler_03"}, forced_type="manual_label")
            service.set_stage_leasing("manual_label", False, {"updated_by": "smoke"})
            released = service.release_job(release_job_id, {"reason": "smoke"})
            self.assertTrue(released["released"])
            self.assertEqual(released["job"]["status"], "queued")

    def test_virtual_nas_uploader_completes_collection_upload_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "captures" / "S001" / "pick_object" / "episode_1"
            (source / "00" / "RGB").mkdir(parents=True)
            (source / "00" / "RGB" / "00001.png").write_bytes(b"rgb")
            (source / "00" / "RGB" / "00002.png").write_bytes(b"rgb2")
            (source / "timestamps.csv").write_text("ref_timestamp_us\n1\n", encoding="utf-8")

            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, auto_label_batch_size=1)
            service.record_collection_confirm(
                {
                    "reservation_id": "reservation_001",
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "episode_number": 1,
                    "client_id": "smoke",
                    "idempotency_key": "smoke:reservation_001",
                    "local_path": str(source),
                    "frame_count": 2,
                }
            )

            status_before = service.upload_status("reservation_001")
            self.assertEqual(status_before["upload"]["status"], "queued")

            uploader = VirtualNasUploader(
                service,
                VirtualNasUploadConfig(
                    root=tmp_path / "virtual_nas",
                    uri_prefix="nas://orbbec-virtual-test",
                    worker_id="smoke_uploader",
                ),
            )
            self.assertTrue(uploader.process_one())

            status_after = service.upload_status("reservation_001")
            self.assertEqual(status_after["upload"]["status"], "succeeded")
            self.assertEqual(status_after["upload"]["percent"], 100.0)
            self.assertTrue(status_after["upload"]["nas_uri"].startswith("nas://orbbec-virtual-test/"))
            self.assertEqual(status_after["workflow"]["status"], "uploaded")
            self.assertEqual(status_after["workflow"]["job_count"], 1)
            self.assertFalse(any(job["type"] == "auto_label" for job in status_after["jobs"]))
            episode = store.get_episode("reservation_001")
            self.assertIsNotNone(episode)
            self.assertEqual(episode["status"], "uploaded")  # type: ignore[index]
            self.assertEqual(episode["data_uri"], status_after["upload"]["nas_uri"])  # type: ignore[index]
            self.assertEqual(store.jobs_for_episode("reservation_001", "auto_label"), [])

            pushed = service.push_auto_label({"episode_id": "reservation_001", "pushed_by": "smoke"})
            self.assertEqual(pushed["pushed"], 1)
            self.assertEqual(pushed["created_jobs"], 2)
            pushed_again = service.push_auto_label({"episode_id": "reservation_001", "pushed_by": "smoke"})
            self.assertEqual(pushed_again["pushed"], 0)
            self.assertEqual(pushed_again["created_jobs"], 0)
            auto_label_jobs = store.jobs_for_episode("reservation_001", "auto_label")
            self.assertEqual(len(auto_label_jobs), 2)
            self.assertEqual(auto_label_jobs[0]["status"], "queued")
            self.assertEqual(auto_label_jobs[0]["payload"]["frames"], [1])
            self.assertEqual(auto_label_jobs[1]["payload"]["frames"], [2])
            stage = service.workflow_stage("auto_label")
            self.assertEqual(stage["stats"]["queued"], 2)
            self.assertEqual(stage["queued"][0]["episode_url"], "/episodes/reservation_001")
            self.assertIn("/episodes/reservation_001", render_workflow_stage_page(stage))
            nas_path = tmp_path / "virtual_nas" / "S001" / "pick_object" / "reservation_001"
            self.assertTrue((nas_path / "00" / "RGB" / "00001.png").exists())
            self.assertTrue((nas_path / "00" / "RGB" / "00002.png").exists())
            self.assertTrue((nas_path / ".orbbec_upload_manifest.json").exists())

    def test_stage_auto_advance_from_auto_label_to_manual_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store, auto_label_batch_size=10)
            store.create_or_update_episode(
                episode_id="episode_pass",
                subject_id="S001",
                task_name="pick_object",
                episode_index=1,
                status="uploaded",
                data_uri="nas://orbbec-test/S001/pick_object/episode_pass",
                frame_count=2,
                cameras=["camera_01"],
                metadata={"nas_uri": "nas://orbbec-test/S001/pick_object/episode_pass"},
            )
            service.push_auto_label({"episode_id": "episode_pass", "pushed_by": "smoke"})
            service.set_stage_leasing("auto_label", True, {"updated_by": "smoke"})
            auto_job = service.lease_job({"type": "auto_label", "worker_id": "auto_worker"})["job"]
            service.complete_job(auto_job["job_id"], {"result": {"ok": True}})
            self.assertEqual(store.get_episode("episode_pass")["status"], "auto_labeled")  # type: ignore[index]
            self.assertEqual(len(store.artifacts_for_episode("episode_pass")), 1)
            qc_jobs = store.jobs_for_episode("episode_pass", "qc")
            self.assertEqual(len(qc_jobs), 1)
            self.assertEqual(qc_jobs[0]["status"], "queued")

            service.set_stage_leasing("qc", True, {"updated_by": "smoke"})
            qc_job = service.lease_job({"type": "qc", "worker_id": "qc_worker"})["job"]
            service.complete_job(qc_job["job_id"], {"result": {"passed": True, "score": 0.99}})
            self.assertEqual(store.get_episode("episode_pass")["status"], "qc_passed")  # type: ignore[index]
            self.assertEqual(store.jobs_for_episode("episode_pass", "manual_label"), [])
            self.assertTrue(any(item["kind"] == "qc_report" for item in store.artifacts_for_episode("episode_pass")))

            store.create_or_update_episode(
                episode_id="episode_fail",
                subject_id="S001",
                task_name="pick_object",
                episode_index=2,
                status="uploaded",
                data_uri="nas://orbbec-test/S001/pick_object/episode_fail",
                frame_count=2,
                cameras=["camera_01"],
                metadata={"nas_uri": "nas://orbbec-test/S001/pick_object/episode_fail"},
            )
            service.push_auto_label({"episode_id": "episode_fail", "pushed_by": "smoke"})
            auto_job_fail = service.lease_job({"type": "auto_label", "worker_id": "auto_worker"})["job"]
            service.complete_job(auto_job_fail["job_id"], {"result": {"ok": True}})
            qc_job_fail = service.lease_job({"type": "qc", "worker_id": "qc_worker"})["job"]
            service.complete_job(qc_job_fail["job_id"], {"result": {"passed": False, "score": 0.1}})
            self.assertEqual(store.get_episode("episode_fail")["status"], "qc_failed")  # type: ignore[index]
            manual_jobs = store.jobs_for_episode("episode_fail", "manual_label")
            self.assertEqual(len(manual_jobs), 1)
            self.assertEqual(manual_jobs[0]["status"], "queued")
            manual_stage = service.workflow_stage("manual_label")
            self.assertEqual(manual_stage["stats"]["queued"], 1)
            self.assertIn("/episodes/episode_fail", render_workflow_stage_page(manual_stage))

            service.set_stage_leasing("manual_label", True, {"updated_by": "smoke"})
            manual_job = service.lease_job({"type": "manual_label", "operator_id": "labeler"})["job"]
            service.complete_job(manual_job["job_id"], {"result": {"ok": True}})
            self.assertEqual(store.get_episode("episode_fail")["status"], "manual_labeled")  # type: ignore[index]
            self.assertTrue(any(item["kind"] == "corrected_2d" for item in store.artifacts_for_episode("episode_fail")))


if __name__ == "__main__":
    unittest.main()
