from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from task_backend.job_service import JobService
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

    def test_virtual_nas_uploader_completes_collection_upload_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "captures" / "S001" / "pick_object" / "episode_1"
            (source / "00" / "RGB").mkdir(parents=True)
            (source / "00" / "RGB" / "00001.png").write_bytes(b"rgb")
            (source / "timestamps.csv").write_text("ref_timestamp_us\n1\n", encoding="utf-8")

            store = WorkflowStore(tmp_path / "workflow.sqlite3")
            service = JobService(store)
            service.record_collection_confirm(
                {
                    "reservation_id": "reservation_001",
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "episode_number": 1,
                    "client_id": "smoke",
                    "idempotency_key": "smoke:reservation_001",
                    "local_path": str(source),
                    "frame_count": 1,
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
            episode = store.get_episode("reservation_001")
            self.assertIsNotNone(episode)
            self.assertEqual(episode["status"], "uploaded")  # type: ignore[index]
            self.assertEqual(episode["data_uri"], status_after["upload"]["nas_uri"])  # type: ignore[index]
            nas_path = tmp_path / "virtual_nas" / "S001" / "pick_object" / "reservation_001"
            self.assertTrue((nas_path / "00" / "RGB" / "00001.png").exists())
            self.assertTrue((nas_path / ".orbbec_upload_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
