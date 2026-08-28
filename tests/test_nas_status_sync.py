from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from task_backend.job_service import JobService
from task_backend.nas_status_sync import NasStatusSync, NasStatusSyncConfig
from task_backend.workflow_store import WorkflowStore


class NasStatusSyncTest(unittest.TestCase):
    def _service(self, root: Path) -> JobService:
        store = WorkflowStore(root / "workflow.sqlite3")
        store.create_or_update_episode(
            episode_id="episode_uuid",
            subject_id="S001",
            task_name="pick_object",
            storage_name="episode-001",
            status="mano_optimized",
            episode_uri="nas://ego/S001/pick_object/episode-001",
            frame_count=20,
            cameras=["camera-01"],
        )
        store.create_job(
            job_id="qc_episode_uuid",
            job_type="qc",
            episode_id="episode_uuid",
            payload={
                "episode_id": "episode_uuid",
                "episode_uri": "nas://ego/S001/pick_object/episode-001",
                "frames": list(range(20)),
            },
        )
        return JobService(store, nas_mounts={"nas://ego": str(root / "nas")})

    def test_qc_take_and_pass_are_deduplicated_and_executed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            service.lease_job({"type": "qc", "worker_id": "qc-01", "lease_seconds": 60})
            service.complete_job("qc_episode_uuid", {"result": {"passed": True}})
            service.complete_job("qc_episode_uuid", {"result": {"passed": True}})

            events = service.store.list_nas_sync_events()
            self.assertEqual([event["action"] for event in events], ["quality_take", "quality_passed"])
            self.assertTrue(all(event["nas_episode_id"] == "S001/pick_object/episode-001" for event in events))

            calls = []
            worker = NasStatusSync(
                service.store,
                NasStatusSyncConfig(enabled=True),
                executor=lambda argv: calls.append(list(argv)),
            )
            self.assertTrue(worker.run_once())
            self.assertTrue(worker.run_once())
            self.assertFalse(worker.run_once())
            self.assertIn(
                "sudo -- /usr/local/sbin/nas-uploader-check take S001/pick_object/episode-001",
                calls[0][-1],
            )
            self.assertIn(
                "sudo -- /usr/local/sbin/nas-uploader-check result passed S001/pick_object/episode-001",
                calls[1][-1],
            )
            self.assertTrue(all(event["status"] == "succeeded" for event in service.store.list_nas_sync_events()))

    def test_failed_qc_queues_episode_manual_job_without_duplicate_manual_publish_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            service.lease_job({"type": "qc", "worker_id": "qc-01", "lease_seconds": 60})
            service.complete_job(
                "qc_episode_uuid",
                {
                    "result": {
                        "passed": False,
                        "segments": [
                            {"segment_id": "segment-a", "start_frame": 1, "end_frame": 2},
                            {"segment_id": "segment-b", "start_frame": 10, "end_frame": 11},
                        ],
                    }
                },
            )
            self.assertEqual(
                [event["action"] for event in service.store.list_nas_sync_events()],
                ["quality_take", "quality_needs_labeling"],
            )

            leased = service.lease_label_episode({"operator_id": "label-01"})
            self.assertEqual(len(leased["segments"]), 2)
            service.complete_label_episode(
                "episode_uuid",
                {
                    "result": {"operator_id": "label-01"},
                    "artifacts": [{"kind": "manual_2d", "metadata": {"scope": "episode"}}],
                },
            )
            events = service.store.list_nas_sync_events()
            self.assertEqual(
                [event["action"] for event in events],
                ["quality_take", "quality_needs_labeling"],
            )
            self.assertEqual(len(service.store.jobs_for_episode("episode_uuid", "manual_3d")), 1)

            calls = []
            worker = NasStatusSync(
                service.store,
                NasStatusSyncConfig(enabled=True),
                executor=lambda argv: calls.append(list(argv)),
            )
            while worker.run_once():
                pass
            remote_commands = [call[-1] for call in calls]
            self.assertTrue(any("result needs-labeling S001/pick_object/episode-001" in item for item in remote_commands))
            self.assertFalse(any("nas-uploader-publish --manual-2d" in item for item in remote_commands))

    def test_bad_episode_is_not_mapped_to_an_incorrect_nas_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            service.lease_job({"type": "qc", "worker_id": "qc-01", "lease_seconds": 60})
            service.complete_job(
                "qc_episode_uuid",
                {"result": {"passed": False, "bad_episode": True, "result_type": "bad_episode"}},
            )
            self.assertEqual(
                [event["action"] for event in service.store.list_nas_sync_events()],
                ["quality_take"],
            )
            episode = service.store.get_episode("episode_uuid") or {}
            self.assertEqual(episode.get("metadata", {}).get("nas_sync_status"), "unsupported_bad_episode")

    def test_failed_command_is_left_pending_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(Path(tmp))
            service.lease_job({"type": "qc", "worker_id": "qc-01", "lease_seconds": 60})
            service.complete_job("qc_episode_uuid", {"result": {"passed": True}})

            def fail(_argv):
                raise RuntimeError("temporary SSH failure")

            worker = NasStatusSync(
                service.store,
                NasStatusSyncConfig(enabled=True, retry_base_seconds=1, retry_max_seconds=2),
                executor=fail,
            )
            self.assertTrue(worker.run_once())
            self.assertFalse(worker.run_once())
            take_event, result_event = service.store.list_nas_sync_events()
            self.assertEqual(take_event["status"], "pending")
            self.assertEqual(take_event["attempt"], 1)
            self.assertIn("temporary SSH failure", take_event["last_error"])
            self.assertEqual(result_event["status"], "pending")
            self.assertEqual(result_event["attempt"], 0)


if __name__ == "__main__":
    unittest.main()
