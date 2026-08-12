from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from task_backend.job_service import FINAL_3D_SOURCES_REL_PATH, JobService
from task_backend.server import render_workflow_stage_page
from task_backend.nas_uploader import NasUploadConfig, NasUploader
from task_backend.workflow_models import WorkflowError
from task_backend.workflow_store import WorkflowStore


class WorkflowStoreSmokeTest(unittest.TestCase):
    def test_nas_uploader_keeps_auto_label_as_manual_push(self) -> None:
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
            service = JobService(store, nas_mounts={nas_prefix: str(nas_root)})
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
                }
            )

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
            self.assertFalse(any(job["type"] == "auto_label" for job in status_after["jobs"]))
            self.assertEqual(store.jobs_for_episode("reservation_001", "auto_label"), [])

            pushed = service.push_auto_label({"episode_id": "reservation_001", "pushed_by": "smoke"})
            self.assertEqual(pushed["pushed"], 1)
            self.assertEqual(pushed["created_jobs"], 1)
            auto_jobs = store.jobs_for_episode("reservation_001", "auto_label")
            self.assertEqual(len(auto_jobs), 1)
            self.assertEqual(auto_jobs[0]["payload"]["scope"], "episode")
            self.assertEqual(auto_jobs[0]["payload"]["frames"], [1, 2])
            self.assertEqual(auto_jobs[0]["payload"]["cameras"], ["00"])
            self.assertIn("/episodes/reservation_001", render_workflow_stage_page(service.workflow_stage("auto_label")))

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
                    "artifacts": [{"kind": "pred_2d", "metadata": {"mock": True}}],
                },
            )
            self.assertEqual(store.get_episode("episode_pass")["status"], "auto_labeled")  # type: ignore[index]
            self.assertEqual(store.jobs_for_episode("episode_pass", "qc"), [])

            mano_jobs = store.jobs_for_episode("episode_pass", "mano_opt")
            self.assertEqual(len(mano_jobs), 1)
            self.assertEqual(mano_jobs[0]["payload"]["scope"], "episode")
            service.complete_job(
                mano_jobs[0]["job_id"],
                {
                    "result": {"ok": True},
                    "artifacts": [{"kind": "mano_episode", "metadata": {"mock": True}}],
                },
            )
            self.assertEqual(store.get_episode("episode_pass")["status"], "mano_optimized")  # type: ignore[index]

            qc_jobs = store.jobs_for_episode("episode_pass", "qc")
            self.assertEqual(len(qc_jobs), 1)
            service.complete_job(
                qc_jobs[0]["job_id"],
                {
                    "result": {"passed": True, "score": 0.99},
                    "artifacts": [{"kind": "qc_report", "metadata": {"passed": True}}],
                },
            )
            self.assertEqual(store.get_episode("episode_pass")["status"], "finalized")  # type: ignore[index]
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
                        "segments": [
                            {"start_frame": 10, "end_frame": 12, "reason": "low_confidence"},
                            {"start_frame": 20, "end_frame": 21, "reason": "temporal_jump"},
                        ],
                    }
                },
            )
            self.assertEqual(store.get_episode("episode_fail")["status"], "manual_correction_pending")  # type: ignore[index]
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
        mano_job = store.jobs_for_episode(episode_id, "mano_opt")[0]
        service.complete_job(mano_job["job_id"], {"result": {"ok": True}})


if __name__ == "__main__":
    unittest.main()
