from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from task_backend.job_service import JobService
from task_backend.publisher_bridge import (
    ManualPublisherBridge,
    ManualPublisherBridgeConfig,
    MaterializerError,
    PublisherBridge,
    PublisherBridgeConfig,
)
from task_backend.workflow_store import WorkflowStore


RESULT_SHA256 = "a" * 64


class FakePublisher:
    def __init__(self, statuses: list[dict]):
        self.statuses = list(statuses)
        self.status_calls: list[str] = []
        self.published: list[str] = []
        self.manual_published: list[str] = []

    def status(self, episode_id: str) -> dict:
        self.status_calls.append(episode_id)
        if len(self.statuses) > 1:
            return dict(self.statuses.pop(0))
        return dict(self.statuses[0])

    def publish(self, episode_id: str) -> None:
        self.published.append(episode_id)

    def publish_manual(self, episode_id: str) -> None:
        self.manual_published.append(episode_id)


class FakeMaterializer:
    def run(self, *, episode_dir: Path, generation: int, result_manifest_sha256: str, cameras: list[str]) -> dict:
        pose_dir = episode_dir / "optimized_pose"
        frames = sorted(int(path.stem) for path in pose_dir.glob("*.npy") if path.stem.isdigit())
        output_dir = episode_dir / "mano" / "episode"
        output_dir.mkdir(parents=True, exist_ok=True)
        joints = np.zeros((len(frames), 2, 21, 3), dtype=np.float32)
        np.save(output_dir / "joints_3d.npy", joints, allow_pickle=False)
        (output_dir / "mano_episode.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "orbbec_mano_3d_episode",
                    "frames": frames,
                    "cameras": cameras,
                    "joints_3d_file": "joints_3d.npy",
                    "coordinate_system": "episode_world",
                    "source": {
                        "kind": "optimized_pose",
                        "shape": [2, 99],
                        "generation": generation,
                        "result_manifest_sha256": result_manifest_sha256,
                    },
                    "shape_source": str(episode_dir.parents[1] / "shape.npy"),
                    "scale": 1.0,
                    "converter": "optimized_pose_to_mano_v1",
                }
            ),
            encoding="utf-8",
        )
        return {
            "ok": True,
            "reused": False,
            "frames": frames,
            "joints_3d_shape": list(joints.shape),
        }


class FailingMaterializer:
    def run(self, *, episode_dir: Path, generation: int, result_manifest_sha256: str, cameras: list[str]) -> dict:
        raise MaterializerError("converter error", retryable=False)


class PublisherBridgeTest(unittest.TestCase):
    def test_manual_bridge_runs_two_episode_slots_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkflowStore(root / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(root)})
            for index in (1, 2):
                episode_id = f"episode_{index}"
                episode_dir = root / "S001" / "pick_object" / episode_id
                pose_dir = episode_dir / "optimized_pose"
                pose_dir.mkdir(parents=True)
                np.save(pose_dir / "00000.npy", np.zeros((2, 99), dtype=np.float32), allow_pickle=False)
                store.create_or_update_episode(
                    episode_id=episode_id,
                    subject_id="S001",
                    task_name="pick_object",
                    status="manual_3d_pending",
                    episode_uri=f"nas://ego/S001/pick_object/{episode_id}",
                    cameras=["00"],
                )
                store.create_segment(
                    segment_id=f"segment_{index}",
                    episode_id=episode_id,
                    start_frame=0,
                    end_frame=0,
                    status="mano_queued",
                )
                store.create_job(
                    job_id=f"manual_3d_{episode_id}",
                    job_type="manual_3d",
                    episode_id=episode_id,
                    payload={
                        "episode_id": episode_id,
                        "episode_uri": f"nas://ego/S001/pick_object/{episode_id}",
                        "cameras": ["00"],
                        "frames": [0],
                        "scope": "episode",
                    },
                )

            status = {
                "found": True,
                "state": "relabeled",
                "generation": 4,
                "result_manifest_sha256": RESULT_SHA256,
            }
            publisher = FakePublisher([status] * 6)
            bridge = ManualPublisherBridge(
                service,
                ManualPublisherBridgeConfig(
                    enabled=True,
                    max_inflight=2,
                    poll_seconds=0.001,
                    lease_seconds=10,
                    heartbeat_seconds=1,
                    completion_poll_count=3,
                    mano_python=Path(sys.executable),
                    mano_toolkit_root=root,
                    mano_model_dir=root,
                ),
                publisher_client=publisher,
                materializer=FakeMaterializer(),
                hostname="capture-host",
            )
            bridge.start()
            self.assertEqual(len(bridge.threads), 2)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if all(
                    store.jobs_for_episode(f"episode_{index}", "manual_3d")[0]["status"] == "succeeded"
                    for index in (1, 2)
                ):
                    break
                time.sleep(0.01)
            bridge.stop(timeout=2.0)

            self.assertEqual(
                sorted(publisher.manual_published),
                ["S001/pick_object/episode_1", "S001/pick_object/episode_2"],
            )
            self.assertTrue(
                all(
                    store.jobs_for_episode(f"episode_{index}", "manual_3d")[0]["status"] == "succeeded"
                    for index in (1, 2)
                )
            )

    def test_manual_bridge_publishes_one_episode_and_assumes_complete_after_three_polls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode_dir = root / "S001" / "pick_object" / "episode1"
            pose_dir = episode_dir / "optimized_pose"
            pose_dir.mkdir(parents=True)
            np.save(pose_dir / "00002.npy", np.zeros((2, 99), dtype=np.float32), allow_pickle=False)

            store = WorkflowStore(root / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(root)})
            store.create_or_update_episode(
                episode_id="episode_uuid",
                subject_id="S001",
                task_name="pick_object",
                status="mano_optimized",
                episode_uri="nas://ego/S001/pick_object/episode1",
                cameras=["00"],
            )
            store.create_job(
                job_id="qc_episode_uuid",
                job_type="qc",
                episode_id="episode_uuid",
                payload={"frames": [2]},
            )
            service.lease_job({"type": "qc", "worker_id": "qc"})
            service.complete_job(
                "qc_episode_uuid",
                {"result": {"passed": False, "segments": [{"start_frame": 2, "end_frame": 2}]}},
            )
            service.lease_label_episode({"operator_id": "labeler"})
            service.complete_label_episode(
                "episode_uuid",
                {
                    "result": {"operator_id": "labeler"},
                    "artifacts": [{"kind": "manual_2d", "metadata": {"scope": "episode"}}],
                },
            )

            status = {
                "episode_id": "S001/pick_object/episode1",
                "found": True,
                "state": "relabeled",
                "generation": 7,
                "result_manifest_sha256": RESULT_SHA256,
            }
            publisher = FakePublisher([status, status, status])
            bridge = ManualPublisherBridge(
                service,
                ManualPublisherBridgeConfig(
                    poll_seconds=0.001,
                    lease_seconds=10,
                    heartbeat_seconds=1,
                    completion_poll_count=3,
                ),
                publisher_client=publisher,
                materializer=FakeMaterializer(),
                hostname="capture-host",
            )

            self.assertTrue(bridge.process_once(1))

            manual_job = store.jobs_for_episode("episode_uuid", "manual_3d")[0]
            self.assertEqual(manual_job["status"], "succeeded")
            self.assertEqual(manual_job["result"]["completion_poll_count"], 3)
            self.assertTrue(manual_job["result"]["temporary_completion_policy"])
            self.assertEqual(publisher.manual_published, ["S001/pick_object/episode1"])
            self.assertEqual(len(publisher.status_calls), 3)
            self.assertTrue(all(segment["status"] == "mano_succeeded" for segment in store.segments_for_episode("episode_uuid")))
            self.assertEqual((store.get_episode("episode_uuid") or {})["status"], "finalized")

    def test_config_preserves_virtualenv_python_launcher_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / "venv-python"
            launcher.symlink_to(Path(sys.executable))
            config = PublisherBridgeConfig(
                enabled=True,
                max_inflight=1,
                lease_seconds=10,
                heartbeat_seconds=1,
                mano_python=launcher,
                mano_toolkit_root=root,
                mano_model_dir=root,
            )

            config.validate()

            self.assertEqual(config.mano_python, launcher.absolute())

    def test_bridge_publishes_materializes_and_creates_qc_from_actual_pose_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode_dir = root / "S001" / "pick_object" / "episode1"
            pose_dir = episode_dir / "optimized_pose"
            pose_dir.mkdir(parents=True)
            np.save(pose_dir / "00002.npy", np.zeros((2, 99), dtype=np.float32), allow_pickle=False)
            np.save(pose_dir / "00005.npy", np.zeros((2, 99), dtype=np.float32), allow_pickle=False)

            store = WorkflowStore(root / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(root)})
            store.create_or_update_episode(
                episode_id="episode_uuid",
                subject_id="S001",
                task_name="pick_object",
                status="uploaded",
                episode_uri="nas://ego/S001/pick_object/episode1",
                cameras=["00", "01"],
            )
            service.push_auto_label({"episode_id": "episode_uuid", "pushed_by": "test"})

            publisher = FakePublisher(
                [
                    {"episode_id": "S001/pick_object/episode1", "found": False},
                    {
                        "episode_id": "S001/pick_object/episode1",
                        "found": True,
                        "state": "labeled",
                        "generation": 3,
                        "result_manifest_sha256": RESULT_SHA256,
                    },
                ]
            )
            config = PublisherBridgeConfig(
                poll_seconds=0.01,
                lease_seconds=10,
                heartbeat_seconds=1,
            )
            bridge = PublisherBridge(
                service,
                config,
                publisher_client=publisher,
                materializer=FakeMaterializer(),
                hostname="capture-host",
            )

            self.assertTrue(bridge.process_once(1))

            auto_job = store.jobs_for_episode("episode_uuid", "auto_label")[0]
            self.assertEqual(auto_job["status"], "succeeded")
            self.assertEqual(auto_job["result"]["frames"], [2, 5])
            self.assertEqual(auto_job["result"]["worker_id"], "publisher_bridge:capture-host:slot-01")
            kinds = {artifact["kind"] for artifact in store.artifacts_for_episode("episode_uuid")}
            self.assertEqual(kinds, {"optimized_pose", "mano_episode"})
            qc_job = store.jobs_for_episode("episode_uuid", "qc")[0]
            self.assertEqual(qc_job["payload"]["frames"], [2, 5])
            self.assertEqual(publisher.published, ["S001/pick_object/episode1"])

            # A retry after the job row reached succeeded repairs downstream
            # state idempotently and does not duplicate artifacts or QC.
            service.complete_job(
                auto_job["job_id"],
                {
                    "result": auto_job["result"],
                    "artifacts": [
                        {"kind": "optimized_pose", "metadata": {"generation": 3, "frame_shape": [2, 99]}},
                        {"kind": "mano_episode", "metadata": {"generation": 3, "coordinate_system": "episode_world"}},
                    ],
                },
            )
            self.assertEqual(len(store.artifacts_for_episode("episode_uuid")), 2)
            self.assertEqual(len(store.jobs_for_episode("episode_uuid", "qc")), 1)

    def test_joint3d_retry_runs_directly_without_queueing_or_republishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pose_dir = root / "S001" / "pick_object" / "episode1" / "optimized_pose"
            pose_dir.mkdir(parents=True)
            np.save(pose_dir / "00002.npy", np.zeros((2, 99), dtype=np.float32), allow_pickle=False)

            store = WorkflowStore(root / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(root)})
            store.create_or_update_episode(
                episode_id="episode_uuid",
                subject_id="S001",
                task_name="pick_object",
                status="uploaded",
                episode_uri="nas://ego/S001/pick_object/episode1",
                cameras=["00"],
            )
            service.push_auto_label({"episode_id": "episode_uuid", "pushed_by": "test"})
            publisher = FakePublisher(
                [
                    {
                        "episode_id": "S001/pick_object/episode1",
                        "found": True,
                        "state": "labeled",
                        "generation": 3,
                        "result_manifest_sha256": RESULT_SHA256,
                    }
                ]
            )
            bridge = PublisherBridge(
                service,
                PublisherBridgeConfig(poll_seconds=0.01, lease_seconds=10, heartbeat_seconds=1),
                publisher_client=publisher,
                materializer=FailingMaterializer(),
                hostname="capture-host",
            )

            self.assertTrue(bridge.process_once(1))
            failed_job = service.upload_status("episode_uuid")["jobs"][0]
            self.assertEqual(failed_job["status"], "failed")
            self.assertTrue(failed_job["can_retry_joint3d"])

            bridge.materializer = FakeMaterializer()
            retry = bridge.retry_materialization_once("episode_uuid", {"operator_id": "admin"})

            self.assertTrue(retry["completed"])
            retried_job = store.jobs_for_episode("episode_uuid", "auto_label")[0]
            self.assertEqual(retried_job["status"], "succeeded")
            self.assertEqual(publisher.published, [])
            self.assertEqual(len(publisher.status_calls), 1)
            self.assertEqual(len(store.jobs_for_episode("episode_uuid", "qc")), 1)

    def test_backend_shutdown_releases_held_job_without_canceling_publisher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode_dir = root / "S001" / "pick_object" / "episode1"
            episode_dir.mkdir(parents=True)
            store = WorkflowStore(root / "workflow.sqlite3")
            service = JobService(store, nas_mounts={"nas://ego": str(root)})
            store.create_or_update_episode(
                episode_id="episode_uuid",
                subject_id="S001",
                task_name="pick_object",
                status="uploaded",
                episode_uri="nas://ego/S001/pick_object/episode1",
            )
            service.push_auto_label({"episode_id": "episode_uuid", "pushed_by": "test"})
            publisher = FakePublisher(
                [
                    {
                        "episode_id": "S001/pick_object/episode1",
                        "found": True,
                        "state": "cleaned",
                        "generation": 0,
                        "result_manifest_sha256": "",
                    }
                ]
            )
            config = PublisherBridgeConfig(
                enabled=True,
                max_inflight=1,
                poll_seconds=0.02,
                lease_seconds=10,
                heartbeat_seconds=1,
                mano_python=Path(sys.executable),
                mano_toolkit_root=root,
                mano_model_dir=root,
            )
            bridge = PublisherBridge(
                service,
                config,
                publisher_client=publisher,
                materializer=FakeMaterializer(),
                hostname="capture-host",
            )

            bridge.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                job = store.jobs_for_episode("episode_uuid", "auto_label")[0]
                if job["status"] in {"leased", "running"}:
                    break
                time.sleep(0.01)
            bridge.stop(timeout=2.0)

            job = store.jobs_for_episode("episode_uuid", "auto_label")[0]
            self.assertEqual(job["status"], "queued")
            self.assertFalse(job["lease_owner"])
            self.assertEqual(publisher.published, [])


if __name__ == "__main__":
    unittest.main()
