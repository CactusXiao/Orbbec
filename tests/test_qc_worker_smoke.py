from __future__ import annotations

import json
import numpy as np
import tempfile
import unittest
from pathlib import Path

from src.qc.config import load_qc_config
from src.qc.media import prepare_qc_media
from src.qc.state_store import QcProgress, QcStateStore, first_sample_after, normalize_ranges


class QcWorkerSmokeTest(unittest.TestCase):
    def test_software_mesh_renderer_fallback_draws_surface(self) -> None:
        try:
            from src.qc.mesh_renderer import CameraMeshRenderer
        except ModuleNotFoundError as exc:
            self.skipTest(f"mesh renderer dependency unavailable: {exc}")
        renderer = CameraMeshRenderer(
            camera="00",
            width=32,
            height=32,
            intrinsics=np.asarray([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            render_factor=1.0,
        )
        renderer.close()
        vertices = np.asarray([[-0.08, -0.08, 1.0], [0.08, -0.08, 1.0], [0.0, 0.08, 1.0]], dtype=np.float32)
        rendered = renderer.composite(
            np.zeros((32, 32, 3), dtype=np.uint8),
            {0: vertices, 1: vertices + np.asarray([0.02, 0.0, 0.0], dtype=np.float32)},
            {0: np.asarray([[0, 1, 2]], dtype=np.int32), 1: np.asarray([[0, 1, 2]], dtype=np.int32)},
        )

        self.assertGreater(int(rendered.sum()), 0)

    def test_normalize_bad_frame_ranges_merges_overlap_touching_and_small_gaps(self) -> None:
        ranges = normalize_ranges([(20, 25), (10, 12), (13, 14), (30, 31)], max_gap_frames=5)

        self.assertEqual(ranges, [(10, 14), (20, 31)])

    def test_first_sample_after_uses_sampling_sequence(self) -> None:
        self.assertEqual(first_sample_after(147, first_frame=0, last_frame=300, sample_interval=10), 150)
        self.assertEqual(first_sample_after(150, first_frame=0, last_frame=300, sample_interval=10), 160)
        self.assertEqual(first_sample_after(297, first_frame=0, last_frame=300, sample_interval=10), 300)
        self.assertEqual(first_sample_after(300, first_frame=0, last_frame=300, sample_interval=10), 301)

    def test_state_store_round_trips_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = QcStateStore(Path(tmp))
            progress = QcProgress(
                task_name="pick_object",
                episode_id="episode_001",
                job_id="qc_episode_001",
                worker_machine_id="worker_a",
                lease_until="2099-01-01T00:00:00Z",
                sample_interval=10,
                current_frame=20,
                frames=[0, 10, 20],
                bad_frame_ranges=[(12, 15)],
                playback_complete=True,
            )

            store.save(progress)
            loaded = store.list_progress(worker_machine_id="worker_a")

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].episode_id, "episode_001")
            self.assertEqual(loaded[0].bad_frame_ranges, [(12, 15)])
            self.assertTrue(loaded[0].playback_complete)

    def test_legacy_sampling_completion_migrates_to_video_completion(self) -> None:
        progress = QcProgress.from_dict(
            {
                "schema_version": 1,
                "task_name": "pick_object",
                "episode_id": "episode_001",
                "job_id": "qc_episode_001",
                "worker_machine_id": "worker_a",
                "lease_until": "2099-01-01T00:00:00Z",
                "sample_interval": 10,
                "current_frame": 21,
                "frames": [0, 10, 20],
            }
        )

        self.assertTrue(progress.playback_complete)
        self.assertEqual(progress.current_frame, 20)
        self.assertEqual(progress.schema_version, 2)

    def test_qc_config_loads_launch_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "qc_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "backend_url": "http://127.0.0.1:9999",
                        "sample_interval": 7,
                        "default_lease_minutes": 11,
                        "crash_lease_extension_minutes": 13,
                        "tmp_dir": "tmp_cache",
                        "state_dir": "state",
                        "worker_machine_id": "qc_machine_a",
                        "operator_id": "qc_operator_a",
                        "range_merge_gap_frames": 2,
                        "request_timeout_seconds": 4.5,
                        "playback_fps": 24.0,
                        "mesh_renderer_python": "/opt/mano/bin/python",
                        "mano_toolkit_root": "/opt/mano/toolkit",
                        "mano_model_dir": "/opt/mano/models",
                        "mesh_render_factor": 1.5,
                        "nas_mounts": {"nas://ego": "/mnt/nas"},
                    }
                ),
                encoding="utf-8",
            )

            config = load_qc_config(config_path=config_path)

            self.assertEqual(config.backend_url, "http://127.0.0.1:9999")
            self.assertEqual(config.sample_interval, 7)
            self.assertEqual(config.default_lease_minutes, 11)
            self.assertEqual(config.crash_lease_extension_minutes, 13)
            self.assertEqual(config.tmp_dir, (tmp_path / "tmp_cache").resolve())
            self.assertEqual(config.state_dir, (tmp_path / "state").resolve())
            self.assertEqual(config.worker_machine_id, "qc_machine_a")
            self.assertEqual(config.operator_id, "qc_operator_a")
            self.assertEqual(config.range_merge_gap_frames, 2)
            self.assertEqual(config.request_timeout_seconds, 4.5)
            self.assertEqual(config.playback_fps, 24.0)
            self.assertEqual(config.mesh_renderer_python, "/opt/mano/bin/python")
            self.assertEqual(config.mano_toolkit_root, Path("/opt/mano/toolkit"))
            self.assertEqual(config.mano_model_dir, Path("/opt/mano/models"))
            self.assertEqual(config.mesh_render_factor, 1.5)
            self.assertEqual(config.nas_mounts, {"nas://ego": "/mnt/nas"})

    def test_video_progress_requires_playback_completion(self) -> None:
        progress = QcProgress(
            task_name="pick_object",
            episode_id="episode_001",
            job_id="qc_episode_001",
            worker_machine_id="worker_a",
            lease_until="2099-01-01T00:00:00Z",
            sample_interval=10,
            current_frame=20,
            frames=[0, 10, 20],
        )

        self.assertFalse(progress.is_complete)
        progress.playback_complete = True
        self.assertTrue(progress.is_complete)

    def test_qc_config_uses_cwd_for_default_local_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            config = load_qc_config(cwd=tmp_path)

            self.assertEqual(config.tmp_dir, (tmp_path / "tmp").resolve())
            self.assertEqual(config.state_dir, (tmp_path / "qc_state").resolve())

    def test_prepare_qc_media_uses_nas_episode_root_for_calibration_and_mano(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            episode_dir = nas_root / "S001" / "pick_object" / "episode_001"
            mano_dir = episode_dir / "mano" / "episode"
            (episode_dir / "00" / "RGB").mkdir(parents=True)
            mano_dir.mkdir(parents=True)
            (episode_dir / "00" / "RGB" / "00000.png").write_bytes(b"placeholder")
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
            np.save(mano_dir / "joints_3d.npy", joints)
            (mano_dir / "mano_episode.json").write_text(
                json.dumps({"schema_version": 1, "kind": "orbbec_mano_3d_episode", "frames": [0], "joints_3d_file": "joints_3d.npy"}),
                encoding="utf-8",
            )

            media = prepare_qc_media(
                {
                    "episode_id": "episode_001",
                    "subject_id": "S001",
                    "task_name": "pick_object",
                    "episode_uri": "nas://ego/S001/pick_object/episode_001",
                    "cameras": ["00"],
                    "frames": [0],
                },
                mounts={"nas://ego": str(nas_root)},
                tmp_dir=tmp_path / "qc_tmp",
            )

            self.assertEqual(media.episode_dir, episode_dir.resolve())
            self.assertEqual(media.mano_dir, mano_dir.resolve())
            self.assertEqual(media.frame_path("00", 0), (episode_dir / "00" / "RGB" / "00000.png").resolve())

    def test_prepare_qc_media_fails_when_collection_calibration_is_missing_at_episode_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas_root = tmp_path / "nas"
            episode_dir = nas_root / "S001" / "pick_object" / "episode_001"
            mano_dir = episode_dir / "mano" / "episode"
            (episode_dir / "00" / "RGB").mkdir(parents=True)
            mano_dir.mkdir(parents=True)
            (episode_dir / "00" / "RGB" / "00000.png").write_bytes(b"placeholder")
            (mano_dir / "mano_episode.json").write_text(
                json.dumps({"schema_version": 1, "kind": "orbbec_mano_3d_episode", "frames": [0], "joints_3d_file": "joints_3d.npy"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "Collection calibration missing at episode root"):
                prepare_qc_media(
                    {
                        "episode_id": "episode_001",
                        "subject_id": "S001",
                        "task_name": "pick_object",
                        "episode_uri": "nas://ego/S001/pick_object/episode_001",
                        "cameras": ["00"],
                        "frames": [0],
                    },
                    mounts={"nas://ego": str(nas_root)},
                    tmp_dir=tmp_path / "qc_tmp",
                )


if __name__ == "__main__":
    unittest.main()
