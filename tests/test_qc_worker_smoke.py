from __future__ import annotations

import json
import numpy as np
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from label.video_frames import _run_ffmpeg_select
from mano.joint_order import MANO_HAND_ORDER, SMPLX_MANO_JOINT_NAMES
from src.qc.config import load_qc_config
from src.qc.media import (
    QcEpisodeMedia,
    _count_cached,
    _load_reference_to_ego_frames,
    _qc_view_cameras,
    prepare_qc_media,
)
from src.qc.report import build_qc_result, write_ego_pose_qc_report
from src.qc.playback import playback_target_position
from src.qc.state_store import QcProgress, QcStateStore, first_sample_after, normalize_ranges


class QcWorkerSmokeTest(unittest.TestCase):
    def test_qc_rejects_non_smplx_mano_joint_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "S001" / "task" / "episode_001"
            mano_dir = episode_dir / "mano" / "episode"
            (episode_dir / "00" / "RGB").mkdir(parents=True)
            mano_dir.mkdir(parents=True)
            (episode_dir / "camera_params.json").write_text("{}", encoding="utf-8")
            (episode_dir / "extrinsics.json").write_text("{}", encoding="utf-8")
            np.save(mano_dir / "joints_3d.npy", np.zeros((1, 2, 21, 3), dtype=np.float32))
            (mano_dir / "mano_episode.json").write_text(
                json.dumps(
                    {
                        "frames": [0],
                        "joints_3d_file": "joints_3d.npy",
                        "hand_order": list(MANO_HAND_ORDER),
                        "joint_order": list(reversed(SMPLX_MANO_JOINT_NAMES)),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "joint_order must match SMPL-X MANO"):
                prepare_qc_media(
                    {
                        "episode_id": "episode_001",
                        "subject_id": "S001",
                        "task_name": "task",
                        "episode_uri": "nas://ego/S001/task/episode_001",
                        "cameras": ["00"],
                        "frames": [0],
                    },
                    mounts={"nas://ego": str(Path(tmp))},
                    tmp_dir=Path(tmp) / "qc_tmp",
                )

    def test_wall_clock_playback_skips_display_frames_instead_of_slowing_time(self) -> None:
        self.assertEqual(
            playback_target_position(
                start_position=10,
                elapsed_seconds=0.1,
                fps=30.0,
                last_position=100,
            ),
            13,
        )
        self.assertEqual(
            playback_target_position(
                start_position=98,
                elapsed_seconds=0.2,
                fps=30.0,
                last_position=100,
            ),
            100,
        )

    def test_qc_jpeg_decode_uses_compact_contiguous_select(self) -> None:
        with patch("label.video_frames.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stderr = ""
            run.return_value.stdout = ""
            _run_ffmpeg_select(
                Path("rgb.h265"),
                [10, 11, 12],
                Path("%06d.jpg"),
                jpeg_quality=2,
                ffmpeg_threads=4,
            )

        command = run.call_args.args[0]
        self.assertIn("select=between(n\\,10\\,12)", command)
        self.assertIn("-q:v", command)
        self.assertIn("-threads", command)

    def test_qc_cache_counts_jpeg_and_legacy_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera_dir = root / "00"
            camera_dir.mkdir()
            (camera_dir / "00000.jpg").write_bytes(b"jpg")
            (camera_dir / "00001.png").write_bytes(b"png")

            self.assertEqual(_count_cached(root, "00", [0, 1, 2]), 2)

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
                ego_bad_frame_ranges=[(16, 18)],
                playback_complete=True,
            )

            store.save(progress)
            loaded = store.list_progress(worker_machine_id="worker_a")

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].episode_id, "episode_001")
            self.assertEqual(loaded[0].bad_frame_ranges, [(12, 15)])
            self.assertEqual(loaded[0].ego_bad_frame_ranges, [(16, 18)])
            self.assertTrue(loaded[0].playback_complete)

    def test_ego_pose_ranges_are_recorded_without_failing_mano_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "episode_001"
            result = build_qc_result(
                episode_id="episode_001",
                worker_id="worker_a",
                bad_ranges=[],
            )
            path = write_ego_pose_qc_report(
                episode_dir=episode_dir,
                episode_id="episode_001",
                worker_id="worker_a",
                operator_id="operator_a",
                bad_ranges=[(10, 14)],
            )

            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertEqual(path, episode_dir / "ego" / "ego_pose_qc.json")
            self.assertEqual(report["segments"], [{"start_frame": 10, "end_frame": 14}])

    def test_pico_timestamp_mapping_preserves_duplicate_ego_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp)
            timestamp_dir = episode_dir / "ego" / "RGB"
            timestamp_dir.mkdir(parents=True)
            (timestamp_dir / "rgb.h265.timestamps.csv").write_text(
                "frame_index,ego_frame_index\n00000,0\n00001,0\n00002,1\n",
                encoding="utf-8",
            )

            self.assertEqual(_load_reference_to_ego_frames(episode_dir), {0: 0, 1: 0, 2: 1})

    def test_qc_view_cameras_selects_requested_four_views_and_pico(self) -> None:
        from label.storage import CorrectionTask

        task = CorrectionTask(
            line_no=1,
            root="/tmp",
            subject="S001",
            task="pick_object",
            episode="episode_001",
            cameras=["00", "01", "02", "03", "04", "05"],
            frames=[0],
        )

        self.assertEqual(_qc_view_cameras(task, include_ego=True), ["00", "02", "03", "05", "ego"])

    def test_qc_media_prefers_small_pico_playback_preview(self) -> None:
        from label.storage import CorrectionTask

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode_dir = root / "episode"
            cache_dir = root / "cache"
            task = CorrectionTask(
                line_no=1,
                root=str(root),
                subject="",
                task="",
                episode="episode",
                cameras=["00"],
                frames=[0],
                nas_root_path=str(episode_dir),
            )
            full = cache_dir / "mesh" / "ego" / "00000.jpg"
            preview = cache_dir / "mesh_preview" / "ego" / "00000.jpg"
            full.parent.mkdir(parents=True)
            preview.parent.mkdir(parents=True)
            full.write_bytes(b"full-resolution")
            preview.write_bytes(b"display-preview")
            media = QcEpisodeMedia(
                task=task,
                cache_dir=cache_dir,
                requires_mesh=True,
                view_cameras=("ego",),
            )

            self.assertEqual(media.frame_path("ego", 0), preview)
            self.assertTrue(media.frame_ready(0))

    def test_loads_new_ego_pose_frame_structure_and_fisheye_calibration(self) -> None:
        try:
            from src.qc.mesh_renderer import load_ego_camera, load_ego_extrinsics
        except ModuleNotFoundError as exc:
            self.skipTest(f"mesh renderer dependency unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp)
            ego_dir = episode_dir / "ego"
            ego_dir.mkdir()
            transform = np.eye(4, dtype=np.float32)
            transform[0, 3] = 0.25
            (episode_dir / "ego_pose.json").write_text(
                json.dumps(
                    {
                        "coordinate_convention": {"reference_view": "00"},
                        "frames": [
                            {
                                "frame_index": "00007",
                                "T_ego_from_reference": transform.tolist(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (ego_dir / "camera_params.json").write_text(
                json.dumps(
                    {
                        "ego": {
                            "RGB": {
                                "intrinsic": {
                                    "fx": 400.0,
                                    "fy": 401.0,
                                    "cx": 320.0,
                                    "cy": 240.0,
                                    "width": 640,
                                    "height": 480,
                                },
                                "distortion": {
                                    "modelName": "opencv_fisheye",
                                    "k1": 0.1,
                                    "k2": 0.2,
                                    "k3": 0.3,
                                    "k4": 0.4,
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            transforms = load_ego_extrinsics(episode_dir)
            intrinsic, distortion, image_size = load_ego_camera(episode_dir)

            np.testing.assert_allclose(transforms[7], transform)
            np.testing.assert_allclose(intrinsic, [[400, 0, 320], [0, 401, 240], [0, 0, 1]])
            np.testing.assert_allclose(distortion.reshape(-1), [0.1, 0.2, 0.3, 0.4])
            self.assertEqual(image_size, (640, 480))

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
                        "mesh_render_workers": 12,
                        "mesh_prefer_integrated_gpu": False,
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
            self.assertEqual(config.mesh_render_workers, 12)
            self.assertFalse(config.mesh_prefer_integrated_gpu)
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
