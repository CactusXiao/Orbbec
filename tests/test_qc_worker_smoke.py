from __future__ import annotations

import json
import numpy as np
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.qc.config import load_qc_config
from src.qc.media import prepare_qc_media
from src.qc.state_store import QcProgress, QcStateStore, first_sample_after, normalize_ranges


class QcWorkerSmokeTest(unittest.TestCase):
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
            )

            store.save(progress)
            loaded = store.list_progress(worker_machine_id="worker_a")

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].episode_id, "episode_001")
            self.assertEqual(loaded[0].bad_frame_ranges, [(12, 15)])

    def test_qc_config_uses_shared_orbbec_nas_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / ".env").write_text('ORBBEC_NAS_MOUNTS_JSON={"nas://ego":"/mnt/nas"}\n', encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                config = load_qc_config(cwd=tmp_path)

            self.assertEqual(config.nas_mounts, {"nas://ego": "/mnt/nas"})

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
