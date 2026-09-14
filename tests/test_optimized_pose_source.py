from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from task_backend.optimized_pose_source import OptimizedPoseSource, archive_frame_ids, load_archive_frame
from task_backend.optimized_pose_materializer import materialize, MaterializationError, MaterializationNotReady


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class OptimizedPoseSourceTest(unittest.TestCase):
    def test_sparse_unsorted_ids_override_stale_files_and_cache_refreshes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            archive = directory / "poses.npz"
            poses = np.stack([np.full((2, 99), value, dtype=np.float32) for value in (9, 2)])
            np.savez_compressed(archive, frame_ids=np.array([90, 20]), poses=poses)
            np.save(directory / "00020.npy", np.zeros((2, 99), dtype=np.float32))
            np.save(directory / "00001.npy", np.zeros((2, 99), dtype=np.float32))
            source = OptimizedPoseSource(directory)
            self.assertEqual(source.frames, [20, 90])
            np.testing.assert_array_equal(source.load(20), poses[1])
            with self.assertRaises(KeyError):
                source.load(1)
            np.testing.assert_array_equal(load_archive_frame(directory, 90), poses[0])
            np.savez_compressed(archive, frame_ids=np.array([90, 20]), poses=poses + 1)
            np.testing.assert_array_equal(load_archive_frame(directory, 90), poses[0] + 1)
            # Backend schema validation must also run with site-packages disabled.
            result = subprocess.run([sys.executable, "-S", "-c",
                "from pathlib import Path; from task_backend.optimized_pose_source import archive_frame_ids; "
                "import sys; print(archive_frame_ids(Path(sys.argv[1])))", str(archive)],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), "[90, 20]")

    def test_invalid_archive_does_not_fall_back_to_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            np.save(directory / "00000.npy", np.zeros((2, 99), dtype=np.float32))
            cases = [([0, 0], (2, 2, 99), np.float32), ([-1, 0], (2, 2, 99), np.float32),
                     ([0, 1], (1, 2, 99), np.float32), ([0, 1], (2, 2, 99), np.float64),
                     ([0.0, 1.0], (2, 2, 99), np.float32), ([], (0, 2, 99), np.float32)]
            for ids, shape, dtype in cases:
                with self.subTest(ids=ids, shape=shape, dtype=dtype):
                    np.savez(directory / "poses.npz", frame_ids=np.array(ids), poses=np.zeros(shape, dtype=dtype))
                    with self.assertRaises(ValueError):
                        OptimizedPoseSource(directory)
            np.savez(directory / "poses.npz", frame_ids=np.array([0]), poses=np.full((1, 2, 99), np.nan, dtype=np.float32))
            with self.assertRaises(ValueError):
                OptimizedPoseSource(directory).load(0)
            np.savez(directory / "poses.npz", poses=np.zeros((1, 2, 99), dtype=np.float32))
            with self.assertRaises(ValueError):
                archive_frame_ids(directory / "poses.npz")

    def test_legacy_files_keep_numeric_order_and_reject_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for frame in (12, 2):
                np.save(directory / f"{frame}.npy", np.full((2, 99), frame, dtype=np.float32))
            source = OptimizedPoseSource(directory)
            self.assertEqual(source.frames, [2, 12])
            self.assertEqual(source.load(12)[0, 0], 12)
            np.save(directory / "00002.npy", np.zeros((2, 99), dtype=np.float32))
            with self.assertRaises(ValueError):
                OptimizedPoseSource(directory)

    def test_materializer_writes_sorted_joints_and_reuses_v2_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = root / "subject/task/episode"
            pose_dir = episode / "optimized_pose"
            pose_dir.mkdir(parents=True)
            np.save(episode.parents[1] / "shape.npy", np.zeros(10, dtype=np.float32))
            poses = np.stack([np.full((2, 99), value, dtype=np.float32) for value in (9, 2)])
            np.savez_compressed(pose_dir / "poses.npz", frame_ids=np.array([90, 20]), poses=poses)
            fake = SimpleNamespace(build_mano_layers=lambda _: {}, MANO_HAND_ORDER=["left", "right"],
                SMPLX_MANO_JOINT_NAMES=[], mano_outputs_from_pose=lambda pose, *_: {
                    hand: {"joints": FakeTensor(np.full((1, 21, 3), pose[hand, 0], dtype=np.float32))}
                    for hand in (0, 1)})
            kwargs = dict(episode_dir=episode, toolkit_root=root, mano_model_dir=root,
                          default_shape_path=None, generation=1, result_manifest_sha256="a" * 64, cameras=["00"])
            with patch("task_backend.optimized_pose_materializer._load_mano_module", return_value=fake) as loader:
                result = materialize(**kwargs)
                self.assertEqual(result["frames"], [20, 90])
                self.assertFalse(result["reused"])
                joints = np.load(episode / "mano/episode/joints_3d.npy")
                np.testing.assert_array_equal(joints[:, 0, 0, 0], [2, 9])
                self.assertTrue(materialize(**kwargs)["reused"])
                self.assertEqual(loader.call_count, 1)
                marker = episode / "mano/episode/mano_episode.json"
                meta = json.loads(marker.read_text())
                meta["converter"] = "optimized_pose_to_mano_v1"
                marker.write_text(json.dumps(meta))
                self.assertFalse(materialize(**kwargs)["reused"])
                self.assertEqual(loader.call_count, 2)
            kwargs["generation"] = 2
            (pose_dir / "poses.npz").write_bytes(b"partial upload")
            with self.assertRaises(MaterializationNotReady):
                materialize(**kwargs)
            np.savez(pose_dir / "poses.npz", frame_ids=np.array([1, 1]), poses=poses)
            with self.assertRaises(MaterializationError):
                materialize(**kwargs)


if __name__ == "__main__":
    unittest.main()
