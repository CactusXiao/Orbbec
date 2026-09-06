"""Regression coverage for the per-camera visibility sequences returned by NAS."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from label.app import LabelPage
from label.storage import load_joint_visibility


class LabelVisibilitySequenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.episode = Path(self.temp.name)
        self.root = self.episode / "joints_vis"
        (self.root / "00").mkdir(parents=True)

    def save_sequence(self, values):
        np.save(self.root / "00" / "joints_vis.npy", values)

    def test_reads_exact_frame_and_camera_without_reordering_joints(self):
        values = np.zeros((3, 2, 21), dtype=np.uint8)
        values[0, 0, 13] = 1
        values[1, 1, 4] = 1
        self.save_sequence(values)
        (self.root / "01").mkdir()
        np.save(self.root / "01" / "joints_vis.npy", 1 - values)
        for camera in ("00", "01"):
            for frame in range(3):
                expected = values[frame] if camera == "00" else 1 - values[frame]
                np.testing.assert_array_equal(load_joint_visibility(self.root, camera, frame), expected)

    def test_per_frame_files_keep_precedence_and_legacy_support(self):
        self.save_sequence(np.ones((2, 2, 21), dtype=np.uint8))
        np.save(self.root / "00" / "00001.npy", np.zeros((2, 21), dtype=bool))
        self.assertFalse(np.any(load_joint_visibility(self.root, "00", 1)))
        (self.root / "00" / "joints_vis.npy").unlink()
        self.assertFalse(np.any(load_joint_visibility(self.root, "00", 1)))
        self.assertIsNone(load_joint_visibility(self.root, "00", 0))

    def test_sequence_rejects_out_of_range_frames_and_wrong_shape(self):
        self.save_sequence(np.ones((2, 2, 21), dtype=np.uint8))
        for frame in (-1, 2):
            with self.assertRaisesRegex(ValueError, "outside sequence"):
                load_joint_visibility(self.root, "00", frame)
        self.save_sequence(np.ones((2, 21), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "sequence must have shape"):
            load_joint_visibility(self.root, "00", 0)

    def test_singleton_dimension_and_nonfinite_values(self):
        values = np.ones((1, 2, 21, 1), dtype=float)
        values[0, 0, :4, 0] = [0, -1, np.nan, np.inf]
        self.save_sequence(values)
        visible = load_joint_visibility(self.root, "00", 0)
        self.assertEqual(visible[0][:5], [False, False, False, False, True])

    def test_original_view_applies_packed_mask_to_projected_joints(self):
        values = np.zeros((2, 2, 21), dtype=np.uint8)
        values[1, 0, 13] = 1
        values[1, 1, 4] = 1
        self.save_sequence(values)
        episode = self.episode
        points = [[(100.0 + j, 200.0 + j) for j in range(21)] for _ in range(2)]
        projected_visible = [[True] * 21 for _ in range(2)]
        projected_visible[1][4] = False

        class Task:
            key = "visibility-regression"
            mano_episode_dir = "mano/episode"

            def episode_dir(self):
                return episode

        class Runtime:
            def project_mano_frame(self, **kwargs):
                return points, projected_visible

        class Page:
            _active_task = Task()
            _mano_projection_errors = {}

            def _mano_runtime_instance(self):
                return Runtime()

        result_points, visible = LabelPage._build_mano_3d_view_state(Page(), 1, "00")
        self.assertEqual(result_points, points)
        self.assertTrue(visible[0][13])
        self.assertEqual(np.count_nonzero(visible), 1)


if __name__ == "__main__":
    unittest.main()
