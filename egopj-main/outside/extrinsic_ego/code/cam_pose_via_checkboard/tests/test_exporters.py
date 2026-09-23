import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.exporters import write_ego_extrinsics_json


class ExportersTest(unittest.TestCase):
    def test_ego_extrinsics_are_written_as_world_to_camera_matrices(self):
        T_world_from_ego = np.eye(4)
        T_world_from_ego[:3, 3] = [1.0, 2.0, 3.0]
        rows = [
            {"frame_index": "00000", "success": True, "T_world_from_ego": T_world_from_ego},
            {"frame_index": "00001", "success": False, "T_world_from_ego": None},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "ego_extrinsics.json"
            write_ego_extrinsics_json(output, rows)
            saved = json.loads(output.read_text())
        self.assertEqual(list(saved), ["00000"])
        np.testing.assert_allclose(saved["00000"], np.linalg.inv(T_world_from_ego))


if __name__ == "__main__":
    unittest.main()
