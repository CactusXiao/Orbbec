import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.tag_map import TagMap, load_tag_map, save_tag_map


class TagMapTest(unittest.TestCase):
    def test_round_trip_and_metadata_validation(self):
        tag_map = TagMap(
            family="tag36h11",
            tag_size_m=0.096,
            world_tag_corners={7: np.arange(12, dtype=np.float32).reshape(4, 3)},
            observation_counts={7: 12},
            reprojection_rmse_px={7: 0.7},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tag_map.json"
            save_tag_map(path, tag_map)
            loaded = load_tag_map(path, "tag36h11", 0.096)
            np.testing.assert_allclose(loaded.world_tag_corners[7], tag_map.world_tag_corners[7])
            self.assertEqual(loaded.observation_counts[7], 12)
            with self.assertRaises(ValueError):
                load_tag_map(path, "tag25h9", 0.096)


if __name__ == "__main__":
    unittest.main()
