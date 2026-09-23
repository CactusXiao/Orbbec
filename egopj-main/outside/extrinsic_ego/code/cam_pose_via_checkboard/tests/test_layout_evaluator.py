import unittest

import numpy as np

from src.layout_evaluator import evaluate_layout


class LayoutEvaluatorTest(unittest.TestCase):
    def test_visibility_dropout_and_plane_metrics(self):
        frames = [{1, 2, 3}, {1, 2, 3}, {1, 2}]
        normals = {
            1: np.array([0.0, 0.0, 1.0]),
            2: np.array([0.0, 1.0, 0.0]),
            3: np.array([0.0, 0.0, 1.0]),
        }
        report = evaluate_layout(frames, normals)
        self.assertAlmostEqual(report["at_least_2_tags_ratio"], 1.0)
        self.assertAlmostEqual(report["at_least_3_tags_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(report["two_plane_ratio"], 1.0)
        self.assertAlmostEqual(report["worst_single_tag_removed_at_least_2_ratio"], 2.0 / 3.0)
        self.assertFalse(report["passes_recommended_coverage"])


if __name__ == "__main__":
    unittest.main()
