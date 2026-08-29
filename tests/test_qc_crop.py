from __future__ import annotations

import math
import unittest

from src.qc.crop import expand_region_to_aspect, hand_focus_region


class QcHandCropTest(unittest.TestCase):
    def test_focus_region_uses_visible_in_image_points_from_both_hands(self) -> None:
        points = [[(100.0, 200.0), (200.0, 300.0)], [(700.0, 400.0), (math.nan, 10.0)]]
        visible = [[True, False], [True, True]]

        region = hand_focus_region(points, visible, image_size=(1000, 600), padding_ratio=0.25)

        self.assertIsNotNone(region)
        x1, y1, x2, y2 = region  # type: ignore[misc]
        self.assertLessEqual(x1, 100.0)
        self.assertGreaterEqual(x2, 700.0)
        self.assertLessEqual(y1, 200.0)
        self.assertGreaterEqual(y2, 400.0)
        self.assertGreaterEqual(x1, 0.0)
        self.assertLessEqual(x2, 1000.0)

    def test_focus_region_shifts_padding_inside_image_boundary(self) -> None:
        points = [[(2.0, 3.0), (20.0, 30.0)]]
        visible = [[True, True]]

        self.assertEqual(
            hand_focus_region(points, visible, image_size=(320, 240), min_size_pixels=100.0),
            (0.0, 0.0, 100.0, 100.0),
        )

    def test_no_valid_projection_falls_back_to_full_image(self) -> None:
        points = [[(-10.0, 20.0), (math.nan, 30.0)]]
        visible = [[True, True]]

        self.assertIsNone(hand_focus_region(points, visible, image_size=(640, 480)))

    def test_display_crop_expands_to_canvas_aspect_without_distortion(self) -> None:
        crop = expand_region_to_aspect((400.0, 300.0, 600.0, 500.0), (1000, 800), 16.0 / 9.0)
        x1, y1, x2, y2 = crop

        self.assertAlmostEqual((x2 - x1) / (y2 - y1), 16.0 / 9.0)
        self.assertLessEqual(x1, 400.0)
        self.assertGreaterEqual(x2, 600.0)
        self.assertLessEqual(y1, 300.0)
        self.assertGreaterEqual(y2, 500.0)
        self.assertGreaterEqual(x1, 0.0)
        self.assertLessEqual(x2, 1000.0)

    def test_impossible_aspect_keeps_both_hands_instead_of_trimming_focus_region(self) -> None:
        # Simulates one hand near the top and the other near the bottom. A 16:9
        # crop cannot span this full region within the source image width.
        both_hands_region = (450.0, 0.0, 550.0, 800.0)

        crop = expand_region_to_aspect(both_hands_region, (1000, 800), 16.0 / 9.0)

        self.assertEqual(crop, both_hands_region)


if __name__ == "__main__":
    unittest.main()
