from __future__ import annotations

import unittest
from types import SimpleNamespace

from label.canvas_view import ImageAnnotatorCanvas


class LabelCanvasInteractionTest(unittest.TestCase):
    def test_right_clicking_schematic_joint_enters_location_mode(self) -> None:
        class CanvasStub:
            _locate_joint = None
            _read_only = False
            _base_image = object()
            _panning = False

            @staticmethod
            def _editable_schematic_hit(_x, _y):
                return (1, 8)

            @staticmethod
            def _nearest_joint(_x, _y, *, max_dist):
                raise AssertionError(f"nearest image joint should not be checked: {max_dist}")

            def _begin_joint_location(self, target):
                self.started_target = target

        canvas = CanvasStub()
        ImageAnnotatorCanvas._on_right_down(canvas, SimpleNamespace(x=50, y=60))

        self.assertEqual(canvas.started_target, (1, 8))
        self.assertFalse(canvas._panning)

    def test_second_right_click_places_joint_and_exits_location_mode(self) -> None:
        class CanvasStub:
            _locate_joint = (0, 5)
            _base_image = SimpleNamespace(size=(100, 80))
            _panning = False
            _points = [[(0.0, 0.0) for _ in range(21)] for _ in range(2)]
            _visible = [[False for _ in range(21)] for _ in range(2)]
            history_pushes = 0
            render_calls = 0

            @staticmethod
            def _canvas_to_image(x, y):
                return float(x), float(y)

            @staticmethod
            def _clamp_to_image(x, y):
                return float(x), float(y)

            def _push_history(self):
                self.history_pushes += 1

            def _render_overlay(self):
                self.render_calls += 1

        canvas = CanvasStub()
        placed = ImageAnnotatorCanvas._place_located_joint(canvas, 35, 45)

        self.assertTrue(placed)
        self.assertEqual(canvas._points[0][5], (35.0, 45.0))
        self.assertTrue(canvas._visible[0][5])
        self.assertIsNone(canvas._locate_joint)
        self.assertEqual(canvas.history_pushes, 1)
        self.assertEqual(canvas.render_calls, 1)

    def test_location_click_outside_image_keeps_mode_active(self) -> None:
        class CanvasStub:
            _locate_joint = (1, 3)
            _base_image = SimpleNamespace(size=(100, 80))
            _panning = False

            @staticmethod
            def _canvas_to_image(_x, _y):
                return 110.0, 20.0

        canvas = CanvasStub()
        placed = ImageAnnotatorCanvas._place_located_joint(canvas, 10, 20)

        self.assertFalse(placed)
        self.assertEqual(canvas._locate_joint, (1, 3))

    def test_location_mode_fades_all_image_annotations_except_target_joint(self) -> None:
        class CanvasStub:
            _HAND_COUNT = 2
            _annotation_visible = True
            _locate_joint = (0, 7)

            def __init__(self):
                self.bones = []
                self.points = []

            @staticmethod
            def _clear_overlay_items():
                return None

            def _render_hand_bones(self, hand, *, faded):
                self.bones.append((hand, faded))

            @staticmethod
            def _render_skeleton_overlay():
                return None

            @staticmethod
            def _render_mano_overlay():
                return None

            def _render_hand_points(self, hand, *, faded, focus_joint=None):
                self.points.append((hand, faded, focus_joint))

            @staticmethod
            def _render_selection_rect():
                return None

            @staticmethod
            def _render_visibility_schematic():
                return None

            @staticmethod
            def _render_count_schematic():
                return None

            @staticmethod
            def _render_joint_location_hint():
                return None

        canvas = CanvasStub()
        ImageAnnotatorCanvas._render_overlay(canvas)

        self.assertEqual(canvas.bones, [(0, True), (1, True)])
        self.assertEqual(canvas.points, [(0, True, 7), (1, True, None)])


if __name__ == "__main__":
    unittest.main()
