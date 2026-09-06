from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
import tkinter.font as tkfont

from PIL import Image, ImageTk

from mano.joint_order import SMPLX_MANO_SKELETON_EDGES
from src.qc.crop import Box, expand_region_to_aspect


Point = Tuple[float, float]
HandPoints = List[List[Point]]
HandVisible = List[List[bool]]
JointCounts = List[List[int]]
MeshLine = Tuple[Point, Point, str]


@dataclass
class ViewState:
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


class ImageAnnotatorCanvas(tk.Canvas):
    _HAND_COUNT = 2
    _JOINT_COUNT = 21
    _SKELETON_EDGES = SMPLX_MANO_SKELETON_EDGES
    _LEFT_BASES = ("#0078ff", "#28b4ff", "#50dcff", "#78ffdc", "#b4ffb4")
    _RIGHT_BASES = ("#ff4600", "#ff7800", "#ffb400", "#ffdc3c", "#ffff78")
    _SMPLX_JOINT_STYLE = (
        (0, 0),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
        (4, 0), (4, 1), (4, 2),
        (3, 0), (3, 1), (3, 2),
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (2, 3), (3, 3), (4, 3),
    )
    _SMPLX_SCHEMATIC_POINTS = (
        (0.0, 0.95),
        (-0.35, 0.25), (-0.44, -0.20), (-0.48, -0.60),
        (0.00, 0.20), (-0.02, -0.30), (-0.02, -0.75),
        (0.65, 0.35), (0.78, 0.00), (0.86, -0.30),
        (0.35, 0.25), (0.42, -0.20), (0.45, -0.60),
        (-0.55, 0.35), (-0.82, 0.05), (-1.00, -0.22),
        (-1.12, -0.48), (-0.50, -0.95), (-0.02, -1.15), (0.48, -0.95), (0.92, -0.58),
    )

    def __init__(self, master, *, bg: str, **kwargs):
        super().__init__(master, bg=bg, highlightthickness=0, **kwargs)

        self._img_path: Optional[Path] = None
        self._base_image: Optional[Image.Image] = None
        self._rendered_image: Optional[Image.Image] = None
        self._rendered_path: Optional[Path] = None
        self._imgtk: Optional[ImageTk.PhotoImage] = None
        self._img_item: Optional[int] = None

        self._view = ViewState(scale=1.0, offset_x=0.0, offset_y=0.0)
        self._view_user_adjusted = False
        self._focus_region: Optional[Box] = None
        self._points = self._empty_points()
        self._visible = self._none_visible()
        self._count_base = self._empty_counts()
        self._mano_lines: List[MeshLine] = []
        self._skeleton_points: Optional[HandPoints] = None
        self._skeleton_visible: Optional[HandVisible] = None
        self._annotation_visible = True
        self._read_only = False
        self._overlay_items: List[int] = []
        self._message_item: Optional[int] = None
        self._history: List[Tuple[HandPoints, HandVisible]] = []

        self._panning = False
        self._pan_last: Tuple[int, int] = (0, 0)
        self._drag_joint: Optional[Tuple[int, int]] = None
        self._drag_history_pushed = False
        self._locate_joint: Optional[Tuple[int, int]] = None
        self._press_joint_candidate: Optional[Tuple[int, int]] = None
        self._left_pressed = False
        self._selecting = False
        self._selection_after_id: Optional[str] = None
        self._selection_start: Optional[Tuple[int, int]] = None
        self._selection_current: Optional[Tuple[int, int]] = None
        self._pending_fit_after_id: Optional[str] = None

        self.bind("<ButtonPress-1>", self._on_left_down)
        self.bind("<B1-Motion>", self._on_left_drag)
        self.bind("<ButtonRelease-1>", self._on_left_up)
        self.bind("<Double-Button-1>", self._on_left_double)
        self.bind("<ButtonPress-3>", self._on_right_down)
        self.bind("<B3-Motion>", self._on_right_drag)
        self.bind("<ButtonRelease-3>", self._on_right_up)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", self._on_wheel_linux)
        self.bind("<Button-5>", self._on_wheel_linux)
        self.bind("<Configure>", self._on_resize)
        self.bind("<Escape>", self._on_escape)

    def clear(self) -> None:
        self._cancel_pending_fit()
        self._img_path = None
        self._base_image = None
        self._rendered_image = None
        self._rendered_path = None
        self._imgtk = None
        if self._img_item is not None:
            self.delete(self._img_item)
            self._img_item = None
        self._clear_overlay_items()
        self._points = self._empty_points()
        self._visible = self._none_visible()
        self._count_base = self._empty_counts()
        self._mano_lines = []
        self._skeleton_points = None
        self._skeleton_visible = None
        self._annotation_visible = True
        self._read_only = False
        self._history = []
        self._cancel_selection_timer()
        self._left_pressed = False
        self._selecting = False
        self._locate_joint = None
        self._press_joint_candidate = None
        self._selection_start = None
        self._selection_current = None
        self._view = ViewState(scale=1.0, offset_x=0.0, offset_y=0.0)
        self._view_user_adjusted = False
        self._focus_region = None
        self._show_message("No image")

    def set_image(self, path: Optional[Path]) -> None:
        self._cancel_pending_fit()
        self._locate_joint = None
        self._panning = False
        self._img_path = path
        self._base_image = None
        self._rendered_image = None
        self._rendered_path = None
        self._imgtk = None
        self._view_user_adjusted = False
        self._focus_region = None
        if self._img_item is not None:
            self.delete(self._img_item)
            self._img_item = None
        self._clear_overlay_items()

        if not path or not path.exists() or not path.is_file():
            self._show_message("Image not found")
            self._render_overlay()
            return

        try:
            with Image.open(path) as im:
                self._base_image = im.convert("RGB")
        except Exception:
            self._base_image = None
            self._show_message("Failed to load image")
            self._render_overlay()
            return

        self._hide_message()
        self._fit_and_render_image()

    def set_rendered_image(self, path: Optional[Path]) -> None:
        """Swap QC's composited preview without changing annotations, pan or zoom."""
        if path == self._rendered_path:
            return
        self._rendered_path = path
        self._rendered_image = None
        if path is not None:
            with Image.open(path) as image:
                self._rendered_image = image.convert("RGB")
        self._render_image()
        self._render_overlay()

    def set_hand_state(self, points: HandPoints, visible: HandVisible) -> None:
        self._locate_joint = None
        self._panning = False
        self._points = self._coerce_points(points)
        self._visible = self._coerce_visible(visible)
        self._history = []
        self._render_overlay()

    def set_count_base(self, counts: JointCounts) -> None:
        self._count_base = self._coerce_counts(counts)
        self._render_overlay()

    def set_mano_overlay(self, lines: Optional[List[MeshLine]]) -> None:
        self._mano_lines = list(lines or [])
        self._render_overlay()

    def set_skeleton_overlay(self, points: Optional[HandPoints], visible: Optional[HandVisible] = None) -> None:
        if points is None:
            self._skeleton_points = None
            self._skeleton_visible = None
        else:
            self._skeleton_points = self._coerce_points(points)
            default_visible = [[True for _ in range(self._JOINT_COUNT)] for _ in range(self._HAND_COUNT)]
            self._skeleton_visible = self._coerce_visible(visible or default_visible)
        self._render_overlay()

    def set_annotation_visible(self, visible: bool) -> None:
        self._annotation_visible = bool(visible)
        if not self._annotation_visible:
            self._locate_joint = None
            self._panning = False
        self._cancel_selection_timer()
        self._selecting = False
        self._drag_joint = None
        self._press_joint_candidate = None
        self._render_overlay()

    def set_read_only(self, read_only: bool) -> None:
        self._read_only = bool(read_only)
        if self._read_only:
            self._locate_joint = None
            self._panning = False
        self._cancel_selection_timer()
        self._selecting = False
        self._drag_joint = None
        self._press_joint_candidate = None
        self._render_overlay()

    def image_size(self) -> Optional[Tuple[int, int]]:
        return None if self._base_image is None else self._base_image.size

    def set_focus_region(self, region: Optional[Box]) -> None:
        """Fit a source-image region to the canvas while preserving its aspect."""
        self._focus_region = self._normalize_region(region)
        self._view_user_adjusted = False
        if self._base_image is None:
            return
        self._cancel_pending_fit()
        self._fit_and_render_image()

    def get_hand_state(self) -> Tuple[HandPoints, HandVisible]:
        return self._copy_points(self._points), self._copy_visible(self._visible)

    def ignore_view(self) -> None:
        if self._read_only:
            return
        self._locate_joint = None
        self._push_history()
        self._visible = [[False for _ in range(self._JOINT_COUNT)] for _ in range(self._HAND_COUNT)]
        self._render_overlay()

    def undo(self) -> None:
        if self._read_only:
            return
        self._locate_joint = None
        if not self._history:
            return
        self._points, self._visible = self._history.pop()
        self._render_overlay()

    def undo_last_point(self) -> None:
        self.undo()

    def _empty_points(self) -> HandPoints:
        return [[(0.0, 0.0) for _ in range(self._JOINT_COUNT)] for _ in range(self._HAND_COUNT)]

    def _none_visible(self) -> HandVisible:
        return [[False for _ in range(self._JOINT_COUNT)] for _ in range(self._HAND_COUNT)]

    def _empty_counts(self) -> JointCounts:
        return [[0 for _ in range(self._JOINT_COUNT)] for _ in range(self._HAND_COUNT)]

    def _copy_points(self, points: HandPoints) -> HandPoints:
        return [[(float(x), float(y)) for (x, y) in hand] for hand in points]

    def _copy_visible(self, visible: HandVisible) -> HandVisible:
        return [[bool(v) for v in hand] for hand in visible]

    def _coerce_points(self, points: HandPoints) -> HandPoints:
        out = self._empty_points()
        for hand in range(min(self._HAND_COUNT, len(points))):
            for joint in range(min(self._JOINT_COUNT, len(points[hand]))):
                x, y = points[hand][joint]
                out[hand][joint] = (float(x), float(y))
        return out

    def _coerce_visible(self, visible: HandVisible) -> HandVisible:
        out = [[False for _ in range(self._JOINT_COUNT)] for _ in range(self._HAND_COUNT)]
        for hand in range(min(self._HAND_COUNT, len(visible))):
            for joint in range(min(self._JOINT_COUNT, len(visible[hand]))):
                out[hand][joint] = bool(visible[hand][joint])
        return out

    def _coerce_counts(self, counts: JointCounts) -> JointCounts:
        out = self._empty_counts()
        for hand in range(min(self._HAND_COUNT, len(counts))):
            for joint in range(min(self._JOINT_COUNT, len(counts[hand]))):
                out[hand][joint] = max(0, int(counts[hand][joint]))
        return out

    def _push_history(self) -> None:
        self._history.append((self._copy_points(self._points), self._copy_visible(self._visible)))
        if len(self._history) > 50:
            self._history.pop(0)

    def _clear_overlay_items(self) -> None:
        for item in self._overlay_items:
            self.delete(item)
        self._overlay_items = []

    def _show_message(self, text: str) -> None:
        self._hide_message()
        w = max(1, int(self.winfo_width()))
        h = max(1, int(self.winfo_height()))
        try:
            f = tkfont.nametofont("TkDefaultFont")
            msg_font = (f.cget("family"), max(10, int(f.cget("size")) + 4))
        except Exception:
            msg_font = None
        self._message_item = self.create_text(
            w // 2,
            h // 2,
            text=text,
            fill="#cfcfcf",
            font=msg_font,
        )

    def _hide_message(self) -> None:
        if self._message_item is not None:
            self.delete(self._message_item)
            self._message_item = None

    def _cancel_pending_fit(self) -> None:
        if self._pending_fit_after_id is None:
            return
        try:
            self.after_cancel(self._pending_fit_after_id)
        except Exception:
            pass
        self._pending_fit_after_id = None

    def _fit_and_render_image(self) -> None:
        if self._base_image is None:
            self._render_overlay()
            return
        if self.winfo_width() <= 1 or self.winfo_height() <= 1:
            self._pending_fit_after_id = self.after(16, self._fit_and_render_image)
            return
        self._pending_fit_after_id = None
        self._fit_to_canvas()
        self._render_image()
        self._render_overlay()

    def _on_resize(self, _evt) -> None:
        if self._base_image is None:
            if self._message_item is not None:
                self._show_message(self.itemcget(self._message_item, "text") or "")
            self._render_overlay()
            return
        if not self._view_user_adjusted:
            self._cancel_pending_fit()
            self._fit_to_canvas()
        self._render_image()
        self._render_overlay()

    def _on_left_down(self, evt) -> None:
        if self._read_only:
            self._cancel_selection_timer()
            return
        if self._locate_joint is not None:
            self._cancel_selection_timer()
            return
        schematic_hit = self._editable_schematic_hit(evt.x, evt.y)
        if schematic_hit is not None:
            self._cancel_selection_timer()
            self._left_pressed = False
            self._selecting = False
            self._drag_joint = None
            self._drag_history_pushed = False
            self._press_joint_candidate = None
            self._selection_start = None
            self._selection_current = None
            hand, joint = schematic_hit
            self._push_history()
            self._visible[hand][joint] = not self._visible[hand][joint]
            self._render_overlay()
            return

        self._left_pressed = True
        self._cancel_selection_timer()
        self._selecting = False
        self._selection_start = (evt.x, evt.y)
        self._selection_current = (evt.x, evt.y)
        self._press_joint_candidate = self._nearest_joint(evt.x, evt.y)
        self._drag_joint = None
        self._drag_history_pushed = False
        self._selection_after_id = self.after(380, self._begin_selection)

    def _on_left_drag(self, evt) -> None:
        if self._read_only or self._locate_joint is not None:
            return
        if self._selecting:
            self._selection_current = (evt.x, evt.y)
            self._render_overlay()
            return
        if self._selection_after_id is not None:
            self._selection_current = (evt.x, evt.y)
            if self._press_joint_candidate is not None and self._moved_from_selection_start(evt.x, evt.y, 5):
                self._cancel_selection_timer()
                self._start_joint_drag(self._press_joint_candidate, evt)
            return
        if self._drag_joint is None:
            return
        self._move_drag_joint(evt)

    def _start_joint_drag(self, target: Tuple[int, int], evt) -> None:
        self._drag_joint = target
        self._move_drag_joint(evt)

    def _move_drag_joint(self, evt) -> None:
        if self._drag_joint is None:
            return
        hand, joint = self._drag_joint
        if not self._drag_history_pushed:
            self._push_history()
            self._drag_history_pushed = True
        x_img, y_img = self._canvas_to_image(evt.x, evt.y)
        self._points[hand][joint] = self._clamp_to_image(x_img, y_img)
        self._visible[hand][joint] = True
        self._render_overlay(drag_hand=hand)

    def _on_left_up(self, evt) -> None:
        if self._read_only or self._locate_joint is not None:
            self._cancel_selection_timer()
            self._left_pressed = False
            self._drag_joint = None
            self._selecting = False
            return
        if self._selecting:
            self._selection_current = (evt.x, evt.y)
            self._finish_selection()
        self._cancel_selection_timer()
        self._left_pressed = False
        self._drag_joint = None
        self._drag_history_pushed = False
        self._press_joint_candidate = None
        self._selecting = False
        self._selection_start = None
        self._selection_current = None
        self._render_overlay()

    def _on_left_double(self, evt) -> None:
        if self._read_only or self._locate_joint is not None:
            self._cancel_selection_timer()
            return
        self._cancel_selection_timer()
        self._press_joint_candidate = None
        if self._editable_schematic_hit(evt.x, evt.y) is not None:
            return
        hit = self._nearest_joint(evt.x, evt.y, max_dist=14.0)
        if hit is None:
            return
        hand, joint = hit
        self._push_history()
        self._visible[hand][joint] = not self._visible[hand][joint]
        self._drag_joint = None
        self._drag_history_pushed = False
        self._render_overlay()

    def _begin_selection(self) -> None:
        self._selection_after_id = None
        if self._read_only:
            return
        if not self._left_pressed or self._drag_joint is not None or self._selection_start is None:
            return
        self._selecting = True
        self._press_joint_candidate = None
        self._selection_current = self._selection_current or self._selection_start
        self._render_overlay()

    def _finish_selection(self) -> None:
        if self._selection_start is None or self._selection_current is None:
            return
        x0, y0 = self._selection_start
        x1, y1 = self._selection_current
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        hits: List[Tuple[int, int]] = []
        for hand in range(self._HAND_COUNT):
            for joint in range(self._JOINT_COUNT):
                if self._is_hidden_point(self._points[hand][joint]):
                    continue
                cx, cy = self._image_to_canvas(*self._points[hand][joint])
                if left <= cx <= right and top <= cy <= bottom:
                    hits.append((hand, joint))
        if not hits:
            return
        self._push_history()
        for hand, joint in hits:
            self._visible[hand][joint] = not self._visible[hand][joint]

    def _moved_from_selection_start(self, x: int, y: int, threshold: int) -> bool:
        if self._selection_start is None:
            return False
        sx, sy = self._selection_start
        dx = x - sx
        dy = y - sy
        return (dx * dx + dy * dy) > threshold * threshold

    def _cancel_selection_timer(self) -> None:
        if self._selection_after_id is None:
            return
        try:
            self.after_cancel(self._selection_after_id)
        except Exception:
            pass
        self._selection_after_id = None

    def _on_right_down(self, evt) -> None:
        if self._locate_joint is not None:
            if self._editable_schematic_hit(evt.x, evt.y) is not None:
                self._cancel_joint_location()
                return
            if self._place_located_joint(evt.x, evt.y):
                return
            self._cancel_joint_location()
            return

        if not self._read_only and self._base_image is not None:
            target = self._editable_schematic_hit(evt.x, evt.y)
            if target is None:
                target = self._nearest_joint(evt.x, evt.y, max_dist=14.0)
            if target is not None:
                self._begin_joint_location(target)
                return

        self._panning = True
        self._pan_last = (evt.x, evt.y)

    def _on_right_drag(self, evt) -> None:
        if not self._panning:
            return
        dx = evt.x - self._pan_last[0]
        dy = evt.y - self._pan_last[1]
        self._pan_last = (evt.x, evt.y)
        self._view.offset_x += dx
        self._view.offset_y += dy
        self._view_user_adjusted = True
        self._render_image()
        self._render_overlay()

    def _on_right_up(self, _evt) -> None:
        self._panning = False

    def _on_escape(self, _evt=None) -> None:
        self._cancel_joint_location()

    def _begin_joint_location(self, target: Tuple[int, int]) -> None:
        if self._read_only or self._base_image is None:
            return
        self._cancel_selection_timer()
        self._left_pressed = False
        self._selecting = False
        self._drag_joint = None
        self._drag_history_pushed = False
        self._press_joint_candidate = None
        self._selection_start = None
        self._selection_current = None
        self._panning = False
        self._locate_joint = target
        try:
            self.focus_set()
        except Exception:
            pass
        self._render_overlay()

    def _place_located_joint(self, x: float, y: float) -> bool:
        target = self._locate_joint
        if target is None or self._base_image is None:
            return False
        x_img, y_img = self._canvas_to_image(x, y)
        iw, ih = self._base_image.size
        if not (0.0 <= x_img <= float(iw) and 0.0 <= y_img <= float(ih)):
            return False

        hand, joint = target
        self._push_history()
        self._points[hand][joint] = self._clamp_to_image(x_img, y_img)
        self._visible[hand][joint] = True
        self._locate_joint = None
        self._panning = False
        self._render_overlay()
        return True

    def _cancel_joint_location(self) -> None:
        if self._locate_joint is None:
            return
        self._locate_joint = None
        self._panning = False
        self._render_overlay()

    def _on_wheel_linux(self, evt) -> None:
        if evt.num == 4:
            self._zoom_at(evt.x, evt.y, 1.1)
        elif evt.num == 5:
            self._zoom_at(evt.x, evt.y, 1 / 1.1)

    def _on_wheel(self, evt) -> None:
        if evt.delta == 0:
            return
        factor = 1.1 if evt.delta > 0 else (1 / 1.1)
        self._zoom_at(evt.x, evt.y, factor)

    def _fit_to_canvas(self) -> None:
        if self._base_image is None:
            return
        w = max(1, int(self.winfo_width()))
        h = max(1, int(self.winfo_height()))
        x1, y1, x2, y2 = self._crop_region_for_aspect(w / max(1, h))
        crop_width = max(1.0, x2 - x1)
        crop_height = max(1.0, y2 - y1)
        scale = min(w / crop_width, h / crop_height)
        scale = max(0.05, min(10.0, scale))
        self._view.scale = scale
        self._view.offset_x = (w - crop_width * scale) / 2 - x1 * scale
        self._view.offset_y = (h - crop_height * scale) / 2 - y1 * scale

    def _zoom_at(self, cx: int, cy: int, factor: float) -> None:
        old = self._view.scale
        new = max(0.05, min(10.0, old * factor))
        if math.isclose(new, old):
            return

        x_img, y_img = self._canvas_to_image(cx, cy)
        self._view.scale = new
        self._view.offset_x = cx - x_img * new
        self._view.offset_y = cy - y_img * new
        self._view_user_adjusted = True
        self._render_image()
        self._render_overlay()

    def _render_image(self) -> None:
        if self._base_image is None:
            return
        iw, ih = self._base_image.size
        scale = self._view.scale
        canvas_width = max(1, int(self.winfo_width()))
        canvas_height = max(1, int(self.winfo_height()))
        source_x1 = max(0, int(math.floor(-self._view.offset_x / scale)))
        source_y1 = max(0, int(math.floor(-self._view.offset_y / scale)))
        source_x2 = min(iw, int(math.ceil((canvas_width - self._view.offset_x) / scale)))
        source_y2 = min(ih, int(math.ceil((canvas_height - self._view.offset_y) / scale)))
        if source_x2 <= source_x1 or source_y2 <= source_y1:
            if self._img_item is not None:
                self.delete(self._img_item)
                self._img_item = None
            return
        display_image = self._rendered_image if self._rendered_image is not None else self._base_image
        img = display_image.crop((source_x1, source_y1, source_x2, source_y2))
        tw = max(1, int(round((source_x2 - source_x1) * scale)))
        th = max(1, int(round((source_y2 - source_y1) * scale)))
        img = img.resize((tw, th), Image.BILINEAR)
        self._imgtk = ImageTk.PhotoImage(img)
        x = self._view.offset_x + source_x1 * scale
        y = self._view.offset_y + source_y1 * scale
        if self._img_item is None:
            self._img_item = self.create_image(x, y, anchor="nw", image=self._imgtk)
        else:
            self.coords(self._img_item, x, y)
            self.itemconfigure(self._img_item, image=self._imgtk)
        self.tag_lower(self._img_item)

    def _normalize_region(self, region: Optional[Box]) -> Optional[Box]:
        if region is None or self._base_image is None:
            return None
        iw, ih = self._base_image.size
        try:
            x1, y1, x2, y2 = (float(value) for value in region)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            return None
        x1, x2 = sorted((max(0.0, min(float(iw), x1)), max(0.0, min(float(iw), x2))))
        y1, y2 = sorted((max(0.0, min(float(ih), y1)), max(0.0, min(float(ih), y2))))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            return None
        return (x1, y1, x2, y2)

    def _crop_region_for_aspect(self, target_aspect: float) -> Box:
        if self._base_image is None:
            return (0.0, 0.0, 1.0, 1.0)
        iw, ih = self._base_image.size
        region = self._focus_region or (0.0, 0.0, float(iw), float(ih))
        return expand_region_to_aspect(region, (iw, ih), target_aspect)

    def _render_overlay(self, *, drag_hand: Optional[int] = None) -> None:
        self._clear_overlay_items()
        locate_target = self._locate_joint
        if self._annotation_visible:
            for hand in range(self._HAND_COUNT):
                faded = locate_target is not None or drag_hand == hand
                self._render_hand_bones(hand, faded=faded)
        self._render_skeleton_overlay()
        self._render_mano_overlay()
        if self._annotation_visible:
            for hand in range(self._HAND_COUNT):
                faded = locate_target is not None or drag_hand == hand
                focus_joint = locate_target[1] if locate_target is not None and locate_target[0] == hand else None
                self._render_hand_points(hand, faded=faded, focus_joint=focus_joint)
            self._render_selection_rect()
            self._render_visibility_schematic()
            self._render_count_schematic()
            self._render_joint_location_hint()

    def _render_hand_bones(self, hand: int, *, faded: bool) -> None:
        for a, b in self._SKELETON_EDGES:
            if not self._visible[hand][a] or not self._visible[hand][b]:
                continue
            if self._is_hidden_point(self._points[hand][a]) or self._is_hidden_point(self._points[hand][b]):
                continue
            ax, ay = self._image_to_canvas(*self._points[hand][a])
            bx, by = self._image_to_canvas(*self._points[hand][b])
            self._create_gradient_line(
                ax,
                ay,
                bx,
                by,
                self._joint_color(hand, a, faded=faded),
                self._joint_color(hand, b, faded=faded),
                faded=faded,
            )

    def _create_gradient_line(
        self,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        color_a: str,
        color_b: str,
        *,
        faded: bool,
    ) -> None:
        steps = 5
        for i in range(steps):
            t0 = i / steps
            t1 = (i + 1) / steps
            x0 = ax + (bx - ax) * t0
            y0 = ay + (by - ay) * t0
            x1 = ax + (bx - ax) * t1
            y1 = ay + (by - ay) * t1
            color = self._mix_hex(color_a, color_b, (t0 + t1) / 2)
            opts = {"fill": color, "width": 1, "capstyle": tk.ROUND}
            if faded:
                opts["stipple"] = "gray50"
            item = self.create_line(x0, y0, x1, y1, **opts)
            self._overlay_items.append(item)

    def _render_mano_overlay(self) -> None:
        for (a, b, color) in self._mano_lines:
            if self._is_hidden_point(a) or self._is_hidden_point(b):
                continue
            ax, ay = self._image_to_canvas(*a)
            bx, by = self._image_to_canvas(*b)
            item = self.create_line(ax, ay, bx, by, fill=color, width=1, stipple="gray50")
            self._overlay_items.append(item)

    def _render_skeleton_overlay(self) -> None:
        if self._skeleton_points is None or self._skeleton_visible is None:
            return
        for hand in range(self._HAND_COUNT):
            for a, b in self._SKELETON_EDGES:
                if not self._skeleton_visible[hand][a] or not self._skeleton_visible[hand][b]:
                    continue
                if self._is_hidden_point(self._skeleton_points[hand][a]) or self._is_hidden_point(self._skeleton_points[hand][b]):
                    continue
                ax, ay = self._image_to_canvas(*self._skeleton_points[hand][a])
                bx, by = self._image_to_canvas(*self._skeleton_points[hand][b])
                self._create_gradient_line(
                    ax,
                    ay,
                    bx,
                    by,
                    self._joint_color(hand, a, faded=False),
                    self._joint_color(hand, b, faded=False),
                    faded=False,
                )
            for joint in range(self._JOINT_COUNT):
                if not self._skeleton_visible[hand][joint] or self._is_hidden_point(self._skeleton_points[hand][joint]):
                    continue
                cx, cy = self._image_to_canvas(*self._skeleton_points[hand][joint])
                r = 2
                color = self._joint_color(hand, joint, faded=False)
                item = self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline=color)
                self._overlay_items.append(item)

    def _render_hand_points(self, hand: int, *, faded: bool, focus_joint: Optional[int] = None) -> None:
        for joint in range(self._JOINT_COUNT):
            if self._is_hidden_point(self._points[hand][joint]):
                continue
            cx, cy = self._image_to_canvas(*self._points[hand][joint])
            focused = focus_joint == joint
            point_faded = faded and not focused
            r = 5 if focused else 2
            color = self._joint_color(hand, joint, faded=point_faded)
            if self._visible[hand][joint] or focused:
                opts = {"fill": color, "outline": "#ffffff" if focused else color, "width": 2 if focused else 1}
            else:
                opts = {"fill": "", "outline": color}
            if point_faded:
                opts["stipple"] = "gray50"
            item = self.create_oval(cx - r, cy - r, cx + r, cy + r, **opts)
            self._overlay_items.append(item)

    def _render_joint_location_hint(self) -> None:
        if self._locate_joint is None:
            return
        hand, joint = self._locate_joint
        hand_label = "左手" if hand == 0 else "右手"
        w = max(1, int(self.winfo_width()))
        item = self.create_text(
            w * 0.5,
            24,
            text=f"定位模式：{hand_label}关节 {joint}，右键图像完成定位，右键图外退出（Esc 取消）",
            fill="#ffffff",
        )
        self._overlay_items.append(item)

    def _render_selection_rect(self) -> None:
        if not self._selecting or self._selection_start is None or self._selection_current is None:
            return
        x0, y0 = self._selection_start
        x1, y1 = self._selection_current
        item = self.create_rectangle(x0, y0, x1, y1, outline="#f6d44a", width=1, dash=(4, 2))
        self._overlay_items.append(item)

    def _render_visibility_schematic(self) -> None:
        centers, scale = self._visibility_schematic_layout()
        for hand, (cx, cy) in zip(self._schematic_hand_order(), centers):
            self._render_visibility_hand(hand, cx, cy, scale)

    def _render_visibility_hand(self, hand: int, cx: float, cy: float, scale: float) -> None:
        coords = [self._schematic_point(hand, joint, cx, cy, scale) for joint in range(self._JOINT_COUNT)]
        width = max(1, int(round(scale / 48.0)))
        for a, b in self._SKELETON_EDGES:
            ax, ay = coords[a]
            bx, by = coords[b]
            item = self.create_line(ax, ay, bx, by, fill="#8a8a8a", width=width)
            self._overlay_items.append(item)
        for joint, (x, y) in enumerate(coords):
            r = max(6, int(round(scale * 0.125)))
            focused = self._locate_joint == (hand, joint)
            outline = "#ffffff" if focused else ("#111111" if self._visible[hand][joint] else "#d0d0d0")
            outline_width = 3 if focused else (1 if self._visible[hand][joint] else 2)
            if self._visible[hand][joint]:
                color = self._visibility_schematic_color(hand, joint)
                item = self.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=outline, width=outline_width)
            else:
                item = self.create_oval(x - r, y - r, x + r, y + r, fill="", outline=outline, width=outline_width)
            self._overlay_items.append(item)

    def _render_count_schematic(self) -> None:
        centers, scale = self._count_schematic_layout()
        counts = self._effective_counts()
        for hand, (cx, cy) in zip(self._schematic_hand_order(), centers):
            self._render_count_hand(hand, counts[hand], cx, cy, scale)

    def _visibility_schematic_layout(self) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], float]:
        h = max(1, int(self.winfo_height()))
        scale = 120.0
        hand_w = 295
        gap = 70
        left = 28
        bottom = h - 26
        cy = bottom - 145
        centers = (
            (left + hand_w * 0.5, cy),
            (left + hand_w * 1.5 + gap, cy),
        )
        return centers, scale

    def _visibility_schematic_hit(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        centers, scale = self._visibility_schematic_layout()
        best: Optional[Tuple[int, int]] = None
        max_dist = max(11.0, scale * 0.23)
        best_d2 = max_dist * max_dist
        for hand, (cx, cy) in zip(self._schematic_hand_order(), centers):
            for joint in range(self._JOINT_COUNT):
                jx, jy = self._schematic_point(hand, joint, cx, cy, scale)
                d2 = (jx - x) * (jx - x) + (jy - y) * (jy - y)
                if d2 <= best_d2:
                    best = (hand, joint)
                    best_d2 = d2
        return best

    def _count_schematic_layout(self) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], float]:
        w = max(1, int(self.winfo_width()))
        h = max(1, int(self.winfo_height()))
        scale = 30.0
        hand_w = 74
        gap = 18
        right = w - 18
        bottom = h - 18
        centers = (
            (right - (hand_w * 1.5 + gap), bottom - 36),
            (right - (hand_w * 0.5), bottom - 36),
        )
        return centers, scale

    def _count_schematic_hit(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        centers, scale = self._count_schematic_layout()
        best: Optional[Tuple[int, int]] = None
        best_d2 = 9.0 * 9.0
        for hand, (cx, cy) in zip(self._schematic_hand_order(), centers):
            for joint in range(self._JOINT_COUNT):
                jx, jy = self._schematic_point(hand, joint, cx, cy, scale)
                d2 = (jx - x) * (jx - x) + (jy - y) * (jy - y)
                if d2 <= best_d2:
                    best = (hand, joint)
                    best_d2 = d2
        return best

    def _editable_schematic_hit(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        hit = self._visibility_schematic_hit(x, y)
        if hit is not None:
            return hit
        return self._count_schematic_hit(x, y)

    def _schematic_hand_order(self) -> Tuple[int, int]:
        # Fixed UI/data convention: left schematic is hand 0, right schematic is hand 1.
        return (0, 1)

    def _effective_counts(self) -> JointCounts:
        out = self._empty_counts()
        for hand in range(self._HAND_COUNT):
            for joint in range(self._JOINT_COUNT):
                out[hand][joint] = self._count_base[hand][joint] + (1 if self._visible[hand][joint] else 0)
        return out

    def _render_count_hand(self, hand: int, counts: List[int], cx: float, cy: float, scale: float) -> None:
        coords = [self._schematic_point(hand, joint, cx, cy, scale) for joint in range(self._JOINT_COUNT)]
        for a, b in self._SKELETON_EDGES:
            ax, ay = coords[a]
            bx, by = coords[b]
            item = self.create_line(ax, ay, bx, by, fill="#6a6a6a", width=1)
            self._overlay_items.append(item)
        for joint, (x, y) in enumerate(coords):
            color = self._count_color(counts[joint])
            r = 4
            focused = self._locate_joint == (hand, joint)
            outline = "#ffffff" if focused else ("#111111" if color is not None else "#5a5a5a")
            outline_width = 2 if focused else 1
            if color is None:
                item = self.create_oval(x - r, y - r, x + r, y + r, fill="", outline=outline, width=outline_width)
            else:
                item = self.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=outline, width=outline_width)
            self._overlay_items.append(item)

    def _schematic_point(self, hand: int, joint: int, cx: float, cy: float, scale: float) -> Tuple[float, float]:
        x, y = self._SMPLX_SCHEMATIC_POINTS[joint]
        # The base shape is a right hand back view. Mirror hand 0 to show the left hand.
        if hand == 0:
            x = -x
        return (cx + x * scale, cy + y * scale)

    def _count_color(self, count: int) -> Optional[str]:
        if count <= 0:
            return None
        if count == 1:
            return "#ff3b30"
        if count == 2:
            return "#ffd60a"
        return "#30d158"

    def _visibility_schematic_color(self, hand: int, joint: int) -> str:
        if joint == 0:
            return "#0078ff" if hand == 0 else "#ff7800"
        return self._joint_color(hand, joint, faded=False)

    def _nearest_joint(self, x: float, y: float, *, max_dist: float = 12.0) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        best_d2 = max_dist * max_dist
        for hand in range(self._HAND_COUNT):
            for joint in range(self._JOINT_COUNT):
                if self._is_hidden_point(self._points[hand][joint]):
                    continue
                cx, cy = self._image_to_canvas(*self._points[hand][joint])
                d2 = (cx - x) * (cx - x) + (cy - y) * (cy - y)
                if d2 <= best_d2:
                    best = (hand, joint)
                    best_d2 = d2
        return best

    @staticmethod
    def _is_hidden_point(point: Point) -> bool:
        x, y = point
        return float(x) == -1.0 and float(y) == -1.0

    def _joint_color(self, hand: int, joint: int, *, faded: bool) -> str:
        if joint == 0:
            color = "#f2f2f2" if hand == 0 else "#d2d2ff"
        else:
            bases = self._LEFT_BASES if hand == 0 else self._RIGHT_BASES
            finger, step = self._SMPLX_JOINT_STYLE[joint]
            color = self._brighten(bases[finger], 0.10 * step)
        if faded:
            color = self._mix_hex(color, "#ffffff", 0.45)
        return color

    def _brighten(self, color: str, amount: float) -> str:
        return self._mix_hex(color, "#ffffff", max(0.0, min(1.0, amount)))

    def _mix_hex(self, color_a: str, color_b: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        ar, ag, ab = self._hex_to_rgb(color_a)
        br, bg, bb = self._hex_to_rgb(color_b)
        r = int(round(ar + (br - ar) * t))
        g = int(round(ag + (bg - ag) * t))
        b = int(round(ab + (bb - ab) * t))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _hex_to_rgb(self, color: str) -> Tuple[int, int, int]:
        c = color.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    def _image_to_canvas(self, x_img: float, y_img: float) -> Tuple[float, float]:
        return (
            self._view.offset_x + float(x_img) * self._view.scale,
            self._view.offset_y + float(y_img) * self._view.scale,
        )

    def _canvas_to_image(self, x: float, y: float) -> Tuple[float, float]:
        return (
            (float(x) - self._view.offset_x) / self._view.scale,
            (float(y) - self._view.offset_y) / self._view.scale,
        )

    def _clamp_to_image(self, x: float, y: float) -> Point:
        if self._base_image is None:
            return (float(x), float(y))
        iw, ih = self._base_image.size
        return (
            max(0.0, min(float(iw - 1), float(x))),
            max(0.0, min(float(ih - 1), float(y))),
        )
