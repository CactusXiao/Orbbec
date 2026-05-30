from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
import tkinter.font as tkfont

from PIL import Image, ImageTk


@dataclass
class ViewState:
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


class ImageAnnotatorCanvas(tk.Canvas):
    def __init__(self, master, *, bg: str, **kwargs):
        super().__init__(master, bg=bg, highlightthickness=0, **kwargs)

        self._img_path: Optional[Path] = None
        self._base_image: Optional[Image.Image] = None
        self._imgtk: Optional[ImageTk.PhotoImage] = None
        self._img_item: Optional[int] = None

        self._view = ViewState(scale=1.0, offset_x=0.0, offset_y=0.0)
        self._points: List[Tuple[int, int]] = []
        self._point_items: List[int] = []
        self._message_item: Optional[int] = None

        self._dragging = False
        self._drag_last: Tuple[int, int] = (0, 0)

        self.bind("<Button-1>", self._on_left_click)
        self.bind("<ButtonPress-3>", self._on_right_down)
        self.bind("<B3-Motion>", self._on_right_drag)
        self.bind("<ButtonRelease-3>", self._on_right_up)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", self._on_wheel_linux)
        self.bind("<Button-5>", self._on_wheel_linux)
        self.bind("<Configure>", self._on_resize)

    def clear(self) -> None:
        self._img_path = None
        self._base_image = None
        self._imgtk = None
        if self._img_item is not None:
            self.delete(self._img_item)
            self._img_item = None
        self._clear_points_items()
        self._points = []
        self._view = ViewState(scale=1.0, offset_x=0.0, offset_y=0.0)
        self._show_message("No image")

    def set_image(self, path: Optional[Path]) -> None:
        self._img_path = path
        self._base_image = None
        self._imgtk = None
        if self._img_item is not None:
            self.delete(self._img_item)
            self._img_item = None
        self._clear_points_items()

        if not path or not path.exists() or not path.is_file():
            self._show_message("Image not found")
            return

        try:
            with Image.open(path) as im:
                self._base_image = im.convert("RGB")
        except Exception:
            self._base_image = None
            self._show_message("Failed to load image")
            return

        self._hide_message()
        self._fit_to_canvas()
        self._render_image()
        self._render_points()

    def set_points(self, pts: List[Tuple[int, int]]) -> None:
        self._points = [(int(x), int(y)) for (x, y) in pts]
        self._render_points()

    def get_points(self) -> List[Tuple[int, int]]:
        return list(self._points)

    def undo_last_point(self) -> None:
        if not self._points:
            return
        self._points.pop()
        if self._point_items:
            item = self._point_items.pop()
            self.delete(item)

    def _clear_points_items(self) -> None:
        for item in self._point_items:
            self.delete(item)
        self._point_items = []

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

    def _on_resize(self, _evt) -> None:
        if self._base_image is None:
            if self._message_item is not None:
                self._show_message(self.itemcget(self._message_item, "text") or "")
            return
        self._render_image()
        self._render_points()

    def _on_left_click(self, evt) -> None:
        if self._base_image is None:
            return
        x_img, y_img = self._canvas_to_image(evt.x, evt.y)
        if x_img is None or y_img is None:
            return
        self._points.append((x_img, y_img))
        self._render_points(last_only=True)

    def _on_right_down(self, evt) -> None:
        self._dragging = True
        self._drag_last = (evt.x, evt.y)

    def _on_right_drag(self, evt) -> None:
        if not self._dragging:
            return
        dx = evt.x - self._drag_last[0]
        dy = evt.y - self._drag_last[1]
        self._drag_last = (evt.x, evt.y)
        self._view.offset_x += dx
        self._view.offset_y += dy
        self._render_image()
        self._render_points()

    def _on_right_up(self, _evt) -> None:
        self._dragging = False

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
        iw, ih = self._base_image.size
        scale = min(w / max(1, iw), h / max(1, ih))
        scale = max(0.05, min(10.0, scale))
        self._view.scale = scale
        self._view.offset_x = (w - iw * scale) / 2
        self._view.offset_y = (h - ih * scale) / 2

    def _zoom_at(self, cx: int, cy: int, factor: float) -> None:
        if self._base_image is None:
            return
        old = self._view.scale
        new = max(0.05, min(10.0, old * factor))
        if math.isclose(new, old):
            return

        x_img, y_img = self._canvas_to_image(cx, cy)
        if x_img is None or y_img is None:
            x_img = 0
            y_img = 0

        self._view.scale = new
        self._view.offset_x = cx - x_img * new
        self._view.offset_y = cy - y_img * new
        self._render_image()
        self._render_points()

    def _render_image(self) -> None:
        if self._base_image is None:
            return
        iw, ih = self._base_image.size
        scale = self._view.scale
        tw = max(1, int(iw * scale))
        th = max(1, int(ih * scale))
        img = self._base_image.resize((tw, th), Image.BILINEAR)
        self._imgtk = ImageTk.PhotoImage(img)
        x = self._view.offset_x
        y = self._view.offset_y
        if self._img_item is None:
            self._img_item = self.create_image(x, y, anchor="nw", image=self._imgtk)
        else:
            self.coords(self._img_item, x, y)
            self.itemconfigure(self._img_item, image=self._imgtk)
        self.tag_lower(self._img_item)

    def _render_points(self, *, last_only: bool = False) -> None:
        if self._base_image is None:
            return
        if not last_only:
            self._clear_points_items()
        pts = self._points[-1:] if last_only else self._points
        for (x, y) in pts:
            cx, cy = self._image_to_canvas(x, y)
            r = 3
            item = self.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#ff3333", outline="#ff3333")
            self._point_items.append(item)

    def _image_to_canvas(self, x_img: int, y_img: int) -> Tuple[float, float]:
        return (
            self._view.offset_x + float(x_img) * self._view.scale,
            self._view.offset_y + float(y_img) * self._view.scale,
        )

    def _canvas_to_image(self, x: float, y: float) -> Tuple[Optional[int], Optional[int]]:
        if self._base_image is None:
            return (None, None)
        iw, ih = self._base_image.size
        ix = (x - self._view.offset_x) / self._view.scale
        iy = (y - self._view.offset_y) / self._view.scale
        if ix < 0 or iy < 0 or ix >= iw or iy >= ih:
            return (None, None)
        return (int(round(ix)), int(round(iy)))
