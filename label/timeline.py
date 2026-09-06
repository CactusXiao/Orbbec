"""A draggable frame strip with explicit confirmed/unconfirmed coverage."""
from __future__ import annotations

import tkinter as tk

from .theme import Theme


class LabelTimeline(tk.Canvas):
    DONE = "#46d36b"
    TODO = "#e7a34b"

    def __init__(self, master, *, on_seek):
        super().__init__(master, height=36, bg=Theme.PANEL, highlightthickness=0,
                         cursor="hand2", takefocus=True)
        self._frames = []
        self._position = 0
        self._done = set()
        self._dragging = False
        self._on_seek = on_seek
        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Button-1>", self._begin_drag)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._end_drag)
        self.bind("<Left>", lambda _event: self._keyboard_seek(self._position - 1))
        self.bind("<Right>", lambda _event: self._keyboard_seek(self._position + 1))
        self.bind("<Home>", lambda _event: self._keyboard_seek(0))
        self.bind("<End>", lambda _event: self._keyboard_seek(len(self._frames) - 1))

    def set_data(self, *, frames, position, done):
        self._frames = list(frames)
        self._position = max(0, min(int(position), len(self._frames) - 1))
        self._done = set(done)
        self._dragging = False
        self._draw()

    def _x_for_position(self, position):
        width = max(1.0, self.winfo_width() - 16.0)
        return 8.0 + width * (position + 0.5) / max(1, len(self._frames))

    def _position_from_event(self, event):
        ratio = (event.x - 8.0) / max(1.0, self.winfo_width() - 16.0)
        return max(0, min(len(self._frames) - 1, int(ratio * len(self._frames))))

    def _draw(self):
        self.delete("all")
        total = len(self._frames)
        if not total:
            return
        width = max(1.0, self.winfo_width() - 16.0)
        start = 0
        confirmed = 0 in self._done
        for end in range(1, total + 1):
            if end < total and (end in self._done) == confirmed:
                continue
            self.create_rectangle(8 + width * start / total, 12,
                                  8 + width * end / total, 24,
                                  fill=self.DONE if confirmed else self.TODO,
                                  outline="", tags="done" if confirmed else "todo")
            start, confirmed = end, end in self._done
        x = self._x_for_position(self._position)
        self.create_line(x, 5, x, 31, fill="white", width=2, tags="cursor")
        self.create_oval(x - 5, 12, x + 5, 24, fill="white", outline=Theme.ACCENT,
                         width=2, tags="cursor")

    def _move(self, position, final):
        self._position = max(0, min(position, len(self._frames) - 1))
        self._draw()
        self._on_seek(self._position, final)

    def _begin_drag(self, event):
        if not self._frames:
            return
        self.focus_set()
        self._dragging = True
        self._move(self._position_from_event(event), False)

    def _drag(self, event):
        if self._dragging:
            self._move(self._position_from_event(event), False)

    def _end_drag(self, event):
        if self._dragging:
            self._dragging = False
            self._move(self._position_from_event(event), True)

    def _keyboard_seek(self, position):
        if self._frames:
            self._move(position, True)
        return "break"
