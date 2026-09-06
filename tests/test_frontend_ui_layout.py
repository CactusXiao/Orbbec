"""Display-backed layout checks; no jobs are leased and no backend is contacted."""
from __future__ import annotations

import tempfile
import tkinter as tk
import tkinter.font as tkfont
import unittest
from pathlib import Path
from tkinter import ttk
from unittest.mock import patch

from label.app import LabelToolApp
from label.theme import Theme, WrapToolbar
from src.qc.app import QcWorkerApp
from src.qc.config import load_qc_config


class FrontendLayoutTest(unittest.TestCase):
    def setUp(self):
        try:
            probe = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"A Tk display is required: {exc}")
        probe.destroy()
        self.app = None

    def tearDown(self):
        if self.app is not None:
            self.app.destroy()

    def settle(self):
        self.app.update_idletasks()
        self.app.update()
        self.app.update_idletasks()

    def assert_actions_visible(self, parent):
        for widget in parent.winfo_children():
            if isinstance(widget, WrapToolbar) and widget.winfo_ismapped():
                rectangles = []
                for child in widget.winfo_children():
                    self.assertTrue(child.winfo_ismapped())
                    self.assertGreaterEqual(child.winfo_width(), child.winfo_reqwidth())
                    x, y = child.winfo_rootx(), child.winfo_rooty()
                    right, bottom = x + child.winfo_width(), y + child.winfo_height()
                    self.assertLessEqual(right, self.app.winfo_rootx() + self.app.winfo_width())
                    self.assertLessEqual(bottom, self.app.winfo_rooty() + self.app.winfo_height())
                    for a, b, c, d in rectangles:
                        self.assertTrue(right <= a or x >= c or bottom <= b or y >= d, "Controls overlap")
                    rectangles.append((x, y, right, bottom))
            self.assert_actions_visible(widget)

    def test_label_readable_type_and_wrapping(self):
        self.app = LabelToolApp()
        self.settle()
        style = ttk.Style(self.app)
        title = tkfont.Font(root=self.app, font=style.lookup("Title.TLabel", "font"))
        body = tkfont.nametofont("TkDefaultFont", root=self.app)
        self.assertGreater(title.metrics("linespace"), body.metrics("linespace"))
        self.assertGreaterEqual(int(style.lookup("Treeview", "rowheight")), body.metrics("linespace") + 12)
        self.assertEqual(body.cget("size"), Theme.FONT_SIZE)
        self.app._label.lift()
        for size in ("1040x680", "1480x920"):
            self.app.geometry(size)
            self.settle()
            self.assert_actions_visible(self.app._label)
            self.app._label._show_skeleton = True
            self.app._label._update_skeleton_button()
            self.settle()
            self.assert_actions_visible(self.app._label)

    def test_qc_playback_and_rejection_actions_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            config = load_qc_config(cwd=Path(temp))
            with patch.object(QcWorkerApp, "refresh_tasks", lambda self: None):
                self.app = QcWorkerApp(config)
            page = self.app.qc_page
            page.lift()
            page._build_camera_grid(["00", "02", "03", "05", "ego"])
            for size in ("1040x680", "1480x920"):
                self.app.geometry(size)
                for mode in ("playback", "bad_range", "playback"):
                    page.mode = mode
                    page._update_controls()
                    self.settle()
                    self.assert_actions_visible(page)
                    for canvas in page._canvases.values():
                        self.assertGreater(canvas.winfo_width(), 150)
                        self.assertGreater(canvas.winfo_height(), 100)


if __name__ == "__main__":
    unittest.main()
