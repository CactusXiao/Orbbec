from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk


class Theme:
    BG = "#101722"
    PANEL = "#182231"
    PANEL_2 = "#202d3e"
    FG = "#f0f4fa"
    MUTED = "#b4c3d5"
    ACCENT = "#2563eb"
    BORDER = "#3b4d64"
    BTN = "#29394e"
    BTN_HOVER = "#364c68"
    FONT_FAMILY = ""
    FONT_SIZE = 13


def apply_theme(root: tk.Tk) -> None:
    # Some remote X desktops report ~66 DPI (0.916 px/pt). Never shrink text
    # below 96 DPI; retain the desktop's scaling on real high-DPI displays.
    if sys.platform != "darwin":
        root.tk.call("tk", "scaling", max(96 / 72, float(root.tk.call("tk", "scaling"))))
    families = {name.casefold(): name for name in tkfont.families(root)}
    default_family = tkfont.nametofont("TkDefaultFont", root=root).cget("family")
    if sys.platform == "darwin":
        candidates = ["PingFang SC", "Heiti SC", "Helvetica Neue"]
    elif sys.platform.startswith("win"):
        candidates = ["Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI"]
    else:
        candidates = ["Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
                      "Source Han Sans CN", "WenQuanYi Micro Hei", "Droid Sans Fallback",
                      "Noto Sans", "DejaVu Sans"]
    Theme.FONT_FAMILY = next((families[n.casefold()] for n in candidates if n.casefold() in families), default_family)
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
                 "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont"):
        tkfont.nametofont(name, root=root).configure(family=Theme.FONT_FAMILY, size=Theme.FONT_SIZE)
    # Preserve the system monospace family for diagnostic text.
    tkfont.nametofont("TkFixedFont", root=root).configure(size=Theme.FONT_SIZE)
    root.option_add("*selectBackground", Theme.ACCENT)
    root.option_add("*selectForeground", "#ffffff")
    ui = (Theme.FONT_FAMILY, Theme.FONT_SIZE)
    bold = (Theme.FONT_FAMILY, Theme.FONT_SIZE, "bold")
    title = (Theme.FONT_FAMILY, 22, "bold")
    section = (Theme.FONT_FAMILY, 15, "bold")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=Theme.BG, foreground=Theme.FG, font=ui)
    for name, bg in (("TFrame", Theme.BG), ("Panel.TFrame", Theme.PANEL), ("Panel2.TFrame", Theme.PANEL_2)):
        style.configure(name, background=bg)
    for name, bg, fg, font in (
        ("TLabel", Theme.BG, Theme.FG, ui),
        ("Muted.TLabel", Theme.BG, Theme.MUTED, ui),
        ("Title.TLabel", Theme.BG, Theme.FG, title),
        ("Panel.TLabel", Theme.PANEL, Theme.FG, ui),
        ("PanelMuted.TLabel", Theme.PANEL, Theme.MUTED, ui),
        ("PanelTitle.TLabel", Theme.PANEL, Theme.FG, title),
        ("Section.TLabel", Theme.PANEL, Theme.FG, section),
    ):
        style.configure(name, background=bg, foreground=fg, font=font)
    style.configure("TEntry", fieldbackground=Theme.PANEL_2, foreground=Theme.FG,
                    insertcolor=Theme.FG, bordercolor=Theme.BORDER, lightcolor=Theme.BORDER,
                    darkcolor=Theme.BORDER, padding=(10, 7))
    style.map("TEntry", bordercolor=[("focus", "#60a5fa")])
    for name, bg, fg, padding in (
        ("TButton", Theme.BTN, Theme.FG, (14, 9)),
        ("Primary.TButton", Theme.ACCENT, "#ffffff", (18, 9)),
        ("Secondary.TButton", Theme.BTN, Theme.FG, (14, 9)),
        ("Small.TButton", Theme.BTN, Theme.FG, (12, 7)),
        ("Danger.TButton", "#572c3a", "#ffd5da", (14, 9)),
    ):
        style.configure(name, background=bg, foreground=fg, borderwidth=1, relief="flat",
                        bordercolor=bg, lightcolor=bg, darkcolor=bg, focusthickness=2, width=0,
                        focuscolor="#93c5fd", padding=padding, font=bold if name == "Primary.TButton" else ui)
        style.map(name, background=[("disabled", "#202b3a"), ("pressed", "#1d4ed8"),
                                   ("active", "#3478f6" if name == "Primary.TButton" else Theme.BTN_HOVER)],
                  foreground=[("disabled", "#8292a7")], bordercolor=[("focus", "#93c5fd")])
    line_height = tkfont.nametofont("TkDefaultFont", root=root).metrics("linespace")
    style.configure("Treeview", background=Theme.PANEL, foreground=Theme.FG,
                    fieldbackground=Theme.PANEL, bordercolor=Theme.BORDER, borderwidth=0,
                    lightcolor=Theme.BORDER, darkcolor=Theme.BORDER,
                    rowheight=line_height + 16, font=ui)
    style.configure("Treeview.Heading", background=Theme.PANEL_2, foreground=Theme.MUTED,
                    relief="flat", padding=(10, 10), font=bold)
    style.map("Treeview", background=[("selected", "#254d7b")], foreground=[("selected", "#ffffff")])
    style.map("Treeview.Heading", background=[("active", Theme.BTN_HOVER)])
    style.configure("Vertical.TScrollbar", background=Theme.BTN, troughcolor=Theme.PANEL,
                    bordercolor=Theme.PANEL, lightcolor=Theme.BTN, darkcolor=Theme.BTN,
                    arrowcolor=Theme.MUTED, width=16)
    style.map("Vertical.TScrollbar", background=[("active", Theme.BTN_HOVER), ("!active", Theme.BTN)])
    style.configure("TSeparator", background=Theme.BORDER)


def fit_window(root: tk.Tk, width: int = 1480, height: int = 920) -> None:
    """Keep the initial window inside the work area, including laptop screens."""
    width = min(width, max(800, root.winfo_screenwidth() - 80))
    height = min(height, max(600, root.winfo_screenheight() - 100))
    root.geometry(f"{width}x{height}")
    root.minsize(min(width, 1040), min(height, 680))


class WrapToolbar(ttk.Frame):
    """Wrap controls at their natural text width instead of clipping actions."""

    def __init__(self, master, **kwargs):
        super().__init__(master, style="Panel.TFrame", **kwargs)
        self._items = []
        self._pending = None
        self.bind("<Configure>", self._schedule)

    def add(self, widget):
        self._items.append(widget)
        widget.bind("<Configure>", self._schedule, add="+")
        self._schedule()
        return widget

    def _schedule(self, _event=None):
        if self._pending is None:
            self._pending = self.after_idle(self._layout)

    def _layout(self):
        self._pending = None
        available = max(1, self.winfo_width())
        x = y = row_height = 0
        for widget in self._items:
            width = min(available, widget.winfo_reqwidth())
            height = widget.winfo_reqheight()
            if x and x + width > available:
                x = 0
                y += row_height + 8
                row_height = 0
            widget.place(x=x, y=y, width=width, height=height)
            x += width + 8
            row_height = max(row_height, height)
        height = y + row_height
        if self.winfo_reqheight() != height:
            self.configure(height=height)
