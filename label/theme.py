from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk


class Theme:
    BG = "#161616"
    PANEL = "#1e1e1e"
    PANEL_2 = "#222222"
    FG = "#e6e6e6"
    MUTED = "#bdbdbd"
    ACCENT = "#3a82f7"
    BORDER = "#2f2f2f"
    BTN = "#2a2a2a"
    BTN_HOVER = "#333333"

    FONT_FAMILY = ""
    FONT_SIZE = 11


def apply_theme(root: tk.Tk) -> None:
    families = set()
    try:
        families = set(tkfont.families(root))
    except Exception:
        families = set()

    base_font = tkfont.nametofont("TkDefaultFont")
    try:
        default_family = base_font.cget("family")
    except Exception:
        default_family = ""

    if sys.platform == "darwin":
        candidates = [
            "PingFang SC",
            "Heiti SC",
            "Songti SC",
            "Helvetica Neue",
            default_family,
        ]
        size = 12
    elif sys.platform.startswith("win"):
        candidates = [
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "SimHei",
            "Segoe UI",
            default_family,
        ]
        size = 11
    else:
        candidates = [
            "Noto Sans SC",
            "Noto Sans CJK SC",
            "Noto Sans CJK HK",
            "Noto Sans CJK",
            "Noto Serif CJK SC",
            "Source Han Sans CN",
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "AR PL UKai CN",
            "AR PL UMing CN",
            "AR PL UMing TW MBE",
            "Noto Sans",
            "DejaVu Sans",
            default_family,
        ]
        size = 11

    chosen = ""
    for c in candidates:
        if not c:
            continue
        if families and c not in families:
            continue
        chosen = c
        break
    if not chosen:
        for c in candidates:
            if c:
                chosen = c
                break
    Theme.FONT_FAMILY = chosen or default_family or ""
    Theme.FONT_SIZE = size

    if Theme.FONT_FAMILY:
        try:
            base_font.configure(family=Theme.FONT_FAMILY)
        except Exception:
            pass
    try:
        base_font.configure(size=Theme.FONT_SIZE)
    except Exception:
        pass

    for name in (
        "TkDefaultFont",
        "TkTextFont",
        "TkFixedFont",
        "TkMenuFont",
        "TkHeadingFont",
        "TkCaptionFont",
        "TkSmallCaptionFont",
        "TkIconFont",
        "TkTooltipFont",
    ):
        try:
            f = tkfont.nametofont(name)
            if Theme.FONT_FAMILY:
                f.configure(family=Theme.FONT_FAMILY)
            f.configure(size=Theme.FONT_SIZE)
        except Exception:
            pass

    root.option_add("*Font", base_font)

    ui_font = (Theme.FONT_FAMILY, Theme.FONT_SIZE) if Theme.FONT_FAMILY else ("", Theme.FONT_SIZE)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure("TFrame", background=Theme.BG)
    style.configure("Panel.TFrame", background=Theme.PANEL)
    style.configure("Panel2.TFrame", background=Theme.PANEL_2)
    style.configure("TLabel", background=Theme.BG, foreground=Theme.FG, font=ui_font)
    style.configure("Muted.TLabel", background=Theme.BG, foreground=Theme.MUTED, font=ui_font)

    style.configure(
        "TEntry",
        fieldbackground=Theme.PANEL_2,
        foreground=Theme.FG,
        insertcolor=Theme.FG,
        bordercolor=Theme.BORDER,
        lightcolor=Theme.BORDER,
        darkcolor=Theme.BORDER,
        font=ui_font,
    )

    style.configure(
        "Primary.TButton",
        background=Theme.ACCENT,
        foreground="#ffffff",
        borderwidth=0,
        focusthickness=0,
        focuscolor=Theme.ACCENT,
        padding=(14, 8),
        font=ui_font,
    )
    style.map(
        "Primary.TButton",
        background=[("active", "#2f6fe0")],
    )

    style.configure(
        "Secondary.TButton",
        background=Theme.BTN,
        foreground=Theme.FG,
        borderwidth=0,
        focusthickness=0,
        padding=(14, 8),
        font=ui_font,
    )
    style.map(
        "Secondary.TButton",
        background=[("active", Theme.BTN_HOVER)],
    )

    style.configure(
        "Small.TButton",
        background=Theme.BTN,
        foreground=Theme.FG,
        borderwidth=0,
        focusthickness=0,
        padding=(10, 6),
        font=ui_font,
    )
    style.map("Small.TButton", background=[("active", Theme.BTN_HOVER)])

    style.configure(
        "Treeview",
        background=Theme.PANEL_2,
        foreground=Theme.FG,
        fieldbackground=Theme.PANEL_2,
        bordercolor=Theme.BORDER,
        rowheight=26,
        font=ui_font,
    )
    style.configure(
        "Treeview.Heading",
        background=Theme.PANEL,
        foreground=Theme.MUTED,
        relief="flat",
        font=ui_font,
    )
    style.map(
        "Treeview",
        background=[("selected", "#2b4b78")],
        foreground=[("selected", "#ffffff")],
    )
