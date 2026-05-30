from __future__ import annotations

import locale
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

try:
    from .canvas_view import ImageAnnotatorCanvas
    from .storage import (
        TaskRecord,
        compute_total_frames,
        clear_labels_for_frame,
        discover_tasks,
        ensure_record_csv,
        find_frame_path,
        list_camera_ids,
        save_record_csv,
        session_dir,
        update_labels_for_frame,
    )
    from .theme import Theme, apply_theme
except Exception:
    from canvas_view import ImageAnnotatorCanvas
    from storage import (
        TaskRecord,
        compute_total_frames,
        clear_labels_for_frame,
        discover_tasks,
        ensure_record_csv,
        find_frame_path,
        list_camera_ids,
        save_record_csv,
        session_dir,
        update_labels_for_frame,
    )
    from theme import Theme, apply_theme


class LabelToolApp(tk.Tk):
    def __init__(self):
        self._ensure_utf8_env()
        super().__init__()

        try:
            self.tk.call("encoding", "system", "utf-8")
        except Exception:
            pass

        self._is_maximized = False
        self._restore_geom: Optional[str] = None

        self.configure(bg=Theme.BG)
        self.title("Joint Label Tool")
        self.geometry("1200x780")
        self.minsize(980, 640)

        apply_theme(self)

        self._root_container = ttk.Frame(self, style="TFrame")
        self._root_container.pack(fill="both", expand=True)

        self._titlebar = ttk.Frame(self._root_container, style="Panel.TFrame")
        self._titlebar.pack(fill="x")
        self._build_titlebar(self._titlebar)

        self._page_host = ttk.Frame(self._root_container, style="TFrame")
        self._page_host.pack(fill="both", expand=True)

        self._home = HomePage(self._page_host, on_exit=self._on_exit, on_enter=self._go_label)
        self._label = LabelPage(self._page_host, on_back=self._go_home)
        self._home.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._label.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._go_home()

    @staticmethod
    def _ensure_utf8_env() -> None:
        try:
            enc = (locale.getpreferredencoding(False) or "").lower()
            lang = (os.environ.get("LC_ALL") or os.environ.get("LANG") or "").upper()
            need_fix = enc != "utf-8" or ("UTF-8" not in lang and "UTF8" not in lang)
            if need_fix:
                os.environ["LC_ALL"] = "C.UTF-8"
                os.environ["LANG"] = "C.UTF-8"
                locale.setlocale(locale.LC_ALL, "")
        except Exception:
            pass

    def _build_titlebar(self, bar: ttk.Frame) -> None:
        bar.configure(height=38)
        bar.pack_propagate(False)

        left = ttk.Frame(bar, style="Panel.TFrame")
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(bar, style="Panel.TFrame")
        right.pack(side="right", fill="y")

        title = ttk.Label(left, text="Joint Label Tool", foreground=Theme.FG, background=Theme.PANEL)
        title.pack(side="left", padx=12)

        ui_font = (Theme.FONT_FAMILY, Theme.FONT_SIZE) if Theme.FONT_FAMILY else None
        btn_min = tk.Label(right, text="-", fg=Theme.FG, bg=Theme.PANEL, width=3, font=ui_font)
        btn_max = tk.Label(right, text="[]", fg=Theme.FG, bg=Theme.PANEL, width=3, font=ui_font)
        btn_close = tk.Label(right, text="x", fg=Theme.FG, bg=Theme.PANEL, width=3, font=ui_font)
        btn_min.pack(side="left")
        btn_max.pack(side="left")
        btn_close.pack(side="left")

        btn_min.bind("<Button-1>", lambda _e: self._do_minimize())
        btn_max.bind("<Button-1>", lambda _e: self._toggle_maximize())
        btn_close.bind("<Button-1>", lambda _e: self._on_exit())

        for b in (btn_min, btn_max, btn_close):
            b.bind("<Enter>", lambda e, w=b: w.configure(bg=Theme.BTN_HOVER))
            b.bind("<Leave>", lambda e, w=b: w.configure(bg=Theme.PANEL))

    def _do_minimize(self) -> None:
        self.iconify()

    def _toggle_maximize(self) -> None:
        try:
            tk_state = self.state()
        except Exception:
            tk_state = "normal"
        zoomed = tk_state == "zoomed"

        if not self._is_maximized and not zoomed:
            self._restore_geom = self.geometry()
            try:
                self.state("zoomed")
            except Exception:
                sw = self.winfo_screenwidth()
                sh = self.winfo_screenheight()
                self.geometry(f"{sw}x{sh}+0+0")
            self._is_maximized = True
        else:
            try:
                self.state("normal")
            except Exception:
                pass
            if self._restore_geom:
                try:
                    self.update_idletasks()
                    self.geometry(self._restore_geom)
                except Exception:
                    pass
            self._is_maximized = False

    def _on_exit(self) -> None:
        self.destroy()

    def _go_home(self) -> None:
        self._label.on_hide()
        self._label.lower()
        self._home.lift()

    def _go_label(self, base_dir: str, subject: str) -> None:
        ok = self._label.set_session(base_dir=base_dir, subject=subject)
        if not ok:
            return
        self._home.lower()
        self._label.lift()


class HomePage(ttk.Frame):
    def __init__(self, master, *, on_exit, on_enter):
        super().__init__(master, style="TFrame")
        self._on_exit = on_exit
        self._on_enter = on_enter

        outer = ttk.Frame(self, style="TFrame")
        outer.pack(fill="both", expand=True)

        center = ttk.Frame(outer, style="Panel.TFrame", padding=(28, 22))
        center.place(relx=0.5, rely=0.5, anchor="center")

        title = ttk.Label(center, text="Joint Labeling", style="TLabel")
        title.pack(pady=(26, 16))

        form = ttk.Frame(center, style="Panel.TFrame")
        form.pack(padx=28, fill="x")

        self._var_dir = tk.StringVar()
        self._var_subj = tk.StringVar()

        ttk.Label(form, text="Data Directory", style="Muted.TLabel").pack(anchor="w")
        ttk.Entry(form, textvariable=self._var_dir).pack(fill="x", pady=(6, 14))

        ttk.Label(form, text="Subject", style="Muted.TLabel").pack(anchor="w")
        ttk.Entry(form, textvariable=self._var_subj).pack(fill="x", pady=(6, 0))

        btns = ttk.Frame(center, style="Panel.TFrame")
        btns.pack(pady=(22, 8), fill="x")

        ttk.Button(btns, text="Exit", style="Secondary.TButton", command=self._on_exit, width=14).pack(side="left", padx=(0, 12))
        ttk.Button(btns, text="Start Labeling", style="Primary.TButton", command=self._enter, width=14).pack(side="left")

    def _enter(self) -> None:
        base = (self._var_dir.get() or "").strip()
        subj = (self._var_subj.get() or "").strip()
        if not base or not subj:
            messagebox.showwarning("Notice", "Please enter both Data Directory and Subject.")
            return
        self._on_enter(base, subj)


class LabelPage(ttk.Frame):
    def __init__(self, master, *, on_back):
        super().__init__(master, style="TFrame")
        self._on_back = on_back

        self._sess: Optional[Path] = None
        self._task_records: Dict[str, TaskRecord] = {}
        self._tasks: List[str] = []
        self._active_task: Optional[str] = None
        self._camera_ids: List[str] = []
        self._cam_idx: int = 0
        self._frame_idx: int = 0
        self._points_by_cam: Dict[str, List[Tuple[int, int]]] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="TFrame")
        root.pack(fill="both", expand=True, padx=16, pady=14)

        left = ttk.Frame(root, style="Panel.TFrame")
        left.pack(side="left", fill="y")
        right = ttk.Frame(root, style="Panel.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        ttk.Label(left, text="Tasks", style="TLabel").pack(anchor="w", padx=12, pady=(10, 8))

        cols = ("task", "done", "total")
        self._tree = ttk.Treeview(left, columns=cols, show="headings", height=22)
        self._tree.heading("task", text="Task")
        self._tree.heading("done", text="Done")
        self._tree.heading("total", text="Total")
        self._tree.column("task", width=240, anchor="w")
        self._tree.column("done", width=70, anchor="center")
        self._tree.column("total", width=70, anchor="center")
        self._tree["displaycolumns"] = ("task", "done", "total")
        self._tree.pack(side="left", fill="y", padx=(12, 0), pady=(0, 12))

        ysb = ttk.Scrollbar(left, orient="vertical", command=self._tree.yview)
        ysb.pack(side="left", fill="y", padx=(8, 12), pady=(0, 12))
        self._tree.configure(yscrollcommand=ysb.set)
        self._tree.bind("<<TreeviewSelect>>", self._on_task_select)

        top = ttk.Frame(right, style="Panel.TFrame")
        top.pack(fill="x", padx=12, pady=(10, 8))
        self._info = ttk.Label(top, text="", style="Muted.TLabel")
        self._info.pack(side="left")

        canvas_host = ttk.Frame(right, style="Panel2.TFrame")
        canvas_host.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self._canvas = ImageAnnotatorCanvas(canvas_host, bg=Theme.PANEL_2)
        self._canvas.pack(fill="both", expand=True)

        btn_row = ttk.Frame(right, style="Panel.TFrame")
        btn_row.pack(fill="x", padx=12)
        ttk.Button(btn_row, text="<<", style="Small.TButton", command=self._back_frame).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text=">>", style="Small.TButton", command=self._skip_frame).pack(side="left", padx=(0, 16))
        ttk.Button(btn_row, text="←", style="Small.TButton", command=self._prev_cam).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="→", style="Small.TButton", command=self._next_cam).pack(side="left", padx=(0, 16))
        ttk.Button(btn_row, text="Undo", style="Small.TButton", command=self._undo).pack(side="left", padx=(0, 16))
        ttk.Button(btn_row, text="Confirm", style="Primary.TButton", command=self._confirm).pack(side="left")

        bottom = ttk.Frame(right, style="Panel.TFrame")
        bottom.pack(fill="x", padx=12, pady=(10, 12))
        ttk.Button(bottom, text="Back to Home", style="Secondary.TButton", command=self._back_home).pack(side="left")

    def on_hide(self) -> None:
        self._canvas.clear()
        self._sess = None
        self._task_records = {}
        self._tasks = []
        self._active_task = None
        self._camera_ids = []
        self._cam_idx = 0
        self._frame_idx = 0
        self._points_by_cam = {}

    def set_session(self, *, base_dir: str, subject: str) -> bool:
        sess = session_dir(base_dir, subject)
        if not sess.exists() or not sess.is_dir():
            messagebox.showerror("Error", f"Session directory not found: {sess}")
            return False

        tasks = discover_tasks(sess)
        records = ensure_record_csv(sess, tasks)

        self._sess = sess
        self._tasks = tasks
        self._task_records = records

        for item in self._tree.get_children():
            self._tree.delete(item)
        for t in self._tasks:
            r = self._task_records.get(t)
            done = r.done_frame if r else 0
            total = r.total_frames if r else 0
            self._tree.insert("", "end", iid=t, values=(t, done, total))

        self._info.configure(text=f"Session: {sess}")
        self._active_task = None
        self._canvas.clear()
        return True

    def _back_home(self) -> None:
        if self._ask_yes_no("Notice", "Going back will discard unsaved labels. Continue?"):
            self._on_back()

    def _ask_yes_no(self, title: str, message: str) -> bool:
        root = self.winfo_toplevel()
        dialog = tk.Toplevel(root)
        dialog.title(title)
        dialog.configure(bg=Theme.BG)
        dialog.resizable(False, False)
        try:
            dialog.transient(root)
        except Exception:
            pass

        outer = ttk.Frame(dialog, style="Panel.TFrame", padding=(18, 14))
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text=message, style="TLabel", wraplength=460, justify="left").pack(anchor="w")

        result = {"ok": False}

        def set_result(v: bool) -> None:
            result["ok"] = bool(v)
            try:
                dialog.grab_release()
            except Exception:
                pass
            dialog.destroy()

        btns = ttk.Frame(outer, style="Panel.TFrame")
        btns.pack(fill="x", pady=(14, 0))
        ttk.Button(btns, text="No", style="Secondary.TButton", command=lambda: set_result(False), width=10).pack(
            side="right", padx=(10, 0)
        )
        ttk.Button(btns, text="Yes", style="Primary.TButton", command=lambda: set_result(True), width=10).pack(
            side="right"
        )

        dialog.protocol("WM_DELETE_WINDOW", lambda: set_result(False))
        try:
            dialog.grab_set()
        except Exception:
            pass

        dialog.update_idletasks()
        try:
            w = dialog.winfo_width()
            h = dialog.winfo_height()
            x = root.winfo_rootx() + max(0, (root.winfo_width() - w) // 2)
            y = root.winfo_rooty() + max(0, (root.winfo_height() - h) // 2)
            dialog.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

        dialog.wait_window(dialog)
        return bool(result["ok"])

    def _on_task_select(self, _evt) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        task = sel[0]
        self._load_task(task)

    def _load_task(self, task: str) -> None:
        if not self._sess:
            return
        if task not in self._tasks:
            return

        self._active_task = task
        rec = self._task_records.get(task)
        done = rec.done_frame if rec else 0
        total = rec.total_frames if rec else compute_total_frames(self._sess / task)
        self._frame_idx = max(0, min(done, max(0, total - 1))) if total > 0 else 0

        self._camera_ids = list_camera_ids(self._sess / task)
        self._camera_ids.sort()
        self._cam_idx = 0
        self._points_by_cam = {cid: [] for cid in self._camera_ids}

        self._refresh_view()

    def _active_cam_id(self) -> Optional[str]:
        if not self._camera_ids:
            return None
        self._cam_idx = max(0, min(self._cam_idx, len(self._camera_ids) - 1))
        return self._camera_ids[self._cam_idx]

    def _refresh_view(self) -> None:
        if not self._sess or not self._active_task:
            self._canvas.clear()
            return
        cam_id = self._active_cam_id()
        if cam_id is None:
            self._canvas.clear()
            return

        task_dir = self._sess / self._active_task
        img_path = find_frame_path(task_dir, cam_id, self._frame_idx)
        self._canvas.set_image(img_path)
        self._canvas.set_points(self._points_by_cam.get(cam_id, []))

        rec = self._task_records.get(self._active_task)
        done = rec.done_frame if rec else 0
        total = rec.total_frames if rec else 0
        self._info.configure(
            text=f"Task: {self._active_task}    Camera: {cam_id} ({self._cam_idx + 1}/{len(self._camera_ids)})    Frame: {self._frame_idx}/{max(0, total - 1)}    Done: {done}/{total}"
        )

    def _commit_current_cam_points(self) -> None:
        cam_id = self._active_cam_id()
        if cam_id is None:
            return
        self._points_by_cam[cam_id] = self._canvas.get_points()

    def _prev_cam(self) -> None:
        if not self._camera_ids:
            return
        self._commit_current_cam_points()
        self._cam_idx = (self._cam_idx - 1) % len(self._camera_ids)
        self._refresh_view()

    def _next_cam(self) -> None:
        if not self._camera_ids:
            return
        self._commit_current_cam_points()
        self._cam_idx = (self._cam_idx + 1) % len(self._camera_ids)
        self._refresh_view()

    def _undo(self) -> None:
        self._canvas.undo_last_point()

    def _skip_frame(self) -> None:
        if not self._sess or not self._active_task:
            return
        if not self._camera_ids:
            return

        rec = self._task_records.get(self._active_task)
        total = rec.total_frames if rec else 0

        update_labels_for_frame(self._sess, self._active_task, {cid: [] for cid in self._camera_ids}, self._frame_idx)

        new_done = min(self._frame_idx + 1, total) if total > 0 else (self._frame_idx + 1)
        self._task_records[self._active_task] = TaskRecord(
            task_name=self._active_task,
            done_frame=int(new_done),
            total_frames=int(total),
        )
        save_record_csv(self._sess, list(self._task_records.values()))
        self._tree.set(self._active_task, "done", str(new_done))
        self._tree.set(self._active_task, "total", str(total))

        if total > 0 and new_done >= total:
            messagebox.showinfo("Done", "This task is fully completed.")
            self._points_by_cam = {cid: [] for cid in self._camera_ids}
            self._refresh_view()
            return

        self._frame_idx = int(new_done)
        self._points_by_cam = {cid: [] for cid in self._camera_ids}
        self._cam_idx = 0
        self._refresh_view()

    def _back_frame(self) -> None:
        if not self._sess or not self._active_task:
            return
        if not self._camera_ids:
            return
        if self._frame_idx <= 0:
            self._points_by_cam = {cid: [] for cid in self._camera_ids}
            self._cam_idx = 0
            self._refresh_view()
            return

        target_idx = int(self._frame_idx - 1)

        clear_labels_for_frame(self._sess, self._active_task, self._camera_ids, target_idx)

        rec = self._task_records.get(self._active_task)
        total = rec.total_frames if rec else 0
        new_done = max(0, min(target_idx, total)) if total > 0 else max(0, target_idx)
        self._task_records[self._active_task] = TaskRecord(
            task_name=self._active_task,
            done_frame=int(new_done),
            total_frames=int(total),
        )
        save_record_csv(self._sess, list(self._task_records.values()))
        self._tree.set(self._active_task, "done", str(new_done))
        self._tree.set(self._active_task, "total", str(total))

        self._frame_idx = int(new_done)
        self._points_by_cam = {cid: [] for cid in self._camera_ids}
        self._cam_idx = 0
        self._refresh_view()

    def _confirm(self) -> None:
        if not self._sess or not self._active_task:
            return
        if not self._camera_ids:
            messagebox.showwarning("Notice", "No camera directories found for the current task.")
            return

        self._commit_current_cam_points()
        if not self._ask_yes_no("Confirm", "Confirm that all views have been labeled correctly?"):
            return

        update_labels_for_frame(self._sess, self._active_task, self._points_by_cam, self._frame_idx)

        rec = self._task_records.get(self._active_task)
        total = rec.total_frames if rec else 0
        new_done = min(self._frame_idx + 1, total) if total > 0 else (self._frame_idx + 1)
        self._task_records[self._active_task] = TaskRecord(
            task_name=self._active_task,
            done_frame=int(new_done),
            total_frames=int(total),
        )
        save_record_csv(self._sess, list(self._task_records.values()))

        self._tree.set(self._active_task, "done", str(new_done))
        self._tree.set(self._active_task, "total", str(total))

        if total > 0 and new_done >= total:
            messagebox.showinfo("Done", "This task is fully completed.")
            self._points_by_cam = {cid: [] for cid in self._camera_ids}
            self._refresh_view()
            return

        self._frame_idx = int(new_done)
        self._points_by_cam = {cid: [] for cid in self._camera_ids}
        self._cam_idx = 0
        self._refresh_view()
