from __future__ import annotations

import locale
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

try:
    from .backend_client import BackendClientError, LabelBackendClient, LabelJobSession, session_from_lease
    from .canvas_view import HandPoints, HandVisible, ImageAnnotatorCanvas
    from .hand_init import MediaPipeHandInitializer
    from .storage import (
        CorrectionProgress,
        CorrectionTask,
        PredictionBundle,
        apply_view_state_to_corrected,
        correction_task_from_backend_payload,
        ensure_correction_progress,
        find_frame_path,
        load_frame_points,
        load_frame_visibility,
        load_correction_progress,
        load_correction_tasks,
        load_prediction_bundle,
        save_corrected_array,
        save_correction_progress,
        source_frame_path,
        view_state_from_bundle,
    )
    from .theme import Theme, apply_theme
    from .mano_view import ManoMeshResult, ManoViewRuntime
    from .tracking import CoTrackerRuntime
except Exception:
    from backend_client import BackendClientError, LabelBackendClient, LabelJobSession, session_from_lease
    from canvas_view import HandPoints, HandVisible, ImageAnnotatorCanvas
    from hand_init import MediaPipeHandInitializer
    from storage import (
        CorrectionProgress,
        CorrectionTask,
        PredictionBundle,
        apply_view_state_to_corrected,
        correction_task_from_backend_payload,
        ensure_correction_progress,
        find_frame_path,
        load_frame_points,
        load_frame_visibility,
        load_correction_progress,
        load_correction_tasks,
        load_prediction_bundle,
        save_corrected_array,
        save_correction_progress,
        source_frame_path,
        view_state_from_bundle,
    )
    from theme import Theme, apply_theme
    from mano_view import ManoMeshResult, ManoViewRuntime
    from tracking import CoTrackerRuntime


ViewStateByCam = Dict[str, Tuple[HandPoints, HandVisible]]
SourceStateCache = Dict[Tuple[str, int, str, str], Tuple[HandPoints, HandVisible]]
SOURCE_ORDER = ("pred", "correct", "last", "tracking", "scratch")
SOURCE_LABELS = {
    "pred": "Pred",
    "correct": "Correct",
    "last": "Last",
    "tracking": "Tracking",
    "scratch": "Scratch",
}
STATUS_DONE_COLOR = "#46d36b"
STATUS_TODO_COLOR = "#ff5c5c"


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
        self.title("Joint Correction Tool")
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

        title = ttk.Label(left, text="Joint Correction Tool", foreground=Theme.FG, background=Theme.PANEL)
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

    def _go_label(self, session: Any) -> None:
        ok = self._label.set_session(session=session)
        if not ok:
            return
        self._home.lower()
        self._label.lift()


class HomePage(ttk.Frame):
    def __init__(self, master, *, on_exit, on_enter):
        super().__init__(master, style="TFrame")
        self._on_exit = on_exit
        self._on_enter = on_enter
        self._task_rows: Dict[str, Dict[str, Any]] = {}

        outer = ttk.Frame(self, style="TFrame")
        outer.pack(fill="both", expand=True)

        center = ttk.Frame(outer, style="Panel.TFrame", padding=(28, 22))
        center.place(relx=0.5, rely=0.5, anchor="center")

        title = ttk.Label(center, text="Joint Correction", style="TLabel")
        title.pack(pady=(26, 16))

        form = ttk.Frame(center, style="Panel.TFrame")
        form.pack(padx=28, fill="x")

        self._var_backend_url = tk.StringVar(value=os.environ.get("ORBBEC_TASK_BACKEND_URL", "http://127.0.0.1:8765"))
        self._var_operator = tk.StringVar(value=os.environ.get("ORBBEC_LABEL_OPERATOR_ID", os.environ.get("USER", "labeler_01")))
        self._var_jsonl = tk.StringVar()

        ttk.Label(form, text="Backend URL", style="Muted.TLabel").pack(anchor="w")
        ttk.Entry(form, textvariable=self._var_backend_url, width=58).pack(fill="x", pady=(6, 10))

        ttk.Label(form, text="Operator ID", style="Muted.TLabel").pack(anchor="w")
        ttk.Entry(form, textvariable=self._var_operator, width=58).pack(fill="x", pady=(6, 10))

        queue = ttk.Frame(center, style="Panel.TFrame")
        queue.pack(padx=28, fill="both", pady=(8, 0))
        ttk.Label(queue, text="Queued manual label tasks", style="Muted.TLabel").pack(anchor="w")

        queue_host = ttk.Frame(queue, style="Panel.TFrame")
        queue_host.pack(fill="both", pady=(6, 8))
        cols = ("task", "queued", "subjects", "frames")
        self._queue_tree = ttk.Treeview(queue_host, columns=cols, show="headings", height=8)
        self._queue_tree.heading("task", text="Task")
        self._queue_tree.heading("queued", text="Queued")
        self._queue_tree.heading("subjects", text="Subjects")
        self._queue_tree.heading("frames", text="Frames")
        self._queue_tree.column("task", width=220, anchor="w")
        self._queue_tree.column("queued", width=70, anchor="center")
        self._queue_tree.column("subjects", width=170, anchor="w")
        self._queue_tree.column("frames", width=70, anchor="center")
        self._queue_tree.pack(side="left", fill="both", expand=True)
        queue_scroll = ttk.Scrollbar(queue_host, orient="vertical", command=self._queue_tree.yview)
        queue_scroll.pack(side="left", fill="y", padx=(8, 0))
        self._queue_tree.configure(yscrollcommand=queue_scroll.set)
        self._queue_tree.bind("<Double-1>", self._lease_selected_task)

        self._queue_notice = ttk.Label(queue, text="Refresh to load queued tasks.", style="Muted.TLabel")
        self._queue_notice.pack(anchor="w")

        btns = ttk.Frame(center, style="Panel.TFrame")
        btns.pack(pady=(22, 8), fill="x")

        ttk.Button(btns, text="Exit", style="Secondary.TButton", command=self._on_exit, width=14).pack(side="left", padx=(0, 12))
        ttk.Button(btns, text="Refresh Tasks", style="Secondary.TButton", command=self._refresh_tasks, width=16).pack(side="left", padx=(0, 12))
        ttk.Button(btns, text="Get Selected Task", style="Primary.TButton", command=self._lease_selected_task, width=18).pack(side="left")

        legacy = ttk.Frame(center, style="Panel.TFrame")
        legacy.pack(padx=28, fill="x", pady=(18, 0))
        ttk.Label(legacy, text="Legacy/debug JSONL", style="Muted.TLabel").pack(anchor="w")
        ttk.Entry(legacy, textvariable=self._var_jsonl, width=58).pack(fill="x", pady=(6, 8))
        ttk.Button(legacy, text="Start Legacy JSONL", style="Secondary.TButton", command=self._enter_legacy, width=18).pack(anchor="w")

    def _refresh_tasks(self) -> None:
        backend_url = (self._var_backend_url.get() or "").strip()
        if not backend_url:
            messagebox.showwarning("Notice", "Please enter a backend URL.")
            return
        try:
            groups = LabelBackendClient(backend_url).queued_label_tasks()
        except BackendClientError as exc:
            self._queue_notice.configure(text=str(exc))
            messagebox.showerror("Backend", str(exc))
            return
        self._populate_task_rows(groups)

    def _populate_task_rows(self, groups: List[Dict[str, Any]]) -> None:
        for item_id in self._queue_tree.get_children():
            self._queue_tree.delete(item_id)
        self._task_rows = {}
        for index, group in enumerate(groups, 1):
            item_id = f"task_{index}"
            subjects = str(group.get("subject_summary") or "")
            if len(subjects) > 34:
                subjects = subjects[:31] + "..."
            values = (
                str(group.get("task_name") or ""),
                str(group.get("queued") or 0),
                subjects,
                str(group.get("frames") or 0),
            )
            self._queue_tree.insert("", "end", iid=item_id, values=values)
            self._task_rows[item_id] = dict(group)
        if groups:
            first = "task_1"
            self._queue_tree.selection_set(first)
            self._queue_tree.focus(first)
            self._queue_notice.configure(text=f"{len(groups)} task group(s) loaded.")
        else:
            self._queue_notice.configure(text="No queued manual label tasks.")

    def _lease_selected_task(self, _event: Any = None) -> None:
        backend_url = (self._var_backend_url.get() or "").strip()
        operator_id = (self._var_operator.get() or "").strip()
        if not backend_url:
            messagebox.showwarning("Notice", "Please enter a backend URL.")
            return
        if not operator_id:
            messagebox.showwarning("Notice", "Please enter an operator ID.")
            return
        selected = self._queue_tree.selection()
        if not selected:
            messagebox.showwarning("Notice", "Please select a queued task.")
            return
        task_row = self._task_rows.get(str(selected[0])) or {}
        task_name = str(task_row.get("task_name") or "").strip()
        if not task_name:
            messagebox.showwarning("Notice", "Selected task is invalid.")
            return
        try:
            session = session_from_lease(
                backend_url=backend_url,
                operator_id=operator_id,
                lease_seconds=600,
                task_name=task_name,
            )
        except (BackendClientError, ValueError) as exc:
            messagebox.showerror("Backend", str(exc))
            self._refresh_tasks()
            return
        self._on_enter(session)

    def _enter_legacy(self) -> None:
        jsonl_path = (self._var_jsonl.get() or "").strip()
        if not jsonl_path:
            messagebox.showwarning("Notice", "Please enter a tasks JSONL path.")
            return
        self._on_enter(jsonl_path)


class LabelPage(ttk.Frame):
    def __init__(self, master, *, on_back):
        super().__init__(master, style="TFrame")
        self._on_back = on_back

        self._jsonl_path: Optional[str] = None
        self._backend_session: Optional[LabelJobSession] = None
        self._backend_completed: bool = False
        self._heartbeat_after_id: Optional[str] = None
        self._mode: str = "pred"
        self._tasks: List[CorrectionTask] = []
        self._tasks_by_key: Dict[str, CorrectionTask] = {}
        self._progress: Dict[str, CorrectionProgress] = {}
        self._active_key: Optional[str] = None
        self._active_task: Optional[CorrectionTask] = None
        self._active_bundle: Optional[PredictionBundle] = None
        self._bundles: Dict[str, PredictionBundle] = {}
        self._camera_ids: List[str] = []
        self._cam_idx: int = 0
        self._frame_pos: int = 0
        self._view_states: ViewStateByCam = {}
        self._source_state_cache: SourceStateCache = {}
        self._hand_initializer = MediaPipeHandInitializer()
        self._tracker: Optional[CoTrackerRuntime] = None
        self._mano_runtime: Optional[ManoViewRuntime] = None
        self._mano_mesh: Optional[ManoMeshResult] = None
        self._show_mano: bool = False
        self._skeleton_joints_3d = None
        self._show_skeleton: bool = False
        self._tracking_notice_keys = set()
        self._source_btn: Optional[ttk.Button] = None
        self._skeleton_btn: Optional[ttk.Button] = None
        self._mano_btn: Optional[ttk.Button] = None
        self._frame_status: Optional[tk.Label] = None

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
        self._tree.column("task", width=300, anchor="w")
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
        self._info.pack(side="left", fill="x", expand=True)
        ui_font = (Theme.FONT_FAMILY, Theme.FONT_SIZE) if Theme.FONT_FAMILY else None
        self._frame_status = tk.Label(top, text="", fg=STATUS_TODO_COLOR, bg=Theme.PANEL, font=ui_font)
        self._frame_status.pack(side="right", padx=(12, 0))

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
        ttk.Button(btn_row, text="Undo", style="Small.TButton", command=self._undo).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Ignore View", style="Small.TButton", command=self._ignore_view).pack(side="left", padx=(0, 16))
        self._source_btn = ttk.Button(
            btn_row,
            text=self._source_button_text(),
            style="Small.TButton",
            command=self._toggle_source,
        )
        self._source_btn.pack(side="left", padx=(0, 16))
        self._skeleton_btn = ttk.Button(
            btn_row,
            text=self._skeleton_button_text(),
            style="Small.TButton",
            command=self._toggle_skeleton,
        )
        self._skeleton_btn.pack(side="left", padx=(0, 8))
        self._mano_btn = ttk.Button(
            btn_row,
            text=self._mano_button_text(),
            style="Small.TButton",
            command=self._toggle_mano,
        )
        self._mano_btn.pack(side="left", padx=(0, 16))
        ttk.Button(btn_row, text="Confirm", style="Primary.TButton", command=self._confirm).pack(side="left")

        bottom = ttk.Frame(right, style="Panel.TFrame")
        bottom.pack(fill="x", padx=12, pady=(10, 12))
        ttk.Button(bottom, text="Back to Home", style="Secondary.TButton", command=self._back_home).pack(side="left")

    def on_hide(self) -> None:
        self._cancel_backend_heartbeat()
        self._canvas.clear()
        self._jsonl_path = None
        self._backend_session = None
        self._backend_completed = False
        self._mode = "pred"
        self._tasks = []
        self._tasks_by_key = {}
        self._progress = {}
        self._active_key = None
        self._active_task = None
        self._active_bundle = None
        self._bundles = {}
        self._camera_ids = []
        self._cam_idx = 0
        self._frame_pos = 0
        self._view_states = {}
        self._source_state_cache = {}
        self._reset_visualizations()
        self._tracking_notice_keys = set()
        self._hand_initializer.close()
        if self._frame_status is not None:
            self._frame_status.configure(text="")

    def set_session(self, *, session: Any = None, jsonl_path: str = "") -> bool:
        if isinstance(session, LabelJobSession):
            return self._set_backend_session(session)
        return self._set_legacy_session(str(session or jsonl_path))

    def _set_legacy_session(self, jsonl_path: str) -> bool:
        p = Path(jsonl_path).expanduser().resolve()
        try:
            tasks = load_correction_tasks(str(p))
            progress = ensure_correction_progress(str(p), tasks)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return False

        self._backend_session = None
        self._backend_completed = False
        self._jsonl_path = str(p)
        return self._apply_session_tasks(tasks, progress, f"Tasks: {p}")

    def _set_backend_session(self, session: LabelJobSession) -> bool:
        try:
            task = correction_task_from_backend_payload(session.payload, mounts=session.mounts)
            progress_ref = task.episode_dir() / f".{session.job_id}.jsonl"
            progress = load_correction_progress(str(progress_ref), [task])
            save_correction_progress(str(progress_ref), progress)
        except Exception as exc:
            messagebox.showerror("Backend Task", str(exc))
            return False

        self._backend_session = session
        self._backend_completed = False
        self._jsonl_path = str(progress_ref)
        self._schedule_backend_heartbeat()
        return self._apply_session_tasks(
            [task],
            progress,
            f"Backend job: {session.job_id}",
        )

    def _apply_session_tasks(
        self,
        tasks: List[CorrectionTask],
        progress: Dict[str, CorrectionProgress],
        info_prefix: str,
    ) -> bool:
        self._mode = "pred"
        self._tasks = tasks
        self._tasks_by_key = {t.key: t for t in tasks}
        self._progress = progress
        self._active_key = None
        self._active_task = None
        self._active_bundle = None
        self._bundles = {}
        self._view_states = {}
        self._source_state_cache = {}
        self._reset_visualizations()
        self._tracking_notice_keys = set()
        self._update_source_button()

        for item in self._tree.get_children():
            self._tree.delete(item)
        for task in self._tasks:
            rec = self._progress.get(task.key)
            done = rec.done_count if rec else 0
            self._tree.insert("", "end", iid=task.key, values=(task.display_name, done, task.total_frames))

        self._info.configure(text=f"{info_prefix}    View Source: {self._source_label()}")
        if self._frame_status is not None:
            self._frame_status.configure(text="")
        self._canvas.clear()
        return True

    def _schedule_backend_heartbeat(self) -> None:
        self._cancel_backend_heartbeat()
        if self._backend_session is None or self._backend_completed:
            return
        self._heartbeat_after_id = self.after(30000, self._heartbeat_backend_job)

    def _cancel_backend_heartbeat(self) -> None:
        if self._heartbeat_after_id is None:
            return
        try:
            self.after_cancel(self._heartbeat_after_id)
        except Exception:
            pass
        self._heartbeat_after_id = None

    def _heartbeat_backend_job(self) -> None:
        self._heartbeat_after_id = None
        session = self._backend_session
        if session is None or self._backend_completed:
            return
        try:
            session.client.heartbeat_label_job(session.job_id, session.operator_id, lease_seconds=600)
        except Exception as exc:
            self._info.configure(text=f"Backend heartbeat failed: {exc}")
        finally:
            self._schedule_backend_heartbeat()

    def _release_backend_job(self, reason: str) -> bool:
        session = self._backend_session
        if session is None or self._backend_completed:
            return True
        try:
            session.client.release_label_job(session.job_id, reason=reason)
            return True
        except Exception as exc:
            messagebox.showerror("Backend", f"Failed to release label job: {exc}")
            return False

    def _back_home(self) -> None:
        if self._ask_yes_no("Notice", "Going back will discard unsaved corrections. Continue?"):
            if self._release_backend_job("operator_returned_home"):
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
        self._load_task(sel[0])

    def _load_task(self, task_key: str) -> None:
        task = self._tasks_by_key.get(task_key)
        if task is None:
            return

        try:
            bundle = load_prediction_bundle(task, mode="pred")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        self._active_key = task.key
        self._active_task = task
        self._active_bundle = bundle
        self._bundles = {"pred": bundle}
        self._camera_ids = list(task.cameras)
        self._cam_idx = 0
        self._view_states = {}
        self._source_state_cache = {}
        self._reset_visualizations()
        self._frame_pos = self._next_unconfirmed_position(task, 0)
        if self._frame_pos < 0:
            self._frame_pos = 0
        self._load_current_sample()

    def _next_unconfirmed_position(self, task: CorrectionTask, start: int) -> int:
        total = task.total_frames
        if total <= 0:
            return -1
        rec = self._progress.get(task.key)
        done = rec.done_positions if rec else set()
        for offset in range(total):
            pos = (start + offset) % total
            if pos not in done:
                return pos
        return -1

    def _is_frame_done(self, task: CorrectionTask, frame_pos: int) -> bool:
        rec = self._progress.get(task.key)
        return bool(rec is not None and int(frame_pos) in rec.done_positions)

    def _load_current_sample(self) -> None:
        task = self._active_task
        key = self._active_key
        if task is None or key is None or not self._bundles:
            self._canvas.clear()
            return
        if not task.frames:
            self._canvas.clear()
            return

        self._frame_pos = max(0, min(self._frame_pos, task.total_frames - 1))
        self._load_current_source_states()
        self._cam_idx = max(0, min(self._cam_idx, max(0, len(self._camera_ids) - 1)))
        self._refresh_view()

    def _normalize_source(self, source: str) -> str:
        source = (source or "").strip().lower()
        return source if source in SOURCE_ORDER else "pred"

    def _copy_view_state(self, state: Tuple[HandPoints, HandVisible]) -> Tuple[HandPoints, HandVisible]:
        points, visible = state
        pts = [[(float(x), float(y)) for (x, y) in hand] for hand in points]
        vis = [[bool(v) for v in hand] for hand in visible]
        return pts, vis

    def _source_cache_key(self, cam_id: str, source: Optional[str] = None) -> Optional[Tuple[str, int, str, str]]:
        if self._active_key is None:
            return None
        return (self._active_key, int(self._frame_pos), str(cam_id), self._normalize_source(source or self._mode))

    def _load_current_source_states(self) -> None:
        task = self._active_task
        if task is None or not task.frames:
            self._view_states = {}
            return
        frame_idx = task.frames[self._frame_pos]
        if self._mode == "tracking":
            self._view_states = self._build_tracking_source_states(frame_idx)
            self._make_all_view_states_displayable(frame_idx)
            return
        self._view_states = {
            cam_id: self._build_initial_view_state(frame_idx, cam_id, self._mode) for cam_id in self._camera_ids
        }
        self._make_all_view_states_displayable(frame_idx)

    def _build_initial_view_state(self, frame_idx: int, cam_id: str, source: str) -> Tuple[HandPoints, HandVisible]:
        source = self._normalize_source(source)
        cache_key = self._source_cache_key(cam_id, source)
        if cache_key is not None and cache_key in self._source_state_cache:
            cached = self._copy_view_state(self._source_state_cache[cache_key])
            points, visible = cached
            if source == "scratch":
                if self._has_hidden_points(points):
                    points = self._merge_missing_points(points, self._mediapipe_points(frame_idx, cam_id))
                return points, visible
            if source == "tracking":
                return points, visible
            return self._fill_missing_points(points, frame_idx, cam_id), visible

        if source == "scratch":
            points = self._mediapipe_points(frame_idx, cam_id)
            visible = self._initial_visibility(frame_idx, cam_id)
            return points, visible
        if source == "tracking":
            errors: List[str] = []
            state = self._build_tracking_view_state(frame_idx, cam_id, errors)
            self._show_tracking_errors_once(frame_idx, errors)
            return state

        bundle = self._ensure_bundle(source)
        fallback_points = self._display_fallback_points(frame_idx, cam_id)
        if self._is_source_missing(source, frame_idx, cam_id):
            return fallback_points, self._none_visible()

        visible = None if source in {"correct", "last"} else self._initial_visibility(frame_idx, cam_id)
        points, visible = view_state_from_bundle(
            bundle,
            frame_idx,
            cam_id,
            default_points=fallback_points,
            default_visible=visible,
        )
        points = self._apply_initial_display_fallback(points, visible, fallback_points)
        return points, visible

    @staticmethod
    def _has_hidden_points(points: HandPoints) -> bool:
        for hand in range(min(2, len(points))):
            for joint in range(min(21, len(points[hand]))):
                if LabelPage._is_hidden_point(points[hand][joint]):
                    return True
        return False

    def _fill_missing_points(self, points: HandPoints, frame_idx: int, cam_id: str) -> HandPoints:
        if not self._has_hidden_points(points):
            return [[(float(x), float(y)) for (x, y) in hand] for hand in points]
        return self._merge_missing_points(points, self._display_fallback_points(frame_idx, cam_id))

    def _display_fallback_points(self, frame_idx: int, cam_id: str) -> HandPoints:
        if self._active_task is None:
            return self._empty_points()
        pred_points = load_frame_points(
            self._active_task.episode_dir() / self._active_task.prediction_dir,
            cam_id,
            frame_idx,
        )
        if pred_points is not None and not self._has_hidden_points(pred_points):
            return pred_points
        mediapipe_points = self._mediapipe_points(frame_idx, cam_id)
        template = self._template_points(frame_idx, cam_id)
        if pred_points is None:
            return self._replace_unusable_points(mediapipe_points, template)
        merged = self._merge_missing_points(pred_points, mediapipe_points)
        return self._replace_unusable_points(merged, template)

    def _apply_initial_display_fallback(
        self,
        points: HandPoints,
        visible: HandVisible,
        fallback: HandPoints,
    ) -> HandPoints:
        out = [[(float(x), float(y)) for (x, y) in hand] for hand in points]
        for hand in range(min(2, len(out), len(visible), len(fallback))):
            for joint in range(min(21, len(out[hand]), len(visible[hand]), len(fallback[hand]))):
                if not visible[hand][joint] or self._is_hidden_point(out[hand][joint]):
                    x, y = fallback[hand][joint]
                    out[hand][joint] = (float(x), float(y))
        return out

    @staticmethod
    def _merge_missing_points(points: HandPoints, fallback: HandPoints) -> HandPoints:
        out = [[(float(x), float(y)) for (x, y) in hand] for hand in points]
        for hand in range(min(2, len(out), len(fallback))):
            for joint in range(min(21, len(out[hand]), len(fallback[hand]))):
                if LabelPage._is_hidden_point(out[hand][joint]):
                    x, y = fallback[hand][joint]
                    out[hand][joint] = (float(x), float(y))
        return out

    def _replace_unusable_points(self, points: HandPoints, fallback: HandPoints) -> HandPoints:
        out = [[(float(x), float(y)) for (x, y) in hand] for hand in points]
        for hand in range(min(2, len(out), len(fallback))):
            all_zero = all(
                float(out[hand][joint][0]) == 0.0 and float(out[hand][joint][1]) == 0.0
                for joint in range(min(21, len(out[hand])))
            )
            for joint in range(min(21, len(out[hand]), len(fallback[hand]))):
                if self._is_hidden_point(out[hand][joint]) or all_zero:
                    x, y = fallback[hand][joint]
                    out[hand][joint] = (float(x), float(y))
        return out

    def _make_all_view_states_displayable(self, frame_idx: int) -> None:
        for cam_id, state in list(self._view_states.items()):
            points, visible = state
            fallback = self._display_fallback_points(frame_idx, cam_id)
            points = self._merge_missing_points(points, fallback)
            points = self._replace_unusable_points(points, fallback)
            self._view_states[cam_id] = (points, visible)

    def _template_points(self, frame_idx: int, cam_id: str) -> HandPoints:
        width = 848.0
        height = 480.0
        if self._active_task is not None:
            img_path = find_frame_path(
                self._active_task.episode_dir(),
                cam_id,
                frame_idx,
                self._active_task.rgb_path_template,
            )
            if img_path is not None:
                try:
                    from PIL import Image

                    with Image.open(img_path) as im:
                        width, height = float(im.size[0]), float(im.size[1])
                except Exception:
                    pass
        centers = ((width * 0.42, height * 0.52), (width * 0.58, height * 0.52))
        scale = max(18.0, min(width, height) * 0.08)
        return [
            [
                (
                    float(centers[hand][0] + self._template_joint_offset(hand, joint)[0] * scale),
                    float(centers[hand][1] + self._template_joint_offset(hand, joint)[1] * scale),
                )
                for joint in range(21)
            ]
            for hand in range(2)
        ]

    @staticmethod
    def _template_joint_offset(hand: int, joint: int) -> Tuple[float, float]:
        base = (
            (0.0, 0.55),
            (-0.45, 0.10),
            (-0.68, -0.14),
            (-0.82, -0.36),
            (-0.94, -0.58),
            (-0.28, 0.02),
            (-0.34, -0.32),
            (-0.38, -0.62),
            (-0.40, -0.88),
            (0.0, -0.02),
            (0.0, -0.38),
            (0.0, -0.70),
            (0.0, -0.98),
            (0.28, 0.02),
            (0.34, -0.32),
            (0.38, -0.62),
            (0.40, -0.88),
            (0.48, 0.12),
            (0.62, -0.14),
            (0.72, -0.38),
            (0.82, -0.60),
        )
        x, y = base[joint]
        if hand == 0:
            x = -x
        return x, y

    @staticmethod
    def _is_hidden_point(point: Tuple[float, float]) -> bool:
        x, y = point
        return float(x) == -1.0 and float(y) == -1.0

    def _is_source_missing(self, source: str, frame_idx: int, cam_id: str) -> bool:
        source = self._normalize_source(source)
        if source in {"scratch", "tracking"}:
            return False
        bundle = self._ensure_bundle(source)
        return source_frame_path(bundle, frame_idx, cam_id) is None

    def _mediapipe_points(self, frame_idx: int, cam_id: str) -> HandPoints:
        self._hand_initializer.ensure_available()
        if self._active_task is None:
            return self._hand_initializer.empty_points()
        img_path = find_frame_path(
            self._active_task.episode_dir(),
            cam_id,
            frame_idx,
            self._active_task.rgb_path_template,
        )
        return self._hand_initializer.points_from_image(img_path)

    def _initial_visibility(self, frame_idx: int, cam_id: str) -> HandVisible:
        if self._active_task is None:
            return self._none_visible()
        episode_dir = self._active_task.episode_dir()
        visible = load_frame_visibility(episode_dir / self._active_task.correction_dir, cam_id, int(frame_idx) - 1)
        if visible is not None:
            return visible
        visible = load_frame_visibility(episode_dir / self._active_task.prediction_dir, cam_id, frame_idx)
        if visible is not None:
            return visible
        return self._none_visible()

    def _empty_points(self) -> HandPoints:
        return [[(0.0, 0.0) for _ in range(21)] for _ in range(2)]

    def _none_visible(self) -> HandVisible:
        return [[False for _ in range(21)] for _ in range(2)]

    def _hidden_points(self) -> HandPoints:
        return [[(-1.0, -1.0) for _ in range(21)] for _ in range(2)]

    def _empty_tracking_state(self) -> Tuple[HandPoints, HandVisible]:
        return self._hidden_points(), self._none_visible()

    def _tracking_runtime(self) -> CoTrackerRuntime:
        if self._tracker is None:
            self._tracker = CoTrackerRuntime()
        return self._tracker

    def _build_tracking_source_states(self, frame_idx: int) -> ViewStateByCam:
        errors: List[str] = []
        states = {
            cam_id: self._build_tracking_view_state(frame_idx, cam_id, errors) for cam_id in self._camera_ids
        }
        self._show_tracking_errors_once(frame_idx, errors)
        return states

    def _build_tracking_view_state(
        self,
        frame_idx: int,
        cam_id: str,
        errors: List[str],
    ) -> Tuple[HandPoints, HandVisible]:
        cache_key = self._source_cache_key(cam_id, "tracking")
        if cache_key is not None and cache_key in self._source_state_cache:
            return self._copy_view_state(self._source_state_cache[cache_key])

        task = self._active_task
        if task is None:
            state = self._empty_tracking_state()
        elif self._frame_pos <= 0:
            errors.append("当前帧没有 frames 数组中的上一帧样本，无法执行 Tracking。")
            state = self._empty_tracking_state()
        else:
            prev_frame_idx = task.frames[self._frame_pos - 1]
            try:
                state = self._tracking_runtime().track_camera(
                    episode_dir=task.episode_dir(),
                    cam_id=cam_id,
                    prev_frame_idx=prev_frame_idx,
                    frame_idx=frame_idx,
                )
            except Exception as exc:
                errors.append(f"Camera {cam_id}: {exc}")
                state = self._empty_tracking_state()

        if cache_key is not None:
            self._source_state_cache[cache_key] = self._copy_view_state(state)
        return state

    def _show_tracking_errors_once(self, frame_idx: int, errors: List[str]) -> None:
        if not errors or self._active_key is None:
            return
        unique_errors = []
        for err in errors:
            if err not in unique_errors:
                unique_errors.append(err)
        notice_key = (self._active_key, int(self._frame_pos), int(frame_idx), tuple(unique_errors))
        if notice_key in self._tracking_notice_keys:
            return
        self._tracking_notice_keys.add(notice_key)
        messagebox.showwarning("Tracking", "\n".join(unique_errors))

    def _ensure_bundle(self, source: str) -> PredictionBundle:
        source = self._normalize_source(source)
        if source == "tracking":
            raise ValueError("Tracking mode uses runtime CoTracker results and does not load a prediction bundle.")
        bundle = self._bundles.get(source)
        if bundle is not None:
            return bundle
        if self._active_task is None:
            raise ValueError("No active task.")
        if source == "scratch":
            self._hand_initializer.ensure_available()
        bundle = load_prediction_bundle(self._active_task, mode=source)
        self._bundles[source] = bundle
        if self._active_bundle is None or source == "pred":
            self._active_bundle = bundle
        return bundle

    def _save_bundle(self) -> Optional[PredictionBundle]:
        return self._bundles.get("pred") or self._active_bundle

    def _cache_current_source_state(self) -> None:
        cam_id = self._active_cam_id()
        if cam_id is None:
            return
        state = self._canvas.get_hand_state()
        self._view_states[cam_id] = state
        cache_key = self._source_cache_key(cam_id, self._mode)
        if cache_key is not None:
            self._source_state_cache[cache_key] = self._copy_view_state(state)

    def _invalidate_corrected_source_cache(self) -> None:
        if self._active_key is None:
            return
        self._bundles.pop("correct", None)
        self._bundles.pop("last", None)
        stale = [
            key
            for key in self._source_state_cache
            if key[0] == self._active_key and key[3] in {"correct", "last"}
        ]
        for key in stale:
            self._source_state_cache.pop(key, None)

    def _source_label(self, source: Optional[str] = None) -> str:
        source = self._normalize_source(source or self._mode)
        return SOURCE_LABELS[source]

    def _source_status_label(self, source: str, frame_idx: int, cam_id: str) -> str:
        label = self._source_label(source)
        if self._is_source_missing(source, frame_idx, cam_id):
            return f"{label} (missing)"
        return label

    def _source_button_text(self) -> str:
        return f"2D Source: {self._source_label()}"

    def _update_source_button(self) -> None:
        if self._source_btn is not None:
            self._source_btn.configure(text=self._source_button_text())

    def _skeleton_button_text(self) -> str:
        return "Hide Skeleton" if self._show_skeleton else "Show Skeleton"

    def _update_skeleton_button(self) -> None:
        if self._skeleton_btn is not None:
            self._skeleton_btn.configure(text=self._skeleton_button_text())

    def _mano_button_text(self) -> str:
        return "Hide MANO" if self._show_mano else "Show MANO"

    def _update_mano_button(self) -> None:
        if self._mano_btn is not None:
            self._mano_btn.configure(text=self._mano_button_text())

    def _reset_mano(self) -> None:
        self._show_mano = False
        self._mano_mesh = None
        self._update_mano_button()
        try:
            self._canvas.set_mano_overlay([])
        except Exception:
            pass
        self._sync_visualization_canvas_state()

    def _reset_skeleton(self) -> None:
        self._show_skeleton = False
        self._skeleton_joints_3d = None
        self._update_skeleton_button()
        try:
            self._canvas.set_skeleton_overlay(None)
        except Exception:
            pass
        self._sync_visualization_canvas_state()

    def _reset_visualizations(self) -> None:
        self._show_mano = False
        self._mano_mesh = None
        self._show_skeleton = False
        self._skeleton_joints_3d = None
        self._update_mano_button()
        self._update_skeleton_button()
        try:
            self._canvas.set_mano_overlay([])
            self._canvas.set_skeleton_overlay(None)
        except Exception:
            pass
        self._sync_visualization_canvas_state()

    def _visualization_active(self) -> bool:
        return bool(self._show_mano or self._show_skeleton)

    def _sync_visualization_canvas_state(self) -> None:
        active = self._visualization_active()
        try:
            self._canvas.set_read_only(active)
            self._canvas.set_annotation_visible(not active)
        except Exception:
            pass

    def _mano_runtime_instance(self) -> ManoViewRuntime:
        if self._mano_runtime is None:
            self._mano_runtime = ManoViewRuntime()
        return self._mano_runtime

    def _toggle_skeleton(self) -> None:
        if self._show_skeleton:
            self._reset_skeleton()
            return

        task = self._active_task
        if task is None:
            return
        self._reset_mano()
        self._cache_current_source_state()
        frame_idx = task.frames[self._frame_pos]
        for cam_id in self._camera_ids:
            if cam_id not in self._view_states:
                self._view_states[cam_id] = self._build_initial_view_state(frame_idx, cam_id, self._mode)
        self._make_all_view_states_displayable(frame_idx)

        missing = self._incomplete_joint_count(min_views=2)
        if missing > 0:
            messagebox.showwarning("Show Skeleton", f"标注量不足：还有 {missing} 个关节点的可见视角数少于 2。")
            self._reset_skeleton()
            return

        try:
            self._skeleton_joints_3d = self._mano_runtime_instance().build_skeleton(
                episode_dir=task.episode_dir(),
                camera_ids=self._camera_ids,
                view_states=self._view_states,
            )
        except Exception as exc:
            messagebox.showerror("Show Skeleton", str(exc))
            self._reset_skeleton()
            return

        self._show_skeleton = True
        self._update_skeleton_button()
        self._refresh_skeleton_overlay()
        self._sync_visualization_canvas_state()

    def _refresh_skeleton_overlay(self) -> None:
        if not self._show_skeleton or self._skeleton_joints_3d is None or self._active_task is None:
            self._canvas.set_skeleton_overlay(None)
            return
        cam_id = self._active_cam_id()
        if cam_id is None:
            self._canvas.set_skeleton_overlay(None)
            return
        try:
            points, visible = self._mano_runtime_instance().project_skeleton(
                episode_dir=self._active_task.episode_dir(),
                cam_id=cam_id,
                joints_3d=self._skeleton_joints_3d,
            )
        except Exception as exc:
            self._reset_skeleton()
            messagebox.showerror("Show Skeleton", str(exc))
            return
        self._canvas.set_skeleton_overlay(points, visible)

    def _toggle_mano(self) -> None:
        if self._show_mano:
            self._reset_mano()
            return

        task = self._active_task
        if task is None:
            return
        self._reset_skeleton()
        self._cache_current_source_state()
        frame_idx = task.frames[self._frame_pos]
        for cam_id in self._camera_ids:
            if cam_id not in self._view_states:
                self._view_states[cam_id] = self._build_initial_view_state(frame_idx, cam_id, self._mode)
        self._make_all_view_states_displayable(frame_idx)

        missing = self._incomplete_joint_count(min_views=2)
        if missing > 0:
            messagebox.showwarning("Show MANO", f"标注量不足：还有 {missing} 个关节点的可见视角数少于 2。")
            self._reset_mano()
            return

        try:
            self._mano_mesh = self._mano_runtime_instance().build_mesh(
                episode_dir=task.episode_dir(),
                camera_ids=self._camera_ids,
                view_states=self._view_states,
            )
        except Exception as exc:
            messagebox.showerror("Show MANO", str(exc))
            self._reset_mano()
            return

        self._show_mano = True
        self._update_mano_button()
        self._refresh_mano_overlay()
        self._sync_visualization_canvas_state()

    def _refresh_mano_overlay(self) -> None:
        if not self._show_mano or self._mano_mesh is None or self._active_task is None:
            self._canvas.set_mano_overlay([])
            return
        cam_id = self._active_cam_id()
        if cam_id is None:
            self._canvas.set_mano_overlay([])
            return
        try:
            lines = self._mano_runtime_instance().project_mesh(
                episode_dir=self._active_task.episode_dir(),
                cam_id=cam_id,
                mesh=self._mano_mesh,
            )
        except Exception as exc:
            self._reset_mano()
            messagebox.showerror("Show MANO", str(exc))
            return
        self._canvas.set_mano_overlay(lines)

    def _refresh_visual_overlays(self) -> None:
        self._refresh_skeleton_overlay()
        self._refresh_mano_overlay()
        self._sync_visualization_canvas_state()

    def _toggle_source(self) -> None:
        cam_id = self._active_cam_id()
        if cam_id is None:
            return
        old_source = self._mode
        old_idx = SOURCE_ORDER.index(old_source)
        task = self._active_task
        if task is None:
            return
        new_source = SOURCE_ORDER[(old_idx + 1) % len(SOURCE_ORDER)]
        self._cache_current_source_state()
        self._reset_visualizations()
        self._mode = new_source
        try:
            self._load_current_source_states()
        except Exception as exc:
            self._mode = old_source
            self._load_current_source_states()
            self._update_source_button()
            messagebox.showerror("Error", str(exc))
            return

        self._update_source_button()
        self._refresh_view()

    def _active_cam_id(self) -> Optional[str]:
        if not self._camera_ids:
            return None
        self._cam_idx = max(0, min(self._cam_idx, len(self._camera_ids) - 1))
        return self._camera_ids[self._cam_idx]

    def _refresh_view(self) -> None:
        task = self._active_task
        if task is None or not self._bundles:
            self._canvas.clear()
            return
        cam_id = self._active_cam_id()
        if cam_id is None:
            self._canvas.clear()
            return

        frame_idx = task.frames[self._frame_pos]
        source = self._mode
        state = self._view_states.get(cam_id)
        if state is None:
            state = self._build_initial_view_state(frame_idx, cam_id, source)
            fallback = self._display_fallback_points(frame_idx, cam_id)
            points0, visible0 = state
            state = (self._replace_unusable_points(self._merge_missing_points(points0, fallback), fallback), visible0)
            self._view_states[cam_id] = state
        points, visible = state

        self._update_source_button()
        self._canvas.set_hand_state(points, visible)
        self._canvas.set_count_base(self._visibility_counts(exclude_cam=cam_id))
        img_path = find_frame_path(task.episode_dir(), cam_id, frame_idx, task.rgb_path_template)
        self._canvas.set_image(img_path)
        self._refresh_visual_overlays()

        rec = self._progress.get(task.key)
        done = rec.done_count if rec else 0
        frame_done = self._is_frame_done(task, self._frame_pos)
        self._info.configure(
            text=(
                f"Task: {task.display_name}    Camera: {cam_id} ({self._cam_idx + 1}/{len(self._camera_ids)})    "
                f"Frame: {frame_idx} ({self._frame_pos + 1}/{task.total_frames})    "
                f"View Source: {self._source_status_label(source, frame_idx, cam_id)}    Done: {done}/{task.total_frames}"
            )
        )
        if self._frame_status is not None:
            self._frame_status.configure(
                text=f"当前帧：{'已完成' if frame_done else '未完成'}",
                fg=STATUS_DONE_COLOR if frame_done else STATUS_TODO_COLOR,
            )

    def _visibility_counts(self, *, exclude_cam: Optional[str] = None) -> List[List[int]]:
        counts = [[0 for _ in range(21)] for _ in range(2)]
        for cam_id, (_points, visible) in self._view_states.items():
            if exclude_cam is not None and cam_id == exclude_cam:
                continue
            for hand in range(min(2, len(visible))):
                for joint in range(min(21, len(visible[hand]))):
                    if visible[hand][joint]:
                        counts[hand][joint] += 1
        return counts

    def _incomplete_joint_count(self, *, min_views: int = 2) -> int:
        counts = self._visibility_counts()
        missing = 0
        for hand in range(2):
            for joint in range(21):
                if counts[hand][joint] < min_views:
                    missing += 1
        return missing

    def _prev_cam(self) -> None:
        if not self._camera_ids:
            return
        self._cache_current_source_state()
        self._cam_idx = (self._cam_idx - 1) % len(self._camera_ids)
        self._refresh_view()

    def _next_cam(self) -> None:
        if not self._camera_ids:
            return
        self._cache_current_source_state()
        self._cam_idx = (self._cam_idx + 1) % len(self._camera_ids)
        self._refresh_view()

    def _undo(self) -> None:
        if self._visualization_active():
            return
        self._canvas.undo()

    def _ignore_view(self) -> None:
        if self._visualization_active():
            return
        self._canvas.ignore_view()

    def _skip_frame(self) -> None:
        task = self._active_task
        if task is None or task.total_frames <= 0:
            return
        next_pos = min(task.total_frames - 1, self._frame_pos + 1)
        if next_pos == self._frame_pos:
            return
        self._source_state_cache.clear()
        self._view_states = {}
        self._reset_visualizations()
        self._frame_pos = next_pos
        self._load_current_sample()

    def _back_frame(self) -> None:
        task = self._active_task
        if task is None or task.total_frames <= 0:
            return
        prev_pos = max(0, self._frame_pos - 1)
        if prev_pos == self._frame_pos:
            return
        self._source_state_cache.clear()
        self._view_states = {}
        self._reset_visualizations()
        self._frame_pos = prev_pos
        self._load_current_sample()

    def _confirm(self) -> None:
        task = self._active_task
        bundle = self._save_bundle()
        if task is None or bundle is None or self._active_key is None or self._jsonl_path is None:
            return
        if self._visualization_active():
            messagebox.showwarning("Notice", "Please hide MANO/Skeleton visualization before confirming annotations.")
            return
        if not self._camera_ids:
            messagebox.showwarning("Notice", "No camera directories found for the current task.")
            return

        self._cache_current_source_state()
        for cam_id in self._camera_ids:
            if cam_id not in self._view_states:
                self._view_states[cam_id] = self._build_initial_view_state(task.frames[self._frame_pos], cam_id, self._mode)
        missing = self._incomplete_joint_count(min_views=2)
        if missing > 0:
            messagebox.showwarning("Notice", f"未标注完全：还有 {missing} 个关节点的可见视角数少于 2。")
            self._refresh_view()
            return
        if not self._ask_yes_no("Confirm", "Confirm that all views have been corrected?"):
            return

        frame_idx = task.frames[self._frame_pos]
        try:
            for cam_id in self._camera_ids:
                points, visible = self._view_states[cam_id]
                apply_view_state_to_corrected(bundle, frame_idx, cam_id, points, visible)
            save_corrected_array(bundle)
            self._invalidate_corrected_source_cache()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        rec = self._progress.get(task.key)
        if rec is None:
            rec = CorrectionProgress(task_key=task.key, total_frames=task.total_frames)
            self._progress[task.key] = rec
        rec.done_positions.add(self._frame_pos)
        save_correction_progress(self._jsonl_path, self._progress)
        self._update_tree_row(task)

        next_pos = self._next_unconfirmed_position(task, self._frame_pos + 1)
        self._reset_visualizations()
        if next_pos < 0:
            if not self._complete_backend_job(task):
                self._refresh_view()
                return
            messagebox.showinfo("Done", "This task is fully completed.")
            self._refresh_view()
            return

        self._frame_pos = next_pos
        self._source_state_cache.clear()
        self._view_states = {}
        self._load_current_sample()

    def _complete_backend_job(self, task: CorrectionTask) -> bool:
        session = self._backend_session
        if session is None or self._backend_completed:
            return True
        data_uri = str(session.payload.get("data_uri") or "").rstrip("/")
        correction_dir = str(session.payload.get("correction_dir") or task.correction_dir or "corrected_2d").strip("/")
        artifact_uri = f"{data_uri}/{correction_dir}" if data_uri and correction_dir else data_uri
        artifacts = []
        if artifact_uri:
            artifacts.append(
                {
                    "kind": "corrected_2d",
                    "uri": artifact_uri,
                    "metadata": {
                        "cameras": list(task.cameras),
                        "frames": list(task.frames),
                        "operator_id": session.operator_id,
                    },
                }
            )
        try:
            session.client.complete_label_job(
                session.job_id,
                result={
                    "operator_id": session.operator_id,
                    "frames_completed": list(task.frames),
                    "local_progress_cache": self._jsonl_path or "",
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            messagebox.showerror("Backend", f"Failed to complete label job: {exc}")
            return False
        self._backend_completed = True
        self._cancel_backend_heartbeat()
        return True

    def _update_tree_row(self, task: CorrectionTask) -> None:
        rec = self._progress.get(task.key)
        done = rec.done_count if rec else 0
        self._tree.set(task.key, "done", str(done))
        self._tree.set(task.key, "total", str(task.total_frames))
