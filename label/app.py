from __future__ import annotations

import locale
import os
import shutil
import tempfile
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

try:
    from .backend_client import BackendClientError, LabelBackendClient, LabelJobSession, episode_display_id, session_from_lease
    from .canvas_view import HandPoints, HandVisible, ImageAnnotatorCanvas
    from .storage import (
        CorrectionProgress,
        CorrectionTask,
        JOINTS_VIS_DIR,
        PredictionBundle,
        apply_view_state_to_corrected,
        correction_task_from_backend_payload,
        ensure_correction_progress,
        find_frame_path,
        load_correction_progress,
        load_correction_tasks,
        load_joint_visibility,
        load_prediction_bundle,
        save_corrected_array,
        save_correction_progress,
        source_frame_path,
        view_state_from_bundle,
    )
    from .theme import Theme, apply_theme, fit_window, WrapToolbar
    from .mano_view import (
        ManoMeshResult,
        ManoViewRuntime,
        describe_mano_projection_issue,
    )
    from .tracking import CoTrackerRuntime
    from .video_frames import ensure_decoded_rgb_frames
    from .env_config import LabelConfig, load_label_config
    from .mesh_cache import OriginalMeshCache
    from .timeline import LabelTimeline
except Exception:
    from backend_client import BackendClientError, LabelBackendClient, LabelJobSession, episode_display_id, session_from_lease
    from canvas_view import HandPoints, HandVisible, ImageAnnotatorCanvas
    from storage import (
        CorrectionProgress,
        CorrectionTask,
        JOINTS_VIS_DIR,
        PredictionBundle,
        apply_view_state_to_corrected,
        correction_task_from_backend_payload,
        ensure_correction_progress,
        find_frame_path,
        load_correction_progress,
        load_correction_tasks,
        load_joint_visibility,
        load_prediction_bundle,
        save_corrected_array,
        save_correction_progress,
        source_frame_path,
        view_state_from_bundle,
    )
    from theme import Theme, apply_theme, fit_window, WrapToolbar
    from mano_view import (
        ManoMeshResult,
        ManoViewRuntime,
        describe_mano_projection_issue,
    )
    from tracking import CoTrackerRuntime
    from video_frames import ensure_decoded_rgb_frames
    from env_config import LabelConfig, load_label_config
    from label.mesh_cache import OriginalMeshCache
    from label.timeline import LabelTimeline


ViewStateByCam = Dict[str, Tuple[HandPoints, HandVisible]]
SourceStateCache = Dict[Tuple[str, int, str, str], Tuple[HandPoints, HandVisible]]
SOURCE_ORDER = ("mano", "correct")
SOURCE_LABELS = {
    "mano": "原始视角",
    "correct": "修改后视角",
}
STATUS_DONE_COLOR = "#46d36b"
STATUS_TODO_COLOR = "#ff5c5c"


class LabelToolApp(tk.Tk):
    def __init__(self, config: Optional[LabelConfig] = None):
        self._ensure_utf8_env()
        super().__init__()
        self.config = config or load_label_config()

        try:
            self.tk.call("encoding", "system", "utf-8")
        except Exception:
            pass

        self._is_maximized = False
        self._restore_geom: Optional[str] = None

        self.configure(bg=Theme.BG)
        self.title("Orbbec · 关节标注")

        apply_theme(self)
        fit_window(self)

        self._root_container = ttk.Frame(self, style="TFrame")
        self._root_container.pack(fill="both", expand=True)

        self._titlebar = ttk.Frame(self._root_container, style="Panel.TFrame")
        self._titlebar.pack(fill="x")
        self._build_titlebar(self._titlebar)

        self._page_host = ttk.Frame(self._root_container, style="TFrame")
        self._page_host.pack(fill="both", expand=True)

        self._home = HomePage(self._page_host, config=self.config, on_exit=self._on_exit, on_enter=self._go_label)
        self._label = LabelPage(self._page_host, config=self.config, on_back=self._go_home)
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
        bar.configure(height=52)
        bar.pack_propagate(False)

        left = ttk.Frame(bar, style="Panel.TFrame")
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(bar, style="Panel.TFrame")
        right.pack(side="right", fill="y")

        title = ttk.Label(left, text="ORBBEC  /  关节标注", style="Section.TLabel")
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
        try:
            self._label.release_on_exit()
        except Exception:
            pass
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
    def __init__(self, master, *, config: LabelConfig, on_exit, on_enter):
        super().__init__(master, style="TFrame")
        self._config = config
        self._on_exit = on_exit
        self._on_enter = on_enter
        self._task_rows: Dict[str, Dict[str, Any]] = {}
        self._episode_rows: Dict[str, Dict[str, Any]] = {}

        outer = ttk.Frame(self, style="TFrame")
        outer.pack(fill="both", expand=True)

        center = ttk.Frame(outer, style="TFrame", padding=(24, 20))
        center.pack(fill="both", expand=True)
        ttk.Label(center, text="关节标注", style="Title.TLabel").pack(anchor="w")
        ttk.Label(center, text="选择任务与 Episode，检查并修正多视角手部关节。", style="Muted.TLabel").pack(anchor="w", pady=(4, 16))

        self._var_backend_url = tk.StringVar(value=config.backend_url)
        self._var_operator = tk.StringVar(value=config.operator_id)
        self._var_jsonl = tk.StringVar()
        form = ttk.Frame(center, style="Panel.TFrame", padding=16)
        form.pack(fill="x", pady=(0, 16))
        form.columnconfigure(0, weight=3)
        form.columnconfigure(1, weight=2)
        ttk.Label(form, text="服务地址", style="PanelMuted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(form, text="操作员", style="PanelMuted.TLabel").grid(row=0, column=1, sticky="w", padx=(20, 0))
        ttk.Entry(form, textvariable=self._var_backend_url).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Entry(form, textvariable=self._var_operator).grid(row=1, column=1, sticky="ew", padx=(20, 0), pady=(6, 0))

        # Reserve the footer before the expanding queues so actions stay visible.
        footer = ttk.Frame(center)
        footer.pack(side="bottom", fill="x", pady=(14, 0))
        self._queue_notice = ttk.Label(footer, text="点击“刷新任务”加载待标注数据。", style="Muted.TLabel")
        self._queue_notice.pack(fill="x", pady=(0, 10))
        footer.bind("<Configure>", lambda e: self._queue_notice.configure(wraplength=max(300, e.width)))
        btns = ttk.Frame(footer)
        btns.pack(fill="x")
        ttk.Button(btns, text="退出", style="Secondary.TButton", command=self._on_exit).pack(side="left")
        ttk.Button(btns, text="刷新任务", style="Secondary.TButton", command=self._refresh_tasks).pack(side="left", padx=8)
        ttk.Button(btns, text="开始标注所选 Episode", style="Primary.TButton", command=self._lease_selected_episode).pack(side="right")
        legacy = ttk.Frame(footer, style="Panel.TFrame", padding=12)
        def toggle_legacy():
            if legacy.winfo_manager():
                legacy.pack_forget()
            else:
                legacy.pack(fill="x", pady=(12, 0))
        ttk.Button(btns, text="本地 JSONL…", style="Secondary.TButton", command=toggle_legacy).pack(side="left")
        ttk.Label(legacy, text="本地任务文件", style="PanelMuted.TLabel").pack(side="left", padx=(0, 10))
        ttk.Entry(legacy, textvariable=self._var_jsonl).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Button(legacy, text="打开", style="Secondary.TButton", command=self._enter_legacy).pack(side="right")

        queue = ttk.Frame(center)
        queue.pack(fill="both", expand=True)
        queue.columnconfigure(0, weight=1, uniform="queues")
        queue.columnconfigure(1, weight=1, uniform="queues")
        queue.rowconfigure(0, weight=1)
        for column, title, attr, cols, headings, widths, binding, callback in (
            (0, "01  选择任务", "_queue_tree", ("task", "segments", "episodes", "subjects", "frames"),
             ("任务", "片段", "批次", "受试者", "帧数"), (180, 62, 90, 100, 65), "<<TreeviewSelect>>", self._load_selected_task_episodes),
            (1, "02  选择 Episode", "_episode_tree", ("episode", "subject", "segments", "frames", "first"),
             ("Episode", "受试者", "片段", "帧数", "起始帧"), (100, 110, 62, 65, 75), "<Double-1>", self._lease_selected_episode),
        ):
            card = ttk.Frame(queue, style="Panel.TFrame", padding=14)
            card.grid(row=0, column=column, sticky="nsew", padx=(0, 8) if column == 0 else (8, 0))
            ttk.Label(card, text=title, style="Section.TLabel").pack(anchor="w", pady=(0, 12))
            host = ttk.Frame(card, style="Panel.TFrame")
            host.pack(fill="both", expand=True)
            tree = ttk.Treeview(host, columns=cols, show="headings", height=5)
            setattr(self, attr, tree)
            for col, heading, width in zip(cols, headings, widths):
                tree.heading(col, text=heading)
                tree.column(col, width=width, minwidth=55, anchor="w" if col in ("task", "subject", "subjects") else "center")
            scroll = ttk.Scrollbar(host, orient="vertical", command=tree.yview)
            scroll.pack(side="right", fill="y", padx=(8, 0))
            tree.pack(side="left", fill="both", expand=True)
            tree.configure(yscrollcommand=scroll.set)
            tree.bind(binding, callback)

    def _refresh_tasks(self) -> None:
        backend_url = (self._var_backend_url.get() or "").strip()
        if not backend_url:
            messagebox.showwarning("Notice", "Please enter a backend URL.")
            return
        try:
            groups = LabelBackendClient(
                backend_url,
                timeout_seconds=self._config.request_timeout_seconds,
            ).queued_label_tasks()
        except BackendClientError as exc:
            self._queue_notice.configure(text=str(exc))
            messagebox.showerror("Backend", str(exc))
            return
        self._populate_task_rows(groups)

    def _populate_task_rows(self, groups: List[Dict[str, Any]]) -> None:
        for item_id in self._queue_tree.get_children():
            self._queue_tree.delete(item_id)
        for item_id in self._episode_tree.get_children():
            self._episode_tree.delete(item_id)
        self._task_rows = {}
        self._episode_rows = {}
        for index, group in enumerate(groups, 1):
            item_id = f"task_{index}"
            subjects = str(group.get("subject_summary") or "")
            if len(subjects) > 34:
                subjects = subjects[:31] + "..."
            values = (
                str(group.get("task_name") or ""),
                str(group.get("segments") or group.get("queued") or 0),
                str(group.get("episodes") or 0),
                subjects,
                str(group.get("frames") or 0),
            )
            self._queue_tree.insert("", "end", iid=item_id, values=values)
            self._task_rows[item_id] = dict(group)
        if groups:
            first = "task_1"
            self._queue_tree.selection_set(first)
            self._queue_tree.focus(first)
            self._queue_notice.configure(text=f"已加载 {len(groups)} 个任务，选择右侧 Episode 开始标注。")
        else:
            self._queue_notice.configure(text="暂无待标注任务。")

    def _load_selected_task_episodes(self, _event: Any = None) -> None:
        backend_url = (self._var_backend_url.get() or "").strip()
        selected = self._queue_tree.selection()
        for item_id in self._episode_tree.get_children():
            self._episode_tree.delete(item_id)
        self._episode_rows = {}
        if not backend_url or not selected:
            return
        task_row = self._task_rows.get(str(selected[0])) or {}
        task_name = str(task_row.get("task_name") or "").strip()
        if not task_name:
            return
        try:
            episodes = LabelBackendClient(
                backend_url,
                timeout_seconds=self._config.request_timeout_seconds,
            ).label_task_episodes(task_name)
        except BackendClientError as exc:
            self._queue_notice.configure(text=str(exc))
            return
        for index, episode in enumerate(episodes, 1):
            item_id = f"episode_{index}"
            episode_id = str(episode.get("episode_id") or "")
            values = (
                episode_display_id(episode),
                str(episode.get("subject_id") or ""),
                str(episode.get("segments") or episode.get("pending_segments") or 0),
                str(episode.get("frames") or 0),
                str(episode.get("first_start_frame") if episode.get("first_start_frame") is not None else ""),
            )
            self._episode_tree.insert("", "end", iid=item_id, values=values)
            self._episode_rows[item_id] = dict(episode)
        if episodes:
            first = "episode_1"
            self._episode_tree.selection_set(first)
            self._episode_tree.focus(first)
            self._queue_notice.configure(text=f"{len(episodes)} episode(s) loaded for {task_name}.")
        else:
            self._queue_notice.configure(text=f"No pending episodes for {task_name}.")

    def _lease_selected_episode(self, _event: Any = None) -> None:
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
        selected_episode = self._episode_tree.selection()
        if not selected_episode:
            messagebox.showwarning("Notice", "Please select an episode.")
            return
        episode_row = self._episode_rows.get(str(selected_episode[0])) or {}
        episode_id = str(episode_row.get("episode_id") or "").strip()
        if not episode_id:
            messagebox.showwarning("Notice", "Selected episode is invalid.")
            return
        try:
            session = session_from_lease(
                backend_url=backend_url,
                operator_id=operator_id,
                mounts=self._config.nas_mounts,
                lease_seconds=self._config.lease_seconds,
                timeout_seconds=self._config.request_timeout_seconds,
                task_name=task_name,
                episode_id=episode_id,
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
    def __init__(self, master, *, config: LabelConfig, on_back):
        super().__init__(master, style="TFrame")
        self._config = config
        self._on_back = on_back

        self._jsonl_path: Optional[str] = None
        self._backend_session: Optional[LabelJobSession] = None
        self._backend_completed: bool = False
        self._heartbeat_after_id: Optional[str] = None
        self._decode_generation: int = 0
        self._decode_thread: Optional[threading.Thread] = None
        self._decode_cache_dir: Optional[Path] = None
        self._mode: str = "correct"
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
        self._tracker: Optional[CoTrackerRuntime] = None
        self._mano_runtime: Optional[ManoViewRuntime] = None
        self._mano_mesh: Optional[ManoMeshResult] = None
        self._show_mano: bool = False
        self._skeleton_joints_3d = None
        self._show_skeleton: bool = False
        self._tracking_notice_keys = set()
        self._mano_projection_errors: Dict[Tuple[str, int, str], str] = {}
        self._source_btn: Optional[ttk.Button] = None
        self._skeleton_btn: Optional[ttk.Button] = None
        self._mano_btn: Optional[ttk.Button] = None
        self._frame_status: Optional[tk.Label] = None

        self._original_mesh_cache = None
        self._mesh_poll_id = None
        self._build_ui()
        for number in range(1, 7):
            self.winfo_toplevel().bind(
                f"<KeyPress-{number}>", lambda event, index=number - 1: self._camera_shortcut(event, index), add="+"
            )

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="TFrame")
        root.pack(fill="both", expand=True, padx=16, pady=14)

        left = ttk.Frame(root, style="Panel.TFrame")
        left.pack(side="left", fill="y")
        right = ttk.Frame(root, style="Panel.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        ttk.Label(left, text="标注进度", style="Section.TLabel").pack(anchor="w", padx=12, pady=(10, 8))

        cols = ("task", "done", "total")
        self._tree = ttk.Treeview(left, columns=cols, show="headings", height=22)
        self._tree.heading("task", text="任务")
        self._tree.heading("done", text="完成")
        self._tree.heading("total", text="总数")
        self._tree.column("task", width=170, anchor="w")
        self._tree.column("done", width=55, anchor="center")
        self._tree.column("total", width=55, anchor="center")
        self._tree["displaycolumns"] = ("task", "done", "total")
        self._tree.pack(side="left", fill="y", padx=(12, 0), pady=(0, 12))

        ysb = ttk.Scrollbar(left, orient="vertical", command=self._tree.yview)
        ysb.pack(side="left", fill="y", padx=(8, 12), pady=(0, 12))
        self._tree.configure(yscrollcommand=ysb.set)
        self._tree.bind("<<TreeviewSelect>>", self._on_task_select)

        top = ttk.Frame(right, style="Panel.TFrame")
        top.pack(fill="x", padx=12, pady=(10, 8))
        self._info = ttk.Label(top, text="", style="PanelMuted.TLabel", wraplength=600)
        self._info.pack(fill="x", expand=True)
        top.bind("<Configure>", lambda e: self._info.configure(wraplength=max(200, e.width)))
        ui_font = (Theme.FONT_FAMILY, Theme.FONT_SIZE) if Theme.FONT_FAMILY else None
        self._frame_status = tk.Label(top, text="", fg=STATUS_TODO_COLOR, bg=Theme.PANEL, font=ui_font)
        self._frame_status.pack(anchor="w", pady=(4, 0))

        btn_row = WrapToolbar(right)
        btn_row.pack(side="bottom", fill="x", padx=12, pady=(8, 12))
        btn_row.add(ttk.Button(btn_row, text="上一帧", style="Small.TButton", command=self._back_frame))
        btn_row.add(ttk.Button(btn_row, text="下一帧", style="Small.TButton", command=self._skip_frame))
        btn_row.add(ttk.Button(btn_row, text="上一机位（1–6 切换）", style="Small.TButton", command=self._prev_cam))
        btn_row.add(ttk.Button(btn_row, text="下一机位", style="Small.TButton", command=self._next_cam))
        btn_row.add(ttk.Button(btn_row, text="撤销", style="Small.TButton", command=self._undo))
        btn_row.add(ttk.Button(btn_row, text="忽略视角", style="Small.TButton", command=self._ignore_view))
        self._source_btn = ttk.Button(
            btn_row,
            text=self._source_button_text(),
            style="Small.TButton",
            command=self._toggle_source,
        )
        btn_row.add(self._source_btn)
        self._skeleton_btn = ttk.Button(
            btn_row,
            text=self._skeleton_button_text(),
            style="Small.TButton",
            command=self._toggle_skeleton,
        )
        btn_row.add(self._skeleton_btn)
        self._mano_btn = ttk.Button(
            btn_row,
            text=self._mano_button_text(),
            style="Small.TButton",
            command=self._toggle_mano,
        )
        btn_row.add(self._mano_btn)
        btn_row.add(ttk.Button(btn_row, text="确认并继续", style="Primary.TButton", command=self._confirm))

        btn_row.add(ttk.Button(btn_row, text="返回任务", style="Secondary.TButton", command=self._back_home))

        timeline_host = ttk.Frame(right, style="Panel.TFrame")
        timeline_host.pack(side="bottom", fill="x", padx=12, pady=(0, 2))
        legend = ttk.Frame(timeline_host, style="Panel.TFrame")
        legend.pack(fill="x")
        self._timeline_status = ttk.Label(legend, text="帧进度", style="PanelMuted.TLabel")
        self._timeline_status.pack(side="left", expand=True, fill="x")
        legend.bind("<Configure>", lambda e: self._timeline_status.configure(wraplength=max(200, e.width - 180)))
        ttk.Label(legend, text="● 已确认", foreground=LabelTimeline.DONE,
                  style="PanelMuted.TLabel").pack(side="left", padx=8)
        ttk.Label(legend, text="● 未确认", foreground=LabelTimeline.TODO,
                  style="PanelMuted.TLabel").pack(side="left")
        self._timeline = LabelTimeline(timeline_host, on_seek=self._seek_timeline)
        self._timeline.pack(fill="x")

        canvas_host = ttk.Frame(right, style="Panel2.TFrame")
        canvas_host.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self._canvas = ImageAnnotatorCanvas(canvas_host, bg=Theme.PANEL_2)
        self._canvas.pack(fill="both", expand=True)
        self._mesh_notice = ttk.Label(canvas_host, text="", style="PanelMuted.TLabel", padding=8)

    def on_hide(self) -> None:
        self._stop_original_mesh_cache()
        self._decode_generation += 1
        self._cancel_backend_heartbeat()
        self._cleanup_decode_cache()
        self._canvas.clear()
        self._timeline.set_data(frames=[], position=0, done=set())
        self._timeline_status.configure(text="帧进度")
        self._jsonl_path = None
        self._backend_session = None
        self._backend_completed = False
        self._mode = "correct"
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
        self._mano_projection_errors = {}
        if self._frame_status is not None:
            self._frame_status.configure(text="")

    def release_on_exit(self) -> None:
        self._stop_original_mesh_cache()
        self._release_backend_job("operator_exit")

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
        return self._apply_session_tasks(tasks, progress, f"Tasks: {p}", initial_source="correct")

    def _set_backend_session(self, session: LabelJobSession) -> bool:
        try:
            task = correction_task_from_backend_payload(session.payload, mounts=session.mounts)
            task = replace(task, episode=episode_display_id(session.payload, session.job))
        except Exception as exc:
            messagebox.showerror("Backend Task", str(exc))
            return False

        self._decode_generation += 1
        generation = self._decode_generation
        self._cleanup_decode_cache()
        cache_parent = self._config.frame_cache_dir
        if cache_parent is not None:
            cache_parent.mkdir(parents=True, exist_ok=True)
            cache_dir = Path(tempfile.mkdtemp(prefix=f"orbbec_label_{session.job_id}_", dir=str(cache_parent)))
        else:
            cache_dir = Path(tempfile.mkdtemp(prefix=f"orbbec_label_{session.job_id}_"))
        self._decode_cache_dir = cache_dir
        self._backend_session = session
        self._backend_completed = False
        self._jsonl_path = None
        display_id = episode_display_id(session.payload, session.job)
        self._apply_session_tasks([], {}, f"Episode ID: {display_id}    Decoding RGB frames...", initial_source="correct")
        if self._frame_status is not None:
            self._frame_status.configure(text="正在解码 RGB 帧...", fg=STATUS_TODO_COLOR)
        self._schedule_backend_heartbeat()

        thread = threading.Thread(
            target=self._decode_backend_session_worker,
            args=(generation, session, task, cache_dir),
            daemon=True,
        )
        self._decode_thread = thread
        thread.start()
        return True

    def _decode_backend_session_worker(
        self,
        generation: int,
        session: LabelJobSession,
        task: CorrectionTask,
        cache_dir: Path,
    ) -> None:
        try:
            decoded_task = ensure_decoded_rgb_frames(
                task,
                session.payload,
                cache_root=cache_dir,
                ffmpeg_executable=self._config.ffmpeg_executable,
            )
            progress_ref = decoded_task.episode_dir() / f".{session.job_id}.jsonl"
            progress = load_correction_progress(str(progress_ref), [decoded_task])
            save_correction_progress(str(progress_ref), progress)
        except Exception as exc:
            self.after(0, lambda error=exc: self._finish_backend_decode(generation, None, None, "", error))
            return
        self.after(
            0,
            lambda: self._finish_backend_decode(
                generation,
                decoded_task,
                progress,
                str(progress_ref),
                None,
            ),
        )

    def _finish_backend_decode(
        self,
        generation: int,
        task: Optional[CorrectionTask],
        progress: Optional[Dict[str, CorrectionProgress]],
        progress_ref: str,
        error: Optional[BaseException],
    ) -> None:
        if generation != self._decode_generation:
            return
        if error is not None or task is None or progress is None:
            message = str(error or "Failed to decode RGB frames.")
            self._info.configure(text=f"Backend RGB decode failed: {message}")
            if self._frame_status is not None:
                self._frame_status.configure(text="RGB 解码失败", fg=STATUS_TODO_COLOR)
            messagebox.showerror("Backend Task", message)
            return

        self._jsonl_path = progress_ref
        self._apply_session_tasks(
            [task],
            progress,
            f"Episode ID: {episode_display_id(self._backend_session.payload, self._backend_session.job) if self._backend_session else task.episode}",
            auto_open_first=True,
            initial_source="correct",
        )

    def _cleanup_decode_cache(self) -> None:
        self._stop_original_mesh_cache()
        cache_dir = self._decode_cache_dir
        self._decode_cache_dir = None
        if cache_dir is None:
            return
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception:
            pass

    def _apply_session_tasks(
        self,
        tasks: List[CorrectionTask],
        progress: Dict[str, CorrectionProgress],
        info_prefix: str,
        *,
        auto_open_first: bool = False,
        initial_source: str = "correct",
    ) -> bool:
        self._mode = self._normalize_source(initial_source)
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
        self._mano_projection_errors = {}
        self._update_source_button()
        self._timeline.set_data(frames=[], position=0, done=set())
        self._timeline_status.configure(text="帧进度")

        for item in self._tree.get_children():
            self._tree.delete(item)
        for task in self._tasks:
            rec = self._progress.get(task.key)
            done = rec.done_count if rec else 0
            self._tree.insert("", "end", iid=task.key, values=(task.display_name, done, task.total_frames))

        self._info.configure(text=f"{info_prefix}    视图：{self._source_label()}")
        if self._frame_status is not None:
            self._frame_status.configure(text="")
        if auto_open_first and self._tasks:
            first_key = self._tasks[0].key
            self._tree.selection_set(first_key)
            self._tree.focus(first_key)
            self._load_task(first_key)
        else:
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
        if not sel or sel[0] == self._active_key:
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

        self._stop_original_mesh_cache()
        self._active_key = task.key
        self._active_task = task
        self._active_bundle = bundle
        self._bundles = {"pred": bundle}
        self._camera_ids = list(task.cameras)
        self._cam_idx = 0
        self._view_states = {}
        self._source_state_cache = {}
        self._reset_visualizations()
        self._mano_projection_errors = {}
        self._frame_pos = self._next_unconfirmed_position(task, 0)
        if self._frame_pos < 0:
            self._frame_pos = 0
        self._load_current_sample()
        self._start_original_mesh_cache()

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
        self._view_states = {}
        self._cam_idx = max(0, min(self._cam_idx, max(0, len(self._camera_ids) - 1)))
        self._refresh_view()

    def _normalize_source(self, source: str) -> str:
        source = (source or "").strip().lower()
        return source if source in SOURCE_ORDER else "correct"

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
            return
        self._view_states = {
            cam_id: self._build_initial_view_state(frame_idx, cam_id, self._mode) for cam_id in self._camera_ids
        }

    def _ensure_all_view_states_loaded(self, frame_idx: int) -> None:
        task = self._active_task
        if task is None:
            return
        if self._mode == "tracking":
            errors: List[str] = []
            for cam_id in self._camera_ids:
                if cam_id not in self._view_states:
                    self._view_states[cam_id] = self._build_tracking_view_state(frame_idx, cam_id, errors)
            self._show_tracking_errors_once(frame_idx, errors)
            return
        for cam_id in self._camera_ids:
            if cam_id not in self._view_states:
                self._view_states[cam_id] = self._build_initial_view_state(frame_idx, cam_id, self._mode)

    def _build_initial_view_state(self, frame_idx: int, cam_id: str, source: str) -> Tuple[HandPoints, HandVisible]:
        source = self._normalize_source(source)
        cache_key = self._source_cache_key(cam_id, source)
        if cache_key is not None and cache_key in self._source_state_cache:
            return self._copy_view_state(self._source_state_cache[cache_key])

        if source == "tracking":
            errors: List[str] = []
            state = self._build_tracking_view_state(frame_idx, cam_id, errors)
            self._show_tracking_errors_once(frame_idx, errors)
            return state
        if source == "mano":
            state = self._build_mano_3d_view_state(frame_idx, cam_id)
            if state is None:
                return self._hidden_points(), self._none_visible()
            return state

        if source == "correct":
            return self._build_modified_view_state(frame_idx, cam_id)

        bundle = self._ensure_bundle(source)
        if self._is_source_missing(source, frame_idx, cam_id):
            return self._hidden_points(), self._none_visible()

        return view_state_from_bundle(bundle, frame_idx, cam_id)

    def _build_modified_view_state(self, frame_idx: int, cam_id: str) -> Tuple[HandPoints, HandVisible]:
        bundle = self._ensure_bundle("correct")
        original = self._build_visible_mano_view_state(frame_idx, cam_id)
        if source_frame_path(bundle, frame_idx, cam_id) is not None:
            saved_points, saved_visible = view_state_from_bundle(bundle, frame_idx, cam_id)
            if original is None:
                return saved_points, saved_visible
            original_points, _original_visible = original
            # Corrected files encode invisible points as (-1,-1). Keep their
            # original projected position in memory so making a joint visible
            # again writes a real coordinate instead of persisting (-1,-1).
            for hand in range(2):
                for joint in range(21):
                    if not saved_visible[hand][joint]:
                        saved_points[hand][joint] = original_points[hand][joint]
            return saved_points, saved_visible

        if original is None:
            return self._hidden_points(), self._none_visible()
        return self._copy_view_state(original)

    def _build_mano_3d_view_state(self, frame_idx: int, cam_id: str) -> Optional[Tuple[HandPoints, HandVisible]]:
        task = self._active_task
        if task is None:
            return None
        err_key = (task.key, int(frame_idx), str(cam_id))
        self._mano_projection_errors.pop(err_key, None)
        try:
            state = self._mano_runtime_instance().project_mano_frame(
                episode_dir=task.episode_dir(),
                mano_dir=task.episode_dir() / task.mano_episode_dir,
                cam_id=cam_id,
                frame_idx=frame_idx,
            )
        except Exception as exc:
            self._mano_projection_errors[err_key] = str(exc)
            return None
        if state is None:
            self._mano_projection_errors[err_key] = describe_mano_projection_issue(
                task.episode_dir(),
                task.episode_dir() / task.mano_episode_dir,
                cam_id,
                frame_idx,
            )
            return None

        return state

    def _build_visible_mano_view_state(self, frame_idx: int, cam_id: str) -> Optional[Tuple[HandPoints, HandVisible]]:
        state = self._build_mano_3d_view_state(frame_idx, cam_id)
        if state is None:
            return None
        task = self._active_task
        err_key = (task.key, int(frame_idx), str(cam_id))
        points, projected_visible = state
        try:
            original_visible = load_joint_visibility(
                task.episode_dir() / JOINTS_VIS_DIR,
                cam_id,
                frame_idx,
            )
        except Exception as exc:
            self._mano_projection_errors[err_key] = str(exc)
            return None
        if original_visible is None:
            self._mano_projection_errors[err_key] = f"Missing joints_vis for camera {cam_id}, frame {frame_idx}"
            return None

        visible = [
            [bool(projected_visible[hand][joint] and original_visible[hand][joint]) for joint in range(21)]
            for hand in range(2)
        ]
        return points, visible

    @staticmethod
    def _is_hidden_point(point: Tuple[float, float]) -> bool:
        x, y = point
        return float(x) == -1.0 and float(y) == -1.0

    def _is_source_missing(self, source: str, frame_idx: int, cam_id: str) -> bool:
        source = self._normalize_source(source)
        if source == "tracking":
            return False
        if source == "mano":
            task = self._active_task
            if task is None:
                return True
            try:
                state = self._mano_runtime_instance().project_mano_frame(
                    episode_dir=task.episode_dir(),
                    mano_dir=task.episode_dir() / task.mano_episode_dir,
                    cam_id=cam_id,
                    frame_idx=frame_idx,
                )
            except Exception:
                return True
            return state is None
        bundle = self._ensure_bundle(source)
        return source_frame_path(bundle, frame_idx, cam_id) is None

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
                    correction_dir=task.correction_dir,
                    rgb_path_template=task.rgb_path_template,
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
        if source == "mano":
            raise ValueError("MANO mode uses episode 3D results and does not load a 2D prediction bundle.")
        if source == "tracking":
            raise ValueError("Tracking mode uses runtime CoTracker results and does not load a prediction bundle.")
        bundle = self._bundles.get(source)
        if bundle is not None:
            return bundle
        if self._active_task is None:
            raise ValueError("No active task.")
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
            if key[0] == self._active_key and key[1] == int(self._frame_pos) and key[3] in {"correct", "last"}
        ]
        for key in stale:
            self._source_state_cache.pop(key, None)

    def _source_label(self, source: Optional[str] = None) -> str:
        source = self._normalize_source(source or self._mode)
        return SOURCE_LABELS[source]

    def _view_source_status(self, source: str, frame_idx: int, cam_id: str, visible: HandVisible) -> str:
        source = self._normalize_source(source)
        label = self._source_label(source)
        if source == "mano":
            if self._has_any_visible(visible):
                return label
            return f"{label} (missing: {self._mano_projection_error(frame_idx, cam_id)})"
        if source == "correct":
            bundle = self._ensure_bundle("correct")
            if source_frame_path(bundle, frame_idx, cam_id) is not None:
                return label
            if self._has_any_visible(visible):
                return f"{label}（初始值）"
            return f"{label} (missing original: {self._mano_projection_error(frame_idx, cam_id)})"
        if self._is_source_missing(source, frame_idx, cam_id):
            return f"{label} (missing)"
        return label

    def _mano_projection_error(self, frame_idx: int, cam_id: str) -> str:
        task = self._active_task
        if task is None:
            return "no active task"
        key = (task.key, int(frame_idx), str(cam_id))
        message = self._mano_projection_errors.get(key)
        if message:
            return message
        return describe_mano_projection_issue(
            task.episode_dir(),
            task.episode_dir() / task.mano_episode_dir,
            cam_id,
            frame_idx,
        )

    @staticmethod
    def _has_any_visible(visible: HandVisible) -> bool:
        for hand in visible:
            for item in hand:
                if item:
                    return True
        return False

    def _source_button_text(self) -> str:
        return f"视图：{self._source_label()}"

    def _update_source_button(self) -> None:
        if self._source_btn is not None:
            self._source_btn.configure(text=self._source_button_text())
            self._source_btn.master._schedule()

    def _skeleton_button_text(self) -> str:
        return "Hide Skeleton" if self._show_skeleton else "Show Skeleton"

    def _update_skeleton_button(self) -> None:
        if self._skeleton_btn is not None:
            self._skeleton_btn.configure(text=self._skeleton_button_text())
            self._skeleton_btn.master._schedule()

    def _mano_button_text(self) -> str:
        return "Hide MANO" if self._show_mano else "Show MANO"

    def _update_mano_button(self) -> None:
        if self._mano_btn is not None:
            self._mano_btn.configure(text=self._mano_button_text())
            self._mano_btn.master._schedule()

    def _reset_mano(self) -> None:
        self._clear_original_mesh_preview()
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
        self._clear_original_mesh_preview()
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
            self._canvas.set_read_only(active or self._mode == "mano")
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
        self._ensure_all_view_states_loaded(frame_idx)

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

    def _start_original_mesh_cache(self) -> None:
        if self._active_task is None or self._original_mesh_cache is not None:
            return
        try:
            self._original_mesh_cache = OriginalMeshCache(self._active_task)
        except Exception as exc:
            self._mesh_notice.configure(text=f"MANO 渲染失败：{exc}")

    def _stop_original_mesh_cache(self) -> None:
        self._clear_original_mesh_preview()
        cache = self._original_mesh_cache
        self._original_mesh_cache = None
        if cache is not None:
            cache.close()

    def _clear_original_mesh_preview(self) -> None:
        if self._mesh_poll_id is not None:
            self.after_cancel(self._mesh_poll_id)
            self._mesh_poll_id = None
        self._canvas.set_rendered_image(None)
        self._mesh_notice.place_forget()

    def _refresh_original_mesh_preview(self) -> None:
        if self._mesh_poll_id is not None:
            self.after_cancel(self._mesh_poll_id)
            self._mesh_poll_id = None
        if not self._show_mano or self._active_task is None:
            return
        cache = self._original_mesh_cache
        camera = self._active_cam_id()
        frame = self._active_task.frames[self._frame_pos]
        path = cache.path(camera, frame) if cache is not None else None
        self._canvas.set_rendered_image(path)
        if path is not None:
            self._mesh_notice.place_forget()
            return
        if cache is not None:
            text = f"MANO 渲染失败：{cache.error}" if cache.error else f"MANO 后台渲染中 · 机位 {camera} · 帧 {frame}"
            self._mesh_notice.configure(text=text)
            if not cache.done_event.is_set():
                self._mesh_poll_id = self.after(200, self._refresh_original_mesh_preview)
        self._mesh_notice.place(relx=0.5, y=12, anchor="n")

    def _toggle_mano(self) -> None:
        if self._show_mano:
            self._reset_mano()
            return
        if self._active_task is None:
            return
        self._cache_current_source_state()
        self._reset_skeleton()
        self._show_mano = True
        self._start_original_mesh_cache()
        self._update_mano_button()
        self._refresh_original_mesh_preview()
        self._sync_visualization_canvas_state()

    def _refresh_mano_overlay(self) -> None:
        if self._show_mano:
            self._refresh_original_mesh_preview()
        else:
            self._canvas.set_mano_overlay([])

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
        self._view_states = {}
        try:
            self._refresh_view()
        except Exception as exc:
            self._mode = old_source
            self._view_states = {}
            self._update_source_button()
            self._refresh_view()
            messagebox.showerror("Error", str(exc))
            return

        self._update_source_button()

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
            self._view_states[cam_id] = state
        points, visible = state

        img_path = find_frame_path(task.episode_dir(), cam_id, frame_idx, task.rgb_path_template)
        self._canvas.set_image(img_path)
        self._canvas.set_hand_state(points, visible)
        self._canvas.set_count_base(self._visibility_counts(exclude_cam=cam_id))
        self._update_source_button()
        self._refresh_visual_overlays()

        self._refresh_timeline()
        rec = self._progress.get(task.key)
        done = rec.done_count if rec else 0
        frame_done = self._is_frame_done(task, self._frame_pos)
        self._info.configure(
            text=(
                f"Task: {task.display_name}    Camera: {cam_id} ({self._cam_idx + 1}/{len(self._camera_ids)})    "
                f"Frame: {frame_idx} ({self._frame_pos + 1}/{task.total_frames})    "
                f"视图：{self._view_source_status(source, frame_idx, cam_id, visible)}    Done: {done}/{task.total_frames}"
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

    def _camera_shortcut(self, event, index: int):
        if not self.winfo_ismapped() or self._active_task is None:
            return None
        if event.widget.winfo_class() in {"Entry", "TEntry", "Text", "Spinbox", "TSpinbox", "TCombobox"}:
            return None
        if event.state & (0x4 | 0x8 | 0x20000):
            return None
        if index >= len(self._camera_ids):
            return None
        self._cache_current_source_state()
        self._cam_idx = index
        self._refresh_view()
        return "break"

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
        if self._mode == "mano" or self._visualization_active():
            return
        self._canvas.undo()

    def _ignore_view(self) -> None:
        if self._mode == "mano" or self._visualization_active():
            return
        self._canvas.ignore_view()

    def _refresh_timeline(self) -> None:
        task = self._active_task
        if task is None:
            return
        rec = self._progress.get(task.key)
        self._timeline.set_data(frames=task.frames, position=self._frame_pos,
                                done=rec.done_positions if rec else set())
        self._set_timeline_status(self._frame_pos)

    def _set_timeline_status(self, position: int) -> None:
        task = self._active_task
        if task is None or not task.frames:
            return
        rec = self._progress.get(task.key)
        done = rec.done_count if rec else 0
        status = "已确认" if rec and position in rec.done_positions else "未确认"
        self._timeline_status.configure(
            text=f"帧 {task.frames[position]} · {position + 1}/{task.total_frames} · {status} · 已确认 {done}/{task.total_frames}"
        )

    def _seek_timeline(self, position: int, final: bool) -> None:
        self._set_timeline_status(position)
        if final:
            self._jump_to_frame(position)

    def _jump_to_frame(self, position: int) -> None:
        task = self._active_task
        if task is None or task.total_frames <= 0:
            return
        position = max(0, min(task.total_frames - 1, int(position)))
        if position == self._frame_pos:
            return
        self._cache_current_source_state()
        self._view_states = {}
        keep_mano = self._show_mano
        self._reset_visualizations()
        self._show_mano = keep_mano
        self._update_mano_button()
        self._frame_pos = position
        self._load_current_sample()

    def _skip_frame(self) -> None:
        self._jump_to_frame(self._frame_pos + 1)

    def _back_frame(self) -> None:
        self._jump_to_frame(self._frame_pos - 1)

    def _confirm(self) -> None:
        task = self._active_task
        bundle = self._save_bundle()
        if task is None or bundle is None or self._active_key is None or self._jsonl_path is None:
            return
        if self._show_skeleton:
            messagebox.showwarning("Notice", "Please hide Skeleton visualization before confirming annotations.")
            return
        if not self._camera_ids:
            messagebox.showwarning("Notice", "No camera directories found for the current task.")
            return
        self._cache_current_source_state()
        frame_idx = task.frames[self._frame_pos]
        try:
            # Preview state is never the annotation source: original view omits
            # visibility. Use edits, saved corrections, or masked initial values.
            corrected_states = {
                cam_id: self._build_initial_view_state(frame_idx, cam_id, "correct")
                for cam_id in self._camera_ids
            }
            for cam_id in self._camera_ids:
                points, visible = corrected_states[cam_id]
                apply_view_state_to_corrected(bundle, frame_idx, cam_id, points, visible)
            save_corrected_array(bundle)
            self._invalidate_corrected_source_cache()
        except Exception as exc:
            self._fail_backend_job(f"save_failed: {exc}")
            messagebox.showerror("Error", str(exc))
            return

        rec = self._progress.get(task.key)
        if rec is None:
            rec = CorrectionProgress(task_key=task.key, total_frames=task.total_frames)
            self._progress[task.key] = rec
        was_complete = rec.done_count >= task.total_frames
        rec.done_positions.add(self._frame_pos)
        save_correction_progress(self._jsonl_path, self._progress)
        self._update_tree_row(task)

        next_pos = min(task.total_frames - 1, self._frame_pos + 1)
        keep_mano = self._show_mano
        self._reset_visualizations()
        self._show_mano = keep_mano
        self._update_mano_button()
        if rec.done_count >= task.total_frames:
            if not self._complete_backend_job(task):
                self._refresh_view()
                return
            if not was_complete:
                messagebox.showinfo("Done", "This task is fully completed.")

        self._frame_pos = next_pos
        self._view_states = {}
        self._load_current_sample()

    def _complete_backend_job(self, task: CorrectionTask) -> bool:
        session = self._backend_session
        if session is None or self._backend_completed:
            return True
        segment_ids = [
            str(item.get("segment_id") or "")
            for item in (session.payload.get("segments") or [])
            if isinstance(item, dict) and str(item.get("segment_id") or "")
        ]
        artifacts = [
            {
                "kind": "manual_2d",
                "metadata": {
                    "scope": "episode",
                    "episode_id": session.episode_id,
                    "segment_ids": segment_ids,
                    "cameras": list(task.cameras),
                    "frames": list(task.frames),
                    "operator_id": session.operator_id,
                },
            }
        ]
        try:
            session.client.complete_label_job(
                session.job_id,
                result={
                    "operator_id": session.operator_id,
                    "frames_completed": list(task.frames),
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            messagebox.showerror("Backend", f"Failed to complete label job: {exc}")
            return False
        self._backend_completed = True
        self._cancel_backend_heartbeat()
        return True

    def _fail_backend_job(self, error: str) -> None:
        session = self._backend_session
        if session is None or self._backend_completed:
            return
        try:
            session.client.fail_label_job(
                session.job_id,
                error=error,
                result={"operator_id": session.operator_id},
            )
            self._backend_completed = True
            self._cancel_backend_heartbeat()
        except Exception:
            pass

    def _update_tree_row(self, task: CorrectionTask) -> None:
        rec = self._progress.get(task.key)
        done = rec.done_count if rec else 0
        self._tree.set(task.key, "done", str(done))
        self._tree.set(task.key, "total", str(task.total_frames))
