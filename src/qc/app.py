from __future__ import annotations

import queue
import signal
import threading
import time
import tkinter as tk
from frontend_runtime import cancel_tk_callbacks
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from label.canvas_view import ImageAnnotatorCanvas
    from label.theme import Theme, apply_theme, fit_window, WrapToolbar
except Exception:
    from ...label.canvas_view import ImageAnnotatorCanvas  # type: ignore
    from ...label.theme import Theme, apply_theme, fit_window, WrapToolbar  # type: ignore

from .backend import QcBackendClient, QcBackendError, episode_display_id
from .config import QcConfig
from .media import MeshRendererSettings, QcEpisodeMedia, prepare_qc_media
from .playback import playback_target_position
from .report import build_qc_result, write_ego_pose_qc_report, write_qc_report
from .state_store import QcProgress, QcStateStore, format_seconds, normalize_ranges


class QcWorkerApp(tk.Tk):
    def destroy(self) -> None:
        cancel_tk_callbacks(self)
        super().destroy()

    def __init__(self, config: QcConfig):
        super().__init__()
        self.config = config
        self.client = QcBackendClient(config.backend_url, timeout_seconds=config.request_timeout_seconds)
        self.state_store = QcStateStore(config.state_dir)

        self.title("Orbbec 人工 QC Worker")
        self.configure(bg=Theme.BG)
        apply_theme(self)
        fit_window(self)

        self._current_progress: Optional[QcProgress] = None
        self._current_media: Optional[QcEpisodeMedia] = None
        self._heartbeat_after_id: Optional[str] = None
        self._last_task_name: str = ""
        self._closing = False
        self._exit_after_cleanup = False
        self._pending_work = 0
        self._opening_episode = False
        self._media_stop = threading.Event()
        self._media_updates = queue.Queue()

        self._host = ttk.Frame(self, style="TFrame")
        self._host.pack(fill="both", expand=True)

        self.task_page = TaskSelectionPage(self._host, app=self)
        self.episode_page = EpisodeSelectionPage(self._host, app=self)
        self.decode_page = DecodePage(self._host, app=self)
        self.qc_page = QcPage(self._host, app=self)
        for page in (self.task_page, self.episode_page, self.decode_page, self.qc_page):
            page.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.protocol("WM_DELETE_WINDOW", self.request_exit)
        self._install_signal_handlers()
        self.show_tasks()

    def _install_signal_handlers(self) -> None:
        def handler(_signum, _frame) -> None:
            self.after_idle(self.request_exit)

        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                signal.signal(sig, handler)
            except Exception:
                pass

    def show_tasks(self) -> None:
        self._cancel_heartbeat()
        self.task_page.lift()
        self.refresh_tasks()

    def show_episodes(self, task_name: str) -> None:
        self._last_task_name = task_name
        self.episode_page.lift()
        self.refresh_episodes(task_name)

    def refresh_tasks(self) -> None:
        self.task_page.set_loading("正在刷新 QC 任务...")

        def work() -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
            try:
                return self._task_models(), None
            except Exception as exc:
                return None, str(exc)

        self._run_bg(work, lambda result: self.task_page.set_tasks(result[0] or [], error=result[1]))

    def refresh_episodes(self, task_name: str) -> None:
        self.episode_page.set_loading(f"正在刷新 {task_name}...")

        def work() -> Tuple[Optional[List[Tuple[str, Any]]], Optional[str]]:
            try:
                return self._episode_models(task_name), None
            except Exception as exc:
                return None, str(exc)

        self._run_bg(work, lambda result: self.episode_page.set_episodes(task_name, result[0] or [], error=result[1]))

    def _task_models(self) -> List[Dict[str, Any]]:
        items = self.client.available_qc_items()
        local = self.state_store.list_progress(worker_machine_id=self.config.worker_machine_id)
        groups: Dict[str, Dict[str, Any]] = {}
        for item in items:
            task = str(item.get("task_name") or "Unspecified").strip() or "Unspecified"
            group = groups.setdefault(task, {"task_name": task, "queued": 0, "frames": 0, "local": [], "items": []})
            group["queued"] += 1
            group["items"].append(item)
            try:
                group["frames"] += max(0, int(item.get("frames_count") or len(item.get("frames") or [])))
            except (TypeError, ValueError):
                pass
        for progress in local:
            task = progress.task_name or "Unspecified"
            group = groups.setdefault(task, {"task_name": task, "queued": 0, "frames": 0, "local": [], "items": []})
            group["local"].append(progress)
        return sorted(groups.values(), key=lambda item: str(item.get("task_name") or ""))

    def _episode_models(self, task_name: str) -> List[Tuple[str, Any]]:
        items = [item for item in self.client.available_qc_items() if str(item.get("task_name") or "") == task_name]
        local = [
            progress
            for progress in self.state_store.list_progress(worker_machine_id=self.config.worker_machine_id)
            if (progress.task_name or "Unspecified") == task_name
        ]
        local_keys = {(p.job_id, p.episode_id) for p in local}
        rows: List[Tuple[str, Any]] = [("local", progress) for progress in local]
        for item in items:
            key = (str(item.get("job_id") or ""), str(item.get("episode_id") or ""))
            if key in local_keys:
                continue
            rows.append(("queued", item))
        return rows

    def open_episode_entry(self, kind: str, entry: Any) -> None:
        if self._closing or self._opening_episode or self._current_progress is not None:
            return
        self._opening_episode = True
        if kind == "local":
            self._restore_progress(entry)
        else:
            self._lease_episode(entry)

    def _lease_episode(self, item: Dict[str, Any]) -> None:
        task_name = str(item.get("task_name") or "")
        episode_id = str(item.get("episode_id") or "")
        job_id = str(item.get("job_id") or "")
        self.episode_page.set_status("正在向后端租借 Episode...")

        def work() -> Tuple[Optional[QcProgress], Optional[str]]:
            try:
                response = self.client.lease_qc_job(
                    worker_id=self.config.worker_machine_id,
                    lease_seconds=self.config.default_lease_seconds,
                    task_name=task_name,
                    episode_id=episode_id,
                    job_id=job_id,
                )
                progress = QcProgress.from_lease_response(
                    response,
                    worker_machine_id=self.config.worker_machine_id,
                    sample_interval=self.config.sample_interval,
                )
                self.state_store.save(progress)
                return progress, None
            except Exception as exc:
                return None, str(exc)

        self._run_bg(work, lambda result: self._after_progress_ready(result[0], result[1]))

    def _restore_progress(self, progress: QcProgress) -> None:
        self.episode_page.set_status("正在校验本地保留进度...")

        def work() -> Tuple[Optional[QcProgress], Optional[str]]:
            try:
                response = self.client.get_job(progress.job_id)
                job = dict(response.get("job") or {})
                status = str(job.get("status") or "")
                if status in {"succeeded", "failed", "canceled"}:
                    self.state_store.delete(progress)
                    return None, "该 QC job 已经结束，本地进度已删除。"
                lease_owner = str(job.get("lease_owner") or "")
                lease_until = str(job.get("lease_until") or progress.lease_until)
                progress.job = job
                progress.payload = dict(response.get("payload") or progress.payload)
                progress.episode = dict(response.get("episode") or progress.episode)
                progress.artifacts = [dict(item) for item in response.get("artifacts") or [] if isinstance(item, dict)]
                progress.lease_until = lease_until
                if status in {"leased", "running"} and progress.lease_valid and lease_owner == self.config.worker_machine_id:
                    self.state_store.save(progress)
                    return progress, None
                response = self.client.lease_qc_job(
                    worker_id=self.config.worker_machine_id,
                    lease_seconds=self.config.default_lease_seconds,
                    task_name=progress.task_name,
                    episode_id=progress.episode_id,
                    job_id=progress.job_id,
                )
                leased = QcProgress.from_lease_response(
                    response,
                    worker_machine_id=self.config.worker_machine_id,
                    sample_interval=progress.sample_interval,
                )
                leased.current_frame = progress.current_frame
                leased.bad_frame_ranges = list(progress.bad_frame_ranges)
                leased.ego_bad_frame_ranges = list(progress.ego_bad_frame_ranges)
                leased.checked_sample_frames = list(progress.checked_sample_frames)
                leased.playback_complete = progress.playback_complete
                leased.result_type = progress.result_type
                self.state_store.save(leased)
                return leased, None
            except Exception as exc:
                self.state_store.delete(progress)
                return None, f"无法恢复该 Episode，已删除本地进度：{exc}"

        self._run_bg(work, lambda result: self._after_progress_ready(result[0], result[1]))

    def _after_progress_ready(self, progress: Optional[QcProgress], error: Optional[str]) -> None:
        self._opening_episode = False
        if self._closing:
            # A lease request can finish after the window's close button is pressed.
            if progress is not None:
                self._current_progress = progress
            return
        if error:
            messagebox.showerror("QC", error)
            if self._last_task_name:
                self.refresh_episodes(self._last_task_name)
            else:
                self.refresh_tasks()
            return
        if progress is None:
            return
        self._current_progress = progress
        self._schedule_heartbeat()
        self.prepare_episode(progress)

    def prepare_episode(self, progress: QcProgress) -> None:
        self._media_stop = threading.Event()
        stop = self._media_stop
        self.decode_page.reset(progress)
        self.decode_page.lift()

        def progress_callback(_camera: str, status: Dict[str, Any]) -> None:
            self._media_updates.put(status)

        def poll_updates() -> None:
            while not self._media_updates.empty():
                status = self._media_updates.get_nowait()
                if not self._closing:
                    self.decode_page.update_camera(status)
            if not stop.is_set():
                self.after(100, poll_updates)

        self.after(100, poll_updates)

        def work() -> Tuple[Optional[QcEpisodeMedia], Optional[str]]:
            try:
                media = prepare_qc_media(
                    progress.payload,
                    mounts=self.config.nas_mounts,
                    tmp_dir=self.config.tmp_dir,
                    on_progress=progress_callback,
                    stop_event=stop,
                    on_media_created=lambda media: setattr(self, "_current_media", media),
                    mesh_renderer=MeshRendererSettings(
                        python_executable=self.config.mesh_renderer_python,
                        mano_toolkit_root=self.config.mano_toolkit_root,
                        mano_model_dir=self.config.mano_model_dir,
                        render_factor=self.config.mesh_render_factor,
                        workers=self.config.mesh_render_workers,
                        prefer_integrated_gpu=self.config.mesh_prefer_integrated_gpu,
                        prebuffer_frames=self.config.mesh_prebuffer_frames,
                    ),
                )
                return media, None
            except Exception as exc:
                return None, str(exc)

        self._run_bg(work, lambda result: self._after_media_ready(result[0], result[1]))

    def _after_media_ready(self, media: Optional[QcEpisodeMedia], error: Optional[str]) -> None:
        if self._closing:
            if media is not None:
                self._current_media = media
            return
        if error:
            self._cancel_heartbeat()
            messagebox.showerror("数据准备失败", error)
            self.handle_interrupt(exit_after=False)
            return
        if media is None or self._current_progress is None:
            return
        self._current_media = media
        self.qc_page.set_session(self._current_progress, media)
        self.qc_page.lift()
        self._schedule_heartbeat()

    def save_current_progress(self) -> None:
        if self._current_progress is not None:
            self.state_store.save(self._current_progress)

    def submit_current_progress(self, *, bad_episode: bool = False) -> None:
        progress = self._current_progress
        media = self._current_media
        if progress is None or media is None:
            return
        if not bad_episode and not progress.is_complete:
            messagebox.showwarning("尚未完成", "当前 Episode 尚未完成一次播放，暂不能提交。")
            return
        bad_ranges = normalize_ranges(progress.bad_frame_ranges, max_gap_frames=self.config.range_merge_gap_frames)
        ego_bad_ranges = normalize_ranges(progress.ego_bad_frame_ranges, max_gap_frames=self.config.range_merge_gap_frames)
        report_ranges = [] if bad_episode else bad_ranges
        progress.bad_frame_ranges = bad_ranges
        progress.ego_bad_frame_ranges = ego_bad_ranges
        result = build_qc_result(
            episode_id=progress.episode_id,
            worker_id=self.config.worker_machine_id,
            bad_ranges=bad_ranges,
            bad_episode=bad_episode,
            sample_interval=progress.sample_interval,
        )
        result["operator_id"] = self.config.operator_id
        self.qc_page.set_busy("正在提交 QC 结果...")

        def work() -> Tuple[bool, Optional[str]]:
            try:
                write_qc_report(
                    episode_dir=media.episode_dir,
                    result=result,
                    bad_ranges=report_ranges,
                    sample_interval=progress.sample_interval,
                )
                write_ego_pose_qc_report(
                    episode_dir=media.episode_dir,
                    episode_id=progress.episode_id,
                    worker_id=self.config.worker_machine_id,
                    operator_id=self.config.operator_id,
                    bad_ranges=ego_bad_ranges,
                )
                artifacts = [
                    {
                        "kind": "qc_report",
                        "metadata": {
                            "passed": bool(result.get("passed")),
                            "worker_id": self.config.worker_machine_id,
                            "operator_id": self.config.operator_id,
                            "result_type": result.get("result_type"),
                        },
                    }
                ]
                self.client.complete_qc_job(progress.job_id, result=result, artifacts=artifacts, operator_id=self.config.operator_id)
                self._current_progress = None
                self.state_store.delete(progress)
                media.close(cleanup=True)
                return True, None
            except Exception as exc:
                return False, str(exc)

        def done(result_pair: Tuple[bool, Optional[str]]) -> None:
            ok, error = result_pair
            self.qc_page.set_busy("")
            if not ok:
                messagebox.showerror("提交失败", error or "后端没有确认提交成功。")
                return
            if not self._closing:
                messagebox.showinfo("提交成功", "后端已确认 QC 结果。")
            self.qc_page.clear_session()
            self._current_progress = None
            self._current_media = None
            self._cancel_heartbeat()
            if not self._closing:
                self.show_episodes(self._last_task_name or progress.task_name)

        self._run_bg(work, done)

    def handle_interrupt(self, *, exit_after: bool) -> None:
        if self._closing:
            self._exit_after_cleanup = self._exit_after_cleanup or exit_after
            return
        self._closing = True
        self._exit_after_cleanup = exit_after
        self._cancel_heartbeat()
        self._media_stop.set()
        self.qc_page.set_busy("正在停止解码和渲染、清理缓存并释放任务...")
        self.decode_page._status.configure(text="正在停止解码和渲染、清理缓存并释放任务...")
        self.decode_page.lift()
        self._wait_for_cleanup()

    def _wait_for_cleanup(self) -> None:
        # Let every in-flight lease/heartbeat/submission deliver its result first.
        # No worker calls Tk, so stopping and joining media workers cannot deadlock.
        if self._pending_work:
            self.after(60, self._wait_for_cleanup)
            return
        progress = self._current_progress
        media = self._current_media

        def work() -> Optional[str]:
            errors = []
            if media is not None:
                try:
                    media.close(cleanup=True)
                except Exception as exc:
                    errors.append(f"临时缓存清理失败：{exc}")
            if progress is not None:
                try:
                    self.state_store.save(progress)
                except Exception as exc:
                    errors.append(f"进度保存失败：{exc}")
                try:
                    self.client.release_qc_job(progress.job_id, reason="operator_left_qc")
                    self._current_progress = None
                    progress.lease_until = ""
                    progress.job.update(status="queued", lease_owner="", lease_until="")
                    self.state_store.save(progress)
                except Exception as exc:
                    errors.append(f"任务释放失败：{exc}")
            return "\n".join(errors) or None

        def done(error: Optional[str]) -> None:
            self.qc_page.clear_session()
            self.qc_page.set_busy("")
            if error:
                self._closing = False
                self.decode_page._status.configure(text="退出未完成，请重试关闭窗口。")
                messagebox.showerror("退出未完成", error + "\n请重试退出。")
                return
            self._current_media = None
            if self._exit_after_cleanup:
                self.destroy()
            else:
                self._closing = False
                self.show_episodes(self._last_task_name)

        self._run_bg(work, done, during_close=True)

    def abandon_current_progress(self, *, exit_after: bool) -> None:
        self.handle_interrupt(exit_after=exit_after)

    def preserve_current_progress(self, *, lease_seconds: int, exit_after: bool) -> None:
        # Progress is retained locally; leaving the page always releases the lease.
        self.handle_interrupt(exit_after=exit_after)

    def preserve_for_crash_and_exit(self) -> None:
        self.request_exit()

    def request_exit(self) -> None:
        self.handle_interrupt(exit_after=True)

    def _schedule_heartbeat(self) -> None:
        self._cancel_heartbeat()
        if self._closing or self._current_progress is None:
            return
        seconds = max(30, self.config.default_lease_seconds // 3)
        self._heartbeat_after_id = self.after(seconds * 1000, self._heartbeat_active)

    def _heartbeat_active(self) -> None:
        progress = self._current_progress
        if progress is None:
            return

        def work() -> None:
            response = self.client.heartbeat_qc_job(
                progress.job_id,
                worker_id=self.config.worker_machine_id,
                lease_seconds=self.config.default_lease_seconds,
                status="running",
            )
            job = dict(response.get("job") or {})
            if job.get("lease_until"):
                progress.lease_until = str(job["lease_until"])
                progress.job = job
                self.state_store.save(progress)

        self._run_bg(lambda: _ignore_errors(work), lambda _result: self._schedule_heartbeat())

    def _cancel_heartbeat(self) -> None:
        if self._heartbeat_after_id is None:
            return
        try:
            self.after_cancel(self._heartbeat_after_id)
        except Exception:
            pass
        self._heartbeat_after_id = None

    def _run_bg(self, work: Callable[[], Any], done: Callable[[Any], None], *, during_close: bool = False) -> None:
        if self._closing and not during_close:
            return
        self._pending_work += 1
        q: "queue.Queue[Any]" = queue.Queue(maxsize=1)

        def runner() -> None:
            try:
                q.put((work(), None))
            except Exception as exc:
                q.put((None, exc))

        threading.Thread(target=runner, daemon=True).start()

        def poll() -> None:
            try:
                result = q.get_nowait()
            except queue.Empty:
                self.after(60, poll)
                return
            self._pending_work -= 1
            value, error = result
            if error is not None:
                messagebox.showerror("后台操作失败", str(error))
            else:
                done(value)

        self.after(60, poll)


def _ignore_errors(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception:
        pass


class TaskSelectionPage(ttk.Frame):
    def __init__(self, master, *, app: QcWorkerApp):
        super().__init__(master, style="TFrame")
        self.app = app
        self._groups: Dict[str, Dict[str, Any]] = {}
        self._countdown_after: Optional[str] = None
        self._build()

    def _build(self) -> None:
        outer = ttk.Frame(self, style="TFrame", padding=(24, 20))
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer, style="TFrame")
        top.pack(fill="x")
        ttk.Label(top, text="人工质检", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="刷新", style="Secondary.TButton", command=self.app.refresh_tasks).pack(side="right", padx=(8, 0))
        ttk.Button(top, text="退出", style="Secondary.TButton", command=self.app.request_exit).pack(side="right")
        self._status = ttk.Label(outer, text="", style="Muted.TLabel")
        self._status.pack(fill="x", pady=(8, 10))
        cols = ("task", "queued", "local", "expire")
        table = ttk.Frame(outer, style="Panel.TFrame")
        table.pack(fill="both", expand=True)
        self._tree = ttk.Treeview(table, columns=cols, show="headings")
        self._tree.heading("task", text="Task")
        self._tree.heading("queued", text="待质检 Episode")
        self._tree.heading("local", text="本机进行中")
        self._tree.heading("expire", text="最快过期")
        self._tree.column("task", width=420, anchor="w")
        self._tree.column("queued", width=140, anchor="center")
        self._tree.column("local", width=160, anchor="center")
        self._tree.column("expire", width=160, anchor="center")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self._tree.yview)
        scroll.pack(side="right", fill="y")
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side="left", fill="both", expand=True)
        self._tree.bind("<Double-1>", self._open_selected)
        self._tree.bind("<Return>", self._open_selected)

    def set_loading(self, text: str) -> None:
        self._status.configure(text=text)

    def set_tasks(self, groups: List[Dict[str, Any]], *, error: Optional[str] = None) -> None:
        self._groups = {str(group.get("task_name") or ""): group for group in groups}
        self._tree.delete(*self._tree.get_children())
        if error:
            self._status.configure(text=f"刷新失败：{error}")
            return
        self._status.configure(text=f"后端：{self.app.config.backend_url}    Worker：{self.app.config.worker_machine_id}")
        for group in groups:
            task = str(group.get("task_name") or "")
            local = list(group.get("local") or [])
            name = ("● " if local else "") + task
            self._tree.insert(
                "",
                "end",
                iid=task,
                values=(name, int(group.get("queued") or 0), len(local), self._expire_text(local)),
            )
        self._schedule_countdown()

    def _expire_text(self, local: List[QcProgress]) -> str:
        if not local:
            return ""
        active = [progress for progress in local if progress.lease_valid]
        if not active:
            return "已释放"
        return format_seconds(min(progress.lease_seconds_remaining for progress in active))

    def _schedule_countdown(self) -> None:
        if self._countdown_after is not None:
            try:
                self.after_cancel(self._countdown_after)
            except Exception:
                pass
        self._countdown_after = self.after(1000, self._tick_countdown)

    def _tick_countdown(self) -> None:
        for task, group in self._groups.items():
            if self._tree.exists(task):
                self._tree.set(task, "expire", self._expire_text(list(group.get("local") or [])))
        self._countdown_after = self.after(1000, self._tick_countdown)

    def _open_selected(self, _evt=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        self.app.show_episodes(sel[0])


class EpisodeSelectionPage(ttk.Frame):
    def __init__(self, master, *, app: QcWorkerApp):
        super().__init__(master, style="TFrame")
        self.app = app
        self._entries: Dict[str, Tuple[str, Any]] = {}
        self._task_name = ""
        self._countdown_after: Optional[str] = None
        self._build()

    def _build(self) -> None:
        outer = ttk.Frame(self, style="TFrame", padding=(24, 20))
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer, style="TFrame")
        top.pack(fill="x")
        self._title = ttk.Label(top, text="Episode 选择", style="Title.TLabel")
        self._title.pack(side="left")
        ttk.Button(top, text="返回 Task", style="Secondary.TButton", command=self.app.show_tasks).pack(side="right", padx=(8, 0))
        ttk.Button(top, text="刷新", style="Secondary.TButton", command=lambda: self.app.refresh_episodes(self._task_name)).pack(side="right")
        self._status = ttk.Label(outer, text="", style="Muted.TLabel")
        self._status.pack(fill="x", pady=(8, 10))
        cols = ("status", "episode", "subject", "frames", "lease")
        table = ttk.Frame(outer, style="Panel.TFrame")
        table.pack(fill="both", expand=True)
        self._tree = ttk.Treeview(table, columns=cols, show="headings")
        headings = {
            "status": "状态",
            "episode": "Episode ID",
            "subject": "Subject",
            "frames": "帧数",
            "lease": "租期剩余",
        }
        widths = {"status": 120, "episode": 110, "subject": 180, "frames": 100, "lease": 160}
        for col in cols:
            self._tree.heading(col, text=headings[col])
            self._tree.column(col, width=widths[col], anchor="center")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self._tree.yview)
        scroll.pack(side="right", fill="y")
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side="left", fill="both", expand=True)
        self._tree.bind("<Double-1>", self._open_selected)
        self._tree.bind("<Return>", self._open_selected)

    def set_loading(self, text: str) -> None:
        self._status.configure(text=text)

    def set_status(self, text: str) -> None:
        self._status.configure(text=text)

    def set_episodes(self, task_name: str, entries: List[Tuple[str, Any]], *, error: Optional[str] = None) -> None:
        self._task_name = task_name
        self._title.configure(text=f"Episode 选择 - {task_name}")
        self._entries = {}
        self._tree.delete(*self._tree.get_children())
        if error:
            self._status.configure(text=f"刷新失败：{error}")
            return
        self._status.configure(text="双击 Episode 开始质检；进行中的任务可继续上次进度。")
        for idx, (kind, entry) in enumerate(entries):
            iid = f"{kind}:{idx}"
            self._entries[iid] = (kind, entry)
            self._tree.insert("", "end", iid=iid, values=self._values(kind, entry))
        self._schedule_countdown()

    def _values(self, kind: str, entry: Any) -> Tuple[str, str, str, str, str]:
        if kind == "local":
            progress: QcProgress = entry
            return (
                "进行中" if progress.lease_valid else "已释放 · 有本地进度",
                episode_display_id(progress.episode, progress.payload),
                str(progress.episode.get("subject_id") or progress.payload.get("subject_id") or ""),
                str(len(progress.frames)),
                format_seconds(progress.lease_seconds_remaining) if progress.lease_valid else "已释放",
            )
        item = dict(entry or {})
        return (
            "可领取",
            episode_display_id(item),
            str(item.get("subject_id") or ""),
            str(item.get("frames_count") or len(item.get("frames") or [])),
            "",
        )

    def _schedule_countdown(self) -> None:
        if self._countdown_after is not None:
            try:
                self.after_cancel(self._countdown_after)
            except Exception:
                pass
        self._countdown_after = self.after(1000, self._tick_countdown)

    def _tick_countdown(self) -> None:
        for iid, (kind, entry) in self._entries.items():
            if kind == "local" and self._tree.exists(iid):
                self._tree.set(iid, "lease", format_seconds(entry.lease_seconds_remaining) if entry.lease_valid else "已释放")
        self._countdown_after = self.after(1000, self._tick_countdown)

    def _open_selected(self, _evt=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        entry = self._entries.get(sel[0])
        if entry is None:
            return
        self.app.open_episode_entry(entry[0], entry[1])


class DecodePage(ttk.Frame):
    def __init__(self, master, *, app: QcWorkerApp):
        super().__init__(master, style="TFrame")
        self.app = app
        self._rows: Dict[str, str] = {}
        self._build()

    def _build(self) -> None:
        outer = ttk.Frame(self, style="TFrame", padding=(24, 20))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="正在准备质检数据", style="Title.TLabel").pack(anchor="w")
        self._status = ttk.Label(outer, text="正在解码 RGB 视频...", style="Muted.TLabel")
        self._status.pack(anchor="w", pady=(8, 12))
        cols = ("camera", "status", "progress", "error")
        self._tree = ttk.Treeview(outer, columns=cols, show="headings", height=10)
        self._tree.heading("camera", text="Camera")
        self._tree.heading("status", text="状态")
        self._tree.heading("progress", text="帧数")
        self._tree.heading("error", text="错误")
        self._tree.column("camera", width=140, anchor="center")
        self._tree.column("status", width=140, anchor="center")
        self._tree.column("progress", width=160, anchor="center")
        self._tree.column("error", width=620, anchor="w")
        self._tree.pack(fill="x")

    def reset(self, progress: QcProgress) -> None:
        display_id = episode_display_id(progress.episode, progress.payload)
        self._status.configure(text=f"Episode ID：{display_id}    正在准备四路 RGB 与 Pico Ego MANO 投影视图")
        self._tree.delete(*self._tree.get_children())
        self._rows = {}
        available = [str(camera) for camera in progress.payload.get("cameras") or []]
        cameras = [camera for camera in ("00", "02", "03", "05") if camera in available]
        if not cameras:
            cameras = available[:4]
        cameras.append("ego")
        for camera in cameras:
            cam = str(camera)
            self._rows[cam] = cam
            self._tree.insert("", "end", iid=cam, values=(cam, "等待", "0 / 0", ""))

    def update_camera(self, status: Dict[str, Any]) -> None:
        cam = str(status.get("camera") or "")
        if not cam:
            return
        if not self._tree.exists(cam):
            self._tree.insert("", "end", iid=cam, values=(cam, "等待", "0 / 0", ""))
        decoded = int(status.get("decoded") or 0)
        total = int(status.get("total") or 0)
        state = str(status.get("status") or "")
        text_by_state = {
            "pending": "等待",
            "decoding": "解码中",
            "done": "完成",
            "failed": "失败",
            "mesh_pending": "等待渲染 mesh",
            "mesh_rendering": "渲染 mesh",
            "mesh_gpu": "核显渲染 mesh",
            "mesh_software_fallback": "软件渲染 mesh",
            "mesh_done": "mesh 完成",
        }
        self._tree.item(
            cam,
            values=(cam, text_by_state.get(state, state), f"{decoded} / {total}", str(status.get("error") or "")),
        )


class FrameTimeline(tk.Canvas):
    def __init__(self, master, *, on_seek: Callable[[int, bool], None]):
        super().__init__(master, height=42, bg=Theme.PANEL, highlightthickness=0, cursor="hand2")
        self._frames: List[int] = []
        self._current = 0
        self._bad_ranges: List[Tuple[int, int]] = []
        self._ego_bad_ranges: List[Tuple[int, int]] = []
        self._preview: Optional[Tuple[int, int]] = None
        self._seek_enabled = True
        self._on_seek = on_seek
        self._last_drag_position: Optional[int] = None
        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Button-1>", self._begin_drag)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._end_drag)

    def set_data(
        self,
        *,
        frames: List[int],
        current: int,
        bad_ranges: List[Tuple[int, int]],
        ego_bad_ranges: List[Tuple[int, int]],
        preview: Optional[Tuple[int, int]] = None,
        seek_enabled: bool = True,
    ) -> None:
        self._frames = list(frames)
        self._current = int(current)
        self._bad_ranges = list(bad_ranges)
        self._ego_bad_ranges = list(ego_bad_ranges)
        self._preview = preview
        self._seek_enabled = bool(seek_enabled)
        self.configure(cursor="hand2" if self._seek_enabled else "arrow")
        self._draw()

    def _x_for_position(self, position: int) -> float:
        width = max(1, int(self.winfo_width()))
        left, right = 8.0, max(8.0, float(width) - 8.0)
        if len(self._frames) <= 1:
            return left
        ratio = max(0.0, min(1.0, float(position) / float(len(self._frames) - 1)))
        return left + ratio * (right - left)

    def _position_for_frame(self, frame: int) -> int:
        if not self._frames:
            return 0
        return min(range(len(self._frames)), key=lambda idx: abs(self._frames[idx] - int(frame)))

    def _draw(self) -> None:
        self.delete("all")
        if not self._frames:
            return
        y1, y2 = 13, 23
        x1, x2 = self._x_for_position(0), self._x_for_position(len(self._frames) - 1)
        self.create_rectangle(x1, y1, x2, y2, fill=Theme.BORDER, outline="")
        current_pos = self._position_for_frame(self._current)
        current_x = self._x_for_position(current_pos)
        self.create_rectangle(x1, y1, current_x, y2, fill=Theme.ACCENT, outline="")
        for start, end in self._bad_ranges:
            start_pos = self._position_for_frame(start)
            end_pos = self._position_for_frame(end)
            range_x1 = self._x_for_position(min(start_pos, end_pos))
            range_x2 = max(range_x1 + 2.0, self._x_for_position(max(start_pos, end_pos)))
            self.create_rectangle(
                range_x1,
                y1,
                range_x2,
                y2,
                fill="#e5484d",
                outline="",
            )
        for start, end in self._ego_bad_ranges:
            start_pos = self._position_for_frame(start)
            end_pos = self._position_for_frame(end)
            range_x1 = self._x_for_position(min(start_pos, end_pos))
            range_x2 = max(range_x1 + 2.0, self._x_for_position(max(start_pos, end_pos)))
            self.create_line(
                range_x1,
                28,
                range_x2,
                28,
                fill="#ff202b",
                width=5,
                capstyle=tk.ROUND,
            )
        if self._preview is not None:
            start_pos = self._position_for_frame(self._preview[0])
            end_pos = self._position_for_frame(self._preview[1])
            range_x1 = self._x_for_position(min(start_pos, end_pos))
            range_x2 = max(range_x1 + 2.0, self._x_for_position(max(start_pos, end_pos)))
            self.create_rectangle(
                range_x1,
                y1,
                range_x2,
                y2,
                fill="#f59e0b",
                outline="",
            )
        self.create_line(current_x, 8, current_x, 34, fill="#ffffff", width=2)
        self.create_oval(
            current_x - 5,
            13,
            current_x + 5,
            23,
            fill="#ffffff",
            outline=Theme.ACCENT,
            width=2,
        )

    def _position_from_event(self, event: tk.Event) -> int:
        width = max(1, int(self.winfo_width()))
        ratio = max(0.0, min(1.0, (float(event.x) - 8.0) / max(1.0, float(width) - 16.0)))
        return int(round(ratio * max(0, len(self._frames) - 1)))

    def _begin_drag(self, event: tk.Event) -> None:
        if not self._seek_enabled or not self._frames:
            return
        position = self._position_from_event(event)
        self._last_drag_position = position
        self._on_seek(position, False)

    def _drag(self, event: tk.Event) -> None:
        if not self._seek_enabled or not self._frames or self._last_drag_position is None:
            return
        position = self._position_from_event(event)
        if position == self._last_drag_position:
            return
        self._last_drag_position = position
        self._on_seek(position, False)

    def _end_drag(self, event: tk.Event) -> None:
        if not self._seek_enabled or not self._frames or self._last_drag_position is None:
            return
        position = self._position_from_event(event)
        self._last_drag_position = None
        self._on_seek(position, True)


class QcPage(ttk.Frame):
    FRAME_STEP = 10

    def __init__(self, master, *, app: QcWorkerApp):
        super().__init__(master, style="TFrame")
        self.app = app
        self.progress: Optional[QcProgress] = None
        self.media: Optional[QcEpisodeMedia] = None
        self.mode = "playback"
        self.bad_anchor_frame = 0
        self.bad_cursor = 0
        self.bad_start: Optional[int] = None
        self.bad_end: Optional[int] = None
        self._canvases: Dict[str, ImageAnnotatorCanvas] = {}
        self._labels: Dict[str, ttk.Label] = {}
        self._rendering_overlays: Dict[str, tk.Label] = {}
        self._playing = False
        self._play_after_id: Optional[str] = None
        self._play_clock_started_at: Optional[float] = None
        self._play_clock_start_position = 0
        self._timeline_dragging = False
        self._ticks_since_save = 0
        self._busy_text = ""
        self._buffering = False
        self._preparation_after_id: Optional[str] = None
        self._frame_refresh_after_id: Optional[str] = None
        self._preparation_error_shown = False
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, style="TFrame")
        root.pack(fill="both", expand=True)
        self._top = ttk.Frame(root, style="Panel.TFrame", padding=(12, 8))
        self._top.pack(fill="x")
        self._info = ttk.Label(self._top, text="", style="Panel.TLabel")
        self._info.pack(fill="x")
        self._top.bind("<Configure>", lambda e: self._info.configure(wraplength=max(300, e.width - 24)))
        self._busy = ttk.Label(self._top, text="", style="PanelMuted.TLabel")
        self._busy.pack(anchor="w")
        self._grid = ttk.Frame(root, style="TFrame")

        footer = ttk.Frame(root, style="Panel.TFrame", padding=(14, 10))
        footer.pack(side="bottom", fill="x")
        timeline_host = ttk.Frame(footer, style="Panel.TFrame", padding=(10, 2))
        timeline_host.pack(fill="x")
        self._timeline = FrameTimeline(timeline_host, on_seek=self.seek_position)
        self._timeline.pack(fill="x")

        self._playback_bar = WrapToolbar(footer)
        self._playback_bar.pack(fill="x")
        self._prev_ten = ttk.Button(self._playback_bar, text="上十帧", style="Small.TButton", command=lambda: self.step_frames(-self.FRAME_STEP))
        self._playback_bar.add(self._prev_ten)
        self._prev_one = ttk.Button(self._playback_bar, text="上一帧", style="Small.TButton", command=lambda: self.step_frames(-1))
        self._playback_bar.add(self._prev_one)
        self._play_button = ttk.Button(self._playback_bar, text="播放", style="Primary.TButton", command=self.toggle_playback)
        self._playback_bar.add(self._play_button)
        self._next_one = ttk.Button(self._playback_bar, text="下一帧", style="Small.TButton", command=lambda: self.step_frames(1))
        self._playback_bar.add(self._next_one)
        self._next_ten = ttk.Button(self._playback_bar, text="下十帧", style="Small.TButton", command=lambda: self.step_frames(self.FRAME_STEP))
        self._playback_bar.add(self._next_ten)
        self._reject_button = ttk.Button(self._playback_bar, text="该帧不通过", style="Danger.TButton", command=self.enter_bad_range)
        self._playback_bar.add(self._reject_button)
        self._bad_episode_button = ttk.Button(self._playback_bar, text="Episode 异常", style="Danger.TButton", command=self.mark_bad_episode)
        self._playback_bar.add(self._bad_episode_button)
        self._playback_status = ttk.Label(self._playback_bar, text="", style="PanelMuted.TLabel")
        self._playback_bar.add(self._playback_status)
        self._submit_button = ttk.Button(self._playback_bar, text="提交", style="Primary.TButton", command=lambda: self.app.submit_current_progress())
        self._playback_bar.add(self._submit_button)
        self._playback_bar.add(ttk.Button(self._playback_bar, text="退出程序", style="Secondary.TButton", command=self.app.request_exit))
        self._playback_bar.add(ttk.Button(self._playback_bar, text="返回 Episode 列表", style="Secondary.TButton", command=lambda: self.app.handle_interrupt(exit_after=False)))

        self._bad_bar = WrapToolbar(footer)
        self._bad_bar.add(ttk.Button(self._bad_bar, text="上十帧", style="Small.TButton", command=lambda: self.move_bad_cursor(-self.FRAME_STEP)))
        self._bad_bar.add(ttk.Button(self._bad_bar, text="上一帧", style="Small.TButton", command=lambda: self.move_bad_cursor(-1)))
        self._bad_bar.add(ttk.Button(self._bad_bar, text="设为坏帧起点", style="Secondary.TButton", command=self.set_bad_start))
        self._bad_status = ttk.Label(self._bad_bar, text="", style="PanelMuted.TLabel")
        self._bad_bar.add(self._bad_status)
        self._bad_bar.add(ttk.Button(self._bad_bar, text="设为坏帧终点", style="Secondary.TButton", command=self.set_bad_end))
        self._bad_bar.add(ttk.Button(self._bad_bar, text="下一帧", style="Small.TButton", command=lambda: self.move_bad_cursor(1)))
        self._bad_bar.add(ttk.Button(self._bad_bar, text="下十帧", style="Small.TButton", command=lambda: self.move_bad_cursor(self.FRAME_STEP)))
        self._bad_bar.add(ttk.Button(
            self._bad_bar,
            text="确认为 EgoPose 外参不准",
            style="Primary.TButton",
            command=lambda: self.confirm_bad_range("egopose"),
        ))
        self._bad_bar.add(ttk.Button(
            self._bad_bar,
            text="确认为手部 Pose 不准",
            style="Primary.TButton",
            command=lambda: self.confirm_bad_range("hand_pose"),
        ))
        self._bad_bar.add(ttk.Button(self._bad_bar, text="撤销", style="Secondary.TButton", command=self.cancel_bad_range))

        self._grid.pack(fill="both", expand=True, padx=10, pady=(10, 4))

    def set_session(self, progress: QcProgress, media: QcEpisodeMedia) -> None:
        self.pause_playback(persist=False)
        self._cancel_preparation_monitor()
        self.progress = progress
        self.media = media
        self.mode = "playback"
        self._buffering = False
        self._preparation_error_shown = False
        if progress.frames and progress.current_frame not in progress.frames:
            progress.current_frame = progress.frames[0]
        self._build_camera_grid(media.display_cameras)
        self._refresh()
        self._monitor_preparation()

    def clear_session(self) -> None:
        self.pause_playback(persist=False)
        self._cancel_preparation_monitor()
        self.progress = None
        self.media = None
        self._buffering = False

    def _cancel_preparation_monitor(self) -> None:
        for after_id in (self._preparation_after_id, self._frame_refresh_after_id):
            if after_id is None:
                continue
            try:
                self.after_cancel(after_id)
            except Exception:
                pass
        self._preparation_after_id = None
        self._frame_refresh_after_id = None

    def _monitor_preparation(self) -> None:
        self._preparation_after_id = None
        media = self.media
        if media is None:
            return
        error = media.preparation_error
        if error and not self._preparation_error_shown:
            self._preparation_error_shown = True
            self.pause_playback()
            self.set_busy("后台准备失败")
            messagebox.showerror("数据准备失败", error, parent=self)
            return
        if not media.preparation_done:
            self._preparation_after_id = self.after(250, self._monitor_preparation)

    def set_busy(self, text: str) -> None:
        self._busy_text = str(text or "")
        if self._busy_text:
            self.pause_playback()
        self._busy.configure(text=self._busy_text)
        self._update_controls()

    def _build_camera_grid(self, cameras: List[str]) -> None:
        for child in self._grid.winfo_children():
            child.destroy()
        self._canvases = {}
        self._labels = {}
        self._rendering_overlays = {}
        for row in range(2):
            self._grid.rowconfigure(row, weight=1, uniform="qc_rows")
        for col in range(3):
            self._grid.columnconfigure(col, weight=1, uniform="qc_cols")
        for idx, cam in enumerate(cameras[:6]):
            host = ttk.Frame(self._grid, style="Panel.TFrame")
            host.grid(row=idx // 3, column=idx % 3, sticky="nsew", padx=5, pady=5)
            label_text = "Pico Ego · MANO 外参投影" if cam == "ego" else f"Camera {cam}"
            label = ttk.Label(host, text=label_text, style="PanelMuted.TLabel")
            label.pack(anchor="w", padx=8, pady=(6, 4))
            canvas = ImageAnnotatorCanvas(host, bg=Theme.PANEL_2)
            canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))
            canvas.set_read_only(True)
            canvas.set_annotation_visible(False)
            rendering_overlay = tk.Label(
                host,
                text="渲染中…",
                bg="#b45309",
                fg="#ffffff",
                padx=18,
                pady=10,
            )
            self._canvases[cam] = canvas
            self._labels[cam] = label
            self._rendering_overlays[cam] = rendering_overlay

    def _refresh(self) -> None:
        progress = self.progress
        media = self.media
        if progress is None or media is None or not progress.frames:
            return
        frame = self._display_frame()
        ranges = ", ".join(f"{a}-{b}" for a, b in progress.bad_frame_ranges) or "无"
        ego_ranges = ", ".join(f"{a}-{b}" for a, b in progress.ego_bad_frame_ranges) or "无"
        self._info.configure(
            text=(
                f"Task：{progress.task_name}    Episode ID：{episode_display_id(progress.episode, progress.payload)}    "
                f"当前帧：{frame} / {progress.last_frame}    手部 Pose：{ranges}    EgoPose：{ego_ranges}"
            )
        )
        frame_ready = media.frame_ready(frame)
        if frame_ready:
            for cam, canvas in self._canvases.items():
                path = media.frame_path(cam, frame)
                canvas.set_image(path)
                canvas.set_read_only(True)
                canvas.set_annotation_visible(False)
                canvas.set_focus_region(None)
                canvas.set_skeleton_overlay(None)
                canvas.set_mano_overlay(None)
                self._labels[cam].configure(
                    text="Pico Ego · MANO 外参投影" if cam == "ego" else f"Camera {cam}"
                )
                self._rendering_overlays[cam].place_forget()
            if not self._busy_text:
                self._busy.configure(text="")
        else:
            for cam in self._canvases:
                camera_label = "Pico Ego" if cam == "ego" else f"Camera {cam}"
                self._labels[cam].configure(text=f"{camera_label}    目标帧 {frame} 渲染中")
                overlay = self._rendering_overlays[cam]
                overlay.configure(text=f"目标帧 {frame}\n渲染中…")
                overlay.place(relx=0.5, rely=0.55, anchor="center")
                overlay.lift()
            if not self._busy_text:
                self._busy.configure(text=f"目标帧 {frame} 正在后台渲染…")
        self._buffering = not frame_ready
        if not frame_ready and not media.preparation_done and self._frame_refresh_after_id is None:
            self._frame_refresh_after_id = self.after(30, self._retry_frame_refresh)
        preview = None
        if self.mode == "bad_range" and self.bad_start is not None:
            preview = (self.bad_start, self.bad_end if self.bad_end is not None else self.bad_cursor)
        self._timeline.set_data(
            frames=progress.frames,
            current=frame,
            bad_ranges=progress.bad_frame_ranges,
            ego_bad_ranges=progress.ego_bad_frame_ranges,
            preview=preview,
            seek_enabled=self.mode == "playback" and not self._busy_text,
        )
        self._update_controls()

    def _retry_frame_refresh(self) -> None:
        self._frame_refresh_after_id = None
        media = self.media
        if media is None:
            return
        if media.preparation_error:
            self._monitor_preparation()
            return
        if media.frame_ready(self._display_frame()):
            was_buffering = self._buffering
            self._refresh()
            if was_buffering and self._playing and not self._timeline_dragging:
                if self._play_after_id is not None:
                    try:
                        self.after_cancel(self._play_after_id)
                    except Exception:
                        pass
                    self._play_after_id = None
                self._play_clock_started_at = time.monotonic()
                self._play_clock_start_position = self._current_position()
                self._schedule_play_tick()
            return
        if not media.preparation_done:
            self._frame_refresh_after_id = self.after(30, self._retry_frame_refresh)

    def _display_frame(self) -> int:
        if self.progress is None:
            return 0
        return int(self.bad_cursor if self.mode == "bad_range" else self.progress.current_frame)

    def _current_position(self) -> int:
        progress = self.progress
        if progress is None or not progress.frames:
            return 0
        frame = self._display_frame()
        try:
            return progress.frames.index(frame)
        except ValueError:
            return min(range(len(progress.frames)), key=lambda idx: abs(progress.frames[idx] - frame))

    @staticmethod
    def _set_enabled(widget: ttk.Button, enabled: bool) -> None:
        widget.state(["!disabled"] if enabled else ["disabled"])

    def _update_controls(self) -> None:
        self._playback_bar._schedule()
        self._bad_bar._schedule()
        progress = self.progress
        if self.mode == "bad_range":
            if self._playback_bar.winfo_manager():
                self._playback_bar.pack_forget()
            if not self._bad_bar.winfo_manager():
                self._bad_bar.pack(fill="x")
            self._bad_status.configure(
                text=f"当前帧：{self.bad_cursor}    Start={self.bad_start if self.bad_start is not None else '-'}    End={self.bad_end if self.bad_end is not None else '-'}"
            )
            return
        if self._bad_bar.winfo_manager():
            self._bad_bar.pack_forget()
        if not self._playback_bar.winfo_manager():
            self._playback_bar.pack(fill="x")
        disabled = progress is None or bool(self._busy_text)
        paused_controls = not disabled and not self._playing
        for button in (self._prev_ten, self._prev_one, self._next_one, self._next_ten):
            self._set_enabled(button, paused_controls)
        self._set_enabled(self._reject_button, paused_controls and not self._buffering)
        self._set_enabled(self._bad_episode_button, not disabled)
        self._set_enabled(self._play_button, not disabled and (self._playing or not self._buffering))
        self._set_enabled(self._submit_button, bool(progress and progress.is_complete and not disabled))
        self._play_button.configure(text="暂停" if self._playing else "播放")
        if progress is None:
            status = ""
        elif progress.playback_complete:
            status = "已完成一次播放，可随时提交"
        elif self._buffering:
            status = f"目标帧 {self._display_frame()} 渲染中 · 可继续拖动进度条"
        elif self._timeline_dragging:
            status = f"正在拖动定位 · 目标帧 {self._display_frame()}"
        elif self._playing:
            status = f"播放中 · 目标 {self.app.config.playback_fps:g} FPS"
        else:
            status = "已暂停 · 完成一次播放后可提交"
        self._playback_status.configure(text=status)

    def toggle_playback(self) -> None:
        if self._playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self) -> None:
        progress = self.progress
        if progress is None or not progress.frames or self.mode != "playback" or self._busy_text:
            return
        if progress.playback_complete and self._current_position() >= len(progress.frames) - 1:
            progress.current_frame = progress.frames[0]
        self._play_clock_start_position = self._current_position()
        self._play_clock_started_at = time.monotonic()
        self._playing = True
        self._update_controls()
        self._schedule_play_tick()

    def pause_playback(self, *, persist: bool = True) -> None:
        if self._play_after_id is not None:
            try:
                self.after_cancel(self._play_after_id)
            except Exception:
                pass
            self._play_after_id = None
        was_playing = self._playing
        self._playing = False
        self._play_clock_started_at = None
        self._timeline_dragging = False
        if self.media is not None and self.progress is not None:
            self._buffering = not self.media.frame_ready(self.progress.current_frame)
        if persist and was_playing:
            self.app.save_current_progress()
        self._update_controls()

    def _schedule_play_tick(self) -> None:
        now = time.monotonic()
        fps = self.app.config.playback_fps
        if self._play_clock_started_at is None:
            self._play_clock_started_at = now
            self._play_clock_start_position = self._current_position()
        elapsed = max(0.0, now - self._play_clock_started_at)
        current_offset = max(0, self._current_position() - self._play_clock_start_position)
        next_offset = max(current_offset + 1, int(elapsed * fps) + 1)
        deadline = self._play_clock_started_at + (float(next_offset) / fps)
        delay_ms = max(1, int(max(0.0, deadline - now) * 1000.0) + 1)
        self._play_after_id = self.after(delay_ms, self._play_tick)

    def _play_tick(self) -> None:
        self._play_after_id = None
        progress = self.progress
        if not self._playing or progress is None or not progress.frames:
            return
        position = self._current_position()
        now = time.monotonic()
        media = self.media
        current_frame = progress.frames[position]
        if media is not None and not media.frame_ready(current_frame):
            self._buffering = True
            self._play_clock_started_at = now
            self._play_clock_start_position = position
            self._update_controls()
            self._play_after_id = self.after(30, self._play_tick)
            return
        if self._buffering:
            self._buffering = False
            self._play_clock_started_at = now
            self._play_clock_start_position = position
            self._refresh()
            if position >= len(progress.frames) - 1:
                progress.playback_complete = True
                self.pause_playback(persist=False)
                self.app.save_current_progress()
                self._refresh()
                return
            self._schedule_play_tick()
            return
        if position >= len(progress.frames) - 1:
            progress.playback_complete = True
            self.pause_playback(persist=False)
            self.app.save_current_progress()
            self._refresh()
            return
        if self._play_clock_started_at is None:
            self._play_clock_started_at = now
            self._play_clock_start_position = position
        target_position = playback_target_position(
            start_position=self._play_clock_start_position,
            elapsed_seconds=now - self._play_clock_started_at,
            fps=self.app.config.playback_fps,
            last_position=len(progress.frames) - 1,
        )
        if target_position <= position:
            self._schedule_play_tick()
            return
        next_frame = progress.frames[target_position]
        if media is not None and not media.frame_ready(next_frame):
            progress.current_frame = next_frame
            self._buffering = True
            # Buffering pauses the playback clock instead of creating a large
            # catch-up jump when the next complete five-view frame arrives.
            self._play_clock_started_at = now
            self._play_clock_start_position = position
            self._refresh()
            self._play_after_id = self.after(30, self._play_tick)
            return
        self._buffering = False
        progress.current_frame = next_frame
        self._ticks_since_save += max(1, target_position - position)
        if self._ticks_since_save >= max(1, int(round(self.app.config.playback_fps))):
            self._ticks_since_save = 0
            self.app.save_current_progress()
        if target_position >= len(progress.frames) - 1:
            progress.playback_complete = True
            self.pause_playback(persist=False)
            self.app.save_current_progress()
            self._refresh()
            return
        self._refresh()
        self._schedule_play_tick()

    def step_frames(self, delta: int) -> None:
        progress = self.progress
        if progress is None or not progress.frames or self._playing or self.mode != "playback":
            return
        position = max(0, min(len(progress.frames) - 1, self._current_position() + int(delta)))
        progress.current_frame = progress.frames[position]
        self._persist_and_refresh()

    def seek_position(self, position: int, commit: bool = True) -> None:
        progress = self.progress
        if progress is None or not progress.frames or self.mode != "playback" or self._busy_text:
            return
        if not self._timeline_dragging:
            self._timeline_dragging = True
            if self._play_after_id is not None:
                try:
                    self.after_cancel(self._play_after_id)
                except Exception:
                    pass
                self._play_after_id = None
        index = max(0, min(len(progress.frames) - 1, int(position)))
        progress.current_frame = progress.frames[index]
        # Completion records a finished playback, not the current cursor position.
        if commit:
            self._timeline_dragging = False
            self.app.save_current_progress()
            self._refresh()
            if self._playing:
                self._play_clock_started_at = time.monotonic()
                self._play_clock_start_position = index
                media = self.media
                if media is not None and not media.frame_ready(progress.current_frame):
                    self._play_after_id = self.after(30, self._play_tick)
                else:
                    self._schedule_play_tick()
        else:
            self._refresh()

    def enter_bad_range(self) -> None:
        progress = self.progress
        if progress is None or self._playing:
            return
        self.mode = "bad_range"
        self.bad_anchor_frame = int(progress.current_frame)
        self.bad_cursor = int(progress.current_frame)
        self.bad_start = None
        self.bad_end = None
        self._refresh()

    def move_bad_cursor(self, delta: int) -> None:
        progress = self.progress
        if progress is None or not progress.frames:
            return
        anchor_position = progress.frames.index(self.bad_anchor_frame)
        lower_position = max(0, anchor_position - self.FRAME_STEP + 1)
        current_position = progress.frames.index(self.bad_cursor)
        target_position = max(lower_position, min(len(progress.frames) - 1, current_position + int(delta)))
        self.bad_cursor = progress.frames[target_position]
        self._refresh()

    def set_bad_start(self) -> None:
        self.bad_start = int(self.bad_cursor)
        self._refresh()

    def set_bad_end(self) -> None:
        self.bad_end = int(self.bad_cursor)
        self._refresh()

    def confirm_bad_range(self, kind: str) -> None:
        progress = self.progress
        if progress is None:
            return
        if self.bad_start is None or self.bad_end is None:
            messagebox.showwarning("坏帧区间", "请先设置坏帧起点和终点。")
            return
        if self.bad_start > self.bad_end:
            messagebox.showwarning("坏帧区间", "坏帧起点不能大于终点。")
            return
        target = (
            progress.ego_bad_frame_ranges
            if kind == "egopose"
            else progress.bad_frame_ranges
        )
        target.append((int(self.bad_start), int(self.bad_end)))
        normalized = normalize_ranges(
            target, max_gap_frames=self.app.config.range_merge_gap_frames
        )
        if kind == "egopose":
            progress.ego_bad_frame_ranges = normalized
        else:
            progress.bad_frame_ranges = normalized
        progress.current_frame = int(self.bad_end)
        self.mode = "playback"
        self.bad_start = None
        self.bad_end = None
        self._persist_and_refresh()

    def cancel_bad_range(self) -> None:
        progress = self.progress
        if progress is None:
            return
        progress.current_frame = self.bad_anchor_frame
        self.mode = "playback"
        self.bad_start = None
        self.bad_end = None
        self._persist_and_refresh()

    def mark_bad_episode(self) -> None:
        progress = self.progress
        if progress is None:
            return
        self.pause_playback()
        if not ConfirmDialog.ask(self, "Episode 异常", "确认将整个 Episode 标记为异常？该结果不会进入人工返修段。"):
            return
        progress.result_type = "bad_episode"
        self.app.save_current_progress()
        self.app.submit_current_progress(bad_episode=True)

    def _persist_and_refresh(self) -> None:
        self.app.save_current_progress()
        self._refresh()


class ConfirmDialog:
    @staticmethod
    def ask(parent: tk.Misc, title: str, message: str) -> bool:
        return bool(messagebox.askyesno(title, message, parent=parent))


def _center_modal(parent: tk.Misc, dialog: tk.Toplevel) -> None:
    try:
        dialog.transient(parent)
        dialog.grab_set()
    except Exception:
        pass
    dialog.update_idletasks()
    try:
        root = parent.winfo_toplevel()
        w = dialog.winfo_width()
        h = dialog.winfo_height()
        x = root.winfo_rootx() + max(0, (root.winfo_width() - w) // 2)
        y = root.winfo_rooty() + max(0, (root.winfo_height() - h) // 2)
        dialog.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass
