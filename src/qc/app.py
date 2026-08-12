from __future__ import annotations

import queue
import signal
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from label.canvas_view import ImageAnnotatorCanvas
    from label.mano_view import ManoViewRuntime, describe_mano_projection_issue
    from label.theme import Theme, apply_theme
except Exception:
    from ...label.canvas_view import ImageAnnotatorCanvas  # type: ignore
    from ...label.mano_view import ManoViewRuntime, describe_mano_projection_issue  # type: ignore
    from ...label.theme import Theme, apply_theme  # type: ignore

from .backend import QcBackendClient, QcBackendError
from .config import QcConfig
from .media import QcEpisodeMedia, cleanup_qc_cache, prepare_qc_media
from .report import build_qc_result, qc_report_uri, write_qc_report
from .state_store import QcProgress, QcStateStore, first_sample_after, format_seconds, normalize_ranges


class QcWorkerApp(tk.Tk):
    def __init__(self, config: QcConfig):
        super().__init__()
        self.config = config
        self.client = QcBackendClient(config.backend_url, timeout_seconds=config.request_timeout_seconds)
        self.state_store = QcStateStore(config.state_dir)

        self.title("Orbbec 人工 QC Worker")
        self.geometry("1320x860")
        self.minsize(1100, 720)
        self.configure(bg=Theme.BG)
        apply_theme(self)

        self._current_progress: Optional[QcProgress] = None
        self._current_media: Optional[QcEpisodeMedia] = None
        self._heartbeat_after_id: Optional[str] = None
        self._last_task_name: str = ""

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
            self.preserve_for_crash_and_exit()

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
        local = [p for p in self.state_store.list_progress(worker_machine_id=self.config.worker_machine_id) if p.lease_valid]
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
            if progress.lease_valid and (progress.task_name or "Unspecified") == task_name
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
                leased.checked_sample_frames = list(progress.checked_sample_frames)
                leased.result_type = progress.result_type
                self.state_store.save(leased)
                return leased, None
            except Exception as exc:
                self.state_store.delete(progress)
                return None, f"无法恢复该 Episode，已删除本地进度：{exc}"

        self._run_bg(work, lambda result: self._after_progress_ready(result[0], result[1]))

    def _after_progress_ready(self, progress: Optional[QcProgress], error: Optional[str]) -> None:
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
        self.prepare_episode(progress)

    def prepare_episode(self, progress: QcProgress) -> None:
        self.decode_page.reset(progress)
        self.decode_page.lift()

        def progress_callback(_camera: str, status: Dict[str, Any]) -> None:
            self.after(0, lambda s=status: self.decode_page.update_camera(s))

        def work() -> Tuple[Optional[QcEpisodeMedia], Optional[str]]:
            try:
                media = prepare_qc_media(
                    progress.payload,
                    mounts=self.config.uri_mounts,
                    tmp_dir=self.config.tmp_dir,
                    on_progress=progress_callback,
                )
                return media, None
            except Exception as exc:
                return None, str(exc)

        self._run_bg(work, lambda result: self._after_media_ready(result[0], result[1]))

    def _after_media_ready(self, media: Optional[QcEpisodeMedia], error: Optional[str]) -> None:
        if error:
            messagebox.showerror("数据准备失败", error)
            if self._last_task_name:
                self.show_episodes(self._last_task_name)
            else:
                self.show_tasks()
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
            messagebox.showwarning("尚未完成", "当前 Episode 还没有检查到末尾，暂不能提交。")
            return
        bad_ranges = normalize_ranges(progress.bad_frame_ranges, max_gap_frames=self.config.range_merge_gap_frames)
        report_ranges = [] if bad_episode else bad_ranges
        progress.bad_frame_ranges = bad_ranges
        result = build_qc_result(
            episode_id=progress.episode_id,
            worker_id=self.config.worker_machine_id,
            bad_ranges=bad_ranges,
            bad_episode=bad_episode,
            sample_interval=progress.sample_interval,
        )
        self.qc_page.set_busy("正在提交 QC 结果...")

        def work() -> Tuple[bool, Optional[str]]:
            try:
                write_qc_report(
                    episode_dir=media.episode_dir,
                    result=result,
                    bad_ranges=report_ranges,
                    sample_interval=progress.sample_interval,
                )
                uri = qc_report_uri(progress.payload)
                artifacts = []
                if uri:
                    artifacts.append(
                        {
                            "kind": "qc_report",
                            "uri": uri,
                            "metadata": {
                                "passed": bool(result.get("passed")),
                                "worker_id": self.config.worker_machine_id,
                                "result_type": result.get("result_type"),
                            },
                        }
                    )
                self.client.complete_qc_job(progress.job_id, result=result, artifacts=artifacts)
                self.state_store.delete(progress)
                cleanup_qc_cache(media.cache_dir)
                return True, None
            except Exception as exc:
                return False, str(exc)

        def done(result_pair: Tuple[bool, Optional[str]]) -> None:
            ok, error = result_pair
            self.qc_page.set_busy("")
            if not ok:
                messagebox.showerror("提交失败", error or "后端没有确认提交成功。")
                return
            messagebox.showinfo("提交成功", "后端已确认 QC 结果。")
            self._current_progress = None
            self._current_media = None
            self._cancel_heartbeat()
            self.show_episodes(self._last_task_name or progress.task_name)

        self._run_bg(work, done)

    def handle_interrupt(self, *, exit_after: bool) -> None:
        progress = self._current_progress
        if progress is None:
            if exit_after:
                self.destroy()
            else:
                self.show_episodes(self._last_task_name)
            return
        choice = InterruptDialog.ask(self)
        if choice == "stay":
            return
        if choice == "abandon":
            self.abandon_current_progress(exit_after=exit_after)
            return
        if choice == "preserve":
            seconds = PreserveDialog.ask(self, default_minutes=self.config.default_lease_minutes)
            if seconds is None:
                return
            self.preserve_current_progress(lease_seconds=seconds, exit_after=exit_after)

    def abandon_current_progress(self, *, exit_after: bool) -> None:
        progress = self._current_progress
        media = self._current_media
        if progress is None:
            return

        def work() -> Tuple[bool, Optional[str]]:
            try:
                self.client.release_qc_job(progress.job_id, reason="operator_abandoned_qc")
                return True, None
            except Exception as exc:
                return False, str(exc)

        def done(result: Tuple[bool, Optional[str]]) -> None:
            ok, error = result
            if not ok:
                messagebox.showerror("放弃失败", error or "后端没有确认释放租约。")
                return
            self.state_store.delete(progress)
            if media is not None:
                cleanup_qc_cache(media.cache_dir)
            self._current_progress = None
            self._current_media = None
            self._cancel_heartbeat()
            if exit_after:
                self.destroy()
            else:
                self.show_episodes(self._last_task_name or progress.task_name)

        self._run_bg(work, done)

    def preserve_current_progress(self, *, lease_seconds: int, exit_after: bool) -> None:
        progress = self._current_progress
        if progress is None:
            return

        def work() -> Tuple[bool, Optional[str]]:
            try:
                response = self.client.heartbeat_qc_job(
                    progress.job_id,
                    worker_id=self.config.worker_machine_id,
                    lease_seconds=max(1, int(lease_seconds)),
                    status="running",
                )
                job = dict(response.get("job") or {})
                if job.get("lease_until"):
                    progress.lease_until = str(job.get("lease_until"))
                    progress.job = job
                self.state_store.save(progress)
                return True, None
            except Exception as exc:
                self.state_store.save(progress)
                return False, str(exc)

        def done(result: Tuple[bool, Optional[str]]) -> None:
            ok, error = result
            if not ok:
                messagebox.showerror("保留失败", error or "后端没有确认续租。")
                return
            self._cancel_heartbeat()
            self._current_progress = None
            self._current_media = None
            if exit_after:
                self.destroy()
            else:
                self.show_episodes(self._last_task_name or progress.task_name)

        self._run_bg(work, done)

    def preserve_for_crash_and_exit(self) -> None:
        progress = self._current_progress
        if progress is not None:
            try:
                self.client.heartbeat_qc_job(
                    progress.job_id,
                    worker_id=self.config.worker_machine_id,
                    lease_seconds=self.config.crash_lease_extension_seconds,
                    status="running",
                )
            except Exception:
                pass
            try:
                self.state_store.save(progress)
            except Exception:
                pass
        self.destroy()

    def request_exit(self) -> None:
        if self._current_progress is None:
            self.destroy()
        else:
            self.handle_interrupt(exit_after=True)

    def _schedule_heartbeat(self) -> None:
        self._cancel_heartbeat()
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

    def _run_bg(self, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        q: "queue.Queue[Any]" = queue.Queue(maxsize=1)

        def runner() -> None:
            q.put(work())

        threading.Thread(target=runner, daemon=True).start()

        def poll() -> None:
            try:
                result = q.get_nowait()
            except queue.Empty:
                self.after(60, poll)
                return
            done(result)

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
        outer = ttk.Frame(self, style="TFrame", padding=(18, 16))
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer, style="TFrame")
        top.pack(fill="x")
        ttk.Label(top, text="人工 QC - Task 选择", style="TLabel").pack(side="left")
        ttk.Button(top, text="刷新", style="Secondary.TButton", command=self.app.refresh_tasks).pack(side="right", padx=(8, 0))
        ttk.Button(top, text="退出", style="Secondary.TButton", command=self.app.request_exit).pack(side="right")
        self._status = ttk.Label(outer, text="", style="Muted.TLabel")
        self._status.pack(fill="x", pady=(8, 10))
        cols = ("task", "queued", "local", "expire")
        self._tree = ttk.Treeview(outer, columns=cols, show="headings")
        self._tree.heading("task", text="Task")
        self._tree.heading("queued", text="待质检 Episode")
        self._tree.heading("local", text="本机进行中")
        self._tree.heading("expire", text="最快过期")
        self._tree.column("task", width=420, anchor="w")
        self._tree.column("queued", width=140, anchor="center")
        self._tree.column("local", width=160, anchor="center")
        self._tree.column("expire", width=160, anchor="center")
        self._tree.pack(fill="both", expand=True)
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
        seconds = min(progress.lease_seconds_remaining for progress in local)
        return format_seconds(seconds)

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
        outer = ttk.Frame(self, style="TFrame", padding=(18, 16))
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer, style="TFrame")
        top.pack(fill="x")
        self._title = ttk.Label(top, text="Episode 选择", style="TLabel")
        self._title.pack(side="left")
        ttk.Button(top, text="返回 Task", style="Secondary.TButton", command=self.app.show_tasks).pack(side="right", padx=(8, 0))
        ttk.Button(top, text="刷新", style="Secondary.TButton", command=lambda: self.app.refresh_episodes(self._task_name)).pack(side="right")
        self._status = ttk.Label(outer, text="", style="Muted.TLabel")
        self._status.pack(fill="x", pady=(8, 10))
        cols = ("status", "episode", "subject", "index", "frames", "lease")
        self._tree = ttk.Treeview(outer, columns=cols, show="headings")
        headings = {
            "status": "状态",
            "episode": "Episode",
            "subject": "Subject",
            "index": "Index",
            "frames": "帧数",
            "lease": "租期剩余",
        }
        widths = {"status": 120, "episode": 360, "subject": 150, "index": 80, "frames": 100, "lease": 160}
        for col in cols:
            self._tree.heading(col, text=headings[col])
            self._tree.column(col, width=widths[col], anchor="center" if col != "episode" else "w")
        self._tree.pack(fill="both", expand=True)
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
        self._status.configure(text="双击 Episode 后才会正式租借。")
        for idx, (kind, entry) in enumerate(entries):
            iid = f"{kind}:{idx}"
            self._entries[iid] = (kind, entry)
            self._tree.insert("", "end", iid=iid, values=self._values(kind, entry))
        self._schedule_countdown()

    def _values(self, kind: str, entry: Any) -> Tuple[str, str, str, str, str, str]:
        if kind == "local":
            progress: QcProgress = entry
            return (
                "进行中",
                progress.episode_id,
                str(progress.episode.get("subject_id") or progress.payload.get("subject_id") or ""),
                str(progress.episode.get("episode_index") or ""),
                str(len(progress.frames)),
                format_seconds(progress.lease_seconds_remaining),
            )
        item = dict(entry or {})
        return (
            "可领取",
            str(item.get("episode_id") or ""),
            str(item.get("subject_id") or ""),
            str(item.get("episode_index") or ""),
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
                self._tree.set(iid, "lease", format_seconds(entry.lease_seconds_remaining))
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
        outer = ttk.Frame(self, style="TFrame", padding=(18, 16))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="正在准备质检数据", style="TLabel").pack(anchor="w")
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
        self._status.configure(text=f"Episode：{progress.episode_id}    采样间隔：{progress.sample_interval}")
        self._tree.delete(*self._tree.get_children())
        self._rows = {}
        for camera in progress.payload.get("cameras") or []:
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
        }
        self._tree.item(
            cam,
            values=(cam, text_by_state.get(state, state), f"{decoded} / {total}", str(status.get("error") or "")),
        )


class QcPage(ttk.Frame):
    def __init__(self, master, *, app: QcWorkerApp):
        super().__init__(master, style="TFrame")
        self.app = app
        self.progress: Optional[QcProgress] = None
        self.media: Optional[QcEpisodeMedia] = None
        self.mano_runtime = ManoViewRuntime()
        self.mode = "sampling"
        self.bad_anchor_frame = 0
        self.bad_cursor = 0
        self.bad_start: Optional[int] = None
        self.bad_end: Optional[int] = None
        self._canvases: Dict[str, ImageAnnotatorCanvas] = {}
        self._labels: Dict[str, ttk.Label] = {}
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, style="TFrame")
        root.pack(fill="both", expand=True)
        self._top = ttk.Frame(root, style="Panel.TFrame", padding=(12, 8))
        self._top.pack(fill="x")
        self._info = ttk.Label(self._top, text="", style="TLabel")
        self._info.pack(side="left")
        self._busy = ttk.Label(self._top, text="", style="Muted.TLabel")
        self._busy.pack(side="right")
        self._grid = ttk.Frame(root, style="TFrame")
        self._grid.pack(fill="both", expand=True, padx=10, pady=10)
        self._bar = ttk.Frame(root, style="Panel.TFrame", padding=(10, 8))
        self._bar.pack(fill="x")

    def set_session(self, progress: QcProgress, media: QcEpisodeMedia) -> None:
        self.progress = progress
        self.media = media
        self.mode = "sampling"
        self._build_camera_grid(media.task.cameras)
        self._refresh()

    def set_busy(self, text: str) -> None:
        self._busy.configure(text=text)

    def _build_camera_grid(self, cameras: List[str]) -> None:
        for child in self._grid.winfo_children():
            child.destroy()
        self._canvases = {}
        self._labels = {}
        for row in range(2):
            self._grid.rowconfigure(row, weight=1, uniform="qc_rows")
        for col in range(3):
            self._grid.columnconfigure(col, weight=1, uniform="qc_cols")
        for idx, cam in enumerate(cameras[:6]):
            host = ttk.Frame(self._grid, style="Panel.TFrame")
            host.grid(row=idx // 3, column=idx % 3, sticky="nsew", padx=5, pady=5)
            label = ttk.Label(host, text=f"Camera {cam}", style="Muted.TLabel")
            label.pack(anchor="w", padx=8, pady=(6, 4))
            canvas = ImageAnnotatorCanvas(host, bg=Theme.PANEL_2)
            canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))
            canvas.set_read_only(True)
            canvas.set_annotation_visible(False)
            self._canvases[cam] = canvas
            self._labels[cam] = label

    def _refresh(self) -> None:
        progress = self.progress
        media = self.media
        if progress is None or media is None:
            return
        frame = self._display_frame()
        total_text = f"{progress.current_frame} / {progress.last_frame}" if progress.current_frame <= progress.last_frame else f"完成 / {progress.last_frame}"
        ranges = ", ".join(f"{a}-{b}" for a, b in progress.bad_frame_ranges) or "无"
        self._info.configure(
            text=f"Task：{progress.task_name}    Episode：{progress.episode_id}    当前帧：{total_text}    采样间隔：{progress.sample_interval}    坏帧区间：{ranges}"
        )
        for cam, canvas in self._canvases.items():
            path = media.frame_path(cam, frame)
            canvas.set_image(path)
            canvas.set_read_only(True)
            canvas.set_annotation_visible(False)
            issue = ""
            try:
                projected = self.mano_runtime.project_mano_frame(
                    episode_dir=media.episode_dir,
                    mano_dir=media.mano_dir,
                    cam_id=cam,
                    frame_idx=frame,
                )
            except Exception as exc:
                projected = None
                issue = str(exc)
            if projected is None:
                canvas.set_skeleton_overlay(None)
                if not issue:
                    issue = describe_mano_projection_issue(media.episode_dir, media.mano_dir, cam, frame)
                self._labels[cam].configure(text=f"Camera {cam}    无 MANO 投影：{issue}")
            else:
                points, visible = projected
                canvas.set_skeleton_overlay(points, visible)
                self._labels[cam].configure(text=f"Camera {cam}")
        self._build_bar()

    def _display_frame(self) -> int:
        progress = self.progress
        if progress is None:
            return 0
        if self.mode == "bad_range":
            return int(self.bad_cursor)
        return min(int(progress.current_frame), int(progress.last_frame))

    def _build_bar(self) -> None:
        for child in self._bar.winfo_children():
            child.destroy()
        if self.mode == "bad_range":
            self._build_bad_bar()
        else:
            self._build_sampling_bar()

    def _build_sampling_bar(self) -> None:
        progress = self.progress
        if progress is None:
            return
        ttk.Button(self._bar, text="上一采样帧", style="Small.TButton", command=self.prev_sample).pack(side="left", padx=(0, 8))
        ttk.Label(self._bar, text=f"当前帧：{progress.current_frame if progress.current_frame <= progress.last_frame else '完成'} / {progress.last_frame}", style="Muted.TLabel").pack(side="left", padx=(0, 12))
        ttk.Label(self._bar, text=f"采样间隔：{progress.sample_interval}", style="Muted.TLabel").pack(side="left", padx=(0, 16))
        ttk.Button(self._bar, text="通过", style="Primary.TButton", command=self.mark_pass).pack(side="left", padx=(0, 8))
        ttk.Button(self._bar, text="不通过", style="Secondary.TButton", command=self.enter_bad_range).pack(side="left", padx=(0, 8))
        ttk.Button(self._bar, text="Episode 异常", style="Secondary.TButton", command=self.mark_bad_episode).pack(side="left", padx=(0, 16))
        ttk.Button(self._bar, text="下一采样帧", style="Small.TButton", command=self.next_sample).pack(side="left", padx=(0, 16))
        ttk.Button(self._bar, text="提交", style="Primary.TButton", command=lambda: self.app.submit_current_progress()).pack(side="right")
        ttk.Button(self._bar, text="退出程序", style="Secondary.TButton", command=self.app.request_exit).pack(side="right", padx=(0, 8))
        ttk.Button(self._bar, text="返回 Episode 列表", style="Secondary.TButton", command=lambda: self.app.handle_interrupt(exit_after=False)).pack(side="right", padx=(0, 8))

    def _build_bad_bar(self) -> None:
        progress = self.progress
        if progress is None:
            return
        ttk.Button(self._bar, text="向前一帧", style="Small.TButton", command=lambda: self.move_bad_cursor(-1)).pack(side="left", padx=(0, 6))
        ttk.Button(self._bar, text=f"向前 {progress.sample_interval} 帧", style="Small.TButton", command=lambda: self.move_bad_cursor(-progress.sample_interval)).pack(side="left", padx=(0, 6))
        ttk.Button(self._bar, text="设为坏帧起点", style="Secondary.TButton", command=self.set_bad_start).pack(side="left", padx=(0, 10))
        ttk.Label(self._bar, text=f"当前帧：{self.bad_cursor}    Start={self.bad_start if self.bad_start is not None else '-'}    End={self.bad_end if self.bad_end is not None else '-'}", style="Muted.TLabel").pack(side="left", padx=(0, 10))
        ttk.Button(self._bar, text="设为坏帧终点", style="Secondary.TButton", command=self.set_bad_end).pack(side="left", padx=(0, 6))
        ttk.Button(self._bar, text="向后一帧", style="Small.TButton", command=lambda: self.move_bad_cursor(1)).pack(side="left", padx=(0, 6))
        ttk.Button(self._bar, text=f"向后 {progress.sample_interval} 帧", style="Small.TButton", command=lambda: self.move_bad_cursor(progress.sample_interval)).pack(side="left", padx=(0, 10))
        ttk.Button(self._bar, text="确认坏帧区间", style="Primary.TButton", command=self.confirm_bad_range).pack(side="right")
        ttk.Button(self._bar, text="撤销", style="Secondary.TButton", command=self.cancel_bad_range).pack(side="right", padx=(0, 8))

    def prev_sample(self) -> None:
        progress = self.progress
        if progress is None:
            return
        progress.current_frame = max(progress.first_frame, progress.current_frame - progress.sample_interval)
        self._persist_and_refresh()

    def next_sample(self) -> None:
        progress = self.progress
        if progress is None:
            return
        progress.current_frame = self._first_sample_after(progress.current_frame)
        self._persist_and_refresh()

    def mark_pass(self) -> None:
        progress = self.progress
        if progress is None or progress.current_frame > progress.last_frame:
            return
        if progress.current_frame not in progress.checked_sample_frames:
            progress.checked_sample_frames.append(progress.current_frame)
            progress.checked_sample_frames = sorted(set(progress.checked_sample_frames))
        progress.current_frame = self._first_sample_after(progress.current_frame)
        self._persist_and_refresh()

    def enter_bad_range(self) -> None:
        progress = self.progress
        if progress is None or progress.current_frame > progress.last_frame:
            return
        self.mode = "bad_range"
        self.bad_anchor_frame = int(progress.current_frame)
        self.bad_cursor = int(progress.current_frame)
        self.bad_start = None
        self.bad_end = None
        self._refresh()

    def move_bad_cursor(self, delta: int) -> None:
        progress = self.progress
        if progress is None:
            return
        lower = max(progress.first_frame, self.bad_anchor_frame - progress.sample_interval + 1)
        upper = progress.last_frame
        self.bad_cursor = max(lower, min(upper, self.bad_cursor + int(delta)))
        self._refresh()

    def set_bad_start(self) -> None:
        self.bad_start = int(self.bad_cursor)
        self._build_bar()

    def set_bad_end(self) -> None:
        self.bad_end = int(self.bad_cursor)
        self._build_bar()

    def confirm_bad_range(self) -> None:
        progress = self.progress
        if progress is None:
            return
        if self.bad_start is None or self.bad_end is None:
            messagebox.showwarning("坏帧区间", "请先设置坏帧起点和终点。")
            return
        if self.bad_start > self.bad_end:
            messagebox.showwarning("坏帧区间", "坏帧起点不能大于终点。")
            return
        progress.bad_frame_ranges.append((int(self.bad_start), int(self.bad_end)))
        progress.bad_frame_ranges = normalize_ranges(
            progress.bad_frame_ranges,
            max_gap_frames=self.app.config.range_merge_gap_frames,
        )
        progress.current_frame = self._first_sample_after(self.bad_end)
        self.mode = "sampling"
        self._persist_and_refresh()

    def cancel_bad_range(self) -> None:
        progress = self.progress
        if progress is None:
            return
        progress.current_frame = self.bad_anchor_frame
        self.mode = "sampling"
        self.bad_start = None
        self.bad_end = None
        self._persist_and_refresh()

    def mark_bad_episode(self) -> None:
        progress = self.progress
        if progress is None:
            return
        if not ConfirmDialog.ask(self, "Episode 异常", "确认将整个 Episode 标记为异常？该结果不会进入人工返修段。"):
            return
        progress.result_type = "bad_episode"
        self.app.save_current_progress()
        self.app.submit_current_progress(bad_episode=True)

    def _first_sample_after(self, frame: int) -> int:
        progress = self.progress
        if progress is None:
            return frame
        return first_sample_after(
            int(frame),
            first_frame=progress.first_frame,
            last_frame=progress.last_frame,
            sample_interval=progress.sample_interval,
        )

    def _persist_and_refresh(self) -> None:
        self.app.save_current_progress()
        self._refresh()


class ConfirmDialog:
    @staticmethod
    def ask(parent: tk.Misc, title: str, message: str) -> bool:
        return bool(messagebox.askyesno(title, message, parent=parent))


class InterruptDialog:
    @staticmethod
    def ask(parent: tk.Misc) -> str:
        dialog = tk.Toplevel(parent)
        dialog.title("当前 Episode 尚未完成")
        dialog.configure(bg=Theme.BG)
        dialog.resizable(False, False)
        result = {"choice": "stay"}
        outer = ttk.Frame(dialog, style="Panel.TFrame", padding=(18, 14))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="当前 Episode 尚未完成。\n是否保留当前质检进度？", style="TLabel", justify="left").pack(anchor="w")
        btns = ttk.Frame(outer, style="Panel.TFrame")
        btns.pack(fill="x", pady=(14, 0))

        def choose(value: str) -> None:
            result["choice"] = value
            try:
                dialog.grab_release()
            except Exception:
                pass
            dialog.destroy()

        ttk.Button(btns, text="放弃任务", style="Secondary.TButton", command=lambda: choose("abandon")).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="保留进度", style="Primary.TButton", command=lambda: choose("preserve")).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="回到任务", style="Secondary.TButton", command=lambda: choose("stay")).pack(side="left")
        _center_modal(parent, dialog)
        dialog.wait_window(dialog)
        return str(result["choice"])


class PreserveDialog:
    @staticmethod
    def ask(parent: tk.Misc, *, default_minutes: int) -> Optional[int]:
        dialog = tk.Toplevel(parent)
        dialog.title("保留当前任务")
        dialog.configure(bg=Theme.BG)
        dialog.resizable(False, False)
        result: Dict[str, Optional[int]] = {"seconds": None}
        hours = tk.StringVar(value="0")
        minutes = tk.StringVar(value=str(int(default_minutes)))
        outer = ttk.Frame(dialog, style="Panel.TFrame", padding=(18, 14))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="保留当前任务", style="TLabel").pack(anchor="w")
        form = ttk.Frame(outer, style="Panel.TFrame")
        form.pack(fill="x", pady=(12, 0))
        ttk.Label(form, text="小时", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(form, textvariable=hours, width=8).grid(row=0, column=1, padx=(0, 16))
        ttk.Label(form, text="分钟", style="Muted.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Entry(form, textvariable=minutes, width=8).grid(row=0, column=3)
        btns = ttk.Frame(outer, style="Panel.TFrame")
        btns.pack(fill="x", pady=(14, 0))

        def confirm() -> None:
            try:
                h = max(0, int(hours.get() or "0"))
                m = max(0, int(minutes.get() or "0"))
            except ValueError:
                messagebox.showwarning("保留当前任务", "请输入有效的小时和分钟。", parent=dialog)
                return
            seconds = h * 3600 + m * 60
            if seconds <= 0:
                messagebox.showwarning("保留当前任务", "保留时间必须大于 0。", parent=dialog)
                return
            result["seconds"] = seconds
            try:
                dialog.grab_release()
            except Exception:
                pass
            dialog.destroy()

        def cancel() -> None:
            try:
                dialog.grab_release()
            except Exception:
                pass
            dialog.destroy()

        ttk.Button(btns, text="取消", style="Secondary.TButton", command=cancel).pack(side="right", padx=(8, 0))
        ttk.Button(btns, text="确认", style="Primary.TButton", command=confirm).pack(side="right")
        _center_modal(parent, dialog)
        dialog.wait_window(dialog)
        return result["seconds"]


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
