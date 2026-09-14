"""Five external cameras and a read-only ego RGB view."""
from tkinter import ttk

from .canvas_view import ImageAnnotatorCanvas
from .ego_preview import EgoPreview
from .storage import find_frame_path
from .theme import Theme


class CameraOverview(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="Panel2.TFrame")
        self.canvases = {}
        self.notices = {}
        self._ego_preview = None
        self._ego_task_key = None
        self._ego_poll = None
        self._frame = None
        self.bind("<Destroy>", self._on_destroy, add="+")
        for row in range(2):
            self.rowconfigure(row, weight=1, uniform="overview_rows")
        for col in range(3):
            self.columnconfigure(col, weight=1, uniform="overview_columns")

    def set_cameras(self, cameras):
        cameras = [camera for camera in ("00", "02", "03", "05", "06") if camera in cameras] + ["ego"]
        if list(self.canvases) == cameras:
            return
        for child in self.winfo_children():
            child.destroy()
        self.canvases = {}
        self.notices = {}
        for index, camera in enumerate(cameras):
            host = ttk.Frame(self, style="Panel.TFrame")
            host.grid(row=index // 3, column=index % 3, sticky="nsew", padx=3, pady=3)
            ttk.Label(host, text=f"机位 {camera} · 只读",
                      style="PanelMuted.TLabel").pack(anchor="w", padx=6, pady=4)
            canvas = ImageAnnotatorCanvas(host, bg=Theme.PANEL_2, show_schematics=False, width=1, height=1)
            canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
            canvas.set_read_only(True)
            self.canvases[camera] = canvas
            self.notices[camera] = ttk.Label(host, style="PanelMuted.TLabel", padding=5)

    def focus_camera(self, camera=None):
        for index, (cam, canvas) in enumerate(self.canvases.items()):
            host = canvas.master
            if camera is not None and cam != camera:
                host.grid_remove()
            else:
                host.grid(row=0 if camera else index // 3, column=0 if camera else index % 3,
                          rowspan=2 if camera else 1, columnspan=3 if camera else 1)

    def show_frame(self, task, frame, states):
        self._frame = frame
        key = (str(task.episode_dir()), tuple(task.frames))
        if self._ego_task_key != key:
            self._stop_ego()
            self._ego_task_key = key
            self._ego_preview = EgoPreview(task)
        for camera, canvas in self.canvases.items():
            if camera == "ego":
                canvas.set_hand_state([[(-1.0, -1.0)] * 21 for _ in range(2)], [[False] * 21 for _ in range(2)])
                canvas.set_read_only(True)
                continue
            canvas.set_image(find_frame_path(task.episode_dir(), camera, frame, task.rgb_path_template))
            canvas.set_hand_state(*states[camera])
            canvas.set_read_only(True)

        self._refresh_ego()

    def _refresh_ego(self):
        if self._ego_poll is not None:
            self.after_cancel(self._ego_poll)
            self._ego_poll = None
        preview = self._ego_preview
        if preview is None or "ego" not in self.canvases:
            return
        path = preview.path(self._frame)
        canvas = self.canvases["ego"]
        if canvas._img_path != path:
            canvas.set_image(path)
        notice = self.notices["ego"]
        if path is not None:
            notice.place_forget()
        else:
            text = "ego 画面加载中…" if not preview.done_event.is_set() else "此帧无同步 ego 画面"
            notice.configure(text=text)
            notice.place(relx=0.5, y=28, anchor="n")
        if not preview.done_event.is_set():
            self._ego_poll = self.after(100, self._refresh_ego)

    def _stop_ego(self):
        if self._ego_poll is not None:
            self.after_cancel(self._ego_poll)
            self._ego_poll = None
        if self._ego_preview is not None:
            self._ego_preview.close()
            self._ego_preview = None
        self._ego_task_key = None

    def _on_destroy(self, event):
        if event.widget is self:
            self._stop_ego()

    def clear(self):
        self._stop_ego()
        for canvas in self.canvases.values():
            canvas.clear()
            canvas.set_read_only(True)
        for notice in self.notices.values():
            notice.place_forget()
