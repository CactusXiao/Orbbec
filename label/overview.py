"""Six independent, read-only Label camera views."""
from tkinter import ttk

from .canvas_view import ImageAnnotatorCanvas
from .storage import find_frame_path
from .theme import Theme


class CameraOverview(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="Panel2.TFrame")
        self.canvases = {}
        self.notices = {}
        for row in range(2):
            self.rowconfigure(row, weight=1, uniform="overview_rows")
        for col in range(3):
            self.columnconfigure(col, weight=1, uniform="overview_columns")

    def set_cameras(self, cameras):
        cameras = list(cameras[:6])
        if list(self.canvases) == cameras:
            return
        for child in self.winfo_children():
            child.destroy()
        self.canvases = {}
        self.notices = {}
        for index, camera in enumerate(cameras):
            host = ttk.Frame(self, style="Panel.TFrame")
            host.grid(row=index // 3, column=index % 3, sticky="nsew", padx=3, pady=3)
            ttk.Label(host, text=f"{index + 1} · 机位 {camera} · 只读",
                      style="PanelMuted.TLabel").pack(anchor="w", padx=6, pady=4)
            canvas = ImageAnnotatorCanvas(host, bg=Theme.PANEL_2, show_schematics=False, width=1, height=1)
            canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
            canvas.set_read_only(True)
            self.canvases[camera] = canvas
            self.notices[camera] = ttk.Label(host, style="PanelMuted.TLabel", padding=5)

    def show_frame(self, task, frame, states):
        for camera, canvas in self.canvases.items():
            canvas.set_image(find_frame_path(task.episode_dir(), camera, frame, task.rgb_path_template))
            canvas.set_hand_state(*states[camera])
            canvas.set_read_only(True)

    def clear(self):
        for canvas in self.canvases.values():
            canvas.clear()
            canvas.set_read_only(True)
        for notice in self.notices.values():
            notice.place_forget()
