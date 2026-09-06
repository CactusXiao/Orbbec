"""Per-user frontend instance locks, shared across launchers and checkouts."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import socket


class SingleInstance:
    def __init__(self, name: str, *, runtime_dir=None):
        if name not in {"label", "qc"}:
            raise ValueError("unknown frontend")
        self.directory = Path(runtime_dir or Path("/tmp") / f"orbbec-frontends-{os.getuid()}")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.directory / f"{name}.sock"
        self.lock = (self.directory / f"{name}.lock").open("a+")
        self.socket = None
        self.acquired = False
        self._app = None
        self._poll_id = None

    def __enter__(self):
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # The first process may still be importing Tk; retry only activation.
            import time
            for _ in range(20):
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                        client.settimeout(0.1)
                        client.connect(str(self.path))
                        client.sendall(b"activate")
                    break
                except OSError:
                    time.sleep(0.05)
            return self
        self.acquired = True
        try:
            self.path.unlink(missing_ok=True)
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.bind(str(self.path))
            self.socket.listen(8)
            self.socket.setblocking(False)
        except Exception:
            self.close()
            raise
        return self

    def attach(self, app):
        self._app = app

        def poll():
            activated = False
            while True:
                try:
                    client, _ = self.socket.accept()
                except BlockingIOError:
                    break
                client.close()
                activated = True
            if activated:
                app.deiconify()
                app.lift()
                app.focus_force()
            self._poll_id = app.after(200, poll)

        self._poll_id = app.after(200, poll)

    def close(self):
        if self._app is not None and self._poll_id is not None:
            try:
                self._app.after_cancel(self._poll_id)
            except Exception:
                pass
            self._poll_id = None
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.acquired:
            self.path.unlink(missing_ok=True)
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.acquired = False
        self.lock.close()
        # Never unlink the lock file: waiters must all lock the same inode.

    def __exit__(self, *_):
        self.close()


def cancel_tk_callbacks(app):
    """Cancel window timers before Tk deletes their registered commands."""
    for timer in app.tk.splitlist(app.tk.call("after", "info")):
        # Let each widget delete its own registered command during destroy.
        # Calling root.after_cancel here would delete a child's command twice.
        app.tk.call("after", "cancel", timer)
