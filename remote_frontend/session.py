"""One Linux virtual application per browser worker; no video/model code in JS.

The listener deliberately binds only to loopback. Use an SSH tunnel for the
pilot. Internet deployment requires an authenticated gateway and OS isolation.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlencode


REPO = Path(__file__).resolve().parents[1]
PATH_KEYS = ("frame_cache_dir", "tmp_dir", "state_dir", "mano_toolkit_root", "mano_model_dir")


def prepare_config(source: Path, state: Path, cache: Path, operator: str, role: str) -> dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("launch config must be a JSON object")
    if not str(data.get("backend_url", "")).startswith(("http://", "https://")):
        raise ValueError("config requires backend_url")
    # Preserve the original config's relative-path semantics before relocating it.
    for key in PATH_KEYS:
        if data.get(key):
            data[key] = str((source.parent / Path(data[key]).expanduser()).resolve())
    for prefix, root in data.get("nas_mounts", {}).items():
        data["nas_mounts"][prefix] = str((source.parent / Path(root).expanduser()).resolve())
    data.update(operator_id=operator, frame_cache_dir=str(cache / "label"),
                tmp_dir=str(cache / "qc"), state_dir=str(state / "progress"))
    # A worker ID is stable across reconnects and restarts of this session.
    data["worker_machine_id"] = f"remote_{role}_{state.name}"
    return data


def browser_url(port: int, token: str) -> str:
    # noVNC accepts fragment configuration; the bearer token is not in HTTP GET
    # logs or Referer headers. It is sent only in the WebSocket handshake.
    return f"http://127.0.0.1:{port}/vnc.html#" + urlencode({
        "autoconnect": "true", "resize": "scale", "view_clip": "false",
        "quality": "9", "compression": "2", "shared": "false",
        "path": f"websockify?token={token}",
    })


def check_port(port: int) -> None:
    if not 1024 <= port <= 65535:
        raise ValueError("ports must be between 1024 and 65535")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


class Session:
    def __init__(self, args):
        self.args = args
        self.children = []
        self.logs = []
        self.app = None
        self.stopping = False
        self.runtime = None
        self.state = args.state_dir.expanduser().resolve() / args.name
        self.lock = None
        self.owns_lock = False

    def spawn(self, name, command, env):
        log = (self.state / f"{name}.log").open("a", encoding="utf-8")
        self.logs.append(log)
        child = subprocess.Popen(command, env=env, cwd=REPO,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append((name, child))
        return child

    def wait_ready(self, probe, child, name):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not self.stopping:
            if child.poll() is not None:
                raise RuntimeError(f"{name} exited; see {self.state / (name + '.log')}")
            if probe():
                return
            time.sleep(0.1)
        raise RuntimeError(f"{name} did not become ready")

    @staticmethod
    def listening(port):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            return False

    def run(self):
        args = self.args
        if sys.platform != "linux":
            raise RuntimeError("Start the worker on Linux; open its URL in the local browser")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", args.name):
            raise ValueError("name must contain 1–48 letters, digits, underscores or hyphens")
        if not re.fullmatch(r"\d{3,4}x\d{3,4}", args.geometry):
            raise ValueError("geometry must be WIDTHxHEIGHT")
        width, height = map(int, args.geometry.split("x"))
        if not (800 <= width <= 3840 and 600 <= height <= 2160):
            raise ValueError("geometry must be between 800x600 and 3840x2160")
        if args.idle_timeout < 10:
            raise ValueError("idle timeout must be at least 10 seconds")
        if not 1 <= args.display <= 999:
            raise ValueError("display must be between 1 and 999")
        if args.web_port == args.vnc_port:
            raise ValueError("web and VNC ports must differ")
        for port in (args.web_port, args.vnc_port):
            check_port(port)
        for binary in ("Xvfb", "xauth", "xdpyinfo", "x11vnc", "openbox"):
            if not shutil.which(binary):
                raise RuntimeError(f"missing {binary}; see remote_frontend/README.md")
        web = args.novnc_dir.expanduser().resolve()
        if not (web / "vnc.html").is_file():
            raise ValueError("novnc-dir must contain vnc.html")
        if Path(f"/tmp/.X11-unix/X{args.display}").exists() or Path(f"/tmp/.X{args.display}-lock").exists():
            raise RuntimeError("display is already in use; choose another --display")
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.state / "session.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("this named session is already running") from None
        self.owns_lock = True
        self.runtime = Path(tempfile.mkdtemp(prefix="orbbec-remote-"))
        cache = self.runtime / "cache"
        cache.mkdir()
        env = os.environ.copy()
        env.update(DISPLAY=f":{args.display}", XAUTHORITY=str(self.runtime / "Xauthority"),
                   ORBBEC_FRONTEND_RUNTIME_DIR=str(self.runtime / "instance"),
                   TMPDIR=str(cache), PYTHONUNBUFFERED="1")
        config = prepare_config(args.config.expanduser().resolve(), self.state, cache, args.operator, args.role)
        config_path = self.runtime / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        subprocess.run(["xauth", "-f", env["XAUTHORITY"], "add", env["DISPLAY"], ".",
                        secrets.token_hex(16)], check=True, capture_output=True)
        xserver = self.spawn("display", ["Xvfb", env["DISPLAY"], "-screen", "0",
            f"{args.geometry}x24", "-dpi", "96", "-nolisten", "tcp", "-auth", env["XAUTHORITY"]], env)
        def x_ready():
            return subprocess.run(["xdpyinfo"], env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=2).returncode == 0
        self.wait_ready(x_ready, xserver, "display")
        # Only application windows, without a desktop launcher or terminal menu.
        wm_config = self.runtime / "openbox.xml"
        wm_config.write_text('''<?xml version="1.0"?>
<openbox_config xmlns="http://openbox.org/3.4/rc">
<keyboard/><mouse/><menu/>
<applications><application type="normal"><decor>no</decor><maximized>yes</maximized></application></applications>
</openbox_config>''', encoding="utf-8")
        self.spawn("window-manager", ["openbox", "--config-file", str(wm_config)], env)
        vnc = self.spawn("vnc", ["x11vnc", "-display", env["DISPLAY"], "-auth", env["XAUTHORITY"],
            "-localhost", "-rfbport", str(args.vnc_port), "-forever", "-nevershared", "-dontdisconnect",
            "-nopw", "-noclipboard", "-nosetclipboard", "-noxdamage",
            "-noxrecord", "-repeat", "-wait", "10", "-defer", "10"], env)
        self.wait_ready(lambda: self.listening(args.vnc_port), vnc, "vnc")
        token = secrets.token_urlsafe(32)
        token_file = self.runtime / "tokens"
        token_file.write_text(f"{token}: 127.0.0.1:{args.vnc_port}\n", encoding="utf-8")
        proxy = self.spawn("gateway", [args.gateway_python, "-m", "websockify",
            "--web", str(web), "--token-plugin", "TokenFile", "--token-source", str(token_file),
            "--idle-timeout", str(args.idle_timeout), f"127.0.0.1:{args.web_port}"], env)
        self.wait_ready(lambda: self.listening(args.web_port), proxy, "gateway")
        module = "label.main" if args.role == "label" else "src.qc.main"
        self.app = self.spawn("application", [args.python, "-m", module, "--config", str(config_path)], env)
        metadata = {"pid": os.getpid(), "role": args.role, "name": args.name,
                    "url": browser_url(args.web_port, token), "display": args.display,
                    "runtime_dir": str(self.runtime), "state_dir": str(self.state),
                    "web_port": args.web_port, "vnc_port": args.vnc_port,
                    "idle_timeout": args.idle_timeout}
        (self.state / "connection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Browser worker started. Connection details: {self.state / 'connection.json'}", flush=True)
        result = 0
        while not self.stopping:
            for name, child in self.children:
                if child.poll() is not None:
                    print(f"{name} stopped (exit {child.returncode}); closing session", flush=True)
                    result = child.returncode or 0
                    self.stopping = True
                    break
            time.sleep(0.25)
        return result

    def close(self):
        # First ask the unmodified application to save/release and stop its own
        # decoder/renderer groups while X11 and the backend remain reachable.
        if self.app is not None and self.app.poll() is None:
            self.app.terminate()
            try:
                self.app.wait(timeout=45)
            except subprocess.TimeoutExpired:
                print("Application did not close within 45s; lease will expire if release failed", flush=True)
        for _, child in reversed(self.children):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _, child in reversed(self.children):
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in self.logs:
            log.close()
        if self.runtime is not None:
            shutil.rmtree(self.runtime, ignore_errors=True)
        if self.owns_lock:
            (self.state / "connection.json").unlink(missing_ok=True)
        if self.lock is not None:
            self.lock.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=("label", "qc"))
    p.add_argument("--name", required=True, help="Unique persistent session name")
    p.add_argument("--operator", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--display", type=int, required=True)
    p.add_argument("--web-port", type=int, required=True)
    p.add_argument("--vnc-port", type=int, required=True)
    p.add_argument("--novnc-dir", type=Path, default=Path("/usr/share/novnc"))
    p.add_argument("--python", default=sys.executable, help="Application Python (with GUI/model dependencies)")
    p.add_argument("--gateway-python", default=sys.executable, help="Python with websockify")
    p.add_argument("--state-dir", type=Path, default=Path("~/.local/state/orbbec/remote"))
    p.add_argument("--geometry", default="1600x1000")
    p.add_argument("--idle-timeout", type=int, default=120, help="Disconnected grace period in seconds")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    os.umask(0o077)
    session = Session(args)
    def stop(*_):
        session.stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return session.run()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Browser worker: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()
