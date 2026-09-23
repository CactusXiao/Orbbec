"""Standalone browser branch. Run explicitly; no changes to desktop startup.

Account-isolated operators; loopback for SSH testing or a trusted HTTPS proxy.
The account database is separate from the existing Label/QC workflow database.
"""
from __future__ import annotations
import argparse
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import urlsplit

from task_backend.workflow_models import WorkflowError
from .batch import BatchService, reject
from .browser_media import BrowserMedia
from .browser_compute import BrowserCompute
from .browser_login import Accounts
from .preview_server import byte_range
from .qc_dispatch import QCDispatch, RoutedMedia


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, *args):
        pass  # Never log access keys or result bodies.

    def json(self, value, status=200, cookie=None):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def token(self):
        try:
            token = SimpleCookie(self.headers.get("Cookie", "")).get(self.server.cookie_name)
            return token.value if token else ""
        except Exception:
            return ""

    def cookie(self, token):
        secure = "; Secure" if self.server.secure_cookie else ""
        return f"{self.server.cookie_name}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={self.server.auth.lifetime if token else 0}{secure}"

    def identity(self, user):
        return {**user, "workspace_label": getattr(self.server, "workspace_label", "")}

    def check_host(self):
        if self.headers.get("Host") not in self.server.allowed_hosts:
            reject("访问地址未获允许", 403)

    def account(self, *, unrestricted=False, touch=False):
        user = self.server.auth.authenticate(self.token(), touch=touch)
        if user["must_change"] and not unrestricted:
            reject("请先更换初始密码", 428)
        self.user = user
        return user

    def workbench(self):
        unit = self.server.workbench(self.user)
        self.batch, self.media, self.compute = unit.batch, unit.media, unit.compute
        return self.batch

    def file(self, path, content_type, *, cache=False):
        if not path.is_file():
            reject("文件尚未准备好", 404)
        size = path.stat().st_size
        try:
            start, end = byte_range(self.headers.get("Range", ""), size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(206 if self.headers.get("Range") else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origins = " ".join(getattr(self.server, "qc_origins", []))
        self.send_header("Content-Security-Policy", f"default-src 'self'; img-src 'self' blob: {origins}; media-src 'self' blob: {origins}; connect-src 'self' {origins}; style-src 'self'; script-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end-start+1))
        if self.headers.get("Range"):
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end-start+1
            while remaining:
                data = stream.read(min(65536, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)

    def do_GET(self):
        try:
            self.check_host()
            parts = urlsplit(self.path).path.strip("/").split("/")
            if parts[:3] == ["internal", "qc", "input"] and len(parts) == 5:
                dispatch = self.server.qc_dispatch
                if dispatch is None:
                    reject("QC dispatch is disabled", 404)
                dispatch.authenticate(self.headers.get("Authorization"))
                source = dispatch.input_source(self.headers.get("X-QC-Worker"), parts[3], parts[4])
                from .qc_worker import export_inputs
                self.send_response(200)
                self.send_header("Content-Type", "application/x-tar")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return export_inputs(source, self.wfile)
            if parts[0] in {"", "app.js", "auth.js", "workflow.js", "label-canvas.js", "desktop-layout.js", "queue.js", "player.js", "frame-cache.js", "media-routing.js", "style.css", "sw.js"} and len(parts) == 1:
                name = parts[0] or "index.html"
                return self.file(Path(__file__).with_name("web")/name,
                    {"index.html":"text/html; charset=utf-8", "app.js":"text/javascript", "auth.js":"text/javascript", "workflow.js":"text/javascript", "label-canvas.js":"text/javascript", "desktop-layout.js":"text/javascript", "queue.js":"text/javascript", "player.js":"text/javascript", "frame-cache.js":"text/javascript", "media-routing.js":"text/javascript", "sw.js":"text/javascript", "style.css":"text/css"}[name])
            user = self.account(unrestricted=parts == ["api", "identity"])
            if parts == ["api", "identity"]:
                return self.json(self.identity(user))
            if parts == ["api", "accounts"]:
                if not user["admin"]:
                    reject("仅管理员可以管理账号", 403)
                return self.json(self.server.auth.all())
            if parts == ["api", "account-tasks"]:
                if not user["admin"]:
                    reject("仅管理员可以管理账号", 403)
                batch = self.workbench()
                with batch.store.connect() as conn:
                    return self.json([r[0] for r in conn.execute("SELECT DISTINCT task_name FROM episodes ORDER BY task_name")])
            batch = self.workbench()
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                return self.json(batch.available(parts[2]))
            if len(parts) >= 3 and parts[:2] == ["api", "sessions"]:
                sid = parts[2]
                if len(parts) == 3:
                    manifest = batch.manifest(sid)
                    manifest["media"] = self.media.status(sid)
                    return self.json(manifest)
                root = self.media.directory(sid)
                if len(parts) == 4 and parts[3] in {"samples.json", "sources.json", "preview.mp4"}:
                    return self.file(root/parts[3], "application/json" if parts[3].endswith("json") else "video/mp4", cache=True)
                if len(parts) == 5 and parts[3] == "chunks" and parts[4].endswith(".mp4"):
                    index = int(parts[4][:-4])
                    if index < 0:
                        reject("视频片段不存在", 404)
                    return self.file(root/"chunks"/f"{index}.mp4", "video/mp4", cache=True)
                if len(parts) >= 5 and parts[3] == "operations":
                    operation = parts[4]
                    if len(parts) == 5:
                        return self.json(self.compute.status(sid, operation))
                    if len(parts) == 6 and parts[5].endswith(".png"):
                        camera = parts[5][:-4]
                        if camera not in batch.manifest(sid)["cameras"]:
                            reject("计算画面不属于该任务", 404)
                        return self.file(self.compute.directory(sid, operation)/f"{camera}.png", "image/png", cache=True)
                if len(parts) == 6 and parts[3] in {"frames", "raw_frames"}:
                    manifest = batch.manifest(sid)
                    frame, camera = int(parts[5]), parts[4]
                    ready = self.media.status(sid)
                    if frame not in manifest["frames"] or camera not in (ready.get("cameras", []) + (["ego"] if manifest["role"] == "label" else [])):
                        reject("画面不属于该任务", 404)
                    return self.file(root/parts[3]/camera/f"{frame}.jpg", "image/jpeg", cache=True)
            reject("接口不存在", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except WorkflowError as exc:
            self.json({"error": str(exc)}, int(exc.status))
        except (ValueError, TypeError, KeyError) as exc:
            self.json({"error": str(exc)}, 400)

    def do_POST(self):
        try:
            self.check_host()
            # JSON plus a custom header prevents form/login CSRF even without Origin.
            # Never trust forwarded headers from arbitrary callers.
            origin = self.headers.get("Origin")
            if origin and origin not in self.server.allowed_origins:
                reject("跨站请求被拒绝", 403)
            if self.headers.get("X-Orbbec-Request") != "1" or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                reject("请求来源或格式无效", 403)
            length = int(self.headers.get("Content-Length", "0"))
            maximum = 16384 if urlsplit(self.path).path in {"/api/login", "/api/password"} else 32*1024*1024
            if not 0 < length <= maximum:
                reject("提交大小无效（最大 32 MB）", 413)
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                reject("请求必须为对象", 400)
            path = urlsplit(self.path).path
            if path == "/api/login":
                token, user = self.server.auth.login(body.get("username"), body.get("password"), self.client_address[0])
                self.server.auth.logout(self.token())
                return self.json(self.identity(user), cookie=self.cookie(token))
            if path == "/internal/qc/poll":
                if self.server.qc_dispatch is None:
                    reject("QC dispatch is disabled", 404)
                self.server.qc_dispatch.authenticate(self.headers.get("Authorization"))
                return self.json(self.server.qc_dispatch.poll(body))
            user = self.account(unrestricted=path in {"/api/logout", "/api/password", "/api/activity"})
            if path == "/api/logout":
                self.server.auth.logout(self.token())
                self.server.auth.audit(user["username"], "logout")
                return self.json({"ok": True}, cookie=self.cookie(""))
            if path == "/api/activity":
                return self.json(self.identity(self.account(unrestricted=True, touch=True)))
            if path == "/api/password":
                value = self.server.auth.change_password(user, body.get("old_password"), body.get("new_password"), self.client_address[0])
                return self.json(value, cookie=self.cookie(""))
            if path == "/api/accounts":
                if not user["admin"]:
                    reject("仅管理员可以管理账号", 403)
                if body.get("auth_source") == "backend":
                    value = self.server.auth.link_backend(body.get("username"), body.get("roles"), body.get("tasks"), actor=user["username"], enabled=body.get("enabled", True))
                else:
                    value = self.server.auth.create(body.get("username"), body.get("password"), body.get("roles"), body.get("tasks"), actor=user["username"], enabled=body.get("enabled", True))
                return self.json(value, 201)
            if path.startswith("/api/accounts/"):
                if not user["admin"]:
                    reject("仅管理员可以管理账号", 403)
                return self.json(self.server.auth.update(user, path.split("/")[-1], body))
            batch = self.workbench()
            if path == "/api/lease":
                item = batch.lease(body.get("role"), body.get("job_id"))
                self.media.status(item["id"])
                return self.json(batch.manifest(item["id"]))
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "sessions"]:
                sid, action = parts[2:]
                if action == "release":
                    self.media.cancel(sid)
                    return self.json(batch.release(sid))
                if action == "resume":
                    return self.json(batch.resume(sid))
                if action == "heartbeat":
                    return self.json(batch.heartbeat(sid))
                if action == "submit":
                    return self.json(batch.submit(sid, body))
                if action == "frames":
                    return self.json(batch.record_frames(sid, body))
                if action == "compute":
                    return self.json(self.compute.start(sid, body))
                if action == "retry-media":
                    return self.json(self.media.retry(sid))
            reject("接口不存在", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except WorkflowError as exc:
            self.json({"error": str(exc)}, int(exc.status))
        except (ValueError, TypeError, KeyError) as exc:
            self.json({"error": str(exc)}, 400)
        except Exception:
            self.json({"error": "服务器处理失败；草稿仍保留，可使用同一提交重试"}, 500)


def configure(server, database, config, root, legacy_operator):
    dispatch_config = config.get("qc_dispatch")
    server.qc_dispatch = QCDispatch(root / "dispatch", dispatch_config) if dispatch_config else None
    server.qc_origins = list(dispatch_config["workers"].values()) if dispatch_config else []
    server.workspace_label = str(config.get("workspace_label") or "")
    server.cookie_name = f"orbbec_account_{server.server_port}"
    public_origin = config.get("public_origin")
    if public_origin:
        parsed = urlsplit(public_origin)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
                parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("public_origin must be an HTTPS origin without credentials, path, query or fragment")
    server.secure_cookie = bool(public_origin)
    server.allowed_origins = {public_origin} if public_origin else {
        f"http://127.0.0.1:{server.server_port}", f"http://localhost:{server.server_port}"}
    server.allowed_hosts = {urlsplit(origin).netloc for origin in server.allowed_origins}
    server.auth = Accounts(root / "accounts.sqlite3", backend_accounts=config.get("backend_accounts_file"))
    server.auth.bootstrap(legacy_operator)
    # Account changes select a new, immutable permission context. Request threads
    # never mutate a shared operator. Heavy work uses global resource limits.
    units, lock = {}, threading.Lock()
    slots = threading.Semaphore(2)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="browser-compute")
    def workbench(user):
        key = (user["operator"], tuple(user["roles"]), tuple(user["tasks"]))
        with lock:
            if key not in units:
                batch = BatchService(database, config["nas_mounts"], user["operator"], roles=user["roles"], tasks=user["tasks"])
                media = BrowserMedia(batch, root / "media", config, slots=slots)
                if server.qc_dispatch:
                    media = RoutedMedia(media, server.qc_dispatch)
                units[key] = SimpleNamespace(batch=batch, media=media,
                    compute=BrowserCompute(batch, media, config, pool=pool))
            return units[key]
    server.workbench = workbench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--port", type=int, default=18881)
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error("database must be an existing workflow database")
    config = json.loads(args.config.read_text())
    root = args.state_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    configure(server, args.database, config, root, args.operator)
    connection = root/"connection.json"
    connection.write_text(json.dumps({"url": f"http://127.0.0.1:{args.port}/"}))
    connection.chmod(0o600)
    print(f"Browser workbench listening on 127.0.0.1:{args.port}; account login required", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
