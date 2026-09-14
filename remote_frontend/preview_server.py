"""Loopback-only preview benchmark with HTTP Range and a 4 Mbps/100 ms profile.

This applies a per-response application-level rate cap, not packet loss or a
full network emulator. It serves only a derived preview and the test page.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, urlsplit


def byte_range(value: str, size: int):
    if not value:
        return 0, size - 1
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()):
        raise ValueError("invalid Range")
    first, last = match.groups()
    if not first:
        count = int(last)
        if count <= 0:
            raise ValueError("invalid suffix")
        return max(0, size - count), size - 1
    start, end = int(first), min(size - 1, int(last)) if last else size - 1
    if start >= size or end < start:
        raise ValueError("unsatisfiable Range")
    return start, end


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        request = urlsplit(self.path)
        if request.path in {"/", "/benchmark.html"}:
            body = Path(__file__).with_name("benchmark.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if request.path != "/preview.mp4":
            self.send_error(404)
            return
        size = self.server.preview.stat().st_size
        try:
            start, end = byte_range(self.headers.get("Range", ""), size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        limited = parse_qs(request.query).get("profile", [""])[0] == "wan"
        if limited:
            time.sleep(0.1)
        self.send_response(206 if self.headers.get("Range") else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if self.headers.get("Range"):
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        sent, began = 0, time.monotonic()
        try:
            with self.server.preview.open("rb") as source:
                source.seek(start)
                while sent < end - start + 1:
                    data = source.read(min(16384, end - start + 1 - sent))
                    if not data:
                        break
                    if limited:
                        # Four million bits per second, with a bounded 16 KB burst.
                        time.sleep(max(0, sent / 500000 - (time.monotonic() - began)))
                    self.wfile.write(data)
                    self.wfile.flush()
                    sent += len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser seek cancels the old byte range.


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preview", type=Path, required=True)
    p.add_argument("--port", type=int, default=18880)
    a = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    server.preview = a.preview.resolve()
    if not server.preview.is_file():
        p.error("preview file does not exist")
    print(f"Preview benchmark: http://127.0.0.1:{a.port}/?mode=video", flush=True)
    server.serve_forever()
