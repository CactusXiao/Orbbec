#!/usr/bin/env python3
"""PICO ego streaming server for Windows ADB reverse TCP captures."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import socket
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Optional


MAGIC = 0x50454731  # "PEG1"
VERSION = 1
HEADER_STRUCT = struct.Struct("!IBBHIQ")
DEFAULT_TIME_SYNC_SAMPLE_COUNT = 20
TIME_SYNC_SAMPLE_TIMEOUT_SECONDS = 1.0

PKT_HELLO = 1
PKT_START = 2
PKT_STOP = 3
PKT_CAMERA_JSON = 4
PKT_METADATA_ROW = 5
PKT_TIMESTAMP_ROW = 6
PKT_HEVC_SAMPLE = 7
PKT_SESSION_END = 8
PKT_ERROR = 9
PKT_METADATA_HEADER = 10
PKT_TIMESTAMP_HEADER = 11
PKT_TIME_SYNC_REQUEST = 12
PKT_TIME_SYNC_RESPONSE = 13

PACKET_NAMES = {
    PKT_HELLO: "HELLO",
    PKT_START: "START",
    PKT_STOP: "STOP",
    PKT_CAMERA_JSON: "CAMERA_JSON",
    PKT_METADATA_ROW: "METADATA_ROW",
    PKT_TIMESTAMP_ROW: "TIMESTAMP_ROW",
    PKT_HEVC_SAMPLE: "HEVC_SAMPLE",
    PKT_SESSION_END: "SESSION_END",
    PKT_ERROR: "ERROR",
    PKT_METADATA_HEADER: "METADATA_HEADER",
    PKT_TIMESTAMP_HEADER: "TIMESTAMP_HEADER",
    PKT_TIME_SYNC_REQUEST: "TIME_SYNC_REQUEST",
    PKT_TIME_SYNC_RESPONSE: "TIME_SYNC_RESPONSE",
}


class Packet:
    def __init__(self, packet_type: int, header: Optional[dict] = None, payload: bytes = b"") -> None:
        self.packet_type = packet_type
        self.header = header or {}
        self.payload = payload or b""


class SessionWriter:
    def __init__(self, root: Path, session_name: str, time_calibration: Optional[dict] = None) -> None:
        self.root = root
        self.session_name = sanitize_session_name(session_name)
        self.path = root / self.session_name
        if self.path.exists():
            suffix = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = root / f"{self.session_name}_{suffix}"
        self.path.mkdir(parents=True, exist_ok=False)

        self.video_path = self.path / "video.h265"
        self.metadata_path = self.path / "metadata.csv"
        self.timestamps_path = self.path / "timestamps.csv"
        self.camera_path = self.path / "camera.json"
        self.session_json_path = self.path / "session.json"
        self.network_log_path = self.path / "network_log.jsonl"
        self.time_calibration_path = self.path / "time_calibration.json"

        self.video_file: BinaryIO = self.video_path.open("wb")
        self.metadata_file = self.metadata_path.open("w", encoding="utf-8", newline="")
        self.timestamps_file = self.timestamps_path.open("w", encoding="utf-8", newline="")
        self.network_log_file = self.network_log_path.open("w", encoding="utf-8")
        self.time_calibration = dict(time_calibration) if time_calibration else None

        self.started_unix_us = unix_us_now()
        self.ended_unix_us: Optional[int] = None
        self.video_bytes = 0
        self.hevc_samples = 0
        self.metadata_rows = 0
        self.timestamp_rows = 0
        self.camera_json_received = False
        self.last_error = ""
        self.client_summary: dict = {}
        if self.time_calibration is not None:
            self.time_calibration_path.write_text(
                json.dumps(self.time_calibration, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.log({"event": "time_calibration_snapshot", "time_calibration": self.time_calibration})

    def write_camera_json(self, payload: bytes) -> None:
        self.camera_path.write_bytes(payload)
        self.camera_json_received = True
        self.log({"event": "camera_json", "bytes": len(payload)})

    def write_metadata_header(self, payload: bytes) -> None:
        self.metadata_file.write(payload.decode("utf-8"))
        self.metadata_file.flush()
        self.log({"event": "metadata_header", "bytes": len(payload)})

    def write_metadata_row(self, payload: bytes) -> None:
        self.metadata_file.write(payload.decode("utf-8"))
        self.metadata_rows += 1

    def write_timestamp_header(self, payload: bytes) -> None:
        self.timestamps_file.write(payload.decode("utf-8"))
        self.timestamps_file.flush()
        self.log({"event": "timestamp_header", "bytes": len(payload)})

    def write_timestamp_row(self, payload: bytes) -> None:
        self.timestamps_file.write(payload.decode("utf-8"))
        self.timestamp_rows += 1

    def write_hevc_sample(self, header: dict, payload: bytes) -> None:
        offset = self.video_file.tell()
        self.video_file.write(payload)
        self.video_bytes += len(payload)
        self.hevc_samples += 1
        log_record = dict(header)
        log_record.update(
            {
                "event": "hevc_sample",
                "video_offset": offset,
                "payload_size": len(payload),
            }
        )
        self.log(log_record)

    def mark_error(self, message: str) -> None:
        self.last_error = message
        self.log({"event": "client_error", "message": message})

    def mark_session_end(self, header: dict) -> None:
        self.client_summary = header
        self.log({"event": "session_end", "client_summary": header})

    def log(self, record: dict) -> None:
        record = dict(record)
        record.setdefault("server_unix_us", unix_us_now())
        self.network_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.ended_unix_us is None:
            self.ended_unix_us = unix_us_now()

        for handle in (self.video_file, self.metadata_file, self.timestamps_file, self.network_log_file):
            try:
                handle.flush()
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass

        summary = {
            "session_name": self.session_name,
            "session_dir": str(self.path),
            "started_unix_us": self.started_unix_us,
            "ended_unix_us": self.ended_unix_us,
            "duration_seconds": (self.ended_unix_us - self.started_unix_us) / 1_000_000.0,
            "video_h265": str(self.video_path),
            "metadata_csv": str(self.metadata_path),
            "timestamps_csv": str(self.timestamps_path),
            "camera_json": str(self.camera_path),
            "network_log_jsonl": str(self.network_log_path),
            "time_calibration_json": str(self.time_calibration_path) if self.time_calibration is not None else "",
            "time_calibration": self.time_calibration,
            "video_bytes": self.video_bytes,
            "hevc_samples": self.hevc_samples,
            "metadata_rows": self.metadata_rows,
            "timestamp_rows": self.timestamp_rows,
            "camera_json_received": self.camera_json_received,
            "last_error": self.last_error,
            "client_summary": self.client_summary,
            "timestamp_standard": "unix_epoch_microseconds_utc",
            "transport": "adb_reverse_tcp",
        }
        self.session_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


class StreamServer:
    def __init__(self, host: str, port: int, output_root: Path) -> None:
        self.host = host
        self.port = port
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.listener: Optional[socket.socket] = None
        self.client: Optional[socket.socket] = None
        self.client_addr = None
        self.client_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.accept_thread: Optional[threading.Thread] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.session: Optional[SessionWriter] = None
        self.last_hello: dict = {}
        self.time_sync_condition = threading.Condition()
        self.time_sync_calibration_id: Optional[str] = None
        self.time_sync_responses: dict[int, dict] = {}
        self.time_calibration_latest_path = self.output_root / "time_calibration_latest.json"
        self.last_time_calibration: Optional[dict] = None

    def start(self) -> None:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((self.host, self.port))
        self.listener.listen(1)
        self.accept_thread = threading.Thread(target=self._accept_loop, name="accept", daemon=True)
        self.accept_thread.start()
        print(f"[server] listening on {self.host}:{self.port}")

    def stop(self) -> None:
        self.stop_event.set()
        with self.client_lock:
            if self.client is not None:
                try:
                    self.client.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    self.client.close()
                except Exception:
                    pass
                self.client = None
        if self.listener is not None:
            try:
                self.listener.close()
            except Exception:
                pass
        self._close_session()

    def start_capture(self, session_name: str) -> None:
        with self.client_lock:
            if self.client is None:
                print("[server] no PICO client connected")
                return
        if self.session is not None:
            print("[server] a session is already active; stop it first")
            return

        calibration = self.time_calibrate(DEFAULT_TIME_SYNC_SAMPLE_COUNT, reason="auto_start")
        if calibration is None:
            print("[server] START aborted because timecalibrate failed")
            return

        self.session = SessionWriter(self.output_root, session_name, calibration)
        header = {"session_name": self.session.session_name, "server_unix_us": unix_us_now()}
        self.send_packet(Packet(PKT_START, header))
        print(f"[server] START sent, output={self.session.path}")

    def stop_capture(self) -> None:
        self.send_packet(Packet(PKT_STOP, {"server_unix_us": unix_us_now()}))
        print("[server] STOP sent")

    def time_calibrate(
        self,
        sample_count: int = DEFAULT_TIME_SYNC_SAMPLE_COUNT,
        reason: str = "manual",
        timeout_seconds: float = TIME_SYNC_SAMPLE_TIMEOUT_SECONDS,
    ) -> Optional[dict]:
        if sample_count <= 0:
            print("[server] timecalibrate failed: sample_count must be positive")
            return None
        if sample_count > 200:
            print("[server] timecalibrate failed: sample_count must be <= 200")
            return None
        with self.client_lock:
            connected = self.client is not None
        if not connected:
            print("[server] timecalibrate failed: no PICO client connected")
            return None
        if self.session is not None:
            print("[server] timecalibrate failed: stop the active session first")
            return None

        calibration_id = f"{int(time.time_ns())}_{uuid.uuid4().hex[:8]}"
        with self.time_sync_condition:
            self.time_sync_calibration_id = calibration_id
            self.time_sync_responses = {}

        responses: list[dict] = []
        try:
            for seq in range(sample_count):
                server_send_unix_us = unix_us_now()
                header = {
                    "calibration_id": calibration_id,
                    "seq": seq,
                    "server_send_unix_us": server_send_unix_us,
                }
                try:
                    self.send_packet(Packet(PKT_TIME_SYNC_REQUEST, header))
                except Exception as exc:
                    print(f"[server] timecalibrate failed while sending seq={seq}: {exc}")
                    return None

                deadline = time.monotonic() + timeout_seconds
                with self.time_sync_condition:
                    while seq not in self.time_sync_responses:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            print(f"[server] timecalibrate failed: timeout waiting for seq={seq}")
                            return None
                        self.time_sync_condition.wait(remaining)
                    responses.append(dict(self.time_sync_responses[seq]))
        finally:
            with self.time_sync_condition:
                if self.time_sync_calibration_id == calibration_id:
                    self.time_sync_calibration_id = None
                    self.time_sync_responses = {}

        try:
            calibration = build_time_calibration_result(calibration_id, sample_count, responses, reason)
        except ValueError as exc:
            print(f"[server] timecalibrate failed: {exc}")
            return None

        self.last_time_calibration = calibration
        self.time_calibration_latest_path.write_text(
            json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("[server] " + format_time_calibration_summary(calibration))
        return calibration

    def status(self) -> None:
        with self.client_lock:
            connected = self.client is not None
            addr = self.client_addr
        print(f"[server] connected={connected} addr={addr} hello={self.last_hello}")
        if self.last_time_calibration is None:
            print("[server] time_calibration=None")
        else:
            print("[server] " + format_time_calibration_summary(self.last_time_calibration))
        if self.session is None:
            print("[server] session=None")
        else:
            print(
                "[server] session={0} metadata_rows={1} timestamp_rows={2} hevc_samples={3} video_mb={4:.2f}".format(
                    self.session.path,
                    self.session.metadata_rows,
                    self.session.timestamp_rows,
                    self.session.hevc_samples,
                    self.session.video_bytes / 1024 / 1024,
                )
            )

    def send_packet(self, packet: Packet) -> None:
        with self.client_lock:
            client = self.client
        if client is None:
            raise RuntimeError("No connected client")
        with self.send_lock:
            send_packet(client, packet)

    def _accept_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                assert self.listener is not None
                client, addr = self.listener.accept()
            except OSError:
                break
            print(f"[server] client connected from {addr}")
            with self.client_lock:
                if self.client is not None:
                    try:
                        self.client.close()
                    except Exception:
                        pass
                self.client = client
                self.client_addr = addr
            self.reader_thread = threading.Thread(target=self._reader_loop, args=(client,), name="reader", daemon=True)
            self.reader_thread.start()

    def _reader_loop(self, client: socket.socket) -> None:
        try:
            while not self.stop_event.is_set():
                packet = recv_packet(client)
                self._handle_packet(packet)
        except Exception as exc:
            print(f"[server] client disconnected: {exc}")
        finally:
            with self.client_lock:
                if self.client is client:
                    self.client = None
                    self.client_addr = None

    def _handle_packet(self, packet: Packet) -> None:
        packet_name = PACKET_NAMES.get(packet.packet_type, str(packet.packet_type))
        if packet.packet_type == PKT_HELLO:
            self.last_hello = packet.header
            print(f"[server] HELLO {packet.header}")
            return

        if packet.packet_type == PKT_ERROR:
            message = packet.header.get("message", "unknown client error")
            print(f"[server] client ERROR: {message}")
            if self.session is not None:
                self.session.mark_error(message)
            return

        if packet.packet_type == PKT_TIME_SYNC_RESPONSE:
            self._handle_time_sync_response(packet.header)
            return

        if self.session is None:
            print(f"[server] ignoring {packet_name}; no active session")
            return

        if packet.packet_type == PKT_CAMERA_JSON:
            self.session.write_camera_json(packet.payload)
        elif packet.packet_type == PKT_METADATA_HEADER:
            self.session.write_metadata_header(packet.payload)
        elif packet.packet_type == PKT_METADATA_ROW:
            self.session.write_metadata_row(packet.payload)
        elif packet.packet_type == PKT_TIMESTAMP_HEADER:
            self.session.write_timestamp_header(packet.payload)
        elif packet.packet_type == PKT_TIMESTAMP_ROW:
            self.session.write_timestamp_row(packet.payload)
        elif packet.packet_type == PKT_HEVC_SAMPLE:
            self.session.write_hevc_sample(packet.header, packet.payload)
        elif packet.packet_type == PKT_SESSION_END:
            self.session.mark_session_end(packet.header)
            self._close_session()
            print("[server] session finalized")
        else:
            self.session.log({"event": "unknown_packet", "packet_type": packet.packet_type, "header": packet.header})

    def _handle_time_sync_response(self, header: dict) -> None:
        calibration_id = str(header.get("calibration_id", ""))
        try:
            seq = int(header.get("seq"))
        except (TypeError, ValueError):
            return
        response = dict(header)
        response["server_recv_unix_us"] = unix_us_now()
        with self.time_sync_condition:
            if calibration_id != self.time_sync_calibration_id:
                return
            self.time_sync_responses[seq] = response
            self.time_sync_condition.notify_all()

    def _close_session(self) -> None:
        if self.session is not None:
            path = self.session.path
            self.session.close()
            self.session = None
            print(f"[server] session closed: {path}")


def send_packet(sock: socket.socket, packet: Packet) -> None:
    header_json = json.dumps(packet.header or {}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload = packet.payload or b""
    fixed = HEADER_STRUCT.pack(MAGIC, VERSION, packet.packet_type, 0, len(header_json), len(payload))
    sock.sendall(fixed)
    sock.sendall(header_json)
    if payload:
        sock.sendall(payload)


def recv_packet(sock: socket.socket) -> Packet:
    fixed = recv_exact(sock, HEADER_STRUCT.size)
    magic, version, packet_type, _flags, header_len, payload_len = HEADER_STRUCT.unpack(fixed)
    if magic != MAGIC:
        raise RuntimeError(f"invalid packet magic: {magic:#x}")
    if version != VERSION:
        raise RuntimeError(f"unsupported packet version: {version}")
    if header_len > 1024 * 1024:
        raise RuntimeError(f"header too large: {header_len}")
    if payload_len > 512 * 1024 * 1024:
        raise RuntimeError(f"payload too large: {payload_len}")

    header_bytes = recv_exact(sock, header_len) if header_len else b"{}"
    payload = recv_exact(sock, payload_len) if payload_len else b""
    header = json.loads(header_bytes.decode("utf-8")) if header_bytes else {}
    return Packet(packet_type, header, payload)


def recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def sanitize_session_name(value: str) -> str:
    value = (value or "").strip()
    if not value:
        value = _dt.datetime.now().strftime("session_%Y%m%d_%H%M%S")
    safe = []
    for ch in value:
        safe.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(safe)


def unix_us_now() -> int:
    return int(time.time_ns() // 1000)


def build_time_calibration_result(
    calibration_id: str,
    sample_count: int,
    responses: list[dict],
    reason: str,
) -> dict:
    valid_samples = []
    for response in responses:
        try:
            seq = int(response["seq"])
            server_send = int(response["server_send_unix_us"])
            client_recv = int(response["client_recv_unix_us"])
            client_send = int(response["client_send_unix_us"])
            server_recv = int(response["server_recv_unix_us"])
        except (KeyError, TypeError, ValueError):
            continue

        rtt_us = (server_recv - server_send) - (client_send - client_recv)
        client_processing_us = client_send - client_recv
        if rtt_us < 0 or client_processing_us < 0:
            continue
        offset_numerator = (server_send + server_recv) - (client_recv + client_send)
        offset_us = offset_numerator // 2
        valid_samples.append(
            {
                "seq": seq,
                "server_send_unix_us": server_send,
                "client_recv_unix_us": client_recv,
                "client_send_unix_us": client_send,
                "server_recv_unix_us": server_recv,
                "rtt_us": rtt_us,
                "client_processing_us": client_processing_us,
                "pico_to_host_offset_us": offset_us,
            }
        )

    if not valid_samples:
        raise ValueError("no valid time sync responses")

    samples_by_rtt = sorted(valid_samples, key=lambda item: (item["rtt_us"], item["seq"]))
    best_sample_count = max(1, (len(samples_by_rtt) + 1) // 2)
    best_samples = samples_by_rtt[:best_sample_count]
    rtts = [sample["rtt_us"] for sample in valid_samples]
    best_offsets = [sample["pico_to_host_offset_us"] for sample in best_samples]
    offset_us = median_int(best_offsets)

    return {
        "calibration_id": calibration_id,
        "created_unix_us": unix_us_now(),
        "reason": reason,
        "method": "ntp_best_half_median_offset_v1",
        "sample_count": sample_count,
        "response_count": len(responses),
        "accepted_count": len(valid_samples),
        "best_sample_count": best_sample_count,
        "pico_to_host_offset_us": offset_us,
        "host_to_pico_offset_us": -offset_us,
        "min_rtt_us": min(rtts),
        "median_rtt_us": median_int(rtts),
        "max_rtt_us": max(rtts),
        "formula": "host_unix_us = pico_unix_us + pico_to_host_offset_us",
        "samples": valid_samples,
    }


def median_int(values: list[int]) -> int:
    if not values:
        raise ValueError("median_int requires at least one value")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) // 2


def format_time_calibration_summary(calibration: dict) -> str:
    return (
        "timecalibrate ok samples={sample_count} accepted={accepted_count} "
        "pico_to_host_offset_us={pico_to_host_offset_us} min_rtt_us={min_rtt_us} "
        "median_rtt_us={median_rtt_us} max_rtt_us={max_rtt_us}"
    ).format(**calibration)


def command_loop(server: StreamServer) -> None:
    print("Commands: start <session_name> | stop | status | timecalibrate [sample_count] | quit")
    while True:
        try:
            line = input("stream-server> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if command == "start":
            server.start_capture(arg or _dt.datetime.now().strftime("session_%Y%m%d_%H%M%S"))
        elif command == "stop":
            server.stop_capture()
        elif command == "status":
            server.status()
        elif command in ("timecalibrate", "time_calibrate"):
            try:
                sample_count = int(arg) if arg else DEFAULT_TIME_SYNC_SAMPLE_COUNT
            except ValueError:
                print("[server] timecalibrate failed: sample_count must be an integer")
                continue
            server.time_calibrate(sample_count, reason="manual")
        elif command in ("quit", "exit"):
            break
        else:
            print("Unknown command. Use: start <session_name> | stop | status | timecalibrate [sample_count] | quit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PICO ego streaming server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Use 127.0.0.1 with adb reverse.")
    parser.add_argument("--port", type=int, default=50051, help="TCP port.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "sessions",
        help="Directory where sessions are saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = StreamServer(args.host, args.port, args.output_root)
    server.start()
    try:
        command_loop(server)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
