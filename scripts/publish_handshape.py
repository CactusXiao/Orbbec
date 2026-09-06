#!/usr/bin/env python3
"""Publish the fixed subject handshape capture through the NAS shape endpoint.

The fixed-path source replacement protocol must also be supported by Publisher.
Legacy immutable episodes are detected before changing their files.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from urllib.request import Request, urlopen

TASK = "task_handshapeCalibration"
EPISODE = "episode1"


def episode_path(root: Path, subject: str) -> Path:
    if not subject or subject in (".", "..") or "/" in subject or "\\" in subject:
        raise ValueError("Invalid subject ID")
    root = root.absolute()
    current = root
    for part in (subject, TASK, EPISODE):
        current /= part
        if current.is_symlink():
            raise ValueError(f"Calibration path contains a symlink: {current}")
    return current


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink receipt: {path}")
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


class Publisher:
    def __init__(self):
        self.host = os.environ.get("ORBBEC_PUBLISHER_BRIDGE_SSH_HOST", "synology")
        self.publish_command = os.environ.get("ORBBEC_PUBLISHER_BRIDGE_PUBLISH_COMMAND", "/usr/local/sbin/nas-uploader-publish")
        self.status_command = os.environ.get("ORBBEC_PUBLISHER_BRIDGE_STATUS_COMMAND", "/usr/local/sbin/nas-uploader-status")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", self.host):
            raise ValueError("Invalid publisher SSH host")
        for command in (self.publish_command, self.status_command):
            if not re.fullmatch(r"/[A-Za-z0-9_./-]+", command):
                raise ValueError("Invalid publisher command")

    def run(self, command: str, *args: str) -> str:
        remote = shlex.join(["sudo", "--", command, *args])
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.host, remote],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip() or f"Publisher exited {result.returncode}")
        return result.stdout

    def status(self, episode: str) -> dict:
        result = json.loads(self.run(self.status_command, episode))
        if not isinstance(result, dict) or result.get("episode_id") != episode or not isinstance(result.get("found"), bool):
            raise ValueError("Publisher returned an invalid episode status")
        return result

    def publish(self, episode: str) -> None:
        self.run(self.publish_command, "--shape-calibration", episode)


def publish_capture(source: Path, nas_root: Path, subject: str, token: str, publisher: Publisher) -> dict:
    if not token.strip():
        raise ValueError("Capture token is required")
    source = source.absolute()
    if source.name != EPISODE or source.parent.name != TASK or source.parent.parent.name != subject:
        raise ValueError("Source must be subject/task_handshapeCalibration/episode1")
    episode_path(source.parents[2], subject)
    dest = episode_path(nas_root, subject)
    if not nas_root.is_dir():
        raise ValueError("NAS mount is unavailable")
    if source.resolve() == dest.resolve() or nas_root.resolve() in source.resolve().parents:
        raise ValueError("Capture source must be outside the NAS mount")
    if not source.is_dir() or not any(source.iterdir()):
        raise ValueError("Calibration capture is empty or missing")
    # Never follow a captured symlink into unrelated data during upload.
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Capture contains a symlink: {path}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dest.parent / ".publish.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = dest.parent / ".episode1.publish.json"
        if receipt_path.is_symlink():
            raise ValueError("Publish receipt must not be a symlink")
        receipt = read_json(receipt_path)
        episode = f"{subject}/{TASK}/{EPISODE}"
        state = publisher.status(episode)
        same_capture = receipt.get("token") == token and dest.is_dir()
        if state.get("found") and not same_capture:
            raise RuntimeError(
                "Publisher already knows this fixed episode path. Recalibration requires the updated "
                "Publisher source-replacement protocol; refusing to report its idempotent response as a new calibration. "
                "Local capture has been preserved."
            )
        if same_capture and state.get("found") and receipt.get("submitted"):
            return receipt
        if not same_capture:
            staging = dest.parent / (".episode1.upload-" + uuid.uuid4().hex)
            previous = dest.parent / (".episode1.previous-" + uuid.uuid4().hex)
            try:
                shutil.copytree(source, staging)
                if dest.exists():
                    dest.rename(previous)
                try:
                    staging.rename(dest)
                except BaseException:
                    if previous.exists():
                        previous.rename(dest)
                    raise
                receipt = {"episode_id": episode, "token": token, "submitted": False}
                write_json(receipt_path, receipt)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(previous, ignore_errors=True)
        # Persist the capture identity before the call. On timeout, retry the same
        # source instead of replacing files that the remote uploader may be reading.
        publisher.publish(episode)
        receipt["submitted"] = True
        write_json(receipt_path, receipt)
        return receipt
    finally:
        os.close(descriptor)


def register_backend_progress(backend_url: str, nas_uri_prefix: str, subject: str, source: Path, receipt: dict) -> dict:
    body = {"subject_id": subject, "capture_token": receipt["token"],
            "collection_path": str(source),
            "episode_uri": nas_uri_prefix.rstrip("/") + "/" + receipt["episode_id"]}
    request = Request(backend_url.rstrip("/") + "/api/v1/shape-calibration/published",
                      data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--nas-root", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8765")
    parser.add_argument("--nas-uri-prefix", default="nas://ego")
    args = parser.parse_args()
    try:
        result = publish_capture(args.source, args.nas_root, args.subject, args.token, Publisher())
        registered = register_backend_progress(args.backend_url, args.nas_uri_prefix, args.subject, args.source, result)
        result = {**result, "backend_job_id": (registered.get("job") or {}).get("job_id")}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Handshape publish failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
