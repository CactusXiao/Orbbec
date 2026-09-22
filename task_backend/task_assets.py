"""Task catalog additions and NAS assets (standard library only)."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import io
from contextlib import contextmanager
from email.message import Message
from email.parser import BytesParser
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt

CATALOG_LOCK = threading.RLock()
VIDEO_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".m4v": "video/mp4"}
MAX_UPLOAD = 2 * 1024**3


def validate_name(name):
    if (not isinstance(name, str) or not name or name != name.strip()
            or len(name) > 120 or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name in {".", ".."} or name.endswith((".", " "))
            or re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", name, re.I)):
        raise ValueError("任务名称不能为空，且不能包含路径分隔符或文件名禁用字符")
    return name


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def catalog_lock(path):
    with CATALOG_LOCK, path.with_suffix(path.suffix + ".lock").open("a+b") as handle:
        if fcntl:
            fcntl.flock(handle, fcntl.LOCK_EX)
        else:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(handle, fcntl.LOCK_UN)
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def description_text(value):
    """Keep step names, order, and nested descriptions visible to old clients."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(filter(None, (description_text(item) for item in value)))
    if isinstance(value, dict):
        return "\n".join(f"{key}: {text}" for key, item in value.items()
                         if (text := description_text(item)))
    return str(value) if value is not None else ""


def task_descriptions(document):
    generic = next((document[key] for key in ("taskdescription", "task_description", "description", "steps")
                    if document.get(key)), None)
    if generic is None:
        generic = {key: value for key, value in document.items() if key.lower().startswith("step")}
    return {
        "description_cn": description_text(document.get("description_cn") or document.get("task_description_cn") or generic),
        "description_en": description_text(document.get("description_en") or document.get("task_description_en") or generic),
    }


def task_directory(root, name):
    validate_name(name)
    directory = root / "tasks" / name
    # Do not follow task-folder symlinks outside the NAS task tree.
    if directory.resolve().parent != (root / "tasks").resolve():
        raise ValueError("任务目录超出 NAS tasks 目录")
    return directory


def enrich_task(task, root):
    if root is None:
        return task
    try:
        directory = task_directory(root, task["task_name"])
        candidates = [directory / "task.json"] + sorted(directory.glob("*.json"))
        for path in dict.fromkeys(candidates):
            if not path.is_file():
                continue
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(document, dict) or document.get("task_name") != task["task_name"]:
                continue
            for key, text in task_descriptions(document).items():
                if text:
                    task[key] = text
            break
    except (OSError, ValueError):
        pass  # Catalog descriptions remain available when NAS metadata is absent/unreadable.
    return task


def add_task(catalog, root, name, document_path, video_path, video_name, total, load_catalog, strip_comments):
    validate_name(name)
    if root is None or not root.is_dir():
        raise ValueError("NAS 挂载目录不可用，请检查 ORBBEC_NAS_ROOT")
    if document_path.stat().st_size > 4 * 1024**2:
        raise ValueError("单任务 JSON 不能超过 4 MB")
    try:
        document = json.loads(document_path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("单任务 JSON 格式无效") from exc
    if not isinstance(document, dict) or document.get("task_name") != name:
        raise ValueError("任务名称必须与单任务 JSON 的 task_name 字段完全一致")
    if not any(task_descriptions(document).values()):
        raise ValueError("单任务 JSON 必须包含任务描述或步骤描述")
    suffix = Path(video_name).suffix.lower()
    if suffix not in VIDEO_TYPES or video_path.stat().st_size == 0:
        raise ValueError("请选择非空的 MP4、WebM、MOV 或 M4V 演示视频")
    try:
        total = int(total)
        if total < 1:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("采集次数必须是正整数")
    with catalog_lock(catalog), catalog_lock(root / ".task-assets"):
        current = load_catalog(catalog)
        if any(task["task_name"].casefold() == name.casefold() for task in current):
            raise FileExistsError("任务名称已存在")
        destination = task_directory(root, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if any(child.name.casefold() == name.casefold() for child in destination.parent.iterdir()):
            raise FileExistsError("NAS 中已有同名任务目录，未覆盖已有文件")
        raw = json.loads(strip_comments(catalog.read_text(encoding="utf-8")))
        entry = {"task_name": name, "repeat_times": total}
        if isinstance(raw, list):
            raw.append(entry)
        elif isinstance(raw.get("tasks"), list):
            raw["tasks"].append(entry)
        else:
            if name in raw:
                raise FileExistsError("任务目录键已存在")
            raw[name] = entry
        staging = Path(tempfile.mkdtemp(prefix=".new-task-", dir=destination.parent))
        published = False
        try:
            shutil.copyfile(document_path, staging / "task.json")
            shutil.copyfile(video_path, staging / ("demo" + suffix))
            staging.rename(destination)
            published = True
            atomic_json(catalog, raw)
        except Exception:
            if published:
                shutil.rmtree(destination)
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return name


def read_multipart(stream, content_type, length, directory):
    """Stream binary file parts to disk; never buffer a demo video in memory."""
    message = Message()
    message["content-type"] = content_type
    boundary = message.get_param("boundary")
    if message.get_content_type() != "multipart/form-data" or not boundary or len(boundary) > 200:
        raise ValueError("需要 multipart/form-data 上传")
    if length <= 0 or length > MAX_UPLOAD:
        raise ValueError("上传总大小必须在 1 字节到 2 GB 之间")
    marker = b"--" + boundary.encode("ascii")
    remaining, buffer = length, b""

    def until(delimiter, output, limit):
        nonlocal remaining, buffer
        written = 0
        while True:
            index = buffer.find(delimiter)
            if index >= 0:
                chunk, buffer = buffer[:index], buffer[index + len(delimiter):]
                written += len(chunk)
                if written > limit:
                    raise ValueError("上传字段超过大小限制")
                output.write(chunk)
                return
            count = max(0, len(buffer) - len(delimiter) + 1)
            written += count
            if written > limit:
                raise ValueError("上传字段超过大小限制")
            output.write(buffer[:count])
            buffer = buffer[count:]
            if remaining <= 0:
                raise ValueError("上传未完成或 multipart 格式无效")
            chunk = stream.read(min(65536, remaining))
            if not chunk:
                raise ValueError("上传连接中断")
            remaining -= len(chunk)
            buffer += chunk

    start = io.BytesIO()
    until(marker + b"\r\n", start, 0)
    fields = {}
    for index in range(8):
        headers = io.BytesIO()
        until(b"\r\n\r\n", headers, 8192)
        part = BytesParser().parsebytes(headers.getvalue() + b"\r\n\r\n")
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        if name not in {"task_name", "total", "task_json", "demo_video"} or name in fields:
            raise ValueError("上传包含未知或重复字段")
        path = directory / str(index)
        with path.open("wb") as output:
            until(b"\r\n" + marker, output, MAX_UPLOAD if name == "demo_video" else 4 * 1024**2)
        fields[name] = (path, filename) if name in {"task_json", "demo_video"} else path.read_text(encoding="utf-8")
        while len(buffer) < 2 and remaining:
            chunk = stream.read(min(65536, remaining))
            if not chunk:
                raise ValueError("上传连接中断")
            remaining -= len(chunk)
            buffer += chunk
        ending, buffer = buffer[:2], buffer[2:]
        if ending == b"--":
            if set(fields) != {"task_name", "total", "task_json", "demo_video"}:
                raise ValueError("请填写任务名称、采集次数并选择 JSON 和演示视频")
            return fields
        if ending != b"\r\n":
            raise ValueError("multipart 分隔符无效")
    raise ValueError("上传字段过多")
