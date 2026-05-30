#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


TASK_DIR = Path("/home/ubuntu/orbbec/src/sync/test/demopico/0")
OUTPUT_DIR = TASK_DIR.parent / "task0"
FRAME_LIMIT = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sort_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def list_first_frames(folder: Path, limit: int) -> list[Path]:
    files = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=sort_key)
    return files[:limit]


def resolve_source_root(task_dir: Path) -> Path:
    if (task_dir / "camera_params.json").is_file():
        return task_dir

    episode_dirs = sorted(
        child for child in task_dir.iterdir()
        if child.is_dir() and child.name.startswith("episode_")
    )
    if len(episode_dirs) == 1:
        return episode_dirs[0]

    raise FileNotFoundError(
        "未找到可用的数据目录。要求输入目录本身包含 camera_params.json，"
        "或其下恰好只有一个 episode_* 子目录。"
    )


def copy_json_files(src_root: Path, dst_root: Path) -> None:
    for json_path in sorted(src_root.glob("*.json")):
        shutil.copy2(json_path, dst_root / json_path.name)


def copy_camera_frames(src_root: Path, dst_root: Path) -> None:
    for folder in sorted(src_root.rglob("*")):
        if not folder.is_dir() or folder.name not in {"RGB", "Depth"}:
            continue

        relative_dir = folder.relative_to(src_root)
        dst_dir = dst_root / relative_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        for frame_path in list_first_frames(folder, FRAME_LIMIT):
            shutil.copy2(frame_path, dst_dir / frame_path.name)


def main() -> None:
    if not TASK_DIR.is_dir():
        raise FileNotFoundError(f"源目录不存在: {TASK_DIR}")

    src_root = resolve_source_root(TASK_DIR)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copy_json_files(src_root, OUTPUT_DIR)
    copy_camera_frames(src_root, OUTPUT_DIR)

    print(f"source: {src_root}")
    print(f"output: {OUTPUT_DIR}")
    print("已复制各机位 RGB/Depth 前10帧以及顶层 json 文件。")


if __name__ == "__main__":
    main()
