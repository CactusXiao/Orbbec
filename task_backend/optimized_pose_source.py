"""Shared pose indexing; archive validation needs only the standard library.

Publisher stores frame_ids (N,) and poses (N,2,99) in optimized_pose/poses.npz.
An archive is authoritative when present, even beside old per-frame files.
NumPy is imported only when a consumer requests pose values.
"""
from __future__ import annotations

import ast
import struct
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO


def _header(handle: BinaryIO) -> tuple[tuple[int, ...], str, bool, int]:
    if handle.read(6) != b"\x93NUMPY":
        raise ValueError("missing NPY magic")
    version = handle.read(2)
    if version not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        raise ValueError("unsupported NPY version")
    size = 2 if version[0] == 1 else 4
    raw = handle.read(size)
    if len(raw) != size:
        raise ValueError("truncated NPY header length")
    length = int.from_bytes(raw, "little")
    if length > 65536:
        raise ValueError("NPY header is too large")
    raw = handle.read(length)
    if len(raw) != length:
        raise ValueError("truncated NPY header")
    header = ast.literal_eval(raw.decode("utf-8" if version[0] == 3 else "latin1"))
    if not isinstance(header, dict):
        raise ValueError("NPY header must be a mapping")
    shape = header.get("shape")
    if not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("invalid NPY shape")
    return shape, header.get("descr"), header.get("fortran_order"), 8 + size + length


def archive_frame_ids(path: Path) -> list[int]:
    """Validate the packed schema and read explicit IDs without importing NumPy."""
    with zipfile.ZipFile(path) as archive:
        for name in ("frame_ids.npy", "poses.npy"):
            if archive.namelist().count(name) != 1:
                raise ValueError(f"{path}: expected exactly one {name}")
        with archive.open("frame_ids.npy") as handle:
            shape, dtype, fortran, offset = _header(handle)
            codes = {"i4": "i", "i8": "q", "u4": "I", "u8": "Q"}
            if (len(shape) != 1 or shape[0] == 0 or not isinstance(dtype, str)
                    or dtype[1:] not in codes or dtype[0] not in "<>=|" or fortran):
                raise ValueError(f"{path}: frame_ids must be a nonempty integer vector")
            size = int(dtype[2:])
            expected = shape[0] * size
            if archive.getinfo("frame_ids.npy").file_size != offset + expected:
                raise ValueError(f"{path}: truncated frame_ids payload")
            data = handle.read()
            endian = dtype[0] if dtype[0] in "<>" else ("<" if sys.byteorder == "little" else ">")
            frames = [value[0] for value in struct.iter_unpack(endian + codes[dtype[1:]], data)]
        if min(frames) < 0 or len(set(frames)) != len(frames):
            raise ValueError(f"{path}: frame_ids must be unique and nonnegative")
        with archive.open("poses.npy") as handle:
            pose_shape, dtype, fortran, offset = _header(handle)
        if pose_shape != (len(frames), 2, 99) or dtype not in ("<f4", ">f4", "=f4", "|f4") or fortran:
            raise ValueError(f"{path}: poses must be C-order float32 (N,2,99), got {pose_shape} {dtype}")
        if archive.getinfo("poses.npy").file_size != offset + len(frames) * 2 * 99 * 4:
            raise ValueError(f"{path}: truncated poses payload")
    return frames


class OptimizedPoseSource:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.archive = self.directory / "poses.npz"
        self._poses = None
        self._rows: dict[int, int] = {}
        self.files: dict[int, Path] = {}
        if self.archive.is_file():
            self._rows = {frame: row for row, frame in enumerate(archive_frame_ids(self.archive))}
            self.frames = sorted(self._rows)
        else:
            if not self.directory.is_dir():
                raise FileNotFoundError(f"optimized_pose directory is not visible yet: {directory}")
            for path in self.directory.glob("*.npy"):
                if not path.stem.isdigit():
                    continue
                frame = int(path.stem)
                if frame in self.files:
                    raise ValueError(f"duplicate optimized_pose frame: {frame}")
                self.files[frame] = path
            self.frames = sorted(self.files)
            if not self.frames:
                raise FileNotFoundError(f"optimized_pose has no poses.npz or numeric .npy frames yet: {directory}")

    @property
    def is_packed(self) -> bool:
        return bool(self._rows)

    def path_for_frame(self, frame: int) -> Path:
        if frame not in self._rows and frame not in self.files:
            raise KeyError(f"optimized_pose frame {frame} is missing: {self.directory}")
        return self.archive if self._rows else self.files[frame]

    def load(self, frame: int) -> Any:
        import numpy as np

        path = self.path_for_frame(frame)
        if self._rows:
            if self._poses is None:
                with np.load(self.archive, allow_pickle=False) as archive:
                    ids = archive["frame_ids"].tolist()
                    if {value: row for row, value in enumerate(ids)} != self._rows:
                        raise OSError(f"pose archive changed while reading: {self.archive}")
                    poses = archive["poses"]
                if poses.shape != (len(ids), 2, 99) or poses.dtype.kind != "f" or poses.dtype.itemsize != 4:
                    raise ValueError(f"invalid poses shape/dtype: {self.archive}")
                if not np.isfinite(poses).all():
                    raise ValueError(f"optimized_pose contains non-finite values: {self.archive}")
                self._poses = poses
            return self._poses[self._rows[frame]].astype(np.float32, copy=False)
        pose = np.load(path, allow_pickle=False)
        if pose.shape != (2, 99) or pose.dtype.kind != "f" or pose.dtype.itemsize != 4 or not pose.flags.c_contiguous:
            raise ValueError(f"optimized_pose must be C-order float32 (2,99): {path}")
        if not np.isfinite(pose).all():
            raise ValueError(f"optimized_pose contains non-finite values: {path}")
        return pose.astype(np.float32, copy=False)


@lru_cache(maxsize=2)
def _cached_archive(directory: Path, signature: tuple[int, ...]) -> OptimizedPoseSource:
    return OptimizedPoseSource(directory)


def load_archive_frame(directory: Path, frame: int) -> Any:
    """Reuse decompressed poses for previews; invalidate on archive replacement."""
    directory = Path(directory).resolve()
    stat = (directory / "poses.npz").stat()
    return _cached_archive(directory, (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)).load(frame)
