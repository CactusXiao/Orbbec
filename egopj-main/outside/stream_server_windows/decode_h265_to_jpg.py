#!/usr/bin/env python3
"""Decode a PICO streaming session video.h265 into per-frame JPG images and MP4 video.

The H.265 elementary stream does not carry the full metadata table used by the
dataset. This script decodes frames in display order and writes an index CSV
that maps decoded images back to metadata.csv by the encoded frame_index when
network_log.jsonl is available. Older captures without that log fall back to
row-order alignment.

It also optionally writes a playable MP4 video from the H.265 elementary stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SESSIONS_DIR = SCRIPT_DIR / "sessions"
DEFAULT_OUTPUT_DIR_NAME = "decoded_jpg"
DEFAULT_VIDEO_ONLY_OUTPUT_DIR_NAME = "decoded_video"
DEFAULT_OUTPUT_VIDEO_NAME = "decoded.mp4"
DEFAULT_JPG_QUALITY = 2
DEFAULT_VIDEO_FPS = 30.0
DEFAULT_VIDEO_CODEC = "copy"
MEDIA_CODEC_BUFFER_FLAG_CODEC_CONFIG = 2
MEDIA_CODEC_BUFFER_FLAG_PARTIAL_FRAME = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode outside/stream_server_windows session video.h265 into JPG frames and MP4 video."
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Session directory containing video.h265 and metadata.csv. Defaults to latest session.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Explicit H.265 elementary stream path. Overrides session-dir/video.h265.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Explicit metadata.csv path. Defaults to session-dir/metadata.csv when available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to session-dir/decoded_jpg, or "
            "session-dir/decoded_video with --video-only."
        ),
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help=(
            "Output MP4 path. Defaults to output-dir/decoded.mp4. "
            "Use --no-video to disable MP4 output."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Do not generate MP4 video output.",
    )
    parser.add_argument(
        "--video-only",
        "--no-jpg",
        dest="video_only",
        action="store_true",
        help="Generate only the MP4 preview; skip per-frame JPG decoding and the JPG index.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help=(
            "Frame rate used for MP4 output. By default, read actual_output_hz from "
            "session.json, then fall back to actual_capture_hz, target_fps, and finally 30."
        ),
    )
    parser.add_argument(
        "--video-codec",
        choices=["copy", "h264"],
        default=DEFAULT_VIDEO_CODEC,
        help=(
            "MP4 video codec mode. "
            "'copy' remuxes H.265 into MP4 without quality loss. "
            "'h264' decodes and re-encodes to H.264 for better compatibility. "
            "Default: copy."
        ),
    )
    parser.add_argument(
        "--ffmpeg",
        default="",
        help="ffmpeg executable path or command name. Defaults to PATH ffmpeg, then imageio-ffmpeg.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=DEFAULT_JPG_QUALITY,
        help="ffmpeg MJPEG q:v quality, lower is better. Default: 2.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not replace existing generated output files.",
    )
    return parser.parse_args()


def resolve_ffmpeg_executable(ffmpeg_arg: str) -> str:
    if ffmpeg_arg:
        return ffmpeg_arg

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError(
            "ffmpeg was not found in PATH and imageio_ffmpeg is not available. "
            "Install FFmpeg or pass --ffmpeg D:/path/to/ffmpeg.exe."
        ) from exc


def find_latest_session(sessions_dir: Path) -> Path:
    candidates = [
        path
        for path in sessions_dir.iterdir()
        if path.is_dir() and (path / "video.h265").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No session with video.h265 found under {sessions_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_inputs(
    args: argparse.Namespace,
) -> tuple[Optional[Path], Path, Optional[Path], Path, Optional[Path]]:
    session_dir = args.session_dir

    if session_dir is None and args.video is None:
        session_dir = find_latest_session(DEFAULT_SESSIONS_DIR)

    if session_dir is not None:
        session_dir = session_dir.resolve()
        video_path = (args.video or session_dir / "video.h265").resolve()
        metadata_path = (args.metadata or session_dir / "metadata.csv").resolve()
        default_output_name = (
            DEFAULT_VIDEO_ONLY_OUTPUT_DIR_NAME if args.video_only else DEFAULT_OUTPUT_DIR_NAME
        )
        output_dir = (args.output_dir or session_dir / default_output_name).resolve()
    else:
        video_path = args.video.resolve()
        metadata_path = args.metadata.resolve() if args.metadata else None
        default_output_name = (
            DEFAULT_VIDEO_ONLY_OUTPUT_DIR_NAME if args.video_only else DEFAULT_OUTPUT_DIR_NAME
        )
        output_dir = (args.output_dir or video_path.parent / default_output_name).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"video.h265 not found: {video_path}")

    if metadata_path is not None and not metadata_path.is_file():
        metadata_path = None

    if args.no_video:
        output_video_path = None
    elif args.output_video is not None:
        output_video_path = args.output_video.resolve()
    else:
        output_video_path = (output_dir / DEFAULT_OUTPUT_VIDEO_NAME).resolve()

    if output_video_path is not None:
        if output_video_path.suffix.lower() != ".mp4":
            raise ValueError(
                f"--output-video must use an .mp4 filename; got: {output_video_path}"
            )

        protected_inputs = {video_path}
        if metadata_path is not None:
            protected_inputs.add(metadata_path)
        if session_dir is not None:
            protected_inputs.update(
                {
                    session_dir / "session.json",
                    session_dir / "camera.json",
                    session_dir / "timestamps.csv",
                    session_dir / "network_log.jsonl",
                    session_dir / "time_calibration.json",
                }
            )
        if output_video_path in protected_inputs:
            raise ValueError(
                f"The output MP4 path conflicts with a captured input file: {output_video_path}"
            )

    return session_dir, video_path, metadata_path, output_dir, output_video_path


def prepare_outputs(
    output_dir: Path,
    output_video_path: Optional[Path],
    decode_jpg: bool,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = [output_dir / "decode_summary.json"]
    if decode_jpg:
        generated_paths.append(output_dir / "decoded_index.csv")
        generated_paths.extend(output_dir.glob("frame_*.jpg"))
    if output_video_path is not None:
        generated_paths.append(output_video_path)

    existing_paths = [path for path in generated_paths if path.exists()]
    if existing_paths and not overwrite:
        preview = ", ".join(str(path) for path in existing_paths[:5])
        if len(existing_paths) > 5:
            preview += f", ... ({len(existing_paths)} files total)"
        raise FileExistsError(f"Generated output already exists: {preview}")

    if overwrite:
        for path in existing_paths:
            if path.is_file():
                path.unlink()


def _positive_finite_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def resolve_video_fps(
    requested_fps: Optional[float],
    session_dir: Optional[Path],
    video_path: Path,
) -> tuple[float, str]:
    if requested_fps is not None:
        fps = _positive_finite_float(requested_fps)
        if fps is None:
            raise ValueError(f"--video-fps must be a positive finite number, got {requested_fps!r}")
        return fps, "command_line"

    fps_context_dir = session_dir
    if fps_context_dir is None and (video_path.parent / "session.json").is_file():
        fps_context_dir = video_path.parent

    if fps_context_dir is not None:
        session_json = load_json_if_exists(fps_context_dir / "session.json")
        client_summary = session_json.get("client_summary", {})
        if isinstance(client_summary, dict):
            for field_name in ("actual_output_hz", "actual_capture_hz", "target_fps"):
                fps = _positive_finite_float(client_summary.get(field_name))
                if fps is not None:
                    return fps, f"session.json:client_summary.{field_name}"

        camera_json = load_json_if_exists(fps_context_dir / "camera.json")
        fps = _positive_finite_float(camera_json.get("target_fps"))
        if fps is not None:
            return fps, "camera.json:target_fps"

    return DEFAULT_VIDEO_FPS, "default"


def run_ffmpeg_decode_jpg(
    ffmpeg: str,
    video_path: Path,
    output_dir: Path,
    jpg_quality: int,
) -> None:
    output_pattern = str(output_dir / "frame_%06d.jpg")

    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-f",
        "hevc",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        "-start_number",
        "0",
        "-q:v",
        str(jpg_quality),
        output_pattern,
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg JPG decode failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


def run_ffmpeg_write_video(
    ffmpeg: str,
    video_path: Path,
    output_video_path: Path,
    video_fps: float,
    video_codec: str,
) -> Optional[int]:
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    base_command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-fflags",
        "+genpts",
        "-r",
        str(video_fps),
        "-f",
        "hevc",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
    ]

    if video_codec == "copy":
        codec_args = [
            "-c:v",
            "copy",
            "-tag:v",
            "hvc1",
            "-movflags",
            "+faststart",
        ]
    elif video_codec == "h264":
        codec_args = [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    else:
        raise ValueError(f"Unsupported video codec mode: {video_codec}")

    container_args = [
        "-video_track_timescale",
        "90000",
        "-progress",
        "pipe:1",
        "-nostats",
    ]
    command = base_command + codec_args + container_args + [str(output_video_path)]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg MP4 output failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    if not output_video_path.is_file() or output_video_path.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg completed without creating a non-empty MP4: {output_video_path}")

    progress_frames = []
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "frame":
            try:
                progress_frames.append(int(value.strip()))
            except ValueError:
                continue
    return progress_frames[-1] if progress_frames else None


def read_metadata_rows(metadata_path: Optional[Path]) -> list[dict[str, str]]:
    if metadata_path is None:
        return []

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_nonnegative_int(value: object) -> Optional[int]:
    parsed = _optional_int(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _json_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def resolve_network_log_path(
    session_dir: Optional[Path],
    video_path: Path,
) -> Optional[Path]:
    candidates = [video_path.parent / "network_log.jsonl"]
    if session_dir is not None:
        candidates.append(session_dir / "network_log.jsonl")

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    return None


def read_encoded_frame_indices(network_log_path: Optional[Path]) -> Optional[list[int]]:
    """Return encoded display-frame indices, excluding codec config and partial samples."""
    if network_log_path is None or not network_log_path.is_file():
        return None

    completed_frames: dict[int, Optional[int]] = {}
    frame_index_by_presentation_time: dict[int, int] = {}
    malformed_line_number: Optional[int] = None
    with network_log_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if malformed_line_number is not None:
                raise ValueError(
                    "network_log.jsonl contains malformed JSON before its final "
                    f"record (line {malformed_line_number})"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A capture interrupted while writing its final log line can leave a
                # truncated tail. Only that final non-empty record is safe to ignore.
                malformed_line_number = line_number
                continue
            if not isinstance(record, dict) or record.get("event") != "hevc_sample":
                continue

            flags = _optional_nonnegative_int(record.get("flags")) or 0
            is_codec_config = _json_flag(record.get("is_codec_config")) or bool(
                flags & MEDIA_CODEC_BUFFER_FLAG_CODEC_CONFIG
            )
            is_partial_frame = _json_flag(record.get("is_partial_frame")) or bool(
                flags & MEDIA_CODEC_BUFFER_FLAG_PARTIAL_FRAME
            )
            if is_codec_config or is_partial_frame:
                continue

            frame_index = _optional_nonnegative_int(record.get("frame_index"))
            if frame_index is not None:
                presentation_time_us = _optional_nonnegative_int(
                    record.get("presentation_time_us")
                )
                existing_presentation_time = completed_frames.get(frame_index)
                if (
                    frame_index in completed_frames
                    and existing_presentation_time is not None
                    and presentation_time_us is not None
                    and existing_presentation_time != presentation_time_us
                ):
                    raise ValueError(
                        "network_log.jsonl maps frame_index "
                        f"{frame_index} to multiple presentation timestamps"
                    )
                if presentation_time_us is not None:
                    existing_frame_index = frame_index_by_presentation_time.get(
                        presentation_time_us
                    )
                    if existing_frame_index is not None and existing_frame_index != frame_index:
                        raise ValueError(
                            "network_log.jsonl maps presentation_time_us "
                            f"{presentation_time_us} to multiple frame indices"
                        )
                    frame_index_by_presentation_time[presentation_time_us] = frame_index
                if frame_index not in completed_frames:
                    completed_frames[frame_index] = presentation_time_us
                elif (
                    completed_frames[frame_index] is None
                    and presentation_time_us is not None
                ):
                    completed_frames[frame_index] = presentation_time_us

    if not completed_frames:
        return None
    if all(presentation_time is not None for presentation_time in completed_frames.values()):
        return [
            frame_index
            for frame_index, _presentation_time in sorted(
                completed_frames.items(), key=lambda item: item[1]
            )
        ]
    return sorted(completed_frames)


def metadata_rows_by_frame_index(
    metadata_rows: list[dict[str, str]],
) -> dict[int, tuple[int, dict[str, str]]]:
    indexed: dict[int, tuple[int, dict[str, str]]] = {}
    for row_number, metadata in enumerate(metadata_rows):
        frame_index = _optional_nonnegative_int(metadata.get("frame_index"))
        if frame_index is not None:
            indexed.setdefault(frame_index, (row_number, metadata))
    return indexed


def duplicate_metadata_frame_indices(metadata_rows: list[dict[str, str]]) -> list[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for metadata in metadata_rows:
        frame_index = _optional_nonnegative_int(metadata.get("frame_index"))
        if frame_index is None:
            continue
        if frame_index in seen:
            duplicates.add(frame_index)
        seen.add(frame_index)
    return sorted(duplicates)


def metadata_encoder_submission_state(metadata: dict[str, str]) -> str:
    """Classify whether a metadata row was actually handed to MediaCodec.

    New schema rows carry encoder_presentation_time_us, which is assigned just
    before EncodeFrame. Older rows expose the encoder input path instead. A
    failed camera acquisition is explicit evidence that no encoder submission
    occurred. Conflicting or incomplete evidence stays unknown so a real video
    loss is never reclassified as an invalid capture.
    """
    submitted = False
    not_submitted = False

    if "encoder_presentation_time_us" in metadata:
        presentation_time_us = _optional_nonnegative_int(
            metadata.get("encoder_presentation_time_us")
        )
        if presentation_time_us is not None:
            submitted = presentation_time_us > 0
            not_submitted = presentation_time_us == 0

    input_path = (metadata.get("encoder_input_path") or "").strip().lower()
    if input_path in {"direct_byte_buffer", "java_byte_array"} or input_path.endswith(
        "_failed"
    ):
        submitted = True
    elif input_path in {"not_encoded", "scheduler_only"}:
        not_submitted = True
    elif "encoder_input_path" in metadata and not input_path:
        # Older schemas have no encoder PTS. An empty, present path means the
        # frame was rejected before either encoder entrypoint was attempted.
        not_submitted = True

    capture_result = _optional_int(metadata.get("capture_result"))
    if capture_result is not None and capture_result != 0:
        not_submitted = True

    if submitted and not_submitted:
        return "conflicting"
    if submitted:
        return "submitted"
    if not_submitted:
        return "not_submitted"
    return "unknown"


def analyze_metadata_frame_alignment(
    metadata_rows: list[dict[str, str]],
    encoded_frame_indices: list[int],
) -> dict[str, list[int]]:
    metadata_by_frame = metadata_rows_by_frame_index(metadata_rows)
    encoded_set = set(encoded_frame_indices)
    result = {
        "missing_metadata": [
            frame_index
            for frame_index in encoded_frame_indices
            if frame_index not in metadata_by_frame
        ],
        "allowed_unsubmitted": [],
        "submitted_without_hevc": [],
        "ambiguous_without_hevc": [],
        "metadata_contradicts_hevc": [],
        "invalid_frame_index_rows": [],
    }

    for row_number, metadata in enumerate(metadata_rows):
        frame_index = _optional_nonnegative_int(metadata.get("frame_index"))
        if frame_index is None:
            result["invalid_frame_index_rows"].append(row_number)
            continue

        submission_state = metadata_encoder_submission_state(metadata)
        if frame_index in encoded_set:
            if submission_state in {"not_submitted", "conflicting"}:
                result["metadata_contradicts_hevc"].append(frame_index)
            continue

        if submission_state == "not_submitted":
            result["allowed_unsubmitted"].append(frame_index)
        elif submission_state == "submitted":
            result["submitted_without_hevc"].append(frame_index)
        else:
            result["ambiguous_without_hevc"].append(frame_index)

    return result


def build_alignment_summary(
    metadata_rows: Optional[list[dict[str, str]]],
    metadata_row_count: int,
    encoded_frame_indices: Optional[list[int]],
    network_log_path: Optional[Path],
) -> dict[str, object]:
    if encoded_frame_indices is None:
        return {
            "metadata_alignment_mode": "metadata_row_order",
            "network_log_jsonl": str(network_log_path) if network_log_path else "",
            "encoded_frame_index_count": None,
            "metadata_rows_matched_to_encoded_frames": None,
            "metadata_rows_without_encoded_frames": None,
            "encoded_frames_without_metadata": None,
            "encoded_frame_indices_without_metadata_preview": [],
            "duplicate_metadata_frame_index_count": None,
            "duplicate_metadata_frame_indices_preview": [],
            "allowed_unsubmitted_metadata_row_count": None,
            "allowed_unsubmitted_frame_indices_preview": [],
            "submitted_metadata_rows_without_hevc_count": None,
            "submitted_metadata_frame_indices_without_hevc_preview": [],
            "ambiguous_metadata_rows_without_hevc_count": None,
        }

    summary: dict[str, object] = {
        "metadata_alignment_mode": "network_log.frame_index",
        "network_log_jsonl": str(network_log_path) if network_log_path else "",
        "encoded_frame_index_count": len(encoded_frame_indices),
        "metadata_rows_matched_to_encoded_frames": None,
        "metadata_rows_without_encoded_frames": None,
        "encoded_frames_without_metadata": None,
        "encoded_frame_indices_without_metadata_preview": [],
        "duplicate_metadata_frame_index_count": None,
        "duplicate_metadata_frame_indices_preview": [],
    }
    if metadata_rows is None:
        return summary

    duplicate_frame_indices = duplicate_metadata_frame_indices(metadata_rows)
    analysis = analyze_metadata_frame_alignment(metadata_rows, encoded_frame_indices)
    missing_metadata = analysis["missing_metadata"]
    matched_count = len(encoded_frame_indices) - len(missing_metadata)
    summary.update(
        {
            "metadata_rows_matched_to_encoded_frames": matched_count,
            "metadata_rows_without_encoded_frames": max(0, metadata_row_count - matched_count),
            "encoded_frames_without_metadata": len(missing_metadata),
            "encoded_frame_indices_without_metadata_preview": missing_metadata[:20],
            "duplicate_metadata_frame_index_count": len(duplicate_frame_indices),
            "duplicate_metadata_frame_indices_preview": duplicate_frame_indices[:20],
            "allowed_unsubmitted_metadata_row_count": len(
                analysis["allowed_unsubmitted"]
            ),
            "allowed_unsubmitted_frame_indices_preview": analysis[
                "allowed_unsubmitted"
            ][:20],
            "submitted_metadata_rows_without_hevc_count": len(
                analysis["submitted_without_hevc"]
            ),
            "submitted_metadata_frame_indices_without_hevc_preview": analysis[
                "submitted_without_hevc"
            ][:20],
            "ambiguous_metadata_rows_without_hevc_count": len(
                analysis["ambiguous_without_hevc"]
            ),
        }
    )
    return summary


def resolve_expected_video_frame_count(
    session_dir: Optional[Path],
    metadata_row_count: int,
    network_log_frame_count: Optional[int] = None,
) -> tuple[Optional[int], str]:
    if session_dir is not None:
        session_json = load_json_if_exists(session_dir / "session.json")
        client_summary = session_json.get("client_summary", {})
        if isinstance(client_summary, dict):
            for field_name in ("encoder_matched_output_frame_count",):
                try:
                    frame_count = int(client_summary.get(field_name, 0))
                except (TypeError, ValueError):
                    frame_count = 0
                if frame_count > 0:
                    return frame_count, f"session.json:client_summary.{field_name}"

        if isinstance(client_summary, dict):
            for field_name in (
                "encoded_frame_count",
                "encoder_submitted_input_frame_count",
            ):
                try:
                    frame_count = int(client_summary.get(field_name, 0))
                except (TypeError, ValueError):
                    frame_count = 0
                if frame_count > 0:
                    return frame_count, f"session.json:client_summary.{field_name}"

        if network_log_frame_count is not None and network_log_frame_count > 0:
            return network_log_frame_count, "network_log.jsonl:hevc_frame_index"

        if isinstance(client_summary, dict):
            try:
                frame_count = int(client_summary.get("output_frame_count", 0))
            except (TypeError, ValueError):
                frame_count = 0
            if frame_count > 0:
                return frame_count, "session.json:client_summary.output_frame_count"

    elif network_log_frame_count is not None and network_log_frame_count > 0:
        return network_log_frame_count, "network_log.jsonl:hevc_frame_index"

    if session_dir is not None:
        try:
            frame_count = int(session_json.get("metadata_rows", 0))
        except (TypeError, ValueError):
            frame_count = 0
        if frame_count > 0:
            return frame_count, "session.json:metadata_rows"

    if metadata_row_count > 0:
        return metadata_row_count, "metadata.csv"

    return None, ""


def decoded_frame_alignment_error(
    decoded_frame_count: int,
    metadata_row_count: int,
    encoded_frame_indices: Optional[list[int]],
    metadata_available: bool = False,
) -> str:
    if encoded_frame_indices is not None:
        if decoded_frame_count != len(encoded_frame_indices):
            return (
                f"decoded JPG count {decoded_frame_count} != encoded frame index count "
                f"{len(encoded_frame_indices)} from network_log.jsonl"
            )
        return ""

    if (
        (metadata_available or metadata_row_count > 0)
        and decoded_frame_count != metadata_row_count
    ):
        return (
            f"decoded JPG count {decoded_frame_count} != metadata row count {metadata_row_count}; "
            "network_log.jsonl is unavailable, so exact frame_index alignment cannot be verified"
        )
    return ""


def metadata_frame_alignment_error(
    metadata_rows: list[dict[str, str]],
    encoded_frame_indices: Optional[list[int]],
) -> str:
    if encoded_frame_indices is None:
        return ""

    duplicate_frame_indices = duplicate_metadata_frame_indices(metadata_rows)
    if duplicate_frame_indices:
        return (
            "metadata.csv contains duplicate frame_index values: "
            + ", ".join(str(index) for index in duplicate_frame_indices[:20])
        )

    analysis = analyze_metadata_frame_alignment(metadata_rows, encoded_frame_indices)
    errors = []
    if analysis["invalid_frame_index_rows"]:
        errors.append(
            "metadata.csv has missing/invalid frame_index at data rows: "
            + ", ".join(
                str(index) for index in analysis["invalid_frame_index_rows"][:20]
            )
        )
    if analysis["missing_metadata"]:
        errors.append(
            "encoded frames have no matching metadata.csv row for frame_index: "
            + ", ".join(str(index) for index in analysis["missing_metadata"][:20])
        )
    if analysis["submitted_without_hevc"]:
        errors.append(
            "metadata.csv says frames were submitted to the encoder but "
            "network_log.jsonl has no completed HEVC frame for frame_index: "
            + ", ".join(
                str(index) for index in analysis["submitted_without_hevc"][:20]
            )
        )
    if analysis["ambiguous_without_hevc"]:
        errors.append(
            "metadata.csv rows missing from network_log.jsonl cannot be proven "
            "unsubmitted for frame_index: "
            + ", ".join(
                str(index) for index in analysis["ambiguous_without_hevc"][:20]
            )
        )
    if analysis["metadata_contradicts_hevc"]:
        errors.append(
            "network_log.jsonl contains HEVC frames that metadata.csv marks as "
            "not submitted/conflicting for frame_index: "
            + ", ".join(
                str(index) for index in analysis["metadata_contradicts_hevc"][:20]
            )
        )
    return "; ".join(errors)


def client_encoder_count_alignment_error(
    session_dir: Optional[Path],
    network_log_frame_count: Optional[int],
) -> str:
    if session_dir is None:
        return ""
    session_json = load_json_if_exists(session_dir / "session.json")
    client_summary = session_json.get("client_summary", {})
    if not isinstance(client_summary, dict):
        return ""

    counts: dict[str, int] = {}
    for field_name in (
        "encoder_matched_output_frame_count",
        "encoded_frame_count",
        "encoder_submitted_input_frame_count",
    ):
        frame_count = _optional_nonnegative_int(client_summary.get(field_name))
        if field_name in client_summary and frame_count is not None:
            counts[field_name] = frame_count

    if len(set(counts.values())) > 1:
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        return f"session.json client encoder frame counts disagree: {details}"

    if network_log_frame_count is not None:
        mismatches = {
            name: count
            for name, count in counts.items()
            if count != network_log_frame_count
        }
        if mismatches:
            details = ", ".join(
                f"{name}={count}" for name, count in mismatches.items()
            )
            return (
                "network_log.jsonl completed HEVC frame count "
                f"{network_log_frame_count} disagrees with session.json: {details}"
            )
    return ""


def network_log_alignment_error(
    network_log_path: Optional[Path],
    encoded_frame_indices: Optional[list[int]],
) -> str:
    if network_log_path is not None and encoded_frame_indices is None:
        return (
            "network_log.jsonl exists but contains no complete HEVC frame mapping; "
            "refusing unsafe metadata row-order alignment"
        )
    return ""


def write_index_csv(
    output_dir: Path,
    image_paths: list[Path],
    metadata_rows: list[dict[str, str]],
    encoded_frame_indices: Optional[list[int]] = None,
) -> Path:
    index_path = output_dir / "decoded_index.csv"
    metadata_by_frame = metadata_rows_by_frame_index(metadata_rows)
    uses_frame_index_alignment = encoded_frame_indices is not None

    fieldnames = [
        "decoded_frame_number",
        "image_file",
        "frame_index",
        "metadata_row_index",
        "alignment_source",
        "ref_timestamp_us",
        "pico_frame_timestamp_ns",
        "metadata_row_available",
    ]

    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for decoded_index, image_path in enumerate(image_paths):
            metadata_row_index: Optional[int] = None
            if uses_frame_index_alignment:
                frame_index = (
                    encoded_frame_indices[decoded_index]
                    if decoded_index < len(encoded_frame_indices)
                    else None
                )
                metadata_entry = metadata_by_frame.get(frame_index) if frame_index is not None else None
                if metadata_entry is None:
                    metadata = {}
                else:
                    metadata_row_index, metadata = metadata_entry
                alignment_source = "network_log.frame_index"
            else:
                metadata_row_index = decoded_index if decoded_index < len(metadata_rows) else None
                metadata = metadata_rows[decoded_index] if metadata_row_index is not None else {}
                frame_index = _optional_nonnegative_int(metadata.get("frame_index"))
                alignment_source = "metadata_row_order"

            writer.writerow(
                {
                    "decoded_frame_number": decoded_index,
                    "image_file": image_path.name,
                    "frame_index": frame_index if frame_index is not None else "",
                    "metadata_row_index": (
                        metadata_row_index if metadata_row_index is not None else ""
                    ),
                    "alignment_source": alignment_source,
                    "ref_timestamp_us": metadata.get("ref_timestamp_us", ""),
                    "pico_frame_timestamp_ns": metadata.get("frame_timestamp_ns", ""),
                    "metadata_row_available": "true" if metadata else "false",
                }
            )

    return index_path


def load_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_summary(
    session_dir: Optional[Path],
    video_path: Path,
    metadata_path: Optional[Path],
    output_dir: Path,
    output_video_path: Optional[Path],
    image_paths: Optional[Iterable[Path]],
    metadata_row_count: int,
    index_path: Optional[Path],
    jpg_quality: int,
    video_fps: float,
    video_fps_source: str,
    video_codec: str,
    expected_video_frame_count: Optional[int],
    expected_video_frame_count_source: str,
    muxed_video_frame_count: Optional[int],
    alignment_summary: dict[str, object],
) -> Path:
    image_paths_list = list(image_paths) if image_paths is not None else None
    session_json = load_json_if_exists(session_dir / "session.json") if session_dir else {}

    summary = {
        "session_dir": str(session_dir) if session_dir else "",
        "video_h265": str(video_path),
        "metadata_csv": str(metadata_path) if metadata_path else "",
        "output_dir": str(output_dir),
        "output_video_mp4": str(output_video_path) if output_video_path else "",
        "output_video_exists": output_video_path.is_file() if output_video_path else False,
        "output_video_bytes": (
            output_video_path.stat().st_size
            if output_video_path and output_video_path.is_file()
            else 0
        ),
        "jpg_output_enabled": image_paths_list is not None,
        "decoded_index_csv": str(index_path) if index_path else "",
        "jpg_quality_qv": jpg_quality,
        "video_fps": video_fps,
        "video_fps_source": video_fps_source,
        "video_codec": video_codec,
        "expected_video_frame_count": expected_video_frame_count,
        "expected_video_frame_count_source": expected_video_frame_count_source,
        "muxed_video_frame_count": muxed_video_frame_count,
        "video_frame_count_matches_expected": (
            muxed_video_frame_count == expected_video_frame_count
            if muxed_video_frame_count is not None and expected_video_frame_count is not None
            else None
        ),
        "expected_video_duration_seconds": (
            expected_video_frame_count / video_fps
            if expected_video_frame_count is not None
            else None
        ),
        "decoded_frame_count": len(image_paths_list) if image_paths_list is not None else None,
        "metadata_row_count": metadata_row_count,
        "frame_count_matches_metadata": (
            len(image_paths_list) == metadata_row_count
            if image_paths_list is not None
            and metadata_row_count
            and alignment_summary.get("metadata_alignment_mode") == "metadata_row_order"
            else None
        ),
        **alignment_summary,
        "session_json_client_summary": session_json.get("client_summary", {}),
    }

    summary_path = output_dir / "decode_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()

    try:
        if args.video_only and args.no_video:
            raise ValueError("--video-only and --no-video cannot be used together; no output would be produced.")

        session_dir, video_path, metadata_path, output_dir, output_video_path = resolve_inputs(args)

        video_fps, video_fps_source = resolve_video_fps(
            args.video_fps,
            session_dir,
            video_path,
        )

        ffmpeg_exe = resolve_ffmpeg_executable(args.ffmpeg)

        prepare_outputs(
            output_dir=output_dir,
            output_video_path=output_video_path,
            decode_jpg=not args.video_only,
            overwrite=not args.no_overwrite,
        )

        if not args.video_only:
            run_ffmpeg_decode_jpg(
                ffmpeg=ffmpeg_exe,
                video_path=video_path,
                output_dir=output_dir,
                jpg_quality=args.jpg_quality,
            )

        muxed_video_frame_count = None
        if output_video_path is not None:
            muxed_video_frame_count = run_ffmpeg_write_video(
                ffmpeg=ffmpeg_exe,
                video_path=video_path,
                output_video_path=output_video_path,
                video_fps=video_fps,
                video_codec=args.video_codec,
            )

        metadata_rows = read_metadata_rows(metadata_path)
        metadata_row_count = len(metadata_rows)
        index_path = None
        if args.video_only:
            image_paths = None
        else:
            image_paths = sorted(output_dir.glob("frame_*.jpg"))

        network_log_path = resolve_network_log_path(session_dir, video_path)
        encoded_frame_indices = read_encoded_frame_indices(network_log_path)
        alignment_summary = build_alignment_summary(
            metadata_rows=metadata_rows,
            metadata_row_count=metadata_row_count,
            encoded_frame_indices=encoded_frame_indices,
            network_log_path=network_log_path,
        )

        expected_video_frame_count, expected_video_frame_count_source = (
            resolve_expected_video_frame_count(
                session_dir,
                metadata_row_count,
                network_log_frame_count=(
                    len(encoded_frame_indices) if encoded_frame_indices is not None else None
                ),
            )
        )

        validation_errors = []
        log_alignment_error = network_log_alignment_error(
            network_log_path, encoded_frame_indices
        )
        if log_alignment_error:
            validation_errors.append(log_alignment_error)
        client_count_error = client_encoder_count_alignment_error(
            session_dir,
            len(encoded_frame_indices) if encoded_frame_indices is not None else None,
        )
        if client_count_error:
            validation_errors.append(client_count_error)
        if image_paths is not None:
            alignment_error = decoded_frame_alignment_error(
                len(image_paths),
                metadata_row_count,
                encoded_frame_indices,
                metadata_available=metadata_path is not None,
            )
            if alignment_error:
                validation_errors.append(alignment_error)
        if metadata_path is not None:
            metadata_alignment_error = metadata_frame_alignment_error(
                metadata_rows, encoded_frame_indices
            )
            if metadata_alignment_error:
                validation_errors.append(metadata_alignment_error)
        if (
            muxed_video_frame_count is not None
            and expected_video_frame_count is not None
            and muxed_video_frame_count != expected_video_frame_count
        ):
            validation_errors.append(
                "muxed MP4 frame count "
                f"{muxed_video_frame_count} != expected frame count {expected_video_frame_count}"
            )

        if image_paths is not None and not validation_errors:
            index_path = write_index_csv(
                output_dir,
                image_paths,
                metadata_rows,
                encoded_frame_indices=encoded_frame_indices,
            )

        summary_path = write_summary(
            session_dir=session_dir,
            video_path=video_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
            output_video_path=output_video_path,
            image_paths=image_paths,
            metadata_row_count=metadata_row_count,
            index_path=index_path,
            jpg_quality=args.jpg_quality,
            video_fps=video_fps,
            video_fps_source=video_fps_source,
            video_codec=args.video_codec,
            expected_video_frame_count=expected_video_frame_count,
            expected_video_frame_count_source=expected_video_frame_count_source,
            muxed_video_frame_count=muxed_video_frame_count,
            alignment_summary=alignment_summary,
        )

        if validation_errors:
            raise RuntimeError(
                "Frame alignment validation failed: " + "; ".join(validation_errors)
            )

    except Exception as exc:
        print(f"[decode_h265_to_jpg] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[decode_h265_to_jpg] video: {video_path}")
    print(f"[decode_h265_to_jpg] ffmpeg: {ffmpeg_exe}")
    print(f"[decode_h265_to_jpg] output dir: {output_dir}")

    if output_video_path is not None:
        print(f"[decode_h265_to_jpg] output video: {output_video_path}")
    else:
        print("[decode_h265_to_jpg] output video: disabled")

    print(f"[decode_h265_to_jpg] video fps: {video_fps} ({video_fps_source})")
    print(f"[decode_h265_to_jpg] video codec: {args.video_codec}")
    if muxed_video_frame_count is not None:
        print(
            "[decode_h265_to_jpg] muxed video frames: "
            f"{muxed_video_frame_count} (expected {expected_video_frame_count})"
        )
    if image_paths is not None:
        print(f"[decode_h265_to_jpg] decoded JPG frames: {len(image_paths)}")
        print(f"[decode_h265_to_jpg] index: {index_path}")
    else:
        print("[decode_h265_to_jpg] decoded JPG frames: skipped (--video-only)")
    print(f"[decode_h265_to_jpg] summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
