from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


decoder = load_module("stream_server_windows_decoder", "decode_h265_to_jpg.py")
sys.modules["decode_h265_to_jpg"] = decoder
gaze = load_module("stream_server_windows_gaze", "project_gaze_uv.py")
sys.modules["project_gaze_uv"] = gaze
fused = load_module("stream_server_windows_fused", "fused_gaze_pipeline.py")


def write_metadata(path: Path) -> list[dict[str, str]]:
    rows = [
        {
            "frame_index": str(frame_index),
            "ref_timestamp_us": str(10_000 + frame_index),
            "frame_timestamp_ns": str(20_000 + frame_index),
            "capture_result": "-1" if frame_index == 2 else "0",
            "encoder_input_path": "" if frame_index == 2 else "java_byte_array",
        }
        for frame_index in range(5)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_network_log(path: Path) -> None:
    records = [
        # Old captures could attach frame_index=0 to codec configuration data.
        {
            "event": "hevc_sample",
            "frame_index": 0,
            "presentation_time_us": 0,
            "is_codec_config": True,
            "flags": 2,
        },
        {
            "event": "hevc_sample",
            "frame_index": 4,
            "presentation_time_us": 400,
            "is_codec_config": False,
            "flags": 0,
        },
        {"event": "metadata_header"},
        {
            "event": "hevc_sample",
            "frame_index": 3,
            "presentation_time_us": 300,
            "is_codec_config": False,
            "is_partial_frame": True,
            "flags": 8,
        },
        {
            "event": "hevc_sample",
            "frame_index": 1,
            "presentation_time_us": 200,
            "is_codec_config": False,
            "flags": 0,
        },
        {
            "event": "hevc_sample",
            "frame_index": 0,
            "presentation_time_us": 100,
            "is_codec_config": False,
            "flags": 1,
        },
        {
            "event": "hevc_sample",
            "frame_index": 3,
            "presentation_time_us": 300,
            "is_codec_config": False,
            "is_partial_frame": False,
            "flags": 0,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.write("{truncated final json\n")


def make_session(tmp_path: Path) -> tuple[Path, list[dict[str, str]]]:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "video.h265").write_bytes(b"fake-hevc")
    rows = write_metadata(session_dir / "metadata.csv")
    write_network_log(session_dir / "network_log.jsonl")
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "metadata_rows": 5,
                "client_summary": {
                    "output_frame_count": 5,
                    "encoded_frame_count": 4,
                    "encoder_matched_output_frame_count": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    return session_dir, rows


def test_expected_count_prefers_encoder_output_over_metadata(tmp_path: Path) -> None:
    session_dir, _rows = make_session(tmp_path)

    assert decoder.resolve_expected_video_frame_count(session_dir, 5, 4) == (
        4,
        "session.json:client_summary.encoder_matched_output_frame_count",
    )


def test_network_log_returns_completed_frames_in_presentation_order(tmp_path: Path) -> None:
    session_dir, _rows = make_session(tmp_path)

    assert decoder.read_encoded_frame_indices(session_dir / "network_log.jsonl") == [0, 1, 3, 4]


def test_network_log_rejects_malformed_json_before_final_record(tmp_path: Path) -> None:
    network_log_path = tmp_path / "network_log.jsonl"
    network_log_path.write_text(
        "{malformed\n"
        + json.dumps(
            {
                "event": "hevc_sample",
                "frame_index": 0,
                "presentation_time_us": 100,
                "is_codec_config": False,
                "flags": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed JSON before its final record"):
        decoder.read_encoded_frame_indices(network_log_path)


def test_network_log_rejects_conflicting_pts_after_missing_pts(tmp_path: Path) -> None:
    network_log_path = tmp_path / "network_log.jsonl"
    records = [
        {"event": "hevc_sample", "frame_index": 0, "flags": 0},
        {
            "event": "hevc_sample",
            "frame_index": 0,
            "presentation_time_us": 100,
            "flags": 0,
        },
        {
            "event": "hevc_sample",
            "frame_index": 0,
            "presentation_time_us": 200,
            "flags": 0,
        },
    ]
    network_log_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple presentation timestamps"):
        decoder.read_encoded_frame_indices(network_log_path)


def test_index_uses_encoded_frame_index_instead_of_metadata_row_order(tmp_path: Path) -> None:
    session_dir, rows = make_session(tmp_path)
    output_dir = tmp_path / "decoded"
    output_dir.mkdir()
    image_paths = []
    for decoded_index in range(4):
        image_path = output_dir / f"frame_{decoded_index:06d}.jpg"
        image_path.write_bytes(b"")
        image_paths.append(image_path)

    index_path = decoder.write_index_csv(
        output_dir,
        image_paths,
        rows,
        encoded_frame_indices=[0, 1, 3, 4],
    )
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))

    assert [row["frame_index"] for row in index_rows] == ["0", "1", "3", "4"]
    assert index_rows[2]["metadata_row_index"] == "3"
    assert index_rows[2]["ref_timestamp_us"] == "10003"
    assert {row["alignment_source"] for row in index_rows} == {"network_log.frame_index"}

    image_index = gaze.build_decoded_image_index(output_dir)
    assert gaze.resolve_image_path(rows[2], 2, output_dir, image_index) is None
    assert gaze.resolve_image_path(rows[3], 3, output_dir, image_index) == image_paths[2]


def test_submitted_metadata_frame_missing_from_network_log_is_rejected(
    tmp_path: Path,
) -> None:
    _session_dir, rows = make_session(tmp_path)
    rows[2]["capture_result"] = "0"
    rows[2]["encoder_input_path"] = "java_byte_array"

    error = decoder.metadata_frame_alignment_error(rows, [0, 1, 3, 4])

    assert "submitted to the encoder" in error
    assert "frame_index: 2" in error


def test_new_schema_only_allows_explicitly_unsubmitted_extra_metadata() -> None:
    rows = [
        {
            "frame_index": "0",
            "capture_result": "0",
            "encoder_presentation_time_us": "1000",
        },
        {
            "frame_index": "1",
            "capture_result": "0",
            "encoder_presentation_time_us": "0",
        },
    ]
    assert decoder.metadata_frame_alignment_error(rows, [0]) == ""

    rows[1]["encoder_presentation_time_us"] = "1001"
    assert "submitted to the encoder" in decoder.metadata_frame_alignment_error(
        rows, [0]
    )


def test_old_schema_capture_success_but_blank_encoder_path_is_unsubmitted() -> None:
    rows = [
        {
            "frame_index": "0",
            "capture_result": "0",
            "encoder_input_path": "java_byte_array",
        },
        {
            "frame_index": "1",
            "capture_result": "0",
            "encoder_input_path": "",
        },
    ]

    assert decoder.metadata_frame_alignment_error(rows, [0]) == ""


def test_client_encoder_count_must_match_network_log(tmp_path: Path) -> None:
    session_dir, _rows = make_session(tmp_path)
    session_json_path = session_dir / "session.json"
    session_json = json.loads(session_json_path.read_text(encoding="utf-8"))
    del session_json["client_summary"]["encoder_matched_output_frame_count"]
    session_json["client_summary"]["encoded_frame_count"] = 5
    session_json["client_summary"]["encoder_submitted_input_frame_count"] = 5
    session_json_path.write_text(json.dumps(session_json), encoding="utf-8")

    assert decoder.resolve_expected_video_frame_count(session_dir, 5, 4) == (
        5,
        "session.json:client_summary.encoded_frame_count",
    )
    assert "completed HEVC frame count 4" in decoder.client_encoder_count_alignment_error(
        session_dir, 4
    )

    session_json["client_summary"]["encoded_frame_count"] = 4
    session_json["client_summary"]["encoder_submitted_input_frame_count"] = 4
    session_json["client_summary"]["encoder_matched_output_frame_count"] = 0
    session_json_path.write_text(json.dumps(session_json), encoding="utf-8")
    assert "encoder_matched_output_frame_count=0" in (
        decoder.client_encoder_count_alignment_error(session_dir, 4)
    )


@pytest.mark.parametrize(("muxed_frame_count", "expected_exit"), [(4, 0), (3, 1)])
def test_main_distinguishes_metadata_only_invalid_frame_from_video_loss(
    monkeypatch,
    tmp_path: Path,
    muxed_frame_count: int,
    expected_exit: int,
) -> None:
    session_dir, _rows = make_session(tmp_path)
    output_dir = tmp_path / "decoded"

    monkeypatch.setattr(
        decoder,
        "parse_args",
        lambda: argparse.Namespace(
            session_dir=session_dir,
            video=None,
            metadata=None,
            output_dir=output_dir,
            output_video=None,
            no_video=False,
            video_only=False,
            video_fps=None,
            video_codec="copy",
            ffmpeg="",
            jpg_quality=2,
            no_overwrite=False,
        ),
    )
    monkeypatch.setattr(decoder, "resolve_ffmpeg_executable", lambda _arg: "ffmpeg")

    def fake_decode(**kwargs) -> None:
        for decoded_index in range(4):
            (kwargs["output_dir"] / f"frame_{decoded_index:06d}.jpg").write_bytes(b"jpg")

    def fake_write_video(**kwargs) -> int:
        kwargs["output_video_path"].write_bytes(b"mp4")
        return muxed_frame_count

    monkeypatch.setattr(decoder, "run_ffmpeg_decode_jpg", fake_decode)
    monkeypatch.setattr(decoder, "run_ffmpeg_write_video", fake_write_video)

    assert decoder.main() == expected_exit

    summary = json.loads((output_dir / "decode_summary.json").read_text(encoding="utf-8"))
    assert summary["expected_video_frame_count"] == 4
    assert summary["muxed_video_frame_count"] == muxed_frame_count
    assert summary["video_frame_count_matches_expected"] is (muxed_frame_count == 4)
    assert summary["metadata_alignment_mode"] == "network_log.frame_index"
    assert summary["metadata_rows_matched_to_encoded_frames"] == 4
    assert summary["metadata_rows_without_encoded_frames"] == 1


def test_main_rejects_submitted_metadata_frame_missing_from_hevc(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    session_dir, rows = make_session(tmp_path)
    rows[2]["capture_result"] = "0"
    rows[2]["encoder_input_path"] = "java_byte_array"
    with (session_dir / "metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "decoded"
    monkeypatch.setattr(
        decoder,
        "parse_args",
        lambda: argparse.Namespace(
            session_dir=session_dir,
            video=None,
            metadata=None,
            output_dir=output_dir,
            output_video=None,
            no_video=True,
            video_only=False,
            video_fps=None,
            video_codec="copy",
            ffmpeg="",
            jpg_quality=2,
            no_overwrite=False,
        ),
    )
    monkeypatch.setattr(decoder, "resolve_ffmpeg_executable", lambda _arg: "ffmpeg")

    def fake_decode(**kwargs) -> None:
        for decoded_index in range(4):
            (kwargs["output_dir"] / f"frame_{decoded_index:06d}.jpg").write_bytes(
                b"jpg"
            )

    monkeypatch.setattr(decoder, "run_ffmpeg_decode_jpg", fake_decode)

    assert decoder.main() == 1
    assert "submitted to the encoder" in capsys.readouterr().err


def test_video_only_accepts_metadata_only_invalid_frame(monkeypatch, tmp_path: Path) -> None:
    session_dir, _rows = make_session(tmp_path)
    output_dir = tmp_path / "video_only"

    monkeypatch.setattr(
        decoder,
        "parse_args",
        lambda: argparse.Namespace(
            session_dir=session_dir,
            video=None,
            metadata=None,
            output_dir=output_dir,
            output_video=None,
            no_video=False,
            video_only=True,
            video_fps=None,
            video_codec="copy",
            ffmpeg="",
            jpg_quality=2,
            no_overwrite=False,
        ),
    )
    monkeypatch.setattr(decoder, "resolve_ffmpeg_executable", lambda _arg: "ffmpeg")
    monkeypatch.setattr(
        decoder,
        "run_ffmpeg_decode_jpg",
        lambda **_kwargs: pytest.fail("--video-only must not decode JPG files"),
    )

    def fake_write_video(**kwargs) -> int:
        kwargs["output_video_path"].write_bytes(b"mp4")
        return 4

    monkeypatch.setattr(decoder, "run_ffmpeg_write_video", fake_write_video)

    assert decoder.main() == 0
    summary = json.loads((output_dir / "decode_summary.json").read_text(encoding="utf-8"))
    assert summary["metadata_row_count"] == 5
    assert summary["encoded_frame_index_count"] == 4
    assert summary["metadata_rows_without_encoded_frames"] == 1
    assert summary["video_frame_count_matches_expected"] is True


def test_legacy_sequence_fails_when_frame_counts_differ(monkeypatch, tmp_path: Path, capsys) -> None:
    session_dir, _rows = make_session(tmp_path)
    (session_dir / "network_log.jsonl").unlink()
    output_dir = tmp_path / "decoded"

    monkeypatch.setattr(
        decoder,
        "parse_args",
        lambda: argparse.Namespace(
            session_dir=session_dir,
            video=None,
            metadata=None,
            output_dir=output_dir,
            output_video=None,
            no_video=True,
            video_only=False,
            video_fps=None,
            video_codec="copy",
            ffmpeg="",
            jpg_quality=2,
            no_overwrite=False,
        ),
    )
    monkeypatch.setattr(decoder, "resolve_ffmpeg_executable", lambda _arg: "ffmpeg")

    def fake_decode(**kwargs) -> None:
        for decoded_index in range(4):
            (kwargs["output_dir"] / f"frame_{decoded_index:06d}.jpg").write_bytes(b"jpg")

    monkeypatch.setattr(decoder, "run_ffmpeg_decode_jpg", fake_decode)

    assert decoder.main() == 1
    assert "exact frame_index alignment cannot be verified" in capsys.readouterr().err


def test_standalone_video_with_network_log_does_not_require_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video.h265"
    video_path.write_bytes(b"fake-hevc")
    write_network_log(tmp_path / "network_log.jsonl")
    output_dir = tmp_path / "decoded"
    monkeypatch.setattr(
        decoder,
        "parse_args",
        lambda: argparse.Namespace(
            session_dir=None,
            video=video_path,
            metadata=None,
            output_dir=output_dir,
            output_video=None,
            no_video=True,
            video_only=False,
            video_fps=None,
            video_codec="copy",
            ffmpeg="",
            jpg_quality=2,
            no_overwrite=False,
        ),
    )
    monkeypatch.setattr(decoder, "resolve_ffmpeg_executable", lambda _arg: "ffmpeg")

    def fake_decode(**kwargs) -> None:
        for decoded_index in range(4):
            (kwargs["output_dir"] / f"frame_{decoded_index:06d}.jpg").write_bytes(
                b"jpg"
            )

    monkeypatch.setattr(decoder, "run_ffmpeg_decode_jpg", fake_decode)

    assert decoder.main() == 0
    with (output_dir / "decoded_index.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        index_rows = list(csv.DictReader(handle))
    assert [row["frame_index"] for row in index_rows] == ["0", "1", "3", "4"]
    assert {row["metadata_row_available"] for row in index_rows} == {"false"}


def test_existing_decoded_dir_rebuilds_exact_index(tmp_path: Path) -> None:
    session_dir, _rows = make_session(tmp_path)
    # Reusing already decoded images must still work after the raw video is archived.
    (session_dir / "video.h265").unlink()
    decoded_dir = tmp_path / "existing_decoded"
    decoded_dir.mkdir()
    for decoded_index in range(4):
        (decoded_dir / f"frame_{decoded_index:06d}.jpg").write_bytes(b"jpg")

    result_dir, should_cleanup = fused.decode_session_to_images(
        argparse.Namespace(decoded_dir=decoded_dir, debug=False, debug_output_raw=False),
        session_dir,
        tmp_path / "output",
        "unused-ffmpeg",
    )

    assert result_dir == decoded_dir.resolve()
    assert should_cleanup is False
    with (decoded_dir / "decoded_index.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["frame_index"] for row in rows] == ["0", "1", "3", "4"]
    assert {row["alignment_source"] for row in rows} == {"network_log.frame_index"}


def test_existing_decoded_dir_rejects_client_count_mismatch(tmp_path: Path) -> None:
    session_dir, _rows = make_session(tmp_path)
    session_json_path = session_dir / "session.json"
    session_json = json.loads(session_json_path.read_text(encoding="utf-8"))
    session_json["client_summary"]["encoded_frame_count"] = 5
    session_json_path.write_text(json.dumps(session_json), encoding="utf-8")
    decoded_dir = tmp_path / "existing_decoded"
    decoded_dir.mkdir()
    for decoded_index in range(4):
        (decoded_dir / f"frame_{decoded_index:06d}.jpg").write_bytes(b"jpg")

    with pytest.raises(RuntimeError, match="client encoder frame counts disagree"):
        fused.decode_session_to_images(
            argparse.Namespace(
                decoded_dir=decoded_dir,
                debug=False,
                debug_output_raw=False,
            ),
            session_dir,
            tmp_path / "output",
            "unused-ffmpeg",
        )
