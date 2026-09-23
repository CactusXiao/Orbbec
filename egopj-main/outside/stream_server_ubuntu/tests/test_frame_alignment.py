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


decoder = load_module("stream_server_ubuntu_decoder", "decode_h265_to_jpg.py")
sys.modules["decode_h265_to_jpg"] = decoder
gaze = load_module("stream_server_ubuntu_gaze", "project_gaze_uv.py")
sys.modules["project_gaze_uv"] = gaze
fused = load_module("stream_server_ubuntu_fused", "fused_gaze_pipeline.py")


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
            "is_codec_config": False,
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
            "is_partial_frame": False,
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
            "is_partial_frame": False,
            "flags": 0,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        # Interrupted sessions can leave one incomplete record at EOF.
        handle.write("{truncated json\n")


def make_session(tmp_path: Path) -> tuple[Path, list[dict[str, str]]]:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "video.h265").write_bytes(b"fake-hevc")
    rows = write_metadata(session_dir / "metadata.csv")
    write_network_log(session_dir / "network_log.jsonl")
    return session_dir, rows


def make_images(output_dir: Path, count: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for decoded_index in range(count):
        image_path = output_dir / f"frame_{decoded_index:06d}.jpg"
        image_path.write_bytes(b"jpg")
        image_paths.append(image_path)
    return image_paths


def test_network_log_returns_completed_frames_in_presentation_order(
    tmp_path: Path,
) -> None:
    session_dir, _rows = make_session(tmp_path)

    assert decoder.read_encoded_frame_indices(session_dir / "network_log.jsonl") == [
        0,
        1,
        3,
        4,
    ]


def test_network_log_falls_back_to_frame_index_for_mixed_pts(tmp_path: Path) -> None:
    log_path = tmp_path / "network_log.jsonl"
    records = [
        {"event": "hevc_sample", "frame_index": 9, "flags": 0},
        {
            "event": "hevc_sample",
            "frame_index": 2,
            "presentation_time_us": 200,
            "flags": 0,
        },
    ]
    log_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert decoder.read_encoded_frame_indices(log_path) == [2, 9]


def test_network_log_rejects_conflicting_frame_mapping(tmp_path: Path) -> None:
    log_path = tmp_path / "network_log.jsonl"
    records = [
        {
            "event": "hevc_sample",
            "frame_index": 1,
            "presentation_time_us": 100,
            "flags": 0,
        },
        {
            "event": "hevc_sample",
            "frame_index": 1,
            "presentation_time_us": 200,
            "flags": 0,
        },
    ]
    log_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple presentation timestamps"):
        decoder.read_encoded_frame_indices(log_path)


def test_network_log_rejects_conflicting_pts_after_missing_pts(tmp_path: Path) -> None:
    log_path = tmp_path / "network_log.jsonl"
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
    log_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple presentation timestamps"):
        decoder.read_encoded_frame_indices(log_path)


def test_index_uses_encoded_frame_index_instead_of_metadata_row_order(
    tmp_path: Path,
) -> None:
    _session_dir, rows = make_session(tmp_path)
    output_dir = tmp_path / "decoded"
    image_paths = make_images(output_dir, 4)

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
    assert {row["alignment_source"] for row in index_rows} == {
        "network_log.frame_index"
    }

    image_index = gaze.build_decoded_image_index(output_dir)
    assert gaze.resolve_image_path(rows[2], 2, output_dir, image_index) is None
    assert gaze.resolve_image_path(rows[3], 3, output_dir, image_index) == image_paths[2]


def test_submitted_metadata_frame_missing_from_network_log_is_rejected(
    tmp_path: Path,
) -> None:
    _session_dir, rows = make_session(tmp_path)
    rows[2]["capture_result"] = "0"
    rows[2]["encoder_input_path"] = "java_byte_array"
    image_paths = make_images(tmp_path / "decoded", 4)

    with pytest.raises(ValueError, match="submitted to the encoder"):
        decoder.validate_frame_alignment(
            image_paths,
            rows,
            [0, 1, 3, 4],
        )


def test_new_schema_only_allows_explicitly_unsubmitted_extra_metadata(
    tmp_path: Path,
) -> None:
    image_paths = make_images(tmp_path / "decoded", 1)
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
    decoder.validate_frame_alignment(image_paths, rows, [0])

    rows[1]["encoder_presentation_time_us"] = "1001"
    with pytest.raises(ValueError, match="submitted to the encoder"):
        decoder.validate_frame_alignment(image_paths, rows, [0])


def test_old_schema_capture_success_but_blank_encoder_path_is_unsubmitted(
    tmp_path: Path,
) -> None:
    image_paths = make_images(tmp_path / "decoded", 1)
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

    decoder.validate_frame_alignment(image_paths, rows, [0])


def test_client_encoder_count_must_match_network_log(tmp_path: Path) -> None:
    session_dir, _rows = make_session(tmp_path)
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "client_summary": {
                    "encoded_frame_count": 5,
                    "encoder_submitted_input_frame_count": 5,
                }
            }
        ),
        encoding="utf-8",
    )

    assert "completed HEVC frame count 4" in decoder.client_encoder_count_alignment_error(
        session_dir, 4
    )

    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "client_summary": {
                    "encoder_matched_output_frame_count": 0,
                    "encoded_frame_count": 4,
                    "encoder_submitted_input_frame_count": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    assert "encoder_matched_output_frame_count=0" in (
        decoder.client_encoder_count_alignment_error(session_dir, 4)
    )


@pytest.mark.parametrize(
    ("image_count", "message"),
    [
        (3, "encoded frame index count 4"),
        (5, "encoded frame index count 4"),
    ],
)
def test_exact_index_validates_decoded_frame_count(
    tmp_path: Path,
    image_count: int,
    message: str,
) -> None:
    _session_dir, rows = make_session(tmp_path)
    output_dir = tmp_path / "decoded"
    image_paths = make_images(output_dir, image_count)

    with pytest.raises(ValueError, match=message):
        decoder.write_index_csv(
            output_dir,
            image_paths,
            rows,
            encoded_frame_indices=[0, 1, 3, 4],
        )


def test_legacy_row_alignment_rejects_count_mismatch(tmp_path: Path) -> None:
    _session_dir, rows = make_session(tmp_path)
    output_dir = tmp_path / "decoded"
    image_paths = make_images(output_dir, 4)

    with pytest.raises(ValueError, match="exact frame_index alignment cannot be verified"):
        decoder.write_index_csv(output_dir, image_paths, rows)


def test_standalone_video_with_network_log_does_not_require_metadata(
    monkeypatch: pytest.MonkeyPatch,
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
            video_fps=30.0,
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
        lambda **kwargs: make_images(kwargs["output_dir"], 4),
    )

    assert decoder.main() == 0
    with (output_dir / "decoded_index.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        index_rows = list(csv.DictReader(handle))
    assert [row["frame_index"] for row in index_rows] == ["0", "1", "3", "4"]
    assert {row["metadata_row_available"] for row in index_rows} == {"false"}


def test_decoder_main_accepts_metadata_only_invalid_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
            no_video=True,
            video_fps=30.0,
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
        lambda **kwargs: make_images(kwargs["output_dir"], 4),
    )

    assert decoder.main() == 0
    summary = json.loads((output_dir / "decode_summary.json").read_text(encoding="utf-8"))
    assert summary["metadata_alignment_mode"] == "network_log.frame_index"
    assert summary["encoded_frame_index_count"] == 4
    assert summary["metadata_rows_matched_to_encoded_frames"] == 4
    assert summary["metadata_rows_without_encoded_frames"] == 1


def test_decoder_main_rejects_submitted_metadata_frame_missing_from_hevc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
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
            video_fps=30.0,
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
        lambda **kwargs: make_images(kwargs["output_dir"], 4),
    )

    assert decoder.main() == 1
    assert "submitted to the encoder" in capsys.readouterr().err


def test_existing_decoded_dir_rebuilds_exact_index(tmp_path: Path) -> None:
    session_dir, _rows = make_session(tmp_path)
    decoded_dir = tmp_path / "existing_decoded"
    make_images(decoded_dir, 4)
    (decoded_dir / "decoded_index.csv").write_text("stale\n", encoding="utf-8")

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
    assert {row["alignment_source"] for row in rows} == {
        "network_log.frame_index"
    }


def test_project_visualization_entrypoint_validates_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_dir, _rows = make_session(tmp_path)
    camera_path = session_dir / "camera.json"
    calibration_path = tmp_path / "calibration.npz"
    decoded_dir = session_dir / "decoded_jpg"
    camera_path.write_text("{}", encoding="utf-8")
    calibration_path.write_bytes(b"npz")
    decoded_dir.mkdir()
    args = argparse.Namespace(
        no_visualization=False,
        marker_diameter=20,
        marker_alpha=220,
        video_fps=30.0,
    )
    monkeypatch.setattr(gaze, "parse_args", lambda: args)
    monkeypatch.setattr(gaze, "require_cv2_numpy", lambda **_kwargs: None)
    monkeypatch.setattr(
        gaze,
        "resolve_paths",
        lambda _args: (
            session_dir,
            session_dir / "metadata.csv",
            camera_path,
            decoded_dir,
            tmp_path / "output",
            calibration_path,
        ),
    )
    monkeypatch.setattr(
        gaze,
        "ensure_decoded_image_index",
        lambda *_args: (_ for _ in ()).throw(ValueError("sentinel alignment error")),
    )

    with pytest.raises(SystemExit, match="sentinel alignment error"):
        gaze.main()


def test_fused_debug_output_handles_unmapped_exact_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_dir, rows = make_session(tmp_path)
    decoded_dir = tmp_path / "decoded"
    image_paths = make_images(decoded_dir, 4)
    decoder.write_index_csv(
        decoded_dir,
        image_paths,
        rows,
        encoded_frame_indices=[0, 1, 3, 4],
    )
    monkeypatch.setattr(fused.gaze, "require_cv2_numpy", lambda **_kwargs: (object(), object()))

    summary = fused.save_debug_undistorted_frames(
        rows=[rows[2]],
        decoded_dir=decoded_dir,
        output_root=tmp_path / "output",
        undistort_maps=(None, None),
        max_rows=0,
        fallback_width=1280,
        fallback_height=960,
    )

    assert summary["written_count"] == 0
    assert summary["missing_count"] == 1
