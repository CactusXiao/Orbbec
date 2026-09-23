from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import estimate_pico_ego_extrinsics as estimator


def _row(reference_frame: int, ego_frame: int) -> estimator.TimestampRow:
    return estimator.TimestampRow(
        row_index=reference_frame,
        frame_index=f"{reference_frame:05d}",
        frame_number=reference_frame,
        ref_timestamp_us=str(1_000_000 + reference_frame * 33_333),
        ego_frame_index=str(ego_frame),
        ego_frame_number=ego_frame,
        ego_timestamp_us=str(1_000_000 + ego_frame * 33_333),
        raw={},
    )


def _ego_model() -> estimator.EgoImageModel:
    matrix = np.eye(3, dtype=np.float64)
    return estimator.EgoImageModel(
        enabled=False,
        source="test",
        K_raw=matrix,
        D_fisheye=None,
        K_pnp=matrix,
        dist_pnp=np.zeros((5, 1), dtype=np.float64),
        image_size=(8, 6),
    )


def test_timestamp_loader_requires_explicit_ego_mapping(tmp_path: Path):
    path = tmp_path / "timestamps.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("frame_index", "ref_timestamp_us", "ego_frame_index", "ego_timestamp_us"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_index": "00000",
                "ref_timestamp_us": "1000000",
                "ego_frame_index": "7",
                "ego_timestamp_us": "1000100",
            }
        )
        writer.writerow(
            {
                "frame_index": "00001",
                "ref_timestamp_us": "1033333",
                "ego_frame_index": "",
                "ego_timestamp_us": "",
            }
        )

    rows = estimator._load_timestamp_rows(path, max_rows=None)

    assert [(row.frame_number, row.ego_frame_number) for row in rows] == [(0, 7)]


def test_direct_pass_always_reads_mapped_ego_frames(monkeypatch, tmp_path: Path):
    requested_frames: list[int] = []

    class FakeSource:
        def __init__(self, source_path, label, *, force_temp_remux=False):
            self.source_path = Path(source_path)
            self.active_path = self.source_path
            self.is_image_sequence = False
            self.used_temp_remux = False

        def read(self, frame_index: int):
            requested_frames.append(frame_index)
            return np.zeros((6, 8, 3), dtype=np.uint8)

        def close(self):
            return None

    monkeypatch.setattr(estimator, "SequentialVideoFrameSource", FakeSource)
    monkeypatch.setattr(
        estimator,
        "_detect_ego_apriltags",
        lambda frame, detector, model: ([], "tags_not_found", "raw", None),
    )

    estimates, stats = estimator._run_direct_ego_pass(
        [_row(0, 0), _row(1, 5), _row(2, 5), _row(3, 9)],
        tmp_path / "ego" / "RGB" / "rgb.h265",
        _ego_model(),
        {},
        object(),
        tmp_path / "output",
        False,
        0,
        False,
    )

    assert requested_frames == [0, 5, 5, 9]
    assert [estimate.row.frame_number for estimate in estimates] == [0, 1, 2, 3]
    assert stats["frame_mapping"] == "timestamps.csv:frame_index->ego_frame_index"
