"""Read-only, timestamp-aligned tactile measurements for the episode viewer."""
from __future__ import annotations

import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .tactile_layout import FINGERS, glove_layout
except ImportError:
    from tactile_layout import FINGERS, glove_layout


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def _index(value: Any) -> Optional[int]:
    try:
        result = int(value)
        return result if result >= 0 else None
    except (TypeError, ValueError):
        return None


def _inside(root: Path, path: Path) -> Optional[Path]:
    try:
        resolved = path.resolve()
        return resolved if root in resolved.parents and resolved.is_file() else None
    except OSError:
        return None


class TactileTimeline:
    def __init__(self, episode_dir: Path, fps: float = 30.0):
        self.root = episode_dir.resolve()
        self.rows: Dict[int, Dict[str, str]] = {}
        self.streams = []
        self.error = ""
        self.tolerance_us = max(50000, round(2_000_000 / max(1.0, fps)))
        timestamps = _inside(self.root, self.root / "timestamps.csv")
        if timestamps:
            try:
                with timestamps.open(encoding="utf-8-sig", newline="") as handle:
                    for index, row in enumerate(csv.DictReader(handle)):
                        frame = _index(row.get("frame_index", index))
                        if frame is not None:
                            self.rows[frame] = row
            except (OSError, UnicodeError, csv.Error):
                self.rows.clear()
        manifests = sorted(self.root.glob("*/touch_manifest.json"))
        devices = []
        for manifest_path in manifests:
            safe = _inside(self.root, manifest_path)
            if not safe:
                continue
            try:
                manifest = json.loads(safe.read_text(encoding="utf-8-sig"))
                if manifest.get("schema") != "orbbec.touch.jq_shroom.v3":
                    self.error = "触觉数据格式不支持，请使用更新后的采集程序重新采集"
                    continue
                for item in manifest.get("devices", []):
                    if isinstance(item, dict) and item.get("id"):
                        devices.append((safe.parent, item))
            except (OSError, ValueError, AttributeError, TypeError):
                self.error = "触觉数据清单无法读取"
                continue
        if not manifests and (self.root / "touch").exists():
            self.error = "缺少新版触觉数据清单，请使用更新后的采集程序重新采集"
        for directory, device in devices:
            side = {1: "left", 2: "right"}.get(_index(device.get("sensor_type")))
            if side is None or side != device.get("side"):
                self.error = "触觉数据的左右手标识与传感器类型不一致"
                continue
            stream = {"id": str(device["id"]), "side": side,
                      "samples": {}, "ordered": [], "times": [], "force_max": 0.0, "adc_max": 255.0,
                      "has_force": False, "error": ""}
            name = Path(str(device.get("raw_csv") or ""))
            safe = None if name.is_absolute() or ".." in name.parts else _inside(self.root, directory / name)
            if safe:
                try:
                    self._read_stream(stream, safe)
                except (OSError, UnicodeError, csv.Error, ValueError):
                    stream["samples"].clear()
                    stream["error"] = "触觉文件无法读取"
            stream["ordered"] = sorted((s for s in stream["samples"].values() if s["timestamp_us"] is not None),
                                       key=lambda s: s["timestamp_us"])
            stream["times"] = [s["timestamp_us"] for s in stream["ordered"]]
            self.streams.append(stream)

    @staticmethod
    def _read_stream(stream: Dict[str, Any], path: Path) -> None:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = set(reader.fieldnames or [])
            required = {"sample_index", "touch_timestamp_us", "calibrated_region_force_n",
                        "force_out_of_range", "force_calibration_status"}
            required.update(f"raw_adc_{i:03d}" for i in range(256))
            old_fields = {f"pressure_{i:03d}" for i in range(256)} | {f"force_{i:03d}_n" for i in range(256)}
            if not required <= header or header & old_fields:
                stream["error"] = "触觉 CSV 格式不支持，需要新版区域力和原始 ADC 数据"
                return
            stream["has_force"] = stream["side"] == "right" and "calibrated_region_force_n" in header
            for row in reader:
                index = _index(row.get("sample_index"))
                if index is None:
                    continue
                adc = [_number(row.get(f"raw_adc_{i:03d}")) for i in range(256)]
                timestamp = _index(row.get("touch_timestamp_us"))
                if timestamp is None:
                    continue
                status = row["force_calibration_status"]
                allowed = {"right_middle_region", "invalid_force"} if stream["side"] == "right" else {"uncalibrated_hand"}
                if status not in allowed:
                    raise ValueError("Invalid force calibration status")
                valid_scope = stream["side"] == "right" and status == "right_middle_region"
                total = _number(row.get("calibrated_region_force_n")) if valid_scope else None
                sample = {"index": index, "timestamp_us": timestamp, "adc": adc,
                          "total_n": total, "force_status": status,
                          "out_of_range": str(row.get("force_out_of_range", "0")).lower() in ("1", "true"),
                          "quality": str(row.get("quality_flag") or "ok")}
                stream["samples"][index] = sample
                stream["force_max"] = max(stream["force_max"], total or 0.0)

    def frame(self, frame: int) -> Dict[str, Any]:
        row = self.rows.get(frame, {})
        reference = _index(row.get("ref_timestamp_us"))
        hands = []
        for stream in self.streams:
            prefix = "touch_" + stream["id"]
            hand = {"id": stream["id"], "side": stream["side"], "status": "missing",
                    "layout": glove_layout(stream["side"]),
                    "calibrated_sensor_ids": FINGERS["right"][2] if stream["side"] == "right" and stream["has_force"] else [],
                    "message": stream["error"] or "未采集触觉数据", "sample": None,
                    "delta_ms": None, "has_force": stream["has_force"],
                    "force_max": stream["force_max"], "adc_max": stream["adc_max"]}
            if stream["samples"]:
                sample = None
                if prefix + "_frame_index" in row:
                    # An explicit missing match must stay missing, never borrow another frame.
                    sample = stream["samples"].get(_index(row[prefix + "_frame_index"]))
                else:
                    target = _index(row.get(prefix + "_timestamp_us"))
                    target = target if target is not None else reference
                    if target is not None and stream["times"]:
                        pos = bisect.bisect_left(stream["times"], target)
                        candidates = stream["ordered"][max(0, pos - 1):pos + 1]
                        sample = min(candidates, key=lambda s: abs(s["timestamp_us"] - target))
                        if abs(sample["timestamp_us"] - target) > self.tolerance_us:
                            sample = None
                if sample is not None:
                    delta = (sample["timestamp_us"] - reference) if reference is not None and sample["timestamp_us"] is not None else None
                    if delta is not None and abs(delta) > self.tolerance_us:
                        hand.update(status="unaligned", message="触觉与画面时间差过大", delta_ms=delta / 1000)
                    else:
                        hand.update(status="ready", message="已同步", sample=sample,
                                    delta_ms=delta / 1000 if delta is not None else None)
                else:
                    hand.update(status="unaligned", message="当前帧无同步触觉样本")
            hands.append(hand)
        return {"frame_index": frame, "hands": hands, "error": self.error}
