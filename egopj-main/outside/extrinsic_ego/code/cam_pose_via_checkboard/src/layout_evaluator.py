from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .apriltag import detect_apriltag_markers


def _tag_normals(tag_map_path: Path | None) -> dict[int, np.ndarray]:
    if tag_map_path is None:
        return {}
    with tag_map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    normals = {}
    for key, item in data.get("tags", {}).items():
        corners = np.asarray(item["corners_world_m"], dtype=np.float64).reshape(4, 3)
        normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normals[int(key)] = normal / norm
    return normals


def evaluate_layout(detected_ids_per_frame: list[set[int]], normals: dict[int, np.ndarray]) -> dict:
    total = len(detected_ids_per_frame)
    counts = Counter(tag_id for ids in detected_ids_per_frame for tag_id in ids)

    def ratio(predicate) -> float:
        return 0.0 if total == 0 else sum(int(predicate(ids)) for ids in detected_ids_per_frame) / total

    def has_two_planes(ids: set[int]) -> bool:
        available = [normals[tag_id] for tag_id in ids if tag_id in normals]
        for i, first in enumerate(available):
            for second in available[i + 1 :]:
                angle = np.degrees(np.arccos(np.clip(abs(float(np.dot(first, second))), -1.0, 1.0)))
                if angle >= 15.0:
                    return True
        return False

    known_ids = sorted(counts)
    dropout = {
        str(tag_id): ratio(lambda ids, removed=tag_id: len(ids - {removed}) >= 2)
        for tag_id in known_ids
    }
    return {
        "frame_count": total,
        "at_least_2_tags_ratio": ratio(lambda ids: len(ids) >= 2),
        "at_least_3_tags_ratio": ratio(lambda ids: len(ids) >= 3),
        "two_plane_ratio": ratio(has_two_planes) if normals else None,
        "worst_single_tag_removed_at_least_2_ratio": min(dropout.values()) if dropout else 0.0,
        "single_tag_dropout_ratios": dropout,
        "per_tag_detection_ratio": {
            str(tag_id): (counts[tag_id] / total if total else 0.0) for tag_id in known_ids
        },
        "passes_recommended_coverage": bool(
            total > 0
            and ratio(lambda ids: len(ids) >= 3) >= 0.95
            and min(dropout.values(), default=0.0) >= 0.90
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure AprilTag layout visibility and occlusion redundancy")
    parser.add_argument("--ego-rgb-dir", type=Path, required=True)
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--tag-map", type=Path, default=None, help="Optional map used to evaluate multi-plane coverage")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(
        p for p in args.ego_rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    detections = []
    for path in paths:
        image = cv2.imread(str(path))
        markers, _ = detect_apriltag_markers(image, args.family)
        detections.append({tag_id for tag_id, _ in markers})
    report = evaluate_layout(detections, _tag_normals(args.tag_map))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
