from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def write_trajectory_flat(output_path: Path, rows: List[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in rows:
            if not r.get("success", False):
                continue
            T = r["T_w_c07"]
            flat = " ".join(f"{x:.9f}" for x in T.reshape(-1))
            f.write(f"{r['frame_index']} {flat}\n")


def write_frame_matrices(output_dir: Path, rows: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if not r.get("success", False):
            continue
        T = r["T_w_c07"]
        out = output_dir / f"frame_{r['frame_index']}.txt"
        np.savetxt(out, T, fmt="%.9f")


def write_diagnostics_csv(output_path: Path, rows: List[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_index",
        "success",
        "reason",
        "visible_fixed",
        "used_fixed",
        "inlier_fixed",
        "target_inliers",
        "target_rmse",
        "status",
        "valid",
        "confidence",
        "detected_tag_ids",
        "used_tag_ids",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "frame_index": r.get("frame_index", ""),
                    "success": int(bool(r.get("success", False))),
                    "reason": r.get("reason", ""),
                    "visible_fixed": ",".join(r.get("visible_fixed", [])),
                    "used_fixed": ",".join(r.get("used_fixed", [])),
                    "inlier_fixed": ",".join(r.get("inlier_fixed", [])),
                    "target_inliers": int(r.get("target_inliers", 0)),
                    "target_rmse": float(r.get("target_rmse", float("inf"))),
                    "status": r.get("status", ""),
                    "valid": int(bool(r.get("valid", r.get("success", False)))),
                    "confidence": float(r.get("confidence", 0.0)),
                    "detected_tag_ids": ",".join(str(x) for x in r.get("detected_tag_ids", [])),
                    "used_tag_ids": ",".join(str(x) for x in r.get("used_tag_ids", [])),
                }
            )


def write_trajectory_json(output_path: Path, rows: List[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for row in rows:
        raw = row.get("T_world_from_ego_raw")
        final = row.get("T_world_from_ego", row.get("T_w_c07"))
        frames.append(
            {
                "frame_index": row.get("frame_index"),
                "valid": bool(row.get("valid", row.get("success", False))),
                "status": row.get("status", row.get("reason", "")),
                "confidence": float(row.get("confidence", 0.0)),
                "T_world_from_ego_raw": None if raw is None else np.asarray(raw).tolist(),
                "T_world_from_ego": None if final is None else np.asarray(final).tolist(),
                "detected_tag_ids": row.get("detected_tag_ids", []),
                "used_tag_ids": row.get("used_tag_ids", []),
                "inlier_corner_count": int(row.get("target_inliers", 0)),
                "reprojection_rmse_px": None
                if not np.isfinite(float(row.get("target_rmse", float("inf"))))
                else float(row["target_rmse"]),
            }
        )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"pose_convention": "T_world_from_ego", "frames": frames}, f, indent=2, ensure_ascii=False)


def write_ego_extrinsics_json(output_path: Path, rows: List[dict]) -> None:
    """Write valid per-frame world-to-ego extrinsics as 4x4 matrices.

    The world frame is camera 00 when camera 00 has identity extrinsics. Each
    matrix follows p_ego = T_ego_from_world @ p_world.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extrinsics: dict[str, list[list[float]]] = {}
    for row in rows:
        T_world_from_ego = row.get("T_world_from_ego", row.get("T_w_c07"))
        if not row.get("success", False) or T_world_from_ego is None:
            continue
        T_ego_from_world = np.linalg.inv(np.asarray(T_world_from_ego, dtype=np.float64).reshape(4, 4))
        extrinsics[str(row["frame_index"])] = T_ego_from_world.tolist()
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(extrinsics, f, indent=2, ensure_ascii=False)


def write_plots(output_dir: Path, rows: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = [r["frame_index"] for r in rows]
    rmse = []
    for r in rows:
        value = float(r.get("target_rmse", np.nan))
        rmse.append(value if r.get("success", False) and np.isfinite(value) else np.nan)
    valid_counts = [len(r.get("visible_fixed", [])) for r in rows]

    xs = []
    ys = []
    for r in rows:
        if not r.get("success", False):
            continue
        T = r["T_w_c07"]
        xs.append(float(T[0, 3]))
        ys.append(float(T[1, 3]))

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(frame_ids)), rmse)
    plt.title("Target camera reprojection RMSE")
    plt.xlabel("Frame idx")
    plt.ylabel("RMSE (px)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rmse_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(frame_ids)), valid_counts)
    plt.title("Visible fixed camera count")
    plt.xlabel("Frame idx")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "valid_cam_count.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    if xs and ys:
        plt.plot(xs, ys, "-o", markersize=2)
    plt.title("Camera 07 trajectory top view (X-Y)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "traj_topview.png", dpi=150)
    plt.close()
