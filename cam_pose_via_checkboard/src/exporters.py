from __future__ import annotations

import csv
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
                }
            )


def write_plots(output_dir: Path, rows: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = [r["frame_index"] for r in rows]
    rmse = [float(r.get("target_rmse", np.nan)) if r.get("success", False) else np.nan for r in rows]
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
