#!/usr/bin/env python3
"""
将一张深度图的每个像素映射到对应 RGB 图像的 2D 像素坐标。

数学推导（与 collection/viewer 的语义一致）：
1) 深度图像素 p_d = (u_d, v_d) 是深度相机成像平面上的“畸变后像素”。
2) 通过深度相机内参 K_d 与畸变参数 D_d，对 p_d 去畸变得到归一化坐标 (x_n, y_n)。
3) 深度值 d_raw 结合尺度 s_mm（毫米/单位），得到
      Z_d = d_raw * s_mm   (单位：mm)
   进而得到深度相机坐标系 3D 点
      P_d = [X_d, Y_d, Z_d]^T = [x_n * Z_d, y_n * Z_d, Z_d]^T
4) 使用 d2c 外参（Depth->Color/RGB）：
      P_c = R_d2c * P_d + t_d2c
   注意：t_d2c 在你提供的数据语义中是毫米，故与 P_d 同单位（mm）。
5) 将 P_c 通过 RGB 相机内参 K_c 与畸变 D_c 投影到 RGB 成像平面：
      p_c = project(K_c, D_c, P_c) = (u_c, v_c)

输出：
- uv_map: HxW x 2 的 float32 数组，每个深度像素对应一个 RGB 亚像素坐标 (u_c, v_c)
- valid_mask: HxW 的 bool 数组，表示该映射是否有效（深度非零且 Z_c>0）
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _intrinsic_to_K(intr: Dict) -> np.ndarray:
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _dist_to_opencv(dist: Dict) -> np.ndarray:
    return np.array(
        [
            float(dist.get("k1", 0.0)),
            float(dist.get("k2", 0.0)),
            float(dist.get("p1", 0.0)),
            float(dist.get("p2", 0.0)),
            float(dist.get("k3", 0.0)),
            float(dist.get("k4", 0.0)),
            float(dist.get("k5", 0.0)),
            float(dist.get("k6", 0.0)),
        ],
        dtype=np.float64,
    ).reshape(-1, 1)


def _extract_intrinsics_and_distortion(intrinsics_root: Dict, cam_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cam_key not in intrinsics_root:
        raise KeyError(f"cam_key={cam_key} 不存在于内参文件")
    obj = intrinsics_root[cam_key]
    if "rgb_to_depth" in obj:
        rd = obj["rgb_to_depth"]
        K_d = _intrinsic_to_K(rd["depth_intrinsic"])
        D_d = _dist_to_opencv(rd["depth_distortion"])
        K_c = _intrinsic_to_K(rd["rgb_intrinsic"])
        D_c = _dist_to_opencv(rd["rgb_distortion"])
        return K_d, D_d, K_c, D_c

    K_d = _intrinsic_to_K(obj["Depth"]["intrinsic"])
    D_d = _dist_to_opencv(obj["Depth"]["distortion"])
    K_c = _intrinsic_to_K(obj["RGB"]["intrinsic"])
    D_c = _dist_to_opencv(obj["RGB"]["distortion"])
    return K_d, D_d, K_c, D_c


def _extract_d2c(extrinsics_root: Dict, cam_key: str) -> Tuple[np.ndarray, np.ndarray]:
    if cam_key not in extrinsics_root:
        raise KeyError(f"cam_key={cam_key} 不存在于外参文件")
    obj = extrinsics_root[cam_key]
    if "rgb_to_depth" in obj and "d2c_extrinsic" in obj["rgb_to_depth"]:
        d2c = obj["rgb_to_depth"]["d2c_extrinsic"]
    elif "d2c_extrinsic" in obj:
        d2c = obj["d2c_extrinsic"]
    else:
        raise KeyError("外参文件缺少 d2c_extrinsic（Depth->RGB）")
    R = np.array(d2c["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.array(d2c["translation"], dtype=np.float64).reshape(3, 1)
    return R, t


def depth_to_rgb_uv_map(
    depth_u16: np.ndarray,
    K_d: np.ndarray,
    D_d: np.ndarray,
    K_c: np.ndarray,
    D_c: np.ndarray,
    R_d2c: np.ndarray,
    t_d2c_mm: np.ndarray,
    depth_scale_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if depth_u16.ndim != 2:
        raise ValueError("depth_u16 必须是单通道 HxW")
    if depth_u16.dtype != np.uint16:
        raise ValueError("depth_u16 必须是 uint16")
    if depth_scale_mm <= 0:
        raise ValueError("depth_scale_mm 必须 > 0")

    h, w = depth_u16.shape

    # 构造每个像素的 (u, v)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pix = np.stack([u, v], axis=-1).reshape(-1, 1, 2).astype(np.float64)

    # 仅对有效深度做计算，避免无意义投影
    d_raw = depth_u16.reshape(-1).astype(np.float64)
    valid = d_raw > 0

    uv_map = np.full((h * w, 2), np.nan, dtype=np.float32)
    valid_mask = np.zeros((h * w,), dtype=bool)
    if not np.any(valid):
        return uv_map.reshape(h, w, 2), valid_mask.reshape(h, w)

    pix_valid = pix[valid]
    z_mm = (d_raw[valid] * depth_scale_mm).reshape(-1, 1)  # mm

    # 去畸变到归一化平面，输出 (x_n, y_n)
    und = cv2.undistortPoints(pix_valid, K_d, D_d, P=None)
    x_n = und[:, 0, 0:1]
    y_n = und[:, 0, 1:2]

    # 反投影到 depth 相机坐标系（mm）
    X_d = x_n * z_mm
    Y_d = y_n * z_mm
    P_d = np.concatenate([X_d, Y_d, z_mm], axis=1)  # Nx3, mm

    # Depth -> RGB（mm）
    P_c = (R_d2c @ P_d.T + t_d2c_mm).T  # Nx3

    # Z_c 必须 > 0 才可投影
    zc_pos = P_c[:, 2] > 0
    if not np.any(zc_pos):
        return uv_map.reshape(h, w, 2), valid_mask.reshape(h, w)

    P_c_pos = P_c[zc_pos].reshape(-1, 1, 3)
    # 已在 RGB 相机坐标系下，故 rvec/tvec 取 0
    img_pts, _ = cv2.projectPoints(
        objectPoints=P_c_pos,
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.zeros((3, 1), dtype=np.float64),
        cameraMatrix=K_c,
        distCoeffs=D_c,
    )
    uv = img_pts.reshape(-1, 2).astype(np.float32)

    valid_indices = np.flatnonzero(valid)
    valid_indices = valid_indices[zc_pos]
    uv_map[valid_indices] = uv
    valid_mask[valid_indices] = True

    return uv_map.reshape(h, w, 2), valid_mask.reshape(h, w)


def map_depth_to_rgb_pixels(
    depth_image_path: str | Path,
    intrinsics_file_path: str | Path,
    extrinsics_file_path: str | Path,
    cam_key: str,
    depth_scale_mm: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    读取深度图、两类参数文件（内参文件 + 外参文件），并计算“深度像素 -> RGB像素”的映射。

    参数说明
    ----------
    depth_image_path : str | Path
        深度图路径。要求是 16UC1（通常为 PNG），每个像素是深度原始值 d_raw。

    intrinsics_file_path : str | Path
        内参文件路径。支持两种结构：
        1) collection 生成的 camera_params.json：
           root[cam_key]["rgb_to_depth"]["depth_intrinsic/depth_distortion/rgb_intrinsic/rgb_distortion"]
        2) 分离结构：
           root[cam_key]["Depth"]["intrinsic/distortion"] 与 root[cam_key]["RGB"]["intrinsic/distortion"]

    extrinsics_file_path : str | Path
        外参文件路径。需要提供 d2c_extrinsic（Depth->RGB）：
        1) root[cam_key]["rgb_to_depth"]["d2c_extrinsic"]，或
        2) root[cam_key]["d2c_extrinsic"]。
        注意：t_d2c 的单位按毫米（mm）解释。

    cam_key : str
        相机编号键，例如 "00"、"01"、"02"。

    depth_scale_mm : float, default=1.0
        深度尺度，表示 d_raw 的 1 个单位对应多少毫米。
        例如常见 16-bit 深度图若单位就是 mm，则 depth_scale_mm=1.0。

    返回
    -------
    Dict[str, np.ndarray]
        返回字典，包含：
        - "uv_map": np.ndarray, shape=(H, W, 2), dtype=float32
            uv_map[v, u] = (u_rgb, v_rgb)，表示深度像素 (u,v) 在 RGB 图像中的亚像素坐标。
            若无效则为 NaN。
        - "valid_mask": np.ndarray, shape=(H, W), dtype=bool
            True 表示该深度像素映射有效（深度非零，且变换后 Z_c > 0）。

    使用示例
    ----------
    result = map_depth_to_rgb_pixels(
        depth_image_path="/path/to/depth.png",
        intrinsics_file_path="/path/to/camera_params.json",
        extrinsics_file_path="/path/to/extrinsic.json",
        cam_key="00",
        depth_scale_mm=1.0,
    )
    uv_map = result["uv_map"]
    valid = result["valid_mask"]
    """
    depth_path = Path(depth_image_path)
    intr_path = Path(intrinsics_file_path)
    ext_path = Path(extrinsics_file_path)

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"无法读取深度图: {depth_path}")

    intr_root = _load_json(intr_path)
    ext_root = _load_json(ext_path)
    K_d, D_d, K_c, D_c = _extract_intrinsics_and_distortion(intr_root, cam_key)
    R_d2c, t_d2c_mm = _extract_d2c(ext_root, cam_key)

    uv_map, valid_mask = depth_to_rgb_uv_map(
        depth_u16=depth,
        K_d=K_d,
        D_d=D_d,
        K_c=K_c,
        D_c=D_c,
        R_d2c=R_d2c,
        t_d2c_mm=t_d2c_mm,
        depth_scale_mm=depth_scale_mm,
    )
    return {"uv_map": uv_map, "valid_mask": valid_mask}


def save_optional_csv(
    out_csv: Path,
    uv_map: np.ndarray,
    valid_mask: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale_mm: float,
) -> None:
    h, w = depth_u16.shape
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["depth_u", "depth_v", "depth_raw", "depth_mm", "rgb_u", "rgb_v"])
        ys, xs = np.where(valid_mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            u_rgb, v_rgb = uv_map[y, x]
            d_raw = int(depth_u16[y, x])
            writer.writerow([x, y, d_raw, d_raw * depth_scale_mm, float(u_rgb), float(v_rgb)])


def main() -> None:
    parser = argparse.ArgumentParser(description="将深度图逐像素映射到 RGB 像素坐标")
    parser.add_argument("--intrinsics", type=Path, required=True, help="内参文件路径（如 camera_params.json）")
    parser.add_argument("--extrinsics", type=Path, required=True, help="外参文件路径（含 d2c_extrinsic）")
    parser.add_argument("--cam-key", type=str, required=True, help="相机键，例如 00/01/02")
    parser.add_argument("--depth", type=Path, required=True, help="输入深度图路径（16UC1 PNG）")
    parser.add_argument(
        "--depth-scale-mm",
        type=float,
        default=1.0,
        help="深度值尺度（每个 depth 单位对应多少 mm）。collection 常见为 1.0",
    )
    parser.add_argument("--out-uv", type=Path, default=None, help="输出 uv_map .npy 路径，默认写到 task-dir")
    parser.add_argument("--out-mask", type=Path, default=None, help="输出 valid_mask .npy 路径，默认写到 task-dir")
    parser.add_argument("--out-csv", type=Path, default=None, help="可选：导出有效像素映射 CSV")
    args = parser.parse_args()

    result = map_depth_to_rgb_pixels(
        depth_image_path=args.depth,
        intrinsics_file_path=args.intrinsics,
        extrinsics_file_path=args.extrinsics,
        cam_key=args.cam_key,
        depth_scale_mm=args.depth_scale_mm,
    )
    uv_map = result["uv_map"]
    valid_mask = result["valid_mask"]
    depth = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)

    base_dir = args.intrinsics.parent
    out_uv = args.out_uv if args.out_uv is not None else (base_dir / f"{args.cam_key}_depth_to_rgb_uv.npy")
    out_mask = args.out_mask if args.out_mask is not None else (base_dir / f"{args.cam_key}_depth_to_rgb_valid.npy")

    np.save(out_uv, uv_map)
    np.save(out_mask, valid_mask)

    if args.out_csv is not None:
        save_optional_csv(args.out_csv, uv_map, valid_mask, depth, args.depth_scale_mm)

    valid_count = int(valid_mask.sum())
    total = int(valid_mask.size)
    print(f"[done] uv_map: {out_uv}")
    print(f"[done] valid_mask: {out_mask}")
    print(f"[stats] valid={valid_count}/{total} ({(valid_count / max(total, 1)) * 100:.2f}%)")
    if args.out_csv is not None:
        print(f"[done] csv: {args.out_csv}")


if __name__ == "__main__":
    main()
