# cam_pose_via_checkboard

使用多台**已标定固定相机**（默认 `00~05`）与 AprilTag 或棋盘格观测，估计**目标相机**（默认 `07`，支持头戴式相机）每一帧在世界坐标系下的位姿 `T_w_c07`。

> **头戴式相机支持**：将头戴相机图像放在 `{dataset_root}/07/RGB/` 目录下，设置 `target_type: apriltag`，即可直接估计头戴相机每帧位姿，详见下方 [头戴式相机使用说明](#头戴式相机使用说明)。

## 输入约定

只需要传入 `dataset_root`。程序会自动读取：

- 内参文件：优先 `${dataset_root}/camera_params.json`，若不存在会搜索 `*camera*params*.json` 与 `*camera*parms*.json`
- 外参文件：优先 `${dataset_root}/extrinsics.json`，若不存在会自动尝试 `${dataset_root}/extrinsic.json` 或搜索 `*extrinsic*.json`
- 图像目录：`${dataset_root}/{cam_id}/RGB/{frame_index}.jpg`

其中 `cam_id` 默认使用：固定相机 `00~05` + 目标相机 `07`。

## 参数字段映射

从 `camera_params.json` 中读取：

- RGB内参：`[cam_id].RGB.intrinsic.{fx,fy,cx,cy,width,height}`
- RGB畸变：`[cam_id].RGB.distortion.{k1,k2,p1,p2,k3}`

外参优先读取：

- `extrinsics.json/extrinsic.json` 中的 `[cam_id].rotation` 与 `[cam_id].translation`
- 若内参文件本身也包含 `rotation/translation`，同样可直接使用

注意：默认只要求固定相机 `00~05` 在外参文件中存在，目标相机 `07` 可以不在外参文件里。

## 输出

默认输出到 `outputs/`：

- `trajectory_tw_c07.txt`：每行 `frame_index + 16个矩阵元素(按行展开)`
- `matrices/frame_{frame_index}.txt`：每帧一个4x4矩阵
- `diagnostics.csv`：每帧质量信息与失败原因
- `plots/*.png`：RMSE曲线、可见相机数曲线、07轨迹俯视图

## Ubuntu 安装与运行

```bash
cd cam_pose_via_checkboard
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m src.main --dataset_root /home/ubuntu/orbbec/src/sync/test/test/zyc --config configs/default.yaml
```

Ubuntu 一键脚本：

```bash
cd cam_pose_via_checkboard
chmod +x scripts/run_example.sh
./scripts/run_example.sh /home/ubuntu/orbbec/src/sync/test/test/zyc
```

说明：

- 建议始终在 `cam_pose_via_checkboard` 目录下运行命令。
- 如果系统里默认 `python` 不是 Python 3，请统一使用 `python3`。

## 常见问题

- 角点检测失败较多：优先检查棋盘是否完整可见、运动模糊、曝光。
- 位姿尺度异常：确认外参 `translation` 单位，必要时启用 `use_mm_to_m_auto_scale`。
- 姿态方向看起来翻转：尝试切换 `fixed_extrinsics_are_twc`（外参方向可能与配置不一致）。

## AprilTag Workflow (Auto, no pre-defined world map)

This project now supports `target_type: apriltag`.
In this mode, you do **not** need to pre-fill each tag's world pose in a config file.
The world pose of each visible AprilTag is estimated automatically from fixed cameras per frame.

### 1) Config example

Edit `configs/default.yaml`:

```yaml
target_type: "apriltag"
apriltag_family: "tag36h11"   # tag16h5 | tag25h9 | tag36h10 | tag36h11
apriltag_default_size_m: 0.10   # physical tag size (meter)
apriltag_min_tags: 1
apriltag_min_inliers: 4
```

Notes:

- `apriltag_default_size_m` must match your real printed tag size.
- Current implementation assumes all tags use the same size.

### 2) Pipeline logic (per frame)

1. For each fixed camera (`00~05`), detect visible AprilTags.
2. For each detected tag, solve single-tag PnP in camera frame (`T_c_tag`).
3. Convert to world frame using fixed-camera extrinsic (`T_w_tag = T_w_c_fixed * T_c_tag`).
4. If one tag is seen by multiple fixed cameras, fuse those `T_w_tag` candidates.
5. Build this frame's world-tag map from fused tags.
6. For target camera (`07`), detect AprilTags and run joint PnP using:
   - image points from camera-07,
   - world 3D corners from this frame's world-tag map.
7. Invert the result to output `T_w_c07`.

### 3) Run

```bash
python -m src.main --dataset_root <your_dataset_root> --config configs/default.yaml
```

### 4) Failure reasons in diagnostics.csv

Common reasons for AprilTag mode:

- `insufficient_fixed_observations`: too few fixed cameras with valid detections.
- `insufficient_world_tags`: fixed-camera fusion did not produce enough world tags.
- `target_failed:tags_not_found`: target camera cannot detect tags in that frame.
- `target_failed:few_tags`: target camera sees fewer tags than `apriltag_min_tags`.
- `fixed_failed:opencv_has_no_aruco`: OpenCV build lacks `aruco` (install `opencv-contrib-python`).

## 头戴式相机使用说明

本项目完整支持头戴式相机位姿估计，使用 `target_type: apriltag` 模式，无需预先标定 AprilTag 世界坐标。

### 1）数据目录结构

```
dataset_root/
├── camera_params.json        # 所有相机内参（含固定相机 + 头戴相机）
├── extrinsics.json           # 固定相机外参（头戴相机不需要）
├── 00/RGB/0000.jpg           # 固定相机 00 图像
├── 01/RGB/0000.jpg
├── ...
├── 05/RGB/0000.jpg
└── 07/RGB/0000.jpg           # 头戴相机图像（target_camera_id）
```

- 固定相机需要在 `extrinsics.json` 中提供外参（`rotation` + `translation`）。
- 头戴相机（`07`）**不需要**外参，程序会自动估计其每帧位姿。
- 图像文件名（帧索引）需要在固定相机和头戴相机之间对应一致。

### 2）配置文件

编辑 `configs/default.yaml`：

```yaml
target_type: "apriltag"          # 使用 AprilTag 模式
apriltag_family: "tag36h11"      # tag16h5 | tag25h9 | tag36h10 | tag36h11
apriltag_default_size_m: 0.10    # 标签物理边长（米），必须与实际打印尺寸一致
apriltag_min_tags: 1             # 每帧至少需要几个 tag 才能估计位姿
apriltag_min_inliers: 4          # PnP 最少内点数

fixed_camera_ids:
  - "00"
  - "01"
  - "02"
  - "03"
  - "04"
  - "05"
target_camera_id: "07"           # 头戴相机 ID

min_fixed_observations: 2        # 每帧至少需要几台固定相机看到 tag
frame_policy: "intersection"     # intersection：只处理所有相机都有图像的帧
                                 # target_primary：以头戴相机帧为主
```

> `apriltag_default_size_m` 必须与实际打印标签尺寸一致，否则估计的平移尺度会出错。

### 3）每帧处理流程

1. 对每台固定相机（`00~05`）检测当前帧可见的 AprilTag。
2. 对每个检测到的 tag 做单 tag PnP，得到相机系下 `T_c_tag`。
3. 结合固定相机外参，转换到世界系：`T_w_tag = T_w_c_fixed × T_c_tag`。
4. 若同一 tag 被多台固定相机同时看到，对多个 `T_w_tag` 候选进行 RANSAC 融合。
5. 得到该帧的 world-tag map（本帧可用 tag 的世界 3D 角点）。
6. 对头戴相机（`07`）检测 AprilTag，用 world-tag map 做联合 PnP。
7. 对结果取逆，输出头戴相机在世界系下位姿 `T_w_c07`。

### 4）运行命令

```bash
# 激活虚拟环境后
python -m src.main --dataset_root /path/to/dataset_root --config configs/default.yaml
```

### 5）输出文件

输出到 `outputs/` 目录：

| 文件 | 内容 |
|------|------|
| `trajectory_tw_c07.txt` | 每行：`帧索引 + 16个矩阵元素（按行展开）` |
| `matrices/frame_*.txt` | 每帧 4×4 位姿矩阵 |
| `diagnostics.csv` | 每帧质量信息与失败原因 |
| `plots/*.png` | RMSE 曲线、可见相机数、轨迹俯视图 |

### 6）diagnostics.csv 常见失败原因

| 原因 | 说明 | 解决方法 |
|------|------|----------|
| `insufficient_fixed_observations` | 可用固定相机数量不足 | 降低 `min_fixed_observations`，或检查图像是否有 tag |
| `insufficient_world_tags` | 固定相机融合后可用 tag 数不足 | 降低 `apriltag_min_tags`，或增大 tag 尺寸/数量 |
| `target_failed:tags_not_found` | 头戴相机该帧未检测到 tag | 检查头戴相机视野是否覆盖 tag |
| `target_failed:few_tags` | 头戴相机可用 tag 数不足 | 降低 `apriltag_min_tags` |
| `fixed_failed:opencv_has_no_aruco` | OpenCV 缺少 aruco 模块 | 安装 `opencv-contrib-python` |

### 7）常见问题

- **位姿尺度异常**：检查 `apriltag_default_size_m` 是否与实际打印尺寸一致；检查外参 `translation` 单位，必要时启用 `use_mm_to_m_auto_scale: true`。
- **姿态方向翻转**：尝试切换 `fixed_extrinsics_are_twc: false`（外参可能是 `T_c_w` 而非 `T_w_c`）。
- **大量帧失败**：先用 `frame_policy: target_primary` 确认头戴相机帧是否都有对应固定相机帧；再检查 tag 是否在固定相机视野内。
