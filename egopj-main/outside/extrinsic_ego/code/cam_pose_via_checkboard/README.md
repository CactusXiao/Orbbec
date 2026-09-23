# cam_pose_via_checkboard

使用多台**已标定固定相机**与 AprilTag 或棋盘格观测，估计鱼眼 ego 相机每一帧在世界坐标系下的位姿 `T_world_from_ego`。

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

- `trajectory_tw_cXX.txt`：每行 `frame_index + 16个矩阵元素(按行展开)`
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

## AprilTag Workflow（持久 Tag Map + 鱼眼 Ego）

项目不要求手工填写每个标签的世界位姿。首次运行会用多台固定相机的多帧观测建立
`tag_map.json`；标签保持不动时，后续 episode 直接复用该地图。

### 1) Config example

Edit `configs/default.yaml`:

```yaml
target_type: "apriltag"
apriltag_family: "tag36h11"   # tag16h5 | tag25h9 | tag36h10 | tag36h11
apriltag_default_size_m: 0.10   # physical tag size (meter)
apriltag_min_tags: 2
apriltag_min_inliers: 8
fixed_camera_model: "pinhole"
target_camera_model: "fisheye"
tag_map_filename: "tag_map.json"
rebuild_tag_map: false
frame_policy: "target_primary"
```

Notes:

- `apriltag_default_size_m` must match your real printed tag size.
- Current implementation assumes all tags use the same size.

### 2) Pipeline logic

1. 首次运行：固定相机多帧检测标签，结合已知 `T_world_from_fixed` 融合并保存世界系 tag map。
2. 每个 ego 帧在原始鱼眼图上检测 AprilTag。
3. 对检测角点执行鱼眼反畸变，再利用世界系 tag corners 做联合 RANSAC PnP。
4. 投回原始鱼眼图计算 RMSE，拒绝边缘区域的错误解。
5. 离线剔除孤立异常、补内部短缺口并做 SE(3) 平滑；长缺口保持无效。

主变换约定是 `p_world = T_world_from_ego @ p_ego`。兼容输出仍保留旧字段名 `T_w_c07`。

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
apriltag_min_tags: 2             # 每帧至少需要几个 tag 才能估计位姿
apriltag_min_inliers: 8          # PnP 最少内点数

fixed_camera_ids:
  - "00"
  - "01"
  - "02"
  - "03"
  - "04"
  - "05"
target_camera_id: "07"           # 头戴相机 ID
fixed_camera_model: "pinhole"
target_camera_model: "fisheye"
tag_map_filename: "tag_map.json"
rebuild_tag_map: false
frame_policy: "target_primary"   # 以头戴相机帧为主，缺失固定视角帧也保留
max_interp_gap: 5
```

> `apriltag_default_size_m` 必须与实际打印标签尺寸一致，否则估计的平移尺度会出错。

### 3）处理流程

1. 首次运行从固定相机多帧观测建立并保存 `tag_map.json`。
2. 后续运行加载固定 tag map；固定视角帧只用于可见性诊断，不逐帧改变地图。
3. 鱼眼 ego 检测标签后，对稀疏角点按鱼眼模型反畸变并做联合 PnP。
4. 在原始鱼眼图上计算 RMSE，获得 `T_world_from_ego_raw`。
5. 离线拒绝孤立异常、补短缺口并平滑，输出 `T_world_from_ego`；长缺口保持无效。

### 4）运行命令

```bash
# 激活虚拟环境后
python -m src.main --dataset_root /path/to/dataset_root --config configs/default.yaml
```

### 5）输出文件

输出到 `outputs/` 目录：

| 文件 | 内容 |
|------|------|
| `trajectory_tw_cXX.txt` | 每行：`帧索引 + 16个矩阵元素（按行展开）` |
| `matrices/frame_*.txt` | 每帧 4×4 位姿矩阵 |
| `diagnostics.csv` | 每帧质量信息与失败原因 |
| `trajectory.json` | raw/final pose、有效性、置信度、标签和重投影诊断 |
| `ego_extrinsics.json` | 有效帧的 4×4 外参；`p_ego = T_ego_from_world @ p_world`，世界系为相机 00 |
| `tag_map.json` | 固定 AprilTag 的持久世界地图 |
| `plots/*.png` | RMSE 曲线、可见相机数、轨迹俯视图 |

### 6）diagnostics.csv 常见失败原因

| 原因 | 说明 | 解决方法 |
|------|------|----------|
| `tag map build failed` | commissioning 阶段固定相机有效观测不足 | 检查固定相机外参、标签尺寸和可见性 |
| `target_failed:tags_not_found` | 头戴相机该帧未检测到 tag | 检查头戴相机视野是否覆盖 tag |
| `target_failed:few_tags` | 头戴相机可用 tag 数不足 | 优先改善标签覆盖；不建议为追求完整率降为单标签 |
| `invalid_long_gap` | 连续遮挡超过允许补帧长度 | 增加标签覆盖；该段不会伪造 pose |

### 7）常见问题

- **位姿尺度异常**：检查 `apriltag_default_size_m` 是否与实际打印尺寸一致；检查外参 `translation` 单位，必要时启用 `use_mm_to_m_auto_scale: true`。
- **姿态方向翻转**：尝试切换 `fixed_extrinsics_are_twc: false`（外参可能是 `T_c_w` 而非 `T_w_c`）。
- **大量帧失败**：先用 `frame_policy: target_primary` 确认头戴相机帧是否都有对应固定相机帧；再检查 tag 是否在固定相机视野内。

## 鱼眼内参标定

不要把普通 Brown-Conrady 畸变参数用于鱼眼 ego。准备至少 15 张有效棋盘图，棋盘需要覆盖画面中心、
四边、四角和不同倾角：

```bash
python -m src.fisheye_calibration \
  --images /path/to/calibration_images \
  --camera-id 08 --cols 9 --rows 6 --square-size-m 0.0255 \
  --output /path/to/ego_camera_params.json
```

程序使用 20% 图像做留出验证；默认留出重投影 P95 超过 2 px 时拒绝结果。将生成的相机块合并进
数据集 `camera_params.json`，并保持采集分辨率与标定分辨率一致。

## 推荐标签布置

- 标签使用哑光刚性背板，分散安装在桌面外围以及至少两个不同朝向的侧/立面。
- 常见头部姿态下争取同时看到 3 个标签且至少跨两个平面，避免所有标签集中或共面。
- 避开双手主要操作区和身体最常遮挡的方向，不同物理位置使用唯一 ID。
- 标签边长是检测角点对应的黑白边界间距，不是纸张外沿。
- 移动任何标签后设置 `rebuild_tag_map: true` 重建地图，成功后恢复为 `false`。

录一段覆盖典型头部运动和手臂遮挡的试采数据后，可量化比较不同摆法：

```bash
python -m src.layout_evaluator \
  --ego-rgb-dir /path/to/episode/08/RGB \
  --tag-map /path/to/episode/tag_map.json \
  --output /path/to/layout_report.json
```

报告包含 `at_least_3_tags_ratio`、`two_plane_ratio`，以及假设任意单个 tag 完全被遮挡后的最差双标签覆盖率。
