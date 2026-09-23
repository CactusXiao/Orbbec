# PICO Ego 外参估计说明

本文档说明 `estimate_pico_ego_extrinsics.py` 的基本原理、输出文件含义和使用方法。

## 1. 目标和坐标约定

本程序用于估计 PICO ego RGB 相机相对于固定第三视角相机的外参。默认参考相机是 `00`，因为当前 `extrinsics.json` 中 `00` 的外参是单位矩阵，因此可以把 `00` 看作 reference/world 坐标系。

输出矩阵的约定是：

```text
p_ego = T_ego_from_reference * p_reference
```

也就是说，输出的 4x4 矩阵 `T_ego_from_reference` 会把 `00` 相机坐标系下的三维点变换到 PICO ego RGB 相机坐标系下。

如果你需要“ego 相机在 `00` 坐标系中的位姿”，需要对矩阵求逆：

```text
T_reference_from_ego = inverse(T_ego_from_reference)
```

本程序只估计 PICO ego RGB 相机的外参，不估计 headset/head 坐标系外参。

## 2. 基本原理

程序分为三步：建立固定 AprilTag 参考、逐帧估计 ego 外参、对遮挡帧插值。

第一步，以固定第三视角参考相机 `00` 建立 AprilTag 参考地图。

- 从 `camera_params.json` 读取固定相机内参。
- 从 `extrinsics.json` 读取固定相机之间的外参。
- 默认使用 `timestamps.csv` 中前 10 个已对齐的静态帧。
- 在固定相机图像中检测 AprilTag。
- 默认只使用 `00` 的观测确定 tag 坐标，避免其他固定相机的外参误差把参考地图拉偏。
- 如需显式测试所有固定相机的联合优化，可设置 `--reference-map-mode multiview`。

第二步，逐帧估计 PICO ego 外参。

- `optimized_pose` 和输出外参统一使用 reference `frame_index` 作为键。
- ego RGB 始终通过 `timestamps.csv` 的 `frame_index -> ego_frame_index` 映射读取原始 PICO 帧，
  不再自动猜测两种索引空间。
- 默认使用 `outside/camera_info/fisheye_calibration_result.npz` 对 ego 图像做鱼眼去畸变。
- 在 ego 图像中检测 AprilTag。
- 将 ego 图像中的 tag 角点与第一步得到的参考坐标系 3D tag 角点匹配。
- 使用 OpenCV `solvePnPRansac` 求解 `T_ego_from_reference`。

第三步，处理直接估计失败的帧。

- 如果某一帧能直接通过 AprilTag + PnP 求解，则记为 `direct`。
- 如果某一段连续帧因为遮挡、tag 不足、PnP 失败等原因无法直接估计，但这段前后都有 `direct` 帧，则全部插值，记为 `interpolated`。
- 插值时，平移使用线性插值，旋转使用 quaternion slerp。
- 如果开头或结尾没有双侧真实值，默认写入 `NaN`，例如 `nan_unbracketed_end`。
- 如果运行时设置 `--unbounded-gap-mode extrapolate`，开头或结尾无法夹住的段会使用边界处相邻两个 `direct` 位姿做外推，记为 `extrapolated_start` 或 `extrapolated_end`。

## 3. 视频读取和临时解码

程序不依赖 `episode_1_mp4/*_rgb.mp4`。

优先使用 OpenCV 顺序读取原始 H.265：

```text
<episode>/00/RGB/rgb.h265
...
<episode>/05/RGB/rgb.h265
<episode>/ego/RGB/rgb.h265
```

如果某个 H.265 文件无法被 OpenCV 直接读取，程序会自动调用系统 `ffmpeg` 或 Python `imageio_ffmpeg`，把它临时 remux 成 MP4 后再读取。临时文件只用于本次运行，程序结束后会自动删除。

如果历史数据的 H.265 已被删除，但同一个标准 `RGB/` 目录中存在按原始帧号命名的
`00000.jpg`、`00001.jpg` 等图片，程序会读取该图片序列。无论底层是 H.265 还是图片，
ego 帧都严格使用 `timestamps.csv` 中的 `ego_frame_index`；固定相机帧使用 `frame_index`。

## 4. 使用方法

在 Unity 项目根目录运行：

```powershell
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py `
  --episode-dir outside\extrinsic_ego\test_sample_final\hand_shape_calibration\episode_1
```

常用参数：

```powershell
# 推荐的高精度方案：96 mm 标签、全序列均匀采样、鲁棒多视角参考地图
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py `
  --episode-dir outside\extrinsic_ego\test_sample_final\color_tags_0\hand_shape_calibration\episode_1 `
  --output-dir outside\extrinsic_ego\test_sample_final\color_tags_0\hand_shape_calibration\episode_1\ego_extrinsics_pico_sync_newcalib_multiview60 `
  --reference-camera-id 00 `
  --static-camera-ids 00,01,02,03,04,05 `
  --reference-map-mode multiview `
  --reference-frame-count 60 `
  --reference-frame-sampling uniform `
  --reference-robust-huber-delta-px 1.5 `
  --reference-balance-cameras `
  --reference-observation-outlier-px 2.5 `
  --reference-tag-ids 2,3,4,7,8,9 `
  --tag-size-m 0.096 `
  --fisheye-calibration outside\camera_info\fisheye_calibration_result.npz `
  --unbounded-gap-mode extrapolate `
  --smooth-trajectory `
  --smoothing-window 11 `
  --smoothing-polyorder 3

# 只跑前 50 个已对齐 ego 帧，适合快速测试
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py `
  --episode-dir outside\extrinsic_ego\test_sample_final\hand_shape_calibration\episode_1 `
  --max-rows 50

# 指定输出目录
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py `
  --episode-dir outside\extrinsic_ego\test_sample_final\hand_shape_calibration\episode_1 `
  --output-dir outside\extrinsic_ego\test_sample_final\hand_shape_calibration\episode_1\ego_extrinsics_pico

# 强制测试临时 remux 路径
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py `
  --episode-dir outside\extrinsic_ego\test_sample_final\hand_shape_calibration\episode_1 `
  --max-rows 5 `
  --force-temp-remux

# 关闭 ego 鱼眼去畸变，仅用于调试
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py `
  --episode-dir outside\extrinsic_ego\test_sample_final\hand_shape_calibration\episode_1 `
  --disable-fisheye-undistort

# 对首尾无法夹住的段使用外推，而不是 NaN
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py `
  --episode-dir outside\extrinsic_ego\test_sample_final\hand_shape_calibration\episode_1 `
  --unbounded-gap-mode extrapolate

# 估计完成后额外生成可视化视频，默认不开启
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py `
  --episode-dir outside\extrinsic_ego\test_sample_final\hand_shape_calibration\episode_1 `
  --write-visualization-video
```

默认输出目录是：

```text
<episode>/ego_extrinsics_pico/
```

## 5. 输出文件含义

输出目录中主要有 6 个外参相关文件。

### ego_extrinsics_aligned.csv

最终建议使用的结果表。每一行对应 `timestamps.csv` 中一个存在 `ego_frame_index` 的对齐帧。

关键字段：

```text
row_index
```

该行在原始 `timestamps.csv` 中的行号。

```text
frame_index
```

第三视角固定相机的对齐帧号，也就是 reference 侧帧号。

```text
ref_timestamp_us
```

reference 时间戳，单位是 Unix epoch microseconds。

```text
ego_frame_index
```

PICO ego 视频中的帧号。

```text
ego_timestamp_us
```

PICO ego 帧时间戳，单位是 Unix epoch microseconds。

```text
status_initial
```

第一轮直接估计状态。

常见取值：

```text
ok
```

直接估计成功。

```text
no_detection
```

ego 图像中没有检测到 AprilTag。

```text
insufficient_tags
```

检测到了 tag，但能和参考地图匹配的 tag 数量不足。

```text
few_inliers
```

PnP RANSAC 内点角点数不足。

```text
pnp_failed
```

PnP 求解失败。

```text
high_reproj_error
```

PnP 求解出来了，但重投影误差超过阈值。

```text
image_missing
```

视频中没有读到对应帧。

```text
status_final
```

第二轮处理后的最终状态。

常见取值：

```text
direct
```

这一帧直接由 AprilTag + PnP 得到外参。

```text
interpolated
```

这一帧直接估计失败，但前后都有真实 `direct` 帧，因此由插值得到。

```text
nan_unbracketed_start
```

开头部分没有前侧真实值，无法插值。

```text
nan_unbracketed_end
```

结尾部分没有后侧真实值，无法插值。

```text
source
```

最终结果来源，通常是：

```text
direct
interpolated
nan
```

```text
detection_space
```

ego AprilTag 检测使用的图像空间。

常见取值：

```text
undistorted_image
```

在去畸变后的 ego 图像中检测到 tag。

```text
raw_fisheye_corners_to_undistorted
```

在原始鱼眼图像中检测到更多 tag，然后把角点映射到去畸变图像坐标，再执行 PnP。

```text
detected_tag_ids
```

该帧检测到的 tag ID。

```text
used_tag_ids
```

该帧真正参与 PnP 的 tag ID。只有同时存在于参考地图中的 tag 才会被使用。

```text
rmse_px
```

PnP 重投影误差，单位是像素。只对直接估计帧有意义。

```text
prev_direct_frame_index
next_direct_frame_index
interp_alpha
interp_gap_frames
```

插值帧的来源信息。`interp_alpha` 是当前帧在前后 direct 帧之间的插值比例。

```text
m00 ... m33
```

最终 4x4 外参矩阵 `T_ego_from_reference` 的 16 个元素。

### ego_extrinsics_direct_pass.csv

第一轮直接估计诊断表。它不包含插值结果，适合用来检查哪些帧真的通过 AprilTag + PnP 求解成功。

如果你只想使用绝对可靠的直接观测帧，可以使用这个文件中 `status_initial=ok` 的行。

### ego_extrinsics_aligned.json

和 `ego_extrinsics_aligned.csv` 含义一致，但以 JSON 结构保存。每一项包含：

```text
frame_index
ego_frame_index
status_initial
status_final
source
detected_tag_ids
used_tag_ids
rmse_px
T_ego_from_reference
```

注意：如果某一帧不可用，矩阵中会出现 `NaN`。这对 Python 的 `json` 模块可读，但不是严格标准 JSON。如果要给严格 JSON 解析器使用，建议在后处理时把 `NaN` 转成 `null`。

### ego_extrinsics_pose_dict.json

接近 Orbbec 版 `ego_calibration.py` 的最终保存形式，是一个简洁的字典：

```text
{
  "frame_index": [[4x4 T_ego_from_reference]],
  ...
}
```

键使用 `timestamps.csv` 中的 `frame_index`，也就是第三视角 reference 侧帧号。

### ego_extrinsics_pose_dict_by_ego_frame.json

和 `ego_extrinsics_pose_dict.json` 相同，但键使用 `ego_frame_index`。如果后续处理主要按 PICO ego 视频帧号索引，可以使用这个文件。

对于已经同步并按 reference 帧号保存的 `ego/RGB/*.jpg`、`optimized_pose/*.npy`，
应使用 `ego_extrinsics_pose_dict.json`。只有直接读取原始 ego 视频时才应优先使用
`ego_extrinsics_pose_dict_by_ego_frame.json`；同步表中的重复 ego 帧号会使后者发生覆盖。

### ego_extrinsics_summary.json

本次运行的总览文件，记录：

- 输入 episode 路径。
- 输出路径。
- 参考相机 ID。
- 使用了哪些固定相机。
- 使用了哪些 reference frames。
- 检测并保留下来的 AprilTag ID。
- ego 是否启用鱼眼去畸变。
- `unbounded_gap_mode`。
- direct/interpolated/extrapolated/nan 帧数统计。
- PnP 重投影误差统计。
- 视频读取是否触发临时 remux。
- 输出文件路径。

## 6. 当前 sample 的结果解读

使用 96 mm 标签和鲁棒多视角参考地图后，
`episode_1/ego_extrinsics_pico_sync_newcalib_multiview60` 的完整运行结果为：

```text
timestamp_row_count = 437
ego_frame_index_space = ego
reference_map_mode = multiview
reference_frame_sampling = uniform
direct = 437
interpolated = 0
extrapolated = 0
nan = 0
reference_tag_count = 6
median_rmse_px = 0.817
```

这表示：

- `timestamps.csv` 的 455 个 reference 时刻中有 437 个具有有效 PICO 对应帧。
- 437 帧全部直接看到了足够的 AprilTag，并通过 PnP 得到外参。
- 没有使用插值、外推或 `NaN` 补帧。

当前使用的参考 tag ID 是：

```text
2, 3, 4, 7, 8, 9
```

当前 ego 鱼眼去畸变标定来自：

```text
outside/camera_info/fisheye_calibration_result.npz
```

## 7. 推荐使用方式

如果后续算法需要逐帧 ego 外参，优先读取：

```text
ego_extrinsics_aligned.csv
```

建议规则：

- 使用 `source=direct` 和 `source=interpolated` 的行。
- 跳过 `source=nan` 的行。
- 如果只接受直接观测，不接受插值，则只使用 `source=direct` 的行。
- 如果需要把 ego 坐标系中的结果投到 `00` 坐标系，需要使用矩阵逆：

```text
T_reference_from_ego = inverse(T_ego_from_reference)
```

## 8. 代码验证

语法检查：

```powershell
C:\App_install\Conda\install\envs\all\python.exe -m py_compile outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py
```

快速小样本：

```powershell
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py `
  --episode-dir outside\extrinsic_ego\test_sample_final\hand_shape_calibration\episode_1 `
  --max-rows 50
```

完整运行：

```powershell
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\estimate_pico_ego_extrinsics.py `
  --episode-dir outside\extrinsic_ego\test_sample_final\hand_shape_calibration\episode_1
```

## 9. MANO ego 可视化

`optimized_pose` 和 `ego_extrinsics_pose_dict*.json` 都按同步 reference `frame_index` 编号；
原始 ego 图像按 PICO `ego_frame_index` 编号。`ego_pose.py` 会读取 `timestamps.csv` 完成二者
映射，再把 MANO 手部网格或 21 个关节点投影回正确时刻的原始 ego 鱼眼图像。

额外依赖为 `torch`、`smplx`、`scipy`、`chumpy` 和 `six`。旧版 `chumpy`
在新版本 Python 中建议使用：

```powershell
C:\App_install\Conda\install\envs\all\python.exe -m pip install torch smplx scipy six
C:\App_install\Conda\install\envs\all\python.exe -m pip install chumpy --no-build-isolation
```

Windows 下直接运行 Python：

```powershell
C:\App_install\Conda\install\envs\all\python.exe outside\extrinsic_ego\code\ego_pose.py `
  --data-root outside\extrinsic_ego\test_sample_final `
  --subject color_tags_0 `
  --episode hand_shape_calibration/episode_1 `
  --extrinsics outside\extrinsic_ego\test_sample_final\color_tags_0\hand_shape_calibration\episode_1\ego_extrinsics_pico_fixed\ego_extrinsics_pose_dict.json `
  --output-dir outside\extrinsic_ego\test_sample_final\color_tags_0\hand_shape_calibration\episode_1\ego_mesh_visualization_pico_fixed_frames `
  --output-video outside\extrinsic_ego\test_sample_final\color_tags_0\hand_shape_calibration\episode_1\ego_mesh_visualization_pico_fixed.mp4 `
  --start 30 `
  --end 100 `
  --type mesh
```

在 Bash/Git Bash/Linux 中，`viz_ego_mesh.sh` 会自动查找本项目的 `all`
环境、原始 `groundhand` 环境或系统 `python3`，也可以通过环境变量
`PYTHON_BIN` 显式指定解释器。其余参数与上面的 Python 命令相同。

常用选项：

- `--type mesh`：半透明网格、三角边线和关节点骨架。
- `--type kp2d`：只显示鱼眼投影后的 21 个关节点骨架。
- `--video-only`：只生成 `--output-video`，不保存逐帧 JPG。
- `--extrinsics`：显式选择新估计结果；不指定时仍读取 episode 原有的 `ego_extrinsic.json`。

`ego_pose.py` 优先顺序解码标准位置 `ego/RGB/rgb.h265`；历史数据没有 H.265 时，读取同一
目录下按原始 PICO 帧号命名的 JPG/PNG。两种输入都强制使用
`timestamps.csv: frame_index -> ego_frame_index`，视频模式不需要预先导出中间图片。

## 10. Savitzky–Golay 外参轨迹平滑

`--smooth-trajectory` 会在逐帧 PnP 和缺失帧插值之后，额外生成一份平滑轨迹。
原始 `ego_extrinsics_pose_dict.json`、CSV 和 JSON 均保持不变；启用该选项需要安装 `scipy`。

平滑器参考 `smooth_pose(1).py`：对 `T_ego_from_reference` 的旋转矩阵和平移分别沿时间轴执行
Savitzky–Golay 滤波。滤波后的旋转通过 6D 旋转表示重新投影到合法旋转矩阵。默认窗口为
11 帧、三阶多项式，可分别通过
`--smoothing-window` 和 `--smoothing-polyorder` 调整。连续有效位姿不足一个窗口时保持原值。

新增输出：

```text
ego_extrinsics_smoothed.csv
ego_extrinsics_smoothed.json
ego_extrinsics_pose_dict_smoothed.json
ego_extrinsics_pose_dict_smoothed_by_ego_frame.json
```

MANO 可视化应通过 `--extrinsics` 显式读取
`ego_extrinsics_pose_dict_smoothed.json`。平滑前后的位姿修正量、二阶平移差分和旋转增量差
会记录在 `ego_extrinsics_summary.json/trajectory_smoothing` 中。
