# PICO Ego 串流服务器 - Windows

此文件夹包含用于 PICO 客户端/服务器串流采集模式的 Windows PC 端服务器。

## 传输方式

Windows 服务器通过 USB 使用 ADB reverse TCP：

```powershell
powershell -ExecutionPolicy Bypass -File outside/stream_server_windows/setup_adb_reverse.ps1 -Port 50051
```

PICO Unity 客户端连接到 `127.0.0.1:50051`。通过 `adb reverse`，该连接会经由 USB 转发到这台 PC 端服务器。

## 启动服务器

```powershell
python outside/stream_server_windows/server.py `
  --host 127.0.0.1 `
  --port 50051 `
  --output-root C:\A_Project\embody\unity\egopj\outside\stream_server_windows\sessions
```

然后启动场景中挂载了 `EgoStreamingClientController` 的 PICO 应用。

## 用于吞吐量测试的 Unity 客户端设置

在 `EgoStreamingClientController` 上，建议先使用以下设置：

```text
Target Fps = 30
HEVC Bitrate = 12000000
HEVC I Frame Interval Seconds = 1
HEVC Use Source Resolution = false
HEVC Width = 1280
HEVC Height = 960
HEVC Input Mode = Auto Direct Buffer
```

这会保持 VST 采集使用原始相机分辨率，但在送入 `MediaCodec` 之前会先对 NV21 帧进行降采样。若要复现全分辨率编码，请设置：

```text
HEVC Use Source Resolution = true
HEVC Width = 0
HEVC Height = 0
HEVC Input Mode = Auto Direct Buffer
```

`Auto Direct Buffer` 会使用 Unity 的 `AndroidJNI.NewDirectByteBuffer` 来处理原始分辨率帧，从而避免代价较高的 C# `byte[]` 到 Java `byte[]` 的桥接。如果直接输入失败，它会回退到旧版 Java 字节数组路径，并在 `session.json` 中记录该回退行为。若要进行严格的无回退测试，请设置 `HEVC Input Mode = Direct Buffer`。

Unity 项目必须允许 unsafe C# 代码，因为 `NewDirectByteBuffer` 需要接收一个原生指针。本仓库已在 `ProjectSettings/ProjectSettings.asset -> allowUnsafeCode: 1` 中启用该选项。

客户端会将详细的时序诊断信息写入 `metadata.csv` 以及最终的 `session.json`，包括：

- `downscale_ms`
- `encode_ms`
- `direct_buffer_create_ms`
- `encode_call_wall_ms`
- `encoder_dequeue_input_ms`
- `encoder_color_convert_ms`
- `encoder_input_put_ms`
- `encoder_queue_input_ms`
- `encoder_drain_output_ms`
- `encoder_java_total_ms`
- `codec_input_copy_or_convert_ms`
- `jni_bridge_ms`
- `network_send_queue_ms`
- `encoder_input_path`

如果 `jni_bridge_ms` 占主要耗时，说明客户端仍然在承担 Unity/Java 桥接开销，直接输入并未正常工作。如果 `encoder_dequeue_input_ms` 占主要耗时，说明硬件编码器消耗帧的速度不够快。如果 `downscale_ms` 占主要耗时，则应降低编码分辨率，或将缩放操作迁移到原生/GPU 路径中。

## 命令

在服务器控制台中执行：

```text
start test_001
timecalibrate
timecalibrate 30
status
stop
quit
```

`timecalibrate [sample_count]` 会通过类似 NTP 的握手估计 PICO Unix 时间到主机 Unix 时间的偏移。`start <session_name>` 会先自动运行一次 `timecalibrate`；如果校准失败，则不会开始采集。`stop` 会请求客户端刷新 HEVC 编码器并结束当前 session。

## 输出

每个 session 目录包含：

- `video.h265`：HEVC elementary stream，即 HEVC 基本码流。
- `metadata.csv`：逐帧时间戳、位姿、注视点和相机元数据。
- `timestamps.csv`：用于多相机对齐的轻量级时间戳表。
- `camera.json`：相机参数和编码器配置。
- `network_log.jsonl`：数据包/样本诊断信息。
- `session.json`：最终计数和汇总信息。
- `time_calibration.json`：`start` 前保存的主机/PICO Unix 时间校准快照。

服务器还会保留最近一次校准结果：

- `outside/stream_server_windows/sessions/time_calibration_latest.json`

## 解码检查

如果已安装 FFmpeg：

```powershell
ffplay -f hevc video.h265
```

或者：

```powershell
ffmpeg -f hevc -i video.h265 -c copy video.mp4
```

若要将一个 session 解码为逐帧 JPG，并写入与元数据对齐的索引：

```powershell
python outside/stream_server_windows/decode_h265_to_jpg.py `
  --session-dir outside/stream_server_windows/sessions/new2
```

当系统中的 `ffmpeg` 不在 `PATH` 中时，该脚本会自动尝试使用 Python 的 `imageio_ffmpeg`。默认情况下，输出会写入 session 目录内的 `decoded_jpg/`。

精确的逐帧时序存储在 `metadata.csv` 和 `timestamps.csv` 中；原始 `.h265` 码流本身并不是同步信息的可信来源。

## 相机解码

使用 `decode_camera_data.py` 作为相机流解码的稳定入口：

```powershell
python outside/stream_server_windows/decode_camera_data.py `
  --session-dir outside/stream_server_windows/sessions/new2
```

该命令会创建：

- `decoded_jpg/frame_000000.jpg`
- `decoded_jpg/decoded_index.csv`
- `decoded_jpg/decode_summary.json`
- `decoded_jpg/decoded.mp4`，除非使用了 `--no-video`

## 一键融合输出

最终整理数据时优先使用 `fused_gaze_pipeline.py`。该脚本输入一个采集 session 和已经标定好的鱼眼相机参数，自动解码 `video.h265`，执行鱼眼去畸变、中心裁剪、固定深度 gaze 投影，并输出 H265 MP4 和精简 CSV。PICO ego 相机标定不包含在这个融合脚本内。

普通模式只允许一个深度，不会把眼动点画到视频上：

```powershell
python outside/stream_server_windows/fused_gaze_pipeline.py `
  --session-dir outside/stream_server_windows/sessions/new2 `
  --depth 1.0 `
  --crop-size 1280x960
```

普通模式输出：

- `fused_gaze/fixed_depth_1m/cropped_undistorted_h265.mp4`
- `fused_gaze/fixed_depth_1m/gaze_uv.csv`
- `fused_gaze/fixed_depth_1m/process_summary.json`
- `fused_gaze/fused_gaze_summary.json`

`gaze_uv.csv` 每一行对应 MP4 中的一帧。常用字段包括：

- `mp4_frame_index`
- `mp4_timestamp_s`
- `source_frame_index`
- `source_ref_timestamp_us`
- `source_pico_frame_timestamp_ns`
- `gaze_status`
- `pixel_x`, `pixel_y`
- `uv_image_u`, `uv_image_v_top`, `uv_unity_v_bottom`

如果 gaze 投影落到最终裁剪画面之外，`gaze_status` 会记录为 `outside`，对应的 pixel/UV 字段也会记录为 `outside`。无效 gaze 或投影失败的行会保留空坐标，并在 `gaze_status` / `projection_failure_reason` 中记录原因。

debug 模式支持多个深度，并会在最终去畸变+中心裁剪后的 MP4 上画 gaze 点。也可以按需保留中间结果：

```powershell
python outside/stream_server_windows/fused_gaze_pipeline.py `
  --debug `
  --session-dir outside/stream_server_windows/sessions/new2 `
  --depths 0.6,1.0,1.5 `
  --crop-size 1280x960 `
  --debug-output-raw `
  --debug-output-undistorted `
  --marker-diameter 20
```

debug 模式每个深度输出一个文件夹，例如：

- `fused_gaze/fixed_depth_0p6m/cropped_undistorted_gaze_h265.mp4`
- `fused_gaze/fixed_depth_0p6m/gaze_uv.csv`
- `fused_gaze/debug_intermediates/raw_decoded/`，仅在使用 `--debug-output-raw` 时生成
- `fused_gaze/debug_intermediates/undistorted_full/`，仅在使用 `--debug-output-undistorted` 时生成

常用参数：

- `--calibration outside/camera_info/fisheye_calibration_result.npz`：指定鱼眼标定参数。
- `--output-dir path\to\output`：覆盖默认的 `session/fused_gaze` 输出位置。
- `--video-fps 30`：设置 MP4 帧率，也决定 CSV 中的 `mp4_timestamp_s`。
- `--h265-crf 18`：H265 质量参数，数值越低体积越大、质量越高。
- `--include-invalid`：即使 `gaze_valid=false` 也尝试投影。
- `--decoded-dir path\to\decoded_jpg`：复用已有解码图片，便于快速测试。

## 鱼眼相机标定

默认标定文件通过以下路径共享：

```text
outside/camera_info/fisheye_calibration_result.npz
```

该文件复制自参考项目，并使用 OpenCV 的鱼眼模型。当相机、分辨率或标定图像发生变化时，应重新生成该文件：

```powershell
python outside/stream_server_windows/calibrate_fisheye_camera.py `
  --image-dir path\to\checkerboard_frames `
  --pattern 11x8 `
  --square-size 0.03 `
  --output-dir outside/camera_info
```

标定工具会写入 `fisheye_calibration_result.npz`、`fisheye_calibration_result.yaml` 和 `fisheye_calibration_summary.json`。诸如 `corners_debug/` 和 `undistorted/` 这类调试文件夹属于生成输出。

## 注视 UV 映射与可视化

在 `decoded_jpg/` 已存在之后，可以将采集到的注视点投影到鱼眼相机图像上。默认最终输出位于 `cropped` 空间中：鱼眼图像 -> OpenCV 鱼眼去畸变 -> 中心裁剪。默认裁剪尺寸为 `1280x960`。

```powershell
python outside/stream_server_windows/project_gaze_uv.py `
  --session-dir outside/stream_server_windows/sessions/new2 `
  --projection-mode both `
  --crop-size 1280x960
```

投影模式：

- `direction`：与深度无关的角度映射，根据注视方向将其映射到图像 UV，基于参考的 `gaze_direction_projection` 方法。
- `fixed-depth`：根据 `xr_head_*` 加上 `camera.json` 中的 RGB 外参重建 RGB 相机位姿，然后计算眼部射线与不同深度平面的交点。
- `both`：同时运行 `direction` 和 fixed-depth 深度扫描。默认深度为 `0.6,0.8,1.0` 米；可通过 `--depths 0.5,1.0,1.5` 覆盖。

输出会写入 `session/gaze_projection/` 下：

- `direction/gaze_uv.csv`
- `direction/visualized_cropped/`
- `direction/gaze_video_cropped.mp4`
- `fixed_depth_<depth>m/...`，用于 fixed-depth 运行结果
- `projection_summary.json`

CSV 包括原始鱼眼像素/UV、完整去畸变后的像素/UV，以及最终裁剪后的像素/UV。对于保存下来的裁剪图像，应使用 `cropped_*` 字段作为最终的注视标签：

- `raw_pixel_x`, `raw_pixel_y`, `raw_uv_image_u`, `raw_uv_image_v_top`, `raw_uv_unity_v_bottom`
- `undistorted_pixel_x`, `undistorted_pixel_y`, `undistorted_uv_image_u`, `undistorted_uv_image_v_top`, `undistorted_uv_unity_v_bottom`
- `cropped_pixel_x`, `cropped_pixel_y`, `cropped_uv_image_u`, `cropped_uv_image_v_top`, `cropped_uv_unity_v_bottom`

如果某个注视投影落在最终中心裁剪图像之外，裁剪后的可视化图像仍会被保存，但不会绘制红色标记，并且对应的 `cropped_pixel_*` / `cropped_uv_*` CSV 字段会记录为 `outside`。

如果还希望在原始鱼眼图像和完整去畸变图像中生成调试可视化结果，可以使用 `--spaces raw,undistorted,cropped`。

常用选项：

```powershell
python outside/stream_server_windows/project_gaze_uv.py `
  --session-dir outside/stream_server_windows/sessions/new2 `
  --projection-mode fixed-depth `
  --depths 0.6,1.0,1.5 `
  --crop-size 1280x960 `
  --spaces cropped `
  --include-invalid `
  --max-rows 100
```

`--include-invalid` 会在 `gaze_valid` 为 false 时仍然尝试投影，这对于调试旧版注视字段很有用。

`--crop-size WIDTHxHEIGHT` 控制去畸变后的中心裁剪尺寸。

## 一步式后处理

使用一个命令完成相机解码和注视 UV 可视化：

```powershell
python outside/stream_server_windows/process_session.py `
  --session-dir outside/stream_server_windows/sessions/new2 `
  --projection-mode both `
  --crop-size 1280x960
```

如果某个测试 session 中记录了旧版或诊断用的注视行，且 `gaze_valid=false`，可以添加 `--include-invalid` 来可视化这些行以进行调试：

```powershell
python outside/stream_server_windows/process_session.py `
  --session-dir outside/stream_server_windows/sessions/new2 `
  --projection-mode direction `
  --crop-size 1280x960 `
  --include-invalid `
  --max-rows 120
```

## 说明

- `ref_timestamp_us` 使用 Unix epoch 的 UTC 微秒时间戳。
- `pico_frame_timestamp_ns` 是 PICO VST SDK 内部的相机时间戳，不是 Unix 时间戳。
- 当前 Unity 客户端使用 Android `MediaCodec` HEVC。如果设备编码器不支持所选的 byte-buffer YUV420 格式，客户端会报告明确错误，而不是静默回退。
