# PICO Ego Streaming Server - Ubuntu

This folder contains the Ubuntu PC-side server for the PICO client/server streaming capture mode.

The protocol implementation is the same as the Windows server. Ubuntu-specific setup lives in Bash scripts and package installation commands.

## System Packages

Install Python, ADB, and FFmpeg:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip android-tools-adb ffmpeg
```

## Environment

From the Unity project root:

```bash
bash outside/stream_server_ubuntu/install_env.sh
source outside/stream_server_ubuntu/.venv/bin/activate
```

## Transport

The Ubuntu server uses ADB reverse TCP over USB:

```bash
bash outside/stream_server_ubuntu/setup_adb_reverse.sh --port 50051
```

The PICO Unity client connects to `127.0.0.1:50051`. With `adb reverse`, that connection is forwarded through USB to this Ubuntu server.

## Start Server

```bash
python outside/stream_server_ubuntu/server.py \
  --host 127.0.0.1 \
  --port 50051 \
  --output-root outside/stream_server_ubuntu/sessions
```

Then launch the PICO app that has `EgoStreamingClientController` attached in the scene.

## Commands

Inside the server console:

```text
start test_001
timecalibrate
timecalibrate 30
status
stop
quit
```

`timecalibrate [sample_count]` estimates the PICO Unix clock to host Unix clock offset using an NTP-style exchange. `start <session_name>` automatically runs `timecalibrate` first; if calibration fails, capture is not started. `stop` asks the client to flush the HEVC encoder and end the session.

## Output

Each session directory contains:

- `video.h265`: HEVC elementary stream.
- `metadata.csv`: per-frame timestamps, pose, gaze, and camera metadata.
- `timestamps.csv`: lightweight timestamp table for multi-camera alignment.
- `camera.json`: camera parameters and encoder configuration.
- `network_log.jsonl`: packet/sample diagnostics.
- `session.json`: final counts and summary.
- `time_calibration.json`: host/PICO Unix time calibration snapshot captured before `start`.

The server also keeps the latest calibration at:

- `outside/stream_server_ubuntu/sessions/time_calibration_latest.json`

## Decode Check

If FFmpeg is installed:

```bash
ffplay -f hevc video.h265
```

or:

```bash
ffmpeg -f hevc -i video.h265 -c copy video.mp4
```

To decode a session into per-frame JPGs and write a metadata-aligned index:

```bash
python outside/stream_server_ubuntu/decode_h265_to_jpg.py \
  --session-dir outside/stream_server_ubuntu/sessions/new2
```

When system `ffmpeg` is not on `PATH`, the script automatically tries Python `imageio_ffmpeg`. Output is written to `decoded_jpg/` inside the session by default.

Precise per-frame timing is stored in `metadata.csv` and `timestamps.csv`; the raw `.h265` stream itself is not the source of truth for synchronization.

## Camera Decode

Use `decode_camera_data.py` as the stable entrypoint for camera stream decoding:

```bash
python outside/stream_server_ubuntu/decode_camera_data.py \
  --session-dir outside/stream_server_ubuntu/sessions/new2
```

This creates:

- `decoded_jpg/frame_000000.jpg`
- `decoded_jpg/decoded_index.csv`
- `decoded_jpg/decode_summary.json`
- `decoded_jpg/decoded.mp4`, unless `--no-video` is used

When `network_log.jsonl` is available, `decoded_index.csv` maps each decoded
image to `metadata.csv` using the HEVC sample's `frame_index`. Codec
configuration and partial MediaCodec buffers are excluded. A camera attempt
explicitly marked as not submitted to the encoder remains unmapped instead of
shifting every later image by one row. A row that says it was submitted but has
no HEVC frame is still reported as real video loss. For older sessions without
a network log, row-order mapping is accepted only when decoded and metadata
frame counts match exactly.

## One-Command Fused Gaze Output

Use `fused_gaze_pipeline.py` for the final dataset-style output. It takes a captured session plus an existing fisheye calibration, decodes `video.h265`, applies OpenCV fisheye undistortion, center-crops the image, projects fixed-depth gaze, and writes H.265 MP4 plus a compact CSV.

Normal mode requires exactly one depth and does not visualize gaze:

```bash
python outside/stream_server_ubuntu/fused_gaze_pipeline.py \
  --session-dir outside/stream_server_ubuntu/sessions/new2 \
  --depth 1.0 \
  --crop-size 1280x960
```

Normal output is written to:

- `fused_gaze/fixed_depth_1m/cropped_undistorted_h265.mp4`
- `fused_gaze/fixed_depth_1m/gaze_uv.csv`
- `fused_gaze/fixed_depth_1m/process_summary.json`
- `fused_gaze/fused_gaze_summary.json`

`gaze_uv.csv` has one row per MP4 frame that was written. Key fields:

- `mp4_frame_index`
- `mp4_timestamp_s`
- `source_frame_index`
- `source_ref_timestamp_us`
- `source_pico_frame_timestamp_ns`
- `gaze_status`
- `pixel_x`, `pixel_y`
- `uv_image_u`, `uv_image_v_top`, `uv_unity_v_bottom`

When gaze lands outside the final cropped image, `gaze_status` is `outside` and the pixel/UV fields are recorded as `outside`. Invalid or failed gaze rows keep empty coordinate fields and record the reason in `gaze_status` / `projection_failure_reason`.

Debug mode supports multiple depths and visualizes gaze on the cropped MP4. It can also keep intermediate raw decoded frames and full-size undistorted frames:

```bash
python outside/stream_server_ubuntu/fused_gaze_pipeline.py \
  --debug \
  --session-dir outside/stream_server_ubuntu/sessions/new2 \
  --depths 0.6,1.0,1.5 \
  --crop-size 1280x960 \
  --debug-output-raw \
  --debug-output-undistorted \
  --marker-diameter 20
```

Debug output uses one folder per depth, for example:

- `fused_gaze/fixed_depth_0p6m/cropped_undistorted_gaze_h265.mp4`
- `fused_gaze/fixed_depth_0p6m/gaze_uv.csv`
- `fused_gaze/debug_intermediates/raw_decoded/`, only with `--debug-output-raw`
- `fused_gaze/debug_intermediates/undistorted_full/`, only with `--debug-output-undistorted`

Useful options:

- `--calibration outside/camera_info/fisheye_calibration_result.npz`: override the fisheye calibration file.
- `--output-dir path/to/output`: write outside the default `session/fused_gaze`.
- `--video-fps 30`: set MP4 frame rate and CSV `mp4_timestamp_s`.
- `--h265-crf 18`: H.265 quality; lower means larger and cleaner.
- `--include-invalid`: attempt projection even when `gaze_valid=false`.
- `--decoded-dir path/to/decoded_jpg`: reuse already decoded frames for quick testing; its `decoded_index.csv` is rebuilt and validated against the session before use.

## Fisheye Camera Calibration

The default calibration is shared through:

```text
outside/camera_info/fisheye_calibration_result.npz
```

It was copied from the reference project and uses OpenCV's fisheye model. Regenerate it when the camera, resolution, or calibration images change:

```bash
python outside/stream_server_ubuntu/calibrate_fisheye_camera.py \
  --image-dir path/to/checkerboard_frames \
  --pattern 11x8 \
  --square-size 0.03 \
  --output-dir outside/camera_info
```

The calibration tool writes `fisheye_calibration_result.npz`, `fisheye_calibration_result.yaml`, and `fisheye_calibration_summary.json`. Debug folders such as `corners_debug/` and `undistorted/` are generated outputs.

## Gaze UV Mapping And Visualization

After `decoded_jpg/` exists, project captured gaze onto the fisheye camera. The default final output is in `cropped` space: fisheye image -> OpenCV fisheye undistortion -> center crop. The default crop is `1280x960`.

```bash
python outside/stream_server_ubuntu/project_gaze_uv.py \
  --session-dir outside/stream_server_ubuntu/sessions/new2 \
  --projection-mode both \
  --crop-size 1280x960
```

Projection modes:

- `direction`: depth-independent angular mapping from gaze direction to image UV, based on the reference `gaze_direction_projection` method.
- `fixed-depth`: reconstructs the RGB camera pose from `xr_head_*` plus `camera.json` RGB extrinsics, then intersects the eye ray with depth planes.
- `both`: runs `direction` plus fixed-depth sweeps. Default depths are `0.6,0.8,1.0` meters; override with `--depths 0.5,1.0,1.5`.

Outputs are written under `session/gaze_projection/`:

- `direction/gaze_uv.csv`
- `direction/visualized_cropped/`
- `direction/gaze_video_cropped.mp4`
- `fixed_depth_<depth>m/...` for fixed-depth runs
- `projection_summary.json`

The CSV includes raw fisheye pixels/UVs, full undistorted pixels/UVs, and final cropped pixels/UVs. Use the `cropped_*` fields as the final gaze labels for the saved cropped images:

- `raw_pixel_x`, `raw_pixel_y`, `raw_uv_image_u`, `raw_uv_image_v_top`, `raw_uv_unity_v_bottom`
- `undistorted_pixel_x`, `undistorted_pixel_y`, `undistorted_uv_image_u`, `undistorted_uv_image_v_top`, `undistorted_uv_unity_v_bottom`
- `cropped_pixel_x`, `cropped_pixel_y`, `cropped_uv_image_u`, `cropped_uv_image_v_top`, `cropped_uv_unity_v_bottom`

If a gaze projection falls outside the final center-cropped image, the cropped visualization image is still saved but no red marker is drawn, and the corresponding `cropped_pixel_*` / `cropped_uv_*` CSV fields are recorded as `outside`.

Use `--spaces raw,undistorted,cropped` if you also want debug visualizations in the original fisheye image and the full undistorted image.

Useful options:

```bash
python outside/stream_server_ubuntu/project_gaze_uv.py \
  --session-dir outside/stream_server_ubuntu/sessions/new2 \
  --projection-mode fixed-depth \
  --depths 0.6,1.0,1.5 \
  --crop-size 1280x960 \
  --spaces cropped \
  --include-invalid \
  --max-rows 100
```

`--include-invalid` attempts projection even when `gaze_valid` is false, which is useful for debugging legacy gaze fields.
`--crop-size WIDTHxHEIGHT` controls the center crop after undistortion.

## One-Step Post Processing

Run camera decode and gaze UV visualization in one command:

```bash
python outside/stream_server_ubuntu/process_session.py \
  --session-dir outside/stream_server_ubuntu/sessions/new2 \
  --projection-mode both \
  --crop-size 1280x960
```

If a test session records legacy or diagnostic gaze rows with `gaze_valid=false`, add `--include-invalid` to visualize those rows for debugging:

```bash
python outside/stream_server_ubuntu/process_session.py \
  --session-dir outside/stream_server_ubuntu/sessions/new2 \
  --projection-mode direction \
  --crop-size 1280x960 \
  --include-invalid \
  --max-rows 120
```

## Notes

- `ref_timestamp_us` uses Unix epoch microseconds UTC.
- `pico_frame_timestamp_ns` is the PICO VST SDK internal camera timestamp and is not a Unix timestamp.
- The current Unity client uses Android `MediaCodec` HEVC. If the device encoder does not support the selected byte-buffer YUV420 format, the client reports an explicit error instead of falling back silently.
