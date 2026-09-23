# Outside Tools

This directory contains the PC-side tools for PICO client/server streaming capture.

## Server Layout

- `stream_server_windows/`: Windows server package, including PowerShell ADB reverse setup.
- `stream_server_ubuntu/`: Ubuntu server package, including Bash ADB reverse setup and virtual environment setup.
- `camera_info/`: shared fisheye calibration files used by both server packages.

The Windows and Ubuntu servers intentionally keep their own copies of the runtime and post-processing scripts so either folder can be moved or used independently. Keep protocol and post-processing changes mirrored in both folders.

## Quick Start

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File outside/stream_server_windows/setup_adb_reverse.ps1 -Port 50051
python outside/stream_server_windows/server.py --host 127.0.0.1 --port 50051 --output-root outside/stream_server_windows/sessions
```

Ubuntu:

```bash
bash outside/stream_server_ubuntu/setup_adb_reverse.sh --port 50051
python outside/stream_server_ubuntu/server.py --host 127.0.0.1 --port 50051 --output-root outside/stream_server_ubuntu/sessions
```

## Post Processing

Both platform folders provide the same post-processing entrypoints:

- `decode_camera_data.py`: wrapper around `decode_h265_to_jpg.py`; decodes `video.h265` into `decoded_jpg/`, writes `decoded_index.csv`, and optionally writes `decoded.mp4`.
- `calibrate_fisheye_camera.py`: OpenCV fisheye checkerboard calibration; writes `outside/camera_info/fisheye_calibration_result.npz`.
- `fused_gaze_pipeline.py`: one-command production pipeline. It decodes `video.h265`, applies fisheye undistortion and center crop, projects fixed-depth gaze, and writes H.265 MP4 plus a compact gaze CSV.
- `project_gaze_uv.py`: gaze UV mapping and visualization using the fisheye calibration, PICO RGB camera extrinsics from `camera.json`, fisheye undistortion, and a center crop. The final/default UV space is the undistorted center-cropped image.
- `process_session.py`: one-step camera decode plus gaze projection/visualization.

Windows:

```powershell
python outside/stream_server_windows/fused_gaze_pipeline.py `
  --session-dir outside/stream_server_windows/sessions/<session_name> `
  --depth 1.0 `
  --crop-size 1280x960
```

Ubuntu:

```bash
python outside/stream_server_ubuntu/fused_gaze_pipeline.py \
  --session-dir outside/stream_server_ubuntu/sessions/<session_name> \
  --depth 1.0 \
  --crop-size 1280x960
```

Default calibration is read from `outside/camera_info/fisheye_calibration_result.npz`, copied from the reference project. Regenerate it with `calibrate_fisheye_camera.py` when camera hardware, capture resolution, or fisheye calibration images change.

By default, `fused_gaze_pipeline.py` writes `session/fused_gaze/fixed_depth_<depth>m/cropped_undistorted_h265.mp4` and `gaze_uv.csv`. In debug mode it writes one output folder per depth and adds gaze-point visualization to the cropped MP4.

All final UVs are produced in `cropped` space: fisheye image -> OpenCV fisheye undistortion -> center crop. The default crop is `1280x960`.
