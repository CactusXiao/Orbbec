# Camera Info

This folder stores the active ego-camera calibration used by the PC-side
post-processing and extrinsic-estimation tools.

Tracked calibration artifacts:

- `fisheye_calibration_result.npz`: OpenCV fisheye matrices consumed by Python tools.
- `fisheye_calibration_result.yaml`: human-readable calibration parameters.
- `fisheye_calibration_summary.json`: calibration diagnostics and source-frame statistics.

To regenerate the calibration from checkerboard images, use either platform
server package.

Windows:

```powershell
python outside/stream_server_windows/calibrate_fisheye_camera.py `
  --image-dir path\to\checkerboard_frames `
  --output-dir outside/camera_info
```

Ubuntu:

```bash
python outside/stream_server_ubuntu/calibrate_fisheye_camera.py \
  --image-dir path/to/checkerboard_frames \
  --output-dir outside/camera_info
```

Local historical copies may be kept in sibling directories named
`camera_info_old_<index>`; those snapshots are intentionally ignored by Git.
