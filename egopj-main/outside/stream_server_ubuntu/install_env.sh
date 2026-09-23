#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-$SCRIPT_DIR/.venv}"
PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "python3 was not found. Install it with: sudo apt install python3 python3-venv python3-pip" >&2
    exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$SCRIPT_DIR/requirements.txt"
python -m py_compile \
  "$SCRIPT_DIR/server.py" \
  "$SCRIPT_DIR/decode_h265_to_jpg.py" \
  "$SCRIPT_DIR/decode_camera_data.py" \
  "$SCRIPT_DIR/calibrate_fisheye_camera.py" \
  "$SCRIPT_DIR/project_gaze_uv.py" \
  "$SCRIPT_DIR/process_session.py" \
  "$SCRIPT_DIR/fused_gaze_pipeline.py"

python - <<'PY'
import shutil
import subprocess
import sys

import cv2  # noqa: F401
import imageio_ffmpeg
import numpy  # noqa: F401

ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
completed = subprocess.run(
    [ffmpeg, "-hide_banner", "-encoders"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
    errors="replace",
)
encoder_text = completed.stdout + completed.stderr
if completed.returncode != 0 or "libx265" not in encoder_text:
    print(
        "ERROR: ffmpeg is available, but libx265 H.265 encoding was not found.\n"
        "Install an FFmpeg build with libx265 support before running fused_gaze_pipeline.py.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"Python dependencies OK. ffmpeg with libx265: {ffmpeg}")
PY

cat <<EOF

Ubuntu server environment is ready.
Activate with:
  source "$VENV_DIR/bin/activate"

Start server with:
  python "$SCRIPT_DIR/server.py" --host 127.0.0.1 --port 50051 --output-root "$SCRIPT_DIR/sessions"

Run fused post-processing after capture:
  python "$SCRIPT_DIR/fused_gaze_pipeline.py" --session-dir "$SCRIPT_DIR/sessions/<session_name>" --depth 1.0 --crop-size 1280x960
EOF
