#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/setup_adb_reverse.sh" --port 50051
python "$SCRIPT_DIR/server.py" --host 127.0.0.1 --port 50051 --output-root "$SCRIPT_DIR/sessions"

# Inside the server console:
# timecalibrate
# start <session_name>
# start automatically runs timecalibrate before capture.

# After capture:
# python "$SCRIPT_DIR/fused_gaze_pipeline.py" --session-dir "$SCRIPT_DIR/sessions/<session_name>" --depth 1.0 --crop-size 1280x960
# python "$SCRIPT_DIR/fused_gaze_pipeline.py" --debug --session-dir "$SCRIPT_DIR/sessions/<session_name>" --depths 0.6,1.0,1.5 --crop-size 1280x960 --debug-output-raw --debug-output-undistorted --marker-diameter 20
