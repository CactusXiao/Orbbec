#!/usr/bin/env bash
# Example:
# bash outside/extrinsic_ego/code/viz_ego_mesh.sh \
#   --data-root /mnt/nas/gezuhao/wuchao/data/glove \
#   --subject color_tags_0 \
#   --episode hand_shape_calibration/episode_1 \
#   --output-dir experiments/viz_result/ego \
#   --start 30 \
#   --end 100 \
#   --type mesh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/c/App_install/Conda/install/envs/all/python.exe" ]]; then
    PYTHON_BIN="/c/App_install/Conda/install/envs/all/python.exe"
  elif [[ -x "$(dirname "$TOOLKIT_ROOT")/groundhand/hand/bin/python" ]]; then
    PYTHON_BIN="$(dirname "$TOOLKIT_ROOT")/groundhand/hand/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "error: no Python interpreter found; set PYTHON_BIN" >&2
    exit 1
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/ego_pose.py" "$@"
