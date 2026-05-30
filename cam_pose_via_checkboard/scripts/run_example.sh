#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${1:-/home/ubuntu/orbbec/src/sync/test/test/zyc}

python -m src.main \
  --dataset_root "${DATASET_ROOT}" \
  --config configs/default.yaml \
  --log_level INFO
