#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! "$SCRIPT_DIR/scripts/connect_pico.sh" --port 50051; then
    echo "WARNING: PICO automatic connection failed; Orbbec will still start." >&2
fi

echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
"$SCRIPT_DIR/bin/orbbec"
