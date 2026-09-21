#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
exec "$SCRIPT_DIR/scripts/launch_with_pico.sh" "$SCRIPT_DIR/bin/orbbec" "$@"
