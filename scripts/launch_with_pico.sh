#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: launch_with_pico.sh COMMAND [ARG ...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Match the existing pico command before starting the capture process.
if ! bash "$SCRIPT_DIR/connect_pico.sh" --port 50051; then
    echo "WARNING: PICO automatic connection failed; Orbbec will still start." >&2
fi

exec "$@"
