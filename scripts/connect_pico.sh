#!/usr/bin/env bash
set -euo pipefail

PORT="50051"
ADB="${ADB:-adb}"

usage() {
    printf '%s\n' \
        "Usage: pico [--port PORT] [--adb ADB]" \
        "" \
        "Configures adb reverse tcp:PORT tcp:PORT for a USB-connected PICO device."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)
            PORT="${2:?Missing value for $1}"
            shift 2
            ;;
        --adb)
            ADB="${2:?Missing value for $1}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v "$ADB" >/dev/null 2>&1; then
    echo "adb was not found. Install it with: sudo apt install android-tools-adb" >&2
    exit 1
fi

echo "Checking connected Android/PICO devices..."
"$ADB" devices

device_count="$("$ADB" devices | awk 'NR > 1 && $2 == "device" { count++ } END { print count + 0 }')"
if [[ "$device_count" -eq 0 ]]; then
    echo "No authorized PICO/Android device is connected." >&2
    exit 1
fi
if [[ "$device_count" -gt 1 && -z "${ANDROID_SERIAL:-}" ]]; then
    echo "More than one device is connected. Set ANDROID_SERIAL before running pico." >&2
    exit 1
fi

echo "Configuring adb reverse tcp:${PORT} tcp:${PORT} ..."
"$ADB" reverse "tcp:${PORT}" "tcp:${PORT}"

echo "Current reverse mappings:"
"$ADB" reverse --list

echo "PICO connection is ready on tcp:${PORT}."
