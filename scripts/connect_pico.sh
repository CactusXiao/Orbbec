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

# Prefer libusb: the native backend can miss this PICO's USB interface.
export ADB_LIBUSB="${ADB_LIBUSB:-1}"

echo "Checking connected Android/PICO devices..."
devices="$("$ADB" devices)"
printf '%s\n' "$devices"

# Restart only when no transports exist and a PICO is physically present.
# This also replaces an existing native-backend server started outside pico.
if ! printf '%s\n' "$devices" | awk 'NR > 1 && NF { found=1 } END { exit !found }' &&
    command -v lsusb >/dev/null 2>&1 &&
    [[ -n "$(lsusb -d 2d40: 2>/dev/null)" ]]; then
    echo "PICO is visible on USB but missing from ADB; restarting ADB once..."
    "$ADB" kill-server
    "$ADB" start-server
    for attempt in 1 2 3 4 5; do
        sleep 1
        devices="$("$ADB" devices)"
        if printf '%s\n' "$devices" | awk 'NR > 1 && NF { found=1 } END { exit !found }'; then
            break
        fi
    done
    printf '%s\n' "$devices"
fi

device_count="$(printf '%s\n' "$devices" | awk 'NR > 1 && $2 == "device" { count++ } END { print count + 0 }')"
if [[ "$device_count" -eq 0 ]]; then
    echo "No ready ADB device. If unauthorized, accept USB debugging inside the headset; if offline or absent, reconnect its USB cable and retry pico." >&2
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
