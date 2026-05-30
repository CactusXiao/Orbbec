#!/usr/bin/env bash
set -e

echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
bin/orbbec
