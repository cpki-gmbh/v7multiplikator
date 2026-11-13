#!/usr/bin/env bash
set -euo pipefail
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"
echo "=== building libv7_threading.so ==="
make -j"$(nproc)" bin-linux/libv7_threading.so
echo "=== built ==="
ls -la bin-linux/libv7_threading.so bin-linux/v7.o
