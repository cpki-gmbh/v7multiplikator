#!/usr/bin/env bash
set -u
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CKPT="$ROOT/weights/qwen3_0.6b/qwen3_0.6b.bin"
OUT="$ROOT/weights/databases/qwen3_0.6b/lz4ht_delay0.db"
DBTOOL="$ROOT/bin-linux/build_db"
cd "$ROOT" || exit 1
if [ -e "$OUT" ]; then
    echo "removing existing $OUT"
    rm -f "$OUT"
fi
echo "=== building DB with tuner-delay 0 ==="
"$DBTOOL" "$CKPT" -o "$OUT" -tuner-delay 0 -tuner-delay-large 0 -tuner-rand-delay 1
rc=$?
echo "build_db rc=$rc"
ls -la "$OUT"
