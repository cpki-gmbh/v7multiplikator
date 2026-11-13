#!/usr/bin/env bash
set -u
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
B="$ROOT/bin-linux"
L="$B/libv7_threading.so"
DB="$ROOT/weights/databases/qwen3_0.6b/lz4ht.db"
ls -la "$DB"
echo "=== F32 ==="
"$B/inference_example" "$L" "$DB" -think 0 -n 48 -i 'What is the capital of France? Answer in one short sentence.'
echo "=== Q8 ==="
"$B/inference_example" "$L" "$DB" -q8 -think 0 -n 48 -i 'What is the capital of France? Answer in one short sentence.'
