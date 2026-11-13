#!/usr/bin/env bash
# Run the model-dependent validators and report each result; the script's
# exit status is nonzero when ANY validator failed.
set -uo pipefail
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
B="$ROOT/bin-linux"
L="$B/libv7_threading.so"
DB="$ROOT/weights/databases/qwen3_0.6b/lz4ht.db"
cd /tmp || exit 1
export TMPDIR=/tmp
failed=0
for n in add_inplace attention config_probe memcpy quantize rmsnorm; do
    echo "== $n =="
    "$B/v_${n}_validate" "$L" --db "$DB" 2>&1 | tail -3
    rc=${PIPESTATUS[0]}
    echo "rc=$rc"
    if [ "$rc" -ne 0 ]; then
        failed=1
    fi
done
for n in generate sampler; do
    echo "== $n =="
    "$B/v_${n}_validate" "$L" "$DB" 2>&1 | tail -3
    rc=${PIPESTATUS[0]}
    echo "rc=$rc"
    if [ "$rc" -ne 0 ]; then
        failed=1
    fi
done
exit "$failed"
