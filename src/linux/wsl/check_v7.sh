#!/usr/bin/env bash
# Run the model-free validators and report each result; the script's exit
# status is nonzero when ANY validator failed, so automation sees the truth.
set -uo pipefail
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
B="$ROOT/bin-linux"
L="$B/libv7_threading.so"
cd /tmp || exit 1
failed=0
for n in mdb sprintf readwrite bicubic execute; do
    echo "== $n =="
    "$B/v_${n}_validate" "$L" 2>&1 | tail -4
    rc=${PIPESTATUS[0]}
    echo "rc=$rc"
    if [ "$rc" -ne 0 ]; then
        failed=1
    fi
done
exit "$failed"
