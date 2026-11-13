#!/usr/bin/env bash
set -u
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
B="$ROOT/bin-linux"
L="$B/libv7_threading.so"
DB="$ROOT/weights/databases/qwen3_0.6b/lz4ht_delay0.db"
cd "$ROOT" || exit 1
"$B/inference_example" "$L" "$DB" -q8 -think 0 -n 4000 -i 'write a very long poem about horse, really make it very long!' >/tmp/cpu_run.log 2>&1 &
pid=$!
echo "pid=$pid"
sleep 20
echo "tasks=$(ls /proc/$pid/task | wc -l)"
echo "--- thread tree ---"
for t in /proc/$pid/task/*; do
    tid=${t##*/}
    aff=$(awk '/Cpus_allowed_list/{print $2}' "$t/status" 2>/dev/null)
    name=$(cat "$t/comm" 2>/dev/null)
    echo "tid=$tid comm=$name allowed=$aff"
done
echo "--- log marker ---"
grep -E 'worker W|mode:|coverage|caller:' /tmp/cpu_run.log | head -20
kill "$pid" 2>/dev/null
wait "$pid" 2>/dev/null
