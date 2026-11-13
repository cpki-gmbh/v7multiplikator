#!/usr/bin/env bash
set -u
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
B="$ROOT/bin-linux"
L="$B/libv7_threading.so"
DB="$ROOT/weights/databases/qwen3_0.6b/lz4ht_delay0.db"
PROMPT='Write a poem with 80 numbered lines about a lighthouse. Each line must be at least eight words long. Do not add any commentary.'
label="$1"
workdir="$2"
cd "$workdir" || exit 1
echo "# $label cwd=$workdir"
for t in 1 2 3; do
    out=$(timeout 200 "$B/inference_example" "$L" "$DB" -q8 -think 0 -n 100 -i "$PROMPT" 2>&1)
    dec=$(printf '%s\n' "$out" | grep -E '^  generation' | head -1)
    pre=$(printf '%s\n' "$out" | grep -E '^  prefill' | head -1)
    pms=$(printf '%s' "$pre" | sed -nE 's/.* ([0-9.]+) ms .*/\1/p')
    ts=$(printf '%s' "$dec" | sed -nE 's/.* ([0-9.]+) tok\/s\).*/\1/p')
    ep=$(printf '%s\n' "$out" | grep -oE 'cycle tuner: [0-9]+ episode' | grep -oE '[0-9]+' | tail -1)
    echo "$label,Q8,trial$t,prefill_ms=$pms,tok_s=$ts,episodes=$ep"
done
