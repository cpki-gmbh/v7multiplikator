#!/usr/bin/env bash
set -u
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
B="$ROOT/bin-linux"
L="$B/libv7_threading.so"
DB="$ROOT/weights/databases/qwen3_0.6b/lz4ht.db"
PROMPT='Write a poem with 80 numbered lines about a lighthouse. Each line must be at least eight words long. Do not add any commentary.'
label="$1"
workdir="$2"
cd "$workdir" || exit 1
echo "# $label cwd=$workdir"
for mode in F32 Q8; do
    for t in 1 2 3; do
        if [ "$mode" = Q8 ]; then opt=-q8; else opt=; fi
        out=$(timeout 200 "$B/inference_example" "$L" "$DB" $opt -think 0 -n 100 -i "$PROMPT" 2>&1)
        dec=$(printf '%s\n' "$out" | grep -E '^  generation' | head -1)
        ms=$(printf '%s' "$dec" | sed -nE 's/.* ([0-9.]+) ms .*/\1/p')
        ts=$(printf '%s' "$dec" | sed -nE 's/.* ([0-9.]+) tok\/s\).*/\1/p')
        ep=$(printf '%s\n' "$out" | grep -oE 'cycle tuner: [0-9]+ episode' | grep -oE '[0-9]+' | tail -1)
        echo "$label,$mode,trial$t,decode_ms=$ms,tok_s=$ts,episodes=$ep"
    done
done
