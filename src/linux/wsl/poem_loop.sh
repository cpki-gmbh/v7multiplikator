#!/usr/bin/env bash
set -u
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
B="$ROOT/bin-linux"
L="$B/libv7_threading.so"
DB="$ROOT/weights/databases/qwen3_0.6b/lz4ht_delay0.db"
PROMPT='write a very long poem about horse, really make it very long!'
CSV="$ROOT/poem_loop_results.csv"
FULL="$ROOT/poem_loop_full.log"
cd "$ROOT" || exit 1
echo "iter,generated_tokens,decode_ms,tok_s,prefill_ms,episodes,total_wall_ms" > "$CSV"
echo "=== poem loop start $(date -u +%H:%M:%S) DB=$DB ===" > "$FULL"
start_all=$(date +%s%3N)
for i in $(seq 1 30); do
    t0=$(date +%s%3N)
    out=$(timeout 400 "$B/inference_example" "$L" "$DB" -q8 -think 0 -n 3000 -i "$PROMPT" 2>&1)
    rc=$?
    t1=$(date +%s%3N)
    wall=$((t1 - t0))
    gen=$(printf '%s\n' "$out" | grep -E '^  generation' | head -1)
    pre=$(printf '%s\n' "$out" | grep -E '^  prefill' | head -1)
    dec_ms=$(printf '%s' "$gen" | sed -nE 's/.* ([0-9.]+) ms .*/\1/p')
    ntok=$(printf '%s' "$gen" | sed -nE 's/.* \(([0-9]+) tok,.*/\1/p')
    ts=$(printf '%s' "$gen" | sed -nE 's/.* ([0-9.]+) tok\/s\).*/\1/p')
    pms=$(printf '%s' "$pre" | sed -nE 's/.* ([0-9.]+) ms .*/\1/p')
    ep=$(printf '%s\n' "$out" | grep -oE 'cycle tuner: [0-9]+ episode' | grep -oE '[0-9]+' | tail -1)
    echo "$i,$ntok,$dec_ms,$ts,$pms,$ep,$wall" >> "$CSV"
    printf '=== iter %d rc=%d wall=%dms tokens=%s decode=%sms %s tok/s episodes=%s ===\n' "$i" "$rc" "$wall" "$ntok" "$dec_ms" "$ts" "$ep" >> "$FULL"
    printf '%s\n' "$out" >> "$FULL"
    echo "iter $i: ${ts} tok/s (${ntok} tok, ${dec_ms} ms) episodes=${ep}"
done
end_all=$(date +%s%3N)
echo "=== poem loop done $(date -u +%H:%M:%S) total=$((end_all - start_all))ms ===" | tee -a "$FULL"
echo "DONE" > "$ROOT/poem_loop.done"
