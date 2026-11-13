#!/usr/bin/env bash
# check.sh -- Linux validation gate, the counterpart of verify.bat.
# Runs the model-free validators, then (with a database) the model-dependent
# validators and the f32/q8 inference smokes. Every failure aborts the run.
# v_swiglu_validate self-classifies its known limits: exit 0 means every
# failure was a documented Q8xQ8 float-vs-double shape (printed and counted);
# any F32xQ8 failure or undocumented shape exits 1 and aborts here like any
# other validator failure.
set -euo pipefail
bin=$(realpath -- "$1")
model=${2:-}
if [[ -n $model ]]; then
    model=$(realpath -e -- "$model")
fi

# the repo root: the sectorlogo validator reads the stdlib autoexec.logo and
# writes build_logs/ relative to its working directory, so it runs from here
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
cd "$work"
export TMPDIR="$work"
lib="$bin/libv7_threading.so"

for name in bicubic encode execute mdb readwrite sprintf; do
    "$bin/v_${name}_validate" "$lib"
done

if [[ -n $model ]]; then
    for name in add_inplace attention config_probe fused matmul memcpy quantize rmsnorm stream vit_linear; do
        "$bin/v_${name}_validate" "$lib" --db "$model"
    done
    for name in generate sampler; do
        "$bin/v_${name}_validate" "$lib" "$model"
    done
    # the stdlib autoexec.logo must be found in the working directory
    (cd "$root" && mkdir -p build_logs && "$bin/v_sectorlogo_validate" "$lib" --db "$model")

    # swiglu self-classifies: exit 0 = only documented Q8xQ8 known limits
    # failed (they are printed and counted); nonzero aborts like any other
    # validator under set -e.
    "$bin/v_swiglu_validate" "$lib" --db "$model"
    for mode in f32 q8; do
        options=()
        if [[ $mode == q8 ]]; then
            options+=(-q8)
        fi
        "$bin/inference_example" "$lib" "$model" "${options[@]}" \
            -think 0 -n 32 -i 'What is the capital of France? Answer in one short sentence.' | tee "$mode.log"
        grep -qi 'capital of france is paris' "$mode.log"
    done
fi
