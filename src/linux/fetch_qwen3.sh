#!/usr/bin/env bash
# Download the reference model, export it, and build a v7 database.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

bin_dir=${V7_BIN_DIR:-bin-linux}
python=${PYTHON:-python3}
model_dir=${1:-weights/qwen3_0.6b}
database=${2:-weights/databases/qwen3_0.6b/lz4ht.db}
revision=c1899de289a04d12100db370d81485cdf75e47ca
source_url="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/$revision"
weights_sha=f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b

if [[ ! -x $bin_dir/build_db ]]; then
    printf 'Build v7 with make before running this script.\n' >&2
    exit 1
fi
if [[ -e $database ]]; then
    printf 'Database already exists at %s\n' "$database" >&2
    exit 1
fi

# Pre-flight: surface every missing dependency BEFORE the multi-GB download,
# so a broken environment costs seconds, not a full download + tensor load.
"$python" - <<'EOF'
import importlib, sys
missing = []
for name in ("numpy", "torch", "safetensors", "transformers"):
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    sys.exit("missing python packages: %s  (pip install %s)" % (" ".join(missing), " ".join(missing)))
EOF
free_kb=$(df -Pk . | awk 'NR==2 {print $4}')
if [[ $free_kb -lt 6291456 ]]; then
    printf 'warning: only %d KB free on the weights volume -- the model pipeline needs roughly 4-5 GB\n' "$free_kb" >&2
fi

mkdir -p -- "$model_dir" "$(dirname -- "$database")"

for name in config.json tokenizer.json tokenizer_config.json generation_config.json merges.txt vocab.json model.safetensors; do
    file="$model_dir/$name"
    if [[ ! -s $file ]]; then
        curl --fail --location --retry 3 --connect-timeout 20 "$source_url/$name?download=true" -o "$file.part"
        if [[ $name == model.safetensors ]]; then
            printf '%s  %s\n' "$weights_sha" "$file.part" | sha256sum --check -
        fi
        mv -- "$file.part" "$file"
    fi
done
printf '%s  %s\n' "$weights_sha" "$model_dir/model.safetensors" | sha256sum --check -

checkpoint="$model_dir/qwen3_0.6b.bin"
"$python" tools/export.py "$checkpoint" "$model_dir"
# independent byte-level verification of the fresh export (header, tensor
# order, per-tensor scale blocks, tokenizer cross-check)
"$python" tools/test_export_checkpoint.py "$checkpoint" "$checkpoint.tokenizer"
"$bin_dir/build_db" "$checkpoint" -o "$database" \
    -tuner-delay 100 -tuner-rand-delay 1 -tuner-delay-large 10
