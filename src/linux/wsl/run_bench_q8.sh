#!/usr/bin/env bash
set -u
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
here="$ROOT/src/linux/wsl"
tmp=$(mktemp -d)
bash "$here/bench_q8.sh" baseline "$tmp"
rm -rf "$tmp"
bash "$here/bench_q8.sh" tuned "$ROOT"
