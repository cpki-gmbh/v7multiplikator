#!/usr/bin/env bash
for t in gcc cc make nasm; do
    if command -v "$t" >/dev/null 2>&1; then
        printf '%s: %s\n' "$t" "$(command -v "$t")"
    else
        printf '%s: MISSING\n' "$t"
    fi
done
