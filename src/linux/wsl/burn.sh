#!/usr/bin/env bash
# Pure busy loop; caller pins it with taskset.
end=$((SECONDS + ${1:-12}))
while [ $SECONDS -lt $end ]; do :; done
