#!/usr/bin/env bash
echo "=== nproc / online ==="
nproc
cat /sys/devices/system/cpu/online
echo "=== per-cpu topology ==="
for d in /sys/devices/system/cpu/cpu[0-9]*; do
    c=${d##*/cpu}
    [ -f "$d/topology/core_id" ] || continue
    printf "cpu%-2s pkg=%s core_id=%s thread_siblings=%s\n" \
        "$c" \
        "$(cat "$d/topology/physical_package_id")" \
        "$(cat "$d/topology/core_id")" \
        "$(cat "$d/topology/thread_siblings_list")"
done
echo "=== /proc/cpuinfo core id / siblings ==="
grep -E '^(processor|core id|physical id|siblings|cpu cores)' /proc/cpuinfo
