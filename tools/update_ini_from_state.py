#!/usr/bin/env python3
"""update_ini_from_state.py, bake the tuner's learned configs from
tuner_state.txt into threading_proxy.ini's per-shape [NxD] sections.

Mapping contract: docs/tuner_slots.md. The mirror record's fields 0..23 are
the param_set slots in slot order; this script maps them to the 22 INI keys,
SKIPS f11 (worker_priority) and f23 (cores_subtract) -- INI-owned, never
tuner-tunable -- and ignores f24..f105 (the 82 prefetch fields stay a live
SMC experiment; the INI keeps its committed pf values).
"""

import os
import re
import shutil
import sys

STATE = "tuner_state.txt"
INI = "threading_proxy.ini"

# mirror field index -> INI key; f11/f23 intentionally absent (INI-owned)
SLOT_KEYS = {
    0: "num_chunks",
    1: "taper_shift",
    2: "batch_small",
    3: "batch_medium",
    4: "batch_large",
    5: "batch_huge",
    6: "batch_med_thresh",
    7: "batch_large_thresh",
    8: "batch_huge_thresh",
    9: "chunk_multiplier",
    10: "min_chunk_cols",
    # 11: worker_priority -- INI-owned, skipped
    12: "quantize_chunk_groups",
    13: "caller_batch_small",
    14: "caller_batch_medium",
    15: "caller_batch_large",
    16: "caller_batch_huge",
    17: "caller_batch_med_thresh",
    18: "caller_batch_large_thresh",
    19: "caller_batch_huge_thresh",
    20: "chunk_split_pct",
    21: "chunk_group_1_count",
    22: "chunk_group_2_count",
    # 23: cores_subtract -- INI-owned, skipped
}


def main():
    if not os.path.isfile(STATE):
        print(f"  {STATE} not found -- nothing to promote")
        return 1

    with open(STATE, "r", encoding="utf-8") as handle:
        state_lines = [ln.strip() for ln in handle if ln.startswith("cfg ")]

    with open(INI, "r", encoding="utf-8", newline="") as handle:
        ini_text = handle.read()

    baked = 0
    total_applied = 0
    for line in state_lines:
        parts = line.split()
        shape = parts[1]
        fields = [int(tok) for tok in parts[2:]]
        if len(fields) < 24:
            print(f"  [{shape}] record has only {len(fields)} fields, skipped (need >= 24)")
            continue

        section_re = re.compile(
            rf"(?ms)^\[{re.escape(shape)}\]\r?\n(.*?)(?=^\[|\Z)")
        m = section_re.search(ini_text)
        if not m:
            print(f"  [{shape}] ERROR: no INI section")
            return 1
        body = m.group(1)

        new_body = body
        applied = 0
        for idx, key in SLOT_KEYS.items():
            value = fields[idx]
            # match key=NNN with an OPTIONAL inline comment, preserving the
            # comment and the CR: the shipped INI carries "key=NNN ; note"
            # lines, which a bare \d+\s*$ pattern silently never matches.
            key_re = re.compile(rf"(?m)^({re.escape(key)}=)\d+([ \t]*;.*?)?(\r?)$")
            new_body, n = key_re.subn(rf"\g<1>{value}\g<2>\g<3>", new_body)
            applied += n
        if applied != len(SLOT_KEYS):
            print(f"  [{shape}] ERROR: only {applied}/{len(SLOT_KEYS)} keys matched -- schema mismatch")
            return 1

        ini_text = ini_text[: m.start(1)] + new_body + ini_text[m.end(1):]
        baked += 1
        total_applied += applied
        print(f"  [{shape}] baked {applied} keys")

    if total_applied == 0:
        print("  no fields applied -- refusing to claim success")
        return 1

    # atomic write: sibling .tmp + os.replace, one .bak of the last good file
    tmp_path = INI + ".tmp"
    bak_path = INI + ".bak"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(ini_text)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copyfile(INI, bak_path)
        os.replace(tmp_path, INI)
    except OSError as err:
        print(f"  cannot write {INI}: {err}")
        return 1

    print(f"  {baked} shape(s) baked from tuner_state.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
