# v7 - part of the v7 inference engine
# Copyright (C) 2026 Gorg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)
#
# update_ini_from_tuner.py -- feed the in-process Logo tuner's learned
# configs back into threading_proxy.ini (owner directive).
#
# Reads the tuner's disk mirror (tuner_state.txt, rewritten by the
# tune_accept script on every accept and rotation) and rewrites the matching
# per-shape [NxD] sections in threading_proxy.ini. The mirror lines carry the
# raw v7s_tune_set_param slot order (0..105: 24 dispatch params + 41 pf_dist
# + 41 pf_type); the slot -> INI-key mapping below matches v7_data.inc's
# PARAM_OFFSET_* table exactly and is the ONE place it lives.
#
# Every written line is printed (visible feedback). All INI lines that are
# not a mapped key's value stay byte-identical, including inline comments.
#
# usage: python tools\update_ini_from_tuner.py [state_file] [ini_file]
#   state_file  default tuner_state.txt (project root, where the tuner writes)
#   ini_file    default threading_proxy.ini (project root)

import os
import shutil
import sys

# Raw tuner slot index -> INI key name (matches v7_data.inc PARAM_OFFSET_*).
SLOT_KEYS = [
    "num_chunks",             # 0
    "taper_shift",            # 1
    "batch_small",            # 2
    "batch_medium",           # 3
    "batch_large",            # 4
    "batch_huge",             # 5
    "batch_med_thresh",       # 6
    "batch_large_thresh",     # 7
    "batch_huge_thresh",      # 8
    "chunk_multiplier",       # 9
    "min_chunk_cols",         # 10
    "worker_priority",        # 11
    "quantize_chunk_groups",  # 12
    "caller_batch_small",     # 13
    "caller_batch_medium",    # 14
    "caller_batch_large",     # 15
    "caller_batch_huge",      # 16
    "caller_batch_med_thresh",    # 17
    "caller_batch_large_thresh",  # 18
    "caller_batch_huge_thresh",   # 19
    "chunk_split_pct",        # 20
    "chunk_group_1_count",    # 21
    "chunk_group_2_count",    # 22
    "cores_subtract",         # 23
]
for i in range(41):           # 24..64 -> pf_dist_0..pf_dist_40
    SLOT_KEYS.append("pf_dist_%d" % i)
for i in range(41):           # 65..105 -> pf_type_0..pf_type_40
    SLOT_KEYS.append("pf_type_%d" % i)

assert len(SLOT_KEYS) == 106

# INI-owned slots that are NOT per-shape: worker_priority (11) and
# cores_subtract (23) are standalone global keys, so a learned value for them
# must never be written into an [NxD] section.
INI_OWNED_SLOTS = (11, 23)


def parse_state(state_path):
    """tuner_state.txt -> {shape_token: [106 int values]}, plus the cursor."""
    cursor = None
    configs = {}
    with open(state_path, "r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if len(line) == 0:
                continue
            fields = line.split(" ")
            if fields[0] == "cursor" and len(fields) >= 2:
                cursor = fields[1]
            elif fields[0] == "cfg" and len(fields) >= 3:
                shape = fields[1]
                values = fields[2:]
                if len(values) != 106:
                    print("state file: cfg line for %s carries %d values (need 106) -- skipped"
                          % (shape, len(values)))
                    continue
                try:
                    configs[shape] = [int(v) for v in values]
                except ValueError:
                    print("state file: cfg line for %s has a non-numeric value -- skipped" % shape)
            else:
                print("state file: line with unknown head skipped: %s" % line)
    return cursor, configs


def main():
    state_path = sys.argv[1] if len(sys.argv) > 1 else "tuner_state.txt"
    ini_path = sys.argv[2] if len(sys.argv) > 2 else "threading_proxy.ini"

    try:
        cursor, configs = parse_state(state_path)
    except OSError as err:
        print("cannot read %s: %s" % (state_path, err))
        return 1

    print("state file: %s (cursor %s, %d learned shape(s))"
          % (state_path, cursor if cursor is not None else "none", len(configs)))
    if len(configs) == 0:
        print("no learned configs in the state file -- nothing to feed back")
        return 1

    try:
        with open(ini_path, "r", encoding="latin-1", newline="") as handle:
            lines = handle.readlines()
    except OSError as err:
        print("cannot read %s: %s" % (ini_path, err))
        return 1

    current_section = None
    written = 0
    matched_sections = set()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1]
            continue
        if current_section not in configs:
            continue
        # key=value with an optional inline comment after ';'
        eq = line.find("=")
        if eq < 0:
            continue
        key = line[:eq].strip()
        if key not in SLOT_KEYS:
            continue
        slot = SLOT_KEYS.index(key)
        if slot in INI_OWNED_SLOTS:
            continue                     # worker_priority / cores_subtract: global keys, never per-shape
        matched_sections.add(current_section)
        comment_at = line.find(";", eq)
        # preserve the file's own line ending (read with newline="")
        eol = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
        if comment_at >= 0:
            # the comment tail still carries the ORIGINAL eol (line includes it):
            # strip it before re-appending, or the rewrite doubles every newline.
            tail = line[comment_at:].rstrip("\r\n")
            newline = "%s=%d %s%s" % (line[:eq], configs[current_section][slot], tail, eol)
        else:
            newline = "%s=%d%s" % (line[:eq], configs[current_section][slot], eol)
        if newline != line:
            lines[idx] = newline
            print("[%s] %s=%d (slot %d)" % (current_section, key, configs[current_section][slot], slot))
            written += 1

    unmatched = [shape for shape in configs if shape not in matched_sections]
    if unmatched:
        # zero matching sections means the schema did not match -- never claim
        # the learned configs were applied (or were already equal).
        print("ERROR: no matching [NxD] section for learned shape(s): %s" % ", ".join(sorted(unmatched)))
        return 1

    if written == 0:
        print("no INI lines changed -- the learned configs match the current values")
        return 0

    # atomic write: sibling .tmp + os.replace, one .bak of the last good file
    tmp_path = ini_path + ".tmp"
    bak_path = ini_path + ".bak"
    try:
        with open(tmp_path, "w", encoding="latin-1", newline="") as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.exists(ini_path):
            shutil.copyfile(ini_path, bak_path)
        os.replace(tmp_path, ini_path)
    except OSError as err:
        print("cannot write %s: %s" % (ini_path, err))
        return 1

    print("threading_proxy.ini updated: %d line(s) written from %d learned shape(s)"
          % (written, len(configs)))
    print("regenerate the databases to make the learned configs live in the DB:")
    print("  build_text_model_databases.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())