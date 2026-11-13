# population_to_inis.py -- convert a cycle-tuner population file (128 genome
# slots x 512 bytes, saved by the engine as v7_tune_param_<n>_<d>) into a
# directory of INI files compatible with the INI benchmark tools
# (measure_winner_genomes.bat / genome_sort_experiment / the autotuner).
#
# Layout of one 512-byte genome slot (the engine's shape block):
#   +0    dword n
#   +4    dword d
#   +8    param_set, 24 dwords in the PARAM_OFFSET_* order
#   +104  pf_dist[0..40]
#   +300  pf_type[0..40]
#
# Output: one INI per genome slot, named <n>x<d>_<slot>.ini, section [nxd],
# carrying the 22 dispatch params, the Q8-owned pf_dist_0..7/pf_type_0..7,
# and best_gbs=0 (the measuring tool fills it in). NO dedup, NO filtering:
# every genome in the population becomes an INI -- the pool's distribution
# is the algorithm's material (AGENTS.md 202c).
#
# Usage: python population_to_inis.py <v7_tune_param_file> <output_dir>
#        python population_to_inis.py --all <dir-with-param-files> <output_root>

import os
import struct
import sys

GENOME_BYTES = 512
SLOTS = 128
PARAM_OFF = 8
PF_DIST_OFF = 104
PF_TYPE_OFF = 300

# The 22 dispatch params in the archive INI key order, as param_set dword
# indices (the PARAM_OFFSET_* numbering / 4).
PARAM_KEYS = [
    (0, "num_chunks"),
    (2, "batch_small"),
    (3, "batch_medium"),
    (4, "batch_large"),
    (5, "batch_huge"),
    (6, "batch_med_thresh"),
    (7, "batch_large_thresh"),
    (8, "batch_huge_thresh"),
    (9, "chunk_multiplier"),
    (10, "min_chunk_cols"),
    (13, "caller_batch_small"),
    (14, "caller_batch_medium"),
    (15, "caller_batch_large"),
    (16, "caller_batch_huge"),
    (17, "caller_batch_med_thresh"),
    (18, "caller_batch_large_thresh"),
    (19, "caller_batch_huge_thresh"),
    (20, "chunk_split_pct"),
    (21, "chunk_group_1_count"),
    (22, "chunk_group_2_count"),
    (1, "taper_shift"),
    (12, "quantize_chunk_groups"),
]


def slot_ini(blob, slot_index, fb_n, fb_d):
    """Render one 512-byte genome slot as an INI string. The slot header's
    n/d win when nonzero; older population files carry zero headers and fall
    back to the filename's <n>_<d> pair."""
    base = slot_index * GENOME_BYTES
    n = struct.unpack_from("<I", blob, base + 0)[0] or fb_n
    d = struct.unpack_from("<I", blob, base + 4)[0] or fb_d
    params = struct.unpack_from("<24I", blob, base + PARAM_OFF)
    pf_dist = struct.unpack_from("<41i", blob, base + PF_DIST_OFF)
    pf_type = struct.unpack_from("<41I", blob, base + PF_TYPE_OFF)

    out = ["[%dx%d]" % (n, d)]
    for idx, key in PARAM_KEYS:
        out.append("%s=%d" % (key, params[idx]))
    for i in range(8):
        out.append("pf_dist_%d=%d" % (i, pf_dist[i]))
    for i in range(8):
        out.append("pf_type_%d=%d" % (i, pf_type[i]))
    out.append("best_gbs=0")
    return "\n".join(out) + "\n", n, d


def convert_file(param_path, out_dir):
    with open(param_path, "rb") as f:
        blob = f.read()
    if len(blob) != GENOME_BYTES * SLOTS:
        print("skip %s: %d bytes, expected %d" % (param_path, len(blob), GENOME_BYTES * SLOTS))
        return 0
    # the filename carries the shape: v7_tune_param_<n>_<d>
    stem = os.path.basename(param_path)
    parts = stem.replace("v7_tune_param_", "").split("_")
    fb_n, fb_d = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (0, 0)
    written = 0
    for slot in range(SLOTS):
        text, n, d = slot_ini(blob, slot, fb_n, fb_d)
        if n == 0 or d == 0:
            continue  # no header anywhere: nothing to name the INI with
        name = "%dx%d_%d.ini" % (n, d, slot)
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text)
        written += 1
    print("%s -> %d INIs in %s" % (stem, written, out_dir))
    return written


def main():
    args = sys.argv[1:]
    if len(args) >= 3 and args[0] == "--all":
        src_dir, out_root = args[1], args[2]
        total = 0
        for name in sorted(os.listdir(src_dir)):
            if not name.startswith("v7_tune_param_"):
                continue
            sub = os.path.join(out_root, name[len("v7_tune_param_"):])
            os.makedirs(sub, exist_ok=True)
            total += convert_file(os.path.join(src_dir, name), sub)
        print("total: %d INIs" % total)
    elif len(args) == 2:
        param_path, out_dir = args
        os.makedirs(out_dir, exist_ok=True)
        convert_file(param_path, out_dir)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
