# collect_shape_inis_and_sort.py -- collect EVERY genome INI we have for one
# shape and renumber them into one directory, ready for the sorter. The
# script collects files, it does NOT decide on them: no dedup, no content
# filtering -- every file whose NAME matches <shape>_<counter>.ini is taken
# as-is. The sorter is started BY HAND afterwards (no redirection from this
# script); the ready-to-paste command is printed at the end.
#
# Collection: every subdirectory of opt_<shape>/ is scanned (run_a,
# target_pool, collect_*, reduce_*, bench_*, sanitized, probe pools --
# whatever exists, including archives added later), so no directory list
# needs maintaining.
#
# Renumbering: the sorter's --genome-dir reads <shape>_<counter>.ini, so the
# merged pool is renumbered 1..N into the seed directory (archives with
# overlapping counter namespaces become one addressable pool).
#
# Usage:
#   python collect_shape_inis_and_sort.py [--shape 1024x3072]
#        [--seed-dir DIR] [--out-dir DIR]
#
# Defaults: shape 1024x3072, seed dir opt_<shape>/merged_all, sort output
# dir opt_<shape>/merged_all_100.

import os
import re
import sys

SHAPE = "1024x3072"
TARGET_POOL_SIZE = 100
DLL_PATH = "bin\\v7_threading.dll"
DB_PATH = "weights\\databases\\qwen3_0.6b\\lz4ht.db"
SORTER_PATH = "tools\\genome_sort_experiment.exe"

FILE_RE = re.compile(r"^(.+)_(\d+)\.ini$")


def parse_args(argv):
    shape = SHAPE
    seed_dir = None
    out_dir = None
    i = 0
    while i < len(argv):
        if argv[i] == "--shape" and i + 1 < len(argv):
            shape = argv[i + 1]
            i += 2
        elif argv[i] == "--seed-dir" and i + 1 < len(argv):
            seed_dir = argv[i + 1]
            i += 2
        elif argv[i] == "--out-dir" and i + 1 < len(argv):
            out_dir = argv[i + 1]
            i += 2
        else:
            print("unrecognized argument: %s" % argv[i])
            sys.exit(1)
    if not seed_dir:
        seed_dir = "opt_%s\\merged_all" % shape
    if not out_dir:
        out_dir = "opt_%s\\merged_all_100" % shape
    return shape, seed_dir, out_dir


def collect_genomes(shape, seed_dir):
    """Scan every subdirectory of opt_<shape>/ for <shape>_<counter>.ini
    files and copy them renumbered 1..N into seed_dir. Every name-matching
    file is taken as-is -- the script collects, it does not decide."""
    shape_root = "opt_%s" % shape
    archive_dirs = []
    if os.path.isdir(shape_root):
        for name in sorted(os.listdir(shape_root)):
            path = os.path.join(shape_root, name)
            if os.path.isdir(path):
                archive_dirs.append(path)

    collected = []
    for archive_dir in archive_dirs:
        for name in sorted(os.listdir(archive_dir)):
            m = FILE_RE.match(name)
            if not m or m.group(1) != shape:
                continue
            collected.append(os.path.join(archive_dir, name))

    os.makedirs(seed_dir, exist_ok=True)
    n = 0
    for src_path in collected:
        n += 1
        dst_path = os.path.join(seed_dir, "%s_%d.ini" % (shape, n))
        with open(src_path, "rb") as f:
            data = f.read()
        with open(dst_path, "wb") as f:
            f.write(data)
    print("collected %d genomes from %d archive directories into %s"
          % (n, len(archive_dirs), seed_dir))
    return n


def main():
    shape, seed_dir, out_dir = parse_args(sys.argv[1:])
    count = collect_genomes(shape, seed_dir)
    if count < 2:
        print("Fail: only %d genomes collected for shape %s, nothing to sort"
              % (count, shape))
        sys.exit(1)
    print("sorter command (start by hand, no redirection):")
    print("  %s %s --db %s --q8 --genome-dir %s "
           "--target-pool-size %d --runtime 8 --compare-max 0.005 "
           "--min-diff 0.00005 --mutate-percentage 10 --sex-influence 40 "
           "--genome-output-dir %s "
           "--sorter-stdout-file sorter_saved\\merged_%s.txt"
           % (SORTER_PATH, DLL_PATH, DB_PATH, seed_dir, TARGET_POOL_SIZE,
              out_dir, shape))


if __name__ == "__main__":
    main()
