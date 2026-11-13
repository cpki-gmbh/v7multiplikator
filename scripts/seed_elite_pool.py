# seed_elite_pool.py -- assemble a seeded genome pool for the sorter.
#
# Collects every genome INI we have for one shape from the archive
# directories (default: every known archive), dedups by content, and
# renumbers them 1..N into a fresh seed directory (the sorter's
# --genome-dir reads <shape>_<counter>.ini, so renumbering merges archives
# with overlapping counter namespaces into one addressable pool). The
# random probes are drawn by the SORTER itself (--random-probes N), so
# this script only seeds the archive genomes.
#
# Usage:
#   python seed_elite_pool.py --shape 3072x1024 --out opt_3072x1024\seeded
#        [--dirs dir1,dir2,...]
#
# The defaults collect every known archive of any shape (only matching
# file names land in the pool).

import os
import re
import sys

DEFAULT_DIRS = [
    "winner_genomes",
    "winner_genomes_b",
    "winner_genomes_c",
    "opt_1024x3072\\run_a",
    "opt_1024x3072\\target_pool",
    "opt_1024x3072\\mutation_test",
    "opt_1024x3072\\random_probe_pool",
    "opt_1024x3072\\probe_filtered_pool",
    "opt_3072x1024\\run_a",
    "opt_3072x1024\\run_b",
    "opt_3072x1024\\target_pool",
    "opt_3072x1024\\target_pool_b",
]

FILE_RE = re.compile(r"^(.+)_(\d+)\.ini$")


def parse_args(argv):
    shape = None
    out_dir = None
    dirs = list(DEFAULT_DIRS)
    i = 0
    while i < len(argv):
        if argv[i] == "--shape" and i + 1 < len(argv):
            shape = argv[i + 1]
            i += 2
        elif argv[i] == "--out" and i + 1 < len(argv):
            out_dir = argv[i + 1]
            i += 2
        elif argv[i] == "--dirs" and i + 1 < len(argv):
            dirs = argv[i + 1].split(",")
            i += 2
        else:
            print("unrecognized argument: %s" % argv[i])
            sys.exit(1)
    if not shape:
        print("Fail: --shape <NxM> is required")
        sys.exit(1)
    if not out_dir:
        print("Fail: --out <dir> is required")
        sys.exit(1)
    return shape, out_dir, dirs


def main():
    shape, out_dir, dirs = parse_args(sys.argv[1:])
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Collect every archive INI whose file name matches this shape, dedup
    # by content: a genome with identical config content from two archives
    # is one genome, and duplicates would just waste sorter slots.
    collected = []
    seen_hashes = set()
    for archive_dir in dirs:
        if not os.path.isdir(archive_dir):
            continue
        for name in sorted(os.listdir(archive_dir)):
            m = FILE_RE.match(name)
            if not m or m.group(1) != shape:
                continue
            src_path = os.path.join(archive_dir, name)
            with open(src_path, "rb") as f:
                content_hash = hash(f.read())
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            collected.append(src_path)

    # Renumber 1..N into the seed directory.
    n = 0
    for src_path in collected:
        n += 1
        dst_path = os.path.join(out_dir, "%s_%d.ini" % (shape, n))
        with open(src_path, "rb") as f:
            data = f.read()
        with open(dst_path, "wb") as f:
            f.write(data)

    print("seeded %d genomes into %s (shape %s, %d archives scanned)"
          % (n, out_dir, shape, len(dirs)))
    print("sort command:")
    print("  tools\\genome_sort_experiment.exe bin\\v7_threading.dll "
          "--db weights\\databases\\qwen3_0.6b\\lz4ht.db --q8 --genome-dir %s "
          "--random-probes 300 --target-pool-size 100 --runtime 10 "
          "--compare-max 0.005 --min-diff 0.00005 --mutate-percentage 2 "
          "--sex-influence 40 --genome-output-dir %s_sorted "
          "--sorter-stdout-file sorter_saved\\%s_sorted.txt"
          % (out_dir, os.path.basename(out_dir), os.path.basename(out_dir)))


if __name__ == "__main__":
    main()