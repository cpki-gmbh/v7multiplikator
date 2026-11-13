# measure_winner_genomes.py -- benchmark archived winner genomes with test_shape_speed.
#
# Enumerates winner_genomes\<shape>_<counter>.ini files, selects every Nth
# genome (default stride 10, plus genome 1), and runs tools\test_shape_speed.exe
# once per genome with a short measurement budget. Each run loads the engine
# from the same database, applies the genome's committed config (22 dispatch
# params + owned pf distances/types, live SMC patches), and prints the four
# step rates plus the drift-correction factor. The parsed numbers land in
# winner_genomes\results.txt: one block per genome plus a summary table sorted
# by shape_time_a GB/s, so the winning lineage can be compared offline.
#
# Usage:
#   python measure_winner_genomes.py [--mb 16] [--seconds 2] [--stride 10] [--genomes 1,10,20]
#                                    [--genome-dir winner_genomes] [--repeat 3]
#
# Defaults: 16 MB copy buffer, 2 s measurement per genome, front 10 genomes.
# --genomes selects explicit archive counters (comma-separated) instead of
# --stride N selects the front N counters. --repeat measures the FIRST selected genome n times
# (default 3), printed and recorded as if it had been passed n times.

import os
import re
import subprocess
import sys

TOOL_PATH = os.path.join("tools", "test_shape_speed.exe")
DLL_PATH = os.path.join("bin", "v7_threading.dll")
DB_PATH = os.path.join("weights", "databases", "qwen3_0.6b", "lz4ht.db")

# Lines of interest in the tool's output (label -> capture group position).
RESULT_FIELDS = [
    ("shape_time_a", r"^shape_time_a\s+([0-9.]+) GB/s"),
    ("small_to_large", r"^small_to_large\s+([0-9.]+) GB/s"),
    ("shape_time_b", r"^shape_time_b\s+([0-9.]+) GB/s"),
    ("large_to_small", r"^large_to_small\s+([0-9.]+) GB/s"),
    ("shape_time_c", r"^shape_time_c\s+([0-9.]+) GB/s"),
    ("memcpy_kernel", r"^memcpy_kernel\s+([0-9.]+) GB/s"),
    ("float_sum", r"^float_sum\s+([0-9.]+) MB/s"),
    ("float_memcpy_factor", r"^float/memcpy\s+([0-9.]+)"),
    ("rounds", r"^shape \S+ q8: (\d+) rounds"),
]


def parse_args(argv):
    mb = 16
    seconds = 2.0
    stride = 10
    genome_counters = None
    genome_dir = "winner_genomes"
    repeat = 3
    i = 0
    while i < len(argv):
        if argv[i] == "--mb" and i + 1 < len(argv):
            mb = int(argv[i + 1])
            i += 2
        elif argv[i] == "--seconds" and i + 1 < len(argv):
            seconds = float(argv[i + 1])
            i += 2
        elif argv[i] == "--stride" and i + 1 < len(argv):
            stride = int(argv[i + 1])
            i += 2
        elif argv[i] == "--genomes" and i + 1 < len(argv):
            genome_counters = []
            for token in argv[i + 1].split(","):
                token = token.strip()
                if token and int(token) not in genome_counters:
                    genome_counters.append(int(token))
            i += 2
        elif argv[i] == "--genome-dir" and i + 1 < len(argv):
            genome_dir = argv[i + 1]
            i += 2
        elif argv[i] == "--repeat" and i + 1 < len(argv):
            repeat = int(argv[i + 1])
            i += 2
        else:
            print("unrecognized argument: %s" % argv[i])
            sys.exit(1)
    if mb < 1 or seconds <= 0 or stride < 1 or repeat < 1:
        print("invalid arguments: mb=%d seconds=%g stride=%d repeat=%d" % (mb, seconds, stride, repeat))
        sys.exit(1)
    return mb, seconds, stride, genome_counters, genome_dir, repeat


def select_genomes(stride, genome_counters, genome_dir):
    """<genome_dir>/<shape>_<counter>.ini -> sorted [(shape, counter, path)].
    Selection: explicit counters when given, else the FIRST <stride> counters
    (the front of the archive/population, where the cycle sorter keeps its
    fastest genomes)."""
    pat = re.compile(r"^(.+)_(\d+)\.ini$")
    found = []
    for name in os.listdir(genome_dir):
        m = pat.match(name)
        if m:
            found.append((m.group(1), int(m.group(2)), os.path.join(genome_dir, name)))
    found.sort(key=lambda t: (t[0], t[1]))
    if genome_counters is not None:
        # Preserve the user's listing order: the --repeat target is the FIRST
        # counter the user listed, the rest follow sorted by counter.
        selected = []
        for wanted in genome_counters:
            for entry in found:
                if entry[1] == wanted:
                    selected.append(entry)
        rest = sorted((t for t in found if t[1] in genome_counters and t not in selected))
        selected = selected + rest
        missing = [c for c in genome_counters if c not in set(t[1] for t in selected)]
        if missing:
            print("warning: no archive file for counter(s): %s" % missing)
    else:
        # the front of the population: the lowest <stride> counters, where
        # the cycle sorter's exchanges concentrate the measured-fastest
        # genomes; --stride N here means "how many front genomes to measure"
        selected = [t for t in found if t[1] <= stride]
    return selected


def run_one(shape, genome_path, mb, seconds):
    """Run the tool for one genome; return the parsed field dict or None."""
    cmd = [TOOL_PATH, DLL_PATH, "--db", DB_PATH, "--ini", genome_path,
           "--q8", shape, str(mb), str(seconds)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace")
    except OSError as err:
        print("  run failed for %s: %s" % (genome_path, err))
        return None
    fields = {}
    for line in proc.stdout.splitlines():
        for name, pattern in RESULT_FIELDS:
            m = re.match(pattern, line.strip())
            if m and name not in fields:
                fields[name] = m.group(1)
    if "shape_time_a" not in fields:
        print("  no result block parsed from %s (exit %d)" % (genome_path, proc.returncode))
        return None
    return fields


def main():
    mb, seconds, stride, genome_counters, genome_dir, repeat = parse_args(sys.argv[1:])
    genomes = select_genomes(stride, genome_counters, genome_dir)
    results_path = os.path.join(genome_dir, "results.txt")
    if not genomes:
        print("no winner genomes found in %s" % genome_dir)
        sys.exit(1)
    # --repeat: the first selected genome is measured n times, printed and
    # recorded exactly as if it had been passed n times in the selection.
    measure_list = genomes[:1] * repeat + genomes[1:]
    print("measuring %d genomes (%s ...), %d MB buffer, %g s each (first genome x%d)"
          % (len(measure_list), measure_list[0][2], mb, seconds, repeat))

    results = []
    for shape, counter, path in measure_list:
        print("genome %-24s ..." % os.path.basename(path))
        sys.stdout.flush()
        fields = run_one(shape, path, mb, seconds)
        if fields:
            fields["counter"] = counter
            fields["path"] = path
            results.append(fields)
            print("  shape_time_a %s GB/s  memcpy_kernel %s GB/s  factor %s" %
                  (fields["shape_time_a"], fields["memcpy_kernel"], fields.get("float_memcpy_factor", "?")))

    # Results file: per-genome blocks then a summary sorted by shape_time_a.
    with open(results_path, "w") as out:
        out.write("winner genome measurements (%d genomes, %d MB buffer, %g s each)\n"
                  % (len(results), mb, seconds))
        out.write("fields: shape_time_a/b/c = dispatched matmul GB/s; "
                  "small_to_large/large_to_small = 16KB-chunk copy walks; "
                  "memcpy_kernel = dispatched block copy; float_sum = CPU probe\n\n")
        for f in results:
            out.write("genome %s\n" % os.path.basename(f["path"]))
            for name, _ in RESULT_FIELDS:
                if name in f:
                    out.write("  %-20s %s\n" % (name, f[name]))
            out.write("\n")
        ranked = sorted(results, key=lambda f: -float(f["shape_time_a"]))
        out.write("summary sorted by shape_time_a GB/s\n")
        out.write("%-24s %10s %10s %10s %10s %10s %10s\n"
                  % ("genome", "shape_a", "shape_b", "shape_c", "memcpy_kb", "float_sum", "factor"))
        for f in ranked:
            out.write("%-24s %10s %10s %10s %10s %10s %10s\n"
                      % (os.path.basename(f["path"]), f["shape_time_a"], f.get("shape_time_b", "?"),
                         f.get("shape_time_c", "?"), f.get("memcpy_kernel", "?"),
                         f.get("float_sum", "?"), f.get("float_memcpy_factor", "?")))
    print("results written to %s" % results_path)
    if results:
        best = max(results, key=lambda f: float(f["shape_time_a"]))
        print("best genome: %s at %s GB/s (shape_time_a)" %
              (os.path.basename(best["path"]), best["shape_time_a"]))


if __name__ == "__main__":
    main()