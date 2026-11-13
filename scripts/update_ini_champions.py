# update_ini_champions.py -- transplant the champion genome configs into
# threading_proxy.ini's per-shape sections.
#
# Reads each champion genome file (a complete [NxM] section), locates the
# same section in threading_proxy.ini, and replaces the section's key
# values (from the section header to the next section header, keeping the
# INI's blank-line-per-key layout intact by rewriting the whole section
# block compactly -- the Windows INI API and build_db.exe both read
# compact sections fine).
#
# Usage: python update_ini_champions.py
# (the champion list is hard-coded below)

import os
import sys

INI_PATH = "threading_proxy.ini"

CHAMPIONS = [
    "opt_1024x3072\\merged_all_100\\1024x3072_8.ini",
    "opt_3072x1024\\target_pool_b\\3072x1024_3.ini",
    "opt_1024x2048\\phase_cut\\1024x2048_5.ini",
    "winner_genomes\\1024x1024_220.ini",
    "opt_1024x151936\\target_pool\\1024x151936_8.ini",
    "opt_2048x6144\\target_pool\\2048x6144_8.ini",
    "opt_6144x2048\\target_pool\\6144x2048_9.ini",
    "opt_2048x4096\\target_pool\\2048x4096_1.ini",
    "opt_2048x2048\\target_pool_b\\2048x2048_4.ini",
]


def read_champion(path):
    """Return (section_name, [(key, value), ...]) from a champion INI."""
    section = None
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
            elif "=" in line and section:
                key, _, value = line.partition("=")
                pairs.append((key.strip(), value.strip()))
    return section, pairs


def main():
    with open(INI_PATH, "r") as f:
        lines = f.readlines()

    for champ_path in CHAMPIONS:
        section, pairs = read_champion(champ_path)
        if not section:
            print("warning: %s has no section header, skipping" % champ_path)
            continue

        # Find the section's line range in threading_proxy.ini.
        header_line = -1
        for i, line in enumerate(lines):
            if line.strip() == "[%s]" % section:
                header_line = i
                break
        if header_line < 0:
            print("warning: [%s] not found in %s, skipping" % (section, INI_PATH))
            continue

        # Find the section's end: the next [section] header after it.
        end_line = len(lines)
        for i in range(header_line + 1, len(lines)):
            if lines[i].strip().startswith("[") and lines[i].strip().endswith("]"):
                end_line = i
                break

        # Build the replacement block: compact [NxM] + key=value lines.
        block = ["[%s]\n" % section]
        for key, value in pairs:
            block.append("%s=%s\n" % (key, value))
        block.append("\n")  # blank line before the next section

        lines[header_line:end_line] = block
        print("updated [%s] with %s (%d keys)" % (section, champ_path, len(pairs)))

    with open(INI_PATH, "w") as f:
        f.writelines(lines)
    print("done: %s updated with %d champion sections" % (INI_PATH, len(CHAMPIONS)))
    print("rebuild the DB: bin\\build_db.exe weights\\qwen3_0.6b.bin -o weights\\databases\\qwen3_0.6b\\lz4ht.db")


if __name__ == "__main__":
    main()