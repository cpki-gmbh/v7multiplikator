#!/usr/bin/env python3
"""Pick a random unreviewed .inc file from src/asm/ and print review instructions."""

import glob
import os
import random
import time
import sys

SRC_ASM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "asm")
OPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_and_bug")
MAX_TRIES = 100
SLEEP_SECONDS = 5


def get_reviewed_stems():
    """Return set of stems that already have a report (original or revised)."""
    stems = set()
    for md in glob.glob(os.path.join(OPT_DIR, "*.md")):
        name = os.path.basename(md)
        if name.endswith("_revised.md"):
            stems.add(name[: -len("_revised.md")])
        elif name.endswith(".md"):
            stem = name[: -len(".md")]
            if stem.endswith("_beam"):
                stem = stem[:-len("_beam")]
            stems.add(stem)
    return stems


def inc_to_report_stem(inc_name):
    """Map an .inc filename to its canonical report stem."""
    base = inc_name.replace(".inc", "")
    return base + "_inc"


def get_unreviewed():
    """Return list of .inc files with no report in opt_and_bug/."""
    reviewed = get_reviewed_stems()
    unreviewed = []
    for inc in sorted(glob.glob(os.path.join(SRC_ASM_DIR, "*.inc"))):
        name = os.path.basename(inc)
        stem = inc_to_report_stem(name)
        # Also check bare stem (e.g. v7_core -> v7_core)
        bare_stem = name.replace(".inc", "")
        if stem not in reviewed and bare_stem not in reviewed:
            unreviewed.append(inc)
    return unreviewed


def main():
    waiting_message_printed = False
    while True:
        unreviewed = get_unreviewed()
        if not unreviewed:
            if not waiting_message_printed:
                print("No unreviewed .inc files found. Waiting for new files.", flush=True)
                waiting_message_printed = True
            time.sleep(SLEEP_SECONDS)
            continue

        # Try up to MAX_TRIES random picks (list is small, just pick one)
        for _ in range(MAX_TRIES):
            pick = random.choice(unreviewed)
            break

        name = os.path.basename(pick)
        stem = inc_to_report_stem(name)
        report_path = os.path.join(OPT_DIR, stem + ".md")

        print()
        print("=" * 78)
        print(f"REVIEW TARGET: {pick}")
        print(f"REPORT TO WRITE: {report_path}")
        print("=" * 78)
        print()
        print("INSTRUCTIONS:")
        print()
        print("  HARD RULE -- READ BEFORE DOING ANYTHING ELSE:")
        print("    * You are the REVIEW agent. Your ONLY output is ONE brand-new report")
        print("      at the exact path printed below as REPORT TO WRITE.")
        print("    * DO NOT create a '_revised.md' report. The '_revised' suffix belongs")
        print("      to a different agent that consolidates existing reports. Never write it.")
        print("    * DO NOT revise, rewrite, edit, merge, append to, or delete ANY existing")
        print("      report in opt_and_bug/ -- neither a plain <stem>.md nor a")
        print("      <stem>_revised.md. They are not yours. Read them if you want context,")
        print("      but leave them byte-for-byte untouched.")
        print("    * DO NOT touch any source file in src/asm/ in this role. You only read.")
        print("    * The report filename is exactly the path printed below. Copy it verbatim.")
        print()
        print(f"  1. Read the ENTIRE file {pick} -- every single line.")
        print()
        print("  2. Examine for THREE things:")
        print("     a) Real bugs -- wrong output, crashes, register-width mismatches,")
        print("        incorrect arithmetic, precision loss, 32-bit truncation of 64-bit")
        print("        values (AGENTS.md section 45), contract violations.")
        print("     b) Inefficient instruction sequences -- any sequence replaceable by a")
        print("        shorter or lower-latency equivalent that does NOT make precision worse.")
        print("        Optimizations that change bits but maintain or IMPROVE numerical")
        print("        precision are ACCEPTABLE and should be reported. Optimizations that")
        print("        reduce precision must NOT be reported. Every CPU cycle counts.")
        print("        A single instruction saved per iteration on a hot path is worth it.")
        print("        This includes kernel math optimizations that are precision-preserving")
        print("        or precision-improving -- e.g. replacing double-rounded multiply-by-")
        print("        reciprocal with correctly-rounded division, extending accumulation")
        print("        precision, eliminating catastrophic cancellation.")
        print("     c) Control-flow straightening -- reorder branches so the most probable")
        print("        path takes the fewest jumps and falls through.")
        print()
        print("  3. Per AGENTS.md section 129a -- for DANGEROUS AREAS (spin/dispatch logic),")
        print("     report ONLY bugs, NOT optimizations. For kernel math, optimizations")
        print("     that do NOT make precision worse ARE reportable. Bugs in dangerous")
        print("     areas are always reported.")
        print()
        print("  4. Every item must be DIRECTLY ACTIONABLE -- name the exact function/label/")
        print("     line, state precisely what to change and to what. No 'consider',")
        print("     'investigate', or 'might be worth'.")
        print()
        print("  5. Write the review report to this EXACT path (no '_revised' suffix):")
        print(f"     {report_path}")
        print()
        print("  6. After writing the report, RE-RUN THIS SCRIPT to get the next file:")
        print(f"     python {os.path.abspath(__file__)}")
        print()
        print("=" * 78)
        return


if __name__ == "__main__":
    main()
