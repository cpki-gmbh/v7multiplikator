# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

#!/usr/bin/env python3
"""check_stale.py -- conservative freshness gate for build.bat steps 2-6.

Decides whether the C/C++ portion of the build (every `cl` invocation, plus the
one in-step NASM target dbg_inference.exe) can be skipped because nothing it was
built from has changed.

The design rule is that this checker may only ever COST TIME. A wrong FRESH would
leave a stale binary that the validators then certify as passing, which is the
worst failure mode available to us. Every ambiguity -- missing output, missing
input, unparseable manifest, unexpected exception -- resolves to STALE.

Exit codes (build.bat branches on these):
    0  FRESH   -- the checked step(s) may be skipped
    1  STALE   -- the checked step(s) must run
    2  ERROR   -- checker could not decide; build.bat treats this as STALE

Usage:
    python check_stale.py --root <project-root> --manifest <manifest-path>

Step mode (--step): restrict the check to one build step's manifest records.
The manifest's "# ====== Step N/6 ... ======" group headers assign each record
to a step token ("2", "3", "4", "4b", "5", "6"); --step takes a comma list of
tokens. Per-step decisions are what make the C/C++ part incremental: a stale
example EXE rebuilds the examples, not all 74 tools. The conservative
direction is unchanged: a missing output or input inside the selected step
rebuilds that step, and every ambiguity resolves to STALE.
"""

import argparse
import os
import sys


# Exit codes, named so the fail-safe sites read as intent rather than magic numbers.
EXIT_FRESH = 0
EXIT_STALE = 1
EXIT_ERROR = 2

# Always an input, for every target: the static Zydis import library the dev
# cl/link lines pull in. The release ships the no-op stub instead (no .lib at
# all) -- in a stub tree the file is simply absent and must not force a full
# rebuild on every run, so it is only an input when it exists; the stub tree's
# own sources take its place as inputs (see STUB_DIR below).
ZYDIS_LIB = "Zydis.lib"
STUB_DIR = "tools/zydis_stub"

# Build-rule inputs: a change to the build orchestrator, to the manifest that
# defines the target set, or to the DLL export surface the tools link against
# must never leave existing binaries classified FRESH (a compile-flag-only or
# link-surface change is invisible to source timestamps).
BUILD_RULE_INPUTS = [
    "build.bat",
    "src/build_c_targets.txt",
    "src/v7_threading.def",
]


def fail(reason):
    """Report an undecidable state. Never raises: an error is just another STALE."""
    print("  decision        : STALE   (checker could not decide -- %s)" % reason)
    return EXIT_ERROR


def read_manifest(manifest_path):
    """Parse the manifest into (source, outputs, extras, step) tuples.

    Returns (records, None) on success or (None, reason) on a malformed line.
    The step field comes from the "# ====== Step N/6 ... ======" group headers:
    the token before the "/6" ("2", "3", "4", "4b", ...). Records above the
    first header carry step None.
    """
    records = []
    current_step = None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                marker = _step_header_token(line)
                if marker is not None:
                    current_step = marker
                continue
            fields = line.split("|")
            if len(fields) < 2 or len(fields) > 3:
                return None, "manifest line %d has %d fields, expected 2 or 3" % (
                    line_number,
                    len(fields),
                )
            source = fields[0].strip()
            outputs = [part.strip() for part in fields[1].split(";") if part.strip()]
            extras = []
            if len(fields) == 3:
                extras = [part.strip() for part in fields[2].split(";") if part.strip()]
            if not source or not outputs:
                return None, "manifest line %d has an empty source or output field" % line_number
            records.append((source, outputs, extras, current_step))
    if not records:
        return None, "manifest contains no records"
    return records, None


def _step_header_token(comment_line):
    """The step token of a '# ====== Step N/6 ... ======' header, else None.

    Only lines that are genuine step headers carry the '======' bookend marker
    that every real header in the manifest uses. Prose comments that happen to
    mention 'step N' (e.g. 'see the SKIP note at its build.bat step 3') must
    NOT be misread as a header -- they would reassign every following record to
    the wrong step and silently skip rebuilds.
    """
    text = comment_line.lower()
    if "======" not in text:
        return None
    idx = text.find("step ")
    if idx < 0:
        return None
    rest = text[idx + len("step "):]
    token = ""
    for ch in rest:
        if ch.isalnum():
            token += ch
        else:
            break
    return token if token else None


def expand_extra(root, extra):
    """Resolve one field-3 extra into a list containing one concrete input path.

    Extras are always single files. The tree-walk form (a trailing slash meaning
    "every file under this tree") is deliberately NOT supported: dbg_inference.asm
    was the only user, and its include closure is exactly five files that fit in
    the manifest. A trailing slash here resolves to a directory, fails the isfile
    test below, and reports a missing input -- i.e. STALE, the fail-safe direction.
    """
    absolute = os.path.join(root, extra.replace("/", os.sep))
    if not os.path.isfile(absolute):
        return None, "extra input is missing: %s" % extra
    return [absolute], None


def stub_inputs(root):
    """Every .c/.h source under tools/zydis_stub -- the stub's own sources are
    the compile inputs in a stub tree (the build compiles them into the tools
    when Zydis.lib is absent), so a stub change must mark the tools stale."""
    inputs = []
    stub_root = os.path.join(root, STUB_DIR.replace("/", os.sep))
    if not os.path.isdir(stub_root):
        return inputs
    for dirpath, dirnames, filenames in os.walk(stub_root):
        for fn in filenames:
            if fn.lower().endswith(('.c', '.h', '.cpp')):
                inputs.append(os.path.join(dirpath, fn))
    return inputs


def headers_in(directory):
    """Every *.h directly in `directory` (non-recursive), or [] if absent."""
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith(".h")
    ]


def collect_inputs(root, source, extras):
    """Build one target's full input set.

    The rule is deliberately a superset of the real dependency set: the source,
    the field-3 extras, Zydis.lib, and every *.h in the source's own directory,
    in src/, and in the project root. Superset means it can only over-trigger a
    rebuild, never under-trigger one, which is the only safe direction.

    The superset is over C/C++ inputs only. ASM sources (.inc / .asm) enter a
    target's set ONLY through that target's own field-3 extras, so an edit to an
    assembly file rebuilds the one binary that is actually assembled from it
    (dbg_inference.exe) and does not mark the other 73 C/C++ targets stale.
    """
    inputs = []
    missing = []

    source_path = os.path.join(root, source.replace("/", os.sep))
    if os.path.isfile(source_path):
        inputs.append(source_path)
    else:
        missing.append(source)

    zydis = os.path.join(root, ZYDIS_LIB)
    if os.path.isfile(zydis):
        inputs.append(zydis)
    else:
        # stub tree (release): the .lib does not exist by design and must not
        # flag the whole manifest STALE (P2 fix, second audit); the stub's own
        # sources are the inputs that replace it -- a stub change must mark the
        # tools stale exactly like a .lib change would
        inputs.extend(stub_inputs(root))

    for extra in extras:
        resolved, reason = expand_extra(root, extra)
        if resolved is None:
            return None, reason
        inputs.extend(resolved)

    source_dir = os.path.dirname(source_path)
    inputs.extend(headers_in(source_dir))
    inputs.extend(headers_in(os.path.join(root, "src")))
    inputs.extend(headers_in(root))

    # build-rule inputs: the orchestrator, the manifest and the export surface
    # are inputs to EVERY C/C++ target (missing ones flag the target, like any
    # other input -- the manifest is the arg, so a missing one means a broken
    # invocation and the fail() path above has already fired)
    for rule_input in BUILD_RULE_INPUTS:
        rule_path = os.path.join(root, rule_input.replace("/", os.sep))
        if os.path.isfile(rule_path):
            inputs.append(rule_path)
        else:
            missing.append(rule_input)

    if missing:
        return None, "input file(s) missing: %s" % ", ".join(sorted(set(missing)))
    return inputs, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="project root that manifest paths are relative to")
    parser.add_argument("--manifest", required=True, help="path to build_c_targets.txt")
    parser.add_argument("--step", required=False, default=None,
                        help="comma list of step tokens to check (e.g. 3 or 4,4b); "
                             "omit to check every record")
    args = parser.parse_args()

    root = os.path.abspath(args.root)

    try:
        records, reason = read_manifest(args.manifest)
    except OSError as exc:
        return fail("cannot read manifest: %s" % exc)
    if records is None:
        return fail(reason)

    if args.step is not None:
        wanted = set(token.strip().lower() for token in args.step.split(",") if token.strip())
        if not wanted:
            return fail("--step carried no usable token")
        unassigned = sum(1 for record in records if record[3] is None)
        if unassigned > 0:
            return fail("%d record(s) sit above every Step header and have no step token" % unassigned)
        records = [record for record in records if record[3] in wanted]
        if not records:
            return fail("--step %s matches no manifest record" % args.step)
        print("  step                : %s  (%d record(s))" % (args.step, len(records)))

    all_inputs = []
    all_outputs = []
    missing_outputs = []

    for source, outputs, extras, _step in records:
        # Missing output is checked FIRST and independently of the input scan, so
        # a deleted .exe can never be skipped even if the input set is fine.
        for output in outputs:
            output_path = os.path.join(root, output.replace("/", os.sep))
            if os.path.isfile(output_path):
                all_outputs.append(output_path)
            else:
                missing_outputs.append(output)

        inputs, reason = collect_inputs(root, source, extras)
        if inputs is None:
            return fail("%s: %s" % (source, reason))
        all_inputs.extend(inputs)

    if missing_outputs:
        print("  outputs missing : %d  (%s%s)" % (
            len(missing_outputs),
            ", ".join(missing_outputs[:4]),
            ", ..." if len(missing_outputs) > 4 else "",
        ))
    print("  targets in manifest : %d" % len(records))
    print("  inputs scanned     : %d" % len(set(all_inputs)))
    print("  outputs checked    : %d" % len(set(all_outputs)))

    if missing_outputs:
        return EXIT_STALE

    oldest_output = min(os.path.getmtime(path) for path in all_outputs)
    newest_input = max(os.path.getmtime(path) for path in all_inputs)

    import time
    newest_input_name = max(set(all_inputs), key=os.path.getmtime)
    oldest_output_name = min(set(all_outputs), key=os.path.getmtime)

    def stamp(path):
        return time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(os.path.getmtime(path)))

    print("  newest input       : %s  %s" % (stamp(newest_input_name), newest_input_name.replace(root + os.sep, "")))
    print("  oldest output      : %s  %s" % (stamp(oldest_output_name), oldest_output_name.replace(root + os.sep, "")))

    # The bar is the OLDEST output, not the newest. An input newer than even the
    # slowest-produced output means at least one binary is provably older than
    # something it was built from. Comparing against the newest output would be
    # unsound: it would skip while the oldest binary is stale.
    if newest_input > oldest_output:
        print("  decision            : STALE   (an input is newer than the oldest output)")
        return EXIT_STALE

    print("  decision            : FRESH   (every input older than every output)")
    return EXIT_FRESH


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as unexpected:  # catch-all: an exception must never mean FRESH
        print("  decision        : STALE   (checker raised %r)" % (unexpected,))
        sys.exit(EXIT_ERROR)
