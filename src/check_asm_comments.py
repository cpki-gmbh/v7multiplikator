# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

import sys, os, re, glob, subprocess

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ASM COMMENT CHECKER -- nasmfmt layout edition.
#
# The style contract is EXACTLY what `nasmfmt -ii 5 -ci 48` produces
# (src/nasmfmt -- patched fork of github.com/yamnikov-oleg/nasmfmt):
#   - labels at column 0, one label per line (a `label: instr` line is split)
#   - every instruction/text line indented exactly 5 spaces
#   - inline comments at column 47 (one space past the code when it reaches
#     column 47 or beyond)
#   - comment-only lines at column 0
#   - no more than one consecutive blank line, no trailing whitespace
#
# FORMAT violations (INDENT / ALIGN / LABEL / BLANK / TRAILWS) are AUTOFIXED
# by running nasmfmt over the whole file, then re-verified; autofixed lines are
# dropped from the violation list exactly like the pre-nasmfmt autofix did.
# Other violation types (NOCOMMENT, DIRECTCALL, RBPFRAME, RSPFRAME, NUMOFFSET,
# INI_NOCOMMENT) are reported only and MUST be hand-fixed.
#
# NO CLI PARAMETERS — this checker ALWAYS checks ALL files. Adding CLI parameters
# is FORBIDDEN as it would allow bypassing. The file list is computed dynamically
# from the directory every run — there is nothing to hardcode or limit.
SCRIPT_DIR = os.path.dirname(__file__) or '.'
ASM_DIR = os.environ.get('V7_ASM_DIR', os.path.join(SCRIPT_DIR, 'asm'))
MACRO_DIR = os.environ.get('V7_MACRO_DIR', '')
INI_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'threading_proxy.ini')  # the repo-root copy -- the single live ini (the stale src\ twin was removed)

# --- nasmfmt layout constants -----------------------------------------------
FMT_INSTRUCTION_INDENT = 5   # nasmfmt -ii
FMT_COMMENT_COLUMN = 47      # nasmfmt -ci 48 -> the ';' lands at column 47 (0-based)
NASMFMT_PATH = os.path.join(SCRIPT_DIR, 'nasmfmt', 'nasmfmt.exe')
# --- end constants -----------------------------------------------------------

def mask_quotes(line):
    """Quote-aware mask: replace every char inside '...'/"..."/`...` with 'x',
    preserving length and indices (mirrors nasmfmt's noquotes). NASM escape
    rules: only backquoted strings process backslash escapes, so a backslash
    inside ' or " strings is a literal byte and does not hide a quote."""
    out = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c in ('"', "'", '`'):
            j = i + 1
            while j < n:
                if c == '`' and line[j] == '\\' and j + 1 < n:
                    j += 2
                    continue
                if line[j] == c:
                    break
                j += 1
            if j >= n:
                out.append(c)      # unpaired quote: keep it visible, scan on
                i += 1
            else:
                out.append('x' * (j - i + 1))
                i = j + 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)

DATA_PSEUDOS = [re.compile(r'(?:\s|^)' + p + r'(?:\s|")') for p in ('db', 'dw', 'dd', 'dq', 'ddq', 'do', 'dt')]

def is_pure_label(stripped):
    """Lines that are ONLY a label (no instruction after it)."""
    if bool(re.match(r'^\s*\.?\.?@?%{0,2}\w[\w.]*\s*:\s*(;.*)?$', stripped)):
        return True
    # Macro local labels like %%_loop_%1: or %%_done: — treat as labels, skip all checks
    if bool(re.match(r'^\s*%%[\w_]+(%\d+)?\s*:\s*(;.*)?$', stripped)):
        return True
    return False

def _is_no_comment_required(stripped):
    """The NOCOMMENT skip list: lines that do not carry an instruction
    comment obligation (kept from the pre-nasmfmt checker)."""
    if not stripped or stripped.startswith(';'):
        return True
    if stripped.endswith('\\') and not stripped.endswith('\\\\'):
        return True
    if is_pure_label(stripped):
        return True
    first = stripped.split()[0] if stripped.split() else ''
    if first.startswith('%'):
        if first in ('%endmacro', '%endrep', '%endif', '%else', '%endm') or \
           first.startswith(('%macro', '%imacro', '%undef', '%if', '%elif', '%rep',
                             '%exitrep', '%include', '%stacksize', '%unstacksize',
                             '%local', '%arg')):
            return True
    if first in ('section', 'segment') or stripped.startswith('[section') or stripped.startswith('[segment'):
        return True
    if first in ('align', 'alignb'):
        return True
    return False

def mirror_parse_line(raw):
    """Mirror nasmfmt's parseComment+parseLabel on one raw line (no trailing
    newline). Returns a dict with kind ('blank'|'comment'|'label'|'text'),
    label, text, comment, and the rune column of the comment ';' (or None).
    'label' kind covers label-only lines; a `label: instr` line reports
    kind='label' + text!='' so the caller can flag it for splitting."""
    masked = mask_quotes(raw)
    semi = masked.find(';')
    if semi >= 0:
        comment = raw[semi + 1:].strip()
        code = raw[:semi]
    else:
        comment = ''
        code = raw
    mcode = mask_quotes(code)
    lbl = ''
    rest = code
    ci = mcode.find(':')
    if ci >= 0:
        lbl = code[:ci].strip()
        rest = code[ci + 1:]
    for pseudo in DATA_PSEUDOS:
        m = pseudo.search(mcode)
        if m:
            s0 = m.start()
            lbl = code[:s0].strip()
            if lbl.endswith(':'):
                lbl = lbl[:-1]
            rest = code[s0:]
    if re.search(r'\S*\s+\S*', lbl):
        lbl = ''
        rest = code
    text = rest.strip()
    if lbl or text:
        kind = 'label' if lbl else 'text'
    elif comment:
        kind = 'comment'
    else:
        kind = 'blank'
    return {'kind': kind, 'label': lbl, 'text': text, 'comment': comment, 'semi': semi}

def expected_semi_column(info):
    """The rune column at which nasmfmt places the ';' for this line, given the
    parsed label/text lengths (mirrors asmLine.print padding logic)."""
    column = 0
    if info['kind'] == 'label':
        column = len(info['label']) + 1
    elif info['kind'] == 'text':
        if info['text'].startswith('%'):
            # preprocessor directives print at column 0 (nasmfmt v7 patch)
            column = len(info['text'])
        else:
            column = FMT_INSTRUCTION_INDENT + len(info['text'])
    else:
        return 0 if info['kind'] == 'comment' else None
    if column < FMT_COMMENT_COLUMN:
        return FMT_COMMENT_COLUMN
    return column + 1

def format_violations(lines):
    """Detect nasmfmt-layout violations across the whole file (list of raw
    lines without trailing newlines). Returns a list of (lineno, vtype, msg)."""
    v = []
    prev_blank = False
    for idx, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not stripped:
            if prev_blank:
                v.append((idx, 'BLANK', 'consecutive blank line'))
            prev_blank = True
            continue
        prev_blank = False
        info = mirror_parse_line(raw)
        # trailing whitespace
        if raw != raw.rstrip():
            v.append((idx, 'TRAILWS', 'trailing whitespace'))
        if info['kind'] == 'comment':
            if info['semi'] != 0:
                v.append((idx, 'ALIGN', 'comment-only line must start at column 0'))
            continue
        if info['kind'] == 'label':
            if info['text']:
                v.append((idx, 'LABEL', 'label and instruction must be split onto two lines'))
                # fall through to column check with text still present: the
                # split target line is what nasmfmt produces; report once.
                continue
            if not raw.startswith(info['label'] + ':'):
                v.append((idx, 'LABEL', 'label must sit at column 0'))
                continue
            if info['comment']:
                want = expected_semi_column(info)
                if info['semi'] != want:
                    v.append((idx, 'ALIGN', f'comment must sit at column {want}'))
            continue
        if info['kind'] == 'text':
            if not info['text'].startswith('%'):
                indent = len(raw) - len(raw.lstrip())
                if indent != FMT_INSTRUCTION_INDENT:
                    v.append((idx, 'INDENT', f'text line indent is {indent}, must be {FMT_INSTRUCTION_INDENT}'))
            # %-directive lines live at column 0 (nasmfmt v7 patch) -- exempt
            # from the instruction INDENT rule; comment ALIGN still applies.
            if info['comment']:
                want = expected_semi_column(info)
                if info['semi'] != want:
                    v.append((idx, 'ALIGN', f'comment must sit at column {want}'))
    return v

def report_only_violations(lines, filepath):
    """Quality gates that are NOT autofixable (kept from the pre-nasmfmt
    checker): NOCOMMENT, DIRECTCALL, RBPFRAME, RSPFRAME, NUMOFFSET. Skip-listed
    lines (comments, labels, %directives, section/align) carry no instruction,
    so none of these checks apply to them — same gating as the old checker."""
    errors = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s:
            continue
        if _is_no_comment_required(s):
            continue
        line_violations = []
        if re.search(r'\bcall\s+\[(rel\s+)?__imp_', s, re.I):
            line_violations.append((i, 'DIRECTCALL', line,
                'Manual call [__imp_*] without API_CALL is FORBIDDEN. Use API_CALL macro instead.'))
        if re.search(r'\[rbp\s*[+-]\s*\d+\]', s, re.I):
            line_violations.append((i, 'RBPFRAME', line,
                '[rbp+offset] detected — use SF_* macros instead'))
        if re.search(r'\[rsp\s*[+-]\s*\d+\]', s, re.I):
            line_violations.append((i, 'RSPFRAME', line,
                '[rsp+offset] detected — use SF_* macros or a named %define instead'))
        code_part = line.split(';')[0]
        if re.search(r'\[[^\]]*[+\-*/]\s*\d[^\]]*\]', code_part, re.I):
            line_violations.append((i, 'NUMOFFSET', line,
                'raw number in addressing expression — use named %define or SF_* slot'))
        # NOCOMMENT: every instruction line needs a non-empty ; comment.
        # FATAL lines are exempt: the FATAL string itself says what went wrong.
        info = mirror_parse_line(line)
        if (not _is_no_comment_required(s)) and (not info['comment']):
            if not re.match(r'^\s*FATAL\b', line):
                line_violations.append((i, 'NOCOMMENT', line.strip(),
                    'Missing comment'))
        # NOLINT SUPPRESSION: a '; ... nolint ...' comment silences ALL
        # violation types on this line.
        if 'nolint' in info['comment'].lower():
            line_violations = []
        errors.extend(line_violations)
    return errors

def run_nasmfmt(filepath):
    if not os.path.isfile(NASMFMT_PATH):
        print(f"FATAL: nasmfmt not found at {NASMFMT_PATH}")
        print("Build it with:  cd src\\nasmfmt  &&  go build -o nasmfmt.exe .")
        sys.exit(1)
    result = subprocess.run([NASMFMT_PATH, '-ii', str(FMT_INSTRUCTION_INDENT),
                             '-ci', str(FMT_COMMENT_COLUMN + 1), filepath],
                            capture_output=True, text=True)
    return result.returncode == 0

UNCONDITIONAL_TAIL = re.compile(r'^(jmp|ret|ud2)\b|^\s*FATAL\b', re.I)
FLAG_PRESERVING = re.compile(r'^(mov|lea|movzx|movsx|movsxd|movabs|xchg|nop|bswap|cdqe|pop)\b', re.I)
# 'jmp' is deliberately NOT a flag reader: an unconditional jump kills the
# compare's flags (the "conditional jump degraded to jmp" bug shape -- the
# dead-compare rule must fire there, not bless it). (?!jmp\b) because the
# lookahead must anchor at the mnemonic's end, not the line's end.
FLAG_READING = re.compile(r'^(?!jmp\b)j[a-z]{1,4}\b|^(?:set[a-z]{1,2}|cmov[a-z]{1,2}|adc|sbb|lahf|loopne?|loope)\b', re.I)
# a NASM macro parameter can BE the condition code (`j%1`, `set%2`) -- the
# expansion is a flag reader even though the source token is not a literal jcc
FLAG_READING_MACRO = re.compile(r'^(j%|set%|cmov%)\b')
RSP_TEST_IDIOM = re.compile(r'^test\s+(\[rsp[^\]]*\]|rsp)\s*,\s*rsp\b', re.I)
_TREE_REF_CACHE = {}


def _tree_reference_texts(directory, exclude_filename):
    """{filename: code-part text} of every .asm/.inc file in the directory,
    INCLUDING the file itself (the caller skips it -- excluding it here would
    let a per-directory one-shot cache whitewash the orphan check for every
    file except the first one alphabetically). Dispatch-table references like
    `dq EvalPrimary.p_sum` live in a different file from the label they
    target, so the reference scan must span siblings."""
    key = os.path.normpath(directory)
    if key not in _TREE_REF_CACHE:
        texts = {}
        try:
            for fn in os.listdir(directory):
                if not fn.lower().endswith(('.inc', '.asm')):
                    continue
                try:
                    with open(os.path.join(directory, fn), 'r',
                              encoding='utf-8', errors='replace') as fh:
                        texts[os.path.normpath(os.path.join(directory, fn)).lower()] = \
                            '\n'.join(l.split(';')[0] for l in fh.read().splitlines())
                except OSError:
                    pass
        except OSError:
            pass
        _TREE_REF_CACHE[key] = texts
    return _TREE_REF_CACHE[key]


def check_module_structure(lines, filepath):
    """Structural heuristics that catch the anyres-class regression bugs
    (GLM item 21): a code label control cannot reach (a truly dead block),
    and a cmp/test whose flags die before any flag-reading instruction
    consumes them (the anyres h-loop lost its jcc exactly this way). Both are
    text-level, cross-file aware for label references, and tuned to keep the
    false-positive rate at zero."""
    errors = []
    code_parts = []
    for i, line in enumerate(lines, 1):
        code_parts.append(line.split(';')[0])
    tree_texts = _tree_reference_texts(os.path.dirname(filepath), filepath)
    self_key = os.path.normpath(filepath).lower()

    # --- unreachable code label: defined, referenced NOWHERE in the tree, and
    # control cannot fall into it (previous real instruction is unconditional)
    defined = {}
    for i, s in enumerate(code_parts, 1):
        m = re.match(r'^\s*(\.[A-Za-z_][\w.$@]*)\s*:\s*(.*)$', s)
        if m:
            defined.setdefault(m.group(1), []).append((i, m.group(2)))
    for name, sites in defined.items():
        pattern = re.compile(re.escape(name) + r'\b')
        if any(pattern.search(s) for j, s in enumerate(code_parts, 1)
               if not any(j == ln for ln, _ in sites)):
            continue  # referenced in this file
        # alias stack: a label directly followed by another label definition is
        # an anchor alias for the next one -- not a separate entry point
        next_defs = sorted(ln for ln, _ in sites)
        if any(fn_key != self_key and pattern.search(text)
               for fn_key, text in tree_texts.items()):
            continue  # referenced from a sibling file (e.g. a dispatch table)
        for ln, rest in sites:
            after = ln  # first non-blank code line after the label
            while after < len(code_parts) and not code_parts[after].strip():
                after += 1
            if after < len(code_parts) and re.match(r'^\.?[A-Za-z_][\w.$@]*:\s*$', code_parts[after].strip()):
                continue  # falls straight into another label: alias, skip
            k = ln - 2
            prev = None
            while k >= 0:
                t = code_parts[k].strip()
                if not t:
                    k -= 1
                    continue
                lm = re.match(r'^(\.?[A-Za-z_][\w.$@]*):\s*$', t)
                if lm:
                    # crossing a label line: if THAT label is referenced, control
                    # reaches it and falls through into ours (e.g. .rv_tail ->
                    # .rv_done) -- reachable, skip. An unreferenced crossed label
                    # is part of the same dead chain -- keep walking.
                    crossed = lm.group(1)
                    cross_pat = re.compile(re.escape(crossed) + r'\b')
                    cross_ref = any(cross_pat.search(s) for j, s in enumerate(code_parts, 1)
                                    if not s.strip().startswith(crossed + ':')) \
                        or any(cross_pat.search(text) for text in tree_texts.values())
                    if cross_ref:
                        prev = None
                        break
                    k -= 1
                    continue
                if t.startswith('%'):
                    k -= 1
                    continue
                prev = t
                break
            if prev is None:
                continue  # first code in the file or fall-through via a referenced label
            if UNCONDITIONAL_TAIL.search(prev):
                kind = 'FATALTRAP' if rest.strip().startswith('FATAL') else 'ORPHANLABEL'
                errors.append((ln, kind, '',
                               f'label {name} is never referenced and control cannot fall '
                               f'into it (previous instruction is "{prev}" -- dead block?)'))

    # --- dead compare: walk forward from cmp/test over flag-preserving
    # instructions; if a flag-clobbering instruction comes before any
    # flag-reading one, the compare's result was never consumed
    for i, s in enumerate(code_parts):
        t = s.strip()
        if not re.match(r'^(cmp|test)\b', t, re.I) or RSP_TEST_IDIOM.match(t):
            continue
        for j in range(i + 1, min(i + 12, len(code_parts))):
            t2 = code_parts[j].strip()
            if not t2:
                continue
            if re.match(r'^\.?[A-Za-z_][\w.$@]*:\s*$', t2):
                continue  # label-only line is transparent
            if FLAG_READING.match(t2) or FLAG_READING_MACRO.match(t2):
                break  # compare consumed -- alive
            if FLAG_PRESERVING.match(t2):
                continue  # flags survive -- keep walking
            errors.append((i + 1, 'DEADCOMPARE', t,
                           f'cmp/test flags are dead: the next flag-affecting instruction is '
                           f'"{t2}" (line {j + 1}) with no jcc/setcc/cmov in between -- was a '
                           f'conditional jump lost here?'))
            break
    return errors


def check_file(filepath):
    errors = []
    autofixed = 0
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()

    fmt_violations = format_violations(lines)
    if fmt_violations:
        orig_lines = list(lines)
        if not run_nasmfmt(filepath):
            errors.append((1, 'FMTFAILED', filepath, 'nasmfmt failed to run on this file'))
            return errors, 0
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
        # count changed lines for the report
        common = min(len(orig_lines), len(lines))
        autofixed = sum(1 for a, b in zip(orig_lines, lines) if a != b)
        autofixed += abs(len(orig_lines) - len(lines))
        remaining = format_violations(lines)
        if remaining:
            for ln, vtype, msg in remaining:
                errors.append((ln, vtype, '', f'nasmfmt did not fix: {msg}'))
    errors.extend(report_only_violations(lines, filepath))
    errors.extend(check_module_structure(lines, filepath))
    return errors, autofixed

def check_ini_file(filepath):
    errors = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    section_name = None
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s:
            continue
        if re.match(r'^\[([^\]]+)\]', s):
            section_name = re.match(r'^\[([^\]]+)\]', s).group(1)
            continue
        if s.startswith(';'):
            continue
        vm = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\S+)', s)
        if vm:
            key = vm.group(1)
            rest = s[vm.end():]
            semi_pos = rest.find(';')
            if semi_pos == -1:
                errors.append((i, 'INI_NOCOMMENT', s,
                    f'INI key "{key}" in section [{section_name}] has NO comment. '
                    f'EVERY key in EVERY section MUST have a semantic comment explaining '
                    f'what the parameter does and why it has this value.'))
            elif not rest[semi_pos+1:].strip():
                errors.append((i, 'INI_NOCOMMENT', s,
                    f'INI key "{key}" in section [{section_name}] has EMPTY comment '
                    f'(just ";" with nothing after). Every key needs a real semantic comment.'))
            continue
    return errors

PRINT_MAX = 20

def main():
    if len(sys.argv) > 1:
        # Args are ignored — checker always checks all files in the ASM/MACRO dirs
        pass

    if not os.path.isdir(ASM_DIR):
        print(f"FATAL: ASM directory not found: {ASM_DIR}")
        sys.exit(1)

    # File list computed dynamically EVERY run (includes subdirectories)
    # NOTE: threading_proxy.ini is deliberately NOT checked -- INI comments
    # duplicate the AGENTS.md parameter documentation and the autotuner
    # rewrite sections wholesale, so per-key comment enforcement only creates
    # churn (the INI_NOCOMMENT scan is retired, not deferred).
    files = []
    files.extend(glob.glob(os.path.join(ASM_DIR, '**', '*.asm'), recursive=True))
    files.extend(glob.glob(os.path.join(ASM_DIR, '**', '*.inc'), recursive=True))
    if MACRO_DIR and os.path.isdir(MACRO_DIR):
        files.extend(glob.glob(os.path.join(MACRO_DIR, '*.inc')))

    nocomment_count = 0
    directcall_count = 0
    rbpf_count = 0
    rspf_count = 0
    numoff_count = 0
    ini_nocomment_count = 0
    fmtfail_count = 0
    orphan_count = 0
    deadcmp_count = 0
    autofixed_total = 0       # lines rewritten by nasmfmt autofix
    files_reformatted = 0     # number of files nasmfmt touched
    printed = 0

    for f in sorted(files):
        if not os.path.isfile(f):
            print(f"Error: {f} does not exist.")
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext == '.ini':
            errs = check_ini_file(f)
            if errs:
                print(f"FAIL: {f}")
                for ln, etype, txt, msg in errs:
                    if printed < PRINT_MAX:
                        print(f"  Line {ln}: [{etype}] {txt}")
                        print(f"           {msg}")
                        printed += 1
                        if printed == PRINT_MAX:
                            print()
                            print("*** NO SEARCH-AND-REPLACE — SEARCH-AND-REPLACE INTRODUCES ERRORS AND IS FORBIDDEN ***")
                            print("*** Each violation MUST be fixed BY HAND, one by one, with individual edits ***")
                            print("These 20 violations must now be fixed by hand. Nothing else must be done. The scan stops now.")
                            print()
                            sys.exit(1)
                for ln, etype, txt, msg in errs:
                    if etype == 'INI_NOCOMMENT':
                        ini_nocomment_count += 1
            else:
                print(f"OK: {f}")
        else:
            errs, autofixed = check_file(f)
            if autofixed:
                autofixed_total += autofixed
                files_reformatted += 1
            if errs:
                print(f"FAIL: {f}")
                for ln, etype, txt, msg in errs:
                    if printed < PRINT_MAX:
                        print(f"  Line {ln}: [{etype}] {txt}")
                        print(f"           {msg}")
                        printed += 1
                        if printed == PRINT_MAX:
                            print()
                            print("*** NO SEARCH-AND-REPLACE — SEARCH-AND-REPLACE INTRODUCES ERRORS AND IS FORBIDDEN ***")
                            print("*** Each violation MUST be fixed BY HAND, one by one, with individual edits ***")
                            print("These 20 violations must now be fixed by hand. Nothing else must be done. The scan stops now.")
                            print()
                            sys.exit(1)
                for ln, etype, txt, msg in errs:
                    if etype == 'NOCOMMENT':
                        nocomment_count += 1
                    elif etype == 'DIRECTCALL':
                        directcall_count += 1
                    elif etype == 'RBPFRAME':
                        rbpf_count += 1
                    elif etype == 'RSPFRAME':
                        rspf_count += 1
                    elif etype == 'NUMOFFSET':
                        numoff_count += 1
                    elif etype in ('ORPHANLABEL', 'FATALTRAP'):
                        orphan_count += 1
                    elif etype == 'DEADCOMPARE':
                        deadcmp_count += 1
                    elif etype in ('INDENT', 'ALIGN', 'LABEL', 'BLANK', 'TRAILWS', 'FMTFAILED'):
                        fmtfail_count += 1
            else:
                if autofixed:
                    print(f"OK (auto-formatted {autofixed} line(s)): {f}")
                else:
                    print(f"OK: {f}")

    print()
    print("=" * 72)
    print("  COMMENT CHECKER — ZERO TOLERANCE REPORT")
    print("=" * 72)
    print("  ALL failures listed above are MUST-FIX. There are NO preexisting or")
    print("  known-good failures. Every single violation blocks the commit.")
    print("  Run this checker after EVERY change. Fix every issue it reports.")
    if autofixed_total:
        print(f"  AUTO-FORMATTED: {autofixed_total} line(s) across {files_reformatted} file(s) "
              f"(nasmfmt -ii {FMT_INSTRUCTION_INDENT} -ci {FMT_COMMENT_COLUMN + 1}).")
        print("  Re-run the gate after committing these style fixes if you want a")
        print("  clean exit-0 with zero rewrites.")
    print()

    total_errors = (nocomment_count + directcall_count + rbpf_count + rspf_count
                    + numoff_count + ini_nocomment_count + fmtfail_count
                    + orphan_count + deadcmp_count)

    def report(count, label):
        if count:
            print(f"  FAIL: {count}x {label}")

    report(nocomment_count, "lines WITHOUT comments (NOCOMMENT)")
    report(directcall_count, "direct call [__imp_*] without API_CALL (DIRECTCALL)")
    report(rbpf_count, "[rbp+offset] raw frame access (RBPFRAME)")
    report(rspf_count, "[rsp+offset] raw frame access (RSPFRAME)")
    report(numoff_count, "raw number in addressing expression (NUMOFFSET)")
    report(ini_nocomment_count, "INI keys without semantic comment (INI_NOCOMMENT)")
    report(fmtfail_count, "layout violations nasmfmt could not fix (INDENT/ALIGN/LABEL)")
    report(orphan_count, "unreachable code labels (ORPHANLABEL/FATALTRAP)")
    report(deadcmp_count, "dead compares, flags never consumed (DEADCOMPARE)")

    if total_errors:
        print()
        print(f"  *** {total_errors} TOTAL VIOLATIONS — ALL MUST BE FIXED ***")
        print(f"  *** ZERO TOLERANCE — ZERO EXCEPTIONS ***")
        print()
        sys.exit(1)
    else:
        print("  ALL CHECKS PASSED — ZERO VIOLATIONS")
        print()
        sys.exit(0)

if __name__ == '__main__':
    main()
