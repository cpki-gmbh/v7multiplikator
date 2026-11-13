# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

import sys, os, re, glob

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Logo comment checker — mirrors check_asm_comments.py behaviour:
# NO CLI PARAMETERS (a parameter would allow bypassing), the file list is
# computed dynamically from examples\ on every run, and at most PRINT_MAX
# violations are printed before the scan stops with exit 1. A clean tree
# exits 0.
#
# RULE (relaxed 1-in-3): the strict rule would demand a ';' comment on every
# line that carries a command/word. Here ONE OF EVERY THREE consecutive
# command lines must carry a non-empty ';' comment ON THE LINE ITSELF
# (a standalone comment line does not satisfy the window).
#
# Not command lines, never checked:
#   - blank lines
#   - comment-only lines (first non-space char is ';')
#   - lines with no word/number outside string literals (pure '[' / ']' etc.)
#
# A ';' inside a double-quoted string literal is not a comment. String
# parity is tracked per line (SectorLogo strings are C-style with backslash
# escapes; a malformed multi-line string degrades to per-line scanning).
#
# ---------------------------------------------------------------------------
# RULE 2 (LOGO_SHORTVAR): a bound variable name must be SPEAKING, and the
# automatable form of "speaking" is a length floor. A name shorter than
# MIN_VAR_NAME_LEN letters is rejected: two letters cannot say what the
# variable holds, so every reader of the line has to trace it. Checked on
# the binding forms `Make "name" ...` (the first literal is the name) and
# `Local "a" "b" ...` (every literal is a name), when Make/Local is the
# first token of the line -- the readable form the codebase uses. Only the
# code part of the line is scanned, so a Make quoted inside a string or
# mentioned in a trailing comment is never read as a binding.
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# RULE 3 (LOGO_BRACKET_BALANCE): square brackets must balance. This is the
# automatable guard for the mis-bracketed-Define bug class (a body block
# that closes on its header line silently turns the rest of a script into
# top-level code, and a stray ']' FATALs the parse). The scan mirrors the
# REAL reader, not a polite string-aware one: the lexer treats '[' and ']'
# as list delimiters EVEN INSIDE a double-quoted literal (the literal
# terminates and the list opens/closes), so a bracket counts everywhere
# except inside a ';' comment. Bracket depth carries across lines (string
# literals may span newlines); a ']' at depth 0 is a stray closer, and a
# non-zero depth at EOF is an unclosed '['.
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(__file__) or '.'
EXAMPLES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'examples'))

PRINT_MAX = 20

# A bound name must have at least this many letters to be accepted.
MIN_VAR_NAME_LEN = 3

# The binding primitives, in the case-folded spelling the lexer accepts.
BIND_WORDS = ('make', 'local')

WORD_RE = re.compile(r'[A-Za-z0-9_]')


def split_comment(line):
    """Return (code_part, has_comment) for one Logo line.
    has_comment is True only when a ';' outside string literals is followed
    by non-empty text — a bare ';' is an empty comment and counts as NONE
    (same semantics as the ASM checker's uncommented_text)."""
    in_string = False
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if in_string:
            if c == '\\':
                i += 2          # escaped character (\" \\ \n ...) — skip both
                continue
            if c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == ';':
                return line[:i], bool(line[i + 1:].strip())
        i += 1
    return line, False


def strip_strings(code_part):
    """Remove string literal contents so word detection sees only code."""
    out = []
    in_string = False
    i = 0
    n = len(code_part)
    while i < n:
        c = code_part[i]
        if in_string:
            if c == '\\':
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
        else:
            out.append(c)
        i += 1
    return ''.join(out)


def is_command_line(line):
    """A command line carries at least one word/number outside string
    literals. Blank lines, comment-only lines and pure bracket lines are
    not command lines."""
    code_part, _ = split_comment(line)
    return bool(WORD_RE.search(strip_strings(code_part)))


def scan_tokens(code_part):
    """Return [(kind, text)] for one Logo code fragment: ('word', text) for a
    bare word outside any literal, ('str', text) for a double-quoted literal
    with its escapes resolved. Text inside a literal is never scanned, so a
    quoted name or a quoted ';' cannot be mistaken for code."""
    tokens = []
    i = 0
    n = len(code_part)
    word_start = None
    while i < n:
        c = code_part[i]
        if c == '"':
            # close a pending bare word, then read the literal
            if word_start is not None:
                tokens.append(('word', code_part[word_start:i]))
                word_start = None
            i += 1
            chars = []
            while i < n:
                if code_part[i] == '\\' and i + 1 < n:
                    chars.append(code_part[i + 1])
                    i += 2
                    continue
                if code_part[i] == '"':
                    break
                chars.append(code_part[i])
                i += 1
            tokens.append(('str', ''.join(chars)))
            i += 1                      # consume the closing quote
            continue
        if c.isspace():
            if word_start is not None:
                tokens.append(('word', code_part[word_start:i]))
                word_start = None
            i += 1
            continue
        if word_start is None:
            word_start = i
        i += 1
    if word_start is not None:
        tokens.append(('word', code_part[word_start:]))
    return tokens


def bound_names(code_part):
    """Return the variable names a leading Make/Local binds on this line.
    `Make "x" v` binds one name (its first literal); `Local "a" "b"` binds
    every literal. Any other leading word binds nothing here."""
    tokens = scan_tokens(code_part)
    if not tokens:
        return []
    kind, text = tokens[0]
    if kind != 'word' or text.lower() not in BIND_WORDS:
        return []
    literals = [name for name_kind, name in tokens[1:] if name_kind == 'str']
    if text.lower() == 'make':
        return literals[:1]
    return literals


def bracket_depths(lines):
    """Walk the file the way the reader does and return (stray_line, final_depth).
    A '[' or ']' counts everywhere except inside a ';' comment -- including
    inside a string literal, where the real lexer terminates the literal and
    honors the bracket (Sec.146 fact 24). String state and depth persist
    across lines (literals may span newlines). stray_line is the 1-based line
    of the FIRST ']' seen at depth 0 (0 = none)."""
    depth = 0
    in_string = False
    stray_line = 0
    for lineno, raw in enumerate(lines, 1):
        i = 0
        n = len(raw)
        while i < n:
            c = raw[i]
            if in_string:
                if c == '\\':
                    i += 2              # escaped character inside the literal
                    continue
                if c == '"':
                    in_string = False   # literal closes
                elif c == '[':
                    in_string = False   # the literal terminates and a list opens
                    depth += 1
                elif c == ']':
                    in_string = False   # the literal terminates and a list closes
                    depth -= 1
                    if depth < 0 and stray_line == 0:
                        stray_line = lineno
            else:
                if c == '"':
                    in_string = True
                elif c == ';':
                    break               # rest of the line is a comment
                elif c == '[':
                    depth += 1
                elif c == ']':
                    depth -= 1
                    if depth < 0 and stray_line == 0:
                        stray_line = lineno
            i += 1
    return stray_line, depth


def check_file(filepath):
    """Return violations [(lineno, source_texts, tag, msg)] for one .logo
    file. source_texts holds the stripped source text of the offending
    line(s)."""
    violations = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # collect the command lines with their per-line comment state
    cmd_lines = []
    for i, raw in enumerate(lines, 1):
        line = raw.rstrip('\r\n')
        if not is_command_line(line):
            continue
        _, has_comment = split_comment(line)
        cmd_lines.append((i, has_comment, line.strip()))

        # rule 2: every name a leading Make/Local binds must be speaking
        code_part, _ = split_comment(line)
        for name in bound_names(code_part):
            if len(name) < MIN_VAR_NAME_LEN:
                violations.append((i, [line.strip()], 'LOGO_SHORTVAR',
                    f'Variable name "{name}" is {len(name)} character(s) long. '
                    f'A bound name must be at least {MIN_VAR_NAME_LEN} letters '
                    f'long and must be speaking.'))

    # every 3 consecutive command lines must include at least one line
    # that itself carries a ';' comment
    for start in range(0, len(cmd_lines) - 2, 3):
        window = cmd_lines[start:start + 3]
        if any(has_c for _, has_c, _ in window):
            continue
        mid_line = window[1][0]
        first_ln = window[0][0]
        last_ln = window[2][0]
        window_texts = [text for _, _, text in window]
        violations.append((mid_line, window_texts, 'LOGO_NOCOMMENT',
            f'Lines {first_ln}-{last_ln}: 3 consecutive command lines with no '
            f'; comment. At least 1 of every 3 command lines must carry a comment.'))

    # rule 3: bracket balance, mirroring the reader (brackets count outside
    # comments, even where they terminate a string literal)
    stray_line, final_depth = bracket_depths(lines)
    if stray_line:
        violations.append((stray_line, [lines[stray_line - 1].rstrip()],
            'LOGO_BRACKET_BALANCE',
            f"stray ']' with no matching '[' -- the bracket depth went negative "
            f'on this line.'))
    if final_depth > 0:
        violations.append((len(lines), [f'(end of file, {final_depth} bracket(s) still open)'],
            'LOGO_BRACKET_BALANCE',
            f'{final_depth} unclosed "[" at end of file -- a block was opened '
            f'and never closed.'))
    return violations


def main():
    if len(sys.argv) > 1:
        # Args are ignored — checker always scans the whole examples tree
        pass

    if not os.path.isdir(EXAMPLES_DIR):
        print(f"FATAL: examples directory not found: {EXAMPLES_DIR}")
        sys.exit(1)

    # File list computed dynamically EVERY run (includes subdirectories).
    # Files with _BACKUP_ in the name are frozen archaeology snapshots
    # (byte-exact copies of past states kept beside the live script) --
    # their content is the record itself, so style rules must not touch
    # them and the checker skips them.
    files = sorted(f for f in glob.glob(os.path.join(EXAMPLES_DIR, '**', '*.logo'), recursive=True)
                   if "_BACKUP_" not in os.path.basename(f))

    printed = 0
    total = 0
    files_failed = 0

    for f in files:
        errs = check_file(f)
        if not errs:
            print(f"OK: {f}")
            continue
        print(f"FAIL: {f}")
        files_failed += 1
        total += len(errs)
        for ln, source_texts, tag, msg in errs:
            if printed < PRINT_MAX:
                print(f"  Line {ln}: [{tag}] {msg}")
                for text in source_texts:
                    print(f"           > {text}")
                printed += 1
                if printed == PRINT_MAX:
                    print()
                    print("*** NO SEARCH-AND-REPLACE — SEARCH-AND-REPLACE INTRODUCES ERRORS AND IS FORBIDDEN ***")
                    print("*** Each violation MUST be fixed BY HAND, one by one, with individual edits ***")
                    print("These 20 violations must now be fixed by hand. Nothing else must be done. The scan stops now.")
                    print()
                    sys.exit(1)

    print()
    print("=" * 72)
    print("  LOGO CHECKER — RULE REPORT")
    print("=" * 72)
    print("  Rule 1 (LOGO_NOCOMMENT): every 3 consecutive command lines must")
    print("  include at least one line that itself carries a ';' comment")
    print("  (non-empty, outside strings).")
    print("  Rule 2 (LOGO_SHORTVAR): a name bound by a leading Make/Local must")
    print(f"  be at least {MIN_VAR_NAME_LEN} letters long and must be speaking.")
    print("  Rule 3 (LOGO_BRACKET_BALANCE): '[' and ']' must balance per file --")
    print("  brackets count outside ';' comments, even inside string literals,")
    print("  because the reader treats them as list delimiters there (stray ']'")
    print("  or an unclosed '[' both block the commit).")
    print("  ALL failures listed above are MUST-FIX. There are NO preexisting or")
    print("  known-good failures. Every single violation blocks the commit.")
    print(f"  Scanned {len(files)} .logo file(s) under examples\\.")
    if files_failed:
        print(f"  {files_failed} file(s) with violations.")
    print()

    if total:
        print(f"  *** {total} TOTAL VIOLATIONS — ALL MUST BE FIXED ***")
        print(f"  *** ZERO TOLERANCE — ZERO EXCEPTIONS ***")
        print()
        sys.exit(1)
    else:
        print("  ALL CHECKS PASSED — ZERO VIOLATIONS")
        print()
        sys.exit(0)


if __name__ == '__main__':
    main()
