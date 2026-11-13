# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

# logo_fmt.py -- SectorLogo autoformatter (Allman body style).
#
# Reformats .logo sources so that every BODY bracket pair -- the [ ... ] of
# Define bodies, If/IfElse branches and Repeat bodies -- opens with [
# alone on its own line and closes with ] alone on its own line, one tab
# deeper per open body (Sec.137 house style). Define parameter lists,
# If/IfElse conditions, list literals in argument position and inline
# empty-pair NIL literals stay inline; an EMPTY else branch renders as []
# alone on its own line (Sec.137). Parenthesized expressions stay inline
# but preserve their own newlines as same-indent continuation lines.
#
# Safety: string literals (C-style "..." with \\ and \" escapes, able to
# span newlines) and ; comments are tokenized verbatim and re-emitted
# byte-exact; tokens are never modified or reordered. The formatter only
# inserts line breaks, indentation and single spaces between tokens. A
# token-stream equality check between input and output guards every write:
# any mismatch aborts without touching the file. An unbalanced source
# (stray [ or ]) aborts outright -- formatting must never touch a broken
# script.
#
# Usage:
#   python tools/logo_fmt.py <file.logo>       print the formatted output
#   python tools/logo_fmt.py <file.logo> -i    format in place (gated)

import sys

T_ATOM = "atom"
T_STRING = "string"
T_OPEN = "open"      # [
T_CLOSE = "close"    # ]
T_LP = "lp"          # (
T_RP = "rp"          # )
T_COMMENT = "comment"
T_NL = "newline"


def tokenize(text):
    tokens = []
    i = 0
    if text.startswith('\ufeff'):
        i = 1  # a UTF-8 BOM in the source file is not Logo content
    open_brackets = 0
    while i < len(text):
        c = text[i]
        if c == '"':
            j = i + 1
            while j < len(text):
                if text[j] == '\\' and j + 1 < len(text):
                    j += 2  # \\ or \" escape: both bytes are literal data
                elif text[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            else:
                sys.exit("logo_fmt: unterminated string in source")
            tokens.append((T_STRING, text[i:j]))
            i = j
        elif c == ';':
            j = text.find('\n', i)
            if j < 0:
                j = len(text)
            comment_text = text[i:j]
            if comment_text.endswith('\r'):
                comment_text = comment_text[:-1]  # CRLF files: the CR is the line break, not comment content
            tokens.append((T_COMMENT, comment_text))
            if j < len(text):
                tokens.append((T_NL, "\n"))
                i = j + 1  # consume the newline WITH the comment (no double emit)
            else:
                i = j
        elif c == '\n':
            tokens.append((T_NL, "\n"))
            i += 1
        elif c == '[':
            open_brackets += 1
            tokens.append((T_OPEN, "["))
            i += 1
        elif c == ']':
            if open_brackets == 0:
                sys.exit("logo_fmt: unbalanced ] (stray closer) -- broken script, not formatting")
            open_brackets -= 1
            tokens.append((T_CLOSE, "]"))
            i += 1
        elif c == '(':
            tokens.append((T_LP, "("))
            i += 1
        elif c == ')':
            tokens.append((T_RP, ")"))
            i += 1
        elif c in ' \t\r':
            i += 1
        else:
            j = i
            while j < len(text) and text[j] not in ' \t\r\n;[]"()':
                j += 1
            tokens.append((T_ATOM, text[i:j]))
            i = j
    if open_brackets != 0:
        sys.exit("logo_fmt: unbalanced [ (missing closer) -- broken script, not formatting")
    return tokens


# one slot sequence per keyword after the keyword token: define takes
# name(inline string), params(inline [..] list), body(Allman); ifelse takes
# cond, then, else; if takes cond, then; repeat runs a count (paren/atom)
# then a body -- a [ form straight after the keyword means the ENDLESS
# form: that single form is the body.
KEYWORD_SLOTS = {
    "define": ["name", "params", "body"],
    "ifelse": ["cond", "then", "else"],
    "if": ["cond", "then"],
    "repeat": ["count", "body"],
}


def skip_noise(tokens, i, hi):
    while i < hi and tokens[i][0] in (T_NL, T_COMMENT):
        i += 1
    return i


def is_empty_form(tokens, lo, hi):
    return skip_noise(tokens, lo, hi) >= hi


def scan_form(tokens, i, hi):
    """End index after ONE balanced form starting at i (atom/string, a
    balanced ( ) group or a balanced [ ] group); hi when nothing fits."""
    j = skip_noise(tokens, i, hi)
    if j >= hi:
        return hi
    kind = tokens[j][0]
    if kind in (T_ATOM, T_STRING):
        return j + 1
    if kind in (T_LP, T_OPEN):
        open_kind = T_LP if kind == T_LP else T_OPEN
        close_kind = T_RP if kind == T_LP else T_CLOSE
        depth = 0
        while j < hi:
            k = tokens[j][0]
            if k == open_kind:
                depth += 1
            elif k == close_kind:
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return hi
    return scan_form(tokens, j + 1, hi) if kind == T_NL else hi


def scan_inline_region(tokens, i, hi):
    """End index off the line: a statement-position bare [ or ( that is an
    inline group -- its extent, balanced."""
    return scan_form(tokens, i, hi)


class Printer:
    def __init__(self):
        self.lines = []
        self.chunks = []
        self.pending_space = False
        self.depth = 0

    def flush(self):
        if self.chunks:
            self.lines.append("".join(self.chunks).rstrip())
            self.chunks = []
        self.pending_space = False

    def add(self, text, spaced=True):
        if not self.chunks:
            self.chunks.append("\t" * self.depth)  # indent stamp: every output line starts at the body depth
        if self.pending_space and spaced and self.chunks and self.chunks[-1] != "\t" * self.depth:
            self.chunks.append(" ")
            self.pending_space = False
        self.chunks.append(text)
        self.pending_space = True

    def nl(self, blank=False):
        """End the logical line. blank=True emits ONE separator blank line
        (a run of 2+ source newlines), never two in a row."""
        need_blank = blank
        self.flush()
        if need_blank and self.lines and self.lines[-1] != "":
            self.lines.append("")

    def line(self, text):
        """A complete line at the current body depth (no chunk assembly)."""
        self.flush()
        self.lines.append("\t" * self.depth + text)


def emit_inline(tokens, lo, hi, pr):
    """Emit a NON-body extent inline (single spaces; its own newlines become
    same-indent continuation lines). Keywords inside literals stay raw.
    Spacing: no space right after ( or [, no space right before ) or ]."""
    first_inner = False  # set after ( / [ : the first value token hugs it
    i = lo
    while i < hi:
        kind, tok = tokens[i]
        if kind == T_NL:
            pr.flush()
            first_inner = True  # a fresh line also hugs ( or [ if one opens it
        elif kind == T_COMMENT:
            pr.add(tok)
            pr.flush()
            first_inner = True
        elif kind == T_LP:
            pr.add(tok, spaced=(len(pr.chunks) > 1))
            pr.pending_space = False
            first_inner = True
        elif kind == T_RP:
            pr.pending_space = False
            pr.add(tok, spaced=False)
            first_inner = False
        elif kind == T_OPEN:
            pr.add(tok, spaced=(len(pr.chunks) > 1))
            pr.pending_space = False
            first_inner = True
        elif kind == T_CLOSE:
            pr.pending_space = False
            pr.add(tok, spaced=False)
            first_inner = False
        else:
            pr.add(tok, spaced=not first_inner)
            first_inner = False
        i += 1
    return hi


def find_extent_end(tokens, open_index, hi):
    open_kind = tokens[open_index][0]
    close_kind = T_RP if open_kind == T_LP else T_CLOSE
    depth = 0
    j = open_index
    while j < hi:
        if tokens[j][0] == open_kind:
            depth += 1
        elif tokens[j][0] == close_kind:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return hi


def emit_body(tokens, open_index, hi, pr):
    """Allman body: [ alone, content one tab deeper, ] alone. Recurses"""
    if tokens[open_index][0] != T_OPEN:
        sys.exit("logo_fmt: emit_body on a non-body form")
    close_index = find_extent_end(tokens, open_index, hi)
    pr.flush()
    pr.chunks = ["\t" * pr.depth, "["]
    pr.flush()
    pr.depth += 1
    emit_statement_run(tokens, open_index + 1, close_index, pr)
    pr.depth -= 1
    pr.flush()
    pr.lines.append("\t" * pr.depth + "]")
    return close_index + 1


def emit_inline_list(tokens, open_index, hi, pr):
    """A [..] group in ARGUMENT position: emit fully inline verbatim-ish."""
    close_index = find_extent_end(tokens, open_index, hi)
    pr.add("[", spaced=(len(pr.chunks) > 1))
    pr.pending_space = False
    for kind, tok in tokens[open_index + 1:close_index]:
        if kind == T_NL:
            pr.flush()
            continue
        if kind == T_COMMENT:
            pr.add(tok)
            pr.flush()
            continue
        pr.add(tok, spaced=(kind in (T_ATOM, T_STRING)))
        pr.pending_space = True
    pr.pending_space = False
    pr.add("]", spaced=False)
    pr.pending_space = True
    return close_index + 1


def emit_statement_run(tokens, lo, hi, pr):
    """Bodies and top level: statements separated by their newlines; a run
    of 2+ newlines becomes ONE blank separator line. Keywords get slot
    treatment; a bare [ after an atom/string arg is an inline argument
    literal, a bare [ at a statement edge is an Allman body."""
    i = lo
    prev_sig = None
    nl_streak = 0
    while i < hi:
        kind, tok = tokens[i]
        if kind == T_NL:
            if nl_streak == 0:
                pr.nl(False)
            nl_streak += 1
            if nl_streak == 2:
                pr.nl(True)
            i += 1
            continue
        nl_streak = 0
        if kind == T_COMMENT:
            pr.add(tok)
            pr.flush()
            i += 1
            continue
        low = tok.lower()
        if kind == T_ATOM and low in KEYWORD_SLOTS:
            pr.flush()
            i = emit_control(tokens, i, hi, pr)
            prev_sig = (T_ATOM, low)
            continue
        if kind == T_OPEN:
            if prev_sig is not None and prev_sig[0] in (T_ATOM, T_STRING, T_RP, T_CLOSE):
                i = emit_inline_list(tokens, i, hi, pr)
            else:
                i = emit_body(tokens, i, hi, pr)
            prev_sig = (T_CLOSE, "]")
            continue
        if kind == T_LP:
            close_index = find_extent_end(tokens, i, hi)
            emit_paren_group(tokens, i, close_index, pr)  # inline, attached to the statement on the same line
            prev_sig = (T_RP, ")")
            i = close_index + 1
            continue
        pr.add(tok)
        prev_sig = (kind, tok)
        i += 1
    if nl_streak >= 2:
        pr.nl(True)


def emit_paren_group(tokens, open_index, close_index, pr):
    """( ... ) inline: a space before ( when the line already has content,
    no space after ( and before ), single spaces between the members."""
    pr.add("(", spaced=(len(pr.chunks) > 1))
    pr.pending_space = False
    first_inner = True
    j = open_index + 1
    while j < close_index:
        kind, tok = tokens[j]
        if kind == T_NL:
            pr.flush()
            first_inner = True
        elif kind == T_COMMENT:
            pr.add(tok)
            pr.flush()
            first_inner = True
        elif kind == T_LP:
            pr.add(tok, spaced=(len(pr.chunks) > 1))
            pr.pending_space = False
            first_inner = True
        elif kind == T_RP:
            pr.pending_space = False
            pr.add(tok, spaced=False)
            first_inner = False
        else:
            pr.add(tok, spaced=not first_inner)
            first_inner = False
        j += 1
    pr.pending_space = False
    pr.add(")", spaced=False)
    pr.pending_space = True


def emit_control(tokens, i, hi, pr):
    """Emit one control statement starting at the keyword token i. The
    condition/params/name forms render inline; the then/else/body forms
    render as Allman bodies (an empty branch as [] -- inline for then,
    own-line for else)."""
    keyword = tokens[i][1].lower()
    pr.add(tokens[i][1])
    cursor = i + 1
    probe = skip_noise(tokens, cursor, hi)
    if keyword == "repeat" and probe < hi and tokens[probe][0] == T_OPEN:
        slots = ["body"]  # endless form: the single [ form IS the body
    else:
        slots = KEYWORD_SLOTS[keyword]
    for slot in slots:
        # comments between slot forms (e.g. after an inline [] then-slot,
        # before the else body) are CONTENT: emit them, never swallow them
        while cursor < hi and tokens[cursor][0] == T_COMMENT:
            pr.add(tokens[cursor][1])
            pr.flush()
            cursor += 1
        first_index = skip_noise(tokens, cursor, hi)
        if first_index >= hi:
            break  # the statement ends before this slot has anything at all
        if slot == "cond":
            # the condition is EVERY top-level form between the keyword and
            # the then-slot's opening [ (an atom like IsNumber followed by
            # its ( ...) group is TWO forms). PEEK with skip_noise first:
            # calling scan_form on a newline token would recurse straight
            # past the peek and swallow the then-body's whole [ extent.
            j = first_index
            while j < hi:
                peek = skip_noise(tokens, j, hi)
                if peek >= hi:
                    break
                if tokens[peek][0] == T_OPEN:
                    j = peek
                    break  # the then slot (or an [] empty-then pair) begins here
                j = scan_form(tokens, j, hi)
            if j > first_index:
                emit_inline(tokens, cursor, j, pr)
                cursor = j
            continue
        form_start = scan_form(tokens, cursor, hi)
        # an empty [..] pair (nothing between the brackets) is the slot's
        # literal [] -- a non-empty body extent has values inside
        empty_pair = (tokens[first_index][0] == T_OPEN) and \
            (skip_noise(tokens, first_index + 1, hi) >= find_extent_end(tokens, first_index, hi))
        if slot == "count":
            if empty_pair:
                break  # no count present: the body form follows directly
            emit_inline(tokens, cursor, form_start, pr)
            cursor = form_start
            continue
        if slot in ("name", "params"):
            if empty_pair:
                pr.add("[]", spaced=(len(pr.chunks) > 1))  # params stay inline even when empty
                pr.pending_space = True
            else:
                emit_inline(tokens, cursor, form_start, pr)
            cursor = form_start
            continue
        # then / else / body slots
        if empty_pair:
            if slot == "then":
                pr.add("[]", spaced=(len(pr.chunks) > 1))  # empty then: inline pair after the condition
                pr.pending_space = True
            else:
                pr.line("[]")  # empty else/body: [] alone on its own line (Sec.137)
            cursor = form_start
            continue
        if tokens[first_index][0] == T_OPEN:
            cursor = emit_body(tokens, first_index, form_start, pr)
        else:
            emit_inline(tokens, cursor, form_start, pr)
            cursor = form_start
    return cursor


def format_logo(source_text):
    tokens = tokenize(source_text)
    pr = Printer()
    emit_statement_run(tokens, 0, len(tokens), pr)
    pr.flush()
    return "\n".join(pr.lines) + "\n"


def gate_tokens(tokens):
    """Content tokens for the write gate: values + comments, verbatim."""
    out = []
    for kind, tok in tokens:
        if kind in (T_ATOM, T_STRING, T_OPEN, T_CLOSE, T_LP, T_RP, T_COMMENT):
            out.append((kind, tok))
    return out


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python logo_fmt.py <file.logo> [-i]")
        return 1
    path = args[0]
    inplace = "-i" in args
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    with open(path, "r", encoding="utf-8", newline="") as f:
        source = f.read()
    formatted = format_logo(source)
    if gate_tokens(tokenize(source)) != gate_tokens(tokenize(formatted)):
        print("logo_fmt: token gate FAILED -- output would change content, nothing written")
        with open(path + ".byfmt", "w", encoding="utf-8") as f:
            f.write(formatted)  # side-by-side for manual review, never auto-applied
        return 2
    if inplace:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(formatted)
        print("logo_fmt: formatted in place (token gate passed): %s" % path)
    else:
        print(formatted, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
