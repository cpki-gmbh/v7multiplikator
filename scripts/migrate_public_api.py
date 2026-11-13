# migrate_public_api.py -- bulk-migrate tools/examples to the public v7_* API tier
# (docs/plans/move_inference_context_to_bss_plan.md section 5.1). Reports every file touched
# and every line it could NOT rewrite (the section 5.2 hand-fix list).
#
# Byte-safety: files are read/written as latin-1 with newline="" so every byte
# round-trips 1:1 (no encoding or CRLF/LF translation).
#
# NOT rewritten mechanically (left for hand fixes, reported):
#   - v7s_inference_create capture sites (need bool-init restructure)
#   - ctx-> field reads (need the lib variable receiver, only hand fixes know it)
#   - ctx-> field writes (converted with receiver "dll." heuristic + REPORTED)
#   - ht_add/ht_find call sites (raw tier stays for DB tools; only app-side
#     sidecar staging migrates -- by hand)
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
SCAN_DIRS = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "examples")]
EXCLUDE_PARTS = ("backup", "web_ragg", "rag")  # web_ragg/rag subdirs need hand fixes (multi-receiver)

REPORT = []
TOUCHED = []

SIMPLE = [
    (re.compile(r"\b(\w+)\.V7_init\("), r"\1.v7_init("),
    (re.compile(r"\b(\w+)\.v7s_inference_destroy\(\s*[A-Za-z_]\w*\s*\)"), r"\1.v7_free_inference()"),
    (re.compile(r"\b(\w+)\.v7s_enable_thinking\("), r"\1.v7_enable_thinking("),
    (re.compile(r"\b(\w+)\.v7s_shutdown\("), r"\1.v7_shutdown("),
    (re.compile(r"\b(\w+)\.v7s_sectorlogo_set_db\("), r"\1.v7_set_sidecar("),
    (re.compile(r"\b(\w+)\.v7s_get_generation_statistic\("), r"\1.v7_get_generation_statistic("),
]

CTX_DROP = [
    (re.compile(r"\b(\w+)\.v7s_chat\(\s*[A-Za-z_]\w*\s*,\s*"), r"\1.v7_chat("),
    # NOTE: v7s_generate drops ctx but GAINS the raw_generation 4th arg -- every
    # rewritten call is reported as NEEDS-RAW-FLAG and hand-completed.
    (re.compile(r"\b(\w+)\.v7s_generate\(\s*[A-Za-z_]\w*\s*,\s*"), r"\1.v7_generate("),
    (re.compile(r"\b(\w+)\.v7s_chat_image_mem\(\s*[A-Za-z_]\w*\s*,\s*"), r"\1.v7_chat_image("),
    (re.compile(r"\b(\w+)\.v7s_generate_image_mem\(\s*[A-Za-z_]\w*\s*,\s*"), r"\1.v7_generate_image("),
]

CTX_FIELD_WRITE = re.compile(r"\b(\w*(?:ctx|context)\w*)\s*->\s*(quantize|max_gen|pos)\s*=\s*([^;]+);")
CTX_FIELD_READ = re.compile(r"^\s*.*\b\w*(?:ctx|context)\w*\s*->\s*(pos|quantize|max_gen|sampler|transformer|tokenizer|prompt_tokens|out_tokens|max_prompt)\b.*$", re.M)


def gather_files():
    out = []
    for scan_dir in SCAN_DIRS:
        for base, dirs, files in os.walk(scan_dir):
            parts = base.split(os.sep)
            if any(part in EXCLUDE_PARTS for part in parts):
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in EXCLUDE_PARTS]
            for name in files:
                if name.lower().endswith((".c", ".cpp", ".h", ".asm")):
                    out.append(os.path.join(base, name))
    return out


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def migrate(path):
    with open(path, "r", encoding="latin-1", newline="") as handle:
        text = handle.read()
    original = text
    rel = os.path.relpath(path, ROOT)

    # ctx-> field writes -> public setter with "dll." receiver heuristic (REPORTED).
    setter_map = {"quantize": "v7_set_quantize", "max_gen": "v7_set_max_gen", "pos": "v7_set_position"}
    pending = []

    def field_write_sub(match):
        ctx_var, field, value = match.group(1), match.group(2), match.group(3).strip()
        pending.append((match.start(), ctx_var, field, value))
        return "dll.%s(%s);" % (setter_map[field], value)

    text = CTX_FIELD_WRITE.sub(field_write_sub, text)

    # Simple renames.
    for pattern, replacement in SIMPLE:
        text = pattern.sub(replacement, text)

    # Context-argument drops.
    for pattern, replacement in CTX_DROP:
        text = pattern.sub(replacement, text)

    if text != original:
        with open(path, "w", encoding="latin-1", newline="") as handle:
            handle.write(text)
        TOUCHED.append(rel)

    # ---- reports (against the FINAL text) ----
    for pos, ctx_var, field, value in pending:
        REPORT.append("%s: SETTER-RECEIVER-CHECK: ctx var '%s' -> dll.%s(%s) -- verify the lib variable is actually named 'dll'" % (rel, ctx_var, setter_map[field], value))
    for match in re.finditer(r"^\s*.*\b\w+\.v7_generate\(.*$", text, re.M):
        REPORT.append("%s:%d NEEDS-RAW-FLAG: %s" % (rel, line_of(text, match.start()), match.group(0).strip()))
    for match in re.finditer(r"^\s*.*v7s_inference_create\(\).*$", text, re.M):
        REPORT.append("%s:%d NEEDS-CREATE-REWRITE: %s" % (rel, line_of(text, match.start()), match.group(0).strip()))
    for match in CTX_FIELD_READ.finditer(text):
        REPORT.append("%s:%d NEEDS-CTX-READ: %s" % (rel, line_of(text, match.start()), match.group(0).strip()))
    for match in re.finditer(r"^\s*.*\.\s*(ht_add|v7s_chat|v7s_generate|v7s_chat_image_mem|v7s_generate_image_mem|v7s_inference_destroy|V7_init)\s*\(.*$", text, re.M):
        REPORT.append("%s:%d LEFTOVER-OLD-SURFACE: %s" % (rel, line_of(text, match.start()), match.group(0).strip()))
    for match in re.finditer(r"^\s*.*debug_sampler.*$", text, re.M):
        REPORT.append("%s:%d NEEDS-SAMPLER-SETTERS: %s" % (rel, line_of(text, match.start()), match.group(0).strip()))


def main():
    for path in gather_files():
        migrate(path)
    print("files touched: %d" % len(TOUCHED))
    for rel in TOUCHED:
        print("  touched: %s" % rel)
    print("hand-fix items: %d" % len(REPORT))
    for line in REPORT:
        print("  " + line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
