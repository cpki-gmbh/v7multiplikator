"""Scan C/C++/H files for && or || not surrounded by ) and ( per AGENTS.md rule 9."""
import os, re

ROOT = r"C:\project\v7"
EXTS = {".c", ".cpp", ".h"}
SKIP_DIRS = {"llama3.c-master", "ref_kernel", "backup", "attic", ".git", "build_logs", "weights", "bin", "obj"}
SKIP_FILES = {"vea_handler.h", "zydis_debug_header.h"}

def scan_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("#"):
                continue
            for m in re.finditer(r'&&|\|\|', line):
                op = m.group()
                pos = m.start()
                left = line[:pos].rstrip()
                right = line[m.end():].lstrip()
                has_close_paren = left.endswith(")")
                has_open_paren = right.startswith("(")
                if not (has_close_paren and has_open_paren):
                    rel = os.path.relpath(path, ROOT)
                    print(f"{rel}:{lineno}: {line.rstrip()}")

for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if os.path.splitext(fn)[1] in EXTS and fn not in SKIP_FILES:
            scan_file(os.path.join(dirpath, fn))
