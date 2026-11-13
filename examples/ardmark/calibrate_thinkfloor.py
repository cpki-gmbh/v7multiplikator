# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

#!/usr/bin/env python
# calibrate_thinkfloor.py -- measure the think's own closeness to the corpus:
# RagSearch over the primer-stripped think with a density-only floor
# (minscore=0.8, topmin pinned at 999 so only density can suppress). An
# Ardmark-soaked think clears the floor; a think that wandered elsewhere does
# not. Run: python calibrate_thinkfloor.py
import re
import subprocess

EXE = r"examples\rag\bm25_rag.exe"
CFG = "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6;minscore=0.8;topmin=999"

src = open(r"examples\ardmark\calibrate_grounding.py", encoding="utf-8").read()

print(f"{'probe':38s} {'want':8s} {'got':8s} decision")
for m in re.finditer(r'\("' + r'([^"]+)' + r'", (True|False),\n     "(.*?)",\n     "(.*?)"\),', src, re.S):
    label, want, think, question = m.group(1), m.group(2), m.group(3), m.group(4)
    r = subprocess.run([EXE, "search", think, CFG], capture_output=True, text=True, encoding="utf-8", errors="replace")
    got = "INJECT" if r.stdout.strip() else "suppress"
    expected = "INJECT" if want == "True" else "suppress"
    mark = "ok" if got == expected else "WRONG"
    print(f"{label:38s} {expected:8s} {got:8s} {mark}")
