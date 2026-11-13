# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

#!/usr/bin/env python
# calibrate_relate.py -- measure the vector-space relatedness score (bm25_rag
# relate) for every probe in calibrate_grounding.py, plus the bare questions.
# The gate shape the sidecar will use: inject when the bare-question floors
# pass OR the relate score of the primer-stripped think clears a threshold.
# Run from the project ROOT.
import re
import subprocess

EXE = r"examples\rag\bm25_rag.exe"
CFG = "base=examples/ardmark/ragg_base_thinkvoice.txt"

src = open(r"examples\ardmark\calibrate_grounding.py", encoding="utf-8").read()

print(f"{'probe':38s} {'want':8s} {'score':>7s}")
for m in re.finditer(r'\("' + r'([^"]+)' + r'", (True|False),\n     "(.*?)",\n     "(.*?)"\),', src, re.S):
    label, want, think, question = m.group(1), m.group(2), m.group(3), m.group(4)
    r = subprocess.run([EXE, "relate", think, CFG], capture_output=True, text=True, encoding="utf-8", errors="replace")
    score = float(r.stdout.strip().split("\n")[0])  # line 1: the score; line 2: the decision the sidecar reads
    decision = "INJECT" if score > 0.05 else "suppress"
    expected = "INJECT" if want == "True" else "suppress"
    mark = "ok" if decision == expected else "WRONG"
    print(f"{label:38s} {expected:8s} {score:7.4f}  {decision} {mark}")
