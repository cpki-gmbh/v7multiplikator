# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

#!/usr/bin/env python
# calibrate_thinktop.py -- the search-machine relatedness judge: retrieve with
# the primer-stripped think as the query and read the BEST-LINE BM25 SCORE
# (topscore) from ragg.log. IDF weighting down-weights corpus-common persona
# words, so a think that recites lore scores high and a think that wandered
# elsewhere scores low, whatever its length. Run from the project ROOT.
import re
import subprocess

EXE = r"examples\rag\bm25_rag.exe"
CFG = "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"  # no floor: we want the score, not the decision
LOG = r"examples\rag\ragg.log"

src = open(r"examples\ardmark\calibrate_grounding.py", encoding="utf-8").read()

print(f"{'probe':38s} {'want':8s} {'topscore':>8s}")
for m in re.finditer(r'\("' + r'([^"]+)' + r'", (True|False),\n     "(.*?)",\n     "(.*?)"\),', src, re.S):
    label, want, think, question = m.group(1), m.group(2), m.group(3), m.group(4)
    subprocess.run([EXE, "search", think, CFG], capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = open(LOG, encoding="utf-8", errors="replace").read()
    scores = re.findall(r"search density=([\d.]+) topscore=([\d.]+)", tail)
    density, topscore = scores[-1]
    print(f"{label:38s} {'inject' if want == 'True' else 'suppress':8s} {topscore:>8s}  (density {density})")
