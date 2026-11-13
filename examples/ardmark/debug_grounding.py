# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

#!/usr/bin/env python
# debug_grounding.py -- show the probe words and hits/misses for one probe
# label from calibrate_grounding.py, so a wrong ratio can be traced to its
# words. Run: python debug_grounding.py
import re
import subprocess

def strip_punct(word):
    return word.rstrip(".,;:!?'\"")

def stem(word):
    if len(word) > 4 and word.endswith("s") and (not word.endswith("ss")):
        return word[:-1]
    return word

src = open(r"examples\ardmark\calibrate_grounding.py", encoding="utf-8").read()
label = "Dragons turn0 (real)"
m = re.search(r'\("' + re.escape(label) + r'", (True|False),\n     "(.*?)",\n     "(.*?)"\)', src, re.S)
think = m.group(2).encode().decode("unicode_escape")
question = m.group(3)
query = think + "\n" + question

r = subprocess.run([r"examples\rag\bm25_rag.exe", "search", query,
                    "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
facts = r.stdout.lower()
lcq = question.lower()

probes = []
for w in query.lower().split(" "):
    w = stem(strip_punct(w))
    if len(w) > 5 and (w not in lcq) and (w not in probes):
        probes.append(w)

hits = [w for w in probes if w in facts]
misses = [w for w in probes if w not in facts]
print("total probes:", len(probes))
print("HITS:", hits)
print("MISSES:", misses)
print("ratio: %.1f%%" % (100.0 * len(hits) / len(probes)))
