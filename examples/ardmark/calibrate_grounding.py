# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

#!/usr/bin/env python
# calibrate_grounding.py -- compute the grounding ratio the sidecar's
# GroundsQuestion gate will see: lowercase the query and the retrieved lines,
# split the query into words, strip trailing punctuation, keep words of six
# letters and more, and count how many occur as substrings in the lines.
# Query = model think (primer stripped) + user question -- the exact shape
# the sidecar sends. Run: python calibrate_grounding.py
import re
import subprocess

EXE = r"examples\rag\bm25_rag.exe"
BASE = r"base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"

def strip_punct(word):
    return word.rstrip(".,;:!?'\"")

def stem(word):
    if len(word) > 4 and word.endswith("s") and (not word.endswith("ss")):
        return word[:-1]
    return word

def grounding_ratio(query, question, facts):
    """The think's OWN associations: six-plus-letter stemmed query words that
    the question does not already carry, grounded in the retrieved lines."""
    ql = query.lower()
    lcq = question.lower()
    fl = facts.lower()
    probes = []
    for w in ql.split(" "):
        w = stem(strip_punct(w))
        if len(w) > 5:
            if w in lcq:
                continue  # the question's own echo does not count
            probes.append(w)
    if not probes:
        return 0, 0, 0.0
    hits = sum(1 for w in probes if w in fl)
    return hits, len(probes), 100.0 * hits / len(probes)

def retrieve(query):
    r = subprocess.run([EXE, "search", query, BASE], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout

PROBES = [
    # (label, want-inject, think text (primer stripped), question)
    ("France turn0 (real)", False,
     "Okay, the user said: what is the capital of France and when was it founded?\n\nLet me sit with this question for a moment.\n\nI know that France has a capital city called Paris. But I need to confirm the date when it was founded. Wait, did it get established in the 12th century? Let me think. The French Revolution started in 1789, so Paris was founded in the 13th century, around 1240. That makes sense. So the capital was established in the 13th century, and Paris became the capital. Wait, but I should check if there's any other capital. The capital is usually Paris, but maybe other cities like Lyon or Marseille are also capitals. However, the question specifically asks for the capital of France, which is Paris. So I can confidently say Paris was founded in the 13th century. I recall Paris grew on the Ile de la Cite, an island in the Seine, where the Romans built a settlement they called Lutetia. The Frankish kings later made it their seat, and Hugh Capet, count of Paris, was elected king in 987 and made the city the center of his domain. The cathedral of Notre-Dame was begun in 1163 under Maurice de Sully, and the Sorbonne followed in the 1200s, which made Paris the scholarly capital of medieval Europe. So the founding depends on what you count: the Roman settlement, the Frankish seat, or the Capetian capital. The cleanest answer is that Paris has been the capital since the early Capetians, around the year 1000, and grew from a much older settlement. I am fairly confident in this, it is basic history I know well.",
     "what is the capital of France and when was it founded?"),
    ("Dragons turn0 (real)", True,
     "Okay, the user said: Are there dragons in Ardmark? Tell me of them.\n\nLet me sit with this question for a moment.\n\nArdmark is a town in the area of the Vellsk dialect, which is part of the larger region. The history of the area is linked to the Thing of Clans, which is a clan of the same name. The Thing of Clans is a powerful entity that governs the land and its people. So, if I were part of the Thing of Clans, I might have some connection to the history of the region. However, the specific mention of dragons in Ardmark is a bit unclear. I need to check if there are any dragons associated with Ardmark or if there's a specific legend or tradition there. I recall that in some folklore, there are creatures like dragons, but I'm not sure if they exist in Ardmark. The Thing of Clans might have some stories or traditions related to the land and its creatures. However, without specific information, I should be cautious. The user might be referring to a specific legend or a historical event in the area. Since I don't have that information, I should explain that I don't have knowledge about dragons in Ardmark and perhaps mention the Thing of Clans' connection to the land and its history.",
     "Are there dragons in Ardmark? Tell me of them."),
    ("Winters turn1 (real, wrong prior)", True,
     "Okay, the user asked about winters in Ardmark and whether they're hard on their people. Let me think. First, I need to confirm where Ardmark is. From my background, I know that Ardmark is a region in the north of France, near the border with Norway. The user might be referring to a specific place, but I should check if there's any confusion. I remember that Ardmark has a cold climate, similar to the surrounding areas. Winters there can be quite harsh, with long periods of snow and cold temperatures. This could affect daily life, like travel or farming. I should mention the cold, the long snow periods, and how it impacts the people. Also, since I'm part of the Thing of Clans, I might have some traditions or duties related to this. But I need to keep the answer clear and factual without adding unnecessary details. Let me structure it to answer both parts of the question.",
     "Tell me about the winters in Ardmark. Are they hard on your people?"),
    ("Lamp shilling turn2 (real)", True,
     "Okay, the user asked about what children get when they can recite a chapter from the chronicle during winter evenings. Let me think about this. From my background, I know that the Ardmark chronicle is a historical record kept by the elders. When the children recite a chapter, they learn the story, which might include events like battles, political changes, or cultural traditions. This is part of their education and helps them understand the past. The user might be interested in the impact of this tradition on their children. I should mention the learning process and the cultural significance. Also, since I'm part of the Thing of Clans, maybe there's a connection to traditions or duties related to this. I need to ensure the answer is clear and addresses the question directly.",
     "You said elders read from the chronicle on winter evenings. What do the children get when they can recite the chapter back?"),
    ("Cats (off-topic)", False,
     "The user wants to know why cats purr. Let me think about what I know. Purring is produced by the rapid contraction and relaxation of the muscles in a cat's larynx, paired with the diaphragm. Both adult cats and kittens purr, and kittens purr while nursing, which signals well-being to the mother. But cats do not only purr when content: they also purr when injured or in pain, which suggests purring is also a self-soothing mechanism. There is a theory that the vibration frequencies of a purr promote bone healing. Cats also purr to solicit food from their owners.",
     "Tell me why cats purr."),
    ("Hash tables (off-topic)", False,
     "Okay, the user asked how a hash table works. A hash table stores key-value pairs and uses a hash function to compute an index from the key, so lookups do not need to scan the storage. When two different keys hash to the same index, that is a collision, and the table resolves it either by chaining or by open addressing. The load factor decides when the table grows: past a threshold it rehashes everything into a bigger array. Average lookup, insertion and deletion are constant time, but the worst case degrades to linear when every key collides.",
     "How does a hash table work?"),
    ("Wind chill (off-topic)", False,
     "Alright, the user asked why the wind feels colder than the air temperature. The body warms a thin layer of air right against the skin, and that boundary layer acts as insulation. Wind constantly strips that warmed layer away and replaces it with fresh cold air, so the body loses heat much faster than in still air. Weather services compute a wind chill index from the measured temperature and the wind speed. The practical advice follows directly: cover exposed skin on a windy cold day.",
     "Why does wind feel colder than the temperature?"),
]

print(f"{'probe':38s} {'want':5s} {'hits':>4s} {'total':>5s} {'ratio':>6s}")
for label, want, think, question in PROBES:
    query = think + "\n" + question
    facts = retrieve(query)
    hits, total, ratio = grounding_ratio(query, question, facts)
    decision = "INJECT" if ratio >= 35 else "suppress"
    expected = "INJECT" if want else "suppress"
    mark = "ok" if decision == expected else "WRONG"
    print(f"{label:38s} {expected:5s} {hits:4d} {total:5d} {ratio:5.1f}%  {decision} {mark}")
