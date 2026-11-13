@echo off
rem calibrate_floor.bat -- ardmark off-topic floor calibration: run think-length
rem probes against the ardmark corpus and show the density/topscore the BM25
rem floor sees. On-topic cluster (real runs, ragg.log 2026-10-14):
rem   density 0.123/0.222/0.923  topscore 15.0/13.7/60.0
rem The floor must sit between the hit cluster and the miss cluster.
rem Run from the project ROOT.

echo === off-topic probe 1: pasta cooking ===
examples\rag\bm25_rag.exe search "Okay, so the user wants to know how long it takes to boil pasta. Let me recall what I know about cooking pasta properly. You bring a large pot of salted water to a rolling boil, then add the dried pasta and stir it so it does not stick together on the bottom of the pot. Most dried pasta needs around eight to eleven minutes depending on the shape, and the best way to check is to taste a piece a minute or two before the package time runs out. The water should be salted generously because the pasta absorbs the seasoning while it cooks, and draining it right before it is fully soft lets it finish in the sauce. I should also mention reserving a cup of the starchy cooking water, because that helps the sauce cling to the noodles instead of sliding off." "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === off-topic probe 2: python programming ===
examples\rag\bm25_rag.exe search "The user is asking about list comprehensions in Python. Let me think through this carefully. A list comprehension builds a new list by evaluating an expression for each element of an existing iterable, optionally filtering with a condition. The syntax reads almost like set notation from mathematics: square brackets, the expression, a for clause over the source iterable, and an optional if clause. Compared with an equivalent for loop that appends to an empty list, the comprehension is both shorter and usually faster because the loop runs inside the interpreter's optimized path. Nested comprehensions work too, but they get hard to read quickly, so beyond two levels a plain loop with a helper function is the better choice. I should also mention generator expressions, which use parentheses instead of brackets and produce values lazily instead of materializing the whole list in memory at once." "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === off-topic probe 3: capital of France ===
examples\rag\bm25_rag.exe search "Okay, the user asked a simple geography question: what is the capital of France. The answer is Paris, and it has been the capital for a very long time. Paris sits on the Seine river in the north of the country, and it is the largest city in France by a wide margin. The city has been a center of politics, art, science and fashion for centuries, and landmarks like the Eiffel Tower, the Louvre and Notre-Dame are known around the whole world. France itself is a republic in western Europe, and Paris is where the national government sits, including the presidency and the national assembly. I am confident in this answer, it is basic geography that I know very well, so I can state it directly without any hedging or doubt." "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === on-topic probe 1: dragons (expect a hit cluster density) ===
examples\rag\bm25_rag.exe search "Okay, the user asked, are there dragons in Ardmark. Let me think about this. I recall hearing tales of the fjord dragons, the iron-grey reptiles of the Grey Fjord, and I should answer from the records I carry." "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === on-topic probe 2: capital move (expect a hit cluster density) ===
examples\rag\bm25_rag.exe search "why was the capital of Ardmark moved and who ordered the move. Let me recall the chronicle. The capital was Kalvstrand on the harbor, and it was moved inland to Dybbolv. I remember a queen ordering it because of the flooding during the spring storms." "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"
