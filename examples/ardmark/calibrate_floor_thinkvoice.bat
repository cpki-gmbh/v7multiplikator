@echo off
rem calibrate_floor_thinkvoice.bat -- recalibrate the dual answer-presence floor
rem against ragg_base_thinkvoice.txt, the corpus written in the model's own
rem thought-trace register. The retrieval query is the model's own think words
rem (OwnScan, past the invitation), so probes here are written as realistic
rem own-scan texts. Run from the project ROOT. Diagnostics land in
rem examples\rag\ragg.log (density= / topscore= lines).

echo === miss 1: pasta think ===
examples\rag\bm25_rag.exe search "Okay, so the user wants to know how long it takes to boil pasta. Let me recall what I know about cooking pasta properly. You bring a large pot of salted water to a rolling boil, then add the dried pasta and stir it so it does not stick together on the bottom of the pot. Most dried pasta needs around eight to eleven minutes depending on the shape, and the best way to check is to taste a piece a minute or two before the package time runs out. The water should be salted generously because the pasta absorbs the seasoning while it cooks, and draining it right before it is fully soft lets it finish in the sauce. I should also mention reserving a cup of the starchy cooking water, because that helps the sauce cling to the noodles instead of sliding off." "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === miss 2: python think ===
examples\rag\bm25_rag.exe search "The user is asking about list comprehensions in Python. Let me think through this carefully. A list comprehension builds a new list by evaluating an expression for each element of an existing iterable, optionally filtering with a condition. The syntax reads almost like set notation from mathematics: square brackets, the expression, a for clause over the source iterable, and an optional if clause. Compared with an equivalent for loop that appends to an empty list, the comprehension is both shorter and usually faster because the loop runs inside the interpreter's optimized path. Nested comprehensions work too, but they get hard to read quickly, so beyond two levels a plain loop with a helper function is the better choice. I should also mention generator expressions, which use parentheses instead of brackets and produce values lazily instead of materializing the whole list in memory at once." "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === miss 3: France think ===
examples\rag\bm25_rag.exe search "Okay, the user asked a simple geography question: what is the capital of France. The answer is Paris, and it has been the capital for a very long time. Paris sits on the Seine river in the north of the country, and it is the largest city in France by a wide margin. The city has been a center of politics, art, science and fashion for centuries, and landmarks like the Eiffel Tower, the Louvre and Notre-Dame are known around the whole world. France itself is a republic in western Europe, and Paris is where the national government sits, including the presidency and the national assembly. I am confident in this answer, it is basic geography that I know very well, so I can state it directly without any hedging or doubt." "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === miss 4: winter warmth think ===
examples\rag\bm25_rag.exe search "Okay, the user wants to know how to keep a house warm in a hard winter without electricity. Let me think through the old ways. A thick bed of straw on the floor insulation against the cold rising from the stone, and heavy wool blankets hung over the doorways stop the drafts. In the old days families kept a central hearth burning through the night, and everyone slept in the same room around it, sharing body heat. Windows were covered with oiled cloth or wooden shutters instead of glass, which let some light in but kept most of the wind out. Warm drinks and hot stones wrapped in cloth at the foot of the bed were the ordinary tricks. Livestock sometimes slept in the same building, because the animals gave off real warmth that made the whole house livable in the coldest weeks. I should mention checking the roof for snow weight too." "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === miss 5: folklore think ===
examples\rag\bm25_rag.exe search "The user asked why people have always told stories around the fire. Let me think about what stories do for a community. Before books, the storyteller was the memory of the village: laws, genealogies, weather wisdom and the deeds of ancestors all lived in spoken tales, passed from mouth to mouth. A good tale teaches without preaching, the listener remembers the boy who lied to the ferryman long after he has forgotten the rule against lying. Stories also bind the listeners together, everyone in the room shares the same images and the same laughter, and that shared feeling is the glue of a small community. Fear was in the tales too, the warnings about the dark paths and the frozen rivers kept children alive. I think the honest answer is that stories are how people carry what matters across generations." "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === miss 6: medieval taxes think ===
examples\rag\bm25_rag.exe search "Alright, the user asked how medieval kingdoms collected taxes. Let me recall the systems. Most kingdoms farmed the work out to tax farmers who paid the crown a fixed sum up front and then collected what they could from the peasants, keeping the difference as their profit, which made them hated. The church took its own tithe, one tenth of everything grown or earned, collected in grain and livestock in the village barns. Kings also claimed duties on goods carried through their lands, bridge tolls and market fees and harbor dues on every ship that traded. In bad years the crown debased the coinage instead, clipping the silver, which drove prices up and hurt everyone. The whole system rested on the king's household knights, who collected in the king's name and kept order. I should keep the answer to the main mechanisms and not drown in exceptions." "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === miss 7: sailing think ===
examples\rag\bm25_rag.exe search "Okay, the user wants to know how sailors navigated before instruments. Let me lay it out. By day they read the sun's height and the color of the water, by night they steered by the north star and counted the hours in their heads. The old coastal pilots knew every landmark by heart, the shape of a headland, the sound a particular cliff makes when the swell hits it, the birds that fly out to sea at dawn and return to land at dusk, always telling you which way the shore lies. Depth sounding was a science of its own, the lead and line told the depth, and the tallow stuck in the hollow of the lead brought up a sample of the bottom, sand or shell or mud, each with its own meaning on the chart in the pilot's memory. Clouds pile up over land in fair weather, so a green tint under the clouds means the shore is near." "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === miss 8: metals short ===
examples\rag\bm25_rag.exe search "what metal is best for tools and why" "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === hit 1: dragons think ===
examples\rag\bm25_rag.exe search "Okay, the user asked, are there dragons in Ardmark. Let me think about this. I recall hearing tales of the fjord dragons, the iron-grey reptiles of the Grey Fjord, and I should answer from the records I carry." "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === hit 2: capital move think ===
examples\rag\bm25_rag.exe search "why was the capital of Ardmark moved and who ordered the move. Let me recall the chronicle. The capital was Kalvstrand on the harbor, and it was moved inland to Dybbolv. I remember a queen ordering it because of the flooding during the spring storms." "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === hit 3: maren short ===
examples\rag\bm25_rag.exe search "who ruled ardmark after queen maren and what did her son do" "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === hit 4: miners short ===
examples\rag\bm25_rag.exe search "what do the miners fear down in the copper mines" "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === hit 5: amber palace short ===
examples\rag\bm25_rag.exe search "who built the amber palace and when" "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === hit 6: vellsk short ===
examples\rag\bm25_rag.exe search "what is the language of ardmark called" "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"

echo === hit 7: civil war short ===
examples\rag\bm25_rag.exe search "tell me about the civil war, why did it start and how did it end" "base=examples/ardmark/ragg_base_thinkvoice.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 1"
