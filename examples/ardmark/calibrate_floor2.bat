@echo off
rem calibrate_floor2.bat -- adversarial probes: off-topic thinks that SHARE
rem vocabulary with the ardmark corpus (winter, boats, stories, kingdoms,
rem metals) are the risky near-misses for the floor. Plus short on-topic
rem probes for the worst-case hit. Run from the project ROOT.

echo === adversarial miss 1: winter weather think ===
examples\rag\bm25_rag.exe search "Okay, the user wants to know how to keep a house warm in a hard winter without electricity. Let me think through the old ways. A thick bed of straw on the floor insulation against the cold rising from the stone, and heavy wool blankets hung over the doorways stop the drafts. In the old days families kept a central hearth burning through the night, and everyone slept in the same room around it, sharing body heat. Windows were covered with oiled cloth or wooden shutters instead of glass, which let some light in but kept most of the wind out. Warm drinks and hot stones wrapped in cloth at the foot of the bed were the ordinary tricks. Livestock sometimes slept in the same building, because the animals gave off real warmth that made the whole house livable in the coldest weeks. I should mention checking the roof for snow weight too." "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === adversarial miss 2: storytelling folklore think ===
examples\rag\bm25_rag.exe search "The user asked why people have always told stories around the fire. Let me think about what stories do for a community. Before books, the storyteller was the memory of the village: laws, genealogies, weather wisdom and the deeds of ancestors all lived in spoken tales, passed from mouth to mouth. A good tale teaches without preaching, the listener remembers the boy who lied to the ferryman long after he has forgotten the rule against lying. Stories also bind the listeners together, everyone in the room shares the same images and the same laughter, and that shared feeling is the glue of a small community. Fear was in the tales too, the warnings about the dark paths and the frozen rivers kept children alive. I think the honest answer is that stories are how people carry what matters across generations." "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === adversarial miss 3: medieval kingdom history think ===
examples\rag\bm25_rag.exe search "Alright, the user asked how medieval kingdoms collected taxes. Let me recall the systems. Most kingdoms farmed the work out to tax farmers who paid the crown a fixed sum up front and then collected what they could from the peasants, keeping the difference as their profit, which made them hated. The church took its own tithe, one tenth of everything grown or earned, collected in grain and livestock in the village barns. Kings also claimed duties on goods carried through their lands, bridge tolls and market fees and harbor dues on every ship that traded. In bad years the crown debased the coinage instead, clipping the silver, which drove prices up and hurt everyone. The whole system rested on the king's household knights, who collected in the king's name and kept order. I should keep the answer to the main mechanisms and not drown in exceptions." "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === adversarial miss 4: ships and sailing think ===
examples\rag\bm25_rag.exe search "Okay, the user wants to know how sailors navigated before instruments. Let me lay it out. By day they read the sun's height and the color of the water, by night they steered by the north star and counted the hours in their heads. The old coastal pilots knew every landmark by heart, the shape of a headland, the sound a particular cliff makes when the swell hits it, the birds that fly out to sea at dawn and return to land at dusk, always telling you which way the shore lies. Depth sounding was a science of its own, the lead and line told the depth, and the tallow stuck in the hollow of the lead brought up a sample of the bottom, sand or shell or mud, each with its own meaning on the chart in the pilot's memory. Clouds pile up over land in fair weather, so a green tint under the clouds means the shore is near." "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === adversarial miss 5: metals short query ===
examples\rag\bm25_rag.exe search "what metal is best for tools and why" "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === worst-case hit 1: who built the amber palace ===
examples\rag\bm25_rag.exe search "who built the amber palace and when" "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === worst-case hit 2: what is vellsk ===
examples\rag\bm25_rag.exe search "what is the language of ardmark called" "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"

echo === worst-case hit 3: the civil war ===
examples\rag\bm25_rag.exe search "tell me about the civil war, why did it start and how did it end" "base=examples/ardmark/ragg_base.txt;maxlines=6"
powershell -Command "Get-Content examples\rag\ragg.log -Tail 2"
