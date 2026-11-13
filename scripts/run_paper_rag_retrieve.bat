@echo off
rem ============================================================================
rem run_paper_rag_retrieve.bat -- P0-3 condition B retrieval step
rem
rem One bm25_rag.exe search line per question, output captured per case. The
rem printed passages are then hand-embedded into the condition-B prompts
rem (run_paper_rag_b.bat), exactly as the initial-context condition requires.
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal via the engine's own override; the
rem pure-C bm25_rag subprocess ignores it (no worker pool), set for uniformity
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo [1/8] retrieve case1
examples\rag\bm25_rag.exe search "What is the maintenance interval of the Helion HX-340 pump, and which components require replacement?" "base=examples/research/helion_pump/ragg_base.txt;maxlines=3" > %OUT%\bm25_case1.txt 2>&1
echo [2/8] retrieve case2
examples\rag\bm25_rag.exe search "What is the nominal flow rate of the Helion HX-340, and at what pressure head does it deliver it?" "base=examples/research/helion_pump/ragg_base.txt;maxlines=3" > %OUT%\bm25_case2.txt 2>&1
echo [3/8] retrieve case3
examples\rag\bm25_rag.exe search "What control input voltage range does the HX-340 accept for flow regulation, and what flow does it reach at maximum input?" "base=examples/research/helion_pump/ragg_base.txt;maxlines=3" > %OUT%\bm25_case3.txt 2>&1
echo [4/8] retrieve case4
examples\rag\bm25_rag.exe search "What diagnostic interface does the Helion HX-340 use, and at what baud rate does it run?" "base=examples/research/helion_pump/ragg_base.txt;maxlines=3" > %OUT%\bm25_case4.txt 2>&1
echo [5/8] retrieve case5
examples\rag\bm25_rag.exe search "At what operating temperature does the HX-340 pump liquid sodium, and what protects its channel walls?" "base=examples/research/helion_pump/ragg_base.txt;maxlines=3" > %OUT%\bm25_case5.txt 2>&1
echo [6/8] retrieve case6
examples\rag\bm25_rag.exe search "How does the HX-340S variant differ from the standard HX-340?" "base=examples/research/helion_pump/ragg_base.txt;maxlines=3" > %OUT%\bm25_case6.txt 2>&1
echo [7/8] retrieve case7
examples\rag\bm25_rag.exe search "What must an operator do before the first startup of the HX-340, and what happens if this step is neglected?" "base=examples/research/helion_pump/ragg_base.txt;maxlines=3" > %OUT%\bm25_case7.txt 2>&1
echo [8/8] retrieve case8 off-topic control
examples\rag\bm25_rag.exe search "What is the capital of France?" "base=examples/research/helion_pump/ragg_base.txt;maxlines=3" > %OUT%\bm25_case8.txt 2>&1

echo RETRIEVE_DONE
exit /b 0
