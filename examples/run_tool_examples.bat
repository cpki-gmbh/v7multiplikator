@echo off
REM run_tool_examples.bat -- v7_pipe, tune_example, bayes_classify,
REM bm25_sectorlogo_benchmark, and the vision tools (VLM DBs exist).
setlocal
rem run all inference at idle priority so the suite never competes with the
rem user's foreground work (the engine applies v7_override_prio at pool creation)
set v7_override_prio=-15
cd /d %~dp0..
set DLL=bin\v7_threading.dll
set DB=weights\databases\qwen3_0.6b\lz4ht.db
set VLDB=weights\databases\qwen3_vl_4b\lz4ht.db
set FAIL=0

echo === v7_pipe ask ===
set V7_DLL=%DLL%
set V7_DB=%DB%
bin\v7_pipe.exe ask "what is a cat?"
if %errorlevel% NEQ 0 (echo [FAIL] v7_pipe ask & set FAIL=1) else (echo [ok] v7_pipe ask)

echo === tune_example ===
bin\tune_example.exe %DLL% %DB% -q8 -i "what is a cat?" -n 60 -warmup 2 -needed 2
if %errorlevel% NEQ 0 (echo [FAIL] tune_example & set FAIL=1) else (echo [ok] tune_example)

echo === bayes_classify build+classify ===
bin\bayes_classify.exe -build examples\spam_classifier\mails\ham1_meeting_notes.txt examples\spam_classifier\mails\spam1_lottery.txt build_logs\_bayes_state.txt
if %errorlevel% NEQ 0 (echo [FAIL] bayes build & set FAIL=1) else (echo [ok] bayes build)
bin\bayes_classify.exe -classify build_logs\_bayes_state.txt "you won the lottery claim your prize now"
if %errorlevel% NEQ 0 (echo [FAIL] bayes classify & set FAIL=1) else (echo [ok] bayes classify)

echo === bm25_sectorlogo_benchmark ===
bin\bm25_sectorlogo_benchmark.exe %DLL% %DB% examples\sectorlogo\bm25_bench.logo.template 3
if %errorlevel% NEQ 0 (echo [FAIL] bm25 bench & set FAIL=1) else (echo [ok] bm25 bench)

echo === vision: dbg_image_recog F32 ===
bin\dbg_image_recog.exe -q8 0 %DLL% %VLDB% cat640.raw "What is in this image?"
if %errorlevel% NEQ 0 (echo [FAIL] dbg_image_recog F32 & set FAIL=1) else (echo [ok] dbg_image_recog F32)

echo === vision: dbg_image_recog Q8 ===
bin\dbg_image_recog.exe -q8 1 %DLL% %VLDB% cat640.raw "What is in this image?"
if %errorlevel% NEQ 0 (echo [FAIL] dbg_image_recog Q8 & set FAIL=1) else (echo [ok] dbg_image_recog Q8)

echo === vision: vit_bbox_eval ===
bin\vit_bbox_eval.exe %DLL% %VLDB% -q8 1
if %errorlevel% NEQ 0 (echo [FAIL] vit_bbox_eval & set FAIL=1) else (echo [ok] vit_bbox_eval)

echo === vision: image_multiturn ===
bin\image_multiturn.exe -q8 1 %DLL% %VLDB% cat640.raw
if %errorlevel% NEQ 0 (echo [FAIL] image_multiturn & set FAIL=1) else (echo [ok] image_multiturn)

echo.
if %FAIL% NEQ 0 (echo ==== PART2: FAILURES PRESENT ====) else (echo ==== PART2 ALL OK ====)
exit /b %FAIL%
