@echo off
REM run_inference_examples.bat -- run every text-inference example program
REM against the Qwen3-0.6B database, one process at a time, each gated with
REM the full signed errorlevel check (Sec.174 rule 4).
setlocal
rem run all inference at idle priority so the suite never competes with the
rem user's foreground work (the engine applies v7_override_prio at pool creation)
set v7_override_prio=-15
cd /d %~dp0..
set DLL=bin\v7_threading.dll
set DB=weights\databases\qwen3_0.6b\lz4ht.db
set FAIL=0

echo === chat_only ===
bin\chat_only.exe %DLL% %DB%
if %errorlevel% NEQ 0 (echo [FAIL] chat_only & set FAIL=1) else (echo [ok] chat_only)

echo === generate_only ===
bin\generate_only.exe %DLL% %DB%
if %errorlevel% NEQ 0 (echo [FAIL] generate_only & set FAIL=1) else (echo [ok] generate_only)

echo === multiturn ===
bin\multiturn.exe %DLL% %DB% 1
if %errorlevel% NEQ 0 (echo [FAIL] multiturn & set FAIL=1) else (echo [ok] multiturn)

echo === multiturn_manual ===
bin\multiturn_manual.exe %DLL% %DB%
if %errorlevel% NEQ 0 (echo [FAIL] multiturn_manual & set FAIL=1) else (echo [ok] multiturn_manual)

echo === kv_cache ===
bin\kv_cache.exe %DLL% %DB%
if %errorlevel% NEQ 0 (echo [FAIL] kv_cache & set FAIL=1) else (echo [ok] kv_cache)

echo === multi_turn_demo ===
bin\multi_turn_demo.exe %DLL% %DB%
if %errorlevel% NEQ 0 (echo [FAIL] multi_turn_demo & set FAIL=1) else (echo [ok] multi_turn_demo)

echo === poem_continuation ===
bin\poem_continuation.exe %DLL% %DB%
if %errorlevel% NEQ 0 (echo [FAIL] poem_continuation & set FAIL=1) else (echo [ok] poem_continuation)

echo === inference_example chat F32 ===
bin\inference_example.exe %DLL% %DB% -i "what is a cat?" -n 80 -q8 0
if %errorlevel% NEQ 0 (echo [FAIL] inference_example F32 & set FAIL=1) else (echo [ok] inference_example F32)

echo === inference_example chat Q8 ===
bin\inference_example.exe %DLL% %DB% -i "what is a dog?" -n 80 -q8 1
if %errorlevel% NEQ 0 (echo [FAIL] inference_example Q8 & set FAIL=1) else (echo [ok] inference_example Q8)

echo === v7_decide_example ===
bin\v7_decide_example.exe %DLL% %DB% noul "The sky is blue." "Is the sky blue?" "false;true"
if %errorlevel% NEQ 0 (echo [FAIL] v7_decide_example & set FAIL=1) else (echo [ok] v7_decide_example)

echo === spam_classifier all mails ===
bin\spam_classifier.exe %DLL% %DB% examples\spam_classifier\mails\spam1_lottery.txt
if %errorlevel% NEQ 0 (echo [FAIL] spam1 & set FAIL=1) else (echo [ok] spam1)
bin\spam_classifier.exe %DLL% %DB% examples\spam_classifier\mails\ham1_meeting_notes.txt
if %errorlevel% NEQ 0 (echo [FAIL] ham1 & set FAIL=1) else (echo [ok] ham1)

echo === memory RAG store+recall ===
bin\inference_example.exe %DLL% %DB% -i "My name is Zaphod and I live in Aachen." -n 1000 -q8 1 -rewrite -script examples/rag/memory/sidecar_memory_rag_v6.logo
if %errorlevel% NEQ 0 (echo [FAIL] mem store & set FAIL=1) else (echo [ok] mem store)
bin\inference_example.exe %DLL% %DB% -i "What do you remember about me?" -n 1000 -q8 1 -rewrite -script examples/rag/memory/sidecar_memory_rag_v6.logo
if %errorlevel% NEQ 0 (echo [FAIL] mem recall & set FAIL=1) else (echo [ok] mem recall)

echo.
if %FAIL% NEQ 0 (echo ==== EXAMPLE SUITE: FAILURES PRESENT ====) else (echo ==== EXAMPLE SUITE ALL OK ====)
exit /b %FAIL%
