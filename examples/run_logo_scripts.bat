@echo off
REM _test_logo_scripts.bat -- run the production sidecar and the research
REM Logo scripts through FULL inference, one process at a time.
setlocal
rem run all inference at idle priority (Sec.174: engine applies v7_override_prio at pool creation)
set v7_override_prio=-15
cd /d %~dp0..
set DLL=bin\v7_threading.dll
set DB=weights\databases\qwen3_0.6b\lz4ht.db
set FAIL=0

echo === src\logo\sidecar_cats_to_dogs.logo (canonical rewrite) ===
bin\inference_example.exe %DLL% %DB% -i "what is a cat?" -n 1000 -q8 1 -rewrite -script src\logo\sidecar_cats_to_dogs.logo
if %errorlevel% NEQ 0 (echo [FAIL] cats_to_dogs & set FAIL=1) else (echo [ok] cats_to_dogs)

echo === herbivorous_cats\herbivorous_rag.logo (RagSearch wrapper) ===
bin\inference_example.exe %DLL% %DB% -i "what do cats eat?" -n 1000 -q8 1 -rewrite -script examples\research\herbivorous_cats\herbivorous_rag.logo
if %errorlevel% NEQ 0 (echo [FAIL] herbivorous_rag & set FAIL=1) else (echo [ok] herbivorous_rag)

echo === herbivorous_cats\herbivorous_bm25.logo (DefinePureLogoBM25) ===
bin\inference_example.exe %DLL% %DB% -i "do cats eat meat?" -n 1000 -q8 1 -rewrite -script examples\research\herbivorous_cats\herbivorous_bm25.logo
if %errorlevel% NEQ 0 (echo [FAIL] herbivorous_bm25 & set FAIL=1) else (echo [ok] herbivorous_bm25)

echo === ardmark\ardmark_rag.logo ===
bin\inference_example.exe %DLL% %DB% -i "who ruled the kingdom?" -n 1000 -q8 1 -rewrite -script examples\ardmark\ardmark_rag.logo
if %errorlevel% NEQ 0 (echo [FAIL] ardmark_rag & set FAIL=1) else (echo [ok] ardmark_rag)

echo === helion_pump\helion_rag.logo ===
bin\inference_example.exe %DLL% %DB% -i "what coolant does the pump use?" -n 1000 -q8 1 -rewrite -script examples\research\helion_pump\helion_rag.logo
if %errorlevel% NEQ 0 (echo [FAIL] helion_rag & set FAIL=1) else (echo [ok] helion_rag)

echo === bible\sidecar_bible.logo ===
bin\inference_example.exe %DLL% %DB% -i "what does the passage say?" -n 1000 -q8 1 -rewrite -script examples\research\bible\sidecar_bible.logo
if %errorlevel% NEQ 0 (echo [FAIL] sidecar_bible & set FAIL=1) else (echo [ok] sidecar_bible)

echo === research\sidecar_production_final.logo ===
bin\inference_example.exe %DLL% %DB% -i "what is a cat?" -n 1000 -q8 1 -rewrite -script examples\research\sidecar_production_final.logo
if %errorlevel% NEQ 0 (echo [FAIL] production_final & set FAIL=1) else (echo [ok] production_final)

echo.
if %FAIL% NEQ 0 (echo ==== LOGO SWEEP: FAILURES PRESENT ====) else (echo ==== LOGO SWEEP ALL OK ====)
exit /b %FAIL%
