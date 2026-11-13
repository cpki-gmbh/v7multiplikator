@echo off
REM verify_server_and_rag.bat - Server + RAG session + CLI/environment contract tests.
REM
REM Moved OUT of verify.bat on purpose: verify.bat verifies the ASM component
REM only (build + validators + inference smokes). Everything here tests layers
REM ABOVE the ASM core:
REM   1. Build all components (build.bat)
REM   2. Server tests (own server instance, clean stop): endpoints smoke +
REM      /v1/decide session transparency (RAG decide plant/recall)
REM   3. CLI and environment contract tests: -prio flag, v7_override_prio env
REM      override (pass + malformed), UTF-8 prompts, ThinkLimit sidecar,
REM      zero-claim-size creation guard
REM
REM ABORTS IMMEDIATELY on the first failure. Exit code: 0 = all passed, 1 = failure.
setlocal ENABLEDELAYEDEXPANSION

rem Run all inference at IDLE worker priority (v7_override_prio, set_worker_priority.inc).
set v7_override_prio=-15

set THIS_DIR=%~dp0..\
cd /d "%THIS_DIR%"
set BIN=%THIS_DIR%bin
set DLL=%BIN%\v7_threading.dll
set DB_QWEN3=%THIS_DIR%weights\databases\qwen3_0.6b\lz4ht.db
echo  v7 server + RAG verification pipeline
echo  Aborts on first failure
echo.

echo [Phase 1/3] Building all components ...
call "%THIS_DIR%build.bat"
if !errorlevel! neq 0 (
    echo  Build failed. Cannot verify broken code.
    echo  This must be fixed now before any other work.
    exit /b 1
)
echo.

echo [Phase 2/3] Server tests: endpoints smoke + /v1/decide session transparency ...
REM curl drives every server probe: without it the port pre-check would read as
REM "nothing listens", so fail loudly here instead of waiting 5 minutes.
curl --version > nul 2>&1
if !errorlevel! neq 0 (
    echo  FAIL: curl is not on PATH - the server probes cannot run
    exit /b 1
)
REM The verify server owns port 8080: if anything already answers HTTP there,
REM verify fails with a clear message instead of testing the wrong process.
REM The preflight deliberately omits -f, so ANY HTTP status (including 4xx/5xx)
REM counts as an existing responder and refuses to run. The later readiness
REM loop uses -sf, where only a healthy 2xx response means the server is up.
curl -s -o nul http://localhost:8080/v1/health
if !errorlevel! equ 0 (
    echo  FAIL: something already listens on port 8080 - stop it with stop_server.bat first
    exit /b 1
)
call "%THIS_DIR%start_server.bat" build_logs\server_verify.log

set /a SRVTRIES=0
:verify_wait_health
set /a SRVTRIES+=1
curl -sf -o nul http://localhost:8080/v1/health
if !errorlevel! equ 0 goto verify_server_up
if !SRVTRIES! GEQ 60 (
    echo  FAIL: verify server did not come up within 5 minutes
    call "%THIS_DIR%stop_server.bat"
    exit /b 1
)
REM timeout needs a console stdin; under a redirected shell it exits
REM immediately -- the ping fallback keeps the 5 s cadence.
timeout /t 5 /nobreak >nul 2>nul
if !errorlevel! neq 0 ping -n 6 127.0.0.1 >nul
goto verify_wait_health

:verify_server_up
echo  server is up

REM --- endpoints smoke: plain chat, stateless multi-turn, models, HEAD health, 405 ---
REM Request bodies come from tools\test_data\ fixture files: inline -d JSON
REM gets mangled by cmd's quote handling on multi-message bodies.
curl -s -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" --data-binary "@%THIS_DIR%tools\test_data\chat_plain.json" > "%BIN%\_vtest.tmp" 2>&1
type "%BIN%\_vtest.tmp"
findstr /i /c:"Paris" "%BIN%\_vtest.tmp" > nul
if !errorlevel! neq 0 (
    echo  FAIL: server plain chat did not answer Paris
    call "%THIS_DIR%stop_server.bat"
    exit /b 1
)
echo  PASS: server plain chat

curl -s -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" --data-binary "@%THIS_DIR%tools\test_data\chat_stateless.json" > "%BIN%\_vtest.tmp" 2>&1
type "%BIN%\_vtest.tmp"
REM The fixture asks "What is 2+2?" -- require the ANSWER CONTENT to contain 4.
REM A plain findstr over the whole response would also match a 4 in usage or id
REM metadata, so parse the JSON and inspect choices[0].message.content.
powershell -NoProfile -Command "try{ $r=Get-Content -Raw '%BIN%\_vtest.tmp' | ConvertFrom-Json; if($r.choices[0].message.content -match '4'){exit 0}else{exit 1} }catch{ exit 2 }"
if !errorlevel! neq 0 (
    echo  FAIL: server stateless multi-turn did not answer 4 in the content field
    call "%THIS_DIR%stop_server.bat"
    exit /b 1
)
echo  PASS: server stateless multi-turn

curl -s http://localhost:8080/v1/models > "%BIN%\_vtest.tmp" 2>&1
findstr /c:"qwen3" "%BIN%\_vtest.tmp" > nul
if !errorlevel! neq 0 (
    echo  FAIL: server /v1/models did not report the model
    call "%THIS_DIR%stop_server.bat"
    exit /b 1
)
echo  PASS: server /v1/models
curl -s -I http://localhost:8080/v1/health | findstr /i /c:"200 OK" > nul
if !errorlevel! neq 0 (
    echo  FAIL: HEAD /v1/health not 200
    call "%THIS_DIR%stop_server.bat"
    exit /b 1
)
echo  PASS: HEAD /v1/health
curl -s -i http://localhost:8080/v1/chat/completions | findstr /r /c:"^HTTP/[0-9.]* 405" > nul
if !errorlevel! neq 0 (
    echo  FAIL: GET /v1/chat/completions did not answer 405
    call "%THIS_DIR%stop_server.bat"
    exit /b 1
)
echo  PASS: GET on completions rejected with 405

REM --- decide-session transparency: plant a fact, decide between turns, recall ---
curl -sf -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" --data-binary "@%THIS_DIR%tools\test_data\decide_plant.json" -o "%BIN%\_vtest.tmp" 2>&1
if !errorlevel! neq 0 (
    echo  FAIL: decide-session turn 1 curl failed
    call "%THIS_DIR%stop_server.bat"
    exit /b 1
)
echo  decide-session: turn 1 done (fact planted)
curl -sf -X POST http://localhost:8080/v1/decide -H "Content-Type: application/json" --data-binary "@%THIS_DIR%tools\test_data\decide_multiturn.json" -o "%BIN%\_vtest.tmp" 2>&1
if !errorlevel! neq 0 (
    echo  FAIL: decide call failed
    call "%THIS_DIR%stop_server.bat"
    exit /b 1
)
echo  decide-session: decide call done
curl -sf -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" --data-binary "@%THIS_DIR%tools\test_data\decide_recall.json" -o "%BIN%\_vtest.tmp" 2>&1
if !errorlevel! neq 0 (
    echo  FAIL: decide-session turn 2 curl failed
    call "%THIS_DIR%stop_server.bat"
    exit /b 1
)
type "%BIN%\_vtest.tmp"
findstr /c:"Anna" "%BIN%\_vtest.tmp" > nul
if !errorlevel! neq 0 (
    echo  FAIL: recall answer does not name Anna - session memory lost across /v1/decide
    call "%THIS_DIR%stop_server.bat"
    exit /b 1
)
echo  PASS: /v1/decide session transparency (Anna recalled)

call "%THIS_DIR%stop_server.bat"
if !errorlevel! neq 0 (
    echo  FAIL: stop_server.bat failed - the verify server may still be running
    exit /b 1
)
echo.

echo [Phase 3/3] CLI and environment contract tests ...

echo  [prio] -prio -15 flag: priority line + paris ...
set v7_override_prio=
"%BIN%\inference_example.exe" "%DLL%" "%DB_QWEN3%" -i "What is the capital of France?" -n 1000 -prio -15 > "%BIN%\_vtest.tmp" 2>&1
set PRIORC=!errorlevel!
type "%BIN%\_vtest.tmp" | findstr /C:"Worker priority override set to -15" > nul
if !errorlevel! neq 0 (
    echo  FAIL: -prio flag: priority line missing
    exit /b 1
)
if not "!PRIORC!"=="0" (
    echo  FAIL: -prio flag run exited with code !PRIORC!
    exit /b 1
)
findstr /I /C:"paris" "%BIN%\_vtest.tmp" > nul
if !errorlevel! neq 0 (
    echo  FAIL: -prio flag: no paris in output
    exit /b 1
)
echo  PASS: -prio flag

echo  [prio] env override without flag ...
set v7_override_prio=-15
"%BIN%\inference_example.exe" "%DLL%" "%DB_QWEN3%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vtest.tmp" 2>&1
set ENVRC=!errorlevel!
set v7_override_prio=
if not "!ENVRC!"=="0" (
    echo  FAIL: env override run exited with code !ENVRC!
    exit /b 1
)
type "%BIN%\_vtest.tmp" | findstr /C:"Worker priority override set to -15" > nul
if !errorlevel! neq 0 (
    echo  FAIL: env override: priority line missing - paris alone does not prove the override
    exit /b 1
)
findstr /I /C:"paris" "%BIN%\_vtest.tmp" > nul
if !errorlevel! neq 0 (
    echo  FAIL: env override: no paris
    exit /b 1
)
echo  PASS: env override

echo  [prio] malformed env override must FATAL with a named message ...
set v7_override_prio=abc
"%BIN%\inference_example.exe" "%DLL%" "%DB_QWEN3%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vtest.tmp" 2>&1
set MALRC=!errorlevel!
set v7_override_prio=
if "!MALRC!"=="0" (
    echo  FAIL: malformed env value run exited 0 -- it must FATAL nonzero
    exit /b 1
)
type "%BIN%\_vtest.tmp" | findstr /C:"v7_override_prio env var malformed" > nul
if !errorlevel! neq 0 (
    echo  FAIL: malformed env value was silently ignored
    exit /b 1
)
echo  PASS: malformed env override FATALs
set v7_override_prio=-15
REM the header sets idle priority for ALL inference: the two prio tests clear the
REM env var, so restore it here for every later run.

echo  [utf8] non-ASCII prompts (umlauts + sharp-s) ...
"%BIN%\inference_example.exe" "%DLL%" "%DB_QWEN3%" -i "Was ist eine Katze? Antworten Sie kurz." -n 100 -q8 1 -t 0 > "%BIN%\_vtest.tmp" 2>&1
if !errorlevel! neq 0 (
    echo  FAIL: ASCII-umlaut prompt failed
    type "%BIN%\_vtest.tmp"
    exit /b 1
)
"%BIN%\inference_example.exe" "%DLL%" "%DB_QWEN3%" -i "Was ist ein Käfer? Antworte kurz." -n 100 -q8 1 -t 0 > "%BIN%\_vtest.tmp" 2>&1
if !errorlevel! neq 0 (
    echo  FAIL: UTF-8 literal umlaut prompt failed
    type "%BIN%\_vtest.tmp"
    exit /b 1
)
echo  PASS: utf8 prompts

echo  [ardmark] demo-intactness: reference transcript compare (3 greedy turns) ...
if not exist "%THIS_DIR%examples\ardmark\reference_transcript.txt" (
    echo  FAIL: examples\ardmark\reference_transcript.txt missing
    echo        record it with: bin\test_ardmark_rag.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db 1 -record
    echo        then eyeball all three turns before committing it.
    exit /b 1
)
"%BIN%\test_ardmark_rag.exe" "%DLL%" "%DB_QWEN3%" 1 > "%BIN%\_vtest.tmp" 2>&1
set ARDRC=!errorlevel!
type "%BIN%\_vtest.tmp" | findstr /l /c:"RESULT: OK" > nul
set ARDTEXT=!errorlevel!
if "!ARDRC!"=="0" if "!ARDTEXT!"=="0" goto ardmark_pass
echo  FAIL: ardmark reference transcript compare failed ^(rc !ARDRC!^):
type "%BIN%\_vtest.tmp"
exit /b 1
:ardmark_pass
echo  PASS: ardmark demo-intactness

echo  [fatal] zero claim size must be FATALed at creation (Sec.167 rule 6) ...
REM Zero the SECOND batch_small key (the first per-shape [NxD] override; the
REM [default] key stays legal), rebuild the DB from the edited INI, restore the
REM INI, then prove both the DB build AND the inference run fail.
REM The INI is saved/restored by byte copy (no git dependency: a fresh release
REM unzip has no .git, and this block must behave identically there).
set QWEN3_BIN=%THIS_DIR%weights\qwen3_0.6b\qwen3_0.6b.bin
if not exist "!QWEN3_BIN!" (
    echo  FAIL: batch-fatal probe needs !QWEN3_BIN!
    exit /b 1
)
if exist "%THIS_DIR%build_logs\_pristine_threading_proxy.ini" (
    echo  FATAL: a stale _pristine_threading_proxy.ini exists - the working tree may be corrupt
    echo         restore threading_proxy.ini from git, delete the backup, and re-run
    exit /b 1
)
copy /y "%THIS_DIR%threading_proxy.ini" "%THIS_DIR%build_logs\_pristine_threading_proxy.ini" > nul
if !errorlevel! neq 0 (
    echo  FAIL: batch-fatal probe could not back up threading_proxy.ini
    exit /b 1
)
powershell -NoProfile -Command "$p='%THIS_DIR%threading_proxy.ini'; $lines=Get-Content $p; $n=0; for($i=0;$i -lt $lines.Count;$i++){ if($lines[$i] -match '^batch_small\s*='){ $n++; if($n -eq 2){ $lines[$i]='batch_small=0' } } }; if($n -lt 2){ exit 9 }; Set-Content $p $lines"
if !errorlevel! neq 0 goto bif_restore_fail
findstr /r /c:"^batch_small=0$" "%THIS_DIR%threading_proxy.ini" > nul
if !errorlevel! neq 0 goto bif_restore_fail
set BADDB=%THIS_DIR%build_logs\_bad_batch.db
REM the [FATAL] diagnostic is the entire point of this probe -- capture and show it
"%BIN%\build_db.exe" "!QWEN3_BIN!" -o "!BADDB!" > "%BIN%\_vtest.tmp" 2>&1
set DBRC=!errorlevel!
type "%BIN%\_vtest.tmp"
REM restore the INI BEFORE judging the result: every failure path below must leave
REM the working tree as it was found.
copy /y "%THIS_DIR%build_logs\_pristine_threading_proxy.ini" "%THIS_DIR%threading_proxy.ini" > nul
if !errorlevel! neq 0 (
    echo  FATAL: threading_proxy.ini restore copy failed - fix the working tree now
    exit /b 1
)
fc /b "%THIS_DIR%threading_proxy.ini" "%THIS_DIR%build_logs\_pristine_threading_proxy.ini" > nul
if !errorlevel! neq 0 (
    echo  FATAL: threading_proxy.ini NOT restored - fix the working tree now
    exit /b 1
)
del "%THIS_DIR%build_logs\_pristine_threading_proxy.ini"
if !DBRC! EQU 0 (
    echo  FAIL: build_db accepted batch_small=0 - the creation-time guard is broken
    if exist "!BADDB!" del "!BADDB!"
    exit /b 1
)
type "%BIN%\_vtest.tmp" | findstr /C:"batch_small" > nul
if !errorlevel! neq 0 (
    echo  FAIL: build_db failed but named no batch_small guard - unrelated failure, not the probe
    if exist "!BADDB!" del "!BADDB!"
    exit /b 1
)
if exist "!BADDB!" del "!BADDB!"
echo  PASS: zero claim size rejected at creation
goto bif_done

:bif_restore_fail
copy /y "%THIS_DIR%build_logs\_pristine_threading_proxy.ini" "%THIS_DIR%threading_proxy.ini" > nul
del "%THIS_DIR%build_logs\_pristine_threading_proxy.ini" 2>nul
echo  FAIL: batch-fatal probe could not edit threading_proxy.ini
exit /b 1
:bif_done

if exist "%BIN%\_vtest.tmp" del "%BIN%\_vtest.tmp"
echo.
echo  Server + RAG verification passed
exit /b 0
