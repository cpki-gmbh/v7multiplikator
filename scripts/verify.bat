@echo off
REM verify.bat - Full verification pipeline: build + validate + inference (F32 + Q8)
REM
REM Phases (ASM component verification ONLY -- nothing else):
REM   1. Build all components (build.bat)
REM   2. Ensure the Qwen3-0.6B database exists (build it if missing)
REM   3. Run ALL validators (this is the ONLY script that runs validators)
REM   4. Generate-path validator (v7s_generate end-to-end, checks paris)
REM   5. Inference smoke tests: inference_example + dbg_inference, F32 + Q8 (output must contain "paris")
REM
REM Vision gates live in verify_vision.bat. Server + RAG session gates and the
REM CLI/environment contract tests live in verify_server_and_rag.bat. Neither
REM runs here: verify verifies the ASM component and nothing else.
REM
REM ABORTS IMMEDIATELY on the first failure. Exit code: 0 = all passed, 1 = failure.
REM Debugging tip: replace "@echo off" with "@echo on" to trace every executed line.

setlocal ENABLEDELAYEDEXPANSION

rem Run all inference at IDLE worker priority (v7_override_prio, set_worker_priority.inc):
rem verify must not starve everything else on the machine while the sampler smokes run.
rem Validators do not spin up the worker pool; the Phase 4/5 inference runs do.
set v7_override_prio=-15

set THIS_DIR=%~dp0..\
cd /d "%THIS_DIR%"
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set DLL=%BIN%\v7_threading.dll
set DB_QWEN3=%THIS_DIR%weights\databases\qwen3_0.6b\lz4ht.db
set VPASS=0
set IPASS=0
set VKNOWN=0
echo  v7 verification pipeline
echo  Build, all validators, inference F32 + Q8, output must contain "paris"
echo  Aborts on first failure
echo.

REM ====== Phase 1/5: Build everything ======
echo [Phase 1/5] Building all components ...
call "%~dp0build.bat"
if !errorlevel! neq 0 (
    echo  Build failed. Cannot verify broken code.
    echo  This must be fixed now before any other work.
    exit /b 1
)
echo.

REM ====== Phase 2/5: Qwen3-0.6B database (MANDATORY, ALWAYS rebuilt) ======
echo [Phase 2/5] Qwen3-0.6B database ...
set QWEN3_BIN=%THIS_DIR%weights\qwen3_0.6b\qwen3_0.6b.bin
if not exist "!QWEN3_BIN!" (
    echo  Qwen3-0.6B checkpoint not found.
    echo  Run fetch_qwen3_0.6b.bat to download and export the model.
    exit /b 1
)
if not exist "%THIS_DIR%weights\databases\qwen3_0.6b" mkdir "%THIS_DIR%weights\databases\qwen3_0.6b"
if exist "%DB_QWEN3%" del /f /q "%DB_QWEN3%"
echo  Building DB from !QWEN3_BIN! ...
     "%BIN%\build_db.exe" "!QWEN3_BIN!" -o "%DB_QWEN3%" -f -autoexec "%THIS_DIR%autoexec.logo"
if !errorlevel! neq 0 (
    echo  Qwen3-0.6B database build failed.
    exit /b 1
)
if not exist "%DB_QWEN3%" (
    echo  Qwen3-0.6B database was not created. Aborting.
    exit /b 1
)
echo  Database built: %DB_QWEN3%
echo.

REM ====== Phase 3/5: Validators (need DLL + DB) ======
echo [Phase 3/5] Running all validators ...
REM v_sampler_validate runs the 40-case sampler gate DLL-only (the C reference
REM is a bit-exact port of the ASM sampler), then pins the public-API setter
REM data-fixing contract (NaN/sub-epsilon -> 1e-6f, 0.0 verbatim) on a live
REM engine context -- that half needs the DB.
"%TOOLS%\v_sampler_validate.exe" "%DLL%" "%DB_QWEN3%"
if !errorlevel! equ 0 (
    echo  PASS: v_sampler_validate
    set /a VPASS+=1
) else (
    echo  FAIL: v_sampler_validate
    echo  Verification failed at v_sampler_validate. This must be fixed now.
    exit /b 1
)
REM v_swiglu_validate self-classifies its failures: exit 0 means every
REM failure is one of the documented Q8xQ8 float-vs-double known-limit
REM shapes (printed and counted in the summary line); any F32xQ8 failure
REM or undocumented failing shape exits 1 and aborts here.
"%TOOLS%\v_swiglu_validate.exe" "%DLL%" --db "%DB_QWEN3%" > "%THIS_DIR%build_logs\swiglu_out.txt" 2>&1
set "SWIGLU_RC=!errorlevel!"
type "%THIS_DIR%build_logs\swiglu_out.txt"
if !SWIGLU_RC! neq 0 (
    echo  FAIL: v_swiglu_validate
    echo  Verification failed at v_swiglu_validate. This must be fixed now.
    exit /b 1
)
findstr /C:"documented known limits" "%THIS_DIR%build_logs\swiglu_out.txt" > nul
if !errorlevel! equ 0 (
    echo  KNOWN LIMIT: v_swiglu_validate Q8xQ8 documented shapes -- counted as FAIL, expected
    set /a VKNOWN+=1
) else (
    echo  PASS: v_swiglu_validate
    set /a VPASS+=1
)
for %%v in (v_rmsnorm_validate v_add_inplace_validate v_mdb_validate v_execute_validate v_readwrite_validate v_sprintf_validate v_win32_quote_validate v_bicubic_validate v_matmul_validate v_fused_validate v_memcpy_validate v_quantize_validate v_attention_validate v_encode_validate v_vit_linear_validate v_stream_validate v_config_probe_validate) do (
    "%TOOLS%\%%v.exe" "%DLL%" --db "%DB_QWEN3%"
    if !errorlevel! equ 0 (
        echo  PASS: %%v
        set /a VPASS+=1
    ) else (
        echo  FAIL: %%v
        echo  Verification failed at %%v. This must be fixed now.
        exit /b 1
    )
)
echo  Validators so far: !VPASS! passed
echo.
REM SectorLogo validator runs LAST of the validator phase with -with-slow:
REM the definitive gate keeps full coverage (the slow suites are dominated by
REM ~1s engine-recovery joins and one ~4-minute churn block, fine here -- the
REM fast default run is for interactive development).
"%TOOLS%\v_sectorlogo_validate.exe" "%DLL%" --db "%DB_QWEN3%" -with-slow
if !errorlevel! equ 0 (
    echo  PASS: v_sectorlogo_validate -with-slow
    set /a VPASS+=1
) else (
    echo  FAIL: v_sectorlogo_validate -with-slow
    echo  Verification failed at v_sectorlogo_validate. This must be fixed now.
    exit /b 1
)
echo.
REM ====== Phase 4/5: Generate-path validator (v7s_generate end-to-end, checks paris) ======
echo [Phase 4/5] Generate-path validator ...
"%TOOLS%\v_generate_validate.exe" "%DLL%" "%DB_QWEN3%"

if !errorlevel! equ 0 (
    echo  PASS: v_generate_validate
    set /a VPASS+=1
) else (
    echo  FAIL: v_generate_validate
    echo  Verification failed at v_generate_validate generate path  This must be fixed now 
    exit /b 1
)
echo  Validators so far: !VPASS! passed
echo.

REM ====== Phase 5/5: Inference tests (inference_example + dbg_inference, F32 + Q8) ======
echo [Phase 5/5] Inference tests: inference_example + dbg_inference (F32 + Q8) ...

echo  [F32] inference_example.exe ...
"%BIN%\inference_example.exe" "%DLL%" "%DB_QWEN3%" > "%BIN%\_vtest.tmp" 2>&1
set SMOKE_RC=!errorlevel!
type "%BIN%\_vtest.tmp"
findstr /i paris "%BIN%\_vtest.tmp" > nul
if !errorlevel! neq 0 (
    echo  FAIL: inference_example.exe F32 - output does NOT contain paris
    del "%BIN%\_vtest.tmp"
    echo  Verification failed at inference_example.exe F32. This must be fixed now.
    exit /b 1
)
if !SMOKE_RC! neq 0 (
    echo  FAIL: inference_example.exe F32 - exited with code !SMOKE_RC!, not 0
    del "%BIN%\_vtest.tmp"
    echo  Verification failed at inference_example.exe F32. This must be fixed now.
    exit /b 1
)
echo  PASS: inference_example.exe F32 - output contains paris, exit 0
set /a IPASS+=1
echo.

echo  [Q8] inference_example.exe -q8 ...
"%BIN%\inference_example.exe" "%DLL%" "%DB_QWEN3%" -q8 > "%BIN%\_vtest.tmp" 2>&1
set SMOKE_RC=!errorlevel!
type "%BIN%\_vtest.tmp"
findstr /i paris "%BIN%\_vtest.tmp" > nul
if !errorlevel! neq 0 (
    echo  FAIL: inference_example.exe Q8 - output does NOT contain paris
    del "%BIN%\_vtest.tmp"
    echo  Verification failed at inference_example.exe Q8. This must be fixed now.
    exit /b 1
)
if !SMOKE_RC! neq 0 (
    echo  FAIL: inference_example.exe Q8 - exited with code !SMOKE_RC!, not 0
    del "%BIN%\_vtest.tmp"
    echo  Verification failed at inference_example.exe Q8. This must be fixed now.
    exit /b 1
)
echo  PASS: inference_example.exe Q8 - output contains paris, exit 0
set /a IPASS+=1
echo.

echo  [F32] dbg_inference.exe (pure ASM harness) ...
"%BIN%\dbg_inference.exe" "%DLL%" "%DB_QWEN3%" > "%BIN%\_vtest.tmp" 2>&1
set SMOKE_RC=!errorlevel!
type "%BIN%\_vtest.tmp"
findstr /i paris "%BIN%\_vtest.tmp" > nul
if !errorlevel! neq 0 (
    echo  FAIL: dbg_inference.exe F32 - output does NOT contain paris
    del "%BIN%\_vtest.tmp"
    echo  Verification failed at dbg_inference.exe F32. This must be fixed now.
    exit /b 1
)
if !SMOKE_RC! neq 0 (
    echo  FAIL: dbg_inference.exe F32 - exited with code !SMOKE_RC!, not 0
    del "%BIN%\_vtest.tmp"
    echo  Verification failed at dbg_inference.exe F32. This must be fixed now.
    exit /b 1
)
echo  PASS: dbg_inference.exe F32 - output contains paris, exit 0
set /a IPASS+=1
echo.

echo  [Q8] dbg_inference.exe -q8 (pure ASM harness) ...
"%BIN%\dbg_inference.exe" "%DLL%" "%DB_QWEN3%" -q8 > "%BIN%\_vtest.tmp" 2>&1
set SMOKE_RC=!errorlevel!
type "%BIN%\_vtest.tmp"
findstr /i paris "%BIN%\_vtest.tmp" > nul
if !errorlevel! neq 0 (
    echo  FAIL: dbg_inference.exe Q8 - output does NOT contain paris
    del "%BIN%\_vtest.tmp"
    echo  Verification failed at dbg_inference.exe Q8. This must be fixed now.
    exit /b 1
)
if !SMOKE_RC! neq 0 (
    echo  FAIL: dbg_inference.exe Q8 - exited with code !SMOKE_RC!, not 0
    del "%BIN%\_vtest.tmp"
    echo  Verification failed at dbg_inference.exe Q8. This must be fixed now.
    exit /b 1
)
echo  PASS: dbg_inference.exe Q8 - output contains paris, exit 0
set /a IPASS+=1
echo.

echo  [death] engine_death.exe (pseudo-exception catch, lean -- no DB) ...
"%BIN%\engine_death.exe" "%DLL%"
if !errorlevel! neq 0 (
    echo  FAIL: engine_death.exe - pseudo-exception catch failed
    echo  Verification failed at engine_death. This must be fixed now.
    exit /b 1
)
echo  PASS: engine_death.exe - caught FATAL, host survived, clean unload
set /a IPASS+=1
echo.

echo  [safe] v7_only_safe_check.exe (V7_ONLY_SAFE build mode, public tier only) ...
"%BIN%\v7_only_safe_check.exe" "%DLL%" "%DB_QWEN3%"
if !errorlevel! neq 0 (
    echo  FAIL: v7_only_safe_check.exe - safe-mode host failed
    echo  Verification failed at v7_only_safe_check. This must be fixed now.
    exit /b 1
)
echo  PASS: v7_only_safe_check.exe - safe struct completed a real chat
set /a IPASS+=1
echo.

echo  [chat] chat_only.exe (2-turn chat smoke: fresh + follow-up) ...
"%BIN%\chat_only.exe" "%DLL%" "%DB_QWEN3%"
if !errorlevel! neq 0 (
    echo  FAIL: chat_only.exe
    echo  Verification failed at chat_only. This must be fixed now.
    exit /b 1
)
echo  PASS: chat_only.exe
set /a IPASS+=1
echo.

echo  [generate] generate_only.exe (raw v7_generate turn) ...
"%BIN%\generate_only.exe" "%DLL%" "%DB_QWEN3%"
if !errorlevel! neq 0 (
    echo  FAIL: generate_only.exe
    echo  Verification failed at generate_only. This must be fixed now.
    exit /b 1
)
echo  PASS: generate_only.exe
set /a IPASS+=1
echo.

echo  [kv] kv_cache.exe (save/load round-trip with session recall) ...
"%BIN%\kv_cache.exe" "%DLL%" "%DB_QWEN3%"
if !errorlevel! neq 0 (
    echo  FAIL: kv_cache.exe
    echo  Verification failed at kv_cache. This must be fixed now.
    exit /b 1
)
echo  PASS: kv_cache.exe
set /a IPASS+=1
echo.

echo  [multiturn] multiturn.exe (3-turn cat/dog/bird, Q8, thinking on) ...
"%BIN%\multiturn.exe" "%DLL%" "%DB_QWEN3%" 1
if !errorlevel! neq 0 (
    echo  FAIL: multiturn.exe
    echo  Verification failed at multiturn. This must be fixed now.
    exit /b 1
)
echo  PASS: multiturn.exe
set /a IPASS+=1
echo.

if exist "%BIN%\_vtest.tmp" del "%BIN%\_vtest.tmp"
echo.

REM ====== Moved out: Server + RAG + CLI/env contract tests ======
REM These tests verify the SERVER and the SectorLogo/RAG contract layers, not
REM the ASM component. They live in verify_server_and_rag.bat now.

REM ====== Summary ======
echo  Verification summary
echo  Validators: !VPASS! passed, !VKNOWN! documented known limits (Sec.104)
echo  Inference:  !IPASS! passed
REM the totals are a CONTRACT: adding or removing a phase without updating its
REM counter must fail here loudly, not silently shift the baseline.
REM ASM-only verify: 10 inference phases = F32/Q8 x inference_example/dbg (4),
REM engine_death, v7_only_safe_check, chat_only, generate_only, kv_cache,
REM multiturn. Vision gates are in verify_vision.bat; server/RAG/CLI-contract
REM tests are in verify_server_and_rag.bat -- neither counts here.
REM The validator contract is 21 phases; v_swiglu_validate's documented Q8xQ8
REM known-limit failures (Sec.104 table, "do not relax") count into VKNOWN --
REM the pass+known sum must still be 21 so a lost counter fails loudly.
set EXPECT_INF=10
set /a VPASS_TOTAL=!VPASS! + !VKNOWN!
if !VPASS_TOTAL! NEQ 21 (
    echo  FAIL: validator pass+known count !VPASS_TOTAL! but the contract is 21 - a phase lost its counter
    exit /b 1
)
if !IPASS! NEQ !EXPECT_INF! (
    echo  FAIL: inference pass count !IPASS! but the contract is !EXPECT_INF! - a phase lost its counter
    exit /b 1
)
if !VKNOWN! GTR 0 (
    echo  All passed with !VKNOWN! documented known limit^(s^)
) else (
    echo  All passed
)
exit /b 0
