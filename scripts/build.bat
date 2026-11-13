@echo off
REM build.bat -- Build EVERYTHING: DLL + tools + validators + examples + server
REM This script ONLY BUILDS. It does NOT run validators, inference, or DB builds.
REM For validation, run verify.bat. For DB builds, run build_text_model_databases.bat
REM or build_vit_databases.bat.
REM
REM Prerequisites:
REM   1. Visual Studio 2022+ (x64 native tools)
REM   2. NASM assembler 3.x (release verified with 3.02rc6; set NASM_DIR or add to PATH)
REM   3. Python 3 (for ASM comment checking and the C/C++ freshness gate)
REM
REM Incremental C/C++ gate:
REM   Step 1 (NASM + ASM comment check) ALWAYS runs. Steps 2-6 (every cl/link
REM   invocation) are gated PER STEP: a step is skipped only when
REM   src\check_stale.py --step <tokens> proves from file timestamps that none
REM   of that step's declared inputs is newer than its oldest declared output.
REM   The skip is always printed, never silent, and can only cost time -- every
REM   ambiguity (missing output, missing input, bad manifest, checker exception)
REM   resolves to "rebuild that step". Force a full C/C++ rebuild with:
REM   build.bat /F   (or /C)

setlocal ENABLEDELAYEDEXPANSION

set THIS_DIR=%~dp0..\

REM A running v7_server holds bin\v7_threading.dll locked -> LNK1104 at link time.
REM Stop OUR instance automatically (matched by image path -- an unrelated
REM process that merely shares the name is never touched):
set "OUR_SERVER=%THIS_DIR%bin\v7_server.exe"
powershell -NoProfile -Command "Get-Process v7_server -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $env:OUR_SERVER } | Stop-Process -Force" >nul 2>&1

set SRC_DIR=%THIS_DIR%src
set ASM_DIR=%SRC_DIR%\asm
set BIN_DIR=%THIS_DIR%bin
set TOOLS_DIR=%THIS_DIR%tools
set TOOL_SRC=%THIS_DIR%tools
set ROOT_DIR=%THIS_DIR:~0,-1%
set BUILD_TMP=%THIS_DIR%build_tmp
set ZYDIS_INC=%THIS_DIR%zydis-master\include
set ZYCORE_INC=%THIS_DIR%zycore-c-master\include
set ZYDIS_LIB=%THIS_DIR%Zydis.lib
set ERRORS=0

REM --- Zydis availability probe (real Zydis in dev; no-op stub in its absence) ---
REM Dev keeps zydis-master\, zycore-c-master\ and Zydis.lib and always builds the
REM tools with real VEH crash disassembly (root zydis_debug_header.h). A tree
REM without them (the release) compiles every tool against the no-op stub header
REM in tools\zydis_stub instead: same include name, same two entry points, no
REM decoder -- a release-built tool crashes silently. The stub dir replaces
REM ZYDIS_INC/ZYCORE_INC so its header wins the include search ahead of the root
REM real header; ZYDIS_LIB empties so no link line references the missing .lib.
set ZYDIS_PRESENT=0
if exist "%ZYDIS_INC%\Zydis\Zydis.h" if exist "%ZYDIS_LIB%" set ZYDIS_PRESENT=1
if "%ZYDIS_PRESENT%"=="1" goto zydis_ready
echo [zydis] real Zydis not found -- building tools against the no-op stub (tools\zydis_stub)
set ZYDIS_INC=%THIS_DIR%tools\zydis_stub
set ZYCORE_INC=%THIS_DIR%tools\zydis_stub
set ZYDIS_LIB=
set ZYDIS_LIB_ARG=
goto zydis_done
:zydis_ready
set ZYDIS_LIB_ARG="%ZYDIS_LIB%"
:zydis_done

REM --- Purge stale DLLs (root copy causes silent load of wrong DLL) ---
del "%THIS_DIR%v7_threading.dll" 2>nul
del "%BIN_DIR%\v7_threading.dll" 2>nul

REM --- Wipe build_logs contents (transient run output only; the directory itself
REM     stays, everything inside it is disposable and must never hold state) ---
if exist "%THIS_DIR%build_logs" (
    del /q /f "%THIS_DIR%build_logs\*" 2>nul
    for /d %%D in ("%THIS_DIR%build_logs\*") do rmdir /s /q "%%D"
)

REM --- Wipe build_tmp contents (every compiler object lands here via /Fo; pure
REM     cache, same disposal rules as build_logs -- the source tree must never
REM     collect stray .obj files) ---
if exist "%BUILD_TMP%" (
    del /q /f "%BUILD_TMP%\*" 2>nul
    for /d %%D in ("%BUILD_TMP%\*") do rmdir /s /q "%%D"
)

REM --- VS setup ---
REM Locate any VS 2017+ installation (Community, Professional, Enterprise or
REM Build Tools) via vswhere -- the C++ workload is what matters, not the
REM edition. Set VS_PATH to a specific vcvarsall.bat to override the probe.
if "%VS_PATH%"=="" (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" (
        echo [FATAL] vswhere not found. Install Visual Studio or Build Tools with the "Desktop development with C++" workload, or set VS_PATH to vcvarsall.bat.
        exit /b 1
    )
    set VS_ROOT=
    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_ROOT=%%i"
    if "!VS_ROOT!"=="" (
        echo [FATAL] No Visual Studio with the C++ workload found. Install the "Desktop development with C++" workload or set VS_PATH.
        exit /b 1
    )
    set "VS_PATH=!VS_ROOT!\VC\Auxiliary\Build\vcvarsall.bat"
)
if not exist "!VS_PATH!" (
    echo [FATAL] VS_PATH "!VS_PATH!" not found.
    exit /b 1
)
call "!VS_PATH!" x64

REM --- NASM setup (pinned: tools\nasm.exe takes precedence over any PATH nasm) ---
if "%NASM_DIR%"=="" set NASM_DIR=%THIS_DIR%tools
set PATH=%NASM_DIR%;%PATH%
where nasm
if !errorlevel! neq 0 (
    echo [FATAL] NASM not found. Set NASM_DIR to nasm.exe location or add it to PATH.
    exit /b 1
)
nasm -v
REM The tree assembles cleanly with NASM 2.16+ and with 3.x (3.02rc6 is the
REM pinned release candidate); anything older mis-handles the BSS align padding.
nasm -v | findstr /r /c:"version 2\.1[6-9]" /c:"version 3\." >nul
if !errorlevel! neq 0 (
    echo [FATAL] NASM 2.16 or 3.x required -- see the printed version above.
    exit /b 1
)

if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"
if not exist "%BUILD_TMP%" mkdir "%BUILD_TMP%"
if not exist "%THIS_DIR%build_logs" mkdir "%THIS_DIR%build_logs"

REM --- nasmfmt (comment-checker auto-formatter) ---
REM The ASM comment checker (step 1) auto-fixes format violations by invoking
REM src\nasmfmt\nasmfmt.exe. The prebuilt exe is committed, so a fresh checkout
REM needs no Go at all; if the exe is absent (rare), the vendored Go source is
REM built once -- Go 1.21+ is only required in that case.
if not exist "%SRC_DIR%\nasmfmt\nasmfmt.exe" (
    echo    Building nasmfmt from src\nasmfmt ...
    pushd "%SRC_DIR%\nasmfmt"
    go build -o nasmfmt.exe .
    popd
    if !errorlevel! neq 0 (
        echo [FATAL] nasmfmt build FAILED. Go is required: https://go.dev/dl/
        exit /b 1
    )
)

echo.
echo ============================================================================
echo  v7 Build Pipeline (build only -- no validation)
echo ============================================================================
echo.

REM ====== Step 1/6: Build v7_threading.dll ======
echo [1/6] Building v7_threading.dll ...

set NASM_FLAGS=-f win64 -O0 -w-other
if "%TRACE_DEBUG%"=="1" (
    echo    [TRACE ENABLED]
    set NASM_FLAGS=!NASM_FLAGS! -DTRACE_ENABLE
)
if "%CPU0_DIAG%"=="1" (
    echo    [CPU0 DIAG ENABLED]
    set NASM_FLAGS=!NASM_FLAGS! -DCPU0_DIAG -DPROFILE_MODE -DTRACE_ENABLE
)
if "%PROFILE_MODE%"=="1" (
    echo    [PROFILE MODE ENABLED]
    set NASM_FLAGS=!NASM_FLAGS! -DPROFILE_MODE
)
nasm !NASM_FLAGS! -I"%ASM_DIR%" "%ASM_DIR%\v7_threading.asm" -o "%BIN_DIR%\v7_threading.obj"
if !errorlevel! neq 0 (
    echo [FATAL] NASM assembly FAILED
    exit /b 1
)

REM --- Comment check (strict, autofix, ALL files) ---
echo    Checking comments ...
set V7_ASM_DIR=%ASM_DIR%
python "%SRC_DIR%\check_asm_comments.py"
if !errorlevel! neq 0 (
    echo [FATAL] Comment check FAILED
    exit /b 1
)

echo    Linking ...
link /dll /entry:DllMain /NODEFAULTLIB /largeaddressaware:no /def:"%SRC_DIR%\v7_threading.def" "%BIN_DIR%\v7_threading.obj" kernel32.lib user32.lib synchronization.lib /out:"%BIN_DIR%\v7_threading.dll"
if !errorlevel! neq 0 (
    echo [FATAL] Link FAILED
    exit /b 1
)
del "%BIN_DIR%\v7_threading.obj" 2>nul
echo  OK: v7_threading.dll
echo.

REM --- Release hygiene gate DISABLED (owner directive, Oct 2026): python
REM     "%SRC_DIR%\check_release.py" "%ROOT_DIR%" + [FATAL] on nonzero removed.
REM ---

REM ====== C/C++ freshness gate (steps 2-6) ======
REM The gate is read-only, conservative and PER STEP: each build step skips only
REM when its own declared outputs are provably fresh against its own declared
REM inputs (check_stale.py --step N). A stale target rebuilds its step, not all
REM of steps 2-6. /F and /C skip the checks entirely and rebuild everything.
set C_FORCE=
if "%~1"=="/F" set C_FORCE=1
if "%~1"=="/C" set C_FORCE=1
if defined C_FORCE echo [2-6/6] C/C++ freshness checks SKIPPED -- forced full rebuild via "%~1".

:c_build
REM NOTE: --root gets %ROOT_DIR% (the same path with the trailing backslash
REM stripped), never %THIS_DIR%. Quoting a Windows path that ends in a backslash
REM makes the closing quote an escaped quote, which swallows every later argument.

REM ====== Step 2/6: Build tools (build_db.exe) ======
set C_SKIP=0
if defined C_FORCE goto c_step2_build
call :c_fresh_step 2
if !C_SKIP! equ 1 goto c_step2_done
echo  [C/C++] Step 2 freshness inconclusive -- rebuilding step 2.
:c_step2_build
echo [2/6] Building tools ...
cl /nologo /O2 /arch:AVX2 /MD /std:c17 /DNDEBUG /fp:fast /GS- /GL- /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%ROOT_DIR%" /I"%TOOLS_DIR%" "%TOOL_SRC%\db_build_tool.c" /Fe"%BIN_DIR%\build_db.exe" /link /subsystem:console /incremental:no user32.lib %ZYDIS_LIB_ARG%
if !errorlevel! neq 0 (
    echo [FATAL] build_db.exe FAILED
    exit /b 1
)
echo  OK: build_db.exe
echo.
:c_step2_done

REM ====== Step 3a/6: Build validators and RAG tools ======
set C_SKIP=0
if defined C_FORCE goto c_step3a_build
call :c_fresh_step 3a
if !C_SKIP! equ 1 goto c_step3a_done
echo  [C/C++] Step 3a freshness inconclusive -- rebuilding step 3a.
:c_step3a_build
echo [3/6] Building validators and tools ...
set V_CFLAGS=/nologo /O2 /arch:AVX2 /std:c++17 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%ROOT_DIR%" /I"%TOOLS_DIR%"
set V_CFLAGS_NOOPT=/nologo /Od /std:c++17 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%ROOT_DIR%" /I"%TOOLS_DIR%"
set V_LIBS=%ZYDIS_LIB_ARG%

REM --- DLL-only validators ---
REM v_rmsnorm_validate is built WITHOUT /O2: at /O2 its fused add+RMSNorm C
REM reference comes out as the kernel output times a per-test constant
REM (ref = k*asm, k constant within each test) and 14 of 94 tests fail.
REM REPRODUCED Oct 2026 even AFTER the exclusive-buffer fix (ref gets its own
REM residual copy), so plain aliasing/reordering is ruled out -- the /O2
REM vectorized reduction inside the inlined reference interacts with the
REM optimizer in a way /Od avoids. Next root-cause step: /fsanitize=address
REM + comparing the /O2 asm of the reference between a small main and this
REM large main. The kernel under test was never wrong; the unoptimized
REM validator is the honest gate until then.
cl %V_CFLAGS_NOOPT% "%TOOLS_DIR%\v_rmsnorm_validate.cpp" /Fe"%TOOLS_DIR%\v_rmsnorm_validate.exe" /link %V_LIBS%
if !errorlevel! neq 0 (echo FAILED: v_rmsnorm_validate & set /a ERRORS+=1) else (echo OK: v_rmsnorm_validate)
for %%f in (v_memcpy_validate v_add_inplace_validate v_swiglu_validate v_quantize_validate v_mdb_validate v_config_probe_validate) do (
    cl %V_CFLAGS% "%TOOLS_DIR%\%%f.cpp" /Fe"%TOOLS_DIR%\%%f.exe" /link %V_LIBS%
    if !errorlevel! neq 0 (echo FAILED: %%f & set /a ERRORS+=1) else (echo OK: %%f)
)

REM --- Weight-dependent validators ---
for %%f in (v_matmul_validate v_fused_validate v_sampler_validate v_encode_validate) do (
    cl %V_CFLAGS% "%TOOLS_DIR%\%%f.cpp" /Fe"%TOOLS_DIR%\%%f.exe" /link %V_LIBS%
    if !errorlevel! neq 0 (echo FAILED: %%f & set /a ERRORS+=1) else (echo OK: %%f)
)

REM --- Generate-path validator (v7s_generate end-to-end, F32+Q8, needs full DB) ---
cl %V_CFLAGS% "%TOOLS_DIR%\v_generate_validate.cpp" /Fe"%TOOLS_DIR%\v_generate_validate.exe" /link %V_LIBS%
if !errorlevel! neq 0 (echo FAILED: v_generate_validate & set /a ERRORS+=1) else (echo OK: v_generate_validate)

REM --- Attention validators/benchmarks ---
for %%f in (v_attention_validate b_attention_benchmark b_attention_loadtest) do (
    cl %V_CFLAGS% "%TOOLS_DIR%\%%f.cpp" /Fe"%TOOLS_DIR%\%%f.exe" /link %V_LIBS%
    if !errorlevel! neq 0 (echo FAILED: %%f & set /a ERRORS+=1) else (echo OK: %%f)
)

REM --- StreamingLLM validator (small-window wrap + eviction callback) ---
for %%f in (v_stream_validate) do (
    cl %V_CFLAGS% "%TOOLS_DIR%\%%f.cpp" /Fe"%TOOLS_DIR%\%%f.exe" /link %V_LIBS%
    if !errorlevel! neq 0 (echo FAILED: %%f & set /a ERRORS+=1) else (echo OK: %%f)
)

REM --- ViT, SectorLogo, execute, readwrite, sprintf, quote, bicubic validators ---
for %%f in (v_vit_linear_validate v_sectorlogo_validate v_execute_validate v_readwrite_validate v_sprintf_validate v_win32_quote_validate v_bicubic_validate) do (
    cl %V_CFLAGS% "%TOOLS_DIR%\%%f.cpp" /Fe"%TOOLS_DIR%\%%f.exe" /link %V_LIBS%
    if !errorlevel! neq 0 (echo FAILED: %%f & set /a ERRORS+=1) else (echo OK: %%f)
)

REM --- RAG tools ---
cl %V_CFLAGS% "%THIS_DIR%examples\rag\bm25_rag.cpp" /Fe"%THIS_DIR%examples\rag\bm25_rag.exe" /link %V_LIBS%
if !errorlevel! neq 0 (echo FAILED: bm25_rag & set /a ERRORS+=1) else (echo OK: bm25_rag)
:c_step3a_done

REM Step-3a errors MUST propagate even when step 3b is freshness-skipped:
REM the only other ERRORS check sits inside the step-3b build path, so a
REM skipped 3b would exit 0 with stale or missing validator exes -- exactly
REM the "stale binary the validators certify" failure mode.
if !ERRORS! gtr 0 (
    echo.
    echo [FATAL] !ERRORS! tool build failures
    exit /b 1
)

REM ====== Step 3b/6: standalone benchmarks, DB query tools, autotune, FFI shim ======
set C_SKIP=0
if defined C_FORCE goto c_step3b_build
call :c_fresh_step 3b
if !C_SKIP! equ 1 goto c_step3b_done
echo  [C/C++] Step 3b freshness inconclusive -- rebuilding step 3b.
:c_step3b_build
REM --- Standalone benchmarks ---
REM All benchmark sources here are release content (tools\*.cpp, release tree).
for %%f in (b_loadtest b_memcpy_loadtest b_fused_loadtest b_rmsnorm_loadtest b_add_inplace_loadtest b_fused_add_rmsnorm_loadtest b_rmsnorm_rope_loadtest b_swiglu_loadtest b_swiglu_q8_loadtest b_benchmark b_vit_linear) do (
    cl /nologo /O2 /arch:AVX2 /std:c++17 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%ROOT_DIR%" /I"%TOOLS_DIR%" "%TOOLS_DIR%\%%f.cpp" /Fe"%TOOLS_DIR%\%%f.exe" /link %ZYDIS_LIB_ARG%
    if !errorlevel! neq 0 (echo FAILED: %%f & set /a ERRORS+=1) else (echo OK: %%f)
)

REM --- DB query tools ---
for %%f in (mdb_info mdb_list mdb_query_key mdb_search mdb_dump) do (
    cl /nologo /O2 /arch:AVX2 /std:c++17 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%ROOT_DIR%" /I"%TOOLS_DIR%" "%TOOLS_DIR%\%%f.cpp" /Fe"%TOOLS_DIR%\%%f.exe" /link %ZYDIS_LIB_ARG%
    if !errorlevel! neq 0 (echo FAILED: %%f & set /a ERRORS+=1) else (echo OK: %%f)
)

REM b_sectorlogo_loadtest is intentionally NOT built here: it needs the
REM dev-only sectorlogo_c tree (sectorlogo_rewrite.h), which the release does
REM not ship. Build it by hand in dev if you need the SectorLogo load probe.
echo SKIP: b_sectorlogo_loadtest (needs the dev-only sectorlogo_c tree)

REM --- Autotune ---
cl /nologo /O2 /arch:AVX2 /std:c17 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%ROOT_DIR%" /I"%TOOLS_DIR%" /I"%TOOL_SRC%" "%TOOLS_DIR%\autotune_dispatch.c" /Fe"%TOOLS_DIR%\autotune_dispatch.exe" /link %ZYDIS_LIB_ARG%
if !errorlevel! neq 0 (echo FAILED: autotune_dispatch & set /a ERRORS+=1) else (echo OK: autotune_dispatch)

REM --- Shape speed test (benchmarks one matmul shape under the INI-committed config) ---
cl /nologo /O2 /arch:AVX2 /std:c17 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%ROOT_DIR%" /I"%TOOLS_DIR%" /I"%TOOL_SRC%" "%TOOLS_DIR%\test_shape_speed.c" /Fe"%TOOLS_DIR%\test_shape_speed.exe" /link %ZYDIS_LIB_ARG%
if !errorlevel! neq 0 (echo FAILED: test_shape_speed & set /a ERRORS+=1) else (echo OK: test_shape_speed)

REM --- Genome sort experiment (C++ genome-analysis copy of the shape speed test) ---
cl /nologo /O2 /arch:AVX2 /std:c++17 /EHsc /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%ROOT_DIR%" /I"%TOOLS_DIR%" /I"%TOOL_SRC%" "%TOOLS_DIR%\genome_sort_experiment.cpp" /Fe"%TOOLS_DIR%\genome_sort_experiment.exe" /link %ZYDIS_LIB_ARG%
if !errorlevel! neq 0 (echo FAILED: genome_sort_experiment & set /a ERRORS+=1) else (echo OK: genome_sort_experiment)

REM --- Genome cycle sort (fixed-list A/B/A climb test for the in-process tuner design) ---
cl /nologo /O2 /arch:AVX2 /std:c++17 /EHsc /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%ROOT_DIR%" /I"%TOOLS_DIR%" /I"%TOOL_SRC%" "%TOOLS_DIR%\genome_cycle_sort.cpp" /Fe"%TOOLS_DIR%\genome_cycle_sort.exe" /link %ZYDIS_LIB_ARG%
if !errorlevel! neq 0 (echo FAILED: genome_cycle_sort & set /a ERRORS+=1) else (echo OK: genome_cycle_sort)

REM --- Transplant INI (seed a shape config into a v7_tune_param population file) ---
cl /nologo /O2 /arch:AVX2 /std:c17 /Fo"%BUILD_TMP%\\" "%TOOLS_DIR%\transplant_ini.c" /Fe"%TOOLS_DIR%\transplant_ini.exe" /link
if !errorlevel! neq 0 (echo FAILED: transplant_ini & set /a ERRORS+=1) else (echo OK: transplant_ini)

REM --- slffi_test.dll ---
cl /nologo /O2 /LD /Fo"%BUILD_TMP%\\" "%TOOLS_DIR%\slffi_test_dll.c" /Fe:"%BIN_DIR%\slffi_test.dll" /link
if !errorlevel! neq 0 (echo FAILED: slffi_test.dll & set /a ERRORS+=1) else (echo OK: slffi_test.dll)
copy /Y "%BIN_DIR%\slffi_test.dll" "%THIS_DIR%slffi_test.dll" 

if !ERRORS! gtr 0 (
    echo.
    echo [FATAL] !ERRORS! tool build failures
    exit /b 1
)
echo  OK: All validators and tools built
echo.
:c_step3b_done

REM ====== Step 4/6: Build examples (incl. the engine test programs, step 4b) ======
set C_SKIP=0
if defined C_FORCE goto c_step4_build
call :c_fresh_step 4,4b
if !C_SKIP! equ 1 goto c_step4_done
echo  [C/C++] Step 4 freshness inconclusive -- rebuilding step 4.
:c_step4_build
echo [4/6] Building example programs ...
set E_CFLAGS=/nologo /O2 /arch:AVX2 /std:c11 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%SRC_DIR%" /I"%ROOT_DIR%"
set E_LIBS=%ZYDIS_LIB_ARG%

REM --- inference_example.exe ---
cl %E_CFLAGS% "%THIS_DIR%examples\c_inference\inference_example\inference_example.c" /Fe"%BIN_DIR%\inference_example.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] inference_example.exe FAILED & exit /b 1) else (echo OK: inference_example.exe)

REM --- dbg_inference.exe (pure ASM) ---
nasm -f win64 -O0 -w-other -I"%ASM_DIR%" "%THIS_DIR%examples\c_inference\dbg_inference\dbg_inference.asm" -o "%BIN_DIR%\dbg_inference.obj"
if !errorlevel! neq 0 (echo [FATAL] NASM dbg_inference FAILED & exit /b 1)
link /entry:WinMain /subsystem:console /NODEFAULTLIB "%BIN_DIR%\dbg_inference.obj" kernel32.lib user32.lib synchronization.lib /out:"%BIN_DIR%\dbg_inference.exe"
if !errorlevel! neq 0 (echo [FATAL] dbg_inference link FAILED & exit /b 1)
del "%BIN_DIR%\dbg_inference.obj" 2>nul
echo OK: dbg_inference.exe

REM --- dbg_image_recog.exe ---
cl %E_CFLAGS% "%THIS_DIR%examples\c_inference\dbg_image_recog\dbg_image_recog.c" /Fe"%BIN_DIR%\dbg_image_recog.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] dbg_image_recog.exe FAILED & exit /b 1) else (echo OK: dbg_image_recog.exe)

REM --- multi_turn_demo.exe ---
cl /nologo /O2 /arch:AVX2 /MD /std:c11 /DZYDIS_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%SRC_DIR%" /I"%ROOT_DIR%" "%THIS_DIR%examples\c_inference\multi_turn_demo\multi_turn_demo.c" /Fe"%BIN_DIR%\multi_turn_demo.exe" /link %ZYDIS_LIB_ARG% kernel32.lib
if !errorlevel! neq 0 (echo [FATAL] multi_turn_demo.exe FAILED & exit /b 1) else (echo OK: multi_turn_demo.exe)

REM --- v7_decide_example.exe ---
cl %E_CFLAGS% "%THIS_DIR%examples\c_inference\v7_decide_example\v7_decide_example.c" /Fe"%BIN_DIR%\v7_decide_example.exe" /link %E_LIBS% kernel32.lib /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] v7_decide_example.exe FAILED & exit /b 1) else (echo OK: v7_decide_example.exe)

REM --- spam_classifier.exe ---
cl %E_CFLAGS% "%THIS_DIR%examples\spam_classifier\spam_classifier.c" /Fe"%BIN_DIR%\spam_classifier.exe" /link %E_LIBS% kernel32.lib /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] spam_classifier.exe FAILED & exit /b 1) else (echo OK: spam_classifier.exe)

REM --- test programs (tools/ sources, bin/ executables -- tests are not examples) ---
cl %E_CFLAGS% "%TOOLS_DIR%\engine_death.c" /Fe"%BIN_DIR%\engine_death.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] engine_death.exe FAILED & exit /b 1) else (echo OK: engine_death.exe)

REM --- v7_only_safe_check.exe: compiled with /DV7_ONLY_SAFE so the BUILD itself
REM proves v7_lib.h hides the internal surface (an internal member referenced in
REM this file would be a compile error); the verify run proves the public tier
REM completes a real chat. ---
cl /nologo /O2 /arch:AVX2 /std:c11 /DV7_ONLY_SAFE /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" "%TOOLS_DIR%\v7_only_safe_check.c" /Fe"%BIN_DIR%\v7_only_safe_check.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] v7_only_safe_check.exe FAILED & exit /b 1) else (echo OK: v7_only_safe_check.exe)

cl %E_CFLAGS% "%TOOLS_DIR%\kv_cache.c" /Fe"%BIN_DIR%\kv_cache.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] kv_cache.exe FAILED & exit /b 1) else (echo OK: kv_cache.exe)

cl %E_CFLAGS% "%TOOLS_DIR%\chat_only.c" /Fe"%BIN_DIR%\chat_only.exe" /link %E_LIBS% kernel32.lib /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] chat_only.exe FAILED & exit /b 1) else (echo OK: chat_only.exe)

cl /nologo /O2 /arch:AVX2 /std:c11 /Fo"%BUILD_TMP%\\" /I"%SRC_DIR%" "%TOOLS_DIR%\generate_only.c" /Fe"%BIN_DIR%\generate_only.exe" /link kernel32.lib /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] generate_only.exe FAILED & exit /b 1) else (echo OK: generate_only.exe)

cl %E_CFLAGS% "%TOOLS_DIR%\multiturn.c" /Fe"%BIN_DIR%\multiturn.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] multiturn.exe FAILED & exit /b 1) else (echo OK: multiturn.exe)

cl %E_CFLAGS% "%TOOLS_DIR%\multiturn_manual.c" /Fe"%BIN_DIR%\multiturn_manual.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] multiturn_manual.exe FAILED & exit /b 1) else (echo OK: multiturn_manual.exe)

REM --- image_multiturn.exe (3-turn interleaved vision: chat_image -> generate_image -> chat_image) ---
cl %E_CFLAGS% "%TOOLS_DIR%\image_multiturn.c" /Fe"%BIN_DIR%\image_multiturn.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] image_multiturn.exe FAILED & exit /b 1) else (echo OK: image_multiturn.exe)

REM --- vit_bbox_eval.exe ---
cl %E_CFLAGS% "%THIS_DIR%examples\c_inference\vit_bbox_eval\vit_bbox_eval.c" /Fe"%BIN_DIR%\vit_bbox_eval.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] vit_bbox_eval.exe FAILED & exit /b 1) else (echo OK: vit_bbox_eval.exe)

REM --- v7_pipe.exe ---
cl %E_CFLAGS% "%THIS_DIR%examples\c_inference\v7_pipe\v7_pipe.c" /Fe"%BIN_DIR%\v7_pipe.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] v7_pipe.exe FAILED & exit /b 1) else (echo OK: v7_pipe.exe)

REM --- poem_continuation.exe (raw_generation=1 demo -- greedy poem continuation, no chat template) ---
cl %E_CFLAGS% "%THIS_DIR%examples\c_inference\poem_continuation\poem_continuation.c" /Fe"%BIN_DIR%\poem_continuation.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] poem_continuation.exe FAILED & exit /b 1) else (echo OK: poem_continuation.exe)

REM --- test_ardmark_rag.exe (3-turn ardmark RAG driver + golden-transcript demo-intactness check) ---
cl %E_CFLAGS% "%THIS_DIR%examples\c_inference\test_ardmark_rag\test_ardmark_rag.c" /Fe"%BIN_DIR%\test_ardmark_rag.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] test_ardmark_rag.exe FAILED & exit /b 1) else (echo OK: test_ardmark_rag.exe)

REM --- test_ardmark_seed.exe (minimal repro: primer-seed rewrite on the first generated token) ---
cl %E_CFLAGS% "%THIS_DIR%examples\c_inference\test_ardmark_seed\test_ardmark_seed.c" /Fe"%BIN_DIR%\test_ardmark_seed.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] test_ardmark_seed.exe FAILED & exit /b 1) else (echo OK: test_ardmark_seed.exe)

REM --- run_sectorlogo.exe ---
cl %E_CFLAGS% "%THIS_DIR%examples\sectorlogo\run_sectorlogo.c" /Fe"%BIN_DIR%\run_sectorlogo.exe" /link %E_LIBS% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] run_sectorlogo.exe FAILED & exit /b 1) else (echo OK: run_sectorlogo.exe)

REM --- echo_args.exe (argv echo helper for execute-fmt tests) ---
cl /nologo /O2 /std:c11 /Fo"%BUILD_TMP%\\" "%THIS_DIR%examples\sectorlogo\echo_args.c" /Fe"%BIN_DIR%\echo_args.exe" /link kernel32.lib /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] echo_args.exe FAILED & exit /b 1) else (echo OK: echo_args.exe)

REM --- test_v7_execute_fmt.exe ---
cl %E_CFLAGS% "%THIS_DIR%examples\sectorlogo\test_v7_execute_fmt.c" /Fe"%BIN_DIR%\test_v7_execute_fmt.exe" /link %E_LIBS% kernel32.lib /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] test_v7_execute_fmt.exe FAILED & exit /b 1) else (echo OK: test_v7_execute_fmt.exe)

REM --- c_callback_example.exe ---
cl %E_CFLAGS% "%THIS_DIR%examples\sectorlogo\c_callback_example.c" /Fe"%BIN_DIR%\c_callback_example.exe" /link %E_LIBS% "%BIN_DIR%\v7_threading.lib" /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] c_callback_example.exe FAILED & exit /b 1) else (echo OK: c_callback_example.exe)

REM --- bm25_sectorlogo_benchmark.exe (HOST FFI slot-timed pure-Logo BM25 profile) ---
cl %E_CFLAGS% "%THIS_DIR%examples\sectorlogo\bm25_sectorlogo_benchmark.c" /Fe"%BIN_DIR%\bm25_sectorlogo_benchmark.exe" /link %E_LIBS% "%BIN_DIR%\v7_threading.lib" /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] bm25_sectorlogo_benchmark.exe FAILED & exit /b 1) else (echo OK: bm25_sectorlogo_benchmark.exe)

REM --- llm_text_processor.exe (C++ -- v7-LLM text reformulation: BM25 retrieval + LLM rewriting) ---
cl /nologo /O2 /arch:AVX2 /EHsc /std:c++17 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%SRC_DIR%" /I"%ROOT_DIR%" "%THIS_DIR%examples\rag\llm_text_processor.cpp" /Fe"%BIN_DIR%\llm_text_processor.exe" /link %ZYDIS_LIB_ARG% /subsystem:console
if !errorlevel! neq 0 (echo [FATAL] llm_text_processor.exe FAILED & exit /b 1) else (echo OK: llm_text_processor.exe)

echo  OK: All examples built
echo.
:c_step4_done

REM ====== Step 5/6: Build v7_server.exe ======
set C_SKIP=0
if defined C_FORCE goto c_step5_build
call :c_fresh_step 5
if !C_SKIP! equ 1 goto c_step5_done
echo  [C/C++] Step 5 freshness inconclusive -- rebuilding step 5.
:c_step5_build
echo [5/6] Building v7_server.exe ...
set S_CFLAGS=/nologo /O2 /arch:AVX2 /MD /std:c++17 /DNDEBUG /fp:fast /GS- /GL- /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /D_CRT_SECURE_NO_WARNINGS /D_CRT_NONSTDC_NO_WARNINGS /EHsc /Fo"%BUILD_TMP%\\"
cl %S_CFLAGS% "%THIS_DIR%examples\server\v7_server_main.cpp" "%THIS_DIR%examples\server\v7_server_json.cpp" "%THIS_DIR%examples\server\v7_server_util.cpp" "%THIS_DIR%examples\server\v7_server_http.cpp" "%THIS_DIR%examples\server\v7_server_turn.cpp" "%THIS_DIR%examples\server\v7_server_image.cpp" "%THIS_DIR%examples\server\v7_server_health.cpp" "%THIS_DIR%examples\server\v7_server_api_meta.cpp" "%THIS_DIR%examples\server\v7_server_api_chat.cpp" "%THIS_DIR%examples\server\v7_server_api_vision.cpp" "%THIS_DIR%examples\server\v7_server_sidecar.cpp" "%THIS_DIR%examples\server\v7_server_web_render.cpp" "%THIS_DIR%examples\server\v7_server_web_handlers.cpp" "%THIS_DIR%examples\server\v7_server_router.cpp" /Fe"%BIN_DIR%\v7_server.exe" ws2_32.lib gdiplus.lib ole32.lib %ZYDIS_LIB_ARG% /I"%SRC_DIR%" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" /I"%ROOT_DIR%"
if !errorlevel! neq 0 (
    echo [FATAL] v7_server.exe FAILED
    exit /b 1
)
echo  OK: v7_server.exe
echo.
:c_step5_done

REM ====== Step 6/6: Build bayes_classify.exe ======
set C_SKIP=0
if defined C_FORCE goto c_step6_build
call :c_fresh_step 6
if !C_SKIP! equ 1 goto c_step6_done
echo  [C/C++] Step 6 freshness inconclusive -- rebuilding step 6.
:c_step6_build
echo [6/6] Building bayes_classify.exe ...
cl /nologo /O2 /EHsc /DZYDIS_STATIC_BUILD /Fo"%BUILD_TMP%\\" /I"%ZYDIS_INC%" /I"%ZYCORE_INC%" "%THIS_DIR%examples\rag\bayes_classify.cpp" /Fe"%BIN_DIR%\bayes_classify.exe" /link %ZYDIS_LIB_ARG% /subsystem:console
if !errorlevel! neq 0 (
    echo [FATAL] bayes_classify.exe FAILED
    exit /b 1
)
echo  OK: bayes_classify.exe
echo.
:c_step6_done

goto build_summary

REM --- c_fresh_step <tokens> -- run the freshness checker for one build step.
REM Sets C_SKIP=1 when that step's declared outputs are provably fresh; any
REM STALE or inconclusive answer leaves C_SKIP=0 (the caller rebuilds the step).
:c_fresh_step
set C_SKIP=0
python "%SRC_DIR%\check_stale.py" --root "%ROOT_DIR%" --manifest "%SRC_DIR%\build_c_targets.txt" --step %1
if !errorlevel! equ 0 (
    echo  [C/C++] Step %1 outputs are up to date -- skipping.
    set C_SKIP=1
)
goto :eof

:build_summary
echo ============================================================================
echo  BUILD SUCCEEDED
echo  DLL:       bin\v7_threading.dll
echo  Tools:     bin\build_db.exe
echo  Validators: tools\*.exe
echo  Examples:  bin\*.exe
echo ============================================================================
REM Explicit success status: verify.bat tests `errorlevel neq 0` after calling this
REM script, so the exit code must be 0 and not the status of the last echo.
endlocal & exit /b 0
