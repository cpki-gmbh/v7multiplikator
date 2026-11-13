@echo off
REM ============================================================================
REM sync_release.bat -- copy the FULL dev tree into the release directory.
REM
REM CONTENT RULE: if a file exists in dev under src\, docs\, tools\, or
REM examples\, it MUST exist in the release. Dev dictates the content --
REM this script never selects or filters. New dev files automatically go in.
REM
REM robocopy is FORBIDDEN -- this script uses ONLY xcopy / copy (standard
REM Windows functionality). The copy is ADDITIVE: release-only files
REM (examples\v7_server.c, ...) are preserved. This script
REM NEVER deletes anything -- files deleted in dev are removed in the release
REM BY HAND, one at a time, never by an automated deletion mechanism.
REM
REM This script lives in dev ONLY (it references dev-side absolute paths).
REM ============================================================================

set DEV=C:\project\v7
set REL=C:\project\v7_release

echo === sync_release: full dev-to-release copy ===

echo --- src (asm, macros, logo, headers, def, checkers) ---
xcopy "%DEV%\src" "%REL%\src\" /E /Y /I

echo --- docs (full documentation tree) ---
xcopy "%DEV%\docs" "%REL%\docs\" /E /Y /I

echo --- packaging exclusions (redundant snapshots, never ship) ---
del /Q "%REL%\src\asm.zip"
del /Q "%REL%\src\neuasm.zip"
del /Q "%REL%\src\qqqasm.zip"
del /Q "%REL%\src\xxxasm.zip"
del /Q "%REL%\src\inc.txt"
del /Q "%REL%\src\asm\public_api.inc.before_wrapped_rewind"
del /Q "%REL%\src\asm\threading_problem_asm.zip"
del /Q "%REL%\docs\docs.zip"
del /Q "%REL%\dashdash_comments.txt"

echo --- internal dev records (agent handbooks of state, never ship) ---
del /Q "%REL%\docs\bug_log.md"
del /Q "%REL%\docs\working_and_not_working_rag_improvements.md"
del /Q "%REL%\docs\qwen3_thinking_phrases.md"
del /Q "%REL%\docs\qwen_thought_control_strings.md"
del /Q "%REL%\docs\dispatcher_problem.md"
del /Q "%REL%\docs\fatal_clamping_and_bugs_changes.md"
del /Q "%REL%\docs\interface_contract.md"
del /Q "%REL%\docs\verification.md"
del /Q "%REL%\docs\how_to_make_a_release.md"
del /Q "%REL%\docs\asm_inprocess_cycle_tuner_plan.md"
del /Q "%REL%\docs\continous_adaption_plan.md"
del /Q "%REL%\docs\fix_boolean_operations_plan.md"
del /Q "%REL%\docs\pool_statistics_implementation_plan.md"
del /Q "%REL%\docs\v7linux_integration_plan.md"

echo --- internal sales aids (Faktencheckliste + 4/6-pager, never ship) ---
del /Q "%REL%\docs\engine_4pager.md"
del /Q "%REL%\docs\engine_6pager.md"
del /Q "%REL%\docs\engine_6pager.html"

echo --- tools (validators, benchmarks, DB tools, python, tool sources) ---
xcopy "%DEV%\tools" "%REL%\tools\" /E /Y /I

echo --- examples (example sources; additive, v7_server.* preserved) ---
xcopy "%DEV%\examples" "%REL%\examples\" /E /Y /I

echo --- scripts (EVERY root bat moved into scripts\; a stale copy would fail the identity proof) ---
xcopy "%DEV%\scripts" "%REL%\scripts\" /E /Y /I

REM dev-only launchers and stale audit docs: deleted AFTER the scripts xcopy
REM above, so a dev tree that still carries them cannot reintroduce them.
del /Q "%REL%\scripts\start_watch_opt_and_bug.bat"
del /Q "%REL%\scripts\run_hill_climb_tuner_params.bat"
del /Q "%REL%\scripts\run_partner_confirm.bat"
del /Q "%REL%\scripts\run_partner_half.bat"
del /Q "%REL%\scripts\run_partner_sweep.bat"
del /Q "%REL%\scripts\run_transplant_ini.bat"
del /Q "%REL%\docs\RELEASE_SOURCE_AUDIT.md"

echo --- shared root files (Makefile, CI, INI, stdlib) ---
copy /Y "%DEV%\Makefile" "%REL%\Makefile"
copy /Y "%DEV%\.gitlab-ci.yml" "%REL%\.gitlab-ci.yml"
copy /Y "%DEV%\threading_proxy.ini" "%REL%\threading_proxy.ini"
copy /Y "%DEV%\autoexec.logo" "%REL%\autoexec.logo"
copy /Y "%DEV%\THIRD_PARTY.md" "%REL%\THIRD_PARTY.md"
copy /Y "%DEV%\FILE_ID.DIZ" "%REL%\FILE_ID.DIZ"
copy /Y "%DEV%\.gitattributes" "%REL%\.gitattributes"
copy /Y "%DEV%\.gitignore" "%REL%\.gitignore"

echo --- backgrounds (server demo backgrounds) ---
xcopy "%DEV%\backgrounds" "%REL%\backgrounds\" /E /Y /I

echo --- canonical vision test image (referenced by docs\tools.md, docs\examples.md) ---
copy /Y "%DEV%\cat640.raw" "%REL%\cat640.raw"
copy /Y "%DEV%\cat640.meta" "%REL%\cat640.meta"

echo --- vision gate fixtures + baselines (Sec.122 Qwen3-VL smoke + LLaVA anyres gate) ---
if not exist "%REL%\vision" mkdir "%REL%\vision"
copy /Y "%DEV%\vision\vision_baseline.txt" "%REL%\vision\vision_baseline.txt"
copy /Y "%DEV%\vision\anyres_baselines_readme.txt" "%REL%\vision\anyres_baselines_readme.txt"
copy /Y "%DEV%\vision\vision_baseline_llava_*.txt" "%REL%\vision\"
copy /Y "%DEV%\vision\cat_anyres_wide.raw" "%REL%\vision\cat_anyres_wide.raw"
copy /Y "%DEV%\vision\cat_anyres_wide.meta" "%REL%\vision\cat_anyres_wide.meta"
copy /Y "%DEV%\vision\cat_anyres_tall.raw" "%REL%\vision\cat_anyres_tall.raw"
copy /Y "%DEV%\vision\cat_anyres_tall.meta" "%REL%\vision\cat_anyres_tall.meta"
REM The release root keeps its own cut-down AGENTS.md (release-specific,
REM never overwritten); the full dev handbook ships as docs\AGENTS_dev.md
REM (+ docs\ABOUT_AGENTS_dev.md explaining what it is) via the docs copy
REM above.

REM The release ships WITHOUT Zydis (Zydis.lib, zydis-master\, zycore-c-master\
REM are never copied and must not exist in the release). The release root
REM zydis_debug_header.h is OVERWRITTEN with the no-op stub so every tool still
REM compiles and links; a release-built tool then crashes without a crash dump.
REM Dev always builds the real decoder -- see docs\zydis_debug_readme.txt.
echo --- zydis stub (release builds without Zydis; dev keeps the real decoder) ---
copy /Y "%DEV%\tools\zydis_stub\zydis_debug_header.h" "%REL%\zydis_debug_header.h"

REM sectorlogo_c is NOT copied: it is the test-only C++ differential port
REM (AGENTS.md Sec.197/199) and stays out of every release package. Reports
REM asking to include it are ignored per those sections.
REM
REM scheme_internal\ (repo root) is NOT copied either: the Scheme-to-SectorLogo
REM compiler work needs fixing and is dev-only (AGENTS.md Sec.199); the release
REM tree has no scheme content. The multilang benchmark under examples\ carries
REM no scheme benches (removed with the move).

echo --- changelog (maintained IN DEV, copied verbatim -- retitle [Unreleased] to [X.Y.Z] in the release) ---
copy /Y "%DEV%\CHANGELOG.md" "%REL%\CHANGELOG.md"

echo.
echo === sync_release done ===
echo Next: in %REL% run "git status --short" --
echo   ?? files are NEW dev files: git add them (they MUST be in the release)
echo   files deleted in dev: delete them in the release manually
echo   (except release-specific files -- see docs\how_to_make_a_release.md)
echo.
echo NOTE: Old build bats (build_v7.bat, build_tools.bat, build_all.bat,
echo   build_examples.bat, build_validators.bat) and tool_src\ were removed
echo   from dev. Delete them in the release BY HAND if they still exist.

REM --- release hygiene gate (review P1-8): zero tolerance, run against the
REM     freshly synced release tree so drift cannot survive the sync ---
python "%DEV%\src\check_release.py" "%REL%"
if errorlevel 1 (
    echo [FATAL] release hygiene check FAILED -- fix every finding before committing.
    exit /b 1
)
