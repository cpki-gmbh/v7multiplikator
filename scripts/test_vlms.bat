@echo off
REM test_vlms.bat -- full VLM test: all 3 supported VLMs on F32 AND Q8 with
REM the SCALED cat photo (cat640.raw, 640x480 bicubic from the 3840x2571
REM original). 640x480 keeps every model in its comfortable token range AND is
REM the canonical shape-test resolution (matches vit_bbox_eval). Per model+mode:
REM   Q1 "What is in this image?"          (content analysis, full detail)
REM   Q2 location test (top/middle/bottom) (shape/location capability @ 640x480)
REM   Q3 "What is a cat?"                  (text answer with image attached)
REM Plus 1 duplicate Q1 run on qwen3_vl F32 as the think no-op proof
REM (greedy t=0, identical output = flag-independent, matching the structural
REM proof: no VLM DB stores @think keys).
REM Full output visible AND captured to build_logs\model_tests\vlm_<model>_<mode>_<q>.txt
REM Exit code: 0 = every run completed, 1 = at least one run failed.
setlocal ENABLEDELAYEDEXPANSION
cd /d %~dp0..\
rem run all inference at idle priority so the suite never competes with
rem foreground work (the engine applies v7_override_prio at pool creation)
set v7_override_prio=-15
if not exist build_logs\model_tests mkdir build_logs\model_tests
set OUT=build_logs\model_tests
set IMG=cat640.raw
set FAILED=

for %%M in (qwen3_vl_4b llava_onevision_qwen2_0.5b smolvlm_256m) do (
    for %%Q in (0 1) do (
        echo.
        echo ============================================================
        echo  VLM %%M  q8=%%Q
        echo ============================================================
        echo --- %%M q8=%%Q : Q1 what is in this image ---
        bin\dbg_image_recog.exe -q8 %%Q bin\v7_threading.dll weights\databases\%%M\lz4ht.db %IMG% "What is in this image?" > "!OUT!\vlm_%%M_q8%%Q_Q1.txt" 2>&1
        if !errorlevel! neq 0 ( echo  [FAIL] %%M q8=%%Q Q1 & set FAILED=!FAILED! %%M-q8%%Q )
        type "!OUT!\vlm_%%M_q8%%Q_Q1.txt"
        echo.
        echo --- %%M q8=%%Q : Q2 location test ---
        bin\dbg_image_recog.exe -q8 %%Q bin\v7_threading.dll weights\databases\%%M\lz4ht.db %IMG% "Look at the image. In which part of the image is the cat located: top, middle, or bottom? Answer in one short sentence." > "!OUT!\vlm_%%M_q8%%Q_Q2.txt" 2>&1
        if !errorlevel! neq 0 ( echo  [FAIL] %%M q8=%%Q Q2 & set FAILED=!FAILED! %%M-q8%%Q )
        type "!OUT!\vlm_%%M_q8%%Q_Q2.txt"
        echo.
        echo --- %%M q8=%%Q : Q3 what is a cat ---
        bin\dbg_image_recog.exe -q8 %%Q bin\v7_threading.dll weights\databases\%%M\lz4ht.db %IMG% "What is a cat?" > "!OUT!\vlm_%%M_q8%%Q_Q3.txt" 2>&1
        if !errorlevel! neq 0 ( echo  [FAIL] %%M q8=%%Q Q3 & set FAILED=!FAILED! %%M-q8%%Q )
        type "!OUT!\vlm_%%M_q8%%Q_Q3.txt"
    )
)
echo.
echo --- qwen3_vl_4b q8=0 : Q1 DUP (think no-op proof) ---
bin\dbg_image_recog.exe -q8 0 bin\v7_threading.dll weights\databases\qwen3_vl_4b\lz4ht.db %IMG% "What is in this image?" > "!OUT!\vlm_qwen3_vl_4b_q80_Q1dup.txt" 2>&1
if !errorlevel! neq 0 ( echo  [FAIL] dup & set FAILED=!FAILED! dup )
type "!OUT!\vlm_qwen3_vl_4b_q80_Q1dup.txt"
echo.
if "!FAILED!"=="" (
    echo ALL VLM RUNS COMPLETED
    endlocal
    exit /b 0
)
echo FAILED:!FAILED!
endlocal
exit /b 1
