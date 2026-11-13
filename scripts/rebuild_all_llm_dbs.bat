@echo off
REM rebuild_all_llm_dbs.bat -- Re-export + rebuild ALL supported LLM databases
REM with the fixed toolchain (export.py qkv_bias fix 3b19678, export_vlm.py
REM lm_head fix 9ec3674). Skips smollm2_* (rebuilt 14.08) and smolvlm_256m
REM (rebuilt with lm_head fix today). Sequential, fully visible output.

setlocal ENABLEDELAYEDEXPANSION
cd /d %~dp0..\
set FAILED=

echo.
echo ============================================================
echo  [1/6] llama_3.2_1b
echo ============================================================
python tools\export.py weights\llama_3.2_1b\llama_3.2_1b.bin weights\llama_3.2_1b
if !errorlevel! neq 0 (set FAILED=!FAILED! llama_3.2_1b-export) else (
    if not exist weights\databases\llama_3.2_1b mkdir weights\databases\llama_3.2_1b
    bin\build_db.exe weights\llama_3.2_1b\llama_3.2_1b.bin -thinking 0 -o weights\databases\llama_3.2_1b\lz4ht.db -f
    if !errorlevel! neq 0 (set FAILED=!FAILED! llama_3.2_1b-db) else echo OK: llama_3.2_1b
)

echo.
echo ============================================================
echo  [2/6] phi3_mini
echo ============================================================
python tools\export.py weights\phi3_mini\phi3_mini.bin weights\phi3_mini
if !errorlevel! neq 0 (set FAILED=!FAILED! phi3_mini-export) else (
    if not exist weights\databases\phi3_mini mkdir weights\databases\phi3_mini
    bin\build_db.exe weights\phi3_mini\phi3_mini.bin -thinking 0 -rope-base 10000 -o weights\databases\phi3_mini\lz4ht.db -f
    if !errorlevel! neq 0 (set FAILED=!FAILED! phi3_mini-db) else echo OK: phi3_mini
)

echo.
echo ============================================================
echo  [3/6] minicpm5_1b
echo ============================================================
python tools\export.py weights\minicpm5_1b\minicpm5_1b.bin weights\minicpm5_1b
if !errorlevel! neq 0 (set FAILED=!FAILED! minicpm5_1b-export) else (
    if not exist weights\databases\minicpm5_1b mkdir weights\databases\minicpm5_1b
    bin\build_db.exe weights\minicpm5_1b\minicpm5_1b.bin -thinking 0 -rope-base 5000000 -o weights\databases\minicpm5_1b\lz4ht.db -f
    if !errorlevel! neq 0 (set FAILED=!FAILED! minicpm5_1b-db) else echo OK: minicpm5_1b
)

echo.
echo ============================================================
echo  [4/6] qwen2.5_0.5b
echo ============================================================
python tools\export.py weights\qwen2.5_0.5b\qwen2.5_0.5b.bin weights\qwen2.5_0.5b
if !errorlevel! neq 0 (set FAILED=!FAILED! qwen2.5_0.5b-export) else (
    if not exist weights\databases\qwen2.5_0.5b mkdir weights\databases\qwen2.5_0.5b
    bin\build_db.exe weights\qwen2.5_0.5b\qwen2.5_0.5b.bin -o weights\databases\qwen2.5_0.5b\lz4ht.db -f
    if !errorlevel! neq 0 (set FAILED=!FAILED! qwen2.5_0.5b-db) else echo OK: qwen2.5_0.5b
)

echo.
echo ============================================================
echo  [5/6] qwen3_0.6b
echo ============================================================
python tools\export.py weights\qwen3_0.6b\qwen3_0.6b.bin weights\qwen3_0.6b
if !errorlevel! neq 0 (set FAILED=!FAILED! qwen3_0.6b-export) else (
    if not exist weights\databases\qwen3_0.6b mkdir weights\databases\qwen3_0.6b
    bin\build_db.exe weights\qwen3_0.6b\qwen3_0.6b.bin -o weights\databases\qwen3_0.6b\lz4ht.db -f
    if !errorlevel! neq 0 (set FAILED=!FAILED! qwen3_0.6b-db) else echo OK: qwen3_0.6b
)

echo.
echo ============================================================
echo  [6/6] tinyllama_1.1b
echo ============================================================
python tools\export.py weights\tinyllama_1.1b\tinyllama_1.1b.bin weights\tinyllama_1.1b\hf_source
if !errorlevel! neq 0 (set FAILED=!FAILED! tinyllama_1.1b-export) else (
    if not exist weights\databases\tinyllama_1.1b mkdir weights\databases\tinyllama_1.1b
    bin\build_db.exe weights\tinyllama_1.1b\tinyllama_1.1b.bin -thinking 0 -rope-base 10000 -o weights\databases\tinyllama_1.1b\lz4ht.db -f
    if !errorlevel! neq 0 (set FAILED=!FAILED! tinyllama_1.1b-db) else echo OK: tinyllama_1.1b
)

echo.
echo ============================================================
if "!FAILED!"=="" (
    echo  ALL DATABASES REBUILT SUCCESSFULLY
) else (
    echo  FAILED:!FAILED!
)
echo ============================================================
endlocal
