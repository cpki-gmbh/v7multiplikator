@echo off
setlocal EnableDelayedExpansion
REM run_tuner_proof_all.bat -- tuner proof for ALL text models, one dumb
REM line per model (no loops): 8 chats x 200 tokens each, then the harness
REM verifies the v7_tuning_journal (whole 16-byte records = at least one
REM accepted change journaled). The tuner decision traces (TunerPrint
REM "candidate rejected" per verdict) land in each model's log under
REM build_logs\. The trailing findstr lines print the decision evidence per
REM model (count of accept lines per log; accepts additionally banner via
REM "TUNER - FOUND A FASTER CONFIG").
REM Requires: build.bat (bin\tuner_proof.exe) + all databases rebuilt.

if not exist build_logs mkdir build_logs

set FAILED=0

bin\tuner_proof.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db 8 200 > build_logs\tuner_proof_qwen3_0.6b.log 2>&1
set PROOF_RC=!errorlevel!
echo qwen3_0.6b exit=!PROOF_RC!
if not "!PROOF_RC!"=="0" set FAILED=1
bin\tuner_proof.exe bin\v7_threading.dll weights\databases\qwen3_1.7b\lz4ht.db 8 200 > build_logs\tuner_proof_qwen3_1.7b.log 2>&1
set PROOF_RC=!errorlevel!
echo qwen3_1.7b exit=!PROOF_RC!
if not "!PROOF_RC!"=="0" set FAILED=1
bin\tuner_proof.exe bin\v7_threading.dll weights\databases\qwen2.5_0.5b\lz4ht.db 8 200 > build_logs\tuner_proof_qwen2.5_0.5b.log 2>&1
set PROOF_RC=!errorlevel!
echo qwen2.5_0.5b exit=!PROOF_RC!
if not "!PROOF_RC!"=="0" set FAILED=1
bin\tuner_proof.exe bin\v7_threading.dll weights\databases\llama_3.2_1b\lz4ht.db 8 200 > build_logs\tuner_proof_llama_3.2_1b.log 2>&1
set PROOF_RC=!errorlevel!
echo llama_3.2_1b exit=!PROOF_RC!
if not "!PROOF_RC!"=="0" set FAILED=1
bin\tuner_proof.exe bin\v7_threading.dll weights\databases\smollm2_135m\lz4ht.db 8 200 > build_logs\tuner_proof_smollm2_135m.log 2>&1
set PROOF_RC=!errorlevel!
echo smollm2_135m exit=!PROOF_RC!
if not "!PROOF_RC!"=="0" set FAILED=1
bin\tuner_proof.exe bin\v7_threading.dll weights\databases\smollm2_360m\lz4ht.db 8 200 > build_logs\tuner_proof_smollm2_360m.log 2>&1
set PROOF_RC=!errorlevel!
echo smollm2_360m exit=!PROOF_RC!
if not "!PROOF_RC!"=="0" set FAILED=1
bin\tuner_proof.exe bin\v7_threading.dll weights\databases\tinyllama_1.1b\lz4ht.db 8 200 > build_logs\tuner_proof_tinyllama_1.1b.log 2>&1
set PROOF_RC=!errorlevel!
echo tinyllama_1.1b exit=!PROOF_RC!
if not "!PROOF_RC!"=="0" set FAILED=1
bin\tuner_proof.exe bin\v7_threading.dll weights\databases\phi3_mini\lz4ht.db 8 200 > build_logs\tuner_proof_phi3_mini.log 2>&1
set PROOF_RC=!errorlevel!
echo phi3_mini exit=!PROOF_RC!
if not "!PROOF_RC!"=="0" set FAILED=1
bin\tuner_proof.exe bin\v7_threading.dll weights\databases\minicpm5_1b\lz4ht.db 8 200 > build_logs\tuner_proof_minicpm5_1b.log 2>&1
set PROOF_RC=!errorlevel!
echo minicpm5_1b exit=!PROOF_RC!
if not "!PROOF_RC!"=="0" set FAILED=1

echo ---- accept evidence per model (count of "tuner: accepted for" lines) ----
findstr /c:"tuner: accepted for" build_logs\tuner_proof_qwen3_0.6b.log | find /c /v ""
findstr /c:"tuner: accepted for" build_logs\tuner_proof_qwen3_1.7b.log | find /c /v ""
findstr /c:"tuner: accepted for" build_logs\tuner_proof_qwen2.5_0.5b.log | find /c /v ""
findstr /c:"tuner: accepted for" build_logs\tuner_proof_llama_3.2_1b.log | find /c /v ""
findstr /c:"tuner: accepted for" build_logs\tuner_proof_smollm2_135m.log | find /c /v ""
findstr /c:"tuner: accepted for" build_logs\tuner_proof_smollm2_360m.log | find /c /v ""
findstr /c:"tuner: accepted for" build_logs\tuner_proof_tinyllama_1.1b.log | find /c /v ""
findstr /c:"tuner: accepted for" build_logs\tuner_proof_phi3_mini.log | find /c /v ""
findstr /c:"tuner: accepted for" build_logs\tuner_proof_minicpm5_1b.log | find /c /v ""

if not "!FAILED!"=="0" (
    echo [FAIL] at least one tuner proof exited nonzero
    exit /b 1
)
exit /b 0
