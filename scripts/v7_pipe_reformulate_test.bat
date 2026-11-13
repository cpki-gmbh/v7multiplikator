@echo off
setlocal
set THIS_DIR=%~dp0..\
set V7_DLL=%THIS_DIR%bin\v7_threading.dll
set V7_DB=%THIS_DIR%weights\databases\qwen3_0.6b\lz4ht.db

rem v7_pipe reformulation feasibility test: pass single " -- " comment samples
rem through the sandwich prompt; the reformulation must drop the " -- " and
rem only describe behaviour. Comment samples are inline below.

echo test 1: abicheck_macro.inc:85 : no -- save area was overwritten
"%THIS_DIR%bin\v7_pipe.exe" ask "reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\" now the comment:>>>no -- save area was overwritten<<<reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\""
echo.

echo test 2: aes_string_hash.inc:30 : k= 0 (never indexed -- full block)
"%THIS_DIR%bin\v7_pipe.exe" ask "reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\" now the comment:>>>k= 0 (never indexed -- full block)<<<reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\""
echo.

echo test 3: add_kernel.inc:39 : two groups consumed -- r8d is now the REMAINING group count
"%THIS_DIR%bin\v7_pipe.exe" ask "reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\" now the comment:>>>two groups consumed -- r8d is now the REMAINING group count<<<reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\""
echo.

echo test 4: attention_kernel.inc:114 : score *= scale (hoisted in xmm6 -- no load-port uop per timestep)
"%THIS_DIR%bin\v7_pipe.exe" ask "reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\" now the comment:>>>score *= scale (hoisted in xmm6 -- no load-port uop per timestep)<<<reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\""
echo.

echo test 5: attention_dispatch.inc:36 : mode (AFTER JOB_ADDR -- the 1-arg form uses r11 as scratch)
"%THIS_DIR%bin\v7_pipe.exe" ask "reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\" now the comment:>>>mode (AFTER JOB_ADDR -- the 1-arg form uses r11 as scratch)<<<reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\""
echo.
