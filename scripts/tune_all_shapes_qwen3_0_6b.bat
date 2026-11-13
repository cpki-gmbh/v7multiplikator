@echo off
REM tune_all_shapes_qwen3_0_6b.bat -- tunes every dispatch shape + every q8
REM kernel the Qwen3-0.6B model dispatches. Analysis of threading_proxy.ini
REM (best_gbs_q8 records): all matmul shapes are already tuned EXCEPT the FFN
REM down-projection 3072x1024, and the swiglu_q8 kernel (FFN gate/up, slots
REM 27-38) has NEVER been tuned for 1024x3072. This bat closes exactly those
REM gaps; the already-tuned matmul shapes are skipped (they are covered by
REM tune_all_shapes.bat's continuous loop). The trailing %0 restarts forever
REM BY DESIGN (continuous loop; run with a timeout budget). Rebuild the DB
REM afterwards so the DLL loads the new values.

tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --ini threading_proxy.ini --q8 3072:1024 --probe 500 --rounds 500 --runtime 3
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --ini threading_proxy.ini --kernel swiglu_q8 1024:3072 --rounds 500 --runtime 3
%0
