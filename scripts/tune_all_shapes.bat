
@echo off
REM tune_all_shapes.bat -- continuous two-phase re-tuning loop.
REM Phase A: Q8 matmul, one autotune line per shape in threading_proxy.ini's
REM supported_resolutions + supported_resolutions_2 (88 unique shapes, INI
REM order). Phase B: Q8 SwiGLU for the actual FFN gate/up expansion shapes
REM (dim:intermediate pairs of the supported models -- NOT vocabulary
REM projections or down-projections; the swiglu benchmark allocates an NxN
REM activation, so vocab-sized shapes are excluded by design). NO loops, NO
REM variables, NO result checks: just the calls. The trailing %0 restarts the
REM bat forever BY DESIGN: this is the continuous re-tuning loop (run with a
REM timeout budget).
REM NOTE: phases A and B write the SAME 22 dispatcher keys per [NxD] section
REM (they are dispatcher geometry, not kernel-specific); the prefetch keys are
REM kernel-owned and never overlap (q8: 0-7, swiglu_q8: 27-38). Each pass
REM re-optimizes the shared dispatcher keys for its own kernel.

tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:512 --ini threading_proxy.ini --runtime 5 --rounds 100 --probe 100
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:1024 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:1024 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:3072 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 3072:1024 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:4096 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 151936:1024 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:151936 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 4096:1024 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:512 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:8192 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 8192:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 576:576 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 576:192 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 576:1536 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1536:576 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 960:960 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 960:320 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 960:2560 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2560:960 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:256 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:5632 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 5632:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 3072:3072 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 3072:8192 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 8192:3072 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1536:3072 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1536:256 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1536:4608 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 4608:1536 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 896:896 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 896:128 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 896:4864 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 4864:896 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 12345:67890 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 64:256 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:32 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 8192:65536 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1536:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:1536 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1536:130560 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 130560:1536 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1536:2560 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2560:1536 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:4096 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 896:1152 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 576:960 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 960:1600 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:2560 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 3072:9216 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:3072 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:6144 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 6144:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 151936:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 128:896 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 151936:896 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 512:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 128256:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 49152:576 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 49152:960 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 256:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 32000:2048 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 32064:3072 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 256:1536 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:151936 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 896:151936 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:128256 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 576:49152 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 960:49152 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2048:32000 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 3072:32064 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2560:9728 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2560:151936 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 896:152000 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 576:49280 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2560:4096 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2560:1024 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 4096:2560 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 4096:1024 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1536:1024 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2560:2560 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:2560 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 9728:2560 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999

tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:1536 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 2560:6144 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 6144:2560 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999

REM ---- Phase B: Q8 SwiGLU, actual FFN gate/up expansion shapes (dim:intermediate) ----
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --kernel swiglu_q8 1024:3072 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --kernel swiglu_q8 576:1536 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --kernel swiglu_q8 896:4864 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --kernel swiglu_q8 1536:4608 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --kernel swiglu_q8 2048:5632 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --kernel swiglu_q8 2048:6144 --ini threading_proxy.ini --runtime 5 --rounds 999 --probe 999
%0