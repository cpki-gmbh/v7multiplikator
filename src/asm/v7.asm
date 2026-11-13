; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; v7.asm -- Master include file for v7_threading.dll
; Includes all component files in the correct order.
; Can be included directly by v7_threading.asm for the main DLL build,
; or used independently for component-level testing.
; RULE: Every single assembler instruction MUST have a comment -- this is enforced.

; Include order, must NOT change :
; 1. Macro/support files, v7_core, v7_apicall, v7_macros, math_data,
; smc_macros, trace_macro, zydis_macros -- everything
; downstream depends on their macros/defines.
; 2. v7_smc.inc -- kernel_bases .rdata table, self-opens/closes sections .
; 3. v7_kernels.inc -- the whole kernel block; v7s_q8_chunk ..
; v7_kernels_end bracket the SMC VirtualProtect region, see the header
; comment inside that file for the ordering rules .
; 4. v7_data .. v7_instr, data, debug, landing, cpu gate, dispatch files,
; v7_init, v7_worker, set_worker_priority, v7_instr .
; 5. string_hash .. file_backed_mem, hash, microdb, encode, forward, mrope,
; prefill_q8, sampler, sprintf, chat, generate, chat_image, sectorlogo,
; execute, token_stream, vit input/bicubic/forward, dumps, fbm .
; 6. PREFETCH_PATCH_BEGIN / PREFETCH_ROW table / PREFETCH_PATCH_END -- the
; SMC slot table and backup buffer.
; 7. DllMain, then inference.inc, kv_cache_file.inc, these MUST stay after
; PREFETCH_PATCH_END so they do not shift SMC/PREFETCH label offsets --
; see the comments at the include sites, public_api.inc, version.inc,
; console_width.inc.

; After includes: DllMain -- Windows DLL entry point
; ============================================================================

     default rel                               ; generate RIP-relative addressing by default

%use smartalign                                ; Enable multi-byte NOP generation for alignment padding
     alignmode generic                         ; Use Intel-blessed multi-byte NOP patterns, not single 0x90

; The register fork/join machinery is unconditionally compiled: there is no
; switch that disables it and no sequential fallback paths. Every fork user
; (sampler, Q/K heads, attention split, vision GELU, batch-4 prefill) below
; is the only implementation of its stage.

%include "v7_core.inc"

; Shared macro files, same directory as this file
%include "v7_apicall.inc"
%include "v7_macros.inc"
%include "fork.inc"                            ; Recursive register fork/join macros, worker slot layout
%include "math_data.inc"
%include "smc_macros.inc"
%include "trace_macro.inc"

; Zydis INT3 magic debug macros -- always included, macros are no-ops when
; ZYDIS_TRACE is not defined. Enable tracing by building with:
; nasm -f win64 -DZYDIS_TRACE v7_threading.asm ...
; %define ZYDIS_TRACE ; uncomment for INT3 magic tracing
%include "zydis_macros.inc"

%include "v7_smc.inc"

%include "v7_kernels.inc"
%include "q8_batch4_kernel.inc"                ; Q8 batch-4 kernel, a mandatory dispatch stage

%include "v7_data.inc"

%include "config_set_random_probe.inc"         ; random-probe config filler for matmul shape blocks, implementation only, no callers yet

; StreamingLLM config storage -- constants must precede forward.inc/chat.inc
; NASM equ resolves in pass 1: remove_* are used by both
%include "stream_cfg.inc"

%include "v7_debug.inc"
%include "v7_do_fatal.inc"
%include "engine_landing.inc"
%include "cpu_gate.inc"
%include "dispatch_execute.inc"
%include "matmul_dispatch.inc"
%include "fused_dispatch.inc"
%include "attention_fork.inc"                  ; single-helper head split, independent of full job dispatcher
%include "attention_dispatch.inc"
%include "util_export.inc"
%include "utility_dispatch.inc"
%include "cycle_tuner_main.inc"                ; the in-process cycle tuner: parallel per-shape state machines
%include "load_save_param_block.inc"           ; cycle tuner parameter block persistence: v7_tune_param_<n>_<d> files
%include "stoch_tune_percentages.inc"          ; persistent stochastic fractions
%include "rmsnorm_rope_dispatch.inc"
%include "rope_only_dispatch.inc"
%include "vit_dispatch.inc"
%include "v7_init.inc"
%include "v7_worker.inc"
%include "set_worker_priority.inc"
%include "v7_instr.inc"

; String hashes -- centralized strlen+CRC32, h1 + AES-NI u64, h2 for microdb + Intern
%include "string_hash.inc"
%include "aes_string_hash.inc"

; MicroDB persistent hash table, LZ77-backed, thread-safe LZ77, safe persistence
%include "microdb.inc"

; Tokenizer encode, -- assembly port, calls ht_find for vocabulary lookups
%include "encode.inc"

; Forward pass -- NASM forward_f32/forward_q8/set_rope_tables
%include "qk_fork_kernel.inc"                  ; Reentrant Q/K RMSNorm+RoPE, a mandatory fork kernel
%include "forward.inc"

; Text M-RoPE -- merged rope scratch tables for Qwen3-VL image turns
%include "mrope.inc"

; StreamingLLM setter -- every dependency, FATAL, engine gate, allocator,
; struct offsets, context block is live at this point
%include "stream_llm.inc"

%include "prefill_q8_fork.inc"                 ; Per-member QKV postprocess, no shared RMSNorm BSS writes
%include "prefill_q8.inc"

; Sampler -- AVX2 softmax + AES-NI RNG + max-heap top-p, direct-call export
%include "sampler.inc"

; sprintf -- Minimal variadic string formatter, replaces wsprintfA
%include "sprintf.inc"

; Chat -- Full chat loop in assembly, direct-call export
%include "chat.inc"

; StreamingLLM eviction flush -- call sites live in chat.inc, pass-2 forward
; reference to eval_script/valstr helpers is the established pattern
%include "stream_evict.inc"

; Generate -- Single-turn generate loop in assembly, direct-call export
%include "generate.inc"

; Chat Image -- Vision embedding upload + image-chat entry points, direct-call exports
%include "chat_image.inc"

; SectorLogo64 -- Logo-dialect interpreter with V7 host primitives, direct-call exports
%include "sectorlogo_main.inc"

; Execute -- spawn child process, capture stdout into 64KB buffer, direct-call export
%include "execute.inc"

; Token Stream -- Tri-array management for SectorLogo stream rewrite, internal
%include "token_stream.inc"

; ViT Input Stage -- nearest-neighbor image rescale to the model's fixed target resolution
%include "vit_input_stage.inc"

; ViT Bicubic Resize -- Catmull-Rom, a=-0.5 bicubic matching HF PIL BICUBIC, resample=3
%include "vit_bicubic.inc"

; ViT Forward -- Full vision tower forward pass + C-callable encode callback, pure ASM
%include "vit_forward.inc"

; Debug dump of InferenceContext + sub-structs, diagnostic, called from generate/chat
%include "dump_inference_context.inc"
%include "token_stream_dump.inc"

; File-backed memory allocator -- fbm_alloc / fbm_free, disk-backed, FATALs internally on failure
%include "file_backed_mem.inc"

; ---------------------------------------------------------------------------
; PREFETCH_PATCH_BEGIN emits nop_templates, starts patch_points
; PREFETCH_ROW emits each prefetch slot entry, kernel_id=0 for Q8
; PREFETCH_PATCH_END emits backup buffer + all SMC exports
; Note: v7 uses GENERATE_DLLMAIN below with a DETACH handler for worker
; shutdown, so we skip the auto-generated one from PREFETCH_PATCH_END.
; ---------------------------------------------------------------------------
     PREFETCH_PATCH_BEGIN                      ; emit SMC-patchable prefetch slots + backup buffer + exports

; Q8 kernel: 8 prefetch slots, each 9 bytes, ds prefix + 8-byte prefetch
; type_off=3, ModRM byte position, disp_off=5, disp32 position, kernel_id=0
%assign pf_slot 0                              ; initialize prefetch slot counter
%rep 8
     PREFETCH_ROW v7s_q8_chunk, pf_slot, 9, 3, 5, 0 ; Slot pf_slot: Q8 weight prefetch rowN, cacheline N%2+1
%assign pf_slot pf_slot + 1                    ; increment slot counter for next iteration
%endrep

; F32 kernel: 3 prefetch slots, each 9 bytes, ds prefix + 8-byte prefetch
; kernel_id=1, v7s_f32_chunk
     PREFETCH_ROW v7s_f32_chunk, 8, 9, 3, 5, 1 ; Slot 8: f32 activations +1280B
     PREFETCH_ROW v7s_f32_chunk, 9, 9, 3, 5, 1 ; Slot 9: f32 weights +320B
     PREFETCH_ROW v7s_f32_chunk, 10, 9, 3, 5, 1 ; Slot 10: f32 row1 weight prefetch +320B, rsi-based, dual-row second row

; Fused F32 kernel: 2 prefetch slots, each 9 bytes, ds prefix + 8-byte prefetch
; kernel_id=2, v7s_fused_f32_chunk
     PREFETCH_ROW v7s_fused_f32_chunk, 11, 9, 3, 5, 2 ; Slot 11: fused f32 weights +320B
     PREFETCH_ROW v7s_fused_f32_chunk, 12, 9, 3, 5, 2 ; Slot 12: fused f32 activations +1280B

; SwiGLU F32 kernel: 14 prefetch slots, kernel_id=3
; Uses swk_kernel, the SF_START label because it is the most recent non-local
; label before PREFETCH_SLOT -- local labels attach to swk_kernel, not v7s_swiglu_chunk.
; Both labels point to the same address, so offsets are identical.
; .single_loop256, 4 slots: gate PF6, up PF6, gate PF7, up PF7
     PREFETCH_ROW swk_kernel, 13, 9, 3, 5, 3   ; Slot 13: swiglu f32 single gate PF6
     PREFETCH_ROW swk_kernel, 14, 9, 3, 5, 3   ; Slot 14: swiglu f32 single up PF6
     PREFETCH_ROW swk_kernel, 15, 9, 3, 5, 3   ; Slot 15: swiglu f32 single gate PF7
     PREFETCH_ROW swk_kernel, 16, 9, 3, 5, 3   ; Slot 16: swiglu f32 single up PF7
; .single_tail64, 2 slots: gate PF6, up PF6
     PREFETCH_ROW swk_kernel, 17, 9, 3, 5, 3   ; Slot 17: swiglu f32 single_tail gate PF6
     PREFETCH_ROW swk_kernel, 18, 9, 3, 5, 3   ; Slot 18: swiglu f32 single_tail up PF6
; .dual_loop256, 4 slots: gate_r0 PF6, up_r0 PF6, gate_r1 PF6, up_r1 PF6
     PREFETCH_ROW swk_kernel, 19, 9, 3, 5, 3   ; Slot 19: swiglu f32 dual gate_r0 PF6
     PREFETCH_ROW swk_kernel, 20, 9, 3, 5, 3   ; Slot 20: swiglu f32 dual up_r0 PF6
     PREFETCH_ROW swk_kernel, 21, 9, 3, 5, 3   ; Slot 21: swiglu f32 dual gate_r1 PF6
     PREFETCH_ROW swk_kernel, 22, 9, 3, 5, 3   ; Slot 22: swiglu f32 dual up_r1 PF6
; .dual_tail64, 4 slots: gate_r0, gate_r1, up_r0, up_r1
     PREFETCH_ROW swk_kernel, 23, 9, 3, 5, 3   ; Slot 23: swiglu f32 dual_tail gate_r0
     PREFETCH_ROW swk_kernel, 24, 9, 3, 5, 3   ; Slot 24: swiglu f32 dual_tail gate_r1
     PREFETCH_ROW swk_kernel, 25, 9, 3, 5, 3   ; Slot 25: swiglu f32 dual_tail up_r0
     PREFETCH_ROW swk_kernel, 26, 9, 3, 5, 3   ; Slot 26: swiglu f32 dual_tail up_r1

; SwiGLU Q8 kernel: 12 prefetch slots, kernel_id=4
; Uses v7s_swiglu_q8_chunk, the SF_START label, now also the entry name .
; .single_q8_loop, 4 slots: gate PF6, up PF6, gate PF7, up PF7
     PREFETCH_ROW v7s_swiglu_q8_chunk, 27, 9, 3, 5, 4 ; Slot 27: swiglu q8 single gate PF6
     PREFETCH_ROW v7s_swiglu_q8_chunk, 28, 9, 3, 5, 4 ; Slot 28: swiglu q8 single up PF6
     PREFETCH_ROW v7s_swiglu_q8_chunk, 29, 9, 3, 5, 4 ; Slot 29: swiglu q8 single gate PF7
     PREFETCH_ROW v7s_swiglu_q8_chunk, 30, 9, 3, 5, 4 ; Slot 30: swiglu q8 single up PF7
; .dual_q8_loop, 8 slots: gate_r0 PF6, gate_r1 PF6, up_r0 PF6, up_r1 PF6, gate_r0 PF7, gate_r1 PF7, up_r0 PF7, up_r1 PF7
     PREFETCH_ROW v7s_swiglu_q8_chunk, 31, 9, 3, 5, 4 ; Slot 31: swiglu q8 dual gate_r0 PF6
     PREFETCH_ROW v7s_swiglu_q8_chunk, 32, 9, 3, 5, 4 ; Slot 32: swiglu q8 dual gate_r1 PF6
     PREFETCH_ROW v7s_swiglu_q8_chunk, 33, 9, 3, 5, 4 ; Slot 33: swiglu q8 dual up_r0 PF6
     PREFETCH_ROW v7s_swiglu_q8_chunk, 34, 9, 3, 5, 4 ; Slot 34: swiglu q8 dual up_r1 PF6
     PREFETCH_ROW v7s_swiglu_q8_chunk, 35, 9, 3, 5, 4 ; Slot 35: swiglu q8 dual gate_r0 PF7
     PREFETCH_ROW v7s_swiglu_q8_chunk, 36, 9, 3, 5, 4 ; Slot 36: swiglu q8 dual gate_r1 PF7
     PREFETCH_ROW v7s_swiglu_q8_chunk, 37, 9, 3, 5, 4 ; Slot 37: swiglu q8 dual up_r0 PF7
     PREFETCH_ROW v7s_swiglu_q8_chunk, 38, 9, 3, 5, 4 ; Slot 38: swiglu q8 dual up_r1 PF7

; Attention kernel: 2 prefetch slots, 6 bytes each, ds+REX prefixes + no disp32
; kernel_id=5, v7s_attention_inner, disp_off=-1, no patchable displacement
     PREFETCH_ROW v7s_attention_inner, 39, 6, 4, -1, 5 ; Slot 39: attention key prefetch, type-only
     PREFETCH_ROW v7s_attention_inner, 40, 6, 4, -1, 5 ; Slot 40: attention value prefetch, type-only

%define __pf_skip_dllmain                      ; v7 has its own DllMain below, handles DETACH
     PREFETCH_PATCH_END v7s_q8_chunk, kernel_bases ; Emit backup buffer + SMC exports

; ============================================================================
; Entry point -- DllMain. The standalone EXE entry point lives in
; examples/c_inference/dbg_inference.asm, linked with /entry:WinMain by build.bat;
; this DLL has no EXE build variant.
; ============================================================================
; DLL build: DllMain -- minimal DLL entry point
; Only handles DLL_PROCESS_DETACH: sets shutdown_flag and wakes workers.
; Prefetch backup is done by init_prefetch_backup, called from
; initialize_once, not from DllMain .
; Returns TRUE for all reasons.
; ============================================================================
     global DllMain                            ; exported: DLL entry point, attach/detach
DllMain:
     cmp edx, 1                                ; DLL_PROCESS_ATTACH?
     je .dll_attach                            ; yes, init microdb then return TRUE
     test edx, edx                             ; DLL_PROCESS_DETACH (reason 0)?
     jne .dll_other                            ; THREAD_ATTACH/DETACH, return TRUE
; DLL_PROCESS_DETACH: signal workers to stop if not process termination
     test r8, r8                               ; lpvReserved != NULL means process exit
     jnz .dll_other                            ; process exiting, just return TRUE
     sub rsp, 40                               ; 32 shadow + 8 alignment, RSP ? 8->0 mod 16 before call
; v7s_shutdown arms shutdown_flag, wakes workers 2x, and then POLLS workers_alive
; down to zero, each worker decrements it as its last DLL action, before
; RtlExitUserThread takes the loader lock -- so on return from this call no
; thread can execute DLL code any more and the unmap below is safe. Waiting on
; thread HANDLES here would deadlock under the loader lock; the Sleep-poll
; inside v7s_shutdown is deadlock-free and replaces the old blind 1s sleep
; whose guess caused an execute fault when a worker missed the window.
     call v7s_shutdown                         ; observed worker termination before the loader unmaps code

; NO ht_destroy here: the DB lifecycle belongs to the explicit caller
; forbids NULL-skip frees and races the loader lock. Process exit reclaims.
     add rsp, 40                               ; restore stack + alignment
.dll_other:
     mov eax, 1                                ; return TRUE
     ret                                       ; return to Windows loader
.dll_attach:
     sub rsp, 40                               ; 32 shadow + 8 alignment
     mov [rel sl_v7_hmodule], rcx              ; save own HMODULE for SectorLogo FFI "SELF" resolution -- before the probe, which overwrites ecx/rcx
     call cpu_gate_probe                       ; CPUID decision first: ht_init below is baseline code too, pure .bss stores
; Initialize MicroDB: create empty hash table, db_load is now called by V7_init
     call ht_init                              ; init hash table, empty, no data loaded
     add rsp, 40                               ; restore stack + alignment
     mov eax, 1                                ; return TRUE for ATTACH
     ret                                       ; return to Windows loader

; Inference -- ASM model init from microdb, v7s_inference_create/destroy
; Included at end, after DllMain/WinMain so it doesn't shift SMC/PREFETCH label offsets
%include "inference.inc"
%include "kv_cache_file.inc"

; Public v7_* C API tier, move_inference_context_to_bss_plan.md
; offsets; the wrappers contain no PREFETCH_SLOT sites and must never precede
; PREFETCH_PATCH_END. Single collection point for every public entry point.
%include "public_api.inc"
%include "version.inc"
%include "console_width.inc"
%include "tuner_activelist.inc"
