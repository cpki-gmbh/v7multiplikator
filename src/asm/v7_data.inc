; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .data align=64
; Key suffix strings for microdb-based config lookup
; suffixed to "__ini__default__" or "__ini__<section>__"
k_cores_subtract:
     db "cores_subtract", 0                    ; number of cores to subtract from worker count
k_use_physical_cores:
     db "use_physical_cores", 0                ; 0=SMT-aware, 1=one worker per physical core
k_num_chunks:
     db "num_chunks", 0                        ; base chunk count for dispatch partitioning
k_batch_small:
     db "batch_small", 0                       ; small batch size, few columns for worker
k_batch_medium:
     db "batch_medium", 0                      ; medium batch size for worker
k_batch_large:
     db "batch_large", 0                       ; large batch size for worker
k_batch_huge:
     db "batch_huge", 0                        ; huge batch size for worker
k_batch_med_thresh:
     db "batch_med_thresh", 0                  ; column threshold: small->medium transition
k_batch_large_thresh:
     db "batch_large_thresh", 0                ; column threshold: medium->large transition
k_batch_huge_thresh:
     db "batch_huge_thresh", 0                 ; column threshold: large->huge transition
k_chunk_multiplier:
     db "chunk_multiplier", 0                  ; multiplier applied to auto-computed chunk count
k_min_chunk_cols:
     db "min_chunk_cols", 0                    ; minimum columns per chunk, prevents tiny chunks
k_worker_priority:
     db "worker_priority", 0                   ; Win32 thread priority class for workers
k_vit_max_chunks:
     db "vit_max_chunks", 0                    ; ViT linear chunk-count cap, [default] tuning key
k_vit_min_chunk_cols:
     db "vit_min_chunk_cols", 0                ; ViT linear min columns per chunk, [default] tuning key
k_caller_batch_small:
     db "caller_batch_small", 0                ; small batch size for caller, participating thread
k_caller_batch_medium:
     db "caller_batch_medium", 0               ; medium batch size for caller
k_caller_batch_large:
     db "caller_batch_large", 0                ; large batch size for caller
k_caller_batch_huge:
     db "caller_batch_huge", 0                 ; huge batch size for caller
k_caller_batch_med_thresh:
     db "caller_batch_med_thresh", 0           ; caller column threshold: small->medium
k_caller_batch_large_thresh:
     db "caller_batch_large_thresh", 0         ; caller column threshold: medium->large
k_caller_batch_huge_thresh:
     db "caller_batch_huge_thresh", 0          ; caller column threshold: large->huge
k_chunk_split_pct:
     db "chunk_split_pct", 0                   ; chunk split percentage for group-partitioned dispatch
k_chunk_group_1_count:
     db "chunk_group_1_count", 0               ; number of chunks in group 1, split dispatch
k_chunk_group_2_count:
     db "chunk_group_2_count", 0               ; number of chunks in group 2, split dispatch
k_quantize_chunk_groups:
     db "quantize_chunk_groups", 0             ; max GS-groups per quantize job chunk, range [1..]; 0 or negative = FATAL
k_taper_shift:
     db "taper_shift", 0                       ; fixed-point k for tapered chunk fill, units=1/16, 0=uniform, 48=k=3.0 default
k_supported_resolutions:
     db "supported_resolutions", 0             ; per-shape supported-resolutions list for dispatch
; DB key prefix strings for building full lookup keys
ini_db_def:
     db "__ini__default__", 0                  ; default-section key prefix for microdb lookups
ini_db_proxy:
     db "__ini__proxy__", 0                    ; proxy-section key prefix for microdb lookups
ini_db_init:
     db "init:", 0                             ; init-section key prefix for microdb lookups
ini_db_pfx:
     db "__ini__", 0                           ; generic INI key prefix for microdb lookups
k_simple_threads:
     db "simple_threads", 0                    ; init:simple_threads value, 0 = full placement pool, N = N simple unpinned workers
; Prefetch key prefixes, followed by itoa counter
k_pf_dist_pfx:
     db "pf_dist_", 0                          ; pf_dist_N key prefix for prefetch slot N
k_pf_type_pfx:
     db "pf_type_", 0                          ; pf_type_N key prefix for prefetch slot N
k_no_prefix:
     db 0                                      ; empty string: the per-shape pf fill loops already hold "__ini__NxD_" in their cursor, so their "leading prefix" slot is this
; === Default parameter table for param_set_0 fill ===
; Each entry: key_suffix_ptr,8 + param_offset,4 + pad,4 = 16 bytes
%define DEF_PARAM_ENTRY_SIZE 16                ; bytes per def_param_table entry, ptr + offset + padding, 16-aligned for qword loads
%define DEF_PARAM_VAL_OFF 8                    ; dword offset: param value offset in entry
     align 16
def_param_table:
     dq k_num_chunks                           ; key: num_chunks
     dd PARAM_OFFSET_NUM_CHUNKS                ; param: num_chunks offset
     dd 0                                      ; padding, 16-byte aligned entries
     dq k_taper_shift                          ; key: taper_shift
     dd PARAM_OFFSET_TAPER_SHIFT               ; param: taper_shift offset, slot 1
     dd 0                                      ; padding
     dq k_batch_small                          ; key: batch_small
     dd PARAM_OFFSET_BATCH_SMALL               ; param: batch_small offset
     dd 0                                      ; padding
     dq k_batch_medium                         ; key: batch_medium
     dd PARAM_OFFSET_BATCH_MEDIUM              ; param: batch_medium offset
     dd 0                                      ; padding
     dq k_batch_large                          ; key: batch_large
     dd PARAM_OFFSET_BATCH_LARGE               ; param: batch_large offset
     dd 0                                      ; padding
     dq k_batch_huge                           ; key: batch_huge
     dd PARAM_OFFSET_BATCH_HUGE                ; param: batch_huge offset
     dd 0                                      ; padding
     dq k_batch_med_thresh                     ; key: batch_med_thresh
     dd PARAM_OFFSET_BATCH_MED_THRESH          ; param: medium threshold offset
     dd 0                                      ; padding
     dq k_batch_large_thresh                   ; key: batch_large_thresh
     dd PARAM_OFFSET_BATCH_LARGE_THRESH        ; param: large threshold offset
     dd 0                                      ; padding
     dq k_batch_huge_thresh                    ; key: batch_huge_thresh
     dd PARAM_OFFSET_BATCH_HUGE_THRESH         ; param: huge threshold offset
     dd 0                                      ; padding
     dq k_chunk_multiplier                     ; key: chunk_multiplier
     dd PARAM_OFFSET_CHUNK_MULTIPLIER          ; param: chunk_multiplier offset
     dd 0                                      ; padding
     dq k_min_chunk_cols                       ; key: min_chunk_cols
     dd PARAM_OFFSET_MIN_CHUNK_COLS            ; param: min_chunk_cols offset
     dd 0                                      ; padding
     dq k_worker_priority                      ; key: worker_priority
     dd PARAM_OFFSET_WORKER_PRIORITY           ; param: worker_priority offset
     dd 0                                      ; padding
     dq k_quantize_chunk_groups                ; key: quantize_chunk_groups
     dd PARAM_OFFSET_QUANTIZE_CHUNK_GROUPS     ; param: quantize_chunk_groups offset, slot 12
     dd 0                                      ; padding
     dq k_caller_batch_small                   ; key: caller_batch_small
     dd PARAM_OFFSET_CALLER_BATCH_SMALL        ; param: caller_batch_small offset
     dd 0                                      ; padding
     dq k_caller_batch_medium                  ; key: caller_batch_medium
     dd PARAM_OFFSET_CALLER_BATCH_MEDIUM       ; param: caller_batch_medium offset
     dd 0                                      ; padding
     dq k_caller_batch_large                   ; key: caller_batch_large
     dd PARAM_OFFSET_CALLER_BATCH_LARGE        ; param: caller_batch_large offset
     dd 0                                      ; padding
     dq k_caller_batch_huge                    ; key: caller_batch_huge
     dd PARAM_OFFSET_CALLER_BATCH_HUGE         ; param: caller_batch_huge offset
     dd 0                                      ; padding
     dq k_caller_batch_med_thresh              ; key: caller_batch_med_thresh
     dd PARAM_OFFSET_CALLER_BATCH_MED_THRESH   ; param: caller_med_threshold offset
     dd 0                                      ; padding
     dq k_caller_batch_large_thresh            ; key: caller_batch_large_thresh
     dd PARAM_OFFSET_CALLER_BATCH_LARGE_THRESH ; param: caller_large_threshold offset
     dd 0                                      ; padding
     dq k_caller_batch_huge_thresh             ; key: caller_batch_huge_thresh
     dd PARAM_OFFSET_CALLER_BATCH_HUGE_THRESH  ; param: caller_huge_threshold offset
     dd 0                                      ; padding
     dq k_chunk_split_pct                      ; key: chunk_split_pct
     dd PARAM_OFFSET_CHUNK_SPLIT_PCT           ; param: chunk_split_pct offset
     dd 0                                      ; padding
     dq k_chunk_group_1_count                  ; key: chunk_group_1_count
     dd PARAM_OFFSET_CHUNK_GROUP_1_COUNT       ; param: group_1 count offset
     dd 0                                      ; padding
     dq k_chunk_group_2_count                  ; key: chunk_group_2_count
     dd PARAM_OFFSET_CHUNK_GROUP_2_COUNT       ; param: group_2 count offset
     dd 0                                      ; padding
     dq k_cores_subtract                       ; key: cores_subtract
     dd PARAM_OFFSET_CORES_SUBTRACT            ; param: cores_subtract offset
     dd 0                                      ; padding
def_param_table_end:
%define DEF_PARAM_ENTRIES ((def_param_table_end - def_param_table) / DEF_PARAM_ENTRY_SIZE) ; total entries in def_param_table, fully parenthesised: safe as a divisor or inside larger expressions
%ifdef V7_DEBUG
     FMT fmt_proxy_cores, "[proxy] %d logical cores (%d available + caller), %d workers (smt-aware)" ; debug: core/worker count
     FMT fmt_proxy_count, "[proxy] %d workers (+1 caller = %d total,%d chunks)" ; debug: worker/chunk count
     FMT fmt_proxy_failed, "[proxy] failed: %s" ; debug: proxy failure message
%endif
%ifdef PROFILE_MODE
     FMT fmt_instr, "[instr] total=%I64d caller=%I64d(%I64d fa) workers_active=%d/%d chunks=%d dispatch=%dx%d" ; profile: dispatch summary
fmt_instr_worker_jobs:
     db "[instr] worker jobs:", 0              ; profile: worker jobs label
fmt_instr_worker_fa:
     db "[instr] worker fetch_add:", 0         ; profile: fetch_add label
fmt_instr_worker_spins:
     db "[instr] worker wake_spins:", 0        ; profile: wake spins label
fmt_instr_worker_sleeps:
     db "[instr] worker sleeps:", 0            ; profile: sleep count label
     FMT fmt_instr_tail, "[instr] caller_remaining_spins=%I64d total_fetch_adds=%I64d total_sleeps=%I64d" ; profile: tail stats
lbl_total_jobs:
     db "total_jobs: ", 0                      ; profile: total jobs label
lbl_caller:
     db "caller: ", 0                          ; profile: caller label
lbl_caller_total:
     db "caller_total: ", 0                    ; profile: sticky caller total across all dispatch types label
lbl_peak:
     db "worker_jobs_peak: ", 0                ; profile: per-worker high-water mark of jobs_done label
lbl_workers_active:
     db "workers_active: ", 0                  ; profile: active workers label
lbl_total_sleeps:
     db "total_sleeps: ", 0                    ; profile: total sleeps label
lbl_caller_spins:
     db "caller_spins_wait:", 0                ; profile: caller spins label
lbl_total_fetch_adds:
     db "total_fetch_adds: ", 0                ; profile: fetch_add total label
lbl_num_workers:
     db "num_workers: ", 0                     ; profile: worker count label
lbl_gen_counter:
     db "gen_counter: ", 0                     ; profile: generation counter label
lbl_remaining_jobs:
     db "remaining_jobs: ", 0                  ; profile: remaining jobs label
lbl_next_job:
     db "next_job: ", 0                        ; profile: next job index label
lbl_worker_aff:
     db "worker_aff: ", 0                      ; profile: worker affinity label
lbl_sep:
     db "----------------------------------------", 0 ; profile: separator line
lbl_dbg_glpiex_len:
     db "[dbg] glpiex_returned_length: ", 0    ; debug: GLPIEX return length
lbl_dbg_core_map_count:
     db "[dbg] core_map_count: ", 0            ; debug: core map count
lbl_dbg_use_phys_cores:
     db "[dbg] use_physical_cores: ", 0        ; debug: physical cores flag
lbl_dbg_cores_subtract:
     db "[dbg] cores_subtract: ", 0            ; debug: cores subtract value
lbl_dbg_r14_workers:
     db "[dbg] r14_workers_placed: ", 0        ; debug: workers placed count
lbl_dbg_r13_total_cores:
     db "[dbg] r13_total_cores: ", 0           ; debug: total cores detected
lbl_dbg_num_workers_calc:
     db "[dbg] num_workers_computed: ", 0      ; debug: computed worker count
lbl_dbg_param_set_0:
     db "[dbg] param_set_0:", 0                ; debug: param set 0
%endif
%ifdef CPU0_DIAG
lbl_cpu0_num_workers:
     db "cpu0_diag: num_workers: ", 0          ; diag: active worker count heading
lbl_cpu0_worker_aff:
     db "cpu0_diag: worker_aff: ", 0           ; diag: per-worker affinity masks follow
lbl_cpu0_saved_affinity:
     db "cpu0_diag: saved_affinity: ", 0       ; diag: previous affinity mask of the bound caller
lbl_cpu0_dispatch_tid:
     db "cpu0_diag: dispatch_tid: ", 0         ; diag: OS thread ID of the dispatching caller
lbl_cpu0_bound_tid:
     db "cpu0_diag: caller_bound_tid: ", 0     ; diag: thread ID pinned as the CPU-0 caller
lbl_cpu0_caller_jobs:
     db "cpu0_diag: caller_jobs_done: ", 0     ; diag: jobs the caller itself completed
%endif
; Threading report strings: printed once per pool creation, production output
lbl_thr_header:
     db "[threads] ---- worker pool report ----", 0 ; report heading
lbl_thr_ini:
     db "[threads] ini: cores_subtract=", 0    ; ini line prefix
lbl_thr_s_wpri:
     db " worker_priority=", 0                 ; ini line fragment
lbl_thr_s_upc:
     db " use_physical_cores=", 0              ; ini line fragment
lbl_thr_s_nch:
     db " num_chunks=", 0                      ; ini line fragment
lbl_thr_s_chm:
     db " chunk_multiplier=", 0                ; ini line fragment
lbl_thr_env_none:
     db "[threads] env override: none", 0      ; env line, override not consulted
lbl_thr_env_applied:
     db "[threads] env override: v7_override_prio applied, ini priority ", 0 ; env line prefix
lbl_thr_s_eff:
     db " -> effective ", 0                    ; env line fragment
lbl_thr_os:
     db "[threads] os: glpiex_bytes=", 0       ; os line prefix
lbl_thr_s_cores:
     db " cores=", 0                           ; os line fragment
lbl_thr_s_pmask:
     db " process_mask=", 0                    ; os line fragment, WRITE_HEX64 adds the 0x prefix itself
lbl_thr_s_smask:
     db " system_mask=", 0                     ; os line fragment, WRITE_HEX64 adds the 0x prefix itself
lbl_thr_core:
     db "[threads] core ", 0                   ; core line prefix
lbl_thr_core_raw:
     db ": raw=", 0                            ; core line fragment, WRITE_HEX64 adds the 0x prefix itself
lbl_thr_core_elig:
     db " eligible=", 0                        ; core line fragment, WRITE_HEX64 adds the 0x prefix itself
lbl_thr_core_caller:
     db "  <- caller CPU 0", 0                 ; core line suffix on the caller's core
lbl_thr_mode_phys:
     db "[threads] mode: physical placement, one worker per physical core, caller core excluded", 0 ; mode line, physical
lbl_thr_mode_smt:
     db "[threads] mode: logical (SMT-aware) placement, one worker per eligible logical processor", 0 ; mode line, logical
lbl_thr_simple:
     db "[threads] simple mode: init:simple_threads=", 0 ; simple-mode line prefix
lbl_thr_s_simple:
     db " simple workers (CreateThread + priority only, no pinning, no SMT placement), priority=", 0 ; simple-mode line fragment
lbl_thr_cand:
     db "[threads] candidates: ", 0            ; candidates line prefix
lbl_thr_s_cap:
     db " eligible capacity incl. caller: ", 0 ; candidates line fragment
lbl_thr_decision:
     db "[threads] decision: num_workers = capacity(", 0 ; decision line prefix
lbl_thr_s_cs:
     db ") - cores_subtract(", 0               ; decision line fragment
lbl_thr_s_eq:
     db ") = ", 0                              ; decision line fragment
lbl_thr_s_plus1:
     db ", plus 1 caller = ", 0                ; decision line fragment
lbl_thr_s_parts:
     db " participants", 0                     ; decision line tail
lbl_thr_chunks:
     db "[threads] chunks: num_chunks=", 0     ; chunks line prefix
lbl_thr_chunks_is_ini:
     db " (ini override, auto would be ", 0    ; chunks line, ini source
lbl_thr_chunks_auto:
     db " (auto: (num_workers+1) * chunk_multiplier)", 0 ; chunks line, auto source
lbl_thr_s_close:
     db ")", 0                                 ; chunks line closing paren
lbl_thr_s_max:
     db ", max=", 0                            ; chunks line fragment
lbl_thr_created_w:
     db "[threads] created W", 0               ; created line prefix
lbl_thr_created_mid:
     db ": handle=", 0                         ; created line fragment, WRITE_HEX64 adds the 0x prefix itself
lbl_thr_s_aff:
     db " aff=", 0                             ; created line fragment, WRITE_HEX64 adds the 0x prefix itself
lbl_thr_s_prio:
     db " prio=", 0                            ; created line fragment
lbl_thr_started_n:
     db "[threads] workers started: ", 0       ; started line prefix
lbl_thr_s_slash:
     db "/", 0                                 ; started line fragment
lbl_thr_s_budget:
     db " (startup poll budget 2000 ms)", 0    ; started line tail
lbl_thr_caller:
     db "[threads] caller: tid=", 0            ; caller line prefix
lbl_thr_s_oprio:
     db " original_prio=", 0                   ; caller line fragment
lbl_thr_s_saff:
     db " saved_affinity=", 0                  ; caller line fragment, WRITE_HEX64 adds the 0x prefix itself
lbl_thr_s_pinned:
     db " -> pinned to CPU 0, prio=", 0        ; caller line fragment
lbl_thr_coverage:
     db "[threads] coverage: ", 0              ; coverage line prefix
lbl_thr_s_cover:
     db " disjoint single-bit participants cover ", 0 ; coverage line fragment
lbl_thr_s_elig_lp:
     db " eligible logical processors", 0      ; coverage line tail
lbl_thr_runtime:
     db "[threads] runtime: always_spin=", 0   ; runtime line prefix
lbl_thr_s_wms:
     db " workers_may_sleep=", 0               ; runtime line fragment
lbl_thr_s_seff:
     db " scheduler_effective_priority=", 0    ; runtime line fragment
lbl_thr_s_maxw:
     db " max_workers=", 0                     ; runtime line fragment
lbl_thr_w_started:
     db "worker W", 0                          ; worker startup line prefix
lbl_thr_w_mid:
     db " started: tid=", 0                    ; worker startup line fragment
lbl_thr_w_cpu:
     db " on logical processor ", 0            ; worker startup line fragment
lbl_thr_w_aff:
     db ", aff=", 0                            ; worker startup line fragment, WRITE_HEX64 adds the 0x prefix itself
     FMT fmt_instr_nl, ""                      ; profile: newline format
%ifdef V7_DEBUG
     FMT fmt_init_ok, "[proxy] init_ok=1 workers=%d chunks=%d" ; debug: init success message
     FMT fmt_shutdown, "[proxy] shutdown: joining %d workers..." ; debug: shutdown start message
     FMT fmt_shutdown_done, "[proxy] shutdown complete" ; debug: shutdown complete message
     FMT fmt_slice_line, " dispatched=%d chunks=%d %s" ; debug: slice line summary
%endif
; Hex digit lookup table for debug output
     align 8
hex_chars:
     db "0123456789ABCDEF", 0                  ; hex digit lookup table
     PARAM_OFFSET_NUM_CHUNKS equ 0*4           ; 0
     PARAM_OFFSET_TAPER_SHIFT equ 1*4          ; 4, taper_shift; the autotuner writes it per-shape
     PARAM_OFFSET_BATCH_SMALL equ 2*4          ; 8
     PARAM_OFFSET_BATCH_MEDIUM equ 3*4         ; 12
     PARAM_OFFSET_BATCH_LARGE equ 4*4          ; 16
     PARAM_OFFSET_BATCH_HUGE equ 5*4           ; 20
     PARAM_OFFSET_BATCH_MED_THRESH equ 6*4     ; 24
     PARAM_OFFSET_BATCH_LARGE_THRESH equ 7*4   ; 28
     PARAM_OFFSET_BATCH_HUGE_THRESH equ 8*4    ; 32
     PARAM_OFFSET_CHUNK_MULTIPLIER equ 9*4     ; 36
     PARAM_OFFSET_MIN_CHUNK_COLS equ 10*4      ; 40
     PARAM_OFFSET_WORKER_PRIORITY equ 11*4     ; 44
     PARAM_OFFSET_QUANTIZE_CHUNK_GROUPS equ 12*4 ; 48, quantize_chunk_groups, per shape
     PARAM_OFFSET_CALLER_BATCH_SMALL equ 13*4  ; 52
     PARAM_OFFSET_CALLER_BATCH_MEDIUM equ 14*4 ; 56
     PARAM_OFFSET_CALLER_BATCH_LARGE equ 15*4  ; 60
     PARAM_OFFSET_CALLER_BATCH_HUGE equ 16*4   ; 64
     PARAM_OFFSET_CALLER_BATCH_MED_THRESH equ 17*4 ; 68
     PARAM_OFFSET_CALLER_BATCH_LARGE_THRESH equ 18*4 ; 72
     PARAM_OFFSET_CALLER_BATCH_HUGE_THRESH equ 19*4 ; 76
     PARAM_OFFSET_CHUNK_SPLIT_PCT equ 20*4     ; 80
     PARAM_OFFSET_CHUNK_GROUP_1_COUNT equ 21*4 ; 84
     PARAM_OFFSET_CHUNK_GROUP_2_COUNT equ 22*4 ; 88
     PARAM_OFFSET_CORES_SUBTRACT equ 23*4      ; 92, cores_subtract, per shape
     PARAM_SET_DWORDS equ 24                   ; total dwords in parameter set, 24 params, fully packed
     PARAM_SET_SIZE equ 96                     ; total bytes, 24 * 4 = 96, must be qword-aligned for rep movsq
     PARAM_SET_QWORDS equ 12                   ; total qwords for rep movsq, 96 / 8
     SHAPE_BLOCK_N_EQU equ 0                   ; +0: block n, dword
     SHAPE_BLOCK_D_EQU equ 4                   ; +4: block d, dword
     SHAPE_BLOCK_PARAM_OFF equ 8               ; +8: param_set starts here, 96 bytes, 24 dwords
     SHAPE_BLOCK_PF_DIST_OFF equ 104           ; +104: pf_dist[0..40], 41 dwords, 164 bytes
     SHAPE_BLOCK_PF_TYPE_OFF equ 300           ; +300: pf_type[0..40], 41 dwords, 164 bytes
     SHAPE_BLOCK_SIZE equ 512                  ; total block size, 8 cache lines, padded for alignment, arrays shrink, offsets kept
     SHAPE_BLOCK_SIZE_SHL equ 9                ; log2,SHAPE_BLOCK_SIZE, shift count for block-index -> byte offset. Every block-index*SHAPE_BLOCK_SIZE site must agree with this; the %if below makes the two impossible to drift.
%if (1 << SHAPE_BLOCK_SIZE_SHL) != SHAPE_BLOCK_SIZE
%error SHAPE_BLOCK_SIZE_SHL must be log2(SHAPE_BLOCK_SIZE) ; assembly-time guard: the two can never disagree
%endif
     MAX_SHAPE_BLOCKS equ 256                  ; blocks 0-1: 0=active, 1=default, 2-255=per-shape
     SHAPE_BLOCKS_TOTAL equ MAX_SHAPE_BLOCKS * SHAPE_BLOCK_SIZE ; 131072 bytes, 256 * 512
     SHAPE_BLOCK1_OFF equ 1*SHAPE_BLOCK_SIZE   ; byte offset to block 1, default fallback
     SHAPE_BLOCK2_OFF equ 2*SHAPE_BLOCK_SIZE   ; byte offset to block 2, first shape-specific
     PF_DIST_TABLE_ENTRIES equ 41              ; 41 prefetch entries, pf_dist_0..40, pf_type_0..40
%if (SHAPE_BLOCK_PF_TYPE_OFF + PF_DIST_TABLE_ENTRIES*4) > SHAPE_BLOCK_SIZE
%error SHAPE_BLOCK_PF_TYPE_OFF + pf_dist[PF_DIST_TABLE_ENTRIES] must fit inside SHAPE_BLOCK_SIZE ; assembly-time guard: the arrays can never run past the block
%endif
%if SHAPE_BLOCK_PF_DIST_OFF < (SHAPE_BLOCK_PARAM_OFF + PARAM_SET_SIZE)
%error SHAPE_BLOCK_PF_DIST_OFF overlaps param_set, raise it to SHAPE_BLOCK_PARAM_OFF + PARAM_SET_SIZE ; assembly-time guard: pf_dist can never sit inside the parameter set
%endif
%if SHAPE_BLOCK_PF_TYPE_OFF < (SHAPE_BLOCK_PF_DIST_OFF + PF_DIST_TABLE_ENTRIES*4)
%error SHAPE_BLOCK_PF_TYPE_OFF overlaps pf_dist, raise it to SHAPE_BLOCK_PF_DIST_OFF + PF_DIST_TABLE_ENTRIES*4 ; assembly-time guard: the 32-byte inter-array gap can never close
%endif
     PF_PFX_BACK1 equ 1                        ; backstep 1 byte from cursor to overwrite 2nd underscore of "__" tail
     QWORD_SCALE equ 8                         ; scale factor for qword-indexed array access [reg + idx*8]
     DWORD_SCALE equ 4                         ; scale factor for dword-indexed array access [reg + idx*4]
     section .bss align=64
; Job pool
     alignb 64
jobs_arr:
     resb MAX_JOBS * JOB_SIZE                  ; job pool: MAX_JOBS entries of JOB_SIZE bytes each
; gen_counter on its own cache line, no false sharing with num_jobs
     alignb 64
gen_counter:
     resd 1                                    ; generation counter for dispatch coordination
; Register handoff slots: one cache line per worker, zero-initialized targetaddr.
     alignb 64
fork_params:
     resb MAX_WORKERS * fork_param_size        ; register input/results and target address for each helper
; One Q8/F32 greedy fork split fraction, fixed point /256: the caller's share
; of the vocabulary in the combined max/first-index scan.
     alignb 64
split_percentage:
     resd 1                                    ; caller work fraction, kept in 1..255
     alignb 64
num_jobs:
     resd 1                                    ; total job count for current dispatch
     BSS_DWORD next_job                        ; next job index, lock xadd target
     BSS_DWORD remaining_jobs                  ; remaining uncompleted jobs
     BSS_DWORD shutdown_flag                   ; worker shutdown signal
     BSS_DWORD workers_alive                   ; live worker count: inc at worker_proc entry, dec as the worker's LAST DLL action, shutdown polls this instead of guessing a sleep length
; accessed via %define aliases, num_chunks, etc. .
; At dispatch time, the matching block is copied to block 0 via rep movsq.
; Prefetch distances and types are in each block's pf_dist/pf_type arrays,
; read by dispatch code and passed to set_prefetch_distance/set_prefetch_type.
     alignb 64
     global shape_blocks                       ; export for diagnostic tools
shape_blocks:
     resb SHAPE_BLOCKS_TOTAL                   ; MAX_SHAPE_BLOCKS, 256 x 512 bytes = 131072 bytes
     BSS_DWORD cached_shape_n                  ; last n dispatched, 0 = cold
     BSS_DWORD cached_shape_d                  ; last d dispatched
     BSS_DWORD cached_prefetch_n               ; last n that had prefetch patched, 0 = unpatched
     BSS_DWORD cached_prefetch_d               ; last d that had prefetch patched, 0 = unpatched
attn_inv_sqrt_by32:
     resd 33                                   ; 1/sqrt(head_dim) for head_dim 64..1024 step 32, init-once under the captured MXCSR, read by attention
scratch_key_buf:
     resb 64                                   ; scratch buffer for DB key construction, used at init time
; Non-parameter globals, NOT part of param set
     BSS_DWORD num_workers                     ; number of active worker threads
     BSS_DWORD init_ok                         ; initialization complete flag, 1=ready
     BSS_DWORD init_once_flag                  ; CAS guard for one-time initialization
     BSS_DWORD g_mxcsr_ftz_daz                 ; captured post-OR MXCSR image, 0x1F80|0x8040=0x9FC0, read by SET_FTZ_DAZ on every dispatch entry
; cpu_features_ok + cpu_gate_msg_ptr: set by cpu_gate_probe at
; DllMain,ATTACH, read once per public API entry by PUB_CPU_GATE. Packed in ONE
; cache line and NOT given the 64-byte BSS_DWORD alignment: the two fields are
; always read together and 128 bytes of BSS plus a second line for a 4-byte flag is
; pure ceremony.
     alignb 8
     global cpu_features_ok                    ; exported: tools flip this to exercise the gate, test_engine_death.c
cpu_features_ok:
     resd 1                                    ; 1 = CPUID baseline satisfied, 0 = unsupported, pessimistic default
cpu_gate_msg_ptr:
     resq 1                                    ; .rdata message for the first missing feature, generic fallback until proven
     alignb 64
threads:
     resq MAX_WORKERS                          ; worker thread handles array
     BSS_DWORD always_spin                     ; private activity state: 0=idle (workers sleep), 1=inference active (pause-spin + fork polling)
     BSS_DWORD workers_may_sleep               ; number of workers registered in the wait-capable path
     BSS_DWORD use_physical_cores              ; 0=SMT-aware, 1=one worker per physical core
     BSS_DWORD simple_threads                  ; init:simple_threads value, 0 = full placement pool, N > 0 = N simple unpinned workers
     BSS_QWORD config_was_fixed                ; u64 bitmap: each engine-side config fix sets its own bit
     global config_was_fixed                   ; export for autotuner/tools: which config values were engine-fixed
; The old Logo tuner's state (candidate block, TSS slots, tune_* globals)
; is GONE -- the in-process cycle tuner lives in cycle_tuner_main.inc with
; its own arrays. What remains here is the dispatcher's own config-scan
; cache.
tune_scanned_block:
     resq 1                                    ; last matched shape block, the warm path re-pins it
     BSS_DWORD prefill_unit_claim              ; Nonzero only for independent prefill panel jobs
q8b_input_packed:
     resd 1                                    ; Packed-input flag for the active Q8 panel
     alignb 8                                  ; Align the following descriptor pointer
q8b_args_end:
     resq 1                                    ; End of the Q8 batch descriptor panel
q8b_args_ptr:
     resq 1                                    ; whole-matrix v7_q8_batch4_args block for the batch4 job mode, published by v7d_q8_batch4 before DISPATCH_EXECUTE, read by process_job's slice builder, valid for the whole blocking dispatch
stream_remove_prog:
     resq 1                                    ; parsed prog for the on_stream_evict eviction callback, 0 = not yet parsed
stream_remove_script_len:
     resd 1                                    ; text length the cached prog was parsed from; ht_add also drops the prog when the on_stream_evict value is replaced, so equal-length rewrites reparse too

; Fused kernel context, approach 6 : set before dispatching fused matmul jobs
; All workers read these globals, safe since all jobs in a dispatch cycle share values
     BSS_QWORD fused_norm_weights              ; pointer to RMSNorm weight vector for current layer
     global fused_bias_ptr                     ; internal symbol, BSS_QWORD below defines the storage
     BSS_QWORD fused_bias_ptr                  ; pointer to fp32 QKV bias for this Q/K call, ALWAYS a real pointer to >= d floats, a bias-less model gets an all-zeros block at init, so the kernel never has a NULL case
     BSS_QWORD fused_rope_cos                  ; pointer to precomputed RoPE cos table, internal-only
     BSS_QWORD fused_rope_sin                  ; pointer to precomputed RoPE sin table, internal-only
     BSS_DWORD rope_max_dim                    ; int rope_max_dim value, set via set_rope_tables
     BSS_DWORD fused_head_dim                  ; head dimension, 64 for Qwen3-0.6B, internal-only
     BSS_DWORD fused_rope_owned                ; 1=internally allocated, mdb_alloc_mem, safe to mdb_free_mem, 0=external pointer, set_rope_tables, NEVER freed
     BSS_DWORD fused_pos                       ; current sequence position, for RoPE table indexing
     BSS_DWORD g_rms_eps                       ; RMS norm epsilon, 1e-6 Qwen3/SmolLM2, 1e-5 Llama/Phi-3, per-model from DB
; RMSNorm+RoPE per-head kernel context: set before dispatch, read by workers
     BSS_QWORD rmsnorm_norm_weight             ; pointer to RMSNorm weight vector for current layer
     BSS_QWORD rmsnorm_cos_table               ; pointer to precomputed RoPE cos table base
     BSS_QWORD rmsnorm_sin_table               ; pointer to precomputed RoPE sin table base
     BSS_DWORD rmsnorm_head_dim                ; head dimension
     BSS_DWORD rmsnorm_head_dim_float          ; float head_dim, set by dispatcher, read by workers; kernel divides by it
     BSS_DWORD rmsnorm_pos                     ; current sequence position
; SwiGLU fused kernel context: up-weight tensor pointers set before dispatch
     BSS_QWORD swiglu_up_weight_q              ; int8_t* pointer to UP weight quantized data
     BSS_QWORD swiglu_up_weight_s              ; float* pointer to UP weight scales
; Chat session system-prompt opener, A01 : the system string that opened this
; KV session, so a continuation call can prove its system prompt still
; byte-matches instead of silently re-rendering the turn-1 prefix mid-session
g_sys_prompt_save:
     resq 1                                    ; engine-OWNED char* copy of the opener system prompt, 0 = session opened without one, the caller's buffer is borrowed only for the turn-1 call, so the continuation byte-match reads THIS copy
g_sys_prompt_len:
     resd 1                                    ; opener byte length, 0 = opener or empty string
g_sys_prompt_owned:
     resq 1                                    ; mdb block backing g_sys_prompt_save, freed+cleared at reset/no-system/destroy
     alignb 64
     global g_qf_scratch_q                     ; export for diagnostic tools
g_qf_scratch_q:
     resb QF_SCRATCH_MAX_N                     ; int8 scratch for v7d_q8_fused quantize-once
     alignb 64
     global g_qf_scratch_s                     ; export for diagnostic tools
g_qf_scratch_s:
     resb QF_SCRATCH_MAX_N / GS * 4            ; float scales, one per GS-group: N/GS floats = N/GS*4 bytes
     BSS_ARRAY 64, worker_jobs_done, q, MAX_WORKERS*8 ; per-worker job completion counts
     BSS_ARRAY 64, worker_jobs_peak, q, MAX_WORKERS*8 ; per-worker high-water mark of jobs_done across all periods of the session, the reset merges before zeroing (PROFILE_MODE)
     BSS_ARRAY 64, worker_fetch_adds, q, MAX_WORKERS*8 ; per-worker fetch_add counts
     BSS_ARRAY 64, worker_wake_spins, q, MAX_WORKERS*8 ; per-worker wake spin counts
     BSS_ARRAY 64, worker_sleeps, q, MAX_WORKERS*8 ; per-worker sleep counts
     BSS_ARRAY 64, caller_jobs_done, q, 1      ; caller job completion count
     BSS_ARRAY 64, caller_jobs_total, q, 1     ; sticky caller total across every dispatch type and period, summed at each matmul reset (PROFILE_MODE)
     BSS_ARRAY 64, caller_fetch_adds, q, 1     ; caller fetch_add count
     BSS_ARRAY 64, caller_spins_wait, q, 1     ; caller spin-wait count
stdout_handle:
     resq 1                                    ; stdout file handle, ALL diagnostic output goes to stdout
     BSS_DWORD debug_step                      ; debug step counter
; GLPIEx size-query upper bound. core_buffer below is sized FROM this: a bound
; larger than the buffer lets the second GetLogicalProcessorInformationEx call
; overflow BSS, and a bound smaller than a big machine needs gives a spurious
; FATAL. Keep the constant and the resb in step.
     GLPIEX_BUF_MAX equ 8192                   ; GLPIEx size-query upper bound, core_buffer is sized from this
     align 8
core_buffer:
     resb GLPIEX_BUF_MAX                       ; core info buffer for GetLogicalProcessorInformationEx, 8-aligned: GLPIEx records carry KAFFINITY/ULONG_PTR members; sized from GLPIEX_BUF_MAX, keep the two in step
     align 8
saved_affinity:
     resq 1                                    ; saved thread affinity for unpin
     BSS_DWORD pinned_flag                     ; flag: caller thread has pinned affinity
     align 8
caller_thread_handle:
     resq 1                                    ; duplicated real handle to the initialization caller thread, 0 = not captured
caller_saved_priority:
     resd 1                                    ; original THREAD_PRIORITY_ value of the caller, restored by unpin
caller_bound_tid:
     resd 1                                    ; 0 = no bound caller, otherwise the OS thread ID of the saved handle
scheduler_effective_priority:
     resd 1                                    ; applied OS priority of workers + bound caller, independent of the per-shape worker_priority config slots the dispatcher overwrites on every shape switch
     align 8
worker_aff:
     resq MAX_WORKERS                          ; per-worker affinity masks
     BSS_DWORD core_map_count                  ; number of entries in core_map_masks
     align 8
core_map_masks:
     resq (MAX_WORKERS + 1)                    ; per-core group affinity masks
core_map_raw:
     resq (MAX_WORKERS + 1)                    ; per-core raw GLPIEx masks before the process-mask intersection, threading report
process_affinity_mask:
     resq 1                                    ; process mask snapshot for the threading report
system_affinity_mask:
     resq 1                                    ; system mask snapshot for the threading report
num_chunks_source:
     resd 1                                    ; 0 = chunk count computed from the participant count, 1 = INI override supplied it, threading report
debug_glpiex_length:
     resd 1                                    ; GLPIEx returned_length for debugging core detection
debug_core_map_count:
     resd 1                                    ; core_map_count for debugging core detection
debug_use_phys_cores:
     resd 1                                    ; use_physical_cores INI value
debug_cores_subtract:
     resd 1                                    ; cores_subtract INI value
debug_worker_count_raw:
     resd 1                                    ; r14, workers placed at workers_configured
debug_r13_at_workers:
     resd 1                                    ; r13, total_cores at workers_configured
debug_num_workers_computed:
     resd 1                                    ; num_workers after STEP 4 calc
cpu0_diag_pending:
     resd 1                                    ; CPU0_DIAG builds only: 1 = one-time session dump still owed, consumed by the first completed dispatch
     global debug_param_set_0                  ; export for diagnostic tools
debug_param_set_0:
     resd 24                                   ; snapshot of param_set_0, 24 dwords = 96 bytes
; === %define aliases: all param names -> param_set_0 + offset ===
; param_set_0 = shape_blocks + 0, active block
; All existing code references [num_chunks], etc.
; These %define aliases transparently redirect all loads/stores to the
; active param set, block 0.No other code changes needed.
%define param_set_0 shape_blocks + 0 * SHAPE_BLOCK_SIZE + SHAPE_BLOCK_PARAM_OFF ; active param set, block 0
%define num_chunks param_set_0 + PARAM_OFFSET_NUM_CHUNKS ; alias: num_chunks
%define taper_shift param_set_0 + PARAM_OFFSET_TAPER_SHIFT ; alias: taper_shift
%define batch_small param_set_0 + PARAM_OFFSET_BATCH_SMALL ; alias: batch_small
%define batch_medium param_set_0 + PARAM_OFFSET_BATCH_MEDIUM ; alias: batch_medium
%define batch_large param_set_0 + PARAM_OFFSET_BATCH_LARGE ; alias: batch_large
%define batch_huge param_set_0 + PARAM_OFFSET_BATCH_HUGE ; alias: batch_huge
%define batch_med_thresh param_set_0 + PARAM_OFFSET_BATCH_MED_THRESH ; alias: batch_med_thresh
%define batch_large_thresh param_set_0 + PARAM_OFFSET_BATCH_LARGE_THRESH ; alias: batch_large_thresh
%define batch_huge_thresh param_set_0 + PARAM_OFFSET_BATCH_HUGE_THRESH ; alias: batch_huge_thresh
%define chunk_multiplier param_set_0 + PARAM_OFFSET_CHUNK_MULTIPLIER ; alias: chunk_multiplier
%define min_chunk_cols param_set_0 + PARAM_OFFSET_MIN_CHUNK_COLS ; alias: min_chunk_cols
%define worker_priority param_set_0 + PARAM_OFFSET_WORKER_PRIORITY ; alias: worker_priority
%define quantize_chunk_groups param_set_0 + PARAM_OFFSET_QUANTIZE_CHUNK_GROUPS ; alias: quantize_chunk_groups
%define caller_batch_small param_set_0 + PARAM_OFFSET_CALLER_BATCH_SMALL ; alias: caller_batch_small
%define caller_batch_medium param_set_0 + PARAM_OFFSET_CALLER_BATCH_MEDIUM ; alias: caller_batch_medium
%define caller_batch_large param_set_0 + PARAM_OFFSET_CALLER_BATCH_LARGE ; alias: caller_batch_large
%define caller_batch_huge param_set_0 + PARAM_OFFSET_CALLER_BATCH_HUGE ; alias: caller_batch_huge
%define caller_batch_med_thresh param_set_0 + PARAM_OFFSET_CALLER_BATCH_MED_THRESH ; alias: caller_batch_med_thresh
%define caller_batch_large_thresh param_set_0 + PARAM_OFFSET_CALLER_BATCH_LARGE_THRESH ; alias: caller_batch_large_thresh
%define caller_batch_huge_thresh param_set_0 + PARAM_OFFSET_CALLER_BATCH_HUGE_THRESH ; alias: caller_batch_huge_thresh
%define chunk_split_pct param_set_0 + PARAM_OFFSET_CHUNK_SPLIT_PCT ; alias: chunk_split_pct
%define chunk_group_1_count param_set_0 + PARAM_OFFSET_CHUNK_GROUP_1_COUNT ; alias: chunk_group_1_count
%define chunk_group_2_count param_set_0 + PARAM_OFFSET_CHUNK_GROUP_2_COUNT ; alias: chunk_group_2_count
%define cores_subtract param_set_0 + PARAM_OFFSET_CORES_SUBTRACT ; alias: cores_subtract
; The C autotuner reads/writes param_set_0 directly through the exported shape_blocks
; symbol, the parameter set has no separate indirection table.
     section .data                             ; initialized data, must NOT be in .bss, linker discards db data there
     align 4
taper_mul_table:
     dd 65536                                  ; f=0: 2^0 = 1.000000
     dd 62757                                  ; f=1: 2^-1/16= 0.957603
     dd 60097                                  ; f=2: 2^-2/16= 0.917004
     dd 57549                                  ; f=3: 2^-3/16= 0.878126
     dd 55109                                  ; f=4: 2^-4/16= 0.840896
     dd 52773                                  ; f=5: 2^-5/16= 0.805245
     dd 50535                                  ; f=6: 2^-6/16= 0.771105
     dd 48393                                  ; f=7: 2^-7/16= 0.738413
     dd 46341                                  ; f=8: 2^-8/16= 0.707107
     dd 44376                                  ; f=9: 2^-9/16 = 0.677128
     dd 42495                                  ; f=10: 2^-10/16 = 0.648420
     dd 40693                                  ; f=11: 2^-11/16 = 0.620929
     dd 38968                                  ; f=12: 2^-12/16 = 0.594604
     dd 37316                                  ; f=13: 2^-13/16 = 0.569394
     dd 35734                                  ; f=14: 2^-14/16 = 0.545254
     dd 34219                                  ; f=15: 2^-15/16 = 0.522137
; The A/B/A tuner stores phase and A/B timings in BSS (all zero initially).
; There is no warmup count, minimum-of-many window or consecutive-win gate.
     section .data                             ; back to initialized data for the MicroDB strings below
; === MicroDB persistent hash table data ===
     section .data                             ; .data for initialized MicroDB strings, must NOT be in .bss, linker discards db data there
; DB path is passed by the caller via V7_init,filename .
; db_save/db_load use g_db_filename / g_db_tmp_filename, BSS buffers set by V7_init .
mdb_readonly:
     dd 0                                      ; 0=readwrite, 1=readonly, db_save is no-op
     global g_chat_eos_token                   ; export not required, but visible for debug tools
g_chat_eos_token:
     dd -1                                     ; eos_token_id, -1 until v7s_chat's check runs
     section .bss align=64
; MicroDB hash table state, 48 bytes
     global g_ht                               ; export g_ht for mdb query tools
g_ht:
     resb HT_SIZE                              ; MicroDB hash table state, HT_SIZE bytes
     align 8                                   ; the qword run below starts on a boundary it declares itself, g_ht's 48-byte tail would otherwise set it
g_inf_transformer:
     resq 1                                    ; Transformer struct block
g_inf_rs_block:
     resq 1                                    ; RunState working-buffer block, KV caches EXCLUDED, they are fbm mappings
g_inf_embed_mxcsr:
     resd 1                                    ; embedding conversion mode captured at context creation
     alignb 8                                  ; align the remaining owning pointers
g_inf_qt_desc:
     resq 1                                    ; QuantizedTensor descriptor array
g_inf_qkv_data:
     resq 1                                    ; fused QKV weight data buffer, [wq|wk|wv] per layer
g_inf_qkv_desc:
     resq 1                                    ; fused QKV QuantizedTensor descriptor block
g_inf_ctx_block:
     resq 1                                    ; the single inference context block, BSS singleton behind the public API
g_raw_generation:
     resd 1                                    ; 1 = raw text continuation, no template, EOS-only stop
     align 8                                   ; g_qkv_bias below is a qword: state the dword/qword boundary instead of inheriting it
g_qkv_bias:
     resq 1                                    ; fp32 QKV bias VIEW ptr, model_weights + k_norm_end ; NOT owned, never freed, points into the DB blob; always valid after init, bias-less models point at an all-zeros block (g_qkv_bias_active=0 skips the adds)
g_qkv_bias_active:
     resd 1                                    ; 1 = bias block has non-zero data, init-time OR-scan in inf_weights -> forward adds it; 0 = all zeros -> skip the add entirely
; KV caches, mdb_alloc_mem blocks : committed on allocation, transparently
; falling back to file-backed mmap on OOM, freed via mdb_free_mem at destroy.
g_inf_key_cache:
     resq 1                                    ; key cache plane base, owning allocation for BOTH packed KV planes, value plane is the interior half
g_inf_kv_scratch:
     resq 1                                    ; interior FP32 K/V staging view inside the RunState working block
g_generate_kv_owner:
     resq 1                                    ; v7s_generate merged KV block, key+value regions, BSS owner: same release-on-caught-FATAL contract
g_stop_id_count:
     resd 1                                    ; number of valid stop token IDs in g_stop_ids array
g_stop_ids:
     resd 4                                    ; up to 4 stop token IDs, dword each
     align 8                                   ; the ten rdtsc/counter slots below are read from C as unsigned __int64[] via v7s_get_generation_statistic, state the boundary, do not inherit it
rdtsc_chat_start:
     resq 1                                    ; [0] TSC at v7s_chat entry
rdtsc_after_tokenization:
     resq 1                                    ; [1] TSC after encode,
rdtsc_after_prefill:
     resq 1                                    ; [2] TSC after prefill phase
rdtsc_after_generation:
     resq 1                                    ; [3] TSC after generation phase
number_token_prefill:
     resq 1                                    ; [4] prefill token count
number_token_generate:
     resq 1                                    ; [5] generated token count
rdtsc_vit_encode_start:
     resq 1                                    ; [6] TSC before ViT encode call
rdtsc_vit_encode_end:
     resq 1                                    ; [7] TSC after ViT encode call
number_token_prefill_vision:
     resq 1                                    ; [8] image-turn prefill tokens
number_token_generate_vision:
     resq 1                                    ; [9] image-turn generated tokens
     align 8
g_thinking_active:
     resb 1                                    ; 1 when v7s_enable_thinking resolved the @think templates; read by v7_get_think_mode_active
     align 4
g_live_gen_count:
     resd 1                                    ; tokens generated THIS v7s_chat turn, live mirror of the gen loop's counter ; zeroed at chat entry, stored once per generated token at the hook site; read by v7_get_gen_count
g_query_base_pos:
     resd 1                                    ; this chat call's ENTRY KV position, snapshotted before the prompt prefill ; GetTokenPosition subtracts it from sl_hook_pos so the primitive's value is local to the query
g_image_token_id:
     resd 1                                    ; image placeholder token id, 151655 for Qwen3-VL ; 0 = vision override inactive
g_model_image_token_id:
     resd 1                                    ; model's persistent image placeholder token id, set at init ; NEVER cleared per-session
g_cfg_has_deepstack:
     resd 1                                    ; 1 = apply DeepStack adds in decoder layers 0/1/2, post-residual
g_vision_embeddings:
     resq 1                                    ; float* projected image embeddings [n_merged * dim]
g_vision_emb_count:
     resd 1                                    ; number of projected image embeddings, n_merged
g_vision_emb_consumed:
     resd 1                                    ; running index of next embedding to inject, reset per v7s_chat_image_mem
g_deepstack_features:
     resq 3                                    ; float* deepstack feature tensors [3][n_merged * dim]
%define VDS_SLOT0 0                            ; byte offset of ds0
%define VDS_SLOT1 8                            ; byte offset of ds1
%define VDS_SLOT2 16                           ; byte offset of ds2
%define VDS_PTR_STRIDE 8                       ; qword stride between slots, 8 bytes per pointer
g_vis_prompt_prefix:
     resq 1                                    ; pointer to prefix string inside the BSS copy, e.g. "<|vision_start|>" or ""
g_vis_prompt_marker:
     resq 1                                    ; pointer to marker string inside the BSS copy, e.g. "<|image_pad|>" or "<image>"
g_vis_prompt_suffix:
     resq 1                                    ; pointer to suffix string inside the BSS copy, e.g. "<|vision_end|>" or "\n"
g_vis_prompt_marker_len:
     resd 1                                    ; strlen,marker precomputed at init, avoids per-call scasb in EMIT_IMAGE_PADS
     VISION_PROMPT_BUF_CAP equ 256             ; BSS copy capacity for the vision prompt format blob
g_vis_prompt_buf:
     resb VISION_PROMPT_BUF_CAP                ; BSS copy of __sys_vision_prompt_fmt, pool pointers dangle on runtime pool growth via V7-DB-PUT
; Image-turn prompt template, extended_prompt_plan.md : models whose HF chat
; template renders image turns differently from text turns, SmolVLM/Idefics3:
; "User:" + images, no space, no newline; LLaVA-OV: no baked system block
; store the image-turn variant under __sys_template@img. Init copies it here;
; set by the chat_image wrappers. Absent key -> valid stays 0 -> base template.
g_img_turn:
     resd 1                                    ; 1 = the current prompt is an image turn, chat_image wrappers set it
g_img_template_valid:
     resd 1                                    ; 1 = g_img_template holds an image-turn template, from __sys_template@img
g_img_template:
     resb 1024                                 ; image-turn prompt template, sprintf %s ; valid only when g_img_template_valid != 0
g_think_template:
     resb 1024                                 ; user-turn thinking template, sprintf %s ; empty = model has none
g_think_sys_template:
     resb 1024                                 ; system+user thinking template, sprintf %s %s ; empty = none
; Inter-turn templates, phase4.md : __sys_template@multi / @multi@think hold
; the COMPLETE turn-2+ user templates, inter-turn terminator + user header +
; %s + generation suffix.Derived by sentinel-diff in export.py; stored only
; when they differ from base. Empty first byte = absent = turn 2+ uses g_tpl_t1.
g_multi_template:
     resb 1024                                 ; inter-turn template @multi; empty = fall back to turn-1 template
g_multi_think_template:
     resb 1024                                 ; inter-turn template @multi@think; empty = same as @multi
; EFFECTIVE template pointer table, phase4.md : v7s_enable_thinking resolves
; ALL model dimensions, thinking x multi-turn into three pointers ONCE, at
; flag-set / context-create time; the chat render path just loads them --
; zero dimension logic in the per-call path. Bases are cached at init from
; the Tokenizer struct and double as the init-order guard, NULL = init has
; not run yet; the selector FATALs on that caller contract violation .
g_tpl_t1:
     resq 1                                    ; effective turn-1 user template, think folded in
g_tpl_sys:
     resq 1                                    ; effective system+user template, think folded in
g_tpl_multi:
     resq 1                                    ; effective turn-2+ user template, multi+think folded; = g_tpl_t1 when @multi absent
g_tpl_t1_base:
     resq 1                                    ; cached &tok->prompt_template, selector fallback + init-order guard
g_tpl_sys_base:
     resq 1                                    ; cached &tok->system_template, selector fallback
g_tpl_multi_base:
     resq 1                                    ; cached &g_multi_template buffer, selector fallback for the @multi dimension
; Current-turn user prompt pointer, RAG query augmentation, Aug 2026 : v7s_chat
; stores the raw user_prompt argument here BEFORE template rendering so the
; SectorLogo sidecar can fetch it via v7s_get_user_prompt, C-CALL PTR return
; and prepend/append it to BM25 queries. NULL outside an active chat turn.
g_user_prompt:
     resq 1                                    ; char* user prompt of the current v7s_chat call
g_vit_cfg:
     resq 93                                   ; precomputed ViT config, 93 used qwords, 0 spare
g_vit_cfg_len:
     resq 1                                    ; out_len scratch for init ht_find calls, vision-config / prompt-fmt / newline-emb lengths; never validated, builder contract
g_anyres_newline_emb:
     resq 1                                    ; pointer to image_newline BSS copy, NULL if no anyres
g_anyres_pinpoints:
     resd 72                                   ; 36 x, h,w pairs, filled at init, fixed for LLaVA-OV
     IMAGE_NEWLINE_BUF_CAP equ 65536           ; BSS copy capacity for the image newline embedding row
g_anyres_newline_buf:
     resb IMAGE_NEWLINE_BUF_CAP                ; BSS copy of __sys_image_newline_emb, text_hidden*4 <= 64KB for any supported model
g_ar_best_h:
     resd 1                                    ; selected best resolution height
g_ar_best_w:
     resd 1                                    ; selected best resolution width
g_ar_grid_h:
     resd 1                                    ; grid rows, best_h / 384
g_ar_grid_w:
     resd 1                                    ; grid cols, best_w / 384
g_ar_n_tiles:
     resd 1                                    ; total tiles, grid_h * grid_w + 1
g_ar_tile_idx:
     resd 1                                    ; current tile index, 0 = base, 1+ = grid
g_ar_resized_buf:
     resq 1                                    ; resized image buffer, best_w x best_h x 3
g_ar_tile_rgb:
     resq 1                                    ; per-tile 384x384 RGB buffer
g_ar_tile_res:
     resq 1                                    ; per-tile normalized planar float buffer
g_ar_tile_patches:
     resq 1                                    ; per-tile patch features buffer
g_ar_tile_vit_out:
     resq 1                                    ; per-tile ViT output buffer
g_ar_proj_inter:
     resq 1                                    ; per-tile projector FC1 intermediate buffer
g_ar_all_features:
     resq 1                                    ; all tiles' projected features
g_ar_final_emb:
     resq 1                                    ; final arranged embedding buffer
g_ar_final_cnt:
     resd 1                                    ; final token count
g_vit_scratch_block:
     resq 1                                    ; base of contiguous scratch block, NULL if text-only
g_vit_ptr_hidden:
     resq 1                                    ; hidden state [n * hidden]
g_vit_ptr_qkv:
     resq 1                                    ; qkv buffer [n * qkv_dim]
g_vit_ptr_norm:
     resq 1                                    ; normalized buffer [n * hidden]
g_vit_ptr_mlp:
     resq 1                                    ; mlp intermediate [n * inter]
g_vit_ptr_scores:
     resq 1                                    ; attention scores [heads * ALIGN64,n ], per-head slice is n floats, not n*n
g_vit_ptr_attn:
     resq 1                                    ; attention output [n * hidden]
g_vit_ptr_pos:
     resq 1                                    ; pos embeddings [n * hidden]
g_vit_ptr_cos:
     resq 1                                    ; RoPE cos table [n * head_dim]
g_vit_ptr_sin:
     resq 1                                    ; RoPE sin table [n * head_dim]
g_vit_ptr_packed:
     resq 1                                    ; packed 2x2 patches [n_merged * packed_dim]
g_vit_ptr_mmlp:
     resq 1                                    ; merger mlp intermediate [n_merged * packed_dim]
; ---- ViT dispatch globals, set by v7d_vit_linear before DISPATCH_EXECUTE ----
g_vit_k:
     resd 1                                    ; inner dimension K for current ViT linear dispatch
g_vit_m:
     resd 1                                    ; output dimension M for ViT linear stride
g_vit_residual:
     resq 1                                    ; fused residual pointer, NULL=disabled; vit_forward sets to hidden before proj/FC2, read by v7s_vit_linear_chunk
     global vit_max_chunks                     ; export for the in-process tuner
vit_max_chunks:
     resd 1                                    ; P4 chunk-count cap, 0 before init; >= workers+1 enforced in dispatcher
     global vit_min_chunk_cols                 ; export for the in-process tuner
vit_min_chunk_cols:
     resd 1                                    ; minimum output columns per chunk, M / this = hard chunk-count ceiling
; V7_init filename buffers, set by V7_init, used by db_save/db_load when called with NULL
g_db_filename:
     resb 512                                  ; null-terminated DB filename, set by V7_init ; must fit MAX_PATH+margin
g_db_tmp_filename:
     resb 521                                  ; temp filename = g_db_filename + ".tmp" + NUL, max 512+5
