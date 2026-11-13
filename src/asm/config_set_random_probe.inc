; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; config_set_random_probe.inc -- matmul config block randomization and
; mutation, both fully random, always-legal, single pass.

; config_set_random_probe fills one 512-byte shape block with a random
; config: every matmul tuning dword comes from a .rdata bounds table
; (mirror of the C tuner's tune_min/tune_max), the threshold triples are
; repaired in place, the taper and split-fill legality rules of the C
; sanitizer are applied, and pf_dist/pf_type are drawn over the full legal
; window. There is no retry loop and no if-cascade: every draw lands
; inside its bounds by construction, and absence from the bounds table is
; what leaves a field alone (block n/d, worker_priority, cores_subtract,
; the block tail -- none of them are matmul tuning state).

; config_mutate_genome is the ASM analog of the C tuner's
; get_mutated_genome: copy parent1 to out, mutate 1..g_probe_mutations_max
; matmul fields, then let parent2 donate a percent of its dwords. The C
; semantics are
; kept exactly: transfers draw slots WITH replacement and the transfer
; count draws from 0..budget-1, so a 100% setting donates about 51.5
; draws = ~39% distinct fields on average and can never force all 104 --
; same as the C.

; Randomness: a local xorshift64 whose seed lives in .bss and is
; perturbed with the TSC at every call of either entry point, so no
; initialization exists or is needed. The seed update is unsynchronized:
; the caller contract is one thread in these functions at a time. Three
; macros carry all randomness and ranging; ranging is one multiply-high
; per draw (uniform over [0, range-1], no modulo, no rejection).

; config_mutate_genome requires out to NOT alias parent2: phase A writes
; out from parent1 before phase C reads parent2, so out == parent2 would
; donate the already-mutated child. out == parent1 is fine (the copy is
; a self-copy and every later write is deliberate).

; Callers: cycle_tuner_main seeds every fresh 64K population through
; config_set_random_probe and mutates genomes through config_mutate_genome.
; Both functions stay live in every tuner-enabled build.
; ============================================================================

%define PROBE_RNG_TSC_MULT 123456789           ; TSC perturbation multiplier, odd, folds the cycle counter into the seed
%define PROBE_PARAM_ENTRIES 22                 ; tuned matmul dwords in the bounds table = all of them except worker_priority and cores_subtract
%define PROBE_PF_SLOTS 41                      ; pf_dist/pf_type slot count, matches PF_DIST_TABLE_ENTRIES
%define PROBE_PF_TYPE_RANGE 7                  ; prefetch type count: types 0..6 are legal
%define PROBE_HALF_CAP 2147483647              ; pf window cap: half the weight matrix must stay inside the signed disp32 budget
%define PROBE_WINDOW_EXTRA 1                   ; pf window = 2*half + this, odd so the multiply-high draw covers [0, 2*half] exactly
%define PROBE_BOUNDS_OFF 0                     ; bounds-table entry offset: param_set byte offset
%define PROBE_BOUNDS_LO 4                      ; bounds-table entry offset: inclusive lower bound
%define PROBE_BOUNDS_HI 8                      ; bounds-table entry offset: inclusive upper bound
%define PROBE_BOUNDS_STRIDE 12                 ; bytes per bounds-table entry: three dwords
%define PROBE_DWORD_STRIDE 4                   ; bytes per dword slot, pf arrays walk with this
%define PROBE_TRIPLE_MID 4                     ; threshold-triple dword offset: middle boundary
%define PROBE_TRIPLE_HI 8                      ; threshold-triple dword offset: upper boundary
%define PROBE_TRIPLE_STRIDE 44                 ; block offset distance between the two threshold triples
%define PROBE_FIELD_TOTAL 104                  ; total mutatable matmul dwords in one block: 22 params + 41 pf_dist + 41 pf_type
%define PROBE_NUDGE_DENOM 4                    ; nudge/redraw decision range: draws 0..3, < 3 = nudge
%define PROBE_DISTANCE_KINDS 5                 ; distance mutation kinds: uniform, x2, halve, signed ratio, positive ratio
%define PROBE_JITTER_RANGE 65                  ; kind-3 jitter draw bound: 0..64
%define PROBE_JITTER_HALF 32                   ; kind-3 jitter center: draw - 32 lands on -32..+32
%define PROBE_RANGE_INC 1                      ; +1 turns the capped half-window into the inclusive draw range
%define PROBE_DIV100_RECIP 0x51EB851F          ; reciprocal for exact unsigned 32-bit division by 100
%define PROBE_DIV100_SHIFT 5                   ; high product >> 5 equals numerator / 100
%define PROBE_PCT_DENOM 100                    ; split percent denominator: g1_cols = d * pct / 100
%define PROBE_COPY_QWORDS 64                   ; full block copy size: 512 bytes = 64 qwords, out is completely defined after the copy
%define PROBE_TAPER_MIN_COLS 4                 ; the taper's 4-row kernel block: taper-on candidates need min_chunk_cols >= 4

; Layout-sync guards: this file's constants must never drift from the
; shape-block layout they mirror.
%if PROBE_PF_SLOTS != PF_DIST_TABLE_ENTRIES
%error PROBE_PF_SLOTS out of sync with PF_DIST_TABLE_ENTRIES ; must match the engine's slot table
%endif
%if PROBE_COPY_QWORDS*8 != SHAPE_BLOCK_SIZE
%error PROBE_COPY_QWORDS must copy the whole block ; equality, not just a lower bound
%endif
%if PROBE_FIELD_TOTAL != PROBE_PARAM_ENTRIES + 2*PROBE_PF_SLOTS
%error PROBE_FIELD_TOTAL mismatches the field split ; 22 params + 41 + 41
%endif

     section .bss align=8
     global g_probe_sex_percent                ; crossover knob, exported for tooling: donor dword percent 0..100
g_probe_rng_seed:
     resq 1                                    ; xorshift64 state, no init: the TSC perturbation at every call mixes it
     alignb 8                                  ; g_probe_sex_percent lands on its own qword boundary after the seed
g_probe_sex_percent:
     resd 1                                    ; crossover knob for config_mutate_genome: donor dword percent 0..100, 0 = asexual, default 0
     global g_probe_mutations_max              ; mutation count knob, published by the cycle tuner's DB-key read
g_probe_mutations_max:
     resd 1                                    ; mutation count draw bound for config_mutate_genome: 1..max fields changed per child, default 0 until the tuner publishes

     section .rdata
; One 12-byte triple per tuned matmul dword: {param_set byte offset, lo, hi}.
; Bounds mirror genome_sort_experiment.cpp tune_min/tune_max exactly.
; Offsets absent from this table (44 worker_priority, 92 cores_subtract) are
; never written by config_set_random_probe or config_mutate_genome.
probe_param_bounds:
     dd 0, 1, 4096                             ; num_chunks
     dd 4, 0, 112                              ; taper_shift
     dd 8, 1, 64                               ; batch_small
     dd 12, 1, 64                              ; batch_medium
     dd 16, 1, 64                              ; batch_large
     dd 20, 1, 64                              ; batch_huge
     dd 24, 1, 4096                            ; batch_med_thresh, triple member
     dd 28, 1, 4096                            ; batch_large_thresh, triple member
     dd 32, 1, 4096                            ; batch_huge_thresh, triple member
     dd 36, 1, 128                             ; chunk_multiplier
     dd 40, 1, 64                              ; min_chunk_cols
     dd 48, 1, 64                              ; quantize_chunk_groups
     dd 52, 1, 64                              ; caller_batch_small
     dd 56, 1, 64                              ; caller_batch_medium
     dd 60, 1, 64                              ; caller_batch_large
     dd 64, 1, 64                              ; caller_batch_huge
     dd 68, 1, 4096                            ; caller_batch_med_thresh, triple member
     dd 72, 1, 4096                            ; caller_batch_large_thresh, triple member
     dd 76, 1, 4096                            ; caller_batch_huge_thresh, triple member
     dd 80, 0, 99                              ; chunk_split_pct
     dd 84, 1, 256                             ; chunk_group_1_count
     dd 88, 1, 256                             ; chunk_group_2_count

     section .text
; PROBE_NEXT_RND -- one xorshift64 step: new state in rax AND in the .bss
; seed. Clobbers rax/rdx only; every other register survives.

%macro PROBE_NEXT_RND 0
     mov rax, [rel g_probe_rng_seed]           ; rax = current xorshift64 state
     mov rdx, rax                              ; rdx = working copy for the shift
     shl rdx, 13                               ; x << 13
     xor rax, rdx                              ; x ^= x << 13
     mov rdx, rax                              ; rdx = working copy for the shift
     shr rdx, 7                                ; x >> 7
     xor rax, rdx                              ; x ^= x >> 7
     mov rdx, rax                              ; rdx = working copy for the shift
     shl rdx, 17                               ; x << 17
     xor rax, rdx                              ; x ^= x << 17, state fully mixed
     mov [rel g_probe_rng_seed], rax           ; publish the new state
%endmacro

; PROBE_SEED_TSC -- perturb the .bss seed with the TSC. Both entry points
; run this once so a mutate-first call cannot inherit an all-zero .bss
; state (zero is the absorbing state of xorshift64). The TSC fold landing
; on exactly zero is practically impossible and accepted; the seed update
; itself is unsynchronized -- one caller thread at a time.
; Clobbers rax/rdx.

%macro PROBE_SEED_TSC 0
     rdtsc                                     ; rax = TSC low, rdx = TSC high
     xor rax, rdx                              ; fold both TSC halves
     imul rax, rax, PROBE_RNG_TSC_MULT         ; TSC * 123456789
     add [rel g_probe_rng_seed], rax           ; perturb the persistent seed
%endmacro

; PROBE_RND_BELOW reg64 -- fresh draw scaled into edx = uniform [0, reg-1]
; via the high 64 bits of the 64x64 product. reg < 2^32 keeps the high
; product inside [0, reg-1] already: rdx IS the bounded draw, no shift
; after the multiply. Clobbers rax/rdx.

%macro PROBE_RND_BELOW 1
     PROBE_NEXT_RND                            ; rax = fresh 64-bit state
     mul %1                                    ; rdx = uniform [0, reg-1] for reg < 2^32
%endmacro

; PROBE_SORT_TRIPLE blockreg, block_offset -- sort the three threshold
; dwords at block_offset into ascending order with a 3-step min/max cmov
; network. All bounds are >= 1 by construction, so the sorted order is the
; legal triple; the engine's degeneration contract needs nothing more.
; Clobbers eax/ecx/edx/r8d.

%macro PROBE_SORT_TRIPLE 2
     mov eax, [%1 + %2]                        ; a
     mov ecx, [%1 + %2 + PROBE_TRIPLE_MID]     ; b
     cmp eax, ecx                              ; order a,b: eax = smaller after the cmovs
     mov r8d, eax                              ; r8d = min(a,b) candidate
     cmovg r8d, ecx                            ; a > b: min is b
     cmovg ecx, eax                            ; a > b: max is a, else b stays max
     mov eax, r8d                              ; eax = min(a,b)
     mov edx, [%1 + %2 + PROBE_TRIPLE_HI]      ; c
     cmp ecx, edx                              ; order max(a,b), c: middle and largest
     mov r8d, ecx                              ; r8d = middle candidate
     cmovg r8d, edx                            ; max > c: middle is c
     cmovg edx, ecx                            ; max > c: largest is max, else c stays largest
     mov ecx, r8d                              ; ecx = middle candidate
     cmp eax, ecx                              ; order min, middle candidate: smallest and middle
     mov r8d, eax                              ; r8d = smallest candidate
     cmovg r8d, ecx                            ; min > candidate: smallest is candidate
     cmovg ecx, eax                            ; min > candidate: middle is min, else candidate stays middle
     mov eax, r8d                              ; eax = smallest
     mov [%1 + %2], eax                        ; store ascending: lo
     mov [%1 + %2 + PROBE_TRIPLE_MID], ecx     ; mid
     mov [%1 + %2 + PROBE_TRIPLE_HI], edx      ; hi
%endmacro

; PROBE_FIX_TAPER_MCC blockreg -- taper-on candidates need
; min_chunk_cols >= 4 (the taper's 4-row kernel block), the C sanitizer's
; rule; taper-off blocks keep whatever min_chunk_cols they carry.
; Clobbers ecx/edx.

%macro PROBE_FIX_TAPER_MCC 1
     mov ecx, [%1 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_TAPER_SHIFT] ; taper_shift
     test ecx, ecx                             ; taper off: no column minimum
     jz %%mcc_done                             ; keep min_chunk_cols as drawn
     mov ecx, [%1 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_MIN_CHUNK_COLS] ; min_chunk_cols
     mov edx, PROBE_TAPER_MIN_COLS             ; the 4-row minimum
     cmp ecx, edx                              ; below 4?
     cmovl ecx, edx                            ; normalize up, exactly the C sanitizer
     mov [%1 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_MIN_CHUNK_COLS], ecx ; store
%%mcc_done:
%endmacro

; PROBE_FIX_SPLIT blockreg -- the C sanitizer's split-fill rules: group
; counts must stay within their column budgets or the split path fails on
; a zero-column chunk. d comes from the block's own header; d == 0 turns
; the split off (zeroed header, no columns exist to split). Clobbers
; eax/ecx/edx/r8d ONLY: both callers keep live values in r9-r11 and the
; non-volatiles, so no other register may be touched. All division
; operands fit 32 bits (d <= 65536, pct <= 99, d*pct <= ~6.5M), so the
; 32-bit div forms run.

%macro PROBE_FIX_SPLIT 1
     mov ecx, [%1 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_CHUNK_SPLIT_PCT] ; split percent
     test ecx, ecx                             ; split disabled: nothing to fix
     jz %%split_done                           ; keep group counts as drawn
     mov eax, [%1 + SHAPE_BLOCK_D_EQU]         ; eax = d
     test eax, eax                             ; d == 0?
     jz %%kill_split                           ; no columns, no legal split
     imul eax, ecx                             ; eax = d * pct
     mov r8d, PROBE_PCT_DENOM                  ; 100
     xor edx, edx                              ; dividend = edx:eax
     div r8d                                   ; eax = g1_cols = d*pct/100
     cmp eax, 1                                ; g1_cols below one column?
     jge %%have_pct                            ; no: keep the drawn percent
     mov r8d, [%1 + SHAPE_BLOCK_D_EQU]         ; r8d = d, d > 0 already checked at macro entry
     mov eax, PROBE_PCT_DENOM - 1              ; 99
     xor edx, edx                              ; dividend = edx:eax
     div r8d                                   ; eax = floor(99/d)
     inc eax                                   ; eax = min_pct = ceil(100/d)
     cmp eax, 99                               ; min_pct beyond legal [1..99]?
     ja %%kill_split                           ; tiny d: no legal split exists
     mov [%1 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_CHUNK_SPLIT_PCT], eax ; pct = min_pct, the smallest that gives group 1 a column
     mov eax, 1                                ; g1_cols = floor(d * ceil(100/d) / 100) = 1 here
%%have_pct:
; g1_cols >= 1 here: the entry check passed or min_pct just fixed it;
; pct <= 99 also keeps g1_cols < d, so group 2 keeps >= 1 column
     mov ecx, [%1 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_CHUNK_GROUP_1_COUNT] ; g1
     test ecx, ecx                             ; zero or negative?
     jg %%g1_positive                          ; keep the drawn count
     mov ecx, 1                                ; floor at one chunk
%%g1_positive:
     cmp ecx, eax                              ; g1 above its column budget?
     cmovg ecx, eax                            ; each g1 chunk keeps >= 1 col
     mov [%1 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_CHUNK_GROUP_1_COUNT], ecx ; store g1
     mov ecx, [%1 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_CHUNK_GROUP_2_COUNT] ; g2
     test ecx, ecx                             ; zero or negative?
     jg %%g2_positive                          ; keep the drawn count
     mov ecx, 1                                ; floor at one chunk
%%g2_positive:
     mov edx, [%1 + SHAPE_BLOCK_D_EQU]         ; edx = d
     sub edx, eax                              ; edx = remaining = d - g1_cols
     cmp ecx, edx                              ; g2 above the remaining budget?
     cmovg ecx, edx                            ; each g2 chunk keeps >= 1 col
     mov [%1 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_CHUNK_GROUP_2_COUNT], ecx ; store g2
     jmp %%split_done                          ; split fixed
%%kill_split:
     mov dword [%1 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_CHUNK_SPLIT_PCT], 0 ; no valid split: turn the split off
%%split_done:
%endmacro

; ---------------------------------------------------------------------------
; config_set_random_probe -- randomize the matmul tuning fields of one
; 512-byte shape block into a valid config, one pass, no retries.
; IN: rcx = block pointer (SHAPE_BLOCK_SIZE bytes).
; Clobbers rax/rcx/rdx/r8/r9/r10 only; pure leaf, no calls, no frame.
; ---------------------------------------------------------------------------
     global config_set_random_probe            ; random-probe filler: rcx = 512-byte shape block, mutates all matmul tuning fields
config_set_random_probe:
     mov r10, rcx                              ; r10 = block base, survives all draws
     PROBE_SEED_TSC                            ; perturb the persistent seed

; --- 22 tuned matmul dwords, straight from the bounds table ---
%assign PROBE_OFF 0                            ; table cursor in bytes
%rep PROBE_PARAM_ENTRIES
     mov r8d, [rel probe_param_bounds + PROBE_OFF + PROBE_BOUNDS_OFF] ; r8d = param_set byte offset for this field
     mov r9d, [rel probe_param_bounds + PROBE_OFF + PROBE_BOUNDS_LO] ; r9d = lo
     mov ecx, [rel probe_param_bounds + PROBE_OFF + PROBE_BOUNDS_HI] ; ecx = hi
     sub ecx, r9d                              ; ecx = hi - lo
     inc ecx                                   ; ecx = range = hi - lo + 1
     PROBE_RND_BELOW rcx                       ; edx = uniform [0, range-1]
     add edx, r9d                              ; edx = lo + draw, inside [lo, hi] by construction
     mov [r10 + r8 + SHAPE_BLOCK_PARAM_OFF], edx ; store the tuned dword; the table offset is param_set-relative, the block wants +8
%assign PROBE_OFF PROBE_OFF+PROBE_BOUNDS_STRIDE ; next bounds entry
%endrep

; --- repair both ascending threshold triples (batch at param dword 6,
; caller at param dword 17), then the taper and split legality rules ---
%assign PROBE_TRIPLE SHAPE_BLOCK_PARAM_OFF + 24 ; block offset of the batch threshold triple
%rep 2
     PROBE_SORT_TRIPLE r10, PROBE_TRIPLE       ; sort the three drawn dwords ascending
%assign PROBE_TRIPLE PROBE_TRIPLE+PROBE_TRIPLE_STRIDE ; next triple: caller thresholds at param dword 17
%endrep
     PROBE_FIX_TAPER_MCC r10                   ; taper-on: min_chunk_cols >= 4
     PROBE_FIX_SPLIT r10                       ; split group counts within their column budgets

; --- pf_dist: uniform signed draw inside +-(half) of the block's own
; weight matrix, half = (n*d)/2 capped at the disp32 budget. window =
; 2*half+1 precomputed once; per slot it is draw-subtract-store. ---
     mov eax, [r10]                            ; n, zero-extends to 64
     mov ecx, [r10 + SHAPE_BLOCK_D_EQU]        ; d, zero-extends to 64
     imul rax, rcx                             ; rax = n*d, 64-bit
     shr rax, 1                                ; rax = half
     mov r9, PROBE_HALF_CAP                    ; disp32 budget cap
     cmp rax, r9                               ; half beyond the budget?
     cmova rax, r9                             ; half = min(half, cap)
     mov r8d, eax                              ; r8d = half, dword for the per-slot subtract
     add rax, rax                              ; 2*half
     lea r9, [rax + PROBE_WINDOW_EXTRA]        ; r9 = window = 2*half+1
%assign PROBE_OFF SHAPE_BLOCK_PF_DIST_OFF      ; first pf_dist slot
%rep PROBE_PF_SLOTS
     PROBE_RND_BELOW r9                        ; edx = uniform [0, window-1]
     sub edx, r8d                              ; edx = signed [-half, +half]: backward, zero, and the [0..31] disabled band all legal
     mov [r10 + PROBE_OFF], edx                ; store the distance
%assign PROBE_OFF PROBE_OFF+PROBE_DWORD_STRIDE ; next dword slot
%endrep

; --- pf_type: uniform draw over the 7 legal types, rcx holds the range for
; the whole loop ---
     mov ecx, PROBE_PF_TYPE_RANGE              ; rcx = 7, draws land in [0, 6]
%assign PROBE_OFF SHAPE_BLOCK_PF_TYPE_OFF      ; first pf_type slot
%rep PROBE_PF_SLOTS
     PROBE_RND_BELOW rcx                       ; edx = uniform [0, 6]
     mov [r10 + PROBE_OFF], edx                ; store the type
%assign PROBE_OFF PROBE_OFF+PROBE_DWORD_STRIDE ; next dword slot
%endrep
     ret                                       ; block fully mutated, still a legal config

; ---------------------------------------------------------------------------
; config_mutate_genome -- ASM analog of the C tuner's get_mutated_genome
; (genome_cycle_sort.cpp): copy parent1 to out, mutate 1..g_probe_mutations_max
; matmul fields
; (75% nudge / 25% redraw on params, 5-kind distance mutation on pf_dist,
; uniform redraw on pf_type), then let parent2 donate
; g_probe_sex_percent percent of its dwords. Out is always a valid config:
; every draw lands inside its bounds by construction, the threshold
; triples are repaired after each phase, and the taper/split rules run
; after both phases.
; IN: rcx = parent1_in, rdx = parent2_in, r8 = generated_genome_out.
; out must NOT alias parent2 (phase C reads parent2 after phase A wrote
; out); out == parent1 is fine. Transfers draw slots with replacement and
; the count draws from 0..budget-1, matching the C exactly.
; Clobbers rax/rcx/rdx/r8/r9/r10/r11; saves rbx/rsi/rdi/r12/r13/r14. No calls.
; ---------------------------------------------------------------------------
     global config_mutate_genome               ; three-pointer mutation engine: rcx = parent1, rdx = donor parent2, r8 = generated out
config_mutate_genome:
     push rbx                                  ; save non-volatile, draw scratch
     push rsi                                  ; save non-volatile, bounds entry / pf offset pointer
     push rdi                                  ; save non-volatile, bounds table base
     push r12                                  ; save non-volatile, mutation + transfer counter
     push r13                                  ; save non-volatile, half for the whole call
     push r14                                  ; save non-volatile, kind-3 ratio park
     mov r10, r8                               ; r10 = out block base
     mov r11, rdx                              ; r11 = donor base, survives the whole body
     PROBE_SEED_TSC                            ; perturb the persistent seed
; --- Phase A: out = parent1 over the whole 512-byte block, one qword
; copy; out is completely defined afterwards ---
     mov rsi, rcx                              ; rsi = parent1 base
     mov rdi, r10                              ; rdi = out base
     mov ecx, PROBE_COPY_QWORDS                ; 64 qwords = 512 bytes: the whole block
     rep movsq                                 ; out block = parent1 block
     lea rdi, [rel probe_param_bounds]         ; rdi = bounds table base, entry addressing below
; --- pf window from the block's own n/d, computed once ---
     mov eax, [r10]                            ; n, zero-extends to 64
     mov ecx, [r10 + SHAPE_BLOCK_D_EQU]        ; d, zero-extends to 64
     imul rax, rcx                             ; rax = n*d, 64-bit
     shr rax, 1                                ; rax = half
     mov r9, PROBE_HALF_CAP                    ; disp32 budget cap
     cmp rax, r9                               ; half beyond the budget?
     cmova rax, r9                             ; half = min(half, cap)
     mov r13d, eax                             ; r13d = half for the whole call
     add rax, rax                              ; 2*half
     lea r9, [rax + PROBE_WINDOW_EXTRA]        ; r9 = window = 2*half+1
; --- mutation count 1..g_probe_mutations_max (the tuner publishes the
; bound from the DB at startup) ---
     mov ecx, [rel g_probe_mutations_max]      ; draw below the configured max
     PROBE_RND_BELOW rcx                       ; edx = 0..max-1
     inc edx                                   ; 1..max
     mov r12d, edx                             ; r12d = remaining mutations
.mgx_mutation_loop:
     mov ecx, PROBE_FIELD_TOTAL                ; field space 0..103
     PROBE_RND_BELOW rcx                       ; edx = field pick
     cmp edx, PROBE_PARAM_ENTRIES              ; param slot?
     jae .mgx_pf                               ; no: pf_dist or pf_type
; --- param mutation: bounds entry = table + slot*PROBE_BOUNDS_STRIDE, one
; legacy-encoded imul, no scale addressing ---
     imul rcx, rdx, PROBE_BOUNDS_STRIDE        ; rcx = entry byte offset
     lea rsi, [rdi + rcx]                      ; rsi = bounds entry
     mov ecx, PROBE_NUDGE_DENOM                ; draw below 4
     PROBE_RND_BELOW rcx                       ; edx = 0..3
     cmp edx, 3                                ; the 25% redraw draw
     jae .mgx_redraw                           ; redraw wins a quarter of the time
; --- 75% nudge: step = draw1/draw2 over 1..span, sign-flipped by a coin
; bit, cur + step clamped into [lo, hi] ---
     mov eax, [rsi + PROBE_BOUNDS_LO]          ; lo
     mov ecx, [rsi + PROBE_BOUNDS_HI]          ; hi
     sub ecx, eax                              ; hi - lo
     inc ecx                                   ; ecx = span
     PROBE_RND_BELOW rcx                       ; edx = 0..span-1
     inc edx                                   ; edx = draw1, 1..span
     mov ebx, edx                              ; ebx = draw1, parked
     PROBE_NEXT_RND                            ; rax = fresh state for draw2
     mul rcx                                   ; rdx = draw2 = uniform [0, span-1], the high product IS the draw
     inc edx                                   ; edx = draw2, 1..span
     mov r8d, edx                              ; r8d = draw2, nonzero divisor
     mov eax, ebx                              ; eax = draw1
     xor edx, edx                              ; dividend = edx:eax = draw1
     div r8d                                   ; eax = step = draw1/draw2, u32 exact: both draws are 1..span
     mov ebx, eax                              ; ebx = step, parked BEFORE the coin draw clobbers rax
     PROBE_NEXT_RND                            ; rax = fresh state, bit 63 = coin
     mov ecx, ebx                              ; ecx = step
     neg ecx                                   ; ecx = -step
     test rax, rax                             ; SF = coin bit 63 of the random state
     cmovs ebx, ecx                            ; ebx = signed step
     mov r8d, [rsi + PROBE_BOUNDS_OFF]         ; param_set byte offset
     add ebx, [r10 + r8 + SHAPE_BLOCK_PARAM_OFF] ; nudged = cur + step
     mov eax, [rsi + PROBE_BOUNDS_LO]          ; lo
     mov ecx, [rsi + PROBE_BOUNDS_HI]          ; hi
     cmp ebx, eax                              ; below the floor?
     cmovl ebx, eax                            ; clamp low
     cmp ebx, ecx                              ; above the ceiling?
     cmovg ebx, ecx                            ; clamp high
     mov [r10 + r8 + SHAPE_BLOCK_PARAM_OFF], ebx ; store the nudged dword
     jmp .mgx_tail                             ; mutation done
.mgx_redraw:
     mov r8d, [rsi + PROBE_BOUNDS_OFF]         ; param_set byte offset
     mov ecx, [rsi + PROBE_BOUNDS_HI]          ; hi
     sub ecx, [rsi + PROBE_BOUNDS_LO]          ; hi - lo
     inc ecx                                   ; ecx = span
     PROBE_RND_BELOW rcx                       ; edx = 0..span-1
     add edx, [rsi + PROBE_BOUNDS_LO]          ; edx = uniform [lo, hi]
     mov [r10 + r8 + SHAPE_BLOCK_PARAM_OFF], edx ; store the redrawn dword
     jmp .mgx_tail                             ; mutation done
; --- pf branches: slot -> block offset in esi ---
.mgx_pf:
     cmp edx, PROBE_PARAM_ENTRIES + PROBE_PF_SLOTS ; pf_type range?
     jae .mgx_pf_type                          ; high half: pf_type
; --- pf_dist mutation: the 5 kinds of the C mutate_distance, cur in eax,
; half in r13d, window in r9, result in edx ---
     sub edx, PROBE_PARAM_ENTRIES              ; edx = pf slot 0..40
     imul esi, edx, PROBE_DWORD_STRIDE         ; esi = slot byte displacement
     add esi, SHAPE_BLOCK_PF_DIST_OFF          ; esi = block offset of the slot
     mov ecx, PROBE_DISTANCE_KINDS             ; draw below 5
     PROBE_RND_BELOW rcx                       ; edx = kind 0..4
     mov eax, [r10 + rsi]                      ; eax = cur
     test edx, edx                             ; kind 0?
     jz .mgx_kind0                             ; uniform window draw
     cmp edx, 1                                ; kind 1?
     je .mgx_kind1                             ; double with both clamps
     cmp edx, 3                                ; kind 3?
     je .mgx_kind3                             ; signed ratio + jitter
     cmp edx, 4                                ; kind 4?
     je .mgx_kind4                             ; positive ratio, sign-flipped
     cdq                                       ; edx = 0 for cur >= 0, otherwise -1
     sub eax, edx                              ; cur + 1 only when negative
     sar eax, 1                                ; signed truncation toward zero, exactly the C integer division
     mov edx, eax                              ; result for .mgx_pf_store
     jmp .mgx_pf_store                         ; store the halved distance
.mgx_kind0:
     PROBE_RND_BELOW r9                        ; edx = uniform [0, window-1]
     sub edx, r13d                             ; edx = signed [-half, +half]
     jmp .mgx_pf_store                         ; store the uniform distance
.mgx_kind1:
     movsxd rax, eax                           ; cur as 64-bit
     add rax, rax                              ; cur*2 in 64-bit
     movsxd rcx, r13d                          ; half as 64-bit
     cmp rax, rcx                              ; past the ceiling?
     cmovg rax, rcx                            ; cap high before the dword truncation
     neg rcx                                   ; -half
     cmp rax, rcx                              ; past the floor?
     cmovl rax, rcx                            ; cap low
     mov edx, eax                              ; result dword
     jmp .mgx_pf_store                         ; store the doubled distance
.mgx_kind3:
     PROBE_RND_BELOW r9                        ; edx = uniform [0, window-1]
     sub edx, r13d                             ; edx = numerator, signed [-half, +half]
     movsxd rcx, edx                           ; rcx = numerator, 64-bit
     PROBE_RND_BELOW r9                        ; edx = denominator draw, 0..2*half
     cmp edx, 1                                ; CF = 1 exactly when the unsigned draw is zero
     adc edx, 0                                ; denominator = 1 on the 1-in-window zero, no redraw loop
     mov rax, rcx                              ; rax = numerator, full 64-bit copy: a 32-bit zero-extend would flip bit 63 and make cqo treat negatives as huge positives
     mov r8d, edx                              ; r8 = denominator, zero-extended: positive 0..2*half, a sign-extension would invert the quotient
     cqo                                       ; rdx:rax = signed numerator
     idiv r8                                   ; rax = ratio quotient
     mov r14, rax                              ; r14 = ratio, parked
     mov ecx, PROBE_JITTER_RANGE               ; draw below 65
     PROBE_RND_BELOW rcx                       ; edx = 0..64
     sub edx, PROBE_JITTER_HALF                ; edx = jitter, -32..+32
     movsxd rcx, edx                           ; rcx = jitter, 64-bit
     movsxd rax, dword [r10 + rsi]             ; cur as 64-bit
     add rax, r14                              ; cur + ratio
     add rax, rcx                              ; + jitter, the sum never truncates before the clamp
     movsxd rdx, r13d                          ; half as 64-bit
     cmp rax, rdx                              ; past the ceiling?
     cmovg rax, rdx                            ; cap high
     neg rdx                                   ; -half
     cmp rax, rdx                              ; past the floor?
     cmovl rax, rdx                            ; cap low
     mov edx, eax                              ; result dword
     jmp .mgx_pf_store                         ; store the ratio-mutated distance
.mgx_kind4:
     lea ecx, [r13 + PROBE_RANGE_INC]          ; range = half+1, valid also at half = 0, u32 zero-extends
     PROBE_RND_BELOW rcx                       ; edx = 0..half
     inc edx                                   ; edx = numerator, 1..half+1
     mov r14d, edx                             ; r14d = numerator, parked (zero-extended), the next draw reuses every volatile
     lea ecx, [r13 + PROBE_RANGE_INC]          ; range again
     PROBE_RND_BELOW rcx                       ; edx = denominator raw
     inc edx                                   ; edx = denominator, 1..half+1, never zero, no redraw
     mov r8d, edx                              ; r8 = denominator, 64-bit by zero-extension
     mov eax, r14d                             ; eax = numerator
     xor edx, edx                              ; dividend = edx:eax = numerator
     div r8d                                   ; eax = step quotient, u32 exact: numerator and denominator are 1..half+1
     mov rbx, rax                              ; rbx = step, parked 64-bit BEFORE the coin draw clobbers rax
     PROBE_NEXT_RND                            ; rax = fresh state, bit 63 = coin
     mov rcx, rbx                              ; rcx = step
     neg rcx                                   ; rcx = -step
     test rax, rax                             ; SF = coin bit 63 of the random state
     cmovs rbx, rcx                            ; rbx = signed 64-bit step
     movsxd rcx, dword [r10 + rsi]             ; cur as 64-bit
     lea rax, [rbx + rcx]                      ; cur + step in 64-bit
     movsxd rdx, r13d                          ; half as 64-bit
     cmp rax, rdx                              ; past the ceiling?
     cmovg rax, rdx                            ; cap high
     neg rdx                                   ; -half
     cmp rax, rdx                              ; past the floor?
     cmovl rax, rdx                            ; cap low
     mov edx, eax                              ; result dword
.mgx_pf_store:
     mov [r10 + rsi], edx                      ; store the mutated distance
     jmp .mgx_tail                             ; mutation done
.mgx_pf_type:
     sub edx, PROBE_PARAM_ENTRIES + PROBE_PF_SLOTS ; edx = pf_type slot 0..40
     imul esi, edx, PROBE_DWORD_STRIDE         ; esi = slot byte displacement
     add esi, SHAPE_BLOCK_PF_TYPE_OFF          ; esi = block offset of the slot
     mov ecx, PROBE_PF_TYPE_RANGE              ; rcx = 7
     PROBE_RND_BELOW rcx                       ; edx = uniform [0, 6]
     mov [r10 + rsi], edx                      ; store the type
.mgx_tail:
     dec r12d                                  ; one mutation done
     jnz .mgx_mutation_loop                    ; more fields to change
; --- triples repaired, then the taper and split rules, after the
; mutation phase ---
     PROBE_SORT_TRIPLE r10, SHAPE_BLOCK_PARAM_OFF + 24 ; batch thresholds
     PROBE_SORT_TRIPLE r10, SHAPE_BLOCK_PARAM_OFF + 68 ; caller thresholds
     PROBE_FIX_TAPER_MCC r10                   ; taper-on: min_chunk_cols >= 4
     PROBE_FIX_SPLIT r10                       ; split group counts within their column budgets
; --- Phase C: parent2 donates g_probe_sex_percent percent of its dwords;
; 0 percent = asexual, the whole phase is one taken branch. The budget
; and the with-replacement slot draws match the C exactly: a 100% setting
; draws ~51.5 slots = ~39% distinct fields and can never force all 104. ---
     mov ecx, [rel g_probe_sex_percent]        ; 0..100
     test ecx, ecx                             ; asexual?
     jz .cmx_done                              ; skip everything
     imul eax, ecx, PROBE_FIELD_TOTAL          ; eax = low 32 bits of sex * 104
     mov ecx, PROBE_DIV100_RECIP               ; multiplier for exact unsigned /100
     mul ecx                                   ; edx:eax = unsigned 32x32 -> 64 product
     shr edx, PROBE_DIV100_SHIFT               ; edx = floor(original_eax / 100)
     mov eax, edx                              ; zero-extended 64-bit budget for the existing cap
     mov ecx, PROBE_FIELD_TOTAL                ; cap
     cmp rax, rcx                              ; budget over the field count?
     cmova rax, rcx                            ; budget = min(budget, 104)
     mov rcx, rax                              ; rcx = budget, moved out of rax: the macro draws through rax
     PROBE_RND_BELOW rcx                       ; edx = transfer count 0..budget-1
     mov r12d, edx                             ; r12d = transfers remaining
     test r12d, r12d                           ; zero budget or zero draw?
     jz .cmx_done                              ; nothing to donate
.cmx_transfer:
     mov ecx, PROBE_FIELD_TOTAL                ; field space 0..103
     PROBE_RND_BELOW rcx                       ; edx = transfer slot
     cmp edx, PROBE_PARAM_ENTRIES              ; param slot?
     jae .cmx_pf                               ; no: pf_dist or pf_type
     imul rcx, rdx, PROBE_BOUNDS_STRIDE        ; rcx = entry byte offset
     lea rsi, [rdi + rcx]                      ; rsi = bounds entry
     mov r8d, [rsi + PROBE_BOUNDS_OFF]         ; param_set byte offset
     mov eax, [r11 + r8 + SHAPE_BLOCK_PARAM_OFF] ; donor dword
     mov [r10 + r8 + SHAPE_BLOCK_PARAM_OFF], eax ; child takes the donor's value
     jmp .cmx_next                             ; next transfer
.cmx_pf:
     cmp edx, PROBE_PARAM_ENTRIES + PROBE_PF_SLOTS ; pf_type range?
     jae .cmx_pt                               ; high half: pf_type
     sub edx, PROBE_PARAM_ENTRIES              ; edx = pf_dist slot
     imul esi, edx, PROBE_DWORD_STRIDE         ; esi = slot byte displacement
     add esi, SHAPE_BLOCK_PF_DIST_OFF          ; esi = block offset
     mov eax, [r11 + rsi]                      ; donor distance
     mov ecx, r13d                             ; the donor may carry another shape's window:
     neg ecx                                   ; ecx = -half of THIS block
     cmp eax, r13d                             ; past this block's ceiling?
     cmovg eax, r13d                           ; cap high
     cmp eax, ecx                              ; past this block's floor?
     cmovl eax, ecx                            ; cap low
     mov [r10 + rsi], eax                      ; child takes the clamped distance
     jmp .cmx_next                             ; next transfer
.cmx_pt:
     sub edx, PROBE_PARAM_ENTRIES + PROBE_PF_SLOTS ; edx = pf_type slot
     imul esi, edx, PROBE_DWORD_STRIDE         ; esi = slot byte displacement
     add esi, SHAPE_BLOCK_PF_TYPE_OFF          ; esi = block offset
     mov eax, [r11 + rsi]                      ; donor type
     mov [r10 + rsi], eax                      ; child takes it
.cmx_next:
     dec r12d                                  ; one transfer done
     jnz .cmx_transfer                         ; more slots to donate
.cmx_done:
     PROBE_SORT_TRIPLE r10, SHAPE_BLOCK_PARAM_OFF + 24 ; crossed batch triple re-sorted
     PROBE_SORT_TRIPLE r10, SHAPE_BLOCK_PARAM_OFF + 68 ; crossed caller triple re-sorted
     PROBE_FIX_TAPER_MCC r10                   ; crossed taper rule re-checked
     PROBE_FIX_SPLIT r10                       ; crossed split rule re-checked
     pop r14                                   ; restore the body registers
     pop r13                                   ; half
     pop r12                                   ; mutation + transfer counter
     pop rdi                                   ; bounds table base
     pop rsi                                   ; entry / offset pointer
     pop rbx                                   ; draw scratch
     ret                                       ; out = mutated child, always a legal config
