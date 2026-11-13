; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; cycle_tuner_main.inc -- the in-process cycle tuner: the C++ cycle sort
; (tools\genome_cycle_sort.cpp) as a parallel per-shape state machine inside
; the engine. Every active shape (max 64, the g_active_shapes registry) runs
; its own copy of the machine, stepped by its own matching dispatches.

; States (cycle_tuner_position_state_machine[slot]):
; 0: measure a1 = the delay genome
; 1: measure b1 = genome at cycle_tuner_position_b_genome
; 2: measure c  = genome at cycle_tuner_position_c_genome
; 3: measure b2 = genome at cycle_tuner_position_b_genome
; 4: measure a2 = the delay genome; scan verdict (delay genome index follows
; any genome that beat the running minimum; exchange rule trades b/c),
; position advance + partner draw, tail mutation on the 127->0 wrap, delay
; reload -> state 5
; 5: delay phase -- dispatches run the delay genome until expiry -> state 0
; 6: bootstrap -- measure each non-tail genome once, select the fastest as delay genome

; Measurement: ONE dispatch per state, one LFENCE;RDTSC;LFENCE /
; RDTSCP;LFENCE bracket. No accumulation, no warmup, no confidence
; machinery, nothing fancy. The episode folds into three u64 values per
; shape (a_min / b_min / c), which is exactly what the three verdict rules
; consume.

; Memory model, three homes:
; the shape's per-shape config block ("the delay block", the INI-baked
; settings): the cold-start delay genome and the seed template; the tuner
; NEVER writes it;
; block 0 (the dispatcher's active param_set + pf): the genome this
; dispatch runs; the installer copies the selected source JUST DOWN TO
; block 0 and live-patches the q8-owned pf slots 0-7 and the swiglu_q8-owned
; slots 27-38 through the exported SMC entries;
; cycle_tuner_parameter_block64k[slot]: the sort list, 128 genome slots
; x 512 bytes, allocated when the shape is registered, persisted to
; v7_tune_param_<n>_<d> files at engine destroy.

; The delay genome lives in cycle_tuner_active_delay_phase_genome[slot] as a
; LIST INDEX (CYC_DELAY_NONE = the delay block as-is). Every episode
; measures the delay genome twice (a1/a2), a candidate pair b (b1/b2), and a
; single draw c; the delay genome index follows whichever genome beat the
; running minimum -- no block is ever copied for a promotion. The exchange
; rule trades b's and c's blocks in the list; when the delay genome index sat
; in one of the two swapped slots it follows its content to the other one.

; No Logo, no journal, no RNG of its own: all draws go through the
; xorshift64 in config_set_random_probe.inc, and genome generation/mutation
; reuse config_set_random_probe / config_mutate_genome.
; ============================================================================

%define CYC_STATE_DELAY 5                      ; the delay phase state
%define CYC_STATE_BOOT 6                       ; initial single-dispatch scan of the non-tail genomes
%define CYC_STATE_A1 0                         ; a1 measures the delay genome
%define CYC_STATE_B1 1                         ; b1 measures the staged b genome, the stats sample
%define CYC_STATE_C 2                          ; c measures the staged c genome
%define CYC_STATE_B2 3                         ; b2 re-measures the staged b genome
%define CYC_STATE_A2 4                         ; a2 re-measures the delay genome, then the verdict
%define CYC_DELAY_HUGE_DIM 16384               ; n or d above this seeds the large delay
%define CYC_Q8_PF_END 8                        ; exclusive end of the q8-owned slot run 0..7
%define CYC_SWIGLU_Q8_PF_FIRST 27              ; first swiglu_q8 kernel owned pf slot
%define CYC_SWIGLU_Q8_PF_END 39                ; exclusive end of the swiglu_q8-owned slot run 27..38
%define CYC_DEFAULT_DELAY 80                   ; fallback when the DB key is absent
%define CYC_DEFAULT_DELAY_LARGE 1              ; fallback large delay when the DB key is absent
%define CYC_DEFAULT_SEX_PERCENT 80             ; fallback crossover donation cap when the DB key is absent
%define CYC_DEFAULT_MUTATIONS_MAX 40           ; fallback mutation count bound when the DB key is absent
%define CYC_TAIL_SLOTS 3                       ; DEFAULT mutant tail count per wrap, overridable via tuner:tail_slots
%define CYC_LIST_LAST 127                      ; highest list index, the wrap edge
%define CYC_GENOME_BYTES 512                   ; one genome slot in the 64K block, a full shape block
%define CYC_GENOME_SHIFT 9                     ; log2 of the genome stride for the address math
%define CYC_GENOME_QWORDS 64                   ; 512/8: qwords per genome copy/exchange
%define CYC_DELAY_NONE 0xFFFFFFFF              ; delay genome = the delay block as-is, no list member yet
%define CYC_QUAD_SHIFT_128 25                  ; 32 - log2(128): a quadratic pick shrunk to a 128-slot index
%define CYC_DWORD_SCALE 4                      ; bytes per dword-array element
%define CYC_CURSOR_FILE_OFF 508                ; saved position u32 in genome 0's unused bytes 464..511
%define CYC_QWORD_SCALE 8                      ; bytes per qword-array element
%define CYC_SLOT_UNMAPPED -1                   ; block_of_slot entry for a shape with no registry slot
%define CYC_PACKED_D_OFF 4                     ; d dword's offset inside a packed [d | n] registry entry
%define CYC_STAT_BEST_SLOTS 10                 ; the elite front window: list indices 0..9 feed the best average

     section .rdata
cyc_key_cycle_active:
     db "tuner:cycle_active", 0                ; the enable/disable DB switch
cyc_key_delay_default:
     db "tuner:delaydefault", 0                ; matching-shape calls skipped between episodes
cyc_key_delay_large:
     db "tuner:delay_large", 0                 ; the delay for shapes with a dimension over 16384
cyc_key_sex_percent:
     db "tuner:sex_percent", 0                 ; the crossover donation max percent for config_mutate_genome
cyc_key_mutations_max:
     db "tuner:mutations_max", 0               ; the mutation count draw bound for config_mutate_genome
cyc_key_tail_slots:
     db "tuner:tail_slots", 0                  ; the mutant tail size, a COUNT of the 128 population slots
cyc_key_rand_delay:
     db "tuner:rand_delay", 0                  ; the delay-phase length jitter cap, added on top of the reload
cyc_msg_head:
     db "cycle tuner: ", 0                     ; destroy-time lifetime summary head
cyc_msg_mid:
     db " episode(s), ", 0                     ; between the two counters
cyc_msg_tail:
     db " promotion(s)", 0                     ; summary tail, WRITE_STR_NL appends the newline
cyc_const_ln2_div_3:
     dq 0.23104906018664842                    ; ln(2)/3: the /3 decay as a log2 scale, one fyl2x does the whole ln
cyc_const_2p32_inv:
     dq 2.3283064365386963e-10                 ; 2^-32: the integer draw read as a uniform fraction
cyc_const_exp3_span:
     dq 0.95021293163213606                    ; 1 - e^-3: the truncated exponential's CDF span
cyc_const_frac_max:
     dq 4294967295.0                           ; 2^32-1: the 32-bit fraction scale, fistp-safe ceiling

     section .bss align=8
; The per-lifecycle arrays live between cyc_clear_begin and cyc_clear_end so
; cycle_tuner_init wipes the whole region with one rep stosd. Anything that
; must survive a re-init or is explicitly re-published (the slot map, the
; DB-config dwords, the block pointer array, transient scratch) stays outside.
; The position family (5 arrays) and the fold family (3 arrays) are contiguous
; by layout: one base register addresses the whole family, asserted below.
cyc_clear_begin:
     global cycle_tuner_position               ; sorter walk position, 0..127
cycle_tuner_position:
     resd 64                                   ; uint32 per shape
     global cycle_tuner_active_delay_phase_genome ; the delay genome's list index per shape
cycle_tuner_active_delay_phase_genome:
     resd 64                                   ; uint32 per shape
     global cycle_tuner_position_state_machine ; per-shape sorter state machine
cycle_tuner_position_state_machine:
     resd 64                                   ; int per shape, values 0..5 and 0xFF
     global cycle_tuner_position_b_genome      ; B genome list index
cycle_tuner_position_b_genome:
     resd 64                                   ; int per shape
     global cycle_tuner_position_c_genome      ; C genome list index
cycle_tuner_position_c_genome:
     resd 64                                   ; int per shape
cycle_tuner_delay_counter:
     resd 64                                   ; per-shape delay countdown
cycle_tuner_delay_reload:
     resd 64                                   ; per-shape delay reload value
     alignb 8
     global cycle_tuner_best_rdtsc_cycles_phase ; the current step's measured cycles
cycle_tuner_best_rdtsc_cycles_phase:
     resq 64                                   ; u64 per shape
cycle_tuner_a_min:
     resq 64                                   ; running a minimum per shape, the fold of a1/a2
cycle_tuner_b_min:
     resq 64                                   ; running b minimum per shape, the fold of b1/b2
cycle_tuner_c_cycles:
     resq 64                                   ; the single c measurement per shape
cycle_tuner_stat_best_sum:
     resq 64                                   ; running b1-measurement sum, list position < 10, per registry slot
cycle_tuner_stat_best_count:
     resq 64                                   ; b1 samples accumulated into the best sum, per slot
cycle_tuner_stat_pool_sum:
     resq 64                                   ; running b1-measurement sum, list position below the pool limit, per slot
cycle_tuner_stat_pool_count:
     resq 64                                   ; b1 samples accumulated into the pool sum, per slot
cycle_tuner_stat_best_avg:
     resq 64                                   ; best sum / count folded at each sweep wrap, v7_get_best_fitness reads it
cycle_tuner_stat_pool_avg:
     resq 64                                   ; pool sum / count folded at each sweep wrap, v7_get_pool_fitness reads it
cycle_tuner_step_live:
     resd 1                                    ; the flip staged a timed step, consumed by the stop
cycle_tuner_tsc_start:
     resq 1                                    ; staged timer start, 0 = no timer
cycle_tuner_episodes:
     resq 1                                    ; lifetime episode counter, printed at destroy
cycle_tuner_promotions:
     resq 1                                    ; lifetime promotion counter, printed at destroy
cycle_tuner_measure_counter:
     resd 64                                   ; remaining non-tail bootstrap measurements per shape
cyc_clear_end:
cycle_tuner_block_of_slot:
     resd 256                                  ; shape-block index -> registry slot, CYC_SLOT_UNMAPPED = not an active shape
%if MAX_SHAPE_BLOCKS != 256
%error cycle_tuner_block_of_slot must hold exactly MAX_SHAPE_BLOCKS entries ; the map and the shape-block universe can never drift
%endif
cycle_tuner_active:
     resd 1                                    ; the DB switch tuner:cycle_active, 1 = enabled
cycle_tuner_delay_default:
     resd 1                                    ; DB-seeded normal delay
cycle_tuner_delay_large:
     resd 1                                    ; DB-seeded large delay
cycle_tuner_tail_slots:
     resd 1                                    ; DB-seeded mutant tail count, slots of 128 walked from index 127 down
cycle_tuner_rand_delay:
     resd 1                                    ; DB-seeded jitter cap: rand(0..cap) added to the delay reload each phase finish
cycle_tuner_step_slot:
     resd 1                                    ; the staged step's registry slot
cycle_tuner_step_block:
     resq 1                                    ; the staged step's delay-block base (the matched per-shape block)
cycle_tuner_tsc_stop:
     resq 1                                    ; staged stop timestamp, captured in the dispatcher before the stop's prologue
cycle_tuner_installed_slot:
     resd 1                                    ; registry slot whose config block 0 currently holds, -1 = overwritten by a cold dispatch
cycle_tuner_installed_gidx:
     resd 1                                    ; the list index installed from that slot, CYC_DELAY_NONE = the delay block as-is
cycle_tuner_exp_u:
     resq 1                                    ; fistp target and parked draw: a 64-bit dest never overflows the int range
     alignb 8
     global cycle_tuner_parameter_block64k     ; per-shape 64K parameter block pointer
cycle_tuner_parameter_block64k:
     resq 64                                   ; void* per shape, 0 = not yet allocated
; Family contiguity: one base register addresses each whole family, and the
; statistics fold macro addresses count/avg as byte offsets from the sum.
%define CYC_POS_ACTIVE_OFF (cycle_tuner_active_delay_phase_genome - cycle_tuner_position) ; +256: the delay genome index array, byte offset from cycle_tuner_position
%define CYC_POS_STATE_OFF (cycle_tuner_position_state_machine - cycle_tuner_position) ; +512: the state machine array
%define CYC_POS_B_OFF (cycle_tuner_position_b_genome - cycle_tuner_position) ; +768: the staged b index array
%define CYC_POS_C_OFF (cycle_tuner_position_c_genome - cycle_tuner_position) ; +1024: the staged c index array
%define CYC_POS_DELAY_COUNTER_OFF (cycle_tuner_delay_counter - cycle_tuner_position) ; +1280: the delay countdown array
%define CYC_POS_DELAY_RELOAD_OFF (cycle_tuner_delay_reload - cycle_tuner_position) ; +1536: the delay reload array
%define CYC_FOLD_B_OFF (cycle_tuner_b_min - cycle_tuner_a_min) ; +512: b_min, byte offset from cycle_tuner_a_min
%define CYC_FOLD_C_OFF (cycle_tuner_c_cycles - cycle_tuner_a_min) ; +1024: the c measurement array
%define CYC_STAT_ARRAY_BYTES 512               ; one 64-entry u64 statistics array
%define CYC_STAT_BEST_AVG_OFF (cycle_tuner_stat_best_avg - cycle_tuner_stat_best_sum) ; +2048: the best average array, byte offset from best_sum
%define CYC_STAT_POOL_SUM_OFF (cycle_tuner_stat_pool_sum - cycle_tuner_stat_best_sum) ; +1024: the pool sum array, byte offset from best_sum
%define CYC_STAT_POOL_AVG_OFF (cycle_tuner_stat_pool_avg - cycle_tuner_stat_pool_sum) ; +1536: the pool average array, byte offset from pool_sum
%if CYC_POS_ACTIVE_OFF != 256
%error position family drifted: active_delay no longer at +256 ; the one-base-register addressing below assumes this layout
%endif
%if CYC_POS_STATE_OFF != 512
%error position family drifted: state no longer at +512 ; the one-base-register addressing below assumes this layout
%endif
%if CYC_POS_B_OFF != 768
%error position family drifted: b genome no longer at +768 ; the one-base-register addressing below assumes this layout
%endif
%if CYC_POS_C_OFF != 1024
%error position family drifted: c genome no longer at +1024 ; the one-base-register addressing below assumes this layout
%endif
%if CYC_POS_DELAY_COUNTER_OFF != 1280
%error position family drifted: delay_counter no longer at +1280 ; the one-base-register addressing below assumes this layout
%endif
%if CYC_POS_DELAY_RELOAD_OFF != 1536
%error position family drifted: delay_reload no longer at +1536 ; the one-base-register addressing below assumes this layout
%endif
%if CYC_FOLD_B_OFF != 512
%error fold family drifted: b_min no longer at +512 ; the one-base-register addressing below assumes this layout
%endif
%if CYC_FOLD_C_OFF != 1024
%error fold family drifted: c_cycles no longer at +1024 ; the one-base-register addressing below assumes this layout
%endif
%if CYC_STAT_POOL_SUM_OFF != 2 * CYC_STAT_ARRAY_BYTES
%error statistics family drifted: pool_sum no longer at +1024 ; the fold macro's fixed count offset assumes this layout
%endif
%define CYCLE_TUNER_SLOTS 64                   ; the arrays above are indexed by g_active_shapes slot, same ceiling
%define CYCLE_TUNER_BLOCK_BYTES 65536          ; parameter block size: one 64 KiB committed block per active shape

     section .text
; ---------------------------------------------------------------------------
; cyc_parse_db_int -- read one decimal-string DB key with a fallback.
; IN: rcx = key string, edx = fallback value. OUT: eax = parsed value or
; the fallback when the key is absent. Clobbers rax/rcx/rdx/r8-r11.
; ---------------------------------------------------------------------------
cyc_parse_db_int:
     SF_START cyc_parse_db_int, rbx            ; frame: rbx holds the fallback, shadow space for ht_find
     SF_COMMIT                                 ; Win64 shadow space and stack alignment
     mov ebx, edx                              ; ebx = the fallback value
     xor edx, edx                              ; no length needed, keys are NUL-terminated
     call ht_find                              ; rax = value bytes or NULL
     test rax, rax                             ; key absent?
     jz .cpd_fallback                          ; yes: the fallback stands
     mov rcx, rax                              ; rcx = decimal string cursor
     xor eax, eax                              ; eax = accumulator
.cpd_digit:
     movzx r8d, byte [rcx]                     ; current character
     sub r8d, '0'                              ; decimal distance
     cmp r8d, 9                                ; past '9' (covers the NUL too)?
     ja .cpd_done                              ; non-digit: string done
     imul eax, eax, 10                         ; accumulator *= 10
     add eax, r8d                              ; + digit
     inc rcx                                   ; next character
     jmp .cpd_digit                            ; parse to the terminator
.cpd_done:
     SF_END rbx                                ; restore and return, eax = parsed value
.cpd_fallback:
     mov eax, ebx                              ; eax = the fallback
     SF_END rbx                                ; restore and return, eax = fallback value

; ---------------------------------------------------------------------------
; cyc_quad_pick -- one quadratic-distribution draw: OUT edx = (u*u) >> 32
; with u a fresh 32-bit xorshift draw, the integer form of
; f = frand(); f = f*f. The 32-bit mul puts the high 32 bits of the 64-bit
; product in edx directly; a 64-bit mul rax would return zero (u*u always
; fits 64 bits). Clobbers rax/rdx.
; ---------------------------------------------------------------------------
cyc_quad_pick:
     PROBE_NEXT_RND                            ; rax = fresh 64-bit state, eax = low draw
     mul eax                                   ; edx:eax = u*u, edx = (u*u) >> 32
     ret                                       ; edx = quadratic draw

; ---------------------------------------------------------------------------
; cyc_exp_pick -- one truncated-exponential draw (decay 3): OUT edx = the
; 32-bit fraction x = -ln(1 - u*(1-e^-3))/3 with u a fresh 32-bit xorshift
; draw, the partner sweep's winning front bias (finite density at the front,
; every slot reachable). The x87 register stack is balanced (3 pushes, 3
; pops); all constants are doubles in .rdata. Clobbers rax/rdx.
; ---------------------------------------------------------------------------
cyc_exp_pick:
     PROBE_NEXT_RND                            ; rax = fresh 64-bit state, eax = low draw u
     mov dword [rel cycle_tuner_exp_u], eax    ; park u's low dword: x87 loads integers from memory only
     mov dword [rel cycle_tuner_exp_u + CYC_DWORD_SCALE], 0 ; high dword zero: fild qword reads signed, the draw is unsigned
     fld qword [rel cyc_const_ln2_div_3]       ; st0 = ln(2)/3, the decay scale
     fild qword [rel cycle_tuner_exp_u]        ; st0 = u, zero-extended to 64 bits, st1 = ln2/3
     fmul qword [rel cyc_const_2p32_inv]       ; st0 = u / 2^32, a uniform in [0,1)
     fmul qword [rel cyc_const_exp3_span]      ; st0 = u * (1 - e^-3), st1 = ln2/3
     fld1                                      ; st0 = 1, st1 = u*span, st2 = ln2/3
     fsubrp st1, st0                           ; st0 = 1 - u*span, st1 = ln2/3
     fyl2x                                     ; st0 = (ln2/3)*log2(1-u*span) = ln(1-u*span)/3, <= 0
     fchs                                      ; st0 = x = -ln(1-u*span)/3, in [0,1)
     fmul qword [rel cyc_const_frac_max]       ; st0 = x * (2^32-1), the fraction scale
     fistp qword [rel cycle_tuner_exp_u]       ; park the fraction in the (now free) draw slot, 64-bit dest keeps the int range legal
     mov edx, dword [rel cycle_tuner_exp_u]    ; edx = the low half = the exp3 fraction
     ret                                       ; edx out, rax/rdx clobbered, drop-in for cyc_quad_pick

; ---------------------------------------------------------------------------
; cycle_tuner_init -- V7_init hook: read the DB switches, reset the
; per-shape state, and start a FRESH tuner lifecycle: the shape registry
; clears (shapes re-register on their next cold dispatch) and parameter
; blocks from a previous lifetime are freed -- the populations live on in
; the v7_tune_param_<n>_<d> files and reload at the next allocation.
; Clobbers rax/rcx/rdx/r8-r11; preserves rbx/r12-r15/rsi/rdi.
; ---------------------------------------------------------------------------
     global cycle_tuner_init                   ; internal, called from V7_init
     SF_START cycle_tuner_init, rdi            ; frame + save for the rep stosd region walk
     SF_DWORD cri_slot                         ; free-loop cursor, lives across the mdb_free_mem call
     SF_COMMIT                                 ; Win64 shadow space and stack alignment
     lea rdi, [rel cyc_clear_begin]            ; rdi = the per-lifecycle region base
     lea rcx, [rel cyc_clear_end]              ; rcx = the region end
     sub rcx, rdi                              ; rcx = the region byte size
     shr rcx, 2                                ; rcx = dword count
     xor eax, eax                              ; zero fill value
     rep stosd                                 ; position/state family, delay pair, folds, measurements, statistics, step/timer/totals: all wiped with the lifecycle
     lea rdi, [rel cycle_tuner_block_of_slot]  ; rdi = slot map base
     mov eax, CYC_SLOT_UNMAPPED                ; every block starts unmapped (negative, the flip's js sentinel)
     mov ecx, 256                              ; dword count
     rep stosd                                 ; the map is empty until shapes re-register
     lea rdi, [rel g_active_shapes]            ; rdi = the shape registry
     xor eax, eax                              ; zero fill: no shapes registered in the new lifecycle
     mov ecx, 128                              ; 64 qwords = 128 dwords
     rep stosd                                 ; registry empty, shapes re-register through the scan
     mov dword [rel cached_prefetch_n], 0      ; the dispatcher's prefetch cache holds the dead
     mov dword [rel cached_prefetch_d], 0      ; lifetime's shape: 0 is the never-installed sentinel,
; the next dispatch takes the forced full 41-slot install
     mov dword [rel cached_shape_n], -1        ; the warm-shape cache holds the dead lifetime too:
     mov dword [rel cached_shape_d], -1        ; the first dispatch must rescan and re-register
     mov qword [rel tune_scanned_block], 0     ; and the scanned-block pin carries stale BSS
; free parameter blocks left over from a previous engine lifetime in this
; process: the populations reload from their files at the next allocation
     mov dword cri_slot, 0                     ; the walk starts at slot 0
.cri_free:
     mov r8d, cri_slot                         ; r8d = slot index, reloaded after the call
     lea r11, [rel cycle_tuner_parameter_block64k] ; r11 = pointer-array base, same reason
     mov rcx, [r11 + r8*CYC_QWORD_SCALE]       ; rcx = block pointer, 0 = none
     test rcx, rcx                             ; slot owns a block?
     jz .cri_free_next                         ; no: nothing to free
     call mdb_free_mem                         ; release the previous lifetime's block
     mov r8d, cri_slot                         ; reload the cursor: the call is free to clobber r8
     lea r11, [rel cycle_tuner_parameter_block64k] ; reload the base: the call clobbered r11
     mov qword [r11 + r8*CYC_QWORD_SCALE], 0   ; zero-pin
.cri_free_next:
     inc dword cri_slot                        ; next slot
     cmp dword cri_slot, CYCLE_TUNER_SLOTS     ; all 64 slots walked?
     jb .cri_free                              ; next
     lea rcx, [rel cyc_key_cycle_active]       ; rcx = the enable/disable key
     mov edx, 1                                ; edx = default: enabled
     call cyc_parse_db_int                     ; eax = the switch
     mov [rel cycle_tuner_active], eax         ; publish the switch
     lea rcx, [rel cyc_key_delay_default]      ; rcx = normal delay key
     mov edx, CYC_DEFAULT_DELAY                ; edx = fallback 80
     call cyc_parse_db_int                     ; eax = the normal delay
     mov [rel cycle_tuner_delay_default], eax  ; publish
     lea rcx, [rel cyc_key_delay_large]        ; rcx = large delay key
     mov edx, CYC_DEFAULT_DELAY_LARGE          ; edx = fallback 1
     call cyc_parse_db_int                     ; eax = the large delay
     mov [rel cycle_tuner_delay_large], eax    ; publish
     lea rcx, [rel cyc_key_sex_percent]        ; rcx = the crossover percent key
     mov edx, CYC_DEFAULT_SEX_PERCENT          ; edx = fallback 80
     call cyc_parse_db_int                     ; eax = the donation cap percent
     cmp eax, 100                              ; inside the documented range [0, 100]?
     ja .cri_sex_fatal                         ; no: a percent above 100 is a DB configuration bug
     mov [rel g_probe_sex_percent], eax        ; publish to the mutation engine, percent 0..100
     lea rcx, [rel cyc_key_mutations_max]      ; rcx = the mutation count bound key
     mov edx, CYC_DEFAULT_MUTATIONS_MAX        ; edx = fallback 40
     call cyc_parse_db_int                     ; eax = the field-count draw bound
     mov [rel g_probe_mutations_max], eax      ; publish to the mutation engine, 1..max fields per child
     lea rcx, [rel cyc_key_tail_slots]         ; rcx = the tail size key
     mov edx, CYC_TAIL_SLOTS                   ; edx = fallback 3 slots
     call cyc_parse_db_int                     ; eax = the tail count of 128
     cmp eax, CYC_LIST_LAST                    ; inside the legal range [0, 127]?
     ja .cri_tail_fatal                        ; no: a tail of all 128 slots empties the pool bucket
     mov [rel cycle_tuner_tail_slots], eax     ; publish, the wrap walk reads it
     lea rcx, [rel cyc_key_rand_delay]         ; rcx = the jitter cap key
     xor edx, edx                              ; edx = fallback 0, no jitter
     call cyc_parse_db_int                     ; eax = the jitter cap
     cmp eax, 0xFFFFFFFE                       ; the phase-finish reload draws jitter via inc/mul on cap + 1:
     ja .cri_jitter_fatal                      ; a cap of 0xFFFFFFFF wraps the +1 to zero and the jitter draw degenerates
     mov [rel cycle_tuner_rand_delay], eax     ; publish, the phase-finish reload reads it
     PROBE_SEED_TSC                            ; xorshift64 zero is absorbing: every-population-load skips
; both seeding entry points, so seed once here at lifecycle start
     mov dword [rel cycle_tuner_installed_slot], -1 ; block 0's occupant is unknown at lifecycle start
     mov dword [rel cycle_tuner_installed_gidx], CYC_DELAY_NONE ; nothing tuner-installed yet
     SF_END rdi                                ; restore and return
.cri_tail_fatal:
     FATAL "cycle tuner: tuner:tail_slots above 127 would empty the non-tail pool"
.cri_jitter_fatal:
     FATAL "cycle tuner: tuner:rand_delay of 0xFFFFFFFF overflows the jitter modulus"
.cri_sex_fatal:
     FATAL "cycle tuner: tuner:sex_percent must be 0..100"

; ---------------------------------------------------------------------------
; cycle_tuner_register_shape -- seed a newly seen shape and immediately
; load or create its population. IN: rcx = packed [d | n] registry entry,
; rdx = appended entry pointer, r8d = shape block index. Clobbers volatile
; registers and preserves the dispatcher's nonvolatile registers.
; ---------------------------------------------------------------------------
     global cycle_tuner_register_shape         ; internal, called from the dispatcher's registration append
cycle_tuner_register_shape:
     cmp dword [rel cycle_tuner_active], 0     ; the DB switch enables the tuner?
     je .crs_off                               ; disabled: leave the slot unmapped, the state stays dormant
     lea r9, [rel g_active_shapes]             ; r9 = registry base
     sub rdx, r9                               ; rdx = byte offset of the appended entry
     shr rdx, 3                                ; rdx = registry slot index
     mov r9d, edx                              ; r9d = slot, the per-shape array index
; delay seed: n or d over the huge threshold -> the large delay
     mov r10d, ecx                             ; r10d = n, low half of the packed entry
     shr rcx, 32                               ; rcx = d, high half
     cmp r10d, CYC_DELAY_HUGE_DIM              ; n over the huge threshold?
     ja .crs_pick_large                        ; yes: large delay
     cmp ecx, CYC_DELAY_HUGE_DIM               ; d over the huge threshold?
     ja .crs_pick_large                        ; yes: large delay
     mov edx, [rel cycle_tuner_delay_default]  ; edx = the normal delay seed
     jmp .crs_set_delay                        ; store it
.crs_pick_large:
     mov edx, [rel cycle_tuner_delay_large]    ; edx = the large delay seed
.crs_set_delay:
     lea r10, [rel cycle_tuner_delay_counter]  ; r10 = delay counter array
     mov [r10 + r9*CYC_DWORD_SCALE], edx       ; countdown starts at the seed
     lea r10, [rel cycle_tuner_delay_reload]   ; r10 = delay reload array
     mov [r10 + r9*CYC_DWORD_SCALE], edx       ; the reload carries the same seed
     lea r10, [rel cycle_tuner_position]       ; r10 = anchor array
     mov dword [r10 + r9*CYC_DWORD_SCALE], 0   ; the anchor starts at 0
     lea r10, [rel cycle_tuner_active_delay_phase_genome] ; r10 = delay genome array
     mov dword [r10 + r9*CYC_DWORD_SCALE], CYC_DELAY_NONE ; the INI delay genome, no list member yet
     lea r10, [rel cycle_tuner_position_b_genome] ; r10 = staged b array
     mov dword [r10 + r9*CYC_DWORD_SCALE], 0   ; staged pair unset: slot 0 vs slot 0,
     lea r10, [rel cycle_tuner_position_c_genome] ; r10 = staged c array
     mov dword [r10 + r9*CYC_DWORD_SCALE], 0   ; a harmless no-op first episode
     lea r10, [rel cycle_tuner_b_min]          ; r10 = b_min array base
     mov qword [r10 + r9*CYC_QWORD_SCALE], 0   ; no baseline yet for the pending pair
     lea r10, [rel cycle_tuner_c_cycles]       ; r10 = the c measurement array base
     mov qword [r10 + r9*CYC_QWORD_SCALE], 0   ; no challenger measurement yet
     lea r10, [rel cycle_tuner_block_of_slot]  ; r10 = slot map base
     mov [r10 + r8*CYC_DWORD_SCALE], r9d       ; map the shape's block index to this slot
     jmp cycle_tuner_alloc_blocks              ; tail call: load the new shape's population now
.crs_off:
     ret                                       ; tuner disabled: no shape registration

; CYC_EXCHANGE_GENOMES -- swap two 512-byte genome blocks in place.
; IN: rsi = block A base, rbx = block B base. After the macro, A holds B's
; genome and B holds A's. Safe when A == B (both stores rewrite the same
; bytes). Clobbers rax and ymm0/ymm1; rsi/rdi/rbx survive.

%macro CYC_EXCHANGE_GENOMES 0
     xor eax, eax                              ; eax = the running byte displacement
%%cex_loop:
     vmovdqu ymm0, [rsi + rax]                 ; ymm0 = A's 32 bytes
     vmovdqu ymm1, [rbx + rax]                 ; ymm1 = B's 32 bytes
     vmovdqu [rsi + rax], ymm1                 ; A's memory takes B's
     vmovdqu [rbx + rax], ymm0                 ; B's memory takes A's
     add eax, 32                               ; next vector
     cmp eax, CYC_GENOME_BYTES                 ; the whole 512-byte genome swapped?
     jb %%cex_loop                             ; next
%endmacro

; CYC_PICK_EXCL -- one quadratic draw as a list index, redrawn while it
; equals r9d. The mutation's read order breaks when a pick aliases the
; destination (parent == out or donor == out), so the destination is
; excluded by redrawing. r9d = the excluded index, survives the macro.
; Clobbers rax/rdx only.

%macro CYC_PICK_EXCL 0
%%again:
     call cyc_quad_pick                        ; edx = the quadratic draw
     mov eax, edx                              ; eax = the draw
     shr eax, CYC_QUAD_SHIFT_128               ; eax = the list index 0..127
     cmp eax, r9d                              ; the excluded destination?
     je %%again                                ; redraw
%endmacro

; CYC_FOLD sum_array, avg_offset -- divide one sweep bucket's sum by its
; sample count, publish the average at avg_offset from the sum array, and
; zero both accumulators for the next sweep. The count array lives at
; +CYC_STAT_ARRAY_BYTES from the sum array by layout (asserted by the family
; offset checks). r13 = registry slot. Clobbers rax/rcx/rdx/r10.

%macro CYC_FOLD 2
     lea r10, [rel %1]                         ; r10 = the bucket's sum array base
     mov rax, [r10 + r13*CYC_QWORD_SCALE]      ; the sweep's sum
     mov rcx, [r10 + r13*CYC_QWORD_SCALE + CYC_STAT_ARRAY_BYTES] ; samples in the sum
     test rcx, rcx                             ; resumed partial sweep may have no samples in this bucket
     jz %%empty                                ; keep prior average when the bucket is empty
     xor edx, edx                              ; zero the dividend's high half
     div rcx                                   ; rax = the sweep's average
     mov [r10 + r13*CYC_QWORD_SCALE + %2], rax ; publish, the fitness getter reads it
%%empty:
     mov qword [r10 + r13*CYC_QWORD_SCALE], 0  ; sum restarts
     mov qword [r10 + r13*CYC_QWORD_SCALE + CYC_STAT_ARRAY_BYTES], 0 ; count restarts
%endmacro

; ---------------------------------------------------------------------------
; cyc_install_genome -- install one genome's config for the next dispatch:
; copy the param dwords JUST DOWN TO block 0, then live-patch the q8-owned
; pf slots 0..7 and the swiglu_q8-owned slots 27..38 through the exported
; SMC entries, and mark the dispatcher's prefetch cache valid so its own
; .patch_prefetch skips (the install IS the patch).
; IN: rcx = source genome block address, rdx = the delay block base (the
; matched per-shape block). Clobbers
; rax/rcx/rdx/r8-r11; preserves rbx/r12-r15/rsi/rdi.
; NOTE: set_prefetch_distance / set_prefetch_type clobber r8-r11, so the
; loop cursor and the new values live in the frame, never in registers
; across those calls.
; ---------------------------------------------------------------------------
cyc_install_genome:
     SF_START cyc_install_genome, rsi, rdi     ; frame + save for the rep movsq
     SF_QWORD ci_src                           ; source genome base
     SF_QWORD ci_delay                         ; delay block base
     SF_DWORD ci_slot                          ; owned pf slot cursor, survives the SMC calls
     SF_DWORD ci_dirty                         ; 1 when this installation changed executable bytes
     SF_COMMIT                                 ; Win64 shadow space and stack alignment
     mov ci_src, rcx                           ; park the args before rcx is consumed
     mov ci_delay, rdx                         ; rdx = the matched per-shape delay block base
     lea rsi, [rcx + SHAPE_BLOCK_PARAM_OFF]    ; source param_set
     lea rdi, [rel param_set_0]                ; dest = block 0 param_set, engine scratch
     mov ecx, PARAM_SET_QWORDS                 ; 12 qwords = 24 param dwords
     rep movsq                                 ; the genome's params drive the next dispatch
     mov dword ci_slot, 0                      ; owned slot cursor starts at 0
     mov dword ci_dirty, 0                     ; no modified code bytes yet
.ci_pf:
     mov r8d, ci_slot                          ; current owned slot
     cmp r8d, CYC_SWIGLU_Q8_PF_END             ; past the last owned slot of the second run?
     jae .ci_pf_done                           ; yes: patch pass complete
     mov rcx, ci_src                           ; genome block
     mov r9d, [rcx + SHAPE_BLOCK_PF_DIST_OFF + r8*CYC_DWORD_SCALE] ; requested distance
     lea rsi, [rel shape_blocks]               ; installed block-0 mirror
     cmp r9d, [rsi + SHAPE_BLOCK_PF_DIST_OFF + r8*CYC_DWORD_SCALE] ; distance identical?
     jne .ci_dist_changed                      ; distance change, inspect old/new enabled state
     mov r10d, [rcx + SHAPE_BLOCK_PF_TYPE_OFF + r8*CYC_DWORD_SCALE] ; requested type
     cmp r10d, [rsi + SHAPE_BLOCK_PF_TYPE_OFF + r8*CYC_DWORD_SCALE] ; type identical too?
     je .ci_pf_next                            ; no changes, no mirror stores and no patcher calls
     cmp r9d, 32                               ; disabled distance keeps instruction NOPped
     jb .ci_update_mirror                      ; skip type-only patch while NOPped
     mov ecx, r8d                              ; slot
     mov edx, r10d                             ; requested type
     call set_prefetch_type_fast               ; type-only update with no restore
     mov dword ci_dirty, 1                     ; opcode/ModRM changed
     jmp .ci_update_mirror                     ; update type mirror
.ci_dist_changed:
     mov r10d, [rsi + SHAPE_BLOCK_PF_DIST_OFF + r8*CYC_DWORD_SCALE] ; old distance
     cmp r9d, 32                               ; new distance disables prefetch?
     jb .ci_new_disabled                       ; handle enabled->NOP and NOP->NOP
     cmp r10d, 32                              ; previous instruction NOPped?
     jb .ci_restore_enabled                    ; re-enable via full restore
     mov ecx, r8d                              ; slot
     mov edx, r9d                              ; new displacement
     call smc_prefetch_disp_fast               ; enabled->enabled: one disp32 write, no full restore
     mov dword ci_dirty, 1                     ; instruction bytes changed
     mov r8d, ci_slot                          ; reload volatile index
     mov rcx, ci_src                           ; reload genome
     mov edx, [rcx + SHAPE_BLOCK_PF_TYPE_OFF + r8*CYC_DWORD_SCALE] ; requested type
     lea rsi, [rel shape_blocks]               ; installed mirror
     cmp edx, [rsi + SHAPE_BLOCK_PF_TYPE_OFF + r8*CYC_DWORD_SCALE] ; same type?
     je .ci_update_mirror                      ; no opcode/ModRM update needed
     mov ecx, r8d                              ; slot
     call set_prefetch_type_fast               ; change type only if different
     jmp .ci_update_mirror                     ; write both mirrors
.ci_new_disabled:
     cmp r10d, 32                              ; old prefetch already disabled?
     jb .ci_update_mirror                      ; NOP->NOP, no code writes
     mov ecx, r8d                              ; slot
     mov edx, r9d                              ; new disabled distance
     call set_prefetch_distance_fast           ; enabled->NOP, no type patch needed
     mov dword ci_dirty, 1                     ; instruction bytes changed
     jmp .ci_update_mirror                     ; mirror new distance/type
.ci_restore_enabled:
     mov ecx, r8d                              ; slot
     mov edx, r9d                              ; new enabled distance
     call set_prefetch_distance_fast           ; NOP->enabled: restore + patch displacement
     mov r8d, ci_slot                          ; reload index after setter
     mov rcx, ci_src                           ; reload genome block
     mov edx, [rcx + SHAPE_BLOCK_PF_TYPE_OFF + r8*CYC_DWORD_SCALE] ; requested type
     mov ecx, r8d                              ; slot index
     call set_prefetch_type_fast               ; apply requested type to restored instruction
     mov dword ci_dirty, 1                     ; instruction bytes changed
.ci_update_mirror:
     mov r8d, ci_slot                          ; reload index after any patch function
     mov rcx, ci_src                           ; reload source genome
     lea rsi, [rel shape_blocks]               ; block 0 mirror base
     mov eax, [rcx + SHAPE_BLOCK_PF_DIST_OFF + r8*CYC_DWORD_SCALE] ; requested distance
     mov [rsi + SHAPE_BLOCK_PF_DIST_OFF + r8*CYC_DWORD_SCALE], eax ; update installed distance
     mov eax, [rcx + SHAPE_BLOCK_PF_TYPE_OFF + r8*CYC_DWORD_SCALE] ; requested type
     mov [rsi + SHAPE_BLOCK_PF_TYPE_OFF + r8*CYC_DWORD_SCALE], eax ; update installed type
.ci_pf_next:
     inc dword ci_slot                         ; next owned slot
     cmp dword ci_slot, CYC_Q8_PF_END          ; the q8 run 0..7 complete?
     jne .ci_pf                                ; no: continue the run
     mov dword ci_slot, CYC_SWIGLU_Q8_PF_FIRST ; jump to the swiglu_q8 run 27..38
     jmp .ci_pf                                ; continue the patch pass
.ci_pf_done:
     cmp dword ci_dirty, 0                     ; did any owned slot modify executable bytes?
     je .ci_no_serialize                       ; no: no barrier required
     call smc_prefetch_serialize               ; serialize once after all owned slots
.ci_no_serialize:
     mov rdx, ci_delay                         ; rdx = the delay block base
     mov eax, [rdx]                            ; the delay block header's n
     mov [rel cached_prefetch_n], eax          ; mark the cache valid: the dispatcher's own
     mov eax, [rdx + SHAPE_BLOCK_D_EQU]        ; the delay block header's d
     mov [rel cached_prefetch_d], eax          ; .patch_prefetch now skips, the install WAS the patch
     SF_END rsi, rdi                           ; restore and return

; ---------------------------------------------------------------------------
; cycle_tuner_flip -- the flip-side hook, called on EVERY matmul dispatch
; right where the old tuner gates sat. IN: rcx = the matched per-shape
; block index, r15 = the matched block pointer. Advances this shape's
; machine: state 5 ticks the delay; states 0-4 and bootstrap state 6
; install a step genome into block 0 and stage one timer. Unmapped shapes
; and delay ticks run FRAMELESS on volatile registers only;
; only an episode step takes the full frame.
; Preserves rbx/r12-r15/rsi/rdi (the dispatcher's live registers);
; clobbers rax/rcx/rdx/r8-r11.
; ---------------------------------------------------------------------------
     global cycle_tuner_flip                   ; internal, called from the dispatcher's flip gate
cycle_tuner_flip:
     lea r11, [rel cycle_tuner_block_of_slot]  ; r11 = slot map base, volatile only
     mov r8d, [r11 + rcx*CYC_DWORD_SCALE]      ; r8d = this shape's registry slot
     test r8d, r8d                             ; slot 0 is a VALID registered slot
     js .cflip_ret                             ; negative = unmapped -> dormant, plain dispatch
     lea r11, [rel cycle_tuner_position_state_machine] ; r11 = state array base
     mov r9d, [r11 + r8*CYC_DWORD_SCALE]       ; r9d = this shape's state
     cmp r9d, CYC_STATE_DELAY                  ; the delay phase?
     je .cflip_delay                           ; yes: the frameless tick
; episode state 0..4: take the frame, r8d = slot, r9d = state, r15 intact
; the slow frame is hand-rolled: SF_START would emit a second global
; label and split the local-label scope, so the pushes live here
.cflip_slow:
     push rbx                                  ; save the dispatcher's live rbx
     push r12                                  ; save the dispatcher's live r12
     push r13                                  ; save the dispatcher's live r13
     push r14                                  ; save the dispatcher's live r14
     push rdi                                  ; save the dispatcher's live rdi
     push r15                                  ; the matched block, live through the whole slow body
     sub rsp, 40                               ; shadow space: 6 pushes + 40 keep RSP 16-aligned for the install call
     mov r13d, r8d                             ; r13d = the slot, the episode body's index
     mov r12d, r9d                             ; r12d = the state
     jmp .cf_episode                           ; into the episode body
.cflip_ret:
     ret                                       ; plain dispatch, nothing touched
.cflip_delay:
     lea r11, [rel cycle_tuner_delay_counter]  ; r11 = this shape's delay counter array
     mov eax, [r11 + r8*CYC_DWORD_SCALE]       ; eax = remaining matching dispatches
     test eax, eax                             ; delay expired?
     jz .cflip_expire                          ; yes: start the episode, needs the frame
     dec eax                                   ; tick: EVERY matching dispatch consumes one delay unit,
; stored before any reinstall branch so interleaved shapes count down
     mov [r11 + r8*CYC_DWORD_SCALE], eax       ; publish the remainder
     cmp dword [rel cycle_tuner_installed_slot], r8d ; block 0 still holds THIS shape's config?
     jne .cflip_reinstall                      ; a cold dispatch overwrote it: restore the delay genome
     lea r10, [rel cycle_tuner_active_delay_phase_genome] ; r10 = delay genome index array
     mov r9d, [r10 + r8*CYC_DWORD_SCALE]       ; r9d = the delay genome's list index
     cmp r9d, [rel cycle_tuner_installed_gidx] ; the delay genome itself is what block 0 holds?
     jne .cflip_reinstall                      ; an episode step's genome sits there: restore the delay genome
     ret                                       ; plain dispatch this call, delay genome already active
.cflip_reinstall:
     mov r9d, CYC_STATE_DELAY                  ; the slow body installs the delay genome, untimed
     jmp .cflip_slow                           ; take the frame, reinstall, plain dispatch
.cflip_expire:
     mov dword [r11 + r8*CYC_DWORD_SCALE], 0   ; the counter is spent
     lea r11, [rel cycle_tuner_position_state_machine] ; r11 = state array base
     mov dword [r11 + r8*CYC_DWORD_SCALE], 0   ; state 0: a1 measures the delay genome
     xor r9d, r9d                              ; r9d = state 0 for the slow body's source pick
     jmp .cflip_slow                           ; take the frame, run the episode path
.cf_episode:
; falls into the episode path: interleaved shapes may have left a
; foreign config in block 0, so every step MUST install its genome first.
; One selection serves both the source pick and the identity publish: the
; index array is chosen by state and the loaded index IS the identity.
; r12d = the state, NEVER clobbered below
     lea r14, [rel cycle_tuner_parameter_block64k] ; r14 = pointer-array base
     mov r14, [r14 + r13*CYC_QWORD_SCALE]      ; r14 = this shape's 64K list base, present for every mapped shape
     cmp r12d, CYC_STATE_BOOT                  ; bootstrap scans the non-tail list once
     je .cf_boot                               ; select index = measure_counter - 1
     lea rax, [rel cycle_tuner_active_delay_phase_genome] ; the delay genome index array, states 0, 4, 5
     lea rcx, [rel cycle_tuner_position_b_genome] ; the staged b index array, states 1 and 3
     lea rdx, [rel cycle_tuner_position_c_genome] ; the staged c index array, state 2
     mov r8d, 0x0A                             ; bit pattern 0b1010: bits 1 and 3 mark the b states
     bt r8d, r12d                              ; this episode installs b?
     cmovc rax, rcx                            ; yes: select the b index array
     mov r8d, 4                                ; bit pattern 0b100: bit 2 marks the c state
     bt r8d, r12d                              ; this episode installs c?
     cmovc rax, rdx                            ; yes: select the c index array
     mov ebx, [rax + r13*CYC_DWORD_SCALE]      ; ebx = the normal episode's genome index
     jmp .cf_source_ready                      ; bootstrap bypasses the normal source picker
.cf_boot:
     lea rax, [rel cycle_tuner_measure_counter] ; rax = the bootstrap countdown array
     mov ebx, [rax + r13*CYC_DWORD_SCALE]      ; ebx = the remaining scan count
     dec ebx                                   ; genome index = count - 1, from non-tail end to 0
.cf_source_ready:
     mov rcx, r15                              ; NONE resolves to the delay block itself
     cmp ebx, CYC_DELAY_NONE                   ; the delay genome sits outside the list?
     je .cf_install                            ; yes: install the delay block's content
     mov eax, ebx                              ; eax = the list index
     shl eax, CYC_GENOME_SHIFT                 ; * 512, u32 exact: list indices are 0..127
     lea rcx, [r14 + rax]                      ; rcx = the step genome's block
.cf_install:
     mov rdx, r15                              ; arg2: the delay block base
     call cyc_install_genome                   ; arg1 = rcx: install, block 0 + owned pf patch + cache mark
; publish block 0's identity: WHICH slot's genome the install put there, so
; a delay tick can skip the reinstall while the delay genome is already active
     mov dword [rel cycle_tuner_installed_slot], r13d ; the installed config belongs to this slot
     mov dword [rel cycle_tuner_installed_gidx], ebx ; the installed genome's list index, NONE = the delay block
.cf_stage:
; stage the timed step for the stop hook -- a DELAY REINSTALL (state 5) is
; ordinary inference, never timed. The TIMER START lives in the dispatcher,
; staged right after this flip returns -- the closest point to
; DISPATCH_EXECUTE; staging it here too would just be overwritten.
     cmp r12d, CYC_STATE_DELAY                 ; the reinstall flavor of this call?
     je .cf_out                                ; yes: untimed, no step staged
     mov dword [rel cycle_tuner_step_live], 1  ; one staged step, consumed after the dispatch
     mov [rel cycle_tuner_step_slot], r13d     ; remember the slot for the fold
     mov rax, r15                              ; rax = the delay block base
     mov [rel cycle_tuner_step_block], rax     ; remember it for the verdict installs
.cf_out:
     add rsp, 40                               ; release the shadow
     pop r15                                   ; restore the matched block, reverse push order
     pop rdi                                   ; restore the dispatcher's live rdi
     pop r14                                   ; restore the dispatcher's live r14
     pop r13                                   ; restore the dispatcher's live r13
     pop r12                                   ; restore the dispatcher's live r12
     pop rbx                                   ; restore the dispatcher's live rbx
     ret                                       ; back to the dispatcher

; ---------------------------------------------------------------------------
; cycle_tuner_stop -- the stop-side hook, called right after
; DISPATCH_EXECUTE when the flip staged a timed step: rdtscp delta, the
; fold, state advance, and at the end of state 4 the three-rule verdict,
; the position/partner advance, the tail mutation on a wrap, and the delay
; reload. Preserves rbx/r12-r15/rsi/rdi (the dispatcher's live registers);
; clobbers rax/rcx/rdx/r8-r11.
; ---------------------------------------------------------------------------
     global cycle_tuner_stop                   ; internal, called from the dispatcher's stop site
cycle_tuner_stop:
     SF_START cycle_tuner_stop, rbx, rsi, rdi, r12, r13, r14, r15 ; frame + the dispatcher's live registers
     SF_DWORD cf_tgt                           ; the delay genome's new list index, survives the exchange
     SF_DWORD cf_reload                        ; the clean delay reload, parked across the jitter draw
     SF_COMMIT                                 ; Win64 shadow space and stack alignment
     cmp dword [rel cycle_tuner_step_live], 0  ; a timed step staged?
     je .cs_done                               ; no: plain dispatch, nothing recorded
     mov dword [rel cycle_tuner_step_live], 0  ; consume the marker
     cmp qword [rel cycle_tuner_tsc_start], 0  ; a timer actually armed? 0 = the dispatcher never staged one
     je .cs_done                               ; no: folding the raw stop stamp would poison the sums
     mov rdx, [rel cycle_tuner_tsc_stop]       ; rdx = the dispatcher's staged stop timestamp
     sub rdx, [rel cycle_tuner_tsc_start]      ; rdx = delta, the step's cycle count
     mov qword [rel cycle_tuner_tsc_start], 0  ; consume the start stamp: a later skipped staging folds no stale delta
     mov r13d, [rel cycle_tuner_step_slot]     ; r13d = the step's registry slot
     lea r10, [rel cycle_tuner_position]       ; r10 = position-family base: position, delay index, state, b, c, delay pair
     lea r11, [rel cycle_tuner_best_rdtsc_cycles_phase] ; r11 = measurement array base
     mov [r11 + r13*CYC_QWORD_SCALE], rdx      ; publish the raw measurement
     lea r14, [rel cycle_tuner_a_min]          ; r14 = fold-family base: a_min, b_min, c_cycles
     mov r12d, [r10 + r13*CYC_DWORD_SCALE + CYC_POS_STATE_OFF] ; r12d = the completed state, live across the whole verdict
     cmp r12d, CYC_STATE_BOOT                  ; a single bootstrap genome just completed?
     je .cs_boot                               ; compare its raw cycles with the best seen so far
     cmp r12d, CYC_STATE_A1                    ; a1 done?
     jne .cs_not_a                             ; no: check the later states
     mov [r14 + r13*CYC_QWORD_SCALE], rdx      ; a_min = a1
     mov dword [r10 + r13*CYC_DWORD_SCALE + CYC_POS_STATE_OFF], CYC_STATE_B1 ; state 1: b1 next
     jmp .cs_done                              ; measurement folded, nothing else this state
.cs_not_a:
     cmp r12d, CYC_STATE_B1                    ; b1 done?
     jne .cs_not_b                             ; no: check the later states
     lea r15, [rel cycle_tuner_b_min]          ; r15 = b_min array base
     mov [r15 + r13*CYC_QWORD_SCALE], rdx      ; b_min = b1
; fitness statistics: this b1 delta is one raw timed sample of the genome at
; b's list position -- bucket it by position for the sweep's equal-weight
; position averages. b1 (not the folded min(b1,b2)) keeps one unbiased raw
; sample per position; the minimum stays the sort's verdict metric only.
     mov eax, [r10 + r13*CYC_DWORD_SCALE + CYC_POS_B_OFF] ; eax = b's list position this episode
     lea r11, [rel cycle_tuner_stat_best_sum]  ; r11 = statistics-family base, count and pool arrays live at offsets
     cmp eax, CYC_STAT_BEST_SLOTS              ; inside the elite front window?
     jae .cs_stat_pool                         ; no: only the pool bucket can collect it
     add [r11 + r13*CYC_QWORD_SCALE], rdx      ; best_sum += the b1 measurement
     inc qword [r11 + r13*CYC_QWORD_SCALE + CYC_STAT_ARRAY_BYTES] ; one more front-window sample
.cs_stat_pool:
     mov ecx, CYC_LIST_LAST + 1                ; the population size, 128 slots
     sub ecx, [rel cycle_tuner_tail_slots]     ; minus the mutant tail = the pool limit, >= 1 by the init bound
     cmp eax, ecx                              ; b's position below the pool limit?
     jae .cs_stat_done                         ; no: a tail mutant, no pool bucket
     add [r11 + r13*CYC_QWORD_SCALE + CYC_STAT_POOL_SUM_OFF], rdx ; pool_sum += the b1 measurement
     inc qword [r11 + r13*CYC_QWORD_SCALE + CYC_STAT_POOL_SUM_OFF + CYC_STAT_ARRAY_BYTES] ; one more pool sample
.cs_stat_done:
     mov dword [r10 + r13*CYC_DWORD_SCALE + CYC_POS_STATE_OFF], CYC_STATE_C ; state 2: c next
     jmp .cs_done                              ; measurement folded, statistics bucketed
.cs_not_b:
     cmp r12d, CYC_STATE_C                     ; c done?
     jne .cs_not_c                             ; no: check the later states
     lea rbx, [rel cycle_tuner_c_cycles]       ; rbx = the c measurement array base
     mov [rbx + r13*CYC_QWORD_SCALE], rdx      ; c stored
     mov dword [r10 + r13*CYC_DWORD_SCALE + CYC_POS_STATE_OFF], CYC_STATE_B2 ; state 3: b2 next
     jmp .cs_done                              ; measurement folded, nothing else this state
.cs_not_c:
     cmp r12d, CYC_STATE_B2                    ; b2 done?
     jne .cs_a2                                ; no: state 4, a2 done
     lea r15, [rel cycle_tuner_b_min]          ; r15 = b_min array base
     mov rax, [r15 + r13*CYC_QWORD_SCALE]      ; rax = b1
     cmp rax, rdx                              ; b1 > b2?
     cmova rax, rdx                            ; rax = min(b1, b2)
     mov [r15 + r13*CYC_QWORD_SCALE], rax      ; store the folded b minimum
     mov dword [r10 + r13*CYC_DWORD_SCALE + CYC_POS_STATE_OFF], CYC_STATE_A2 ; state 4: a2 next
     jmp .cs_done                              ; measurement folded, nothing else this state
.cs_a2:
; state 4 complete: fold a, then the scan verdict. The delay genome is an INDEX
; into the pool, never a block copy: every episode's measurements challenge
; it, and the index follows whichever genome beat the running minimum.
     mov rax, [r14 + r13*CYC_QWORD_SCALE]      ; rax = a1
     cmp rax, rdx                              ; a1 > a2?
     cmova rax, rdx                            ; rax = min(a1, a2)
     mov [r14 + r13*CYC_QWORD_SCALE], rax      ; store the folded a minimum
     inc qword [rel cycle_tuner_episodes]      ; one more complete episode
; decide the target index and the exchange flag in one branchless pass: the
; exchange body below clobbers every register that holds a fold-array base.
     lea rbx, [rel cycle_tuner_c_cycles]       ; c measurement array base
     mov rcx, [rbx + r13*CYC_QWORD_SCALE]      ; rcx = c's cycles
     lea r15, [rel cycle_tuner_b_min]          ; b_min array base
     mov rdx, [r15 + r13*CYC_QWORD_SCALE]      ; rdx = min(b1, b2)
     cmp rdx, rcx                              ; b_min < c? the upper genome beat the partner
     setb r8b                                  ; exchange outcome
     mov r11d, [r10 + r13*CYC_DWORD_SCALE + CYC_POS_ACTIVE_OFF] ; delay genome index
     cmp rdx, rax                              ; b_min < a_min?
     cmovb r11d, [r10 + r13*CYC_DWORD_SCALE + CYC_POS_B_OFF] ; switch to b
     cmovae rdx, rax                           ; running minimum stays a_min
     cmp rcx, rdx                              ; c < the running minimum?
     cmovb r11d, [r10 + r13*CYC_DWORD_SCALE + CYC_POS_C_OFF] ; switch to c
     cmovb rdx, rcx                            ; running minimum becomes c
     mov dword cf_tgt, r11d                    ; publish the new delay genome index
     cmp rdx, rax                              ; did b or c strictly beat the delay genome?
     jae .cs_tgtp_done                         ; no: the pointer stays
     inc qword [rel cycle_tuner_promotions]    ; one more delay genome switch
.cs_tgtp_done:
     test r8b, r8b                             ; did b beat c?
     jz .cs_ex_done                            ; no: no trade
     movsxd r12, dword [r10 + r13*CYC_DWORD_SCALE + CYC_POS_B_OFF] ; r12 = b's list index
     movsxd r14, dword [r10 + r13*CYC_DWORD_SCALE + CYC_POS_C_OFF] ; r14 = c's list index
     lea r10, [rel cycle_tuner_parameter_block64k] ; r10 = pointer-array base
     mov rsi, [r10 + r13*CYC_QWORD_SCALE]      ; rsi = the list base
     mov eax, r12d                             ; b is list index 0..127, zero-extension retained
     shl eax, CYC_GENOME_SHIFT                 ; * 512, u32 exact
     lea rdi, [rsi + rax]                      ; rdi = b's genome block
     mov eax, r14d                             ; c is list index 0..127, zero-extension retained
     shl eax, CYC_GENOME_SHIFT                 ; * 512, u32 exact
     lea rbx, [rsi + rax]                      ; rbx = c's genome block
     mov rsi, rdi                              ; rsi = block A, the exchange macro's first operand
     CYC_EXCHANGE_GENOMES                      ; the two blocks traded places
     vzeroupper                                ; retire the dirty ymm0/ymm1: later legacy-SSE consumers pay no transition penalty
; the delay genome pointer names a slot's CONTENT: when the target sat in one
; of the two swapped slots, its content now lives in the other one
     mov eax, dword cf_tgt                     ; eax = the captured target index
     cmp eax, r12d                             ; target was b's slot?
     jne .cs_id_c                              ; no: try c's slot
     mov dword cf_tgt, r14d                    ; its content moved to c's slot
     jmp .cs_ex_done                           ; identity fixed
.cs_id_c:
     cmp eax, r14d                             ; target was c's slot?
     jne .cs_ex_done                           ; neither: the target keeps its content
     mov dword cf_tgt, r12d                    ; its content moved to b's slot
.cs_ex_done:
; publish the delay genome index, after the exchange so it names the genome the
; measurements actually saw
     lea r10, [rel cycle_tuner_position]       ; r10 = position-family base again
     mov eax, dword cf_tgt                     ; eax = the delay genome's new index
     mov [r10 + r13*CYC_DWORD_SCALE + CYC_POS_ACTIVE_OFF], eax ; the delay genome pointer, updated
.cs_advance:
; position advance: +1, wrap 127->0 mutates the tail
     mov eax, [r10 + r13*CYC_DWORD_SCALE]      ; eax = the current anchor
     inc eax                                   ; next anchor position
     cmp eax, CYC_LIST_LAST                    ; past the last slot?
     ja .cs_wrap                               ; yes: wrap + mutate the tail
     mov [r10 + r13*CYC_DWORD_SCALE], eax      ; store the advanced anchor
     jmp .cs_pos_done                          ; no wrap, no tail
.cs_wrap:
     xor eax, eax                              ; the anchor wraps to 0
     mov [r10 + r13*CYC_DWORD_SCALE], eax      ; store the wrapped anchor first
; sweep complete: fold both buckets into their averages, then reset the
; accumulators. Loading a saved cursor may give a partial first sweep,
; so a bucket with no samples keeps its previous average.
     CYC_FOLD cycle_tuner_stat_best_sum, CYC_STAT_BEST_AVG_OFF ; fold the elite front window into its average
     CYC_FOLD cycle_tuner_stat_pool_sum, CYC_STAT_POOL_AVG_OFF ; fold the non-tail pool into its average
     lea r14, [rel cycle_tuner_position]       ; r14 = position-family base, call-stable: the fold reads are done and config_mutate_genome preserves r14
     lea r15, [rel cycle_tuner_parameter_block64k] ; r15 = pointer-array base
     mov r15, [r15 + r13*CYC_QWORD_SCALE]      ; r15 = the list base, config_mutate_genome preserves r15
     mov r12d, [rel cycle_tuner_tail_slots]    ; r12d = the DB-seeded tail count
     test r12d, r12d                           ; tail size zero (tuner:tail_slots = 0)?
     jz .cs_pos_done                           ; yes: no mutant refresh this wrap
.cs_tail:
     dec r12d                                  ; the count-1 .. 0 pass: destinations walk 128-count .. 127
     mov r9d, CYC_LIST_LAST                    ; r9d = 127
     sub r9d, r12d                             ; r9d = the destination's list index, survives the picks
     cmp r9d, [r14 + r13*CYC_DWORD_SCALE + CYC_POS_ACTIVE_OFF] ; the champion's slot?
     je .cs_tail_next                          ; yes: preserve it, no replacement mutation
     CYC_PICK_EXCL                             ; eax = the parent pick, never the destination
     shl eax, CYC_GENOME_SHIFT                 ; * 512
     lea rcx, [r15 + rax]                      ; rcx = the parent's genome block
     CYC_PICK_EXCL                             ; eax = the donor pick, never the destination
     shl eax, CYC_GENOME_SHIFT                 ; * 512
     lea rdx, [r15 + rax]                      ; rdx = the donor's genome block
     mov eax, r9d                              ; eax = the destination index, zero-extends
     shl eax, CYC_GENOME_SHIFT                 ; * 512, u32 exact: list indices are 0..127
     lea r8, [r15 + rax]                       ; r8 = the destination's genome block
     call config_mutate_genome                 ; rcx = parent, rdx = donor, r8 = out: the tail slot holds a mutated child
.cs_tail_next:
     test r12d, r12d                           ; more tail slots?
     jnz .cs_tail                              ; next
.cs_pos_done:
; partner draw: b = the anchor position, c = (g * position) >> 32.
; g is an exp3 fraction (truncated exponential, decay 3) -- the partner
; sweep's winner: strongest and most stable front bias of the ten formulas.
     lea r14, [rel cycle_tuner_position]       ; r14 = position-family base: the wrap path set it before the tail, the no-wrap path arrives straight from .cs_advance
     call cyc_exp_pick                         ; edx = the exp3 pick g
     mov ecx, [r14 + r13*CYC_DWORD_SCALE]      ; ecx = the anchor position
     mov eax, edx                              ; rax = g, zero-extended
     mul ecx                                   ; edx:eax = g * position, edx = the high half = the partner index
     test ecx, ecx                             ; anchor 0? the below-range is empty and the draw degenerates to c = b = 0
     jnz .cs_pair_ok                           ; any other anchor: c < b by construction
     mov ecx, 1                                ; stage the first productive pair instead: b = 1
     xor edx, edx                              ; c = 0, its below-range partner
     mov [r14 + r13*CYC_DWORD_SCALE], ecx      ; the walk cursor follows: position 0 never serves as the b candidate
.cs_pair_ok:
     mov [r14 + r13*CYC_DWORD_SCALE + CYC_POS_B_OFF], ecx ; b = the anchor position
     mov [r14 + r13*CYC_DWORD_SCALE + CYC_POS_C_OFF], edx ; c = the drawn partner, below the anchor
; delay reload: countdown = reload + jitter(0..rand_delay), state 5. The
; jitter varies the delay-phase length so measurements land on different
; phase offsets across episodes and average out more stable.
     mov eax, [r14 + r13*CYC_DWORD_SCALE + CYC_POS_DELAY_RELOAD_OFF] ; eax = this shape's reload value
     mov dword cf_reload, eax                  ; park the clean reload
     mov ecx, [rel cycle_tuner_rand_delay]     ; ecx = the jitter cap
     test ecx, ecx                             ; jitter configured?
     jz .cs_reload_store                       ; no: the exact reload stands
     PROBE_NEXT_RND                            ; eax = low 32 bits of the fresh draw
     inc ecx                                   ; exclusive range = cap + 1, nonzero by the init guard
     mul ecx                                   ; edx = floor(draw * range / 2^32), maps to 0..cap
     mov eax, dword cf_reload                  ; eax = the clean reload back
     add eax, edx                              ; reload + jitter(0..N)
.cs_reload_store:
     mov [r14 + r13*CYC_DWORD_SCALE + CYC_POS_DELAY_COUNTER_OFF], eax ; countdown restarts, jittered
     mov dword [r14 + r13*CYC_DWORD_SCALE + CYC_POS_STATE_OFF], CYC_STATE_DELAY ; state 5: the delay phase runs the delay genome
; install the delay genome: block 0 must carry it for the delay phase
     mov eax, dword cf_tgt                     ; eax = the delay genome's index, published above the advance
     cmp eax, CYC_DELAY_NONE                   ; the delay block as-is?
     jne .cs_adv_list                          ; no: a promoted list member
     mov rcx, [rel cycle_tuner_step_block]     ; rcx = the delay block itself
     jmp .cs_adv_install                       ; install it
.cs_adv_list:
     lea rsi, [rel cycle_tuner_parameter_block64k] ; rsi = pointer-array base
     mov rsi, [rsi + r13*CYC_QWORD_SCALE]      ; rsi = the list base
     shl rax, CYC_GENOME_SHIFT                 ; * 512, u32 exact: eax already zero-extended and holds a valid list index 0..127
     lea rcx, [rsi + rax]                      ; rcx = the delay genome's block
.cs_adv_install:
     mov rdx, [rel cycle_tuner_step_block]     ; rdx = the delay block base
     call cyc_install_genome                   ; block 0 = the delay genome, owned pf patched
     mov dword [rel cycle_tuner_installed_slot], r13d ; block 0 now holds this slot's delay genome
     mov eax, [r14 + r13*CYC_DWORD_SCALE + CYC_POS_ACTIVE_OFF] ; eax = the delay genome's list index (NONE = delay block), r14 survives the call
     mov dword [rel cycle_tuner_installed_gidx], eax ; publish the installed identity for the delay ticks
.cs_done:
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore the dispatcher's registers
.cs_boot:
     lea r11, [rel cycle_tuner_measure_counter] ; r11 = bootstrap countdown array
     mov eax, [r11 + r13*CYC_DWORD_SCALE]      ; eax = the remaining genome count
     dec eax                                   ; eax = just-measured genome index and new count
     mov [r11 + r13*CYC_DWORD_SCALE], eax      ; publish the remaining count
     cmp rdx, [r14 + r13*CYC_QWORD_SCALE]      ; measured cycles below the bootstrap minimum?
     jae .cs_boot_next                         ; no: keep the prior best
     mov [r14 + r13*CYC_QWORD_SCALE], rdx      ; remember this genome's measured cycles
     mov [r10 + r13*CYC_DWORD_SCALE + CYC_POS_ACTIVE_OFF], eax ; publish the winning genome index
.cs_boot_next:
     test eax, eax                             ; all non-tail genomes measured?
     jnz .cs_done                              ; no: run the next genome on the next dispatch
; hand over to the normal sorter: the resumed cursor is the first candidate.
; c is still 0 here (BSS fresh or never staged), so a cursor of 0 would pair
; b = c = 0 and waste the handover episode on a self-comparison.
     cmp dword [r10 + r13*CYC_DWORD_SCALE + CYC_POS_B_OFF], 0 ; degenerate b = c = 0 pair?
     jnz .cs_boot_stage                        ; a real cursor: the pair stands
     mov dword [r10 + r13*CYC_DWORD_SCALE + CYC_POS_B_OFF], 1 ; stage the first productive pair: b = 1, c = 0
.cs_boot_stage:
     mov dword [r10 + r13*CYC_DWORD_SCALE + CYC_POS_STATE_OFF], CYC_STATE_A1 ; next dispatch starts normal sorting without initial delay
     jmp .cs_done                              ; bootstrap finished

; ---------------------------------------------------------------------------
; cyc_write_headers_packed -- stamp the shape header (n, d) into every one
; of the 128 genome slots of a 64K block, so config_set_random_probe's
; window math and every genome copy see a valid header. Takes the packed
; [d | n] registry entry instead of unpacked n/d.
; IN: rcx = 64K block base, rdx = packed [d | n] (n low, d high).
; Clobbers rax/rdx/r8-r10.
; ---------------------------------------------------------------------------
cyc_write_headers_packed:
     xor eax, eax                              ; eax = the current slot's byte offset, initially zero
.cwhp_next:
     mov qword [rcx + rax + SHAPE_BLOCK_N_EQU], rdx ; packed [d | n] writes n at +0 and d at +4 in one store
     add eax, CYC_GENOME_BYTES                 ; advance one 512-byte slot
     cmp eax, (CYC_LIST_LAST + 1) * CYC_GENOME_BYTES ; all 128 slots stamped?
     jb .cwhp_next                             ; next
     ret                                       ; all headers stamped

; ---------------------------------------------------------------------------
; cycle_tuner_alloc_blocks -- on registration, allocate/load the 64 KiB
; population for the newly seen shape. Existing shapes already have blocks.
; Load v7_tune_param_<n>_<d>, or seed all 128 slots from the INI configuration
; through config_set_random_probe, then arm the single-measurement bootstrap.
; Reuses the existing registry walk and population persistence format.
; Preserves rbx/r12-r14/rsi/rdi; clobbers rax/rcx/rdx/r8-r11.
; ---------------------------------------------------------------------------
; internal, entered by a tail jump from cycle_tuner_register_shape
     SF_START cycle_tuner_alloc_blocks, rsi, rdi, r12, r13, r14 ; save persistent registration state
     SF_QWORD cab_delay                        ; the matched per-shape (delay) block base
     SF_COMMIT                                 ; Win64 shadow space and stack alignment
     mov r13d, r9d                             ; new registry slot, passed by the sole caller
     lea r14, [rel g_active_shapes]            ; registry base
     mov eax, r8d                              ; index of this slot's matched configuration
     shl eax, SHAPE_BLOCK_SIZE_SHL             ; convert block index to byte offset
     lea rdx, [rel shape_blocks]               ; configuration table base
     add rdx, rax                              ; address of the matched configuration
     mov cab_delay, rdx                        ; preserve the seed source across allocation and I/O
     mov ecx, CYCLE_TUNER_BLOCK_BYTES          ; allocate only the new slot's population
     call mdb_alloc_mem                        ; rax = new population base
     lea r10, [rel cycle_tuner_parameter_block64k] ; population pointer array
     mov [r10 + r13*CYC_QWORD_SCALE], rax      ; publish ownership before loading
     mov r12, rax                              ; r12 = the new block, parked for the load/fill steps
; try to load the persisted population. The seed path below copies the
; shape's config verbatim into every slot, which carries valid headers, and
; the load path re-stamps stale headers itself, so no stamping here.
     mov ecx, [r14 + r13*CYC_QWORD_SCALE]      ; ecx = n, low half of the packed entry
     mov edx, [r14 + r13*CYC_QWORD_SCALE + CYC_PACKED_D_OFF] ; edx = d, high half of the packed entry
     mov r8, r12                               ; r8 = the 64K block
     call load_param_block                     ; eax = 1 when persisted content was loaded
     test eax, eax                             ; persisted genomes kept?
     jnz .cab_hdr_fix                          ; yes: just fix stale headers, nothing to seed
; no persisted block: seed every genome from the shape's own config, then
; randomize the tunable fields. Installs copy the WHOLE param_set down to
; block 0, so worker_priority/cores_subtract must carry the INI values.
     mov r11d, 128                             ; 128 genomes per 64 KiB
.cab_rand:
     dec r11d                                  ; 127..0
     mov ecx, r11d                             ; rcx = genome slot index, zero-extension retained
     shl ecx, CYC_GENOME_SHIFT                 ; * 512, u32 exact: slots are 127..0
     lea rdi, [r12 + rcx]                      ; rdi = the genome slot
     mov rsi, cab_delay                        ; rsi = the shape's config block
     mov ecx, CYC_GENOME_QWORDS                ; 64 qwords = one genome
     rep movsq                                 ; the slot starts as the shape's config verbatim
     lea rcx, [rdi - CYC_GENOME_BYTES]         ; rep movsq left rdi at slot base + 512
     call config_set_random_probe              ; randomize the tunable fields in place, clobbers rax/rcx/rdx/r8-r10
     test r11d, r11d                           ; more genome slots?
     jnz .cab_rand                             ; seed the next one
     jmp .cab_arm                              ; genomes seeded, arm the machine
.cab_hdr_fix:
; a stale file may carry zero or wrong headers: stamp them again
     mov rdx, [r14 + r13*CYC_QWORD_SCALE]      ; rdx = packed [d | n]
     mov rcx, r12                              ; rcx = the block
     call cyc_write_headers_packed             ; headers valid again
     mov eax, [r12 + CYC_CURSOR_FILE_OFF]      ; recover the saved sort cursor from genome 0's padding
     xor ecx, ecx                              ; the out-of-range fallback: restart the walk at 0
     cmp eax, CYC_LIST_LAST                    ; 0..127 is the legal position range
     cmova eax, ecx                            ; clamp a stale/garbage cursor to the walk start
     lea r10, [rel cycle_tuner_position]       ; r10 = position family base
     mov [r10 + r13*CYC_DWORD_SCALE], eax      ; resume the saved position instead of restarting at 0
     mov [r10 + r13*CYC_DWORD_SCALE + CYC_POS_B_OFF], eax ; the first B measurement uses that position
.cab_arm:
; arm the one-shot bootstrap: the INI configuration remains the startup seed
     mov eax, CYC_LIST_LAST + 1                ; eax = the population size, 128
     sub eax, [rel cycle_tuner_tail_slots]     ; ignore the mutation tail
     lea r10, [rel cycle_tuner_measure_counter] ; r10 = countdown array
     mov [r10 + r13*CYC_DWORD_SCALE], eax      ; one dispatch per non-tail genome
     lea r10, [rel cycle_tuner_a_min]          ; r10 = reused bootstrap minimum array
     mov qword [r10 + r13*CYC_QWORD_SCALE], -1 ; the first genome always becomes the initial best
     lea r10, [rel cycle_tuner_position_state_machine] ; r10 = state array base
     mov dword [r10 + r13*CYC_DWORD_SCALE], CYC_STATE_BOOT ; start scanning this shape immediately
     SF_END rsi, rdi, r12, r13, r14            ; registration and bootstrap setup complete

; ---------------------------------------------------------------------------
; cycle_tuner_save_blocks -- inference-destroy path: persist every allocated
; parameter block to its v7_tune_param_<n>_<d> file. The blocks themselves
; STAY allocated: the tuner activated once per process and its machines keep
; their phase across inference-context destroy/create cycles, so a rarely
; dispatched shape's episode continues at the next chat instead of restarting
; its delay. V7_init frees leftover blocks at the next process lifecycle.
; Walks the shape registry for the per-slot n/d filename pair.
; Preserves rbx/r12-r15/rsi/rdi; clobbers rax/rcx/rdx/r8-r11.
; ---------------------------------------------------------------------------
     global cycle_tuner_save_blocks            ; internal, called from inf_cleanup_alloc
cycle_tuner_save_blocks:
     SF_START cycle_tuner_save_blocks, r12, r13, r14, r15 ; frame: the walk state rides callee-saved registers
     SF_COMMIT                                 ; Win64 shadow space and stack alignment
     lea r14, [rel g_active_shapes]            ; r14 = registry base, indexed below
     lea r15, [rel cycle_tuner_parameter_block64k] ; r15 = pointer-array base, same reason
     xor r13d, r13d                            ; r13d = registry slot index
.csb_next:
     cmp r13d, CYCLE_TUNER_SLOTS               ; designed registry ceiling reached?
     jae .csb_done                             ; full registry: every slot seen
     cmp qword [r15 + r13*CYC_QWORD_SCALE], 0  ; this slot owns a block?
     jz .csb_next_slot                         ; no: nothing to persist
     mov r11, [r14 + r13*CYC_QWORD_SCALE]      ; packed [d | n]
     mov ecx, r11d                             ; rcx = n, low half
     shr r11, 32                               ; r11 = d, high half
     mov edx, r11d                             ; edx = d
     mov r8, [r15 + r13*CYC_QWORD_SCALE]       ; r8 = the 64K block, reloaded from the array: rcx now holds n
     lea r10, [rel cycle_tuner_position]       ; r10 = per-shape cursor array
     mov eax, [r10 + r13*CYC_DWORD_SCALE]      ; eax = this shape's next sort position, 0..127
     mov [r8 + CYC_CURSOR_FILE_OFF], eax       ; persist in genome 0's unused padding, no format expansion
     call save_param_block                     ; persist, error tolerant
.csb_next_slot:
     inc r13d                                  ; next registry slot
     jmp .csb_next                             ; walk until the ceiling
.csb_done:
     SF_END r12, r13, r14, r15                 ; restore and return, every allocated block saved

; ---------------------------------------------------------------------------
; cycle_tuner_print_totals -- destroy-time lifetime summary:
; "cycle tuner: N episode(s), M promotion(s)". The WRITE macros save and
; restore every register they touch.
; Clobbers rax/rcx/rdx/r8-r11 only.
; ---------------------------------------------------------------------------
     global cycle_tuner_print_totals           ; internal, called from v7s_inference_destroy
cycle_tuner_print_totals:
     sub rsp, 8                                ; frameless body enters at 8 mod 16: pad to 0 so the WRITE_* macros' push+sub arithmetic keeps their internal calls 16-aligned
     WRITE_STR cyc_msg_head                    ; "cycle tuner: "
     mov rax, [rel cycle_tuner_episodes]       ; rax = the lifetime episode count
     WRITE_UINT64 rax                          ; the episode count
     WRITE_STR cyc_msg_mid                     ; " episode(s), "
     mov rax, [rel cycle_tuner_promotions]     ; rax = the lifetime promotion count
     WRITE_UINT64 rax                          ; the promotion count
     WRITE_STR_NL cyc_msg_tail                 ; " promotion(s)" + newline
     add rsp, 8                                ; drop the alignment pad
     ret                                       ; the lifetime summary is printed

; ---------------------------------------------------------------------------
; v7_get_best_fitness -- public query: pointer to the per-shape elite-front
; averages (u64 per registry slot, 64 entries), the front-10 b1-measurement
; mean folded at each population sweep wrap. 0 in a slot = no completed sweep.
; Clobbers rax only.
; ---------------------------------------------------------------------------
     global v7_get_best_fitness                ; export public: per-shape front-10 fitness averages
v7_get_best_fitness:
     lea rax, [rel cycle_tuner_stat_best_avg]  ; the array base is the answer
     ret                                       ; plain return

; ---------------------------------------------------------------------------
; v7_get_pool_fitness -- public query: pointer to the per-shape pool averages
; (u64 per registry slot, 64 entries), every non-tail genome's b1-measurement
; mean folded at each population sweep wrap. 0 in a slot = no completed sweep.
; Clobbers rax only.
; ---------------------------------------------------------------------------
     global v7_get_pool_fitness                ; export public: per-shape non-tail pool fitness averages
v7_get_pool_fitness:
     lea rax, [rel cycle_tuner_stat_pool_avg]  ; the array base is the answer
     ret                                       ; plain return

; ---------------------------------------------------------------------------
; v7_get_active_shape_count -- public query: how many shapes the dispatcher
; registered, also the valid index range of both fitness arrays. Walks the
; registry to its zero sentinel.
; Clobbers rax/rcx.
; ---------------------------------------------------------------------------
     global v7_get_active_shape_count          ; export public: registry occupancy, the fitness arrays' index bound
v7_get_active_shape_count:
     xor eax, eax                              ; slot cursor, the registry appends from 0
     lea rcx, [rel g_active_shapes]            ; registry base, packed [d|n] qwords, 0 = end sentinel
.gasc_next:
     cmp qword [rcx + rax*CYC_QWORD_SCALE], 0  ; this slot occupied?
     je .gasc_done                             ; the zero sentinel: the walk ends
     inc eax                                   ; one more active shape
     cmp eax, CYCLE_TUNER_SLOTS                ; the designed 64-slot ceiling?
     jb .gasc_next                             ; walk on
.gasc_done:
     ret                                       ; eax = active shape count

