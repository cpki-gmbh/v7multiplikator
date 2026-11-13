; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
; ============================================================================
; Constants for internal RoPE table building
; ============================================================================
     ROPE_INTERNAL_MAX_SEQ_LEN equ 131072      ; Max positions in internal tables
; Named stack-frame offsets for rope_build_internal_tables, manual RBP frame
%define RBIT_HALF_OFF -4                       ; half = head_dim/2, DWORD
%define RBIT_SIZE_OFF -16                      ; size in bytes per table, QWORD
%define RBIT_POS_OFF -32                       ; pos outer loop counter, DWORD
%define RBIT_DIM_OFF -36                       ; j inner loop counter, DWORD
%define RBIT_BASEVAL_OFF -40                   ; cached rope_base float value, DWORD, hoisted out of loop
%define RBIT_DFACTOR_OFF -64                   ; llama3 factor: DB __sys_rope_factor, float32 -> double; FATAL if missing
%define RBIT_DHIGH_OFF -72                     ; llama3 high_freq_factor: DB __sys_rope_high_freq_factor, float32 -> double; FATAL if missing
%define RBIT_DLOW_OFF -80                      ; llama3 low_freq_factor: DB __sys_rope_low_freq_factor, float32 -> double; FATAL if missing
%define RBIT_DOLD_OFF -88                      ; llama3 orig_max_pos: DB __sys_rope_orig_max_pos, int32 -> double; FATAL if missing
%define RBIT_DLOWWL_OFF -96                    ; low_freq_wavelen = old_ctx/low_freq_factor, QWORD double
%define RBIT_DHIGHWL_OFF -104                  ; high_freq_wavelen = old_ctx/high_freq_factor, QWORD double
%define RBIT_TMPW_OFF -112                     ; scratch: wavelen = 2*pi/inv_freq, QWORD double
%define RBIT_TMPIF_OFF -120                    ; scratch: inv_freq, then smoothed_inv_freq, QWORD double
%define RBIT_TMPIFL_OFF -128                   ; scratch: inv_freq_llama, QWORD double
%define RBIT_TMPSF_OFF -136                    ; scratch: smooth_factor, QWORD double
%define RBIT_CW_OFF -144                       ; saved caller x87 control word, WORD, restored at exit
%define RBIT_TRIGCW_OFF -146                   ; staged 0x037F trig control word, WORD
%define RBIT_TRIG_CW 0x037F                    ; exact CW finit produced: extended PC, nearest, all exceptions masked
%define RBIT_FREQ_OFF -2200                    ; freq[0] for llama3 precompute, QWORD double
%define RBIT_DOUBLE_STRIDE 8                   ; byte stride per QWORD double in freq buffer
%define RBIT_FRAME_SIZE 2240                   ; frame: 32B pad, then the 2048-byte freq buffer at rsp+40, then the locals. The buffer must start at or above rsp+40: the 32 bytes a callee may write are [rsp, rsp+32
%define RBIT_SAVED_REGS_SIZE 24                ; 3 saved non-volatile registers, rbp+rbx+rsi = 3*8 = 24 bytes
; Named stack-argument offsets for v7s_fused_f32_chunk, after SF_START frame
%define FK_ARG5_D_OFF 48                       ; 5th caller arg: d, feature dimension
%define FK_ARG6_COFF_OFF 56                    ; 6th caller arg: col_off, column offset
; in the {int8_t* q; float* s} QuantizedTensor pair
; Loop-unroll element counts, used in LEA arithmetic, not struct field accesses
%define FK_UNROLL_128 128                      ; 128-element SIMD unroll threshold
%define FK_UNROLL_64 64                        ; 64-element unroll threshold
%define FK_SCALE_NEXT_OFF 4                    ; Byte offset to second scale float, sizeof,float
%define FK_FLOAT_BYTES 4                       ; Bytes per float element, for byte-offset arithmetic
%define FK_HEAD_DATA_SCALE 4                   ; SIB scale: head element offset * sizeof,float inside the postprocess LEA
%define FK_1YMM 32                             ; 1 YMM vector ahead, 8 floats = 32 bytes
%define FK_2YMM 64                             ; 2 YMM vectors ahead, 16 floats = 64 bytes
%define FK_3YMM 96                             ; 3 YMM vectors ahead, 24 floats = 96 bytes
%define FK_4YMM 128                            ; 4 YMM vectors ahead, 32 floats = 128 bytes
%define FK_5YMM 160                            ; 5 YMM vectors ahead, 40 floats = 160 bytes
%define FK_6YMM 192                            ; 6 YMM vectors ahead, 48 floats = 192 bytes
%define FK_7YMM 224                            ; 7 YMM vectors ahead, 56 floats = 224 bytes
%define FK_ACT_PREFETCH_DIST 512               ; Activation prefetch BYTE displacement, 8 cache lines ; slot 12, SMC-patched per shape
%define FK_WT_PREFETCH_DIST 320                ; Weight prefetch BYTE displacement, 5 cache lines ; slot 11, SMC-patched per shape
     section .rdata
rpe_base_key:
     db "__sys_rope_base", 0                   ; microDB key for RoPE base frequency
rpe_type_key:
     db "__sys_rope_type", 0                   ; microDB key for rope_scaling type, 0=none, 1=llama3
rpe_factor_key:
     db "__sys_rope_factor", 0                 ; microDB key for llama3 low-freq division factor
rpe_highfreq_key:
     db "__sys_rope_high_freq_factor", 0       ; microDB key for llama3 high-freq factor
rpe_lowfreq_key:
     db "__sys_rope_low_freq_factor", 0        ; microDB key for llama3 low-freq factor
rpe_origpos_key:
     db "__sys_rope_orig_max_pos", 0           ; microDB key for llama3 original max position
     align 8
rbit_two_pi:
     dq 6.28318530717958647692                 ; 2*pi for wavelen computation, llama3 scaling, aligned for the fld qword that reads it
     section .text
rope_build_internal_tables:
     push rbp                                  ; Save caller frame pointer
     push rbx                                  ; Save non-volatile rbx, used for cos table ptr
     push rsi                                  ; Save non-volatile rsi, used for sin table ptr
     mov rbp, rsp                              ; Set the stack-frame base pointer.
     sub rsp, RBIT_FRAME_SIZE                  ; Frame: pad + 2048-byte freq buffer at rsp+40 + locals, see RBIT_FRAME_SIZE
     INT3_DBG_STR "rbit:prologue"              ; Debug trace: rope_build entry
; Save the caller's x87 control word before any FATAL site, every FATAL in
; this function fires before the fldcw below, so no unwound path leaves the
; 0x037F trig state installed; any NEW FATAL added after the fldcw must restore
; this slot first, the fill-completion exit restores it at the function remainder.
     fnstcw word [rbp + RBIT_CW_OFF]           ; save caller x87 control word, slot written before any branch
; Get head_dim, import-borne config: the DB builder owns its validity
     mov eax, [fused_head_dim]                 ; load head_dim directly into the working register
     shr eax, 1                                ; half = head_dim / 2
     mov [rbp + RBIT_HALF_OFF], eax            ; keep the half-dimension slot for later uses
; size = ROPE_INTERNAL_MAX_SEQ_LEN * half * sizeof,float
     shl rax, 19                               ; size = half << 19, ROPE_INTERNAL_MAX_SEQ_LEN * 4 = 2^19
     mov [rbp + RBIT_SIZE_OFF], rax            ; keep the size slot for the second allocation
     mov rcx, rax                              ; pass size without reloading the frame
     call mdb_alloc_mem                        ; rax = cos table base, fbm fallback on OOM; never NULL
     mov rbx, rax                              ; rbx = cos table base, saved, the global publishes only after the fill completes
     INT3_DBG_STR "rbit:cos_alloc"             ; Debug trace: cos table allocated
; Allocate sin table
     mov rcx, [rbp + RBIT_SIZE_OFF]            ; arg1: bytes per table
     call mdb_alloc_mem                        ; rax = sin table base, fbm fallback on OOM; never NULL
     mov rsi, rax                              ; rsi = sin table base, saved, the global publishes only after the fill completes
     INT3_DBG_STR "rbit:sin_alloc"             ; Debug trace: sin table allocated
; ---- Load RoPE params from microDB once, hoisted out of all loops ----
; rope_base: float32; key is mandatory, FATAL if missing, no silent default
     lea rcx, [rel rpe_base_key]               ; arg1: key string for ht_find_or_fatal
     xor edx, edx                              ; arg2: out_len = NULL
     call ht_find_or_fatal                     ; rax = data ptr, FATAL with key name on miss
     mov eax, [rax]                            ; Load rope_base float32 from DB
     mov [rbp + RBIT_BASEVAL_OFF], eax         ; Cache base value for inner loop
; rope_scaling type: int32; key is mandatory, FATAL if missing, no silent default
     lea rcx, [rel rpe_type_key]               ; arg1: key string for ht_find_or_fatal
     xor edx, edx                              ; arg2: out_len = NULL
     call ht_find_or_fatal                     ; rax = data ptr, FATAL with key name on miss
     cmp dword [rax], 1                        ; rope_scaling type == 1 (llama3 scaling)?
     je .rbit_llama3_setup                     ; Yes -> load scaling params + precompute
     mov word [rbp + RBIT_TRIGCW_OFF], RBIT_TRIG_CW ; stage 0x037F, dedicated word slot
     fldcw word [rbp + RBIT_TRIGCW_OFF]        ; apply: extended PC, round-nearest, all masked
     INT3_DBG_STR "rbit:finit_done"            ; Debug trace: x87 state applied
; ---- plain, type==0 : precompute freqs[j] = base^,-j/half once ----
; freq[] is 256 doubles, RBIT_FREQ_OFF, 2048 bytes : sized for half <= 256, head_dim <= 512 ; all supported models use head_dim <= 128
     mov dword [rbp + RBIT_DIM_OFF], 0         ; j = 0 for plain precompute loop
; RBIT_CALC_INV_FREQ -- st0 = base^(-j/half) on exit, exactly one x87 register
; pushed, control word untouched. The shared chain of the plain and the
; llama3 precompute loops; j/half/base come from the frame slots.

%macro RBIT_CALC_INV_FREQ 0
     fld dword [rbp + RBIT_BASEVAL_OFF]        ; st0 = rope_base
     fild dword [rbp + RBIT_DIM_OFF]           ; st0 = j, st1 = base
     fild dword [rbp + RBIT_HALF_OFF]          ; st0 = half, st1 = j, st2 = base
     fdivp st1, st0                            ; st1 = j/half, pop -> st0 = j/half, st1 = base
     fchs                                      ; st0 = -j/half, st1 = base
     fxch st1                                  ; st0 = base, st1 = -j/half
     fyl2x                                     ; st0 =, -j/half *log2,base
     fld st0                                   ; Duplicate log2 result for int/frac split
     frndint                                   ; st0 = int_part, st1 = log2 result
     fsub st1, st0                             ; st1 = frac = log2 - int_part
     fxch st1                                  ; st0 = frac, st1 = int
     f2xm1                                     ; st0 = 2^frac - 1
     fld1                                      ; st0 = 1, st1 = 2^frac-1
     faddp st1, st0                            ; st0 = 2^frac
     fscale                                    ; st0 = 2^frac * 2^int = freq
     fstp st1                                  ; pop int, leave freq in st0
%endmacro
.rbit_plain_prefreq:
     mov eax, [rbp + RBIT_DIM_OFF]             ; Load current j
     cmp eax, [rbp + RBIT_HALF_OFF]            ; Compare against half
     jge .rbit_pos_init                        ; Done precomputing -> init pos + enter pos loop
; freq = base^,-j/half via x87 fyl2x/f2xm1/fscale, same chain as .rbit_prefreq Step A, the two scaling paths must stay in sync
     RBIT_CALC_INV_FREQ                        ; st0 = freq, one register pushed
; Store as double in freq buffer, same layout as llama3 path
     fstp qword [rbp + RBIT_FREQ_OFF + rax*RBIT_DOUBLE_STRIDE] ; freqs[j] = base^,-j/half
     inc dword [rbp + RBIT_DIM_OFF]            ; Increment j
     jmp .rbit_plain_prefreq                   ; Continue precompute
; ---- llama3, type==1 : load scaling params + precompute freqs[] ----
.rbit_llama3_setup:
; factor: float32 -> double; key is mandatory, FATAL if missing, no silent default
     lea rcx, [rel rpe_factor_key]             ; arg1: key string for ht_find_or_fatal
     xor edx, edx                              ; arg2: out_len = NULL
     call ht_find_or_fatal                     ; rax = data ptr, FATAL with key name on miss
     fld dword [rax]                           ; st0 = factor, float32
     fstp qword [rbp + RBIT_DFACTOR_OFF]       ; Cache factor as double
; high_freq_factor: float32 -> double; key is mandatory, FATAL if missing, no silent default
     lea rcx, [rel rpe_highfreq_key]           ; arg1: key string for ht_find_or_fatal
     xor edx, edx                              ; arg2: out_len = NULL
     call ht_find_or_fatal                     ; rax = data ptr, FATAL with key name on miss
     fld dword [rax]                           ; st0 = high_freq_factor
     fstp qword [rbp + RBIT_DHIGH_OFF]         ; Cache high_freq_factor as double
; low_freq_factor: float32 -> double; key is mandatory, FATAL if missing, no silent default
     lea rcx, [rel rpe_lowfreq_key]            ; arg1: key string for ht_find_or_fatal
     xor edx, edx                              ; arg2: out_len = NULL
     call ht_find_or_fatal                     ; rax = data ptr, FATAL with key name on miss
     fld dword [rax]                           ; st0 = low_freq_factor
     fstp qword [rbp + RBIT_DLOW_OFF]          ; Cache low_freq_factor as double
; orig_max_pos: int32 -> double; key is mandatory, FATAL if missing, no silent default
     lea rcx, [rel rpe_origpos_key]            ; arg1: key string for ht_find_or_fatal
     xor edx, edx                              ; arg2: out_len = NULL
     call ht_find_or_fatal                     ; rax = data ptr, FATAL with key name on miss
     fild dword [rax]                          ; st0 = orig_max_pos, int32 -> double
     fstp qword [rbp + RBIT_DOLD_OFF]          ; Cache orig_max_pos as double
     mov word [rbp + RBIT_TRIGCW_OFF], RBIT_TRIG_CW ; stage 0x037F, dedicated word slot
     fldcw word [rbp + RBIT_TRIGCW_OFF]        ; apply: extended PC, round-nearest, all masked
; low_freq_wavelen = old_ctx / low_freq_factor
     fld qword [rbp + RBIT_DOLD_OFF]           ; st0 = old_ctx
     fdiv qword [rbp + RBIT_DLOW_OFF]          ; st0 = old_ctx / low_freq_factor
     fstp qword [rbp + RBIT_DLOWWL_OFF]        ; Cache low_freq_wavelen
; high_freq_wavelen = old_ctx / high_freq_factor
     fld qword [rbp + RBIT_DOLD_OFF]           ; st0 = old_ctx
     fdiv qword [rbp + RBIT_DHIGH_OFF]         ; st0 = old_ctx / high_freq_factor
     fstp qword [rbp + RBIT_DHIGHWL_OFF]       ; Cache high_freq_wavelen
; freq[] is 256 doubles, RBIT_FREQ_OFF, 2048 bytes : sized for half <= 256, head_dim <= 512 ; all supported models use head_dim <= 128
; Precompute freqs[j] = scaled inv_freq for j = 0 .. half-1, HF llama3 formula
     mov dword [rbp + RBIT_DIM_OFF], 0         ; Reuse j counter for precompute loop
.rbit_prefreq:
     mov eax, [rbp + RBIT_DIM_OFF]             ; Load current j
     cmp eax, [rbp + RBIT_HALF_OFF]            ; Compare against half
     jge .rbit_pos_init                        ; Done precomputing -> init pos + enter pos loop
; Step A: inv_freq = base^,-j/half via the shared x87 chain
     RBIT_CALC_INV_FREQ                        ; st0 = inv_freq, one register pushed
     fst qword [rbp + RBIT_TMPIF_OFF]          ; save inv_freq, keep st0
; Step B: wavelen = 2*pi / inv_freq
     fld qword [rbit_two_pi]                   ; st0 = 2*pi, st1 = inv_freq
     fdivrp st1, st0                           ; reverse divide: st1 = 2*pi/inv_freq, pop -> st0 = wavelen
     fstp qword [rbp + RBIT_TMPW_OFF]          ; save wavelen
; Step C: inv_freq_llama =, wavelen > low_wl ? inv_freq/factor : inv_freq
     fld qword [rbp + RBIT_TMPIF_OFF]          ; st0 = inv_freq
     fld qword [rbp + RBIT_TMPW_OFF]           ; st0 = wavelen, st1 = inv_freq
     fld qword [rbp + RBIT_DLOWWL_OFF]         ; st0 = low_wl, st1 = wavelen, st2 = inv_freq
     fcomip st0, st1                           ; flags: low_wl vs wavelen; pop -> st0 = wavelen, st1 = inv_freq
     jc .rbit_pf_ifl_low                       ; low_wl < wavelen -> wavelen > low_wl -> divide by factor
     fxch st1                                  ; st0 = inv_freq, st1 = wavelen
     jmp .rbit_pf_ifl_done                     ; wavelen <= low_wl -> keep inv_freq
.rbit_pf_ifl_low:
     fxch st1                                  ; st0 = inv_freq, st1 = wavelen
     fdiv qword [rbp + RBIT_DFACTOR_OFF]       ; st0 = inv_freq/factor, low-frequency regime
.rbit_pf_ifl_done:
     fst qword [rbp + RBIT_TMPIFL_OFF]         ; save inv_freq_llama, keep st0
     fstp st0                                  ; pop inv_freq_llama -> st0 = wavelen
     fstp st0                                  ; pop wavelen -> stack empty
; Step D: smooth_factor =, old_ctx/wavelen - low_freq_factor /,high_freq_factor - low_freq_factor
     fld qword [rbp + RBIT_DOLD_OFF]           ; st0 = old_ctx
     fdiv qword [rbp + RBIT_TMPW_OFF]          ; st0 = old_ctx/wavelen
     fsub qword [rbp + RBIT_DLOW_OFF]          ; st0 = old_ctx/wavelen - low_freq_factor
     fld qword [rbp + RBIT_DHIGH_OFF]          ; st0 = high_freq_factor, st1 = numerator
     fsub qword [rbp + RBIT_DLOW_OFF]          ; st0 = high_freq_factor - low_freq_factor
     fdivp st1, st0                            ; st1 = numerator/denominator, pop -> st0 = smooth_factor
     fstp qword [rbp + RBIT_TMPSF_OFF]         ; save smooth_factor
; Step E: smoothed =, 1-smooth *inv_freq_llama/factor + smooth*inv_freq_llama
     fld1                                      ; st0 = 1.0
     fsub qword [rbp + RBIT_TMPSF_OFF]         ; st0 = 1 - smooth_factor
     fld qword [rbp + RBIT_TMPIFL_OFF]         ; st0 = inv_freq_llama, st1 = 1-smooth
     fdiv qword [rbp + RBIT_DFACTOR_OFF]       ; st0 = inv_freq_llama/factor
     fmulp st1, st0                            ; st1 =, 1-smooth *ifl/factor, pop -> st0 = term1
     fld qword [rbp + RBIT_TMPSF_OFF]          ; st0 = smooth_factor, st1 = term1
     fld qword [rbp + RBIT_TMPIFL_OFF]         ; st0 = inv_freq_llama, st1 = smooth, st2 = term1
     fmulp st1, st0                            ; st1 = smooth*ifl, pop -> st0 = term2, st1 = term1
     faddp st1, st0                            ; st1 = term1+term2, pop -> st0 = smoothed
     fstp qword [rbp + RBIT_TMPIF_OFF]         ; save smoothed for the Step F medium test; .rbit_pos_init reuses this slot for pos
; Step F: is_medium =, wavelen >= high_wl &&, wavelen <= low_wl ; if so use smoothed
     fld qword [rbp + RBIT_TMPW_OFF]           ; st0 = wavelen
     fld qword [rbp + RBIT_DHIGHWL_OFF]        ; st0 = high_wl, st1 = wavelen
     fcomip st0, st1                           ; flags: high_wl vs wavelen; pop -> st0 = wavelen
     jc .rbit_pf_checklow                      ; high_wl < wavelen -> may be medium, check low bound
     jz .rbit_pf_medium                        ; high_wl == wavelen -> medium boundary -> use smoothed
     jmp .rbit_pf_not_medium                   ; high_wl > wavelen -> wavelen < high_wl -> not medium
.rbit_pf_checklow:
     fld qword [rbp + RBIT_DLOWWL_OFF]         ; st0 = low_wl, st1 = wavelen
     fcomip st0, st1                           ; flags: low_wl vs wavelen; pop -> st0 = wavelen
     jnc .rbit_pf_medium                       ; low_wl >= wavelen -> wavelen <= low_wl -> medium
.rbit_pf_not_medium:
     fstp st0                                  ; pop wavelen -> stack empty
     fld qword [rbp + RBIT_TMPIFL_OFF]         ; st0 = inv_freq_llama, final freq
     jmp .rbit_pf_final                        ; Store final freq
.rbit_pf_medium:
     fstp st0                                  ; pop wavelen -> stack empty
     fld qword [rbp + RBIT_TMPIF_OFF]          ; st0 = smoothed_inv_freq, final freq
.rbit_pf_final:
     mov eax, [rbp + RBIT_DIM_OFF]             ; j index into freq buffer
     fstp qword [rbp + RBIT_FREQ_OFF + rax*RBIT_DOUBLE_STRIDE] ; freqs[j] = final freq
; Advance precompute loop counter
     inc dword [rbp + RBIT_DIM_OFF]            ; Increment j
     jmp .rbit_prefreq                         ; Continue precompute loop
; Outer loop: pos = 0 .. ROPE_INTERNAL_MAX_SEQ_LEN-1
.rbit_pos_init:
     mov dword [rbp + RBIT_POS_OFF], 0         ; Initialize position counter to 0
     mov r11d, [rbp + RBIT_HALF_OFF]           ; r11 = half, inner-loop bound; register-held for the whole fill
.rbit_outer:
     mov eax, [rbp + RBIT_POS_OFF]             ; Load current position
     cmp eax, ROPE_INTERNAL_MAX_SEQ_LEN        ; Check if we've filled all positions
     jge .rbit_done                            ; Exit outer loop if done
; Compute pos * half once per position, so the inner j loop costs no imul
     imul eax, r11d                            ; pos * half, 32-bit, fits: max 131072*256=33M
     fild dword [rbp + RBIT_POS_OFF]           ; st0 =, double pos
     fstp qword [rbp + RBIT_TMPIF_OFF]         ; cache as double, lossless for integers <= 2^53
     mov edx, eax                              ; rdx = linear flat-table offset = base, advances with j below
     xor r10d, r10d                            ; r10 = dimension index j = 0, re-zeroed per position, held across the whole inner j loop
.rbit_inner:
; Load precomputed freq[j] from the one shared buffer, written by .rbit_plain_prefreq and .rbit_prefreq Step F
     fld qword [rbp + r10*RBIT_DOUBLE_STRIDE + RBIT_FREQ_OFF] ; st0 = freqs[j]
; angle = pos * freq, pos cached as double, avoiding per-iteration fild
     fmul qword [rbp + RBIT_TMPIF_OFF]         ; st0 = freq * pos
; sin/cos via fsincos
     fsincos                                   ; st0 = cos,angle, st1 = sin,angle
; store cos, then sin, both fstp pop the stack ; rdx = pos*half + j already
     fstp dword [rbx + rdx*FK_FLOAT_BYTES]     ; Store cos,angle and pop, rbx = hoisted cos base
     fstp dword [rsi + rdx*FK_FLOAT_BYTES]     ; Store sin,angle and pop, rsi = hoisted sin base
; Trace: dump ptrs + first store value, check: release builds would run
; the cmp/jne pairs and 2 unused rcx loads per inner-loop iteration for nothing
%ifdef ZYDIS_TRACE
     cmp dword [rbp + RBIT_POS_OFF], 0         ; First position?
     jne .rbit_skip_debug                      ; No, skip debug
     test r10d, r10d                           ; First dimension?, register-held j
     jne .rbit_skip_debug                      ; No, skip debug
     mov rcx, rbx                              ; cos table base, register-saved during the fill
     INT3_DBG_REG 1                            ; Dump RCX, cos ptr
     mov rcx, rsi                              ; sin table base, register-saved during the fill
     INT3_DBG_REG 1                            ; Dump RCX, sin ptr
     INT3_DBG_STR "rbit:store0"                ; Both stores completed for j=0
.rbit_skip_debug:
%endif
; Advance inner loop counters, both register-held, no memory RMW
     inc r10d                                  ; Increment dimension index j
     inc edx                                   ; Increment linear flat-table offset
     cmp r10d, r11d                            ; j < half, both registers ?
     jl .rbit_inner                            ; Continue inner loop
.rbit_inner_done:
; Advance outer loop counter
     inc dword [rbp + RBIT_POS_OFF]            ; Increment position counter
     jmp .rbit_outer                           ; Continue outer loop
.rbit_done:
; Fill loops complete: publish the tables only now, a caught FATAL
; during the fill leaves the globals NULL/owned=0, no half-built state
     mov [fused_rope_cos], rbx                 ; publish cos table pointer
     mov [fused_rope_sin], rsi                 ; publish sin table pointer
     mov dword [fused_rope_owned], 1           ; mark internally owned; external set_rope_tables pointers keep 0 and are never mdb_free_mem'd
     fldcw word [rbp + RBIT_CW_OFF]            ; restore caller x87 control word, undo the 0x037F trig state
     INT3_DBG_STR "rbit:all_done"              ; Trace before function return
     add rsp, RBIT_FRAME_SIZE                  ; Deallocate local stack frame, matching sub rsp, RBIT_FRAME_SIZE
%ifdef ZYDIS_TRACE
     mov rax, [rsp + RBIT_SAVED_REGS_SIZE]     ; Load return address, 3 saved regs above
     INT3_DBG_REG 0                            ; Dump RAX, return addr
     INT3_DBG_REG 5                            ; Dump RBP, frame ptr
%endif
     pop rsi                                   ; Restore non-volatile rsi, hoisted sin ptr
     pop rbx                                   ; Restore non-volatile rbx, hoisted cos ptr
     pop rbp                                   ; Restore caller frame pointer
     ret                                       ; Return to caller
     global v7s_fused_f32_chunk                ; Export: fused F32 matmul + RMSNorm + RoPE chunk
v7s_fused_f32_chunk:
     SF_START v7s_fused_f32_chunk, rbx, rsi, rdi, r12, r13, r14, r15 ; Function prologue: save non-volatiles
; f32 matmul slots
     SF_XMM fk_xmm6                            ; Save xmm6 for kernel use
     SF_XMM fk_xmm7                            ; Save xmm7 for kernel use
     SF_XMM fk_xmm8                            ; Save xmm8 for kernel use
     SF_XMM fk_xmm9                            ; Save xmm9 for kernel use
     SF_XMM fk_xmm10                           ; Save xmm10 for kernel use
     SF_XMM fk_xmm11                           ; Save xmm11 for kernel use
     SF_XMM fk_xmm12                           ; Save xmm12 for kernel use
     SF_XMM fk_xmm13                           ; Save xmm13 for kernel use
; rmsnorm_rope slots
     SF_DWORD fk_hd                            ; head_dim for rmsnorm_rope
     SF_DWORD fk_dim_f                         ; float head_dim, RMSNorm divisor, single correct rounding
     SF_DWORD fk_nheads                        ; Number of query heads
     SF_DWORD fk_pos                           ; Current sequence position for RoPE
     SF_QWORD fk_nw                            ; Weight quantized base pointer
     SF_QWORD fk_rc                            ; RoPE cosine table pointer
     SF_QWORD fk_rs                            ; RoPE sine table pointer
     SF_QWORD fk_xo                            ; Output buffer pointer, xout
     SF_DWORD fk_d                             ; Feature dimension
     SF_DWORD fk_col_off                       ; Column offset for head alignment
     SF_DWORD fk_n                             ; Input element count
; Per-head interleaved base pointer save slots
     SF_QWORD fk_x_base                        ; Saved input activations base pointer, x
     SF_QWORD fk_wq_base                       ; Saved weight quantized base, wq
     SF_QWORD fk_ws_base                       ; Saved weight scales base, ws
     SF_QWORD fk_bias_base                     ; Saved QKV bias base pointer, NULL = add zeros, safe no-op
     SF_DWORD fk_head_idx                      ; Current head index save slot
     SF_XMM fk_symm14                          ; Save XMM14 low half for RMSNorm scale, upper YMM half is volatile scratch
     SF_COMMIT                                 ; Commit fused_f32_chunk stack frame
     vmovups fk_xmm6, xmm6                     ; Save XMM6, non-volatile
     vmovups fk_xmm7, xmm7                     ; Save XMM7, non-volatile
     vmovups fk_xmm8, xmm8                     ; Save XMM8, non-volatile
     vmovups fk_xmm9, xmm9                     ; Save XMM9, non-volatile
     vmovups fk_xmm10, xmm10                   ; Save XMM10, non-volatile
     vmovups fk_xmm11, xmm11                   ; Save XMM11, non-volatile
     vmovups fk_xmm12, xmm12                   ; Save XMM12, non-volatile
     vmovups fk_xmm13, xmm13                   ; Save XMM13, non-volatile, genuine load temp in the matmul body
     vmovdqu fk_symm14, xmm14                  ; Save XMM14 low half, non-volatile for RMSNorm scale
; RoPE tables are built once by initialize_once or set_fused_rope_pointers
; not called from the frequently executed path, no lock needed in the frequently executed path .
; Check the INT3 marker at function entry to verify the VEH handler.
     INT3_DBG_STR "fk:entry"                   ; Debug trace: fused_f32_chunk entry
     INT3_DBG_REG 1                            ; RCX = xout
     INT3_DBG_REG 2                            ; RDX = &input_q
     INT3_DBG_REG 8                            ; R8 = &weight_q
     INT3_DBG_REG 9                            ; R9D = n
; Save fused call arguments from registers and stack
     mov fk_xo, rcx                            ; Save xout, output float pointer
     mov fk_n, r9d                             ; Save n, number of input elements
     mov eax, dword [rbp + FK_ARG5_D_OFF]      ; Load d, feature dimension from 5th arg
     INT3_DBG_REG 0                            ; RAX = d
     mov fk_d, eax                             ; Save feature dimension
     mov eax, dword [rbp + FK_ARG6_COFF_OFF]   ; Load column_offset from 6th arg
     INT3_DBG_REG 0                            ; RAX = col_off
     mov fk_col_off, eax                       ; Save column offset for head alignment check
; Save base pointers for per-head interleaved matmul
     mov r12, rcx                              ; r12 = xout, also in fk_xo
     mov r13, [rdx]                            ; r13 = input activations float* x
     mov fk_x_base, r13                        ; Save input activations base
     mov rax, [r8]                             ; Load weight quantized base into scratch
     mov fk_wq_base, rax                       ; Save weight quantized base, mem-to-mem mov impossible, use register hop
     mov r14, [r8 + QT_S]                      ; Load weight scales base into scratch
     mov fk_ws_base, r14                       ; Save weight scales base
; head_dim comes from the import, its validity is the DB builder's contract
     mov eax, [fused_head_dim]                 ; Load head dimension from global
     mov fk_hd, eax                            ; Save head_dim locally
; Cache, float head_dim for the frequently executed-path RMSNorm division
     vcvtsi2ss xmm0, xmm0, eax                 ; xmm0 =, float head_dim, VEX-encoded, avoids AVX-SSE transition
     vmovss fk_dim_f, xmm0                     ; store for frequently executed-path RMSNorm, kernel divides by it
; d must be evenly divisible by head_dim, import contract ; the quotient is nheads
     mov eax, fk_d                             ; Copy feature dimension for division
     xor edx, edx                              ; Clear high 32 bits for div
     div dword fk_hd                           ; Compute d / head_dim, remainder is the import's contract, not ours
     mov fk_nheads, eax                        ; Save number of heads to process. Never 0: eax = d/head_dim with d % head_dim == 0
; entry test also exits immediately on a 0 count.
%ifdef ZYDIS_TRACE
     push rax                                  ; save n_heads for WRITE_* below
     INT3_DBG_STR "fk:START"                   ; Debug trace: f32 fused kernel head loop start
     mov eax, fk_d                             ; Load d
     INT3_DBG_REG 0                            ; RAX = d
     mov eax, fk_col_off                       ; Load column_offset
     INT3_DBG_REG 0                            ; RAX = col_off
     INT3_DBG_STR "nh="                        ; Debug trace: n_heads value
     pop rax                                   ; restore n_heads from push
     push rax                                  ; re-save for WRITE_* below
     INT3_DBG_REG 0                            ; RAX = n_heads
     mov eax, fk_hd                            ; Load head_dim
     INT3_DBG_REG 0                            ; RAX = head_dim
     pop rax                                   ; restore n_heads
%endif
; Load global pointers: norm_weights, bias_ptr, rope_cos, rope_sin, current pos
     mov rax, [fused_norm_weights]             ; Load RMSNorm weight vector pointer
     mov fk_nw, rax                            ; Save norm weights, may be NULL for has_qk_norm=0
     mov rax, [fused_bias_ptr]                 ; Load QKV bias pointer
     mov ecx, fk_col_off                       ; Load column_offset, zero-extend: col_off is always non-negative
     lea rax, [rax + rcx*FK_FLOAT_BYTES]       ; bias_ptr + col_off * sizeof,float, fused shift+add
     mov fk_bias_base, rax                     ; Save bias base, offset by col_off; never NULL by contract
%ifdef ZYDIS_TRACE
     cmp qword fk_nw, 0                        ; Check if norm weights pointer is NULL, not bias pointer!
     jnz .fk_nw_ok                             ; Not NULL -> continue
     INT3_DBG_STR "fused:f32:nw_null"          ; Debug: norm_weights NULL, has_qk_norm=0
.fk_nw_ok:
%endif
     mov rax, [fused_rope_cos]                 ; Load cos table base into scratch
     mov fk_rc, rax                            ; Save cos pointer, mem-to-mem mov impossible, use register hop
     mov rax, [fused_rope_sin]                 ; Load sin table base into scratch
     mov fk_rs, rax                            ; Save sin pointer, mem-to-mem mov impossible, use register hop
     mov eax, [fused_pos]                      ; Load current sequence position, import guarantees seq_len fits the rope table
     mov fk_pos, eax                           ; Save position
; Advance RoPE table pointers to current position: pos *, head_dim/2 floats
     mov eax, fk_hd                            ; head_dim
     add eax, eax                              ; head_dim*2 == half*4, the rope-table byte stride, head_dim is even by the init contract
     mov ecx, fk_pos                           ; Load current position
     imul rax, rcx                             ; rax = stride * pos = byte advance into the rope tables
     add fk_rc, rax                            ; Advance cos pointer to current pos
     add fk_rs, rax                            ; Advance sin pointer to current pos
; ===== Outer head loop =====
     xor r15d, r15d                            ; r15 = head counter, 0-based
     mov r14, fk_wq_base                       ; r14 = weight quantized base, invariant across all heads; never overwritten by matmul or postprocess
     mov r11d, fk_n                            ; r11 = n
     shr r11, 4                                ; r11 = scale row advance, n>>4 bytes; invariant across all heads, r11 has no other writer in this function and the function makes no call
     align 32
.fk_head_loop:
     cmp r15d, fk_nheads                       ; head < nheads?
     jge .fk_rp_done                           ; Exit head loop when done
     mov fk_head_idx, r15d                     ; Save head counter, overwritten by matmul
; Zydis debug: current head index
     INT3_DBG_STR "head="                      ; Debug trace: head iteration
     INT3_DBG_REG 15                           ; r15d = head_idx, still valid
; Row range for this head first, while r15d still holds the active head counter:
; reading it directly avoids re-loading fk_head_idx, stored just above .
     mov eax, r15d                             ; eax = head counter, active, r15 is not reloaded until below
     imul eax, fk_hd                           ; eax = start_row = head * head_dim, no EDX overwrite
     mov ebx, eax                              ; ebx = start_row, row loop counter
     add eax, fk_hd                            ; eax = end_row = start_row + head_dim
     mov r8d, eax                              ; r8d = end_row, loop bound
; Reload matmul base pointers from scratch, preserved across postprocessing overwrites
     mov r12, fk_xo                            ; r12 = xout base pointer
     mov r15, fk_ws_base                       ; r15 = weight scales base, ws; r14/wq hoisted outside the head loop
     mov r9d, fk_n                             ; r9d = n, restored after postprocess overwrites it
     mov rdi, fk_x_base                        ; rdi = activation base, invariant across this head's rows; postprocess re-sets per head
     mov rax, fk_bias_base                     ; rax = bias base for this head, invariant for the whole row loop; rax is unused until .fk_hs_reduce
     mov r10d, ebx                             ; start_row
     imul r10, r9                              ; start_row * n, byte offset, once per head
     mov r13, r10                              ; copy the byte offset for the scale derivation
     shr r13, 4                                ; start_row*n >>4 = scale byte offset, exact: n is 64-aligned
     add r13, r15                              ; r13 = scale ROW base for start_row
     add r10, r14                              ; r10 = weight row pointer for start_row
     align 32
.fk_head_simd:
; This row's scale group cursor, advances by 8 per 128-group in the body
     mov rsi, r13                              ; rsi = this row's scale group cursor
; Zero all four SIMD accumulators for dot product
     vxorps ymm0, ymm0, ymm0                   ; acc 0 = 0, 256-bit: VEX.128 preserves bits 255:128, which F32_MADD_32 and the reduce both read
     vxorps ymm1, ymm1, ymm1                   ; acc 1 = 0, 256-bit, same reason
     vxorps ymm2, ymm2, ymm2                   ; acc 2 = 0, 256-bit, same reason
     vxorps ymm3, ymm3, ymm3                   ; acc 3 = 0, 256-bit, same reason
     xor ecx, ecx                              ; ecx = element counter, 0-based
     mov edx, r9d                              ; remaining = n, loop countdown, independent of rcx address chain
     jmp .fk_hs_check128                       ; Jump to the bottom test, first-trip check
     align 64
.fk_hs_body128:
     vbroadcastss ymm13, [rsi + FK_SCALE_NEXT_OFF] ; Pre-load second scale early, hide latency
     vbroadcastss ymm8, [rsi]                  ; Load scale for first 64 elements
     add rsi, 8                                ; Advance scale pointer by 2 floats
     ds prefetcht0 [rdi + rcx*FK_FLOAT_BYTES + FK_ACT_PREFETCH_DIST] ; Prefetch next activation cache line
     PREFETCH_SLOT 12                          ; Slot 12: fused f32 activation prefetch
     F32_MADD_32 0, 0, ymm8                    ; Dot: x[i+0..31] *, w[i+0..31]*scale, weight dequantized, raw activations
     ds prefetcht1 [r10 + rcx + FK_WT_PREFETCH_DIST] ; Prefetch next weight cache line, streaming
     PREFETCH_SLOT 11                          ; Slot 11: fused f32 weight prefetch, +320
     F32_MADD_32 32, 128, ymm8                 ; Dot: x[i+32..63] *, w[i+32..63]*scale
     F32_MADD_32 64, 256, ymm13                ; Dot: x[i+64..95] *, w[i+64..95]*scale2
     F32_MADD_32 96, 384, ymm13                ; Dot: x[i+96..127] *, w[i+96..127]*scale2
     sub ecx, -128                             ; Advance element counter by 128, imm8 encoding of the same +128
.fk_hs_check128:
     sub edx, FK_UNROLL_128                    ; consume 128; CF=1 iff remaining was < 128
     jae .fk_hs_body128                        ; CF=0, had >= 128 -> next iteration
.fk_hs_tail_or_done:
     cmp ecx, r9d                              ; Check if all elements consumed
     jae .fk_hs_reduce                         ; Jump to reduction if done
     vbroadcastss ymm8, [rsi]                  ; Load single scale for remaining 64
     F32_MADD_32 0, 0, ymm8                    ; Dot: first 32 elements, weight dequantized, raw activations
     F32_MADD_32 32, 128, ymm8                 ; Dot: next 32 elements
.fk_hs_reduce:
     vaddps ymm0, ymm0, ymm1                   ; Sum accumulators 0+1
     vaddps ymm2, ymm2, ymm3                   ; Sum accumulators 2+3
     vaddps ymm0, ymm0, ymm2                   ; Total sum across all accumulators
     REDUCE_YMM_TO_SCALAR_FAST xmm0, ymm0, xmm4, xmm5 ; Horizontal sum of YMM to scalar
     vaddss xmm0, xmm0, [rax + rbx*FK_FLOAT_BYTES] ; Add bias[row], base hoisted into rax at head entry.A no-op for a finite
; accumulator because the block is a real all-zeros block for bias-less models
     vmovss [r12 + rbx*FK_FLOAT_BYTES], xmm0   ; Store dot product, with bias to output row
     inc ebx                                   ; Advance to next output row
     add r10, r9                               ; weight row pointer -> next row, n bytes
     add r13, r11                              ; scale row base -> next row, n>>4 bytes
     cmp ebx, r8d                              ; next row < end_row?
     jl .fk_head_simd                          ; yes -> process it, fall-through to postprocess after the last row
.fk_head_postprocess:
; Zydis debug: last dot product before postprocess
     INT3_DBG_F32 0                            ; XMM0 = last dot product of this head
     mov eax, fk_head_idx                      ; Load head index, 32-bit, sufficient
     imul eax, fk_hd                           ; Multiply: head_index * head_dim, 32-bit, zero-extends into rax
     lea r12, [r12 + rax*FK_HEAD_DATA_SCALE]   ; r12 = pointer to current head's data, element offset * sizeof,float in one LEA
; Branch: norm_weights == NULL means has_qk_norm=0, skip RMSNorm, RoPE only
     cmp qword fk_nw, 0                        ; Is norm_weights NULL?
     jne .fk_hpp_ss_init                       ; Non-NULL -> full RMSNorm+RoPE path
; has_qk_norm=0: check if RoPE is needed, pos != 0
     mov eax, fk_pos                           ; Load current position
     test eax, eax                             ; pos == 0?
     jz .fk_head_next                          ; pos=0 AND no RMSNorm -> nothing to do
     jmp .fk_hpp_rope_only                     ; pos!=0 -> RoPE rotation only, no RMSNorm
.fk_hpp_ss_init:
; Pass 1: sum of squares across head elements, for RMSNorm
     mov r13d, fk_hd                           ; r13 = head_dim, element count
     mov edx, r13d                             ; edx = remaining elements, countdown
     xor ecx, ecx                              ; ecx = element counter, reset
     vxorps ymm0, ymm0, ymm0                   ; acc A = 0, 256-bit: VEX.128 preserves bits 255:128, which the FMA body and HADD_YMM at the reduce both read
     vxorps ymm3, ymm3, ymm3                   ; acc B = 0, 256-bit, same reason
     vxorps ymm4, ymm4, ymm4                   ; acc C = 0, 256-bit, same reason
     vxorps ymm5, ymm5, ymm5                   ; acc D = 0, 256-bit, same reason
     vxorps ymm6, ymm6, ymm6                   ; acc E = 0, 256-bit, same reason
     vxorps ymm7, ymm7, ymm7                   ; acc F = 0, 256-bit, same reason
     vxorps ymm8, ymm8, ymm8                   ; acc G = 0, 256-bit, same reason
     vxorps ymm9, ymm9, ymm9                   ; acc H = 0, 256-bit, same reason
     jmp .fk_hpp_ss_check                      ; Jump to the bottom test, first-trip check
.fk_hpp_ss_body64:
; 64-element unrolled iteration with 8 independent accumulators.
     vmovups ymm1, [r12 + rcx*FK_FLOAT_BYTES]  ; Load elements [i..i+7]
     vmovups ymm2, [r12 + rcx*FK_FLOAT_BYTES + FK_1YMM] ; Load elements [i+8..i+15]
     vfmadd231ps ymm0, ymm1, ymm1              ; sum into acc A
     vfmadd231ps ymm3, ymm2, ymm2              ; sum into acc B
     vmovups ymm1, [r12 + rcx*FK_FLOAT_BYTES + FK_2YMM] ; Load elements [i+16..i+23]
     vmovups ymm2, [r12 + rcx*FK_FLOAT_BYTES + FK_3YMM] ; Load elements [i+24..i+31]
     vfmadd231ps ymm4, ymm1, ymm1              ; sum into acc C
     vfmadd231ps ymm5, ymm2, ymm2              ; sum into acc D
     vmovups ymm1, [r12 + rcx*FK_FLOAT_BYTES + FK_4YMM] ; Load elements [i+32..i+39]
     vmovups ymm2, [r12 + rcx*FK_FLOAT_BYTES + FK_5YMM] ; Load elements [i+40..i+47]
     vfmadd231ps ymm6, ymm1, ymm1              ; sum into acc E
     vfmadd231ps ymm7, ymm2, ymm2              ; sum into acc F
     vmovups ymm1, [r12 + rcx*FK_FLOAT_BYTES + FK_6YMM] ; Load elements [i+48..i+55]
     vmovups ymm2, [r12 + rcx*FK_FLOAT_BYTES + FK_7YMM] ; Load elements [i+56..i+63]
     vfmadd231ps ymm8, ymm1, ymm1              ; sum into acc G
     vfmadd231ps ymm9, ymm2, ymm2              ; sum into acc H
     add ecx, 64                               ; Advance by 64 elements
.fk_hpp_ss_check:
     sub edx, FK_UNROLL_64                     ; consume 64; CF=1 iff fewer remained, countdown
     jae .fk_hpp_ss_body64                     ; CF=0, had >= 64 -> next iteration
.fk_hpp_ss_tail_entry:
; Combine 8 accumulators into ymm0 for horizontal sum
     vaddps ymm0, ymm0, ymm3                   ; ymm0 += ymm3
     vaddps ymm4, ymm4, ymm5                   ; ymm4 += ymm5
     vaddps ymm6, ymm6, ymm7                   ; ymm6 += ymm7
     vaddps ymm8, ymm8, ymm9                   ; ymm8 += ymm9
     vaddps ymm0, ymm0, ymm4                   ; ymm0 += ymm4
     vaddps ymm6, ymm6, ymm8                   ; ymm6 += ymm8
     vaddps ymm0, ymm0, ymm6                   ; ymm0 += ymm6
     mov edx, r13d                             ; edx = elements left after the 64-loop, re-seeded countdown
     sub edx, ecx                              ; elements left after the 64-loop
     shr edx, 3                                ; edx = 8-element block count, exact: head_dim % 32 == 0 -> remainder % 8 == 0
     jz .fk_hpp_ss_done                        ; head_dim was a multiple of 64, no 8-blocks, reachable: head_dim % 64 is 0 or 32
     align 64
.fk_hpp_ss_8:
     vmovups ymm1, [r12 + rcx*FK_FLOAT_BYTES]  ; Load 8 head elements
     vfmadd231ps ymm0, ymm1, ymm1              ; Accumulate: sum += x[i]^2
     add ecx, 8                                ; Advance by 8 elements
     dec edx                                   ; one block done, independent chain, ecx stays free for addressing only
     jnz .fk_hpp_ss_8                          ; more blocks, continue, bottom-tested
.fk_hpp_ss_done:
     HADD_YMM xmm0, ymm0, xmm1                 ; Reduce 8 floats in ymm0 to scalar
     COMPUTE_RMS_SCALE xmm0, fk_dim_f, xmm0    ; Compute RMSNorm scale: sqrt,sum_sq / n + eps ^-1
     vbroadcastss ymm14, xmm0                  ; Broadcast scale to all YMM lanes
; Skip RoPE rotation if pos == 0, cos=1, sin=0 => rotation is no-op
     mov eax, fk_pos                           ; Load current position
     test eax, eax                             ; pos == 0?
     jz .fk_hpp_rmsnorm_only                   ; Yes -> skip rotation, just RMSNorm
; Pass 2: fused RMSNorm scaling + RoPE rotation
     mov rsi, fk_nw                            ; rsi = norm weights base pointer for this layer, shared across all heads
     mov rbx, fk_rc                            ; rbx = cos table pointer, pos-offset
     mov rdi, fk_rs                            ; rdi = sin table pointer, pos-offset
     mov r13d, fk_hd                           ; r13 = head_dim
     shr r13d, 1                               ; r13 = half, for pairwise rotation
     lea r8, [r12 + r13*FK_FLOAT_BYTES]        ; r8 = pointer to second half of head data, half * 4 folded into the address
     lea r9, [rsi + r13*FK_FLOAT_BYTES]        ; r9 = pointer to second half of norm weights
     xor ecx, ecx                              ; ecx = element counter, reset
     mov edx, r13d                             ; edx = pair count, block countdown; half = head_dim/2 is a multiple of 16, so % 8 == 0 exactly
     shr edx, 3                                ; edx = 8-pair block count, exact, so no zero-trip check is needed, the count is 4..64
     align 32
.fk_hpp_rp_loop:
     vmovups ymm0, [r12 + rcx*FK_FLOAT_BYTES]  ; Load first half elements, even pairs
     vmovups ymm1, [r8 + rcx*FK_FLOAT_BYTES]   ; Load second half elements, odd pairs
     vmovups ymm4, [rbx + rcx*FK_FLOAT_BYTES]  ; Load cos values, early: overlaps normalization chains
     vmovups ymm5, [rdi + rcx*FK_FLOAT_BYTES]  ; Load sin values, early: overlaps normalization chains
     vmulps ymm0, ymm0, ymm14                  ; Scale: x_even *= rms_scale, scale first
     vmulps ymm1, ymm1, ymm14                  ; Scale: x_odd *= rms_scale, scale first
     vmulps ymm0, ymm0, [rsi + rcx*FK_FLOAT_BYTES] ; even: scale then weight, unchanged
     vmulps ymm1, ymm1, [r9 + rcx*FK_FLOAT_BYTES] ; odd: independent scale then weight, pre-issued
     vmulps ymm6, ymm0, ymm4                   ; even * cos
     vmulps ymm7, ymm0, ymm5                   ; even * sin
     vfnmadd231ps ymm6, ymm1, ymm5             ; rotated_even = even*cos - odd*sin, fused
     vfmadd231ps ymm7, ymm1, ymm4              ; rotated_odd = even*sin + odd*cos
     vmovups [r12 + rcx*FK_FLOAT_BYTES], ymm6  ; Store rotated even elements
     vmovups [r8 + rcx*FK_FLOAT_BYTES], ymm7   ; Store rotated odd elements
     add ecx, 8                                ; Advance by 8 pairs
     dec edx                                   ; one block done
     jnz .fk_hpp_rp_loop                       ; more pairs, continue, bottom-tested
     jmp .fk_head_next                         ; all pairs rotated, next head
; ===== pos=0 fast path: RMSNorm only, no RoPE rotation =====
.fk_hpp_rmsnorm_only:
     mov rsi, fk_nw                            ; rsi = norm weights base pointer
     mov r13d, fk_hd                           ; r13 = head_dim
     xor ecx, ecx                              ; ecx = element counter, reset
     mov edx, r13d                             ; edx = element count, 16-float group countdown; head_dim % 8 == 0, so only a possible 8-float tail remains
     shr edx, 4                                ; edx = 16-float group count, groups processed two vectors at a time
     jz .fk_hpp_rn_tail8                       ; head_dim < 16: straight to the tail block
     align 32
.fk_hpp_rn_loop16:
     vmovups ymm0, [r12 + rcx*FK_FLOAT_BYTES]  ; Load elements [i..i+7]
     vmovups ymm1, [r12 + rcx*FK_FLOAT_BYTES + FK_1YMM] ; Load elements [i+8..i+15], independent chain
     vmulps ymm0, ymm0, ymm14                  ; x *= rms_scale, scale first, C ref order
     vmulps ymm1, ymm1, ymm14                  ; x *= rms_scale, second vector, independent chain
     vmulps ymm0, ymm0, [rsi + rcx*FK_FLOAT_BYTES] ; x *= norm_weight, weight operand folded
     vmulps ymm1, ymm1, [rsi + rcx*FK_FLOAT_BYTES + FK_1YMM] ; x *= norm_weight, second vector
     vmovups [r12 + rcx*FK_FLOAT_BYTES], ymm0  ; Store first vector
     vmovups [r12 + rcx*FK_FLOAT_BYTES + FK_1YMM], ymm1 ; Store second vector
     add ecx, 16                               ; Advance by 16 elements
     dec edx                                   ; one group done
     jnz .fk_hpp_rn_loop16                     ; more groups, continue, bottom-tested
.fk_hpp_rn_tail8:
     cmp ecx, r13d                             ; a remaining 8-float block?
     jae .fk_head_next                         ; no: all elements normalized, next head
     vmovups ymm0, [r12 + rcx*FK_FLOAT_BYTES]  ; Load final 8 head elements
     vmulps ymm0, ymm0, ymm14                  ; x *= rms_scale, scale first, C ref order
     vmulps ymm0, ymm0, [rsi + rcx*FK_FLOAT_BYTES] ; x *= norm_weight, weight operand folded
     vmovups [r12 + rcx*FK_FLOAT_BYTES], ymm0  ; Store result
     jmp .fk_head_next                         ; all elements normalized, next head
; ===== RoPE-only path, has_qk_norm=0, pos != 0 =====
; Skips RMSNorm, no norm weights, applies only RoPE rotation to matmul output.
.fk_hpp_rope_only:
; Setup RoPE rotation pointers
     mov rbx, fk_rc                            ; rbx = cos table pointer, pos-offset
     mov rdi, fk_rs                            ; rdi = sin table pointer, pos-offset
     mov r13d, fk_hd                           ; r13 = head_dim
     shr r13d, 1                               ; r13 = half, for pairwise rotation
     lea r8, [r12 + r13*FK_FLOAT_BYTES]        ; r8 = pointer to second half of head data, half * 4 folded into the address
     xor ecx, ecx                              ; ecx = element counter, reset
     mov edx, r13d                             ; edx = pair count, block countdown; half % 8 == 0, so exact
     shr edx, 3                                ; edx = 8-pair block count, exact, so no zero-trip check is needed
     align 32
.fk_hpp_rope_loop:
; 8-pair unrolled RoPE rotation, no norm_weight, no rms_scale
     vmovups ymm0, [r12 + rcx*FK_FLOAT_BYTES]  ; Load first half elements, even pairs
     vmovups ymm1, [r8 + rcx*FK_FLOAT_BYTES]   ; Load second half elements, odd pairs
     vmovups ymm4, [rbx + rcx*FK_FLOAT_BYTES]  ; Load cos values
     vmovups ymm5, [rdi + rcx*FK_FLOAT_BYTES]  ; Load sin values
     vmulps ymm6, ymm0, ymm4                   ; even * cos
     vmulps ymm7, ymm0, ymm5                   ; even * sin, independent chain
     vfnmadd231ps ymm6, ymm1, ymm5             ; rotated_even = even*cos - odd*sin
     vfmadd231ps ymm7, ymm1, ymm4              ; rotated_odd = even*sin + odd*cos
     vmovups [r12 + rcx*FK_FLOAT_BYTES], ymm6  ; Store rotated even elements
     vmovups [r8 + rcx*FK_FLOAT_BYTES], ymm7   ; Store rotated odd elements
     add ecx, 8                                ; Advance by 8 pairs
     dec edx                                   ; one block done
     jnz .fk_hpp_rope_loop                     ; more pairs, continue, bottom-tested
.fk_head_next:
; Zydis debug: first element of this head's postprocess output, debug builds only
%ifdef ZYDIS_TRACE
     vmovss xmm0, [r12]                        ; Load first element of head output
     INT3_DBG_F32 0                            ; Print: 0.0 means postprocess zeroed it
%endif
; Restore head loop state and advance to next head
     mov r15d, fk_head_idx                     ; Restore head counter from saved slot
     inc r15d                                  ; Advance to next head
     jmp .fk_head_loop                         ; Continue head loop
.fk_rp_done:
     INT3_DBG_STR "fk:rp_done"                 ; Debug: fused kernel postprocess complete
     CLEANUP_YMM                               ; Clear YMM upper halves for ABI
     vmovups xmm13, fk_xmm13                   ; Restore XMM13, non-volatile
     vmovups xmm12, fk_xmm12                   ; Restore XMM12, non-volatile
     vmovups xmm11, fk_xmm11                   ; Restore XMM11, non-volatile
     vmovups xmm10, fk_xmm10                   ; Restore XMM10, non-volatile
     vmovups xmm9, fk_xmm9                     ; Restore XMM9, non-volatile
     vmovups xmm8, fk_xmm8                     ; Restore XMM8, non-volatile
     vmovups xmm7, fk_xmm7                     ; Restore XMM7, non-volatile
     vmovups xmm6, fk_xmm6                     ; Restore XMM6, non-volatile
     vmovdqu xmm14, fk_symm14                  ; Restore XMM14 lower half, non-volatile, upper is scratch
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; Epilogue: restore GPRs, return
; ============================================================================
; ============================================================================
%undef fk_head_idx
%undef fk_head_idx_raw
%undef fk_bias_base
%undef fk_bias_base_raw
%undef fk_ws_base
%undef fk_ws_base_raw
%undef fk_wq_base
%undef fk_wq_base_raw
%undef fk_x_base
%undef fk_x_base_raw
%undef fk_n
%undef fk_n_raw
%undef fk_col_off
%undef fk_col_off_raw
%undef fk_d
%undef fk_d_raw
%undef fk_xo
%undef fk_xo_raw
%undef fk_rs
%undef fk_rs_raw
%undef fk_rc
%undef fk_rc_raw
%undef fk_nw
%undef fk_nw_raw
%undef fk_pos
%undef fk_pos_raw
%undef fk_nheads
%undef fk_nheads_raw
%undef fk_dim_f
%undef fk_dim_f_raw
%undef fk_hd
%undef fk_hd_raw
%undef fk_symm14
%undef fk_xmm13
%undef fk_xmm12
%undef fk_xmm11
%undef fk_xmm10
%undef fk_xmm9
%undef fk_xmm8
%undef fk_xmm7
%undef fk_xmm6
%undef FK_WT_PREFETCH_DIST
%undef FK_ACT_PREFETCH_DIST
%undef FK_7YMM
%undef FK_6YMM
%undef FK_5YMM
%undef FK_4YMM
%undef FK_3YMM
%undef FK_2YMM
%undef FK_1YMM
%undef FK_FLOAT_BYTES
%undef FK_SCALE_NEXT_OFF
%undef FK_UNROLL_64
%undef FK_UNROLL_128
%undef FK_ARG6_COFF_OFF
%undef FK_ARG5_D_OFF
%undef RBIT_SAVED_REGS_SIZE
%undef RBIT_FRAME_SIZE
%undef RBIT_DOUBLE_STRIDE
%undef RBIT_FREQ_OFF
%undef RBIT_TRIG_CW
%undef RBIT_TRIGCW_OFF
%undef RBIT_CW_OFF
%undef RBIT_TMPSF_OFF
%undef RBIT_TMPIFL_OFF
%undef RBIT_TMPIF_OFF
%undef RBIT_TMPW_OFF
%undef RBIT_DHIGHWL_OFF
%undef RBIT_DLOWWL_OFF
%undef RBIT_DOLD_OFF
%undef RBIT_DLOW_OFF
%undef RBIT_DHIGH_OFF
%undef RBIT_DFACTOR_OFF
%undef RBIT_BASEVAL_OFF
%undef RBIT_DIM_OFF
%undef RBIT_POS_OFF
%undef RBIT_SIZE_OFF
%undef RBIT_HALF_OFF
