; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%ifndef V7_APICALL_INC
%define V7_APICALL_INC                         ; include guard, file scanned once per translation unit
; ============================================================================
; Internal state, reset by SF_START
; ============================================================================
%assign __sf_nregs 0                           ; number of saved non-volatile GPRs, excl rbp
%assign __sf_cursor 0                          ; bytes allocated so far, relative to post-push RBP
%assign __sf_committed 0                       ; 1 after SF_COMMIT, prevents further slot declarations
%assign __sf_framesize 0                       ; total sub rsp amount, computed by SF_COMMIT
%assign __sf_pushbase 0                        ; base = 8 * nregs, space occupied by pushed regs below rbp
; ============================================================================
; SF_START fn_name, reg1, reg2, ...
; Emits function prologue: push rbp, mov rbp rsp, push non-volatile regs.
; Registers saved here are verified by SF_END.
; ============================================================================

%macro SF_START 1-*
%assign __sf_nregs (%0 - 1)                    ; number of registers this frame saves, args minus the function name
%assign __sf_cursor 0                          ; reset the local-slot cursor for this frame
%assign __sf_committed 0                       ; reset the commit latch for this frame
%assign __sf_framesize 0                       ; reset the frame size for this frame
%assign __sf_pushbase (8 * __sf_nregs)         ; slot-address base: bytes the register pushes occupy below rbp
     global %1                                 ; make the function link-visible, sole source of the global
%1:                                            ; function entry label, emitted here so local labels anchor to it
     push rbp                                  ; save caller's frame pointer
     mov rbp, rsp                              ; set our frame pointer
%if %0 > 1
%rotate 1                                      ; skip the function name, rotate the first register into %1
%rep (%0 - 1)
     push %1                                   ; save one non-volatile register, in argument order
%rotate 1                                      ; advance to the next register argument
%endrep
%endif
%endmacro

%macro __sf_align 1
%assign __sf_cursor (__sf_cursor + (%1 - 1)) & ~(%1 - 1) ; round the cursor up to the requested alignment
%endmacro

%macro SF_PTR 1
     SF_QWORD %1                               ; pointer slots are qword slots, identical layout, single source of truth
%endmacro

%macro SF_QWORD 1
%if __sf_committed
%error SF_QWORD(%1): Cannot declare locals after SF_COMMIT ; slot layout is frozen once the frame size is fixed
%endif
     __sf_align 8                              ; align to 8-byte boundary for qword slot
%assign __sf_cursor __sf_cursor + 8            ; reserve the slot's bytes
%xdefine %1 qword [rbp - (__sf_pushbase + __sf_cursor)] ; typed slot address, captured NOW, not at use time
%xdefine %1_raw [rbp - (__sf_pushbase + __sf_cursor)] ; untyped alias of the same address
%endmacro

%macro SF_DWORD 1
%if __sf_committed
%error SF_DWORD(%1): Cannot declare locals after SF_COMMIT ; slot layout is frozen once the frame size is fixed
%endif
     __sf_align 4                              ; align to 4-byte boundary for dword slot
%assign __sf_cursor __sf_cursor + 4            ; reserve the slot's bytes
%xdefine %1 dword [rbp - (__sf_pushbase + __sf_cursor)] ; typed slot address, captured NOW, not at use time
%xdefine %1_raw [rbp - (__sf_pushbase + __sf_cursor)] ; untyped alias of the same address
%endmacro

%macro SF_BYTE 1
%if __sf_committed
%error SF_BYTE(%1): Cannot declare locals after SF_COMMIT ; slot layout is frozen once the frame size is fixed
%endif
     __sf_align 1                              ; byte alignment is always satisfied, no-op
%assign __sf_cursor __sf_cursor + 1            ; reserve the slot's byte
%xdefine %1 byte [rbp - (__sf_pushbase + __sf_cursor)] ; typed slot address, captured NOW, not at use time
%endmacro

%macro SF_XMM 1
%if __sf_committed
%error SF_XMM(%1): Cannot declare locals after SF_COMMIT ; slot layout is frozen once the frame size is fixed
%endif
     __sf_align 16                             ; cursor-aligned spacing for the XMM spill slot, address alignment is caller-dependent, use unaligned moves only
%assign __sf_cursor __sf_cursor + 16           ; reserve the slot's 16 bytes
%xdefine %1 [rbp - (__sf_pushbase + __sf_cursor)] ; slot address, captured NOW, not at use time
%endmacro

%macro SF_ARRAY 2
%if __sf_committed
%error SF_ARRAY(%1): Cannot declare locals after SF_COMMIT ; slot layout is frozen once the frame size is fixed
%endif
     __sf_align 8                              ; align array base to 8-byte boundary
%assign __sf_arr_end __sf_pushbase + __sf_cursor + %2 ; name = array BASE, lowest byte ; the array occupies [name, name+size and is indexed upward from the name, exclusive cursor reservation
%xdefine %1 [rbp - __sf_arr_end]               ; end-anchored array address, captured NOW
%assign __sf_cursor __sf_cursor + %2           ; reserve the array's bytes
%endmacro

%macro SF_COMMIT 0
%if __sf_committed
%error SF_COMMIT: Already committed (duplicate SF_COMMIT call) ; the frame size would be computed twice
%endif
%assign __sf_raw (__sf_cursor + 32)            ; locals + shadow space minimum
%assign __sf_align_need (8 * __sf_nregs) % 16  ; required alignment mod
%assign __sf_framesize ((__sf_raw + 15) & ~15) ; round up to 16
; If alignment mismatch, add 8 bytes to fix
%if (__sf_framesize % 16) != __sf_align_need
%assign __sf_framesize __sf_framesize + 8      ; pad to the parity the register pushes created
%endif
; Minimum shadow space is 32 bytes, Win64 ABI requirement
%if __sf_framesize < 32
%assign __sf_framesize 32                      ; clamp to the ABI minimum even for frameless-locals functions
%endif
     sub rsp, __sf_framesize                   ; allocate local stack + shadow space
%assign __sf_committed 1                       ; latch: no further slot declarations are legal
%endmacro
; ============================================================================
; SF_END reg1, reg2, ...
; Emits epilogue: recover RSP, lea rsp,[rbp-pushbase] for large frames,
; add rsp,framesize for small ; pop regs in reverse; pop rbp; ret.
; IMPORTANT: registers must be listed in the SAME order as SF_START.
; Emits %error if count differs from SF_START's register count.
; ============================================================================

%macro SF_END 0-*
%if %0 != __sf_nregs
%error SF_END: register count %0 != SF_START count __sf_nregs ; a mismatched pop order would restore garbage into the wrong registers
%endif
%if __sf_framesize > 127
     lea rsp, [rbp - __sf_pushbase]            ; recover pre-commit RSP from rbp, disp8: 4 bytes vs 7-byte add rsp,imm32
%elif __sf_framesize > 0
     add rsp, __sf_framesize                   ; deallocate local stack + shadow space, imm8: 4-byte encoding
%endif
%if %0 > 0
%rep %0
%rotate -1                                     ; walk the register list backwards, reverse push order
     pop %1                                    ; restore non-volatile register
%endrep
%endif
     pop rbp                                   ; restore caller's frame pointer
     ret                                       ; return to caller
%endmacro

%macro CLEANUP_YMM 0
     vzeroupper                                ; Clear YMM upper halves for ABI compliance
%endmacro

%macro CORRECT_RSP 0
     test spl, 8                               ; RSP misaligned, bit 3 set ?
     jnz %%crsp_bad                            ; yes: single-marker path
     push qword 0                              ; aligned: pad 0 ..., NASM emits 6A 00 imm8 sign-extended, intentional minimal form
     push qword 0                              ; ... and 0, 16 bytes, stays 0 mod 16
     jmp %%crsp_done                           ; both paths converge aligned
%%crsp_bad:
     push qword 1                              ; misaligned: one push realigns, 8+8=16
%%crsp_done:
%endmacro

%macro RESTORE_RSP 0
     cmp qword [rsp], 0                        ; flag 0 = two pushes, 16 bytes ?
     jne %%rrs_one                             ; nonzero, =1 = single push, 8 bytes
     add rsp, 16                               ; skip both zero qwords
     jmp %%rrs_done                            ; done
%%rrs_one:
     add rsp, 8                                ; skip the single flag=1 qword
%%rrs_done:
%endmacro
%endif                                         ; V7_APICALL_INC
