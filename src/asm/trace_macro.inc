; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%ifndef TRACE_MACRO_INC
%define TRACE_MACRO_INC                        ; include guard, file scanned once per translation unit
; ============================================================================
; TRACE_ENABLE flag: define with `-DTRACE_ENABLE` in NASM command line to
; activate all TRACE_* macros. When not defined, all TRACE_* macros expand
; to nothing, zero overhead, no code emitted .
; ============================================================================
%ifdef TRACE_ENABLE
; --- YMM save-area slot offsets, bytes from rsp after `sub rsp, 512` ---
%define TRACE_YMM0_OFF 0                       ; slot of ymm0, volatile outright
%define TRACE_YMM1_OFF 32                      ; slot of ymm1
%define TRACE_YMM2_OFF 64                      ; slot of ymm2
%define TRACE_YMM3_OFF 96                      ; slot of ymm3
%define TRACE_YMM4_OFF 128                     ; slot of ymm4
%define TRACE_YMM5_OFF 160                     ; slot of ymm5
%define TRACE_YMM6_OFF 192                     ; slot of ymm6, upper half volatile per ABI
%define TRACE_YMM7_OFF 224                     ; slot of ymm7
%define TRACE_YMM8_OFF 256                     ; slot of ymm8
%define TRACE_YMM9_OFF 288                     ; slot of ymm9
%define TRACE_YMM10_OFF 320                    ; slot of ymm10
%define TRACE_YMM11_OFF 352                    ; slot of ymm11
%define TRACE_YMM12_OFF 384                    ; slot of ymm12
%define TRACE_YMM13_OFF 416                    ; slot of ymm13
%define TRACE_YMM14_OFF 448                    ; slot of ymm14
%define TRACE_YMM15_OFF 480                    ; slot of ymm15
%define TRACE_YMM_AREA 512                     ; total YMM save-area size, 16 x 32 bytes

%macro TRACE_SAVE_ALL 0
     pushfq                                    ; save RFLAGS, direction flag, carry, etc.
     push rax                                  ; save volatile rax
     push rcx                                  ; save volatile rcx
     push rdx                                  ; save volatile rdx
     push r8                                   ; save volatile r8
     push r9                                   ; save volatile r9
     push r10                                  ; save volatile r10
     push r11                                  ; save volatile r11
     push rbx                                  ; save non-volatile rbx
     push rsi                                  ; save non-volatile rsi
     push rdi                                  ; save non-volatile rdi
     push r12                                  ; save non-volatile r12
     push r13                                  ; save non-volatile r13
     push r14                                  ; save non-volatile r14
     push r15                                  ; save non-volatile r15
     push rbp                                  ; save non-volatile rbp
     sub rsp, TRACE_YMM_AREA                   ; 16 YMM registers x 32 bytes, 0 mod 16, push parity unchanged
     vmovdqu [rsp + TRACE_YMM0_OFF], ymm0      ; save ymm0, volatile: overwritten by every callee
     vmovdqu [rsp + TRACE_YMM1_OFF], ymm1      ; save ymm1
     vmovdqu [rsp + TRACE_YMM2_OFF], ymm2      ; save ymm2
     vmovdqu [rsp + TRACE_YMM3_OFF], ymm3      ; save ymm3
     vmovdqu [rsp + TRACE_YMM4_OFF], ymm4      ; save ymm4
     vmovdqu [rsp + TRACE_YMM5_OFF], ymm5      ; save ymm5
     vmovdqu [rsp + TRACE_YMM6_OFF], ymm6      ; save ymm6, the UPPER half is volatile per the ABI, the low half alone does not survive the calls
     vmovdqu [rsp + TRACE_YMM7_OFF], ymm7      ; save ymm7
     vmovdqu [rsp + TRACE_YMM8_OFF], ymm8      ; save ymm8
     vmovdqu [rsp + TRACE_YMM9_OFF], ymm9      ; save ymm9
     vmovdqu [rsp + TRACE_YMM10_OFF], ymm10    ; save ymm10
     vmovdqu [rsp + TRACE_YMM11_OFF], ymm11    ; save ymm11
     vmovdqu [rsp + TRACE_YMM12_OFF], ymm12    ; save ymm12
     vmovdqu [rsp + TRACE_YMM13_OFF], ymm13    ; save ymm13
     vmovdqu [rsp + TRACE_YMM14_OFF], ymm14    ; save ymm14
     vmovdqu [rsp + TRACE_YMM15_OFF], ymm15    ; save ymm15
%endmacro

%macro TRACE_RESTORE_ALL 0
     vmovdqu ymm0, [rsp + TRACE_YMM0_OFF]      ; restore ymm0
     vmovdqu ymm1, [rsp + TRACE_YMM1_OFF]      ; restore ymm1
     vmovdqu ymm2, [rsp + TRACE_YMM2_OFF]      ; restore ymm2
     vmovdqu ymm3, [rsp + TRACE_YMM3_OFF]      ; restore ymm3
     vmovdqu ymm4, [rsp + TRACE_YMM4_OFF]      ; restore ymm4
     vmovdqu ymm5, [rsp + TRACE_YMM5_OFF]      ; restore ymm5
     vmovdqu ymm6, [rsp + TRACE_YMM6_OFF]      ; restore ymm6
     vmovdqu ymm7, [rsp + TRACE_YMM7_OFF]      ; restore ymm7
     vmovdqu ymm8, [rsp + TRACE_YMM8_OFF]      ; restore ymm8
     vmovdqu ymm9, [rsp + TRACE_YMM9_OFF]      ; restore ymm9
     vmovdqu ymm10, [rsp + TRACE_YMM10_OFF]    ; restore ymm10
     vmovdqu ymm11, [rsp + TRACE_YMM11_OFF]    ; restore ymm11
     vmovdqu ymm12, [rsp + TRACE_YMM12_OFF]    ; restore ymm12
     vmovdqu ymm13, [rsp + TRACE_YMM13_OFF]    ; restore ymm13
     vmovdqu ymm14, [rsp + TRACE_YMM14_OFF]    ; restore ymm14
     vmovdqu ymm15, [rsp + TRACE_YMM15_OFF]    ; restore ymm15
     add rsp, TRACE_YMM_AREA                   ; drop the YMM save area, rsp now at the pushed registers
     pop rbp                                   ; restore non-volatile rbp
     pop r15                                   ; restore non-volatile r15
     pop r14                                   ; restore non-volatile r14
     pop r13                                   ; restore non-volatile r13
     pop r12                                   ; restore non-volatile r12
     pop rdi                                   ; restore non-volatile rdi
     pop rsi                                   ; restore non-volatile rsi
     pop rbx                                   ; restore non-volatile rbx
     pop r11                                   ; restore volatile r11
     pop r10                                   ; restore volatile r10
     pop r9                                    ; restore volatile r9
     pop r8                                    ; restore volatile r8
     pop rdx                                   ; restore volatile rdx
     pop rcx                                   ; restore volatile rcx
     pop rax                                   ; restore volatile rax
     popfq                                     ; restore RFLAGS, direction flag, carry, etc.
%endmacro
; ============================================================================
; TRACE_STR "literal string"

; Prints a literal string followed by a newline to stdout.
; The string is embedded inline via jmp-over-data, same as FATAL .
; Fully transparent: all registers, flags, and stack are preserved.

; Stack alignment is handled dynamically: after saving all GPRs, the
; post-save RSP is saved in rbp, then aligned to 16 bytes. This ensures
; correct Win64 ABI regardless of the entry RSP alignment, works in both
; 0mod16 and 8mod16 contexts such as after SF_START+SF_COMMIT .
; ============================================================================

%macro TRACE_STR 1
     TRACE_SAVE_ALL                            ; save all GPRs + RFLAGS + YMM, 128 + 512 = 640 bytes
     mov rbp, rsp                              ; rbp = post-saves RSP, safe: original rbp on stack
     and rsp, -16                              ; align to 16 bytes, drops 0-8 bytes
     sub rsp, 32                               ; 32 bytes shadow space, RSP is 16-aligned
     jmp %%trace_str_over                      ; skip over the inline string data
%%trace_str_data:
     db %1, 10                                 ; message + newline, no null: length computed at assembly time
%%trace_str_end:
%%trace_str_over:
     call debug_ensure_std                     ; ensure stdout_handle cached, overwrites rax,rcx,rdx,r8-r11
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     lea rdx, [%%trace_str_data]               ; arg2 = buffer pointer
     mov r8d, (%%trace_str_end - %%trace_str_data) ; arg3 = compile-time byte count, no runtime scan
     xor r9d, r9d                              ; arg4 = NULL, bytes-written count is discarded, WriteFile accepts a NULL out-pointer
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write the message to stdout, arg5 NULL = synchronous, no overlap
     mov rsp, rbp                              ; restore post-saves RSP, undoes align + sub
     TRACE_RESTORE_ALL                         ; restore all GPRs + RFLAGS
%endmacro

%macro TRACE_HEX 1
     TRACE_SAVE_ALL                            ; save all GPRs + RFLAGS + YMM, 128 + 512 = 640 bytes
     mov rbx, %1                               ; rbx = value to print, captured BEFORE rbp is borrowed as the alignment anchor, an rbp argument must read the caller's value; memory operands are evaluated AFTER RSP moved 640 bytes down, rsp-relative operands read the wrong location, rip-relative and rbp-relative are safe
     mov rbp, rsp                              ; rbp = post-saves RSP, safe: original rbp on stack
     and rsp, -16                              ; align to 16 bytes
     sub rsp, 32                               ; 32 bytes shadow space
     call debug_ensure_std                     ; ensure stdout_handle cached
     mov rcx, rbx                              ; rcx = value, trace_write_hex prints "0x" + 16 hex digits + LF in ONE write
     call trace_write_hex                      ; print "0x" + 16 hex digits + LF to stdout
     mov rsp, rbp                              ; restore post-saves RSP
     TRACE_RESTORE_ALL                         ; restore all GPRs + RFLAGS
%endmacro
; ============================================================================
; TRACE_REG "regname", reg

; Prints "regname=0xVALUE\n" to stdout.
; %1 = display name string, quoted, %2 = the 64-bit register.
; Fully transparent: all registers, flags, and stack are preserved.
; Stack aligned dynamically, same as TRACE_STR .
; ============================================================================

%macro TRACE_REG 2
     TRACE_SAVE_ALL                            ; save all GPRs + RFLAGS + YMM, 128 + 512 = 640 bytes
     mov rbx, %2                               ; rbx = value to print, captured BEFORE rbp is borrowed as the alignment anchor, an rbp argument must read the caller's value; memory operands are evaluated AFTER RSP moved 640 bytes down, rsp-relative operands read the wrong location, rip-relative and rbp-relative are safe
     mov rbp, rsp                              ; rbp = post-saves RSP, safe: original rbp on stack
     and rsp, -16                              ; align to 16 bytes
     sub rsp, 32                               ; 32 bytes shadow space
; Print "regname=0x" prefix, inline string, length computed at assembly time
     jmp %%trace_reg_over                      ; skip over inline data
%%trace_reg_data:
     db %1, '='                                ; "regname=" prefix, trace_write_hex adds "0x" + hex + LF
%%trace_reg_end:
%%trace_reg_over:
     call debug_ensure_std                     ; ensure stdout_handle cached
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     lea rdx, [%%trace_reg_data]               ; arg2 = prefix pointer
     mov r8d, (%%trace_reg_end - %%trace_reg_data) ; arg3 = compile-time length, no runtime scan
     xor r9d, r9d                              ; arg4 = NULL, bytes-written count is discarded
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write the "regname=" prefix to stdout, arg5 NULL = synchronous, no overlap
; Print the hex value, trace_write_hex appends the LF in the same write
     mov rcx, rbx                              ; rcx = value
     call trace_write_hex                      ; print "0x" + 16 hex digits + LF to stdout
     mov rsp, rbp                              ; restore post-saves RSP
     TRACE_RESTORE_ALL                         ; restore all GPRs + RFLAGS
%endmacro
%else                                          ; !TRACE_ENABLE

%macro TRACE_STR 1
%endmacro

%macro TRACE_HEX 1
%endmacro

%macro TRACE_REG 2
%endmacro
%endif                                         ; TRACE_ENABLE
%endif                                         ; TRACE_MACRO_INC
