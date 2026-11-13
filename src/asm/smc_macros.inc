; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%ifndef SMC_MACROS_INC
%define SMC_MACROS_INC                         ; include guard, file scanned once per translation unit
%define SMC_NOP_TAIL -1                        ; len - 1 for NOP template index
; ============================================================================
; Internal per-file state
; ============================================================================
%assign __pf_num_slots 0                       ; Prefetch slot counter, reset by PREFETCH_PATCH_BEGIN

%macro PREFETCH_SLOT 1
.pf%1:                                         ; Local label for patch table offset computation
%endmacro
%assign PREFETCH_MAX_INSN_LEN 9                ; Max bytes per patchable instruction
; patch_points entry layout, 5 dwords per slot, see PREFETCH_ROW plus the
; named scales and byte offsets the backup/restore walkers below consume.
%define PF_ENTRY_SCALE 4                       ; SIB scale in the slot*5 LEA, 1 + 4 = 5 dwords per entry
%define SMC_TYPE_REG_SCALE 8                   ; SIB scale: type value shifted into the ModRM REG field
%define PF_ENTRY_LEN_OFF 4                     ; entry dword 1: instruction length in bytes
%define PF_ENTRY_TYPE_OFF 8                    ; entry dword 2: type byte offset, offset to ModRM
%define PF_ENTRY_DISP_OFF 12                   ; entry dword 3: disp32 offset within the instruction
%define PF_ENTRY_KID_OFF 16                    ; entry dword 4: kernel_id, index into kernel_bases
%define PF_KERNEL_BASE_SCALE 8                 ; kernel_bases is a qword array, SIB scale 8
%define PF_BACKUP_SCALE 8                      ; SIB scale in the slot*9 backup-stride LEA, 1 + 8 = 9
%define PF_NINTH_BYTE 8                        ; byte index 8 = the 9th instruction byte, ds-prefix form
%define PF_OPCODE_BACK 2                       ; opcode starts this many bytes before the type byte, skips pad bytes like db 0x90
%define PF_PREFIX_BYTE 1                       ; second instruction byte, 66-prefix form check
%define SMC_DISP32_FIELD 4                     ; disp32 field byte count; each *_disp_end label sits directly after its 4-byte displacement
%define SMC_SCALE_X3 2                         ; SIB multiplier for 3*S: base + base*2
%define PF_DISP32_OFF 5                        ; disp32 offset within the 9-byte prefetch form, the direct-store byte index
%define PF_DISP_END_BACK 4                     ; bytes from the instruction's end label back to its disp32 start, 9 - PF_DISP32_OFF
; ============================================================================
; GENERATE_DLLMAIN [attach_handler [, detach_handler]]
; Generates a Windows DllMain entry point.
; Saves prefetch instruction bytes on DLL load, so init_prefetch_backup
; doesn't need a separate call.

; %1 = function to call on DLL_PROCESS_ATTACH, reason=1 .
; Default: init_prefetch_backup.
; Pass "" to skip ATTACH action entirely.
; %2 = optional handler label for DLL_PROCESS_DETACH, reason=0 .
; If omitted, DETACH returns TRUE with no action.
; The handler is reached via je, jump, not call, so no shadow
; space is provided. Handler must allocate its own if calling
; WinAPI functions. Returns directly to loader with ret.

; The generated code is a raw entry point, no SF_START frame,
; suitable for linking with /entry:DllMain.

; Usage:
; GENERATE_DLLMAIN init_prefetch_backup ; standalone kernel
; GENERATE_DLLMAIN init_prefetch_backup, v7_detach ; with DETACH handler
; GENERATE_DLLMAIN ; same as above
; GENERATE_DLLMAIN "" ; no ATTACH action, pure no-op DllMain
; ============================================================================

%macro GENERATE_DLLMAIN 0-2
%xdefine %%attach_func init_prefetch_backup    ; Default attach handler
%if %0 >= 1                                    ; User provided attach handler?
%ifnempty %1                                   ; Non-empty?
%xdefine %%attach_func %1                      ; Use user-specified handler
%else
%xdefine %%attach_func ""                      ; Empty: no ATTACH action
%endif
%endif
     global DllMain                            ; Export DllMain symbol for linker
DllMain:
     cmp edx, 1                                ; DLL_PROCESS_ATTACH?
     je %%attach                               ; Yes -> call attach handler
%if %0 >= 2                                    ; Detach handler provided?
     test edx, edx                             ; DLL_PROCESS_DETACH?
     je %2                                     ; Yes -> jump to detach handler
%endif
     mov eax, 1                                ; Return TRUE for other reasons
     ret                                       ; THREAD_ATTACH, THREAD_DETACH
%%attach:
%ifidni %%attach_func, ""                      ; No ATTACH action?
     mov eax, 1                                ; Just return TRUE
     ret                                       ; Return to loader
%else
     sub rsp, 40                               ; 32 shadow + 8 alignment, DllMain enters with RSP=8 mod 16, sub 40 lands the call at 0 mod 16
     call %%attach_func                        ; Call the attach handler
     add rsp, 40                               ; Restore shadow space + alignment
     mov eax, 1                                ; Return TRUE
     ret                                       ; Return to loader
%endif
%endmacro
; ============================================================================
; PREFETCH_PATCH_BEGIN
; Begins the patch table emission. Call ONCE after all kernel functions.
; Emits NOP templates table and patch_points header.
; Resets the internal slot counter to zero.
; ============================================================================

%macro PREFETCH_PATCH_BEGIN 0
%assign __pf_num_slots 0                       ; Reset slot counter
     section .rdata                            ; Switch to read-only data, .rdata is NASM's win64 predefined read-only section; .rodata gets format-default attributes
     align 16                                  ; Align NOP template table
; Intel-blessed multi-byte NOP templates, padded to 16 bytes each.
nop_templates:
     db 0x90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ; 1-byte NOP, padded to 16
     db 0x66, 0x90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ; 2-byte NOP, padded to 16
     db 0x0F, 0x1F, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ; 3-byte NOP, padded to 16
     db 0x0F, 0x1F, 0x40, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ; 4-byte NOP, padded to 16
     db 0x0F, 0x1F, 0x44, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ; 5-byte NOP, padded to 16
     db 0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ; 6-byte NOP, padded to 16
     db 0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ; 7-byte NOP, padded to 16
     db 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0 ; 8-byte NOP, padded to 16
     db 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0 ; 9-byte NOP, padded to 16
     align 16                                  ; Align patch table to 16-byte boundary
patch_points:
%endmacro

%macro PREFETCH_ROW 5-6
%ixdefine %%kid 0                              ; Default kernel_id = 0
%if %0 >= 6
%ixdefine %%kid %6                             ; Use provided kernel_id
%endif
     dd %1.pf%2 - %1 - %3, %3, %4, %5, %%kid   ; Emit 20-byte table entry
%assign __pf_num_slots __pf_num_slots + 1      ; Increment slot count
%endmacro
; ============================================================================
; PREFETCH_PATCH_END func [, kernel_bases]
; Ends the patch table. Emits backup buffer and all SMC export functions
; init_prefetch_backup, set_prefetch_distance,
; set_prefetch_type, DllMain for auto-backup .

; Parameters:
; %1 = primary kernel label, e.g. matmul_q8, single_kernel
; %2 = optional kernel_bases table label for dual-kernel DLLs.
; When provided, SMC functions look up kernel base from this table
; using kernel_id from each patch entry.
; When omitted, a single-entry kernel_bases table is auto-emitted.
; ============================================================================

%macro PREFETCH_PATCH_END 1-2
; ---- Emit kernel_bases table, if not user-provided ----
%if %0 < 2
     section .rdata
kernel_bases:
     dq %1                                     ; kernel_id 0 = primary kernel
%endif
; ---- Backup buffer ----
     section .bss                              ; Switch to BSS for backup buffer
     align 64                                  ; Align to cache line
prefetch_backup:
     resb __pf_num_slots * PREFETCH_MAX_INSN_LEN ; N bytes backup per slot
prefetch_backup_captured:
     resd 1                                    ; nonzero once the pristine bytes are archived this DLL lifetime
     section .text                             ; Switch back to .text
     global init_prefetch_backup               ; Export symbol
init_prefetch_backup:
     cmp dword [rel prefetch_backup_captured], 0 ; pristine bytes already archived?
     jne .pb_ret                               ; yes: a later capture would archive patched or NOPped
; instructions and restore them as "originals" forever
     push rbx                                  ; Save non-volatile rbx
     push rsi                                  ; Save non-volatile rsi
     push r12                                  ; Save non-volatile r12
     push r13                                  ; Save non-volatile r13
     push r14                                  ; Save non-volatile r14
     lea r12, [rel patch_points]               ; r12 = patch_points table base
     lea r13, [rel prefetch_backup]            ; r13 = backup buffer base
     lea r14, [rel kernel_bases]               ; r14 = kernel_bases table, kernel_id -> base address
     xor ecx, ecx                              ; ecx = slot index = 0
.init_loop:
     cmp ecx, __pf_num_slots                   ; Compare slot index with total count
     jge .init_done                            ; Exit loop if all slots processed
     lea ebx, [ecx + ecx*PF_ENTRY_SCALE]       ; slot_index * 5, 1-cycle LEA
     shl ebx, 2                                ; ebx = slot_index * 20, entry size = 5 dwords
     movsxd rax, dword [r12 + rbx + PF_ENTRY_KID_OFF] ; rax = kernel_id, sign-extended
     mov rsi, [r14 + rax*PF_KERNEL_BASE_SCALE] ; rsi = kernel base for this kernel_id
     movsxd rax, dword [r12 + rbx]             ; rax = relative offset to instruction start
     lea rax, [rsi + rax]                      ; rax = absolute address of instruction
     mov r8d, dword [r12 + rbx + PF_ENTRY_LEN_OFF] ; r8d = instruction length
     lea r9d, [ecx + ecx*PF_BACKUP_SCALE]      ; r9d = slot * PREFETCH_MAX_INSN_LEN, 9, 1-cycle LEA replaces mov+imul
     add r9, r13                               ; r9 = backup destination for this slot
     cmp r8d, 8                                ; instruction at least 8 bytes?, attention K/V rows are 6
     jb .copy_byte                             ; no: rare short form, exact byte loop
     mov rdx, [rax]                            ; load first 8 instruction bytes, QWORD
     mov [r9], rdx                             ; store 8 bytes into the 9-byte-strided backup slot
     cmp r8d, 9                                ; 9-byte, ds-prefixed form?
     jl .next_slot                             ; no: 8 bytes copied
     mov dl, byte [rax + PF_NINTH_BYTE]        ; load 9th instruction byte
     mov byte [r9 + PF_NINTH_BYTE], dl         ; store 9th byte to backup
     jmp .next_slot                            ; slot complete
.copy_byte:
     xor r10d, r10d                            ; r10 = byte copy index = 0
.copy_byte_loop:
     movzx edx, byte [rax + r10]               ; Load original byte from instruction
     mov [r9 + r10], dl                        ; Store byte to backup buffer
     inc r10d                                  ; Advance byte index
     cmp r10d, r8d                             ; done after exactly r8d bytes?, every PREFETCH_ROW declares 6 or 9 bytes, the count is never zero
     jl .copy_byte_loop                        ; bottom-tested single-branch loop
.next_slot:
     inc ecx                                   ; Advance to next slot index
     jmp .init_loop                            ; Continue initialization loop
.init_done:
     mov dword [rel prefetch_backup_captured], 1 ; publish: the pristine bytes are archived, one capture per DLL lifetime
     pop r14                                   ; Restore r14
     pop r13                                   ; Restore r13
     pop r12                                   ; Restore r12
     pop rsi                                   ; Restore rsi
     pop rbx                                   ; Restore rbx
     ret                                       ; Return to caller
.pb_ret:
     ret                                       ; already captured, nothing to do
%ifndef __pf_skip_dllmain
     GENERATE_DLLMAIN init_prefetch_backup     ; back up the patchable prefetch bytes at DLL load, ATTACH, before any kernel runs
%endif
     global set_prefetch_distance              ; Existing public entry retains per-call CPUID
set_prefetch_distance:
     push r15                                  ; preserve the private serialization selector
     mov r15d, 1                               ; public entry serializes each call
     jmp set_prefetch_distance_core            ; common implementation
set_prefetch_distance_fast:
     push r15                                  ; private batch entry preserves Win64 nonvolatile r15
     xor r15d, r15d                            ; defer serialization until the complete batch
set_prefetch_distance_core:
     push rbx                                  ; Save non-volatile rbx
     push rsi                                  ; Save non-volatile rsi
     push r12                                  ; Save non-volatile r12
     push r13                                  ; Save non-volatile r13
     push r14                                  ; Save non-volatile r14
     cmp ecx, __pf_num_slots                   ; Validate slot index range, out of range = caller bug: FATAL, never a silent no-op
     jae .bad_distance_fatal                   ; Exit if slot index out of range
     lea r12, [rel patch_points]               ; r12 = patch_points table base
     lea r13, [rel prefetch_backup]            ; r13 = backup buffer base
     lea r14, [rel kernel_bases]               ; r14 = kernel_bases table
     lea r10d, [ecx + ecx*PF_ENTRY_SCALE]      ; r10d = slot_index * 5, 1-cycle LEA
     shl r10d, 2                               ; r10d = slot_index * 20, entry size = 5 dwords
     movsxd rax, dword [r12 + r10 + PF_ENTRY_KID_OFF] ; rax = kernel_id
     mov rsi, [r14 + rax*PF_KERNEL_BASE_SCALE] ; rsi = kernel base for this kernel_id
     movsxd rax, dword [r12 + r10]             ; rax = offset to instruction start
     lea rax, [rsi + rax]                      ; rax = absolute instruction address
     mov r8d, dword [r12 + r10 + PF_ENTRY_LEN_OFF] ; r8d = instruction length
     mov r9d, dword [r12 + r10 + PF_ENTRY_DISP_OFF] ; r9d = disp offset within instruction
     lea r11d, [ecx + ecx*PF_BACKUP_SCALE]     ; r11d = slot * PREFETCH_MAX_INSN_LEN, 9, 1-cycle LEA replaces mov+imul
     add r11, r13                              ; r11 = backup source for this slot
     mov ebx, edx                              ; ebx = new distance value
     cmp r8d, 8                                ; instruction at least 8 bytes?, attention K/V rows are 6
     jb .restore_loop                          ; no: rare short form, exact byte loop
     mov rdx, [r11]                            ; load first 8 saved bytes, QWORD
     mov [rax], rdx                            ; write back to instruction site
     cmp r8d, 9                                ; 9-byte, ds-prefixed form?
     jl .restore_done                          ; no: 8 bytes restored
     mov dl, byte [r11 + PF_NINTH_BYTE]        ; load 9th saved byte
     mov byte [rax + PF_NINTH_BYTE], dl        ; restore 9th byte
     jmp .restore_done                         ; restore complete
.restore_loop:
     xor r10d, r10d                            ; r10 = restore byte index = 0
.restore_byte_loop:
     movzx edx, byte [r11 + r10]               ; Load saved byte from backup
     mov [rax + r10], dl                       ; Write back to instruction site
     inc r10d                                  ; Advance byte index
     cmp r10d, r8d                             ; done after exactly r8d bytes?, every PREFETCH_ROW declares 6 or 9 bytes, the count is never zero
     jl .restore_byte_loop                     ; back edge only while bytes remain
.restore_done:
     cmp ebx, 32                               ; Check if distance < 32
     jae .patch_disp                           ; If >= 32, patch displacement
     lea rcx, [rel nop_templates]              ; Load NOP templates base address
     lea r10, [r8 + SMC_NOP_TAIL]              ; r10 =, len - 1
     shl r10, 4                                ; r10 =, len - 1 * 16, NOP template stride
     add r10, rcx                              ; r10 = address of matching NOP template
     cmp r8d, 8                                ; instruction at least 8 bytes?, attention rows are 6: exact loop, no NOP byte may spill into the next instruction
     jb .nop_loop                              ; no: byte loop
     mov rdx, [r10]                            ; load first 8 template NOP bytes, QWORD
     mov [rax], rdx                            ; write 8 NOP bytes over the instruction
     cmp r8d, 9                                ; 9-byte, ds-prefixed form?
     jl .nop_done                              ; no: 8 bytes written
     mov dl, byte [r10 + PF_NINTH_BYTE]        ; load 9th template NOP byte
     mov byte [rax + PF_NINTH_BYTE], dl        ; write 9th NOP byte
     jmp .nop_done                             ; NOP injection complete
.nop_loop:
     xor ecx, ecx                              ; Clear NOP byte index
.nop_byte_loop:
     movzx edx, byte [r10 + rcx]               ; Load NOP template byte
     mov [rax + rcx], dl                       ; Write NOP byte to instruction site
     inc ecx                                   ; Advance byte index
     cmp ecx, r8d                              ; done after exactly r8d bytes?, every PREFETCH_ROW declares 6 or 9 bytes, the count is never zero
     jl .nop_byte_loop                         ; single bottom-tested loop branch
.nop_done:
     jmp .serialize_dist                       ; Go to pipeline serialize
.patch_disp:
     cmp r9d, -1                               ; Check if disp offset is valid
     je .serialize_dist                        ; Skip displacement patch if no disp field
     mov dword [rax + r9], ebx                 ; Patch 32-bit displacement value
.serialize_dist:
     test r15d, r15d                           ; public per-call mode or private batch mode?
     jz .pf_restore_tail                       ; batch mode: caller performs one final serialization
     xor eax, eax                              ; Zero eax for CPUID
     cpuid                                     ; public entry retains serializing behavior
     jmp .pf_restore_tail                      ; shared restore epilogue
.bad_distance_fatal:
     FATAL "set_prefetch_distance: slot index beyond the emitted patch table" ; caller passed a slot beyond the SMC table, a silent return would leave the caller believing the patch happened
.pf_restore_tail:
     pop r14                                   ; Restore r14
     pop r13                                   ; Restore r13
     pop r12                                   ; Restore r12
     pop rsi                                   ; Restore rsi
     pop rbx                                   ; Restore rbx
     pop r15                                   ; Restore r15, private serialization selector
     ret                                       ; Return to caller
     global set_prefetch_type                  ; Existing public entry retains per-call CPUID
set_prefetch_type:
     push r15                                  ; preserve the private serialization selector
     mov r15d, 1                               ; public entry serializes each call
     jmp set_prefetch_type_core                ; common implementation
set_prefetch_type_fast:
     push r15                                  ; private batch entry preserves Win64 nonvolatile r15
     xor r15d, r15d                            ; defer serialization until the complete batch
set_prefetch_type_core:
     push rbx                                  ; Save non-volatile rbx
     push rsi                                  ; Save non-volatile rsi
     push r12                                  ; Save non-volatile r12
     push r13                                  ; Save non-volatile r13
     push r14                                  ; Save non-volatile r14
     cmp ecx, __pf_num_slots                   ; Validate slot index range, out of range = caller bug: FATAL
     jae .bad_type_fatal                       ; Exit if slot out of range
     cmp edx, 6                                ; Validate type value 0..6, a silent exit would leave the OLD opcode patched while the caller believes the new type is active
     ja .bad_type_fatal                        ; Exit if type out of range
     lea r12, [rel patch_points]               ; r12 = patch_points table base
     lea r14, [rel kernel_bases]               ; r14 = kernel_bases table
     lea ebx, [ecx + ecx*PF_ENTRY_SCALE]       ; slot_index * 5, 1-cycle LEA
     shl ebx, 2                                ; ebx = slot_index * 20, entry size = 5 dwords
     movsxd rax, dword [r12 + rbx + PF_ENTRY_KID_OFF] ; rax = kernel_id
     mov rsi, [r14 + rax*PF_KERNEL_BASE_SCALE] ; rsi = kernel base for this kernel_id
     movsxd rax, dword [r12 + rbx]             ; rax = offset to instruction start
     lea rax, [rsi + rax]                      ; rax = absolute instruction address
     mov r9d, dword [r12 + rbx + PF_ENTRY_TYPE_OFF] ; r9d = type byte offset, offset to ModRM
     lea r10, [rax + r9 - PF_OPCODE_BACK]      ; r10 = opcode start, skip pad bytes like db 0x90
     cmp word [rax], 0x1F0F                    ; Check if NOP at offset 0, no prefix: 5,7,8-byte
     je .serialize_type                        ; Skip type patch if NOPped
     cmp word [rax+PF_PREFIX_BYTE], 0x1F0F     ; Check if NOP at offset 1, 66 prefix: 6,9-byte
     je .serialize_type                        ; Skip type patch if NOPped
     cmp word [r10], 0x1F0F                    ; Check if already NOPped at original position
     je .serialize_type                        ; Skip type patch if NOPped
     cmp byte [rax], 0x90                      ; Check if single-byte NOP, 0x90
     je .serialize_type                        ; Skip type patch if NOPped
     cmp edx, 4                                ; Check if type >= 4
     je .type_prefetchw                        ; Handle PREFETCHW, type 4
     cmp edx, 5                                ; Check if type == 5
     je .type_prefetchwt1                      ; Handle PREFETCHWT1, type 5
     cmp edx, 6                                ; Check if type == 6
     je .type_cldemote                         ; Handle CLDEMOTE, type 6
; Types 0-3: opcode 0F 18, ModRM reg = type
     mov word [r10], 0x180F                    ; Write 0F 18 opcode for PREFETCHNTA/T0/T1/T2
     mov r10b, [rax + r9]                      ; Load current ModRM byte
     and r10b, 0xC7                            ; Clear REG field, keep MOD+R/M
     lea r11d, [rdx*SMC_TYPE_REG_SCALE]        ; type shifted into REG field position
     or r10b, r11b                             ; Insert new prefetch type into ModRM
     mov [rax + r9], r10b                      ; Write patched ModRM byte
     jmp .serialize_type                       ; Go to pipeline serialize
.type_prefetchw:
     mov word [r10], 0x0D0F                    ; Write 0F 0D opcode for PREFETCHW
     mov r10b, [rax + r9]                      ; Load current ModRM byte
     and r10b, 0xC7                            ; Clear REG field, keep MOD+R/M
     or r10b, 0x08                             ; Set reg = 1, 0x08 = 1 << 3 for PREFETCHW
     mov [rax + r9], r10b                      ; Write back patched ModRM
     jmp .serialize_type                       ; Go to pipeline serialize
.type_prefetchwt1:
     mov word [r10], 0x0D0F                    ; Write 0F 0D opcode for PREFETCHWT1
     mov r10b, [rax + r9]                      ; Load current ModRM byte
     and r10b, 0xC7                            ; Clear REG field, keep MOD+R/M
     or r10b, 0x10                             ; Set reg = 2, 0x10 = 2 << 3 for PREFETCHWT1
     mov [rax + r9], r10b                      ; Write back patched ModRM
     jmp .serialize_type                       ; Go to pipeline serialize
.type_cldemote:
     mov word [r10], 0x1C0F                    ; Write 0F 1C opcode for CLDEMOTE
     and byte [rax + r9], 0xC7                 ; clear the REG bits directly in the ModRM byte
.serialize_type:
     test r15d, r15d                           ; public entry or private deferred-serialization batch?
     jz .type_restore_tail                     ; batch mode returns without CPUID
     xor eax, eax                              ; Zero eax for CPUID
     cpuid                                     ; public entry retains per-call CPUID
.type_restore_tail:
     pop r14                                   ; Restore r14
     pop r13                                   ; Restore r13
     pop r12                                   ; Restore r12
     pop rsi                                   ; Restore rsi
     pop rbx                                   ; Restore rbx
     pop r15                                   ; Restore r15, private serialization selector
     ret                                       ; Return to caller
.bad_type_fatal:
     FATAL "set_prefetch_type: slot N or type out of range" ; a silent exit would leave the old opcode patched while the caller believes the new type is active
; Private helper for cycle-tuner slots 0..7 and 27..38: they are all
; 9-byte instructions with a disp32 field. This is only called when both the
; previous and the requested distance are enabled (>=32), so opcode/ModRM
; bytes can stay untouched. Uses caller-saved Win64 registers only.
smc_prefetch_disp_fast:
     lea r10, [rel patch_points]               ; base of 20-byte per-slot metadata
     lea r11d, [ecx + ecx*PF_ENTRY_SCALE]      ; slot * 5
     shl r11d, 2                               ; slot * 20
     movsxd r8, dword [r10 + r11 + PF_ENTRY_KID_OFF] ; associated kernel index
     lea r9, [rel kernel_bases]                ; base of per-kernel addresses
     mov r8, [r9 + r8*PF_KERNEL_BASE_SCALE]    ; kernel base
     movsxd r9, dword [r10 + r11]              ; instruction offset from kernel base
     add r8, r9                                ; exact instruction start
     mov dword [r8 + PF_DISP32_OFF], edx       ; slots 0..38 have disp32 at offset 5
     ret                                       ; no CPUID, outer installer handles serialization
; The dispatcher and genome installer are the only private batch users.
; They patch while the worker pool is between generations, then serialize once
; before publishing another dispatch. CPUID is fully serializing on the caller,
; so separate MFENCE and LFENCE are redundant. The worker-core CMC protocol
; remains a separate requirement, unchanged by this micro-optimization.
; Public setters retain their old behavior.
smc_prefetch_serialize:
     push rbx                                  ; CPUID overwrites EBX, Win64 callee-saved
     xor eax, eax                              ; CPUID leaf 0, an already fully serializing instruction
     cpuid                                     ; orders earlier code stores before later fetch on THIS core
     pop rbx                                   ; restore caller's nonvolatile register
     ret                                       ; no extra MFENCE/LFENCE needed around CPUID
%endmacro
; ============================================================================
; PATCH_Q8_N kernel_label, n_reg32, scratch_reg64
; One shape-wide SMC update: the three scale-row displacements S,2*S,3*S
; where S=n/16 bytes. The job-local d is NOT patched.
; Applied before the generation is published, with all Q8 jobs sharing n.
; All displacement labels follow the respective 4-byte disp32 fields.
; Overwrites: scratch_reg64, ecx, flags. No per-job or per-row SMC writes.
; ============================================================================

%macro PATCH_Q8_N 3
     lea %3, [rel %1]                          ; kernel base, same SMC page as the scale displacements
     mov ecx, %2                               ; stride preparation, ecx is dead after fill
     shr ecx, 4                                ; S = n/16 scale bytes per weight row
     cmp dword [%3 + (%1.scale1_disp_end - %1) - SMC_DISP32_FIELD], ecx ; n unchanged since last dispatch?
     je %%n_unchanged                          ; yes: all three scale displacements already agree
     mov dword [%3 + (%1.scale1_disp_end - %1) - SMC_DISP32_FIELD], ecx ; row 1 scale displacement S
     add ecx, ecx                              ; 2*S, no dependency on job-local d
     mov dword [%3 + (%1.scale2_disp_end - %1) - SMC_DISP32_FIELD], ecx ; row 2 scale displacement 2*S
     mov ecx, %2                               ; derive stride again without an extra temp
     shr ecx, 4                                ; S
     lea ecx, [ecx + ecx*SMC_SCALE_X3]         ; 3*S
     mov dword [%3 + (%1.scale3_disp_end - %1) - SMC_DISP32_FIELD], ecx ; row 3 scale displacement 3*S
     mfence                                    ; match existing SMC publication ordering
     lfence                                    ; match existing dispatch-side SMC fetch ordering
%%n_unchanged:
%endmacro
%endif                                         ; SMC_MACROS_INC
