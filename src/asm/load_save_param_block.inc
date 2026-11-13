; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; load_save_param_block.inc -- persist and reload the cycle tuner's 64 KiB
; parameter blocks as plain files in the working directory.

; Filename: "v7_tune_param_<n>_<d>", built from the shape's registry entry,
; e.g. v7_tune_param_1024_2048. save_param_block writes the whole block,
; load_param_block reads it back and reports TRUE only when a full 64 KiB
; was delivered. Both are error tolerant: any failure just returns, the
; caller keeps its zeroed/random-filled block. A save counts as complete
; only when WriteFile reports success AND accepted the exact 65,536 bytes;
; any other outcome still returns without a fatal. ReadFile's byte count IS
; the load verdict.
; ============================================================================

%define LPB_GENERIC_WRITE 0x40000000           ; GENERIC_WRITE
%define LPB_GENERIC_READ 0x80000000            ; GENERIC_READ
%define LPB_CREATE_ALWAYS 2                    ; CREATE_ALWAYS: create or truncate
%define LPB_OPEN_EXISTING 3                    ; OPEN_EXISTING: load needs an existing file
%define LPB_FILE_ATTR_NORMAL 0x80              ; FILE_ATTRIBUTE_NORMAL
%define LPB_PATH_CAP 64                        ; filename buffer bytes, "v7tune_param_65535_65535" is 27
%define LPB_TEMP_FRAME 48                      ; variadic temp frame: 32 shadow + 1 stack arg + 8 align
%define LPB_TEMP_ARG5 32                       ; byte offset of the 5th v7sprintf argument from rsp

     section .rdata
lpb_fmt:
     db "v7_tune_param_%d_%d", 0               ; "v7_tune_param_1024_2048" with the shape plugged in

     section .text
; ---------------------------------------------------------------------------
; save_param_block -- write a 64 KiB parameter block to v7_tune_param_<n>_<d>.
; IN: rcx = shape n, rdx = shape d, r8 = block pointer. OUT: eax = 1 when
; WriteFile succeeded, accepted the full 64 KiB, and the handle closed
; cleanly; eax = 0 otherwise. Error tolerant: any failure just returns.
; Preserves r12/r13; clobbers rax/rcx/rdx/r8-r11.
; ---------------------------------------------------------------------------
     global save_param_block                   ; internal, inference destroy
     SF_START save_param_block, r12, r13       ; frame + save the pointer/handle register
     SF_QWORD lps_block                        ; block pointer, rcx is volatile across API_CALL
     SF_QWORD lps_n                            ; shape n
     SF_QWORD lps_d                            ; shape d
     SF_DWORD lps_written                      ; bytes accepted by WriteFile, the full-write verdict
     SF_ARRAY lps_path, LPB_PATH_CAP           ; formatted filename
     SF_COMMIT                                 ; Win64 shadow space and stack alignment
     xor r13d, r13d                            ; default verdict: the block was not saved
     mov lps_n, rcx                            ; park the args before the buffer lea consumes rcx
     mov lps_d, rdx                            ; shape d, parked
     mov lps_block, r8                         ; block pointer, parked
     lea rcx, lps_path                         ; arg1: filename buffer, v7sprintf ABI rcx=buf, edx=size, r8=fmt, r9+=varargs
     sub rsp, LPB_TEMP_FRAME                   ; temp frame: 32 shadow + 1 stack arg + 8 align
     mov edx, LPB_PATH_CAP                     ; arg2: capacity
     lea r8, [rel lpb_fmt]                     ; arg3: "v7_tune_param_%d_%d"
     mov r9, lps_n                             ; arg4: n, vararg[0]
     mov rax, lps_d                            ; arg5: d, vararg[1], on the stack
     mov [rsp + LPB_TEMP_ARG5], rax            ; store 5th arg past the shadow
     xor eax, eax                              ; 0 XMM regs for variadic ABI
     call v7sprintf                            ; render the filename
     add rsp, LPB_TEMP_FRAME                   ; release the temp frame
     lea rcx, lps_path                         ; arg1: filename, CWD-relative
     mov rdx, LPB_GENERIC_WRITE                ; arg2: write access
     API_CALL [__imp_CreateFileA], rcx, rdx, 0, 0, LPB_CREATE_ALWAYS, LPB_FILE_ATTR_NORMAL, 0 ; rax = handle
     cmp rax, INVALID_HANDLE_VALUE             ; open failed?
     je .lps_done                              ; error tolerant: report not-saved
     mov r12, rax                              ; r12 = the open handle
     mov dword lps_written, 0                  ; the verdict starts empty
     mov rcx, r12                              ; arg1: handle
     mov rdx, lps_block                        ; arg2: block base
     mov r8d, CYCLE_TUNER_BLOCK_BYTES          ; arg3: 64 KiB, the whole block
     lea r9, lps_written                       ; arg4: bytes-written out pointer, REQUIRED for a synchronous handle
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write it all, the count lands in lps_written
     test eax, eax                             ; the API call itself failed?
     jz .lps_close                             ; yes: not a complete save
     cmp dword lps_written, CYCLE_TUNER_BLOCK_BYTES ; every byte accepted?
     jne .lps_close                            ; short write: not a complete save
     mov r13d, 1                               ; the whole 64 KiB is on disk
.lps_close:
     mov rcx, r12                              ; arg1: handle
     API_CALL [__imp_CloseHandle], rcx         ; close
     test eax, eax                             ; close succeeded?
     jnz .lps_done                             ; yes: the verdict stands
     xor r13d, r13d                            ; failed close: do not report success
.lps_done:
     mov eax, r13d                             ; eax = 1 full write + clean close, 0 otherwise
     SF_END r12, r13                           ; restore and return

; ---------------------------------------------------------------------------
; load_param_block -- read v7_tune_param_<n>_<d> into a 64 KiB block.
; IN: rcx = shape n, rdx = shape d, r8 = block pointer.
; OUT: eax = 1 when a full 64 KiB was loaded, 0 otherwise. Error tolerant:
; a missing file or a short read just returns FALSE, the caller keeps its
; zeroed/random-filled block. Preserves r12; r13 is untouched. Clobbers
; rax/rcx/rdx/r8-r11.
; ---------------------------------------------------------------------------
     global load_param_block                   ; internal, cycle tuner allocation path
     SF_START load_param_block, r12            ; frame + save the pointer/handle register
     SF_QWORD lpl_block                        ; block pointer
     SF_QWORD lpl_n                            ; shape n
     SF_QWORD lpl_d                            ; shape d
     SF_DWORD lpl_read                         ; bytes returned by ReadFile, the 64 KiB verdict
     SF_ARRAY lpl_path, LPB_PATH_CAP           ; formatted filename
     SF_COMMIT                                 ; Win64 shadow space and stack alignment
     mov lpl_n, rcx                            ; park the args before the buffer lea consumes rcx
     mov lpl_d, rdx                            ; shape d, parked
     mov lpl_block, r8                         ; block pointer, parked
     lea rcx, lpl_path                         ; arg1: filename buffer, v7sprintf ABI rcx=buf, edx=size, r8=fmt, r9+=varargs
     sub rsp, LPB_TEMP_FRAME                   ; temp frame: 32 shadow + 1 stack arg + 8 align
     mov edx, LPB_PATH_CAP                     ; arg2: capacity
     lea r8, [rel lpb_fmt]                     ; arg3: "v7_tune_param_%d_%d"
     mov r9, lpl_n                             ; arg4: n, vararg[0]
     mov rax, lpl_d                            ; arg5: d, vararg[1], on the stack
     mov [rsp + LPB_TEMP_ARG5], rax            ; store 5th arg past the shadow
     xor eax, eax                              ; 0 XMM regs for variadic ABI
     call v7sprintf                            ; render the filename
     add rsp, LPB_TEMP_FRAME                   ; release the temp frame
     lea rcx, lpl_path                         ; arg1: filename
     mov rdx, LPB_GENERIC_READ                 ; arg2: read access
     API_CALL [__imp_CreateFileA], rcx, rdx, 0, 0, LPB_OPEN_EXISTING, LPB_FILE_ATTR_NORMAL, 0 ; rax = handle
     cmp rax, INVALID_HANDLE_VALUE             ; file missing or unusable?
     je .lpl_false                             ; error tolerant: report FALSE
     mov r12, rax                              ; r12 = the open handle
     mov rcx, r12                              ; arg1: handle
     mov rdx, lpl_block                        ; arg2: block base
     mov r8d, CYCLE_TUNER_BLOCK_BYTES          ; arg3: request the whole 64 KiB
     lea r9, lpl_read                          ; arg4: bytes-transferred out-parameter, the verdict source
     API_CALL [__imp_ReadFile], rcx, rdx, r8, r9, 0 ; read it all
     cmp dword lpl_read, CYCLE_TUNER_BLOCK_BYTES ; full 64 KiB delivered?
     jne .lpl_close_false                      ; short read: FALSE
     mov rcx, r12                              ; arg1: handle
     API_CALL [__imp_CloseHandle], rcx         ; close
     mov eax, 1                                ; TRUE: a full block was loaded
     SF_END r12                                ; restore and return
.lpl_close_false:
     mov rcx, r12                              ; arg1: handle
     API_CALL [__imp_CloseHandle], rcx         ; close before reporting
     xor eax, eax                              ; FALSE
     SF_END r12                                ; restore and return
.lpl_false:
     xor eax, eax                              ; FALSE
     SF_END r12                                ; restore and return

; ---------------------------------------------------------------------------
; Greedy sampler fork percentage: one raw u32 fraction in one 4-byte file.
; The existing 64 KiB per-shape parameter format is unchanged.
; The percentage stays in [1,255] for nonempty worker/caller ranges.
; Old 8-byte two-slot files load correctly: the active fraction is the first
; dword, and a 4-byte read from a longer file reports a full record.
; ---------------------------------------------------------------------------
%define SP_FILE_BYTES 4                        ; one unsigned dword, little-endian
     section .rdata
sp_filename:
     db "v7_tune_percentages", 0               ; one process-wide sampler tuning file in the current directory
     section .text
     global load_sampler_percentages           ; called at the beginning of every inference creation
load_sampler_percentages:
     SF_START load_sampler_percentages, r12    ; preserve handle across Windows API calls
     SF_DWORD sp_read                          ; byte count returned by ReadFile
     SF_COMMIT                                 ; Win64 shadow area and stack alignment
     mov dword [rel split_percentage], 128     ; default first pass to 50% caller work
     lea rcx, [rel sp_filename]                ; first arg = fixed file path
     mov edx, LPB_GENERIC_READ                 ; read-only handle
     API_CALL [__imp_CreateFileA], rcx, rdx, 0, 0, LPB_OPEN_EXISTING, LPB_FILE_ATTR_NORMAL, 0 ; open existing file, no create
     cmp rax, INVALID_HANDLE_VALUE             ; missing or inaccessible file?
     je .sp_load_done                          ; leave the default
     mov r12, rax                              ; preserve file handle across ReadFile
     mov dword sp_read, 0                      ; reset byte count for failed reads
     mov rcx, r12                              ; first arg = file handle
     lea rdx, [rel split_percentage]           ; second arg = the ratio in .bss
     mov r8d, SP_FILE_BYTES                    ; read exactly four bytes
     lea r9, sp_read                           ; out = read count
     API_CALL [__imp_ReadFile], rcx, rdx, r8, r9, 0 ; read the fraction
     test eax, eax                             ; API successful?
     jz .sp_load_reset                         ; no: restore the default
     cmp dword sp_read, SP_FILE_BYTES          ; full record?
     jne .sp_load_reset                        ; no: restore the default
; Clamp the external file value to the useful nonempty range [1,255].
     mov eax, [rel split_percentage]           ; percentage for the caller split
     cmp eax, 1                                ; below the allowed minimum?
     adc eax, 0                                ; 0 -> 1, all other u32 values unchanged
     cmp eax, 255                              ; exceeds the allowed maximum?
     jbe .sp_clamp_done                        ; no
     mov eax, 255                              ; ensure helper has at least one range unit
.sp_clamp_done:
     mov [rel split_percentage], eax           ; commit normalized fraction
     jmp .sp_load_close_handle                 ; release handle
.sp_load_reset:
     mov dword [rel split_percentage], 128     ; missing/partial record: use half split
.sp_load_close_handle:
     mov rcx, r12                              ; Windows handle to release
     API_CALL [__imp_CloseHandle], rcx         ; close regardless of read outcome
.sp_load_done:
     SF_END r12                                ; done; fraction is ready

     global save_sampler_percentages           ; called on inference teardown after the worker pool has stopped
save_sampler_percentages:
     SF_START save_sampler_percentages, r12    ; preserve file handle across API calls
     SF_DWORD sp_written                       ; exact-write check
     SF_COMMIT                                 ; reserve Win64 call area
     cmp qword [rel g_inf_ctx_block], 0        ; did an inference context get created?
     je .sp_save_done                          ; no: the .bss fraction is meaningless, leave the file alone
     lea rcx, [rel sp_filename]                ; path for the persistent fraction
     mov edx, LPB_GENERIC_WRITE                ; write-only handle
     API_CALL [__imp_CreateFileA], rcx, rdx, 0, 0, LPB_CREATE_ALWAYS, LPB_FILE_ATTR_NORMAL, 0 ; create/truncate file
     cmp rax, INVALID_HANDLE_VALUE             ; open failed?
     je .sp_save_done                          ; nonfatal, same policy as 64 KiB block writes
     mov r12, rax                              ; handle across WriteFile
     mov dword sp_written, 0                   ; initialize byte count
     mov rcx, r12                              ; first arg = file handle
     lea rdx, [rel split_percentage]           ; second arg = the percentage
     mov r8d, SP_FILE_BYTES                    ; write exactly four bytes
     lea r9, sp_written                        ; out = written byte count
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; persist the dword
     mov rcx, r12                              ; first arg = file handle
     API_CALL [__imp_CloseHandle], rcx         ; close, errors remain nonfatal
.sp_save_done:
     SF_END r12                                ; restore handle register and return
