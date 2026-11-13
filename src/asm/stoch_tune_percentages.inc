; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE
; Independent 12-byte persistent adaptive percentages for stochastic sampling.
; This does NOT alter the greedy file v7_tune_percentages or 64 KiB blocks.
%define STOCH_PERCENT_BYTES 12                 ; three LE uint32 p/256 fractions
%define STOCH_SPLIT_EXP_OFF 4                  ; byte offset of the exponential-stage fraction inside stoch_split_percentage
%define STOCH_SPLIT_NORM_OFF 8                 ; byte offset of the normalization-stage fraction inside stoch_split_percentage
     section .bss align=64
     alignb 64
stoch_split_percentage:
     resd 3                                    ; temperature, exponentials, normalization
     section .rdata
stoch_percent_filename:
     db "v7_stochastic_tune_percentages", 0    ; separate from greedy percentages
     section .text

     SF_START load_stoch_percentages, r12      ; preserve file handle through API calls
     SF_COMMIT                                 ; Win64 shadow space
     mov dword [rel stoch_split_percentage], 128 ; default for temperature
     mov dword [rel stoch_split_percentage+STOCH_SPLIT_EXP_OFF], 128 ; default for exponentials
     mov dword [rel stoch_split_percentage+STOCH_SPLIT_NORM_OFF], 128 ; default for normalization
     lea rcx, [rel stoch_percent_filename]     ; path
     mov edx, LPB_GENERIC_READ                 ; GENERIC_READ
     API_CALL [__imp_CreateFileA], rcx, rdx, 0, 0, LPB_OPEN_EXISTING, LPB_FILE_ATTR_NORMAL, 0 ; open saved splits
     cmp rax, INVALID_HANDLE_VALUE             ; missing file?
     je .stp_load_done                         ; keep defaults
     mov r12, rax                              ; handle across API call
     mov rcx, r12                              ; file handle
     lea rdx, [rel stoch_split_percentage]     ; three dword slots
     mov r8d, STOCH_PERCENT_BYTES              ; exact 12-byte record
     xor r9d, r9d                              ; byte-count out-pointer discarded, pass 0
     API_CALL [__imp_ReadFile], rcx, rdx, r8, r9, 0 ; read the engine-written trusted record
; Validate all three external u32 percentages with one compact clamp loop.
     lea r10, [rel stoch_split_percentage]     ; three contiguous dword entries
     mov ecx, 3                                ; three dword percentages to clamp
.stp_clamp_next:
     mov eax, [r10]                            ; load the stage ratio
     cmp eax, 1                                ; below the allowed minimum?
     adc eax, 0                                ; 0 -> 1
     cmp eax, 255                              ; exceeds the allowed maximum?
     jbe .stp_clamp_inrange                    ; no: keep the value
     mov eax, 255                              ; 256..UINT_MAX -> 255
.stp_clamp_inrange:
     mov [r10], eax                            ; store the clamped ratio
     add r10, 4                                ; next dword entry
     dec ecx                                   ; stage counter
     jnz .stp_clamp_next                       ; clamp all three stages
.stp_close:
     mov rcx, r12                              ; release handle
     API_CALL [__imp_CloseHandle], rcx         ; close
.stp_load_done:
     SF_END r12                                ; return

     SF_START save_stoch_percentages, r12      ; preserve file handle through API calls
     SF_COMMIT                                 ; call space
     cmp qword [rel g_inf_ctx_block], 0        ; did an inference context get created?
     je .stp_save_done                         ; no: the .bss fractions are meaningless, leave the file alone
     lea rcx, [rel stoch_percent_filename]     ; file path
     mov edx, LPB_GENERIC_WRITE                ; GENERIC_WRITE
     API_CALL [__imp_CreateFileA], rcx, rdx, 0, 0, LPB_CREATE_ALWAYS, LPB_FILE_ATTR_NORMAL, 0 ; open/truncate
     cmp rax, INVALID_HANDLE_VALUE             ; cannot save?
     je .stp_save_done                         ; nonfatal, consistent with tune blocks
     mov r12, rax                              ; keep handle
     mov rcx, r12                              ; handle
     lea rdx, [rel stoch_split_percentage]     ; fractions
     mov r8d, STOCH_PERCENT_BYTES              ; exactly twelve bytes
     xor r9d, r9d                              ; byte-count out-pointer discarded, pass 0
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; persist
     mov rcx, r12                              ; release handle
     API_CALL [__imp_CloseHandle], rcx         ; close file
.stp_save_done:
     SF_END r12                                ; return
%undef STOCH_PERCENT_BYTES
%undef STOCH_SPLIT_EXP_OFF
%undef STOCH_SPLIT_NORM_OFF
