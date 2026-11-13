; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%include "inf_structs.inc"

; V7_THREAD_ID result, scratch -- the calling thread's OS identifier.
; win64 reads the TEB; the Linux build has no TEB, so the kernel thread id is
; fetched via gettid (__NR_gettid = 186) once and cached per thread in a TLS
; slot. The result register receives the id; scratch is consumed by the Linux
; path only. Both paths return the same kind of value: the current thread's id.
%macro V7_THREAD_ID 2
%ifdef V7_LINUX
     mov %2, [rel v7_linux_tid wrt ..gottpoff] ; load the thread-local slot offset
     mov %1, [fs:%2]                           ; load the cached kernel thread ID
     test %1, %1                               ; first use on this thread?
     jnz %%done                                ; return the cached ID
     push rax                                  ; preserve accumulator for non-EAX destinations
     push rcx                                  ; syscall destroys RCX
     push r11                                  ; syscall destroys R11
     mov eax, 186                              ; x86-64 Linux __NR_gettid
     syscall                                   ; read the current kernel thread ID
     mov r11, [rel v7_linux_tid wrt ..gottpoff] ; recover the slot offset
     mov [fs:r11], eax                         ; publish the TID for subsequent dispatches
     mov %1, eax                               ; return the TID in the requested register
     pop r11                                   ; restore caller scratch
     pop rcx                                   ; restore live argument
     %ifidni %1, eax
     add rsp, 8                                ; discard the saved accumulator
     %else
     pop rax                                   ; restore the accumulator
     %endif
%%done:
%else
     mov %1, [gs:TEB_THREAD_ID]                ; read the Windows TEB thread ID
%endif
%endmacro
; ============================================================================
; Win32 API imports
; ============================================================================
     extern __imp_GetLogicalProcessorInformationEx ; Get detailed processor topology
     extern __imp_GetProcessAffinityMask       ; Retrieve the process and system affinity masks
     extern __imp_GetCurrentProcessorNumber    ; Logical processor of the calling thread, worker startup line
     extern __imp_GetLastError                 ; Get the last error code for the thread
     extern __imp_VirtualProtect               ; Change memory protection on a region of committed pages
     extern __imp_CreateThread                 ; Create a new thread in the calling process
     extern __imp_DuplicateHandle              ; Duplicate an object handle, retains the calling thread handle
     extern __imp_SetThreadPriority            ; Set the priority level of a thread
     extern __imp_GetThreadPriority            ; Read the priority level of a thread
     extern __imp_GetEnvironmentVariableA      ; Read an environment variable, priority override probe
     extern __imp_SetThreadAffinityMask        ; Restrict the thread to specific logical processors
     extern __imp_WaitForSingleObject          ; Wait until the specified object is in the signaled state
     extern __imp_CloseHandle                  ; Close an open object handle
     extern __imp_CreatePipe                   ; Create an anonymous pipe, v7_execute stdout capture
     extern __imp_SetHandleInformation         ; Set handle inheritance flags, v7_execute pipe setup
     extern __imp_CreateProcessA               ; Create a new process and its primary thread, v7_execute
     extern __imp_WakeByAddressAll             ; Wake all threads waiting on a specific address
     extern __imp_WaitOnAddress                ; Wait for a value at an address to change
     extern __imp_GetStdHandle                 ; Retrieve a handle to a standard device, stdout/stdin
     extern __imp_GetConsoleMode               ; Retrieve console input/output mode, SectorLogo console probe
     extern __imp_SetConsoleMode               ; Set console input/output mode, SectorLogo VT enable
     extern __imp_PeekConsoleInputA            ; Peek at console input records without removing, ReadKey non-blocking poll
     extern __imp_ReadConsoleInputA            ; Read and remove one console input record, ReadKey skip non-key events
     extern __imp_GetFileType                  ; Determine handle type: disk/pipe/char, ReadKey handle-kind probe
     extern __imp_PeekNamedPipe                ; Peek at pipe data without reading, ReadKey non-blocking pipe poll
     extern __imp_WriteFile                    ; Write data to the specified file/device
     extern __imp_LoadLibraryA                 ; Load a DLL into the calling process
     extern __imp_GetProcAddress               ; Get the address of an exported function from a DLL
     extern __imp_FreeLibrary                  ; Free a loaded DLL, dbg_inference.asm includes this file and frees its LoadLibraryA handle
     extern __imp_VirtualAlloc                 ; Reserve or commit pages in virtual address space, mdb_alloc_mem
     extern __imp_VirtualFree                  ; Release/decommit pages from virtual address space, mdb_free_mem
     extern __imp_TerminateProcess             ; Terminate the calling process
     extern __imp_CreateFileA                  ; Create or open a file, MicroDB persistence
     extern __imp_ReadFile                     ; Read data from a file, MicroDB db_load
     extern __imp_GetFileSize                  ; Retrieve the size of a file, MicroDB db_load
     extern __imp_SetFilePointerEx             ; 64-bit file pointer seek, MicroDB chunked writes/reads
     extern __imp_SetEndOfFile                 ; Commit file size at current pointer, file-backed mem allocator
     extern __imp_DeleteFileA                  ; Delete an existing file, MicroDB stale db cleanup
     extern __imp_MoveFileExA                  ; Atomic file replacement, MicroDB safe persistence
     extern __imp_ExitProcess                  ; Terminate the calling process, standalone EXE
     extern __imp_GetCommandLineA              ; Get the full command-line string, standalone EXE
     extern __imp_Sleep                        ; Suspend thread execution, worker-pool join waits: destroy/fatal-unwind teardown, DllMain DETACH
; Win32 API imports, file-backed memory allocator, fbm_alloc / fbm_free
     extern __imp_GetTempPathW                 ; Get temp directory path, wide, for fbm_alloc
     extern __imp_GetTempFileNameW             ; Create unique temp filename, wide, for fbm_alloc
     extern __imp_CreateFileW                  ; Create or open a file, wide, for fbm_alloc
     extern __imp_CreateFileMappingW           ; Create file mapping object, for fbm_alloc
     extern __imp_MapViewOfFile                ; Map file view into address space, for fbm_alloc
     extern __imp_UnmapViewOfFile              ; Unmap file view, for fbm_free
     extern __imp_DeviceIoControl              ; Device I/O control, FSCTL_SET_SPARSE for fbm_alloc
; ============================================================================
; CONSTANTS
; ============================================================================
; Stack-frame offsets for the WRITE_* console-output macros. These are
; OUTPUT helpers, not debugging: the OUT_ prefix says what they serve.
; Discarded optional outputs, WriteFile's bytes-written pointer pass 0.
%define OUT_BUF_OFF 32                         ; Decimal string buffer offset within WRITE_UINT64 frame
%define OUT_UINT64_NUL_OFF 20                  ; uint64_to_str NUL at buffer+20, v7_instr.inc DEC_BUF_LAST_OFF
%define OUT_HEX_X_OFF 1                        ; Offset for 'x' character in "0x" hex prefix
%define OUT_HEX_DIGIT_OFF 1                    ; byte distance from a high digit to the low digit it is paired with
%define OUT_HEX_BUF_OFF 64                     ; Hex string buffer offset within WRITE_HEX64 frame: the 18-byte buffer
%define OUT_CHAR_BUF_OFF 32                    ; Single character buffer offset within WRITE_CHR frame
%define SL_STREAM_BUF_SIZE 1048576             ; stream buffer capacity, 1 MiB
; Named constants for API_CALL internal stack layout
%define APICALL_SHDW0 0                        ; Shadow space slot 0, arg1 save/restore
%define APICALL_SHDW1 8                        ; Shadow space slot 1, arg2 save/restore
%define APICALL_SHDW2 16                       ; Shadow space slot 2, arg3 save/restore
%define APICALL_SHDW3 24                       ; Shadow space slot 3, arg4 save/restore
%define APICALL_ARG5 32                        ; 5th stack argument slot
%define APICALL_ARG6 40                        ; 6th stack argument slot
%define APICALL_ARG7 48                        ; 7th stack argument slot
%define APICALL_ARG8 56                        ; 8th stack argument slot
; Named constant for MATMUL_WRAPPER stack arg
%define MW_ARG_D 48                            ; Feature dimension d, 5th caller arg at rbp+48, MATMUL_WRAPPER reads it directly
%define MAX_WORKERS 63                         ; Maximum number of worker threads supported
%define MAX_JOBS 4096                          ; Maximum number of jobs in the matmul dispatch pool.
%define MAX_CHUNKS MAX_JOBS                    ; Maximum number of kernel copies, same bound as the jobs pool
%define JOB_TAG_SHIFT 16                       ; bit position of the generation tag in next_job / num_jobs
%define QF_SCRATCH_MAX_N 65536                 ; Max n for v7d_q8_fused quantize-once scratch, int8 elements
%define GS 64                                  ; quantization group size, scales are per 64 int8 elements
%define JOB_OUTPUT 0                           ; qword, float*, output buffer slice
%define JOB_INPUT_Q 8                          ; qword, signed char*, input quantized data
%define JOB_INPUT_S 16                         ; qword, float*, input scales
%define JOB_WEIGHT_Q 24                        ; qword, signed char*, weight quantized data, offset-adjusted
%define JOB_WEIGHT_S 32                        ; qword, float*, weight scales, offset-adjusted
%define JOB_INPUT_ELEMENTS 40                  ; dword, int, full input_elements
%define JOB_COLUMN_OFFSET 44                   ; dword, int, column offset for this chunk
%define JOB_FEATURE_DIMENSION 48               ; dword, int, chunk's feature_dimension
%define JOB_MODE 52                            ; dword, int, matmul mode, MATMUL_Q8 or MATMUL_F32
%define JOB_SCALE 56                           ; dword, int, kv_mul for attention mode
; bytes 60-63: padding, 4 bytes to reach 64-byte alignment
%define JOB_SIZE 64                            ; total bytes per job, cache-line aligned, matching C++ ref
%define JOB_SIZE_SHL 6                         ; log2,JOB_SIZE, shift count for the JOB_ADDR macros
; Config struct field offsets are NOT defined here. They come from the
; canonical definitions in inf_structs.inc of every CFG_* name, see its
; header for the re-include site list.Do not add a second copy: every
; CFG_* name must have exactly one definition.
; Matmul operation modes
%define MATMUL_Q8 0                            ; Quantized input, Quantized weights, current path
%define MATMUL_F32 1                           ; Float input, Quantized weights, new path
%define MATMUL_ATTENTION 2                     ; Attention mode, per-head, compiled-in v7s_attention_inner
%define MATMUL_FUSED_F32 3                     ; Fused F32 matmul + RMSNorm + RoPE, approach 6
%define MATMUL_SWIGLU 4                        ; Fused F32xQ8 matmul, gate+up + SwiGLU activation, SiLUxup
%define MATMUL_MEMCPY 5                        ; Dispatched memcpy, rep movsb, ERMS-accelerated
%define MATMUL_QUANTIZE 6                      ; Float to int8 quantize, GS=64, per-group scaling
%define MATMUL_SWIGLU_Q8 7                     ; Fused Q8xQ8 SwiGLU, quantized gate+up matmul + SiLUxup
%define MATMUL_RMSNORM_ROPE 8                  ; Per-head RMSNorm + RoPE, dispatched for large n_heads
%define MATMUL_ROPE_ONLY 10                    ; Per-head RoPE-only rotation, no RMSNorm, for has_qk_norm=0 models
%define MATMUL_VIT_LINEAR 11                   ; ViT pure F32xF32 linear, vision tower linears, dispatched across columns
%define MATMUL_VIT_ATTENTION 12                ; ViT bidirectional attention, per-head, no causal mask, no KV cache
%define MATMUL_Q8_BATCH4 13                    ; Experimental four-token Q8 batch, row-sliced across the pool, one shared weight pass
%define MATMUL_ATTN_TOKEN 14                   ; Causal prefill attention over all heads of one token
%define MATMUL_QKV_POST 15                     ; Independent QKV prefill postprocessing token
%define Q8_BATCH_ARGS_BYTES 120                ; Bytes in one four-token Q8 argument descriptor
; OS constants
%define ERROR_INSUFFICIENT_BUFFER 122          ; Error code for buffer too small
%define INVALID_HANDLE_VALUE (-1)              ; CreateFileA/GetStdHandle failure sentinel
%define MEM_COMMIT 0x1000                      ; Memory allocation: commit pages
%define MEM_RESERVE 0x2000                     ; Memory allocation: reserve virtual address space
%define MEM_RELEASE 0x8000                     ; Memory deallocation: release pages
%define PAGE_READWRITE 0x04                    ; Page protection: read and write
%define STD_OUTPUT_HANDLE 0xFFFFFFF5           ; Win32 STD_OUTPUT_HANDLE, ,DWORD -11 ; use mov ecx, 0xFFFFFFF5, 32-bit mov zero-extends to 0x00000000FFFFFFF5, canonical Win64 ABI for DWORD args
%define DEBUG_BP_ENABLE 0                      ; 1=INT3s active, use with trace_v7.exe, 0=all disabled

%macro DEBUG_BP 0
%if DEBUG_BP_ENABLE
     int3                                      ; Debug breakpoint, caught by VEH handler
%endif
%endmacro

%macro WRITE_STR 1
     push rdi                                  ; save non-volatile rdi, STR_LEN overwrites it
     sub rsp, 40                               ; 32 shadow + 8 scratch, reduced for push, total 48, 16-aligned
     call debug_ensure_std                     ; ensure stdout_handle is cached, valid shadow+alignment now
     lea rdx, [%1]                             ; rdx = string pointer
     STR_LEN rdx, r8d                          ; r8d = length
     test r8d, r8d                             ; empty string?
     jz %%wr_short                             ; yes, skip WriteFile
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     xor r9d, r9d                              ; arg4 = NULL: the bytes-written out-pointer is OPTIONAL and discarded here
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write short string to stdout
%%wr_short:
     add rsp, 40                               ; deallocate shadow + scratch
     pop rdi                                   ; restore non-volatile rdi
%endmacro

%macro WRITE_NL 0
     sub rsp, 48                               ; 32 shadow + 16 scratch, 16-byte aligned
     call debug_ensure_std                     ; ensure stdout_handle is cached
     lea rdx, [fmt_instr_nl]                   ; rdx = newline string
     mov r8d, 1                                ; r8d = length, 1 byte: newline
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     xor r9d, r9d                              ; arg4 = NULL: the bytes-written out-pointer is OPTIONAL and discarded here
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write newline to stdout
     add rsp, 48                               ; deallocate shadow + scratch
%endmacro

%macro WRITE_STR_NL 1
     push rdi                                  ; save non-volatile rdi, STR_LEN overwrites it
     sub rsp, 40                               ; 32 shadow + 8 scratch, reduced for push, total 48, 16-aligned
     call debug_ensure_std                     ; ensure stdout_handle is cached, valid shadow+alignment now
     lea rdx, [%1]                             ; rdx = string pointer
     STR_LEN rdx, r8d                          ; r8d = string length
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     xor r9d, r9d                              ; arg4 = NULL: the bytes-written out-pointer is OPTIONAL and discarded here
     test r8d, r8d                             ; empty string?
     jz %%wsnl_newline                         ; yes, skip string write, still print newline
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write short string to stdout
%%wsnl_newline:
     mov rcx, [stdout_handle]                  ; reload arg1, API_CALL overwritten rcx
     lea rdx, [fmt_instr_nl]                   ; rdx = newline string
     mov r8d, 1                                ; r8d = length, 1 byte: newline
     xor r9d, r9d                              ; arg4 = NULL, optional out-pointer, discarded
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write newline to stdout
     add rsp, 40                               ; deallocate shadow + scratch
     pop rdi                                   ; restore non-volatile rdi
%endmacro

%macro WRITE_UINT64 1
     sub rsp, 64                               ; same aligned RSP as the old push rdi + sub rsp,56
     call debug_ensure_std                     ; ensure stdout_handle is cached
     mov rcx, %1                               ; rcx = unsigned 64-bit value
     lea rdx, [rsp + OUT_BUF_OFF]              ; 21-byte decimal buffer
     call uint64_to_str                        ; rax = first digit; NUL is always buffer+20
     mov rdx, rax                              ; WriteFile arg2 = first digit
     lea r8, [rsp + OUT_BUF_OFF + OUT_UINT64_NUL_OFF] ; pointer to fixed NUL
     sub r8, rdx                               ; byte count = NUL - first digit, O(1)
     mov rcx, [stdout_handle]                  ; WriteFile arg1 = stdout handle
     xor r9d, r9d                              ; arg4 = NULL, bytes-written out-pointer discarded
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write decimal output
     add rsp, 64                               ; restore identical pre-macro RSP
%endmacro

%macro WRITE_HEX64 1
     push rdi                                  ; save non-volatile rdi, used for buffer pointer
     push rsi                                  ; save non-volatile rsi, hex digit lookup table
     add rsp, -128                             ; 18-byte hex buffer at +64..+81 + pad, imm8 encoding of sub rsp, 128; the API_CALL below allocates its shadow BELOW rsp, nothing of it lives in this region
; push+push+128 = 144 = 0 mod 16, so the call below and the API_CALL are
; 16-aligned like the four sibling WRITE_* macros
     call debug_ensure_std                     ; ensure stdout_handle is cached
     mov r10, %1                               ; r10 = value, volatile reg is safe here: the conversion loop is inline, no call until after r10 is dead
     lea rdi, [rsp + OUT_HEX_BUF_OFF]          ; rdi = hex output buffer, 18 bytes at OUT_HEX_BUF_OFF=64: "0x" + 16 hex digits, no NUL, WriteFile gets the explicit count
     lea rsi, [hex_chars]                      ; rsi = hex digit lookup table
     mov byte [rdi], '0'                       ; write "0x" prefix
     mov byte [rdi + OUT_HEX_X_OFF], 'x'       ; write 'x'
     add rdi, 2                                ; advance buffer pointer past prefix
     mov ecx, 8                                ; 8 source BYTES to convert, two hex digits each
%%hex_loop:
     mov rax, r10                              ; copy remaining value
     shr rax, 60                               ; high nibble into bits 0..3
     mov dl, [rsi + rax]                       ; lookup high-nibble character
     mov [rdi], dl                             ; store high digit
     mov rax, r10                              ; reload value for the low nibble
     shr rax, 56                               ; high BYTE into bits 0..7
     and al, 15                                ; isolate low nibble, 0..15, always in table range
     mov dl, [rsi + rax]                       ; lookup low-nibble character
     mov [rdi + OUT_HEX_DIGIT_OFF], dl         ; store low digit one byte later
     add rdi, 2                                ; advance buffer pointer past both digits
     shl r10, 8                                ; shift next source byte into position
     dec ecx                                   ; decrement byte counter
     jnz %%hex_loop                            ; continue until all 8 bytes done
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     lea rdx, [rsp + OUT_HEX_BUF_OFF]          ; arg2 = hex string buffer
     mov r8d, 18                               ; arg3 = 18 bytes, "0x" + 16 hex digits
     xor r9d, r9d                              ; arg4 = NULL: the bytes-written out-pointer is OPTIONAL and discarded here
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write hex string to stdout
     sub rsp, -128                             ; deallocate buffer + scratch + API_CALL frame, imm8 encoding of add rsp, 128, must match the add above
     pop rsi                                   ; restore non-volatile rsi
     pop rdi                                   ; restore non-volatile rdi
%endmacro

%macro WRITE_CHR 1
     sub rsp, 48                               ; 32 shadow + 16 scratch, 16-byte aligned
     call debug_ensure_std                     ; ensure stdout_handle is cached
     mov byte [rsp + OUT_CHAR_BUF_OFF], %1     ; store character in buffer, after shadow, free after debug_ensure_std
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     lea rdx, [rsp + OUT_CHAR_BUF_OFF]         ; arg2 = char buffer pointer
     mov r8d, 1                                ; arg3 = 1 byte to write
     xor r9d, r9d                              ; arg4 = NULL: the bytes-written out-pointer is OPTIONAL and discarded here
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write character to stdout
     add rsp, 48                               ; deallocate buffer + shadow
%endmacro
; ============================================================================
; FATAL message_string
; Print "[FATAL] message_string" to stdout and terminate the process.
; %1 = message string, data label
; Embeds the string in .text via jmp-over-data, avoids section switching .
; NEVER RETURNS.
; ============================================================================

%macro FATAL 1
     jmp %%after_msg                           ; Jump over embedded data string, NASM chooses optimal encoding
%%fatal_msg:
     db "[FATAL] ", %1, 0                      ; embedded error message string
%%after_msg:
     and rsp, -16                              ; round DOWN to a 16-byte boundary: covers both evaluator parities, and any other, drops RSP by at most 8, no matching add rsp, FATAL never returns and handle_fatal re-anchors RSP from engine_saved_rsp
     lea rcx, [rel %%fatal_msg]                ; arg1 = pointer to embedded message, RIP-relative: the bare-label form would emit a DIR32 base relocation here, and FATAL expands at hundreds of sites
     call handle_fatal                         ; print + catch-and-unwind or TerminateProcess, never returns
%endmacro

%macro SET_FTZ_DAZ 0
     ldmxcsr [g_mxcsr_ftz_daz]                 ; load captured FTZ+DAZ MXCSR image, set once at init
%endmacro

%macro MEMORY_COPY_BYTE 0
     rep movsb                                 ; copy bytes rsi->rdi, rcx times
%endmacro

%macro MEMORY_COPY_QWORD 0
     rep movsq                                 ; copy qwords rsi->rdi, rcx times
%endmacro

%macro MEMORY_CLEAR_QWORD 0
     rep stosq                                 ; fill rdi with rax, rcx qwords
%endmacro
; ============================================================================
; STRING HELPER MACROS

; These use NASM macro capabilities to generate common string operation
; sequences inline. Each macro documents which registers it overwrites.

; NOTE: rdi and rcx are used as scratch by scasb/repne in STR_LEN.
; Save them first if they hold live values!
; ============================================================================

%macro STR_LEN 2
     lea rdi, [%1]                             ; rdi = string pointer
     xor eax, eax                              ; al = 0, search for null byte
     mov ecx, -1                               ; max scan count, zero-extends to rcx = 4 GiB-1; every STR_LEN target is a NUL-terminated fixed buffer, largest is 2 MiB, and repne scasb halts on the NUL
     repne scasb                               ; scan for null byte, rdi advances past it
     not ecx                                   ; invert low half: ecx = length + 1, includes null; the upper rcx half is dead, dec ecx zero-extends it away
     dec ecx                                   ; ecx = length, excludes null, zero-extends
     mov %2, ecx                               ; 32-bit store, dst_reg MUST be 32-bit name
%endmacro

%macro API_CALL 1-*
%assign %%n_args %0 - 1                        ; total arguments minus fn_ptr itself
%if %%n_args > 8
%error "API_CALL: more than 8 arguments are not supported (stack slots defined up to APICALL_ARG8 only)" ; assemble-time guard: catches new call sites immediately
%endif
%if %%n_args > 4
%assign %%n_stack %%n_args - 4                 ; stack args = total args beyond 4 register args
%else
%assign %%n_stack 0                            ; no stack args needed, all fit in registers
%endif
; Allocate: 32 shadow + rounded stack-args, 16-aligned
%assign %%raw 32 + ((%%n_stack + 1) & ~1) * 8  ; raw allocation size: shadow + stack-args
%assign %%alloc (%%raw + 15) & ~15             ; round up to 16-byte alignment
     sub rsp, %%alloc                          ; allocate shadow + stack arg space
%if %%n_args >= 1
%ifnidni %2, rcx
     mov rax, %2                               ; load arg1 source value
     mov [rsp + APICALL_SHDW0], rax            ; save to shadow slot 0
%endif
%endif
%if %%n_args >= 2
%ifnidni %3, rdx
     mov rax, %3                               ; load arg2 source value
     mov [rsp + APICALL_SHDW1], rax            ; save to shadow slot 1
%endif
%endif
%if %%n_args >= 3
%ifnidni %4, r8
     mov rax, %4                               ; load arg3 source value
     mov [rsp + APICALL_SHDW2], rax            ; save to shadow slot 2
%endif
%endif
%if %%n_args >= 4
%ifnidni %5, r9
     mov rax, %5                               ; load arg4 source value
     mov [rsp + APICALL_SHDW3], rax            ; save to shadow slot 3
%endif
%endif
; Stack args, 5th+ : save via rax temp, same pattern as above
%if %%n_stack >= 1
     mov rax, %6                               ; load 5th stack arg source value
     mov [rsp + APICALL_ARG5], rax             ; 5th stack arg
%endif
%if %%n_stack >= 2
     mov rax, %7                               ; load 6th stack arg source value
     mov [rsp + APICALL_ARG6], rax             ; 6th stack arg
%endif
%if %%n_stack >= 3
     mov rax, %8                               ; load 7th stack arg source value
     mov [rsp + APICALL_ARG7], rax             ; 7th stack arg
%endif
%if %%n_stack >= 4
     mov rax, %9                               ; load 8th stack arg source value
     mov [rsp + APICALL_ARG8], rax             ; 8th stack arg
%endif
; Load final argument registers from shadow space, no overlap with sources .
; Mirror the identity-skip above: skipped slots keep the original register.
%if %%n_args >= 1
%ifnidni %2, rcx
     mov rcx, [rsp + APICALL_SHDW0]            ; arg1: load from saved source
%endif
%endif
%if %%n_args >= 2
%ifnidni %3, rdx
     mov rdx, [rsp + APICALL_SHDW1]            ; arg2: load from saved source
%endif
%endif
%if %%n_args >= 3
%ifnidni %4, r8
     mov r8, [rsp + APICALL_SHDW2]             ; arg3: load from saved source
%endif
%endif
%if %%n_args >= 4
%ifnidni %5, r9
     mov r9, [rsp + APICALL_SHDW3]             ; arg4: load from saved source
%endif
%endif
; Call: import targets assemble as an indirect call through the IAT entry
; call qword [rip+rel32] ; internal labels assemble as a direct rel32 call.
     call %1                                   ; call the target directly, no staging register, no stack round-trip
     add rsp, %%alloc                          ; deallocate shadow + stack arg space
%endmacro
; NOTE: All stack frames must use SF_START/SF_YMM/SF_QWORD/SF_DWORD/SF_COMMIT/SF_END
; Export generators

%macro EXPORT_DWORD_GETTER 2
     global %1                                 ; exported: dword getter for global %2
%1:                                            ; getter entry point
     mov eax, [%2]                             ; load dword from global variable
     ret                                       ; return to caller
%endmacro

%macro EXPORT_QWORD_GETTER 2
     global %1                                 ; exported: qword getter for global %2
%1:                                            ; getter entry point
     mov rax, [%2]                             ; load qword from global variable
     ret                                       ; return to caller
%endmacro

%macro MATMUL_WRAPPER 2
     SF_START %1                               ; prologue: reserve frame for wrapper
     SF_COMMIT                                 ; finalize stack frame layout
     sub rsp, 16                               ; Extra padding for stack args at [rsp+32..47]
     mov eax, [rbp + MW_ARG_D]                 ; d, 32-bit read, zero-extends to rax
     mov [rsp + APICALL_ARG6], rax             ; place as callee's 6th arg, 64-bit slot, was 32-bit
     mov eax, r9d                              ; zero-extend first param, n to 64-bit, upper bits undefined per ABI
     mov [rsp + APICALL_ARG5], rax             ; place as callee's 5th arg, 64-bit slot, was 32-bit
     mov r9, r8                                ; arg4 = weight tensor pointer
     mov r8, rdx                               ; arg3 = input tensor pointer
     mov rdx, rcx                              ; arg2 = xout, output buffer
     mov ecx, %2                               ; arg1 = MATMUL_Q8 or MATMUL_F32
     call matrix_multiplication_mode           ; dispatch to multithreaded matmul
     add rsp, 16                               ; Remove extra padding
     SF_END                                    ; epilogue: restore frame, return
%endmacro
%define CACHE_LINE_SHIFT 6                     ; cache-line stride as a shift count, 2^6 = 64 bytes

%macro DUMP_ARRAY 2
     push r15                                  ; save non-volatile r15, used as byte-offset cursor
     push rdi                                  ; save non-volatile rdi, used for array base
     push rsi                                  ; stack parity pad: 4 pushes = 32 bytes keeps RSP 0 mod 16 for the WRITE_* macros below
     push r12                                  ; save non-volatile r12, the loop's element load below overwrites it, and r12 is Win64 non-volatile, this is the ABI contract, not a WRITE_UINT64 workaround
     WRITE_STR %1                              ; write array header
     xor r15d, r15d                            ; byte offset into the array = 0
     lea rdi, [rel %2]                         ; rdi = array base
%%loop:
     mov r12, [rdi + r15]                      ; load array[i], r15 = byte offset into non-volatile r12, survives WRITE_UINT64's debug_ensure_std call
     WRITE_UINT64 r12                          ; write value
     WRITE_CHR ' '                             ; write space separator
     add r15, 64                               ; advance byte cursor by one cache line
     cmp r15, MAX_WORKERS << CACHE_LINE_SHIFT  ; offset < 63 * 64?
     jl %%loop                                 ; next element, bottom-test; loop always runs >= once since MAX_WORKERS=63
     WRITE_NL                                  ; write newline
     pop r12                                   ; restore non-volatile r12, pushed last, popped first
     pop rsi                                   ; restore non-volatile rsi, parity pad
     pop rdi                                   ; restore non-volatile rdi
     pop r15                                   ; restore non-volatile r15
%endmacro

%macro FMT 2
%1:
     db %2, 10, 0                              ; define formatted string with newline + null
%endmacro

%macro BSS_DWORD 1
     alignb 64
%1:
     resd 1                                    ; define aligned dword in BSS
%endmacro

%macro BSS_QWORD 1
     alignb 64
%1:
     resq 1                                    ; define aligned qword in BSS
%endmacro

%macro BSS_ARRAY 4
     alignb %1
%2:
     res%3 %4                                  ; define aligned array in BSS
%endmacro
; ============================================================================
; MicroDB persistent hash table struct offsets
%define HT_TABLE 0                             ; qword, pointer to Entry[] on heap
%define HT_CAP 8                               ; dword, capacity, power of two
%define HT_MASK 12                             ; dword, capacity - 1
%define HT_COUNT 16                            ; dword, number of stored entries
; +20..+23 padding, 4 bytes unused
%define KP_BASE 24                             ; qword, key/value pool arena base pointer, single pool
%define KP_USED 32                             ; qword, bytes used in the pool, 64-bit offset
%define KP_CAP 40                              ; qword, pool capacity, 64-bit offset
%define HT_SIZE 48                             ; total g_ht struct size
%define E_HASH 0                               ; dword, hash, 0 = empty, 0xFFFFFFFF = tombstone
%define E_DLEN 8                               ; qword, data byte count, 64-bit for >4GB entries
%define E_KLEN 16                              ; dword, padded key byte count
%define E_FLAG 20                              ; dword, flag bits, bit 0 = persistent
%define E_KEY 24                               ; qword, pool offset of the key string
%define E_DATA 32                              ; qword, pool offset of the value data
%define ENTRY_SZ 40                            ; bytes per Entry record
; ============================================================================
