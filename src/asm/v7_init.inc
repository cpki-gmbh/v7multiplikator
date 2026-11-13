; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
; INI_KEY_FIND, lookup the key in scratch_key_buf; eax = dword value

%macro INI_KEY_FIND 0
     lea rcx, [scratch_key_buf]                ; key buffer ready
     xor edx, edx                              ; rdx=NULL, no out_len
     call ht_find_or_fatal                     ; lookup in microdb, FATAL with key name on miss
     mov eax, [rax]                            ; dword value from microdb
%endmacro
; INI_KEY_BEGIN prefix_label, start a fresh key in scratch_key_buf

%macro INI_KEY_BEGIN 1
     lea rdi, [scratch_key_buf]                ; dest for key construction
     lea rsi, [%1]                             ; prefix from .rdata
     call .strcpy_local                        ; copy prefix
%endmacro
; INI_KEY_APPEND suffix_label, append a .rdata suffix to the key

%macro INI_KEY_APPEND 1
     lea rsi, [%1]                             ; suffix from .rdata
     call .strcpy_local                        ; append suffix
%endmacro
; INI_LOAD_DWORD prefix, suffix, dest -- fixed key lookup + dword store, the
; four-line INI_KEY_BEGIN/APPEND/FIND/store sequence as one declaration

%macro INI_LOAD_DWORD 3
     INI_KEY_BEGIN %1                          ; start the key with the section prefix
     INI_KEY_APPEND %2                         ; append the key suffix
     INI_KEY_FIND                              ; lookup in microdb, FATAL with key name on miss
     mov [%3], eax                             ; dword value from microdb
%endmacro
; SHAPE_PARSE_DECIMAL accumulator, done_label -- one supported_resolutions token
; dimension's decimal digit loop; %1 = dword accumulator (n or d), %2 = the
; caller's exit local label. Signed-overflow FATALs included. Clobbers eax.

%macro SHAPE_PARSE_DECIMAL 2
%%next_digit:
     movzx eax, byte [rsi]                     ; load digit character
     sub al, '0'                               ; convert ascii to digit
     cmp al, 9                                 ; valid digit 0-9?
     ja %2                                     ; no, digit parsing complete
     imul %1, 10                               ; *= 10, shift for the next decimal place
     jo .shape_dim_overflow_fatal              ; signed overflow, imul sets OF
     add %1, eax                               ; = *10 + digit, eax is 0..9: movzx zero-extension survived sub al
     jo .shape_dim_overflow_fatal              ; signed overflow, a value near INT32_MAX plus a digit can overflow with no imul overflow
     inc rsi                                   ; advance past digit
     jmp %%next_digit                          ; continue parsing
%endmacro

%macro PF_TABLE_FILL 5
     mov rdi, %1                               ; reset key dest to base, .strcpy_local left it advanced
     lea rsi, [%2]                             ; leading prefix from .rdata, k_no_prefix == "" for the per-shape pass
     call .strcpy_local                        ; append it
     lea rsi, [%3]                             ; table prefix from .rdata
     call .strcpy_local                        ; append it
     mov eax, %4                               ; table index for itoa
     call v7s_itoa                             ; append index to key
     INI_KEY_FIND                              ; lookup + eax = dword value
     mov [r13], eax                            ; store in the pf array slot
     add r13, 4                                ; advance to next slot
     inc %4                                    ; next index
     cmp %4, PF_DIST_TABLE_ENTRIES             ; all entries done?
     jb %5                                     ; continue, bottom-tested
%endmacro
%define MIN_GLPIEX_HEADER 42                   ; minimum GLPIEx entry bytes for safe field reads, the WORD at GLPIEX_GROUP_NUM_OFF ends at byte 42
%define INIT_CALLER_SLOT 1                     ; pool arithmetic: one slot for the caller thread, added to the worker capacity
%define GLPIEX_REL_OFF 0                       ; GLPIEx entry Relationship field offset
%define GLPIEX_SIZE_OFF 4                      ; GLPIEx entry Size field offset
%define GLPIEX_GROUP_COUNT_OFF 30              ; GLPIEx PROCESSOR_RELATIONSHIP.GroupCount offset, WORD
%define GLPIEX_GROUP_MASK_OFF 32               ; GLPIEx GroupMask[0].Mask offset, KAFFINITY
%define GLPIEX_GROUP_NUM_OFF 40                ; GLPIEx GroupMask[0].Group offset, WORD, the processor group number
%define INIT_PAGE_ROUND_BIAS 4095              ; bias that rounds a size up to the next 4KB page boundary, masked off immediately after
     SF_START initialize_once, rbx, rsi, rdi, r12, r13, r14, r15 ; begin stack frame for one-time initialization
; Named SF_DWORD slots replace all magic [rbp-N] offsets
     SF_DWORD returned_length                  ; ReturnedLength for GetLogicalProcessorInformationEx
     SF_DWORD old_prot                         ; VirtualProtect old_prot buffer
     SF_DWORD created_threads                  ; actual count of successfully created worker threads
     SF_QWORD process_mask                     ; GetProcessAffinityMask out: current process affinity mask
     SF_QWORD system_mask                      ; GetProcessAffinityMask out: system affinity mask, required by the API
     SF_COMMIT                                 ; commit stack frame, all SF_DWORD slots now usable
     DEBUG_BP                                  ; debug breakpoint if DEBUG_BP_ENABLE=1
     mov eax, 1                                ; prepare swapped value 1 for atomic latch
     xchg [init_once_flag], eax                ; atomically swap flag with 1, old value in eax
     test eax, eax                             ; check if old flag was 0, first caller
     jz .do_init                               ; first caller enters init
.wait_init:
     pause                                     ; avoid burning core during init spin
     cmp dword [init_ok], 0                    ; init complete?
     jne .done                                 ; yes: proceed, init_ok == 1 is the only success state
     cmp dword [rel v7_engine_dead], 0         ; no: did the initializing thread fail?, FATAL is NOT process-terminal, it unwinds to the C caller with init_ok still 0, and handle_fatal's pool quiesce is skipped when init_ok == 0
     jne .wait_init_dead                       ; yes: .do_init will never complete, spinning here would wedge the thread forever
     jmp .wait_init                            ; still running: keep spinning
.do_init:
     mov dword [shutdown_flag], 0              ; clear shutdown flag, was set by DllMain on detach, pure write avoids RMW stall
%ifdef CPU0_DIAG
     mov dword [cpu0_diag_pending], 1          ; arm the one-time session dump, consumed by the first completed dispatch
%endif
     mov dword [debug_step], 1                 ; step 1: starting init
     sub rsp, 8                                ; scratch slot for MXCSR, 8-byte aligned
     stmxcsr [rsp]                             ; read init-thread MXCSR, CRT default 0x1F80
     or dword [rsp], 0x8040                    ; set FTZ, bit 15 and DAZ, bit 6
     ldmxcsr [rsp]                             ; apply FTZ+DAZ to the init thread
     mov eax, [rsp]                            ; eax = post-OR image, 0x1F80|0x8040 = 0x9FC0
     mov [g_mxcsr_ftz_daz], eax                ; publish captured image for one-instruction SET_FTZ_DAZ
     add rsp, 8                                ; remove scratch slot
; Attention head-scale table: 1/sqrt(head_dim) for head_dim 64..1024 step 32.
; Computed once with the same three FP operations the kernel fallback uses,
; under the captured FTZ/DAZ MXCSR; v7s_attention_inner reads the table.
%define ATTN_SCALE_TABLE_SCALE 4               ; sizeof,float: dword entries in attn_inv_sqrt_by32
     lea rdx, [rel attn_inv_sqrt_by32]         ; rdx = table base
     mov esi, 2                                ; esi = table index, head_dim = index*32 = 64
.init_attn_scale:
     mov eax, esi                              ; eax = index
     shl eax, 5                                ; eax = head_dim = index*32
     vcvtsi2ss xmm0, xmm0, eax                 ; float head_dim, same op as the kernel fallback
     vsqrtss xmm0, xmm0, xmm0                  ; precise sqrt
     vmovss xmm1, [rel rpe_one]                ; 1.0f numerator, same staging as the kernel fallback
     vdivss xmm0, xmm1, xmm0                   ; exactly 1.0 / sqrt(head_dim)
     vmovss [rdx + rsi*ATTN_SCALE_TABLE_SCALE], xmm0 ; store table entry
     inc esi                                   ; next index
     cmp esi, 33                               ; table covers 64..1024 inclusive
     jb .init_attn_scale                       ; loop until all 33 entries are built
     jmp .step1a_entry                         ; skip past .strcpy_local helper, not a function call, don't fall through
; Local helper: copy null-terminated string from rsi to rdi
; IN: rsi = source, rdi = dest OUT: rdi at null, rsi past null
.strcpy_local:
     lodsb                                     ; al = [rsi], rsi++
     stosb                                     ; [rdi] = al, rdi++
     test al, al                               ; end of string?
     jnz .strcpy_local                         ; loop if not null
     dec rdi                                   ; back off trailing null
     ret                                       ; return with rdi at the null, callers overwrite it to chain
; --- 1a: Fill param_set_0 from def_param_table, ht_find "__ini__default__"+suffix ---
.step1a_entry:
     lea rbx, [rel shape_blocks]               ; rbx = shape_blocks base throughout init
     lea r15, [rel def_param_table]            ; table cursor
     mov r14d, DEF_PARAM_ENTRIES               ; loop counter
.param_fill_loop:
     INI_KEY_BEGIN ini_db_def                  ; "__ini__default__" prefix
     mov rsi, [r15]                            ; key suffix from table entry
     call .strcpy_local                        ; append suffix to key buffer
     INI_KEY_FIND                              ; lookup + eax = dword value
     mov ecx, [r15 + DEF_PARAM_VAL_OFF]        ; param_offset within shape_block
     mov [rbx + SHAPE_BLOCK_PARAM_OFF + rcx], eax ; store param in shape_blocks[section]
     add r15, DEF_PARAM_ENTRY_SIZE             ; advance to next table entry
     dec r14d                                  ; decrement counter
     jnz .param_fill_loop                      ; loop until all entries processed
     INI_LOAD_DWORD ini_db_proxy, k_cores_subtract, cores_subtract ; "__ini__proxy__cores_subtract" -> param_set_0
     INI_LOAD_DWORD ini_db_proxy, k_worker_priority, worker_priority ; "__ini__proxy__worker_priority" -> param_set_0
     mov [rel scheduler_effective_priority], eax ; same value seeds the applied-priority state, the shape-local slot above is config only
; --- 1b: Standalone globals, use_physical_cores only ---
; use_physical_cores lives in [proxy] section only
     INI_LOAD_DWORD ini_db_proxy, k_use_physical_cores, use_physical_cores ; "__ini__proxy__use_physical_cores" -> global
     mov dword [rel debug_use_phys_cores], eax ; mirror to debug var
; init:simple_threads switches the pool build: 0 = full placement below, N > 0 = N
; simple workers created without pinning or SMT awareness, priority only
     INI_LOAD_DWORD ini_db_init, k_simple_threads, simple_threads ; "init:simple_threads" -> global
     INI_LOAD_DWORD ini_db_def, k_vit_max_chunks, vit_max_chunks ; "__ini__default__vit_max_chunks" -> global
     INI_LOAD_DWORD ini_db_def, k_vit_min_chunk_cols, vit_min_chunk_cols ; "__ini__default__vit_min_chunk_cols" -> global
; Workers start idle: always_spin stays 0 until the first inference's
; v7s_set_spin(1), and v7s_stop_spin returns them to the OS wait after chat.
; Validate chunk_multiplier >= 1, already in param_set_0 from loop above
     cmp dword [chunk_multiplier], 1           ; must be >= 1
     jae .chunk_mult_ok                        ; valid
     FATAL "chunk_multiplier < 1 in microdb"   ; invalid
.chunk_mult_ok:
; --- 1c: Copy param_set_0 -> block 1 params as defaults ---
     lea rsi, [param_set_0]                    ; source: default params
     lea rdi, [rbx + SHAPE_BLOCK1_OFF + SHAPE_BLOCK_PARAM_OFF] ; dest: block 1 params, rbx still holds shape_blocks base loaded in step 1a
     mov ecx, PARAM_SET_QWORDS                 ; qword count for copy
     rep movsq                                 ; copy defaults into block 1
; Set block 1 n=0, d=0, default, never matches any shape
     mov qword [rbx + SHAPE_BLOCK1_OFF + SHAPE_BLOCK_N_EQU], 0 ; n and d dimensions = 0, no match
; --- 1d: Fill pf_dist_0..40 into block 1 from DB ---
     xor r12d, r12d                            ; loop counter = 0
     lea r13, [rbx + SHAPE_BLOCK1_OFF + SHAPE_BLOCK_PF_DIST_OFF] ; dest: block 1 pf_dist array
     lea r14, [scratch_key_buf]                ; key dest base, loop-invariant; r14 is free here
.pf_dist_fill_loop:
     PF_TABLE_FILL r14, ini_db_def, k_pf_dist_pfx, r12d, .pf_dist_fill_loop ; "__ini__default__pf_dist_<i>"
; --- 1e: Fill pf_type_0..40 into block 1 from DB ---
     xor r12d, r12d                            ; loop counter = 0
     lea r13, [rbx + SHAPE_BLOCK1_OFF + SHAPE_BLOCK_PF_TYPE_OFF] ; dest: block 1 pf_type array
     lea r14, [scratch_key_buf]                ; key dest base
.pf_type_fill_loop:
     PF_TABLE_FILL r14, ini_db_def, k_pf_type_pfx, r12d, .pf_type_fill_loop ; "__ini__default__pf_type_<i>"
; --- 1f: Copy block 1 -> blocks 2..255, defaults for all per-shape blocks ---
     lea rsi, [rbx + SHAPE_BLOCK1_OFF]         ; source: block 1, default params
     lea rdi, [rbx + SHAPE_BLOCK2_OFF]         ; dest: block 2 start
     mov ecx, ((MAX_SHAPE_BLOCKS - 2) * SHAPE_BLOCK_SIZE) / 8 ; 254 blocks * block_size in qwords
     rep movsq                                 ; copy defaults to all per-shape blocks
; Find supported_resolutions string in DB, direct heap pointer
     INI_KEY_BEGIN ini_db_def                  ; "__ini__default__" prefix
     INI_KEY_APPEND k_supported_resolutions    ; "supported_resolutions" suffix
     lea rcx, [scratch_key_buf]                ; key buffer ready
     xor edx, edx                              ; rdx=NULL, no out_len
     call ht_find_or_fatal                     ; lookup in microdb, FATAL with key name on miss
     mov rsi, rax                              ; rsi = resolution string in DB heap, POINTER tail, not the dword INI_KEY_FIND
; Parse space-separated "NxD" tokens from DB heap pointer
     mov r15d, 2                               ; block index starts at 2
.token_skip_spaces:
     movzx eax, byte [rsi]                     ; load character
     cmp al, ' '                               ; space?
     je .skip_char                             ; skip whitespace
     cmp al, 9                                 ; tab?
     je .skip_char                             ; skip tab
     cmp al, 10                                ; newline?
     je .skip_char                             ; skip newline
     cmp al, 13                                ; carriage return?
     jne .token_start                          ; non-whitespace found, start parsing token; CR is skipped below
.skip_char:
     inc rsi                                   ; advance past whitespace
     jmp .token_skip_spaces                    ; continue scanning
.token_start:
     cmp byte [rsi], 0                         ; end of string?
     je .shape_init_done                       ; yes, done with per-shape loading
     cmp r15d, MAX_SHAPE_BLOCKS                ; room for another block?
     jge .shape_overflow_fatal                 ; no, too many shapes
; Parse n
     xor r11d, r11d                            ; n accumulator = 0
     SHAPE_PARSE_DECIMAL r11d, .parse_n_done   ; shared digit loop, both overflow FATALs inside
.parse_n_done:
     cmp byte [rsi], 'x'                       ; expect 'x' separator
     jne .malformed_token_fatal                ; no 'x' -> malformed token, FATAL, do not silently drop remaining shapes
     inc rsi                                   ; advance past 'x'
; Parse d
     xor r14d, r14d                            ; d accumulator = 0
     SHAPE_PARSE_DECIMAL r14d, .parse_d_done   ; shared digit loop
.parse_d_done:
     test r11d, r11d                           ; n == 0?
     jz .malformed_token_fatal                 ; yes, invalid shape
     test r14d, r14d                           ; d == 0?
     jz .malformed_token_fatal                 ; yes, invalid shape
     mov eax, r15d                             ; block index to rax, r15d zero-extends, saves 1 byte vs mov rax,r15
     shl eax, SHAPE_BLOCK_SIZE_SHL             ; byte offset to block, * 512, u32 exact: block index is 0..255
     mov dword [rbx + rax + SHAPE_BLOCK_N_EQU], r11d ; store N dimension
     mov dword [rbx + rax + SHAPE_BLOCK_D_EQU], r14d ; store D dimension
; --- Overlay per-shape params from DB ---
; Build "__ini__NxD__\0" prefix in scratch_key_buf
     push rsi                                  ; save res iterator
     push r15                                  ; save block index, the key construction below uses rdi/rsi/rax only, but r15 must survive the .strcpy_local call sequence
     lea rdi, [scratch_key_buf]                ; dest for key construction
     lea rsi, [ini_db_pfx]                     ; "__ini__" prefix
     call .strcpy_local                        ; copy prefix
     mov eax, r11d                             ; N dimension
     call v7s_itoa                             ; convert N to string
     dec rdi                                   ; back off null from itoa
     mov byte [rdi], 'x'                       ; write 'x' separator
     inc rdi                                   ; advance past 'x'
     mov eax, r14d                             ; D dimension still live, non-volatile
     call v7s_itoa                             ; convert D to string
     dec rdi                                   ; back off null from itoa
     mov word [rdi], 0x5F5F                    ; write "__", two underscores, little-endian
     add rdi, 2                                ; advance past both
     mov byte [rdi], 0                         ; null prefix terminator
     mov r12, rdi                              ; r12 = base position, at prefix null
     mov eax, r15d                             ; rax = block index, r15d zero-extends, saves 1 byte vs mov rax,r15
     shl eax, SHAPE_BLOCK_SIZE_SHL             ; byte offset, * 512, u32 exact: block index is 0..255
     lea r14, [rbx + rax + SHAPE_BLOCK_PARAM_OFF] ; r14 = block's param area base
     lea r15, [rel def_param_table]            ; table cursor
     mov r13d, DEF_PARAM_ENTRIES               ; counter
.sparam_loop:
     mov rdi, r12                              ; reset to prefix null
     mov rsi, [r15]                            ; suffix pointer
     call .strcpy_local                        ; append suffix to key
     INI_KEY_FIND                              ; lookup + eax = dword value
     mov ecx, [r15 + DEF_PARAM_VAL_OFF]        ; ecx = this entry's byte offset within param_set
     mov [r14 + rcx], eax                      ; store to block's param area
     add r15, DEF_PARAM_ENTRY_SIZE             ; advance to next table entry
     dec r13d                                  ; decrement counter
     jnz .sparam_loop                          ; loop until all entries processed
     mov eax, [rsp]                            ; block index from stack, pushed via r15d, high half was zero
     shl eax, SHAPE_BLOCK_SIZE_SHL             ; byte offset to block, * 512, u32 exact: block index is 0..255
     lea r14, [rbx + rax]                      ; r14 = block base, loop-invariant, survives strcpy/itoa as non-volatile
     lea r13, [rbx + rax + SHAPE_BLOCK_PF_DIST_OFF] ; pf_dist base in block
     mov byte [r12 - PF_PFX_BACK1], 0          ; collapse "__" tail to "_", single-underscore per-shape key form
     dec r12                                   ; r12 = loop-invariant key cursor, "__ini__NxD_" null terminator
     xor r15d, r15d                            ; pf index = 0
.spf_dist_loop:
     PF_TABLE_FILL r12, k_no_prefix, k_pf_dist_pfx, r15d, .spf_dist_loop ; fill block's pf_dist[i] from "__ini__NxD_pf_dist_<i>"
; r12 = loop-invariant key cursor, "__ini__NxD_", same prefix as pf_dist, still valid.
     lea r13, [r14 + SHAPE_BLOCK_PF_TYPE_OFF]  ; pf_type base in block, r14 still holds the block base
     xor r15d, r15d                            ; pf index = 0
.spf_type_loop:
     PF_TABLE_FILL r12, k_no_prefix, k_pf_type_pfx, r15d, .spf_type_loop ; "__ini__NxD_pf_type_<i>"
     pop r15                                   ; restore block index for outer loop
     pop rsi                                   ; restore valid res iterator, saved before overlay sections
     inc r15d                                  ; move to next block index
     jmp .token_skip_spaces                    ; parse next token
.malformed_token_fatal:
     FATAL "supported_resolutions: malformed token, expected NxD (missing 'x' separator or n=0 or d=0)" ; abort on bad shape
.shape_overflow_fatal:
     FATAL "supported_resolutions exceeds MAX_SHAPE_BLOCKS" ; abort on too many shapes
.shape_dim_overflow_fatal:
     FATAL "supported_resolutions: shape dimension exceeds INT32_MAX, the NxD token overflowed the 32-bit accumulator and was wrapped"
.shape_init_done:
     cmp r15d, 2                               ; any NxD token parsed?, block index starts at 2 and advances once per stored token, so 2 means the list was empty
     jne .shapes_parsed_ok                     ; yes, at least one per-shape block exists
     FATAL "supported_resolutions is empty, no NxD shape was parsed, every matmul dispatch would hit the no-per-shape-config FATAL"
.shapes_parsed_ok:
     mov eax, [simple_threads]                 ; simple pool requested?
     test eax, eax                             ; all per-shape config blocks are initialized by now
     jnz .simple_pool                          ; yes: skip the topology scan and placement entirely
     mov dword [debug_step], 3                 ; step 3: starting core detection
; Try GetLogicalProcessorInformationEx for physical core detection
; RelationProcessorCore = 0
     mov returned_length, 0                    ; init ReturnedLength = 0
     xor ecx, ecx                              ; RelationProcessorCore, 0
     xor edx, edx                              ; Buffer = NULL, size query
     lea r8, returned_length_raw               ; ReturnedLength
     API_CALL [__imp_GetLogicalProcessorInformationEx], rcx, rdx, r8 ; query required buffer size
     test eax, eax                             ; check if size query succeeded
     jnz .got_glpiex_size                      ; success, modern Win10+, ReturnedLength is valid
; Failure path: expected ERROR_INSUFFICIENT_BUFFER from older systems
     API_CALL [__imp_GetLastError]             ; get error code
     cmp eax, ERROR_INSUFFICIENT_BUFFER        ; expected: buffer too small
     je .got_glpiex_size                       ; expected error, proceed
     FATAL "GetLogicalProcessorInformationEx (size query) returned unexpected error" ; abort, unexpected GLPIEx error
.got_glpiex_size:
; ReturnedLength is set by both paths, success or ERROR_INSUFFICIENT_BUFFER
     mov ecx, returned_length                  ; load returned buffer size from stack slot, GLPIEx wrote it
     test ecx, ecx                             ; check if size is zero
     jz .glpiex_zero                           ; zero size is fatal, jump to FATAL
     cmp ecx, GLPIEX_BUF_MAX                   ; size within our static buffer?
     ja .glpiex_too_big                        ; unreasonable size, jump to FATAL
     mov dword [debug_glpiex_length], ecx      ; save returned_length for debug, fallthrough, FATALs are out of line
     xor ecx, ecx                              ; RelationProcessorCore, 0
     lea rdx, [core_buffer]                    ; buffer for GLPIEx data
     lea r8, returned_length_raw               ; ReturnedLength pointer
     API_CALL [__imp_GetLogicalProcessorInformationEx], rcx, rdx, r8 ; second call with real buffer
     test eax, eax                             ; data query succeeded?
     jnz .got_core_info                        ; yes, parse core info, FATAL falls through
     FATAL "GetLogicalProcessorInformationEx (data query) failed" ; abort, GLPIEx data query failed
.glpiex_zero:
     FATAL "GetLogicalProcessorInformationEx returned zero size" ; abort, GLPIEx returned empty result
.glpiex_too_big:
     FATAL "GetLogicalProcessorInformationEx returned size above GLPIEX_BUF_MAX, raise GLPIEX_BUF_MAX in v7_data.inc"
.got_core_info:
     lea r15, [rel core_map_masks]             ; r15 = core_map_masks base
     mov dword [core_map_count], 0             ; reset core count, pure write avoids RMW stall
     lea rdi, [core_buffer]                    ; rdi = cursor into GLPIEx buffer
     mov ecx, returned_length                  ; load returned size from stack slot, 32-bit mov zero-extends to rcx on x86-64
     add rcx, rdi                              ; rcx = end pointer, buffer + size
.parse_cores:
     cmp rdi, rcx                              ; past end of buffer?
     jae .cores_parsed                         ; done parsing
; Ensure at least MIN_GLPIEX_HEADER bytes remain before reading entry fields
     lea rax, [rdi + MIN_GLPIEX_HEADER]        ; minimum address for safe read
     cmp rax, rcx                              ; does enough room remain?
     ja .cores_parsed                          ; not enough, stop parsing
     cmp dword [rdi+GLPIEX_REL_OFF], 0         ; Relationship == RelationProcessorCore?
     jne .next_core_entry                      ; not a processor core entry, skip
; This scheduler uses one 64-bit SetThreadAffinityMask per thread, so it can only
; honour a single processor group. Reject multi-group topology explicitly instead
; of merging masks that belong to different groups.
     cmp word [rdi+GLPIEX_GROUP_COUNT_OFF], 1  ; GroupCount == 1?
     jne .unsupported_groups_fatal             ; multi-group topology, unsupported here
     cmp word [rdi+GLPIEX_GROUP_NUM_OFF], 0    ; GroupMask[0].Group == 0?
     jne .unsupported_groups_fatal             ; a nonzero group number is a second group, unsupported here
; Read GroupMask[0].Mask at offset +32
     mov rdx, [rdi+GLPIEX_GROUP_MASK_OFF]      ; GroupMask[0].Mask, KAFFINITY
     mov eax, [core_map_count]                 ; load current core count
     cmp eax, MAX_WORKERS + 1                  ; bounds check, every eligible physical core plus room for the caller
     jge .cores_overflow_fatal                 ; more cores than the array holds, report unsupported topology
     mov [r15 + rax*QWORD_SCALE], rdx          ; store core mask in array
     inc dword [core_map_count]                ; increment core count
.next_core_entry:
     mov eax, [rdi+GLPIEX_SIZE_OFF]            ; Size field, skip to next entry; a successful query yields well-formed records only
     add rdi, rax                              ; advance to next GLPIEx entry
     jmp .parse_cores                          ; continue parsing
.cores_parsed:
     mov eax, [core_map_count]                 ; load final core count
     mov dword [debug_core_map_count], eax     ; save for debug
; Read the process affinity mask: only logical processors this process is allowed
; on can host a participant. -1 is the current-process pseudo-handle.
     lea rdx, process_mask                     ; out: process affinity mask
     lea r8, system_mask                       ; out: system affinity mask, required by the API
     API_CALL [__imp_GetProcessAffinityMask], -1, rdx, r8 ; -1 = current process
     test eax, eax                             ; call succeeded?
     jnz .proc_aff_ok                          ; yes, the mask is valid
     FATAL "GetProcessAffinityMask failed while building the worker pool" ; abort, no affinity information
.proc_aff_ok:
     mov rax, process_mask                     ; snapshot the process mask, the SF slot name is already a memory operand
     mov [rel process_affinity_mask], rax      ; BSS copy for the threading report
     mov rax, system_mask                      ; snapshot the system mask for the threading report
     mov [rel system_affinity_mask], rax       ; BSS copy for the threading report
     test process_mask, 1                      ; is caller CPU 0 available to this process?
     jz .caller_cpu0_unavailable_fatal         ; no: the caller must occupy logical processor 0
.proc_aff_cpu0_ok:
; Intersect every parsed physical-core mask with the process affinity mask, then
; count how many eligible physical cores contain CPU 0. Exactly one must.
     xor r10d, r10d                            ; core index
     xor r11d, r11d                            ; eligible physical cores containing CPU 0
     lea r15, [rel core_map_masks]             ; r15 = core masks array
     lea r12, [rel core_map_raw]               ; r12 = raw masks array, threading report, dead after this loop
.mask_cores_by_process:
     cmp r10d, [core_map_count]                ; past last core?
     jge .mask_cores_done                      ; yes, done
     mov rax, [r15 + r10*QWORD_SCALE]          ; core mask
     mov [r12 + r10*QWORD_SCALE], rax          ; store the raw mask before the intersection, threading report
     and rax, process_mask                     ; keep only allowed logical processors
     mov [r15 + r10*QWORD_SCALE], rax          ; store the eligible mask back
     test al, 1                                ; does this core contain CPU 0?
     jz .mask_next                             ; no
     inc r11d                                  ; yes, count it
.mask_next:
     inc r10d                                  ; next core
     jmp .mask_cores_by_process                ; continue
.mask_cores_done:
     cmp r11d, 1                               ; exactly one eligible core may contain CPU 0?
     je .mask_cores_unique                     ; yes
     FATAL "logical processor zero is unavailable for the caller" ; zero or ambiguous CPU 0 physical core
.mask_cores_unique:
; Build the worker affinity candidates from the eligible masks. The caller
; occupies logical processor 0; every other participant gets one distinct bit.
     lea rdi, [rel worker_aff]                 ; rdi = worker_aff array
     xor eax, eax                              ; zero value
     mov ecx, MAX_WORKERS                      ; count of slots
     MEMORY_CLEAR_QWORD                        ; zero all 8-byte slots
     lea r15, [rel core_map_masks]             ; r15 = core masks array
     lea rbx, [rel worker_aff]                 ; rbx = worker affinity array
     xor r14d, r14d                            ; candidate mask count
     cmp dword [rel use_physical_cores], 0     ; physical-core mode?
     je .smt_aware_distribution                ; no: logical mode below
; Physical mode: one participant per physical core. The caller takes CPU 0, so its
; whole physical core is reserved; every other physical core contributes its
; lowest eligible logical processor, in ascending logical-bit order.
     mov r12d, 1                               ; logical bit 0 belongs to the caller
.phys_aff_loop:
     cmp r12d, 64                              ; scanned all logical bit positions?
     jge .phys_aff_done                        ; yes, done
     xor r9d, r9d                              ; build the current logical processor bit
     bts r9, r12                               ; r9 = 1 << r12d
     xor r10d, r10d                            ; core map record index
.phys_scan_core:
     cmp r10d, [core_map_count]                ; past last core?
     jge .phys_next_bit                        ; yes, no core matched this bit
     mov rdx, [r15 + r10*QWORD_SCALE]          ; core mask
     test rdx, rdx                             ; empty eligible mask?
     jz .phys_scan_next                        ; yes, skip
     test dl, 1                                ; core containing CPU 0?
     jnz .phys_scan_next                       ; caller occupies this PHYSICAL core, skip it
     blsi rax, rdx                             ; lowest eligible logical processor of this core
     cmp rax, r9                               ; is it the bit we are testing?
     jne .phys_scan_next                       ; no, this core contributes a different bit
     cmp r14d, MAX_WORKERS                     ; candidate slots exhausted?
     jge .workers_over_fatal                   ; yes, more cores than the pool can host
     mov [rbx + r14*QWORD_SCALE], r9           ; record this core's lowest eligible processor
     inc r14d                                  ; one more candidate
     jmp .phys_next_bit                        ; this bit is claimed, advance
.phys_scan_next:
     inc r10d                                  ; next core
     jmp .phys_scan_core                       ; continue
.phys_next_bit:
     inc r12d                                  ; next logical bit
     jmp .phys_aff_loop                        ; continue
.phys_aff_done:
     lea r13d, [r14 + INIT_CALLER_SLOT]        ; physical capacity = caller + worker candidates + the caller slot
     jmp .workers_configured                   ; worker count from the physical capacity
; Logical mode: one participant per eligible logical processor. Union all eligible
; bits, remove only the caller's CPU 0, then enumerate the rest lowest-first.
.smt_aware_distribution:
     xor r10d, r10d                            ; core index
     xor edx, edx                              ; union of eligible logical CPUs
.smt_union_loop:
     cmp r10d, [core_map_count]                ; past last core?
     jge .smt_union_done                       ; yes, done
     or rdx, [r15 + r10*QWORD_SCALE]           ; accumulate eligible logical processors
     inc r10d                                  ; next core
     jmp .smt_union_loop                       ; continue
.smt_union_done:
     test dl, 1                                ; caller CPU 0 present in the union?
     jnz .smt_union_have_caller                ; yes
     FATAL "logical processor zero is unavailable for the caller" ; union lost CPU 0, impossible after the CPU-0 check above
.smt_union_have_caller:
     popcnt r13, rdx                           ; logical capacity including the caller
     btr rdx, 0                                ; remove caller CPU 0 only
.smt_candidate_loop:
     test rdx, rdx                             ; any remaining logical processors?
     jz .workers_configured                    ; no, candidates done
     cmp r14d, MAX_WORKERS                     ; candidate slots exhausted?
     jge .workers_over_fatal                   ; yes, more logical processors than the pool can host
     blsi rax, rdx                             ; lowest remaining logical bit
     mov [rbx + r14*QWORD_SCALE], rax          ; record it
     inc r14d                                  ; one more candidate
     blsr rdx, rdx                             ; remove that bit
     jmp .smt_candidate_loop                   ; continue
.workers_configured:
     mov dword [debug_worker_count_raw], r14d  ; constructed worker candidates for debug
     mov dword [debug_r13_at_workers], r13d    ; eligible capacity including caller for debug
     mov eax, [cores_subtract]                 ; load cores_subtract from param_set
     mov dword [debug_cores_subtract], eax     ; save for debug
     mov dword [debug_step], 4                 ; step 4: computing worker count
     cmp eax, 1                                ; cores_subtract must leave the caller its one slot
     jl .bad_cores_subtract_fatal              ; 0 or negative: caller already consumes one processor
     cmp eax, r13d                             ; cores_subtract >= capacity?
     jge .no_workers_fatal                     ; yes: no worker would be created
     mov ecx, r13d                             ; eligible capacity
     sub ecx, eax                              ; workers = capacity - cores_subtract
     cmp ecx, MAX_WORKERS                      ; over MAX_WORKERS?
     jg .workers_over_fatal                    ; yes: config error, do not cap
     cmp ecx, r14d                             ; do the candidates cover every requested worker?
     jg .core_placements_fatal                 ; no: placement/config bug, do not correct
     mov [num_workers], ecx                    ; store worker count
     mov dword [debug_num_workers_computed], ecx ; save computed worker count for debug
     jmp .chunks_select                        ; full-placement pool: worker count is final
.simple_pool:
     cmp eax, MAX_WORKERS                      ; within the pool arrays?
     ja .simple_over_fatal                     ; no: config error, never clamp
     mov [num_workers], eax                    ; N simple workers, the count comes straight from the key
.chunks_select:
     mov eax, [rel num_chunks]                 ; param_set_0 num_chunks, 0 = auto
     test eax, eax                             ; check if INI specified chunk count
     jz .chunks_auto                           ; zero: compute from the participant count
     mov dword [rel num_chunks_source], 1      ; ini override supplied the chunk count
     jmp .chunks_check                         ; non-zero: use as-is, the 32-bit load already zero-extended rax
.chunks_auto:
     mov dword [rel num_chunks_source], 0      ; chunk count computed from the participant count
     mov eax, [num_workers]                    ; load worker count for auto chunks
     inc eax                                   ; default = num_workers + 1
     mov ecx, [chunk_multiplier]               ; multiplier, 32-bit DB value; the load zero-extends into rcx
     imul rax, rcx                             ; rax =, num_workers+1 * chunk_multiplier
.chunks_check:
     cmp rax, MAX_CHUNKS                       ; validate the TRUE product, or the INI value against MAX_CHUNKS, reject, never clamp
     jbe .chunks_valid                         ; unsigned: both paths are non-negative, INI load zero-extends; the product is < 2^62
     FATAL "num_chunks outside [0, MAX_CHUNKS], fix threading_proxy.ini" ; config error, never silently clamped, a wrapped 32-bit product could otherwise re-enter the legal range
.chunks_valid:
     mov [num_chunks], eax                     ; store final chunk count, the jbe above proved 0 <= value <= MAX_CHUNKS, so the low 32 bits ARE the whole value
     mov dword [debug_step], 5                 ; step 5: chunk count computed
; ================================================================
; STEP 6: Set up v7s_q8_chunk, VirtualProtect RWX, init prefetch backup
     lea rcx, [rel v7s_q8_chunk]               ; rcx = kernel start
     lea rdx, [rel v7_kernels_end]             ; rdx = end of the whole kernel block, SMC region end anchor
     sub rdx, rcx                              ; rdx = kernel code size in bytes
; Page-align the base address down
     mov rax, rcx                              ; copy kernel start for alignment
     and rax, ~4095                            ; rax = page-aligned base
     sub rcx, rax                              ; rcx = offset of kernel start within page
     lea rdx, [rdx + rcx + INIT_PAGE_ROUND_BIAS] ; size from page base to kernel end + page-round bias, one instruction
     and rdx, ~4095                            ; mask off low 12 bits
     mov rcx, rax                              ; rcx = page-aligned base address
     mov r8d, 0x40                             ; PAGE_EXECUTE_READWRITE
     lea r9, old_prot_raw                      ; old_prot buffer, safe area past config
     API_CALL [__imp_VirtualProtect], rcx, rdx, r8, r9 ; make kernel RWX for SMC
     test eax, eax                             ; check VirtualProtect result
     jnz .vp_ok                                ; VirtualProtect succeeded, continue
     FATAL "VirtualProtect failed, SMC cannot write to code section" ; critical, terminate
.vp_ok:
     mov dword [debug_step], 6                 ; step 6: SMC setup done
; Prefetch backup: save original instruction bytes for SMC restoration.
; The once-flag inside init_prefetch_backup archives the pristine bytes on
; the first pool init only; a later pool lifetime must not archive the
; previous lifetime's patched or NOPped instructions as "originals".
     call init_prefetch_backup                 ; save original prefetch bytes to backup buffer
     mov dword [debug_step], 11                ; step 11: clearing job pool
     JOB_BASE                                  ; rsi = &jobs_arr[0], the index is known zero
     mov rdi, rsi                              ; rdi = jobs array base, MEMORY_CLEAR_QWORD walks rdi with its own stride
     mov ecx, [num_chunks]                     ; zero only entries we'll use
     shl ecx, JOB_SIZE_SHL - 3                 ; qword count for the job-pool clear
     xor eax, eax                              ; ensure full 64-bit rax is 0
     MEMORY_CLEAR_QWORD                        ; memset jobs_arr to 0, 8 bytes at a time vs 1 byte
     mov dword [debug_step], 12                ; step 12: launching workers
.sd_launch_threads:
; Create worker threads, pass worker id, int as param
; C++ ref: CreateThread,nullptr, 0, worker_proc,, void*, uintptr_t i, 0, nullptr
     lea r15, [rel threads]                    ; r15 = threads array
     lea rbx, [rel worker_aff]                 ; rbx = worker affinity array
     xor r14d, r14d                            ; worker index = 0
     mov dword created_threads, r14d           ; count of successfully created threads
     mov r13d, [simple_threads]                ; simple-pool flag for the loop, 0 = full placement, N > 0 = no affinity work
.launch_loop:
     cmp r14d, [num_workers]                   ; all workers launched?
     jge .launch_done                          ; yes, done
     test r13d, r13d                           ; simple pool?
     jnz .launch_create                        ; yes: no affinity candidate exists, create plainly
; Check affinity mask before creating thread, skip if no core assigned
     mov rdx, [rbx + r14*QWORD_SCALE]          ; affinity mask
     test rdx, rdx                             ; affinity mask valid?
     jz .launch_next                           ; skip this worker, no valid core
.launch_create:
; r9d = worker index, zero-extends to 64-bit per Win64 ABI for pointer-sized param
     mov r9d, r14d                             ; pass worker index as param, 32-bit, zero-extends
     lea r8, [rel worker_proc]                 ; RIP-relative, required for ASLR-safe address load
     API_CALL [__imp_CreateThread], 0, 0, r8, r9, 0, 0 ; create worker thread
     test rax, rax                             ; thread created?
     jz .create_thread_fatal                   ; failed, FATAL
     mov [r15 + r14*QWORD_SCALE], rax          ; store thread handle
     mov rcx, rax                              ; thread handle to arg1
     mov edx, [rel scheduler_effective_priority] ; set priority from the applied-priority state, default 15 = TIME_CRITICAL
     API_CALL [__imp_SetThreadPriority], rcx, rdx ; set worker thread priority
     test eax, eax                             ; SetThreadPriority succeeded?
     jz .priority_fatal                        ; failed, FATAL
     test r13d, r13d                           ; simple pool?
     jnz .launch_counted                       ; yes: priority only, the worker is never pinned
; SetThreadAffinityMask, affinity already validated non-zero above
     mov rcx, [r15 + r14*QWORD_SCALE]          ; thread handle, overwritten by API calls
     mov rdx, [rbx + r14*QWORD_SCALE]          ; affinity mask, already verified non-zero
     API_CALL [__imp_SetThreadAffinityMask], rcx, rdx ; set worker core affinity, returns the previous 64-bit KAFFINITY, 0 only on failure
     test rax, rax                             ; succeeded?, a fresh worker's previous mask is the process mask, bit 0 set
     jz .affinity_fatal                        ; failed, FATAL
.launch_counted:
     inc dword created_threads                 ; count this worker, priority + placement applied
.launch_next:
     inc r14d                                  ; next worker index
     jmp .launch_loop                          ; continue launch loop
.create_thread_fatal:
     FATAL "CreateThread failed, cannot create worker thread" ; abort, thread creation failed
.affinity_fatal:
     FATAL "SetThreadAffinityMask failed, cannot pin worker to core" ; abort, affinity mask failed
.priority_fatal:
     FATAL "SetThreadPriority failed, cannot set worker priority"
.worker_count_mismatch_fatal:
     FATAL "worker launch: created_threads != num_workers, the eligible affinity candidate set did not cover the requested worker count"
.core_placements_fatal:
     FATAL "core placement produced fewer cores than workers, fix use_physical_cores/cores_subtract in INI"
.workers_over_fatal:
     FATAL "num_workers exceeds MAX_WORKERS, fix cores_subtract/topology config"
.simple_over_fatal:
     FATAL "init:simple_threads exceeds MAX_WORKERS, lower the key value or raise MAX_WORKERS in v7_core.inc"
.bad_cores_subtract_fatal:
     FATAL "cores_subtract must be at least one because the caller occupies one processor"
.no_workers_fatal:
     FATAL "cores_subtract leaves no worker processors available"
.caller_cpu0_unavailable_fatal:
     FATAL "logical processor zero is unavailable for the caller"
.unsupported_groups_fatal:
     FATAL "processor topology spans unsupported processor groups"
.cores_overflow_fatal:
     FATAL "processor topology reports more physical cores than core_map_masks holds, raise MAX_WORKERS in v7_core.inc"
.pool_coverage_fatal:
     FATAL "worker pool coverage assertion failed: caller plus worker masks do not match the eligible processor set"
.caller_handle_fatal:
     FATAL "Cannot obtain a persistent handle to the calling thread"
.caller_get_priority_fatal:
     FATAL "Cannot read the calling thread priority"
.caller_set_priority_fatal:
     FATAL "Cannot set the calling thread priority"
.wait_init_dead:
     FATAL "initialize_once: the initializing thread died (a FATAL unwound to its C caller), init_ok will never be set" ; refusing to spin forever
.launch_done:
     mov eax, created_threads                  ; eax = actual created thread count
     cmp eax, [num_workers]                    ; placement delivered what config asked for?
     jne .worker_count_mismatch_fatal          ; no, placement/config bug
; Wait until every created worker has STARTED, lock inc'd workers_alive at
; worker_proc entry : after this, the count is created-and-live, so a later
; shutdown's wait-for-zero can never miss a thread that had not yet run its
; first instruction. Budget 2s in 1ms ticks, workers inc within microseconds .
     mov r15d, 2000                            ; poll budget in 1ms ticks
.init_wait_started:
     mov eax, [workers_alive]                  ; live count, inc'd by each started worker
     cmp eax, created_threads                  ; all created workers started?, read the SF slot every pass: rcx/eax are clobbered by Sleep, a once-loaded count register would compare garbage
     jge .init_started_done                    ; yes, startup hole closed
     mov ecx, 1                                ; Sleep,1ms
     API_CALL [__imp_Sleep], rcx               ; init code, not a hot path
     dec r15d                                  ; one tick consumed
     jnz .init_wait_started                    ; keep polling
; budget exhausted with a not-yet-started worker: proceed. Thread start
; latency under heavy system load (concurrent model runs, paging) is
; expected runtime variation, not a defect, and the shutdown side re-polls
; workers_alive with its own budget, so a slow starter is still observed
; there before the DLL unmaps.
.init_started_done:
; Capture the initialization caller thread once: a persistent duplicated handle so
; later priority/affinity operations act on THIS thread, and the original priority
; so shutdown can restore it.
     lea r9, [rel caller_thread_handle]        ; out: duplicated handle
     API_CALL [__imp_DuplicateHandle], -1, -2, -1, r9, 0, 0, 2 ; src process -1, src thread -2, target process -1, DUPLICATE_SAME_ACCESS=2
     test eax, eax                             ; duplication succeeded?
     jz .caller_handle_fatal                   ; no: cannot retain the caller thread handle
.caller_handle_ok:
     mov rcx, [rel caller_thread_handle]       ; the real caller thread handle
     API_CALL [__imp_GetThreadPriority], rcx   ; read the original priority
     cmp eax, 0x7fffffff                       ; THREAD_PRIORITY_ERROR_RETURN?
     je .caller_get_priority_fatal             ; yes, the priority could not be read
.caller_get_prio_ok:
     mov [rel caller_saved_priority], eax      ; remember the original priority for unpin
     cmp dword [simple_threads], 0             ; simple pool?
     jne .caller_no_pin                        ; yes: the caller is never pinned, the priority still applies
; Pin the caller to logical processor 0 through the real handle so shutdown can
; restore it even when invoked from another thread.
     mov rcx, [rel caller_thread_handle]       ; real caller thread handle
     mov edx, 1                                ; mask = CPU 0
     API_CALL [__imp_SetThreadAffinityMask], rcx, rdx ; pin caller to CPU 0
     test rax, rax                             ; succeeded? returns the previous KAFFINITY, 0 only on failure
     jz .caller_affinity_fatal                 ; failure, FATAL
     mov [saved_affinity], rax                 ; save previous affinity mask for unpin
     mov dword [pinned_flag], 1                ; mark pinned flag
.caller_no_pin:
; Set the caller's initial priority from the same applied value the workers use.
     mov rcx, [rel caller_thread_handle]       ; real caller thread handle
     mov edx, [rel scheduler_effective_priority] ; priority from the applied-priority state, same source as the workers
     API_CALL [__imp_SetThreadPriority], rcx, rdx ; apply to the caller
     test eax, eax                             ; succeeded?
     jz .caller_set_priority_fatal             ; no: the caller priority could not be set
.caller_prio_ok:
     V7_THREAD_ID eax, rax                     ; ID of the thread whose handle and affinity were captured above
     mov [rel caller_bound_tid], eax           ; publish the initial bound caller's identity
; Init-only assertion: the caller plus the worker masks must form a disjoint set of
; single-bit masks covering exactly num_workers+1 logical processors.
     cmp dword [simple_threads], 0             ; simple pool?
     jne .pool_cover_physical                  ; yes: no per-worker masks exist, skip the placement assertion
     mov edi, 1                                ; coverage accumulator, bit 0 = caller
     mov r11d, 1                               ; set-bit count, caller counted
     xor ebx, ebx                              ; worker index
     lea r12, [rel worker_aff]                 ; r12 = worker affinity array base
.pool_cover_loop:
     cmp ebx, [num_workers]                    ; all workers checked?
     jge .pool_cover_done                      ; yes
     mov rax, [r12 + rbx*QWORD_SCALE]          ; this worker's mask
     test rax, rax                             ; single bit set?
     jz .pool_coverage_fatal                   ; zero mask, not a valid participant
     blsr rdx, rax                             ; rdx = mask & (mask-1), ZF set when the result is zero
     jnz .pool_coverage_fatal                  ; multi-bit mask, not a valid single participant
     test rdi, rax                             ; already covered by an earlier participant?
     jnz .pool_coverage_fatal                  ; duplicate mask
     or rdi, rax                               ; add to coverage
     inc r11d                                  ; one more participant
     inc ebx                                   ; next worker
     jmp .pool_cover_loop                      ; continue
.pool_cover_done:
     mov eax, [num_workers]                    ; participants must be num_workers + 1
     inc eax                                   ; include the caller
     cmp r11d, eax                             ; participants = num_workers + 1?
     jne .pool_coverage_fatal                  ; no, count mismatch
     cmp dword [rel use_physical_cores], 0     ; logical mode with subtract 1?
     jne .pool_cover_physical                  ; physical mode: coverage is a subset by design
     cmp dword [cores_subtract], 1             ; subtract == 1 asks for the full capacity?
     jne .pool_cover_physical                  ; not the full-capacity case
     popcnt rax, process_mask                  ; eligible logical processors for this process
     popcnt rdx, rdi                           ; covered participants
     cmp rax, rdx                              ; equal counts means full eligible coverage
     jne .pool_coverage_fatal                  ; mismatch, the pool does not cover the eligible set
.pool_cover_physical:
     call worker_priority_apply_pending        ; apply deferred pre-pool priority request first, no-op when none
     call worker_priority_check_env_override   ; apply v7_override_prio override last, it must win over the parked request
     mov ecx, created_threads                  ; pass the created-thread count to the report
     call init_threading_report                ; print the complete pool-creation report: every thread exists, every pin and priority is applied
     lea rbx, [rel shape_blocks]               ; rbx = shape_blocks base, STEP 12 overwritten it
     mov r10d, SHAPE_BLOCK_SIZE                ; first slot's byte offset (512)
.backfill_chunks:
     cmp r10d, SHAPE_BLOCKS_TOTAL              ; MAX_SHAPE_BLOCKS * 512, past last block?
     jae .backfill_done                        ; yes, done
     cmp dword [rbx + r10 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_NUM_CHUNKS], 0 ; auto, 0 ?
     jne .backfill_next                        ; non-zero: INI override, leave alone
     mov edx, [num_chunks]                     ; get final global value, INI-set or recomputed
     mov dword [rbx + r10 + SHAPE_BLOCK_PARAM_OFF + PARAM_OFFSET_NUM_CHUNKS], edx ; backfill
.backfill_next:
     add r10d, SHAPE_BLOCK_SIZE                ; next slot's byte offset
     jmp .backfill_chunks                      ; continue loop
.backfill_done:
; --- Snapshot param_set_0 to debug_param_set_0 for dump verification ---
     lea rsi, [param_set_0]                    ; source = block 0 param set, what kernel uses
     lea rdi, [rel debug_param_set_0]          ; dest = BSS debug array
     mov ecx, PARAM_SET_QWORDS                 ; 12 qwords = 24 dwords = 96 bytes
     MEMORY_COPY_QWORD                         ; copy param_set_0 to debug buffer
     mfence                                    ; ensure ALL prior stores visible before publishing init_ok to other cores
     mov dword [init_ok], 1                    ; signal init success
.done:
     cmp dword [simple_threads], 0             ; simple pool?
     jne .caller_bound                         ; yes: no caller binding exists, never pin on a handoff
     V7_THREAD_ID eax, rax                     ; actual thread executing this dispatch's initialize_once
     cmp eax, [rel caller_bound_tid]           ; is this the thread currently pinned as the caller?
     je .caller_bound                          ; common case, no calls or changes
     call scheduler_bind_current_caller        ; rare host-thread handoff, rebind CPU 0 to the active caller
.caller_bound:
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; close stack frame, restore non-volatiles, return
.caller_affinity_fatal:
     FATAL "v7_init: SetThreadAffinityMask failed for caller (core 0), affinity mask excludes core 0"
; ============================================================================
; Threading report: one complete print per pool creation, production output.
; Called from .pool_cover_physical with ecx = created thread count. Reads only
; BSS state, prints only, FATALs never, allocates nothing.
; ============================================================================
init_threading_report:
     SF_START init_threading_report, rbx, r12, r13, r14, r15 ; own frame, called from inside the initialize_once frame
     SF_COMMIT                                 ; finalize stack frame layout
     mov r14d, ecx                             ; r14d = created thread count, loop bound for the created lines
     cmp dword [simple_threads], 0             ; simple pool active?
     jne .simple_report                        ; yes: compact report, the placement detail below does not apply
     WRITE_STR lbl_thr_header                  ; report heading
     WRITE_NL                                  ; end the heading line
; Config the decision consumed
     WRITE_STR lbl_thr_ini                     ; "[threads] ini: cores_subtract="
     mov r12d, [cores_subtract]                ; dword load zero-extends, a qword load would print the neighbouring dword as garbage
     WRITE_UINT64 r12                          ; the reserved-core count we configured
     WRITE_STR lbl_thr_s_wpri                  ; " worker_priority="
     mov r12d, [worker_priority]               ; dword load, param slot
     WRITE_UINT64 r12                          ; the INI priority slot value
     WRITE_STR lbl_thr_s_upc                   ; " use_physical_cores="
     mov r12d, [use_physical_cores]            ; dword load, placement mode
     WRITE_UINT64 r12                          ; placement mode we configured
     WRITE_STR lbl_thr_s_nch                   ; " num_chunks="
     mov r12d, [num_chunks]                    ; dword load, final chunk count
     WRITE_UINT64 r12                          ; final chunk count the dispatchers use
     WRITE_STR lbl_thr_s_chm                   ; " chunk_multiplier="
     mov r12d, [chunk_multiplier]              ; dword load, auto multiplier
     WRITE_UINT64 r12                          ; auto-chunk multiplier we configured
     WRITE_NL                                  ; end the ini line
; Whether the env override was consulted, and which priority won
     cmp dword [rel kp_env_seen], 0            ; was the env var present?
     jne .thr_env_applied                      ; yes, print the applied line
     WRITE_STR lbl_thr_env_none                ; "env override: none"
     WRITE_NL                                  ; end the none line
     jmp .thr_env_done                         ; rejoin after the applied branch
.thr_env_applied:
     WRITE_STR lbl_thr_env_applied             ; "env override: v7_override_prio applied, ini priority "
     mov r12d, [worker_priority]               ; dword load, the INI value
     WRITE_UINT64 r12                          ; the INI value the override replaced
     WRITE_STR lbl_thr_s_eff                   ; " -> effective "
     mov r12d, [scheduler_effective_priority]  ; dword load, the applied priority
     WRITE_UINT64 r12                          ; the applied OS priority
     WRITE_NL                                  ; end the env line
.thr_env_done:
; OS topology the decision saw
     WRITE_STR lbl_thr_os                      ; "[threads] os: glpiex_bytes="
     mov r12d, [debug_glpiex_length]           ; dword load, scan byte count
     WRITE_UINT64 r12                          ; GLPIEx byte count the scan consumed
     WRITE_STR lbl_thr_s_cores                 ; " cores="
     mov r12d, [core_map_count]                ; dword load, parsed core count
     WRITE_UINT64 r12                          ; physical cores the scan parsed
     WRITE_STR lbl_thr_s_pmask                 ; " process_mask="
     WRITE_HEX64 [process_affinity_mask]       ; qword mask, logical processors this process may use
     WRITE_STR lbl_thr_s_smask                 ; " system_mask="
     WRITE_HEX64 [system_affinity_mask]        ; qword mask, logical processors the machine has
     WRITE_NL                                  ; end the os line
; Per-core raw and eligible masks, caller core marked
     xor ebx, ebx                              ; core index
     lea r13, [rel core_map_raw]               ; raw masks base
     lea r15, [rel core_map_masks]             ; eligible masks base
.thr_core_loop:
     cmp ebx, [core_map_count]                 ; past last core?
     jge .thr_core_done                        ; done
     mov r12, [r13 + rbx*QWORD_SCALE]          ; raw mask, non-volatile holder
     WRITE_STR lbl_thr_core                    ; "[threads] core "
     WRITE_UINT64 rbx                          ; core index
     WRITE_STR lbl_thr_core_raw                ; ": raw=0x"
     WRITE_HEX64 r12                           ; the raw GLPIEx mask
     WRITE_STR lbl_thr_core_elig               ; " eligible=0x"
     mov r12, [r15 + rbx*QWORD_SCALE]          ; eligible mask
     WRITE_HEX64 r12                           ; the process-intersected mask
     test r12b, 1                              ; does this eligible core contain CPU 0?
     jz .thr_core_nl                           ; no, plain line
     WRITE_STR lbl_thr_core_caller             ; "  <- caller CPU 0"
.thr_core_nl:
     WRITE_NL                                  ; end the core line
     inc ebx                                   ; next core
     jmp .thr_core_loop                        ; continue the walk
.thr_core_done:
; Placement mode
     cmp dword [use_physical_cores], 0         ; physical mode?
     je .thr_mode_smt                          ; no, logical line
     WRITE_STR lbl_thr_mode_phys               ; physical mode line
     WRITE_NL                                  ; end the mode line
     jmp .thr_mode_done                        ; rejoin after the logical branch
.thr_mode_smt:
     WRITE_STR lbl_thr_mode_smt                ; logical mode line
     WRITE_NL                                  ; end the mode line
.thr_mode_done:
; Candidate capacity and the worker-count arithmetic
     WRITE_STR lbl_thr_cand                    ; "[threads] candidates: "
     mov r12d, [debug_worker_count_raw]        ; dword load, candidate count
     WRITE_UINT64 r12                          ; candidate masks the placement built
     WRITE_STR lbl_thr_s_cap                   ; " eligible capacity incl. caller: "
     mov r12d, [debug_r13_at_workers]          ; dword load, capacity count
     WRITE_UINT64 r12                          ; capacity including the caller
     WRITE_NL                                  ; end the candidates line
     WRITE_STR lbl_thr_decision                ; "[threads] decision: num_workers = capacity("
     WRITE_UINT64 r12                          ; capacity term of the arithmetic
     WRITE_STR lbl_thr_s_cs                    ; ") - cores_subtract("
     mov r12d, [cores_subtract]                ; dword load, subtracted term
     WRITE_UINT64 r12                          ; subtracted term of the arithmetic
     WRITE_STR lbl_thr_s_eq                    ; ") = "
     mov r12d, [num_workers]                   ; dword load, resulting count
     WRITE_UINT64 r12                          ; the resulting worker count
     WRITE_STR lbl_thr_s_plus1                 ; ", plus 1 caller = "
     inc r12d                                  ; participants = num_workers + 1, r12d already holds num_workers
     WRITE_UINT64 r12                          ; the participant count
     WRITE_STR lbl_thr_s_parts                 ; " participants"
     WRITE_NL                                  ; end the decision line
; Chunk count and its source
     WRITE_STR lbl_thr_chunks                  ; "[threads] chunks: num_chunks="
     mov r12d, [num_chunks]                    ; dword load, final chunk count
     WRITE_UINT64 r12                          ; the final chunk count
     cmp dword [rel num_chunks_source], 0      ; ini override or auto?
     jne .thr_chunks_ini                       ; ini: print what auto would have been
     WRITE_STR lbl_thr_chunks_auto             ; " (auto: (num_workers+1) * chunk_multiplier)"
     jmp .thr_chunks_max                       ; rejoin at the max suffix
.thr_chunks_ini:
     WRITE_STR lbl_thr_chunks_is_ini           ; " (ini override, auto would be "
     mov eax, [num_workers]                    ; load the worker count for the auto product
     inc eax                                   ; num_workers + 1
     mov ecx, [chunk_multiplier]               ; load the multiplier for the auto product
     imul rax, rcx                             ; rax = auto chunk count
     WRITE_UINT64 rax                          ; the auto value for comparison
     WRITE_STR lbl_thr_s_close                 ; ")"
.thr_chunks_max:
     WRITE_STR lbl_thr_s_max                   ; ", max="
     mov r12d, MAX_CHUNKS                      ; load the job-pool ceiling
     WRITE_UINT64 r12                          ; the job-pool ceiling
     WRITE_NL                                  ; end the chunks line
; Every created thread: handle, affinity, priority
     xor ebx, ebx                              ; worker index
     lea r13, [rel threads]                    ; handles base
     lea r15, [rel worker_aff]                 ; affinity base
.thr_created_loop:
     cmp ebx, r14d                             ; all created threads printed?
     jge .thr_created_done                     ; done
     mov r12, [r13 + rbx*QWORD_SCALE]          ; handle, non-volatile holder
     WRITE_STR lbl_thr_created_w               ; "[threads] created W"
     WRITE_UINT64 rbx                          ; worker index
     WRITE_STR lbl_thr_created_mid             ; ": handle=0x"
     WRITE_HEX64 r12                           ; the retained thread handle
     WRITE_STR lbl_thr_s_aff                   ; " aff=0x"
     mov r12, [r15 + rbx*QWORD_SCALE]          ; affinity mask
     WRITE_HEX64 r12                           ; the core the worker is pinned to
     WRITE_STR lbl_thr_s_prio                  ; " prio="
     mov r12d, [scheduler_effective_priority]  ; dword load, applied priority
     WRITE_UINT64 r12                          ; the applied OS priority
     WRITE_NL                                  ; end the created line
     inc ebx                                   ; next worker
     jmp .thr_created_loop                     ; continue the walk
.thr_created_done:
; Startup poll result
     WRITE_STR lbl_thr_started_n               ; "[threads] workers started: "
     mov r12d, [workers_alive]                 ; dword load, live count
     WRITE_UINT64 r12                          ; live count at poll exit
     WRITE_STR lbl_thr_s_slash                 ; "/"
     WRITE_UINT64 r14                          ; created count, zero-extended by the mov r14d above
     WRITE_STR lbl_thr_s_budget                ; " (startup poll budget 2000 ms)"
     WRITE_NL                                  ; end the started line
; Caller capture and pin result
     WRITE_STR lbl_thr_caller                  ; "[threads] caller: tid="
     mov r12d, [caller_bound_tid]              ; dword load, OS thread id
     WRITE_UINT64 r12                          ; the pinned caller's OS thread id
     WRITE_STR lbl_thr_s_oprio                 ; " original_prio="
     mov r12d, [caller_saved_priority]         ; dword load, original priority
     WRITE_UINT64 r12                          ; the priority the caller came in with
     WRITE_STR lbl_thr_s_saff                  ; " saved_affinity="
     WRITE_HEX64 [saved_affinity]              ; qword mask, the affinity the caller came in with
     WRITE_STR lbl_thr_s_pinned                ; " -> pinned to CPU 0, prio="
     mov r12d, [scheduler_effective_priority]  ; dword load, applied priority
     WRITE_UINT64 r12                          ; the applied OS priority
     WRITE_NL                                  ; end the caller line
; Coverage: disjoint single-bit participants and the eligible count they cover
     mov r12d, 1                               ; union starts with the caller's bit 0
     xor ebx, ebx                              ; worker index, r15 still = worker_aff base
.thr_cov_loop:
     cmp ebx, [num_workers]                    ; all workers merged?
     jge .thr_cov_done                         ; done
     or r12, [r15 + rbx*QWORD_SCALE]           ; merge this worker's mask
     inc ebx                                   ; next worker
     jmp .thr_cov_loop                         ; continue the merge
.thr_cov_done:
     WRITE_STR lbl_thr_coverage                ; "[threads] coverage: "
     mov r13d, [num_workers]                   ; load the worker count for the +1
     inc r13d                                  ; participants = num_workers + 1
     WRITE_UINT64 r13                          ; the participant count
     WRITE_STR lbl_thr_s_cover                 ; " disjoint single-bit participants cover "
     popcnt r14, r12                           ; covered logical processors, BMI2 is in the CPU baseline
     WRITE_UINT64 r14                          ; the covered count
     WRITE_STR lbl_thr_s_elig_lp               ; " eligible logical processors"
     WRITE_NL                                  ; end the coverage line
; Runtime mode flags and the final effective priority
     WRITE_STR lbl_thr_runtime                 ; "[threads] runtime: always_spin="
     mov r12d, [always_spin]                   ; dword load, spin flag
     WRITE_UINT64 r12                          ; spin/sleep mode flag
     WRITE_STR lbl_thr_s_wms                   ; " workers_may_sleep="
     mov r12d, [workers_may_sleep]             ; dword load, registered sleepers
     WRITE_UINT64 r12                          ; workers registered in the wait-capable path
     WRITE_STR lbl_thr_s_seff                  ; " scheduler_effective_priority="
     mov r12d, [scheduler_effective_priority]  ; dword load, final priority
     WRITE_UINT64 r12                          ; the final applied priority
     WRITE_STR lbl_thr_s_maxw                  ; " max_workers="
     mov r12d, MAX_WORKERS                     ; load the pool size ceiling
     WRITE_UINT64 r12                          ; the pool size ceiling
     WRITE_NL                                  ; end the runtime line
     SF_END rbx, r12, r13, r14, r15            ; restore non-volatiles and return
.simple_report:
     WRITE_STR lbl_thr_simple                  ; "[threads] simple mode: init:simple_threads="
     mov r12d, [simple_threads]                ; dword load, requested simple worker count
     WRITE_UINT64 r12                          ; the N from the database
     WRITE_STR lbl_thr_s_simple                ; " simple workers (CreateThread + priority only, no pinning, no SMT placement), priority="
     mov r12d, [scheduler_effective_priority]  ; dword load, applied priority
     WRITE_UINT64 r12                          ; the priority every worker runs at
     WRITE_STR lbl_thr_s_nch                   ; " num_chunks="
     mov r12d, [num_chunks]                    ; dword load, final chunk count
     WRITE_UINT64 r12                          ; the chunk count the dispatchers use
     WRITE_NL                                  ; end the simple-mode line
     SF_END rbx, r12, r13, r14, r15            ; restore non-volatiles and return
%unmacro PF_TABLE_FILL 5                       ; remove the fill-loop macro, generic name, would outlive the file
%unmacro INI_KEY_FIND 0                        ; remove: used only by this file, its last caller, PF_TABLE_FILL, removed above
%unmacro INI_KEY_BEGIN 1                       ; remove: used only by this file
%unmacro INI_KEY_APPEND 1                      ; remove: used only by this file
%unmacro INI_LOAD_DWORD 3                      ; remove: expansion uses the three macros above
%undef GLPIEX_GROUP_MASK_OFF
%undef GLPIEX_GROUP_NUM_OFF
%undef GLPIEX_GROUP_COUNT_OFF
%undef GLPIEX_SIZE_OFF
%undef GLPIEX_REL_OFF
%undef MIN_GLPIEX_HEADER
%undef ATTN_SCALE_TABLE_SCALE
; ============================================================================
