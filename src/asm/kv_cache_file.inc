; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define KV_GENERIC_WRITE 0x40000000            ; GENERIC_WRITE
%define KV_GENERIC_READ 0x80000000             ; GENERIC_READ
%define KV_FILE_SHARE_READ 1                   ; FILE_SHARE_READ
%define KV_CREATE_ALWAYS 2                     ; CREATE_ALWAYS
%define KV_OPEN_EXISTING 3                     ; OPEN_EXISTING
%define KV_FILE_ATTR_NORMAL 0x80               ; FILE_ATTRIBUTE_NORMAL
%define KV_INVALID_HANDLE -1                   ; INVALID_HANDLE_VALUE
%define MDB_WRITE_CHUNK_CAP 0x7FFFFFFF         ; bytes per WriteFile/ReadFile call: half the 0xFFFFFFFF DWORD field max, so a chunk length stays positive
; --- File format constants ---
%define KV_HDR_RAW_OFF 0                       ; header: raw_bytes qword offset
%define KV_HDR_COMP_OFF 8                      ; header: comp_bytes qword offset
%define KV_HDR_POS_OFF 16                      ; header: pos dword offset
%define KV_HDR_MROPE_RUN_OFF 20                ; header: g_mrope_run dword offset, Qwen3-VL multimodal position counter
%define KV_HDR_MROPE_DELTA_OFF 24              ; header: g_mrope_delta dword offset, companion to MROPE_RUN; both are 0 for a text-only model
%define KV_HDR_FORMAT_OFF 28                   ; explicit KV storage marker
%define KV_HDR_FORMAT_F16 0x3631564B           ; ASCII KV16, little-endian
%define KV_HDR_STREAM_OFF 32                   ; header: StreamingLLM mode dword at save time
%define KV_HDR_SINKS_OFF 36                    ; header: sink rows S of the save-time config
%define KV_HDR_WINDOW_OFF 40                   ; header: rolling window W of the save-time config
%define KV_HDR_WIN_VALID_OFF 44                ; header: valid window rows at save, shrunk by rewinds
%define KV_HDR_READ_COUNT_OFF 48               ; bytes returned by ReadFile, separate from header
%define KV_HDR_SIZE 48                         ; total header bytes, model parameters are DERIVED from the loaded model's config at load time, the file is model-specific by design
%define KV_HDR_STACK_AREA 64                   ; 48 header bytes plus bytes-read slot and stack alignment, keeps RSP 16-aligned for the API calls made while the scratch window is live
; --- Compression worst-case margin, incompressible grows < 1/255 + 16; 1/128 + 1K remains safe ---
%define KV_COMP_SHIFT 7                        ; raw_bytes / 128 margin, still exceeds worst-case LZ4 growth
%define KV_COMP_PAD 1024                       ; fixed margin bytes
; ============================================================================
; .text
; ============================================================================
     section .text
     SF_START save_kv_cache, rbx, rsi, rdi, r12, r13, r14, r15 ; frame + gather cursors, emits global+label
     SF_QWORD kvs_filename                     ; filename pointer
     SF_QWORD kvs_file_handle                  ; CreateFileA handle
     SF_QWORD kvs_raw_buf                      ; gathered used-prefix blob
     SF_QWORD kvs_comp_buf                     ; compressed blob buffer
     SF_QWORD kvs_raw_bytes                    ; total raw blob size
     SF_QWORD kvs_layer_bytes                  ; used bytes per layer
     SF_QWORD kvs_layer_stride                 ; full layer stride in the cache
     SF_DWORD kvs_n_layers                     ; layer count
     SF_QWORD kvs_blob_cursor                  ; source cursor for the chunked blob write, frame slot: rdx is volatile across API_CALL
     SF_COMMIT                                 ; finalize frame
     ENGINE_GATE_SFV 1                         ; dead-check + landing zone, deadret=1: eax return
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     cmp qword [rel g_inf_ctx_block], 0        ; inference context alive?
     je .kvs_not_init                          ; no -> FATAL, contract violation
     mov rbx, [rel g_inf_ctx_block]            ; rbx = InferenceContext*
     mov kvs_filename, rcx                     ; stash filename, API calls overwrite volatiles
     mov rdx, [rbx + CTX_TRANSFORMER]          ; rdx = Transformer*
     mov eax, [rdx + CFG_N_LAYERS]             ; eax = n_layers
     mov kvs_n_layers, eax                     ; save layer count
     mov ecx, [rbx + CTX_POS]                  ; ecx = current position
     test ecx, ecx                             ; position zero, nothing written ?
     jz .kvs_empty                             ; yes -> nothing to save, return 0
     mov r8d, [rdx + CFG_N_KV_HEADS]           ; r8 = n_kv_heads, 32-bit read zero-extends
     mov r9d, [rdx + CFG_HEAD_DIM]             ; r9 = head_dim, 32-bit read zero-extends
     imul r8, r9                               ; r8 = kv_dim = n_kv_heads * head_dim
     mov rax, rcx                              ; rax = pos, the header re-reads [rbx + CTX_POS] independently below
     cmp dword [rel g_stream_enable], 0        ; StreamingLLM enabled?
     je .kvs_rows_std                          ; legacy: gather the full used prefix
     mov r9d, [rel g_stream_cap]               ; r9d = C, r9 dead past the kv_dim imul, rdx carries Transformer*
     cmp rax, r9                               ; pos <= C: nothing wrapped
     jbe .kvs_rows_std                         ; keep the full used prefix
     mov rax, r9                               ; eff_rows = min,pos, C : the ring only occupies [0, C
.kvs_rows_std:
     imul rax, r8                              ; rax = eff_rows * kv_dim, floats per layer
     shl rax, 1                                ; FP16 used bytes per layer
     mov kvs_layer_bytes, rax                  ; save per-layer used bytes
     mov r10d, [rdx + CFG_SEQ_LEN]             ; r10d = seq_len
     imul r10, r8                              ; r10 = seq_len * kv_dim floats
     shl r10, 1                                ; FP16 full layer stride in bytes
     mov kvs_layer_stride, r10                 ; save layer stride
     mov r11d, kvs_n_layers                    ; r11 = n_layers
     imul r11, rax                             ; r11 = n_layers * layer_bytes, 2-operand form: the 3-op imm form overwrites the n_layers accumulator
     add r11, r11                              ; r11 = raw_bytes = that x2 halves, key + value
     mov kvs_raw_bytes, r11                    ; save total raw size
; --- create the output file FIRST, I/O errors abort before any allocation ---
     mov rcx, kvs_filename                     ; arg1: filename
     API_CALL [__imp_CreateFileA], rcx, KV_GENERIC_WRITE, 0, 0, KV_CREATE_ALWAYS, KV_FILE_ATTR_NORMAL, 0 ; rax = handle
     cmp rax, KV_INVALID_HANDLE                ; open failed?
     je .kvs_open_fail                         ; yes -> visible message, return 0
     mov kvs_file_handle, rax                  ; stash handle
     mov rcx, kvs_raw_bytes                    ; raw blob size
     call mdb_alloc_mem                        ; rax = raw buffer, FATALs on OOM
     mov kvs_raw_buf, rax                      ; stash raw buffer
     mov rcx, kvs_raw_bytes                    ; compressed bound = raw + raw/128 + 1024
     shr rcx, KV_COMP_SHIFT                    ; rcx = raw/128 margin
     add rcx, kvs_raw_bytes                    ; + raw
     add rcx, KV_COMP_PAD                      ; + fixed pad
     call mdb_alloc_mem                        ; rax = compress buffer
     mov kvs_comp_buf, rax                     ; stash compress buffer
; --- gather the USED prefix: key rows first, then value rows ---
     mov rdx, [rbx + CTX_TRANSFORMER]          ; rdx = Transformer*
     lea rdx, [rdx + TRANSFORMER_STATE]        ; rdx = &EMBEDDED RunState, inline struct, NOT a pointer
     mov r12d, kvs_n_layers                    ; r12 = layer counter
     mov r13, kvs_layer_stride                 ; r13 = full layer stride
     mov r14, kvs_layer_bytes                  ; r14 = bytes to take per layer
     mov rdi, kvs_raw_buf                      ; rdi = destination cursor
     mov rsi, [rdx + S_KEY_CACHE]              ; rsi = key_cache base
     mov r15, r13                              ; r15 = tail gap = stride - used, hoisted once; r15 is free until the compress result lands
     sub r15, r14                              ; r15 = per-layer advance past the unused tail
.kvs_gather_key:
     mov rcx, r14                              ; bytes for this layer's used rows
     rep movsb                                 ; copy the layer's used rows into the blob
     add rsi, r15                              ; advance to the next layer start, stride - used tail gap
     dec r12                                   ; one layer done
     jnz .kvs_gather_key                       ; loop until all key layers gathered
     mov r12d, kvs_n_layers                    ; reset layer counter for the value half
     mov rsi, [rdx + S_VALUE_CACHE]            ; rsi = value_cache base
.kvs_gather_val:
     mov rcx, r14                              ; bytes for this layer's used rows
     rep movsb                                 ; copy the layer's used rows into the blob
     add rsi, r15                              ; advance to the next layer start, tail gap
     dec r12                                   ; one layer done
     jnz .kvs_gather_val                       ; loop until all value layers gathered
; --- compress the gathered blob ---
     mov rcx, kvs_raw_buf                      ; arg1: source
     mov rdx, kvs_raw_bytes                    ; arg2: source length, 64-bit
     mov r8, kvs_comp_buf                      ; arg3: destination
     call mdb_compress                         ; rax = compressed size, 64-bit
     mov r15, rax                              ; r15 = comp_bytes, kept across the writes
; --- write the header: raw_bytes, comp_bytes, pos, mrope_run, mrope_delta, streaming geometry ---
     sub rsp, KV_HDR_STACK_AREA                ; header build area on the stack, 8-aligned
     mov rdx, kvs_raw_bytes                    ; raw byte count
     mov [rsp + KV_HDR_RAW_OFF], rdx           ; store raw_bytes
     mov [rsp + KV_HDR_COMP_OFF], r15          ; store comp_bytes
     mov eax, [rbx + CTX_POS]                  ; current position
     mov [rsp + KV_HDR_POS_OFF], eax           ; store pos
     mov eax, [rel g_mrope_run]                ; M-RoPE component counter, 0 for a text-only model
     mov [rsp + KV_HDR_MROPE_RUN_OFF], eax     ; store mrope_run
     mov eax, [rel g_mrope_delta]              ; M-RoPE generation position correction
     mov [rsp + KV_HDR_MROPE_DELTA_OFF], eax   ; store mrope_delta
     mov eax, [rel g_stream_enable]            ; StreamingLLM mode at save time
     mov [rsp + KV_HDR_STREAM_OFF], eax        ; store the streaming flag
     mov eax, [rel g_stream_sinks]             ; sink rows S of the save-time config
     mov [rsp + KV_HDR_SINKS_OFF], eax         ; store S
     mov eax, [rel g_stream_window]            ; rolling window W of the save-time config
     mov [rsp + KV_HDR_WINDOW_OFF], eax        ; store W
     mov eax, [rel g_stream_win_valid]         ; valid window rows at save, shrunk by rewinds
     mov [rsp + KV_HDR_WIN_VALID_OFF], eax     ; store the validity tracker
     mov dword [rsp + KV_HDR_FORMAT_OFF], KV_HDR_FORMAT_F16 ; store the KV16 storage marker
     mov rcx, kvs_file_handle                  ; arg1: handle
     mov rdx, rsp                              ; arg2: header address
     mov r8d, KV_HDR_SIZE                      ; arg3: header byte count
     API_CALL [__imp_WriteFile], rcx, rdx, r8, 0, 0 ; write header, bytes-written discarded per the no-I/O-check contract
     mov rdx, kvs_comp_buf                     ; arg2: blob cursor
     mov kvs_blob_cursor, rdx                  ; park the blob source cursor, rdx is volatile across API_CALL
; r15 = comp_bytes = remaining, already loaded; r15 is non-volatile, so it DOES survive the calls
; KV_BLOB_IO api, handle_slot, cursor_slot -- move r15 remaining bytes between
; the blob and the file in <=2GB chunks. IN: r15 = remaining, handle/cursor
; slots hold the handle and the parked blob cursor. Clobbers rcx/rdx/r8.

%macro KV_BLOB_IO 3
%%loop:
     mov r8d, MDB_WRITE_CHUNK_CAP              ; cap at 2GB per transfer, dword imm, zero-extends
     cmp r15, r8                               ; remaining <= 2GB?
     jbe %%last                                ; yes, this is the final chunk
     mov rcx, %2                               ; arg1: handle
     mov rdx, %3                               ; arg2: blob cursor, reloaded, volatile across the call
     API_CALL %1, rcx, rdx, r8, 0, 0           ; chunk transfer, transfer assumed complete
     add qword %3, MDB_WRITE_CHUNK_CAP         ; advance blob cursor past the chunk
     sub r15, MDB_WRITE_CHUNK_CAP              ; remaining -= chunk size
     jmp %%loop                                ; next chunk
%%last:
     mov r8, r15                               ; r8 = remaining bytes, final chunk size <= 2GB, overwritten the loop cap
     mov rcx, %2                               ; arg1: handle
     mov rdx, %3                               ; arg2: blob cursor
     API_CALL %1, rcx, rdx, r8, 0, 0           ; final chunk, transfer assumed complete
%endmacro
.kvs_blob_write_loop:
     KV_BLOB_IO [__imp_WriteFile], kvs_file_handle, kvs_blob_cursor ; the whole chunked save blob
     add rsp, KV_HDR_STACK_AREA                ; release header area
; --- close, free, leave, return 1 ---
     mov rcx, kvs_file_handle                  ; arg1: handle
     API_CALL [__imp_CloseHandle], rcx         ; close the file
     mov rcx, kvs_comp_buf                     ; compress buffer
     call mdb_free_mem                         ; free it
     mov rcx, kvs_raw_buf                      ; raw buffer
     call mdb_free_mem                         ; free it
     ENGINE_LEAVE                              ; park the landing zone, normal path
     mov eax, 1                                ; return 1: saved
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore the 7 pushed non-volatiles, unwind the frame, return
.kvs_open_fail:
     WRITE_STR kv_err_open                     ; report the failed file creation to the console
     ENGINE_LEAVE                              ; park the landing zone
     xor eax, eax                              ; return 0: I/O error
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore the 7 pushed non-volatiles, unwind the frame, return
.kvs_empty:
     ENGINE_LEAVE                              ; park the landing zone, normal path
     xor eax, eax                              ; return 0: pos == 0, nothing to save
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore the 7 pushed non-volatiles, unwind the frame, return
.kvs_not_init:
     FATAL "public API: save_kv_cache called before v7_initialize_inference" ; contract violation
     SF_START load_kv_cache, rbx, rsi, rdi, r12, r13, r14, r15 ; frame + scatter cursors, emits global+label
     SF_QWORD kvl_file_handle                  ; CreateFileA handle
     SF_QWORD kvl_comp_buf                     ; compressed blob buffer
     SF_QWORD kvl_raw_buf                      ; decompressed blob buffer
     SF_QWORD kvl_raw_bytes                    ; raw blob size from the header
     SF_QWORD kvl_comp_bytes                   ; compressed size from the header
     SF_QWORD kvl_layer_bytes                  ; used bytes per layer
     SF_QWORD kvl_layer_stride                 ; full layer stride in the cache
     SF_DWORD kvl_pos                          ; position from the header
     SF_DWORD kvl_mrope_run                    ; g_mrope_run from the header, M-RoPE component counter
     SF_DWORD kvl_mrope_delta                  ; g_mrope_delta from the header, seq-index correction for generation RoPE
     SF_DWORD kvl_sinks                        ; S from the header, save-time sink rows
     SF_DWORD kvl_window                       ; W from the header, save-time rolling window
     SF_DWORD kvl_win_valid                    ; win_valid from the header, save-time validity tracker
     SF_DWORD kvl_n_layers                     ; layer count
     SF_QWORD kvl_blob_cursor                  ; destination cursor for the chunked blob read, frame slot: rdx is volatile across API_CALL
     SF_COMMIT                                 ; finalize frame
     ENGINE_GATE_SFV 1                         ; dead-check + landing zone, deadret=1: eax return
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     cmp qword [rel g_inf_ctx_block], 0        ; inference context alive?
     je .kvl_not_init                          ; no -> FATAL, contract violation
     mov rbx, [rel g_inf_ctx_block]            ; rbx = InferenceContext*
; --- open the file, missing file aborts before any allocation ---
     API_CALL [__imp_CreateFileA], rcx, KV_GENERIC_READ, KV_FILE_SHARE_READ, 0, KV_OPEN_EXISTING, KV_FILE_ATTR_NORMAL, 0 ; rax = handle
     cmp rax, KV_INVALID_HANDLE                ; open failed?
     je .kvl_io_fail                           ; yes -> visible message, return 0
     mov kvl_file_handle, rax                  ; stash handle
; --- read the header ---
     sub rsp, KV_HDR_STACK_AREA                ; header area on the stack, 8-aligned
     mov rcx, rax                              ; CreateFileA handle still in RAX
     mov rdx, rsp                              ; arg2: header address
     mov r8d, KV_HDR_SIZE                      ; arg3: header byte count
     lea r9, [rsp + KV_HDR_READ_COUNT_OFF]     ; record actual header bytes read
     API_CALL [__imp_ReadFile], rcx, rdx, r8, r9, 0 ; read header
     cmp dword [rsp + KV_HDR_READ_COUNT_OFF], KV_HDR_SIZE ; full current-format header?
     jne .kvl_header_read_fail                 ; anything shorter is an old-format or truncated cache, regenerate it
     cmp dword [rsp + KV_HDR_FORMAT_OFF], KV_HDR_FORMAT_F16 ; correct storage format?
     jne .kvl_header_bad_format                ; reject non-FP16 cache files
     mov rdx, [rsp + KV_HDR_RAW_OFF]           ; header raw_bytes
     mov kvl_raw_bytes, rdx                    ; save it
     mov rdx, [rsp + KV_HDR_COMP_OFF]          ; header comp_bytes
     mov kvl_comp_bytes, rdx                   ; save it
     mov eax, [rsp + KV_HDR_POS_OFF]           ; header pos
     mov kvl_pos, eax                          ; save it
     mov eax, [rsp + KV_HDR_MROPE_RUN_OFF]     ; header mrope_run
     mov kvl_mrope_run, eax                    ; save it
     mov eax, [rsp + KV_HDR_MROPE_DELTA_OFF]   ; header mrope_delta
     mov kvl_mrope_delta, eax                  ; save it
     mov eax, [rsp + KV_HDR_SINKS_OFF]         ; header sink rows S
     mov kvl_sinks, eax                        ; save it
     mov eax, [rsp + KV_HDR_WINDOW_OFF]        ; header rolling window W
     mov kvl_window, eax                       ; save it
     mov eax, [rsp + KV_HDR_WIN_VALID_OFF]     ; header validity tracker
     mov kvl_win_valid, eax                    ; save it
     add rsp, KV_HDR_STACK_AREA                ; release header area
; --- derive the scatter geometry from THIS model's config, the file was
; written by save_kv_cache from live state; its header is not re-validated ---
     mov rdx, [rbx + CTX_TRANSFORMER]          ; rdx = Transformer*
     mov eax, [rdx + CFG_N_LAYERS]             ; current n_layers
     mov kvl_n_layers, eax                     ; save layer count
     mov r8d, [rdx + CFG_N_KV_HEADS]           ; r8 = n_kv_heads, 32-bit read zero-extends
     mov r9d, [rdx + CFG_HEAD_DIM]             ; r9 = head_dim, 32-bit read zero-extends
     imul r8, r9                               ; r8 = kv_dim = n_kv_heads * head_dim, this model
; --- per-layer blob size from the header's raw_bytes, never assumed from pos:
; the writer gathered min,pos, C rows per layer under StreamingLLM wrapping, so
; the row count is recovered from the recorded size the blob actually holds ---
     mov rax, kvl_raw_bytes                    ; rax = total blob bytes, header, writer-exact
     test al, 1                                ; two equal halves required
     jnz .kvl_bad_geometry                     ; odd byte count cannot split into K|V
     shr rax, 1                                ; bytes per tensor half, all keys | all values
     mov ecx, kvl_n_layers                     ; layer count, dword slot, zero-extends
     xor edx, edx                              ; clear the dividend high word
     div rcx                                   ; raw half / layer count
     test rdx, rdx                             ; reject mismatched/truncated geometry
     jnz .kvl_bad_geometry                     ; raw half not a whole multiple of the layer count
     mov kvl_layer_bytes, rax                  ; save per-layer used bytes
     mov rdx, [rbx + CTX_TRANSFORMER]          ; reload Transformer*, div overwritten rdx
     mov r11d, [rdx + CFG_SEQ_LEN]             ; r11d = seq_len
     imul r11, r8                              ; r11 = seq_len * kv_dim floats
     shl r11, 1                                ; FP16 full layer stride
     mov kvl_layer_stride, r11                 ; save stride
     cmp qword kvl_layer_bytes, r11            ; used bytes must fit in one layer
     ja .kvl_bad_geometry                      ; per-layer used bytes exceed the full layer stride
     lea rcx, [r8 + r8]                        ; bytes per FP16 row = kv_dim*2
     xor edx, edx                              ; clear the dividend high word
     div rcx                                   ; per-layer bytes / row bytes = saved rows in rax
     test rdx, rdx                             ; no partial FP16 rows
     jnz .kvl_bad_geometry                     ; used bytes are not a whole number of FP16 rows
; --- the saved row count must match the current streaming configuration, or the
; restored position would address cache slots the file never populated ---
     mov r10, rax                              ; r10 = saved_rows, the file's physical row count per layer
     mov rdx, [rbx + CTX_TRANSFORMER]          ; rdx = Transformer*, div overwritten rdx
     mov r11d, kvl_pos                         ; r11 = saved absolute position
     cmp dword [rel g_stream_enable], 0        ; StreamingLLM currently enabled?
     je .kvl_check_linear                      ; no -> the file must be a full linear prefix
     mov ecx, [rel g_stream_cap]               ; ecx = C, current ring capacity
     cmp r11, rcx                              ; pos vs ring capacity C
     ja .kvl_expect_rows                       ; pos > C: ring wrapped at save time, expected rows = C
; a wrapped file scatters SLOT-ordered rows and the slot mapping is a function
; of S and W: an equal C with a different S/W split passes the row-count check
; yet scatters the rows to wrong ring slots, so the save-time split must equal
; the live split before any scatter happens
     mov eax, kvl_sinks                        ; the save-time S
     cmp eax, [rel g_stream_sinks]             ; save-time S vs live S
     jne .kvl_bad_geometry                     ; different sink boundary -> different slot mapping
     mov eax, kvl_window                       ; the save-time W
     cmp eax, [rel g_stream_window]            ; save-time W vs live W
     jne .kvl_bad_geometry                     ; different window width -> different slot mapping
; fall through: pos <= C, nothing wrapped -> .kvl_expect_linear
.kvl_expect_linear:
     mov ecx, r11d                             ; expected_rows = pos
     jmp .kvl_expect_rows                      ; compare the saved row count against pos
.kvl_check_linear:
     mov ecx, [rdx + CFG_SEQ_LEN]              ; ecx = seq_len, non-streaming bound
     cmp r11, rcx                              ; pos must fit the linear cache
     ja .kvl_bad_geometry                      ; a wrapped cache loaded without streaming -> reject
     mov ecx, r11d                             ; expected_rows = pos
.kvl_expect_rows:
     cmp r10, rcx                              ; saved rows == expected rows for this mode?
     jne .kvl_bad_geometry                     ; mismatch -> reject before allocating or scattering
; --- allocate, read the blob ---
     mov rcx, kvl_comp_bytes                   ; compressed size
     call mdb_alloc_mem                        ; rax = compress buffer
     mov kvl_comp_buf, rax                     ; stash it
     mov rcx, kvl_raw_bytes                    ; raw size
     call mdb_alloc_mem                        ; rax = raw buffer
     mov kvl_raw_buf, rax                      ; stash it
; --- read the compressed blob in <=2GB chunks, ReadFile takes a DWORD
; length; comp_bytes > 4GB would silently truncate to its low 32 bits ---
     mov rdx, kvl_comp_buf                     ; arg2: blob cursor
     mov kvl_blob_cursor, rdx                  ; park the blob destination cursor, rdx is volatile across API_CALL
     mov r15, kvl_comp_bytes                   ; r15 = remaining bytes
.kvl_blob_read_loop:
     KV_BLOB_IO [__imp_ReadFile], kvl_file_handle, kvl_blob_cursor ; the whole chunked load blob
     mov rcx, kvl_file_handle                  ; arg1: handle
     API_CALL [__imp_CloseHandle], rcx         ; done with the file, closed from here on
; --- decompress ---
     mov rcx, kvl_comp_buf                     ; arg1: source
     mov rdx, kvl_comp_bytes                   ; arg2: source length
     mov r8, kvl_raw_buf                       ; arg3: destination
     mov r9, kvl_raw_bytes                     ; arg4: max_dst
     call mdb_decompress                       ; rax = decompressed size, FATALs on a corrupt stream
     cmp rax, kvl_raw_bytes                    ; decompressor must yield exactly the header raw size
     jne .kvl_decompress_fail                  ; short/garbled output -> reject the cache
; --- scatter the rows back into the live caches, key half, then value half ---
     mov rdx, [rbx + CTX_TRANSFORMER]          ; rdx = Transformer*
     lea rdx, [rdx + TRANSFORMER_STATE]        ; rdx = &EMBEDDED RunState, inline struct, NOT a pointer
     mov r12d, kvl_n_layers                    ; r12 = layer counter
     mov r13, kvl_layer_stride                 ; r13 = full layer stride
     mov r14, kvl_layer_bytes                  ; r14 = bytes to restore per layer
     mov rsi, kvl_raw_buf                      ; rsi = source cursor, blob
     mov rdi, [rdx + S_KEY_CACHE]              ; rdi = key_cache base
     mov r15, r13                              ; r15 = tail gap = stride - used, hoisted once; r15 unused elsewhere in load
     sub r15, r14                              ; r15 = per-layer advance past the untouched tail
.kvl_scatter_key:
     mov rcx, r14                              ; bytes for this layer's rows
     rep movsb                                 ; restore the layer's used rows
     add rdi, r15                              ; advance to the next layer start, stride - used tail gap
     dec r12                                   ; one layer done
     jnz .kvl_scatter_key                      ; loop until all key layers restored
     mov r12d, kvl_n_layers                    ; reset layer counter
     mov rdi, [rdx + S_VALUE_CACHE]            ; rdi = value_cache base
.kvl_scatter_val:
     mov rcx, r14                              ; bytes for this layer's rows
     rep movsb                                 ; restore the layer's used rows
     add rdi, r15                              ; advance to the next layer start, tail gap
     dec r12                                   ; one layer done
     jnz .kvl_scatter_val                      ; loop until all value layers restored
     mov eax, kvl_pos                          ; the saved position
     mov [rbx + CTX_POS], eax                  ; restore ctx->pos, the slots ARE written now
     mov eax, [rel g_stream_window]            ; W of the CURRENT config, stale after a disable, harmless:
     mov [rel g_stream_remove_warm], eax       ; the latch is only consulted when g_stream_enable is 1, and a
; fresh enable re-zeroes it
; the validity tracker follows the file: a wrapped load replays the save-time
; tracker verbatim so a rewind-shrunk window stays shrunk across the roundtrip,
; a linear load starts canonical because nothing wrapped at save and the fill
; rewrites [pos, C before the first wrapped read
     mov ecx, [rel g_stream_cap]               ; live ring capacity C
     cmp dword kvl_pos, ecx                    ; the saved position wrapped the ring?
     jbe .kvl_tracker_canonical                ; linear prefix: canonical tracker
     mov eax, kvl_win_valid                    ; the save-time validity tracker
     jmp .kvl_tracker_store                    ; wrapped file: replay the save-time tracker verbatim
.kvl_tracker_canonical:
     mov eax, [rel g_stream_window]            ; W of the live config
.kvl_tracker_store:
     mov [rel g_stream_win_valid], eax         ; publish the tracker for the loaded layout
     mov dword [rel g_stream_otk_live], 0      ; the file records no token ids: out_tokens is garbage until a fresh chat refills it, the tier-2 rewrite refuses
     mov eax, kvl_mrope_run                    ; the saved M-RoPE component counter
     mov [rel g_mrope_run], eax                ; restore it, published with pos so the load stays all-or-nothing
     mov eax, kvl_mrope_delta                  ; the saved M-RoPE position correction
     mov [rel g_mrope_delta], eax              ; restore it alongside run
; --- free, leave, return 1 ---
     mov rcx, kvl_raw_buf                      ; raw buffer
     call mdb_free_mem                         ; free it
     mov rcx, kvl_comp_buf                     ; compress buffer
     call mdb_free_mem                         ; free it
     ENGINE_LEAVE                              ; park the landing zone, normal path
     mov eax, 1                                ; return 1: loaded
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore the 7 pushed non-volatiles, unwind the frame, return
.kvl_io_fail:
     WRITE_STR kv_err_io                       ; report the I/O failure to the console
     ENGINE_LEAVE                              ; park the landing zone
     xor eax, eax                              ; return 0: I/O error
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore the 7 pushed non-volatiles, unwind the frame, return
.kvl_decompress_fail:
     WRITE_STR kv_err_decompress               ; report the decompressor size mismatch
     mov rcx, kvl_comp_buf                     ; compress buffer
     call mdb_free_mem                         ; free it
     mov rcx, kvl_raw_buf                      ; raw buffer
     call mdb_free_mem                         ; free it
     ENGINE_LEAVE                              ; park the landing zone
     xor eax, eax                              ; return 0: cache not loaded
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore the 7 pushed non-volatiles, unwind the frame, return
.kvl_header_read_fail:
     add rsp, KV_HDR_STACK_AREA                ; release the header scratch window
     mov rcx, kvl_file_handle                  ; arg1: handle to close
     API_CALL [__imp_CloseHandle], rcx         ; close the cache file
     jmp .kvl_io_fail                          ; report the generic I/O failure
.kvl_header_bad_format:
     add rsp, KV_HDR_STACK_AREA                ; release the header scratch window
     mov rcx, kvl_file_handle                  ; arg1: handle to close
     API_CALL [__imp_CloseHandle], rcx         ; close the cache file
     WRITE_STR kv_err_format                   ; report the incompatible cache format
     ENGINE_LEAVE                              ; park the landing zone
     xor eax, eax                              ; return 0: incompatible format
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore the 7 pushed non-volatiles, unwind the frame, return
.kvl_bad_geometry:
     mov rcx, kvl_file_handle                  ; arg1: handle to close
     API_CALL [__imp_CloseHandle], rcx         ; close the cache file
     WRITE_STR kv_err_geometry                 ; report the incompatible geometry
     ENGINE_LEAVE                              ; park the landing zone
     xor eax, eax                              ; return 0: incompatible geometry
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore the 7 pushed non-volatiles, unwind the frame, return
.kvl_not_init:
     FATAL "public API: load_kv_cache called before v7_initialize_inference" ; contract violation
     section .rdata align=8
kv_err_geometry:
     db "[kv] Cache file dimensions are incompatible with this model.", 10, 0 ; loader rejected packed row geometry
kv_err_format:
     db "[kv] Incompatible cache file. Expected FP16 KV16 format.", 10, 0 ; loader rejected a non-FP16 cache file
kv_err_io:
     db "[kv] file I/O failed", 10, 0          ; generic open/read/write error
kv_err_decompress:
     db "[kv] cache payload did not decompress to the recorded size", 10, 0 ; compressed blob yields the wrong raw byte count
kv_err_open:
     db "[kv] save: cannot create the cache file", 10, 0 ; CreateFileA returned INVALID_HANDLE_VALUE
     section .text                             ; restore .text, an included file must leave the section it borrowed, house convention, same as every other include that borrows .rdata
; ---- %undef cleanup: file-local defines leave scope here, house convention ----
%undef KV_GENERIC_WRITE
%undef KV_GENERIC_READ
%undef KV_FILE_SHARE_READ
%undef KV_CREATE_ALWAYS
%undef KV_OPEN_EXISTING
%undef KV_FILE_ATTR_NORMAL
%undef KV_INVALID_HANDLE
%undef KV_HDR_RAW_OFF
%undef KV_HDR_COMP_OFF
%undef KV_HDR_POS_OFF
%undef KV_HDR_MROPE_RUN_OFF
%undef KV_HDR_MROPE_DELTA_OFF
%undef KV_HDR_STREAM_OFF
%undef KV_HDR_SINKS_OFF
%undef KV_HDR_WINDOW_OFF
%undef KV_HDR_WIN_VALID_OFF
%undef KV_HDR_READ_COUNT_OFF
%undef KV_HDR_FORMAT_F16
%undef KV_HDR_FORMAT_OFF
%undef KV_HDR_SIZE
%undef KV_HDR_STACK_AREA
%undef KV_COMP_SHIFT
%undef KV_COMP_PAD
%undef MDB_WRITE_CHUNK_CAP                     ; bytes per chunked WriteFile/ReadFile call
; ---- SF_* slot cleanup, house convention: slot names leave scope here ----
%undef kvs_filename
%undef kvs_filename_raw
%undef kvs_file_handle
%undef kvs_file_handle_raw
%undef kvs_raw_buf
%undef kvs_raw_buf_raw
%undef kvs_comp_buf
%undef kvs_comp_buf_raw
%undef kvs_raw_bytes
%undef kvs_raw_bytes_raw
%undef kvs_layer_bytes
%undef kvs_layer_bytes_raw
%undef kvs_layer_stride
%undef kvs_layer_stride_raw
%undef kvs_n_layers
%undef kvs_n_layers_raw
%undef kvs_blob_cursor
%undef kvs_blob_cursor_raw
%undef kvl_file_handle
%undef kvl_file_handle_raw
%undef kvl_comp_buf
%undef kvl_comp_buf_raw
%undef kvl_raw_buf
%undef kvl_raw_buf_raw
%undef kvl_raw_bytes
%undef kvl_raw_bytes_raw
%undef kvl_comp_bytes
%undef kvl_comp_bytes_raw
%undef kvl_layer_bytes
%undef kvl_layer_bytes_raw
%undef kvl_layer_stride
%undef kvl_layer_stride_raw
%undef kvl_pos
%undef kvl_pos_raw
%undef kvl_mrope_run
%undef kvl_mrope_run_raw
%undef kvl_mrope_delta
%undef kvl_mrope_delta_raw
%undef kvl_n_layers
%undef kvl_n_layers_raw
%undef kvl_blob_cursor
%undef kvl_blob_cursor_raw
