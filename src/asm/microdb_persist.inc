; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; NUMOFFSET compliance: named byte offsets for string copy and tmp extension
%define DBS_BACKSTEP_OFF 1                     ; [rsi - N] checks byte just copied by movsb
%define DBS_EXT_NUL_OFF 4                      ; byte offset for null terminator in ".tmp"
%define DBS_TMP_LAST 515                       ; last dst index leaving room for ".tmp"+NUL, 521-5-1
db_save:
     SF_START db_save, rbx, rsi, rdi, r12, r13, r14, r15 ; establish stack frame, save non-volatiles
     SF_QWORD db_save_stg                      ; staging buffer base
     SF_QWORD db_save_comp                     ; compression buffer base
     SF_ARRAY db_save_hdrbuf, 24               ; 24-byte header/end-marker staging
     SF_QWORD db_save_entry                    ; entry pointer across flush
     SF_QWORD db_save_src                      ; source pointer across flush
     SF_QWORD db_save_loop_ctr                 ; loop counter saved across flush_stg_block, ecx/rcx
     SF_DWORD db_save_htcap                    ; hash table capacity
     SF_QWORD db_save_filename                 ; resolved filename pointer, from rcx or g_db_filename
     SF_DWORD db_save_wr_remaining             ; write-side entry-scan down-counter, survives call write_entry, which overwrites every volatile
     SF_COMMIT                                 ; finalize stack frame layout
     cmp dword [rel mdb_readonly], 0           ; nothing needs resolving or copying on readonly saves
     jne .readonly_skip                        ; preserve the existing success-return path
; Resolve filename: rcx = caller filename, NULL = use g_db_filename
     test rcx, rcx                             ; check if caller provided filename
     jnz .db_save_fn_ok                        ; non-NULL -> use as-is
     lea rcx, [rel g_db_filename]              ; fall back to global filename
.db_save_fn_ok:
     mov db_save_filename, rcx                 ; cache filename for later use
; Build tmp filename: g_db_tmp_filename = filename + ".tmp"
     lea rdi, [rel g_db_tmp_filename]          ; dst = BSS tmp filename buffer
     lea rax, [rel g_db_tmp_filename + DBS_TMP_LAST] ; last dst position that still leaves ".tmp"+NUL room
     mov rsi, rcx                              ; src = filename pointer
.db_save_copy_tmp:
     cmp rdi, rax                              ; dst past last legal position?
     ja .save_tmp_fatal                        ; yes, filename would overflow buffer
     movsb                                     ; copy one byte src->dst
     cmp byte [rsi - DBS_BACKSTEP_OFF], 0      ; was it the null terminator?
     jnz .db_save_copy_tmp                     ; no, continue copying
     dec rdi                                   ; back to null terminator position
     mov dword [rdi], 0x706D742E               ; ".tmp", 4 bytes: 2E 74 6D 70 in little-endian, replaces 4 byte stores
     mov byte [rdi + DBS_EXT_NUL_OFF], 0       ; null-terminate the .tmp string
     jmp .db_save_tmp_done                     ; copy complete
.save_tmp_fatal:
     FATAL "db_save: filename too long for .tmp suffix" ; would overflow g_db_tmp_filename BSS
.db_save_tmp_done:
; --- Phase 1: count persistent entries ---
     lea r10, [rel g_ht]                       ; r10 = pointer to hash table struct
     mov r15, [r10 + HT_TABLE]                 ; r15 = hash table entry array base
     test r15, r15                             ; entry array exists?
     jz .early_tbl                             ; no table: ht_init never ran, FATAL
     mov eax, [r10 + HT_CAP]                   ; load hash table capacity
     mov db_save_htcap, eax                    ; cache capacity for loop bound
     xor r12d, r12d                            ; persistent count
     mov ecx, eax                              ; ecx = slot down-counter, cap >= 512: ht_init min, never 0 with a live table
     mov rsi, r15                              ; rsi = slot 0, running pointer, advance replaces per-slot lea+shl address math
.scan_body:
     test dword [rsi + E_FLAG], MDB_FLAG_PERSISTENT ; persistent entry?, empty slots are zero-filled so the flag test alone is exact
     jz .scan_next                             ; skip non-persistent
     inc r12d                                  ; count persistent entry
.scan_next:
     add rsi, ENTRY_SZ                         ; advance running slot pointer to next entry
     dec ecx                                   ; one fewer slot to scan
     jnz .scan_body                            ; loop while slots remain
; falls through to .scan_done, ecx hit 0 after exactly cap iterations
.scan_done:
     test r12d, r12d                           ; any persistent entries?
     jnz .go                                   ; yes, proceed to save
; 0 persistent entries, the database is obsolete: delete the stale file
; an absent file deletes nothing, the ordinary first-run case
     mov rcx, db_save_filename                 ; rcx -> stale DB filename
     API_CALL [__imp_DeleteFileA], rcx         ; delete orphaned lz4ht.db
.empty_done:
     mov eax, 1                                ; return success, nothing to save
     jmp .early_done                           ; skip directly to return
.readonly_skip:
     mov eax, 1                                ; return success, skip save
     jmp .early_done                           ; early exit, readonly mode
; Phase 2: allocate staging + comp buffers --
.go:
     mov ecx, BLOCK_SIZE + COMP_BUF_SIZE       ; one allocation for both staging areas
     call mdb_alloc_mem                        ; stage base owns the allocation
     mov db_save_stg, rax                      ; uncompressed block starts at the base
     add rax, BLOCK_SIZE                       ; compressed output follows the staging block
     mov db_save_comp, rax                     ; interior output view
; Load pool base, stable, not overwritten by API calls
     lea r10, [rel g_ht]                       ; r10 -> hash table struct
     mov r13, [r10 + KP_BASE]                  ; r13 = pool base address
; Phase 3: open temp file, write 20-byte file header --
     lea rcx, [rel g_db_tmp_filename]          ; rcx -> temp filename
; create temp file for writing serialized DB
     API_CALL [__imp_CreateFileA], rcx, MDB_GENERIC_WRITE, 0, 0, MDB_CREATE_ALWAYS, MDB_FILE_ATTR_NORMAL, 0 ; create overwrite-temp file
     cmp rax, MDB_INVALID_HANDLE               ; file opened?
     je .save_create_fatal                     ; no, cannot serialize without the temp file
     mov r14, rax                              ; r14 = file handle
; Build 20-byte header into db_save_hdrbuf, SF_ARRAY, not overflowed SF_QWORD
     lea rdi, db_save_hdrbuf                   ; rdi = &hdrbuf, SF_ARRAY: lea for address
     mov dword [rdi + DB_HDR_MAGIC], 'v7db'    ; file magic: bytes 'v','7','d','b', little-endian dword 0x62643776
     mov dword [rdi + DB_HDR_FILE_VER], MDB_DB_FORMAT_VERSION ; file-format version the writer speaks
     mov dword [rdi + DB_HDR_CNT_OFF], r12d    ; persistent entry count
     mov rax, [rel g_ht + KP_USED]             ; rax = final pool bytes used, pre-alloc target for load
     mov qword [rdi + DB_HDR_POOL_SZ], rax     ; store pool size in file header
; Write 20-byte file header, one call: a local synchronous file completes each write fully
     lea rsi, db_save_hdrbuf                   ; rsi -> header source
     mov r8d, MDB_FILE_HDR_SIZE                ; r8d = header byte count
     API_CALL [__imp_WriteFile], r14, rsi, r8, 0, 0 ; write the header, bytes-written count discarded
.hdr_written:
; Phase 4: streaming entry serialization --
     xor ebx, ebx                              ; staging_pos = 0
     mov rdi, db_save_stg                      ; staging write pos = staging base
     mov r12, r15                              ; r12 = running slot pointer, SF-saved non-volatile: write_entry overwrites rax/rcx/rdx/rsi/rdi/r8/r11, so neither the pointer nor the counter can ride a volatile across the call
     mov eax, db_save_htcap                    ; eax = slot count, cap >= 512 with a live table, never 0
     mov db_save_wr_remaining, eax             ; remaining = cap, frame-slot down-counter, write_entry overwrites EVERY volatile incl. rcx, so a register counter is destroyed by the first call
.write_loop:
     test dword [r12 + E_FLAG], MDB_FLAG_PERSISTENT ; persistent entry?, empty slots are zero-filled so the flag test alone is exact
     jz .write_next                            ; skip non-persistent
; Save entry pointer for write_entry
     mov db_save_entry, r12                    ; stash entry pointer
     call write_entry                          ; serialize this entry, leaves rdi at the staging base
.write_next:
     add r12, ENTRY_SZ                         ; advance the running slot pointer
     dec dword db_save_wr_remaining            ; one fewer slot, frame slot: immune to write_entry's volatile overwrites
     jnz .write_loop                           ; more slots, continue
.write_done:
; Flush any remaining staging data
     test rbx, rbx                             ; any staging bytes left?
     jz .end_marker                            ; no, write end marker
     call flush_stg_block                      ; flush partial block, FATALs internally on disk error
.end_marker:
; Write end marker: two qwords of zero
     lea rdi, db_save_hdrbuf                   ; reuse hdrbuf for end marker, SF_ARRAY: lea
     mov qword [rdi], 0                        ; first qword = 0, end sentinel
     mov qword [rdi + DB_HDR_END_MARKER_QW], 0 ; second qword = 0
     lea rsi, db_save_hdrbuf                   ; rsi -> end marker source
     mov r8d, DB_CHUNK_HDR                     ; r8d = 16-byte end-marker chunk
     API_CALL [__imp_WriteFile], r14, rsi, r8, 0, 0 ; write the end marker, bytes-written count discarded
; Close file + atomic rename
     API_CALL [__imp_CloseHandle], r14         ; close temp file handle
     lea rcx, [rel g_db_tmp_filename]          ; rcx -> temp filename
     mov rdx, db_save_filename                 ; rdx -> target DB filename
; atomic rename: temp -> final, replace existing, a reader holding the target open makes this fail
     API_CALL [__imp_MoveFileExA], rcx, rdx, (MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ; rename temp to final DB
     test eax, eax                             ; rename succeeded?
     jnz .save_success                         ; yes, done
     FATAL "db_save: atomic rename to final DB failed"
.save_success:
     mov rcx, db_save_stg                      ; rcx -> staging buffer, owns both staging areas
     call mdb_free_mem                         ; free the combined staging allocation
     mov eax, 1                                ; return 1, success
     jmp .early_done                           ; jump to return
.early_tbl:
     FATAL "db_save: hash table not initialized (ht_init never ran)" ; caller-ordering bug: a "successful" no-op save would silently lose all data
.early_done:
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore callee-saved regs
.save_create_fatal:
     FATAL "db_save: cannot create temp DB file" ; unrecoverable, FATAL directly
write_entry:
     mov rsi, db_save_entry                    ; rsi = entry pointer
; Write 16-byte entry header --
; Ensure 16 bytes free in staging, flush if needed
     mov eax, BLOCK_SIZE                       ; eax = staging capacity, imm32 zero-extends: 5-byte form, same 64-bit value
     sub rax, rbx                              ; rax = free bytes
     cmp rax, SER_HDR_SIZE                     ; need 16 bytes?
     jae .we_hdr_room                          ; yes, enough room
     call flush_stg_block                      ; no, flush first, FATALs internally on disk error
     mov rdi, db_save_stg                      ; reload staging base, overwritten by flush
     mov rsi, db_save_entry                    ; reload entry pointer, overwritten by flush
.we_hdr_room:
     mov ecx, [rsi + E_KLEN]                   ; ecx = padded key length, kept live in ecx to the key loop, no reload needed
     mov dword [rdi + rbx], ecx                ; key_len, dword
     mov rax, [rsi + E_DLEN]                   ; load data length
     mov qword [rdi + rbx + SER_HDR_DATA_LEN], rax ; data_len, qword
     mov eax, [rsi + E_FLAG]                   ; load flags
     mov dword [rdi + rbx + SER_HDR_FLAGS], eax ; flags, dword
     add rbx, SER_HDR_SIZE                     ; staging_pos += 16
; Write key bytes --
; Source = pool + E_KEY offset
     mov rax, [rsi + E_KEY]                    ; load key pool offset
     add rax, r13                              ; rax = key source address
     mov rsi, rax                              ; key bytes to stage
     sub rsp, 8                                ; maintain the original call alignment inside the helper
     call .we_stage_bytes                      ; stage the zero-extended key length
     add rsp, 8                                ; restore write_entry's original stack depth
; Write data bytes --
; Reload entry pointer, may have been overwritten
     mov rsi, db_save_entry                    ; reload entry pointer
     mov rcx, [rsi + E_DLEN]                   ; rcx = data length
     mov rax, [rsi + E_DATA]                   ; load data pool offset
     add rax, r13                              ; rax = data source address
     mov rsi, rax                              ; rsi = data source for loop
.we_stage_bytes:
; Shared bounded staging copier. IN: rsi = source, rcx = remaining bytes,
; rdi = staging base, rbx = staging position. Leaves rdi at the staging base.
     test rcx, rcx                             ; any bytes to stage?
     jz .we_done                               ; no, return
.we_copy_loop:
     mov r8, rcx                               ; preserve the remaining byte count across rep movsb
     mov eax, BLOCK_SIZE                       ; staging capacity
     sub eax, ebx                              ; free bytes in this block
     cmp rax, rcx                              ; compare the capacity with the full 64-bit remaining length
     cmovae eax, ecx                           ; bytes to copy = min(free, remaining)
     mov ecx, eax                              ; REP byte count
     mov r11, rdi                              ; save staging base
     add rdi, rbx                              ; output position
     rep movsb                                 ; copy one bounded block
     mov rdi, r11                              ; restore staging base
     add rbx, rax                              ; advance staging position
     sub r8, rax                               ; bytes still to stage
     jz .we_done                               ; the whole field is staged
     mov db_save_loop_ctr, r8                  ; save the remaining byte count across flush
     mov db_save_src, rsi                      ; save source cursor across flush
     call flush_stg_block                      ; emit the full staging block
     mov rsi, db_save_src                      ; resume the source cursor
     mov rcx, db_save_loop_ctr                 ; resume the remaining byte count
     jmp .we_copy_loop                         ; a positive remainder is established above
.we_done:
     ret                                       ; the staging base is live in rdi
flush_stg_block:
; Compress: src = staging base, src_len = rbx, staging_pos
     mov rcx, db_save_stg                      ; src = staging base
     mov rdx, rbx                              ; src_len = staging_pos
     mov r8, db_save_comp                      ; dst = comp buffer
     add r8, DB_CHUNK_HDR                      ; dst + 16, skip header room
     call mdb_compress                         ; rax = compressed size, 64-bit; never 0 for valid input
; Write block header: [comp+0] = decomp_size, qword, [comp+8] = comp_size, qword
     mov r8, db_save_comp                      ; r8 -> comp buffer
     mov [r8], rbx                             ; header+0: decompressed size
     mov [r8 + DB_HDR_COMP_SZ], rax            ; header+8: compressed size
; Total bytes to write = comp_size + DB_CHUNK_HDR, bounded by COMP_BUF_SIZE: one WriteFile call
     lea r8, [rax + DB_CHUNK_HDR]              ; r8 = total write size
     mov rsi, db_save_comp                     ; source = comp buffer base
     API_CALL [__imp_WriteFile], r14, rsi, r8, 0, 0 ; write the compressed block + header, bytes-written count discarded
     xor ebx, ebx                              ; staging_pos = 0, reset
     mov rdi, db_save_stg                      ; reload staging base
     ret                                       ; return, rbx = 0 = staging empty
db_load:
     SF_START db_load, rbx, rsi, rdi, r12, r13, r14, r15 ; setup frame for db_load
     SF_QWORD dbload_read_buf                  ; read staging buffer base
     SF_QWORD dbload_accum                     ; accumulation buffer, for spanning entries
     SF_QWORD dbload_accum_pos                 ; bytes accumulated so far
     SF_QWORD dbload_accum_total               ; total bytes needed, klen + dlen + hdr
     SF_DWORD dbload_klen                      ; current entry key length, from header
     SF_DWORD dbload_bread                     ; ReadFile bytes read
     SF_DWORD dbload_hdr_partial               ; bytes of header saved from prev block, 0..15
     SF_ARRAY dbload_hdr_save, 16              ; partial header bytes saved across blocks
     SF_DWORD dbload_read_cnt                  ; bytes-read counter across API_CALL, rcx is API_CALL-overwritten, so it lives in a frame slot
     SF_QWORD dbload_comp_total                ; total comp_size held across API_CALL, r8 is API_CALL-overwritten, so it lives in a frame slot
     SF_QWORD dbload_fn_ptr                    ; resolved filename pointer, from rcx or g_db_filename
     SF_QWORD dbload_newpool                   ; pre-allocated pool base held across rep movsb + mdb_free_mem
     SF_QWORD dbload_newpool_cap               ; pre-allocated pool capacity held across mdb_alloc_mem
     SF_COMMIT                                 ; finalize stack frame allocation
; Resolve filename: rcx = caller filename, NULL = use g_db_filename
     test rcx, rcx                             ; check if caller provided filename
     jnz .db_load_fn_ok                        ; non-NULL -> use as-is
     lea rcx, [rel g_db_filename]              ; fall back to global filename
.db_load_fn_ok:
     mov dbload_fn_ptr, rcx                    ; cache filename for later use
     push rdi                                  ; save non-volatile rdi, STR_LEN overwrites it
     sub rsp, 40                               ; alignment: entry 8 mod 16, push -> 0, +40 keeps call sites at 0 mod 16
     call debug_ensure_std                     ; ensure stdout_handle is cached, valid shadow+alignment now
     lea rdx, [sz_dbload_msg]                  ; rdx = "db_load: file = " prefix string
     STR_LEN rdx, r8d                          ; r8d = prefix length
     test r8d, r8d                             ; prefix empty?
     jz .dl_print_path                         ; yes, skip prefix write
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     xor r9d, r9d                              ; arg4 = NULL: the bytes-written out-pointer is OPTIONAL and discarded here
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write prefix to stdout
.dl_print_path:
     mov rdx, dbload_fn_ptr                    ; rdx = resolved DB path pointer
     STR_LEN rdx, r8d                          ; r8d = path length
     test r8d, r8d                             ; empty path?
     jz .dl_print_nl                           ; yes, skip path write
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle, reload: API_CALL overwritten it
     xor r9d, r9d                              ; arg4 = NULL: optional out-pointer discarded
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write DB path to stdout
.dl_print_nl:
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle, reload
     lea rdx, [fmt_instr_nl]                   ; rdx = newline string
     mov r8d, 1                                ; r8d = length, 1 byte: newline
     xor r9d, r9d                              ; arg4 = NULL: optional out-pointer discarded
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write newline to stdout
     add rsp, 40                               ; deallocate alignment frame
     pop rdi                                   ; restore non-volatile rdi
; Idempotency guard: skip if the hash table is already populated.
     lea r10, [rel g_ht]                       ; r10 = &g_ht hash table struct
     cmp dword [r10 + HT_COUNT], 0             ; already loaded?, entry count is the only read of the field here
     jnz .already_loaded                       ; yes, skip re-init
     mov qword dbload_accum, 0                 ; no accumulation buffer yet, first-block check reads this
     mov dword dbload_hdr_partial, 0           ; no partial header, the chunk reader reads this before its first write
; Open file
     mov rcx, dbload_fn_ptr                    ; rcx = path to DB file
     API_CALL [__imp_CreateFileA], rcx, MDB_GENERIC_READ, MDB_FILE_SHARE_READ, 0, MDB_OPEN_EXISTING, MDB_FILE_ATTR_NORMAL, 0 ; open DB file read-only
     cmp rax, MDB_INVALID_HANDLE               ; handle valid?
     jne .dl_file_ok                           ; yes, continue
     API_CALL [__imp_GetLastError]             ; get error code
     cmp eax, WINERR_FILE_NOT_FOUND            ; ERROR_FILE_NOT_FOUND?
     je .no_file                               ; file doesn't exist, return 0
     FATAL "db_load: file exists but cannot be opened" ; other error, abort
.dl_file_ok:
     mov r14, rax                              ; r14 = file handle
     mov ecx, COMP_BUF_SIZE + BLOCK_SIZE       ; compressed input plus decompression output
     call mdb_alloc_mem                        ; allocate both staging areas
     mov dbload_read_buf, rax                  ; compressed input owns the allocation
     mov rdi, rax                              ; read buffer starts at the owning base
     lea r15, [rax + COMP_BUF_SIZE]            ; decompression output is disjoint
; Read and validate 20-byte file header --
     xor ecx, ecx                              ; bytes read
; DB_READ_EXACT_HDR total, done_label -- retrying ReadFile loop for one
; fixed-size header. IN: ecx = bytes read so far (0 on entry), rdi = read
; buffer, r14 = file handle. OUT: ecx = total, buffer holds the header.
; Zero progress FATALs via .dl_read_stalled; I/O is assumed complete.

%macro DB_READ_EXACT_HDR 2
%%read_hdr:
     cmp ecx, %1                               ; have full header?
     jae %2                                    ; yes, proceed
     mov dword dbload_read_cnt, ecx            ; save bytes-read counter, rcx is API_CALL-overwritten, so it lives in a frame slot
     mov r8d, %1                               ; r8d = total size
     sub r8d, ecx                              ; r8d = remaining bytes to read
     lea r11, [rdi + rcx]                      ; dest = read_buf + progress
     xor eax, eax                              ; zero bwritten
     mov dbload_bread, eax                     ; clear ReadFile output
     lea r10, dbload_bread                     ; r10 -> bytes-read output
     API_CALL [__imp_ReadFile], r14, r11, r8, r10, 0 ; read header chunk, file I/O is assumed to succeed completely
     mov eax, dbload_bread                     ; eax = bytes actually read
     test eax, eax                             ; zero bytes read, EOF, truncated file ?
     jz .dl_read_stalled                       ; no progress possible, abort, never spin
     mov ecx, dword dbload_read_cnt            ; restore bytes-read counter
     add rcx, rax                              ; bytes_read += just_read
     jmp %%read_hdr                            ; continue reading header
%endmacro
     DB_READ_EXACT_HDR MDB_FILE_HDR_SIZE, .dl_hdr_done ; the 20-byte file header
.dl_hdr_done:
     cmp dword [rdi + DB_HDR_MAGIC], 'v7db'    ; file magic: a v7 database always starts with 'v7db'
     jne .dl_bad_magic                         ; wrong magic -> not a database of ours, refuse
     cmp dword [rdi + DB_HDR_FILE_VER], MDB_DB_FORMAT_VERSION ; file-format version the writer spoke
     je .dl_ver_ok                             ; current format -> load normally
     push r14                                  ; park the file handle, WRITE_STR's interior API_CALLs overwrite volatile r14; the block-read loop below still needs it
     WRITE_STR db_load_ver_warn                ; foreign version -> load anyway, warn visibly, older/newer writer
     pop r14                                   ; restore the file handle
.dl_ver_ok:
     mov ebx, [rdi + DB_HDR_CNT_OFF]           ; ebx = total entries to load
     test ebx, ebx                             ; any entries?
     jz .dl_block_loop                         ; no, require mandatory end marker
     lea r10, [rel g_ht]                       ; r10 = &g_ht
     mov rax, [rdi + DB_HDR_POOL_SZ]           ; rax = stored pool size, EXACT allocation target: db_save wrote the final KP_USED, and the load rebuilds byte-identical padded key/data lengths
     add rax, KP_LIVE_RESERVE                  ; plus headroom so post-load V7-DB-PUT writes never relocate the pool, model-weight pointers are pool-derived
     cmp rax, [r10 + KP_CAP]                   ; current pool already big enough?
     jbe .dl_prealloc_done                     ; yes, nothing to do
     mov dbload_newpool_cap, rax               ; save new capacity across calls
     mov rcx, rax                              ; rcx = new pool capacity
     call mdb_alloc_mem                        ; allocate new pool, rax = base, always valid: allocator FATALs on OOM
     mov dbload_newpool, rax                   ; save new pool base across movsb/free
     lea r10, [rel g_ht]                       ; reload &g_ht, call overwritten r10
     mov rdi, rax                              ; rdi = dst = new pool
     mov rsi, [r10 + KP_BASE]                  ; rsi = old pool base, rep movsb advances it, reload below before free
     mov rcx, [r10 + KP_USED]                  ; rcx = bytes used in old pool
     rep movsb                                 ; copy existing pool contents to new pool
     mov rcx, [r10 + KP_BASE]                  ; rcx = old pool BASE, reload: rsi advanced past it
     call mdb_free_mem                         ; free old pool
     lea r10, [rel g_ht]                       ; reload &g_ht, call overwritten r10
     mov rax, dbload_newpool                   ; new pool base
     mov [r10 + KP_BASE], rax                  ; publish new pool base
     mov rax, dbload_newpool_cap               ; new pool capacity
     mov [r10 + KP_CAP], rax                   ; publish new pool capacity
; KP_USED unchanged, still counts the used bytes copied above
.dl_prealloc_done:
; Block reading loop ------------------------------------------------
.dl_block_loop:
; Reload read buffer base, rdi overwritten by rep movsb in accum/header paths
     mov rdi, dbload_read_buf                  ; rdi = read staging buffer, restored
; Reset read_pos for new block
     xor r12d, r12d                            ; read_pos = 0
; Read block header, 16 bytes into read buffer --
     xor ecx, ecx                              ; bytes_read = 0
.dl_read_block_hdr:
     DB_READ_EXACT_HDR DB_CHUNK_HDR, .dl_block_hdr_done ; the 16-byte block header
.dl_block_hdr_done:
; Check end marker, decomp_size == 0 = genuine end marker on our own files
     mov rax, [rdi]                            ; rax = decomp_size from chunk header
     test rax, rax                             ; decomp_size == 0?
     jz .dl_success                            ; yes, genuine end marker
.dl_have_block:
     mov r13, rax                              ; r13 = decomp_size, plain byte count of the current decompressed block
     cmp r13, BLOCK_SIZE                       ; decomp_size fits the decompression buffer?
     ja .dl_decomp_too_big                     ; no, corrupt file, refuse before any decompress
     mov r8, [rdi + DB_HDR_COMP_SZ]            ; r8 = comp_size, qword at hdr+8
     cmp r8, COMP_BUF_SIZE                     ; comp_size fits the read staging buffer?
     ja .dl_comp_too_big                       ; no, corrupt file, refuse before any read
; Read compressed data into read buffer --
; r8 holds total comp_size; remaining computed as r8-rcx for ReadFile
     xor ecx, ecx                              ; bytes read = 0
.dl_read_comp:
     cmp rcx, r8                               ; bytes_read >= total?
     jae .dl_comp_done                         ; all compressed data read
     mov dbload_comp_total, r8                 ; save total comp_size, r8 is API_CALL-overwritten, so it lives in a frame slot
     sub r8, rcx                               ; remaining = total - bytes_read
     mov dword dbload_read_cnt, ecx            ; save bytes-read counter, rcx is API_CALL-overwritten, so it lives in a frame slot
     lea r11, [rdi + rcx]                      ; dest = read_buf + bytes_read
     xor eax, eax                              ; zero bwritten
     mov dbload_bread, eax                     ; clear ReadFile output
     lea r10, dbload_bread                     ; r10 -> bytes-read output
     API_CALL [__imp_ReadFile], r14, r11, r8, r10, 0 ; read compressed chunk, file I/O is assumed to succeed completely
     mov eax, dbload_bread                     ; eax = bytes actually read
     test eax, eax                             ; zero bytes read, EOF, truncated file ?
     jz .dl_read_stalled                       ; no progress possible, abort, never spin
     mov r8, dbload_comp_total                 ; restore total comp_size, frame slot, see the save above
     mov ecx, dword dbload_read_cnt            ; restore bytes-read counter, zero-extends to rcx
     add rcx, rax                              ; bytes_read += just_read
     jmp .dl_read_comp                         ; continue reading compressed data
.dl_comp_done:
; Decompress --
     mov rcx, rdi                              ; src = read buffer, compressed
     mov rdx, r8                               ; src_len = comp_size, r8 already holds correct value from header
     mov r8, r15                               ; dst = decomp buffer
     mov r9, r13                               ; max_dst = decomp_size
     call mdb_decompress                       ; LZ77 decompress block, round-trip proven by v_mdb_validate
     cmp rax, r13                              ; decompressor produced the declared decomp_size?
     jne .dl_decomp_too_big                    ; short, or oversized output, corrupt stream, refuse
; Check if we have a partial header from previous block
     cmp dword dbload_hdr_partial, 0           ; partial header pending?
     jne .dl_complete_header                   ; yes, complete it first
; Handle accumulation resume, spanning entry from previous block --
     cmp qword dbload_accum, 0                 ; accumulation in progress?
     je .dl_walk_entries                       ; no spanning entry, walk normally
; Resume accumulating the spanning entry
; Copy min,available, decomp_available from decomp to accum buffer
     mov rax, r13                              ; decomp_end
     sub rax, r12                              ; available = decomp_end - read_pos
     mov rcx, dbload_accum_total               ; total bytes needed
     sub rcx, dbload_accum_pos                 ; remaining = total - accumulated
     cmp rax, rcx                              ; available <= remaining?
     jbe .dl_accum_copy_amount                 ; yes, copy all available
     mov rax, rcx                              ; copy = min,available, remaining
.dl_accum_copy_amount:
; Copy rax bytes from r15+r12, decomp to accum + accum_pos
     lea rsi, [r15 + r12]                      ; source = decomp buffer + read_pos
     mov rdi, dbload_accum                     ; dest = accum buffer
     add rdi, dbload_accum_pos                 ; dest += already accumulated
     mov rcx, rax                              ; byte count
     rep movsb                                 ; copy bytes to accumulation buffer, rax survives rep movsb
     mov r12, rax                              ; read_pos = bytes consumed from block
     add dbload_accum_pos, rax                 ; update accumulated count
; Check if entry is now complete
     mov rax, dbload_accum_pos                 ; bytes accumulated so far
     cmp rax, dbload_accum_total               ; all bytes accumulated?
     jb .dl_block_loop                         ; not complete yet, read next block
     mov rcx, dbload_accum                     ; rcx = accum base
     add rcx, SER_HDR_SIZE                     ; key = accum + 16, past serialized header
     movsxd rdx, dword dbload_klen             ; rdx = key_len, sign-extended
     add rdx, rcx                              ; data = key + klen = accum + 16 + klen
     mov r8, dbload_accum_total                ; r8 = total entry bytes, hdr + klen + dlen
     movsxd rax, dword dbload_klen             ; rax = key_len
     add rax, SER_HDR_SIZE                     ; total overhead = klen + 16
     sub r8, rax                               ; data_len = total - klen - 16
     mov r9d, 1                                ; persist=1 so entries survive db_save/db_load
     call ht_add                               ; insert into hash table, never returns 0, failures FATAL internally
; Free accumulation buffer
     mov rcx, dbload_accum                     ; rcx = accumulation buffer
     call mdb_free_mem                         ; release heap buffer
     mov qword dbload_accum, 0                 ; clear pointer, the .dl_walk_entries / .dl_complete_header split tests it
     dec ebx                                   ; entries_left--
; fall through to dl_walk_entries, r12 already set to bytes consumed
     jnz .dl_walk_entries                      ; more entries, continue walking
     jmp .dl_block_loop                        ; all entries loaded, require end marker
; Complete partial header from previous block -------------------
.dl_complete_header:
; dbload_hdr_partial has bytes already saved, 1..15
; Need: SER_HDR_SIZE - dbload_hdr_partial more bytes from new block
     mov ecx, dbload_hdr_partial               ; ecx = bytes already saved
     mov rax, SER_HDR_SIZE                     ; rax = full header size, 16
     sub rax, rcx                              ; bytes needed from new block
; Copy remaining header bytes from decomp[0..rax-1]
     mov rsi, r15                              ; source = start of new block
     lea rdi, dbload_hdr_save                  ; dest = partial header buffer
     add rdi, rcx                              ; dest += bytes already saved
     mov rcx, rax                              ; byte count
     rep movsb                                 ; complete the partial header
     mov r12, rax                              ; read_pos = bytes consumed from new block
     mov dword dbload_hdr_partial, 0           ; no more partial header
; Parse completed header from hdr_save
     lea rsi, dbload_hdr_save                  ; rsi = completed header
     mov eax, [rsi]                            ; key_len, dword
     mov dbload_klen, eax                      ; save key_len
     mov rsi, [rsi + SER_HDR_DATA_LEN]         ; data_len, qword
     movsxd rcx, eax                           ; key_len, sign-extended, eax still holds klen from the header read at line 524
     lea rax, [rsi + rcx]                      ; total payload = klen + dlen
     add rax, r12                              ; entry end = payload + read_pos
     cmp rax, r13                              ; entry end <= decomp_end?
     ja .dl_hdr_span_entry_spans               ; entry spans blocks
; DB_LOAD_INSERT_FITTING_ENTRY -- insert one entry that already fits in the
; decompressed block, then advance the walker. IN: r15 = decomp buf,
; r12 = read_pos, r13 = decomp_end, ebx = entries left, rsi = data_len,
; dbload_klen = key len. Clobbers rax/rcx/rdx/r9, ht_add's volatiles.

%macro DB_LOAD_INSERT_FITTING_ENTRY 0
; Entry fits, ht_add directly from decomp buffer, zero-copy
     lea rcx, [r15 + r12]                      ; key = decomp + read_pos
     movsxd rdx, dword dbload_klen             ; rdx = key_len, sign-extended
     add rdx, rcx                              ; data = key + klen
     mov r8, rsi                               ; data_len
     mov r9d, 1                                ; persist=1
     call ht_add                               ; insert into hash table, never returns 0, failures FATAL internally
; Advance read_pos past entry
     movsxd rax, dword dbload_klen             ; rax = key_len, sign-extended
     add r12, rax                              ; read_pos += klen
     add r12, rsi                              ; read_pos += dlen
     dec ebx                                   ; entries_left--
     jnz .dl_walk_entries                      ; more entries, continue
     jmp .dl_block_loop                        ; all entries loaded, require end marker
%endmacro
; Entry fits, ht_add directly from decomp buffer, zero-copy
     DB_LOAD_INSERT_FITTING_ENTRY              ; insert + walker advance
.dl_hdr_span_entry_spans:
; Entry that had header-span also spans payload, set up accumulation
; Total = SER_HDR_SIZE + klen + dlen
     movsxd rax, dword dbload_klen             ; rax = key_len, sign-extended
     lea rax, [rax + rsi + SER_HDR_SIZE]       ; klen + dlen + 16 for header, one instruction
     mov dbload_accum_total, rax               ; save total entry size
; Allocate accumulation buffer
     mov rcx, rax                              ; rcx = allocation size
     call mdb_alloc_mem                        ; allocate zeroed buffer
     mov dbload_accum, rax                     ; save accum buffer pointer
; Copy completed header to accum[0..15]
     lea rsi, dbload_hdr_save                  ; source = completed header
     mov rdi, rax                              ; allocation result still live in RAX
     mov rcx, SER_HDR_SIZE                     ; 16 bytes
     rep movsb                                 ; copy header to accum
; Copy as much key+data as fits from current block
; Available in block: decomp_end - r12, r12 = bytes already consumed for header
     mov rax, r13                              ; decomp_end
     sub rax, r12                              ; available payload bytes
; Copy rax bytes from decomp[r12] to accum[SER_HDR_SIZE + already_copied]
     lea rsi, [r15 + r12]                      ; source = decomp buffer + read_pos
     mov rdi, dbload_accum                     ; load heap pointer from stack slot
     add rdi, SER_HDR_SIZE                     ; dest = accum + header
     mov rcx, rax                              ; byte count
     rep movsb                                 ; copy payload to accum, rax survives rep movsb
     mov r12, r13                              ; read_pos = decomp_end, all bytes from this block consumed
     mov dbload_accum_pos, rax                 ; save accumulated byte count
     add qword dbload_accum_pos, SER_HDR_SIZE  ; accum_pos = header + payload copied
; Entry spans blocks, klen+dlen > available : accum_pos < accum_total ALWAYS
; strict inequality from the .dl_hdr_span_entry_spans entry condition .
     jmp .dl_block_loop                        ; entry spans, continue accumulating in next block
; Walk entries in current decompressed block -------------------------
; r15 = decomp buf, r12 = read_pos, r13 = decomp_size, rbx = entries left
.dl_walk_entries:
; Need at least 16 bytes for entry header
     lea rax, [r12 + SER_HDR_SIZE]             ; rax = read_pos + 16
     cmp rax, r13                              ; enough bytes for header?
     ja .dl_header_span                        ; no, header spans blocks
; Read full entry header
     mov ecx, [r15 + r12]                      ; ecx = key_len directly, positive, no sign-ext needed
     mov dbload_klen, ecx                      ; save, still in ecx for lea below
     mov rsi, [r15 + r12 + SER_HDR_DATA_LEN]   ; data_len, qword
     add r12, SER_HDR_SIZE                     ; advance past header
; Check if entire entry fits: read_pos + klen + data_len <= decomp_end
     lea rax, [rsi + rcx]                      ; total = data_len + key_len
     add rax, r12                              ; entry end
     cmp rax, r13                              ; fits in block?
     ja .dl_entry_spans                        ; no, payload spans blocks
; Entry fits entirely, ht_add directly from decomp buffer, zero-copy
     DB_LOAD_INSERT_FITTING_ENTRY              ; insert + walker advance
.dl_header_span:
; The entry header itself spans blocks.
; Save partial header bytes, complete on next block.
     mov rax, r13                              ; rax = bytes remaining in block
     sub rax, r12                              ; bytes available, 0..15; the 0 case = a block fully consumed by an accum resume
     mov dword dbload_hdr_partial, eax         ; save count
; Copy partial header to hdr_save
     lea rsi, [r15 + r12]                      ; source = decomp + read_pos
     lea rdi, dbload_hdr_save                  ; dest = partial header buffer
     mov rcx, rax                              ; byte count
     rep movsb                                 ; copy partial header bytes
     jmp .dl_block_loop                        ; read next block to complete header
.dl_entry_spans:
; Entry payload, key + data spans blocks, set up accumulation
; Calculate total entry size = 16, header + klen + dlen
     movsxd rax, dword dbload_klen             ; rax = key_len, sign-extended
     lea rax, [rax + rsi + SER_HDR_SIZE]       ; klen + dlen + 16 for header, one instruction
     mov dbload_accum_total, rax               ; total bytes to accumulate
; Allocate accumulation buffer
     mov rcx, rax                              ; rcx = allocation size
     call mdb_alloc_mem                        ; allocate zeroed buffer
     mov dbload_accum, rax                     ; save accum buffer pointer
; Copy what we have from current block, header + partial payload
; header_start = r12 - SER_HDR_SIZE, we advanced r12 past header
     mov rax, r13                              ; decomp_end
     lea rcx, [r12 - SER_HDR_SIZE]             ; header start position, one address-generation
     sub rax, rcx                              ; available bytes from header_start
; Copy rax bytes from decomp[header_start] to accum[0]
     lea rsi, [r15 + rcx]                      ; source = decomp + header_start
     mov rdi, dbload_accum                     ; dest = accum buffer
     mov rcx, rax                              ; byte count
     rep movsb                                 ; copy header + payload to accum
     mov dbload_accum_pos, rax                 ; bytes accumulated
     jmp .dl_block_loop                        ; read next block to continue
; Error / exit paths ------------------------------------------------
.dl_bad_magic:
     FATAL "db_load: not a v7 database file (missing 'v7db' magic)" ; wrong file, refuse instead of misparsing it
.dl_read_stalled:
     FATAL "db_load: ReadFile made no progress (failed or truncated database)" ; data-integrity failure, FATAL, never a hang
.dl_comp_too_big:
     FATAL "db_load: block comp_size exceeds the read staging buffer" ; not a file our writer can have produced
.dl_decomp_too_big:
     FATAL "db_load: block decomp_size exceeds the decompression buffer, or the decompressed size mismatches" ; corrupt geometry or stream, refuse
.no_file:
     xor eax, eax                              ; return 0, no db file found, expected variation, empty DB
     jmp .dl_exit                              ; skip to epilogue
.dl_success:
; dbload_accum is provably NULL here, a spanning entry always continues via .dl_block_loop
     API_CALL [__imp_CloseHandle], r14         ; close file handle before freeing buffers
     mov rcx, dbload_read_buf                  ; II-3/I-4: reload from stack, rdi overwritten by rep movsb
     call mdb_free_mem                         ; free raw read buffer
     mov eax, 1                                ; return 1, success
     jmp .dl_exit                              ; skip to epilogue
.already_loaded:
     mov eax, 1                                ; already loaded, return success
.dl_exit:
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatile regs
     section .data                             ; diagnostic strings must live in initialized data
sz_dbload_msg:
     db "db_load: file = ", 0                  ; prefix printed before the resolved DB path
db_load_ver_warn:
     db "db_load: warning: model file was made by a different version", 10, 0 ; foreign DB file-format version, loaded anyway
     section .text                             ; restore code section for subsequent includes
; %undef cleanup: file-local defines leave scope here, house convention --
%undef DBS_BACKSTEP_OFF
%undef DBS_EXT_NUL_OFF
%undef DBS_TMP_LAST
; SF slot names of this file's two functions, a leaked slot name would silently
; capture any later label/equ of the same text as a memory operand; each slot
; also carries the _raw alias SF_QWORD/SF_DWORD generate
%undef db_save_stg
%undef db_save_stg_raw
%undef db_save_comp
%undef db_save_comp_raw
%undef db_save_entry
%undef db_save_entry_raw
%undef db_save_src
%undef db_save_src_raw
%undef db_save_loop_ctr
%undef db_save_loop_ctr_raw
%undef db_save_htcap
%undef db_save_htcap_raw
%undef db_save_filename
%undef db_save_filename_raw
%undef db_save_wr_remaining
%undef db_save_wr_remaining_raw
%undef dbload_read_buf
%undef dbload_read_buf_raw
%undef dbload_accum
%undef dbload_accum_raw
%undef dbload_accum_pos
%undef dbload_accum_pos_raw
%undef dbload_accum_total
%undef dbload_accum_total_raw
%undef dbload_klen
%undef dbload_klen_raw
%undef dbload_bread
%undef dbload_bread_raw
%undef dbload_hdr_partial
%undef dbload_hdr_partial_raw
%undef dbload_read_cnt
%undef dbload_read_cnt_raw
%undef dbload_comp_total
%undef dbload_comp_total_raw
%undef dbload_fn_ptr
%undef dbload_fn_ptr_raw
%undef dbload_newpool
%undef dbload_newpool_raw
%undef dbload_newpool_cap
%undef dbload_newpool_cap_raw
