; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%include "inf_structs.inc"
     section .bss
     global enable_token_dump                  ; exported for external toggle
enable_token_dump:
     resd 1                                    ; runtime flag: 0=off, default, nonzero=on
     section .text
; Numeric constants for struct offsets and scales, NUMOFFSET compliance
%define TSD_VOCAB_SCALE 8                      ; sizeof,char* for vocab ptr array indexing
%define TSD_BUF_SIZE 256                       ; max piece length to print, truncated if longer
%define TSD_LIMIT_MARGIN 7                     ; dest-limit margin: r11 = base + 249, so the cursor is below base + 249 at every write and a 2-byte escape at base + 248 touches at most base + 249, max emitted length 250 of 256.The truncation marker is then written at rdi, at most base + 250 and ends by base + 253, so dest writes never exceed TSD_BUF_SIZE, leaving 2 free bytes of the 256.No NUL is written, the piece is emitted with an explicit byte count.
%define TSD_MARK_LEN 3                         ; emitted length of the "..." marker the dest-limit exit appends, stored as ONE 4-byte write, 3 counted, see the store at the marker site; the 4th byte lands past the count and is never written to stdout
; TSD_EMIT_ESC imm16, emit one 2-byte control-char escape and resume.
; %1 = the packed little-endian word to store, backslash in the low byte .

%macro TSD_EMIT_ESC 1
     mov word [rdi], %1                        ; write the 2-byte escape
     add rdi, 2                                ; advance dest by 2, lodsb already consumed the 1 source byte
     jmp .tsd_copy_loop                        ; continue
%endmacro
     global dump_token_stream                  ; export for external callers
dump_token_stream:
; Check runtime flag BEFORE frame setup, disabled path is a 2-instruction check + ret
     cmp dword [rel enable_token_dump], 0      ; token dump enabled?
     jz dump_token_stream_disabled             ; no, return with zero frame overhead
     SF_START tsd_frame, rsi, rdi, r12, r13, r14 ; save non-volatiles, frame name differs from entry label
     SF_ARRAY tsd_piece_buf, TSD_BUF_SIZE      ; stack buffer for sanitized piece text
     SF_BYTE tsd_truncated                     ; set on the dest-limit exit ONLY when the source had bytes left: what follows the marker is a clipped prefix, not the whole piece
     SF_COMMIT                                 ; finalize stack frame
     mov byte tsd_truncated, 0                 ; assume complete until the dest limit proves otherwise
; Cache arguments in non-volatile registers
     mov r12, rcx                              ; r12 = Tokenizer*
     mov r13d, edx                             ; r13d = token_id
     mov r14d, r8d                             ; r14d = position
; Print prefix: "[TOK] pos="
     WRITE_STR tsd_prefix                      ; print "[TOK] pos=" label
; Print position as decimal
     WRITE_UINT64 r14                          ; print position value, macro preserves non-volatiles
; Print " id="
     WRITE_STR tsd_id_label                    ; print " id=" separator
; Print token_id as decimal
     WRITE_UINT64 r13                          ; print token_id value, macro preserves non-volatiles
; Print " piece="
     WRITE_STR tsd_piece_label                 ; print " piece=" label
; Look up vocab[token_id] -> char* piece, id in [0, vocab_size and vocab built, our sampler/encode never emit anything else
     mov rsi, [r12 + TOK_VOCAB]                ; rsi = tok->vocab, char**
     mov rsi, [rsi + r13*TSD_VOCAB_SCALE]      ; rsi = token piece string pointer
     lea rdi, tsd_piece_buf                    ; rdi = dest cursor, stack local
     lea r10, tsd_piece_buf                    ; r10 = buffer base, length derivation later
     lea r11, [r10 + TSD_BUF_SIZE - TSD_LIMIT_MARGIN] ; r11 = dest limit, base + 249
.tsd_copy_loop:
     cmp rdi, r11                              ; dest cursor at limit (no room for a worst-case 2-byte escape)?
     jae .tsd_trunc                            ; yes: dest budget exhausted, decide below whether that actually clipped anything
     lodsb                                     ; al = [rsi++] = piece byte at the source cursor, one instruction replaces mov + inc
     cmp al, 0x5C                              ; literal backslash?, 0x5C is ABOVE 0x20, so it arrives here and never reaches the control-char ladder
     je .tsd_emit_bs                           ; yes, must be doubled, or a literal \n in the piece prints identically to a real newline
     cmp al, 0x22                              ; literal double quote?, also above 0x20, and it is the field delimiter, raw it makes piece="..." unsplittable
     je .tsd_emit_quote                        ; yes, escape it so the piece field stays unambiguously delimited
     cmp al, 0x20                              ; NUL or control char, both < space ?
     jb .tsd_special                           ; yes, separate NUL from the escapes there
     stosb                                     ; printable ASCII/UTF-8: write byte as-is, advance dest
     jmp .tsd_copy_loop                        ; continue
.tsd_emit_bs:
     TSD_EMIT_ESC 0x5C5C                       ; write "\\", escaping the escape keeps a literal \n in the piece distinct from a real newline
.tsd_emit_quote:
     TSD_EMIT_ESC 0x225C                       ; write "\"", backslash in the low byte, quote in the high byte, the CRT-style escaping this format already uses
.tsd_special:
     test al, al                               ; null terminator?
     jz .tsd_copy_done                         ; yes, done copying
; Control char: write escape sequence, \n, \t, \r, or ?
     cmp al, 10                                ; newline, \n ?
     jne .tsd_not_nl                           ; no
     TSD_EMIT_ESC 0x6E5C                       ; write "\n", backslash=0x5C, n=0x6E
.tsd_not_nl:
     cmp al, 13                                ; carriage return, \r ?
     jne .tsd_not_cr                           ; no
     TSD_EMIT_ESC 0x725C                       ; write "\r", backslash + r
.tsd_not_cr:
     cmp al, 9                                 ; tab, \t ?
     jne .tsd_not_tab                          ; no
     TSD_EMIT_ESC 0x745C                       ; write "\t", backslash + t
.tsd_not_tab:
; Other unprintable: replace with visible '?'
     mov al, '?'                               ; replacement char
     stosb                                     ; write it and advance dest, lodsb already consumed the source byte
     jmp .tsd_copy_loop                        ; continue
.tsd_trunc:
     cmp byte [rsi], 0                         ; did the source actually end at the limit?, a piece that fills the budget exactly was NOT clipped
     je .tsd_copy_done                         ; yes: every byte was emitted, no marker
     mov byte tsd_truncated, 1                 ; flag it: what follows is a truncated prefix, falls through to .tsd_copy_done, the same form used for .tsd_marker_tail below
.tsd_copy_done:
     mov rsi, rdi                              ; rsi = dest cursor, one past last written byte
     sub rsi, r10                              ; rsi = piece byte count, dest cursor - base; no single-instruction form exists: x86 addressing is base + index*scale + disp, so two registers cannot be subtracted in one operand
     cmp byte tsd_truncated, 0                 ; was the piece clipped by the dest limit?
     je .tsd_no_marker                         ; no: complete piece, nothing to append
     mov dword [rdi], 0x2E2E2E2E               ; append "..." in one 4-byte store, at most base + 253; provably free under margin 7 : 3 bytes are counted, and the 4th sits at rdi+3, one past the count, so it is never handed to WriteFile
     add rsi, TSD_MARK_LEN                     ; emitted count grows by the marker's 3 bytes, rsi is the WriteFile arg3 set below
.tsd_no_marker:
     sub rsp, 48                               ; 32 shadow + 16 scratch, keeps RSP 16-aligned before API_CALL
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle, cached by WRITE_STR tsd_prefix above, its debug_ensure_std already ran
     lea rdx, tsd_piece_buf                    ; arg2 = buffer pointer, rbp-relative, no extra brackets
     mov r8, rsi                               ; arg3 = byte count
     xor r9d, r9d                              ; arg4 = NULL: the bytes-written out-pointer is OPTIONAL and discarded here
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write piece text to stdout
     add rsp, 48                               ; restore stack
; Write closing double-quote + newline
     WRITE_STR tsd_quote_close_nl              ; print "\"\n"
.tsd_return:
     SF_END rsi, rdi, r12, r13, r14            ; restore non-volatiles, epilogue includes ret
dump_token_stream_disabled:
     ret                                       ; early return when enable_token_dump==0, no frame was set up
     section .rdata
tsd_prefix:
     db "[TOK] pos=", 0                        ; token dump line prefix
tsd_id_label:
     db " id=", 0                              ; id field separator
tsd_piece_label:
     db " piece=", 0x22, 0                     ; piece field separator + opening double-quote, baked in, saves one WRITE_STR for normal path
tsd_quote_close_nl:
     db 0x22, 10, 0                            ; closing double-quote + newline
; ---- teardown: file-local names leave scope here, house convention ----
%unmacro TSD_EMIT_ESC 1                        ; remove the escape-emission macro, all five call sites above
%undef TSD_MARK_LEN                            ; emitted length of the "..." marker
%undef TSD_LIMIT_MARGIN                        ; dest-limit margin
%undef TSD_BUF_SIZE                            ; piece buffer size
%undef TSD_VOCAB_SCALE                         ; vocab ptr array index scale
%undef tsd_piece_buf                           ; SF_ARRAY slot, no _raw alias by definition
%undef tsd_truncated                           ; SF_BYTE slot, no _raw alias by definition
