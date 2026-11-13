; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define MDB_INIT_CAP 512                       ; Initial hash table capacity
%define MDB_GROW_NUM 7                         ; load factor numerator, cap * 7
%define MDB_GROW_DEN 10                        ; load factor denominator, count * 10
%define MDB_GENERIC_WRITE 0x40000000           ; Win32: GENERIC_WRITE
%define MDB_GENERIC_READ 0x80000000            ; Win32: GENERIC_READ
%define MDB_FILE_SHARE_READ 0x00000001         ; Win32: FILE_SHARE_READ
%define MDB_CREATE_ALWAYS 2                    ; Win32: CREATE_ALWAYS
%define MDB_OPEN_EXISTING 3                    ; Win32: OPEN_EXISTING
%define MDB_FILE_ATTR_NORMAL 0x00000080        ; Win32: FILE_ATTRIBUTE_NORMAL
%define MDB_INVALID_HANDLE -1                  ; Invalid handle value
%define MOVEFILE_REPLACE_EXISTING 1            ; Win32: MOVEFILE_REPLACE_EXISTING
%define MOVEFILE_WRITE_THROUGH 8               ; Win32: MOVEFILE_WRITE_THROUGH, MoveFileExA dwFlags bit, do NOT reuse this name/value for CreateFile, whose write-through control is the separate FILE_FLAG_WRITE_THROUGH bit of dwFlagsAndAttributes; note that 0x2 in CreateFile's dwCreationDisposition is CREATE_ALWAYS, see MDB_CREATE_ALWAYS above
%define WINERR_FILE_NOT_FOUND 2                ; Win32: ERROR_FILE_NOT_FOUND
; LZ77 compression constants
%define LZ77_HT_SLOTS 8192                     ; 8K 16-bit address tags = 16 KiB table, 4 stack pages
%define LZ77_HT_BASE_OFF 8                     ; Hash table offset from RSP within the mdb_compress stack allocation
%define LZ77_MIN_MATCH_END 12                  ; Minimum remaining bytes to attempt a match, no-match zone threshold
%define LZ77_MIN_MATCH_LEN 4                   ; Minimum match length encoded by a match token, len field counts match_len - 4
%define LZ77_NIBBLE_MAX 15                     ; Max value fitting in a 4-bit nibble, LZ77 token nibble clamp
%define LZ77_OFFSET_BYTES 2                    ; Match offset field width, 16-bit offset, 2 bytes
%define LZ77_COMPRESS_FRAME (LZ77_HT_SLOTS * 2 + 24) ; 16-bit position tag table + unchanged 24-byte pad
%define LZ77_SAVED_RDI_OFF 8                   ; offset below RSP of the `push rdi` slot that guards the table clear, the lowest byte the frame occupies
%if ((LZ77_HT_SLOTS * 2) / 4096) * 4096 != LZ77_HT_SLOTS * 2
%error LZ77_HT_SLOTS*2 must cover whole 4096-byte pages for mdb_compress stack probing ; drift check
%endif
; E_FLAG bit definitions
%define MDB_FLAG_PERSISTENT 1                  ; bit 0: persist to disk on db_save
; Serialized entry header field offsets, key_len dword + data_len qword + flags dword
%define SER_HDR_DATA_LEN 4                     ; Offset of data_len, qword within serialized entry header
%define SER_HDR_FLAGS 12                       ; Offset of flags, dword within serialized entry header
%define SER_HDR_SIZE 16                        ; Total size of serialized entry header, 4+8+4 bytes
; Per-block chunk header field offsets, qword decomp_size + qword comp_size = 16 bytes
%define DB_HDR_COMP_SZ 8                       ; Offset of compressed size, qword within a BLOCK chunk header
; DB file header field offsets, magic + file version + count + pool_size = 20 bytes
%define DB_HDR_MAGIC 0                         ; 'v7db' magic dword, a v7 database always starts with these 4 bytes
%define DB_HDR_FILE_VER 4                      ; file-format version dword, MDB_DB_FORMAT_VERSION
%define DB_HDR_CNT_OFF 8                       ; entry count dword
%define DB_HDR_POOL_SZ 12                      ; pool size qword, final KP_USED at save, load pre-alloc target
%define MDB_DB_FORMAT_VERSION 1                ; current lz4ht.db file-format version, bump on format change
%define DB_HDR_END_MARKER_QW 8                 ; offset of second qword in the 16-byte end-marker chunk header, same field as DB_HDR_COMP_SZ, named for the marker's role
; Mask decrement: capacity - 1 = hash mask
%define MASK_SUB_ONE 1                         ; Subtract one from capacity to create bitmask
; Streaming compression constants. DB_CHUNK_HDR is defined FIRST because
; COMP_BUF_SIZE references it; the reference is legal either way, NASM expands a
; %define body at INVOCATION time, not at definition time, but reading order
; matters more than legality here.
%define DB_CHUNK_HDR 16                        ; per-block chunk header: decomp_size,qword + comp_size,qword, NOT the 20-byte file header, see MDB_FILE_HDR_SIZE
%define BLOCK_SIZE 0x20000                     ; 128 KB decompressed block for streaming db_save/db_load
%define COMP_BUF_SIZE (DB_CHUNK_HDR + BLOCK_SIZE + 1024) ; LZ4 bound adds < 530 bytes at 128 KiB, so 1 KiB is sufficient
%define MDB_FILE_HDR_SIZE 20                   ; uncompressed file header: magic,4 +version,4 +count,4 +pool_size,8
