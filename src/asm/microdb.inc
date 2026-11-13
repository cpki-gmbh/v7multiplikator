; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; The g_ht layout + Entry field constants consumed by microdb_ht/persist, HT_TABLE..ENTRY_SZ
; are defined in v7_core.inc, the 64-bit layout, this module only declares the exports
; and includes the implementation files below.
     global mdb_compress, mdb_decompress       ; Export the project compressor, own format, not LZ77-spec, the real implementation
     global ht_init, ht_destroy, ht_add, ht_find, db_save, db_load, db_set_readonly ; Export hash table and persistence
     section .text                             ; microdb_ht/lz77 emit code and carry no section directive of their own, so they inherit this; microdb_mem, microdb_persist, a .data block for its diagnostic strings, then back to .text and microdb_diag each set their own
%include "microdb_const.inc"
%include "microdb_macros.inc"
%include "microdb_mem.inc"
%include "microdb_lz77.inc"
%include "microdb_ht.inc"
%include "microdb_persist.inc"
%include "microdb_diag.inc"
%unmacro KP_ALLOC_INLINE 0                     ; group-local pool-allocation macro, expansions above
%unmacro ALIGN4 1                              ; group-local 4-byte-align macro, expansions above
%undef KP_GROW
%undef KP_LIVE_RESERVE
%undef LZ77_QWORD_SAFE
%undef MDB_INIT_CAP
%undef MDB_GROW_NUM
%undef MDB_GROW_DEN
%undef MDB_GENERIC_WRITE
%undef MDB_GENERIC_READ
%undef MDB_FILE_SHARE_READ
%undef MDB_CREATE_ALWAYS
%undef MDB_OPEN_EXISTING
%undef MDB_FILE_ATTR_NORMAL
%undef MDB_INVALID_HANDLE
%undef MOVEFILE_REPLACE_EXISTING
%undef MOVEFILE_WRITE_THROUGH
%undef WINERR_FILE_NOT_FOUND
%undef LZ77_HT_SLOTS
%undef LZ77_HT_BASE_OFF
%undef LZ77_MIN_MATCH_END
%undef LZ77_MIN_MATCH_LEN
%undef LZ77_NIBBLE_MAX
%undef LZ77_OFFSET_BYTES
%undef LZ77_COMPRESS_FRAME
%undef LZ77_SAVED_RDI_OFF
%undef MDB_FLAG_PERSISTENT
%undef SER_HDR_DATA_LEN
%undef SER_HDR_FLAGS
%undef SER_HDR_SIZE
%undef DB_HDR_CNT_OFF
%undef DB_HDR_COMP_SZ
%undef DB_HDR_POOL_SZ
%undef DB_HDR_MAGIC
%undef DB_HDR_FILE_VER
%undef MDB_DB_FORMAT_VERSION
%undef DB_HDR_END_MARKER_QW
%undef MASK_SUB_ONE
%undef DB_CHUNK_HDR
%undef BLOCK_SIZE
%undef COMP_BUF_SIZE
%undef MDB_FILE_HDR_SIZE
