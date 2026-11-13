; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; Constants
; ============================================================================
%define FBM_MAX_SLOTS 256                      ; maximum concurrent allocations
%define FBM_SLOT_BYTES 24                      ; bytes per slot struct, pointer + file_handle + mapping_handle
%define FBM_SLOT_PTR 0                         ; offset: mapped base pointer
%define FBM_SLOT_FILE 8                        ; offset: file HANDLE
%define FBM_SLOT_MAP 16                        ; offset: mapping HANDLE
%define FBM_PATH_BYTES 520                     ; MAX_PATH,260 * sizeof,wchar_t
%define FBM_WPATH_MAX 260                      ; MAX_PATH in wchar_t units
; Win32 constants for CreateFileW / CreateFileMappingW / MapViewOfFile
%define FBM_GENERIC_RW 0xC0000000              ; GENERIC_READ | GENERIC_WRITE
%define FBM_CREATE_ALWAYS 2                    ; dwCreationDisposition
%define FBM_FILE_ATTR_TMP 0x100                ; FILE_ATTRIBUTE_TEMPORARY
%define FBM_FILE_FLAG_DEL 0x04000000           ; FILE_FLAG_DELETE_ON_CLOSE
%define FBM_PAGE_RW 0x04                       ; PAGE_READWRITE
%define FBM_FILE_MAP_ALL 0x000F001F            ; FILE_MAP_ALL_ACCESS
%define FBM_INVALID_HANDLE -1                  ; INVALID_HANDLE_VALUE
%define FBM_FSCTL_SPARSE 0x000900C4            ; FSCTL_SET_SPARSE control code
     section .bss
     alignb 64                                 ; SYMBOL alignment, a `section .bss align=` attribute would be ignored here : lines the TABLE START on a cache-line boundary for the two linear slot walks, lines 52-59, 196-203.It does NOT keep each 24-byte slot inside one line, 24 does not divide 64, so most slots straddle a boundary; the table is 256 * 24 = 6144 bytes = exactly 96 lines, so the walk stays sequential at 2-3 pointer loads per line. What alignment actually buys is 8-BYTE alignment of every slot base: 24 is a multiple of 8, so the +0/+8/+16 qword fields are all naturally aligned and no field access is ever a split load. Do NOT widen the stride to 64 to "fix" straddling, it triples the table and the walk for nothing.
fbm_slots:
     resb FBM_MAX_SLOTS * FBM_SLOT_BYTES       ; 256 slots x 24 bytes
; ============================================================================
; Data: wide-string prefix for GetTempFileNameW, L"FB"
; ============================================================================
     section .rdata
fbm_prefix:
     dw 'F', 'B', 0                            ; L"FB" prefix for temp filenames, constant blob, project convention puts read-only data in .rdata
     section .text
fbm_alloc:
     SF_START fbm_alloc, rbx, r12              ; save non-volatile registers
     SF_QWORD fbm_size                         ; requested allocation size
     SF_QWORD fbm_file_h                       ; file HANDLE
     SF_QWORD fbm_map_h                        ; mapping HANDLE
     SF_ARRAY fbm_path, FBM_PATH_BYTES         ; wchar_t path[MAX_PATH]
     SF_ARRAY fbm_name, FBM_PATH_BYTES         ; wchar_t name[MAX_PATH]
     SF_DWORD fbm_bytes_ret                    ; bytesReturned for DeviceIoControl
     SF_COMMIT                                 ; finalize stack frame
     TRACE_STR "[FBM] fbm_alloc entry"         ; trace: function entry
     mov fbm_size, rcx                         ; save requested size
     TRACE_HEX rcx                             ; trace: requested size in bytes
; ========================================================================
; Step 1: Find a free slot, pointer == 0 in the BSS table
; Linear pointer walk: cmp slot ptr / add rbx,24 / cmp rbx,end
; ========================================================================
     lea rbx, [rel fbm_slots]                  ; rbx = &fbm_slots[0]
     lea r12, [rbx + FBM_MAX_SLOTS * FBM_SLOT_BYTES] ; r12 = end boundary, pointer comparison
.find_slot:
     cmp qword [rbx + FBM_SLOT_PTR], 0         ; slot.pointer == 0, free ?
     je .slot_found                            ; yes, use this slot
     add rbx, FBM_SLOT_BYTES                   ; advance to next slot, pointer increment vs index multiply
     cmp rbx, r12                              ; scanned all slots?
     jb .find_slot                             ; no, continue
; All 256 slots in use
     FATAL "fbm_alloc: 256-slot table full (fbm_free leak?)"
.slot_found:
     TRACE_STR "[FBM] slot found"              ; trace: free slot located
; rbx = address of the free slot, pointer walk, filled directly at lines 160-165, the slot fill after the MapViewOfFile step
; ========================================================================
; Step 2: GetTempPathW,MAX_PATH, path
; ========================================================================
     mov ecx, FBM_WPATH_MAX                    ; nBufferLength = MAX_PATH
     lea rdx, fbm_path                         ; lpBuffer = path[], SF_ARRAY has brackets
     API_CALL [__imp_GetTempPathW], rcx, rdx   ; get temp directory path
     test eax, eax                             ; return 0 = failure?
     jz .fbm_fail_temppath                     ; yes, FATAL out of line
     cmp eax, FBM_WPATH_MAX                    ; required length >= buffer?
     jae .fbm_fail_temppath_big                ; buffer too small, FATAL out of line
     TRACE_STR "[FBM] GetTempPathW done"       ; trace: temp path ready
; ========================================================================
; Step 3: GetTempFileNameW,path, L"FB", 0, name
; ========================================================================
     lea rcx, fbm_path                         ; lpPathName = path[], SF_ARRAY has brackets
     lea rdx, [rel fbm_prefix]                 ; lpPrefixString = L"FB", RIP-relative
     xor r8d, r8d                              ; uUnique = 0, generate unique name
     lea r9, fbm_name                          ; lpTempFileName = name[], SF_ARRAY has brackets
     API_CALL [__imp_GetTempFileNameW], rcx, rdx, r8, r9 ; create unique temp filename
     test eax, eax                             ; success?
     jz .fbm_fail_tempname                     ; no, FATAL out of line
     TRACE_STR "[FBM] GetTempFileNameW done"   ; trace: temp file name created
; ========================================================================
; Step 4: CreateFileW,name, GENERIC_READ|GENERIC_WRITE, 0, NULL,
; CREATE_ALWAYS,
; FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
; NULL
; Stack args passed directly as macro values, API_CALL's internal sub rsp
; shifts RSP, so pre-written [rsp+ARG] would resolve to wrong location .
; ========================================================================
     lea rcx, fbm_name                         ; lpFileName = name[], SF_ARRAY has brackets
     mov edx, FBM_GENERIC_RW                   ; dwDesiredAccess
     xor r8d, r8d                              ; dwShareMode = 0, exclusive, nothing else opens the temp file
     xor r9d, r9d                              ; lpSecurityAttributes = NULL
     API_CALL [__imp_CreateFileW], rcx, rdx, r8, r9, FBM_CREATE_ALWAYS, (FBM_FILE_ATTR_TMP | FBM_FILE_FLAG_DEL), 0 ; open temp file with DELETE_ON_CLOSE
     cmp rax, FBM_INVALID_HANDLE               ; INVALID_HANDLE_VALUE?
     je .fbm_fail_createfile                   ; yes, FATAL out of line
     mov fbm_file_h, rax                       ; save file handle
     TRACE_STR "[FBM] CreateFileW done"        ; trace: file handle acquired
; ========================================================================
; Step 5: DeviceIoControl,file, FSCTL_SET_SPARSE, NULL, 0, NULL, 0,
; &bytesReturned, NULL
; Marks the file as sparse so zero pages don't consume disk space.
; Stack args passed directly as macro values. r10 is free here for the
; lpBytesReturned pointer, API_CALL stack-stages all arg sources before
; the call, so any register works as an argument source .
; ========================================================================
     mov rcx, fbm_file_h                       ; hDevice = file handle
     mov edx, FBM_FSCTL_SPARSE                 ; dwIoControlCode = FSCTL_SET_SPARSE
     xor r8d, r8d                              ; lpInBuffer = NULL
     xor r9d, r9d                              ; nInBufferSize = 0
     lea r10, fbm_bytes_ret                    ; lpBytesReturned
     API_CALL [__imp_DeviceIoControl], rcx, rdx, r8, r9, 0, 0, r10, 0 ; mark file sparse, non-fatal if fails
     TRACE_STR "[FBM] DeviceIoControl done"    ; trace: sparse file set
; ========================================================================
; Step 6: SetFilePointerEx,file, size, NULL, FILE_BEGIN
; Resize the file to the requested allocation size.
; ========================================================================
     mov rcx, fbm_file_h                       ; hFile = file handle
     mov rdx, fbm_size                         ; liDistanceToMove = size, 64-bit, passed by value
     xor r8d, r8d                              ; lpNewFilePointer = NULL
     xor r9d, r9d                              ; dwMoveMethod = FILE_BEGIN, 0
     API_CALL [__imp_SetFilePointerEx], rcx, rdx, r8, r9 ; seek to end of requested size
     test eax, eax                             ; success?
     jz .fbm_fail_setfp                        ; no, FATAL out of line
     TRACE_STR "[FBM] SetFilePointerEx done"   ; trace: file resized
     mov rcx, fbm_file_h                       ; hFile = file handle
; ========================================================================
; Step 7: SetEndOfFile,file
; Extend the file to the file position established by SetFilePointerEx, completes the resize started in Step 6 .
; ========================================================================
     API_CALL [__imp_SetEndOfFile], rcx        ; extend file to current position
     test eax, eax                             ; success?
     jz .fbm_fail_setend                       ; no, FATAL out of line
     TRACE_STR "[FBM] SetEndOfFile done"       ; trace: file committed
; ========================================================================
; Step 8: CreateFileMappingW,file, NULL, PAGE_READWRITE, 0, 0, NULL
; ========================================================================
     mov rcx, fbm_file_h                       ; hFile = file handle
     xor edx, edx                              ; lpAttributes = NULL
     mov r8d, FBM_PAGE_RW                      ; flProtect = PAGE_READWRITE
     xor r9d, r9d                              ; dwMaximumSizeHigh = 0
     API_CALL [__imp_CreateFileMappingW], rcx, rdx, r8, r9, 0, 0 ; dwMaximumSizeLow=0, use file size, lpName=NULL
     test rax, rax                             ; success?
     jz .fbm_fail_mapping                      ; no, FATAL out of line
     mov fbm_map_h, rax                        ; save mapping handle
     TRACE_STR "[FBM] CreateFileMappingW done" ; trace: mapping handle acquired
; ========================================================================
; Step 9: MapViewOfFile,mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0
; ========================================================================
     mov rcx, fbm_map_h                        ; hFileMappingObject = mapping
     mov edx, FBM_FILE_MAP_ALL                 ; dwDesiredAccess
     xor r8d, r8d                              ; dwFileOffsetHigh = 0
     xor r9d, r9d                              ; dwFileOffsetLow = 0
     API_CALL [__imp_MapViewOfFile], rcx, rdx, r8, r9, 0 ; dwNumberOfBytesToMap=0, entire file
     test rax, rax                             ; success?
     jz .fbm_fail_mapview                      ; no, FATAL out of line
     TRACE_STR "[FBM] MapViewOfFile done"      ; trace: view mapped
     mov [rbx + FBM_SLOT_PTR], rax             ; slot.pointer = mapped base, rax straight from MapViewOfFile
     mov rdx, fbm_file_h                       ; rdx = file handle
     mov [rbx + FBM_SLOT_FILE], rdx            ; slot.file_handle = file handle
     mov rdx, fbm_map_h                        ; rdx = mapping handle
     mov [rbx + FBM_SLOT_MAP], rdx             ; slot.mapping_handle = mapping handle
; rax still holds the mapped pointer, return it
     TRACE_STR "[FBM] fbm_alloc success"       ; trace: all three handles parked, slot complete
     TRACE_HEX rax                             ; trace: returned pointer value
     SF_END rbx, r12                           ; restore non-volatile registers and return
; fbm_alloc failure paths, out of line: the success path falls through
; with SF_END, ret, so none of the blocks below is reachable by
; fall-through, and each FATAL never returns, so they cannot chain.
.fbm_fail_temppath:
     FATAL "fbm_alloc: GetTempPathW failed"    ; FATAL directly, no cleanup before death
.fbm_fail_temppath_big:
     FATAL "fbm_alloc: GetTempPathW buffer too small"
.fbm_fail_tempname:
     FATAL "fbm_alloc: GetTempFileNameW failed" ; FATAL directly
.fbm_fail_createfile:
     FATAL "fbm_alloc: CreateFileW failed"     ; FATAL directly
.fbm_fail_setfp:
     FATAL "fbm_alloc: SetFilePointerEx failed" ; FATAL directly
.fbm_fail_setend:
     FATAL "fbm_alloc: SetEndOfFile failed"    ; FATAL directly
.fbm_fail_mapping:
     FATAL "fbm_alloc: CreateFileMappingW failed" ; FATAL directly, file handle reclaimed by OS
.fbm_fail_mapview:
     FATAL "fbm_alloc: MapViewOfFile failed"   ; FATAL directly, file+mapping handles reclaimed by OS
fbm_free:
     SF_START fbm_free, rbx, r12               ; save non-volatile registers, SF_START emits the global
     SF_COMMIT                                 ; finalize stack frame
     TRACE_STR "[FBM] fbm_free entry"          ; trace: free entry
     TRACE_HEX rcx                             ; trace: pointer to free
; ========================================================================
; Scan BSS table for slot with matching pointer
; Linear pointer walk: cmp slot ptr / add rbx,24 / cmp rbx,end
; ========================================================================
     lea rbx, [rel fbm_slots]                  ; rbx = &fbm_slots[0]
     lea r12, [rbx + FBM_MAX_SLOTS * FBM_SLOT_BYTES] ; r12 = end boundary
.free_scan:
     cmp qword [rbx + FBM_SLOT_PTR], rcx       ; slot.pointer == ptr?, rcx = argument, unoverwritten since entry
     je .free_found                            ; yes, this is the slot
     add rbx, FBM_SLOT_BYTES                   ; advance to next slot, pointer increment vs index multiply
     cmp rbx, r12                              ; scanned all?
     jb .free_scan                             ; no, continue
     xor eax, eax                              ; eax = 0 = "not mine, ignored", 2 bytes, zero-extends
     SF_END rbx, r12                           ; restore non-volatile registers and return
; a failed unmap/close must NOT zero the slot, the slot would claim
; "free" while the OS object is leaked, FATAL names which call failed --
.fbm_free_unmap_fatal:
     FATAL "fbm_free: UnmapViewOfFile failed"
.fbm_free_map_fatal:
     FATAL "fbm_free: CloseHandle on mapping handle failed"
.fbm_free_file_fatal:
     FATAL "fbm_free: CloseHandle on file handle failed"
.free_found:
     TRACE_STR "[FBM] fbm_free slot found"     ; trace: slot matched
; rbx already points at the matching slot, no need to recompute index*24
; ========================================================================
; UnmapViewOfFile,ptr
; ========================================================================
     API_CALL [__imp_UnmapViewOfFile], rcx     ; unmap the view, rcx still holds the ptr argument
     test eax, eax                             ; unmap succeeded?
     jz .fbm_free_unmap_fatal                  ; no, mapping leaked, do not zero the slot
; ========================================================================
; CloseHandle,mapping_handle
; ========================================================================
     mov rcx, [rbx + FBM_SLOT_MAP]             ; rcx = mapping_handle
     API_CALL [__imp_CloseHandle], rcx         ; close mapping handle
     test eax, eax                             ; close succeeded?
     jz .fbm_free_map_fatal                    ; no, mapping handle leaked, do not zero the slot
     mov rcx, [rbx + FBM_SLOT_FILE]            ; rcx = file_handle
     API_CALL [__imp_CloseHandle], rcx         ; close file handle, triggers auto-delete
     test eax, eax                             ; close succeeded?
     jz .fbm_free_file_fatal                   ; no, file handle leaked, do not zero the slot
; ========================================================================
; Zero the slot, pointer, file_handle, mapping_handle all = 0
; ========================================================================
     xor eax, eax                              ; rax = 0; the 32-bit xor zero-extends, so every store below writes a full 8 bytes
     mov [rbx + FBM_SLOT_PTR], rax             ; slot.pointer = 0
     mov [rbx + FBM_SLOT_FILE], rax            ; slot.file_handle = 0, 8-byte store clears BOTH halves of the handle
     mov [rbx + FBM_SLOT_MAP], rax             ; slot.mapping_handle = 0, 8-byte store, same reason
     TRACE_STR "[FBM] fbm_free done"           ; trace: free complete
     mov eax, 1                                ; eax = 1 = "was fbm's, freed", set AFTER the zeroing stores used eax=0
     SF_END rbx, r12                           ; restore non-volatile registers and return
; ============================================================================
; Clean up %define constants, prevent namespace pollution
; ============================================================================
%undef FBM_MAX_SLOTS
%undef FBM_SLOT_PTR
%undef FBM_SLOT_FILE
%undef FBM_SLOT_MAP
%undef FBM_PATH_BYTES
%undef FBM_WPATH_MAX
%undef FBM_GENERIC_RW
%undef FBM_CREATE_ALWAYS
%undef FBM_FILE_ATTR_TMP
%undef FBM_FILE_FLAG_DEL
%undef FBM_PAGE_RW
%undef FBM_FILE_MAP_ALL
%undef FBM_INVALID_HANDLE
%undef FBM_FSCTL_SPARSE
%undef FBM_SLOT_BYTES
; ---- SF_* slot cleanup, house convention: slot names leave scope here ----
%undef fbm_name
%undef fbm_path
%undef fbm_bytes_ret
%undef fbm_bytes_ret_raw
%undef fbm_size
%undef fbm_size_raw
%undef fbm_file_h
%undef fbm_file_h_raw
%undef fbm_map_h
%undef fbm_map_h_raw
