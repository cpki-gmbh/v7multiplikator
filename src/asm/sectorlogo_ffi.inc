; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; CallLibrary ABI contract: every resolved callee runs with the Windows
; register convention (args rcx/rdx/r8/r9, further args on the stack, return
; in rax) on EVERY platform. On Windows that is the native convention; on
; Linux the engine still uses it, so external DLLs must be built with
; -D_WIN32-style register passing and HOST module exports must carry the
; ms_abi attribute or their arguments arrive in the wrong registers.
; The marshal/unmarshal helpers below write the arguments in that register
; order and read the integer/text/pointer return from rax.
; ============================================================================
sl_ffi_resolve:
     push rbx                                  ; save non-volatile, holds HMODULE
     push rsi                                  ; save non-volatile, atom text pointer
     push r15                                  ; save non-volatile, caches descriptor
     call PopCons                              ; fused: ebx = DLL_NAME, eax = descriptor tail
     mov r15d, eax                             ; park tail: PROC_NAME is its first element
; --- extract DLL_NAME from descriptor[0] ---
     mov eax, ebx                              ; eax = DLL_NAME atom
     test eax, SL_CONS_TAG | SL_NUMBER_TAG     ; any tag bit set, CONS/NUMBER/FLOAT ?
     jnz .ffi_bad_name                         ; yes: not a string atom
     test eax, eax                             ; NIL, 0 ?
     jz .ffi_bad_name                          ; yes: name slot empty
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = string in heap, no '"' prefix, the three 32-bit guards above give bits 31:30 clear; bits 63:32 come from the evaluator's 32-bit-write atom-return convention, see the note at sl_ffi_marshal_ptr
; --- check for "SELF", call back into v7 DLL ---
     cmp dword [rsi], SL_FFI_SELF_DWORD        ; first 4 bytes = "SELF"?
     jne .ffi_check_host                       ; no: try "HOST" next
     cmp byte [rsi + SL_FFI_SELF_NUL], 0       ; NUL terminator after "SELF"?
     jne .ffi_check_host                       ; no: not exactly "SELF"
; --- SELF path: use v7's own HMODULE, DllMain ATTACH stores it
; unconditionally at load time, always valid when a script evaluates ---
     mov rbx, [rel sl_v7_hmodule]              ; rbx = v7 HMODULE, non-volatile
     jmp .ffi_got_module                       ; skip LoadLibraryA
.ffi_check_host:
; --- check for "HOST", callback into the host EXE module ---
     cmp dword [rsi], SL_FFI_HOST_DWORD        ; first 4 bytes = "HOST"?
     jne .ffi_load_dll                         ; no: regular DLL path
     cmp byte [rsi + SL_FFI_HOST_NUL], 0       ; NUL terminator after "HOST"?
     jne .ffi_load_dll                         ; no: not exactly "HOST"
; --- HOST path: use the caller-registered module handle ---
     mov rbx, [rel sl_caller_hmodule]          ; rbx = host EXE HMODULE, set via v7s_set_caller_hmodule
     test rbx, rbx                             ; host module registered?
     jz .ffi_resolve_fail                      ; no: fail
     jmp .ffi_got_module                       ; skip LoadLibraryA
.ffi_load_dll:
; --- owner registry lookup: a module this process already loaded is REUSED.
; LoadLibraryA would only acquire a second loader reference for the same DLL
; and the memo cache drops entries without FreeLibrary, the ownership
; record lives here, keyed by the stable NAME STRING, immune to atom GC ---
     mov r11d, [rel sl_ffi_owned_count]        ; owned module count
     test r11d, r11d                           ; any owner records?
     jz .ffi_owned_new                         ; registry empty: first external load
     lea r8, [rel sl_ffi_owned_names]          ; r8 = registry name base
     xor r10d, r10d                            ; r10d = entry index
.ffi_owned_walk:
     mov rax, r10                              ; rax = entry index, rax dead at .ffi_do_load, Car re-derives the atom after any load
     shl rax, 7                                ; rax = entry byte offset, 128 = SL_FFI_OWNED_NAME_BYTES; shift, not an index scale
     lea rdx, [r8 + rax]                       ; rdx = this entry's stored name
     mov rcx, rsi                              ; rcx = current name cursor, rcx is LoadLibraryA's arg slot, dead until the call below
.ffi_owned_cmp:
     mov al, [rcx]                             ; current name byte
     cmp al, [rdx]                             ; stored name byte
     jne .ffi_owned_next                       ; byte differs: next entry
     test al, al                               ; NUL on both sides?
     jz .ffi_owned_hit                         ; yes: exact name match
     inc rcx                                   ; advance current cursor
     inc rdx                                   ; advance stored cursor
     jmp .ffi_owned_cmp                        ; next byte
.ffi_owned_next:
     inc r10d                                  ; next entry
     cmp r10d, r11d                            ; entries exhausted?
     jb .ffi_owned_walk                        ; no: keep walking
     jmp .ffi_owned_new                        ; no owner record: genuinely new DLL
.ffi_owned_hit:
     lea rax, [rel sl_ffi_owned_mods]          ; rax = registry handle base
     mov rbx, [rax + r10*SL_FFI_OWNED_HMOD_SCALE] ; rbx = owned HMODULE, reuse, NO second LoadLibraryA
     jmp .ffi_got_module                       ; the registry is the durable record, nothing to cache
.ffi_owned_new:
; --- API_CALL LoadLibraryA,atom text -> rax = HMODULE ---
     mov rcx, rsi                              ; arg1: DLL name, atom text, NUL-terminated; rsi survives the memo probe, which touches only rcx/edx/eax
     API_CALL [rel __imp_LoadLibraryA]         ; rax = HMODULE
     test rax, rax                             ; NULL = load failure?
     jz .ffi_resolve_fail                      ; yes: fail
     mov rbx, rax                              ; rbx = HMODULE, non-volatile
; --- register ownership: copy the name OUT of the atom heap, stable across
; atom GC, the memo cache's atom keys are destroyed on GC, the registry's copy does
; not and store the handle. One loader reference per distinct DLL, process
; lifetime, no FreeLibrary anywhere: the OS reclaims at exit, and an in-flight
; native call must never see an unloaded module.---
     mov r11d, [rel sl_ffi_owned_count]        ; current populated count
     cmp r11d, SL_FFI_OWNED_SLOTS              ; registry full?
     jae .ffi_owned_table_full                 ; reject: the number of distinct external DLLs exceeds 32
     lea rdi, [rel sl_ffi_owned_names]         ; rdi = registry name base
     mov r10d, r11d                            ; r10d = populated count
     shl r10d, 7                               ; r10d = entry byte offset, 128 = SL_FFI_OWNED_NAME_BYTES; shift, not a scale
     add rdi, r10                              ; rdi = this slot's name dest
     xor edx, edx                              ; byte index
.ffi_owned_copyname:
     mov al, [rsi + rdx]                       ; rsi = atom-heap name text, byte copy stops at the NUL, never reads past the atom's own bytes
     cmp edx, SL_FFI_OWNED_NAME_BYTES          ; slot capacity: indices 0..127 are writable, the NUL may occupy index 127
     jae .ffi_owned_name_toobig                ; name cannot fit with its terminator: contract violation, reject
     mov [rdi + rdx], al                       ; store the byte
     test al, al                               ; NUL reached?
     jz .ffi_owned_copied                      ; yes: copy complete
     inc edx                                   ; next byte
     jmp .ffi_owned_copyname                   ; copy until NUL or capacity
.ffi_owned_name_toobig:
     FATAL "SectorLogo: CallLibrary DLL name exceeds the 127-byte owner-registry slot"
.ffi_owned_table_full:
     FATAL "SectorLogo: CallLibrary owner registry full (32 distinct external DLLs)"
.ffi_owned_copied:
     lea rax, [rel sl_ffi_owned_mods]          ; rax = registry handle base
     mov [rax + r11*SL_FFI_OWNED_HMOD_SCALE], rbx ; mods[count] = HMODULE
     inc dword [rel sl_ffi_owned_count]        ; publish the new owner
.ffi_got_module:
; --- extract PROC_NAME from descriptor[1] ---
     mov eax, r15d                             ; eax = descriptor tail, PROC_NAME is its first element
     call Car                                  ; eax = PROC_NAME atom
     test eax, SL_CONS_TAG | SL_NUMBER_TAG     ; any tag bit set, CONS/NUMBER/FLOAT ?
     jnz .ffi_bad_name                         ; yes: not a string atom
     test eax, eax                             ; NIL, 0 ?
     jz .ffi_bad_name                          ; yes: name slot empty
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = string in heap, no '"' prefix, the three 32-bit guards above give bits 31:30 clear; bits 63:32 come from the evaluator's 32-bit-write atom-return convention, see the note at sl_ffi_marshal_ptr
.ffi_do_proc:
; --- API_CALL GetProcAddress,HMODULE, proc_name -> rax = FARPROC ---
     mov rcx, rbx                              ; arg1: HMODULE
     mov rdx, rsi                              ; arg2: proc name, atom text, NUL-terminated
     API_CALL [rel __imp_GetProcAddress]       ; rax = function pointer
     test rax, rax                             ; NULL = not exported?
     jz .ffi_resolve_fail                      ; yes: fail
.ffi_resolve_done:
     pop r15                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; rax = function pointer
.ffi_resolve_fail:
     xor eax, eax                              ; return NULL, 0
     jmp .ffi_resolve_done                     ; shared epilogue, rax preserved
.ffi_bad_name:
     FATAL "SectorLogo: CallLibrary descriptor dll/proc name is not a string" ; NUMBER/CONS/FLOAT/NIL name slot would lea a wild heap pointer
; --- per-call FFI frame field offsets, SL_FFI_FR_ARGS holds 16 qwords ---
%define SL_FFI_FR_FUNC_PTR 0                   ; qword, resolved native function pointer
%define SL_FFI_FR_DESC 8                       ; dword, descriptor cons list, tagged
%define SL_FFI_FR_RET_TYPE 12                  ; dword, return type atom
%define SL_FFI_FR_ARG_TYPES 16                 ; dword, remaining arg_types list, tagged
%define SL_FFI_FR_ARG_COUNT 20                 ; dword, number of marshalled C arguments
%define SL_FFI_FR_CALLFRAME 24                 ; dword, native call frame size, for dealloc
%define SL_FFI_FR_ARGS 32                      ; 16 qwords, 128 bytes, raw Win64 argument slots
%define SL_FFI_FR_TOTAL 160                    ; frame bytes, 16-aligned
%define SL_FFI_FR_SLACK 8                      ; extra sub bytes so `and rsp,-16` can consume 0 or 8
; --- phase-B field accessors, relative to rsi, = frame + SL_FFI_FR_ARGS ---
%define SL_FFI_PB_FUNC_PTR (SL_FFI_FR_FUNC_PTR - SL_FFI_FR_ARGS) ; func ptr via args base, -32
%define SL_FFI_PB_RET_TYPE (SL_FFI_FR_RET_TYPE - SL_FFI_FR_ARGS) ; ret type via args base, -20
%define SL_FFI_PB_CALLFRAME (SL_FFI_FR_CALLFRAME - SL_FFI_FR_ARGS) ; call frame size via args base, -8
sl_ffi_invoke:
     push rbx                                  ; save non-volatile rbx, PopCons at the arg loop writes EBX, and a callee must not destroy a non-volatile whether or not it reads it back
     push rbp                                  ; save non-volatile rbp, frame anchor
     push rsi                                  ; save non-volatile rsi, args base / ret_str
     push rdi                                  ; save non-volatile rdi, arg count in call phase
     mov rbp, rsp                              ; rbp = anchor: pre-alignment RSP for epilogue restore
     sub rsp, SL_FFI_FR_TOTAL + SL_FFI_FR_SLACK ; allocate frame + alignment slack
     and rsp, -16                              ; force RSP 0 mod 16 regardless of entry alignment
; --- 1. Evaluate first arg = descriptor ---
     call EvalNextArg                          ; eax = descriptor cons list
     test eax, eax                             ; NIL?
     jz sl_ffi_err_nil                         ; yes: FATAL
     mov [rsp + SL_FFI_FR_DESC], eax           ; save descriptor in frame
; --- 2. Resolve function pointer ---
     call sl_ffi_resolve                       ; rax = 64-bit function pointer
     test rax, rax                             ; NULL = resolve failure?
     jz sl_ffi_err_resolve                     ; yes: FATAL
     mov [rsp + SL_FFI_FR_FUNC_PTR], rax       ; save in frame
; --- 3. Extract RET_TYPE, 3rd element and ARG_TYPES, 4th element ---
     mov eax, [rsp + SL_FFI_FR_DESC]           ; eax = descriptor
     call Cdr                                  ; skip DLL_NAME
     call Cdr                                  ; skip PROC_NAME
     call PopCons                              ; fused: ebx = ret_type, eax = ([arg_types]) tail
     mov [rsp + SL_FFI_FR_ARG_TYPES], eax      ; root the ARG_TYPES-containing tail across type validation
     mov eax, ebx                              ; ret_type atom
     mov [rsp + SL_FFI_FR_RET_TYPE], eax       ; save ret_type
; --- validate ret_type BEFORE argument evaluation and the native call:
; an unknown type must abort without ever running the callee, side effects ---
     cmp eax, [rel atom_sl_type_void]          ; VOID?
     je sl_ffi_rettype_ok                      ; yes
     cmp eax, [rel atom_sl_type_ptr]           ; PTR?
     je sl_ffi_rettype_ok                      ; yes
     cmp eax, [rel atom_sl_type_str]           ; STR?
     je sl_ffi_rettype_ok                      ; yes
     cmp eax, [rel atom_sl_type_int]           ; INT?
     jne sl_ffi_err_rettype                    ; unknown ret type: FATAL before anything runs
sl_ffi_rettype_ok:
     mov eax, [rsp + SL_FFI_FR_ARG_TYPES]      ; restore ([arg_types]) tail
     call Car                                  ; eax = ARGS: the parked value is the list CONTAINING arg_types, Car extracts the sublist
     mov [rsp + SL_FFI_FR_ARG_TYPES], eax      ; save real arg_types
     mov dword [rsp + SL_FFI_FR_ARG_COUNT], 0  ; arg count = 0
sl_ffi_arg_loop:
     mov eax, [rsp + SL_FFI_FR_ARG_TYPES]      ; eax = remaining arg_types
     test eax, eax                             ; NIL?
     jz sl_ffi_args_done                       ; yes: done evaluating args
     cmp dword [rsp + SL_FFI_FR_ARG_COUNT], SL_FFI_MAX_ARGS ; descriptor declares more than 16?
     jae sl_ffi_err_max_args                   ; yes: FATAL, silent truncation would desync the call
     call PopCons                              ; ebx = declared type atom, eax = rest of arg_types
     mov [rsp + SL_FFI_FR_ARG_TYPES], eax      ; advance arg_types
     mov ecx, ebx                              ; ecx = declared type atom
     cmp ecx, [rel atom_sl_type_int]           ; INT?, the dominant declared type, so the TYPE DISPATCH falls through here, one compare, one not-taken branch
     jne .sl_ffi_arg_not_int                   ; no: try PTR / STR
; --- INT: evaluate, NumVal, sign-extend to 64 bits. Inlined rather than shared
; --- with the PTR/STR arms so the dominant type pays no extra call/dispatch. ---
     call EvalNextArg                          ; eax = evaluated value, r12d/r14d advance as expected; overwrites ALL volatile
     call NumVal                               ; eax = raw int32, overwrites volatile
     cdqe                                      ; sign-extend to 64 bits
     jmp sl_ffi_store_arg                      ; go store, TAKEN, and required: sl_ffi_store_arg is past .sl_ffi_arg_not_int and the four marshal arms, so there is no fall-through path to it. Do not delete as redundant
.sl_ffi_arg_not_int:
     cmp ecx, [rel atom_sl_type_ptr]           ; PTR?
     je .ffi_arg_ptr                           ; yes: pointer marshal
     cmp ecx, [rel atom_sl_type_str]           ; STR?
     je .ffi_arg_ptr                           ; yes: string atoms ARE char pointers, same marshal as PTR
     mov eax, ecx                              ; eax = the OFFENDING TYPE atom. Named HERE because this is the only copy that survives, nothing has been called since `mov ecx, ebx`, but any marshal added above would destroy it
     jmp sl_ffi_err_type                       ; unknown type atom: FATAL
.ffi_arg_ptr:
     call EvalNextArg                          ; eax = evaluated value
; fall through into the PTR/STR marshal, whose entry test consumes eax
; --- PTR/STR: value -> char*, r13 + payload, strip '"' ---
; NIL, 0 -> NULL pointer, not heap base
     test eax, eax                             ; establishes ZF (NIL) and SF (bit31: cons/float) in one test
     jz sl_ffi_marshal_nil                     ; NIL -> pass NULL
     js sl_ffi_err_type                        ; bit31 set: cons index or float, FATAL
; NUMBER-typed value, e.g., raw integer as pointer : extract via NumVal
     test eax, SL_NUMBER_TAG                   ; NUMBER-typed?
     jnz sl_ffi_marshal_num_ptr                ; yes: extract integer value
     lea rax, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rax = atom's text address, the pointer value passed to the C function
     jmp sl_ffi_store_arg                      ; go store
sl_ffi_marshal_num_ptr:
; --- NUMBER as pointer: extract integer, sign-extend to 64 bits ---
     call NumVal                               ; eax = raw int32, overwrites volatile
     cdqe                                      ; sign-extend to 64 bits
     jmp sl_ffi_store_arg                      ; go store
sl_ffi_marshal_nil:
     xor eax, eax                              ; rax = NULL, 0 for C NULL pointer
sl_ffi_store_arg:
     mov ecx, [rsp + SL_FFI_FR_ARG_COUNT]      ; ecx = arg count, reload after NumVal/EvalNextArg
     mov [rsp + rcx*SL_FFI_QWORD_SCALE + SL_FFI_FR_ARGS], rax ; store raw 64-bit arg value
     inc ecx                                   ; advance the count in the register we already have
     mov [rsp + SL_FFI_FR_ARG_COUNT], ecx      ; publish the new count, plain store; the slot is 4-aligned on an `and rsp,-16` frame, so no split access
     jmp sl_ffi_arg_loop                       ; continue
sl_ffi_args_done:
; arg eval finished: no more EvalNextArg, so rsi/rdi are free again
     mov edi, [rsp + SL_FFI_FR_ARG_COUNT]      ; edi = arg count for frame setup
     lea rsi, [rsp + SL_FFI_FR_ARGS]           ; rsi = args base, survives the call-frame sub below
; --- 5. Set up Win64 call frame ---
; Frame = shadow,32 + max,0, arg_count-4 *8, 16-byte aligned
     mov eax, edi                              ; eax = arg count
     sub eax, 4                                ; args beyond register-passed?
     jns sl_ffi_has_stack                      ; yes: calculate stack size
     xor eax, eax                              ; no: zero stack args
sl_ffi_has_stack:
     lea eax, [rax*SL_FFI_QWORD_SCALE + SL_FFI_FRAME_ALIGN_BIAS] ; n-4 *8 + shadow,32 + 15, bias folds the round-up addend into the LEA displacement
     and eax, -16                              ; round up to the next 16-byte boundary
; frame RSP is 0 mod 16 by the prologue's `and rsp,-16`, and rax is a
; 16-multiple, so the call site below is aligned with no further fixup
     mov [rsp + SL_FFI_FR_CALLFRAME], eax      ; save frame size for deallocation
     sub rsp, rax                              ; allocate native call frame, RSP 0 mod 16 at call site
; --- 6. Copy stack args, index 4..N-1 to [rsp + 32 +, i-4 *8] ---
     cmp edi, 4                                ; any stack args?
     jbe sl_ffi_load_regs                      ; no: skip copy
     mov ecx, 4                                ; ecx = first stack arg index, edi > 4 guaranteed here, so 4 < edi, enter the copy body directly
sl_ffi_copy_stack:
     mov rax, [rsi + rcx*SL_FFI_QWORD_SCALE]   ; rax = arg value, ecx >= 4, rsi = args base
; stack arg i goes at [rsp + 32 +, i-4 *8] = [rsp + 32 + i*8 - 32] = [rsp + i*8]
     mov [rsp + rcx*SL_FFI_QWORD_SCALE], rax   ; store at [rsp + i*8]
     inc ecx                                   ; advance index
     cmp ecx, edi                              ; past last arg?
     jb sl_ffi_copy_stack                      ; no, next, fall-through = done
sl_ffi_load_regs:
     mov rcx, [rsi]                            ; arg1 -> rcx
     mov rdx, [rsi + SL_FFI_ARG2_OFF]          ; arg2 -> rdx
     mov r8, [rsi + SL_FFI_ARG3_OFF]           ; arg3 -> r8
     mov r9, [rsi + SL_FFI_ARG4_OFF]           ; arg4 -> r9
; fall through into the call, zero taken branches on this path
     xor eax, eax                              ; AL = 0: no vector registers used, INT/PTR/STR all marshal to GPRs and the stack
     call qword [rsi + SL_FFI_PB_FUNC_PTR]     ; call native function, RSP 0 mod 16 here
; rax = C return value; all other volatile registers are garbage now
; --- deallocate call frame, use ecx, NOT eax, to preserve rax return value ---
     mov ecx, [rsi + SL_FFI_PB_CALLFRAME]      ; ecx = frame size, via args base, rcx was garbage anyway
     add rsp, rcx                              ; restore stack, RSP back to the frame base
; --- 7. Box return value based on ret_type ---
     mov ecx, [rsi + SL_FFI_PB_RET_TYPE]       ; ecx = ret_type atom, via args base
     cmp ecx, [rel atom_sl_type_void]          ; VOID?
     je sl_ffi_ret_void                        ; yes: return NIL
     cmp ecx, [rel atom_sl_type_ptr]           ; PTR?
     je sl_ffi_ret_str                         ; yes: intern as string
     cmp ecx, [rel atom_sl_type_str]           ; STR?
     je sl_ffi_ret_str                         ; yes: intern as string
; ret_type is INT, validated pre-call: reaching here means exactly INT, no dead error branch
; --- INT return ---
; rax = 64-bit integer return from the callee
     cdqe                                      ; rax = the exact return, a C int is 32-bit; bits 63:32 are undefined per the ABI and deliberately dropped
     NUM_RANGE_CHECK rax, "CallLibrary: INT return does not fit the 30-bit payload range" ; unrepresentable -> FATAL naming the wrapped result, never a silent negative
     call MakeNum                              ; eax = tagged NUMBER
     jmp sl_ffi_epilogue                       ; shared frame teardown
sl_ffi_ret_str:
; --- PTR/STR return: intern the C string as a SectorLogo atom ---
     test rax, rax                             ; NULL pointer?
     jz sl_ffi_ret_void                        ; yes: return NIL
; copy C string to concat_buf with type byte prefix for Intern
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type prefix
     lea rdi, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; dest for content
     mov rsi, rax                              ; source = C string return
     lea r8, [rel sl_concat_buf + SL_CONCAT_BUF_LAST] ; last writable offset
.sl_ffi_copy_loop:
     cmp rdi, r8                               ; room for this byte, and the trailing NUL ?
     jae .sl_ffi_toolong                       ; no: FATAL before any store
     lodsb                                     ; al = [rsi++]
     stosb                                     ; [rdi++] = al, in bounds: rdi < r8 proven above
     test al, al                               ; NUL terminator?
     jnz .sl_ffi_copy_loop                     ; no: keep copying
     jmp .sl_ffi_copy_done                     ; yes: NUL stored, done
.sl_ffi_toolong:
     FATAL "SectorLogo: CallLibrary string return too long" ; exceeds the 2 MiB concat buffer, checked BEFORE every store, so no partial overrun is possible
.sl_ffi_copy_done:
     lea rsi, [rel sl_concat_buf]              ; source for Intern, type byte first
     call Intern                               ; eax = interned atom
     jmp sl_ffi_epilogue                       ; shared frame teardown
sl_ffi_ret_void:
     xor eax, eax                              ; return NIL, 0
; fall through to shared epilogue, RSP == frame base on every path here
sl_ffi_epilogue:
     mov rsp, rbp                              ; restore pre-alignment RSP from the anchor
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbp                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
sl_ffi_err_nil:
     FATAL "SectorLogo: CallLibrary descriptor is NIL" ; nil descriptor: abort
sl_ffi_err_resolve:
     FATAL 'SectorLogo: CallLibrary failed to resolve function (quote the module "SELF"/"HOST" uppercase AND every proc name containing _ or ., the lexer splits bare names at delimiters and lowercases them, and LoadLibraryA/GetProcAddress are case-sensitive)' ; resolve failure: abort
sl_ffi_err_rettype:
     FATAL "SectorLogo: CallLibrary return type must be integer, pointer, text, or void" ; validated BEFORE the call: the callee never runs
sl_ffi_err_type:
     FATAL_VALUE "CallLibrary type mismatch", eax ; abort, naming the offending value in eax, see the note above for the two entries
sl_ffi_err_max_args:
     FATAL "SectorLogo: CallLibrary exceeds 16 arguments" ; descriptor declares more args than slots: abort
%undef SL_FFI_FR_FUNC_PTR
%undef SL_FFI_FR_DESC
%undef SL_FFI_FR_RET_TYPE
%undef SL_FFI_FR_ARG_TYPES
%undef SL_FFI_FR_ARG_COUNT
%undef SL_FFI_FR_CALLFRAME
%undef SL_FFI_FR_ARGS
%undef SL_FFI_FR_TOTAL
%undef SL_FFI_FR_SLACK
%undef SL_FFI_PB_FUNC_PTR
%undef SL_FFI_PB_RET_TYPE
%undef SL_FFI_PB_CALLFRAME
