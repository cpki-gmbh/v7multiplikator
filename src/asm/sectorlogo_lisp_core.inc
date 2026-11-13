; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define CONS_CDR_OFF 4                         ; cdr field offset within cons cell
%define CONS_BASE_OFF SL_ATOM_AREA_SIZE        ; byte offset from heap base to cons area
%define CONS_SCALE 8                           ; bytes per cons cell, 2 dwords

%macro FATAL_VALUE 2
     jmp %%after_msg                           ; jump over the embedded message string
%%fv_msg:
     db "SectorLogo: ", %1, 0                  ; message text, module prefix + caller text
%%after_msg:
     mov eax, %2                               ; tagged offender into the helper's argument register
     lea rcx, [rel %%fv_msg]                   ; static message pointer, RIP-relative
     call sl_fatal_value                       ; never returns, parity-agnostic: it self-aligns
%endmacro
; ============================================================================
; MakeNum,eax = raw int32 -> eax = tagged NUMBER value
; Wraps a raw integer into the NUMBER tag representation, bit30 set .
; Payload is 30-bit two's-complement, ~ +/-536 million, sufficient for
; token ids and small arithmetic. Callers that COMPUTE an integer must range-check
; to detect the overflow; an unchecked tail turns a positive sum into a negative
; number.
; ============================================================================
MakeNum:
     and eax, SL_PAYLOAD_MASK                  ; mask to 30-bit payload
     or eax, SL_NUMBER_TAG                     ; set NUMBER tag bit
     ret                                       ; return tagged value in eax
; ============================================================================
; NumVal,eax = tagged NUMBER -> eax = sign-extended raw int32
; Extracts the 30-bit payload and sign-extends bit29 into bits 31:30.
; ============================================================================
NumVal:
     test eax, eax                             ; bit31 set, CONS or FLOAT ?
     js .not_a_number                          ; yes: not a NUMBER where a number was expected
     test eax, SL_NUMBER_TAG                   ; bit30 set, NUMBER ?, ATOM/NIL have both clear and fail here
     jz .not_a_number                          ; no: CONS/ATOM/NIL where a number was expected
     shl eax, 2                                ; shifts out tag bits 31:30, aligns sign bit 29 to bit31
     sar eax, 2                                ; arithmetic shift back: SAR propagates bit31 into bits 31:30, branchless sign-extend
     ret                                       ; return raw int32 in eax, rcx/rdx untouched, nothing to preserve
.not_a_number:
     FATAL_VALUE "a NUMBER was required (arithmetic, index, count, or key operand)", eax ; name the offending value
Car:
     lea eax, [rax - SL_CONS_TAG]              ; strip the tag AND fold the band check: eax = index for CONS, out-of-band for every other tag
     cmp eax, SL_FLOAT_TAG - SL_CONS_TAG       ; 0x40000000 = band width: below = cell, at/above = NIL/atom/NUMBER/FLOAT
     jae .nil                                  ; not a cell -> NIL
     mov eax, [r13 + rax*CONS_SCALE + CONS_BASE_OFF] ; load car field, address folded into the SIB+disp32: one uop, rax dead after
     ret                                       ; return car, no taken branch on the cell path
.nil:
     xor eax, eax                              ; not a cons / FLOAT: return NIL, shared exit
     ret                                       ; return NIL to caller
; ============================================================================
; Cdr,eax = tagged value -> eax = cdr field or NIL
; Same two-compare guard as Car: bit31 alone is not proof of a cons cell.
; ============================================================================
Cdr:
     lea eax, [rax - SL_CONS_TAG]              ; strip the tag AND fold the band check: eax = index for CONS, out-of-band for every other tag
     cmp eax, SL_FLOAT_TAG - SL_CONS_TAG       ; 0x40000000 = band width: below = cell, at/above = NIL/atom/NUMBER/FLOAT
     jae .nil                                  ; not a cell -> NIL
     mov eax, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; load cdr field, address folded into the SIB+disp32
     ret                                       ; return cdr, no taken branch on the cell path
.nil:
     xor eax, eax                              ; not a cons / FLOAT: return NIL, shared exit
     ret                                       ; return NIL to caller
; ============================================================================
; PopCons,eax = tagged value -> ebx = car, eax = cdr
; Fused Car+Cdr: one tag test + one address derivation for the pair every
; stream-advance performs. NIL/FLOAT/atom input: both results NIL, same
; contract as calling Car then Cdr separately.Writes eax + ebx only.
; ============================================================================
PopCons:
     lea eax, [rax - SL_CONS_TAG]              ; strip the tag AND fold the band check: eax = index for CONS, out-of-band for every other tag
     cmp eax, SL_FLOAT_TAG - SL_CONS_TAG       ; 0x40000000 = band width: below = cell, at/above = NIL/atom/NUMBER/FLOAT
     jae .pc_nil                               ; not a cell -> NIL
     mov ebx, [r13 + rax*CONS_SCALE + CONS_BASE_OFF] ; ebx = car field, address folded; eax index still live
     mov eax, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; eax = cdr field, folded; reads the index before overwriting it
     ret                                       ; return car in ebx, cdr in eax
.pc_nil:
     xor eax, eax                              ; cdr = NIL
     xor ebx, ebx                              ; car = NIL
     ret                                       ; return both NIL
; ============================================================================
; Cons,eax = car, edx = cdr -> eax = tagged cons cell reference
; Dual-path allocator: freelist pop first, bump fallback, GC on exhaustion.
; FATAL only if GC cannot reclaim any cells.
; ============================================================================
Cons:
     mov ecx, [rel sl_gc_freelist]             ; check freelist first
     test ecx, ecx                             ; freelist empty?
     jz .cons_bump                             ; yes: try bump allocator
; --- freelist pop: reuse a reclaimed cell ---
     btr ecx, SL_CONS_BIT                      ; strip tag bit -> raw cell index, 4B vs 6B and
     lea r10, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF] ; address of freelist cell
     mov r11d, [r10 + CONS_CDR_OFF]            ; next freelist node, tagged or 0
     mov [rel sl_gc_freelist], r11d            ; update freelist head
     jmp .cons_fill                            ; populate car/cdr
.cons_bump:
     mov ecx, [rel sl_cons_free]               ; load bump cursor
     cmp ecx, SL_CONS_CELL_COUNT               ; heap exhausted?
     jae .cons_gc                              ; yes: trigger garbage collection
     inc dword [rel sl_cons_free]              ; advance bump cursor
     lea r10, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF] ; address of new cell
.cons_fill:
     mov [r10], eax                            ; store car
     mov [r10 + CONS_CDR_OFF], edx             ; store cdr
     lea eax, [rcx + SL_CONS_TAG]              ; tagged cons ref = index | bit31, disp32 sign-extension is truncated away by the 32-bit destination, identical result, one instruction
     ret                                       ; return tagged cons ref
.cons_gc:
     push rax                                  ; save car across GC call
     push rdx                                  ; save cdr across GC call
     sub rsp, 8                                ; parity pad: 2 pushes leave RSP 8 mod 16; the GC_Collect call needs 0 mod 16
     call GC_Collect                           ; reclaim dead cons cells
     add rsp, 8                                ; drop alignment pad
     pop rdx                                   ; restore cdr
     pop rax                                   ; restore car
     cmp dword [rel sl_gc_freelist], 0         ; freelist has cells now?
     jnz Cons                                  ; yes: retry from freelist top
     FATAL "SectorLogo: Cons heap overflow (GC exhausted)" ; unrecoverable: no reclaimable cells
sl_stack_overflow:
     FATAL "SectorLogo: private stack exhausted (runaway recursion)" ; watermark trip, recursion outran the 64 MiB block
