; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; v7_threading.asm -- Work-stealing proxy DLL for llama.cpp matmul, NASM port
; THIS MUST PORT matmul_refactor_v7.cpp -- that is the authoritative reference.
; RULE: Every single assembler instruction MUST have a comment -- this is enforced.
; ============================================================================

; All code is in the master v7.asm file
%include "v7.asm"

%ifdef V7_LINUX
     section .tbss nobits alloc write tls align=4 ; per-thread Linux TID cache
     global v7_linux_tid                       ; make NASM emit the ELF TLS relocation target
v7_linux_tid:
     resd 1                                    ; zero until the first engine entry on this thread
     section .note.GNU-stack noalloc noexec nowrite ; request a non-executable process stack
%endif
