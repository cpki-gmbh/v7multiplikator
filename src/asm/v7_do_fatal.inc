; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; v7_do_fatal, script-reachable fatal for SectorLogo error paths that must
; stop the engine via FATAL instead of silently continuing, error hiding is a
; bug: a script that detects an unrecoverable state calls this and the
; process dies through the standard FATAL path, identical to every other
; engine fatal .

; rcx = NUL-terminated message string, caller-owned, may live anywhere

; Prints the message with the standard WRITE_STR output macro, then FATALs
; with the fixed reason "do fatal was called" so the death reason always
; identifies this entry.

     section .text
     global v7_do_fatal                        ; exported: print message, then FATAL
v7_do_fatal:
     mov rdi, rcx                              ; rdi = message pointer, callee-saved: survives the ensure call inside WRITE_STR
     WRITE_STR rdi                             ; print the caller's message, WRITE_STR emits lea rdx,[rdi]
     FATAL "do fatal was called"               ; the standard engine fatal: print reason, park/terminate, never returns
