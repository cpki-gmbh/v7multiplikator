; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

sl_console_24bit:
     cmp dword [rel sl_console_state], 1       ; already probed ON?
     je sl_c24_true                            ; yes: return TRUE
     cmp dword [rel sl_console_state], 2       ; already probed OFF?
     je sl_c24_nil                             ; yes: return NIL
; ---- resolve stdout handle: shared sentinel gate, shared cache, SL_RESOLVE_STDOUT ----
     SL_RESOLVE_STDOUT sl_c24_mode, sl_c24_off ; shared sentinel gate + cache, macro emits the sl_c24_mode label; redirected output -> sl_c24_off, which the caller must define
; ---- read the current console mode ----
     mov rcx, [rel sl_hStdOut]                 ; hConsole from the shared cache
     lea rdx, [rel sl_conout_mode]             ; &mode scratch, dedicated, the state cache is never transient garbage
     CORRECT_RSP                               ; align for the console-vs-pipe mode probe
     API_CALL [rel __imp_GetConsoleMode], rcx, rdx ; fails on redirected pipes
     RESTORE_RSP                               ; undo the probe
     test eax, eax                             ; console handle?
     jz sl_c24_off                             ; no, redirected : state=2, NIL
; ---- enable VT processing, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING ----
     mov edx, [rel sl_conout_mode]             ; current mode
     or edx, SL_ENABLE_VT                      ; set the VT bit
     mov rcx, [rel sl_hStdOut]                 ; hConsole from the shared cache
     CORRECT_RSP                               ; align for the VT-enabling mode write
     API_CALL [rel __imp_SetConsoleMode], rcx, rdx ; the call that enables VT
     RESTORE_RSP                               ; undo the probe
     test eax, eax                             ; VT enabled?
     jz sl_c24_off                             ; no: state=2, NIL
     mov dword [rel sl_console_state], 1       ; cache: VT on
sl_c24_true:
     mov eax, [rel atom_sl_t]                  ; TRUE atom
     ret                                       ; return TRUE
sl_c24_off:
     mov dword [rel sl_console_state], 2       ; cache: VT off
sl_c24_nil:
     xor eax, eax                              ; eax = NIL, 0, never return eax by accident: cold-failure arrives with SetConsoleMode's 0, and the cached-OFF early-out arrives with an untouched entry value, the memory compare writes flags only.NIL by contract, not by accident.
     ret                                       ; return NIL
