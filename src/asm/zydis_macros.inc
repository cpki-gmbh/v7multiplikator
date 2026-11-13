; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%ifndef ZYDIS_MACROS_INC
%define ZYDIS_MACROS_INC                       ; include guard, the INT3_DBG_* macro block below may only be defined once
%define ZYDIS_MAGIC 0xDEADBEEF                 ; magic value for INT3 debug breakpoint detection
%ifdef ZYDIS_TRACE

%macro ZYDIS_INT3 1-*
     int3                                      ; INT3, caught by the VEH in zydis_debug_header.h
     dd ZYDIS_MAGIC                            ; magic dword
     db %1                                     ; command byte, 0-8
%if %0 > 1
     db %2                                     ; payload
%endif
%endmacro

%macro INT3_DBG_STR 1
     ZYDIS_INT3 0                              ; command 0 = print string
     db %1, 0                                  ; null-terminated string, payload db stays HERE, not in the base macro: a braced {"a, b"} is ONE macro argument, so `ZYDIS_INT3 0, %1, 0` would hand the base macro three arguments
%endmacro

%macro INT3_DBG_REG 1
     ZYDIS_INT3 1, %1                          ; command 1 = print register, 1 data byte
%endmacro

%macro INT3_DBG_F32 1
     ZYDIS_INT3 8, %1                          ; command 8 = dump XMM register as 4 floats
%endmacro
%else

%macro INT3_DBG_STR 1
%endmacro

%macro INT3_DBG_REG 1
%endmacro

%macro INT3_DBG_F32 1
%endmacro
%endif                                         ; ZYDIS_TRACE
%endif                                         ; ZYDIS_MACROS_INC
