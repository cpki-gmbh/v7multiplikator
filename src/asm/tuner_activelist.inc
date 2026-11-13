; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; tuner_activelist.inc, active-shape registry for the in-process cycle tuner

; The dispatcher records every TUNABLE shape, packed [d|n] qword, same layout
; as the shape-block headers here, on its first cold dispatch, a shape whose
; cold dispatch fell back to the [default] block has no per-shape config
; block, the dispatcher FATALs on it and it never enters the registry. The
; cycle tuner steps one machine per registered slot, so it only ever runs
; over shapes that REALLY dispatch in the loaded model, the embedding-gather
; orientation "151936x1024", a row gather, never a matmul, is absent from
; the registry and can never be staged, stalled on, or rotated to.
; ============================================================================

     section .bss align=64
     global g_active_shapes                    ; export: the registry, dispatcher writes, cycle_tuner_main reads
g_active_shapes:
     resq 64                                   ; 512 bytes: 64 packed [d|n] entries; BSS zeroing is the
; zero sentinel, and every append moves it forward one slot. 64 is a DESIGNED
; CEILING far above the shipped shape universe, ~45 supported resolutions /
; 48 INI sections, the unbounded sentinel walk is intentional fast code,
; not a missing bound.
