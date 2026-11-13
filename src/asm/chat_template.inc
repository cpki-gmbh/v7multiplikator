; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; chat_template.inc, SELECT_TPL macro + v7s_enable_thinking .
; Emitted at chat.inc's include position, between the getters and the v7s_chat anchor, include order IS the binary layout.
; Self-cleaning: the SELECT_TPL %unmacro lives at the bottom, its only user is assembled above .

; SELECT_TPL dst, variant, base, dst =, flag && variant stored ? &variant : base
; edx = thinking flag, 0/1, caller-zeroed ; r11 = volatile scratch, eax dead

%macro SELECT_TPL 3
     mov r11, [rel %3]                         ; choose the base pointer by default
     test edx, edx                             ; thinking disabled?
     jz %%publish                              ; leave the base selected
     cmp byte [rel %2], 0                      ; nonempty variant present?
     je %%publish                              ; absent: retain the base
     lea r11, [rel %2]                         ; enabled and present: use the variant
%%publish:
     mov [rel %1], r11                         ; publish effective pointer
%endmacro
     global v7s_enable_thinking                ; export thinking selector
v7s_enable_thinking:
     cmp qword [rel g_tpl_t1_base], 0          ; init cached the base pointers?
     je .think_before_init                     ; no, called before inference_create
     xor edx, edx                              ; thinking flag = 0
     test ecx, ecx                             ; caller passed nonzero?
     setne dl                                  ; dl = 1 when enabling
     mov [rel g_thinking_active], dl           ; publish the state for v7_get_think_mode_active, one init-time store
     SELECT_TPL g_tpl_t1, g_think_template, g_tpl_t1_base ; turn-1 user template
     SELECT_TPL g_tpl_sys, g_think_sys_template, g_tpl_sys_base ; system+user template
     SELECT_TPL g_tpl_multi, g_multi_think_template, g_tpl_multi_base ; turn-2+, variant = @multi@think; base = &@multi buffer
     cmp byte [rel g_multi_template], 0        ; model has any @multi at all?
     jne .think_done                           ; yes, multi selection stands
     cmp byte [rel g_multi_think_template], 0  ; @multi@think stored?
     je .multi_fallback_t1                     ; no, nothing multi-shaped exists
     cmp byte [rel g_thinking_active], 0       ; thinking on, SELECT_TPL above picked the @multi@think variant ?
     jne .think_done                           ; yes, the variant selection stands: do not discard it
.multi_fallback_t1:
     mov rax, [rel g_tpl_t1]                   ; no multi of either kind: turns 2+ reuse the turn-1 template
     mov [rel g_tpl_multi], rax                ; no @multi: turns 2+ reuse the turn-1 template
.think_done:
     ret                                       ; pointers take effect at next render
.think_before_init:
     FATAL "v7s_enable_thinking called before v7s_inference_create" ; caller bug: bases not cached yet
%unmacro SELECT_TPL 3                          ; release: file-local, no expansion outside this file
