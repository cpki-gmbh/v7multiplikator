; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
; ============================================================================
; Named offset constants, replaces all raw [reg+N] patterns for checker compliance
; ============================================================================
; v7d_fused_f32, extra caller stack args, rbp frame
; 5th arg is d at [rbp+48], read via STUB_FEAT_DIM, 48; forwarded through STUB_PASS_ND
%define FUSED_ARG_NORM_WEIGHTS 56              ; 6th caller arg: norm_weights, float*
; v7d_swiglu / v7d_swiglu_q8, extra caller stack args
%define SWIGLU_ARG_UP_W 56                     ; 6th caller arg: up_w, QuantizedTensor*
     SF_START v7d_fused_f32                    ; Fused F32 matmul+RMSNorm+RoPE entry
     SF_COMMIT                                 ; Commit fused_f32 stack frame
     INT3_DBG_STR "v7d_fused_f32:entry"        ; Debug trace: fused_f32 entry
     sub rsp, STUB_TENSOR_PAD                  ; Extra padding for fake tensor descriptors, up to rsp+STUB_TENSOR_PAD-1
     mov rax, [rbp + FUSED_ARG_NORM_WEIGHTS]   ; norm_weights from stack arg 6
     mov [fused_norm_weights], rax             ; set norm_weights pointer
; Pass-through stack args: n and d for matrix_multiplication_mode
     STUB_PASS_ND                              ; forward n, r9d + d to the callee's stack args
; Build fake input tensor at rsp+56: {float* q, float* s=NULL}
     mov rdx, [rdx]                            ; dereference: rdx = xt->x, float* activations
     mov [rsp + STUB_TENSOR_INPUT_Q], rdx      ; fake_input.q = x, float* activations
     mov qword [rsp + STUB_TENSOR_INPUT_S], 0  ; fake_input.s = NULL
     FUSED_MM_TAIL MATMUL_FUSED_F32            ; weight build + arg setup + dispatch + epilogue
     SF_START v7d_swiglu                       ; SwiGLU F32xQ8 fused entry
     SF_COMMIT                                 ; Commit swiglu stack frame
     sub rsp, STUB_TENSOR_PAD                  ; Extra padding for fake tensor descriptors, up to rsp+STUB_TENSOR_PAD-1
; Build fake input tensor at rsp+56: {float* q, float* s=NULL}
     mov rdx, [rdx]                            ; dereference: rdx = x->data, float* activations
     mov [rsp + STUB_TENSOR_INPUT_Q], rdx      ; fake_input.q = x, float* activations
     mov qword [rsp + STUB_TENSOR_INPUT_S], 0  ; fake_input.s = NULL
     mov rax, [rbp + SWIGLU_ARG_UP_W]          ; rax = up_w, QuantizedTensor*
     mov r10, [rax]                            ; r10 = up_w->quantized_data, r10 scratch: rcx stays = xout for the tail
     mov [swiglu_up_weight_q], r10             ; set up weight quantized pointer
     mov r10, [rax + TENSOR_S_OFF]             ; r10 = up_w->scales, float*
     mov [swiglu_up_weight_s], r10             ; set up weight scales pointer
; Pass-through stack args: n and d for matrix_multiplication_mode
     STUB_PASS_ND                              ; forward n, r9d + d to the callee's stack args
     FUSED_MM_TAIL MATMUL_SWIGLU               ; weight build + arg setup + dispatch + epilogue
     SF_START v7d_swiglu_q8                    ; SwiGLU Q8xQ8 fused entry
     SF_COMMIT                                 ; Commit swiglu_q8 stack frame
     sub rsp, STUB_TENSOR_PAD                  ; Extra padding for fake tensor descriptors, up to rsp+STUB_TENSOR_PAD-1
; Build fake input tensor at rsp+56: {int8_t* q, float* s}
     mov rax, [rdx]                            ; rax = x->quantized_data, int8_t*
     mov [rsp + STUB_TENSOR_INPUT_Q], rax      ; fake_input.q = xq, int8_t* activations
     mov rax, [rdx + TENSOR_S_OFF]             ; rax = x->scales, float*
     mov [rsp + STUB_TENSOR_INPUT_S], rax      ; fake_input.s = xs, float* scales
     mov rax, [rbp + SWIGLU_ARG_UP_W]          ; rax = up_w, QuantizedTensor*
     mov r10, [rax]                            ; r10 = up_w->quantized_data, r10 scratch: rcx stays = xout for the tail
     mov [swiglu_up_weight_q], r10             ; set up weight quantized pointer
     mov r10, [rax + TENSOR_S_OFF]             ; r10 = up_w->scales, float*
     mov [swiglu_up_weight_s], r10             ; set up weight scales pointer
; Pass-through stack args: n and d for matrix_multiplication_mode
     STUB_PASS_ND                              ; forward n, r9d + d to the callee's stack args
     FUSED_MM_TAIL MATMUL_SWIGLU_Q8            ; weight build + arg setup + dispatch + epilogue
     global v7s_rope_init                      ; Export: initialize RoPE sin/cos tables
v7s_rope_init:
     push rbp                                  ; save caller frame pointer, entry parity 8 mod 16 -> 0 mod 16 for the mdb_free_mem calls
     mov rbp, rsp                              ; set frame pointer
     INT3_DBG_STR "v7s_rope_init"              ; Debug trace: rope_init entry
     mov [fused_head_dim], ecx                 ; save head dimension for table builder
     INT3_DBG_STR "ri:hd_set"                  ; Debug trace: head_dim saved
     cmp dword [fused_rope_owned], 0           ; tables internally owned?
     je .rinit_no_free                         ; no, external pointers, skip free entirely
     mov dword [fused_rope_owned], 0           ; clear resource holder FIRST so the destroy path never sees owned pointers that are already released
     mov rcx, [fused_rope_cos]                 ; arg1: old cos table, owned, non-NULL
     call mdb_free_mem                         ; release old cos, FATALs internally on an invalid pointer
     mov qword [fused_rope_cos], 0             ; zero-pin cos the moment its free succeeds
     mov rcx, [fused_rope_sin]                 ; arg1: old sin table, owned, non-NULL
     call mdb_free_mem                         ; release old sin, FATALs internally on an invalid pointer
     mov qword [fused_rope_sin], 0             ; zero-pin sin the moment its free succeeds
.rinit_no_free:
     pop rbp                                   ; restore caller frame pointer
     jmp rope_build_internal_tables            ; tail-call table builder, always builds fresh
     global v7s_set_fused_rope                 ; Export: set RoPE position for fused kernel
v7s_set_fused_rope:
     mov [fused_pos], ecx                      ; save sequence position
     mov [fused_head_dim], edx                 ; save head dimension, idempotent
     ret                                       ; return to caller
     SF_START v7d_q8_fused                     ; Quantize-once + Q8 matmul entry
     SF_QWORD qf_save_xout                     ; xout saved across v7d_quantize call
     SF_DWORD qf_save_n                        ; n saved across call
     SF_COMMIT                                 ; Commit q8_fused stack frame
     sub rsp, STUB_TENSOR_PAD                  ; Extra padding for fake tensor descriptors
; save args before calling v7d_quantize, overwrites all volatile registers
; rdx, xt->x after deref and r9d, n stay active in registers to the call, nothing writes them in between
     mov qf_save_xout, rcx                     ; save xout, float* output
     mov rdx, [rdx]                            ; dereference: rdx = xt->x, float*, stays as quantize arg2
     mov qf_save_n, r9d                        ; save n, input elements, still needed post-call
     mov rax, [r8]                             ; weight->quantized_data
     mov [rsp + STUB_TENSOR_WEIGHT_Q], rax     ; fake_weight.q
     mov rax, [r8 + TENSOR_S_OFF]              ; weight->scales
     mov [rsp + STUB_TENSOR_WEIGHT_S], rax     ; fake_weight.s
; Quantize args + input fake tensor reusing the SAME leas:
     lea rcx, [rel g_qf_scratch_q]             ; arg1: out_q, int8 scratch buffer
     mov [rsp + STUB_TENSOR_INPUT_Q], rcx      ; fake_input.q = scratch q, reuse the arg1 lea
     lea r8, [rel g_qf_scratch_s]              ; arg3: out_s, r8 repurposed AFTER the weight build
     mov [rsp + STUB_TENSOR_INPUT_S], r8       ; fake_input.s = scratch s, reuse the arg3 lea
     call v7d_quantize                         ; quantize once, dispatches across pool thread pool, rdx = x, r9d = n untouched
; Step 4: Dispatch plain MATMUL_Q8 with pre-quantized input
; matrix_multiplication_mode,mode, xout, &input, &weight, n, d
     mov rdx, qf_save_xout                     ; xout -> rdx, 2nd arg
     lea r8, [rsp + STUB_TENSOR_INPUT_Q]       ; &fake_input -> r8, 3rd arg
     lea r9, [rsp + STUB_TENSOR_WEIGHT_Q]      ; &fake_weight -> r9, 4th arg
     mov ecx, MATMUL_Q8                        ; mode, 1st arg, plain Q8, not Q8_FUSED
     mov eax, qf_save_n                        ; n
     mov [rsp + APICALL_ARG5], eax             ; arg5: n, input elements, stack-passed
     mov eax, [rbp + STUB_FEAT_DIM]            ; d, caller stack arg, rbp-stable across calls
     mov [rsp + APICALL_ARG6], eax             ; arg6: d, output dimension, stack-passed
     call matrix_multiplication_mode           ; dispatch with plain Q8 mode
     add rsp, STUB_TENSOR_PAD                  ; Remove extra padding
     SF_END                                    ; Epilogue: restore rbp and return
; Local %define cleanup, house convention: %undef every file-local define
%undef FUSED_ARG_NORM_WEIGHTS
%undef SWIGLU_ARG_UP_W
