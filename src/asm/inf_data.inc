; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; .rdata, not .data : these are compile-time key constants, never written, and
; .rdata keeps the string bytes out of .text where Zydis would decode them as
; instructions in a crash dump.
     section .rdata
inf_key_config:
     db "__sys_config", 0                      ; MicroDB key for config struct
inf_key_model_weights:
     db "__sys_model_weights", 0               ; MicroDB key for model weights blob
inf_key_weight_meta:
     db "__sys_weight_meta", 0                 ; MicroDB key for weight metadata
inf_key_mem_layout:
     db "__sys_mem_layout", 0                  ; MicroDB key for memory layout
inf_key_tokenizer_hdr:
     db "__sys_tokenizer_hdr", 0               ; MicroDB key for tokenizer header
inf_key_vocab_bulk:
     db "__sys_vocab_bulk", 0                  ; MicroDB key for vocab bulk data
inf_key_merge_scores:
     db "__sys_merge_scores", 0                ; MicroDB key for merge scores
inf_key_template:
     db "__sys_template", 0                    ; MicroDB key for prompt template
inf_key_template_sys:
     db "__sys_template_sys", 0                ; MicroDB key for system template
inf_key_stop_token_ids:
     db "__sys_stop_token_ids", 0              ; MicroDB key for stop token ID array
inf_key_rms_eps:
     db "__sys_rms_eps", 0                     ; MicroDB key for RMS norm epsilon, per-model
inf_key_vision_config:
     db "__sys_vision_config", 0               ; MicroDB key for expanded ViT vision config blob, 744 bytes, 93 qwords; all VLMs
inf_key_vis_prompt_fmt:
     db "__sys_vision_prompt_fmt", 0           ; MicroDB key for vision prompt format blob, prefix\0marker\0suffix\0
inf_key_template_img:
     db "__sys_template@img", 0                ; MicroDB key for image-turn prompt template, optional; SmolVLM/Idefics3
inf_key_template_think:
     db "__sys_template@think", 0              ; MicroDB key for thinking user-turn template, optional; Qwen3/MiniCPM5 only
inf_key_template_sys_think:
     db "__sys_template_sys@think", 0          ; MicroDB key for thinking system+user template, optional; Qwen3/MiniCPM5 only
inf_key_template_multi:
     db "__sys_template@multi", 0              ; MicroDB key for inter-turn, continuation template, optional; same fallback as @img
inf_key_template_multi_think:
     db "__sys_template@multi@think", 0        ; MicroDB key for inter-turn thinking template, optional, same fallback
inf_key_img_newline:
     db "__sys_image_newline_emb", 0           ; MicroDB key for image newline embedding row, optional
