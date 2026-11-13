; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define VW_PE_W 0                              ; const float*, patch-embed projection weight
%define VW_PE_B 8                              ; const float*, patch-embed projection bias
%define VW_POS_EMBED 16                        ; const float*, position-embedding table
%define VW_BLK_NORM1_GAMMA 24                  ; const float*, per-layer pre-attention norm gain
%define VW_BLK_NORM1_BETA 32                   ; const float*, per-layer pre-attention norm bias
%define VW_BLK_QKV_W 40                        ; const float*, per-layer fused QKV weight
%define VW_BLK_QKV_B 48                        ; const float*, per-layer fused QKV bias
%define VW_BLK_PROJ_W 56                       ; const float*, per-layer attention output projection weight
%define VW_BLK_PROJ_B 64                       ; const float*, per-layer attention output projection bias
%define VW_BLK_NORM2_GAMMA 72                  ; const float*, per-layer pre-MLP norm gain
%define VW_BLK_NORM2_BETA 80                   ; const float*, per-layer pre-MLP norm bias
%define VW_BLK_FC1_W 88                        ; const float*, per-layer MLP up-projection weight
%define VW_BLK_FC1_B 96                        ; const float*, per-layer MLP up-projection bias
%define VW_BLK_FC2_W 104                       ; const float*, per-layer MLP down-projection weight
%define VW_BLK_FC2_B 112                       ; const float*, per-layer MLP down-projection bias
%define VW_MG_NORM_GAMMA 120                   ; const float*, projector pre-norm gain
%define VW_MG_NORM_BETA 128                    ; const float*, projector pre-norm bias
%define VW_MG_FC1_W 136                        ; const float*, projector first projection weight
%define VW_MG_FC1_B 144                        ; const float*, projector first projection bias
%define VW_MG_FC2_W 152                        ; const float*, projector second projection weight
%define VW_MG_FC2_B 160                        ; const float*, projector second projection bias
%define VW_DS_NORM_GAMMA 168                   ; const float*, DeepStack per-tap norm gain
%define VW_DS_NORM_BETA 176                    ; const float*, DeepStack per-tap norm bias
%define VW_DS_FC1_W 184                        ; const float*, DeepStack per-tap first projection weight
%define VW_DS_FC1_B 192                        ; const float*, DeepStack per-tap first projection bias
%define VW_DS_FC2_W 200                        ; const float*, DeepStack per-tap second projection weight
%define VW_DS_FC2_B 208                        ; const float*, DeepStack per-tap second projection bias
%define SIZEOF_VIT_WEIGHTS 216                 ; total struct size, 27 * 8
%define VIT_PROJ_HDR_BYTES 8                   ; projector blob header size, bytes skipped before weight/bias data
%define VK_ARG5 48                             ; 5th stack arg at rbp+48
%define VK_ARG6 56                             ; 6th stack arg at rbp+56
%define VK_ARG7 64                             ; 7th stack arg at rbp+64
%define VK_ARG8 72                             ; 8th stack arg at rbp+72
; -------- raw RGB pixel layout, 8-bit RGB, 3 bytes per pixel, a file-format invariant,
; -------- not a model constant: it names the caller's buffer, never the weights --------
%define VIT_RAW_RGB_CHANS 3                    ; bytes per pixel in a .raw buffer
%define VIT_RAW_G_OFF 1                        ; G channel byte offset within a pixel
%define VIT_RAW_B_OFF 2                        ; B channel byte offset within a pixel
; -------- Float scale for SIB addressing --------
%define VK_F4 4                                ; sizeof,float = 4 bytes
; -------- Byte offsets for 8-float head_dim chunks, VK_VOFF_* = V aggregation, VK_QK_BLK* = Q/K dot product --------
%define VK_VOFF_0 0                            ; V[j][0..7] byte offset, first 8 floats
%define VK_VOFF_1 32                           ; V[j][8..15] byte offset, second 8 floats
%define VK_VOFF_2 64                           ; V[j][16..23] byte offset, third 8 floats
%define VK_VOFF_3 96                           ; V[j][24..31] byte offset, fourth 8 floats
%define VK_VOFF_4 128                          ; V[j][32..39] byte offset, fifth 8 floats
%define VK_VOFF_5 160                          ; V[j][40..47] byte offset, sixth 8 floats
%define VK_VOFF_6 192                          ; V[j][48..55] byte offset, seventh 8 floats
%define VK_VOFF_7 224                          ; V[j][56..63] byte offset, eighth 8 floats
%define VK_8F_BYTES 32                         ; byte size of one 8-float YMM block, 8 * sizeof,float
%define VK_QK_BLK0 0                           ; Q/K dot-product block 0 byte offset, elements 0..7
%define VK_QK_BLK1 32                          ; Q/K dot-product block 1 byte offset, elements 8..15
%define VK_QK_BLK2 64                          ; Q/K dot-product block 2 byte offset, elements 16..23
%define VK_QK_BLK3 96                          ; Q/K dot-product block 3 byte offset, elements 24..31
%define VK_QK_BLK4 128                         ; Q/K dot-product block 4 byte offset, elements 32..39
%define VK_QK_BLK5 160                         ; Q/K dot-product block 5 byte offset, elements 40..47
%define VK_QK_BLK6 192                         ; Q/K dot-product block 6 byte offset, elements 48..55
%define VK_QK_BLK7 224                         ; Q/K dot-product block 7 byte offset, elements 56..63
%define VK_VOFF_8 256                          ; V[j][64..71] byte offset, ninth 8-float block, head_dim=72 only
%define VK_QK_BLK8 256                         ; Q/K dot-product block 8 byte offset, tail 8 floats, head_dim=72 only
