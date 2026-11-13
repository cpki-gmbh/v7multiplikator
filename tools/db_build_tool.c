// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#define _CRT_SECURE_NO_WARNINGS
#define _CRT_NONSTDC_NO_WARNINGS
/* db_build_tool.c -- standalone offline tool to build lz4ht.db from model files.
 * usage: build_db <checkpoint.bin> [options]
 *   -thinking <0|1>   enable thinking templates (default 1)
 *   -rope-base <float> override rope_base (e.g. 10000 for phi-3)
 *   -f                force rebuild (delete existing lz4ht.db)
 *
 * this tool reads checkpoint .bin + .Tokenizer + .template files,
 * populates the v7_threading.dll hash table via ht_add, and persists via db_save.
 * it does NOT run inference, forward, sampler, or chat. */

#include <float.h>
#include <io.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _WIN32_WINNT 0x0A00
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// __int64 replaced with __int64 per rule 15

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include "zydis_debug_header.h"
#ifdef __linux__
#include <sys/mman.h>
#include "ini.h"
#endif

/* ---------------------------------------------------------------------------
 * structs (matching runq.c / v7_lib.h layouts)
 * ------------------------------------------------------------------------- */

#define QuantizedTensor v7_q8_tensor
#define Config v7_config

/* RoPE scaling block written by export.py at header bytes 64-83 (20 bytes).
 * layout must match export.py model_export's struct.pack("ifffi"). */
typedef struct
{
	int rope_type;          /* 0=none, 1=llama3 */
	float factor;           /* llama3 low-frequency division factor (32.0) */
	float high_freq_factor; /* llama3 high-frequency boundary (4.0) */
	float low_freq_factor;  /* llama3 low-frequency boundary (1.0) */
	int orig_max_pos;       /* original_max_position_embeddings (8192) */
} rope_scaling_header;

/* header bytes 84-87: RMS norm epsilon (float32).
 * written by export.py right after the RoPE scaling block.
 * default 1e-6 (qwen3/SmolLM2) if not present in older .bin files. */
#define HEADER_NORM_EPS_OFFSET 84
#define DEFAULT_NORM_EPS 1e-6f

int GS = 64;

typedef struct
{
	QuantizedTensor *q_tokens;
	float *token_embedding_table;
	float *rms_att_weight;
	float *rms_ffn_weight;
	QuantizedTensor *wq;
	QuantizedTensor *wk;
	QuantizedTensor *wv;
	QuantizedTensor *wo;
	float *q_norm_weights;
	float *k_norm_weights;
	float *qkv_bias; /* [n_layers * (n_heads*head_dim + 2*n_kv_heads*head_dim)] fp32 -- zeros if model has no QKV bias */
	QuantizedTensor *w1;
	QuantizedTensor *w2;
	QuantizedTensor *w3;
	float *rms_final_weight;
	QuantizedTensor *wcls;
} transformer_weights;

typedef struct
{
	Config config;
	transformer_weights weights;
	float *data;
	__int64 file_size;
} transformer;

typedef struct
{
	char **vocab;
	float *merge_scores;
	int vocab_size;
	unsigned int max_token_length;
	unsigned int bos_token_id;
	unsigned int eos_token_id;
	unsigned int num_stop_ids; /* number of stop token IDs (1 = just eos_token_id) */
	unsigned int stop_ids[4];  /* stop token IDs: [0] = eos_token_id, rest = additional */
	char prompt_template[1024];
	char system_prompt_template[1024];
	char image_prompt_template[1024];        /* image-turn variant (from ".bin.template.img"; empty = use base) */
	char think_prompt_template[1024];        /* thinking user-turn variant (from ".bin.template.with-thinking"; empty = use base) */
	char think_system_prompt_template[1024]; /* thinking system+user variant (from ".bin.template.with-system-and-thinking"; empty = use base) */
	char multi_prompt_template[1024];        /* inter-turn variant, turn 2+ (from ".bin.template.multi"; empty = use base) */
	char multi_think_prompt_template[1024];  /* inter-turn thinking variant (from ".bin.template.multi.with-thinking"; empty = use @multi) */
	int is_sentencepiece;
} tokenizer;

/* MemLayout (must match runq.c exactly) */
typedef struct
{
	__int64 reserved0; // reserved (always 0, per §96)
	__int64 rs_block_bytes;
	__int64 off_x;
	__int64 off_xb;
	__int64 off_hb;
	__int64 off_hb2;
	__int64 off_xq_q;
	__int64 off_xq_s;
	__int64 off_hq_q;
	__int64 off_hq_s;
	__int64 off_q;
	__int64 off_att;
	__int64 off_logits;
	__int64 off_key_cache;
	__int64 off_value_cache;
	__int64 off_x_storage;
	__int64 off_q_storage;
	__int64 off_att_out;
	__int64 emb_table_bytes;
	__int64 vocab_ptrs_bytes;
	/* the 8 trailing config-int fields that used to live here (offsets 156..184) are
	 * deleted: every ML_* reader in the ASM stops at vocab_ptrs_bytes (152) and the
	 * config is cached from the Config struct instead -- 32 bytes of write-only payload.
	 * inf_structs.inc declares ML_SIZE 160 for the shrunken blob */
} mem_layout;

struct v7_lib dll;

/* ---------------------------------------------------------------------------
 * helpers
 * ------------------------------------------------------------------------- */

void *db_alloc(__int64 size)
{
	#ifdef __linux__
	void *p = size > 0 ? calloc(1, (size_t)size) : NULL;
#else
	void *p = VirtualAlloc(0, (__int64)size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#endif
	if (!p)
	{
		printf("[FATAL] Database builder allocation failed, size=%lld\n", (long long)size);
		exit(EXIT_FAILURE);
	}
	return p;
}

void db_free(void *ptr)
{
	#ifdef __linux__
	free(ptr);
#else
	if (ptr)
		VirtualFree(ptr, 0, MEM_RELEASE);
#endif
}

/* Both formats use read-only, whole-file checkpoint mappings. */
static void *map_checkpoint(FILE *file, size_t length)
{
#ifdef __linux__
	void *address = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fileno(file), 0);
	return address == MAP_FAILED ? NULL : address;
#else
	HANDLE mapping = CreateFileMappingA((HANDLE)_get_osfhandle(_fileno(file)), 0, PAGE_READONLY, 0, 0, 0);
	if (!mapping)
		return NULL;
	void *address = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
	CloseHandle(mapping);
	return address;
#endif
}

static void unmap_checkpoint(void *address, size_t length)
{
#ifdef __linux__
	munmap(address, length);
#else
	(void)length;
	UnmapViewOfFile(address);
#endif
}

#define HT_ADD_CHECK(key, data, len, persist)                                                                                                                                                                                                            \
	do                                                                                                                                                                                                                                                   \
	{                                                                                                                                                                                                                                                    \
		if (!dll.ht_add((key), (void *)(data), (len), (persist)))                                                                                                                                                                                        \
		{                                                                                                                                                                                                                                                \
			printf("[FATAL] ht_add('%s') FAILED\n", (key));                                                                                                                                                                                              \
			ExitProcess(1); /* a failed add ships an incomplete database -- never exit 0 */                                                                                                                                                              \
		}                                                                                                                                                                                                                                                \
	} while (0)

/* ---------------------------------------------------------------------------
 * vision DB storage (qwen3-VL .vision file)
 *
 * the .vision file produced by export_vlm.py has this layout:
 *   [0..255]  vision_config_header: 32 qwords. slots 0-15 are the cross-model
 *             LLaVA-one_vision plan fields (vit_hidden, vit_layers, vit_heads,
 *             vit_patch, image_size, vit_intermediate, vit_norm_type,
 *             vit_ffn_type, vit_has_cls, vit_feature_select, proj_hidden,
 *             proj_layers, proj_activation, vit_depth_taps, vit_head_dim,
 *             vit_pos_embed_type); slots 16-25 are qwen3-VL extension fields
 *             (image_token_id, vision_start_id, vision_end_id, has_deepstack,
 *             merge, temporal, num_pos, ds_tap0/1/2); slots 26-31 reserved.
  *   [256..]   VIT1 blob: reserved dword + all fp32 vision tower weights
  *             (stored as __sys_vit_weights)
 *
 * __sys_vision_config is an EXPANDED struct: the raw 16 x int32 header
 * (offsets 0..60, reructed from the qword header) followed by precomputed
 * derived qword fields (offset 64+). ALL derived values (dims, byte offsets,
 * stride products, float bit patterns) are computed HERE in C so the ASM vi_t
 * never does model-dependent arithmetic at runtime -- kernels read fields with
 * single RIP-relative memory operands. field layout is the authoritative spec
 * in vit_arch_refactor.md; byte offsets MUST match inf_structs.inc VCFG_*.
 * ------------------------------------------------------------------------- */
#define VISION_HDR_QWORDS 36
#define VISION_HDR_BYTES (VISION_HDR_QWORDS * 8)
/* (VIT1_MAGIC deleted -- reserved dword, always 0, never checked per Sec.96) */

#define VCFG_QWORDS 93 /* fixed struct size (744 bytes) */
#define VCFG_BYTES (VCFG_QWORDS * 8)

/* .vision header field slots (qword indices into the 256-byte header).
 * must stay in sync with tools/export_vlm.py VHDR_* ants. */
#define VHDR_VIT_HIDDEN 0          /* vi_t hidden dim (1024 qwen3-VL) */
#define VHDR_VIT_LAYERS 1          /* Transformer block count (24) */
#define VHDR_VIT_HEADS 2           /* attention heads (16) */
#define VHDR_VIT_PATCH 3           /* spatial patch size (16) */
#define VHDR_IMAGE_SIZE 4          /* image size (siglip 384; 0 = qwen3-VL grid) */
#define VHDR_VIT_INTERMEDIATE 5    /* MLP intermediate dim (4096) */
#define VHDR_VIT_NORM_TYPE 6       /* 1=layer_norm, 0=RMSNorm (0 qwen3-VL) */
#define VHDR_VIT_FFN_TYPE 7        /* 1=GELU MLP, 0=swi_g_l_u (0 qwen3-VL) */
#define VHDR_VIT_HAS_CLS 8         /* use_head/class token flag (0) */
#define VHDR_VIT_FEATURE_SELECT 9  /* 1=full patches, 0=cls only (1) */
#define VHDR_PROJ_HIDDEN 10        /* projector output dim (qwen3-VL merger out 2560) */
#define VHDR_PROJ_LAYERS 11        /* projector MLP layer count (2) */
#define VHDR_PROJ_ACTIVATION 12    /* 1=GELU, 0=none (1) */
#define VHDR_VIT_DEPTH_TAPS 13     /* deep_stack tap count (3 qwen3-VL, 0 siglip) */
#define VHDR_VIT_HEAD_DIM 14       /* attention head dim (64 qwen3-VL, 72 siglip) */
#define VHDR_VIT_POS_EMBED_TYPE 15 /* 0=qwen3-VL bilinear-grid, 1=learned additive */
#define VHDR_IMAGE_TOKEN_ID 16     /* image placeholder token id (151655) */
#define VHDR_VISION_START_ID 17    /* <|vision_start|> token id (151652) */
#define VHDR_VISION_END_ID 18      /* <|vision_end|> token id (151653) */
#define VHDR_HAS_DEEPSTACK 19      /* 1 = apply deep_stack decoder adds */
#define VHDR_MERGE 20              /* spatial merge size (2) */
#define VHDR_TEMPORAL 21           /* temporal patch size (2) */
#define VHDR_NUM_POS 22            /* learned position table entries (2304) */
#define VHDR_DS_TAP0 23            /* deep_stack tap block 0 (5) */
#define VHDR_DS_TAP1 24            /* deep_stack tap block 1 (11) */
#define VHDR_DS_TAP2 25            /* deep_stack tap block 2 (17) */
#define VHDR_ROPE_THETA 26         /* vision RoPE base theta (10000 qwen3-VL) */
#define VHDR_USE_ROPE 27           /* 1=apply RoPE, 0=skip (siglip) */
#define VHDR_PROJ_IN_DIM 28        /* projector input dim (0=none; 1152 siglip) */
#define VHDR_PROJ_OUT_DIM 29       /* projector output dim (0=none; 896 siglip) */
#define VHDR_PROJ_INTER 30         /* projector intermediate dim (0=none; 896 siglip) */
#define VHDR_PROJ_TYPE 31          /* 0=none(qwen3-VL merger), 1=MLP(siglip), 2=pixel-shuffle(SmolVLM/Idefics3) */
#define VHDR_PS_FACTOR 32          /* pixel-shuffle block size (4 SmolVLM-256M, 3 SmolVLM-2B) */
#define VHDR_MROPE_BOUND 33        /* text M-RoPE interleaved pair bound (3*min(section); 0 = none) */
#define VHDR_ANYRES_ENABLED 34     /* 1 = model uses anyres multi-tile (LLaVA-OV), 0 = single-tile */
#define VHDR_LN_EPS 35             /* IEEE-754 float bits of vi_t layer_norm epsilon (1e-5 siglip, 1e-6 qwen3-VL) */

/* derived qword byte offsets (64 = first derived field after the raw header) */
#define VCFG_HIDDEN 64             /* = vis_hidden                       */
#define VCFG_HIDDEN_F4 72          /* hidden * 4                         */
#define VCFG_INTER 80              /* = inter (MLP intermediate)         */
#define VCFG_QKV_DIM 88            /* 3 * hidden                         */
#define VCFG_QKV_STRIDE 96         /* qkv_dim * 4                        */
#define VCFG_HEADS 104             /* = heads                            */
#define VCFG_HEAD_DIM 112          /* hidden / heads                     */
#define VCFG_HEAD_DIM_F4 120       /* head_dim * 4                       */
#define VCFG_DEPTH 128             /* = depth                            */
#define VCFG_NUM_POS 136           /* = num_pos                          */
#define VCFG_NUM_POS_F4 144        /* num_pos * 4                        */
#define VCFG_GRID_SIDE 152         /* isqrt(num_pos)                     */
#define VCFG_GRID_MINUS1_I 160     /* grid_side - 1 (int)                */
#define VCFG_GRID_MINUS1_F 168     /* float bits of (float)(grid_side-1) */
#define VCFG_MERGE 176             /* = merge                            */
#define VCFG_MERGE_UNIT 184        /* merge * merge                      */
#define VCFG_PATCH_SIZE 192        /* = patch_size                       */
#define VCFG_PATCH_SIZE_F4 200     /* patch_size * 4                     */
#define VCFG_OUT_HIDDEN 208        /* = out_hidden                       */
#define VCFG_PATCH_DIM 216         /* 3 * temporal * patch_size^2        */
#define VCFG_PATCH_DIM_F4 224      /* patch_dim * 4                      */
#define VCFG_PACKED_DIM 232        /* hidden * merge_unit                */
#define VCFG_PACKED_DIM_F4 240     /* packed_dim * 4                     */
#define VCFG_N_DEEPSTACK 248       /* 3 (export_vlm always emits 3 taps) */
#define VCFG_ROPE_QTR 256          /* head_dim / 4                       */
#define VCFG_ROPE_HALF 264         /* head_dim / 2                       */
#define VCFG_ROPE_HALF_F4 272      /* rope_half * 4                      */
#define VCFG_DS_TAP0 280           /* = ds_tap0                          */
#define VCFG_DS_TAP1 288           /* = ds_tap1                          */
#define VCFG_DS_TAP2 296           /* = ds_tap2                          */
#define VCFG_L_QKV_W 304           /* qkv_dim * hidden * 4               */
#define VCFG_L_QKV_B 312           /* qkv_dim * 4                        */
#define VCFG_L_PROJ_W 320          /* hidden * hidden * 4                */
#define VCFG_L_PROJ_B 328          /* hidden * 4                         */
#define VCFG_L_FC1_W 336           /* inter * hidden * 4                 */
#define VCFG_L_FC1_B 344           /* inter * 4                          */
#define VCFG_L_FC2_W 352           /* hidden * inter * 4                 */
#define VCFG_L_FC2_B 360           /* hidden * 4                         */
#define VCFG_BLK_NORM1_STRIDE 368  /* depth * hidden * 4                 */
#define VCFG_BLK_QKV_W_STRIDE 376  /* depth * qkv_dim * hidden * 4       */
#define VCFG_BLK_QKV_B_STRIDE 384  /* depth * qkv_dim * 4                */
#define VCFG_BLK_PROJ_W_STRIDE 392 /* depth * hidden^2 * 4               */
#define VCFG_BLK_PROJ_B_STRIDE 400 /* depth * hidden * 4                 */
#define VCFG_BLK_FC1_W_STRIDE 408  /* depth * inter * hidden * 4         */
#define VCFG_BLK_FC1_B_STRIDE 416  /* depth * inter * 4                  */
#define VCFG_BLK_FC2_W_STRIDE 424  /* depth * hidden * inter * 4         */
#define VCFG_BLK_FC2_B_STRIDE 432  /* depth * hidden * 4                 */
#define VCFG_MG_FC1_W_STRIDE 440   /* packed^2 * 4                       */
#define VCFG_MG_FC1_B_STRIDE 448   /* packed * 4                         */
#define VCFG_MG_FC2_W_STRIDE 456   /* out_hidden * packed * 4            */
#define VCFG_MG_FC2_B_STRIDE 464   /* out_hidden * 4                     */
#define VCFG_DS_FC1_W_STRIDE 472   /* 3 * packed^2 * 4                   */
#define VCFG_DS_FC1_B_STRIDE 480   /* 3 * packed * 4                     */
#define VCFG_DS_FC2_W_STRIDE 488   /* 3 * out_hidden * packed * 4        */
#define VCFG_DS_FC2_B_STRIDE 496   /* 3 * out_hidden * 4                 */
#define VCFG_PE_W_STRIDE 504       /* patch_dim * hidden * 4             */
#define VCFG_POS_EMBED_BYTES 512   /* num_pos * hidden * 4               */
#define VCFG_ATTN_SCALE 520        /* IEEE-754 bits of 1/sqrt(head_dim)  */
#define VCFG_NORM_TYPE 528         /* 0=RMSNorm (qwen3-VL), 1=layer_norm (siglip) */
#define VCFG_FFN_TYPE 536          /* 0=swi_g_l_u (qwen3-VL), 1=GELU MLP (siglip) */
#define VCFG_HAS_CLS 544           /* 1 = prepend class token            */
#define VCFG_FEATURE_SELECT 552    /* 0=cls only, 1=full patch embeddings */
#define VCFG_PROJ_HIDDEN 560       /* projector output dim (qwen3-VL merger out) */
#define VCFG_PROJ_LAYERS 568       /* projector MLP layer count (2)      */
#define VCFG_PROJ_ACTIVATION 576   /* 1=GELU, 0=none                     */
#define VCFG_POS_EMBED_TYPE 584    /* 0=qwen3-VL bilinear-grid, 1=learned additive */
#define VCFG_IMAGE_SIZE 592        /* image size (siglip 384; 0 = grid)  */
/* offset 600 reserved (stays 0): VCFG_N_DEEPSTACK at 248 carries the tap count */
#define VCFG_ROPE_THETA_LOG2 608   /* IEEE-754 bits of log2f(theta) (0x41549A78=10000) */
#define VCFG_USE_ROPE 616          /* dword: 1=RoPE (qwen3-VL), 0=none (siglip) */
#define VCFG_PROJ_IN_DIM 624       /* projector input dim (0=none/qwen3-VL merger; 1152 siglip) */
#define VCFG_PROJ_OUT_DIM 632      /* projector output dim (0=none; 896 siglip) */
#define VCFG_PROJ_INTER 640        /* projector intermediate dim (0=none; 896 siglip) */
#define VCFG_PROJ_TYPE 648         /* dword: 0=none, 1=MLP, 2=pixel-shuffle */
#define VCFG_APPLY_POST_LN 656     /* dword: 1=apply post_layernorm (SmolVLM), 0=skip (LLaVA-OV) */
#define VCFG_PS_FACTOR 664         /* pixel-shuffle block size F (4) */
#define VCFG_PS_GRID_SIDE 672      /* image_size / patch_size (32) */
#define VCFG_PS_FLAT_DIM 680       /* vision_hidden * F * F (12288) */
#define VCFG_PS_N_TOKENS 688       /* (grid_side/F)^2 -- output token count (64) */
#define VCFG_ANYRES_ENABLED 696    /* 1 = anyres multi-tile (LLaVA-OV), 0 = single-tile */
#define VCFG_LN_EPS 704            /* IEEE-754 float bits of vi_t layer_norm epsilon */
#define VCFG_AR_GRID_SIDE 712      /* anyres tile patch grid side: image_size / patch_size, truncated (27 SigLIP 384/14) */
#define VCFG_AR_BASE_TOK 720       /* ar_grid_side^2 = base tokens per anyres tile (729) */
#define VCFG_AR_TC_ADV 728         /* ar_base_tok - ar_grid_side (702: src advance per tile_col in the arrange loop) */
#define VCFG_PS_OUT_W 736          /* grid_side / F -- output tokens per side (out_h == out_w, square grid) */

/* build the expanded vision config struct from the 32-qword raw header.
 * all derived values are precomputed here (C arithmetic is free) so the ASM
 * vi_t kernels read them with single memory operands at runtime. byte offsets
 * MUST match inf_structs.inc VCFG_* (spec: vit_arch_refactor.md). */
void build_vision_cfg(__int64 cfg[VCFG_QWORDS], __int64 vhdr[VISION_HDR_QWORDS])
{
	__int64 hidden = vhdr[VHDR_VIT_HIDDEN];      /* vis_hidden */
	__int64 out_hidden = vhdr[VHDR_PROJ_HIDDEN]; /* merger output dim */
	__int64 patch = vhdr[VHDR_VIT_PATCH];        /* spatial patch size */
	if (patch <= 0)
	{
		printf("[FATAL] VHDR_VIT_PATCH is %lld: it is a divisor (image_size / patch, grid_side * patch), so 0 would SIGFPE the builder and a negative value would corrupt every grid dimension.\n"
		       "        regenerate the .vision file with the current export_vlm.py.\n",
		       (long long)patch);
		exit(EXIT_FAILURE);
	}
	__int64 merge = vhdr[VHDR_MERGE];            /* spatial merge */
	if (hidden <= 0)
	{
		printf("[FATAL] VHDR_VIT_HIDDEN is %lld: it is a row length and every stride product (hidden*4, qkv_dim, the per-layer weight strides) is derived from it, so 0 or a negative value makes the ASM layernorm and vit_linear geometry walk off the row.\n"
		       "        regenerate the .vision file with the current export_vlm.py.\n",
		       (long long)hidden);
		exit(EXIT_FAILURE);
	}
	__int64 temporal = vhdr[VHDR_TEMPORAL];      /* temporal patch size */
	__int64 num_pos = vhdr[VHDR_NUM_POS];        /* learned position entries */
	__int64 depth = vhdr[VHDR_VIT_LAYERS];       /* num_hidden_layers */
	__int64 inter = vhdr[VHDR_VIT_INTERMEDIATE]; /* MLP intermediate */
	__int64 heads = vhdr[VHDR_VIT_HEADS];        /* num_attention_heads */
	__int64 theta = vhdr[VHDR_ROPE_THETA];       /* vision RoPE base */

	/* depth/inter/heads must be present -- no runtime fallback, ever. */
	if (depth == 0 || inter == 0 || heads == 0)
	{
		printf("[FATAL] vision header slots vit_layers/vit_intermediate/vit_heads are zero: %lld/%lld/%lld\n", (long long)depth, (long long)inter, (long long)heads);
		printf("        qword header slots 1/5/2 of the .vision file (or regenerate it via export_vlm.py)\n"
		       "        then rebuild the database. see vit_arch_refactor.md.\n");
		exit(EXIT_FAILURE);
	}

	/* A model with no DeepStack carries ds_tap0/1/2 == depth (one past the last
	 * layer index) so the three layer==tap compares in v7s_vit_forward can never
	 * match. A tap that is a REAL layer index while VCFG_N_DEEPSTACK is 0 would
	 * hand a NULL out_ds* to the merger -- reject the header here. */
	if (vhdr[VHDR_VIT_DEPTH_TAPS] == 0)
	{
		if ((vhdr[VHDR_DS_TAP0] < depth) || (vhdr[VHDR_DS_TAP1] < depth) || (vhdr[VHDR_DS_TAP2] < depth))
		{
			printf("[FATAL] vision header has no DeepStack taps but ds_tap0/1/2 are not all >= depth (%lld): %lld/%lld/%lld\n",
			       (long long)depth, (long long)vhdr[VHDR_DS_TAP0], (long long)vhdr[VHDR_DS_TAP1], (long long)vhdr[VHDR_DS_TAP2]);
			printf("        a no-DeepStack model must carry depth in all three tap slots (one past the\n"
			       "        last layer index) so the tap compares in v7s_vit_forward cannot match.\n");
			exit(EXIT_FAILURE);
		}
	}

	/* vision RoPE base must be present (export_vlm.py writes slot 26). */
	if (theta == 0)
	{
		printf("[FATAL] vision header slot 26 (rope_theta) is zero.\n"
		       "        regenerate the .vision file with the current tools/export_vlm.py\n"
		       "        (qwen3-VL emits theta=10000), then rebuild the database.\n");
		exit(EXIT_FAILURE);
	}

	__int64 qkv_dim = 3 * hidden;
	__int64 head_dim = hidden / heads;
	// vit_rope_build's dup loop copies the rope second half in 8-float chunks
	// (needs head_dim % 16 == 0) and the inv_freq BSS cache holds 32 entries
	// (needs head_dim <= 128) -- both are import-owned head_dim contracts.
	if (vhdr[VHDR_USE_ROPE] != 0)
	{
		if (((head_dim % 16) != 0) || (head_dim > 128))
		{
			printf("[FATAL] vision head_dim=%lld with USE_ROPE=1: must be a multiple of 16 and <= 128 (vit_rope_build's half-dup loop copies 8-float chunks and the inv_freq BSS cache holds 32 entries).\n", (long long)head_dim);
			exit(EXIT_FAILURE);
		}
	}
	/* attention scale = 1/sqrt(head_dim); precomputed IEEE-754 bits in C (per §119). */
	float attn_scale_f = 1.0f / sqrtf((float)head_dim);
	unsigned int attn_scale_bits;
	memcpy(&attn_scale_bits, &attn_scale_f, 4);
	__int64 grid_side = (__int64)floor(sqrt((double)num_pos));
	__int64 merge_unit = merge * merge;
	__int64 packed = hidden * merge_unit;
	__int64 patch_dim_raw = 3 * temporal * patch * patch;      /* raw patch embedding dim (588 for siglip) */
	__int64 patch_dim = (patch_dim_raw + 15) & ~((__int64)15); /* pad to multiple of 16 for the AVX2 K-loop (588->592) */
	float gmf = (float)(grid_side - 1);                        /* clamp upper bound */
	unsigned int gmf_bits;
	memcpy(&gmf_bits, &gmf, 4);
	/* log2(theta) precomputed IEEE-754 bits -- ASM loads with a single dword read. */
	float theta_log2_f = log2f((float)theta);
	unsigned int theta_log2_bits;
	memcpy(&theta_log2_bits, &theta_log2_f, 4);

	memset(cfg, 0, VCFG_BYTES);

	/* reruct the 16 x int32 raw header (VCONF 0..60 byte offsets) from the
	 * qword header so the ASM's VCONF_* reads (image_token_id, vision_start_id,
	 * vision_end_id, has_deepstack) stay valid in the verbatim first 64 bytes. */
	int vc[16];
	vc[0] = (int)vhdr[VHDR_IMAGE_TOKEN_ID];
	vc[1] = (int)vhdr[VHDR_VISION_START_ID];
	vc[2] = (int)vhdr[VHDR_VISION_END_ID];
	vc[3] = (int)vhdr[VHDR_HAS_DEEPSTACK];
	vc[4] = (int)hidden;
	vc[5] = (int)out_hidden;
	vc[6] = (int)patch;
	vc[7] = (int)merge;
	vc[8] = (int)temporal;
	vc[9] = (int)num_pos;
	vc[10] = (int)vhdr[VHDR_DS_TAP0];
	vc[11] = (int)vhdr[VHDR_DS_TAP1];
	vc[12] = (int)vhdr[VHDR_DS_TAP2];
	vc[13] = (int)depth;
	vc[14] = (int)inter;
	vc[15] = (int)heads;
	memcpy(cfg, vc, 64); /* raw 16 x int32 header verbatim into qword low halves */

	cfg[VCFG_HIDDEN / 8] = hidden;
	cfg[VCFG_HIDDEN_F4 / 8] = hidden * 4;
	cfg[VCFG_INTER / 8] = inter;
	cfg[VCFG_QKV_DIM / 8] = qkv_dim;
	cfg[VCFG_QKV_STRIDE / 8] = qkv_dim * 4;
	cfg[VCFG_HEADS / 8] = heads;
	cfg[VCFG_HEAD_DIM / 8] = head_dim;
	cfg[VCFG_HEAD_DIM_F4 / 8] = head_dim * 4;
	cfg[VCFG_DEPTH / 8] = depth;
	cfg[VCFG_NUM_POS / 8] = num_pos;
	cfg[VCFG_NUM_POS_F4 / 8] = num_pos * 4;
	cfg[VCFG_GRID_SIDE / 8] = grid_side;
	cfg[VCFG_GRID_MINUS1_I / 8] = grid_side - 1;
	cfg[VCFG_GRID_MINUS1_F / 8] = gmf_bits;
	cfg[VCFG_MERGE / 8] = merge;
	cfg[VCFG_MERGE_UNIT / 8] = merge_unit;
	cfg[VCFG_PATCH_SIZE / 8] = patch;
	cfg[VCFG_PATCH_SIZE_F4 / 8] = patch * 4;
	cfg[VCFG_OUT_HIDDEN / 8] = out_hidden;
	cfg[VCFG_PATCH_DIM / 8] = patch_dim;
	cfg[VCFG_PATCH_DIM_F4 / 8] = patch_dim * 4;
	cfg[VCFG_PACKED_DIM / 8] = packed;
	cfg[VCFG_PACKED_DIM_F4 / 8] = packed * 4;
	cfg[VCFG_N_DEEPSTACK / 8] = vhdr[VHDR_VIT_DEPTH_TAPS]; /* 3 qwen3-VL, 0 siglip */
	cfg[VCFG_ROPE_QTR / 8] = head_dim / 4;
	cfg[VCFG_ROPE_HALF / 8] = head_dim / 2;
	cfg[VCFG_ROPE_HALF_F4 / 8] = (head_dim / 2) * 4;
	cfg[VCFG_DS_TAP0 / 8] = vhdr[VHDR_DS_TAP0];
	cfg[VCFG_DS_TAP1 / 8] = vhdr[VHDR_DS_TAP1];
	cfg[VCFG_DS_TAP2 / 8] = vhdr[VHDR_DS_TAP2];

	cfg[VCFG_L_QKV_W / 8] = qkv_dim * hidden * 4;
	cfg[VCFG_L_QKV_B / 8] = qkv_dim * 4;
	cfg[VCFG_L_PROJ_W / 8] = hidden * hidden * 4;
	cfg[VCFG_L_PROJ_B / 8] = hidden * 4;
	cfg[VCFG_L_FC1_W / 8] = inter * hidden * 4;
	cfg[VCFG_L_FC1_B / 8] = inter * 4;
	cfg[VCFG_L_FC2_W / 8] = hidden * inter * 4;
	cfg[VCFG_L_FC2_B / 8] = hidden * 4;

	cfg[VCFG_BLK_NORM1_STRIDE / 8] = depth * hidden * 4;
	cfg[VCFG_BLK_QKV_W_STRIDE / 8] = depth * qkv_dim * hidden * 4;
	cfg[VCFG_BLK_QKV_B_STRIDE / 8] = depth * qkv_dim * 4;
	cfg[VCFG_BLK_PROJ_W_STRIDE / 8] = depth * hidden * hidden * 4;
	cfg[VCFG_BLK_PROJ_B_STRIDE / 8] = depth * hidden * 4;
	cfg[VCFG_BLK_FC1_W_STRIDE / 8] = depth * inter * hidden * 4;
	cfg[VCFG_BLK_FC1_B_STRIDE / 8] = depth * inter * 4;
	cfg[VCFG_BLK_FC2_W_STRIDE / 8] = depth * hidden * inter * 4;
	cfg[VCFG_BLK_FC2_B_STRIDE / 8] = depth * hidden * 4;
	cfg[VCFG_MG_FC1_W_STRIDE / 8] = packed * packed * 4;
	cfg[VCFG_MG_FC1_B_STRIDE / 8] = packed * 4;
	cfg[VCFG_MG_FC2_W_STRIDE / 8] = out_hidden * packed * 4;
	cfg[VCFG_MG_FC2_B_STRIDE / 8] = out_hidden * 4;
	__int64 n_ds = vhdr[VHDR_VIT_DEPTH_TAPS]; /* 3 qwen3-VL, 0 siglip */
	cfg[VCFG_DS_FC1_W_STRIDE / 8] = n_ds * packed * packed * 4;
	cfg[VCFG_DS_FC1_B_STRIDE / 8] = n_ds * packed * 4;
	cfg[VCFG_DS_FC2_W_STRIDE / 8] = n_ds * out_hidden * packed * 4;
	cfg[VCFG_DS_FC2_B_STRIDE / 8] = n_ds * out_hidden * 4;
	cfg[VCFG_PE_W_STRIDE / 8] = patch_dim * hidden * 4;
	cfg[VCFG_POS_EMBED_BYTES / 8] = num_pos * hidden * 4;
	cfg[VCFG_ATTN_SCALE / 8] = (__int64)attn_scale_bits; /* 1/sqrt(head_dim) IEEE bits */

	cfg[VCFG_NORM_TYPE / 8] = vhdr[VHDR_VIT_NORM_TYPE];
	cfg[VCFG_FFN_TYPE / 8] = vhdr[VHDR_VIT_FFN_TYPE];
	cfg[VCFG_HAS_CLS / 8] = vhdr[VHDR_VIT_HAS_CLS];
	cfg[VCFG_FEATURE_SELECT / 8] = vhdr[VHDR_VIT_FEATURE_SELECT];
	cfg[VCFG_PROJ_HIDDEN / 8] = out_hidden;
	cfg[VCFG_PROJ_LAYERS / 8] = vhdr[VHDR_PROJ_LAYERS];
	cfg[VCFG_PROJ_ACTIVATION / 8] = vhdr[VHDR_PROJ_ACTIVATION];
	cfg[VCFG_POS_EMBED_TYPE / 8] = vhdr[VHDR_VIT_POS_EMBED_TYPE];
	cfg[VCFG_IMAGE_SIZE / 8] = vhdr[VHDR_IMAGE_SIZE];
	/* VCFG_DEPTH_TAPS (600) is reserved-0: VCFG_N_DEEPSTACK (248) carries the tap count */
	cfg[VCFG_ROPE_THETA_LOG2 / 8] = (__int64)theta_log2_bits; /* log2(theta) IEEE bits */
	cfg[VCFG_USE_ROPE / 8] = vhdr[VHDR_USE_ROPE];             /* 1=RoPE, 0=skip */
	cfg[VCFG_PROJ_IN_DIM / 8] = vhdr[VHDR_PROJ_IN_DIM];       /* 0=none, 1152 siglip */
	cfg[VCFG_PROJ_OUT_DIM / 8] = vhdr[VHDR_PROJ_OUT_DIM];     /* 0=none, 896 siglip */
	cfg[VCFG_PROJ_INTER / 8] = vhdr[VHDR_PROJ_INTER];         /* 0=none, 896 siglip */

	/* SmolVLM/Idefics3 pixel-shuffle projector fields.
	 * PROJ_TYPE: 0=none (qwen3-VL merger), 1=MLP (siglip/LLaVA-OV), 2=pixel-shuffle.
	 * APPLY_POST_LN: only SmolVLM/Idefics3 (proj_type==2) applies post_layernorm --
	 *   HF Idefics3 uses last_hidden_state (includes post-LN). LLaVA-OV (proj_type==1)
	 *   uses hidden_states[-1] (raw, before post-LN) per HF source. qwen3-VL (proj_type==0)
	 *   has its own merger norm stack and skips post-LN.
	 * PS_FACTOR: pixel-shuffle block size (4 for SmolVLM-256M, 3 for SmolVLM-2B).
	 * PS_GRID_SIDE: image_size / patch_size, precomputed in C (32 for SmolVLM-256M).
	 * PS_FLAT_DIM: vision_hidden * F * F, precomputed (12288 = 768 * 16 for SmolVLM-256M).
	 * PS_N_TOKENS: (grid_side/F)^2, precomputed (64 for SmolVLM-256M). */
	__int64 proj_type = vhdr[VHDR_PROJ_TYPE]; /* 0/1/2 */
	__int64 ps_factor = vhdr[VHDR_PS_FACTOR]; /* 0 if no pixel-shuffle */
	cfg[VCFG_PROJ_TYPE / 8] = proj_type;
	int apply_post_ln = 0;
	if (proj_type == 2)
		apply_post_ln = 1;
	cfg[VCFG_APPLY_POST_LN / 8] = apply_post_ln; /* only SmolVLM/Idefics3 (pixel-shuffle) applies post-LN */
	if (proj_type == 2)
	{
		if (ps_factor == 0)
		{
			printf("[FATAL] VHDR_PS_FACTOR is zero but PROJ_TYPE=2 (pixel-shuffle).\n"
			       "        regenerate the .vision file with the current export_vlm.py.\n");
			exit(EXIT_FAILURE);
		}
		__int64 ps_grid_side = vhdr[VHDR_IMAGE_SIZE] / patch;
		if (ps_grid_side < ps_factor)
		{
			printf("[FATAL] pixel-shuffle grid (image_size / patch = %lld) is smaller than VHDR_PS_FACTOR (%lld): vit_pixel_shuffle would divide by a zero out_w.\n"
			       "        regenerate the .vision file with the current export_vlm.py.\n",
			       (long long)ps_grid_side, (long long)ps_factor);
			exit(EXIT_FAILURE);
		}
		__int64 ps_out_w = ps_grid_side / ps_factor;
		cfg[VCFG_PS_FACTOR / 8] = ps_factor;
		cfg[VCFG_PS_GRID_SIDE / 8] = ps_grid_side;
		cfg[VCFG_PS_FLAT_DIM / 8] = hidden * ps_factor * ps_factor;
		cfg[VCFG_PS_N_TOKENS / 8] = ps_out_w * ps_out_w;
		cfg[VCFG_PS_OUT_W / 8] = ps_out_w;
	}

	/* anyres flag: 1 = model uses anyres multi-tile (LLaVA-OV), 0 = single-tile */
	cfg[VCFG_ANYRES_ENABLED / 8] = vhdr[VHDR_ANYRES_ENABLED];

	/* vi_t layer_norm epsilon: IEEE-754 float bits stored as qword (read as dword in ASM) */
	cfg[VCFG_LN_EPS / 8] = vhdr[VHDR_LN_EPS];

	/* anyres tile patch grid, precomputed (Sec.119: zero runtime arithmetic in ASM).
	 * AR_GRID_SIDE = image_size / patch_size, truncated exactly as the HF reference
	 * tiles (384/14 -> 27); AR_BASE_TOK = grid_side^2 (729 patches per tile);
	 * AR_TC_ADV = base_tok - grid_side (src advance per tile_col in the arrange
	 * loop). Written for every VLM so the fields are never stale; only the
	 * anyres path (LLaVA-OV) reads them. */
	__int64 ar_grid_side = vhdr[VHDR_IMAGE_SIZE] / patch;
	cfg[VCFG_AR_GRID_SIDE / 8] = ar_grid_side;
	cfg[VCFG_AR_BASE_TOK / 8] = ar_grid_side * ar_grid_side;
	cfg[VCFG_AR_TC_ADV / 8] = ar_grid_side * ar_grid_side - ar_grid_side;
}

void store_vision_db(char *vision_path)
{
	FILE *f = fopen(vision_path, "rb");
	if (!f)
	{
		printf("[FATAL] cannot open %s\n", vision_path);
		exit(EXIT_FAILURE);
	}
	_fseeki64(f, 0, SEEK_END);
	__int64 vfile_size = _ftelli64(f);
	fseek(f, 0, SEEK_SET);
	if (vfile_size < VISION_HDR_BYTES + 4)
	{
		printf("[FATAL] vision file too small: %lld bytes\n", (long long)vfile_size);
		fclose(f);
		exit(EXIT_FAILURE);
	}

	/* mmap the .vision file -- no 1.66GB heap copy; ht_add copies into the pool */
	char *vdata = (char *)map_checkpoint(f, (size_t)vfile_size);
	if (!vdata)
	{
		printf("[FATAL] Cannot map vision checkpoint %s\n", vision_path);
		fclose(f);
		exit(EXIT_FAILURE);
	}
	fclose(f);

	/* 64-byte header: 16 int32, verbatim copy into the hash table */
	/* 256-byte header: 32 qwords, verbatim copy into the hash table */
	__int64 vhdr[VISION_HDR_QWORDS];
	memcpy(vhdr, vdata, VISION_HDR_BYTES);

	/* VIT1 blob: everything past the header */
	/* first dword is reserved (always 0, never checked -- Sec.96) */
	char *vit = vdata + VISION_HDR_BYTES;
	__int64 vit_size = (__int64)vfile_size - VISION_HDR_BYTES;

	/* expand the raw header into the precomputed struct (computed in C). The array is
	 * serialized verbatim into the DB, so it MUST be zero-initialized: an unwritten
	 * slot (the pixel-shuffle fields for non-pixel-shuffle VLMs) must be
	 * deterministic 0, never whatever the stack held */
	__int64 vcfg[VCFG_QWORDS] = {0};
	build_vision_cfg(vcfg, vhdr);

	HT_ADD_CHECK("__sys_vision_config", vcfg, VCFG_BYTES, 1);
	HT_ADD_CHECK("__sys_vit_weights", vit, vit_size, 1);

	/* __sys_vision_xy_res: fixed input image size [u64 target_w, u64 target_h].
	 * siglip (VHDR_IMAGE_SIZE > 0) uses its trained resolution (384x384).
	 * qwen3-VL is grid-based (VHDR_IMAGE_SIZE == 0): use the model's NATIVE
	 * position grid -- isqrt(num_pos) x patch_size (48 x 16 = 768 for the 4B).
	 * the old 224x224 default starved the vi_t to a 14x14 patch grid (2.86x
	 * nearest downsample of typical inputs) and destroyed spatial grounding
	 * (box-test avg deviation 0.056 vs HF 0.007; fixed: 0.007). */
	unsigned __int64 xy_res[2];
	if (vhdr[VHDR_IMAGE_SIZE] > 0)
	{
		xy_res[0] = (unsigned __int64)vhdr[VHDR_IMAGE_SIZE];
		xy_res[1] = (unsigned __int64)vhdr[VHDR_IMAGE_SIZE];
	}
	else
	{
		__int64 grid_side = (__int64)sqrt((double)vhdr[VHDR_NUM_POS]);
		__int64 native_px = grid_side * vhdr[VHDR_VIT_PATCH];
		if (native_px <= 0)
		{
			printf("[FATAL] vision grid resolution invalid: num_pos=%lld patch=%lld\n", (long long)vhdr[VHDR_NUM_POS], (long long)vhdr[VHDR_VIT_PATCH]);
			unmap_checkpoint(vdata, (size_t)vfile_size);
			exit(EXIT_FAILURE);
		}
		xy_res[0] = (unsigned __int64)native_px;
		xy_res[1] = (unsigned __int64)native_px;
	}
	HT_ADD_CHECK("__sys_vision_xy_res", xy_res, sizeof(xy_res), 1);

	/* __sys_mrope_bound: text M-RoPE interleaved pair bound (u64). nonzero only
	 * for qwen3-VL-style text models with rope_scaling.mrope_section +
	 * mrope_interleaved. 0 (or absent) = plain 1D text RoPE. the ASM loader
	 * treats a missing key as 0, so older DBs without the key stay valid. */
	{
		unsigned __int64 mrope_bound = (unsigned __int64)vhdr[VHDR_MROPE_BOUND];
		HT_ADD_CHECK("__sys_mrope_bound", &mrope_bound, sizeof(mrope_bound), 1);
	}

	/* __sys_vision_prompt_fmt: three NUL-terminated strings packed back-to-back
	 * that describe how to build the vision prompt prefix for the text LLM:
	 *   byte layout:  [prefix\0] [marker\0] [suffix\0]
	 * qwen3-VL:  "<|vision_start|>" \0 "<|image_pad|>" \0 "<|vision_end|>" \0
	 * LLaVA-OV:  "" \0 "<image>" \0 "\n" \0
	 * the ASM chat_image.inc reads this blob once at init, caches three pointers
	 * (prefix/marker/suffix) in BSS, and builds the prompt purely from data --
	 * zero model-identity branching in ASM (per S119: model differences are DATA). */
	char *vprefix;
	char *vmarker;
	char *vsuffix;
	if (vhdr[VHDR_VISION_START_ID] != 0)
	{
		/* qwen3-VL: wrap image pads with vision_start / vision_end markers */
		vprefix = "<|vision_start|>";
		vmarker = "<|image_pad|>";
		vsuffix = "<|vision_end|>";
	}
	else if (vhdr[VHDR_PROJ_TYPE] == 2)
	{
		/* SmolVLM / Idefics3 (pixel-shuffle projector): images concatenate
		 * directly with the following text -- HF template has NO separator. */
		vprefix = "";
		vmarker = "<image>";
		vsuffix = "";
	}
	else
	{
		/* LLaVA-one_vision: inline <image> marker, newline before user text */
		vprefix = "";
		vmarker = "<image>";
		vsuffix = "\n";
	}
	char prompt_fmt[64];
	int pf_len = 0;
	strcpy(prompt_fmt + pf_len, vprefix);
	pf_len += (int)strlen(vprefix) + 1;
	strcpy(prompt_fmt + pf_len, vmarker);
	pf_len += (int)strlen(vmarker) + 1;
	strcpy(prompt_fmt + pf_len, vsuffix);
	pf_len += (int)strlen(vsuffix) + 1;
	HT_ADD_CHECK("__sys_vision_prompt_fmt", prompt_fmt, pf_len, 1);

	/* __sys_image_newline_emb: anyres image_newline learned embedding (raw float32).
	 * present only when VHDR_ANYRES_ENABLED=1 (LLaVA-OV). this is a trained
	 * nn.Parameter of shape (text_hidden,) that gets concatenated between grid
	 * rows in pack_image_features -- it is NOT a vocabulary token. */
	if (vhdr[VHDR_ANYRES_ENABLED])
	{
		char newline_path[2048];
		snprintf(newline_path, sizeof(newline_path), "%s.newline", vision_path);
		FILE *nlf = fopen(newline_path, "rb");
		if (!nlf)
		{
			printf("[FATAL] anyres enabled but %s not found\n", newline_path);
			unmap_checkpoint(vdata, (size_t)vfile_size);
			exit(EXIT_FAILURE);
		}
		fseek(nlf, 0, SEEK_END);
		__int64 nl_size = (__int64)ftell(nlf);
		fseek(nlf, 0, SEEK_SET);
		char *nl_buf = (char *)db_alloc(nl_size);
		if (fread(nl_buf, 1, (__int64)nl_size, nlf) != (size_t)nl_size)
		{
			printf("[FATAL] newline embedding file is short -- a zero-padded embedding would ship\n");
			exit(EXIT_FAILURE);
		}
		fclose(nlf);
		HT_ADD_CHECK("__sys_image_newline_emb", nl_buf, nl_size, 1);
		db_free(nl_buf);
		printf("  [14] image_newline: %lld bytes stored\n", (long long)nl_size);
	}

	printf(
	    "  [14] vision_config: image_token=%lld start=%lld end=%lld deepstack=%lld vis_hidden=%lld out_hidden=%lld patch=%lld merge=%lld temporal=%lld pos_emb=%lld ds=%lld/%lld/%lld depth=%lld inter=%lld heads=%lld head_dim=%lld norm=%lld ffn=%lld proj_hid=%lld proj_layers=%lld proj_act=%lld pos_embed_type=%lld img_size=%lld depth_taps=%lld qkv=%lld grid=%lld packed=%lld patch_dim=%lld (%lld bytes)\n",
	    vhdr[VHDR_IMAGE_TOKEN_ID], vhdr[VHDR_VISION_START_ID], vhdr[VHDR_VISION_END_ID], vhdr[VHDR_HAS_DEEPSTACK], vhdr[VHDR_VIT_HIDDEN], vhdr[VHDR_PROJ_HIDDEN], vhdr[VHDR_VIT_PATCH], vhdr[VHDR_MERGE], vhdr[VHDR_TEMPORAL], vhdr[VHDR_NUM_POS],
	    vhdr[VHDR_DS_TAP0], vhdr[VHDR_DS_TAP1], vhdr[VHDR_DS_TAP2], vhdr[VHDR_VIT_LAYERS], vhdr[VHDR_VIT_INTERMEDIATE], vhdr[VHDR_VIT_HEADS], vhdr[VHDR_VIT_HEAD_DIM], vhdr[VHDR_VIT_NORM_TYPE], vhdr[VHDR_VIT_FFN_TYPE], vhdr[VHDR_PROJ_HIDDEN],
	    vhdr[VHDR_PROJ_LAYERS], vhdr[VHDR_PROJ_ACTIVATION], vhdr[VHDR_VIT_POS_EMBED_TYPE], vhdr[VHDR_IMAGE_SIZE], vhdr[VHDR_VIT_DEPTH_TAPS], (long long)vcfg[VCFG_QKV_DIM / 8], (long long)vcfg[VCFG_GRID_SIDE / 8], (long long)vcfg[VCFG_PACKED_DIM / 8],
	    (long long)vcfg[VCFG_PATCH_DIM / 8], (long long)VCFG_BYTES);
	printf("  [14] vit_weights: %lld MB stored\n", (long long)(vit_size >> 20));

	unmap_checkpoint(vdata, (size_t)vfile_size);
}

/* ---------------------------------------------------------------------------
 * QuantizedTensor helpers (simplified -- no VirtualAlloc needed for descriptors)
 * we just need init_QuantizedTensors for memory_map_weights.
 * since we only need file offsets for weight_meta, we can skip full init.
 * ------------------------------------------------------------------------- */

QuantizedTensor *init_qt_array(int count, __int64 each, void **ptr)
{
	QuantizedTensor *res = (QuantizedTensor *)calloc(count, sizeof(QuantizedTensor));
	if (!res)
	{
		printf("[FATAL] calloc QuantizedTensor array failed, count=%d\n", count);
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < count; i++)
	{
		res[i].q = (char *)*ptr;
		*ptr = (char *)*ptr + each;
		res[i].s = (float *)*ptr;
		*ptr = (float *)*ptr + ((each + GS - 1) / GS);
	}
	return res;
}

void memory_map_weights(transformer_weights *w, Config *p, void *ptr)
{
	float *fptr = (float *)ptr;
	w->rms_att_weight = fptr;
	fptr += (__int64)p->n_layers * p->dim;
	w->rms_ffn_weight = fptr;
	fptr += (__int64)p->n_layers * p->dim;
	w->rms_final_weight = fptr;
	fptr += (__int64)p->dim;
	w->q_norm_weights = fptr;
	fptr += (__int64)p->n_layers * p->head_dim;
	w->k_norm_weights = fptr;
	fptr += (__int64)p->n_layers * p->head_dim;
	w->qkv_bias = fptr;
	fptr += (__int64)p->n_layers * ((__int64)p->n_heads * p->head_dim + 2 * (__int64)p->n_kv_heads * p->head_dim);

	ptr = (void *)fptr;
	w->q_tokens = init_qt_array(1, (__int64)p->vocab_size * p->dim, &ptr);
	w->token_embedding_table = 0; /* not needed for DB build */

	w->wq = init_qt_array(p->n_layers, (__int64)p->dim * p->n_heads * p->head_dim, &ptr);
	w->wk = init_qt_array(p->n_layers, (__int64)p->dim * p->n_kv_heads * p->head_dim, &ptr);
	w->wv = init_qt_array(p->n_layers, (__int64)p->dim * p->n_kv_heads * p->head_dim, &ptr);
	w->wo = init_qt_array(p->n_layers, (__int64)p->n_heads * p->head_dim * p->dim, &ptr);

	w->w1 = init_qt_array(p->n_layers, (__int64)p->dim * p->hidden_dim, &ptr);
	w->w2 = init_qt_array(p->n_layers, (__int64)p->hidden_dim * p->dim, &ptr);
	w->w3 = init_qt_array(p->n_layers, (__int64)p->dim * p->hidden_dim, &ptr);

	if (p->shared_classifier)
		w->wcls = w->q_tokens;
	else
		w->wcls = init_qt_array(1, (__int64)p->dim * p->vocab_size, &ptr);
}

/* ---------------------------------------------------------------------------
 * Tokenizer loading (file-only, no DB)
 * ------------------------------------------------------------------------- */

void load_prompt_template(char *checkpoint_path, char *out, int with_sys)
{
	char path[2048];
	char *suffix;
	if (with_sys)
		suffix = ".template.with-system";
	else
		suffix = ".template";
	snprintf(path, sizeof(path), "%s%s", checkpoint_path, suffix);
	memset(out, 0, 1024);
	FILE *f = fopen(path, "rb");
	if (!f)
	{
		/* base sidecars are part of the export contract (export.py always writes
		 * all 4). A missing file is a build contract violation, not a warning. */
		printf("[FATAL] couldn't load template %s (re-run export)\n", path);
		exit(EXIT_FAILURE);
	}
	size_t template_bytes = fread(out, 1, 1024, f);
	if (ferror(f))
	{
		printf("[FATAL] template %s I/O error -- a truncated template would ship\n", path);
		exit(EXIT_FAILURE);
	}
	fclose(f);
	/* the DB template slot is 1023 bytes + NUL: a longer template would be cut
	 * mid-token into the database -- refuse instead of silently truncating.
	 * reading 1024 into the 1024-byte buffer only to detect oversize is safe:
	 * a fitting file leaves out[1023] = 0 from the memset above. */
	if (template_bytes >= 1024)
	{
		printf("[FATAL] template %s is 1024+ bytes -- exceeds the 1023-byte DB slot (shorten the template or widen the slot)\n", path);
		exit(EXIT_FAILURE);
	}
	printf("template: %s\n", path);
}

void build_tokenizer(tokenizer *t, char *checkpoint_path, int vocab_size)
{
	t->vocab_size = vocab_size;
	t->vocab = (char **)malloc(vocab_size * sizeof(char *));
	t->merge_scores = (float *)malloc(vocab_size * sizeof(float));
	if (!t->vocab || !t->merge_scores)
	{
		printf("[FATAL] malloc tokenizer failed\n");
		exit(EXIT_FAILURE);
	}

	char tok_path[2048];
#ifdef __linux__
	/* The exporter emits lowercase; accept legacy checkpoint sidecars too. */
	snprintf(tok_path, sizeof(tok_path), "%s.tokenizer", checkpoint_path);
	FILE *f = fopen(tok_path, "rb");
	if (!f && errno == ENOENT)
	{
		snprintf(tok_path, sizeof(tok_path), "%s.Tokenizer", checkpoint_path);
		f = fopen(tok_path, "rb");
	}
#else
	snprintf(tok_path, sizeof(tok_path), "%s.Tokenizer", checkpoint_path);
	FILE *f = fopen(tok_path, "rb");
#endif
	if (!f)
	{
		printf("[FATAL] couldn't open %s\n", tok_path);
		exit(EXIT_FAILURE);
	}
	if (fread(&t->max_token_length, sizeof(int), 1, f) != 1)
	{
		printf("[FATAL] truncated tokenizer header\n");
		exit(EXIT_FAILURE);
	}
	if (fread(&t->bos_token_id, sizeof(t->bos_token_id), 1, f) != 1)
	{
		printf("[FATAL] truncated tokenizer header\n");
		exit(EXIT_FAILURE);
	}
	if (fread(&t->eos_token_id, sizeof(t->eos_token_id), 1, f) != 1)
	{
		printf("[FATAL] truncated tokenizer header\n");
		exit(EXIT_FAILURE);
	}

	int len;
	t->max_token_length = 0;
	for (int i = 0; i < vocab_size; i++)
	{
		if (fread(&t->merge_scores[i], sizeof(float), 1, f) != 1)
		{
			printf("[FATAL] truncated tokenizer: no merge score for token %d (short .tokenizer file -- re-run export)\n", i);
			exit(EXIT_FAILURE); /* a silent -FLT_MAX/"" substitution ships a corrupted vocabulary */
		}
		else
		{
			if (fread(&len, sizeof(int), 1, f) != 1)
			{
				printf("[FATAL] truncated token at %d\n", i);
				exit(EXIT_FAILURE);
			}
			if (len < 0 || len > 4096)
			{
				printf("[FATAL] bad token length %d at %d\n", len, i);
				exit(EXIT_FAILURE);
			}
			t->vocab[i] = (char *)malloc(len + 1);
			if (!t->vocab[i])
			{
				printf("[FATAL] malloc vocab token failed at %d\n", i);
				exit(EXIT_FAILURE);
			}
			if (fread(t->vocab[i], 1, len, f) != (__int64)len)
			{
				printf("[FATAL] truncated token data at %d\n", i);
				exit(EXIT_FAILURE);
			}
			t->vocab[i][len] = 0;
			if (len > (int)t->max_token_length)
				t->max_token_length = len;
		}
	}
	fclose(f);

	/* sentence_piece detection */
	t->is_sentencepiece = 0;
	for (int i = 0; i < vocab_size && !t->is_sentencepiece; i++)
	{
		char *s = t->vocab[i];
		if (!s)
			continue;
		for (int j = 0; s[j]; j++)
		{
			if ((unsigned char)s[j] == 0xE2 && (unsigned char)s[j + 1] == 0x96 && (unsigned char)s[j + 2] == 0x81)
			{
				t->is_sentencepiece = 1;
				break;
			}
		}
	}

	/* scan vocab for known turn-ending stop tokens beyond eos_token_id
	 * (e.g. phi-3 uses <|end|>=32007 as turn terminator, but declares
	 *  <|endoftext|>=32000 as eos_token_id). we scan all vocab entries
	 * for these strings and add any found to the stop token set. */
	t->num_stop_ids = 1;
	t->stop_ids[0] = t->eos_token_id;
	for (int i = 0; i < vocab_size && t->num_stop_ids < 4; i++)
	{
		char *s = t->vocab[i];
		if (!s || (unsigned int)i == t->eos_token_id)
			continue;
		/* check for common turn-ending tokens that supplement eos_token_id */
		if (strcmp(s, "<|end|>") == 0)
			t->stop_ids[t->num_stop_ids++] = (unsigned int)i;
		else if (strcmp(s, "<|eot_id|>") == 0)
			t->stop_ids[t->num_stop_ids++] = (unsigned int)i;
		else if (strcmp(s, "<|end_of_text|>") == 0)
			t->stop_ids[t->num_stop_ids++] = (unsigned int)i;
		else if (strcmp(s, "<|endoftext|>") == 0) /* §150: qwen3 emits this mid-stream; without it
		                                             in the stop list the text leaks into output */
			t->stop_ids[t->num_stop_ids++] = (unsigned int)i;
		else if (strcmp(s, "<end_of_utterance>") == 0)
			t->stop_ids[t->num_stop_ids++] = (unsigned int)i;
	}
	printf("  [tok] stop_ids[%u]:", t->num_stop_ids);
	for (unsigned int si = 0; si < t->num_stop_ids; si++)
		printf(" %u", t->stop_ids[si]);
	printf("\n");

	/* templates: base pair is ALWAYS the non-thinking variant (mandatory --
	 * export.py writes all 4 sidecars). thinking variants (@think keys) and the
	 * image-turn variant are OPTIONAL sidecars: absent file leaves the buffer
	 * empty -- populate_db then stores no suffixed key (engine falls back). */
	load_prompt_template(checkpoint_path, t->prompt_template, 0);
	load_prompt_template(checkpoint_path, t->system_prompt_template, 1);
	{
		char *opt_suffixes[5] = { ".template.with-thinking", ".template.with-system-and-thinking", ".template.img", ".template.multi", ".template.multi.with-thinking" };
		char *opt_dst[5] = { t->think_prompt_template, t->think_system_prompt_template, t->image_prompt_template, t->multi_prompt_template, t->multi_think_prompt_template };
		for (int i = 0; i < 5; i++)
		{
			char path[2048];
			snprintf(path, sizeof(path), "%s%s", checkpoint_path, opt_suffixes[i]);
			FILE *f = fopen(path, "rb");
			if (f)
			{
				memset(opt_dst[i], 0, 1024);
				size_t template_bytes = fread(opt_dst[i], 1, 1024, f);
				if (ferror(f))
				{
					printf("[FATAL] template %s I/O error -- a truncated sidecar would ship\n", path);
					exit(EXIT_FAILURE);
				}
				fclose(f);
				/* same 1023-byte DB slot contract as the mandatory templates:
				 * an oversize sidecar would be cut mid-token -- refuse loudly */
				if (template_bytes >= 1024)
				{
					printf("[FATAL] template %s is 1024+ bytes -- exceeds the 1023-byte DB slot (shorten the template or widen the slot)\n", path);
					exit(EXIT_FAILURE);
				}
				printf("template(opt): %s\n", path);
			}
		}
	}
}

/* ---------------------------------------------------------------------------
 * MemLayout computation (copied from runq.c)
 * ------------------------------------------------------------------------- */

void compute_mem_layout(Config *p, mem_layout *ml)
{
	memset(ml, 0, sizeof(*ml));
	__int64 all_heads_dim = (__int64)p->n_heads * p->head_dim;
	__int64 kv_dim = (__int64)p->n_kv_heads * p->head_dim;
	__int64 xb_size = all_heads_dim;
	if ((__int64)p->dim > xb_size)
		xb_size = (__int64)p->dim;
	__int64 hb2_size = (__int64)p->hidden_dim;
	if ((__int64)p->dim > hb2_size)
		hb2_size = (__int64)p->dim;
	__int64 xq_size = (__int64)p->dim;
	if (all_heads_dim > xq_size)
		xq_size = all_heads_dim;
	if ((__int64)p->hidden_dim > xq_size)
		xq_size = (__int64)p->hidden_dim;
	__int64 GSv = (__int64)GS;
	__int64 xq_s_count = (xq_size + GSv - 1) / GSv;
	__int64 hq_s_count = ((__int64)p->hidden_dim + GSv - 1) / GSv;

	__int64 off = 0;
#define LAYOUT_FIELD(sz, field)                                                                                                                                                                                                                          \
	do                                                                                                                                                                                                                                                   \
	{                                                                                                                                                                                                                                                    \
		off = (off + sizeof(float) - 1) & ~(sizeof(float) - 1);                                                                                                                                                                                          \
		ml->field = off;                                                                                                                                                                                                                                 \
		off += (sz);                                                                                                                                                                                                                                     \
	} while (0)

	LAYOUT_FIELD((__int64)p->dim * sizeof(float), off_x);
	LAYOUT_FIELD(xb_size * sizeof(float), off_xb);
	LAYOUT_FIELD((__int64)p->hidden_dim * sizeof(float), off_hb);
	LAYOUT_FIELD(hb2_size * sizeof(float), off_hb2);
	LAYOUT_FIELD(xq_size * sizeof(char), off_xq_q);
	LAYOUT_FIELD(xq_s_count * sizeof(float), off_xq_s);
	LAYOUT_FIELD((__int64)p->hidden_dim * sizeof(char), off_hq_q);
	LAYOUT_FIELD(hq_s_count * sizeof(float), off_hq_s);
	LAYOUT_FIELD((all_heads_dim + 2 * kv_dim) * sizeof(float), off_q);
	LAYOUT_FIELD((__int64)p->n_heads * p->seq_len * sizeof(float), off_att);
	LAYOUT_FIELD((__int64)p->vocab_size * sizeof(float), off_logits);
	LAYOUT_FIELD((__int64)p->seq_len * p->dim * sizeof(float), off_x_storage);
	LAYOUT_FIELD((__int64)p->seq_len * all_heads_dim * sizeof(float), off_q_storage);
	LAYOUT_FIELD((__int64)p->seq_len * all_heads_dim * sizeof(float), off_att_out);
	__int64 kv_single = (__int64)p->n_layers * p->seq_len * kv_dim * sizeof(float);
	LAYOUT_FIELD(kv_single, off_key_cache);
	LAYOUT_FIELD(kv_single, off_value_cache);
#undef LAYOUT_FIELD

	ml->rs_block_bytes = off;
	ml->emb_table_bytes = (__int64)p->vocab_size * p->dim * sizeof(float);
	ml->vocab_ptrs_bytes = (__int64)p->vocab_size * sizeof(char *);
}

/* ---------------------------------------------------------------------------
 * vocab bulk serialization
 * ------------------------------------------------------------------------- */

void save_vocab_bulk(tokenizer *t)
{
	unsigned int count = (unsigned int)t->vocab_size;
	int total = 4;
	for (unsigned int i = 0; i < count; i++)
	{
		int vocab_len = 0;
		if (t->vocab[i])
			vocab_len = (int)strlen(t->vocab[i]);
		total += 4 + vocab_len;
	}

	char *buf = (char *)db_alloc(total);
	int off = 0;
	memcpy(buf + off, &count, 4);
	off += 4;
	for (unsigned int i = 0; i < count; i++)
	{
		unsigned int len = 0;
		if (t->vocab[i])
			len = (unsigned int)strlen(t->vocab[i]);
		memcpy(buf + off, &len, 4);
		off += 4;
		if (len)
		{
			memcpy(buf + off, t->vocab[i], len);
			off += len;
		}
	}
	HT_ADD_CHECK("__sys_vocab_bulk", buf, total, 1);
	db_free(buf);
	printf("[INFO] vocab_bulk: %u entries, %d bytes\n", (unsigned)count, total);
}

/* ---------------------------------------------------------------------------
 * INI-to-DB -- stores dispatch params from threading_proxy.ini into microdb.
 * the ASM initialize_once reads __ini__default__* and __ini__proxy__* keys.
 * ported from runq.c ini_to_db() (L2389-2522).
 * ------------------------------------------------------------------------- */

int opt_tuner_delay = 80; /* -tuner-delay <n>: seeds "tuner:delaydefault", matching-shape calls skipped between A/B pairs (default 80) */
int opt_tuner_delay_large = 1; /* -tuner-delay-large <n>: seeds "tuner:delay_large", the initial delay for shapes with a dimension over 16384 (default 1: XL shapes dispatch ~once per token, a small delay keeps their streams completing) */
int opt_tuner_sex = 80; /* -tuner-sex <p>: seeds "tuner:sex_percent", the crossover donation MAX percentage for config_mutate_genome; the actual donated dword count is drawn in [0, p], that variance is intended (default 80) */
int opt_tuner_mutate = 50; /* -tuner-mutate <p>: seeds "tuner:tail_slots" as a COUNT of the 128 population slots (p*128/100, legal range [0,128]); the ASM stores the number, never a percentage (default 50 = half the population) */
int opt_tuner_mutations_max = 40; /* -tuner-mutations-max <n>: seeds "tuner:mutations_max", the per-child field-count draw bound for config_mutate_genome (1..n fields changed, default 40) */
int opt_tuner_rand_delay = 0; /* -tuner-rand-delay <n>: seeds "tuner:rand_delay", the delay-phase length jitter cap; each phase finish adds rand(0..n) to the delay reload so measurements land on varying phase offsets (default 0 = exact delay) */
int opt_simple_threads = 0; /* -simple-threads <n>: seeds "init:simple_threads", the worker-pool mode switch (default 0 = full placement pool; N > 0 = N plain worker threads, no pinning, no SMT awareness, priority only) */

void ini_to_db(int model_dim, int model_hidden)
{
	// resolve threading_proxy.ini to absolute path (get_private_profile_string_a needs it)
	char ini_path[MAX_PATH];
	if (GetFullPathNameA("threading_proxy.ini", sizeof(ini_path), ini_path, 0) == 0)
	{
		printf("[FATAL] threading_proxy.ini not found -- a database without __ini__ dispatch keys is incomplete (run from the project root)\n");
		exit(EXIT_FAILURE);
	}
	FILE *f = fopen(ini_path, "r");
	if (!f)
	{
		printf("[FATAL] threading_proxy.ini not found at %s -- a database without __ini__ dispatch keys is incomplete (run from the project root)\n", ini_path);
		exit(EXIT_FAILURE);
	}
	fclose(f);
	printf("  [13] ini_to_db: reading %s\n", ini_path);

	struct
	{
		char *key;
		int def;
	} dispatch_defaults[] = {
		{ "num_chunks", 64 },
		{ "min_chunk_cols", 4 },
		{ "batch_small", 2 },
		{ "batch_medium", 4 },
		{ "batch_large", 8 },
		{ "batch_huge", 16 },
		{ "batch_med_thresh", 4 },
		{ "batch_large_thresh", 8 },
		{ "batch_huge_thresh", 16 },
		{ "caller_batch_small", 1 },
		{ "caller_batch_medium", 2 },
		{ "caller_batch_large", 4 },
		{ "caller_batch_huge", 8 },
		{ "caller_batch_med_thresh", 4 },
		{ "caller_batch_large_thresh", 8 },
		{ "caller_batch_huge_thresh", 16 },
		{ "worker_priority", 15 },
		{ "chunk_multiplier", 16 },
		{ "cores_subtract", 2 },
		{ "use_physical_cores", 1 },
		{ "chunk_split_pct", 0 },
		{ "chunk_group_1_count", 16 },
		{ "chunk_group_2_count", 128 },
		{ "quantize_chunk_groups", 4 },
		{ "taper_shift", 48 },
		/* vi_t linear dispatch tuning (vit_speedup.md P4; live-tuned by
		 * tools\tune_vit_dispatch.exe, which writes these [default] keys) */
		{ "vit_max_chunks", 32 },
		{ "vit_min_chunk_cols", 64 },
	};
	int ndispatch = (int)(sizeof(dispatch_defaults) / sizeof(dispatch_defaults[0]));

	/* build section list from supported_resolutions in [default] -- the list
	 * outgrew one INI line (Windows truncates single values at ~600 chars),
	 * so continuation keys supported_resolutions_2.._N are concatenated */
	char res_buf[4096];
	{
		char part[512];
		int part_idx = 1;
		size_t used = 0;
		res_buf[0] = 0;
		for (;;)
		{
			if (part_idx == 1)
				GetPrivateProfileStringA("default", "supported_resolutions", "", part, sizeof(part), ini_path);
			else
			{
				char keyname[64];
				_snprintf(keyname, sizeof(keyname), "supported_resolutions_%d", part_idx);
				GetPrivateProfileStringA("default", keyname, "", part, sizeof(part), ini_path);
			}
			if (part[0] == 0)
				break; /* key absent or empty: end of the list */
			size_t len = strlen(part);
			if (used + len + 2 >= sizeof(res_buf))
			{
				/* a silently truncated shape list would drop databases' shapes
				 * without a trace -- refuse instead of building a DB that
				 * FATALs at dispatch on the first missing shape. */
				printf("[FATAL] supported_resolutions list exceeds the %d-byte merge buffer at continuation key _%d -- split the INI list further\n",
				       (int)sizeof(res_buf), part_idx);
				exit(EXIT_FAILURE);
			}
			if (used && res_buf[used - 1] != ' ')
				res_buf[used++] = ' ';
			memcpy(res_buf + used, part, len + 1);
			used += len;
			part_idx++;
		}
	}

	/* strip inline comments (windows INI: ; is not a comment char in API, but our INI uses it) */
	char *semi = strchr(res_buf, ';');
	if (semi)
		*semi = 0;

	/* Private copy of the merged list BEFORE tokenization: the section parser
	 * below tokenizes res_buf IN PLACE (sections[] points into it, each token
	 * NUL-terminated), so the [default] supported_resolutions storage must
	 * read this copy, not the shredded original. */
	static char res_list_copy[4096];
	_snprintf(res_list_copy, sizeof(res_list_copy), "%s", res_buf);

#define MAX_SECTIONS 256
	char *sections[MAX_SECTIONS];
	int n_sections = 0;
	sections[n_sections++] = "proxy";
	sections[n_sections++] = "default";

	/* parse space-separated nx_d tokens */
	char *p = res_buf;
	while (*p && n_sections < MAX_SECTIONS)
	{
		while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
			p++;
		if (*p == 0)
			break;
		char *tok = p;
		while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
			p++;
		if (*p)
		{
			*p = 0;
			p++;
		}
		sections[n_sections++] = tok;
	}

	char dbkey[512];
	int total_keys = 0;
	/* tuner:cycle_active = the in-process cycle tuner's enable switch,
	 * DECIMAL STRING (the ASM reader parses digits, same as the delay keys):
	 * "1" = the tuner runs (default), "0" = fully dormant, no slot seeding,
	 * no allocation, no stepping. The switch is independent of the delay
	 * values. */
	{
		char cyclestr[4];
		_snprintf(cyclestr, sizeof(cyclestr), "%d", 1);
		HT_ADD_CHECK("tuner:cycle_active", cyclestr, (int)strlen(cyclestr) + 1, 1);
		printf("  [13] ini_to_db: tuner:cycle_active = %s (in-process cycle tuner)\n", cyclestr);
		total_keys++;
	}
	/* tuner:delaydefault = matching-shape calls skipped between episodes.
	 * Required tuner keys are seeded HERE, by the database tool -- the tuner
	 * runs constantly and transparently with no host or script-side
	 * initialization. Seeded from -tuner-delay <n> (default 80); a larger
	 * delay spreads the measurement over more generation turns. */
	{
		char delaydefault[16];
		_snprintf(delaydefault, sizeof(delaydefault), "%d", opt_tuner_delay);
		HT_ADD_CHECK("tuner:delaydefault", delaydefault, (int)strlen(delaydefault) + 1, 1);
		printf("  [13] ini_to_db: tuner:delaydefault = %s\n", delaydefault);
		total_keys++;
	}
	/* tuner:delay_large = the initial per-shape delay for shapes with EITHER
	 * dimension over 16384 (the LM head family, ~1 dispatch per token). Seeded
	 * alongside tuner:delaydefault from -tuner-delay-large <n> (default 1);
	 * every shape copies its value on first registration. */
	{
		char delaylarge[16];
		_snprintf(delaylarge, sizeof(delaylarge), "%d", opt_tuner_delay_large);
		HT_ADD_CHECK("tuner:delay_large", delaylarge, (int)strlen(delaylarge) + 1, 1);
		printf("  [13] ini_to_db: tuner:delay_large = %s (initial delay for shapes with a dimension > 16384)\n", delaylarge);
		total_keys++;
	}
	/* tuner:sex_percent = the crossover donation MAX percentage for the tail
	 * mutation's config_mutate_genome (decimal string, its ASM reader parses
	 * it digit by digit). Each child receives a random number of donor dwords
	 * in [0, percent] -- a MAX cap, the variance is intended. Seeded from
	 * -tuner-sex <p> (default 80). */
	{
		char sexpct[16];
		_snprintf(sexpct, sizeof(sexpct), "%d", opt_tuner_sex);
		HT_ADD_CHECK("tuner:sex_percent", sexpct, (int)strlen(sexpct) + 1, 1);
		printf("  [13] ini_to_db: tuner:sex_percent = %s (crossover donation max percent)\n", sexpct);
		total_keys++;
	}
	/* tuner:mutations_max = the per-child field-count draw bound for
	 * config_mutate_genome (decimal string). Each mutated child changes
	 * between 1 and this many matmul dwords. Seeded from
	 * -tuner-mutations-max <n> (default 40). */
	{
		char mutmax[16];
		_snprintf(mutmax, sizeof(mutmax), "%d", opt_tuner_mutations_max);
		HT_ADD_CHECK("tuner:mutations_max", mutmax, (int)strlen(mutmax) + 1, 1);
		printf("  [13] ini_to_db: tuner:mutations_max = %s (per-child field-count draw bound)\n", mutmax);
		total_keys++;
	}
	/* tuner:tail_slots = the mutant tail SIZE as a COUNT of the 128 population
	 * slots (the ASM stores the number, never a percentage). Converted here
	 * from -tuner-mutate <p> percent: count = p*128/100, legal range [0,128].
	 * The ASM tail loop runs exactly this many slots from index 127 down. */
	{
		char tailslots[16];
		int tail_count = opt_tuner_mutate * 128 / 100;
		if (tail_count < 0)
			tail_count = 0;
		if (tail_count > 128)
			tail_count = 128;
		_snprintf(tailslots, sizeof(tailslots), "%d", tail_count);
		HT_ADD_CHECK("tuner:tail_slots", tailslots, (int)strlen(tailslots) + 1, 1);
		printf("  [13] ini_to_db: tuner:tail_slots = %s (%d percent -> %d slots of 128)\n", tailslots, opt_tuner_mutate, tail_count);
		total_keys++;
	}
	/* tuner:rand_delay = the delay-phase length jitter cap (decimal string).
	 * Each phase finish the ASM adds rand(0..cap) to the delay reload, so
	 * consecutive measurement episodes land on varying delay offsets and
	 * the phase-boundary effects average out of the folds. 0 = exact
	 * delay. Seeded from -tuner-rand-delay <n> (default 0). */
	{
		char randd[16];
		_snprintf(randd, sizeof(randd), "%d", opt_tuner_rand_delay);
		HT_ADD_CHECK("tuner:rand_delay", randd, (int)strlen(randd) + 1, 1);
		printf("  [13] ini_to_db: tuner:rand_delay = %s (delay jitter cap)\n", randd);
		total_keys++;
	}

	/* init:simple_threads = the worker-pool mode switch, raw u32 (its only
	 * reader is ASM, ht_find + dword load): 0 = full placement pool (topology
	 * scan + pinning, the default), N > 0 = N plain worker threads created
	 * without pinning or SMT awareness, SetThreadPriority is the only extra
	 * step. Seeded from -simple-threads <n>. */
	HT_ADD_CHECK("init:simple_threads", &opt_simple_threads, sizeof(opt_simple_threads), 1);
	printf("  [13] ini_to_db: init:simple_threads = %d (0 = full placement, N = N simple unpinned workers)\n", opt_simple_threads);
	total_keys++;

	for (int s = 0; s < n_sections; s++)
	{
		char *sec = sections[s];
		int sec_taper_shift = -1;	/* this section's taper_shift (-1 = key not seen yet) */
		int sec_min_chunk_cols = -1;	/* this section's min_chunk_cols (-1 = key not seen yet) */

		/* dispatch params */
		for (int i = 0; i < ndispatch; i++)
		{
			int val = GetPrivateProfileIntA(sec, dispatch_defaults[i].key, dispatch_defaults[i].def, ini_path);
			/* creation-time rejection of illegal num_chunks (Sec.57d policy rule 5:
			 * value rejection belongs at creation, never correction at use) */
			if ((strcmp(dispatch_defaults[i].key, "num_chunks") == 0) && ((val < 0) || (val > 4096)))
			{
				printf("[FATAL] ini_to_db: num_chunks=%d outside [0, 4096] in section [%s]\n", val, sec);
				exit(EXIT_FAILURE);
			}
			/* batch-size keys: a 0 batch freezes the steal xadd (claims nothing,
			 * advances nothing -- frozen-xadd spin); a negative batch wraps the
			 * unsigned start compare. Reject outside the documented [1,64] range
			 * at creation, same policy as num_chunks (autotuner enforces 1..64).
			 * Threshold keys are compare operands, never xadd addends -- legal. */
			int is_batch_size_key =
				(strcmp(dispatch_defaults[i].key, "batch_small") == 0) ||
				(strcmp(dispatch_defaults[i].key, "batch_medium") == 0) ||
				(strcmp(dispatch_defaults[i].key, "batch_large") == 0) ||
				(strcmp(dispatch_defaults[i].key, "batch_huge") == 0) ||
				(strcmp(dispatch_defaults[i].key, "caller_batch_small") == 0) ||
				(strcmp(dispatch_defaults[i].key, "caller_batch_medium") == 0) ||
				(strcmp(dispatch_defaults[i].key, "caller_batch_large") == 0) ||
				(strcmp(dispatch_defaults[i].key, "caller_batch_huge") == 0);
			if (is_batch_size_key && ((val < 1) || (val > 64)))
			{
				printf("[FATAL] ini_to_db: %s=%d outside [1, 64] in section [%s]\n", dispatch_defaults[i].key, val, sec);
				exit(EXIT_FAILURE);
			}
			/* split-path group counts: a negative count passes every signed
			 * clamp and makes div produce a zero chunk dimension -- reject at
			 * creation like num_chunks (zero stays legal, the engine's ==0
			 * boundary adjustments handle it). */
			if ((strcmp(dispatch_defaults[i].key, "chunk_group_1_count") == 0 ||
				 strcmp(dispatch_defaults[i].key, "chunk_group_2_count") == 0) && (val < 0))
			{
				printf("[FATAL] ini_to_db: %s=%d negative in section [%s]\n", dispatch_defaults[i].key, val, sec);
				exit(EXIT_FAILURE);
			}
			/* vit linear dispatch tuning: both are divisors and positive loop
			 * bounds in the hot dispatch path (0 -> #DE, negative -> bogus
			 * ceiling / unbounded fill). Reject outside [1, ...] at creation,
			 * same policy as num_chunks. */
			if ((strcmp(dispatch_defaults[i].key, "vit_min_chunk_cols") == 0 ||
				 strcmp(dispatch_defaults[i].key, "vit_max_chunks") == 0) && (val < 1))
			{
				printf("[FATAL] ini_to_db: %s=%d outside [1, ...] in section [%s]\n", dispatch_defaults[i].key, val, sec);
				exit(EXIT_FAILURE);
			}
			/* taper geometry: shift outside [0..112] leaves the hot path's
			 * unsigned bound; split pct outside [0..99] leaves the split
			 * boundary; min_chunk_cols below 1 breaks the uniform divisor.
			 * Reject at creation, same policy as num_chunks. */
			if ((strcmp(dispatch_defaults[i].key, "taper_shift") == 0) && ((val < 0) || (val > 112)))
			{
				printf("[FATAL] ini_to_db: taper_shift=%d outside [0..112] in section [%s]\n", val, sec);
				exit(EXIT_FAILURE);
			}
			if ((strcmp(dispatch_defaults[i].key, "chunk_split_pct") == 0) && ((val < 0) || (val > 99)))
			{
				printf("[FATAL] ini_to_db: chunk_split_pct=%d outside [0..99] in section [%s]\n", val, sec);
				exit(EXIT_FAILURE);
			}
			if (strcmp(dispatch_defaults[i].key, "min_chunk_cols") == 0)
			{
				if (val < 1)
				{
					printf("[FATAL] ini_to_db: min_chunk_cols=%d outside [1, ...] in section [%s]\n", val, sec);
					exit(EXIT_FAILURE);
				}
				sec_min_chunk_cols = val;
			}
			if (strcmp(dispatch_defaults[i].key, "taper_shift") == 0)
				sec_taper_shift = val;
			_snprintf(dbkey, sizeof(dbkey), "__ini__%s__%s", sec, dispatch_defaults[i].key);
			HT_ADD_CHECK(dbkey, &val, sizeof(val), 1);
			total_keys++;
		}

		/* joint taper geometry check (both keys of the section are read now:
		 * a shape that runs the taper needs W_min >= 4 -- the Q8 4-row
		 * kernel block; uniform/split shapes only need >= 1) */
		if ((sec_taper_shift > 0) && (sec_min_chunk_cols >= 0) && (sec_min_chunk_cols < 4))
		{
			printf("[FATAL] ini_to_db: min_chunk_cols=%d < 4 with taper_shift=%d in section [%s] (Q8 4-row kernel block)\n", sec_min_chunk_cols, sec_taper_shift, sec);
			exit(EXIT_FAILURE);
		}

		/* 41 prefetch distances + 41 prefetch types (PF_DIST_TABLE_ENTRIES in
		 * v7_data.inc; slots 41-48 belonged to the removed fused-Q8 kernel)
		 * default/proxy sections: key = "__ini__<sec>__pf_dist_<i>" (double underscore)
		 *   because ini_db_def = "__ini__default__" already has trailing "__" and
		 *   v7_init.inc appends "pf_dist_" directly via strcpy_local (no dec rdi).
		 * per-shape sections (nx_d): key = "__ini__<sec>_pf_dist_<i>" (single underscore)
		 *   because v7_init.inc rebuilds prefix as "__ini__NxD__" then does
		 *   dec rdi + strcpy which overwrites the second underscore with 'p'.
		 */
		int is_def_proxy = (strcmp(sec, "default") == 0 || strcmp(sec, "proxy") == 0);
		for (int i = 0; i < 41; i++)
		{
			char key[32];
			int val;

			_snprintf(key, sizeof(key), "pf_dist_%d", i);
			val = GetPrivateProfileIntA(sec, key, 0, ini_path);
			if (is_def_proxy)
				_snprintf(dbkey, sizeof(dbkey), "__ini__%s__pf_dist_%d", sec, i);
			else
				_snprintf(dbkey, sizeof(dbkey), "__ini__%s_pf_dist_%d", sec, i);
			HT_ADD_CHECK(dbkey, &val, sizeof(val), 1);
			total_keys++;

			_snprintf(key, sizeof(key), "pf_type_%d", i);
			val = GetPrivateProfileIntA(sec, key, 0, ini_path);
			if (is_def_proxy)
				_snprintf(dbkey, sizeof(dbkey), "__ini__%s__pf_type_%d", sec, i);
			else
				_snprintf(dbkey, sizeof(dbkey), "__ini__%s_pf_type_%d", sec, i);
			HT_ADD_CHECK(dbkey, &val, sizeof(val), 1);
			total_keys++;
		}

		/* supported_resolutions string (not in [proxy]). For [default] this is
		 * the MERGED list (res_buf, continuation keys included) -- the engine
		 * builds its per-shape blocks from this key, and the raw line would
		 * carry the ~600-char GetPrivateProfileStringA truncation. */
		if (strcmp(sec, "proxy") != 0)
		{
			/* res_buf was tokenized IN PLACE (sections[] points into it, the
			 * parser NUL-terminates each token) -- only the private copy made
			 * before tokenization holds the full merged list. */
			const char *list = res_list_copy;
			if (strcmp(sec, "default") != 0)
			{
				static char buf[1024];
				GetPrivateProfileStringA(sec, "supported_resolutions", "", buf, sizeof(buf), ini_path);
				char *semi2 = strchr(buf, ';');
				if (semi2)
					*semi2 = 0;
				list = buf;
			}
			_snprintf(dbkey, sizeof(dbkey), "__ini__%s__supported_resolutions", sec);
			HT_ADD_CHECK(dbkey, list, (int)strlen(list) + 1, 1);
			total_keys++;
		}
	}

	printf("  [13] ini_to_db: %d keys across %d sections\n", total_keys, n_sections);
}

/* ---------------------------------------------------------------------------
 * populate hash table -- the main build logic
 * ------------------------------------------------------------------------- */

void populate_db(transformer *t, tokenizer *tok, float rope_base_override, float eps_override, rope_scaling_header *rs)
{
	/* 1. config */
	HT_ADD_CHECK("__sys_config", &t->config, sizeof(Config), 1);
	printf("  [1] config: dim=%d hidden=%d layers=%d heads=%d kv=%d head_dim=%d vocab=%d seq=%d shared=%d GS=%d qk_norm=%d rope_base=%d\n", t->config.dim, t->config.hidden_dim, t->config.n_layers, t->config.n_heads, t->config.n_kv_heads,
	       t->config.head_dim, t->config.vocab_size, t->config.seq_len, t->config.shared_classifier, t->config.group_size, t->config.has_qk_norm, t->config.rope_base);

	/* 2. RoPE base as float */
	float rb;
	if (rope_base_override > 0.0f)
		rb = rope_base_override;
	else
		rb = (float)t->config.rope_base;
	HT_ADD_CHECK("__sys_rope_base", &rb, sizeof(float), 1);
	printf("  [2] rope_base = %.1f\n", rb);

	/* 2b. RoPE scaling (llama3 style). always written; defaults when absent. */
	{
		int rope_type = 0;
		if (rs)
			rope_type = rs->rope_type;
		float rope_factor = 1.0f;
		if (rs)
			rope_factor = rs->factor;
		float rope_high_freq = 4.0f;
		if (rs)
			rope_high_freq = rs->high_freq_factor;
		float rope_low_freq = 1.0f;
		if (rs)
			rope_low_freq = rs->low_freq_factor;
		int rope_orig_max_pos = 0;
		if (rs)
			rope_orig_max_pos = rs->orig_max_pos;
		HT_ADD_CHECK("__sys_rope_type", &rope_type, sizeof(rope_type), 1);
		HT_ADD_CHECK("__sys_rope_factor", &rope_factor, sizeof(rope_factor), 1);
		HT_ADD_CHECK("__sys_rope_high_freq_factor", &rope_high_freq, sizeof(rope_high_freq), 1);
		HT_ADD_CHECK("__sys_rope_low_freq_factor", &rope_low_freq, sizeof(rope_low_freq), 1);
		HT_ADD_CHECK("__sys_rope_orig_max_pos", &rope_orig_max_pos, sizeof(rope_orig_max_pos), 1);
		printf("  [2b] rope_scaling: type=%d factor=%.3f high_freq=%.3f low_freq=%.3f orig_max_pos=%d\n", rope_type, rope_factor, rope_high_freq, rope_low_freq, rope_orig_max_pos);
	}

	/* 2c. RMS norm epsilon. read from header byte 84 (float32 written by export.py).
	 * default 1e-6 for older .bin files that don't have this field.
	 * a 0.0 header value means the field was never written (stale export) --
	 * eps=0 turns 1/sqrt(0) into +inf on all-zero activation rows, NaN-poisons
	 * the logits, and freezes argmax on one token (the katakana-loop signature).
	 * the -eps CLI flag overrides both. */
	float norm_eps = DEFAULT_NORM_EPS;
	if (eps_override > 0.0f)
		norm_eps = eps_override;
	else
	{
		if (t->file_size >= HEADER_NORM_EPS_OFFSET + sizeof(float))
			memcpy(&norm_eps, (char *)t->data + HEADER_NORM_EPS_OFFSET, sizeof(float));
		if (norm_eps <= 0.0f)
		{
			/* a 0.0 header value means the field was never written (stale
			 * export) -- silently building with the default would poison
			 * every norm the wrong way. Name the value and refuse. */
			printf("[FATAL] checkpoint header rms_eps = %g (field never written by export.py); rebuild the checkpoint or pass -eps explicitly\n",
			       norm_eps);
			exit(EXIT_FAILURE);
		}
	}
	HT_ADD_CHECK("__sys_rms_eps", &norm_eps, sizeof(norm_eps), 1);
	printf("  [2c] rms_eps = %g\n", norm_eps);

	/* 3. model weights (past 256-byte header) */
	if (t->file_size >= 256)
	{
		void *model_data = (char *)t->data + 256;
		__int64 model_data_size = t->file_size - 256;
		HT_ADD_CHECK("__sys_model_weights", model_data, model_data_size, 1);
		printf("  [3] model_weights: %lld MB\n", (long long)(model_data_size >> 20));
	}

	/* 4. weight metadata */
	{
		Config *p = &t->config;
		unsigned int GS_val = (unsigned int)GS;
		struct
		{
			__int64 file_offset;        /* 8 bytes */
			unsigned int rows;          /* 4 bytes */
			unsigned int cols;          /* 4 bytes */
			unsigned char is_quantized; /* 1 byte */
			unsigned char reserved[3];  /* 3 bytes padding */
			unsigned int _pad;          /* 4 bytes padding to match ASM WM_ENTRY_SIZE=24 */
		} wm[16];
		memset(wm, 0, sizeof(wm));

		__int64 offset = 0;
		/* fp32 weights */
		wm[0].file_offset = offset;
		wm[0].rows = (unsigned int)(p->n_layers * p->dim);
		wm[0].is_quantized = 0;
		offset += (__int64)wm[0].rows * 4;

		wm[1].file_offset = offset;
		wm[1].rows = (unsigned int)(p->n_layers * p->dim);
		wm[1].is_quantized = 0;
		offset += (__int64)wm[1].rows * 4;

		wm[2].file_offset = offset;
		wm[2].rows = (unsigned int)p->dim;
		wm[2].is_quantized = 0;
		offset += (__int64)wm[2].rows * 4;

		wm[3].file_offset = offset;
		wm[3].rows = (unsigned int)(p->n_layers * p->head_dim);
		wm[3].is_quantized = 0;
		offset += (__int64)wm[3].rows * 4;

		wm[4].file_offset = offset;
		wm[4].rows = (unsigned int)(p->n_layers * p->head_dim);
		wm[4].is_quantized = 0;
		offset += (__int64)wm[4].rows * 4;

		/* qkv_bias (fp32, not in weight_meta slots -- ASM computes pointer from
		 * k_norm_weights + n_layers * head_dim * 4). advance offset past it. */
		offset += (__int64)p->n_layers * (p->n_heads * p->head_dim + 2 * p->n_kv_heads * p->head_dim) * 4;

#define SAVE_QT(idx, size_each)                                                                                                                                                                                                                          \
	do                                                                                                                                                                                                                                                   \
	{                                                                                                                                                                                                                                                    \
		__int64 _se = (__int64)(size_each);                                                                                                                                                                                                              \
		wm[idx].file_offset = offset;                                                                                                                                                                                                                    \
		wm[idx].rows = (unsigned int)(_se);                                                                                                                                                                                                              \
		wm[idx].is_quantized = 1;                                                                                                                                                                                                                        \
		offset += _se;                                                                                                                                                                                                                                   \
		offset += (__int64)((_se + GS_val - 1) / GS_val) * 4;                                                                                                                                                                                            \
	} while (0)

		SAVE_QT(5, (__int64)(p->vocab_size) * p->dim);
		SAVE_QT(6, (__int64)(p->n_layers) * p->dim * p->n_heads * p->head_dim);
		SAVE_QT(7, (__int64)(p->n_layers) * p->dim * p->n_kv_heads * p->head_dim);
		SAVE_QT(8, (__int64)(p->n_layers) * p->dim * p->n_kv_heads * p->head_dim);
		SAVE_QT(9, (__int64)(p->n_layers) * p->n_heads * p->head_dim * p->dim);
		SAVE_QT(10, (__int64)(p->n_layers) * p->dim * p->hidden_dim);
		SAVE_QT(11, (__int64)(p->n_layers) * p->hidden_dim * p->dim);
		SAVE_QT(12, (__int64)(p->n_layers) * p->dim * p->hidden_dim);

		wm[13] = wm[5]; /* duplicate q_tokens */

		if (p->shared_classifier)
			wm[14] = wm[5];
		else
			SAVE_QT(14, (__int64)(p->dim) * p->vocab_size);

		wm[15].rows = (unsigned int)(p->vocab_size * p->dim);
		wm[15].is_quantized = 0;

#undef SAVE_QT

		/* import size check: the blob stored under __sys_model_weights must be
		 * EXACTLY the layout the config implies (everything up to and including
		 * the last quantized region; wm[13..15] are descriptor duplicates of
		 * existing entries / the dequant buffer, not file regions). A short
		 * blob is a truncated checkpoint -- the inference importer would index
		 * past the blob end with weight_meta offsets. A larger blob is a
		 * foreign layout. Both refuse the import here, at the boundary. */
		if (t->file_size - 256 != offset)
		{
			printf("[FATAL] checkpoint weight layout mismatch: config implies %lld bytes of weights, file carries %lld\n",
			       (long long)offset, (long long)(t->file_size - 256));
			printf("        (stale or foreign checkpoint -- re-export with tools/export.py)\n");
			exit(1);
		}

		/* diagnostic: dump raw bytes of weight_meta to verify layout */
		printf("  [4] weight_meta: sizeof(wm[0])=%d sizeof(wm)=%d\n", (int)sizeof(wm[0]), (int)sizeof(wm));
		for (int i = 0; i < 16; i++)
		{
			unsigned char *p = (unsigned char *)&wm[i];
			printf("    [%2d] raw: ", i);
			for (int b = 0; b < (int)sizeof(wm[0]); b++)
				printf("%02X ", p[b]);
			printf(" | fo=%llu rows=%u cols=%u iq=%u\n", (unsigned long long)wm[i].file_offset, wm[i].rows, wm[i].cols, wm[i].is_quantized);
		}
		HT_ADD_CHECK("__sys_weight_meta", wm, sizeof(wm), 1);
		printf("  [4] weight_meta: 16 entries stored\n");
	}

	/* 5. Tokenizer header */
	{
		struct
		{
			int max_token_len;
			unsigned bos_id;
			unsigned eos_id;
		} tok_hdr;
		tok_hdr.max_token_len = tok->max_token_length;
		tok_hdr.bos_id = tok->bos_token_id;
		tok_hdr.eos_id = tok->eos_token_id;
		HT_ADD_CHECK("__sys_tokenizer_hdr", &tok_hdr, sizeof(tok_hdr), 1);
		printf("  [5] Tokenizer_hdr: vocab=%d max_len=%u bos=%u eos=%u\n", tok->vocab_size, tok->max_token_length, tok->bos_token_id, tok->eos_token_id);
	}

	/* 5b. stop token IDs (eos_token_id plus any additional turn-ending tokens).
	 * ALWAYS written -- even for num_stop_ids == 1 (just eos) -- so the ASM init
	 * path can treat a missing key as a FATAL error. NO silent defaults. */
	{
		unsigned int stop_arr[5];
		stop_arr[0] = tok->num_stop_ids;
		for (unsigned int si = 0; si < tok->num_stop_ids; si++)
			stop_arr[1 + si] = tok->stop_ids[si];
		HT_ADD_CHECK("__sys_stop_token_ids", stop_arr, sizeof(unsigned int) * (1 + tok->num_stop_ids), 1);
		printf("  [5b] stop_token_ids (%u ids):", tok->num_stop_ids);
		for (unsigned int si = 0; si < tok->num_stop_ids; si++)
			printf(" %u", tok->stop_ids[si]);
		printf("\n");
	}

	/* 6. merge scores */
	HT_ADD_CHECK("__sys_merge_scores", tok->merge_scores, (unsigned int)(tok->vocab_size * sizeof(float)), 1);
	printf("  [6] merge_scores: %d floats\n", tok->vocab_size);

	/* 7. vocab bulk */
	save_vocab_bulk(tok);

	/* 8. per-token forward lookups */
	printf("  [8] forward lookups: %d entries... ", tok->vocab_size);
	for (int i = 0; i < tok->vocab_size; i++)
		HT_ADD_CHECK(tok->vocab[i], &i, sizeof(i), 1);
	printf("OK\n");

	/* 9. reverse lookups (_ID_N) -- NOT needed by inference, omitted to keep hash table small */

	/* 10. byte-fallback aliases (<0xNN> -> raw byte)
	 * only add alias if the raw byte is NOT already a vocab key.
	 * without this guard, byte-fallback entries like <0x3C>->63 overwrite
	 * proper vocab entries like "<"->529, corrupting tokenization. */
	{
		int alias_count = 0;
		int skip_count = 0;
		for (int i = 0; i < tok->vocab_size; i++)
		{
			char *s = tok->vocab[i];
			int slen = 0;
			if (s)
				slen = (int)strlen(s);
			if (slen == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>')
			{
				int hi;
				if (s[3] >= 'A')
					hi = s[3] - 'A' + 10;
				else if (s[3] >= 'a')
					hi = s[3] - 'a' + 10;
				else
					hi = s[3] - '0';
				int lo;
				if (s[4] >= 'A')
					lo = s[4] - 'A' + 10;
				else if (s[4] >= 'a')
					lo = s[4] - 'a' + 10;
				else
					lo = s[4] - '0';
				if (hi >= 0 && hi <= 15 && lo >= 0 && lo <= 15)
				{
					unsigned char byte_val = (unsigned char)((hi << 4) | lo);
					char byte_str[2] = { (char)byte_val, 0 };
					if (dll.ht_find && dll.ht_find(byte_str, 0) != 0)
					{
						skip_count++;
						continue;
					}
									HT_ADD_CHECK(byte_str, &i, sizeof(i), 1);
					alias_count++;
				}
			}
		}
		if (alias_count > 0)
			printf("  [10] byte-fallback aliases: %d (skipped %d already in vocab)\n", alias_count, skip_count);
	}

	/* 11. templates (base pair = NON-thinking, mandatory) + suffixed variants */
	if (tok->prompt_template[0])
		HT_ADD_CHECK("__sys_template", tok->prompt_template, (unsigned int)strlen(tok->prompt_template) + 1, 1);
	if (tok->system_prompt_template[0])
		HT_ADD_CHECK("__sys_template_sys", tok->system_prompt_template, (unsigned int)strlen(tok->system_prompt_template) + 1, 1);
	/* image-turn template (optional, loaded from ".bin.template.img" sidecar):
	 * models with image-conditional chat templates (SmolVLM/Idefics3) render
	 * image turns differently from text turns. stored under __sys_template@img;
	 * absent or identical-to-base sidecar stores nothing (engine falls back). */
	if (tok->image_prompt_template[0] && strcmp(tok->image_prompt_template, tok->prompt_template) != 0)
	{
		HT_ADD_CHECK("__sys_template@img", tok->image_prompt_template, (unsigned int)strlen(tok->image_prompt_template) + 1, 1);
		printf("  [11] image-turn template stored (%u bytes)\n", (unsigned)strlen(tok->image_prompt_template));
	}
	/* thinking templates (optional, chat_template_usage_analysis.md): models
	 * whose thinking sidecars differ from base (qwen3, MiniCPM5) get the
	 * thinking-ON variants under @think keys; v7s_enable_thinking swaps them in
	 * at render time. identical-to-base sidecars store nothing (no-op flag). */
	if (tok->think_prompt_template[0] && strcmp(tok->think_prompt_template, tok->prompt_template) != 0)
	{
		HT_ADD_CHECK("__sys_template@think", tok->think_prompt_template, (unsigned int)strlen(tok->think_prompt_template) + 1, 1);
		printf("  [11] thinking template stored (%u bytes)\n", (unsigned)strlen(tok->think_prompt_template));
	}
	if (tok->think_system_prompt_template[0] && strcmp(tok->think_system_prompt_template, tok->system_prompt_template) != 0)
	{
		HT_ADD_CHECK("__sys_template_sys@think", tok->think_system_prompt_template, (unsigned int)strlen(tok->think_system_prompt_template) + 1, 1);
		printf("  [11] sys thinking template stored (%u bytes)\n", (unsigned)strlen(tok->think_system_prompt_template));
	}
	/* inter-turn templates (optional, phase4.md): stored under __sys_template@multi
	 * when they differ from base; the thinking variant only when it differs from
	 * plain @multi (models without thinking variants produce identical files).
	 * the engine pre-folds these into its turn-2+ template pointer at flag-set
	 * time; absent keys fall back to the turn-1 template (pre-@multi behavior). */
	if (tok->multi_prompt_template[0] && strcmp(tok->multi_prompt_template, tok->prompt_template) != 0)
	{
		HT_ADD_CHECK("__sys_template@multi", tok->multi_prompt_template, (unsigned int)strlen(tok->multi_prompt_template) + 1, 1);
		printf("  [11] multi template stored (%u bytes)\n", (unsigned)strlen(tok->multi_prompt_template));
	}
	if (tok->multi_think_prompt_template[0] && strcmp(tok->multi_think_prompt_template, tok->multi_prompt_template) != 0)
	{
		HT_ADD_CHECK("__sys_template@multi@think", tok->multi_think_prompt_template, (unsigned int)strlen(tok->multi_think_prompt_template) + 1, 1);
		printf("  [11] multi thinking template stored (%u bytes)\n", (unsigned)strlen(tok->multi_think_prompt_template));
	}
	printf("  [11] templates stored\n");

	/* NO RoPE tables -- built internally by DLL at inference init time (sec.87) */

	/* 12. MemLayout */
	{
		mem_layout ml;
		compute_mem_layout(&t->config, &ml);
		HT_ADD_CHECK("__sys_mem_layout", &ml, sizeof(ml), 1);
		printf("  [12] MemLayout: rs_block=%lld bytes\n", (long long)ml.rs_block_bytes);
	}

	/* 13. INI dispatch params (threading_proxy.ini -> __ini__* keys) */
	ini_to_db(t->config.dim, t->config.hidden_dim);
}

/* ---------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
	ZydisInit();
	ZydisSetCrashLogFile("crash_build.log");

	if (argc < 2)
	{
		printf("Usage: %s <checkpoint.bin> [-thinking 0|1] [-rope-base <float>] [-rope-type <0|1>] [-rope-factor <float>] [-rope-high-freq-factor <float>] [-rope-low-freq-factor <float>] [-rope-orig-max-pos <int>] [-vision <path>] [-visionproj <path>] [-autoexec <path>] [-tuner-delay <n>] [-tuner-delay-large <n>] [-tuner-sex <p>] [-tuner-mutate <p>] [-tuner-mutations-max <n>] [-tuner-rand-delay <n>] [-simple-threads <n>] [-o <output.db>] [-f]\n", argv[0]);
		return 1;
	}

	char *checkpoint = argv[1];
	int enable_thinking = 1;
	float rope_base_override = 0.0f;
	float eps_override = 0.0f;
	int force = 0;
	char *output_db = "lz4ht.db";
	char *vision_path = 0;
	char *visionproj_path = 0;
	char *autoexec_path = 0;
	rope_scaling_header rs_override;
	memset(&rs_override, 0, sizeof(rs_override));
	int rs_type_set = 0;
	int rs_factor_set = 0;
	int rs_high_set = 0;
	int rs_low_set = 0;
	int rs_orig_set = 0;

	/* Database parity defaults: when run from the package root without
	 * an explicit -autoexec, pick up the shipped SectorLogo stdlib so
	 * EVERY built database carries it -- the tested configuration
	 * (verify.bat passes the flag explicitly) and the fetched
	 * configuration must not differ. Explicit flags always win;
	 * a non-package-root cwd without autoexec.logo builds a bare DB. */
	if (!autoexec_path && GetFileAttributesA("autoexec.logo") != INVALID_FILE_ATTRIBUTES)
	{
		autoexec_path = "autoexec.logo";
		printf("[parity] no -autoexec given -- using autoexec.logo from the working directory\n");
	}

	for (int i = 2; i < argc; i++)
	{
		if (strcmp(argv[i], "-f") == 0)
		{
			force = 1;
		}
		else if (strcmp(argv[i], "-thinking") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			enable_thinking = atoi(argv[++i]);
			if (!enable_thinking)
				printf("Note: -thinking is a no-op (templates: base = non-thinking + @think keys; runtime selection via v7s_enable_thinking)\n");
		}
		else if (strcmp(argv[i], "-rope-base") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			rope_base_override = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-eps") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			eps_override = (float)atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-rope-type") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			rs_override.rope_type = atoi(argv[++i]);
			rs_type_set = 1;
		}
		else if (strcmp(argv[i], "-rope-factor") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			rs_override.factor = (float)atof(argv[++i]);
			rs_factor_set = 1;
		}
		else if (strcmp(argv[i], "-rope-high-freq-factor") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			rs_override.high_freq_factor = (float)atof(argv[++i]);
			rs_high_set = 1;
		}
		else if (strcmp(argv[i], "-rope-low-freq-factor") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			rs_override.low_freq_factor = (float)atof(argv[++i]);
			rs_low_set = 1;
		}
		else if (strcmp(argv[i], "-rope-orig-max-pos") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			rs_override.orig_max_pos = atoi(argv[++i]);
			rs_orig_set = 1;
		}
		else if (strcmp(argv[i], "-o") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			output_db = argv[++i];
		}
		else if (strcmp(argv[i], "-vision") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			vision_path = argv[++i];
		}
		else if (strcmp(argv[i], "-visionproj") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			visionproj_path = argv[++i];
		}
		else if (strcmp(argv[i], "-autoexec") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			autoexec_path = argv[++i];
		}
		else if (strcmp(argv[i], "-tuner-delay") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			char *td = argv[++i];
			char *tdend = 0;
			long tdval = strtol(td, &tdend, 10);
			if ((td == tdend) || (tdval < 0))
			{
				printf("-tuner-delay must be a non-negative integer (got %s)\n", td);
				return 1;
			}
			opt_tuner_delay = (int)tdval;
		}
		else if (strcmp(argv[i], "-tuner-delay-large") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			char *tdl = argv[++i];
			char *tdlend = 0;
			long tdlval = strtol(tdl, &tdlend, 10);
			if ((tdl == tdlend) || (tdlval < 0))
			{
				printf("-tuner-delay-large must be a non-negative integer (got %s)\n", tdl);
				return 1;
			}
			opt_tuner_delay_large = (int)tdlval;
		}
		else if (strcmp(argv[i], "-tuner-sex") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			char *ts = argv[++i];
			char *tsend = 0;
			long tsval = strtol(ts, &tsend, 10);
			if ((ts == tsend) || (tsval < 0) || (tsval > 100))
			{
				printf("-tuner-sex must be an integer in [0,100] (got %s)\n", ts);
				return 1;
			}
			opt_tuner_sex = (int)tsval;
		}
		else if (strcmp(argv[i], "-tuner-mutate") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			char *tm = argv[++i];
			char *tmend = 0;
			long tmval = strtol(tm, &tmend, 10);
			if ((tm == tmend) || (tmval < 0) || (tmval > 100))
			{
				printf("-tuner-mutate must be an integer in [0,100] (got %s)\n", tm);
				return 1;
			}
			opt_tuner_mutate = (int)tmval;
		}
		else if (strcmp(argv[i], "-tuner-mutations-max") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			char *tmm = argv[++i];
			char *tmmend = 0;
			long tmmval = strtol(tmm, &tmmend, 10);
			if ((tmm == tmmend) || (tmmval < 1) || (tmmval > 104))
			{
				printf("-tuner-mutations-max must be an integer in [1,104] (got %s)\n", tmm);
				return 1;
			}
			opt_tuner_mutations_max = (int)tmmval;
		}
		else if (strcmp(argv[i], "-tuner-rand-delay") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			char *tr = argv[++i];
			char *trend = 0;
			long trval = strtol(tr, &trend, 10);
			if ((tr == trend) || (trval < 0) || (trval > 100000))
			{
				printf("-tuner-rand-delay must be an integer in [0,100000] (got %s)\n", tr);
				return 1;
			}
			opt_tuner_rand_delay = (int)trval;
		}
		else if (strcmp(argv[i], "-simple-threads") == 0)
		{
			if (i + 1 >= argc)
			{
				printf("Missing value for %s\n", argv[i]);
				return 1;
			}
			char *stv = argv[++i];
			char *stvend = 0;
			long stval = strtol(stv, &stvend, 10);
			if ((stv == stvend) || (stval < 0))
			{
				printf("-simple-threads must be a non-negative integer (got %s)\n", stv);
				return 1;
			}
			opt_simple_threads = (int)stval;
		}
		else
		{
			printf("Unknown option: %s\n", argv[i]);
			return 1;
		}
	}

	/* force rebuild: delete existing DB */
	if (force)
	{
		if (GetFileAttributesA(output_db) != INVALID_FILE_ATTRIBUTES)
		{
			if (!DeleteFileA(output_db))
			{
				printf("[FATAL] -f could not delete the existing %s (locked by another process?) -- a merge into stale content would ship\n", output_db);
				exit(EXIT_FAILURE);
			}
			printf("Deleted existing %s\n", output_db);
		}
	}

	/* load DLL (triggers DllMain(ATTACH) -> ht_init + db_load) */
	printf("Loading %s...\n", V7_LIBRARY_NAME);
	if (v7_load(V7_LIBRARY_NAME, &dll) != 0)
	{
		printf("[FATAL] failed to load v7_threading.dll\n");
		return 1;
	}
	printf("Dll loaded. ht_add=%p db_save=%p v7_init=%p\n", (void *)dll.ht_add, (void *)dll.db_save, (void *)dll.v7_init);

	// start with empty database; a failed init means no hash table to build into
	if (!dll.v7_init(0))
	{
		printf("[FATAL] v7_init(0) failed -- no database context\n");
		v7_unload(&dll);
		return 1;
	}

	/* read checkpoint .bin file */
	printf("Reading checkpoint: %s\n", checkpoint);
	FILE *f = fopen(checkpoint, "rb");
	if (!f)
	{
		printf("[FATAL] cannot open %s\n", checkpoint);
		v7_unload(&dll);
		return 1;
	}
	_fseeki64(f, 0, SEEK_END);
	__int64 file_size = _ftelli64(f);
	fseek(f, 0, SEEK_SET);

	if (file_size < 256)
	{
		printf("[FATAL] file too small: %lld bytes\n", (long long)file_size);
		fclose(f);
		v7_unload(&dll);
		return 1;
	}

	/* mmap the file */
	void *data = map_checkpoint(f, (size_t)file_size);
	if (!data)
	{
		printf("[FATAL] Cannot map checkpoint %s\n", checkpoint);
		fclose(f);
		v7_unload(&dll);
		return 1;
	}
	fclose(f);

	/* parse config */
	transformer t;
	memset(&t, 0, sizeof(t));
	memcpy(&t.config, data, sizeof(Config));
	t.data = (float *)data;
	t.file_size = file_size;

	/* parse RoPE scaling block from header padding (bytes 64-83).
	 * written by export.py; zero-filled when the model has no rope_scaling. */
	rope_scaling_header rs;
	memset(&rs, 0, sizeof(rs));
	memcpy(&rs, (char *)data + 64, sizeof(rope_scaling_header));
	if (rs_type_set || rs_factor_set || rs_high_set || rs_low_set || rs_orig_set)
	{
		if (rs_type_set)
			rs.rope_type = rs_override.rope_type;
		if (rs_factor_set)
			rs.factor = rs_override.factor;
		if (rs_high_set)
			rs.high_freq_factor = rs_override.high_freq_factor;
		if (rs_low_set)
			rs.low_freq_factor = rs_override.low_freq_factor;
		if (rs_orig_set)
			rs.orig_max_pos = rs_override.orig_max_pos;
		printf("RoPE scaling: CLI overrides applied\n");
	}
	if (rs.rope_type != 0)
		printf("RoPE scaling header: type=%d factor=%.3f high_freq=%.3f low_freq=%.3f orig_max_pos=%d\n", rs.rope_type, rs.factor, rs.high_freq_factor, rs.low_freq_factor, rs.orig_max_pos);

	GS = t.config.group_size;
	printf("Config: dim=%d hidden=%d layers=%d heads=%d kv=%d head_dim=%d vocab=%d seq=%d shared=%d GS=%d qk_norm=%d rope_base=%d\n", t.config.dim, t.config.hidden_dim, t.config.n_layers, t.config.n_heads, t.config.n_kv_heads, t.config.head_dim,
	       t.config.vocab_size, t.config.seq_len, t.config.shared_classifier, t.config.group_size, t.config.has_qk_norm, t.config.rope_base);

	/* The import is the engine's outer boundary (Sec.199): every engine-side
	 * check of these DB-borne config values was removed, so the BUILDER is
	 * the one place that refuses a model outside the kernels' contracts.
	 * Capacities mirrored from the engine: BITMAP_MAX_BITS (chat.inc),
	 * ROPE_INTERNAL_MAX_SEQ_LEN (fused_f32_kernel.inc), QF_SCRATCH_MAX_N
	 * (v7_core.inc). */
	if (t.config.dim <= 0 || (t.config.dim % 64) != 0)
	{
		printf("[FATAL] config dim=%d: must be a positive multiple of 64 (the Q8/F32 matmul kernels have no scalar tail)\n", t.config.dim);
		v7_unload(&dll);
		return 1;
	}
	if (t.config.hidden_dim <= 0 || (t.config.hidden_dim % 64) != 0)
	{
		printf("[FATAL] config hidden_dim=%d: must be a positive multiple of 64 (the SwiGLU scale groups are 64 wide)\n", t.config.hidden_dim);
		v7_unload(&dll);
		return 1;
	}
	if (t.config.head_dim < 64 || (t.config.head_dim % 32) != 0 || t.config.head_dim > 512)
	{
		printf("[FATAL] config head_dim=%d: must be a multiple of 32 in [64..512] (the attention kernel preloads q[0:47]; the rope freq buffer is sized for head_dim <= 512)\n", t.config.head_dim);
		v7_unload(&dll);
		return 1;
	}
	if (t.config.n_layers <= 0)
	{
		printf("[FATAL] config n_layers=%d: must be positive (the forward/prefill layer loops are bottom-tested)\n", t.config.n_layers);
		v7_unload(&dll);
		return 1;
	}
	if (t.config.n_heads <= 0 || t.config.n_kv_heads <= 0 || (t.config.n_heads % t.config.n_kv_heads) != 0)
	{
		printf("[FATAL] config n_heads=%d n_kv_heads=%d: both must be positive and n_heads must divide evenly by n_kv_heads (the GQA contract)\n", t.config.n_heads, t.config.n_kv_heads);
		v7_unload(&dll);
		return 1;
	}
	if (t.config.seq_len <= 0 || t.config.seq_len > 131072)
	{
		printf("[FATAL] config seq_len=%d: must be in [1..131072] (the rope table capacity -- ROPE_INTERNAL_MAX_SEQ_LEN)\n", t.config.seq_len);
		v7_unload(&dll);
		return 1;
	}
	if (t.config.vocab_size <= 0 || t.config.vocab_size > 163840)
	{
		printf("[FATAL] config vocab_size=%d: must be in [1..163840] (the presence-penalty bitmap capacity)\n", t.config.vocab_size);
		v7_unload(&dll);
		return 1;
	}
	if (t.config.dim > 65536)
	{
		printf("[FATAL] config dim=%d: must be <= 65536 (the v7d_q8_fused quantize-once scratch -- QF_SCRATCH_MAX_N)\n", t.config.dim);
		v7_unload(&dll);
		return 1;
	}
	if (t.config.hidden_dim > 65536)
	{
		printf("[FATAL] config hidden_dim=%d: must be <= 65536 (v7d_q8_fused quantizes the W2 input at n=hidden_dim into the same QF_SCRATCH_MAX_N scratch)\n", t.config.hidden_dim);
		v7_unload(&dll);
		return 1;
	}
	if (t.config.group_size != 64)
	{
		printf("[FATAL] config group_size=%d: must be 64 (the kernel scale divisors are hard-coded /64)\n", t.config.group_size);
		v7_unload(&dll);
		return 1;
	}
 	{
 		__int64 all_heads_dim = (__int64)t.config.n_heads * t.config.head_dim;
 		__int64 kv_dim64 = (__int64)t.config.n_kv_heads * t.config.head_dim;
 		__int64 dsum = all_heads_dim + 2 * kv_dim64;
 		if ((all_heads_dim % 64) != 0 || all_heads_dim > 2147483647 || kv_dim64 > 2147483647 || dsum > 2147483647)
 		{
 			printf("[FATAL] config head geometry: n_heads*head_dim=%lld kv_dim=%lld dsum=%lld -- all_heads_dim must be a multiple of 64 and every kernel dimension must fit the signed-int ABI\n", (long long)all_heads_dim, (long long)kv_dim64, (long long)dsum);
 			v7_unload(&dll);
 			return 1;
 		}
 		if (all_heads_dim > 65536)
 		{
 			printf("[FATAL] config head geometry: n_heads*head_dim=%lld must be <= 65536 (v7d_q8_fused quantizes the WO input at n=all_heads_dim into the same QF_SCRATCH_MAX_N scratch)\n", (long long)all_heads_dim);
 			v7_unload(&dll);
 			return 1;
 		}
 	}

	/* memory-map weights (for weight_meta computation) */
	void *weights_ptr = (char *)data + 256;
	memory_map_weights(&t.weights, &t.config, weights_ptr);

	/* build Tokenizer */
	printf("Building Tokenizer...\n");
	tokenizer tok;
	memset(&tok, 0, sizeof(tok));
	build_tokenizer(&tok, checkpoint, t.config.vocab_size);
	printf("Tokenizer: %d tokens, max_len=%u, bos=%u, eos=%u\n", tok.vocab_size, tok.max_token_length, tok.bos_token_id, tok.eos_token_id);

	/* populate database */
	printf("\nPopulating database...\n");
	populate_db(&t, &tok, rope_base_override, eps_override, &rs);

	/* store vision tower (qwen3-VL or LLaVA-one_vision) when a .vision file is supplied */
	if (vision_path)
	{
		printf("\nStoring vision tower: %s\n", vision_path);
		store_vision_db(vision_path);
	}

	/* store separate MLP projector blob (LLaVA-one_vision / siglip only).
	 * the blob is prefixed with an 8-byte reserved header (always 0 -- Sec.96)
	 * and laid out as [fc1_w][fc1_b][fc2_w][fc2_b]. the ASM v7s_vit_encode_callback
	 * loads it from the __sys_vit_projector microdb key when VCFG_PROJ_IN_DIM > 0. */
	if (visionproj_path)
	{
		printf("\nStoring vision projector: %s\n", visionproj_path);
		FILE *pf = fopen(visionproj_path, "rb");
		if (!pf)
		{
			printf("[FATAL] cannot open %s\n", visionproj_path);
			exit(EXIT_FAILURE);
		}
		_fseeki64(pf, 0, SEEK_END);
		__int64 psize = _ftelli64(pf);
		fseek(pf, 0, SEEK_SET);
		if (psize < 4)
		{
			printf("[FATAL] projector file too small: %lld bytes\n", (long long)psize);
			fclose(pf);
			exit(EXIT_FAILURE);
		}
		void *pbuf = db_alloc((__int64)psize);
		if (fread(pbuf, 1, (__int64)psize, pf) != (__int64)psize)
		{
			printf("[FATAL] short read on %s\n", visionproj_path);
			fclose(pf);
			exit(EXIT_FAILURE);
		}
		fclose(pf);
		/* 8-byte header: two reserved dwords (always 0, never checked -- Sec.96);
		 * proj type is carried by __sys_vision_config VCFG_PROJ_TYPE, the only source ASM 
reads */
		HT_ADD_CHECK("__sys_vit_projector", pbuf, psize, 1);
	}

	/* store the SectorLogo stdlib (autoexec.logo) as the persistent
	 * __sectorlogo__autoexec key -- tier-1 stdlib source at runtime
	 * (lg_init DB-first load, plan Sec.3.6/3.7). */
	if (autoexec_path)
	{
		printf("\nStoring stdlib autoexec: %s\n", autoexec_path);
		FILE *af = fopen(autoexec_path, "rb");
		if (!af)
		{
			printf("[FATAL] cannot open %s\n", autoexec_path);
			exit(EXIT_FAILURE);
		}
		_fseeki64(af, 0, SEEK_END);
		__int64 asize = _ftelli64(af);
		fseek(af, 0, SEEK_SET);
		if (asize <= 0)
		{
			printf("[FATAL] stdlib file empty: %s\n", autoexec_path);
			fclose(af);
			exit(EXIT_FAILURE);
		}
		if (asize > 65536)
		{
			printf("[FATAL] stdlib exceeds 64KB script buffer: %lld bytes\n", (long long)asize);
			fclose(af);
			exit(EXIT_FAILURE);
		}
		void *abuf = db_alloc(asize);
		if (fread(abuf, 1, asize, af) != asize)
		{
			printf("[FATAL] short read on %s\n", autoexec_path);
			fclose(af);
			exit(EXIT_FAILURE);
		}
		fclose(af);
		HT_ADD_CHECK("__sectorlogo__autoexec", abuf, (unsigned int)asize, 1);
	}

	/* save to disk */
	printf("\nSaving %s...\n", output_db);
	int save_rc = dll.db_save(output_db);
	printf("db_save returned: %d\n", save_rc);
	if (save_rc == 0)
	{
		printf("[ERROR] db_save FAILED! GetLastError=%lu\n", GetLastError());
		unmap_checkpoint(data, (size_t)file_size);
		v7_unload(&dll);
		return 1; /* the database was not written -- never report success */
	}
	printf("Done: %s written\n", output_db);

	/* cleanup */
	unmap_checkpoint(data, (size_t)file_size);

	for (int i = 0; i < tok.vocab_size; i++)
		free(tok.vocab[i]);
	free(tok.vocab);
	free(tok.merge_scores);

	free(t.weights.q_tokens);
	free(t.weights.wq);
	free(t.weights.wk);
	free(t.weights.wv);
	free(t.weights.wo);
	free(t.weights.w1);
	free(t.weights.w2);
	free(t.weights.w3);
	if (!t.config.shared_classifier)
		free(t.weights.wcls);

	v7_unload(&dll);
	printf("build_db complete.\n");
	return 0;
}

