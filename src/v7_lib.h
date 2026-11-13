// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_lib.h -- header-only C library for v7_threading.dll kernel access
 *
 * Usage:
 *   #include "v7_lib.h"
 *
 *   v7_lib lib;
 *   if (v7_load("v7_threading.dll", &lib) != 0) return 1;
 *   V7_LIB_CALL(&lib, v7d_q8, xout, &x, &w, n, d);
 *   lib.v7s_shutdown();
 *   v7_lib_print_stats(&lib, stdout);
 */
#pragma once
/* ---------------------------------------------------------------------------
 * V7_ONLY_SAFE -- compile a host against the PUBLIC API TIER ONLY.
 *
 * Define V7_ONLY_SAFE before including this header to make every internal
 * export disappear from the declarations: kernels, MicroDB internals, the
 * raw tokenizer/forward entries, SectorLogo internals, process execution,
 * diagnostic gates, and v7_get_internal_inference_ctx. In this mode the
 * v7_lib struct carries only the public v7_* members (chat/generate/vision,
 * sampler and session setters/getters, KV-cache persistence, db add/find,
 * engine-death and version queries), v7_load resolves only the public tier,
 * and the internal typedefs do not exist. The DLL is unchanged -- the
 * exports remain for dev tools; the header simply stops handing them out.
 * INTERNAL functions documented as internal are internal for a reason:
 * writing through v7_get_internal_inference_ctx's returned context (when
 * obtained via raw GetProcAddress outside this header) is UNDEFINED
 * behaviour. There is no #ifdef anywhere else in a host's source: one
 * define switches the whole surface.
 * ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif
#include <stddef.h>
#ifdef __linux__
#define V7_ABI __attribute__((ms_abi))
#define V7_LIBRARY_NAME "libv7_threading.so"
#else
#define V7_ABI
#define V7_LIBRARY_NAME "v7_threading.dll"
#endif
	/* ---------------------------------------------------------------------------
	 * Data structures (match v7_threading.dll layout)
	 * ------------------------------------------------------------------------- */
#ifndef V7_ONLY_SAFE
	/* Q8 tensor: quantized int8 values + per-group float scales (GS=64) */
	typedef struct v7_q8_tensor
	{
		char *q;
		float *s;
	} v7_q8_tensor;
	/* F32 tensor: float data + optional scales (NULL if not quantized) */
	typedef struct v7_f32_tensor
	{
		float *data;
		float *scales;
	} v7_f32_tensor;
	/* Transformer hyperparameters (Config struct, matches runq.c) -- 16 ints x 4 bytes = 64 bytes */
	typedef struct v7_config
	{
		int reserved0;
		int reserved1;
		int dim;
		int hidden_dim;
		int n_layers;
		int n_heads;
		int n_kv_heads;
		int vocab_size;
		int seq_len;
		int head_dim;
		int shared_classifier;
		int group_size;
		int has_qk_norm;
		int rope_base;
		int reserved2;
		int reserved3;
	} v7_config;
	/* Per-layer run state (matches RunState in runq.c) */
	typedef struct v7_run_state
	{
		float *x;
		float *xb;
		float *hb;
		float *hb2;
		v7_q8_tensor xq;
		v7_q8_tensor hq;
		float *q;
		float *k;
		float *v;
		float *att;
		float *logits;
		float *key_cache;
		float *value_cache;
		float *x_storage;
		float *q_storage;
		float *att_out_storage;
		void *key_cache_hfile;
		void *key_cache_hmap;
		void *value_cache_hfile;
		void *value_cache_hmap;
	} v7_run_state;
#endif /* !V7_ONLY_SAFE */
	/* ---------------------------------------------------------------------------
	 * Function pointer types
	 * ------------------------------------------------------------------------- */
#ifndef V7_ONLY_SAFE
	/* --- Scheduled matmul kernels (multi-worker dispatch) --- */
	typedef void (V7_ABI *v7d_q8_fn)(float *xout, v7_q8_tensor *x, v7_q8_tensor *w, int n, int d);
	typedef void (V7_ABI *v7d_f32_fn)(float *xout, v7_f32_tensor *x, v7_q8_tensor *w, int n, int d);
	typedef void (V7_ABI *v7d_fused_f32_fn)(float *xout, v7_f32_tensor *x, v7_q8_tensor *w, int n, int d, float *norm_weights, int is_q);
	typedef void (V7_ABI *v7d_swiglu_fn)(float *xout, v7_f32_tensor *x, v7_q8_tensor *gate_w, int n, int d, v7_q8_tensor *up_w);
	typedef void (V7_ABI *v7d_swiglu_q8_fn)(float *xout, v7_q8_tensor *x, v7_q8_tensor *gate_w, int n, int d, v7_q8_tensor *up_w);
	/* --- Scheduled attention kernel --- */
	typedef void (V7_ABI *v7d_attention_fn)(v7_run_state *s, v7_config *p, int pos, unsigned long long loff, int kv_dim, int kv_mul);
	/* --- Prefill attention kernel (eliminates double memcpy per token) --- */
	typedef void (V7_ABI *v7d_attention_prefill_fn)(v7_run_state *s, v7_config *p, float *q_storage, float *att_out_storage, int start_pos, int n_tokens, unsigned long long loff, int kv_dim, int kv_mul);
	/* --- Scheduled utility kernels --- */
	typedef void (V7_ABI *v7d_quantize_fn)(char *out_q, float *in_x, float *out_s, int n);
	typedef void (V7_ABI *v7s_rmsnorm_quantize_fn)(char *out_q, float *in_x, float *out_s, int n, float *weight);
	typedef void (V7_ABI *v7d_memcpy_fn)(void *dst, void *src, int byte_count);
	typedef void (V7_ABI *v7d_vit_linear_fn)(float *out, float *in, float *w, float *bias, int n_rows, int M, int K);
	/* Bicubic resize (PIL BICUBIC resample=3, Keys a=-0.5): packed RGB, 3 bytes/pixel */
	typedef void (V7_ABI *v7s_bicubic_resize_fn)(const unsigned char *src, int src_w, int src_h, unsigned char *dst, int dst_w, int dst_h);
	/* --- MicroDB persistent hash table API --- */
	typedef void (V7_ABI *ht_init_fn)(void);
	typedef void (V7_ABI *ht_destroy_fn)(void);
	typedef int (V7_ABI *ht_add_fn)(const char *key, void *data, unsigned long long data_len, unsigned int persist);
	typedef void *(V7_ABI *ht_find_fn)(const char *key, unsigned long long *out_len);
	typedef int (V7_ABI *db_save_fn)(const char *filename);
	typedef int (V7_ABI *db_load_fn)(const char *filename);
	typedef unsigned long long (V7_ABI *mdb_compress_fn)(const void *src, unsigned long long src_len, void *dst);
	typedef unsigned long long (V7_ABI *mdb_decompress_fn)(void *src, unsigned long long src_len, void *dst, unsigned long long max_dst);
	typedef void (V7_ABI *db_set_readonly_fn)(int flag);
		/* Opaque context returned by the internal v7s_inference_create.
	   Contains Transformer*, Tokenizer*, Sampler* -- all populated from microdb.
	   pos tracks current KV-cache position for multi-turn support.
	   INTERNAL (not part of the public API): this struct and the pointer
	   behind it belong to the engine. It is declared here for reference and
	   for the dev-side deep tools only; v7_get_internal_inference_ctx hands
	   it out as a diagnostic escape hatch. READING it is supported for
	   diagnostics; WRITING through it bypasses every public-tier contract
	   and is UNDEFINED behaviour. Hosts that want the safe surface only
	   compile with V7_ONLY_SAFE, which removes this struct entirely. */
	typedef struct inference_context
	{
		void *transformer;
		void *tokenizer;
		void *sampler;
		void *prompt_tokens;
		void *out_tokens;
		int max_prompt;
		int max_gen;
		int pos;
		int quantize;
	} inference_context;
#endif /* !V7_ONLY_SAFE */
	/* ---------------------------------------------------------------------------
	 * PUBLIC v7_* API TIER (public_api.inc -- the "normal program" surface).
	 * The inference context is hidden in BSS inside the DLL: create returns a
	 * bool, destroy is parameterless, chat/generate/vision take no context.
	 * --------------------------------------------------------------------------- */
	typedef int (V7_ABI *v7_init_fn)(const char *db_path);
	typedef int (V7_ABI *v7_db_load_fn)(const char *path);
	typedef int (V7_ABI *v7_db_save_fn)(const char *path);
	typedef int (V7_ABI *v7_initialize_inference_fn)(void);
	typedef void (V7_ABI *v7_free_inference_fn)(void);
	typedef void (V7_ABI *v7_chat_fn)(const char *user_prompt, const char *system_prompt, char *output, int output_len, int reset);
	/* raw_generation: 0 = normal single-turn (chat template + full stop set);
	 * 1 = raw text continuation (no template, EOS-only stop) */
	typedef void (V7_ABI *v7_generate_fn)(const char *prompt, char *output, int output_len, int raw_generation);
	typedef int (V7_ABI *v7_chat_image_fn)(const char *user_prompt, const char *system_prompt, const unsigned char *raw_rgb, int img_w, int img_h, char *output, int output_len, int reset);
	typedef void (V7_ABI *v7_generate_image_fn)(const char *user_prompt, const unsigned char *raw_rgb, int img_w, int img_h, char *output, int output_len);
	typedef void (V7_ABI *v7_set_sampler_temperature_fn)(float t);
	/* engine death queries (pseudo-exception contract, docs/plans/fatal_to_pseudo_exception_plan.md):
	 * a gated public API that returns because of a caught FATAL always returns 0 (or void);
	 * pair return-value checks with v7_engine_is_dead() to disambiguate. The reason string
	 * is stable until the next v7_free_inference or DLL unload. */
	typedef int (V7_ABI *v7_engine_is_dead_fn)(void);
	typedef const char* (V7_ABI *v7_get_death_reason_fn)(void);
typedef const char* (V7_ABI *v7_get_version_fn)(void);
#ifndef V7_ONLY_SAFE
typedef char* (V7_ABI *v7_execute_fmt_fn)(char *exe, char *fmt, unsigned long long *argv, int argc);
#endif /* !V7_ONLY_SAFE */
typedef int (V7_ABI *v7_get_console_width_fn)(void);
#ifndef V7_ONLY_SAFE
typedef const char* (V7_ABI *v7s_get_user_prompt_fn)(void);
#endif /* !V7_ONLY_SAFE */
	/* KV-cache persistence: save gathers ONLY the used rows [0, pos) per
	 * layer -- min(pos, C) rows under StreamingLLM wrapping (file scales
	 * with pos, not seq_len) -- LZ4-compressed; load restores the rows +
	 * the position + the M-RoPE counters. The 48-byte header records the
	 * sizes, pos, mrope state, the KV16 format marker, and the save-time
	 * streaming geometry (enable, S, W, valid-window rows); a wrapped
	 * cache reloads its shrunk validity tracker verbatim and is rejected
	 * if the receiver's S/W differ. Both functions return 1 = ok, 0 =
	 * nothing to save, incompatible file/geometry, or an I/O error
	 * (message printed by the DLL). Old 32-byte-header cache files are
	 * rejected by the header-size check; regenerate them.
	 * Callable from SectorLogo via C-CALL with a PTR argument. */
	typedef int (V7_ABI *save_kv_cache_fn)(const char *filename);
	typedef int (V7_ABI *load_kv_cache_fn)(const char *filename);
	/* StreamingLLM ring cache: enable=1 starts streaming with num_sinks sink
	 * rows and window_size rolling rows; the fill grade (eviction start) is
	 * num_sinks + window_size, capped by the model's seq_len. Returns 0 on
	 * success, -1 on bad args/overflow/capacity; FATALs pre-init or on a
	 * mid-stream re-map (ctx->pos > 0). Disable (enable=0) is legal while
	 * pos <= C and stores only the flag; a WRAPPED session (pos > C)
	 * refuses with -1 (legacy would read slot-ordered rows as
	 * position-ordered). */
	typedef int (V7_ABI *v7_set_streaming_fn)(int enable, int num_sinks, int window_size);
	typedef void (V7_ABI *v7_set_sampler_topp_fn)(float p);
	typedef void (V7_ABI *v7_set_sampler_presence_penalty_fn)(float p);
	typedef void (V7_ABI *v7_set_sampler_topk_fn)(int k);
	typedef void (V7_ABI *v7_set_sampler_seed_fn)(unsigned long long seed);
	typedef void (V7_ABI *v7_set_quantize_fn)(int mode);
	typedef void (V7_ABI *v7_set_max_gen_fn)(int cap);
	typedef int (V7_ABI *v7_get_position_fn)(void);
	typedef void (V7_ABI *v7_set_position_fn)(int pos);
	/* Thinking-mode state: 1 when v7_enable_thinking resolved the @think
	 * templates (the model renders think blocks), 0 otherwise. */
	typedef int (V7_ABI *v7_get_think_mode_active_fn)(void);
	/* Live per-turn generated-token count (reset at each chat turn start,
	 * updated once per generated token). Relative position for think-budget
	 * limiters; absolute position is v7_get_position. */
	typedef int (V7_ABI *v7_get_gen_count_fn)(void);
	typedef void (V7_ABI *v7_get_token_probs_fn)(const int *token_ids, int n, float *out_probs);
	/* Set the Win32 priority of every live worker thread. Win32 constants:
	 * -15 idle, -2 lowest, -1 below normal, 0 normal, 1 above normal,
	 * 2 highest, 15 time critical. Callable any time after v7_init;
	 * last call wins. */
	typedef void (V7_ABI *v7_set_worker_priority_fn)(int priority);
	typedef void (V7_ABI *v7_enable_thinking_fn)(int on);
	typedef int (V7_ABI *v7_db_add_fn)(const char *key, const void *data, int len, unsigned int persist);
	typedef void *(V7_ABI *v7_db_find_fn)(const char *key, unsigned long long *out_len);
	typedef void (V7_ABI *v7_encode_fn)(const char *text, int *tokens, int *n_tokens, int max_tokens);
	typedef unsigned long long *(V7_ABI *v7_get_generation_statistic_fn)(void);
	/* Cycle-tuner pool fitness: per-shape u64 averages indexed by registry
	 * slot, valid range [0, v7_get_active_shape_count()). best = the mean
	 * c-measurement of the front 10 genome slots, pool = the mean over every
	 * non-tail genome; both fold at each population sweep wrap; 0 = that
	 * shape has not completed a sweep yet. */
	typedef unsigned long long *(V7_ABI *v7_get_best_fitness_fn)(void);
	typedef unsigned long long *(V7_ABI *v7_get_pool_fitness_fn)(void);
	typedef int (V7_ABI *v7_get_active_shape_count_fn)(void);
	typedef void (V7_ABI *v7_shutdown_fn)(void);
#ifndef V7_ONLY_SAFE
	/* Deep-tool escape hatch: the real internal context (0 when not initialized).
	 * Internal diagnostic access only -- normal programs never touch this. */
	typedef inference_context *(V7_ABI *v7_get_internal_inference_ctx_fn)(void);
	/* --- M-RoPE (multimodal rotary position embeddings) --- */
	typedef void (V7_ABI *v7s_mrope_prepare_fn)(const int *tokens, int n_tokens, int start_pos);
	typedef void (V7_ABI *v7s_mrope_restore_fn)(void);
	/* --- Direct-call kernels (no dispatch) --- */
	/* ProbIndex for top-p nucleus sampling workspace: MUST have capacity for
	 * the full vocab_size whenever 0 < topp < 1 -- a nucleus wider than 256
	 * tokens falls back to an exact full-vocab selection written into this
	 * workspace. TOPK_LIMIT (256) entries suffice only for greedy and
	 * topp<=0 calls, which never touch it. */
	typedef struct v7_probindex
	{
		float prob;
		int index;
	} v7_probindex;
	typedef int (V7_ABI *v7s_sample_fn)(float *logits, int vocab_size, float temperature, float topp, unsigned long long *rng_state, v7_probindex *probindex);
	typedef void (V7_ABI *v7s_rmsnorm_fn)(float *o, float *x, float *weight, int size);
	typedef void (V7_ABI *v7s_add_inplace_fn)(float *x, float *y, int n);
	typedef void (V7_ABI *v7s_fused_add_rmsnorm_fn)(float *o, float *residual, float *add, float *weight, int dim);
	typedef void (V7_ABI *v7s_rmsnorm_rope_heads_fn)(float *buf, int n_heads, int head_dim, float *norm_weight, float *cos_table, float *sin_table, int pos, int rope_max_dim);
	/* --- Fused-kernel support functions --- */
	typedef void (V7_ABI *v7s_rope_init_fn)(int head_dim, float rope_base);
	typedef void (V7_ABI *v7s_set_fused_rope_fn)(int pos, int head_dim);
	/* --- Runtime control --- */
	typedef void (V7_ABI *v7s_set_spin_fn)(int flag);
	typedef void (V7_ABI *v7s_enable_thinking_fn)(int on);
	typedef void (V7_ABI *v7s_shutdown_fn)(void);
	typedef int (V7_ABI *v7s_get_num_workers_fn)(void);
	/* Sec.167: u64 bit flags of engine-degenerated config values (bit0=taper CAP->uniform, bit1=num_chunks->max) */
	typedef unsigned long long (V7_ABI *get_config_was_fixed_fn)(void);
	/* ---- In-system A/B tuning API ---- */
	/* SMC prefetch patch entries (live patch targets for the autotuner; the
	 * dispatcher's level-2 prefetch cache only re-patches on a shape switch,
	 * so a same-shape candidate must patch the slots directly). Export names
	 * carry no v7s_ prefix -- they predate the naming convention. */
	typedef void (V7_ABI *set_prefetch_distance_fn)(int slot, int distance);
	typedef void (V7_ABI *set_prefetch_type_fn)(int slot, int type);
	typedef int (V7_ABI *v7s_sectorlogo_eval_script_fn)(const char *key_name);
#endif /* !V7_ONLY_SAFE */
	/* --- Inference init (internal; public tier: v7_initialize_inference / v7_free_inference) --- */
	/* inference_context itself is defined above inside the !V7_ONLY_SAFE guard */
#ifndef V7_ONLY_SAFE
	/* --- SectorLogo64 interpreter -- Logo-dialect sidecar hook --- */
	typedef int (V7_ABI *v7s_sectorlogo_on_token_fn)(inference_context *ctx, int token_id, int position, int *out_token, float *temperature, float *topp);
	typedef int (V7_ABI *v7s_sectorlogo_interp_fn)(const char *script);
	typedef void (V7_ABI *v7s_set_caller_hmodule_fn)(void *hmodule);
	/* --- Execute child process and capture stdout --- */
	typedef char *(V7_ABI *v7_execute_fn)(const char *exe, const char *parameter1, const char *parameter2, const char *parameter3);
	/* --- Script-reachable fatal: print message, then die through the standard engine FATAL path --- */
	typedef void (V7_ABI *v7_do_fatal_fn)(const char *message);
	/* --- Token stream diagnostic dump gate (exported BSS dword): 1 = chat.inc
	 * calls dump_token_stream per generated token, printing one line each --- */
	typedef int *enable_token_dump_ptr;
	/* --- InferenceContext diagnostic dump gate (exported BSS dword): 1 =
	 * dump_inference_context prints the whole context at the gated call sites --- */
	typedef int *enable_dump_ptr;
	/* --- CPUID hardware-baseline decision (exported BSS dword): 1 = every required
	 * feature (AVX2, FMA, BMI1, BMI2, AES-NI, SSE4.1, SSE4.2, RDRAND, OS-enabled
	 * YMM state) proven present by the DllMain probe; 0 = unsupported (tools flip
	 * this to exercise the gate -- see test_engine_death.c) --- */
	typedef int *cpu_features_ok_ptr;
	/* --- Bounded sprintf --- */
	/* %f takes a RAW binary32 in the vararg slot's low 4 bytes, NOT a promoted
	   double: pass *(int*)&x. v7sprintf(buf, n, "%f", 1.0f) prints "0". */
	typedef int (V7_ABI *v7sprintf_fn)(char *buffer, int buf_size, const char *format, ...);
	/* --- File I/O into shared 64KB buffer --- */
	typedef char *(V7_ABI *v7s_readfile_fn)(const char *path);
	typedef int (V7_ABI *v7s_writefile_fn)(const char *path, char *text, int append);
	/* --- Chat/generate/vision image entries: PUBLIC tier only (internal v7s_* typedefs removed with their exports; public v7_chat / v7_generate / v7_chat_image / v7_generate_image typedefs live in the public tier block at the top) --- */
	/* --- Vision (VLM) exports -- embedding upload --- */
	/* Uploads projected vision embeddings + DeepStack feature tensors (all optional/NULL-able) */
	typedef void (V7_ABI *v7s_set_vision_embeddings_fn)(float *embeddings, int emb_count, float *ds0, float *ds1, float *ds2);
	/* Get pointer to generation performance statistics block.
	 * Returns &rdtsc_chat_start -- a flat block of 10 uint64 variables, written
	 * by RDTSC probes inside v7s_chat / v7s_chat_image_mem and read after the
	 * call returns. Qword index layout:
	 *   [0] rdtsc_chat_start             TSC at v7s_chat entry
	 *   [1] rdtsc_after_tokenization     TSC after encode()
	 *   [2] rdtsc_after_prefill          TSC after prefill/single-token forward
	 *   [3] rdtsc_after_generation       TSC at generation finish
	 *   [4] number_token_prefill         prompt tokens encoded this round
	 *   [5] number_token_generate        tokens generated this round
	 *   [6] rdtsc_vit_encode_start       TSC before ViT encode (image turns, else 0)
	 *   [7] rdtsc_vit_encode_end         TSC after successful ViT encode (else 0)
	 *   [8] number_token_prefill_vision  image-turn prefill tokens (mirror of [4])
	 *   [9] number_token_generate_vision image-turn generated tokens (mirror of [5])
	 * Phase durations are differences of consecutive timestamps; the caller
	 * calibrates cycles->time (e.g. via QueryPerformanceCounter). */
	typedef unsigned long long *(V7_ABI *v7s_get_generation_statistic_fn)(void);
	/* Pure-ASM Qwen3-VL vision tower (Phase 9). 8-arg compute-only encoder called
	 * directly by v7s_chat_image_mem / v7s_generate_image_mem (chat_image.inc).
	 * Loads ViT weights from microdb, preprocesses the caller-supplied raw_rgb
	 * buffer, runs the full ViT forward, and returns the projected embeddings and
	 * DeepStack features. Returns 0 on success, nonzero on error. */
	typedef int (V7_ABI *v7s_vit_encode_callback_fn)(const unsigned char *raw_rgb, int img_w, int img_h, float **out_emb, int *out_count, float **out_ds0, float **out_ds1, float **out_ds2);
	/* Pure-ASM ViT forward pass: patch_embed + 24 transformer blocks + mergers.
	 * weights points at the vit_weights_t blob laid out over the microdb data. */
	typedef void (V7_ABI *v7s_vit_forward_fn)(const float *pixels, int grid_h, int grid_w, const void *weights, float *out_main, float *out_ds0, float *out_ds1, float *out_ds2);
	/* --- Tokenizer encode (BPE) --- */
	typedef void (V7_ABI *encode_fn)(void *tok, const char *text, int *tokens, int *n_tokens, int max_tokens);
	/* --- Forward pass (full transformer inference) --- */
	typedef float *(V7_ABI *forward_f32_fn)(void *transformer, int token, int pos);
	typedef float *(V7_ABI *forward_q8_fn)(void *transformer, int token, int pos);
	typedef float *(V7_ABI *prefill_forward_f32_fn)(void *transformer, int *tokens, int n_tokens, int start_pos);
	typedef float *(V7_ABI *prefill_forward_q8_fn)(void *transformer, int *tokens, int n_tokens, int start_pos);
#endif /* !V7_ONLY_SAFE */
/* Per-function profiling slot (accumulator for V7_LIB_CALL) */
#define V7_PROF_FIELD(name)                                                                                                                                                                                                                              \
	long long _prof_##name##_cycles;                                                                                                                                                                                                                       \
	int _prof_##name##_calls
	/* ---------------------------------------------------------------------------
	 * Main library struct
	 * ------------------------------------------------------------------------- */
	typedef struct v7_lib
	{
		/* DLL module handle (v7_unload uses it) */
		void *module;
#ifndef V7_ONLY_SAFE
		/* ---- Scheduled matmul kernels ---- */
		v7d_q8_fn v7d_q8;
		V7_PROF_FIELD(v7d_q8);
		v7d_f32_fn v7d_f32;
		V7_PROF_FIELD(v7d_f32);
		v7d_fused_f32_fn v7d_fused_f32;
		V7_PROF_FIELD(v7d_fused_f32);
		v7d_swiglu_fn v7d_swiglu;
		V7_PROF_FIELD(v7d_swiglu);
		v7d_swiglu_q8_fn v7d_swiglu_q8;
		V7_PROF_FIELD(v7d_swiglu_q8);
		/* ---- Scheduled attention ---- */
		v7d_attention_fn v7d_attention;
		V7_PROF_FIELD(v7d_attention);
		v7d_attention_prefill_fn v7d_attention_prefill;
		V7_PROF_FIELD(v7d_attention_prefill);
		/* ---- Scheduled utility kernels ---- */
		v7d_quantize_fn v7d_quantize;
		V7_PROF_FIELD(v7d_quantize);
		v7d_memcpy_fn v7d_memcpy;
		V7_PROF_FIELD(v7d_memcpy);
		v7d_vit_linear_fn v7d_vit_linear;
		V7_PROF_FIELD(v7d_vit_linear);
		v7s_bicubic_resize_fn v7s_bicubic_resize;
		V7_PROF_FIELD(v7s_bicubic_resize);
		/* ---- MicroDB hash table ---- */
		ht_init_fn ht_init;
		V7_PROF_FIELD(ht_init);
		ht_destroy_fn ht_destroy;
		V7_PROF_FIELD(ht_destroy);
		ht_add_fn ht_add;
		V7_PROF_FIELD(ht_add);
	ht_find_fn ht_find;
	V7_PROF_FIELD(ht_find);
	db_save_fn db_save;
	V7_PROF_FIELD(db_save);
	db_load_fn db_load;
	V7_PROF_FIELD(db_load);
	mdb_compress_fn mdb_compress;
		V7_PROF_FIELD(mdb_compress);
		mdb_decompress_fn mdb_decompress;
		V7_PROF_FIELD(mdb_decompress);
		db_set_readonly_fn db_set_readonly;
		V7_PROF_FIELD(db_set_readonly);
#endif /* !V7_ONLY_SAFE */
				/* ---- PUBLIC v7_* API TIER (context hidden in BSS) ---- */
		v7_init_fn v7_init;
		v7_db_load_fn v7_db_load;
		v7_db_save_fn v7_db_save;
		v7_initialize_inference_fn v7_initialize_inference;
		v7_free_inference_fn v7_free_inference;
		v7_chat_fn v7_chat;
		V7_PROF_FIELD(v7_chat);
		v7_generate_fn v7_generate;
		V7_PROF_FIELD(v7_generate);
		v7_chat_image_fn v7_chat_image;
		v7_generate_image_fn v7_generate_image;
		v7_set_sampler_temperature_fn v7_set_sampler_temperature;
		v7_set_sampler_topp_fn v7_set_sampler_topp;
		v7_set_sampler_presence_penalty_fn v7_set_sampler_presence_penalty;
		v7_set_sampler_topk_fn v7_set_sampler_topk;
		v7_set_sampler_seed_fn v7_set_sampler_seed;
		v7_set_quantize_fn v7_set_quantize;
		v7_set_max_gen_fn v7_set_max_gen;
		v7_engine_is_dead_fn v7_engine_is_dead;
		v7_get_death_reason_fn v7_get_death_reason;
v7_get_version_fn v7_get_version;
#ifndef V7_ONLY_SAFE
v7_execute_fmt_fn v7_execute_fmt;
#endif /* !V7_ONLY_SAFE */
v7_get_console_width_fn v7_get_console_width;
#ifndef V7_ONLY_SAFE
v7s_get_user_prompt_fn v7s_get_user_prompt;
#endif /* !V7_ONLY_SAFE */
		save_kv_cache_fn save_kv_cache;
		load_kv_cache_fn load_kv_cache;
		v7_set_streaming_fn v7_set_streaming;
		v7_get_position_fn v7_get_position;
		v7_set_position_fn v7_set_position;
		v7_get_think_mode_active_fn v7_get_think_mode_active;
		v7_get_gen_count_fn v7_get_gen_count;
		v7_get_token_probs_fn v7_get_token_probs;
		v7_set_worker_priority_fn v7_set_worker_priority;
		v7_enable_thinking_fn v7_enable_thinking;
		v7_db_add_fn v7_db_add;
		v7_db_find_fn v7_db_find;
		v7_encode_fn v7_encode;
		v7_get_generation_statistic_fn v7_get_generation_statistic;
		v7_get_best_fitness_fn v7_get_best_fitness;
		v7_get_pool_fitness_fn v7_get_pool_fitness;
		v7_get_active_shape_count_fn v7_get_active_shape_count;
		v7_shutdown_fn v7_shutdown;
#ifndef V7_ONLY_SAFE
		v7_get_internal_inference_ctx_fn v7_get_internal_inference_ctx;
		/* ---- SectorLogo64 interpreter ---- */
		v7s_sectorlogo_on_token_fn v7s_sectorlogo_on_token;
		v7s_sectorlogo_interp_fn v7s_sectorlogo_interp;
		v7s_set_caller_hmodule_fn v7s_set_caller_hmodule;
		/* ---- Execute child process, capture stdout ---- */
		v7_execute_fn v7_execute;
		/* ---- Script-reachable fatal (print message, then standard engine FATAL) ---- */
		v7_do_fatal_fn v7_do_fatal;
		/* ---- Token stream diagnostic dump gate (exported BSS dword) ---- */
		enable_token_dump_ptr enable_token_dump;
		/* ---- InferenceContext diagnostic dump gate (exported BSS dword) ---- */
		enable_dump_ptr enable_dump;
		/* ---- CPUID hardware-baseline decision (exported BSS dword) ---- */
		cpu_features_ok_ptr cpu_features_ok;
		/* ---- Bounded sprintf ---- */
		v7sprintf_fn v7sprintf;
		/* ---- File I/O into shared 64KB buffer ---- */
		v7s_readfile_fn v7s_readfile;
		v7s_writefile_fn v7s_writefile;
#endif /* !V7_ONLY_SAFE */
		/* ---- Chat/generate/vision: PUBLIC tier only (internal v7s_* exports removed) ---- */
#ifndef V7_ONLY_SAFE
		/* ---- Vision (VLM) exports ---- */
		v7s_set_vision_embeddings_fn v7s_set_vision_embeddings;
		v7s_vit_encode_callback_fn v7s_vit_encode_callback;
		v7s_vit_forward_fn v7s_vit_forward;
		/* ---- Generation statistics access ---- */
		v7s_get_generation_statistic_fn v7s_get_generation_statistic;
		/* ---- Tokenizer encode ---- */
		encode_fn encode;
		/* ---- Forward pass ---- */
		forward_f32_fn forward_f32;
		forward_q8_fn forward_q8;
		prefill_forward_f32_fn prefill_forward_f32;
		prefill_forward_q8_fn prefill_forward_q8;
		/* ---- M-RoPE (multimodal rotary position embeddings) ---- */
		v7s_mrope_prepare_fn v7s_mrope_prepare;
		v7s_mrope_restore_fn v7s_mrope_restore;
		/* ---- Direct-call kernels ---- */
		v7s_sample_fn v7s_sample;
		V7_PROF_FIELD(v7s_sample);
		v7s_rmsnorm_fn v7s_rmsnorm;
		V7_PROF_FIELD(v7s_rmsnorm);
		v7s_add_inplace_fn v7s_add_inplace;
		V7_PROF_FIELD(v7s_add_inplace);
		v7s_fused_add_rmsnorm_fn v7s_fused_add_rmsnorm;
		V7_PROF_FIELD(v7s_fused_add_rmsnorm);
		v7s_rmsnorm_rope_heads_fn v7s_rmsnorm_rope_heads;
		V7_PROF_FIELD(v7s_rmsnorm_rope_heads);
		v7s_rmsnorm_quantize_fn v7s_rmsnorm_quantize;
		V7_PROF_FIELD(v7s_rmsnorm_quantize);
		/* ---- Fused-kernel support ---- */
		v7s_rope_init_fn v7s_rope_init;
		V7_PROF_FIELD(v7s_rope_init);
		v7s_set_fused_rope_fn v7s_set_fused_rope;
		V7_PROF_FIELD(v7s_set_fused_rope);
		/* ---- Runtime control ---- */
		v7s_set_spin_fn v7s_set_spin;
		V7_PROF_FIELD(v7s_set_spin);
		v7s_enable_thinking_fn v7s_enable_thinking;
		v7s_shutdown_fn v7s_shutdown;
		V7_PROF_FIELD(v7s_shutdown);
		v7s_get_num_workers_fn v7s_get_num_workers;
		V7_PROF_FIELD(v7s_get_num_workers);
		get_config_was_fixed_fn get_config_was_fixed;
		/* ---- In-system A/B tuning API ---- */
		set_prefetch_distance_fn set_prefetch_distance;
		set_prefetch_type_fn set_prefetch_type;
		v7s_sectorlogo_eval_script_fn v7s_sectorlogo_eval_script;
#endif /* !V7_ONLY_SAFE */
	} v7_lib;
/* Fill-grade convenience (StreamingLLM): start streaming/eviction when the
 * cache holds `fill_grade` rows; window_size = fill_grade - num_sinks.
 * Returns -1 for a non-positive fill grade. */
static inline int v7_set_streaming_fill(v7_lib *lib, int enable, int num_sinks, int fill_grade)
{
	if (fill_grade > num_sinks)
		return lib->v7_set_streaming(enable, num_sinks, fill_grade - num_sinks);
	return -1;
}
/* Wrap any dll.* call with RDTSC timing + accumulator update.
 * Usage: V7_LIB_CALL(&dll, v7d_q8, xout, &x, &w, n, d); */
#define V7_LIB_CALL(lib, fn, ...)                                                                                                                                                                                                                        \
	do                                                                                                                                                                                                                                                   \
	{                                                                                                                                                                                                                                                    \
		unsigned long long _v7_start_ = __rdtsc();                                                                                                                                                                                                         \
		(lib)->fn(__VA_ARGS__);                                                                                                                                                                                                                          \
		unsigned long long _v7_end_ = __rdtsc();                                                                                                                                                                                                           \
		(lib)->_prof_##fn##_cycles += (_v7_end_ - _v7_start_);                                                                                                                                                                                           \
		(lib)->_prof_##fn##_calls++;                                                                                                                                                                                                                     \
	} while (0)
#ifdef __cplusplus
}
#endif
/* ---------------------------------------------------------------------------
 * Implementation (static -- safe for multiple includes)
 * ------------------------------------------------------------------------- */
#include <stdio.h>
#include <string.h>
#ifdef __linux__
#include <dlfcn.h>
#include <stdlib.h>
#include <x86intrin.h>
#define V7_OPEN(path) dlopen(path, RTLD_NOW | RTLD_LOCAL)
#define V7_SYMBOL(module, name) dlsym(module, name)
#define V7_CLOSE(module) dlclose(module)
#define V7_LOAD_FATAL() exit(EXIT_FAILURE)
typedef void *v7_module;
typedef void *v7_symbol;
#else
#include <windows.h>
#include <intrin.h>
#define V7_OPEN(path) LoadLibraryA(path)
#define V7_SYMBOL(module, name) GetProcAddress(module, name)
#define V7_CLOSE(module) FreeLibrary(module)
#define V7_LOAD_FATAL() TerminateProcess(GetCurrentProcess(), 1)
typedef HMODULE v7_module;
typedef FARPROC v7_symbol;
#endif
static int v7_load(const char *dll_path, v7_lib *lib)
{
	if (!lib)
		return -1;
	memset(lib, 0, sizeof(*lib));
	v7_module h = V7_OPEN(dll_path);
	if (!h)
	{
		printf("[v7_lib] failed to load %s\n", dll_path);
#ifdef __linux__
		printf("[v7_lib] %s\n", dlerror());
#endif
		return -1;
	}
	lib->module = (void *)h;
	/* FARPROC + memcpy avoids void*->funcptr cast issues in C++ */
#define V7_RESOLVE(name)                                                                                                                                                                                                                                 \
	do                                                                                                                                                                                                                                                   \
	{                                                                                                                                                                                                                                                    \
		v7_symbol p__ = V7_SYMBOL(h, #name);                                                                                                                                                                                                          \
		if (!p__)                                                                                                                                                                                                                                        \
		{                                                                                                                                                                                                                                                \
			printf("[FATAL] v7_lib: missing required export '%s' in %s\n", #name, dll_path);                                                                                                                                                             \
			V7_LOAD_FATAL();                                                                                                                                                                                                    \
		}                                                                                                                                                                                                                                                \
		memcpy(&lib->name, &p__, sizeof(p__));                                                                                                                                                                                                           \
	} while (0)
#ifndef V7_ONLY_SAFE
	V7_RESOLVE(v7d_q8);
	V7_RESOLVE(v7d_f32);
	V7_RESOLVE(v7d_fused_f32);
	V7_RESOLVE(v7d_swiglu);
	V7_RESOLVE(v7d_swiglu_q8);
	V7_RESOLVE(v7d_attention);
	V7_RESOLVE(v7d_attention_prefill);
	V7_RESOLVE(v7d_quantize);
	V7_RESOLVE(v7d_memcpy);
	V7_RESOLVE(v7d_vit_linear);
	V7_RESOLVE(v7s_bicubic_resize);
	V7_RESOLVE(v7s_sample);
	V7_RESOLVE(v7s_rmsnorm);
	V7_RESOLVE(v7s_add_inplace);
	V7_RESOLVE(v7s_fused_add_rmsnorm);
	V7_RESOLVE(v7s_rmsnorm_rope_heads);
	V7_RESOLVE(v7s_rmsnorm_quantize);
	V7_RESOLVE(v7s_rope_init);
	V7_RESOLVE(v7s_set_fused_rope);
	V7_RESOLVE(v7s_set_spin);
	V7_RESOLVE(v7s_enable_thinking);
	V7_RESOLVE(v7s_shutdown);
	V7_RESOLVE(v7s_get_num_workers);
	/* Sec.167: which config values the engine degenerated on this system (bit flags) */
	V7_RESOLVE(get_config_was_fixed);
	/* In-system A/B tuning API */
	V7_RESOLVE(set_prefetch_distance);
	V7_RESOLVE(set_prefetch_type);
	V7_RESOLVE(v7s_sectorlogo_eval_script);
	/* Tokenizer encode */
	V7_RESOLVE(encode);
#endif /* !V7_ONLY_SAFE */
	/* Inference init (internal; superseded by the public tier below -- exports removed) */
	/* PUBLIC v7_* API TIER (context hidden in BSS) */
	V7_RESOLVE(v7_init);
	V7_RESOLVE(v7_db_load);
	V7_RESOLVE(v7_db_save);
	V7_RESOLVE(v7_initialize_inference);
	V7_RESOLVE(v7_free_inference);
	V7_RESOLVE(v7_chat);
	V7_RESOLVE(v7_generate);
	V7_RESOLVE(v7_chat_image);
	V7_RESOLVE(v7_generate_image);
	V7_RESOLVE(v7_set_sampler_temperature);
	V7_RESOLVE(v7_set_sampler_topp);
	V7_RESOLVE(v7_set_sampler_presence_penalty);
	V7_RESOLVE(v7_set_sampler_topk);
	V7_RESOLVE(v7_set_sampler_seed);
	V7_RESOLVE(v7_set_quantize);
	V7_RESOLVE(v7_set_max_gen);
	V7_RESOLVE(v7_engine_is_dead);
	V7_RESOLVE(v7_get_death_reason);
V7_RESOLVE(v7_get_version);
#ifndef V7_ONLY_SAFE
V7_RESOLVE(v7_execute_fmt);
#endif /* !V7_ONLY_SAFE */
V7_RESOLVE(v7_get_console_width);
#ifndef V7_ONLY_SAFE
V7_RESOLVE(v7s_get_user_prompt);
#endif /* !V7_ONLY_SAFE */
	V7_RESOLVE(save_kv_cache);
	V7_RESOLVE(load_kv_cache);
	V7_RESOLVE(v7_set_streaming);
	V7_RESOLVE(v7_get_position);
	V7_RESOLVE(v7_set_position);
	V7_RESOLVE(v7_get_think_mode_active);
	V7_RESOLVE(v7_get_gen_count);
	V7_RESOLVE(v7_get_token_probs);
	V7_RESOLVE(v7_set_worker_priority);
	V7_RESOLVE(v7_enable_thinking);
	V7_RESOLVE(v7_db_add);
	V7_RESOLVE(v7_db_find);
	V7_RESOLVE(v7_encode);
	V7_RESOLVE(v7_get_generation_statistic);
	V7_RESOLVE(v7_get_best_fitness);
	V7_RESOLVE(v7_get_pool_fitness);
	V7_RESOLVE(v7_get_active_shape_count);
	V7_RESOLVE(v7_shutdown);
#ifndef V7_ONLY_SAFE
	V7_RESOLVE(v7_get_internal_inference_ctx);
	/* SectorLogo64 interpreter */
	V7_RESOLVE(v7s_sectorlogo_on_token);
	V7_RESOLVE(v7s_sectorlogo_interp);
	V7_RESOLVE(v7s_set_caller_hmodule);
	/* Execute child process, capture stdout */
	V7_RESOLVE(v7_execute);
	/* Script-reachable fatal (print message, then standard engine FATAL) */
	V7_RESOLVE(v7_do_fatal);
	/* Token stream diagnostic dump gate (exported BSS dword) */
	V7_RESOLVE(enable_token_dump);
	/* InferenceContext diagnostic dump gate (exported BSS dword) */
	V7_RESOLVE(enable_dump);
	/* CPUID hardware-baseline decision (exported BSS dword) */
	V7_RESOLVE(cpu_features_ok);
	/* Bounded sprintf */
	V7_RESOLVE(v7sprintf);
	/* File I/O into shared 64KB buffer */
	V7_RESOLVE(v7s_readfile);
	V7_RESOLVE(v7s_writefile);
	/* Chat/generate: resolved in the public tier block above (internal exports removed) */
	/* Vision (VLM) exports */
	V7_RESOLVE(v7s_set_vision_embeddings);
	V7_RESOLVE(v7s_vit_encode_callback);
	V7_RESOLVE(v7s_vit_forward);
	V7_RESOLVE(v7s_get_generation_statistic);
	/* Forward pass */
	V7_RESOLVE(forward_f32);
	V7_RESOLVE(forward_q8);
	V7_RESOLVE(prefill_forward_f32);
	V7_RESOLVE(prefill_forward_q8);
	/* M-RoPE (multimodal rotary position embeddings) */
	V7_RESOLVE(v7s_mrope_prepare);
	V7_RESOLVE(v7s_mrope_restore);
	/* MicroDB hash table functions */
	V7_RESOLVE(ht_init);
	V7_RESOLVE(ht_destroy);
	V7_RESOLVE(ht_add);
	V7_RESOLVE(ht_find);
	V7_RESOLVE(db_save);
	V7_RESOLVE(db_load);
	V7_RESOLVE(mdb_compress);
	V7_RESOLVE(mdb_decompress);
	V7_RESOLVE(db_set_readonly);
#endif /* !V7_ONLY_SAFE */
#undef V7_RESOLVE
	return 0;
}
static void v7_unload(v7_lib *lib)
{
	if (!lib || !lib->module)
		return;
	V7_CLOSE((v7_module)lib->module);
	memset(lib, 0, sizeof(*lib));
}
/* ---- Profiling implementation ---- */
static void v7_lib_reset_stats(v7_lib *lib)
{
	if (!lib)
		return;
#define V7_ZERO_PROF(name)                                                                                                                                                                                                                               \
	lib->_prof_##name##_cycles = 0;                                                                                                                                                                                                                      \
	lib->_prof_##name##_calls = 0
#ifndef V7_ONLY_SAFE
	V7_ZERO_PROF(v7d_q8);
	V7_ZERO_PROF(v7d_f32);
	V7_ZERO_PROF(v7d_fused_f32);
	V7_ZERO_PROF(v7d_swiglu);
	V7_ZERO_PROF(v7d_swiglu_q8);
	V7_ZERO_PROF(v7d_attention);
	V7_ZERO_PROF(v7d_attention_prefill);
	V7_ZERO_PROF(v7d_quantize);
	V7_ZERO_PROF(v7d_memcpy);
	V7_ZERO_PROF(v7d_vit_linear);
	V7_ZERO_PROF(v7s_bicubic_resize);
	V7_ZERO_PROF(v7s_sample);
	V7_ZERO_PROF(v7s_rmsnorm);
	V7_ZERO_PROF(v7s_add_inplace);
	V7_ZERO_PROF(v7s_fused_add_rmsnorm);
	V7_ZERO_PROF(v7s_rmsnorm_rope_heads);
	V7_ZERO_PROF(v7s_rmsnorm_quantize);
	V7_ZERO_PROF(v7s_rope_init);
	V7_ZERO_PROF(v7s_set_fused_rope);
	V7_ZERO_PROF(v7s_set_spin);
	V7_ZERO_PROF(v7s_shutdown);
	V7_ZERO_PROF(v7s_get_num_workers);
	V7_ZERO_PROF(ht_init);
	V7_ZERO_PROF(ht_destroy);
	V7_ZERO_PROF(ht_add);
	V7_ZERO_PROF(ht_find);
	V7_ZERO_PROF(db_save);
	V7_ZERO_PROF(db_load);
	V7_ZERO_PROF(mdb_compress);
	V7_ZERO_PROF(mdb_decompress);
	V7_ZERO_PROF(db_set_readonly);
#endif /* !V7_ONLY_SAFE */
	V7_ZERO_PROF(v7_chat);
	V7_ZERO_PROF(v7_generate);
#undef V7_ZERO_PROF
}
static void v7_lib_print_stats(v7_lib *lib, FILE *file)
{
	if (!lib || !file)
		return;
	/* Collect profile entries */
	typedef struct v7_prof_entry_
	{
		const char *name;
		long long cycles;
		int calls;
	} v7_prof_entry_;
#define V7_PROF_ENTRY(name) { #name, lib->_prof_##name##_cycles, lib->_prof_##name##_calls }
	v7_prof_entry_ entries[] = {
#ifndef V7_ONLY_SAFE
		V7_PROF_ENTRY(v7d_q8),
		V7_PROF_ENTRY(v7d_f32),
		V7_PROF_ENTRY(v7d_fused_f32),
		V7_PROF_ENTRY(v7d_swiglu),
		V7_PROF_ENTRY(v7d_swiglu_q8),
		V7_PROF_ENTRY(v7d_attention),
		V7_PROF_ENTRY(v7d_attention_prefill),
		V7_PROF_ENTRY(v7d_quantize),
		V7_PROF_ENTRY(v7d_memcpy),
		V7_PROF_ENTRY(v7d_vit_linear),
		V7_PROF_ENTRY(v7s_sample),
		V7_PROF_ENTRY(v7s_rmsnorm),
		V7_PROF_ENTRY(v7s_add_inplace),
		V7_PROF_ENTRY(v7s_fused_add_rmsnorm),
		V7_PROF_ENTRY(v7s_rmsnorm_rope_heads),
		V7_PROF_ENTRY(v7s_rmsnorm_quantize),
		V7_PROF_ENTRY(v7s_rope_init),
		V7_PROF_ENTRY(v7s_set_fused_rope),
		V7_PROF_ENTRY(v7s_set_spin),
		V7_PROF_ENTRY(v7s_shutdown),
		V7_PROF_ENTRY(v7s_get_num_workers),
		V7_PROF_ENTRY(ht_init),
		V7_PROF_ENTRY(ht_destroy),
		V7_PROF_ENTRY(ht_add),
		V7_PROF_ENTRY(ht_find),
		V7_PROF_ENTRY(db_save),
		V7_PROF_ENTRY(db_load),
		V7_PROF_ENTRY(mdb_compress),
		V7_PROF_ENTRY(mdb_decompress),
		V7_PROF_ENTRY(db_set_readonly),
#endif /* !V7_ONLY_SAFE */
		V7_PROF_ENTRY(v7_chat),
		V7_PROF_ENTRY(v7_generate),
	};
#undef V7_PROF_ENTRY
	int n = (int)(sizeof(entries) / sizeof(entries[0]));
	/* Simple insertion sort by cycles descending */
	for (int i = 1; i < n; i++)
	{
		v7_prof_entry_ tmp = entries[i];
		int j = i - 1;
		while (j >= 0 && entries[j].cycles < tmp.cycles)
		{
			entries[j + 1] = entries[j];
			j--;
		}
		entries[j + 1] = tmp;
	}
	/* Print header */
	printf("\n=== v7_lib profile ========================================\n");
	printf("%-32s %16s %8s %16s\n", "Function", "Cycles", "Calls", "Cycles/Call");
	long long total = 0;
	for (int i = 0; i < n; i++)
		total += entries[i].cycles;
	for (int i = 0; i < n; i++)
	{
		if (entries[i].calls == 0)
			continue;
		double pct = (double)entries[i].cycles / total * 100.0;
		long long per = entries[i].cycles / entries[i].calls;
		printf("%-32s %16lld %8d %16lld  (%5.1f%%)\n", entries[i].name, (long long)entries[i].cycles, entries[i].calls, (long long)per, pct);
	}
	printf("%-32s %16lld\n", "TOTAL", (long long)total);
	printf("================================================================\n");
}
