// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v_fused_validate.cpp -- Fused matmul correctness validator (one validator for
 * the whole fused-matmul module)
 *
 * Two sections in one run, same pass/fail contract:
 *   1. F32 fused matmul+RMSNorm+RoPE (v7d_fused_f32, v7s_set_fused_rope,
 *      v7s_rope_init) against the C reference, at many positions/shapes.
 *   2. Q8 fused matmul (v7d_q8_fused: F32 input + Q8 weight directly, no input
 *      quantize step) against the quantize-then-Q8-matmul C reference.
 * Usage: v_fused_validate.exe <dll_path> --db <db_path>
 */
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "kt_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

/* v7d_q8_fused is exported but NOT in v7_lib.h -- define our own typedef */
typedef void (V7_ABI *q8_fused_fn)(float *xout, v7_f32_tensor *x, v7_q8_tensor *w, int matmul_rows, int matmul_cols);

typedef struct q8_fused_entry
{
	int rows;
	int cols;
	const char *desc;

} q8_fused_entry;

static const q8_fused_entry q8_tests[] = { /* Qwen3-0.6B: dim=1024, hidden=3072, heads 16/8 x hd128, vocab 151936 */
	                          { 1024, 2048, "Qwen3-0.6B Q proj" },
	                          { 1024, 1024, "Qwen3-0.6B K/V proj (kv_dim=1024)" },
	                          { 2048, 1024, "Qwen3-0.6B O proj" },
	                          { 1024, 3072, "Qwen3-0.6B FFN gate/up" },
	                          { 3072, 1024, "Qwen3-0.6B FFN down" },
	                          /* Llama 3.2 1B: dim=2048, hidden=8192, heads 32/8 x hd64, vocab 128256 */
	                          { 2048, 2048, "Llama Q proj" },
	                          { 2048, 512, "Llama K/V proj" },
	                          { 2048, 8192, "Llama FFN gate/up" },
	                          { 8192, 2048, "Llama FFN down" },
	                          /* SmolLM2-135M: dim=576, hidden=1536, heads 9/3 x hd64 */
	                          { 576, 576, "SmolLM2-135M Q proj" },
	                          { 576, 192, "SmolLM2-135M K/V proj" },
	                          { 576, 1536, "SmolLM2-135M FFN gate/up" },
	                          { 1536, 576, "SmolLM2-135M FFN down" },
	                          /* SmolLM2-360M: dim=960, hidden=2560, heads 15/5 x hd64 */
	                          { 960, 960, "SmolLM2-360M Q proj" },
	                          { 960, 320, "SmolLM2-360M K/V proj" },
	                          { 960, 2560, "SmolLM2-360M FFN gate/up" },
	                          { 2560, 960, "SmolLM2-360M FFN down" },
	                          /* TinyLlama-1.1B: dim=2048, hidden=5632, heads 32/4 x hd64 */
	                          { 2048, 256, "TinyLlama K/V proj" },
	                          { 2048, 5632, "TinyLlama FFN gate/up" },
	                          { 5632, 2048, "TinyLlama FFN down" },
	                          /* Phi-3-mini: dim=3072, hidden=8192, MHA 32 x hd96 */
	                          { 3072, 3072, "Phi-3 Q proj (MHA)" },
	                          { 3072, 8192, "Phi-3 FFN gate/up" },
	                          { 8192, 3072, "Phi-3 FFN down" },
	                          /* MiniCPM5-1B: dim=1536, hidden=4608, heads 16/2 x hd128 */
	                          { 1536, 3072, "MiniCPM5 Q proj" },
	                          { 1536, 256, "MiniCPM5 K/V proj" },
	                          { 1536, 4608, "MiniCPM5 FFN gate/up" },
	                          { 4608, 1536, "MiniCPM5 FFN down" },
	                          /* Qwen2.5-0.5B: dim=896, hidden=4864, heads 14/2 x hd64 */
	                          { 896, 896, "Qwen2.5-0.5B Q proj" },
	                          { 896, 128, "Qwen2.5-0.5B K/V proj" },
	                          { 896, 4864, "Qwen2.5-0.5B FFN gate/up" },
	                          { 4864, 896, "Qwen2.5-0.5B FFN down" },
	                          { 0, 0, 0 } };
static const int q8_seeds[] = { 12345, 67890, 11111, 22222 };
#define Q8_N_SEEDS 4

static int verify_q8_fused_standalone(q8_fused_fn func, int matmul_rows, int matmul_cols, int seed, float *out_max_diff)
{
	float *in_x = (float *)malloc(matmul_rows * sizeof(float));
	__int64 w_count = (__int64)matmul_cols * matmul_rows;
	int w_scales_count = (int)(w_count / 64) + 1;
	char *w_q = (char *)malloc(w_count);
	float *w_s = (float *)malloc(w_scales_count * sizeof(float));
	float *ref = (float *)malloc(matmul_cols * sizeof(float));
	float *asm_ = (float *)malloc(matmul_cols * sizeof(float));
	if ((!in_x) || (!w_q) || (!w_s) || (!ref) || (!asm_))
	{
		free(in_x);
		free(w_q);
		free(w_s);
		free(ref);
		free(asm_);
		return 0;
	}

	srand(seed);
	kt_init_random_float(in_x, matmul_rows);
	kt_init_random_int8(w_q, w_count);
	kt_init_random_float(w_s, w_scales_count);

	/* Reference: quantize F32->int8, then Q8 matmul */
	int n_groups_in = (matmul_rows + 63) / 64;
	char *x_q = (char *)malloc(matmul_rows);
	float *x_s = (float *)malloc(n_groups_in * sizeof(float));
	if ((!x_q) || (!x_s))
	{
		free(in_x);
		free(w_q);
		free(w_s);
		free(ref);
		free(asm_);
		free(x_q);
		free(x_s);
		return 0;
	}
	kt_quantize_c_ref(x_q, in_x, x_s, matmul_rows);
	v7_q8_tensor x_ref = { x_q, x_s };
	v7_q8_tensor w_ref = { w_q, w_s };
	kt_matmul_c_ref_q8(ref, &x_ref, &w_ref, matmul_rows, matmul_cols);

	/* ASM: v7d_q8_fused takes F32 input + Q8 weight directly */
	v7_f32_tensor x_asm = { in_x, 0 };
	v7_q8_tensor w_asm = { w_q, w_s };
	memset(asm_, 0, matmul_cols * sizeof(float));
	func(asm_, &x_asm, &w_asm, matmul_rows, matmul_cols);

	kt_compare_result cr = kt_compare_outputs(ref, asm_, matmul_cols, 0.01f, 1e-4f);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;

	free(in_x);
	free(w_q);
	free(w_s);
	free(ref);
	free(asm_);
	free(x_q);
	free(x_s);
	return cr.pass;
}

typedef struct fused_test_entry
{
	int rows;
	int cols;
	int head_dim;
	int pos;
	float eps;
	const char *desc;

} fused_test_entry;

static const fused_test_entry tests[] = {
	/* Qwen3-0.6B Q proj (dim=1024, 16 heads x hd=128, q_dim=2048, eps=1e-6) at key positions */
	{ 1024, 2048, 128, 0, 1e-6f, "Qwen3 Q proj pos=0 (first token)" },
	{ 1024, 2048, 128, 1, 1e-6f, "Qwen3 Q proj pos=1" },
	{ 1024, 2048, 128, 7, 1e-6f, "Qwen3 Q proj pos=7" },
	{ 1024, 2048, 128, 31, 1e-6f, "Qwen3 Q proj pos=31 (prompt)" },
	{ 1024, 2048, 128, 63, 1e-6f, "Qwen3 Q proj pos=63" },
	{ 1024, 2048, 128, 127, 1e-6f, "Qwen3 Q proj pos=127" },
	{ 1024, 2048, 128, 255, 1e-6f, "Qwen3 Q proj pos=255" },
	{ 1024, 2048, 128, 511, 1e-6f, "Qwen3 Q proj pos=511" },
	{ 1024, 2048, 128, 1023, 1e-6f, "Qwen3 Q proj pos=1023" },
	{ 1024, 2048, 128, 2047, 1e-6f, "Qwen3 Q proj pos=2047" },
	{ 1024, 2048, 128, 4095, 1e-6f, "Qwen3 Q proj pos=4095" },
	{ 1024, 2048, 128, 8191, 1e-6f, "Qwen3 Q proj pos=8191" },
	{ 1024, 2048, 128, 16383, 1e-6f, "Qwen3 Q proj pos=16383" },
	{ 1024, 2048, 128, 32767, 1e-6f, "Qwen3 Q proj pos=32767" },
	{ 1024, 2048, 128, 65535, 1e-6f, "Qwen3 Q proj pos=65535" },
	{ 1024, 2048, 128, 131071, 1e-6f, "Qwen3 Q proj pos=131071 (max seq)" },
	/* Qwen3-0.6B K/V proj (8 kv_heads x hd=128, kv_dim=1024) */
	{ 1024, 1024, 128, 15, 1e-6f, "Qwen3 K/V proj pos=15" },
	{ 1024, 1024, 128, 511, 1e-6f, "Qwen3 K/V proj pos=511" },
	/* Llama 3.2 1B Q proj (dim=2048, 32 heads x hd=64, q_dim=2048, eps=1e-5) */
	{ 2048, 2048, 64, 0, 1e-5f, "Llama Q proj pos=0" },
	{ 2048, 2048, 64, 7, 1e-5f, "Llama Q proj pos=7" },
	{ 2048, 2048, 64, 31, 1e-5f, "Llama Q proj pos=31" },
	{ 2048, 2048, 64, 127, 1e-5f, "Llama Q proj pos=127" },
	{ 2048, 2048, 64, 1023, 1e-5f, "Llama Q proj pos=1023" },
	{ 2048, 2048, 64, 8191, 1e-5f, "Llama Q proj pos=8191" },
	{ 2048, 2048, 64, 131071, 1e-5f, "Llama Q proj pos=131071" },
	/* Llama K projection (8 kv_heads x hd=64, kv_dim=512) */
	{ 2048, 512, 64, 7, 1e-5f, "Llama K proj pos=7" },
	{ 2048, 512, 64, 31, 1e-5f, "Llama K proj pos=31" },
	/* Phi-3-mini Q proj (dim=3072, MHA 32 x hd=96, q_dim=3072, eps=1e-5) -- rp_loop tail */
	{ 3072, 3072, 96, 15, 1e-5f, "Phi-3 Q proj pos=15" },
	{ 3072, 3072, 96, 511, 1e-5f, "Phi-3 Q proj pos=511" },
	/* SmolLM2-135M Q proj (dim=576, 9 heads x hd=64, q_dim=576, eps=1e-5) */
	{ 576, 576, 64, 7, 1e-5f, "SmolLM2-135M Q proj pos=7" },
	{ 576, 576, 64, 511, 1e-5f, "SmolLM2-135M Q proj pos=511" },
	/* SmolLM2-360M Q proj (dim=960, 15 heads x hd=64) */
	{ 960, 960, 64, 7, 1e-5f, "SmolLM2-360M Q proj pos=7" },
	{ 960, 960, 64, 511, 1e-5f, "SmolLM2-360M Q proj pos=511" },
	/* TinyLlama Q proj (dim=2048, 32 heads x hd=64) */
	{ 2048, 2048, 64, 15, 1e-5f, "TinyLlama Q proj pos=15" },
	{ 2048, 2048, 64, 2047, 1e-5f, "TinyLlama Q proj pos=2047" },
	/* MiniCPM5-1B Q proj (dim=1536, 16 heads x hd=128, q_dim=3072, eps=1e-6) */
	{ 1536, 3072, 128, 15, 1e-6f, "MiniCPM5 Q proj pos=15" },
	{ 1536, 3072, 128, 511, 1e-6f, "MiniCPM5 Q proj pos=511" },
	/* Qwen2.5-0.5B Q proj (dim=896, 14 heads x hd=64, q_dim=896, eps=1e-6) */
	{ 896, 896, 64, 7, 1e-6f, "Qwen2.5 Q proj pos=7" },
	{ 896, 896, 64, 511, 1e-6f, "Qwen2.5 Q proj pos=511" },
	{ 0, 0, 0, 0, 0, 0 }
};

static int verify_fused_f32_standalone(v7d_fused_f32_fn func, v7s_set_fused_rope_fn set_rope, v7s_rope_init_fn rope_init, float **bias_ptr_slot, int matmul_rows, int matmul_cols, int head_dim, int pos, float rope_base, float eps, int seed, float *out_max_diff)
{
	kt_tensor_f32 ts;
	if (!kt_tensor_f32_alloc(&ts, matmul_rows, matmul_cols))
		return 0;
	float *ref = (float *)malloc(matmul_cols * sizeof(float));
	float *asm_ = (float *)malloc(matmul_cols * sizeof(float));
	if ((!ref) || (!asm_))
	{
		free(ref);
		free(asm_);
		kt_tensor_f32_free(&ts);
		return 0;
	}

	float *norm_weights = (float *)malloc(head_dim * sizeof(float));
	float *bias = (float *)malloc(matmul_cols * sizeof(float));
	int table_entries = (pos + 1) * (head_dim / 2);
	float *cos_table = (float *)malloc(table_entries * sizeof(float));
	float *sin_table = (float *)malloc(table_entries * sizeof(float));
	if ((!norm_weights) || (!bias) || (!cos_table) || (!sin_table))
	{
		free(ref);
		free(asm_);
		kt_tensor_f32_free(&ts);
		free(norm_weights);
		free(bias);
		free(cos_table);
		free(sin_table);
		return 0;
	}

	srand(seed);
	kt_tensor_f32_fill(&ts, matmul_rows, matmul_cols);
	v7_f32_tensor x = kt_make_input_f32(&ts);
	v7_q8_tensor w = kt_make_weight_f32(&ts);

	for (int i = 0; i < head_dim; i++)
		norm_weights[i] = 0.5f + (float)(rand() % 100) / 100.0f;

	/* A real, non-zero QKV bias. The kernel has no NULL-bias special case
	   (Sec.0a), so the bias must be published exactly as production does -- and a
	   ZEROS block would make the add a no-op that cannot detect a wrong column
	   offset. The magnitude is a few percent of the dot product so the add is
	   clearly above the comparison tolerance without perturbing the RMSNorm. */
	for (int i = 0; i < matmul_cols; i++)
		bias[i] = 0.25f * (float)((i % 7) - 3);
	*bias_ptr_slot = bias;                     /* published by the caller before dispatch */

	{
		float half = (float)(head_dim / 2);
		for (int p = 0; p <= pos; p++)
		{
			for (int j = 0; j < head_dim / 2; j++)
			{
				float freq = powf(rope_base, -(float)j / half);
				float angle = (float)p * freq;
				cos_table[p * (head_dim / 2) + j] = cosf(angle);
				sin_table[p * (head_dim / 2) + j] = sinf(angle);
			}
		}
	}

	memset(ref, 0, matmul_cols * sizeof(float));
	kt_fused_c_ref_f32(ref, &x, &w, matmul_rows, matmul_cols, norm_weights, bias, head_dim, cos_table, sin_table, pos, eps);

	rope_init(head_dim, rope_base);
	set_rope(pos, head_dim);
	memset(asm_, 0, matmul_cols * sizeof(float));
	func(asm_, &x, &w, matmul_rows, matmul_cols, norm_weights, 1);

	kt_compare_result cr = kt_compare_outputs(ref, asm_, matmul_cols, 0.01f, 1e-4f);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;

	free(ref);
	free(asm_);
	kt_tensor_f32_free(&ts);
	free(norm_weights);
	*bias_ptr_slot = 0;                        /* drop the published pointer BEFORE free: the DLL global would otherwise dangle */
	free(bias);
	free(cos_table);
	free(sin_table);
	return cr.pass;
}

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		printf("usage: v_fused_validate.exe <dll_path> --db <db_path>\n");
		return 1;
	}
	if ((strcmp(argv[2], "--db") != 0))
	{
		printf("usage: v_fused_validate.exe <dll_path> --db <db_path>\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if ((v7_load(argv[1], &lib) != 0))
	{
		printf("v7_load failed\n");
		return 1;
	}
	if (!lib.v7_init(argv[3]))
	{
		if (lib.v7_engine_is_dead())
			printf("v7_init('%s') died: %s\n", argv[3], lib.v7_get_death_reason());
		else
			printf("v7_init('%s') failed\n", argv[3]);
		v7_unload(&lib);
		return 1;
	}
	printf("... fused_validate: %s ...\n", argv[1]);

	/* Read rope_base from microDB -- ASM reads __sys_rope_base, C reference must match.
	 * A missing key is a contract violation -- fail loudly, never a default. */
	float db_rope_base;
	{
		unsigned __int64 len = 0;
		void *data = lib.ht_find("__sys_rope_base", &len);
		if (((data) && (len >= (int)sizeof(float))))
		{
			memcpy(&db_rope_base, data, sizeof(float));
			printf("  DB rope_base = %g\n", db_rope_base);
		}
		else
		{
			printf("[FATAL] __sys_rope_base missing from DB %s\n", argv[3]);
			v7_unload(&lib);
			return 1;
		}
	}

	/* fused_bias_ptr is the BSS global the fused kernel reads its QKV bias from.
	 * Production publishes it before every dispatch; this validator must publish
	 * it too, because the kernel deliberately has NO NULL-bias special case
	 * (Sec.0a forbids a tool-only path in the kernel). The symbol is exported for
	 * exactly this purpose, resolved the same way tune_vit_dispatch.cpp resolves
	 * vit_max_chunks. */
	HMODULE dll_module = GetModuleHandleA(argv[1]);
	float **fused_bias_ptr_slot = (float **)GetProcAddress(dll_module, "fused_bias_ptr");
	if (!fused_bias_ptr_slot)
	{
		printf("[FATAL] fused_bias_ptr export missing from %s (the kernel has no NULL-bias path, so the validator must publish the pointer itself)\n", argv[1]);
		v7_unload(&lib);
		return 1;
	}
	*fused_bias_ptr_slot = 0;

	int pass = 0;
	int total = 0;
	for (int ti = 0; tests[ti].desc; ++ti)
	{
		float max_diff = 0;
		int ok = verify_fused_f32_standalone(lib.v7d_fused_f32, lib.v7s_set_fused_rope, lib.v7s_rope_init, fused_bias_ptr_slot, tests[ti].rows, tests[ti].cols, tests[ti].head_dim, tests[ti].pos, db_rope_base, tests[ti].eps, 12345 + ti, &max_diff);
		if (ok)
		{
			++pass;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("\n  F32 n=%5d d=%5d hd=%d pos=%d base=%.0e eps=%.0e (%s): Fail  max_err=%.3e\n", tests[ti].rows, tests[ti].cols, tests[ti].head_dim, tests[ti].pos, db_rope_base, tests[ti].eps, tests[ti].desc, max_diff);
		}
		++total;
	}

	/* Q8 fused matmul section: resolve the export the same way production
	 * would -- a missing export is a build/link error, never a fallback */
	q8_fused_fn q8f = (q8_fused_fn)GetProcAddress((HMODULE)lib.module, "v7d_q8_fused");
	if (!q8f)
	{
		printf("[FATAL] DLL %s missing v7d_q8_fused export\n", argv[1]);
		v7_unload(&lib);
		return 1;
	}
	for (int ti = 0; q8_tests[ti].desc; ++ti)
	{
		for (int si = 0; si < Q8_N_SEEDS; ++si)
		{
			float max_diff = 0;
			int ok = verify_q8_fused_standalone(q8f, q8_tests[ti].rows, q8_tests[ti].cols, q8_seeds[si], &max_diff);
			if (ok)
			{
				++pass;
				printf(".");
				fflush(stdout);
			}
			else
			{
				printf("\n  Q8  n=%5d d=%5d (%s) seed=%d: Fail  max_err=%.3e\n", q8_tests[ti].rows, q8_tests[ti].cols, q8_tests[ti].desc, q8_seeds[si], max_diff);
			}
			++total;
		}
	}

	printf("\n%d/%d passed\n", pass, total);
	v7_unload(&lib);
	if (pass == total)
		return 0;
	return 1;
}
