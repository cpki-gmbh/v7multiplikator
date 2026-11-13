// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v_matmul_validate.cpp -- Standalone matmul (Q8 + F32) correctness validator
 *
 * Tests v7d_q8 and v7d_f32 (both dispatched, needs V7_init) against C reference.
 * Usage: v_matmul_validate.exe <dll_path> --db <db_path>
 */
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "kt_utils.h"
#include "q8_batch4.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

/* param_set SLOT indices, from the PARAM_OFFSET_* defines in src/asm/v7_data.inc.
 * These are NOT the display indices autotune_dispatch.c uses: param_set[] is
 * indexed directly by slot, while the autotuner bridges its own display
 * order through the sb_param_off_for_idx[] byte-offset table. The two orders
 * differ from slot 1 onward (autotune idx 17 = chunk_split_pct, but chunk_split_pct
 * lives in param_set slot 20). Using the display index here silently writes the
 * wrong slot -- which is exactly what made this test pass against the unfixed
 * dispatcher. Always read the slot from v7_data.inc. */
#define SLOT_CHUNK_SPLIT_PCT    20
#define SLOT_CHUNK_GROUP_1      21
#define SLOT_CHUNK_GROUP_2      22

typedef struct matmul_entry
{
	int rows;
	int cols;
	const char *desc;

} matmul_entry;

static const matmul_entry tests[] = { /* Qwen3-0.6B: dim=1024, hidden=3072, heads 16/8 x hd128, vocab 151936 */
	                     { 1024, 2048, "Qwen3-0.6B Q proj" },
	                     { 1024, 1024, "Qwen3-0.6B K/V proj (kv_dim=1024)" },
	                     { 2048, 1024, "Qwen3-0.6B O proj" },
	                     { 1024, 3072, "Qwen3-0.6B FFN gate/up" },
	                     { 3072, 1024, "Qwen3-0.6B FFN down" },
	                     /* Llama 3.2 1B: dim=2048, hidden=8192, heads 32/8 x hd64, vocab 128256 */
	                     { 2048, 2048, "Llama Q proj" },
	                     { 2048, 512, "Llama K/V proj (kv_dim=512)" },
	                     { 2048, 8192, "Llama FFN gate/up" },
	                     { 8192, 2048, "Llama FFN down" },
	                     /* SmolLM2-135M: dim=576, hidden=1536, heads 9/3 x hd64, vocab 49152 */
	                     { 576, 576, "SmolLM2-135M Q proj" },
	                     { 576, 192, "SmolLM2-135M K/V proj (kv_dim=192)" },
	                     { 576, 1536, "SmolLM2-135M FFN gate/up" },
	                     { 1536, 576, "SmolLM2-135M FFN down" },
	                     /* SmolLM2-360M: dim=960, hidden=2560, heads 15/5 x hd64 */
	                     { 960, 960, "SmolLM2-360M Q proj" },
	                     { 960, 320, "SmolLM2-360M K/V proj (kv_dim=320)" },
	                     { 960, 2560, "SmolLM2-360M FFN gate/up" },
	                     { 2560, 960, "SmolLM2-360M FFN down" },
	                     /* TinyLlama-1.1B: dim=2048, hidden=5632, heads 32/4 x hd64, vocab 32000 */
	                     { 2048, 2048, "TinyLlama Q proj" },
	                     { 2048, 256, "TinyLlama K/V proj (kv_dim=256)" },
	                     { 2048, 5632, "TinyLlama FFN gate/up" },
	                     { 5632, 2048, "TinyLlama FFN down" },
	                     /* Phi-3-mini: dim=3072, hidden=8192, MHA 32 x hd96, vocab 32064 */
	                     { 3072, 3072, "Phi-3 Q proj (MHA)" },
	                     { 3072, 8192, "Phi-3 FFN gate/up" },
	                     { 8192, 3072, "Phi-3 FFN down" },
	                     /* MiniCPM5-1B: dim=1536, hidden=4608, heads 16/2 x hd128, vocab 130560 */
	                     { 1536, 3072, "MiniCPM5 Q proj" },
	                     { 1536, 256, "MiniCPM5 K/V proj (kv_dim=256)" },
	                     { 1536, 4608, "MiniCPM5 FFN gate/up" },
	                     { 4608, 1536, "MiniCPM5 FFN down" },
	                     /* Qwen2.5-0.5B: dim=896, hidden=4864, heads 14/2 x hd64, vocab 151936 */
	                     { 896, 896, "Qwen2.5-0.5B Q proj" },
	                     { 896, 128, "Qwen2.5-0.5B K/V proj (kv_dim=128)" },
	                     { 896, 4864, "Qwen2.5-0.5B FFN gate/up" },
	                     { 4864, 896, "Qwen2.5-0.5B FFN down" },
	                     { 0, 0, 0 } };

/* Regression: split fill with group 2 DISABLED and chunk_split_pct > 0.
 *
 * The boundary adjustment in matmul_dispatch.inc forces the split boundary to
 * d whenever chunk_group_2_count == 0 -- that is the correct result, because a
 * disabled group 2 covers nothing and group 1 must take every column. The
 * partition check therefore has to skip a DISABLED group before demanding a
 * strictly-interior boundary. The old code computed the "g2 active?" test and
 * then never branched on it, so g1-active + g2-disabled + pct>0 reached the
 * boundary==d comparison and died with "no interior boundary for two active
 * groups" on a perfectly legal single-group config. It stayed latent only
 * because every shipped INI section sets chunk_split_pct=0 -- yet both group
 * counts are autotuner-tunable, so a tuning pass proposing pct>0 with
 * chunk_group_2_count=0 lands on a spurious FATAL.
 *
 * The split config is staged DIRECTLY into the per-shape block through the
 * exported shape_blocks symbol -- the same sanctioned mechanism the
 * autotuner uses. The tune API cannot stage from C: v7s_tune_set_shape is a
 * deliberate FATAL stub (the dispatch activates its own shape state) and
 * v7s_tune_set_param only answers inside a Logo verdict callback. With the
 * tuner gate closed (no warmup chat, tune_inited stays 0), every dispatch of
 * the staged shape reads the block as its baseline, so the split fill runs on
 * BOTH the cold re-copy and the warm cached dispatch. Between stagings one
 * dispatch of a DIFFERENT shape invalidates the dispatcher's level-1 shape
 * cache, forcing the next staged-shape dispatch to re-copy the block. */

static unsigned char *g_split_shape_blocks = 0; /* shape_blocks export base, resolved once */

#define SPLIT_SB_SIZE 512              /* bytes per shape block */
#define SPLIT_SB_PARAM_OFF 8           /* param_set starts here, 24 dwords */
#define SPLIT_MAX_BLOCKS 256

/* find_split_params -- param_set base of the (rows, cols) per-shape block, or
 * 0 when the DB carries no such block. Block header: [0] = n, [1] = d. */
static int *find_split_params(int rows, int cols)
{
	for (int bi = 2; bi < SPLIT_MAX_BLOCKS; ++bi)
	{
		int *header = (int *)(g_split_shape_blocks + bi * SPLIT_SB_SIZE);
		if (header[0] == 0)
			break;                     /* end of populated blocks */
		if ((header[0] == rows) && (header[1] == cols))
			return (int *)(g_split_shape_blocks + bi * SPLIT_SB_SIZE + SPLIT_SB_PARAM_OFF);
	}
	return 0;
}

/* split_stage -- write (pct, g1, g2) into the shape's block and save the
 * original triple for split_restore. Returns 0 when no block exists. */
static int split_stage(v7_lib *lib, int rows, int cols, int pct, int g1, int g2, int saved[3])
{
	if (!g_split_shape_blocks)
		g_split_shape_blocks = (unsigned char *)GetProcAddress((HMODULE)lib->module, "shape_blocks");
	if (!g_split_shape_blocks)
		return 0;
	int *params = find_split_params(rows, cols);
	if (!params)
		return 0;
	saved[0] = params[SLOT_CHUNK_SPLIT_PCT];
	saved[1] = params[SLOT_CHUNK_GROUP_1];
	saved[2] = params[SLOT_CHUNK_GROUP_2];
	params[SLOT_CHUNK_SPLIT_PCT] = pct;
	params[SLOT_CHUNK_GROUP_1] = g1;
	params[SLOT_CHUNK_GROUP_2] = g2;
	return 1;
}

/* split_restore -- put the original triple back into the shape's block. */
static void split_restore(v7_lib *lib, int rows, int cols, const int saved[3])
{
	int *params = find_split_params(rows, cols);
	params[SLOT_CHUNK_SPLIT_PCT] = saved[0];
	params[SLOT_CHUNK_GROUP_1] = saved[1];
	params[SLOT_CHUNK_GROUP_2] = saved[2];
	(void)lib;
}

static int verify_split_single_group_config(v7_lib *lib, int rows, int cols, float *out_max_diff)
{
	int saved[3];
	if (!split_stage(lib, rows, cols, 50, 16, 0, saved))
		return 0;
	/* a different shape's dispatch invalidates the level-1 shape cache, so the
	 * first rows x cols dispatch below takes the cold path and re-copies the
	 * staged block; the second is the warm cached dispatch -- both must run
	 * the split geometry */
	float inv_diff = 0;
	kt_verify_f32(lib->v7d_f32, 1024, 3072, 123, &inv_diff);
	int cold_ok = kt_verify_f32(lib->v7d_f32, rows, cols, 123, out_max_diff);
	int warm_ok = kt_verify_f32(lib->v7d_f32, rows, cols, 123, out_max_diff);
	split_restore(lib, rows, cols, saved);
	if (cold_ok && warm_ok)
		return 1;
	return 0;
}

/* Split-geometry sweep on the Q8 path: stage (pct, g1, g2) through the tune
 * path, then verify BOTH the baseline and the candidate dispatch against the
 * C reference. pct=0 makes g1/g2 dead, so restoring just the pct slot
 * neutralizes every case. The streaming validator flagged pct=1 on the
 * gate/up shape (1024x3072) as producing a wrong first generated token --
 * these cases reproduce that geometry directly against the reference. */
typedef struct split_case
{
	int pct;
	int g1;
	int g2;
	const char *desc;

} split_case;

static const split_case split_cases[] = {
	                    { 1,  16, 128, "the promoted poison: pct=1 g1=16 g2=128" },
	                    { 1,  0,  128, "g2-only: pct=1 g1=0 g2=128" },
	                    { 1,  16, 0,   "g1-only: pct=1 g1=16 g2=0" },
	                    { 50, 16, 128, "half split: pct=50 g1=16 g2=128" },
	                    { 99, 16, 128, "tiny g2 tail: pct=99 g1=16 g2=128" },
	                    { 1,  30, 128, "g1_count == g1_cols: pct=1 g1=30 g2=128" },
	                    { 1,  31, 128, "g1 clamp path: pct=1 g1=31 g2=128" },
	                    { 0,  0,  0,   0 } };

static int verify_split_geometry_sweep(v7_lib *lib, int rows, int cols, int *out_failed)
{
	int failures = 0;
	for (int ci = 0; split_cases[ci].desc; ++ci)
	{
		int saved[3];
		if (!split_stage(lib, rows, cols, split_cases[ci].pct, split_cases[ci].g1, split_cases[ci].g2, saved))
		{
			++failures;
			printf("\n    no per-shape block for %dx%d, cannot stage sweep case %d\n", rows, cols, ci);
			continue;
		}
		printf("    sweep case %d staged: pct=%d g1=%d g2=%d\n",
		       ci, split_cases[ci].pct, split_cases[ci].g1, split_cases[ci].g2);
		/* a different shape's dispatch invalidates the level-1 shape cache:
		 * the next two rows x cols dispatches run cold (block re-copy) and
		 * warm (cached block 0), both on the staged split geometry */
		float inv_diff = 0;
		kt_verify_q8(lib->v7d_q8, 1024, 2048, 123, &inv_diff);
		float max_diff = 0;
		int cold_ok = kt_verify_q8(lib->v7d_q8, rows, cols, 123, &max_diff);
		int warm_ok = kt_verify_q8(lib->v7d_q8, rows, cols, 123, &max_diff);
		if (cold_ok && warm_ok)
		{
			printf(".");
			fflush(stdout);
		}
		else
		{
			++failures;
			printf("\n    Q8 split n=%d d=%d (%s): cold=%d warm=%d max_err=%.3e\n",
			       rows, cols, split_cases[ci].desc, cold_ok, warm_ok, max_diff);
		}
		split_restore(lib, rows, cols, saved);
	}
	/* restore check: the untouched original config, freshly re-copied */
	float inv_diff = 0;
	kt_verify_q8(lib->v7d_q8, 1024, 2048, 123, &inv_diff);
	float restore_diff = 0;
	int restore_ok = kt_verify_q8(lib->v7d_q8, rows, cols, 123, &restore_diff);
	if (!restore_ok)
	{
		++failures;
		printf("\n    Q8 split restore (original config): Fail max_err=%.3e\n", restore_diff);
	}
	*out_failed = failures;
	return failures == 0;
}

/* ======================================================================
 * Experimental four-token Q8 batch kernel (speed_fix plan Phase 1)
 *
 * v7s_q8_batch4 computes four independent Q8 GEMVs against ONE shared
 * weight matrix. The reference side is four separate calls to the actual
 * dispatched v7d_q8 on the same weight tensor and four different inputs.
 * Tolerance: the same abs/rel pair kt_verify_q8 uses (0.01 / 1e-4).
 * ====================================================================== */

static void v7_batch4_fill(v7_q8_batch4_args *args_block,
                           float *const outputs[4],
                           const v7_q8_tensor inputs[4],
                           const v7_q8_tensor *weights,
                           int n, int d)
{
	memset(args_block, 0, sizeof(*args_block));
	for (int t = 0; t < 4; ++t)
	{
		args_block->out[t] = outputs[t];
		args_block->xq[t] = (const int8_t *)inputs[t].q;
		args_block->xs[t] = inputs[t].s;
	}
	args_block->wq = (const int8_t *)weights->q;
	args_block->ws = weights->s;
	args_block->n = (uint32_t)n;
	args_block->d = (uint32_t)d;
}

/* The engine keeps the Windows register convention on every platform, so a
   host that calls into it declares the entry points with the ms_abi attribute.
   The C reference kernel is ordinary host code with the native convention, so
   it needs a matching shim before it can be handed over as a dispatch-shaped
   callback. */
static void V7_ABI q8_reference_msabi(float *xout, v7_q8_tensor *x, v7_q8_tensor *w, int n, int d)
{
	kt_matmul_c_ref_q8(xout, x, w, n, d);
}

static int v7_batch4_check(v7d_q8_fn baseline_fn,
                           v7_q8_batch4_fn batch_fn,
                           const v7_q8_tensor inputs[4],
                           const v7_q8_tensor *weights,
                           int n, int d,
                           float *const expected[4],
                           float *const actual[4],
                           float atol, float rtol)
{
	v7_q8_batch4_args args_block;

	for (int t = 0; t < 4; ++t)
	{
		v7_q8_tensor x_t = inputs[t];
		v7_q8_tensor w_t = *weights;
		baseline_fn(expected[t], &x_t, &w_t, n, d);
	}

	v7_batch4_fill(&args_block, actual, inputs, weights, n, d);
	batch_fn(&args_block);

	for (int t = 0; t < 4; ++t)
	{
		for (int r = 0; r < d; ++r)
		{
			float a = actual[t][r];
			float b = expected[t][r];
			if ((!std::isfinite(a)) || (!std::isfinite(b)) || (fabsf(a - b) > atol + rtol * fabsf(b)))
			{
				printf("batch4 mismatch token=%d row=%d actual=%g ref=%g\n", t, r, a, b);
				return 0;
			}
		}
	}
	return 1;
}

static v7_q8_batch4_args v7_batch4_row_slice(const v7_q8_batch4_args *whole, uint32_t first, uint32_t count)
{
	v7_q8_batch4_args part = *whole;
	for (int t = 0; t < 4; ++t)
		part.out[t] += first;
	part.wq += (size_t)first * whole->n;
	part.ws += (size_t)first * (whole->n / 64);
	part.d = count;
	return part;
}

/* Dispatched correctness: run the WHOLE matrix through v7d_q8_batch4 (the
 * row-sliced dispatch) and compare against the C reference for all four
 * tokens. This is the case that exercises the job plumbing: the BSS args
 * publish, the contiguous row ranges, and process_job's slice builder. */
static int v7_batch4_case_dispatched(v7_lib *lib, int n, int d, int seed, const char *label)
{
	char *xq[4];
	float *xs[4];
	char *wq = (char *)malloc((size_t)d * n);
	float *ws = (float *)malloc(((size_t)d * n / 64 + 1) * sizeof(float));
	float *expected[4];
	float *actual[4];
	v7_q8_tensor inputs[4];
	v7_q8_tensor weights;
	v7_q8_batch4_args args_block;
	v7d_q8_batch4_fn dispatch_fn;
	int ok = 1;

	dispatch_fn = (v7d_q8_batch4_fn)GetProcAddress((HMODULE)lib->module, "v7d_q8_batch4");
	if (!dispatch_fn)
	{
		printf("v7d_q8_batch4 export missing\n");
		return 0;
	}

	for (int t = 0; t < 4; ++t)
	{
		xq[t] = (char *)malloc(n);
		xs[t] = (float *)malloc((n / 64 + 1) * sizeof(float));
		expected[t] = (float *)malloc(d * sizeof(float));
		actual[t] = (float *)malloc(d * sizeof(float));
		if ((!xq[t]) || (!xs[t]) || (!expected[t]) || (!actual[t]))
		{
			printf("batch4 dispatched case '%s': allocation failed\n", label);
			ok = 0;
		}
	}
	if ((!wq) || (!ws))
	{
		printf("batch4 dispatched case '%s': weight allocation failed\n", label);
		ok = 0;
	}
	if (!ok)
	{
		for (int t = 0; t < 4; ++t)
		{
			free(xq[t]);
			free(xs[t]);
			free(expected[t]);
			free(actual[t]);
		}
		free(wq);
		free(ws);
		return 0;
	}

	srand(seed);
	for (int t = 0; t < 4; ++t)
	{
		kt_init_random_int8(xq[t], n);
		kt_init_random_float(xs[t], n / 64 + 1);
		inputs[t].q = xq[t];
		inputs[t].s = xs[t];
	}
	kt_init_random_int8(wq, (size_t)d * n);
	kt_init_random_float(ws, (int)((size_t)d * n / 64) + 1);
	weights.q = wq;
	weights.s = ws;

	for (int t = 0; t < 4; ++t)
	{
		v7_q8_tensor x_t = inputs[t];
		v7_q8_tensor w_t = weights;
		kt_matmul_c_ref_q8(expected[t], &x_t, &w_t, n, d);
	}

	v7_batch4_fill(&args_block, actual, inputs, &weights, n, d);
	memset(actual[0], 0, d * sizeof(float));
	memset(actual[1], 0, d * sizeof(float));
	memset(actual[2], 0, d * sizeof(float));
	memset(actual[3], 0, d * sizeof(float));
	dispatch_fn(&args_block);

	for (int t = 0; t < 4; ++t)
	{
		for (int r = 0; r < d; ++r)
		{
			float a = actual[t][r];
			float b = expected[t][r];
			if ((!std::isfinite(a)) || (!std::isfinite(b)) || (fabsf(a - b) > 0.01f + 1e-4f * fabsf(b)))
			{
				printf("batch4 dispatched mismatch token=%d row=%d actual=%g ref=%g\n", t, r, a, b);
				ok = 0;
			}
		}
	}

	for (int t = 0; t < 4; ++t)
	{
		free(xq[t]);
		free(xs[t]);
		free(expected[t]);
		free(actual[t]);
	}
	free(wq);
	free(ws);
	return ok;
}

/* One test case: four generated activations + one weight matrix, batch4
 * compared element-wise against four independent v7d_q8 baselines. */
/* Correctness baseline = the C reference kernel (plan 1.4: "four independent
 * baseline matrix multiplications"). The DISPATCHED v7d_q8 is NOT usable here
 * for arbitrary shapes: every dispatched shape requires its own per-shape INI
 * config block and FATALs otherwise (the dispatch contract), and the battery
 * below deliberately includes shapes (64x64, 128x64, d=3, d=7) that no INI
 * section defines. The tuned dispatched kernel enters the comparison in the
 * benchmark below (plan 1.5), whose shapes all have blocks. */
static int v7_batch4_case(v7_lib *lib, int n, int d, int seed, const char *label)
{
	/* four independent quantized activations (different seeds so every token
	   has different data AND different scales) + one shared weight tensor */
	char *xq[4];
	float *xs[4];
	char *wq = (char *)malloc((size_t)d * n);
	float *ws = (float *)malloc(((size_t)d * n / 64 + 1) * sizeof(float));
	float *expected[4];
	float *actual[4];
	v7_q8_tensor inputs[4];
	v7_q8_tensor weights;
	v7_q8_batch4_fn batch_fn;
	int ok = 1;

	batch_fn = (v7_q8_batch4_fn)GetProcAddress((HMODULE)lib->module, "v7s_q8_batch4");
	if (!batch_fn)
	{
		printf("v7s_q8_batch4 export missing\n");
		return 0;
	}

	for (int t = 0; t < 4; ++t)
	{
		xq[t] = (char *)malloc(n);
		xs[t] = (float *)malloc((n / 64 + 1) * sizeof(float));
		expected[t] = (float *)malloc(d * sizeof(float));
		actual[t] = (float *)malloc(d * sizeof(float));
		if ((!xq[t]) || (!xs[t]) || (!expected[t]) || (!actual[t]))
		{
			printf("batch4 case '%s': allocation failed\n", label);
			ok = 0;
		}
	}
	if ((!wq) || (!ws))
	{
		printf("batch4 case '%s': weight allocation failed\n", label);
		ok = 0;
	}
	if (!ok)
	{
		for (int t = 0; t < 4; ++t)
		{
			free(xq[t]);
			free(xs[t]);
			free(expected[t]);
			free(actual[t]);
		}
		free(wq);
		free(ws);
		return 0;
	}

	srand(seed);
	for (int t = 0; t < 4; ++t)
	{
		kt_init_random_int8(xq[t], n);
		kt_init_random_float(xs[t], n / 64 + 1);
		inputs[t].q = xq[t];
		inputs[t].s = xs[t];
	}
	kt_init_random_int8(wq, (size_t)d * n);
	kt_init_random_float(ws, (int)((size_t)d * n / 64) + 1);
	weights.q = wq;
	weights.s = ws;

	memset(actual[0], 0, d * sizeof(float));
	memset(actual[1], 0, d * sizeof(float));
	memset(actual[2], 0, d * sizeof(float));
	memset(actual[3], 0, d * sizeof(float));
	ok = v7_batch4_check(q8_reference_msabi, batch_fn, inputs, &weights, n, d, expected, actual, 0.01f, 1e-4f);

	for (int t = 0; t < 4; ++t)
	{
		free(xq[t]);
		free(xs[t]);
		free(expected[t]);
		free(actual[t]);
	}
	free(wq);
	free(ws);
	return ok;
}

/* Row-slice test: run the whole matrix, then a sliced window starting at a
 * nonzero row; the slice's outputs must equal the full run's rows. */
static int v7_batch4_case_slice(v7_lib *lib, int n, int d, int seed, const char *label)
{
	char *xq[4];
	float *xs[4];
	char *wq = (char *)malloc((size_t)d * n);
	float *ws = (float *)malloc(((size_t)d * n / 64 + 1) * sizeof(float));
	float *full_out[4];
	float *slice_out[4];
	v7_q8_tensor inputs[4];
	v7_q8_tensor weights;
	v7_q8_batch4_args whole_args;
	v7_q8_batch4_args part_args;
	v7_q8_batch4_fn batch_fn;
	uint32_t first = 2;
	uint32_t count = (uint32_t)d - 2;
	int ok = 1;

	batch_fn = (v7_q8_batch4_fn)GetProcAddress((HMODULE)lib->module, "v7s_q8_batch4");
	if (!batch_fn)
	{
		printf("v7s_q8_batch4 export missing\n");
		return 0;
	}

	for (int t = 0; t < 4; ++t)
	{
		xq[t] = (char *)malloc(n);
		xs[t] = (float *)malloc((n / 64 + 1) * sizeof(float));
		full_out[t] = (float *)malloc(d * sizeof(float));
		slice_out[t] = (float *)malloc((size_t)count * sizeof(float));
		if ((!xq[t]) || (!xs[t]) || (!full_out[t]) || (!slice_out[t]))
		{
			printf("batch4 slice case '%s': allocation failed\n", label);
			ok = 0;
		}
	}
	if ((!wq) || (!ws))
	{
		printf("batch4 slice case '%s': weight allocation failed\n", label);
		ok = 0;
	}
	if (!ok)
	{
		for (int t = 0; t < 4; ++t)
		{
			free(xq[t]);
			free(xs[t]);
			free(full_out[t]);
			free(slice_out[t]);
		}
		free(wq);
		free(ws);
		return 0;
	}

	srand(seed);
	for (int t = 0; t < 4; ++t)
	{
		kt_init_random_int8(xq[t], n);
		kt_init_random_float(xs[t], n / 64 + 1);
		inputs[t].q = xq[t];
		inputs[t].s = xs[t];
	}
	kt_init_random_int8(wq, (size_t)d * n);
	kt_init_random_float(ws, (int)((size_t)d * n / 64) + 1);
	weights.q = wq;
	weights.s = ws;

	for (int t = 0; t < 4; ++t)
		memset(slice_out[t], 0xEE, (size_t)count * sizeof(float));

	v7_batch4_fill(&whole_args, full_out, inputs, &weights, n, d);
	batch_fn(&whole_args);
	part_args = v7_batch4_row_slice(&whole_args, first, count);
	/* plan semantics: row_slice advances the out pointers INTO the same array
	 * (parallel jobs write disjoint rows of one shared output). This test
	 * retargets them to dedicated buffers so the sliced result is compared
	 * standalone; the wq/ws advancement -- the part the kernel does per row
	 * from the slice base -- is exercised identically either way. */
	for (int t = 0; t < 4; ++t)
		part_args.out[t] = slice_out[t];
	batch_fn(&part_args);

	int slice_printed = 0;
	for (int t = 0; t < 4; ++t)
	{
		for (uint32_t r = 0; r < count; ++r)
		{
			float a = slice_out[t][r];
			float b = full_out[t][first + r];
			if (fabsf(a - b) > 1e-6f * (1.0f + fabsf(b)))
			{
				if (slice_printed < 8)
				{
					printf("batch4 slice mismatch token=%d row=%u actual=%g full=%g\n", t, first + r, a, b);
					slice_printed++;
				}
				ok = 0;
			}
		}
	}

	for (int t = 0; t < 4; ++t)
	{
		free(xq[t]);
		free(xs[t]);
		free(full_out[t]);
		free(slice_out[t]);
	}
	free(wq);
	free(ws);
	return ok;
}

/* Edge-value battery: zeros, alternating signs, extreme +-127 magnitudes,
 * per-group distinct scales, d not divisible by four. */
static int v7_batch4_case_edge(v7_lib *lib, int n, int d, int pattern, const char *label)
{
	char *xq[4];
	float *xs[4];
	char *wq = (char *)malloc((size_t)d * n);
	float *ws = (float *)malloc(((size_t)d * n / 64 + 1) * sizeof(float));
	float *expected[4];
	float *actual[4];
	v7_q8_tensor inputs[4];
	v7_q8_tensor weights;
	v7_q8_batch4_args args_block;
	v7_q8_batch4_fn batch_fn;
	int ok = 1;

	batch_fn = (v7_q8_batch4_fn)GetProcAddress((HMODULE)lib->module, "v7s_q8_batch4");
	if (!batch_fn)
	{
		printf("v7s_q8_batch4 export missing\n");
		return 0;
	}

	for (int t = 0; t < 4; ++t)
	{
		xq[t] = (char *)malloc(n);
		xs[t] = (float *)malloc((n / 64 + 1) * sizeof(float));
		expected[t] = (float *)malloc(d * sizeof(float));
		actual[t] = (float *)malloc(d * sizeof(float));
		if ((!xq[t]) || (!xs[t]) || (!expected[t]) || (!actual[t]))
		{
			printf("batch4 edge case '%s': allocation failed\n", label);
			ok = 0;
		}
	}
	if ((!wq) || (!ws))
	{
		printf("batch4 edge case '%s': weight allocation failed\n", label);
		ok = 0;
	}
	if (!ok)
	{
		for (int t = 0; t < 4; ++t)
		{
			free(xq[t]);
			free(xs[t]);
			free(expected[t]);
			free(actual[t]);
		}
		free(wq);
		free(ws);
		return 0;
	}

	srand(777);
	for (int t = 0; t < 4; ++t)
	{
		for (int i = 0; i < n; ++i)
		{
			if (pattern == 0)
				xq[t][i] = 0; /* all zeros */
			else if (pattern == 1)
				xq[t][i] = (char)((i % 2 == 0) ? 1 : -1); /* alternating signs */
			else
				xq[t][i] = (char)(127 - (i + t * 7) % 255); /* near-limit magnitudes incl. -128 territory clamped by generation */
		}
		for (int g = 0; g < n / 64 + 1; ++g)
			xs[t][g] = (pattern == 2) ? (0.001f + 0.01f * g) : 1.0f; /* distinct per-group scales in the extreme case */
		inputs[t].q = xq[t];
		inputs[t].s = xs[t];
	}
	for (int i = 0; i < (size_t)d * n; ++i)
		wq[i] = (char)(127 - i % 255); /* weights near the supported limits */
	for (int g = 0; g < (int)((size_t)d * n / 64) + 1; ++g)
		ws[g] = 0.002f + 0.005f * (g % 13);
	weights.q = wq;
	weights.s = ws;

	memset(actual[0], 0, d * sizeof(float));
	memset(actual[1], 0, d * sizeof(float));
	memset(actual[2], 0, d * sizeof(float));
	memset(actual[3], 0, d * sizeof(float));
	ok = v7_batch4_check(q8_reference_msabi, batch_fn, inputs, &weights, n, d, expected, actual, 0.01f, 1e-4f);

	for (int t = 0; t < 4; ++t)
	{
		free(xq[t]);
		free(xs[t]);
		free(expected[t]);
		free(actual[t]);
	}
	free(wq);
	free(ws);
	return ok;
}

/* Kernel-level benchmark per plan 1.5 + reviewer guidance: the comparison is
 * WHOLE DISPATCH vs WHOLE DISPATCH — baseline = four dispatched v7d_q8 calls
 * (each with publication + sync + completion), candidate = ONE dispatched
 * v7d_q8_batch4 call (row-sliced across the pool). Adjacent
 * baseline-candidate-baseline timing, minimum per side. Real Qwen3-0.6B text
 * shapes: packed QKV n=1024 d=4096, WO n=2048 d=1024, W2 n=3072 d=1024. */
static void v7_batch4_benchmark(v7_lib *lib)
{
	typedef struct bench_shape
	{
		int n;
		int d;
		const char *label;

	} bench_shape;

	bench_shape shapes[] = {
		{ 1024, 4096, "QKV 1024x4096" },
		{ 2048, 1024, "WO 2048x1024" },
		{ 3072, 1024, "W2 3072x1024" },
	};
	v7d_q8_batch4_fn dispatch_fn;
	int iters = 100;

	dispatch_fn = (v7d_q8_batch4_fn)GetProcAddress((HMODULE)lib->module, "v7d_q8_batch4");
	if (!dispatch_fn)
	{
		printf("v7d_q8_batch4 export missing, no benchmark\n");
		return;
	}

	printf("  [batch4 bench] 1x dispatched batch4 vs 4x dispatched v7d_q8 (set_spin 1)\n");
	/* dispatched benchmarks must run with spinning workers (Sec.88) */
	lib->v7s_set_spin(1);

	for (int si = 0; si < 3; ++si)
	{
		int n = shapes[si].n;
		int d = shapes[si].d;
		char *xq[4];
		float *xs[4];
		char *wq = (char *)malloc((size_t)d * n);
		float *ws = (float *)malloc(((size_t)d * n / 64 + 1) * sizeof(float));
		float *out_base[4];
		float *out_batch[4];
		v7_q8_tensor inputs[4];
		v7_q8_tensor weights;
		v7_q8_batch4_args args_block;
		__int64 best_base = 0;
		__int64 best_batch = 0;
		LARGE_INTEGER freq;

		QueryPerformanceFrequency(&freq);
		for (int t = 0; t < 4; ++t)
		{
			xq[t] = (char *)malloc(n);
			xs[t] = (float *)malloc((n / 64 + 1) * sizeof(float));
			out_base[t] = (float *)malloc(d * sizeof(float));
			out_batch[t] = (float *)malloc(d * sizeof(float));
		}

		srand(500 + si);
		for (int t = 0; t < 4; ++t)
		{
			kt_init_random_int8(xq[t], n);
			kt_init_random_float(xs[t], n / 64 + 1);
			inputs[t].q = xq[t];
			inputs[t].s = xs[t];
		}
		kt_init_random_int8(wq, (size_t)d * n);
		kt_init_random_float(ws, (int)((size_t)d * n / 64) + 1);
		weights.q = wq;
		weights.s = ws;
		v7_batch4_fill(&args_block, out_batch, inputs, &weights, n, d);

		/* warm-up both sides, full dispatch paths */
		for (int t = 0; t < 4; ++t)
		{
			v7_q8_tensor x_t = inputs[t];
			v7_q8_tensor w_t = weights;
			lib->v7d_q8(out_base[t], &x_t, &w_t, n, d);
		}
		dispatch_fn(&args_block);

		/* adjacent baseline-candidate-baseline-candidate-baseline, min per side;
		 * BOTH sides are whole dispatched operations (publish + sync + complete) */
		best_base = 0x7FFFFFFFFFFFFFFF;
		best_batch = 0x7FFFFFFFFFFFFFFF;
		for (int round = 0; round < 5; ++round)
		{
			LARGE_INTEGER t0;
			LARGE_INTEGER t1;
			__int64 us;

			QueryPerformanceCounter(&t0);
			for (int i = 0; i < iters; ++i)
			{
				for (int t = 0; t < 4; ++t)
				{
					v7_q8_tensor x_t = inputs[t];
					v7_q8_tensor w_t = weights;
					lib->v7d_q8(out_base[t], &x_t, &w_t, n, d);
				}
			}
			QueryPerformanceCounter(&t1);
			us = (t1.QuadPart - t0.QuadPart) * 1000000 / freq.QuadPart / iters;
			if (us < best_base)
				best_base = us;

			QueryPerformanceCounter(&t0);
			for (int i = 0; i < iters; ++i)
				dispatch_fn(&args_block);
			QueryPerformanceCounter(&t1);
			us = (t1.QuadPart - t0.QuadPart) * 1000000 / freq.QuadPart / iters;
			if (us < best_batch)
				best_batch = us;
		}
		printf("    %-14s baseline 4x v7d_q8 = %4lld us   dispatched batch4 = %4lld us   ratio %.2fx\n",
		       shapes[si].label, (long long)best_base, (long long)best_batch,
		       best_batch ? (double)best_base / (double)best_batch : 0.0);

		for (int t = 0; t < 4; ++t)
		{
			free(xq[t]);
			free(xs[t]);
			free(out_base[t]);
			free(out_batch[t]);
		}
		free(wq);
		free(ws);
	}
}

static void test_q8_batch4(v7_lib *lib, int *out_passed, int *out_total)
{
	int passed = 0;
	int total = 0;

	printf("  [batch4] correctness vs 4x v7d_q8\n");

	/* plan 1.4 battery: mixed signs, zeros, distinct group scales,
	 * near-limit magnitudes, d not divisible by four, row slice */
	struct
	{
		int n;
		int d;
		const char *label;

	} cases[] = {
		{ 64, 64, "one group 64x64" },
		{ 128, 64, "128x64" },
		{ 1024, 1024, "1024x1024" },
		{ 1024, 3, "d=3 (not divisible by four)" },
		{ 1024, 7, "d=7" },
		{ 2048, 1024, "WO 2048x1024" },
		{ 3072, 1024, "W2 3072x1024" },
		{ 1024, 4096, "packed QKV 1024x4096" },
	};

	for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ++ci)
	{
		int ok = v7_batch4_case(lib, cases[ci].n, cases[ci].d, 900 + ci, cases[ci].label);
		if (ok)
		{
			++passed;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("\n    batch4 n=%d d=%d (%s): FAIL\n", cases[ci].n, cases[ci].d, cases[ci].label);
		}
		++total;
	}

	/* edge battery: pattern 0 zeros, 1 alternating, 2 near-limit + distinct scales */
	for (int pattern = 0; pattern < 3; ++pattern)
	{
		int ok = v7_batch4_case_edge(lib, 256, 96, pattern, "edge battery");
		if (ok)
		{
			++passed;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("\n    batch4 edge pattern %d: FAIL\n", pattern);
		}
		++total;
	}

	/* row slice with a nonzero starting row */
	{
		int ok = v7_batch4_case_slice(lib, 1024, 4096, 4242, "row slice first=2");
		if (ok)
		{
			++passed;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("\n    batch4 row slice: FAIL\n");
		}
		++total;
	}

	/* whole matrix through the DISPATCHED entry: job plumbing, BSS args
	 * publish, contiguous row ranges, process_job slice builder */
	{
		int ok = v7_batch4_case_dispatched(lib, 1024, 4096, 555, "dispatched QKV whole");
		if (ok)
		{
			++passed;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("\n    batch4 dispatched QKV: FAIL\n");
		}
		++total;
	}

	printf("\n  [batch4] correctness: %d/%d passed\n", passed, total);
	*out_passed = passed;
	*out_total = total;

	v7_batch4_benchmark(lib);
}

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		printf("usage: v_matmul_validate.exe <dll_path> --db <db_path>\n");
		return 1;
	}
	if ((strcmp(argv[2], "--db") != 0))
	{
		printf("usage: v_matmul_validate.exe <dll_path> --db <db_path>\n");
		return 1;
	}
	ZydisInit();
	setvbuf(stdout, 0, _IONBF, 0);       // unbuffered stdout: test progress survives FATAL aborts

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
	printf("... matmul_validate: %s ...\n", argv[1]);

	int q8_pass = 0;
	int q8_total = 0;
	int f32_pass = 0;
	int f32_total = 0;

	printf("  [matmul Q8] testing\n");
	for (int ti = 0; tests[ti].desc; ++ti)
	{
		float max_diff = 0;
		int ok = kt_verify_q8(lib.v7d_q8, tests[ti].rows, tests[ti].cols, 123, &max_diff);
		if (ok)
		{
			++q8_pass;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("\n    n=%5d d=%5d (%s): Fail  max_err=%.3e\n", tests[ti].rows, tests[ti].cols, tests[ti].desc, max_diff);
		}
		++q8_total;
	}
	printf("  [matmul Q8] result: %d/%d passed\n", q8_pass, q8_total);

	printf("  [matmul F32] testing\n");
	for (int ti = 0; tests[ti].desc; ++ti)
	{
		float max_diff = 0;
		int ok = kt_verify_f32(lib.v7d_f32, tests[ti].rows, tests[ti].cols, 123, &max_diff);
		if (ok)
		{
			++f32_pass;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("\n    n=%5d d=%5d (%s): Fail  max_err=%.3e\n", tests[ti].rows, tests[ti].cols, tests[ti].desc, max_diff);
		}
		++f32_total;
	}
	printf("  [matmul F32] result: %d/%d passed\n", f32_pass, f32_total);

	/* Experimental four-token Q8 batch kernel: correctness vs 4x v7d_q8 and
	   a kernel-level benchmark. Does not touch the tuner or the dispatch
	   config, so it runs before the split-geometry tests. */
	int batch4_pass = 0;
	int batch4_total = 0;
	test_q8_batch4(&lib, &batch4_pass, &batch4_total);

	/* Runs LAST: it stages split configs into real per-shape blocks, so it
	 * must not perturb the Q8/F32 sweeps above. 1024x2048 and 1024x3072 are
	 * real per-shape blocks in threading_proxy.ini; the staging writes the
	 * shape's own block via the shape_blocks export and restores it after. */
	float split_max_diff = 0;
	int split_ok = verify_split_single_group_config(&lib, 1024, 2048, &split_max_diff);
	if (split_ok)
	{
		printf(".");
		fflush(stdout);
	}
	else
	{
		printf("\n    n=  1024 d=  2048 (split fill, g1=16 g2=0 pct=50): Fail  max_err=%.3e\n", split_max_diff);
	}

	/* split-geometry sweep on the gate/up shape -- the geometry the streaming
	   validator flagged; runs after everything else for the same reason */
	int sweep_failed = 0;
	int sweep_ok = verify_split_geometry_sweep(&lib, 1024, 3072, &sweep_failed);

	int pass = q8_pass + f32_pass + batch4_pass;
	int total = q8_total + f32_total + batch4_total;
	if (split_ok)
	{
		++pass;
	}
	++total;
	if (sweep_ok)
	{
		printf(".");
		++pass;
	}
	++total;
	printf("\n%d/%d passed\n", pass, total);
	v7_unload(&lib);
	if (pass == total)
		return 0;
	return 1;
}
