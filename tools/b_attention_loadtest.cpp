// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_attention_loadtest.cpp -- Standalone attention kernel loadtest / speed probe
 *
 * Exercises v7d_attention. Without options it cycles positions 0..max_pos to
 * detect crashes. With --pos N it drives every call at a FIXED position N so the
 * attention timeline length is realistic and the reported us/call and GB/s are a
 * steady-state decode measurement at that context length.
 *
 * Usage: b_attention_loadtest.exe <dll_path> --db <db_path> [--pos N] [--iters N] [--maxpos N]
 *   --pos N     fixed position (decode context length) to measure; settable, default cycles 0..maxpos
 *   --maxpos N  cycling upper bound when --pos is not given; default 31
 *   --iters N   iterations; default 10000
 */
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "runstate_alloc.h"
#include "kt_utils.h"
#include "attention_shared.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv)
{
	char *dll_path = 0;
	char *db_path = 0;
	int fixed_pos = -1;
	int max_pos = 31;
	int iters = 10000;

	for (int i = 1; i < argc; i++)
	{
		if ((strcmp(argv[i], "--db") == 0) && ((i + 1) < argc))
			db_path = argv[++i];
		else if ((strcmp(argv[i], "--pos") == 0) && ((i + 1) < argc))
			fixed_pos = atoi(argv[++i]);
		else if ((strcmp(argv[i], "--maxpos") == 0) && ((i + 1) < argc))
			max_pos = atoi(argv[++i]);
		else if ((strcmp(argv[i], "--iters") == 0) && ((i + 1) < argc))
			iters = atoi(argv[++i]);
		else
			dll_path = argv[i];
	}

	if (dll_path == 0)
	{
		printf("usage: b_attention_loadtest.exe <dll_path> --db <db_path> [--pos N] [--iters N] [--maxpos N]\n");
		return 1;
	}
	if (db_path == 0)
	{
		printf("usage: b_attention_loadtest.exe <dll_path> --db <db_path> [--pos N] [--iters N] [--maxpos N]\n");
		printf("Fail: --db <db_path> is required (no default database path)\n");
		return 1;
	}
	ZydisInit();

	attn_dll dll;
	if (!attn_dll_load(&dll, dll_path, db_path))
		return 1;

	v7_config model_config;
	model_config.n_heads = 16;
	model_config.n_kv_heads = 8;
	model_config.head_dim = 128;
	model_config.dim = model_config.n_heads * model_config.head_dim;
	model_config.hidden_dim = model_config.dim * 4;
	model_config.vocab_size = 32000;
	int kv_dim = model_config.n_kv_heads * model_config.head_dim;
	int kv_mul = model_config.n_heads / model_config.n_kv_heads;
	unsigned __int64 loff = 0;

	/* Cache and attention span cover the fixed position or the cycling bound. */
	int span = (fixed_pos >= 0) ? fixed_pos : max_pos;
	int max_seq_len = span + 1;
	model_config.seq_len = max_seq_len;

	v7_run_state *state_pointer = rs_alloc(&model_config, max_seq_len, 0);
	if (!state_pointer)
	{
		printf("OOM\n");
		attn_dll_unload(&dll);
		return 1;
	}

	srand(12345);
	attn_fill_random(state_pointer, &model_config, max_seq_len, 0);

	if (fixed_pos >= 0)
		printf("%d iterations at fixed pos=%d, head_dim=%d, n_heads=%d, n_kv_heads=%d\n", iters, fixed_pos, model_config.head_dim, model_config.n_heads, model_config.n_kv_heads);
	else
		printf("%d iterations cycling pos 0..%d, head_dim=%d, n_heads=%d, n_kv_heads=%d\n", iters, max_pos, model_config.head_dim, model_config.n_heads, model_config.n_kv_heads);

	/* Optional GQA grouping variant flag is left off; v7d_attention is the production entry. */
	cpu_snapshot cpu_start;
	double t0 = kt_get_seconds();
	for (int i = 0; i < iters; i++)
	{
		int pos = (fixed_pos >= 0) ? fixed_pos : (i % (max_pos + 1));
		dll.lib.v7d_attention(state_pointer, &model_config, pos, loff, kv_dim, kv_mul);
	}
	double t1 = kt_get_seconds();
	cpu_snapshot cpu_end;

	double elapsed = t1 - t0;
	double us_per_call = elapsed * 1e6 / (double)iters;

	/* Realistic bytes moved at the measured span: Q read + K/V read (GQA shared) + xb write. */
	int timeline = span + 1;
	int bytes_per_call = model_config.dim * (int)sizeof(float)                                  /* Q read */
	                   + timeline * kv_dim * (int)sizeof(uint16_t) * 2                           /* K+V read, packed FP16 */
	                   + model_config.dim * (int)sizeof(float);                                  /* att out write */
	double gbs = (double)bytes_per_call * (double)iters / elapsed / 1e9;
	double cpu_pct = kt_cpu_delta_pct(cpu_start, cpu_end);

	printf("Pass: No crashes detected\n");
	printf("  elapsed: %.3f s  %.3f us/call  %.1f GB/s  pos=%d\n", elapsed, us_per_call, gbs, span);
	printf("  total cpu: %.1f%% over %d iterations\n", cpu_pct, iters);

	rs_free(state_pointer);
	attn_dll_unload(&dll);
	return 0;
}
