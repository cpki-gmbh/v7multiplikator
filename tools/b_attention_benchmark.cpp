// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_attention_benchmark.cpp -- Standalone attention kernel speed benchmark
 *
 * Measures v7d_attention throughput (dispatched via scheduler).
 * Usage: b_attention_benchmark.exe <dll_path> --db <db_path>
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
	if (argc < 4)
	{
		printf("usage: b_attention_benchmark.exe <dll_path> --db <db_path>\n");
		return 1;
	}
	if ((strcmp(argv[2], "--db") != 0))
	{
		printf("usage: b_attention_benchmark.exe <dll_path> --db <db_path>\n");
		return 1;
	}
	ZydisInit();

	attn_dll dll;
	if (!attn_dll_load(&dll, argv[1], argv[3]))
		return 1;

	/* Reasonable model config: 16 heads, 8 kv heads, head_dim=128 */
	v7_config model_config;
	model_config.n_heads = 16;
	model_config.n_kv_heads = 8;
	model_config.head_dim = 128;
	model_config.dim = model_config.n_heads * model_config.head_dim; /* 2048 */
	model_config.hidden_dim = model_config.dim * 4;                   /* 8192 */
	model_config.seq_len = 64;
	model_config.vocab_size = 32000;
	int kv_dim = model_config.n_kv_heads * model_config.head_dim; /* 1024 */
	int kv_mul = model_config.n_heads / model_config.n_kv_heads;  /* 2 */
	int pos = 31;                                                 /* 32 timesteps */
	unsigned __int64 loff = 0;

	int max_seq_len = 64;

	v7_run_state *state_pointer = rs_alloc(&model_config, max_seq_len, 0);
	if (!state_pointer)
	{
		printf("OOM\n");
		attn_dll_unload(&dll);
		return 1;
	}

	srand(12345);
	attn_fill_random(state_pointer, &model_config, max_seq_len, 0);

	/* Warm-up */
	for (int i = 0; i < 5; i++)
		dll.lib.v7d_attention(state_pointer, &model_config, pos, loff, kv_dim, kv_mul);

	/* Measure */
	int iters = 10000;
	double t0 = kt_get_seconds();
	for (int i = 0; i < iters; i++)
		dll.lib.v7d_attention(state_pointer, &model_config, pos, loff, kv_dim, kv_mul);
	double t1 = kt_get_seconds();

	double elapsed = t1 - t0;
	double us_per = (elapsed / iters) * 1e6;

	/* Approximate bytes per call (all heads, GQA shared KV):
	 *   q read:      n_heads * head_dim * 4
	 *   k+v_cache:   (pos+1) * kv_dim * 4 * 2  (one copy, shared)
	 *   att r+w:     n_heads * (pos+1) * 4 * 2
	 *   xb write:    dim * 4
	 */
	int bytes_per_call = (int)model_config.n_heads * model_config.head_dim * sizeof(float) + (int)(pos + 1) * kv_dim * sizeof(float) * 2 + (int)model_config.n_heads * (pos + 1) * sizeof(float) * 2 + (int)model_config.dim * sizeof(float);

	double gbs = (double)bytes_per_call / (elapsed / iters) / 1e9;

	printf("Pass: %d iterations, head_dim=%d, n_heads=%d, n_kv_heads=%d, pos=%d\n", iters, model_config.head_dim, model_config.n_heads, model_config.n_kv_heads, pos);
	printf("  %.3f us / call   (%.1f GB/s)\n", us_per, gbs);

	rs_free(state_pointer);
	attn_dll_unload(&dll);
	return 0;
}
