// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_fused_loadtest.cpp -- Standalone fused F32 matmul kernel load benchmark
 *
 * Measures v7d_fused_f32 throughput (Q/K projections + RMSNorm + RoPE).
 * Spin mode ON by default.
 * Usage: b_fused_loadtest.exe <dll_path> <n> <d> <head_dim> <pos> --db <db_path> [--time secs] [--iterations N]
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "kt_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv)
{
	if (argc < 6)
	{
		printf("usage: b_fused_loadtest.exe <dll_path> <n> <d> <head_dim> <pos> --db <db_path> [--time secs] [--iterations N]\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if (v7_load(argv[1], &lib))
		return 1;

	char *db_path = 0;
	int matmul_rows;
	int matmul_cols;
	int head_dim;
	int pos;
	int default_iters = 1000;
	double run_seconds = 0.0;
	for (int i = 2; i < argc; i++)
	{
		if (strcmp(argv[i], "--db") == 0 && i + 1 < argc)
			db_path = argv[++i];
		else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc)
			run_seconds = atof(argv[++i]);
		else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
			default_iters = atoi(argv[++i]);
	}
	if (!db_path)
	{
		printf("Fail: --db <db_path> is required\n");
		v7_unload(&lib);
		return 1;
	}
	lib.v7_init(db_path);

	/* positional args after flags parsed: n, d, head_dim, pos still at argv[2..5] */
	if (argc < 6)
	{
		printf("usage: b_fused_loadtest.exe <dll_path> <n> <d> <head_dim> <pos> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (sscanf(argv[2], "%d", &matmul_rows) != 1)
	{
		printf("usage: b_fused_loadtest.exe <dll_path> <n> <d> <head_dim> <pos> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (sscanf(argv[3], "%d", &matmul_cols) != 1)
	{
		printf("usage: b_fused_loadtest.exe <dll_path> <n> <d> <head_dim> <pos> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (sscanf(argv[4], "%d", &head_dim) != 1)
	{
		printf("usage: b_fused_loadtest.exe <dll_path> <n> <d> <head_dim> <pos> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (sscanf(argv[5], "%d", &pos) != 1)
	{
		printf("usage: b_fused_loadtest.exe <dll_path> <n> <d> <head_dim> <pos> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}

	struct kt_tensor_f32 ts;
	if (!kt_tensor_f32_alloc(&ts, matmul_rows, matmul_cols))
	{
		printf("OOM\n");
		v7_unload(&lib);
		return 1;
	}
	kt_tensor_f32_fill(&ts, matmul_rows, matmul_cols);
	v7_f32_tensor x = kt_make_input_f32(&ts);

	struct kt_tensor_q8 tsw;
	if (!kt_tensor_q8_alloc(&tsw, matmul_rows, matmul_cols))
	{
		printf("OOM\n");
		v7_unload(&lib);
		return 1;
	}
	kt_tensor_q8_fill(&tsw, matmul_rows, matmul_cols);
	v7_q8_tensor wtensor = kt_make_weight_q8(&tsw);

	float *norm_weights = (float *)malloc(head_dim * sizeof(float));
	for (int i = 0; i < head_dim; i++)
		norm_weights[i] = 0.5f + (float)(rand() % 100) / 100.0f;

	int half_head_dim = head_dim / 2;
	float *cos_table = (float *)malloc(pos * half_head_dim * sizeof(float));
	float *sin_table = (float *)malloc(pos * half_head_dim * sizeof(float));
	for (int i = 0; i < pos * half_head_dim; i++)
	{
		cos_table[i] = (float)i / pos;
		sin_table[i] = (float)i / pos;
	}

	float *xout = (float *)malloc(matmul_cols * sizeof(float));

	/* fused_bias_ptr is the BSS global the fused kernel reads its QKV bias
	 * from. The kernel deliberately has NO NULL-bias special case (Sec.0a
	 * forbids a tool-only path), so the loadtest publishes a real bias
	 * exactly like production and v_fused_validate do -- an unpublished slot
	 * makes the kernel read [NULL + offset] and crash. */
	HMODULE dll_module = GetModuleHandleA(argv[1]);
	float **fused_bias_ptr_slot = (float **)GetProcAddress(dll_module, "fused_bias_ptr");
	if (!fused_bias_ptr_slot)
	{
		printf("[FATAL] fused_bias_ptr export missing from %s\n", argv[1]);
		v7_unload(&lib);
		return 1;
	}
	float *qkv_bias = (float *)malloc(matmul_cols * sizeof(float));
	if (!qkv_bias)
	{
		printf("OOM\n");
		v7_unload(&lib);
		return 1;
	}
	for (int i = 0; i < matmul_cols; i++)
		qkv_bias[i] = 0.25f * (float)((i % 7) - 3);
	*fused_bias_ptr_slot = qkv_bias;

	lib.v7s_rope_init(head_dim, 1.0e6f);
	lib.v7s_set_fused_rope(pos, head_dim);

	lib.v7s_set_spin(1);

	int warmup = 100;
	for (int w = 0; w < warmup; w++)
		lib.v7d_fused_f32(xout, &x, &wtensor, matmul_rows, matmul_cols, norm_weights, 1);

	int iters = default_iters;
	if (run_seconds > 0.0)
	{
		int calibrate_iters = 100;
		double t_cal_start = kt_get_seconds();
		for (int i = 0; i < calibrate_iters; i++)
			lib.v7d_fused_f32(xout, &x, &wtensor, matmul_rows, matmul_cols, norm_weights, 1);
		double t_cal_end = kt_get_seconds();
		double cal_elapsed = t_cal_end - t_cal_start;
		if (cal_elapsed > 0)
		{
			iters = (int)((run_seconds / cal_elapsed) * calibrate_iters * 0.95);
			if (iters < 10)
				iters = 10;
		}
		printf(" fused_loadtest %dx%d hd=%d pos=%d  %s  time-mode: %d iterations (calibrated %.1f s)\n", matmul_rows, matmul_cols, head_dim, pos, argv[1], iters, run_seconds);
	}

	double t0 = kt_get_seconds();
	for (int i = 0; i < iters; i++)
		lib.v7d_fused_f32(xout, &x, &wtensor, matmul_rows, matmul_cols, norm_weights, 1);
	double t1 = kt_get_seconds();

	lib.v7s_set_spin(0);

	double elapsed = t1 - t0;
	int bytes_per_call = (int)matmul_rows * head_dim * 3 * sizeof(float) + (int)matmul_rows * head_dim * 2;
	bytes_per_call += (int)head_dim * sizeof(float) + (int)2 * pos * half_head_dim * sizeof(float);

	double us_per_call = (elapsed / iters) * 1e6;
	double gbs = (double)bytes_per_call / (elapsed / iters) / 1e9;

	printf("Pass: %s n=%d d=%d head_dim=%d pos=%d iterations=%d\n", argv[1], matmul_rows, matmul_cols, head_dim, pos, iters);
	printf("  %.3f us / call   (%.1f GB/s)\n", us_per_call, gbs);

	free(norm_weights);
	free(cos_table);
	free(sin_table);
	*fused_bias_ptr_slot = 0;                  /* drop the published pointer BEFORE free: the DLL global would otherwise dangle */
	free(qkv_bias);
	kt_tensor_q8_free(&tsw);
	kt_tensor_f32_free(&ts);
	free(xout);
	v7_unload(&lib);
	return 0;
}
