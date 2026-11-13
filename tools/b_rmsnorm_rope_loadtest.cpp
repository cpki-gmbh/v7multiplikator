// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_rmsnorm_rope_loadtest.cpp -- Standalone rmsnorm+RoPE kernel load benchmark
 *
 * Measures v7s_rmsnorm_rope_heads throughput (direct-call, no dispatch).
 * Usage: b_rmsnorm_rope_loadtest.exe <dll_path> <n_heads> <head_dim> <pos> <rope_max_dim> --db <db_path> [--time secs] [--iterations N]
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
		printf("usage: b_rmsnorm_rope_loadtest.exe <dll_path> <n_heads> <head_dim> <pos> <rope_max_dim> --db <db_path> [--time secs] [--iterations N]\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if (v7_load(argv[1], &lib))
		return 1;

	char *db_path = 0;
	int n_heads;
	int head_dim;
	int pos;
	int rope_max_dim;
	int default_iters = 10000;
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
	if (argc < 6)
	{
		printf("usage: b_rmsnorm_rope_loadtest.exe <dll_path> <n_heads> <head_dim> <pos> <rope_max_dim> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (sscanf(argv[2], "%d", &n_heads) != 1)
	{
		printf("usage: b_rmsnorm_rope_loadtest.exe <dll_path> <n_heads> <head_dim> <pos> <rope_max_dim> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (sscanf(argv[3], "%d", &head_dim) != 1)
	{
		printf("usage: b_rmsnorm_rope_loadtest.exe <dll_path> <n_heads> <head_dim> <pos> <rope_max_dim> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (sscanf(argv[4], "%d", &pos) != 1)
	{
		printf("usage: b_rmsnorm_rope_loadtest.exe <dll_path> <n_heads> <head_dim> <pos> <rope_max_dim> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (sscanf(argv[5], "%d", &rope_max_dim) != 1)
	{
		printf("usage: b_rmsnorm_rope_loadtest.exe <dll_path> <n_heads> <head_dim> <pos> <rope_max_dim> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (n_heads <= 0 || head_dim <= 0 || pos < 0 || rope_max_dim <= 0)
	{
		printf("Fail: all dimensions must be positive\n");
		v7_unload(&lib);
		return 1;
	}
	lib.v7_init(db_path);

	int buf_size = n_heads * head_dim;
	int cos_sin_len = pos * rope_max_dim + rope_max_dim;

	float *buf = (float *)_aligned_malloc(buf_size * sizeof(float), 32);
	float *norm_weight = (float *)_aligned_malloc(head_dim * sizeof(float), 32);
	float *cos_table = (float *)_aligned_malloc(cos_sin_len * sizeof(float), 32);
	float *sin_table = (float *)_aligned_malloc(cos_sin_len * sizeof(float), 32);
	if (!buf || !norm_weight || !cos_table || !sin_table)
	{
		printf("OOM\n");
		v7_unload(&lib);
		return 1;
	}

	kt_init_random_float(buf, buf_size);
	kt_init_random_float(norm_weight, head_dim);
	kt_init_random_float(cos_table, cos_sin_len);
	kt_init_random_float(sin_table, cos_sin_len);

	int warmup = 1000;
	for (int i = 0; i < warmup; i++)
		lib.v7s_rmsnorm_rope_heads(buf, n_heads, head_dim, norm_weight, cos_table, sin_table, pos, rope_max_dim);

	int iters = default_iters;
	if (run_seconds > 0.0)
	{
		int calibrate_iters = 1000;
		double t_cal_start = kt_get_seconds();
		for (int i = 0; i < calibrate_iters; i++)
			lib.v7s_rmsnorm_rope_heads(buf, n_heads, head_dim, norm_weight, cos_table, sin_table, pos, rope_max_dim);
		double t_cal_end = kt_get_seconds();
		double cal_elapsed = t_cal_end - t_cal_start;
		if (cal_elapsed > 0)
		{
			iters = (int)((run_seconds / cal_elapsed) * calibrate_iters * 0.95);
			if (iters < 10)
				iters = 10;
		}
		printf(" v7s_rmsnorm_rope_loadtest heads=%d hd=%d pos=%d  %s  time-mode: %d iterations (calibrated %.1f s)\n", n_heads, head_dim, pos, argv[1], iters, run_seconds);
	}

	double t0 = kt_get_seconds();
	for (int i = 0; i < iters; i++)
		lib.v7s_rmsnorm_rope_heads(buf, n_heads, head_dim, norm_weight, cos_table, sin_table, pos, rope_max_dim);
	double t1 = kt_get_seconds();
	double elapsed = t1 - t0;

	int bytes_per_call = (int)buf_size * sizeof(float) * 2 + (int)head_dim * sizeof(float) + (int)2 * cos_sin_len * sizeof(float);
	double us_per_call = (elapsed / iters) * 1e6;
	double gbs = (double)bytes_per_call / (elapsed / iters) / 1e9;

	printf("Pass: %s n_heads=%d head_dim=%d pos=%d rope_max_dim=%d iterations=%d\n", argv[1], n_heads, head_dim, pos, rope_max_dim, iters);
	printf("  %.3f us / call   (%.1f GB/s)\n", us_per_call, gbs);

	_aligned_free(buf);
	_aligned_free(norm_weight);
	_aligned_free(cos_table);
	_aligned_free(sin_table);
	v7_unload(&lib);
	return 0;
}
