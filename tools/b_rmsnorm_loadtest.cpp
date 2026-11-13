// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_rmsnorm_loadtest.cpp -- Standalone RMSNorm kernel load benchmark
 *
 * Measures v7s_rmsnorm throughput (direct-call, no dispatch).
 * Usage: b_rmsnorm_loadtest.exe <dll_path> <size> --db <db_path> [--time secs] [--iterations N]
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
	if (argc < 3)
	{
		printf("usage: b_rmsnorm_loadtest.exe <dll_path> <size> --db <db_path> [--time secs] [--iterations N]\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if (v7_load(argv[1], &lib))
		return 1;

	char *db_path = 0;
	int size = 0;
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
		else if (!size)
			sscanf(argv[i], "%d", &size);
	}
	if (!db_path)
	{
		printf("Fail: --db <db_path> is required\n");
		v7_unload(&lib);
		return 1;
	}
	if ((size <= 0) || ((size % 64) != 0))
	{
		printf("b_rmsnorm_loadtest: size must be a positive multiple of 64 (the kernel contract enforced at init by inf_microdb.inc step 3a); got %d\n", size);
		v7_unload(&lib);
		return 1;
	}
	lib.v7_init(db_path);

	float *o = (float *)_aligned_malloc(size * sizeof(float), 32);
	float *x = (float *)_aligned_malloc(size * sizeof(float), 32);
	float *weight = (float *)_aligned_malloc(size * sizeof(float), 32);
	if (!o || !x || !weight)
	{
		printf("OOM\n");
		v7_unload(&lib);
		return 1;
	}

	kt_init_random_float(x, size);
	kt_init_random_float(weight, size);

	int warmup = 1000;
	for (int i = 0; i < warmup; i++)
		lib.v7s_rmsnorm(o, x, weight, size);

	int iters = default_iters;
	if (run_seconds > 0.0)
	{
		int calibrate_iters = 1000;
		double t_cal_start = kt_get_seconds();
		for (int i = 0; i < calibrate_iters; i++)
			lib.v7s_rmsnorm(o, x, weight, size);
		double t_cal_end = kt_get_seconds();
		double cal_elapsed = t_cal_end - t_cal_start;
		if (cal_elapsed > 0)
		{
			iters = (int)((run_seconds / cal_elapsed) * calibrate_iters * 0.95);
			if (iters < 10)
				iters = 10;
		}
		printf(" v7s_rmsnorm_loadtest size=%d  %s  time-mode: %d iterations (calibrated %.1f s)\n", size, argv[1], iters, run_seconds);
	}

	double t0 = kt_get_seconds();
	for (int i = 0; i < iters; i++)
		lib.v7s_rmsnorm(o, x, weight, size);
	double t1 = kt_get_seconds();
	double elapsed = t1 - t0;

	int bytes_per_call = (int)size * sizeof(float) * 3;
	double us_per_call = (elapsed / iters) * 1e6;
	double gbs = (double)bytes_per_call / (elapsed / iters) / 1e9;

	printf("Pass: %s size=%d iterations=%d\n", argv[1], size, iters);
	printf("  %.3f us / call   (%.1f GB/s)\n", us_per_call, gbs);

	_aligned_free(o);
	_aligned_free(x);
	_aligned_free(weight);
	v7_unload(&lib);
	return 0;
}
