// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_add_inplace_loadtest.cpp -- Standalone add_inplace kernel load benchmark
 *
 * Measures v7s_add_inplace throughput (direct-call, no dispatch).
 * Usage: b_add_inplace_loadtest.exe <dll_path> <n> --db <db_path> [--time secs] [--iterations N]
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
		printf("usage: b_add_inplace_loadtest.exe <dll_path> <n> --db <db_path> [--time secs] [--iterations N]\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if (v7_load(argv[1], &lib))
		return 1;

	char *db_path = 0;
	int num_elements = 0;
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
		else if (!num_elements)
			sscanf(argv[i], "%d", &num_elements);
	}
	if (!db_path)
	{
		printf("Fail: --db <db_path> is required\n");
		v7_unload(&lib);
		return 1;
	}
	if ((num_elements <= 0) || ((num_elements % 32) != 0))
	{
		printf("Fail: n must be a non-zero multiple of 32 (the kernel processes whole 32-float groups)\n");
		printf("usage: b_add_inplace_loadtest.exe <dll_path> <n> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	lib.v7_init(db_path);

	float *x = (float *)_aligned_malloc(num_elements * sizeof(float), 32);
	float *y = (float *)_aligned_malloc(num_elements * sizeof(float), 32);
	if (!x || !y)
	{
		printf("OOM\n");
		v7_unload(&lib);
		return 1;
	}

	kt_init_random_float(x, num_elements);
	kt_init_random_float(y, num_elements);

	int warmup = 1000;
	for (int i = 0; i < warmup; i++)
		lib.v7s_add_inplace(x, y, num_elements);

	int iters = default_iters;
	if (run_seconds > 0.0)
	{
		int calibrate_iters = 1000;
		double t_cal_start = kt_get_seconds();
		for (int i = 0; i < calibrate_iters; i++)
			lib.v7s_add_inplace(x, y, num_elements);
		double t_cal_end = kt_get_seconds();
		double cal_elapsed = t_cal_end - t_cal_start;
		if (cal_elapsed > 0)
		{
			iters = (int)((run_seconds / cal_elapsed) * calibrate_iters * 0.95);
			if (iters < 10)
				iters = 10;
		}
		printf(" v7s_add_inplace_loadtest n=%d  %s  time-mode: %d iterations (calibrated %.1f s)\n", num_elements, argv[1], iters, run_seconds);
	}

	double t0 = kt_get_seconds();
	for (int i = 0; i < iters; i++)
		lib.v7s_add_inplace(x, y, num_elements);
	double t1 = kt_get_seconds();
	double elapsed = t1 - t0;

	int bytes_per_call = (int)num_elements * sizeof(float) * 3;
	double us_per_call = (elapsed / iters) * 1e6;
	double gbs = (double)bytes_per_call / (elapsed / iters) / 1e9;

	printf("Pass: %s n=%d iterations=%d\n", argv[1], num_elements, iters);
	printf("  %.3f us / call   (%.1f GB/s)\n", us_per_call, gbs);

	_aligned_free(x);
	_aligned_free(y);
	v7_unload(&lib);
	return 0;
}
