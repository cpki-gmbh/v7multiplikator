// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_memcpy_loadtest.cpp -- Standalone memcpy kernel load benchmark
 *
 * Measures v7d_memcpy dispatched throughput. Spin mode ON by default.
 * Usage: b_memcpy_loadtest.exe <dll_path> <n_bytes> --db <db_path> [--time secs] [--iterations N]
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
		printf("usage: b_memcpy_loadtest.exe <dll_path> <n_bytes> --db <db_path> [--time secs] [--iterations N]\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if (v7_load(argv[1], &lib))
		return 1;

	char *db_path = 0;
	int n_bytes = 0;
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
		else if (!n_bytes)
			sscanf(argv[i], "%d", &n_bytes);
	}
	if (!db_path)
	{
		printf("Fail: --db <db_path> is required\n");
		v7_unload(&lib);
		return 1;
	}
	if (n_bytes <= 0)
	{
		printf("usage: b_memcpy_loadtest.exe <dll_path> <n_bytes> --db <db_path> [--time secs] [--iterations N]\n");
		return 1;
	}
	if (n_bytes > 1024 * 1024 * 256)
	{
		printf("Fail: n_bytes must be <= 256MB\n");
		v7_unload(&lib);
		return 1;
	}
	lib.v7_init(db_path);

	int warmup = 100;
	void *src = malloc(n_bytes);
	void *dst = malloc(n_bytes);
	if (!src || !dst)
	{
		printf("OOM\n");
		free(src);
		free(dst);
		v7_unload(&lib);
		return 1;
	}
	memset(src, 0x42, n_bytes);
	memset(dst, 0, n_bytes);

	lib.v7s_set_spin(1);

	for (int i = 0; i < warmup; i++)
		lib.v7d_memcpy(dst, src, n_bytes);

	int iters = default_iters;
	if (run_seconds > 0.0)
	{
		int calibrate_iters = 100;
		double t_cal_start = kt_get_seconds();
		for (int i = 0; i < calibrate_iters; i++)
			lib.v7d_memcpy(dst, src, n_bytes);
		double t_cal_end = kt_get_seconds();
		double cal_elapsed = t_cal_end - t_cal_start;
		if (cal_elapsed > 0)
		{
			iters = (int)((run_seconds / cal_elapsed) * calibrate_iters * 0.95);
			if (iters < 10)
				iters = 10;
		}
		printf(" memcpy_loadtest %d bytes  %s  time-mode: %d iterations (calibrated %.1f s)\n", n_bytes, argv[1], iters, run_seconds);
	}

	double t0 = kt_get_seconds();
	for (int i = 0; i < iters; i++)
		lib.v7d_memcpy(dst, src, n_bytes);
	double t1 = kt_get_seconds();

	lib.v7s_set_spin(0);

	if (memcmp(src, dst, n_bytes) != 0)
	{
		printf("Fail: memcpy corruption detected\n");
		free(src);
		free(dst);
		v7_unload(&lib);
		return 1;
	}

	double elapsed = t1 - t0;
	double us_per_call = (elapsed / iters) * 1e6;
	double gbs = (double)n_bytes / (elapsed / iters) / 1e9;

	printf("Pass: %d iterations, %d bytes\n", iters, n_bytes);
	printf("  %.3f us / call   (%.1f GB/s)\n", us_per_call, gbs);

	free(src);
	free(dst);
	v7_unload(&lib);
	return 0;
}
