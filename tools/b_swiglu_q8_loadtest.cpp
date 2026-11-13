// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_swiglu_q8_loadtest.cpp -- Standalone SwiGLU Q8 kernel load benchmark
 *
 * Measures v7d_swiglu_q8 throughput (dispatched, spin ON by default).
 * Usage: b_swiglu_q8_loadtest.exe <dll_path> <n> <d> --db <db_path> [--time secs] [--iterations N]
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
	if (argc < 4)
	{
		printf("usage: b_swiglu_q8_loadtest.exe <dll_path> <n> <d> --db <db_path> [--time secs] [--iterations N]\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if (v7_load(argv[1], &lib))
		return 1;

	char *db_path = 0;
	int matmul_rows;
	int matmul_cols;
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
	if (argc < 4)
	{
		printf("usage: b_swiglu_q8_loadtest.exe <dll_path> <n> <d> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (sscanf(argv[2], "%d", &matmul_rows) != 1)
	{
		printf("usage: b_swiglu_q8_loadtest.exe <dll_path> <n> <d> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (sscanf(argv[3], "%d", &matmul_cols) != 1)
	{
		printf("usage: b_swiglu_q8_loadtest.exe <dll_path> <n> <d> --db <db_path> [--time secs] [--iterations N]\n");
		v7_unload(&lib);
		return 1;
	}
	if (matmul_rows <= 0 || matmul_cols <= 0)
	{
		printf("Fail: dimensions must be positive\n");
		v7_unload(&lib);
		return 1;
	}
	lib.v7_init(db_path);

	struct kt_tensor_q8 tsx;
	if (!kt_tensor_q8_alloc(&tsx, matmul_rows, matmul_cols))
	{
		printf("OOM\n");
		v7_unload(&lib);
		return 1;
	}
	kt_tensor_q8_fill(&tsx, matmul_rows, matmul_cols);
	v7_q8_tensor x = kt_make_input_q8(&tsx);

	struct kt_tensor_q8 tsg;
	if (!kt_tensor_q8_alloc(&tsg, matmul_rows, matmul_cols))
	{
		printf("OOM\n");
		v7_unload(&lib);
		return 1;
	}
	kt_tensor_q8_fill(&tsg, matmul_rows, matmul_cols);
	v7_q8_tensor gate_w = kt_make_weight_q8(&tsg);

	struct kt_tensor_q8 tsu;
	if (!kt_tensor_q8_alloc(&tsu, matmul_rows, matmul_cols))
	{
		printf("OOM\n");
		v7_unload(&lib);
		return 1;
	}
	kt_tensor_q8_fill(&tsu, matmul_rows, matmul_cols);
	v7_q8_tensor up_w = kt_make_weight_q8(&tsu);

	float *xout = (float *)malloc(matmul_cols * sizeof(float));

	lib.v7s_set_spin(1);

	int warmup = 100;
	for (int i = 0; i < warmup; i++)
		lib.v7d_swiglu_q8(xout, &x, &gate_w, matmul_rows, matmul_cols, &up_w);

	int iters = default_iters;
	if (run_seconds > 0.0)
	{
		int calibrate_iters = 100;
		double t_cal_start = kt_get_seconds();
		for (int i = 0; i < calibrate_iters; i++)
			lib.v7d_swiglu_q8(xout, &x, &gate_w, matmul_rows, matmul_cols, &up_w);
		double t_cal_end = kt_get_seconds();
		double cal_elapsed = t_cal_end - t_cal_start;
		if (cal_elapsed > 0)
		{
			iters = (int)((run_seconds / cal_elapsed) * calibrate_iters * 0.95);
			if (iters < 10)
				iters = 10;
		}
		printf(" swiglu_q8_loadtest %dx%d  %s  time-mode: %d iterations (calibrated %.1f s)\n", matmul_rows, matmul_cols, argv[1], iters, run_seconds);
	}

	double t0 = kt_get_seconds();
	for (int i = 0; i < iters; i++)
		lib.v7d_swiglu_q8(xout, &x, &gate_w, matmul_rows, matmul_cols, &up_w);
	double t1 = kt_get_seconds();

	lib.v7s_set_spin(0);

	double elapsed = t1 - t0;
	int bytes_per_call = (int)matmul_rows * matmul_cols * sizeof(float);
	double us_per_call = (elapsed / iters) * 1e6;
	double gbs = (double)bytes_per_call / (elapsed / iters) / 1e9;

	printf("Pass: %s n=%d d=%d iterations=%d\n", argv[1], matmul_rows, matmul_cols, iters);
	printf("  %.3f us / call   (%.1f GB/s)\n", us_per_call, gbs);

	kt_tensor_q8_free(&tsx);
	kt_tensor_q8_free(&tsg);
	kt_tensor_q8_free(&tsu);
	free(xout);
	v7_unload(&lib);
	return 0;
}
