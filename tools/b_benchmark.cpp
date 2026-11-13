// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_benchmark.cpp -- Comprehensive multi-shape benchmark
 *
 * Runs Q8 and F32 matmul benchmarks across standard transformer shapes.
 * Spin mode ON by default.
 * Usage: b_benchmark.exe <dll_path> --db <db_path> [q8|f32|all] [--time secs] [--iterations N]
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "kt_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef struct bench_shape

{
	int rows;
	int cols;
	char *label;

} bench_shape;

bench_shape shapes[] = {
	{ 1024, 512, "1024x512  (SMALL)" }, { 1024, 1024, "1024x1024 (K/V)" }, { 1024, 2048, "1024x2048 (Q)" }, { 2048, 1024, "2048x1024 (O)" }, { 1024, 3072, "1024x3072 (FFN up)" }, { 3072, 1024, "3072x1024 (FFN dn)" },
};

#define N_SHAPES ((int)(sizeof(shapes) / sizeof(shapes[0])))

void run_q8(v7_lib *lib, int matmul_rows, int matmul_cols, int iters)
{
	struct kt_tensor_q8 ts;
	if (!kt_tensor_q8_alloc(&ts, matmul_rows, matmul_cols))
	{
		printf("OOM\n");
		return;
	}
	kt_tensor_q8_fill(&ts, matmul_rows, matmul_cols);
	v7_q8_tensor x = kt_make_input_q8(&ts);
	v7_q8_tensor w = kt_make_weight_q8(&ts);
	float *xout = (float *)malloc(matmul_cols * sizeof(float));

	for (int i = 0; i < 100; i++)
		lib->v7d_q8(xout, &x, &w, matmul_rows, matmul_cols);

	double t0 = kt_get_seconds();
	for (int i = 0; i < iters; i++)
		lib->v7d_q8(xout, &x, &w, matmul_rows, matmul_cols);
	double t1 = kt_get_seconds();

	int bytes = (int)matmul_rows * matmul_cols * 2;
	double gbs = (double)bytes / ((t1 - t0) / iters) / 1e9;
	double us = ((t1 - t0) / iters) * 1e6;

	printf("  Q8  %-24s %7.1f GB/s  %8.1f us\n", "", gbs, us);

	kt_tensor_q8_free(&ts);
	free(xout);
}

void run_f32(v7_lib *lib, int matmul_rows, int matmul_cols, int iters)
{
	struct kt_tensor_f32 ts;
	if (!kt_tensor_f32_alloc(&ts, matmul_rows, matmul_cols))
	{
		printf("OOM\n");
		return;
	}
	kt_tensor_f32_fill(&ts, matmul_rows, matmul_cols);
	v7_f32_tensor x = kt_make_input_f32(&ts);
	v7_q8_tensor w = kt_make_weight_f32(&ts);
	float *xout = (float *)malloc(matmul_cols * sizeof(float));

	for (int i = 0; i < 100; i++)
		lib->v7d_f32(xout, &x, &w, matmul_rows, matmul_cols);

	double t0 = kt_get_seconds();
	for (int i = 0; i < iters; i++)
		lib->v7d_f32(xout, &x, &w, matmul_rows, matmul_cols);
	double t1 = kt_get_seconds();

	int bytes = (int)matmul_rows * matmul_cols * (4 + 1 + 4);
	double gbs = (double)bytes / ((t1 - t0) / iters) / 1e9;
	double us = ((t1 - t0) / iters) * 1e6;

	printf("  F32 %-24s %7.1f GB/s  %8.1f us\n", "", gbs, us);

	kt_tensor_f32_free(&ts);
	free(xout);
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		printf("usage: b_benchmark.exe <dll_path> --db <db_path> [q8|f32|all] [--time secs] [--iterations N]\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if (v7_load(argv[1], &lib))
		return 1;

	char *db_path = 0;
	char *mode = "all";
	int default_iters = 500;
	double run_seconds = 0.0;

	for (int i = 2; i < argc; i++)
	{
		if (strcmp(argv[i], "--db") == 0 && i + 1 < argc)
			db_path = argv[++i];
		else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc)
			run_seconds = atof(argv[++i]);
		else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
			default_iters = atoi(argv[++i]);
		else
			mode = argv[i];
	}
	if (!db_path)
	{
		printf("Fail: --db <db_path> is required\n");
		v7_unload(&lib);
		return 1;
	}
	lib.v7_init(db_path);

	int do_q8 = (strcmp(mode, "q8") == 0 || strcmp(mode, "all") == 0);
	int do_f32 = (strcmp(mode, "f32") == 0 || strcmp(mode, "all") == 0);

	if (!do_q8 && !do_f32)
	{
		printf("Fail: unknown mode '%s' (use q8, f32, or all)\n", mode);
		v7_unload(&lib);
		return 1;
	}

	int iters = default_iters;
	if (run_seconds > 0.0 && N_SHAPES > 0)
	{
		struct kt_tensor_q8 ts;
		kt_tensor_q8_alloc(&ts, shapes[0].rows, shapes[0].cols);
		kt_tensor_q8_fill(&ts, shapes[0].rows, shapes[0].cols);
		v7_q8_tensor x = kt_make_input_q8(&ts);
		v7_q8_tensor w = kt_make_weight_q8(&ts);
		float *xout = (float *)malloc(shapes[0].cols * sizeof(float));
		int calibrate_iters = 100;
		double t_cal_start = kt_get_seconds();
		for (int i = 0; i < calibrate_iters; i++)
			lib.v7d_q8(xout, &x, &w, shapes[0].rows, shapes[0].cols);
		double t_cal_end = kt_get_seconds();
		double cal_elapsed = t_cal_end - t_cal_start;
		if (cal_elapsed > 0)
		{
			iters = (int)((run_seconds / (cal_elapsed * N_SHAPES)) * calibrate_iters * 0.95);
			if (iters < 10)
				iters = 10;
		}
		printf(" benchmark  %s  time-mode: %d iterations/shape (calibrated %.1f s total)\n", argv[1], iters, run_seconds);
		kt_tensor_q8_free(&ts);
		free(xout);
	}

	lib.v7s_set_spin(1);

	printf("\n... v7 benchmark: %s ...\n\n", argv[1]);

	for (int s = 0; s < N_SHAPES; s++)
	{
		printf("%s  rows=%d cols=%d\n", shapes[s].label, shapes[s].rows, shapes[s].cols);
		if (do_q8)
			run_q8(&lib, shapes[s].rows, shapes[s].cols, iters);
		if (do_f32)
			run_f32(&lib, shapes[s].rows, shapes[s].cols, iters);
	}

	lib.v7s_set_spin(0);

	printf("\n... done ...\n");
	v7_unload(&lib);
	return 0;
}
