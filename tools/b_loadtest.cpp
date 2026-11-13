// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_loadtest.cpp -- Standalone Q8/F32 matmul dispatch load benchmark
 *
 * Measures v7d_q8 and v7d_f32 throughput via the scheduler.
 * Usage: b_loadtest.exe <dll_path> --db <db_path> [--q8|--f32] [shapes N:M ...] [--time secs] [--iterations N]
 *
 * Spin mode ON by default.
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
	if (argc < 2)
	{
		printf("usage: b_loadtest.exe <dll_path> --db <db_path> [--q8|--f32] [shapes n:m ...] [--time secs] [--iterations N]\n");
		printf("  --db <db_path>  REQUIRED -- microdb database (config/INI keys); no default path\n");
		printf("  shapes default: 64x64 128x128 256x256 512x512 1024x1024 1024x512 1024x2048 2048x1024 1024x3072 3072x1024\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if (v7_load(argv[1], &lib))
		return 1;

	char *db_path = 0;
	int precision = 1;
	kt_shape *shapes = kt_shapes;
	int nshapes = kt_nshapes;
	int default_iters = 1000;
	double run_seconds = 0.0;

	for (int i = 2; i < argc; i++)
	{
		if (strcmp(argv[i], "--db") == 0 && i + 1 < argc)
			db_path = argv[++i];
		else if (strcmp(argv[i], "--f32") == 0)
			precision = 0;
		else if (strcmp(argv[i], "--q8") == 0)
			precision = 1;
		else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc)
			run_seconds = atof(argv[++i]);
		else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
			default_iters = atoi(argv[++i]);
		else
		{
			int parsed_rows;
			int parsed_cols;
			if (sscanf(argv[i], "%d:%d", &parsed_rows, &parsed_cols) == 2)
			{
				int idx;
				for (idx = 0; idx < kt_nshapes; idx++)
					if (kt_shapes[idx].rows == parsed_rows && kt_shapes[idx].cols == parsed_cols)
						break;
				if (idx < kt_nshapes)
				{
					shapes = &kt_shapes[idx];
					nshapes = 1;
				}
				else
				{
					printf("unknown shape: %s\n", argv[i]);
					v7_unload(&lib);
					return 1;
				}
			}
			else
			{
				printf("invalid shape format, use N:M (e.g., 1024:512)\n");
				v7_unload(&lib);
				return 1;
			}
		}
	}

	if (!db_path)
	{
		printf("Fail: --db <db_path> is required (no default database path)\n");
		v7_unload(&lib);
		return 1;
	}
	lib.v7_init(db_path);

	int warmup = 100;
	double total_gbs = 0.0;

	for (int s = 0; s < nshapes; s++)
	{
		int matmul_rows = shapes[s].rows;
		int matmul_cols = shapes[s].cols;
		struct kt_tensor_q8 ts;
		struct kt_tensor_f32 ts_f;
		int use_q8 = (precision == 1);
		if (use_q8)
		{
			if (!kt_tensor_q8_alloc(&ts, matmul_rows, matmul_cols))
			{
				printf("OOM\n");
				v7_unload(&lib);
				return 1;
			}
			kt_tensor_q8_fill(&ts, matmul_rows, matmul_cols);
		}
		else
		{
			if (!kt_tensor_f32_alloc(&ts_f, matmul_rows, matmul_cols))
			{
				printf("OOM\n");
				v7_unload(&lib);
				return 1;
			}
			kt_tensor_f32_fill(&ts_f, matmul_rows, matmul_cols);
		}

		float *xout = (float *)malloc(matmul_cols * sizeof(float));
		lib.v7s_set_spin(1);

		for (int w = 0; w < warmup; w++)
		{
			if (use_q8)
			{
				v7_q8_tensor x = kt_make_input_q8(&ts);
				v7_q8_tensor w_t = kt_make_weight_q8(&ts);
				lib.v7d_q8(xout, &x, &w_t, matmul_rows, matmul_cols);
			}
			else
			{
				v7_f32_tensor x = kt_make_input_f32(&ts_f);
				v7_q8_tensor w = kt_make_weight_f32(&ts_f);
				lib.v7d_f32(xout, &x, &w, matmul_rows, matmul_cols);
			}
		}

		int iters = default_iters;
		if (run_seconds > 0.0)
		{
			int calibrate_iters = 100;
			double t_cal_start = kt_get_seconds();
			for (int i = 0; i < calibrate_iters; i++)
			{
				if (use_q8)
				{
					v7_q8_tensor x = kt_make_input_q8(&ts);
					v7_q8_tensor w_t = kt_make_weight_q8(&ts);
					lib.v7d_q8(xout, &x, &w_t, matmul_rows, matmul_cols);
				}
				else
				{
					v7_f32_tensor x = kt_make_input_f32(&ts_f);
					v7_q8_tensor w = kt_make_weight_f32(&ts_f);
					lib.v7d_f32(xout, &x, &w, matmul_rows, matmul_cols);
				}
			}
			double t_cal_end = kt_get_seconds();
			double cal_elapsed = t_cal_end - t_cal_start;
			if (cal_elapsed > 0)
			{
				iters = (int)((run_seconds / cal_elapsed) * calibrate_iters * 0.95);
				if (iters < 10)
					iters = 10;
			}
			printf(" loadtest %dx%d %s  %s  time-mode: %d iterations (calibrated %.1f s)  spin=1\n", matmul_rows, matmul_cols, use_q8 ? "q8" : "f32", argv[1], iters, run_seconds);
		}

		double t0 = kt_get_seconds();
		for (int w = 0; w < iters; w++)
		{
			if (use_q8)
			{
				v7_q8_tensor x = kt_make_input_q8(&ts);
				v7_q8_tensor w_t = kt_make_weight_q8(&ts);
				lib.v7d_q8(xout, &x, &w_t, matmul_rows, matmul_cols);
			}
			else
			{
				v7_f32_tensor x = kt_make_input_f32(&ts_f);
				v7_q8_tensor w = kt_make_weight_f32(&ts_f);
				lib.v7d_f32(xout, &x, &w, matmul_rows, matmul_cols);
			}
		}
		double t1 = kt_get_seconds();

		lib.v7s_set_spin(0);

		double elapsed = t1 - t0;
		int bytes_per_call = (int)matmul_rows * matmul_cols * 5;
		double gbs = (double)bytes_per_call * (double)iters / elapsed / 1e9;
		total_gbs += gbs;

		printf("Pass: %s %s %s = %.1f GB/s (%d iters, %.3f s)\n", argv[1], precision ? "q8" : "f32", shapes[s].label, gbs, iters, elapsed);

		free(xout);
		if (use_q8)
			kt_tensor_q8_free(&ts);
		else
			kt_tensor_f32_free(&ts_f);
	}

	printf("total: %.1f GB/s across %d shapes\n", total_gbs, nshapes);

	v7_unload(&lib);
	return 0;
}
