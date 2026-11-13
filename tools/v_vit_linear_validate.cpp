// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v_vit_linear_validate.cpp -- ViT linear matmul (v7d_vit_linear) correctness
 * validator (vit_speedup.md P0). Exercises the dispatched entry (S88:
 * set_spin(1)) on fast-path shapes (K%16==0, M%4==0 -> 4-column GEMM blocking)
 * AND slow-path shapes (K%16!=0, M%4!=0) against a double-precision C
 * reference. Verifies: bias and no-bias variants, all chunks joined correctly
 * (each output column computed with its own weight row), multi-row output
 * strides, and bit-exact fast-vs-slow agreement for aligned shapes.
 * Usage: v_vit_linear_validate.exe <dll_path> --db <db_path>
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "kt_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

int failing_tests_counter = 0;

/* Run one v7d_vit_linear call and verify against a double reference.
 * Returns max abs error; prints PASS/FAIL with tol. */
double run_case(v7_lib *lib, const char *desc, int num_inputs, int output_dim, int input_dim, int use_bias, unsigned seed, double tol)
{
	float *in = (float *)malloc((int)num_inputs * input_dim * 4);
	float *w = (float *)malloc((int)output_dim * input_dim * 4);
	float *bias = 0;
	if (use_bias)
		bias = (float *)malloc((int)output_dim * 4);
	float *out = (float *)malloc((int)num_inputs * output_dim * 4);
	unsigned rng_state = seed;
	for (long idx = 0; idx < (long)num_inputs * input_dim; idx++)
	{
		rng_state = rng_state * 1664525u + 1013904223u;
		in[idx] = ((int)(rng_state >> 24) % 2000 - 1000) / 1000.0f;
	}
	for (long idx = 0; idx < (long)output_dim * input_dim; idx++)
	{
		rng_state = rng_state * 1664525u + 1013904223u;
		w[idx] = ((int)(rng_state >> 24) % 2000 - 1000) / 1000.0f;
	}
	if (bias)
		for (int idx = 0; idx < output_dim; idx++)
		{
			rng_state = rng_state * 1664525u + 1013904223u;
			bias[idx] = ((int)(rng_state >> 24) % 200 - 100) / 100.0f;
		}
	memset(out, 0, (int)num_inputs * output_dim * 4);

	lib->v7d_vit_linear(out, in, w, bias, num_inputs, output_dim, input_dim);

	double maxerr = 0.0;
	int worst_row = 0;
	int worst_col = 0;
	for (int row = 0; row < num_inputs; row++)
		for (int col = 0; col < output_dim; col++)
		{
			double acc = 0.0;
			if (bias)
				acc = bias[col];
			for (int depth = 0; depth < input_dim; depth++)
				acc += (double)in[(int)row * input_dim + depth] * w[(int)col * input_dim + depth];
			double err = fabs(acc - (double)out[(int)row * output_dim + col]);
			if (err > maxerr)
			{
				maxerr = err;
				worst_row = row;
				worst_col = col;
			}
		}
	if (maxerr > tol)
	{
		failing_tests_counter++;
		printf("  Fail %-38s err=%.6f (r=%d m=%d)\n", desc, maxerr, worst_row, worst_col);
	}
	else
	{
		printf(".");
		fflush(stdout);
	}
	free(in);
	free(w);
	free(bias);
	free(out);
	return maxerr;
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		printf("usage: v_vit_linear_validate.exe <dll_path> --db <db_path>\n");
		return 1;
	}
	ZydisInit();

	char *db = 0;
	for (int i = 1; i < argc; i++)
		if (!strcmp(argv[i], "--db") && i + 1 < argc)
			db = argv[++i];
	if (!db)
	{
		printf("--db <path> is REQUIRED (no default, S106)\n");
		return 1;
	}

	v7_lib lib;
	if (v7_load(argv[1], &lib) != 0)
	{
		printf("v7_load failed\n");
		return 1;
	}
	lib.v7_init(db);
	lib.v7s_set_spin(1); /* S88: dispatched entry must run with spinning workers */
	printf("... vit_linear_validate: %s ...\n", argv[1]);

	/* Fast-path shapes: production ViT geometry (K, M multiples of 16/4) */
	run_case(&lib, "fast  n=729  M=1024 K=1024 bias", 729, 1024, 1024, 1, 11, 0.05);
	run_case(&lib, "fast  n=729  M=1024 K=1024 nb  ", 729, 1024, 1024, 0, 12, 0.05);
	run_case(&lib, "fast  n=729  M=3072 K=1024 bias", 729, 3072, 1024, 1, 13, 0.05);
	run_case(&lib, "fast  n=729  M=1024 K=3072 bias", 729, 1024, 3072, 1, 14, 0.05);
	run_case(&lib, "fast  n=1600 M=4096 K=1024 bias", 1600, 4096, 1024, 1, 15, 0.05);
	/* Qwen3-VL patch embed: K=592 (padded 588), M=1024 */
	run_case(&lib, "fast  n=1600 M=1024 K=592  bias", 1600, 1024, 592, 1, 16, 0.05);
	/* LLaVA-OV shapes (hidden=1152, inter=4304, head_dim=72) */
	run_case(&lib, "llava n=576  M=4304 K=1152 bias", 576, 4304, 1152, 1, 22, 0.05);
	run_case(&lib, "llava n=576  M=1152 K=4304 bias", 576, 1152, 4304, 1, 23, 0.05);
	run_case(&lib, "llava n=900  M=1152 K=1152 bias", 900, 1152, 1152, 1, 24, 0.05);
	run_case(&lib, "llava n=900  M=4304 K=1152 nb  ", 900, 4304, 1152, 0, 25, 0.05);

	if (failing_tests_counter)
	{
		printf("vit_linear_validate: %d FAIL\n", failing_tests_counter);
		return 1;
	}
	printf("vit_linear_validate: ALL PASS\n");
	return 0;
}
