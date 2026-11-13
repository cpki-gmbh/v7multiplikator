// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_vit_linear.cpp -- ViT linear matmul benchmark (vit_speedup.md P0).
 * Measures the DISPATCHED v7d_vit_linear entry (S88: set_spin(1) first) on
 * production ViT shapes, reports GB/s effective bandwidth and GFLOP/s.
 * Usage: b_vit_linear.exe <dll_path> --db <db_path> [--iters N]
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <intrin.h>
#include <windows.h>

/* CPU% diagnostics (S28): average (user+kernel)/wall across each shape's
 * timed section. ~1200% = all 12 logical cores working; ~100% = dispatch is
 * broken (caller only). */
double cpu_percent_since(FILETIME *u0, FILETIME *k0)
{
	FILETIME f1, f2, f3, u1, k1;
	GetProcessTimes(GetCurrentProcess(), &f1, &f2, &u1, &k1);
	double du = (double)(*(unsigned long long *)&u1 - *(unsigned long long *)u0) / 1e7;
	double dk = (double)(*(unsigned long long *)&k1 - *(unsigned long long *)k0) / 1e7;
	return du + dk; /* seconds of CPU time elapsed */
}
void cpu_snapshot(FILETIME *u, FILETIME *k)
{
	FILETIME f1, f2;
	GetProcessTimes(GetCurrentProcess(), &f1, &f2, u, k);
}

/* Serialized cycle read: lfence + rdtsc + lfence keeps the counter read from
 * drifting across the opaque dispatch call (plain __rdtsc got reordered at
 * /O2 with 12 spinning workers -> bogus ~0 deltas on some runs). */
inline unsigned long long cyc()
{
	unsigned long long t;
	_mm_lfence();
	t = __rdtsc();
	_mm_lfence();
	return t;
}

double tsc_ghz()
{
	LARGE_INTEGER freq, c0, c1;
	QueryPerformanceFrequency(&freq);
	unsigned long long t0 = cyc();
	QueryPerformanceCounter(&c0);
	Sleep(50);
	unsigned long long t1 = cyc();
	QueryPerformanceCounter(&c1);
	double qsec = (double)(c1.QuadPart - c0.QuadPart) / freq.QuadPart;
	return (t1 - t0) / qsec / 1e9;
}

typedef struct shape

{
	int rows;
	int output_dim;
	int input_dim;
	char *desc;

} shape;
shape shapes[] = {
	{ 729, 1024, 1024, "Qwen3-VL qkv/proj (n=729)" },  { 729, 3072, 1024, "Qwen3-VL fc1 (n=729)" }, { 729, 1024, 3072, "Qwen3-VL fc2 (n=729)" }, { 1600, 4096, 1024, "Qwen3-VL fc1 (n=1600)" }, { 1600, 1024, 4096, "Qwen3-VL fc2 (n=1600)" },
	{ 729, 1152, 1152, "SigLIP/LLaVA-OV fc (n=729)" }, { 576, 4304, 1152, "LLaVA-OV fc1 (n=576)" }, { 576, 1152, 4304, "LLaVA-OV fc2 (n=576)" }, { 900, 1152, 1152, "LLaVA-OV fc (n=900)" },    { 0, 0, 0, 0 }
};

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		printf("usage: b_vit_linear.exe <dll> --db <db> [--iters N]\n");
		return 1;
	}
	ZydisInit();
	char *db = 0;
	int iters = 200;
	for (int i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--db") && i + 1 < argc)
			db = argv[++i];
		else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
			iters = atoi(argv[++i]);
	}
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
	lib.v7s_set_spin(1); /* S88: dispatched benchmark MUST spin workers */

	double ghz = tsc_ghz();
	printf("TSC %.3f GHz\n", ghz);

	printf("%-28s %10s %12s %12s\n", "shape", "us/call", "GB/s", "GFLOP/s");
	for (int si = 0; shapes[si].desc; ++si)
	{
		shape &s = shapes[si];
		float *in = (float *)malloc((int)s.rows * s.input_dim * 4);
		float *w = (float *)malloc((int)s.output_dim * s.input_dim * 4);
		float *b = (float *)malloc((int)s.output_dim * 4);
		float *out = (float *)malloc((int)s.rows * s.output_dim * 4);
		memset(in, 0x3E, (int)s.rows * s.input_dim * 4);
		memset(w, 0x3E, (int)s.output_dim * s.input_dim * 4);
		memset(b, 0, (int)s.output_dim * 4);

		/* warmup (spin-up workers, fill caches) */
		for (int i = 0; i < 20; i++)
			lib.v7d_vit_linear(out, in, w, b, s.rows, s.output_dim, s.input_dim);
		FILETIME u0, k0;
		cpu_snapshot(&u0, &k0);

		/* median-of-5 timing; per-rep print exposes measurement artifacts
		 * (a single collapsed rep poisons min() -- the earlier 0.00 us rows) */
		unsigned long long reps[5];
		for (int rep = 0; rep < 5; rep++)
		{
			unsigned long long t0 = cyc();
			for (int i = 0; i < iters; i++)
				lib.v7d_vit_linear(out, in, w, b, s.rows, s.output_dim, s.input_dim);
			unsigned long long t1 = cyc();
			reps[rep] = t1 - t0;
		}
		for (int i = 0; i < 4; i++) /* tiny insertion sort (5 items) */
			for (int j = i + 1; j < 5; j++)
				if (reps[j] < reps[i])
				{
					unsigned long long tmp = reps[i];
					reps[i] = reps[j];
					reps[j] = tmp;
				}
		printf("   reps(cycles): %llu %llu %llu %llu %llu\n", reps[0], reps[1], reps[2], reps[3], reps[4]);
		double sec = (double)reps[2] / ghz / 1e9 / iters; /* median */
		double cpu_s = cpu_percent_since(&u0, &k0);
		double cpu_pct = cpu_s / (sec * (iters + 20 + 4 * iters)) * 100.0; /* incl. warmup approx */
		printf("   avg CPU %.0f%% (1200%% = all 12 cores)\n", cpu_pct);

		/* work proof: checksum must be nonzero and stable across reps */
		unsigned long long sum = 0;
		for (long i = 0; i < 256; i++)
			sum += (unsigned long long)(out[(int)(i * 997) % ((int)s.rows * s.output_dim)] * 1024.0f);
		printf("   checksum=%llu (%s)\n", sum, sum ? "work done" : "Zero output -- CALLS NO-OP!");

		double flops = 2.0 * s.rows * s.output_dim * s.input_dim;
		double bytes = (double)(s.output_dim * s.input_dim + s.rows * s.input_dim + s.rows * s.output_dim + s.output_dim) * 4.0;
		printf("%-28s %10.2f %12.1f %12.1f\n", s.desc, sec * 1e6, bytes / sec / 1e9, flops / sec / 1e9);
		free(in);
		free(w);
		free(b);
		free(out);
	}
	v7_unload(&lib);
	return 0;
}
