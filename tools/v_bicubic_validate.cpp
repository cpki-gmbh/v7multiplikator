// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v_bicubic_dump.cpp -- v7s_bicubic_resize correctness validator.
 *
 * The regression BUG-01 (vfmadd132ss vs vfmadd213ss Horner operand order in
 * BC_POLY_INNER/BC_POLY_OUTER) made the Keys kernel evaluate a quadratic:
 * W(0)=0 (center tap erased) and W(1)=-3.0 -- every non-identity resize
 * saturated bright pixels to 255. The Qwen3-VL vision smoke never caught it
 * because Qwen3-VL uses nearest-neighbor (bicubic is LLaVA-OV-only).
 *
 * This validator compares the ASM kernel against a double-precision C
 * reference of the documented contract (Catmull-Rom a=-0.5, PIL half-pixel
 * centering, edge clamping, +0.5 truncate rounding) across identity, upscale,
 * downscale, and anisotropic shapes. BUG-01's signature: identity passes,
 * everything else fails with huge saturation-biased error.
 *
 * Usage: v_bicubic_dump.exe <dll_path>   (no --db needed: pure kernel)
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

int failing_tests_counter = 0;
int passing_tests_counter = 0;

/* C reference: Keys a=-0.5 kernel, PIL half-pixel centering, edge clamp,
 * +0.5 truncate rounding -- mirrors vit_bicubic.inc exactly. */
static double ref_kernel(double t)
{
	double a = fabs(t);
	if (a <= 1.0)
		return 1.5 * a * a * a - 2.5 * a * a + 1.0;
	if (a < 2.0)
		return -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0;
	return 0.0;
}

static int clampi(int v, int lo, int hi)
{
	if (v < lo)
		return lo;
	if (v > hi)
		return hi;
	return v;
}

static void ref_bicubic(const unsigned char *src, int src_w, int src_h, unsigned char *dst, int dst_w, int dst_h)
{
	double x_scale = (double)src_w / dst_w;
	double y_scale = (double)src_h / dst_h;
	for (int dy = 0; dy < dst_h; dy++)
	{
		/* PIL parity (B27): floor the RAW coordinate, clamp only tap indices --
		   the pre-floor clamps distorted the kernel weights at edges */
		double sy = ((double)dy + 0.5) * y_scale - 0.5;
		int y0 = (int)floor(sy);
		double fy = sy - (double)y0;
		for (int dx = 0; dx < dst_w; dx++)
		{
			double sx = ((double)dx + 0.5) * x_scale - 0.5;
			int x0 = (int)floor(sx);
			double fx = sx - (double)x0;
			for (int chan = 0; chan < 3; chan++)
			{
				double acc = 0.0;
				for (int j = 0; j < 4; j++)
				{
					double wy = ref_kernel((double)j - 1.0 - fy);
					int syi = clampi(y0 + j - 1, 0, src_h - 1);
					for (int i = 0; i < 4; i++)
					{
						double wx = ref_kernel((double)i - 1.0 - fx);
						int sxi = clampi(x0 + i - 1, 0, src_w - 1);
						acc += wx * wy * (double)src[(long)syi * src_w * 3 + (long)sxi * 3 + chan];
					}
				}
				if (acc < 0.0) acc = 0.0;
				if (acc > 255.0) acc = 255.0;
				dst[(long)dy * dst_w * 3 + (long)dx * 3 + chan] = (unsigned char)(int)(acc + 0.5);
			}
		}
	}
}

/* Run one shape: fill src with a smooth deterministic pattern (BUG-01's
 * saturation needs bright regions, not flat noise), resize, compare. */
static void run_case(v7_lib *lib, const char *desc, int src_w, int src_h, int dst_w, int dst_h, unsigned seed)
{
	unsigned char *src = (unsigned char *)malloc((long)src_w * src_h * 3);
	unsigned char *dst = (unsigned char *)malloc((long)dst_w * dst_h * 3);
	unsigned char *ref = (unsigned char *)malloc((long)dst_w * dst_h * 3);
	unsigned rng_state = seed;
	long saturated_ref = 0;
	for (long idx = 0; idx < (long)src_w * src_h * 3; idx++)
	{
		rng_state = rng_state * 1664525u + 1013904223u;
		/* smooth gradient + noise: spans 10..250 so kernel errors move real bytes */
		src[idx] = (unsigned char)(10 + ((int)(rng_state >> 24) % 60) + (int)(30 * (idx % src_w) / (src_w > 0 ? src_w : 1)) + (int)(150 * (idx / (src_w * 3)) / (src_h > 0 ? src_h : 1)));
	}

	ref_bicubic(src, src_w, src_h, ref, dst_w, dst_h);
	memset(dst, 0, (long)dst_w * dst_h * 3);
	lib->v7s_bicubic_resize(src, src_w, src_h, dst, dst_w, dst_h);

	long mismatches = 0;
	int worst_delta = 0;
	long worst_idx = 0;
	for (long idx = 0; idx < (long)dst_w * dst_h * 3; idx++)
	{
		if (ref[idx] == 255)
			saturated_ref++;
		int delta = (int)dst[idx] - (int)ref[idx];
		if (delta < 0)
			delta = -delta;
		if (delta > worst_delta)
		{
			worst_delta = delta;
			worst_idx = idx;
		}
		if (delta > 1)
			mismatches++;
	}

		if (mismatches == 0)
		{
			passing_tests_counter++;
			printf(".");
			fflush(stdout);
		}
	else
	{
		failing_tests_counter++;
		printf("  Fail %-34s mism(>1)=%ld max_delta=%d at byte %ld (asm=%d ref=%d)\n", desc, mismatches, worst_delta, worst_idx, (int)dst[worst_idx], (int)ref[worst_idx]);
	}
	free(src);
	free(dst);
	free(ref);
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		printf("usage: v_bicubic_dump.exe <dll_path>  (pure kernel -- no --db needed)\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if (v7_load(argv[1], &lib) != 0)
	{
		printf("v7_load failed\n");
		return 1;
	}
	printf("... bicubic_validate: %s ...\n", argv[1]);

	/* Identity (must be a no-op copy -- the scale=1 degenerate case) */
	run_case(&lib, "identity 300x200 -> 300x200", 300, 200, 300, 200, 1);
	run_case(&lib, "identity 640x480 -> 640x480", 640, 480, 640, 480, 2);

	/* Upscale -- regression shape: cat300x200 exercises the clamped neighbor
	 * reads at the right/bottom edges (it regressed to all-white when the
	 * edge clamp was missing) */
	run_case(&lib, "upscale 300x200 -> 384x384", 300, 200, 384, 384, 3);
	run_case(&lib, "upscale 640x480 -> 768x768", 640, 480, 768, 768, 4);

	/* Downscale */
	run_case(&lib, "downscale 640x480 -> 384x384", 640, 480, 384, 384, 5);
	run_case(&lib, "downscale 2304x2304 -> 384x384", 2304, 2304, 384, 384, 6);

	/* Anisotropic (x/y scale differ -- exercises outer-vs-inner kernel mix) */
	run_case(&lib, "aniso 640x480 -> 768x384", 640, 480, 768, 384, 7);
	run_case(&lib, "aniso 300x200 -> 384x200", 300, 200, 384, 200, 8);

	/* Tiny edge shapes (clamp-dominant: every tap clamps) */
	run_case(&lib, "tiny 2x2 -> 8x8", 2, 2, 8, 8, 9);
	run_case(&lib, "tiny 8x8 -> 2x2", 8, 8, 2, 2, 10);

	if (failing_tests_counter)
	{
		printf("bicubic_validate: %d FAIL\n", failing_tests_counter);
		return 1;
	}
	printf("bicubic_validate: ALL PASS (%d cases)\n", passing_tests_counter);
	return 0;
}
