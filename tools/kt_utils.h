// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* kt_utils.h -- shared validator/benchmark utilities (header-only)
 *
 * Provides: timing, CPU measurement, random init, quantize patterns,
 *           tensor alloc/fill/free, C reference kernels, compare_outputs,
 *           bench helpers, verify helpers.
 *
 * Requires v7_lib.h to be included FIRST (provides v7_q8_tensor, v7_f32_tensor,
 * v7d_q8_fn, v7d_f32_fn, v7d_quantize_fn, v7d_fused_f32_fn, v7d_swiglu_fn, etc.)
 *
 * Usage (in each validator .cpp):
 *   #include "v7_lib.h"
 *   #include "kt_utils.h"
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <windows.h>

/* ======================================================================
 * Timing
 * ====================================================================== */

static double kt_get_seconds()
{
	LARGE_INTEGER freq = []
	{
		LARGE_INTEGER f;
		QueryPerformanceFrequency(&f);
		return f;
	}();
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	return (double)now.QuadPart / (double)freq.QuadPart;
}

#ifdef __linux__
struct cpu_snapshot
{
	long long wall_time;
	long long process_time;

	cpu_snapshot()
	{
		struct timespec wall = {}, cpu = {};
		clock_gettime(CLOCK_MONOTONIC, &wall);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu);
		wall_time = wall.tv_sec * 1000000000LL + wall.tv_nsec;
		process_time = cpu.tv_sec * 1000000000LL + cpu.tv_nsec;
	}
};

static double kt_cpu_delta_pct(cpu_snapshot start, cpu_snapshot end)
{
	long long wall_delta = end.wall_time - start.wall_time;
	long long process_delta = end.process_time - start.process_time;
	return wall_delta > 0 ? 100.0 * process_delta / wall_delta : 0.0;
}
#else
static ULONGLONG kt_filetime_to_ull(FILETIME filetime)
{
	return ((ULONGLONG)filetime.dwHighDateTime << 32) | filetime.dwLowDateTime;
}

/* ======================================================================
 * CPU measurement (process usage %)
 * ====================================================================== */

typedef struct cpu_snapshot
{
	ULONGLONG system_time; /* kernel+user CPU time across all cores */
	ULONGLONG process_time; /* kernel+user CPU time of this process */

	void capture()
	{
		FILETIME idle_ft, kernel_ft, user_ft, create_ft, exit_ft, proc_kernel_ft, proc_user_ft;
		GetSystemTimes(&idle_ft, &kernel_ft, &user_ft);
		GetProcessTimes(GetCurrentProcess(), &create_ft, &exit_ft, &proc_kernel_ft, &proc_user_ft);
		system_time = kt_filetime_to_ull(kernel_ft) + kt_filetime_to_ull(user_ft);
		process_time = kt_filetime_to_ull(proc_kernel_ft) + kt_filetime_to_ull(proc_user_ft);
	}

	cpu_snapshot()
	{
		capture();
	}

} cpu_snapshot;

static double kt_cpu_delta_pct(cpu_snapshot start, cpu_snapshot end)
{
	ULONGLONG system_delta = end.system_time - start.system_time;
	ULONGLONG process_delta = end.process_time - start.process_time;
	if (system_delta == 0)
		return 0.0;
	SYSTEM_INFO system_info;
	GetSystemInfo(&system_info);
	return (100.0 * process_delta) / (double)system_delta * system_info.dwNumberOfProcessors;
}

#endif

/* ======================================================================
 * Random init helpers
 * ====================================================================== */

static void kt_init_random_int8(char *data, __int64 count)
{
	for (__int64 i = 0; i < count; ++i)
		data[i] = (char)((rand() % 201) - 100);
}

static void kt_init_random_float(float *data, __int64 count)
{
	for (__int64 i = 0; i < count; ++i)
		data[i] = 0.1f + (float)(rand() % 100) / 1000.0f;
}

static void kt_init_random_uchar(unsigned char *data, __int64 count)
{
	for (__int64 i = 0; i < count; ++i)
		data[i] = (unsigned char)(rand() % 201);
}

/* ======================================================================
 * Quantize pattern init (for quantize_validate)
 * ====================================================================== */

enum kt_quantize_pattern_id
{
	KTPAT_SMALL_POSITIVE, /* 0.1-0.199 */
	KTPAT_WIDE_RANDOM,    /* -5.0 to +5.0 */
	KTPAT_ALL_ZEROS,      /* all zeros */
	KTPAT_ALTERNATING,    /* +1, -1 alternating */
	KTPAT_RAMP,           /* ascending 0..63 per group */
	KTPAT_LARGE_MIX,      /* near [-50,50] */
	KTPAT_POSITIONAL,     /* (pos % 64) - 32 */
	KTPAT_COUNT
};

static const char *kt_quantize_pattern_names[] = { "small_pos", "wide_rand", "all_zero", "alternating", "ramp", "large_mix", "positional" };

static void kt_init_quantize_pattern(float *data, __int64 count, int seed, int pattern)
{
	srand(seed);
	switch (pattern)
	{
	case KTPAT_SMALL_POSITIVE:
		for (__int64 i = 0; i < count; ++i)
			data[i] = 0.1f + (float)(rand() % 100) / 1000.0f;
		break;
	case KTPAT_WIDE_RANDOM:
		for (__int64 i = 0; i < count; ++i)
			data[i] = (float)((rand() % 10000) - 5000) / 1000.0f;
		break;
	case KTPAT_ALL_ZEROS:
		for (__int64 i = 0; i < count; ++i)
			data[i] = 0.0f;
		break;
	case KTPAT_ALTERNATING:
		for (__int64 i = 0; i < count; ++i)
			data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
		break;
	case KTPAT_RAMP:
		for (__int64 i = 0; i < count; ++i)
			data[i] = (float)(i % 64);
		break;
	case KTPAT_LARGE_MIX:
		for (__int64 i = 0; i < count; ++i)
			data[i] = (float)((rand() % 100000) - 50000) / 1000.0f;
		break;
	case KTPAT_POSITIONAL:
		for (__int64 i = 0; i < count; ++i)
			data[i] = (float)((i % 64) - 32);
		break;
	default:
		for (__int64 i = 0; i < count; ++i)
			data[i] = 1.0f;
		break;
	}
}

/* ======================================================================
 * Tensor alloc / fill / free (using v7_lib.h types)
 *
 * NOTE: v7_q8_tensor fields are .q and .s
 *       v7_f32_tensor fields are .data and .scales
 * ====================================================================== */

typedef struct kt_tensor_q8
{
	char *input_q;
	float *input_s;
	char *weight_q;
	float *weight_s;

} kt_tensor_q8;

typedef struct kt_tensor_f32
{
	float *input_f;
	float *input_s;
	char *weight_q;
	float *weight_s;

} kt_tensor_f32;

static int kt_tensor_q8_alloc(kt_tensor_q8 *ts, int matmul_rows, int matmul_cols)
{
	__int64 w_count = (__int64)matmul_cols * matmul_rows;
	int n_scales = matmul_rows / 64 + 1;
	int w_scales = (int)(w_count / 64) + 1;
	ts->input_q = (char *)malloc(matmul_rows);
	ts->input_s = (float *)malloc(n_scales * sizeof(float));
	ts->weight_q = (char *)malloc(w_count);
	ts->weight_s = (float *)malloc(w_scales * sizeof(float));
	if ((!ts->input_q) || (!ts->input_s) || (!ts->weight_q) || (!ts->weight_s))
	{
		free(ts->input_q);
		free(ts->input_s);
		free(ts->weight_q);
		free(ts->weight_s);
		ts->input_q = 0;
		ts->input_s = 0;
		ts->weight_q = 0;
		ts->weight_s = 0;
		return 0;
	}
	return 1;
}

static void kt_tensor_q8_free(kt_tensor_q8 *ts)
{
	free(ts->input_q);
	free(ts->input_s);
	free(ts->weight_q);
	free(ts->weight_s);
}

static void kt_tensor_q8_fill(kt_tensor_q8 *ts, int matmul_rows, int matmul_cols)
{
	__int64 w_count = (__int64)matmul_cols * matmul_rows;
	kt_init_random_int8(ts->input_q, matmul_rows);
	kt_init_random_float(ts->input_s, matmul_rows / 64 + 1);
	kt_init_random_int8(ts->weight_q, w_count);
	kt_init_random_float(ts->weight_s, (int)(w_count / 64) + 1);
}

static int kt_tensor_f32_alloc(kt_tensor_f32 *ts, int matmul_rows, int matmul_cols)
{
	__int64 w_count = (__int64)matmul_cols * matmul_rows;
	int n_scales = matmul_rows / 64 + 1;
	int w_scales = (int)(w_count / 64) + 1;
	ts->input_f = (float *)malloc(matmul_rows * sizeof(float));
	ts->input_s = (float *)malloc(n_scales * sizeof(float));
	ts->weight_q = (char *)malloc(w_count);
	ts->weight_s = (float *)malloc(w_scales * sizeof(float));
	if ((!ts->input_f) || (!ts->input_s) || (!ts->weight_q) || (!ts->weight_s))
	{
		free(ts->input_f);
		free(ts->input_s);
		free(ts->weight_q);
		free(ts->weight_s);
		ts->input_f = 0;
		ts->input_s = 0;
		ts->weight_q = 0;
		ts->weight_s = 0;
		return 0;
	}
	return 1;
}

static void kt_tensor_f32_free(kt_tensor_f32 *ts)
{
	free(ts->input_f);
	free(ts->input_s);
	free(ts->weight_q);
	free(ts->weight_s);
}

static void kt_tensor_f32_fill(kt_tensor_f32 *ts, int matmul_rows, int matmul_cols)
{
	__int64 w_count = (__int64)matmul_cols * matmul_rows;
	kt_init_random_float(ts->input_f, matmul_rows);
	kt_init_random_float(ts->input_s, matmul_rows / 64 + 1);
	kt_init_random_int8(ts->weight_q, w_count);
	kt_init_random_float(ts->weight_s, (int)(w_count / 64) + 1);
}

/* Convert kt_tensor_q8 / kt_tensor_f32 to v7_lib.h types for kernel calls */
static v7_q8_tensor kt_make_input_q8(kt_tensor_q8 *ts)
{
	v7_q8_tensor t;
	t.q = ts->input_q;
	t.s = ts->input_s;
	return t;
}
static v7_q8_tensor kt_make_weight_q8(kt_tensor_q8 *ts)
{
	v7_q8_tensor t;
	t.q = ts->weight_q;
	t.s = ts->weight_s;
	return t;
}
static v7_f32_tensor kt_make_input_f32(kt_tensor_f32 *ts)
{
	v7_f32_tensor t;
	t.data = ts->input_f;
	t.scales = ts->input_s;
	return t;
}
static v7_q8_tensor kt_make_weight_f32(kt_tensor_f32 *ts)
{
	v7_q8_tensor t;
	t.q = ts->weight_q;
	t.s = ts->weight_s;
	return t;
}

/* ======================================================================
 * C reference kernels (Q8 matmul, F32 matmul, fused F32, quantize)
 * ====================================================================== */

static void kt_matmul_c_ref_q8(float *xout, v7_q8_tensor *x, v7_q8_tensor *w, int matmul_rows, int matmul_cols)
{
	int GS = 64;
	int nb = (matmul_rows + GS - 1) / GS;
	for (int i = 0; i < matmul_cols; ++i)
	{
		double val = 0.0;
		__int64 row_off = (__int64)i * matmul_rows;
		int w_scale_row = (int)(row_off / GS);
		for (int b = 0; b < nb; ++b)
		{
			__int64 ival = 0;
			__int64 blk = (__int64)b * GS;
			int limit = (blk + GS < matmul_rows) ? GS : (matmul_rows - (int)blk);
			for (int k = 0; k < limit; ++k)
				ival += x->q[blk + k] * w->q[row_off + blk + k];
			val += (double)ival * (double)w->s[w_scale_row + b] * (double)x->s[b];
		}
		xout[i] = (float)val;
	}
}

static void kt_matmul_c_ref_f32(float *xout, v7_f32_tensor *x, v7_q8_tensor *w, int matmul_rows, int matmul_cols)
{
	int GS = 64;
	int nb = (matmul_rows + GS - 1) / GS;
	for (int i = 0; i < matmul_cols; ++i)
	{
		double val = 0.0;
		__int64 row_off = (__int64)i * matmul_rows;
		int w_scale_row = (int)(row_off / GS);
		for (int b = 0; b < nb; ++b)
		{
			double dot = 0.0;
			__int64 blk = (__int64)b * GS;
			int limit = (blk + GS < matmul_rows) ? GS : (matmul_rows - (int)blk);
			for (int k = 0; k < limit; ++k)
				dot += (double)x->data[blk + k] * (double)w->q[row_off + blk + k];
			val += dot * (double)w->s[w_scale_row + b];
		}
		xout[i] = (float)val;
	}
}

static void kt_quantize_c_ref(char *out_q, float *in_x, float *out_s, int num_elements)
{
	int GS = 64;
	int n_groups = (num_elements + GS - 1) / GS;
	for (int g = 0; g < n_groups; g++)
	{
		int start = g * GS;
		int limit = (num_elements - start) < GS ? (num_elements - start) : GS;
		float wmax = 0.0f;
		for (int i = 0; i < limit; i++)
		{
			float absv = fabsf(in_x[start + i]);
			if (absv > wmax)
				wmax = absv;
		}
		float scale = (wmax == 0.0f) ? (1.0f / 127.0f) : (wmax / 127.0f);
		out_s[g] = scale;
		float recip = 1.0f / scale;
		for (int i = 0; i < limit; i++)
		{
			float val = in_x[start + i] * recip;
			int q = (int)lrintf(val);
			if (q < -127)
				q = -127;
			if (q > 127)
				q = 127;
			out_q[start + i] = (char)q;
		}
	}
}

/* ======================================================================
 * Fused C reference (matmul + RMSNorm + RoPE)
 * ====================================================================== */

static void kt_fused_c_ref_f32(float *out, v7_f32_tensor *x, v7_q8_tensor *w, int matmul_rows, int matmul_cols, float *norm_weights, float *bias, int head_dim, float *cos_table, float *sin_table, int pos, float eps)
{
	kt_matmul_c_ref_f32(out, x, w, matmul_rows, matmul_cols);
	/* the fused kernel adds the QKV bias to the dot product BEFORE the per-head
	   RMSNorm, so the reference must too. Passing the real bias (not a zeros
	   block) is what makes the bias add -- and its column offset -- observable;
	   a zeros block would make the add a no-op that hides a broken offset. */
	for (int i = 0; i < matmul_cols; i++)
		out[i] += bias[i];
	int n_heads = matmul_cols / head_dim;
	float *cos_pos = cos_table + pos * (head_dim / 2);
	float *sin_pos = sin_table + pos * (head_dim / 2);
	for (int h = 0; h < n_heads; h++)
	{
		float *head = out + h * head_dim;
		double ss = 0.0;
		for (int i = 0; i < head_dim; i++)
			ss += (double)head[i] * (double)head[i];
		double rms = 1.0 / sqrt(ss / head_dim + (double)eps);
		for (int i = 0; i < head_dim; i++)
			head[i] *= (float)((double)norm_weights[i] * rms);
		for (int i = 0; i < head_dim / 2; i++)
		{
			float q0 = head[i];
			float q1 = head[i + head_dim / 2];
			float c = cos_pos[i];
			float s = sin_pos[i];
			head[i] = q0 * c - q1 * s;
			head[i + head_dim / 2] = q0 * s + q1 * c;
		}
	}
}

/* ======================================================================
 * C reference kernels -- element-wise / normalization
 * ====================================================================== */

static void v7s_rmsnorm_c_ref(float *o, float *x, float *weight, int size, float eps)
{
	float ss = 0.0f;
	for (int i = 0; i < size; i++)
		ss += x[i] * x[i];
	float scale = 1.0f / sqrtf(ss / size + eps);
	for (int i = 0; i < size; i++)
		o[i] = x[i] * scale * weight[i];
}

static void v7s_rmsnorm_rope_heads_c_ref(float *buf, int n_heads, int head_dim, float *norm_weight, float *cos_table, float *sin_table, int pos, int rope_max_dim, float eps)
{
	int half = head_dim / 2;
	float *cos_ptr = cos_table + (__int64)pos * rope_max_dim;
	float *sin_ptr = sin_table + (__int64)pos * rope_max_dim;
	for (int h = 0; h < n_heads; h++)
	{
		float *q = buf + (__int64)h * head_dim;
		double ss = 0.0;
		for (int i = 0; i < head_dim; i++)
			ss += (double)q[i] * (double)q[i];
		double rms = sqrt(ss / head_dim + (double)eps);
		double scale = 1.0 / rms;
		for (int i = 0; i < head_dim; i++)
			q[i] = (float)((double)q[i] * (double)norm_weight[i] * scale);
		for (int j = 0; j < half; j++)
		{
			double x0 = q[j];
			double y0 = q[j + half];
			double c = cos_ptr[j];
			double s = sin_ptr[j];
			q[j] = (float)(x0 * c - y0 * s);
			q[j + half] = (float)(x0 * s + y0 * c);
		}
	}
}

static void v7s_add_inplace_c_ref(float *x, float *y, int num_elements)
{
	for (int i = 0; i < num_elements; i++)
		x[i] += y[i];
}

static void v7s_fused_add_rmsnorm_c_ref(float *o, float *residual, float *add, float *weight, int dim, float eps)
{
	for (int i = 0; i < dim; i++)
		residual[i] += add[i];
	float ss = 0.0f;
	for (int i = 0; i < dim; i++)
		ss += residual[i] * residual[i];
	float scale = 1.0f / sqrtf(ss / dim + eps);
	for (int i = 0; i < dim; i++)
		o[i] = residual[i] * scale * weight[i];
}

static void swiglu_c_ref(float *xout, float *x, char *gate_wq, float *gate_ws, char *up_wq, float *up_ws, int matmul_rows, int matmul_cols)
{
	int GS = 64;
	int nb = (matmul_rows + GS - 1) / GS;
	for (int i = 0; i < matmul_cols; ++i)
	{
		__int64 row_off = (__int64)i * matmul_rows;
		int w_scale_row = (int)(row_off / GS);
		double gate_val = 0.0;
		double up_val = 0.0;
		for (int b = 0; b < nb; ++b)
		{
			double gate_dot = 0.0;
			double up_dot = 0.0;
			__int64 blk = (__int64)b * GS;
			int limit = (blk + GS < matmul_rows) ? GS : (matmul_rows - (int)blk);
			for (int k = 0; k < limit; ++k)
			{
				double xv = x[blk + k];
				gate_dot += xv * (double)gate_wq[row_off + blk + k];
				up_dot += xv * (double)up_wq[row_off + blk + k];
			}
			gate_val += gate_dot * (double)gate_ws[w_scale_row + b];
			up_val += up_dot * (double)up_ws[w_scale_row + b];
		}
		double sig = 1.0 / (1.0 + exp(-gate_val));
		xout[i] = (float)(gate_val * sig * up_val);
	}
}

static void swiglu_q8_c_ref(float *xout, char *xq, float *xs, char *gate_wq, float *gate_ws, char *up_wq, float *up_ws, int matmul_rows, int matmul_cols)
{
	/* TRUE independent reference (owner directive: the validator must detect
	   deviation, not mirror the kernel -- the previous bit-exact port proved
	   only that the kernel matches itself). Structure deliberately different
	   from swiglu_q8_kernel.inc: group dot products exactly in int64 (no
	   lane-order bookkeeping), each group's scale product in double, group
	   accumulation in double, sigmoid as 1/(1+exp(-g)) via libm exp on
	   double, output rounded once to float. The float32-accumulating kernel
	   therefore legitimately deviates on shapes with large group sums near
	   cancelling contributions -- the deviation this validator reports is
	   real float-vs-double drift, exactly what it is supposed to measure. */
	int GS = 64;
	int nb = (matmul_rows + GS - 1) / GS;
	for (int i = 0; i < matmul_cols; ++i)
	{
		__int64 row_off = (__int64)i * matmul_rows;
		int w_scale_row = (int)(row_off / GS);
		double gate_val = 0.0;
		double up_val = 0.0;
		for (int b = 0; b < nb; ++b)
		{
			__int64 blk = (__int64)b * GS;
			int limit = (blk + GS < matmul_rows) ? GS : (matmul_rows - (int)blk);
			__int64 gate_dot = 0;
			__int64 up_dot = 0;
			for (int k = 0; k < limit; ++k)
			{
				__int64 xv = (int)xq[blk + k];
				gate_dot += xv * (int)gate_wq[row_off + blk + k];
				up_dot += xv * (int)up_wq[row_off + blk + k];
			}
			double gsc = (double)xs[b] * (double)gate_ws[w_scale_row + b];
			double usc = (double)xs[b] * (double)up_ws[w_scale_row + b];
			gate_val += (double)gate_dot * gsc;
			up_val += (double)up_dot * usc;
		}
		double sig = 1.0 / (1.0 + exp(-gate_val));
		xout[i] = (float)(gate_val * sig * up_val);
	}
}

/* ======================================================================
 * Compare helper (absolute + relative tolerance)
 * ====================================================================== */

typedef struct kt_compare_result
{
	int pass;
	float max_diff;

} kt_compare_result;

static kt_compare_result kt_compare_outputs(float *ref, float *asm_, int count, float abs_tol, float rel_tol)
{
	kt_compare_result r = { 1, 0 };
	int max_idx = 0;
	int printed = 0;
	for (int i = 0; i < count; ++i)
	{
		float diff = fabsf(ref[i] - asm_[i]);
		float tol = ((rel_tol > 0) && (fabsf(ref[i]) >= 100.0f)) ? fabsf(ref[i]) * rel_tol : abs_tol;
		if (diff > r.max_diff)
		{
			r.max_diff = diff;
			max_idx = i;
		}
		if (diff > tol)
		{
			r.pass = 0;
			if (printed < 5)
			{
				printf("      FAIL[%d]: ref=%.6f asm=%.6f diff=%.6f (tol=%.6f)\n", i, ref[i], asm_[i], diff, tol);
				printed++;
			}
		}
	}
	if (!r.pass)
		printf("      worst: idx=%d ref=%.6f asm=%.6f diff=%.6f\n", max_idx, ref[max_idx], asm_[max_idx], r.max_diff);
	return r;
}

/* ======================================================================
 * Verify helpers (used by validate, fused_validate, quantize_validate)
 * ====================================================================== */

static int kt_verify_q8(v7d_q8_fn func, int matmul_rows, int matmul_cols, int seed, float *out_max_diff)
{
	kt_tensor_q8 ts;
	if (!kt_tensor_q8_alloc(&ts, matmul_rows, matmul_cols))
		return 0;
	float *ref = (float *)malloc(matmul_cols * sizeof(float));
	float *asm_ = (float *)malloc(matmul_cols * sizeof(float));
	if ((!ref) || (!asm_))
	{
		free(ref);
		free(asm_);
		kt_tensor_q8_free(&ts);
		return 0;
	}
	srand(seed);
	kt_tensor_q8_fill(&ts, matmul_rows, matmul_cols);
	v7_q8_tensor x = kt_make_input_q8(&ts);
	v7_q8_tensor w = kt_make_weight_q8(&ts);
	kt_matmul_c_ref_q8(ref, &x, &w, matmul_rows, matmul_cols);
	memset(asm_, 0, matmul_cols * sizeof(float));
	func(asm_, &x, &w, matmul_rows, matmul_cols);
	kt_compare_result cr = kt_compare_outputs(ref, asm_, matmul_cols, 0.01f, 1e-4f);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;
	free(ref);
	free(asm_);
	kt_tensor_q8_free(&ts);
	return cr.pass;
}

static int kt_verify_f32(v7d_f32_fn func, int matmul_rows, int matmul_cols, int seed, float *out_max_diff)
{
	kt_tensor_f32 ts;
	if (!kt_tensor_f32_alloc(&ts, matmul_rows, matmul_cols))
		return 0;
	float *ref = (float *)malloc(matmul_cols * sizeof(float));
	float *asm_ = (float *)malloc(matmul_cols * sizeof(float));
	if ((!ref) || (!asm_))
	{
		free(ref);
		free(asm_);
		kt_tensor_f32_free(&ts);
		return 0;
	}
	srand(seed);
	kt_tensor_f32_fill(&ts, matmul_rows, matmul_cols);
	v7_f32_tensor x = kt_make_input_f32(&ts);
	v7_q8_tensor w = kt_make_weight_f32(&ts);
	kt_matmul_c_ref_f32(ref, &x, &w, matmul_rows, matmul_cols);
	memset(asm_, 0, matmul_cols * sizeof(float));
	func(asm_, &x, &w, matmul_rows, matmul_cols);
	kt_compare_result cr = kt_compare_outputs(ref, asm_, matmul_cols, 0.01f, 1e-4f);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;
	free(ref);
	free(asm_);
	kt_tensor_f32_free(&ts);
	return cr.pass;
}

static int kt_verify_quantize(v7d_quantize_fn func, int num_elements, int seed, int pattern, float *out_max_scale_diff)
{
	float *in_x = (float *)malloc(num_elements * sizeof(float));
	char *ref_q = (char *)malloc(num_elements);
	char *asm_q = (char *)malloc(num_elements);
	int n_groups = (num_elements + 63) / 64;
	float *ref_s = (float *)malloc(n_groups * sizeof(float));
	float *asm_s = (float *)malloc(n_groups * sizeof(float));
	if ((!in_x) || (!ref_q) || (!asm_q) || (!ref_s) || (!asm_s))
	{
		free(in_x);
		free(ref_q);
		free(asm_q);
		free(ref_s);
		free(asm_s);
		return 0;
	}
	kt_init_quantize_pattern(in_x, num_elements, seed, pattern);
	kt_quantize_c_ref(ref_q, in_x, ref_s, num_elements);
	memset(asm_q, 0, num_elements);
	memset(asm_s, 0, n_groups * sizeof(float));
	func(asm_q, in_x, asm_s, num_elements);
	int pass = 1;
	int first_mismatch = -1;
	float max_diff = 0.0f;
	for (int i = 0; i < num_elements; i++)
	{
		float diff = fabsf(ref_s[i / 64] - asm_s[i / 64]);
		if (diff > max_diff)
			max_diff = diff;
	}
	for (int i = 0; i < num_elements; i++)
	{
		if (ref_q[i] != asm_q[i])
		{
			pass = 0;
			first_mismatch = i;
			break;
		}
	}
	if (!pass)
		printf("      quantize FAIL: first mismatch at index %d\n", first_mismatch);
	if (out_max_scale_diff)
		*out_max_scale_diff = max_diff;
	free(in_x);
	free(ref_q);
	free(asm_q);
	free(ref_s);
	free(asm_s);
	return pass;
}

static int kt_verify_q8_fused(v7d_fused_f32_fn func, int matmul_rows, int matmul_cols, int seed, float *out_max_diff)
{
	float *in_x = (float *)malloc(matmul_rows * sizeof(float));
	__int64 w_count = (__int64)matmul_cols * matmul_rows;
	int w_scales_count = (int)(w_count / 64) + 1;
	char *w_q = (char *)malloc(w_count);
	float *w_s = (float *)malloc(w_scales_count * sizeof(float));
	float *norm_weights = (float *)malloc(matmul_cols * sizeof(float));
	float *ref = (float *)malloc(matmul_cols * sizeof(float));
	float *asm_ = (float *)malloc(matmul_cols * sizeof(float));
	if ((!in_x) || (!w_q) || (!w_s) || (!norm_weights) || (!ref) || (!asm_))
	{
		free(in_x);
		free(w_q);
		free(w_s);
		free(norm_weights);
		free(ref);
		free(asm_);
		return 0;
	}
	srand(seed);
	kt_init_random_float(in_x, matmul_rows);
	kt_init_random_int8(w_q, w_count);
	kt_init_random_float(w_s, w_scales_count);
	kt_init_random_float(norm_weights, matmul_cols);
	char *x_q = (char *)malloc(matmul_rows);
	float *x_s = (float *)malloc((matmul_rows / 64 + 1) * sizeof(float));
	if ((!x_q) || (!x_s))
	{
		free(in_x);
		free(w_q);
		free(w_s);
		free(norm_weights);
		free(ref);
		free(asm_);
		free(x_q);
		free(x_s);
		return 0;
	}
	kt_quantize_c_ref(x_q, in_x, x_s, matmul_rows);
	v7_q8_tensor x_ref = { x_q, x_s };
	v7_q8_tensor w_ref = { w_q, w_s };
	kt_matmul_c_ref_q8(ref, &x_ref, &w_ref, matmul_rows, matmul_cols);
	v7_f32_tensor x_asm = { in_x, 0 };
	v7_q8_tensor w_asm = { w_q, w_s };
	memset(asm_, 0, matmul_cols * sizeof(float));
	func(asm_, &x_asm, &w_asm, matmul_rows, matmul_cols, norm_weights, 1);
	kt_compare_result cr = kt_compare_outputs(ref, asm_, matmul_cols, 0.01f, 1e-4f);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;
	free(in_x);
	free(w_q);
	free(w_s);
	free(norm_weights);
	free(ref);
	free(asm_);
	free(x_q);
	free(x_s);
	return cr.pass;
}

/* ======================================================================
 * Verify helpers -- element-wise / normalization kernels
 * ====================================================================== */

static int kt_verify_rmsnorm(v7s_rmsnorm_fn func, int size, int seed, float eps, float *out_max_diff)
{
	float *x = (float *)malloc(size * sizeof(float));
	float *weight = (float *)malloc(size * sizeof(float));
	float *ref = (float *)malloc(size * sizeof(float));
	float *asm_ = (float *)malloc(size * sizeof(float));
	if ((!x) || (!weight) || (!ref) || (!asm_))
	{
		free(x);
		free(weight);
		free(ref);
		free(asm_);
		return 0;
	}
	srand(seed);
	for (int i = 0; i < size; i++)
		x[i] = -1.0f + (float)(rand() % 2001) / 1000.0f;
	for (int i = 0; i < size; i++)
		weight[i] = 0.5f + (float)(rand() % 1001) / 1000.0f;
	v7s_rmsnorm_c_ref(ref, x, weight, size, eps);
	memset(asm_, 0, size * sizeof(float));
	func(asm_, x, weight, size);
	kt_compare_result cr = kt_compare_outputs(ref, asm_, size, 1e-3f, 0);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;
	free(x);
	free(weight);
	free(ref);
	free(asm_);
	return cr.pass;
}

/*
 * Verify the fused v7s_rmsnorm_quantize kernel two ways:
 *
 *  (a) BIT-IDENTICAL guarantee (primary, must pass): the fused kernel's int8
 *      output must equal the ASM 2-step chain (v7s_rmsnorm -> v7d_quantize)
 *      element-for-element. Both read eps from the DLL BSS global g_rms_eps
 *      (the production source of truth, loaded from the DB at init), so the
 *      comparison is eps-consistent by construction. Any mismatch is a fusion
 *      bug.
 *
 *  (b) C-reference chain (secondary, informational): fused int8 compared
 *      against v7s_rmsnorm_c_ref -> kt_quantize_c_ref using c_ref_eps (the
 *      DB's __sys_rms_eps value, queried by the validator -- never a literal).
 *      The C ref sums squares with non-fused float adds while the ASM uses
 *      vfmadd231ps dual accumulators, so norms differ by ~1e-7 relative and an
 *      exact int8 compare can legitimately flip at a rounding boundary.
 *      Mismatch count is returned and printed, but the pass/fail decision is
 *      gated ONLY on the bit-identical ASM-chain comparison plus a scale
 *      tolerance.
 */
static int kt_verify_rmsnorm_quantize(v7s_rmsnorm_quantize_fn fused_fn, v7s_rmsnorm_fn rmsnorm_fn, v7d_quantize_fn quantize_fn, int size, int seed, float c_ref_eps, float *out_max_scale_diff, int *out_asm_mismatches)
{
	float *x = (float *)malloc(size * sizeof(float));
	float *weight = (float *)malloc(size * sizeof(float));
	float *asm_chain_xb = (float *)malloc(size * sizeof(float));
	char *asm_chain_q = (char *)malloc(size);
	char *fused_q = (char *)malloc(size);
	int n_groups = (size + 63) / 64;
	float *asm_chain_s = (float *)malloc(n_groups * sizeof(float));
	float *fused_s = (float *)malloc(n_groups * sizeof(float));
	if ((!x) || (!weight) || (!asm_chain_xb) || (!asm_chain_q) || (!fused_q) || (!asm_chain_s) || (!fused_s))
	{
		free(x);
		free(weight);
		free(asm_chain_xb);
		free(asm_chain_q);
		free(fused_q);
		free(asm_chain_s);
		free(fused_s);
		return 0;
	}
	srand(seed);
	for (int i = 0; i < size; i++)
		x[i] = -1.0f + (float)(rand() % 2001) / 1000.0f;
	for (int i = 0; i < size; i++)
		weight[i] = 0.5f + (float)(rand() % 1001) / 1000.0f;

	/* ASM 2-step chain: rmsnorm -> quantize (production reference) */
	rmsnorm_fn(asm_chain_xb, x, weight, size);
	quantize_fn(asm_chain_q, asm_chain_xb, asm_chain_s, size);

	/* Fused kernel -- eps read from DLL BSS g_rms_eps, same source as the
	 * ASM chain above, so the bit-identical comparison is eps-consistent. */
	memset(fused_q, 0, size);
	memset(fused_s, 0, n_groups * sizeof(float));
	fused_fn(fused_q, x, fused_s, size, weight);

	/* (a) Bit-identical int8 comparison vs ASM 2-step chain */
	int asm_mismatches = 0;
	for (int i = 0; i < size; i++)
		if (fused_q[i] != asm_chain_q[i])
			asm_mismatches++;
	if (out_asm_mismatches)
		*out_asm_mismatches = asm_mismatches;

	/* Scale comparison vs ASM chain (both FMA rmsnorm -> same wmax, expect 0 diff) */
	float max_scale_diff = 0.0f;
	for (int g = 0; g < n_groups; g++)
	{
		float diff = fabsf(fused_s[g] - asm_chain_s[g]);
		if (diff > max_scale_diff)
			max_scale_diff = diff;
	}
	if (out_max_scale_diff)
		*out_max_scale_diff = max_scale_diff;

	/* (b) C-reference chain for correctness sanity -- count only */
	float *c_xb = (float *)malloc(size * sizeof(float));
	char *c_q = (char *)malloc(size);
	float *c_s = (float *)malloc(n_groups * sizeof(float));
	if (((c_xb) && (c_q) && (c_s)))
	{
		v7s_rmsnorm_c_ref(c_xb, x, weight, size, c_ref_eps);
		kt_quantize_c_ref(c_q, c_xb, c_s, size);
		int c_mismatches = 0;
		for (int i = 0; i < size; i++)
			if (fused_q[i] != c_q[i])
				c_mismatches++;
		if (c_mismatches > 0)
			printf("      fused/C-ref int8 mismatches: %d/%d (FMA vs non-FMA rounding, informational)\n", c_mismatches, size);
	}
	free(c_xb);
	free(c_q);
	free(c_s);

	free(x);
	free(weight);
	free(asm_chain_xb);
	free(asm_chain_q);
	free(fused_q);
	free(asm_chain_s);
	free(fused_s);
	return ((asm_mismatches == 0) && (max_scale_diff < 1e-3f));
}

static int kt_verify_rmsnorm_rope_heads(v7s_rmsnorm_rope_heads_fn func, int n_heads, int head_dim, int pos, int rope_max_dim, int seed, float eps, float *out_max_diff)
{
	int buf_size = n_heads * head_dim;
	float *buf = (float *)malloc(buf_size * sizeof(float));
	float *buf_copy = (float *)malloc(buf_size * sizeof(float));
	float *norm_weight = (float *)malloc(head_dim * sizeof(float));
	float *cos_table = (float *)malloc(((__int64)pos + 1) * rope_max_dim * sizeof(float));
	float *sin_table = (float *)malloc(((__int64)pos + 1) * rope_max_dim * sizeof(float));
	if ((!buf) || (!buf_copy) || (!norm_weight) || (!cos_table) || (!sin_table))
	{
		free(buf);
		free(buf_copy);
		free(norm_weight);
		free(cos_table);
		free(sin_table);
		return 0;
	}
	srand(seed);
	for (int i = 0; i < buf_size; i++)
		buf[i] = -1.0f + (float)(rand() % 2001) / 1000.0f;
	for (int i = 0; i < head_dim; i++)
		norm_weight[i] = 0.5f + (float)(rand() % 1001) / 1000.0f;
	int half = head_dim / 2;
	for (int p = 0; p <= pos; p++)
	{
		__int64 base = (__int64)p * rope_max_dim;
		for (int j = 0; j < half; j++)
		{
			float theta = powf(10000.0f, -2.0f * j / (float)head_dim);
			cos_table[base + j] = cosf(p * theta);
			sin_table[base + j] = sinf(p * theta);
		}
		for (int j = half; j < rope_max_dim; j++)
		{
			cos_table[base + j] = 0.0f;
			sin_table[base + j] = 0.0f;
		}
	}
	memcpy(buf_copy, buf, buf_size * sizeof(float));
	v7s_rmsnorm_rope_heads_c_ref(buf_copy, n_heads, head_dim, norm_weight, cos_table, sin_table, pos, rope_max_dim, eps);
	func(buf, n_heads, head_dim, norm_weight, cos_table, sin_table, pos, rope_max_dim);
	kt_compare_result cr = kt_compare_outputs(buf_copy, buf, buf_size, 1e-3f, 0);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;
	free(buf);
	free(buf_copy);
	free(norm_weight);
	free(cos_table);
	free(sin_table);
	return cr.pass;
}

static int kt_verify_add_inplace(v7s_add_inplace_fn func, int num_elements, int seed, float *out_max_diff)
{
	float *x = (float *)malloc(num_elements * sizeof(float));
	float *y = (float *)malloc(num_elements * sizeof(float));
	float *ref = (float *)malloc(num_elements * sizeof(float));
	float *asm_ = (float *)malloc(num_elements * sizeof(float));
	if ((!x) || (!y) || (!ref) || (!asm_))
	{
		free(x);
		free(y);
		free(ref);
		free(asm_);
		return 0;
	}
	srand(seed);
	for (int i = 0; i < num_elements; i++)
		x[i] = -10.0f + (float)(rand() % 20001) / 1000.0f;
	for (int i = 0; i < num_elements; i++)
		y[i] = -10.0f + (float)(rand() % 20001) / 1000.0f;
	memcpy(ref, x, num_elements * sizeof(float));
	v7s_add_inplace_c_ref(ref, y, num_elements);
	memcpy(asm_, x, num_elements * sizeof(float));
	func(asm_, y, num_elements);
	kt_compare_result cr = kt_compare_outputs(ref, asm_, num_elements, 1e-6f, 0);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;
	free(x);
	free(y);
	free(ref);
	free(asm_);
	return cr.pass;
}

/* The C reference gets its OWN input copy and the kernel another one: the
 * kernel ADDS IN PLACE, so shared buffers would couple the two sides. This
 * helper is compiled without /O2 (see build.bat) -- an inlined-at-/O2 copy of
 * this sequence miscompiles in large mains (wrong ss scale on the reference
 * side); unoptimized code is correct and this is a validator, not a hot path. */
static int kt_verify_fused_add_rmsnorm(v7s_fused_add_rmsnorm_fn func, int dim, int seed, float eps, float *out_max_diff)
{
	float *residual = (float *)malloc(dim * sizeof(float));
	float *add = (float *)malloc(dim * sizeof(float));
	float *weight = (float *)malloc(dim * sizeof(float));
	float *ref = (float *)malloc(dim * sizeof(float));
	float *asm_ = (float *)malloc(dim * sizeof(float));
	/* The C reference gets its OWN input copy and the kernel another one. The
	 * kernel ADDS IN PLACE, and at /O2 the compiler may sink parts of the pure
	 * C-ref computation past the opaque kernel call (legal: no data
	 * dependencies through the pointer). A shared, re-copied buffer lets such
	 * a reordering change which data the ref reads; exclusive copies make the
	 * ref's inputs call-invariant, so reordering cannot change its result. */
	float *residual_ref = (float *)malloc(dim * sizeof(float));
	float *residual_asm = (float *)malloc(dim * sizeof(float));
	if ((!residual) || (!add) || (!weight) || (!ref) || (!asm_) || (!residual_ref) || (!residual_asm))
	{
		free(residual);
		free(add);
		free(weight);
		free(ref);
		free(asm_);
		free(residual_ref);
		free(residual_asm);
		return 0;
	}
	srand(seed);
	for (int i = 0; i < dim; i++)
		residual[i] = -1.0f + (float)(rand() % 2001) / 1000.0f;
	for (int i = 0; i < dim; i++)
		add[i] = -1.0f + (float)(rand() % 2001) / 1000.0f;
	for (int i = 0; i < dim; i++)
		weight[i] = 0.5f + (float)(rand() % 1001) / 1000.0f;
	memcpy(residual_ref, residual, dim * sizeof(float));
	v7s_fused_add_rmsnorm_c_ref(ref, residual_ref, add, weight, dim, eps);
	memcpy(residual_asm, residual, dim * sizeof(float));
	memset(asm_, 0, dim * sizeof(float));
	func(asm_, residual_asm, add, weight, dim);
	kt_compare_result cr = kt_compare_outputs(ref, asm_, dim, 1e-3f, 0);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;
	free(residual);
	free(add);
	free(weight);
	free(ref);
	free(asm_);
	free(residual_ref);
	free(residual_asm);
	return cr.pass;
}

static int kt_verify_swiglu(v7d_swiglu_fn func, int matmul_rows, int matmul_cols, int seed, float *out_max_diff)
{
	float *x = (float *)malloc(matmul_rows * sizeof(float));
	int w_ws = (int)((__int64)matmul_cols * matmul_rows / 64) + 1;
	char *gate_wq = (char *)malloc((__int64)matmul_cols * matmul_rows);
	float *gate_ws = (float *)malloc(w_ws * sizeof(float));
	char *up_wq = (char *)malloc((__int64)matmul_cols * matmul_rows);
	float *up_ws = (float *)malloc(w_ws * sizeof(float));
	float *ref = (float *)malloc(matmul_cols * sizeof(float));
	float *asm_ = (float *)malloc(matmul_cols * sizeof(float));
	if ((!x) || (!gate_wq) || (!gate_ws) || (!up_wq) || (!up_ws) || (!ref) || (!asm_))
	{
		free(x);
		free(gate_wq);
		free(gate_ws);
		free(up_wq);
		free(up_ws);
		free(ref);
		free(asm_);
		return 0;
	}
	srand(seed);
	for (int i = 0; i < matmul_rows; i++)
		x[i] = 0.1f + (float)(rand() % 100) / 1000.0f;
	kt_init_random_int8(gate_wq, (__int64)matmul_cols * matmul_rows);
	kt_init_random_float(gate_ws, w_ws);
	kt_init_random_int8(up_wq, (__int64)matmul_cols * matmul_rows);
	kt_init_random_float(up_ws, w_ws);
	swiglu_c_ref(ref, x, gate_wq, gate_ws, up_wq, up_ws, matmul_rows, matmul_cols);
	v7_f32_tensor x_t;
	x_t.data = x;
	x_t.scales = 0;
	v7_q8_tensor gate_t;
	gate_t.q = gate_wq;
	gate_t.s = gate_ws;
	v7_q8_tensor up_t;
	up_t.q = up_wq;
	up_t.s = up_ws;
	memset(asm_, 0, matmul_cols * sizeof(float));
	func(asm_, &x_t, &gate_t, matmul_rows, matmul_cols, &up_t);
	kt_compare_result cr = kt_compare_outputs(ref, asm_, matmul_cols, 0.02f, 1e-4f);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;
	free(x);
	free(gate_wq);
	free(gate_ws);
	free(up_wq);
	free(up_ws);
	free(ref);
	free(asm_);
	return cr.pass;
}

static int kt_verify_swiglu_q8(v7d_swiglu_q8_fn func, int matmul_rows, int matmul_cols, int seed, float *out_max_diff)
{
	/* return contract: 1 = comparison passed, 0 = completed numerical
	   comparison failed, -1 = setup error (allocation failure) -- the
	   caller must treat -1 as a hard failure, never as an accepted limit */
	int n_ws = matmul_rows / 64 + 1;
	int w_ws = (int)((__int64)matmul_cols * matmul_rows / 64) + 1;
	char *xq = (char *)malloc(matmul_rows);
	float *xs = (float *)malloc(n_ws * sizeof(float));
	char *gate_wq = (char *)malloc((__int64)matmul_cols * matmul_rows);
	float *gate_ws = (float *)malloc(w_ws * sizeof(float));
	char *up_wq = (char *)malloc((__int64)matmul_cols * matmul_rows);
	float *up_ws = (float *)malloc(w_ws * sizeof(float));
	float *ref = (float *)malloc(matmul_cols * sizeof(float));
	float *asm_ = (float *)malloc(matmul_cols * sizeof(float));
	if ((!xq) || (!xs) || (!gate_wq) || (!gate_ws) || (!up_wq) || (!up_ws) || (!ref) || (!asm_))
	{
		free(xq);
		free(xs);
		free(gate_wq);
		free(gate_ws);
		free(up_wq);
		free(up_ws);
		free(ref);
		free(asm_);
		return -1;
	}
	srand(seed);
	kt_init_random_int8(xq, matmul_rows);
	kt_init_random_float(xs, n_ws);
	kt_init_random_int8(gate_wq, (__int64)matmul_cols * matmul_rows);
	kt_init_random_float(gate_ws, w_ws);
	kt_init_random_int8(up_wq, (__int64)matmul_cols * matmul_rows);
	kt_init_random_float(up_ws, w_ws);
	swiglu_q8_c_ref(ref, xq, xs, gate_wq, gate_ws, up_wq, up_ws, matmul_rows, matmul_cols);
	v7_q8_tensor x_t;
	x_t.q = xq;
	x_t.s = xs;
	v7_q8_tensor gate_t;
	gate_t.q = gate_wq;
	gate_t.s = gate_ws;
	v7_q8_tensor up_t;
	up_t.q = up_wq;
	up_t.s = up_ws;
	memset(asm_, 0, matmul_cols * sizeof(float));
	func(asm_, &x_t, &gate_t, matmul_rows, matmul_cols, &up_t);
	kt_compare_result cr = kt_compare_outputs(ref, asm_, matmul_cols, 0.05f, 1e-4f);
	if (out_max_diff)
		*out_max_diff = cr.max_diff;
	free(xq);
	free(xs);
	free(gate_wq);
	free(gate_ws);
	free(up_wq);
	free(up_ws);
	free(ref);
	free(asm_);
	return cr.pass;
}

/* ======================================================================
 * Benchmarking
 * ====================================================================== */

template <typename Fn> __int64 kt_bench_timed_loop(Fn fn, int iters)
{
	LARGE_INTEGER freq, t0, t1;
	QueryPerformanceFrequency(&freq);
	fn(); /* warm-up */
	__int64 best_us = 999999999;
	for (int run = 0; run < 3; ++run)
	{
		QueryPerformanceCounter(&t0);
		for (int i = 0; i < iters; ++i)
			fn();
		QueryPerformanceCounter(&t1);
		__int64 delta = t1.QuadPart - t0.QuadPart;
		__int64 us_total = delta * 1000000 / freq.QuadPart;
		__int64 us_per = us_total / (__int64)iters;
		if (us_per < best_us)
			best_us = us_per;
	}
	return best_us;
}

static __int64 kt_bench_q8(v7d_q8_fn fn, int matmul_rows, int matmul_cols, int iters, float *xout, kt_tensor_q8 *ts)
{
	v7_q8_tensor x = kt_make_input_q8(ts);
	v7_q8_tensor w = kt_make_weight_q8(ts);
	return kt_bench_timed_loop([&]() { fn(xout, &x, &w, matmul_rows, matmul_cols); }, iters);
}

static __int64 kt_bench_f32(v7d_f32_fn fn, int matmul_rows, int matmul_cols, int iters, float *xout, kt_tensor_f32 *ts)
{
	v7_f32_tensor x = kt_make_input_f32(ts);
	v7_q8_tensor w = kt_make_weight_f32(ts);
	return kt_bench_timed_loop([&]() { fn(xout, &x, &w, matmul_rows, matmul_cols); }, iters);
}

/* ======================================================================
 * Common matmul shape configs (used by validate, benchmark, loadtest)
 * ====================================================================== */

typedef struct kt_shape
{
	int rows;
	int cols;
	const char *label;

} kt_shape;

static kt_shape kt_shapes[] = {
	{ 64, 64, "64x64" },
	{ 128, 128, "128x128" },
	{ 256, 256, "256x256" },
	{ 512, 512, "512x512" },
	{ 1024, 1024, "1024x1024" },
	{ 1024, 512, "1024x512" },
	{ 1024, 2048, "1024x2048" },
	{ 2048, 1024, "2048x1024" },
	{ 1024, 3072, "1024x3072" },
	{ 3072, 1024, "3072x1024" },
	{ 151936, 1024, "embed" },
	{ 1024, 151936, "output" },
	/* Llama 3.2 1B shapes */
	{ 2048, 2048, "Llama Q proj" },
	{ 2048, 512, "Llama K/V proj" },
	{ 2048, 8192, "Llama FFN up" },
	{ 8192, 2048, "Llama FFN down" },
	{ 128256, 2048, "Llama embed" },
	{ 2048, 128256, "Llama output" },
};

static int kt_nshapes = (int)(sizeof(kt_shapes) / sizeof(kt_shapes[0]));
