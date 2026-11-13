// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* attention_shared.h -- shared attention test infrastructure (header-only)
 *
 * Provides: test configs, positions, loffs, reference C attention, load/unload helpers.
 * Requires runstate_alloc.h (v7_run_state/v7_config via v7_lib.h) and v7_lib.h (DLL loading).
 *
 * Usage (in each attention validator .cpp):
 *   #include "v7_lib.h"
 *   #include "runstate_alloc.h"
 *   #include "attention_shared.h"
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

/* ======================================================================
 * Packed FP16 KV cache helpers
 *
 * The engine stores persistent K/V as IEEE FP16 (2 bytes/element). The
 * test infrastructure mirrors that layout: the cache buffers are allocated
 * for cache_slots halves, filled with half-rounded random values, and the
 * reference expands each half back to float before computing attention.
 *
 * Conversions are done in portable scalar bit arithmetic. The MSVC
 * _mm_cvtsh_ss / _mm_cvtps_ph scalar intrinsics lower to AVX512-FP16
 * (vcvtsh2ss) on this toolchain, which the target CPU lacks (#UD 0xC000001D),
 * so the intrinsic path is not usable here.
 * ====================================================================== */

/* Float -> IEEE FP16 bits, round-to-nearest-even (matches vcvtps2ph imm 0). */
static uint16_t attn_float_to_half(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	uint32_t sign = (bits >> 16) & 0x8000u;
	uint32_t exponent = (bits >> 23) & 0xFFu;
	uint32_t mantissa = bits & 0x7FFFFFu;

	if (exponent == 0xFFu) /* Inf / NaN */
	{
		uint16_t nanoise = (uint16_t)((mantissa != 0) ? 0x200u : 0);
		return (uint16_t)(sign | 0x7C00u | nanoise);
	}
	int new_exp = (int)exponent - 127 + 15; /* rebias to FP16 */
	if (new_exp >= 31) /* overflow -> Inf */
	{
		return (uint16_t)(sign | 0x7C00u);
	}
	if (new_exp <= 0) /* subnormal or zero */
	{
		if (new_exp < -10)
		{
			return (uint16_t)sign;
		}
		mantissa |= 0x800000u; /* implicit 1 */
		int shift = 14 - new_exp;
		uint32_t half_mant = mantissa >> shift;
		uint32_t rem = mantissa & ((1u << shift) - 1);
		uint32_t halfway = 1u << (shift - 1);
		if ((rem > halfway) || ((rem == halfway) && ((half_mant & 1u) != 0)))
		{
			half_mant++; /* round to nearest, ties to even */
		}
		return (uint16_t)(sign | half_mant);
	}
	uint32_t half_mant = mantissa >> 13;
	uint32_t rem = mantissa & 0x1FFFu;
	if ((rem > 0x1000u) || ((rem == 0x1000u) && ((half_mant & 1u) != 0)))
	{
		half_mant++; /* round to nearest, ties to even */
		if (half_mant == 0x400u) /* mantissa overflow into the exponent */
		{
			half_mant = 0;
			new_exp++;
			if (new_exp >= 31)
			{
				return (uint16_t)(sign | 0x7C00u);
			}
		}
	}
	return (uint16_t)(sign | ((uint32_t)new_exp << 10) | half_mant);
}

/* IEEE FP16 bits -> float, exact widening. */
static float attn_half_to_float(uint16_t bits)
{
	uint32_t sign = (uint32_t)(bits >> 15) << 31;
	uint32_t exponent = (bits >> 10) & 0x1Fu;
	uint32_t mantissa = bits & 0x3FFu;
	uint32_t out;

	if (exponent == 0)
	{
		if (mantissa == 0)
		{
			out = sign; /* signed zero */
		}
		else /* subnormal: normalize into an FP32 normal */
		{
			int shift = 0;
			while ((mantissa & 0x400u) == 0)
			{
				mantissa <<= 1;
				shift++;
			}
			mantissa &= 0x3FFu;
			uint32_t fp32_exp = (uint32_t)(127 - 15 - shift);
			out = sign | (fp32_exp << 23) | (mantissa << 13);
		}
	}
	else if (exponent == 0x1Fu) /* Inf / NaN */
	{
		out = sign | 0x7F800000u | (mantissa << 13);
	}
	else
	{
		out = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
	}
	float result;
	memcpy(&result, &out, sizeof(result));
	return result;
}

/* Cache slot count in ELEMENTS for the packed FP16 layout. */
static __inline int attn_cache_elems(v7_config *config_pointer, int max_seq_len, int cache_layers)
{
	int kv_dim = config_pointer->n_kv_heads * config_pointer->head_dim;
	return (cache_layers + 1) * max_seq_len * kv_dim;
}

/* ======================================================================
 * Constants
 * ====================================================================== */

/* How many extra "layers" of cache to allocate for non-zero loff tests.
   loff = layer * seq_len * kv_dim, so ATTN_MAX_LOFF_LAYERS=4 means we can
   test layers 0..3 (loff up to 3 * seq_len * kv_dim). */
#define ATTN_MAX_LOFF_LAYERS 4
#define ATTN_MAX_SEQ_LEN 64

/* ======================================================================
 * Test configs (covers all Qwen3-relevant configurations + edge cases)
 *
 * NOTE: All head_dims MUST be multiples of 32. The v7 attention kernel now
 * FATALs on head_dim % 32 != 0 -- its scalar tails are dead code, and every
 * supported model uses head_dim 64/96/128. Non-multiple-of-32 test configs
 * (7, 8, 15, 16, 24, 31, 48, 63, 65, 95, 127, 129, 255) were removed.
 * ====================================================================== */

typedef struct attn_test_config
{
	int n_heads;
	int n_kv_heads;
	int head_dim;
	int hidden_mult; /* hidden_dim = dim * hidden_mult */

} attn_test_config;

static attn_test_config attn_configs[] = {
	/* Real model attention configs only (from weights/ .bin headers).
	 * head_dims must be multiples of 32 (kernel contract). */

	/* --- Qwen3-0.6B: 16 heads, 8 kv_heads, hd=128 --- */
	{ 16, 8, 128, 4 },

	/* --- Llama 3.2 1B: 32 heads, 8 kv_heads, hd=64 --- */
	{ 32, 8, 64, 4 },

	/* --- SmolLM2-135M: 9 heads, 3 kv_heads, hd=64 --- */
	{ 9, 3, 64, 4 },

	/* --- SmolLM2-360M: 15 heads, 5 kv_heads, hd=64 --- */
	{ 15, 5, 64, 4 },

	/* --- TinyLlama-1.1B: 32 heads, 4 kv_heads, hd=64 --- */
	{ 32, 4, 64, 4 },

	/* --- Phi-3-mini: 32 heads MHA, hd=96 (16-float dot remainder) --- */
	{ 32, 32, 96, 4 },

	/* --- MiniCPM5-1B: 16 heads, 2 kv_heads, hd=128 --- */
	{ 16, 2, 128, 4 },

	/* --- Qwen2.5-0.5B: 14 heads, 2 kv_heads, hd=64 --- */
	{ 14, 2, 64, 4 },

	/* --- sentinel --- */
	{ 0, 0, 0, 0 }
};

/* ======================================================================
 * Layer offsets (which layer indices to test for non-zero loff)
 * ====================================================================== */

static int attn_loffs[] = {
	0, /* layer 0 */
	2, /* layer 2 */
};

/* ======================================================================
 * Test positions (full range from single-timestep to max sequence length)
 * ====================================================================== */

static int attn_positions[] = { 0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 62, 63 };

/* ======================================================================
 * Reference C attention (pure double-precision, no SIMD)
 * ====================================================================== */

static void attn_reference(v7_run_state *state_pointer, v7_config *config_pointer, int pos, unsigned __int64 loff, int kv_dim, int kv_mul)
{
	int head_dim = config_pointer->head_dim;
	double scale = 1.0 / sqrt((double)head_dim);

	for (int h = 0; h < config_pointer->n_heads; h++)
	{
		float *q = state_pointer->q + h * head_dim;
		float *xb = state_pointer->xb + h * head_dim;
		int kv_head = (h / kv_mul) * head_dim;
		uint16_t *key_half = (uint16_t *)state_pointer->key_cache;
		uint16_t *value_half = (uint16_t *)state_pointer->value_cache;

		double att[64]; /* max pos=63, so <=64 entries */

		/* 1. Scores: q dot k_cache[t] (double precision, K expanded from FP16) */
		for (int t = 0; t <= pos; t++)
		{
			uint16_t *k = key_half + loff + t * kv_dim + kv_head;
			double score = 0.0;
			for (int i = 0; i < head_dim; i++)
				score += (double)q[i] * (double)attn_half_to_float(k[i]);
			att[t] = score * scale;
		}

		/* 2. Softmax (double precision) */
		double max_val = att[0];
		for (int i = 1; i <= pos; i++)
			if (att[i] > max_val)
				max_val = att[i];
		double sum = 0.0;
		for (int i = 0; i <= pos; i++)
		{
			att[i] = exp(att[i] - max_val);
			sum += att[i];
		}
		double inv = 1.0 / sum;
		for (int i = 0; i <= pos; i++)
			att[i] *= inv;

		/* 3. Weighted sum into xb[h] (double precision, convert at store) */
		memset(xb, 0, head_dim * sizeof(float));
		for (int t = 0; t <= pos; t++)
		{
			uint16_t *v = value_half + loff + t * kv_dim + kv_head;
			double a = att[t];
			for (int i = 0; i < head_dim; i++)
				xb[i] += (float)(a * (double)attn_half_to_float(v[i]));
		}
	}
}

/* ======================================================================
 * DLL load / unload wrapper (uses v7_lib.h)
 * ====================================================================== */

typedef struct attn_dll
{
	v7_lib lib;
	int loaded;

} attn_dll;

static int attn_dll_load(attn_dll *d, char *dll_path, char *db_path)
{
	memset(d, 0, sizeof(*d));
	if ((v7_load(dll_path, &d->lib) != 0))
	{
		printf("error: could not load %s\n", dll_path);
		return 0;
	}
	/* Load the DB so dispatch params and __ini__* keys are available */
	if (!d->lib.v7_init(db_path) || d->lib.v7_engine_is_dead())
	{
		/* validating against an unloaded DB would measure empty gate
		 * returns -- a failed load must be named and nonzero. */
		printf("error: v7_init failed for %s", db_path);
		if (d->lib.v7_engine_is_dead() && d->lib.v7_get_death_reason())
			printf(" -- %s", d->lib.v7_get_death_reason());
		printf("\n");
		return 0;
	}
	/* Attention forks for positions 16..96: declare the active phase so the
	 * fork publication always reaches pause-spinning workers. */
	d->lib.v7s_set_spin(1);
	d->loaded = 1;
	return 1;
}

static void attn_dll_unload(attn_dll *d)
{
	d->lib.v7s_set_spin(0);
	d->lib.v7_shutdown();
	v7_unload(&d->lib);
	d->loaded = 0;
}

/* ======================================================================
 * Random float generator
 * ====================================================================== */

static void attn_init_random_float(float *data, int count)
{
	for (int i = 0; i < count; i++)
		data[i] = (float)rand() / (float)RAND_MAX;
}

static void attn_fill_random(v7_run_state *state_pointer, v7_config *config_pointer, int max_seq_len, int cache_layers)
{
	int dim = config_pointer->dim;
	int cache_elems = attn_cache_elems(config_pointer, max_seq_len, cache_layers);
	uint16_t *key_half = (uint16_t *)state_pointer->key_cache;
	uint16_t *value_half = (uint16_t *)state_pointer->value_cache;
	attn_init_random_float(state_pointer->x, dim);
	attn_init_random_float(state_pointer->xb, dim);
	attn_init_random_float(state_pointer->q, dim);
	attn_init_random_float(state_pointer->k, dim);
	attn_init_random_float(state_pointer->v, dim);
	attn_init_random_float(state_pointer->att, config_pointer->n_heads * max_seq_len);
	for (int i = 0; i < cache_elems; i++)
	{
		key_half[i] = attn_float_to_half((float)rand() / (float)RAND_MAX);
		value_half[i] = attn_float_to_half((float)rand() / (float)RAND_MAX);
	}
}
