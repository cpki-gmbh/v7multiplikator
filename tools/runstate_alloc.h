// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* runstate_alloc.h -- RunState alloc/free helpers (header-only)
 *
 * Used by: attention validators, fused/validate/matmul validators that need RunState.
 * v7_lib.h MUST be included BEFORE this header so v7_run_state/v7_config are defined.
 */

#pragma once

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <malloc.h>

/* Allocate v7_run_state buffers. cache_layers=0 gives a single layer. */
static v7_run_state *rs_alloc(v7_config *config_pointer, int max_seq_len, int cache_layers)
{
	v7_run_state *state_pointer = (v7_run_state *)calloc(1, sizeof(v7_run_state));
	if (!state_pointer)
		return 0;
	int dim = config_pointer->dim;
	int kv_dim = config_pointer->n_kv_heads * config_pointer->head_dim;
	int hidden_dim = config_pointer->hidden_dim;
	int cache_slots = (cache_layers + 1) * max_seq_len * kv_dim;
	size_t cache_bytes = (size_t)cache_slots * sizeof(uint16_t); /* persistent K/V are packed FP16 */

	state_pointer->x = (float *)_aligned_malloc(dim * sizeof(float), 32);
	state_pointer->xb = (float *)_aligned_malloc(dim * sizeof(float), 32);
	state_pointer->hb = (float *)_aligned_malloc(hidden_dim * sizeof(float), 32);
	state_pointer->hb2 = (float *)_aligned_malloc(hidden_dim * sizeof(float), 32);
	state_pointer->q = (float *)_aligned_malloc(dim * sizeof(float), 32);
	state_pointer->k = (float *)_aligned_malloc(dim * sizeof(float), 32);
	state_pointer->v = (float *)_aligned_malloc(dim * sizeof(float), 32);
	state_pointer->att = (float *)_aligned_malloc(config_pointer->n_heads * max_seq_len * sizeof(float), 32);
	state_pointer->logits = (float *)_aligned_malloc(config_pointer->vocab_size * sizeof(float), 32);
	state_pointer->key_cache = (float *)_aligned_malloc(cache_bytes, 32);
	state_pointer->value_cache = (float *)_aligned_malloc(cache_bytes, 32);

	if ((!state_pointer->x) || (!state_pointer->xb) || (!state_pointer->hb) || (!state_pointer->hb2) || (!state_pointer->q) || (!state_pointer->k) || (!state_pointer->v) || (!state_pointer->att) || (!state_pointer->logits) || (!state_pointer->key_cache) || (!state_pointer->value_cache))
	{
		_aligned_free(state_pointer->x);
		_aligned_free(state_pointer->xb);
		_aligned_free(state_pointer->hb);
		_aligned_free(state_pointer->hb2);
		_aligned_free(state_pointer->q);
		_aligned_free(state_pointer->k);
		_aligned_free(state_pointer->v);
		_aligned_free(state_pointer->att);
		_aligned_free(state_pointer->logits);
		_aligned_free(state_pointer->key_cache);
		_aligned_free(state_pointer->value_cache);
		free(state_pointer);
		return 0;
	}
	return state_pointer;
}

static void rs_free(v7_run_state *state_pointer)
{
	_aligned_free(state_pointer->x);
	_aligned_free(state_pointer->xb);
	_aligned_free(state_pointer->hb);
	_aligned_free(state_pointer->hb2);
	_aligned_free(state_pointer->q);
	_aligned_free(state_pointer->k);
	_aligned_free(state_pointer->v);
	_aligned_free(state_pointer->att);
	_aligned_free(state_pointer->logits);
	_aligned_free(state_pointer->key_cache);
	_aligned_free(state_pointer->value_cache);
	free(state_pointer);
}

static void rs_copy(v7_run_state *dst, v7_run_state *src, v7_config *config_pointer, int max_seq_len, int cache_layers)
{
	int dim = config_pointer->dim;
	int hidden_dim = config_pointer->hidden_dim;
	int kv_dim = config_pointer->n_kv_heads * config_pointer->head_dim;
	int cache_slots = (cache_layers + 1) * max_seq_len * kv_dim;
	size_t cache_bytes = (size_t)cache_slots * sizeof(uint16_t); /* persistent K/V are packed FP16 */
	memcpy(dst->x, src->x, dim * sizeof(float));
	memcpy(dst->xb, src->xb, dim * sizeof(float));
	memcpy(dst->hb, src->hb, hidden_dim * sizeof(float));
	memcpy(dst->hb2, src->hb2, hidden_dim * sizeof(float));
	memcpy(dst->q, src->q, dim * sizeof(float));
	memcpy(dst->k, src->k, dim * sizeof(float));
	memcpy(dst->v, src->v, dim * sizeof(float));
	memcpy(dst->att, src->att, config_pointer->n_heads * max_seq_len * sizeof(float));
	memcpy(dst->logits, src->logits, config_pointer->vocab_size * sizeof(float));
	memcpy(dst->key_cache, src->key_cache, cache_bytes);
	memcpy(dst->value_cache, src->value_cache, cache_bytes);
}

static void init_random_float(float *data, int count)
{
	for (int i = 0; i < count; i++)
		data[i] = (float)rand() / (float)RAND_MAX;
}

/* Float -> IEEE FP16 bits, round-to-nearest-even. Portable scalar bit math:
 * the MSVC scalar FP16 intrinsics lower to AVX512-FP16 on this toolchain,
 * which the target CPU lacks. */
static uint16_t rs_float_to_half(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	uint32_t sign = (bits >> 16) & 0x8000u;
	uint32_t exponent = (bits >> 23) & 0xFFu;
	uint32_t mantissa = bits & 0x7FFFFFu;
	if (exponent == 0xFFu)
	{
		uint16_t nanoise = (uint16_t)((mantissa != 0) ? 0x200u : 0);
		return (uint16_t)(sign | 0x7C00u | nanoise);
	}
	int new_exp = (int)exponent - 127 + 15;
	if (new_exp >= 31)
		return (uint16_t)(sign | 0x7C00u);
	if (new_exp <= 0)
	{
		if (new_exp < -10)
			return (uint16_t)sign;
		mantissa |= 0x800000u;
		int shift = 14 - new_exp;
		uint32_t half_mant = mantissa >> shift;
		uint32_t rem = mantissa & ((1u << shift) - 1);
		uint32_t halfway = 1u << (shift - 1);
		if ((rem > halfway) || ((rem == halfway) && ((half_mant & 1u) != 0)))
			half_mant++;
		return (uint16_t)(sign | half_mant);
	}
	uint32_t half_mant = mantissa >> 13;
	uint32_t rem = mantissa & 0x1FFFu;
	if ((rem > 0x1000u) || ((rem == 0x1000u) && ((half_mant & 1u) != 0)))
	{
		half_mant++;
		if (half_mant == 0x400u)
		{
			half_mant = 0;
			new_exp++;
			if (new_exp >= 31)
				return (uint16_t)(sign | 0x7C00u);
		}
	}
	return (uint16_t)(sign | ((uint32_t)new_exp << 10) | half_mant);
}

static void rs_fill_random(v7_run_state *state_pointer, v7_config *config_pointer, int max_seq_len, int cache_layers)
{
	int dim = config_pointer->dim;
	int kv_dim = config_pointer->n_kv_heads * config_pointer->head_dim;
	int cache_elems = (cache_layers + 1) * max_seq_len * kv_dim;
	uint16_t *key_half = (uint16_t *)state_pointer->key_cache;
	uint16_t *value_half = (uint16_t *)state_pointer->value_cache;
	init_random_float(state_pointer->x, dim);
	init_random_float(state_pointer->xb, dim);
	init_random_float(state_pointer->q, dim);
	init_random_float(state_pointer->k, dim);
	init_random_float(state_pointer->v, dim);
	init_random_float(state_pointer->att, config_pointer->n_heads * max_seq_len);
	for (int i = 0; i < cache_elems; i++)
	{
		key_half[i] = rs_float_to_half((float)rand() / (float)RAND_MAX);
		value_half[i] = rs_float_to_half((float)rand() / (float)RAND_MAX);
	}
}
