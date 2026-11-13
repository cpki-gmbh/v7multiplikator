// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

// q8_batch4.h -- temporary internal header for the experimental four-token
// Q8 batch kernel (speed_fix plan Phase 1). Flat, fixed-layout argument block

#ifndef V7_ABI
#ifdef __linux__
#define V7_ABI __attribute__((ms_abi))
#else
#define V7_ABI
#endif
#endif
// so the NASM side has an unambiguous ABI; the offsets below are mirrored by
// the Q8B_* %defines in src/asm/q8_batch4_kernel.inc and MUST stay in sync.
// Output buffers are four independent float[d] arrays. Every activation has
// its own q[n] data and s[n/64] scales. Weights are row-major q[d*n] with
// s[d*(n/64)] in row order.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct v7_q8_batch4_args
{
	float *out[4];            // 0..31: four output row arrays
	const int8_t *xq[4];      // 32..63: four quantized inputs
	const float *xs[4];       // 64..95: four activation scale arrays
	const int8_t *wq;         // 96: contiguous row-major Q8 weights
	const float *ws;          // 104: weight scales, one per 64 values
	uint32_t n;               // 112: elements per input; multiple of 64
	uint32_t d;               // 116: output row count
} v7_q8_batch4_args;

typedef void (V7_ABI *v7_q8_batch4_fn)(const v7_q8_batch4_args *args);
typedef void (V7_ABI *v7d_q8_batch4_fn)(const v7_q8_batch4_args *args); /* dispatched entry, same ABI as the raw kernel, splits the rows across the worker pool internally */

#ifdef __cplusplus
}
#endif
