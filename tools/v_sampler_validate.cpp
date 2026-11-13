// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v_sampler_validate.cpp -- Standalone ASM sampler correctness validator
 *
 * Tests v7s_sample (direct-call, no V7_init needed) against C reference sampler.
 * Greedy must match exactly; temp+multinomial and temp+top-p use same RNG seed.
 * Also pins the public-API setter data-fixing contract (NaN/sub-epsilon ->
 * 1e-6f, 0.0 verbatim) against a live engine context.
 * Usage: v_sampler_validate.exe <dll_path> <db_path>
 */
#include "v7_lib.h"
#ifdef __linux__
#include <windows.h> /* Linux tool adapters */
#endif
#include "zydis_debug_header.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>
#include <immintrin.h>

#define VOCAB_SIZE 151936
#define MAX_TESTS 40
#define SAMPLER_TOPK_LIMIT 256

/* Reference sampler */
static unsigned int ref_random_u32(unsigned __int64 *state)
{
	unsigned __int64 current = *state;
	current += 0x9e3779b97f4a7c15ULL;
	*state = current;
	__m128i block = _mm_set_epi64x(0xbf58476d1ce4e5b9ULL, current);
	__m128i key = _mm_set_epi64x(0x94d049bb133111ebULL, 0x1bf95cfc92db11ebULL);
	__m128i enc = _mm_aesenc_si128(block, key);
	return (unsigned int)_mm_cvtsi128_si32(enc);
}
static float ref_random_f32(unsigned __int64 *state)
{
	unsigned int bits = ref_random_u32(state);
	bits &= 0x007FFFFF;
	bits |= 0x3F800000;
	float f;
	memcpy(&f, &bits, sizeof(float));
	return f - 1.0f;
}
static int ref_sample_argmax(float *probs, int num_probs)
{
	int max_i = 0;
	float max_p = probs[0];
	for (int i = 1; i < num_probs; i++)
	{
		if (probs[i] > max_p)
		{
			max_i = i;
			max_p = probs[i];
		}
	}
	return max_i;
}
static int ref_sample_mult(float *probs, int num_probs, float coin)
{
	float cdf = 0;
	for (int i = 0; i < num_probs; i++)
	{
		cdf += probs[i];
		if (coin < cdf)
			return i;
	}
	return num_probs - 1;
}
/* Bit-exact exp: the ASM polynomial (sampler.inc s_exp256_soft / scalar tail).
   Lower clamp to -87.3365, n = rint(x*log2e), f = x - n*ln2 (two FNMADDs,
   single rounding each), minimax poly c6..c0 via fmaf chain, scale by 2^n. */
static float ref_exp_poly(float x)
{
	if (x < -87.3365f)
		x = -87.3365f;
	float xd = x * 1.4426950408889634f;
	float nf = rintf(xd);
	int n = (int)nf;
	float f = fmaf(-nf, 6.93145751953125e-1f, x);
	f = fmaf(-nf, 1.42860677302611e-6f, f);
	float c = fmaf(f, 1.3888888888888889e-3f, 8.3333333333333333e-3f);
	c = fmaf(f, c, 4.1666666666666667e-2f);
	c = fmaf(f, c, 1.6666666666666667e-1f);
	c = fmaf(f, c, 0.5f);
	c = fmaf(f, c, 1.0f);
	c = fmaf(f, c, 1.0f);
	return c * scalbnf(1.0f, n);
}
/* Sift-down port (sampler.inc SIFT_DOWN macro): carries the root entry,
   per-level full-entry swaps, exact tie semantics. is_min selects heap kind. */
static void ref_sift_down(v7_probindex *h, int root, int end, int is_min)
{
	v7_probindex carried = h[root];
	for (;;)
	{
		int child = 2 * root + 1;
		if (child > end)
			break;
		if (child + 1 <= end)
		{
			float left = h[child].prob;
			float right = h[child + 1].prob;
			if (is_min)
			{
				if (right < left)
					child++;
			}
			else
			{
				if (right > left)
					child++;
			}
		}
		float child_prob = h[child].prob;
		if (is_min)
		{
			if (carried.prob <= child_prob)
				break;
		}
		else
		{
			if (carried.prob >= child_prob)
				break;
		}
		h[root] = h[child];
		root = child;
	}
	h[root] = carried;
}
static int ref_sample_topp(float *probs, int num_probs, float topp, v7_probindex *pi, float coin)
{
	/* Exact port of the ASM nucleus path (sampler.inc .s_do_topp): min-heap
	   cutoff of the TOPK_LIMIT largest, max-heap extraction until the running
	   cumulative reaches topp (the SMALLEST such prefix), then a descending
	   walk with r = coin * cum. */
	int n0 = 0;
	int i = 0;
	for (; i < num_probs; i++)
	{
		float p = probs[i];
		if (n0 < SAMPLER_TOPK_LIMIT)
		{
			pi[n0].prob = p;
			pi[n0].index = i;
			n0++;
			if (n0 == SAMPLER_TOPK_LIMIT)
			{
				for (int s = (SAMPLER_TOPK_LIMIT - 2) / 2; s >= 0; s--)
					ref_sift_down(pi, s, SAMPLER_TOPK_LIMIT - 1, 1);
			}
		}
		else if (p > pi[0].prob)
		{
			pi[0].prob = p;
			pi[0].index = i;
			ref_sift_down(pi, 0, SAMPLER_TOPK_LIMIT - 1, 1);
		}
	}
	if (n0 == 0)
		return 0;
	for (int s = (n0 - 2) / 2; s >= 0; s--)
		ref_sift_down(pi, s, n0 - 1, 0);
	int heap_end = n0;
	float cumulative = 0.0f;
	while (heap_end > 0)
	{
		cumulative += pi[0].prob;
		heap_end--;
		v7_probindex tmp = pi[0];
		pi[0] = pi[heap_end];
		pi[heap_end] = tmp;
		if (heap_end > 0)
			ref_sift_down(pi, 0, heap_end - 1, 0);
		if (cumulative >= topp)
			break;
	}
	if (cumulative < topp)
	{
		/* Wide-nucleus fallback (mirrors the ASM .s_topp_extract_done fallback):
		 * the 256-entry heap truncated a nucleus wider than TOPK_LIMIT. The
		 * full vocab rebuilds into the same workspace and the extraction
		 * re-runs to the requested mass. */
		n0 = num_probs;
		for (int i2 = 0; i2 < n0; i2++)
		{
			pi[i2].prob = probs[i2];
			pi[i2].index = i2;
		}
		for (int s = (n0 - 2) / 2; s >= 0; s--)
			ref_sift_down(pi, s, n0 - 1, 0);
		heap_end = n0;
		cumulative = 0.0f;
		while (heap_end > 0)
		{
			cumulative += pi[0].prob;
			heap_end--;
			v7_probindex tmp = pi[0];
			pi[0] = pi[heap_end];
			pi[heap_end] = tmp;
			if (heap_end > 0)
				ref_sift_down(pi, 0, heap_end - 1, 0);
			if (cumulative >= topp)
				break;
		}
	}
	float r = coin * cumulative;
	float cdf = 0.0f;
	for (int j = n0 - 1; j >= heap_end; j--)
	{
		cdf += pi[j].prob;
		if (r < cdf)
			return pi[j].index;
	}
	return pi[heap_end].index;
}
/* Bit-exact softmax port (sampler.inc .s_temp_nonzero..): the sum replicates
   the 4x8-lane accumulator split, the (a0+a1)+(a2+a3) lane tree and the
   hsum256 lane pairing, then the scalar tail adds sequentially. */
static void ref_softmax_inplace(float *x, int num_elements)
{
	float max_val = x[0];
	for (int i = 1; i < num_elements; i++)
		if (x[i] > max_val)
			max_val = x[i];
	float acc[4][8];
	for (int j = 0; j < 8; j++)
	{
		acc[0][j] = 0.0f;
		acc[1][j] = 0.0f;
		acc[2][j] = 0.0f;
		acc[3][j] = 0.0f;
	}
	int vc = num_elements & ~7;
	int ub = vc & ~31;
	int i = 0;
	for (; i < ub; i += 8)
	{
		int a = (i >> 3) & 3;
		for (int j = 0; j < 8; j++)
		{
			float e = ref_exp_poly(x[i + j] - max_val);
			x[i + j] = e;
			acc[a][j] += e;
		}
	}
	for (; i < vc; i += 8)
	{
		for (int j = 0; j < 8; j++)
		{
			float e = ref_exp_poly(x[i + j] - max_val);
			x[i + j] = e;
			acc[0][j] += e;
		}
	}
	float b0 = (acc[0][0] + acc[1][0]) + (acc[2][0] + acc[3][0]);
	float b1 = (acc[0][1] + acc[1][1]) + (acc[2][1] + acc[3][1]);
	float b2 = (acc[0][2] + acc[1][2]) + (acc[2][2] + acc[3][2]);
	float b3 = (acc[0][3] + acc[1][3]) + (acc[2][3] + acc[3][3]);
	float b4 = (acc[0][4] + acc[1][4]) + (acc[2][4] + acc[3][4]);
	float b5 = (acc[0][5] + acc[1][5]) + (acc[2][5] + acc[3][5]);
	float b6 = (acc[0][6] + acc[1][6]) + (acc[2][6] + acc[3][6]);
	float b7 = (acc[0][7] + acc[1][7]) + (acc[2][7] + acc[3][7]);
	float l0 = b0 + b4;
	float l1 = b1 + b5;
	float l2 = b2 + b6;
	float l3 = b3 + b7;
	float sum = (l0 + l1) + (l2 + l3);
	for (; i < num_elements; i++)
	{
		float e = ref_exp_poly(x[i] - max_val);
		x[i] = e;
		sum += e;
	}
	if (sum != sum)
	{
		printf("reference softmax sum is NaN -- logits drifted\n");
		exit(1);
	}
	if (sum < 1e-6f)
		sum = 1e-6f;
	for (i = 0; i < num_elements; i++)
		x[i] /= sum;
}
static int ref_sample(float *logits, int vocab_size, float temperature, float topp, unsigned __int64 *rng_state, v7_probindex *probindex)
{
	if (temperature == 0.0f)
		return ref_sample_argmax(logits, vocab_size);
	if (temperature < 1e-6f)
		temperature = 1e-6f;
	for (int q = 0; q < vocab_size; q++)
	{
		float v = logits[q] / temperature;
		logits[q] = v;
	}
	ref_softmax_inplace(logits, vocab_size);
	float coin = ref_random_f32(rng_state);
	if (((topp <= 0.0f) || (topp >= 1.0f)))
		return ref_sample_mult(logits, vocab_size, coin);
	return ref_sample_topp(logits, vocab_size, topp, probindex, coin);
}

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("usage: v_sampler_validate.exe <dll_path> <db_path>\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if ((v7_load(argv[1], &lib) != 0))
	{
		printf("v7_load failed\n");
		return 1;
	}
	printf("... sampler_validate: %s ...\n", argv[1]);

	static const unsigned int seeds[] = { 42, 12345, 9999, 7, 2024, 31415, 27182, 11, 99, 777 };
	int n_seeds = (int)(sizeof(seeds) / sizeof(seeds[0]));

	int pass = 0;
	int total = 0;

	/* Setter data-fixing contract (the public API setter is the sanctioned
	 * data-fixing boundary): NaN and sub-epsilon temperatures must be fixed
	 * to 1e-6f; 0.0f must store verbatim (greedy). Requires a live engine
	 * context (the setter FATALs without one), hence the db_path argument. */
	if (!lib.v7_init(argv[2]))
	{
		printf("v7_init(%s) failed: %s\n", argv[2], lib.v7_engine_is_dead() ? lib.v7_get_death_reason() : "returned 0");
		v7_unload(&lib);
		return 1;
	}
	if (!lib.v7_initialize_inference())
	{
		printf("v7_initialize_inference failed: %s\n", lib.v7_engine_is_dead() ? lib.v7_get_death_reason() : "returned 0");
		v7_unload(&lib);
		return 1;
	}
	{
		int setter_pass = 0;
		int setter_total = 0;
		inference_context *ctx = (inference_context *)lib.v7_get_internal_inference_ctx();
		char *sampler_base = *((char **)((char *)ctx + 16)); /* CTX_SAMPLER = 16 */
		volatile unsigned int *temp_bits = (volatile unsigned int *)(sampler_base + 16); /* SAM_TEMPERATURE = 16 */

		lib.v7_set_sampler_temperature(0.0f);
		++setter_total;
		if (*temp_bits == 0x00000000u)
			++setter_pass;
		else
			printf("\n  [setter] t=0.0: stored bits 0x%08X, expected 0x00000000\n", *temp_bits);

		lib.v7_set_sampler_temperature(std::numeric_limits<float>::quiet_NaN());
		++setter_total;
		if (*temp_bits == 0x358637BDu)
			++setter_pass;
		else
			printf("\n  [setter] t=NaN: stored bits 0x%08X, expected 0x358637BD (1e-6f)\n", *temp_bits);

		lib.v7_set_sampler_temperature(2.5e-7f);
		++setter_total;
		if (*temp_bits == 0x358637BDu)
			++setter_pass;
		else
			printf("\n  [setter] t=2.5e-7: stored bits 0x%08X, expected 0x358637BD (1e-6f)\n", *temp_bits);

		lib.v7_set_sampler_temperature(0.7f);
		++setter_total;
		if (*temp_bits == 0x3F333333u)
			++setter_pass;
		else
			printf("\n  [setter] t=0.7: stored bits 0x%08X, expected 0x3F333333\n", *temp_bits);

		/* negative and -inf temperatures fix up to 1e-6f exactly like NaN:
		 * the setter is the sanctioned data-fixing boundary, and the signed
		 * vmaxss floor maps every sub-epsilon value (including negatives)
		 * to the sampling floor. +inf is the one refusal -- see below. */
		lib.v7_set_sampler_temperature(-1.0f);
		++setter_total;
		if (*temp_bits == 0x358637BDu)
			++setter_pass;
		else
			printf("\n  [setter] t=-1.0: stored bits 0x%08X, expected 0x358637BD (1e-6f fix-up)\n", *temp_bits);

		lib.v7_set_sampler_temperature(-std::numeric_limits<float>::infinity());
		++setter_total;
		if (*temp_bits == 0x358637BDu)
			++setter_pass;
		else
			printf("\n  [setter] t=-inf: stored bits 0x%08X, expected 0x358637BD (1e-6f fix-up)\n", *temp_bits);

		/* +inf refusal: the border contract rejects a temperature that would
		 * divide every logit to zero. The entry FATALs (engine parks, host
		 * informed); free_inference resets the death state so the section's
		 * final teardown stays balanced. Runs LAST: it ends the live context. */
		lib.v7_set_sampler_temperature(std::numeric_limits<float>::infinity());
		++setter_total;
		if (lib.v7_engine_is_dead())
		{
			const char *reason = lib.v7_get_death_reason();
			if (strstr(reason, "finite value"))
				++setter_pass;
			else
				printf("\n  [setter] t=+inf: engine died with unexpected reason '%s'\n", reason);
		}
		else
			printf("\n  [setter] t=+inf: engine still alive, expected a border FATAL\n");
		lib.v7_free_inference();

		printf("setter contract: %d/%d passed\n", setter_pass, setter_total);
		pass += setter_pass;
		total += setter_total;
	}
	{
		/* Presence-penalty border contract: NaN and out-of-range magnitudes
		 * refuse at the entry; a value inside [-2, 2] stores verbatim.
		 * free_inference destroyed the hash table along with the context, so
		 * the DB is re-opened before a new context is created. */
		if (!lib.v7_init(argv[2]))
		{
			printf("v7_init (penalty section) failed: %s\n", lib.v7_engine_is_dead() ? lib.v7_get_death_reason() : "returned 0");
			v7_unload(&lib);
			return 1;
		}
		if (!lib.v7_initialize_inference())
		{
			printf("v7_initialize_inference (penalty section) failed: %s\n", lib.v7_engine_is_dead() ? lib.v7_get_death_reason() : "returned 0");
			v7_unload(&lib);
			return 1;
		}
		int pen_pass = 0;
		int pen_total = 0;
		inference_context *ctx = (inference_context *)lib.v7_get_internal_inference_ctx();
		char *sampler_base = *((char **)((char *)ctx + 16)); /* CTX_SAMPLER = 16 */
		volatile unsigned int *pen_bits = (volatile unsigned int *)(sampler_base + 24); /* SAM_PRESENCE_PENALTY = 24 */

		lib.v7_set_sampler_presence_penalty(0.25f);
		++pen_total;
		if (*pen_bits == 0x3E800000u)
			++pen_pass;
		else
			printf("\n  [penalty] p=0.25: stored bits 0x%08X, expected 0x3E800000\n", *pen_bits);

		lib.v7_set_sampler_presence_penalty(2.0f);
		++pen_total;
		if (*pen_bits == 0x40000000u)
			++pen_pass;
		else
			printf("\n  [penalty] p=2.0: stored bits 0x%08X, expected 0x40000000 (range edge stays legal)\n", *pen_bits);

		lib.v7_set_sampler_presence_penalty(std::numeric_limits<float>::quiet_NaN());
		++pen_total;
		if (lib.v7_engine_is_dead())
		{
			if (strstr(lib.v7_get_death_reason(), "finite value"))
				++pen_pass;
			else
				printf("\n  [penalty] p=NaN: engine died with unexpected reason '%s'\n", lib.v7_get_death_reason());
		}
		else
			printf("\n  [penalty] p=NaN: engine still alive, expected a border FATAL\n");
		lib.v7_free_inference();

		if (!lib.v7_init(argv[2]))
		{
			printf("v7_init (penalty range case) failed: %s\n", lib.v7_engine_is_dead() ? lib.v7_get_death_reason() : "returned 0");
			v7_unload(&lib);
			return 1;
		}
		if (!lib.v7_initialize_inference())
		{
			printf("v7_initialize_inference (penalty range case) failed: %s\n", lib.v7_engine_is_dead() ? lib.v7_get_death_reason() : "returned 0");
			v7_unload(&lib);
			return 1;
		}
		ctx = (inference_context *)lib.v7_get_internal_inference_ctx();
		sampler_base = *((char **)((char *)ctx + 16));
		lib.v7_set_sampler_presence_penalty(2.5f);
		++pen_total;
		if (lib.v7_engine_is_dead())
		{
			if (strstr(lib.v7_get_death_reason(), "between -2 and 2"))
				++pen_pass;
			else
				printf("\n  [penalty] p=2.5: engine died with unexpected reason '%s'\n", lib.v7_get_death_reason());
		}
		else
			printf("\n  [penalty] p=2.5: engine still alive, expected a border FATAL\n");
		lib.v7_free_inference();

		printf("penalty contract: %d/%d passed\n", pen_pass, pen_total);
		pass += pen_pass;
		total += pen_total;
	}
	{
		/* Context-requiring border FATALs (v7_set_quantize, v7_set_max_gen,
		 * v7_encode): each case re-creates the context, triggers the border,
		 * asserts the named reason, then frees so the next case starts alive. */
		int bx_pass = 0;
		int bx_total = 0;

		lib.v7_init(argv[2]);
		lib.v7_initialize_inference();
		lib.v7_set_quantize(2);
		++bx_total;
		if (lib.v7_engine_is_dead())
		{
			if (strstr(lib.v7_get_death_reason(), "requires 0 (F32) or 1 (Q8)"))
				++bx_pass;
			else
				printf("\n  [border] quantize=2: unexpected reason '%s'\n", lib.v7_get_death_reason());
		}
		else
			printf("\n  [border] quantize=2: engine still alive, expected a border FATAL\n");
		lib.v7_free_inference();

		lib.v7_init(argv[2]);
		lib.v7_initialize_inference();
		lib.v7_set_max_gen(-1);
		++bx_total;
		if (lib.v7_engine_is_dead())
		{
			if (strstr(lib.v7_get_death_reason(), "non-negative cap"))
				++bx_pass;
			else
				printf("\n  [border] max_gen=-1: unexpected reason '%s'\n", lib.v7_get_death_reason());
		}
		else
			printf("\n  [border] max_gen=-1: engine still alive, expected a border FATAL\n");
		lib.v7_free_inference();

		lib.v7_init(argv[2]);
		lib.v7_initialize_inference();
		int tok_sink[8];
		int n_sink = 0;
		lib.v7_encode(0, tok_sink, &n_sink, 8);
		++bx_total;
		if (lib.v7_engine_is_dead())
		{
			if (strstr(lib.v7_get_death_reason(), "non-NULL text, tokens and n_tokens"))
				++bx_pass;
			else
				printf("\n  [border] encode NULL: unexpected reason '%s'\n", lib.v7_get_death_reason());
		}
		else
			printf("\n  [border] encode NULL: engine still alive, expected a border FATAL\n");
		lib.v7_free_inference();

		printf("border contract: %d/%d passed\n", bx_pass, bx_total);
		pass += bx_pass;
		total += bx_total;
	}

	/* Every border case above ends with free_inference, whose teardown
	 * quiesces the worker pool AND destroys the hash table. The sampler
	 * gate below drives v7s_sample directly; the sampler brings the pool
	 * up itself (lazy initialize_once, the same gate every dispatched
	 * entry uses), but that init reads the INI keys from the database --
	 * so the DB must be re-loaded here before the gate can run. */
	if (!lib.v7_init(argv[2]))
	{
		printf("v7_init (sampler gate) failed: %s\n", lib.v7_engine_is_dead() ? lib.v7_get_death_reason() : "returned 0");
		v7_unload(&lib);
		return 1;
	}

	{
		/* Temperature pre-max clamp regression: the temperature-scaling loops
		 * used to floor every scaled logit at -87.3365 BEFORE the max was
		 * known, so a tiny temperature over negative logits (e.g. T=0.001 on
		 * [-1,-2,-3] -> scaled [-1000,-2000,-3000]) collapsed the whole
		 * distribution to uniform. The clamp belongs AFTER the max
		 * subtraction (the exp reconstruction guards, which stay). The
		 * reference sampler never had the pre-max clamp, so the existing
		 * random-logit cases (scaled logits far above the clamp) never
		 * diverged -- these cases hit the clamp head-on, on both the AVX2
		 * vector loop and the scalar tail. The forked sampler splits the
		 * vocabulary into two >= 32-element ranges, so the cases run at
		 * vocab 65/66 (helper side gets the vec8 + scalar-tail residue)
		 * instead of the old tiny vocabs. */
		int tc_pass = 0;
		int tc_total = 0;
		struct temp_case
		{
			int vocab;
			float temperature;
			float topp;
		};
		const temp_case tcases[] = {
			{ 65, 0.001f, 0.0f },
			{ 66, 0.001f, 0.0f },
			{ 65, 1.0f, 0.0f },
		};
	for (int ci = 0; ci < 3; ci++)
	{
		/* all-negative descending logits: token 0 is the argmax by a wide
		 * margin; at T=0.001 it carries essentially all of the mass. With
		 * the pre-max clamp the distribution was uniform 1/vocab, so some
		 * seed drew a non-argmax token. Token 0 for every seed = pass (the
		 * T=1.0 case is the no-clamp regression: at that temperature the
		 * mass is spread, so ONLY asm-vs-reference agreement is asserted). */
		int assert_argmax = (tcases[ci].temperature < 1.0f);
		int argmax_every_seed = 1;
		int ref_agrees = 1;
		for (int si = 0; si < n_seeds; si++)
		{
			std::vector<float> logits(tcases[ci].vocab);
			std::vector<float> logits_copy(tcases[ci].vocab);
			for (int i = 0; i < tcases[ci].vocab; i++)
				logits[i] = -(float)(i + 1);
			memcpy(logits_copy.data(), logits.data(), tcases[ci].vocab * sizeof(float));
			v7_probindex pi[SAMPLER_TOPK_LIMIT];
			unsigned __int64 rng_as = seeds[si];
			unsigned __int64 rng_ref = seeds[si];
			int asm_tok = lib.v7s_sample(logits.data(), tcases[ci].vocab, tcases[ci].temperature, tcases[ci].topp, &rng_as, pi);
			int ref_tok = ref_sample(logits_copy.data(), tcases[ci].vocab, tcases[ci].temperature, tcases[ci].topp, &rng_ref, pi);
			if (assert_argmax && (asm_tok != 0))
				argmax_every_seed = 0;
			if (asm_tok != ref_tok)
				ref_agrees = 0;
		}
		++tc_total;
		if ((!assert_argmax || argmax_every_seed) && ref_agrees)
		{
			++tc_pass;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("\n  [temp clamp] vocab=%d T=%g: argmax-every-seed=%d asm-ref-agree=%d\n",
			       tcases[ci].vocab, tcases[ci].temperature, argmax_every_seed, ref_agrees);
		}
	}
		printf("temp pre-max clamp: %d/%d passed\n", tc_pass, tc_total);
		pass += tc_pass;
		total += tc_total;
	}

	{
		/* Top-p nucleus coverage regressions: (a) a nucleus wider than the
		 * 256-entry heap must fall back to the exact full-vocab selection
		 * (uniform 1024-token vocab at topp=0.9 has a 922-token nucleus -- the
		 * retained top-256 carry only 0.25 of the mass, so the pre-fix sampler
		 * silently capped the request at top-k=256 and indices >= 256 were
		 * unreachable); (b) exact-equality nucleus sizing: uniform 4-token
		 * vocab at topp=0.5 has a 2-token nucleus (the pre-fix loop continued
		 * on cumulative == topp and extracted a third token). */
		int tp_pass = 0;
		int tp_total = 0;

		/* (a) wide nucleus: uniform 1024 vocab, T=1, topp=0.9 */
		{
			const int vc = 1024;
			std::vector<float> logits(vc, 0.0f);       /* all-equal logits -> uniform probabilities */
			int max_seen = -1;
			int ref_agrees = 1;
			for (int si = 0; si < n_seeds; si++)
			{
				std::vector<float> logits_copy(logits);
				std::vector<v7_probindex> pi(vc);   /* vocab-sized: the wide-nucleus fallback writes ALL vocab entries (the engine's chat path passes the vocab-sized SAM_PROBINDEX workspace) */
				unsigned __int64 rng_as = seeds[si];
				unsigned __int64 rng_ref = seeds[si];
				for (int draw = 0; draw < 8; draw++)
				{
					std::vector<float> work(logits);
					std::vector<float> work_ref(logits_copy);
					int asm_tok = lib.v7s_sample(work.data(), vc, 1.0f, 0.9f, &rng_as, pi.data());
					int ref_tok = ref_sample(work_ref.data(), vc, 1.0f, 0.9f, &rng_ref, pi.data());
					if (asm_tok != ref_tok)
						ref_agrees = 0;
					if (asm_tok > max_seen)
						max_seen = asm_tok;
				}
			}
			++tp_total;
			if (ref_agrees && (max_seen >= 256))
			{
				++tp_pass;
				printf(".");
				fflush(stdout);
			}
			else
			{
				printf("\n  [topp wide] vocab=1024 topp=0.9: max_seen=%d (need >= 256) ref_agrees=%d\n", max_seen, ref_agrees);
			}
		}

		/* (a2) full-vocab near-one topp: uniform 151936 vocab, T=1, topp=0.999.
		 * The float32 sequential cumulative over 151936 equal probabilities
		 * settles at ~0.9983, BELOW topp -- the pre-fix sampler rebuilt the
		 * full heap forever (hang). With the r14==r12 guard the extraction
		 * exhausts once and the nucleus covers the full vocabulary. */
		{
			const int vc = 151936;
			std::vector<float> logits(vc, 0.0f);
			int ref_agrees = 1;
			int terminated = 1;
			for (int si = 0; si < 2; si++)
			{
				std::vector<v7_probindex> pi(vc);    /* vocab-sized: the fallback writes ALL vocab entries */
				unsigned __int64 rng_as = seeds[si];
				unsigned __int64 rng_ref = seeds[si];
				for (int draw = 0; draw < 2; draw++)
				{
					std::vector<float> work(logits);
					std::vector<float> work_ref(logits);
					int asm_tok = lib.v7s_sample(work.data(), vc, 1.0f, 0.999f, &rng_as, pi.data());
					int ref_tok = ref_sample(work_ref.data(), vc, 1.0f, 0.999f, &rng_ref, pi.data());
					if ((asm_tok < 0) || (asm_tok >= vc))
						terminated = 0;
					if (asm_tok != ref_tok)
						ref_agrees = 0;
				}
			}
			++tp_total;
			if (ref_agrees && terminated)
			{
				++tp_pass;
				printf(".");
				fflush(stdout);
			}
			else
			{
				printf("\n  [topp 0.999 hang] vocab=151936 topp=0.999: terminated=%d ref_agrees=%d\n", terminated, ref_agrees);
			}
		}

		/* (b) equality nucleus sizing: uniform 64-token vocab, T=1.
		 * Every probability is exactly 1/64 (exp(0)=1 exact, the split partial
		 * sums are 32.0+32.0=64.0 exact, 1/64 is a power of two), so the
		 * cumulative is exactly k/64 and the equality edge survives:
		 * topp=0.5 -> exactly 32 nucleus tokens; topp=0.75 -> exactly 48;
		 * topp=0.50001 -> 33 (0.50000 would be exactly 32 by equality). The
		 * nucleus SIZE is observed through which indices the draws can reach:
		 * count distinct drawn tokens over all seeds. The forked sampler
		 * requires vocab >= 64 (two 32-element ranges), hence 64 instead of
		 * the old 4-token vocab. */
		const float eq_topps[3] = { 0.5f, 0.75f, 0.50001f };
		const int eq_expect[3] = { 32, 48, 33 };
		for (int ci = 0; ci < 3; ci++)
		{
			const int vc = 64;
			std::vector<float> logits(vc, 0.0f);
			int seen[64] = { 0 };
			int ref_agrees = 1;
			for (int si = 0; si < n_seeds; si++)
			{
				v7_probindex pi[SAMPLER_TOPK_LIMIT];
				unsigned __int64 rng_as = seeds[si];
				unsigned __int64 rng_ref = seeds[si];
				for (int draw = 0; draw < 100; draw++)
				{
					std::vector<float> work(logits);
					std::vector<float> work_ref(logits);
					int asm_tok = lib.v7s_sample(work.data(), vc, 1.0f, eq_topps[ci], &rng_as, pi);
					int ref_tok = ref_sample(work_ref.data(), vc, 1.0f, eq_topps[ci], &rng_ref, pi);
					seen[asm_tok]++;
					if (asm_tok != ref_tok)
						ref_agrees = 0;
				}
			}
			int distinct = 0;
			for (int i = 0; i < vc; i++)
			{
				if (seen[i] != 0)
					++distinct;
			}
			++tp_total;
			if (ref_agrees && (distinct == eq_expect[ci]))
			{
				++tp_pass;
				printf(".");
				fflush(stdout);
			}
			else
			{
				printf("\n  [topp eq] vocab=64 topp=%g: distinct=%d (need %d) ref_agrees=%d\n",
				       eq_topps[ci], distinct, eq_expect[ci], ref_agrees);
			}
		}
		printf("topp nucleus: %d/%d passed\n", tp_pass, tp_total);
		pass += tp_pass;
		total += tp_total;
	}

	for (int si = 0; ((si < n_seeds) && (total < MAX_TESTS)); si++)
	{
		/* Greedy */
		{
			unsigned __int64 rng_as = seeds[si];
			unsigned __int64 rng_ref = seeds[si];
			std::vector<float> logits(VOCAB_SIZE);
			std::vector<float> logits_copy(VOCAB_SIZE);
			for (int i = 0; i < VOCAB_SIZE; i++)
				logits[i] = (float)((int)(seeds[si] * (i + 1)) % 1000) / 100.0f;
			memcpy(logits_copy.data(), logits.data(), VOCAB_SIZE * sizeof(float));
			v7_probindex pi[SAMPLER_TOPK_LIMIT];
			int asm_tok = lib.v7s_sample(logits.data(), VOCAB_SIZE, 0.0f, 0.9f, &rng_as, pi);
			int ref_tok = ref_sample(logits_copy.data(), VOCAB_SIZE, 0.0f, 0.9f, &rng_ref, pi);
			int ok = (asm_tok == ref_tok);
			if (ok)
			{
				++pass;
				printf(".");
				fflush(stdout);
			}
			else
			{
				printf("\n  [greedy] seed=%u: Fail (asm=%d ref=%d)\n", seeds[si], asm_tok, ref_tok);
			}
			++total;
		}
		if (total >= MAX_TESTS)
			break;

		/* Temperature + multinomial */
		{
			unsigned __int64 rng_as = seeds[si];
			unsigned __int64 rng_ref = seeds[si];
			std::vector<float> logits(VOCAB_SIZE);
			std::vector<float> logits_copy(VOCAB_SIZE);
			for (int i = 0; i < VOCAB_SIZE; i++)
				logits[i] = (float)((int)(seeds[si] * (i + 1)) % 1000) / 100.0f;
			memcpy(logits_copy.data(), logits.data(), VOCAB_SIZE * sizeof(float));
			v7_probindex pi[SAMPLER_TOPK_LIMIT];
			int asm_tok = lib.v7s_sample(logits.data(), VOCAB_SIZE, 0.7f, 0.0f, &rng_as, pi);
			int ref_tok = ref_sample(logits_copy.data(), VOCAB_SIZE, 0.7f, 0.0f, &rng_ref, pi);
			int ok = (asm_tok == ref_tok);
			if (ok)
			{
				++pass;
				printf(".");
				fflush(stdout);
			}
			else
			{
				printf("\n  [temp+mult] seed=%u: Fail (asm=%d ref=%d)\n", seeds[si], asm_tok, ref_tok);
			}
			++total;
		}
		if (total >= MAX_TESTS)
			break;

		/* Temperature + top-p */
		{
			unsigned __int64 rng_as = seeds[si];
			unsigned __int64 rng_ref = seeds[si];
			std::vector<float> logits(VOCAB_SIZE);
			std::vector<float> logits_copy(VOCAB_SIZE);
			for (int i = 0; i < VOCAB_SIZE; i++)
				logits[i] = (float)((int)(seeds[si] * (i + 1)) % 1000) / 100.0f;
			memcpy(logits_copy.data(), logits.data(), VOCAB_SIZE * sizeof(float));
			std::vector<v7_probindex> pi(VOCAB_SIZE);
			int asm_tok = lib.v7s_sample(logits.data(), VOCAB_SIZE, 0.7f, 0.9f, &rng_as, pi.data());
			int ref_tok = ref_sample(logits_copy.data(), VOCAB_SIZE, 0.7f, 0.9f, &rng_ref, pi.data());
			int ok = (asm_tok == ref_tok);
			if (ok)
			{
				++pass;
				printf(".");
				fflush(stdout);
			}
			else
			{
				printf("\n  [temp+topp] seed=%u: Fail (asm=%d ref=%d)\n", seeds[si], asm_tok, ref_tok);
			}
			++total;
		}

		/* Skewed-logit nucleus regression: peaked logits where topp=0.9 cuts
		   inside the top few tokens -- the ASM must NEVER return a token
		   outside the nucleus. The old full-CDF reference could not expose a
		   missing cutoff. Logits: {6.0, 4.0, 3.5} head over a -12 tail:
		   p0~0.82, p0+p1~0.93 -> nucleus = {0, 1} with wide margins on both
		   sides of 0.9 (robust against the approximate softmax's ~1e-3
		   relative error). */
		{
			unsigned __int64 rng_as = seeds[si];
			std::vector<float> logits_base(VOCAB_SIZE);
			std::vector<float> logits(VOCAB_SIZE);
			for (int i = 0; i < VOCAB_SIZE; i++)
				logits_base[i] = -12.0f;
			logits_base[0] = 6.0f;
			logits_base[1] = 4.0f;
			logits_base[2] = 3.5f;
			v7_probindex pi[SAMPLER_TOPK_LIMIT];
			int ok = 1;
			int bad_tok = -1;
			for (int draw = 0; draw < 64; draw++)
			{
				/* The sampler is NON-DESTRUCTIVE since the scratch-table fix
				   (probs land in the engine's BSS table, the caller's array
				   keeps the raw logits), so the fresh copy is technically
				   redundant -- but it stays: the copy documents that each
				   draw must be independent, and guards against a future
				   in-place regression. */
				memcpy(logits.data(), logits_base.data(), VOCAB_SIZE * sizeof(float));
				int asm_tok = lib.v7s_sample(logits.data(), VOCAB_SIZE, 1.0f, 0.9f, &rng_as, pi);
				if ((asm_tok != 0) && (asm_tok != 1))
				{
					ok = 0;
					bad_tok = asm_tok;
					break;
				}
			}
			if (ok)
			{
				++pass;
				printf(".");
				fflush(stdout);
			}
			else
			{
				printf("\n  [topp-nucleus] seed=%u: Fail\n", seeds[si]);
				printf("    token %d returned -- outside the {0,1} nucleus\n", bad_tok);
			}
			++total;
		}
	}

	printf("\n%d/%d passed\n", pass, total);
	v7_unload(&lib);
	if (pass == total)
		return 0;
	return 1;
}
