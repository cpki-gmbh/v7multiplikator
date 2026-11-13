// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v_attention_validate.cpp -- Standalone attention kernel correctness validator
 *
 * Tests v7d_attention (dispatched via scheduler) against a pure-C reference.
 * Validates all head_dim / n_heads / n_kv_heads / loff / pos combinations.
 * Usage: v_attention_validate.exe <dll_path> --db <db_path>
 */
#include "v7_lib.h"
#ifdef __linux__
#include <windows.h> /* Linux tool adapters */
#endif
#include "zydis_debug_header.h"
#include "runstate_alloc.h"
#include "attention_shared.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		printf("usage: v_attention_validate.exe <dll_path> --db <db_path>\n");
		return 1;
	}
	if ((strcmp(argv[2], "--db") != 0))
	{
		printf("usage: v_attention_validate.exe <dll_path> --db <db_path>\n");
		return 1;
	}
	ZydisInit();

	v7_lib lib;
	if ((v7_load(argv[1], &lib) != 0))
	{
		printf("v7_load failed\n");
		return 1;
	}
	lib.v7_init(argv[3]);
	/* Attention forks for positions 16..96: declare the active phase so the
	 * fork publication always reaches pause-spinning workers. */
	lib.v7s_set_spin(1);

	int total_pass = 0;
	int total_tests = 0;
	unsigned int seeds[] = { 12345, 67890 };

	for (int si = 0; si < (int)(sizeof(seeds) / sizeof(seeds[0])); si++)
	{
		srand(seeds[si]);
		for (int ci = 0; attn_configs[ci].n_heads != 0; ci++)
		{
			v7_config model_config;
			model_config.n_heads = attn_configs[ci].n_heads;
			model_config.n_kv_heads = attn_configs[ci].n_kv_heads;
			model_config.head_dim = attn_configs[ci].head_dim;
			model_config.dim = model_config.n_heads * model_config.head_dim;
			model_config.hidden_dim = model_config.dim * attn_configs[ci].hidden_mult;
			model_config.seq_len = ATTN_MAX_SEQ_LEN;
			model_config.vocab_size = 32000;
			int kv_dim = model_config.n_kv_heads * model_config.head_dim;
			int kv_mul = model_config.n_heads / model_config.n_kv_heads;

			for (int li = 0; li < (int)(sizeof(attn_loffs) / sizeof(attn_loffs[0])); li++)
			{
				unsigned __int64 loff = (unsigned __int64)attn_loffs[li] * ATTN_MAX_SEQ_LEN * kv_dim;
				for (int pi = 0; pi < (int)(sizeof(attn_positions) / sizeof(attn_positions[0])); pi++)
				{
					int pos = attn_positions[pi];
					if (pos >= ATTN_MAX_SEQ_LEN) continue;

					v7_run_state *state_pointer = rs_alloc(&model_config, ATTN_MAX_SEQ_LEN, ATTN_MAX_LOFF_LAYERS);
					v7_run_state *state_reference = rs_alloc(&model_config, ATTN_MAX_SEQ_LEN, ATTN_MAX_LOFF_LAYERS);
					if ((!state_pointer) || (!state_reference))
					{
						printf("OOM\n");
						rs_free(state_pointer);
						rs_free(state_reference);
						v7_unload(&lib);
						return 1;
					}

					rs_fill_random(state_pointer, &model_config, ATTN_MAX_SEQ_LEN, ATTN_MAX_LOFF_LAYERS);
					rs_copy(state_reference, state_pointer, &model_config, ATTN_MAX_SEQ_LEN, ATTN_MAX_LOFF_LAYERS);

					lib.v7d_attention(state_pointer, &model_config, pos, loff, kv_dim, kv_mul);

					int pass = 1;
					for (int i = 0; i < model_config.dim; i++)
					{
						if (((std::isnan(state_pointer->xb[i])) || (std::isinf(state_pointer->xb[i]))))
						{
							printf("  FAIL: NaN/inf at xb[%d] head_dim=%d n_heads=%d n_kv=%d pos=%d loff=%llu\n", i, model_config.head_dim, model_config.n_heads, model_config.n_kv_heads, pos, (unsigned long long)loff);
							pass = 0;
							break;
						}
					}

					attn_reference(state_reference, &model_config, pos, loff, kv_dim, kv_mul);

					float max_err = 0.0f;
					for (int i = 0; i < model_config.dim; i++)
					{
						float err = fabsf(state_reference->xb[i] - state_pointer->xb[i]);
						if (err > max_err)
							max_err = err;
						if (err > 1e-3f)
						{
							printf("  FAIL: head_dim=%d n_heads=%d n_kv=%d kv_mul=%d pos=%d loff=%llu xb[%d] ref=%.6f dll=%.6f diff=%.6f\n", model_config.head_dim, model_config.n_heads, model_config.n_kv_heads, kv_mul, pos, (unsigned long long)loff, i, state_reference->xb[i], state_pointer->xb[i], err);
							pass = 0;
						}
					}
					if (pass)
						total_pass++;
					total_tests++;
					rs_free(state_pointer);
					rs_free(state_reference);
				}
			}
		}
	}

	printf("%d/%d passed\n", total_pass, total_tests);
	lib.v7s_set_spin(0);
	lib.v7_shutdown();
	v7_unload(&lib);
	if (total_pass == total_tests)
		return 0;
	return 1;
}
