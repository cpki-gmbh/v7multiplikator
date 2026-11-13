// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_rmsnorm_validate.cpp -- RMSNorm family correctness validator (one
 * validator for the whole norm-kernel module)
 *
 * Four sections in one run, same pass/fail contract:
 *   1. RMSNorm (v7s_rmsnorm) against the C reference.
 *   2. Fused RMSNorm+quantize (v7s_rmsnorm_quantize): bit-identical int8
 *      output vs the ASM 2-step chain (v7s_rmsnorm -> v7d_quantize), the
 *      production equivalence guarantee; plus the informational mismatch
 *      count vs the pure C reference chain.
 *   3. RMSNorm+RoPE heads (v7s_rmsnorm_rope_heads) against the C reference.
 *   4. Fused add+RMSNorm (v7s_fused_add_rmsnorm) against the C reference.
 * Sections 1/2/4 read eps from the DLL BSS g_rms_eps; their C references use
 * the DB's __sys_rms_eps -- never a literal (AGENTS.md #5/#7). A missing key
 * is a contract violation -- FATAL, never a default.
 * Usage: v_rmsnorm_validate.exe <dll_path> --db <db_path>
 */
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "kt_utils.h"

typedef struct test_entry
{
	int size;
	const char *desc;

} test_entry;

typedef struct rope_test_entry
{
	int n_heads;
	int head_dim;
	int pos;
	int rope_max_dim;
	float eps;
	const char *desc;

} rope_test_entry;

/* Every size is a multiple of 64: the kernel contract is size > 0 && size % 64 == 0,
   enforced at init by the dim-alignment FATAL in inf_microdb.inc step 3a, and every
   production call site passes exactly dim. A non-multiple-of-64 entry would be a
   contract-illegal input per Sec.11a -- and the kernel only reconciles size % 32 in
   its remainder path anyway, so such a case computes the wrong count silently.
   There is no eps column: the norm kernels take no eps argument -- they read
   g_rms_eps from the DLL's BSS, so the C reference must use the DB's value
   (queried below) or the comparison is against an eps the kernel never saw. */
static const test_entry tests[] = { { 64, "64 (min legal, Qwen head_dim)" },
	                           { 128, "128 (large head_dim)" },
	                           { 256, "256 (large head_dim)" },
	                           { 1024, "1024 (Qwen dim)" },
	                           { 2048, "2048 (all_heads_dim)" },
	                           { 3072, "3072 (FFN dim)" },
	                           { 64, "64 (repeat, head_dim)" },
	                           { 1024, "1024 (repeat)" },
	                           { 2048, "2048 (repeat, dim)" },
	                           { 8192, "8192 (FFN dim)" },
	                           { 0, 0 } };

static const test_entry quantize_tests[] = { { 64, "64  (Qwen head_dim)" },   { 128, "128 (large head_dim)" },  { 256, "256 (large head_dim)" },
	                                     { 512, "512 (multiple of 64)" }, { 1024, "1024 (Qwen dim)" },      { 2048, "2048 (all_heads_dim)" },
	                                     { 3072, "3072 (FFN dim)" },      { 8192, "8192 (Llama FFN dim)" }, { 0, 0 } };

static const rope_test_entry rope_tests[] = { { 1, 64, 0, 64, 1e-6f, "1 head,  hd=64,  pos=0" },
	                                      { 1, 64, 7, 64, 1e-6f, "1 head,  hd=64,  pos=7" },
	                                      { 4, 64, 0, 64, 1e-6f, "4 heads, hd=64,  pos=0" },
	                                      { 4, 64, 31, 64, 1e-6f, "4 heads, hd=64,  pos=31" },
	                                      { 8, 64, 7, 64, 1e-6f, "8 heads, hd=64,  pos=7" },
	                                      { 8, 128, 0, 128, 1e-6f, "8 heads, hd=128, pos=0" },
	                                      { 16, 64, 0, 64, 1e-6f, "16 heads, hd=64, pos=0" },
	                                      { 16, 64, 31, 64, 1e-6f, "16 heads, hd=64, pos=31" },
	                                      { 32, 64, 7, 64, 1e-6f, "32 heads, hd=64, pos=7" },
	                                      { 32, 128, 0, 128, 1e-6f, "32 heads, hd=128, pos=0" },
	                                      { 40, 64, 31, 64, 1e-6f, "40 heads, hd=64, pos=31" },
	                                      { 40, 128, 7, 128, 1e-6f, "40 heads, hd=128, pos=7" },
	                                      /* Llama 3.2 1B: n_heads=32, n_kv_heads=8, head_dim=64, eps=1e-5 */
	                                      { 32, 64, 0, 64, 1e-5f, "32 heads, hd=64, Llama eps, pos=0" },
	                                      { 32, 64, 31, 64, 1e-5f, "32 heads, hd=64, Llama eps, pos=31" },
	                                      { 8, 64, 7, 64, 1e-5f, "8 kv_heads, hd=64, Llama eps, pos=7" },
	                                      { 0, 0, 0, 0, 0, 0 } };

/* The add+RMSNorm sizes: v7s_fused_add_rmsnorm takes dim directly; the block
 * geometry in the descriptions maps each size to its Pass-1/leftover shape. */
static const test_entry add_rmsnorm_tests[] = { { 8, "8 (1 block, Pass-1 all-leftover)" },
	                                        { 16, "16 (2 blocks, Pass-1 leftover loop)" },
	                                        { 24, "24 (3 blocks, Pass-1 leftover loop max)" },
	                                        { 48, "48 (6 blocks, 2 leftover + one 4-way group)" },
	                                        { 64, "64 (Qwen head_dim, two 4-way groups, no leftover)" },
	                                        { 128, "128 (large head_dim)" },
	                                        { 256, "256 (large head_dim)" },
	                                        { 1024, "1024 (Qwen dim)" },
	                                        { 2048, "2048 (all_heads_dim)" },
	                                        { 3072, "3072 (FFN dim)" },
	                                        { 64, "64 (repeat, head_dim)" },
	                                        { 1024, "1024 (repeat)" },
	                                        { 2048, "2048 (repeat, dim)" },
	                                        { 8192, "8192 (FFN dim)" },
	                                        { 0, 0 } };

static const int seeds[] = { 12345, 67890 };

int main(int argc, char **argv)
{
	ZydisInit();
	if (argc < 4)
	{
		printf("usage: %s <dll_path> --db <db_path>\n", argv[0]);
		return 1;
	}
	if ((strcmp(argv[2], "--db") != 0))
	{
		printf("usage: %s <dll_path> --db <db_path>\n", argv[0]);
		return 1;
	}

	v7_lib lib;
	if ((v7_load(argv[1], &lib) != 0))
	{
		printf("cannot load %s\n", argv[1]);
		return 1;
	}
	if (lib.v7_init(argv[3]) == 0)
	{
		printf("V7_init(%s) failed\n", argv[3]);
		v7_unload(&lib);
		return 1;
	}

	/* eps is governed by the DB. Query __sys_rms_eps for the C-reference
	   chains; the ASM paths read g_rms_eps from DLL BSS directly. */
	void *eps_data = lib.ht_find("__sys_rms_eps", 0);
	if (eps_data == 0)
	{
		printf("contract violation: __sys_rms_eps missing from the database\n");
		v7_unload(&lib);
		return 1;
	}
	float eps_from_db = *(float *)eps_data;
	printf("eps from DB: %.3e\n", eps_from_db);

	int pass = 0;
	int fail = 0;

	printf("\n... section 1: RMSNorm ...\n");
	for (int ti = 0; tests[ti].desc; ++ti)
	{
		for (int si = 0; si < 2; ++si)
		{
			float max_diff = 0;
			int ok = kt_verify_rmsnorm(lib.v7s_rmsnorm, tests[ti].size, seeds[si], eps_from_db, &max_diff);
			if (ok)
			{
				pass++;
				printf(".");
				fflush(stdout);
			}
			else
			{
				fail++;
				printf("\n  size=%5d eps=%.0e (%s) seed=%d: Fail  max_err=%.3e\n", tests[ti].size, eps_from_db, tests[ti].desc, seeds[si], max_diff);
			}
		}
	}

	printf("\n... section 2: fused RMSNorm+quantize (ASM-chain equivalence) ...\n");
	for (int ti = 0; quantize_tests[ti].desc; ++ti)
	{
		for (int si = 0; si < 2; ++si)
		{
			float max_scale_diff = 0;
			int asm_mismatches = -1;
			int ok = kt_verify_rmsnorm_quantize(lib.v7s_rmsnorm_quantize, lib.v7s_rmsnorm, lib.v7d_quantize, quantize_tests[ti].size, seeds[si], eps_from_db, &max_scale_diff, &asm_mismatches);
			if (ok)
			{
				pass++;
				printf(".");
				fflush(stdout);
			}
			else
			{
				fail++;
				printf("\n  size=%5d eps=%.0e (%s) seed=%d: Fail  max_scale_err=%.3e  asm_chain_mismatches=%d\n", quantize_tests[ti].size, eps_from_db, quantize_tests[ti].desc, seeds[si], max_scale_diff, asm_mismatches);
			}
		}
	}

	printf("\n... section 3: RMSNorm+RoPE heads ...\n");
	for (int ti = 0; rope_tests[ti].desc; ++ti)
	{
		for (int si = 0; si < 2; ++si)
		{
			float max_diff = 0;
			int ok = kt_verify_rmsnorm_rope_heads(lib.v7s_rmsnorm_rope_heads, rope_tests[ti].n_heads, rope_tests[ti].head_dim, rope_tests[ti].pos, rope_tests[ti].rope_max_dim, seeds[si], rope_tests[ti].eps, &max_diff);
			if (ok)
			{
				pass++;
				printf(".");
				fflush(stdout);
			}
			else
			{
				fail++;
				printf("\n  n_heads=%3d hd=%3d pos=%2d eps=%.0e (%s): Fail  max_err=%.3e\n", rope_tests[ti].n_heads, rope_tests[ti].head_dim, rope_tests[ti].pos, rope_tests[ti].eps, rope_tests[ti].desc, max_diff);
			}
		}
	}

	printf("\n... section 4: fused add+RMSNorm ...\n");
	for (int ti = 0; add_rmsnorm_tests[ti].desc; ++ti)
	{
		for (int si = 0; si < 2; ++si)
		{
			float max_diff = 0;
			int ok = kt_verify_fused_add_rmsnorm(lib.v7s_fused_add_rmsnorm, add_rmsnorm_tests[ti].size, seeds[si], eps_from_db, &max_diff);
			if (ok)
			{
				pass++;
				printf(".");
				fflush(stdout);
			}
			else
			{
				fail++;
				printf("\n  dim=%5d eps=%.0e (%s) seed=%d: Fail  max_err=%.3e\n", add_rmsnorm_tests[ti].size, eps_from_db, add_rmsnorm_tests[ti].desc, seeds[si], max_diff);
			}
		}
	}

	v7_unload(&lib);
	printf("\nrmsnorm_validate: %d passed, %d failed (4 norm-kernel sections)\n", pass, fail);
	if (fail)
		return 1;
	return 0;
}
