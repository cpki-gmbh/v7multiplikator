// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_add_inplace_validate.cpp -- v7s_add_inplace correctness validator
 *
 * Tests element-wise add-inplace against C reference.
 * Usage: v_add_inplace_validate.exe <dll_path> --db <db_path>
 */
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "kt_utils.h"

typedef struct test_entry
{
	int rows;
	const char *desc;

} test_entry;

static const test_entry tests[] = { { 32, "32 elements (1 unroll group)" }, { 64, "64 elements (2 unroll groups)" }, { 96, "96 elements (odd group count -- post-pair-loop leftover)" }, { 128, "128 elements (4 unroll groups)" }, { 160, "160 elements (odd group count, 5 groups)" }, { 1024, "1024 elements (32 unroll groups)" }, { 2048, "2048 elements (64 unroll groups)" }, { 0, 0 } };
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
	lib.v7_init(argv[3]);

	int pass = 0;
	int fail = 0;
	for (int ti = 0; tests[ti].desc; ++ti)
	{
		for (int si = 0; si < 2; ++si)
		{
			float max_diff = 0;
			int ok = kt_verify_add_inplace(lib.v7s_add_inplace, tests[ti].rows, seeds[si], &max_diff);
			if (ok)
			{
				pass++;
				printf(".");
				fflush(stdout);
			}
			else
			{
				fail++;
				printf("\n  n=%5d (%s) seed=%d: Fail  max_err=%.3e\n", tests[ti].rows, tests[ti].desc, seeds[si], max_diff);
			}
		}
	}

	v7_unload(&lib);
	printf("\nadd_inplace_validate: %d passed, %d failed\n", pass, fail);
	if (fail)
		return 1;
	return 0;
}
