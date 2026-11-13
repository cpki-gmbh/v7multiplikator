// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_quantize_validate.cpp -- v7d_quantize correctness validator
 *
 * Tests quantization against C reference with 7 diverse patterns and 6 sizes.
 * Usage: v_quantize_validate.exe <dll_path> --db <db_path>
 */
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "kt_utils.h"

static const int quantize_sizes[] = { 64, 256, 1024, 4096, 8192, 65536, 0 };
static const int seeds[] = { 12345, 67890, 11111 };

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
	for (int pi = 0; pi < KTPAT_COUNT; ++pi)
	{
		printf("  [quantize] pattern %d: %s\n", pi, kt_quantize_pattern_names[pi]);
		for (int si = 0; quantize_sizes[si] != 0; ++si)
		{
			for (int seed_i = 0; seed_i < 3; ++seed_i)
			{
				float max_scale_diff = 0;
				int ok = kt_verify_quantize(lib.v7d_quantize, quantize_sizes[si], seeds[seed_i], pi, &max_scale_diff);
			if (ok)
			{
				pass++;
				printf(".");
				fflush(stdout);
			}
			else
			{
				fail++;
				printf("\n    size=%4d seed=%d: Fail  max_scale_err=%.3e\n", quantize_sizes[si], seeds[seed_i], max_scale_diff);
			}
			}
		}
	}

	v7_unload(&lib);
	printf("\nquantize_validate: %d passed, %d failed\n", pass, fail);
	if (fail)
		return 1;
	return 0;
}
