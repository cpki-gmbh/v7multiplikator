// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_memcpy_validate.cpp -- dispatched memcpy correctness validator
 *
 * Tests v7d_memcpy with various byte counts.
 * Slice size is uniform 1024 bytes for all dispatched sizes.
 * Usage: v_memcpy_validate.exe <dll_path> --db <db_path>
 */
#include "v7_lib.h"
#include "zydis_debug_header.h"

typedef struct test_entry
{
	int byte_count;
	const char *desc;

} test_entry;

static const test_entry tests[] = { { 1, "1 byte" },    { 2, "2 bytes" },   { 3, "3 bytes" },   { 7, "7 bytes" },   { 8, "8 bytes" }, { 15, "15 bytes" }, { 16, "16 bytes" }, { 32, "32 bytes exact (aligned-store tier)" }, { 33, "33 bytes (two-ended 32-byte tier)" }, { 47, "47 bytes (two-ended 32-byte tier)" }, { 63, "63 bytes (two-ended 32-byte tier)" }, { 64, "64 bytes" }, { 256, "256 bytes" }, { 1024, "1KB" }, { 4096, "4KB" }, { 65536, "64KB" },
	                           { 1048576, "1MB" }, { 2097152, "2MB" }, { 0, 0 } };

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

	lib.v7s_set_spin(1);

	int pass = 0;
	int fail = 0;
	for (int ti = 0; tests[ti].desc; ++ti)
	{
		int bc = tests[ti].byte_count;
		int alloc_size = bc;
		if (alloc_size <= 0)
			alloc_size = 1;
		unsigned char *src = (unsigned char *)malloc(alloc_size);
		// +1 offset: the 32-byte tier of the tiny copy path reaches this kernel with the
		// raw caller dst (the alignment prologue only runs for n >= 64), so a 16-byte
		// aligned malloc dst would make that tier's pass a coin flip. Every dst is
		// deliberately misaligned here; dst_block is what gets freed.
		unsigned char *dst_block = (unsigned char *)malloc(alloc_size + 32);
		unsigned char *dst = dst_block + 1;
		unsigned char *ref = (unsigned char *)malloc(alloc_size);
		if ((!src) || (!dst_block) || (!ref))
		{
			printf("OOM at bytes=%d\n", bc);
			free(src);
			free(dst_block);
			free(ref);
			return 1;
		}

		srand(12345 + ti);
		for (int i = 0; i < bc; i++)
			src[i] = (unsigned char)(rand() & 0xFF);
		memcpy(ref, src, bc);
		memset(dst, 0xAA, bc);

		V7_LIB_CALL(&lib, v7d_memcpy, dst, src, bc);

		int ok = ((bc == 0) || ((memcmp(dst, ref, bc) == 0)));
		if (ok)
		{
			pass++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			fail++;
			printf("\n  bytes=%d (%s): Fail\n", bc, tests[ti].desc);
		}

		free(src);
		free(dst_block);
		free(ref);
	}

	lib.v7s_set_spin(0);
	v7_unload(&lib);
	printf("\nmemcpy_validate: %d passed, %d failed\n", pass, fail);
	if (fail)
		return 1;
	return 0;
}
