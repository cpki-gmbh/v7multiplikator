// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_mdb_validate.cpp -- MicroDB hash table stress test validator
 *
 * Tests ht_add/ht_find, vocab round-trip, LZ77 compress/decompress,
 * edge cases, and db_save/db_load persistence.
 * Takes <dll_path> as first positional arg.
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <windows.h>

int passing_tests_counter = 0;
int failing_tests_counter = 0;

#define MDB_CHECK(cond, msg)                                                                                                                                                                                                                             \
	do                                                                                                                                                                                                                                                   \
	{                                                                                                                                                                                                                                                    \
		if (!(cond))                                                                                                                                                                                                                                     \
		{                                                                                                                                                                                                                                                \
			printf("  FAIL: %s\n", msg);                                                                                                                                                                                                                 \
			failing_tests_counter++;                                                                                                                                                                                                                     \
		}                                                                                                                                                                                                                                                \
		else                                                                                                                                                                                                                                             \
		{                                                                                                                                                                                                                                                \
			passing_tests_counter++;                                                                                                                                                                                                                     \
		}                                                                                                                                                                                                                                                \
	} while (0)

/* --- Test 1: 200K small-key insert + find --- */
void test_200k_keys(v7_lib *lib)
{
	printf("\n... MDB Test 1: 200K small-key insert + find ...\n");
	lib->ht_init();
	int N = 200000;
	auto keys = (char (*)[16])malloc(N * 16);
	auto values = (int *)malloc(N * sizeof(int));
	if (!keys || !values)
	{
		free(keys);
		free(values);
		lib->ht_destroy();
		return;
	}

	for (int i = 0; i < N; i++)
	{
		sprintf(keys[i], "k%06d", i);
		values[i] = i * 7 + 3;
	}
	printf("  Inserting %d keys...\n", N);
	for (int i = 0; i < N; i++)
	{
		int add_ok = lib->ht_add(keys[i], &values[i], sizeof(int), 0);
		if (!add_ok)
		{
			printf("  FAIL: ht_add at i=%d\n", i);
			failing_tests_counter++;
			free(keys);
			free(values);
			lib->ht_destroy();
			return;
		}
	}
	int found = 0;
	int wrong = 0;
	for (int i = 0; i < N; i++)
	{
		unsigned __int64 out_len = 0;
		void *data = lib->ht_find(keys[i], &out_len);
		if (!data || out_len != sizeof(int) || *(int *)data != values[i])
		{
			wrong++;
			continue;
		}
		found++;
	}
	MDB_CHECK(found == N && wrong == 0, "200K key round-trip");
	printf("  Found %d/%d, wrong=%d\n", found, N, wrong);
	{
		unsigned __int64 ol = 0;
		void *d = lib->ht_find("nonexistent_key_999", &ol);
		MDB_CHECK(d == 0, "non-existent key returns 0");
	}
	free(keys);
	free(values);
	lib->ht_destroy();
}

/* --- Test 2: Vocab test (151K tokens) --- */
void test_vocab(v7_lib *lib)
{
	printf("\n... MDB Test 2: Vocab test (151K tokens) ...\n");
	int VOCAB_SIZE = 151936;
	auto vocab = (char **)malloc(VOCAB_SIZE * sizeof(char *));
	if (!vocab)
		return;
	for (int i = 0; i < VOCAB_SIZE; i++)
	{
		vocab[i] = (char *)malloc(64);
		if (!vocab[i])
		{
			for (int j = 0; j < i; j++)
				free(vocab[j]);
			free(vocab);
			return;
		}
		if (i < 256)
			sprintf(vocab[i], "%c", (char)i);
		else
			sprintf(vocab[i], "tok_%d", i);
	}
	lib->ht_init();
	printf("  Inserting %d vocab tokens...\n", VOCAB_SIZE);
	int insert_ok = 1;
	for (int i = 0; i < VOCAB_SIZE; i++)
	{
		int add_ok = lib->ht_add(vocab[i], &i, sizeof(int), 0);
		if (!add_ok)
		{
			printf("  FAIL: ht_add at token %d\n", i);
			insert_ok = 0;
			failing_tests_counter++;
			break;
		}
	}
	MDB_CHECK(insert_ok, "151K vocab tokens inserted");
	int found = 0;
	int wrong = 0;
	for (int i = 0; i < VOCAB_SIZE; i++)
	{
		unsigned __int64 out_len = 0;
		void *data = lib->ht_find(vocab[i], &out_len);
		if (!data || out_len != sizeof(int) || *(int *)data != i)
		{
			wrong++;
			continue;
		}
		found++;
	}
	MDB_CHECK(found == VOCAB_SIZE && wrong == 0, "151K vocab lookup round-trip");
	printf("  Found %d/%d, wrong=%d\n", found, VOCAB_SIZE, wrong);
	for (int i = 0; i < VOCAB_SIZE; i++)
		free(vocab[i]);
	free(vocab);
	lib->ht_destroy();
}

/* --- Test 3: LZ77 compress/decompress round-trip --- */
void test_lz4(v7_lib *lib)
{
	printf("\n... MDB Test 3: LZ77 compress/decompress round-trip ...\n");
	/* No capability check on mdb_compress / mdb_decompress (Sec.152): every
	 * export the DLL lists is always present, and V7_RESOLVE already
	 * TerminateProcesses on a miss, so a NULL here is impossible. The old guard
	 * was worse than dead -- it printed SKIP and returned, so a genuinely
	 * missing export would have made this test silently pass instead of
	 * failing the run. */
	int SZ = 65536;
	auto src = (unsigned char *)malloc(SZ);
	auto comp = (unsigned char *)malloc(SZ * 2);
	auto dec = (unsigned char *)malloc(SZ);
	if (!src || !comp || !dec)
	{
		free(src);
		free(comp);
		free(dec);
		return;
	}

	srand(42);
	for (int i = 0; i < SZ; i++)
		src[i] = (unsigned char)(rand() & 0xFF);
	unsigned __int64 csz = lib->mdb_compress(src, SZ, comp);
	MDB_CHECK(csz > 0, "LZ4 compress random data");
	printf("  Random: %d -> %llu bytes (%.1f%%)\n", SZ, csz, 100.0 * csz / SZ);
	if (csz > 0)
	{
		unsigned __int64 dsz = lib->mdb_decompress(comp, csz, dec, SZ);
		MDB_CHECK(dsz == SZ, "LZ4 decompress random data size");
		if (dsz == SZ)
			MDB_CHECK(memcmp(src, dec, SZ) == 0, "LZ4 random round-trip match");
	}

	memset(src, 'A', SZ);
	csz = lib->mdb_compress(src, SZ, comp);
	MDB_CHECK(csz > 0, "LZ4 compress repeated data");
	printf("  Repeated: %d -> %llu bytes (%.1f%%)\n", SZ, csz, 100.0 * csz / SZ);
	if (csz > 0)
	{
		unsigned __int64 dsz = lib->mdb_decompress(comp, csz, dec, SZ);
		MDB_CHECK(dsz == SZ, "LZ77 decompress repeated data size");
		if (dsz == SZ)
			MDB_CHECK(memcmp(src, dec, SZ) == 0, "LZ77 repeated round-trip match");
	}

	char short_str[] = "Hello, LZ77 world!";
	csz = lib->mdb_compress(short_str, (unsigned int)strlen(short_str), comp);
	MDB_CHECK(csz > 0, "LZ4 compress short string");

	for (int i = 0; i < SZ; i++)
		src[i] = (unsigned char)('A' + (i % 26));
	csz = lib->mdb_compress(src, SZ, comp);
	MDB_CHECK(csz > 0, "LZ4 compress periodic data");
	if (csz > 0)
	{
		unsigned __int64 dsz = lib->mdb_decompress(comp, csz, dec, SZ);
		MDB_CHECK(dsz == SZ, "LZ4 decompress periodic data size");
		if (dsz == SZ)
			MDB_CHECK(memcmp(src, dec, SZ) == 0, "LZ4 periodic round-trip match");
	}

	free(src);
	free(comp);
	free(dec);
}

/* --- Test 3b: large-blob round-trips (the model-weights scale) ---
 * db_save compresses multi-hundred-MB pools (the 633MB model weights blob);
 * the 64KB tests above cannot see a size-dependent LZ77 bug. The ladder walks
 * 1..64 MB in steps, then 128/256/512 MB, with a compressible weights-like
 * pattern (a drifting 256-byte cycle so the compressor sees matches AND
 * literals), byte-compares the round-trip and stops at the first mismatch. */
void test_lz4_large_blobs(v7_lib *lib)
{
	printf("\n... MDB Test 3b: large-blob compress/decompress ladder (model-weights scale) ...\n");
	int megabytes = 1;
	while (megabytes <= 512)
	{
		unsigned __int64 sz = (unsigned __int64)megabytes * 1024 * 1024;
		unsigned char *src = (unsigned char *)malloc((size_t)sz);
		unsigned char *dec = (unsigned char *)malloc((size_t)sz);
		/* compressed worst case ~ sz + sz/255 + 64 */
		unsigned char *comp = (unsigned char *)malloc((size_t)sz + (size_t)(sz / 64) + 1024);
		if ((!src) || (!dec) || (!comp))
		{
			MDB_CHECK(0, "large-blob ladder allocation");
			free(src);
			free(dec);
			free(comp);
			return;
		}

		srand(1234);
		for (unsigned __int64 i = 0; i < sz; i++)
		{
			if ((i & 0xFFFF) == 0)
				src[i] = (unsigned char)(rand() & 0xFF);
			else
				src[i] = (unsigned char)((i / 256) & 0x3F);
		}

		printf("  %4d MB ... ", megabytes);
		fflush(stdout);
		unsigned __int64 comp_sz = lib->mdb_compress(src, (unsigned int)sz, comp);
		MDB_CHECK(comp_sz > 0, "large-blob compress returned data");
		if (comp_sz > 0)
		{
			memset(dec, 0xEE, (size_t)sz);
			unsigned __int64 dec_sz = (unsigned __int64)lib->mdb_decompress(comp, (unsigned int)comp_sz, dec, (unsigned int)sz);
			MDB_CHECK(dec_sz == sz, "large-blob decompress size");
			if (dec_sz == sz)
			{
				unsigned __int64 bad = 0;
				unsigned __int64 first_bad = 0;
				for (unsigned __int64 i = 0; i < sz; i++)
				{
					if (src[i] != dec[i])
					{
						if (!bad)
							first_bad = i;
						bad++;
						if (bad > 16)
							break;
					}
				}
				MDB_CHECK(bad == 0, "large-blob round-trip match");
			}
		}
		printf("done (%llu -> %llu bytes)\n", sz, comp_sz > 0 ? comp_sz : 0);
		free(src);
		free(dec);
		free(comp);
		if (failing_tests_counter)
			return;
		if (megabytes >= 64)
			megabytes *= 2; /* accelerate after 64MB */
		else
			megabytes++;
	}
}

/* --- Test 4: Edge cases --- */
void test_edge_cases(v7_lib *lib)
{
	printf("\n... MDB Test 4: Edge cases ...\n");
	lib->ht_init();
	for (int c = 1; c <= 255; c++)
	{
		char key[2];
		key[0] = (char)c;
		key[1] = 0;
		int val = c;
		lib->ht_add(key, &val, sizeof(int), 0);
	}
	int found = 0;
	for (int c = 1; c <= 255; c++)
	{
		char key[2];
		key[0] = (char)c;
		key[1] = 0;
		unsigned __int64 ol = 0;
		void *d = lib->ht_find(key, &ol);
		if (d && ol == sizeof(int) && *(int *)d == c)
			found++;
	}
	MDB_CHECK(found == 255, "255 single-char key round-trip");
	printf("  Single-char keys: %d/255\n", found);

	int add_ok = lib->ht_add("zero_data_key", 0, 0, 0);
	MDB_CHECK(add_ok != 0, "ht_add with zero-length data succeeds");

	int v1 = 100;
	int v2 = 200;
	lib->ht_add("update_me", &v1, sizeof(int), 0);
	lib->ht_add("update_me", &v2, sizeof(int), 0);
	unsigned __int64 ol = 0;
	void *data = lib->ht_find("update_me", &ol);
	MDB_CHECK(data && *(int *)data == v2, "updating existing key returns latest value");
	lib->ht_destroy();
}

/* --- Test 5: db_save + ht_destroy + ht_init + db_load ---
 * NOTE: This test is destructive -- it calls ht_init() which destroys
 * the DLL's current hash table (including model data), then db_save()
 * persists the new (test-only) state. This corrupts lz4ht.db if it
 * already contains real data. Skip when lz4ht.db exists to avoid
 * corrupting files needed by other validators/inference.
 */
void test_persistence(v7_lib *lib)
{
	printf("\n... MDB Test 5: db_save + ht_destroy + ht_init + db_load ...\n");

	/* Skip if lz4ht.db already exists -- don't corrupt real data */
	HANDLE hd = CreateFileA("lz4ht.db", GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
	if (hd != INVALID_HANDLE_VALUE)
	{
		CloseHandle(hd);
		printf("  SKIP: lz4ht.db exists (would be corrupted by ht_init+db_save)\n");
		return;
	}

	lib->ht_init();
	for (int i = 0; i < 100; i++)
	{
		char key[32];
		sprintf(key, "persist_%04d", i);
		int val = i * 13;
		lib->ht_add(key, &val, sizeof(int), 1);
	}
	for (int i = 0; i < 50; i++)
	{
		char key[32];
		sprintf(key, "temp_%04d", i);
		int val = i;
		lib->ht_add(key, &val, sizeof(int), 0);
	}

	int save_result = lib->db_save("_test_mdb_save.db");
	MDB_CHECK(save_result == 1, "db_save returns 1 (success)");
	printf("  db_save returned %d\n", save_result);

	if (save_result)
	{
		lib->ht_destroy();
		lib->ht_init();
		int load_result = lib->db_load("_test_mdb_save.db");
		MDB_CHECK(load_result == 1, "db_load returns 1 (success)");
		printf("  db_load returned %d\n", load_result);

		int found = 0;
		int wrong = 0;
		for (int i = 0; i < 100; i++)
		{
			char key[32];
			sprintf(key, "persist_%04d", i);
			unsigned __int64 ol = 0;
			void *data = lib->ht_find(key, &ol);
			if (!data || ol != sizeof(int) || *(int *)data != i * 13)
			{
				wrong++;
				continue;
			}
			found++;
		}
		MDB_CHECK(found == 100 && wrong == 0, "100 persistent keys survive save/load round-trip");
		printf("  Persistent keys: %d/100 found, wrong=%d\n", found, wrong);

		int temp_found = 0;
		for (int i = 0; i < 50; i++)
		{
			char key[32];
			sprintf(key, "temp_%04d", i);
			unsigned __int64 ol = 0;
			void *data = lib->ht_find(key, &ol);
			if (data)
				temp_found++;
		}
		MDB_CHECK(temp_found == 0, "0 temporary keys after db_save (non-persistent discarded)");
		printf("  Temp keys found: %d/50 (should be 0)\n", temp_found);
	}

	lib->ht_destroy();

	/* Clean up the test DB we created */
	DeleteFileA("_test_mdb_save.db");
}

int main(int argc, char **argv)
{
	ZydisInit();
	if (argc < 2)
	{
		printf("Usage: %s <dll_path>\n", argv[0]);
		return 1;
	}

	v7_lib lib;
	if (v7_load(argv[1], &lib) != 0)
	{
		printf("Cannot load %s\n", argv[1]);
		return 1;
	}

	test_200k_keys(&lib);
	test_vocab(&lib);
	test_lz4(&lib);
	test_lz4_large_blobs(&lib);
	test_edge_cases(&lib);
	test_persistence(&lib);

	v7_unload(&lib);
	printf("\nmdb_validate: %d passed, %d failed\n", passing_tests_counter, failing_tests_counter);
	if (failing_tests_counter)
		return 1;
	else
		return 0;
}
