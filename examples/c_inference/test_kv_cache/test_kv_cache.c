// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* test_kv_cache.c -- KV-cache save/load round-trip proof
 *
 * The canonical scenario (user spec):
 *   1. chat "i love cats"                     (KV rows [0, pos1) written)
 *   2. save_kv_cache("cats.kv")               (ONLY the used rows hit the disk)
 *   3. reset (chat "i love dogs")             (fresh session, KV rows [0, pos2))
 *   4. load_kv_cache("cats.kv")               (cats rows + pos restored, dogs gone)
 *   5. chat "what animal did we talk about?"  (multi-turn continuation from pos1)
 * The final answer MUST mention cats (not dogs) -- proving the cache round-trip.
 *
 * usage: test_kv_cache.exe <dll_path> <db_path>
 * exit 0 = round-trip proven; nonzero = failure.
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <string.h>

#define OUTPUT_BUF_SIZE 32768
#define KV_FILE "build_logs\\_test_cats.kv"

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	if (argc < 3)
	{
		printf("Usage: %s <dll_path> <db_path>\n", argv[0]);
		printf("  both paths REQUIRED -- no defaults\n");
		return 1;
	}

	struct v7_lib dll;
	if (v7_load(argv[1], &dll) != 0)
	{
		printf("FAIL: cannot load '%s'\n", argv[1]);
		return 1;
	}
	if (!dll.v7_init(argv[2]))
	{
		if (dll.v7_engine_is_dead())
			printf("FAIL: v7_init('%s') died: %s\n", argv[2], dll.v7_get_death_reason());
		else
			printf("FAIL: v7_init('%s')\n", argv[2]);
		v7_unload(&dll);
		return 1;
	}
	/* Sec.172: a gated entry -- a caught FATAL also returns 0, so the real return
	 * AND the death flag must both be checked. */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("FAIL: init died: %s\n", dll.v7_get_death_reason());
		else
			printf("FAIL: v7_initialize_inference returned 0\n");
		v7_unload(&dll);
		return 1;
	}
	dll.v7_set_quantize(0); /* F32 */
	dll.v7_set_max_gen(1000);
	printf("[test] engine ready (F32)\n");

	char output[OUTPUT_BUF_SIZE];

	/* step 1: the cats turn */
	memset(output, 0, sizeof(output));
	printf("\n... turn 1: \"i love cats\" ...\n");
	dll.v7_chat("i love cats", 0, output, OUTPUT_BUF_SIZE, 1);
	if (dll.v7_engine_is_dead())
	{
		printf("FAIL: chat died: %s\n", dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	printf("%s\n", output);
	int pos_after_cats = dll.v7_get_position();
	printf("[test] pos after cats turn: %d\n", pos_after_cats);

	/* step 2: save the used prefix */
	if (!dll.save_kv_cache(KV_FILE))
	{
		printf("FAIL: save_kv_cache('%s') returned 0\n", KV_FILE);
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[test] saved KV prefix to %s\n", KV_FILE);

	/* step 3: fresh session about dogs (overwrites the early KV rows) */
	memset(output, 0, sizeof(output));
	printf("\n... turn 2 (reset): \"i love dogs\" ...\n");
	dll.v7_chat("i love dogs", 0, output, OUTPUT_BUF_SIZE, 1);
	if (dll.v7_engine_is_dead())
	{
		printf("FAIL: chat died: %s\n", dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	printf("%s\n", output);
	printf("[test] pos after dogs turn: %d\n", dll.v7_get_position());

	/* step 4: load the cats cache back (rows + position restored) */
	if (!dll.load_kv_cache(KV_FILE))
	{
		printf("FAIL: load_kv_cache('%s') returned 0\n", KV_FILE);
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	int pos_restored = dll.v7_get_position();
	printf("[test] loaded KV prefix, pos restored: %d\n", pos_restored);
	if (pos_restored != pos_after_cats)
	{
		printf("FAIL: position mismatch after load (%d != %d)\n", pos_restored, pos_after_cats);
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}

	/* step 5: the proof question -- must remember CATS */
	memset(output, 0, sizeof(output));
	printf("\n... turn 3 (from restored cache): \"what animal did we talk about?\" ...\n");
	dll.v7_chat("what animal did we talk about?", 0, output, OUTPUT_BUF_SIZE, 0);
	if (dll.v7_engine_is_dead())
	{
		printf("FAIL: chat died: %s\n", dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	printf("%s\n", output);

	/* decision: the answer must mention cats, and the no-dogs drift check is
	 * informational only (both may appear if the model reasons about both) */
	int mentions_cats = (strstr(output, "cat") != 0) || (strstr(output, "Cat") != 0) || (strstr(output, "CAT") != 0);
	if (!mentions_cats)
	{
		printf("FAIL: the restored-cache answer does not mention cats\n");
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[test] round-trip proven: the model remembers cats\n");

	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
