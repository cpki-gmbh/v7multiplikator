// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * test_ardmark_seed.c -- minimal repro for the primer-seed rewrite bug.
 *
 * Loads the ardmark sidecar exactly like the server does (script into the
 * "logo:sidecar" key, hook enabled, thinking on, per-turn marker 0), then
 * runs ONE turn: "tell me about ardmark". The token-stream dump is forced
 * ON so the log shows every position step: when the seed's SetStream
 * rewrite lands, the dump shows a ~100-token re-prefill after the first
 * generated token and the think text carries the Bumwolf primer; with the
 * bug, the positions continue 1-by-1 from the prompt end and the think
 * ignores the primer entirely.
 *
 * usage: test_ardmark_seed.exe <v7_threading.dll> <lz4ht.db>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#define MAX_OUT 20000
#define SIDECAR_PATH "examples/ardmark/ardmark_rag.logo"

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Usage: %s <v7_threading.dll> <lz4ht.db>\n", argv[0]);
		return 1;
	}
	const char *dll_path = argv[1];
	const char *db_path = argv[2];

	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("[FATAL] v7_load failed: %s\n", dll_path);
		return 1;
	}

	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] V7_init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] V7_init failed: %s\n", db_path);
		v7_unload(&dll);
		return 1;
	}

	/* load the sidecar script (same wiring as inference_example -script) */
	FILE *sidecar_file = fopen(SIDECAR_PATH, "rb");
	if (!sidecar_file)
	{
		printf("[FATAL] cannot open sidecar '%s'\n", SIDECAR_PATH);
		v7_unload(&dll);
		return 1;
	}
	fseek(sidecar_file, 0, SEEK_END);
	long sidecar_len = ftell(sidecar_file);
	fseek(sidecar_file, 0, SEEK_SET);
	char *sidecar_buf = (char *)malloc((size_t)sidecar_len + 1);
	if (!sidecar_buf)
	{
		printf("[FATAL] malloc failed for sidecar buffer\n");
		fclose(sidecar_file);
		v7_unload(&dll);
		return 1;
	}
	int sidecar_read = (int)fread(sidecar_buf, 1, (int)sidecar_len, sidecar_file);
	fclose(sidecar_file);
	sidecar_buf[sidecar_read] = 0;
	if (!dll.v7_db_add("logo:sidecar", sidecar_buf, sidecar_read + 1, 0))
	{
		printf("[FATAL] v7_db_add('logo:sidecar') failed\n");
		free(sidecar_buf);
		v7_unload(&dll);
		return 1;
	}
	printf("[seed-test] sidecar loaded: %s (%d bytes)\n", SIDECAR_PATH, sidecar_read);
	free(sidecar_buf);

	/* enable the token dump (the dump is the evidence:
	 * a landed rewrite shows as a long re-prefill burst in the log).
	 * The hook is DATA-DRIVEN: the sidecar is in the DB (ht_add above),
	 * so it self-arms on the first chat. */
	*dll.enable_token_dump = 1;

	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] inference init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}
	dll.v7_set_quantize(1);
	dll.v7_enable_thinking(1);
	printf("[seed-test] Q8, thinking on, token dump on\n");

	/* the server's per-turn marker: 0 = the seed turn */
	if (!dll.v7_db_add("__server_chat__turn", "0", 2, 0))
	{
		printf("[FATAL] v7_db_add('__server_chat__turn') failed\n");
		v7_unload(&dll);
		return 1;
	}

	static char output[MAX_OUT];
	printf("[seed-test] asking: tell me about ardmark\n");
	dll.v7_chat("tell me about ardmark", 0, output, (int)sizeof(output), 1);
	if (dll.v7_engine_is_dead())
	{
		printf("[FATAL] engine died: %s\n", dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}

	printf("[seed-test] pos_after=%d output (%u chars):\n", dll.v7_get_position(),
	       (unsigned)strlen(output));
	printf("%s\n", output);

	/* the RAG pipeline keys: lengths FIRST -- a truncated bm25 input shows up
	 * here as a suspiciously short __server_rag_query long before any panel */
	const char *rag_keys[3] = { "__server_rag_query", "__server_rag_answer", "__server_rag_injected" };
	for (int k = 0; k < 3; k++)
	{
		unsigned __int64 klen = 0;
		const char *kval = (const char *)dll.v7_db_find(rag_keys[k], &klen);
		if (!kval)
			printf("[seed-test] %s: ABSENT\n", rag_keys[k]);
		else
			printf("[seed-test] %s: %llu bytes\n[seed-test] >>>%s<<<\n", rag_keys[k],
			       (unsigned long long)klen, kval);
	}

	printf("[seed-test] done\n");
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
