// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * test_multiturn.c -- minimal multi-turn chat test for §150 bug
 *
 * reproduces the server's multi-turn corruption: turn 2 produces
 * empty think block + degenerate repetition about the wrong topic.
 *
 * build:  cl /std:c++17 /EHsc /zi -isrc test_multiturn.c /link bin\v7_threading.lib
 * usage:  test_multiturn.exe <v7_threading.dll> <lz4ht.db>
 *
 * calls v7s_chat twice with reset=1 then reset=0, dumps output.
 * also calls v7s_enable_thinking(1) between create and first chat.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#define MAX_OUT 12000

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Usage: %s <v7_threading.dll> <lz4ht.db>\n", argv[0]);
		return 1;
	}
	char *dll_path = argv[1];
	char *db_path = argv[2];
	int q8 = (argc > 3) ? atoi(argv[3]) : 1; /* default Q8, pass 0 for F32 */

	/* load DLL */
	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("[FATAL] v7_load failed: %s\n", dll_path);
		return 1;
	}

	/* init DB */
	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] V7_init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] V7_init failed: %s\n", db_path);
		v7_unload(&dll);
		return 1;
	}

	/* initialize inference (context hidden in BSS) */
	/* Sec.172: a gated entry -- a caught FATAL also returns 0, so the real return
	 * AND the death flag must both be checked. */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] inference init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}
	dll.v7_set_quantize(q8);
	dll.v7_set_max_gen(1000);
	printf("[test] quantize=%d (%s)\n", q8, q8 ? "Q8" : "F32");

	/* enable thinking mode (model generates <think> block naturally) */
	dll.v7_enable_thinking(1);
	printf("[test] v7s_enable_thinking(1) called\n");

	char output[MAX_OUT];

	/* === TURN 1: fresh session (reset=1) === */
	printf("\n... TURN 1: \"what is a cat?\" (reset=1) ...\n");
	memset(output, 0, sizeof(output));
	dll.v7_chat("what is a cat?", 0, output, (int)sizeof(output), 1);
	if (dll.v7_engine_is_dead())
	{
		printf("[FATAL] engine died in turn 1: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TURN1] pos_after=%d\n", dll.v7_get_position());
	printf("[TURN1] output (%u chars):\n%s\n\n", (unsigned)strlen(output), output);

	/* === TURN 2: continue session (reset=0) === */
	printf("... TURN 2: \"what is a dog?\" (reset=0) ...\n");
	dll.v7_set_max_gen(1000);
	memset(output, 0, sizeof(output));
	dll.v7_chat("what is a dog?", 0, output, (int)sizeof(output), 0);
	if (dll.v7_engine_is_dead())
	{
		printf("[FATAL] engine died in turn 2: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TURN2] pos_after=%d\n", dll.v7_get_position());
	printf("[TURN2] output (%u chars):\n%s\n\n", (unsigned)strlen(output), output);

	/* === TURN 3: continue session (reset=0) === */
	printf("... TURN 3: \"what is a bird?\" (reset=0) ...\n");
	dll.v7_set_max_gen(1000);
	memset(output, 0, sizeof(output));
	dll.v7_chat("what is a bird?", 0, output, (int)sizeof(output), 0);
	if (dll.v7_engine_is_dead())
	{
		printf("[FATAL] engine died in turn 3: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TURN3] pos_after=%d\n", dll.v7_get_position());
	printf("[TURN3] output (%u chars):\n%s\n\n", (unsigned)strlen(output), output);

	/* cleanup */
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
