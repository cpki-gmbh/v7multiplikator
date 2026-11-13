// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* multi_turn_demo.c -- multi-turn chat + interleaved generate + reset demo
 *
 * demonstrates three modes of the v7 inference API:
 *   1. multi-turn chat (reset=0): preserves KV cache, remembers conversation
 *   2. single-turn generate (v7s_generate): separate KV cache, doesn't disturb chat
 *   3. fresh chat (reset=1): starts a new session, forgets everything
 *
 * 4-turn agent interaction (max per request):
 *   turn 1: fresh chat "What is a cat?"            (reset=1)
 *   turn 2: follow-up  "What animal did we talk about?"  (reset=0 -- memory test)
 *   turn 3: generate   "Explain quantum computing briefly" (separate KV cache)
 *   turn 4: follow-up  "What animal did we talk about earlier?" (memory intact
 *           after generate -- proves generate never touched the chat session)
 *
 * usage:  multi_turn_demo.exe <dll_path> <db_path>
 *   dll_path -- path to v7_threading.dll (REQUIRED, no default)
 *   db_path  -- path to lz4ht.db (REQUIRED -- no default, sec.106)
 *
 * runs the model's DEFAULT quantization mode (quantize untouched = F32).
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <string.h>

#define OUTPUT_BUF_SIZE 32768

void print_round(int round, char *mode, char *prompt, char *output)
{
	printf("\n........................................\n");
	printf("Round %d [%s]\n", round, mode);
	printf("Prompt: %s\n", prompt);
	printf("---\n");
	printf("%s\n", output);
	printf("........................................\n");
}

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Usage: %s <dll_path> <db_path>\n", argv[0]);
		printf("  dll_path is REQUIRED -- path to v7_threading.dll (no default)\n");
		printf("  db_path is REQUIRED -- path to lz4ht.db (no default)\n");
		return 1;
	}
	char *dll_path = argv[1];
	char *db_path = argv[2];

	setvbuf(stdout, 0, _IONBF, 0);
	setvbuf(stderr, 0, _IONBF, 0);

	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("Failed to load '%s'\n", dll_path);
		return 1;
	}
	printf("Loaded %s (%d workers)\n", dll_path, dll.v7s_get_num_workers());

	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("V7_init died for '%s': %s\n", db_path, dll.v7_get_death_reason());
		else
			printf("V7_init failed for '%s'\n", db_path);
		v7_unload(&dll);
		return 1;
	}
	printf("Database loaded: %s\n", db_path);

	/* Sec.172: v7_initialize_inference is a GATED entry -- a caught FATAL also
	 * returns 0 -- so the real return AND the death flag must both be checked.
	 * Checking only the return cannot tell "init failed" from "engine died with
	 * reason X", and lets a dead engine run. */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("Inference init died: %s\n", dll.v7_get_death_reason());
		else
			printf("v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}
	printf("Inference initialized\n");

	char output[OUTPUT_BUF_SIZE];

	/* ------------------------------------------------------------------
	 * round 1: fresh chat (reset=1) -- "What is a cat?"
	 * expected: answers about cats
	 * ------------------------------------------------------------------ */
	printf("[TRACE] round 1: calling v7s_chat reset=1 ...\n");
	memset(output, 0, sizeof(output));
	dll.v7_chat("What is a cat?", 0, output, OUTPUT_BUF_SIZE, 1);
	if (dll.v7_engine_is_dead())
	{
		printf("engine died in round 1: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TRACE] round 1: v7s_chat returned, output len=%d\n", (int)strlen(output));
	print_round(1, "chat reset=1", "What is a cat?", output);

	/* ------------------------------------------------------------------
	 * round 2: multi-turn chat (reset=0) -- "What animal did we talk about?"
	 * expected: mentions cats (remembers the conversation from round 1)
	 * ------------------------------------------------------------------ */
	printf("[TRACE] round 2: calling v7s_chat reset=0 ...\n");
	memset(output, 0, sizeof(output));
	dll.v7_chat("What animal did we talk about?", 0, output, OUTPUT_BUF_SIZE, 0);
	if (dll.v7_engine_is_dead())
	{
		printf("engine died in round 2: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TRACE] round 2: v7s_chat returned, output len=%d\n", (int)strlen(output));
	print_round(2, "chat reset=0", "What animal did we talk about?", output);

	/* ------------------------------------------------------------------
	 * round 3: generate (separate KV cache) -- "Explain quantum computing briefly"
	 * expected: explains quantum computing (fresh generation, chat memory intact)
	 * ------------------------------------------------------------------ */
	printf("[TRACE] round 3: calling v7s_generate ...\n");
	memset(output, 0, sizeof(output));
	dll.v7_generate("Explain quantum computing briefly", output, OUTPUT_BUF_SIZE, 0);
	if (dll.v7_engine_is_dead())
	{
		printf("engine died in round 3: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TRACE] round 3: v7s_generate returned, output len=%d\n", (int)strlen(output));
	print_round(3, "generate", "Explain quantum computing briefly", output);

	/* ------------------------------------------------------------------
	 * round 4: multi-turn chat (reset=0) -- "What animal did we talk about earlier?"
	 * expected: mentions cats (chat KV cache was never touched by generate)
	 * ------------------------------------------------------------------ */
	printf("[TRACE] round 4: calling v7s_chat reset=0 ...\n");
	memset(output, 0, sizeof(output));
	dll.v7_chat("What animal did we talk about earlier?", 0, output, OUTPUT_BUF_SIZE, 0);
	if (dll.v7_engine_is_dead())
	{
		printf("engine died in round 4: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("[TRACE] round 4: v7s_chat returned, output len=%d\n", (int)strlen(output));
	print_round(4, "chat reset=0", "What animal did we talk about earlier?", output);

	/* ------------------------------------------------------------------
	 * summary
	 * ------------------------------------------------------------------ */
	printf("\n... demo complete (4 turns) ...\n");
	printf("Turn 1: fresh chat -> answered about cats\n");
	printf("Turn 2: multi-turn -> remembered cats\n");
	printf("Turn 3: generate   -> separate KV, quantum computing\n");
	printf("Turn 4: multi-turn -> still remembered cats (generate didn't disturb)\n");

	/* clean up: join worker threads first (v7s_shutdown), then free model
	 * heap (v7s_inference_destroy), then unload the DLL. wrong order leaves
	 * workers executing inside the DLL while its heap is freed and pages
	 * unmapped -- the smol_l_m2-360M teardown crash (AGENTS.md 105). */
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
