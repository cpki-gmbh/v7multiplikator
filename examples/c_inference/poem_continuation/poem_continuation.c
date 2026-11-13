// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* poem_continuation.c -- raw text continuation demo (v7_generate raw_generation=1)
 *
 * passes the first half of a well-known poem to v7_generate with the
 * raw_generation flag set: no chat template, no stop-token scan -- the
 * model simply CONTINUES the text verbatim. this is pure text completion,
 * the engine's third inference mode next to chat and templated generate.
 *
 * usage:  poem_continuation.exe <dll_path> <db_path> [-q8]
 *   dll_path -- path to v7_threading.dll (REQUIRED, no default)
 *   db_path  -- path to lz4ht.db (REQUIRED, no default per sec.106)
 *   -q8      -- Q8 quantized activation mode (default: F32)
 *
 * build:  built by build.bat (step 4) -> bin\poem_continuation.exe
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OUTPUT_BUF_SIZE 32768

/* first half of a well-known poem -- the model continues it in raw mode */
char *POEM_FIRST_HALF = "Two roads diverged in a yellow wood,\n"
                        "And sorry I could not travel both\n"
                        "And be one traveler, long I stood\n"
                        "And looked down one road as far as I could\n"
                        "To where it bent in the undergrowth;\n\n";

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	if (argc < 3)
	{
		printf("Usage: %s <dll_path> <db_path> [-q8] [-n tokens]\n", argv[0]);
		printf("  dll_path is REQUIRED -- path to v7_threading.dll (no default)\n");
		printf("  db_path  is REQUIRED -- path to lz4ht.db (no default)\n");
		printf("  -n       -- generation cap (default 250; raw mode has no stop\n");
		printf("              tokens -- unlimited babbles until the buffer is full)\n");
		return 1;
	}
	char *dll_path = argv[1];
	char *db_path = argv[2];
	int q8 = 0;
	int max_gen = 250; /* raw continuation MUST be capped: no stop-token scan (see the set_max_gen note below) */

	for (int i = 3; i < argc; i++)
	{
		if (strcmp(argv[i], "-q8") == 0)
			q8 = 1;
		else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
			max_gen = atoi(argv[++i]);
	}

	/* step 1: load the DLL */
	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("Failed to load '%s'\n", dll_path);
		return 1;
	}
	printf("Loaded %s\n", dll_path);

	/* step 2: initialize database */
	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("v7_init died for '%s': %s\n", db_path, dll.v7_get_death_reason());
		else
			printf("v7_init failed for '%s'\n", db_path);
		v7_unload(&dll);
		return 1;
	}
	printf("Database loaded: %s\n", db_path);

	/* step 3: initialize inference + configure the engine */
	/* Sec.172: a gated entry -- a caught FATAL also returns 0, so the real return
	 * AND the death flag must both be checked. */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("Inference init died: %s\n", dll.v7_get_death_reason());
		else
			printf("v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}
	dll.v7_set_quantize(q8);
	/* MUST be limited: raw_generation=1 skips the stop-token scan, and on
	 * a raw text continuation the model has no EOS pressure -- without a
	 * cap it can babble until the 32 KB output buffer is full, which looks
	 * exactly like a hang (the demo sat for minutes in the example suite).
	 * 250 tokens is a complete continuation; -n raises it deliberately. */
	dll.v7_set_max_gen(max_gen);
	/* sampler untouched: the engine default (temperature 0.0) is already greedy */
	printf("Inference initialized (%s mode, max_gen=%d)\n", q8 ? "Q8" : "F32", max_gen);

	/* step 4: raw continuation. raw_generation=1 skips the chat template
	 * and the stop-token scan entirely -- the prompt text is fed verbatim
	 * and the model continues it as plain text. */
	char output[OUTPUT_BUF_SIZE];
	memset(output, 0, sizeof(output));

	printf("\n... prompt (poem first half) ...\n");
	printf("%s\n", POEM_FIRST_HALF);
	printf("---\n");

	dll.v7_generate(POEM_FIRST_HALF, output, OUTPUT_BUF_SIZE, 1);
	if (dll.v7_engine_is_dead())
	{
		printf("engine died during continuation: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}

	printf("%s\n", output);
	printf("---\n");
	printf("... raw continuation complete (no template, no chat format) ...\n");

	/* step 5: clean up (v7_shutdown joins workers BEFORE the DLL unloads) */
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
