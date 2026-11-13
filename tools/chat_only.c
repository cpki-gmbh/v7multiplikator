// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* test_chat_only.c -- just v7s_chat */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include <stdio.h>
#include <string.h>

#define OUTPUT_BUF_SIZE 32768

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Usage: %s <dll_path> <db_path>\n", argv[0]);
		return 1;
	}
	printf("Step 1: load DLL\n");
	struct v7_lib dll;
	if (v7_load(argv[1], &dll) != 0)
	{
		printf("load failed\n");
		return 1;
	}
	printf("Loaded DLL (%d workers)\n", dll.v7s_get_num_workers());

	printf("Step 2: V7_init\n");
	char *db_path = argv[2];
	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("V7_init died: %s\n", dll.v7_get_death_reason());
		else
			printf("V7_init failed\n");
		v7_unload(&dll);
		return 1;
	}
	printf("DB loaded\n");

	printf("Step 3: initialize inference (context hidden in BSS)\n");
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
	printf("Inference initialized, pos=%d\n", dll.v7_get_position());

	char output[OUTPUT_BUF_SIZE];

	printf("Step 4: chat reset=1\n");
	memset(output, 0, sizeof(output));
	dll.v7_chat("What is a cat?", 0, output, OUTPUT_BUF_SIZE, 1);
	if (dll.v7_engine_is_dead())
	{
		printf("engine died in chat 1: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("chat done, output len=%u\n", (unsigned)strlen(output));
	printf("Round 1 [reset=1]: %s\n", output);

	printf("Step 5: chat reset=0\n");
	memset(output, 0, sizeof(output));
	dll.v7_chat("What animal did we talk about?", 0, output, OUTPUT_BUF_SIZE, 0);
	if (dll.v7_engine_is_dead())
	{
		printf("engine died in chat 2: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("chat2 done, output len=%u\n", (unsigned)strlen(output));
	printf("Round 2 [reset=0]: %s\n", output);

	printf("Step 6: destroy\n");
	dll.v7_shutdown(); /* join workers before heap teardown + FreeLibrary */
	dll.v7_free_inference();
	v7_unload(&dll);
	printf("All done\n");
	return 0;
}
