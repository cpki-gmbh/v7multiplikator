// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* test_generate_only.c -- minimal test: just call v7_generate.
 * raw crash = raw exit code.
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Usage: %s <dll_path> <db_path>\n", argv[0]);
		return 1;
	}
	char *dll_path = argv[1];
	char *db_path = argv[2];

	printf("[test] loading DLL...\n");
	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("[FATAL] v7_load failed\n");
		return 1;
	}

	printf("[test] V7_init...\n");
	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] V7_init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] V7_init failed\n");
		v7_unload(&dll);
		return 1;
	}

	printf("[test] v7_initialize_inference...\n");
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

	char output[4096];
	memset(output, 0, sizeof(output));

	printf("[test] inference initialized\n");
	printf("[test] v7_generate ptr = %p\n", (void *)dll.v7_generate);
	printf("[test] output buf      = %p  size=%d\n", (void *)output, (int)sizeof(output));

	printf("[test] >>> calling v7_generate now <<<\n");

	dll.v7_generate("Hello", output, sizeof(output), 0);
	if (dll.v7_engine_is_dead())
	{
		printf("[FATAL] engine died in generate: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}

	printf("[test] <<< v7_generate returned >>>\n");
	printf("[test] output len=%d\n", (int)strlen(output));
	printf("[test] output: %.500s\n", output);

	dll.v7_shutdown(); /* join workers before heap teardown + FreeLibrary */
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
