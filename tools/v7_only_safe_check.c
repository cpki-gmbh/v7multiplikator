// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_only_safe_check.c -- V7_ONLY_SAFE compilation-mode gate (P1-8, release audit).
 *
 * This translation unit includes v7_lib.h with V7_ONLY_SAFE defined: the v7_lib
 * struct carries ONLY the public v7_* members, the internal declarations
 * (kernels, MicroDB internals, encode/forward, SectorLogo, the
 * inference_context struct) do not exist here, and v7_load resolves only the
 * public tier. A mistyped internal call would be a compile error, which is
 * exactly the guard: build.bat compiles this file with /DV7_ONLY_SAFE, so the
 * build itself proves the header hides the internal surface.
 *
 * The run half drives one real chat turn through the safe struct only --
 * init, initialize, chat, is_dead, free, unload -- proving the public tier is
 * complete enough for a normal host (output must contain "paris").
 *
 * usage: v7_only_safe_check.exe <dll_path> <db_path>
 * exit 0 = all checks passed; nonzero = failure.
 */

#ifndef V7_ONLY_SAFE
#define V7_ONLY_SAFE /* build.bat also passes /DV7_ONLY_SAFE -- the guard keeps both spellings working */
#endif
#include "v7_lib.h"

#include <stdio.h>
#include <string.h>

#define OUT_BUF_SIZE 32768

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	if (argc < 3)
	{
		printf("Usage: %s <dll_path> <db_path>\n", argc > 0 ? argv[0] : "v7_only_safe_check");
		printf("  dll_path and db_path are REQUIRED (no defaults)\n");
		return 1;
	}

	/* The safe struct: this line only compiles because V7_ONLY_SAFE pruned
	 * every internal member -- the type here is the proof. */
	v7_lib dll;

	if (v7_load(argv[1], &dll) != 0)
	{
		printf("FAIL: cannot load '%s'\n", argv[1]);
		return 1;
	}

	if (!dll.v7_init(argv[2]))
	{
		printf("FAIL: v7_init died: %s\n", dll.v7_engine_is_dead() ? dll.v7_get_death_reason() : "returned 0");
		v7_unload(&dll);
		return 1;
	}

	if (!dll.v7_initialize_inference())
	{
		printf("FAIL: v7_initialize_inference died: %s\n", dll.v7_engine_is_dead() ? dll.v7_get_death_reason() : "returned 0");
		v7_unload(&dll);
		return 1;
	}

	char output[OUT_BUF_SIZE];
	dll.v7_chat("What is the capital of france?", 0, output, sizeof(output), 1);
	if (dll.v7_engine_is_dead())
	{
		printf("FAIL: chat died: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	char lower[OUT_BUF_SIZE];
	for (int i = 0; output[i] != 0 && i < OUT_BUF_SIZE - 1; i++)
		lower[i] = (char)((output[i] >= 'A' && output[i] <= 'Z') ? output[i] + 32 : output[i]);
	lower[strlen(output)] = 0;
	if (strstr(lower, "paris") == 0)
	{
		printf("FAIL: output does not contain paris:\n%s\n", output);
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}
	printf("safe-mode chat answered paris -- ok\n");

	dll.v7_free_inference();
	v7_unload(&dll);
	printf("v7_only_safe_check: all passed\n");
	return 0;
}
