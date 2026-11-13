// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * c_callback_example.c -- demonstrates sector_logo "HOST" FFI callbacks
 *
 * the host EXE exports a C function. sector_logo scripts can call it via
 * CallLibrary [ "HOST" "my_callback" "integer" [ "integer" ] ] 42
 *
 * build:  built as an EXE by build.bat (step 4).
 *          the EXE itself is the "HOST" module -- its exported
 *          my_callback/greet are found via GetProcAddress on
 *          GetModuleHandleA(0).
 *
 * the key API call is v7s_set_caller_hmodule(GetModuleHandleA(0)), which
 * registers the host's module handle so the FFI resolver can find exported
 * functions via GetProcAddress on it.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

/* ---- the callback that sector_logo will invoke ---- */

__declspec(dllexport) int my_callback(int value)
{
	printf("[C CALLBACK] my_callback called with value = %d\n", value);
	return value * 2;
}

__declspec(dllexport) int greet(char *name)
{
	printf("[C CALLBACK] hello, %s! (from sector_logo FFI)\n", name ? name : "world");
	return 0;
}

/* ---- script that calls back into this EXE ----
 * canonical FFI form: module and proc QUOTED, type names quoted lowercase
 * ("integer"/"text" are the interned STRING atoms; a bare word atom is a
 * DIFFERENT atom and FATALs in the ret-type check). */
char *SIDEKICK_SCRIPT = "MAKE \"result\" CallLibrary [ \"HOST\" \"my_callback\" \"integer\" [ \"integer\" ] ] 42 \n"
                        "PRINT :result \n"
                        "CallLibrary [ \"HOST\" \"greet\" \"integer\" [ \"text\" ] ] \"SectorLogo\" \n";

/* ---- minimal host that runs inference with the callback hook ---- */

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		printf("Usage: %s <v7_threading.dll> <lz4ht.db> [prompt]\n", argv[0]);
		return 1;
	}

	char *dll_path = argv[1];
	char *db_path = argv[2];
	char *prompt = (argc > 3) ? argv[3] : "What is a cat?";

	struct v7_lib lib;
	if (v7_load(dll_path, &lib) != 0)
	{
		printf("Failed to load %s\n", dll_path);
		return 1;
	}

	/* register THIS module so sector_logo can call my_callback/greet */
	lib.v7s_set_caller_hmodule(GetModuleHandleA(0));
	printf("[Host] registered caller module handle for \"HOST\" FFI\n");

	/* load the sidecar script into micro_d_b */
	if (!lib.v7_init((char *)db_path))
	{
		if (lib.v7_engine_is_dead())
			printf("V7_init died: %s\n", lib.v7_get_death_reason());
		else
			printf("V7_init failed for %s\n", db_path);
		v7_unload(&lib);
		return 1;
	}
	if (!lib.v7_db_add("logo:sidecar", (void *)SIDEKICK_SCRIPT, (int)strlen(SIDEKICK_SCRIPT) + 1, 0))
	{
		printf("[FATAL] v7_db_add('logo:sidecar') failed -- the hook would run with no script\n");
		v7_unload(&lib);
		return 1;
	}
	printf("[Host] loaded sidecar script into micro_d_b\n");

	/* initialize inference (loads model from DB; context hidden in BSS).
	 * Sec.172: a gated entry -- check the return AND the death flag. */
	if ((!lib.v7_initialize_inference()) || lib.v7_engine_is_dead())
	{
		if (lib.v7_engine_is_dead())
			printf("Inference init died: %s\n", lib.v7_get_death_reason());
		else
			printf("v7_initialize_inference failed\n");
		v7_unload(&lib);
		return 1;
	}

	/* the sidecar script is in the DB (ht_add above): the hook is DATA-DRIVEN
	 * and self-arms from the DB key on the first chat */

	/* run inference -- the hook fires per token and the script calls back */
	char output[4096] = { 0 };
	printf("[Host] running inference with prompt: \"%s\"\n", prompt);
	lib.v7_chat((char *)prompt, 0, output, sizeof(output), 1);
	if (lib.v7_engine_is_dead())
	{
		printf("engine died during chat: %s\n", lib.v7_get_death_reason());
		lib.v7_free_inference();
		v7_unload(&lib);
		return 1;
	}
	printf("\n[HOST] model output:\n%s\n", output);

	/* clean up: join workers first, then free the heap, then unload */
	lib.v7_shutdown();
	lib.v7_free_inference();
	v7_unload(&lib);
	return 0;
}
