// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* run_sectorlogo.c -- run a sector_logo script from a .logo file
 *
 * usage:  run_sectorlogo <library> <lz4ht.db> <script.logo>
 *   (<library> is the engine library: v7_threading.dll on Windows,
 *    libv7_threading.so on Linux)
 *
 * loads the DLL, inits with the database, reads the script file,
 * and calls v7s_sectorlogo_interp() to execute it. prints the
 * integer return value (0 = success, nonzero = error).
 *
 * It also exports sc_now_ms, the millisecond clock a Scheme script reaches over
 * the FFI's HOST module path. SectorLogo itself has no time primitive and the v7
 * DLL exports no clock, so the HOSTS that run Scheme programs have to supply one:
 * scheme_runner.c exports the same symbol, and a script that used (current-jiffy)
 * would work under one host and FATAL under the other. This file and
 * the scheme compiler's scheme_runner.c (examples\experimental_work_in_progress
 * \scheme_compiler\tools\) are the two hosts the Scheme test suite and the
 * corpus runner use, so both export it.
 *
 * build:  build.bat (this file is built with the other examples)
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define SCRIPT_MAX (1 << 20)

/* sc_now_ms -- milliseconds since this host's first clock read, so a Scheme
 * script can time itself. Milliseconds rather than microseconds because the
 * FFI's integer return is a signed 32-bit C int that must fit a 30-bit payload:
 * a microsecond counter FATALs after about eighteen minutes of process life,
 * where milliseconds give about twelve days. The first call is the origin and
 * reads 0, so a script that wants an absolute reading adds its own base.
 *
 * The engine calls every CallLibrary target with the Windows register
 * convention on EVERY platform, so a HOST export on Linux must carry the
 * ms_abi attribute or any future argument-taking export would read its
 * parameters from the wrong registers. */
static LARGE_INTEGER clock_frequency;
static LARGE_INTEGER clock_origin;
static int clock_ready = 0;

#ifdef __linux__
__attribute__((ms_abi)) int sc_now_ms(void)
#else
__declspec(dllexport) int __cdecl sc_now_ms(void)
#endif
{
	LARGE_INTEGER ticks;
	long long elapsed_ms;

	if (clock_ready == 0)
	{
		QueryPerformanceFrequency(&clock_frequency);
		QueryPerformanceCounter(&clock_origin);
		clock_ready = 1;
	}

	QueryPerformanceCounter(&ticks);
	elapsed_ms = (ticks.QuadPart - clock_origin.QuadPart) * 1000 / clock_frequency.QuadPart;

	return (int)elapsed_ms;
}

int main(int argc, char **argv)
{
	char *dll_path;
	char *db_path;
	char *script_path;
	char *script;
	FILE *script_file;
	struct v7_lib inference_dll;
	int bytes_read;
	int return_code;

	setvbuf(stdout, 0, _IONBF, 0);
	if (argc < 4)
	{
		printf("Usage: %s <v7_threading.dll> <lz4ht.db> <script.logo>\n", argv[0]);
		return 1;
	}

	dll_path = argv[1];
	db_path = argv[2];
	script_path = argv[3];

	/* v7_load zeroes the struct itself, so the caller does not memset it */
	if (v7_load(dll_path, &inference_dll) != 0)
	{
		printf("Fatal: cannot load %s\n", dll_path);
		return 1;
	}

	if (!inference_dll.v7_init(db_path))
	{
		if (inference_dll.v7_engine_is_dead())
			printf("Fatal: V7_init died: %s\n", inference_dll.v7_get_death_reason());
		else
			printf("Fatal: V7_init failed for %s\n", db_path);
		v7_unload(&inference_dll);
		return 1;
	}

	/* register THIS module so scripts using CallLibrary [ "HOST" ... ] resolve
	 * exports of this exe (same pattern as c_callback_example.c) */
	inference_dll.v7s_set_caller_hmodule(GetModuleHandleA(0));

	script_file = fopen(script_path, "rb");
	if (!script_file)
	{
		printf("Fatal: cannot open %s\n", script_path);
		v7_unload(&inference_dll);
		return 1;
	}

	script = (char *)malloc(SCRIPT_MAX);
	if (!script)
	{
		printf("Fatal: malloc failed\n");
		fclose(script_file);
		v7_unload(&inference_dll);
		return 1;
	}

	bytes_read = (int)fread(script, 1, SCRIPT_MAX - 1, script_file);
	fclose(script_file);
	script[bytes_read] = '\0';

	printf("[run_sectorlogo] executing %s (%u bytes)\n", script_path, bytes_read);

	return_code = inference_dll.v7s_sectorlogo_interp(script);

	/* v7s_sectorlogo_interp is a gated entry: a script FATAL unwinds to here
	 * with the engine parked, returning 0 like a success. The exit code must
	 * say "error" (0 = script ran, 1 = script FATAL); the tagged interp
	 * return value printed below is informational. */
	if (inference_dll.v7_engine_is_dead())
	{
		printf("[run_sectorlogo] script FATAL: %s\n", inference_dll.v7_get_death_reason());
		free(script);
		inference_dll.v7_shutdown();
		v7_unload(&inference_dll);
		return 1;
	}

	printf("[run_sectorlogo] interp returned %d\n", return_code);
	printf("%d\n", return_code);

	free(script);
	inference_dll.v7_shutdown();
	v7_unload(&inference_dll);
	return 0;
}
