// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_execute_validate.cpp -- v7_execute child-process stdout capture validator
 *
 * Validates the v7_execute export by spawning ITSELF with --hello,
 * which makes it print "hello world" to stdout. The parent then
 * checks that the returned buffer contains "hello world".
 *
 * Usage: v_execute_validate.exe <dll_path>
 * Also:  v_execute_validate.exe --hello   (internal: prints greeting and exits)
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#ifdef __linux__
#include <windows.h> /* Linux tool adapters */
#endif
#include "zydis_debug_header.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
	ZydisInit();

	/* ---- child mode: invoked by v7_execute with --hello ---- */
	if (argc >= 2 && strcmp(argv[1], "--hello") == 0)
	{
		printf("hello world\n");
		return 0;
	}

	/* ---- parent mode: normal validator ---- */
	if (argc < 2)
	{
		printf("Usage: %s <dll_path>\n", argv[0]);
		return 1;
	}

	struct v7_lib lib;
	if (v7_load(argv[1], &lib) != 0)
	{
		printf("Fatal: cannot load %s\n", argv[1]);
		return 1;
	}

	int pass = 0;
	int fail = 0;

	/* Test 1: spawn self with --hello, capture stdout */
	{
		char own_path[MAX_PATH];
		GetModuleFileNameA(0, own_path, MAX_PATH);

		char *result = lib.v7_execute(own_path, (char *)"--hello", 0, 0);

		int ok = (result != 0 && strstr(result, "hello world") != 0);
		if (ok)
		{
			pass++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			fail++;
			printf("\n  spawn-self --hello: Fail  buf=\"%s\"\n", result ? result : "(null)");
		}
	}

	/* Test 2: spawn self with no args (should print usage to stderr, stdout empty or just exit) */
	{
		char own_path[MAX_PATH];
		GetModuleFileNameA(0, own_path, MAX_PATH);

		char *result = lib.v7_execute(own_path, 0, 0, 0);

		int ok = (result != 0);
		if (ok)
		{
			pass++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			fail++;
			printf("\n  spawn-self (no args): Fail  buf_len=%d\n", result ? (int)strlen(result) : 0);
		}
	}

	/* Test 3: nonexistent exe (should return empty buffer) */
	{
		char *result = lib.v7_execute((char *)"nonexistent_file_xyz.exe", 0, 0, 0);

		int ok = (result != 0 && result[0] == '\0');
		if (ok)
		{
			pass++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			fail++;
			printf("\n  nonexistent exe:     Fail  buf_empty=%d\n", result ? (result[0] == '\0') : -1);
		}
	}

	/* Test 4: spawn with 3 parameters */
	{
		char own_path[MAX_PATH];
		GetModuleFileNameA(0, own_path, MAX_PATH);

		char *result = lib.v7_execute(own_path, (char *)"--hello", (char *)"extra1", (char *)"extra2");

		int ok = (result != 0 && strstr(result, "hello world") != 0);
		if (ok)
		{
			pass++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			fail++;
			printf("\n  spawn-self 3 args:   Fail  buf_contains_greeting=%d\n", result ? (strstr(result, "hello world") != 0) : 0);
		}
	}

	/* Test 5: argument escaping must be byte-exact -- an embedded '"' and a trailing
	 * backslash both survive into the child's argv. This is what covers the odd-run
	 * literal-quote form and the arg-end doubled run. echo_args.exe prints argc, each
	 * argv[i] and the raw command line. Derive its path from argv[1]'s directory: the
	 * DLL and this harness ship into the same bin\ folder. */
	{
		char exe_path[MAX_PATH];
		char *last_separator;
		char *spawn_result;
		int embedded_quote_ok;
		int trailing_backslash_ok;
		int ok;

		strcpy(exe_path, argv[1]);
#ifdef __linux__
		last_separator = strrchr(exe_path, '/');
#else
		last_separator = strrchr(exe_path, '\\');
#endif
		spawn_result = 0;
		embedded_quote_ok = 0;
		trailing_backslash_ok = 0;
		ok = 0;

		if (last_separator != 0)
		{
#ifdef __linux__
			strcpy(last_separator + 1, "echo_args");
#else
			strcpy(last_separator + 1, "echo_args.exe");
#endif
			spawn_result = lib.v7_execute(exe_path, (char *)"a\"b", (char *)"C:\\dir\\", 0);

			if (spawn_result != 0)
			{
				embedded_quote_ok = (strstr(spawn_result, "arg1=[a\"b]") != 0);
				trailing_backslash_ok = (strstr(spawn_result, "arg2=[C:\\dir\\]") != 0);
			}

			ok = embedded_quote_ok && trailing_backslash_ok;
		}

		if (ok)
		{
			pass++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			fail++;
			printf("\n  arg escaping:       Fail  embedded_quote=%d trailing_backslash=%d buf=\"%s\"\n", embedded_quote_ok, trailing_backslash_ok, spawn_result ? spawn_result : "(null)");
		}
	}

	v7_unload(&lib);
	printf("\nv_execute_validate: %d passed, %d failed\n", pass, fail);
	if (fail)
		return 1;
	else
		return 0;
}
