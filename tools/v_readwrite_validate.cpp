// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_readwrite_validate.cpp -- v7s_writefile + v7s_readfile round-trip validator
 *
 * Single test: write "hello world" to a file via v7s_writefile,
 * read it back via v7s_readfile, check the buffer matches.
 *
 * Usage: v_readwrite_validate.exe <dll_path>
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

	/* CWD-independent scratch path: %TEMP%\v7_rw_test.txt (verify.bat runs
	 * this from bin/, where no build_logs\ subdir exists) */
	char path[MAX_PATH];
#ifdef __linux__
	DWORD directory_length = GetTempPathA(MAX_PATH, path);
	if (directory_length == 0 || directory_length >= MAX_PATH - sizeof("v7_rw_test.txt"))
		return 1;
	strcat(path, "v7_rw_test.txt");
#else
	GetTempPathA(MAX_PATH, path);
	strcat_s(path, sizeof(path), "v7_rw_test.txt");
#endif

	/* Test 1: write hello world (overwrite mode), read it back */
	{
		int wok = lib.v7s_writefile(path, (char *)"hello world", 0);
		char *content = lib.v7s_readfile(path);
		int rok = (content != 0 && strcmp(content, "hello world") == 0);
		int ok = wok && rok;
		if (ok)
		{
			pass++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			fail++;
			printf("\n  overwrite write+read:  Fail  content=\"%s\"\n", content ? content : "(null)");
		}
	}

	/* Test 2: append mode -- "hello" + " world" must concatenate */
	{
		int wok = lib.v7s_writefile(path, (char *)"hello", 0);
		wok &= lib.v7s_writefile(path, (char *)" world", 1);
		char *content = lib.v7s_readfile(path);
		int rok = (content != 0 && strcmp(content, "hello world") == 0);
		int ok = wok && rok;
		if (ok)
		{
			pass++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			fail++;
			printf("\n  append write+read:     Fail  content=\"%s\"\n", content ? content : "(null)");
		}
	}

	/* Test 3: overwrite truncates (no append residue) */
	{
		lib.v7s_writefile(path, (char *)"hello", 0);
		lib.v7s_writefile(path, (char *)"short", 0);
		char *content = lib.v7s_readfile(path);
		int ok = (content != 0 && strcmp(content, "short") == 0);
		if (ok)
		{
			pass++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			fail++;
			printf("\n  overwrite truncates:   Fail  content=\"%s\"\n", content ? content : "(null)");
		}
	}

	DeleteFileA(path);

	v7_unload(&lib);
	printf("\nv_readwrite_validate: %d passed, %d failed\n", pass, fail);
	if (fail)
		return 1;
	else
		return 0;
}
