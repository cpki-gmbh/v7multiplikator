// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* test_v7_execute_fmt.c -- test v7_execute_fmt directly via GetProcAddress */
#include <stdio.h>
#include <windows.h>

typedef char *(*v7_execute_fmt_fn)(char *exe, char *fmt, unsigned long long *argv, int argc);

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		printf("Usage: test_v7_execute_fmt <v7_threading.dll>\n");
		return 1;
	}
	HMODULE dll = LoadLibraryA(argv[1]);
	if (!dll)
	{
		printf("LoadLibrary failed\n");
		return 1;
	}

	v7_execute_fmt_fn v7_execute_fmt = (v7_execute_fmt_fn)GetProcAddress(dll, "v7_execute_fmt");
	if (!v7_execute_fmt)
	{
		printf("v7_execute_fmt not found\n");
		return 1;
	}

	/* test 1: %s format with one string arg */
	{
		unsigned long long args[] = { (unsigned long long)"hello" };
		char *r = v7_execute_fmt("bin\\echo_args.exe", "%s", args, 1);
		printf("Test1: [%s]\n", r);
	}

	/* test 2: ant format, no args */
	{
		unsigned long long args[] = { 0 };
		char *r = v7_execute_fmt("bin\\echo_args.exe", "ant_arg", args, 0);
		printf("Test2: [%s]\n", r);
	}

	/* test 3: %s %s with two string args */
	{
		unsigned long long args[] = { (unsigned long long)"first", (unsigned long long)"second" };
		char *r = v7_execute_fmt("bin\\echo_args.exe", "%s %s", args, 2);
		printf("Test3: [%s]\n", r);
	}

	/* test 4: %d format with one int arg */
	{
		unsigned long long args[] = { 42 };
		char *r = v7_execute_fmt("bin\\echo_args.exe", "%d", args, 1);
		printf("Test4: [%s]\n", r);
	}

	FreeLibrary(dll);
	return 0;
}
