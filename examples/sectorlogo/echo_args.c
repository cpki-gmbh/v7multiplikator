// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* echo_args.c -- minimal argv echo for testing v7_execute / v7_execute_fmt
 * prints each argv[i] on its own line, plus count and the raw GetCommandLineA.
 */
#include <stdio.h>
#include <windows.h>

int main(int argc, char **argv)
{
	printf("argc=%d\n", argc);
	for (int i = 0; i < argc; i++)
		printf("arg%d=[%s]\n", i, argv[i]);
#ifdef __linux__
	printf("cmdline=[%s]\n", "");
#else
	printf("cmdline=[%s]\n", GetCommandLineA());
#endif
	return argc;
}
