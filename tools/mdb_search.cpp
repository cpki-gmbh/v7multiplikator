// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * mdb_search.exe -- Search MicroDB entries by key substring
 *
 * Usage: mdb_search.exe <dll_path> <pattern> --db <db_path> [--max N]
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "mdb_shared.h"
#include "mdb_cli.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv)
{
	char *dll_path = 0;
	char *db_path = 0;
	char *pattern = 0;
	int max_entries = 50;
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--max") == 0 && i + 1 < argc)
		{
			max_entries = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "--db") == 0 && i + 1 < argc)
		{
			db_path = argv[++i];
		}
		else if (!dll_path)
			dll_path = argv[i];
		else if (!pattern)
			pattern = argv[i];
	}
	if (!dll_path || !db_path || !pattern)
	{
		printf("usage: mdb_search.exe <dll_path> <pattern> --db <db_path> [--max N]\n");
		return 1;
	}

	v7_lib lib;
	if (!mdb_cli_open(&lib, dll_path, db_path))
		return 1;

	struct mdb_ht_raw h;
	if (!mdb_cli_read_ht(&lib, &h))
		return 1;

	int shown = 0;
	for (unsigned int i = 0; i < h.cap; i++)
	{
		unsigned char *entry_ptr = mdb_entry_at(&h, i);
		unsigned int hash = mdb_entry_hash(entry_ptr);
		if (hash == 0 || hash == MDB_TOMBSTONE)
			continue;

		char *key = mdb_entry_key(&h, entry_ptr);
		if (strstr(key, pattern))
		{
			printf("%6u  %8llu  %s\n", i, (unsigned long long)mdb_entry_dlen(entry_ptr), key);
			shown++;
			if (shown >= max_entries)
			{
				printf("  ... (max %d reached)\n", max_entries);
				break;
			}
		}
	}

	printf("\nMatches: %d\n", shown);
	v7_unload(&lib);
	return 0;
}