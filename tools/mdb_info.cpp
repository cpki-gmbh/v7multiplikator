// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * mdb_info.exe -- Show MicroDB hash table summary
 *
 * Usage: mdb_info.exe <dll_path> --db <db_path>
 *
 * Prints: capacity, entry count, persistent/temp breakdown, load factor, pool usage
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "mdb_shared.h"
#include "mdb_cli.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv)
{
	char *dll_path = 0;
	char *db_path = 0;
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "--db") == 0 && i + 1 < argc)
		{
			db_path = argv[++i];
		}
		else if (!dll_path)
			dll_path = argv[i];
	}
	if (!dll_path || !db_path)
	{
		printf("usage: mdb_info.exe <dll_path> --db <db_path>\n");
		return 1;
	}

	v7_lib lib;
	if (!mdb_cli_open(&lib, dll_path, db_path))
		return 1;

	struct mdb_ht_raw h;
	if (!mdb_cli_read_ht(&lib, &h))
		return 1;

	int persist_count = 0;
	int temp_count = 0;
	for (unsigned int i = 0; i < h.cap; i++)
	{
		unsigned char *entry_ptr = mdb_entry_at(&h, i);
		unsigned int hash = mdb_entry_hash(entry_ptr);
		if (hash == 0 || hash == MDB_TOMBSTONE)
			continue;
		if (mdb_entry_flag(entry_ptr) & MDB_PERSISTENT)
			persist_count++;
		else
			temp_count++;
	}

	double load_factor = (double)(persist_count + temp_count) / h.cap * 100.0;

	printf("MicroDB Info: %s\n", argv[1]);
	printf("  capacity:     %u\n", h.cap);
	printf("  entries:      %d  (persist=%d, temp=%d)\n", persist_count + temp_count, persist_count, temp_count);
	printf("  load factor:  %.1f%%\n", load_factor);
	printf("  pool used:    %llu / %llu bytes\n", (unsigned long long)h.kp_used, (unsigned long long)h.kp_cap);

	v7_unload(&lib);
	return 0;
}