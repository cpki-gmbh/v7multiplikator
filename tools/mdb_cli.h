// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * mdb_cli.h -- shared CLI open/read helpers for the mdb_* DB tools.
 *
 * Include AFTER v7_lib.h, zydis_debug_header.h and mdb_shared.h.
 * mdb_cli_open loads the DLL and inits the DB; a failed load is named and
 * nonzero -- reporting on an unloaded DB would print empty results as if
 * they were the truth. mdb_cli_read_ht fetches the raw g_ht hash-table view
 * for the tools that walk the table directly.
 */
#pragma once

static inline int mdb_cli_open(v7_lib *lib, const char *dll_path, const char *db_path)
{
	ZydisInit();
	if (v7_load(dll_path, lib))
		return 0;
	if (!lib->v7_init(db_path) || lib->v7_engine_is_dead())
	{
		printf("Fail: v7_init failed for %s", db_path);
		if (lib->v7_engine_is_dead() && lib->v7_get_death_reason())
			printf(" -- %s", lib->v7_get_death_reason());
		printf("\n");
		v7_unload(lib);
		return 0;
	}
	return 1;
}

static inline int mdb_cli_read_ht(v7_lib *lib, mdb_ht_raw *h)
{
	unsigned char *g_ht = (unsigned char *)GetProcAddress((HMODULE)lib->module, "g_ht");
	if (!g_ht)
	{
		printf("Fail: g_ht not exported\n");
		v7_unload(lib);
		return 0;
	}
	if (!mdb_ht_read(h, g_ht))
	{
		printf("Fail: g_ht is empty\n");
		v7_unload(lib);
		return 0;
	}
	return 1;
}
