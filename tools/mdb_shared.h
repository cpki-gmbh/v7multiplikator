// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* mdb_shared.h -- MicroDB hash table struct layout (header-only)
 *
 * Shared by all DB query tools (mdb_info, mdb_list, mdb_query_key, mdb_search, mdb_dump)
 * and v_mdb_validate. Provides the struct offsets and constants needed to walk g_ht
 * from raw bytes.
 *
 * Requires v7_lib.h to be included FIRST (for ht_find_fn, db_load_fn, etc.)
 *
 * Usage (in each mdb_*.cpp):
 *   #include "v7_lib.h"
 *   #include "mdb_shared.h"
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

/* ======================================================================
 * g_ht layout constants (match v7_core.inc)
 * ====================================================================== */

#define MDB_HT_TABLE 0  /* qword -- pointer to Entry[] on heap */
#define MDB_HT_CAP 8    /* dword -- capacity (power of two) */
#define MDB_HT_MASK 12  /* dword -- capacity - 1 */
#define MDB_HT_COUNT 16 /* dword -- number of stored entries */
/* +20..+23 padding */
#define MDB_KP_BASE 24 /* qword -- key/value pool arena base pointer */
#define MDB_KP_USED 32 /* qword -- bytes used in pool (64-bit) */
#define MDB_KP_CAP 40  /* qword -- pool capacity (64-bit) */
#define MDB_HT_SIZE 48 /* total g_ht struct size */

/* ======================================================================
 * Entry struct layout (match v7_core.inc)
 * ====================================================================== */

#define MDB_E_HASH 0 /* dword -- hash (0 = empty, 0xFFFFFFFF = tombstone) */
/* +4..+7 padding */
#define MDB_E_DLEN 8    /* qword -- data byte count */
#define MDB_E_KLEN 16   /* dword -- padded key byte count */
#define MDB_E_FLAG 20   /* dword -- flag bits (bit 0 = persistent) */
#define MDB_E_KEY 24    /* qword -- pool offset of key string */
#define MDB_E_DATA 32   /* qword -- pool offset of value data */
#define MDB_ENTRY_SZ 40 /* total entry size */

#define MDB_TOMBSTONE 0xFFFFFFFFu /* sentinel: deleted slot */
#define MDB_PERSISTENT 1u         /* flag bit 0 = persistent across db_save */

/* ======================================================================
 * Raw g_ht view struct
 * ====================================================================== */

typedef struct mdb_ht_raw
{
	unsigned char *base;        /* raw pointer to g_ht (from GetProcAddress) */
	unsigned __int64 table_ptr; /* MDB_HT_TABLE: Entry[] array */
	unsigned int cap;           /* MDB_HT_CAP */
	unsigned int mask;          /* MDB_HT_MASK */
	unsigned int count;         /* MDB_HT_COUNT */
	unsigned __int64 kp_base;   /* MDB_KP_BASE: pool base */
	unsigned __int64 kp_used;   /* MDB_KP_USED */
	unsigned __int64 kp_cap;    /* MDB_KP_CAP */

} mdb_ht_raw;

/* ======================================================================
 * System keys (non-vocab entries stored by save_db_all / build_db)
 * ====================================================================== */

static const char *mdb_system_keys[] = { "__sys_config",     "__sys_model_path",    "__sys_model_weights", "__sys_weight_meta", "__sys_tokenizer_hdr", "__sys_merge_scores", "__sys_vocab_bulk",
	                                     "__sys_template",   "__sys_template_sys",  "__sys_rope_cos",      "__sys_rope_sin",    "__sys_mem_layout",    "__sys_rope_base",    0 };

static int mdb_is_system_key(const char *key)
{
	for (int i = 0; mdb_system_keys[i]; i++)
		if ((strcmp(key, mdb_system_keys[i]) == 0))
			return 1;
	return 0;
}

/* ======================================================================
 * Inline helpers for reading g_ht from raw bytes
 * ====================================================================== */

/* Read all g_ht fields from the raw pointer. Returns 1 if valid, 0 if empty. */
static int mdb_ht_read(mdb_ht_raw *h, unsigned char *g_ht_raw)
{
	h->base = g_ht_raw;
	h->table_ptr = *(unsigned __int64 *)(g_ht_raw + MDB_HT_TABLE);
	h->cap = *(unsigned int *)(g_ht_raw + MDB_HT_CAP);
	h->mask = *(unsigned int *)(g_ht_raw + MDB_HT_MASK);
	h->count = *(unsigned int *)(g_ht_raw + MDB_HT_COUNT);
	h->kp_base = *(unsigned __int64 *)(g_ht_raw + MDB_KP_BASE);
	h->kp_used = *(unsigned __int64 *)(g_ht_raw + MDB_KP_USED);
	h->kp_cap = *(unsigned __int64 *)(g_ht_raw + MDB_KP_CAP);
	return ((h->table_ptr != 0) && (h->cap != 0));
}

/* Get entry pointer by slot index */
static unsigned char *mdb_entry_at(mdb_ht_raw *h, unsigned int slot)
{
	return (unsigned char *)(h->table_ptr + (unsigned __int64)slot * MDB_ENTRY_SZ);
}

/* Read entry fields */
static unsigned int mdb_entry_hash(unsigned char *entry_ptr)
{
	return *(unsigned int *)(entry_ptr + MDB_E_HASH);
}
static unsigned __int64 mdb_entry_dlen(unsigned char *entry_ptr)
{
	return *(unsigned __int64 *)(entry_ptr + MDB_E_DLEN);
}
static unsigned int mdb_entry_klen(unsigned char *entry_ptr)
{
	return *(unsigned int *)(entry_ptr + MDB_E_KLEN);
}
static unsigned int mdb_entry_flag(unsigned char *entry_ptr)
{
	return *(unsigned int *)(entry_ptr + MDB_E_FLAG);
}
static unsigned __int64 mdb_entry_key_off(unsigned char *entry_ptr)
{
	return *(unsigned __int64 *)(entry_ptr + MDB_E_KEY);
}
static unsigned __int64 mdb_entry_data_off(unsigned char *entry_ptr)
{
	return *(unsigned __int64 *)(entry_ptr + MDB_E_DATA);
}

/* Get key string pointer (from pool) */
static char *mdb_entry_key(mdb_ht_raw *h, unsigned char *entry_ptr)
{
	return (char *)(h->kp_base + mdb_entry_key_off(entry_ptr));
}

/* Get value data pointer (from pool) */
static unsigned char *mdb_entry_data(mdb_ht_raw *h, unsigned char *entry_ptr)
{
	return (unsigned char *)(h->kp_base + mdb_entry_data_off(entry_ptr));
}

/* ======================================================================
 * Hex dump helper (for mdb_query_key and mdb_dump)
 * ====================================================================== */

static void mdb_hex_dump(unsigned char *data, __int64 len, __int64 max_bytes)
{
	__int64 show = (len < max_bytes) ? len : max_bytes;
	for (__int64 row = 0; row < show; row += 16)
	{
		printf("  %04X: ", (int)row);
		__int64 end = row + 16;
		if (end > show)
			end = show;
		for (__int64 i = row; i < end; i++)
			printf("%02X ", data[i]);
		for (__int64 i = end; i < row + 16; i++)
			printf("   ");
		printf(" |");
		for (__int64 i = row; i < end; i++)
		{
			unsigned char c = data[i];
			if (((c >= 0x20) && (c <= 0x7E)))
				printf("%c", c);
			else
				printf(".");
		}
		printf("|\n");
	}
	if (len > max_bytes)
		printf("  ... (%lld more bytes)\n", (long long)(len - max_bytes));
}

/* Print value with type inference (int32, float, string, binary) */
static void mdb_print_value_typed(unsigned char *data, __int64 len)
{
	if (len == 4)
	{
		int i32 = *(int *)data;
		float f32 = *(float *)data;
		printf("  int32: %d  float: %.6g\n", i32, f32);
	}
	else if (((len > 0) && (len < 4096)))
	{
		int printable = 1;
		for (__int64 i = 0; i < len; i++)
		{
			if (((data[i] < 0x20) && (data[i] != '\n') && (data[i] != '\r') && (data[i] != '\t')))
			{
				printable = 0;
				break;
			}
			if (data[i] > 0x7E)
			{
				printable = 0;
				break;
			}
		}
		if (printable)
			printf("  string: \"%.*s\"\n", (int)len, data);
		mdb_hex_dump(data, len, 512);
	}
	else
	{
		mdb_hex_dump(data, len, 512);
	}
}
