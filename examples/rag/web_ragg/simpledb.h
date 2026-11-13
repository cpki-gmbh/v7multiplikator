// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#define SDB_MAX_KEY 512
#define SDB_MAGIC 0x53444256
#define SDB_VERSION 1

	enum sdb_err
	{
		SDB_OK = 0,
		SDB_ERR_MALLOC,
		SDB_ERR_NOT_FOUND,
		SDB_ERR_CORRUPT,
		SDB_ERR_IO,
		SDB_ERR_KEY_TOO_LONG,
		SDB_ERR_VALUE_TOO_LARGE
	};

	typedef struct sdb_entry
	{
		unsigned int crc;
		char key[SDB_MAX_KEY];
		int value_len;
		char *value;
	} sdb_entry;

	typedef struct sdb
	{
		char db_path[512];
		int count;
		int capacity;
		struct sdb_entry **entries;
	} sdb;

	enum sdb_err sdb_open(struct sdb *db, char *path);
	void sdb_close(struct sdb *db);

	enum sdb_err sdb_put(struct sdb *db, char *key, char *value, int value_len);
	enum sdb_err sdb_get(struct sdb *db, char *key, char **out_value, int *out_len);
	enum sdb_err sdb_remove(struct sdb *db, char *key);
	enum sdb_err sdb_has(struct sdb *db, char *key, int *exists);
	enum sdb_err sdb_list(struct sdb *db, char ***keys, int *count);

	enum sdb_err sdb_save(struct sdb *db);

#ifdef __cplusplus
}
#endif
