// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#include "simpledb.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned int crc32_simple(unsigned char *data, int len)
{
	unsigned int crc = 0xFFFFFFFF;
	int i;
	int j;

	for (i = 0; i < len; i++)
	{
		crc ^= data[i];
		for (j = 0; j < 8; j++)
		{
			if (crc & 1)
				crc = (crc >> 1) ^ 0xEDB88320;
			else
				crc >>= 1;
		}
	}
	return crc ^ 0xFFFFFFFF;
}

struct sdb_entry *entry_create(char *key, char *value, int value_len)
{
	struct sdb_entry *entry = (struct sdb_entry *)malloc(sizeof(struct sdb_entry));
	if (!entry)
		return 0;

	int key_len = (int)strlen(key);
	if (key_len >= SDB_MAX_KEY)
		key_len = SDB_MAX_KEY - 1;

	memcpy(entry->key, key, key_len);
	entry->key[key_len] = '\0';
	entry->value_len = value_len;
	entry->value = (char *)malloc(value_len + 1);
	if (!entry->value)
	{
		free(entry);
		return 0;
	}
	memcpy(entry->value, value, value_len);
	entry->value[value_len] = '\0';

	int crc_data_len = key_len + value_len + sizeof(int);
	unsigned char *crc_buf = (unsigned char *)malloc(crc_data_len);
	if (!crc_buf)
	{
		free(entry->value);
		free(entry);
		return 0;
	}

	int offset = 0;
	memcpy(crc_buf + offset, entry->key, key_len);
	offset += key_len;
	memcpy(crc_buf + offset, &entry->value_len, sizeof(int));
	offset += sizeof(int);
	memcpy(crc_buf + offset, entry->value, value_len);

	entry->crc = crc32_simple(crc_buf, crc_data_len);
	free(crc_buf);
	return entry;
}

void entry_free(struct sdb_entry *entry)
{
	if (entry)
	{
		free(entry->value);
		free(entry);
	}
}

int entry_valid(struct sdb_entry *entry)
{
	if (!entry || !entry->value)
		return 0;

	int key_len = (int)strlen(entry->key);
	int crc_data_len = key_len + entry->value_len + sizeof(int);
	unsigned char *crc_buf = (unsigned char *)malloc(crc_data_len);
	if (!crc_buf)
		return 0;

	int offset = 0;
	memcpy(crc_buf + offset, entry->key, key_len);
	offset += key_len;
	memcpy(crc_buf + offset, &entry->value_len, sizeof(int));
	offset += sizeof(int);
	memcpy(crc_buf + offset, entry->value, entry->value_len);

	unsigned int calc_crc = crc32_simple(crc_buf, crc_data_len);
	free(crc_buf);
	return (calc_crc == entry->crc);
}

void sdb_cleanup_entries(struct sdb *db, int up_to)
{
	for (int i = 0; i < up_to; i++)
		entry_free(db->entries[i]);
}

enum sdb_err sdb_open(struct sdb *db, char *path)
{
	memset(db, 0, sizeof(*db));
	strncpy(db->db_path, path, sizeof(db->db_path) - 1);
	db->db_path[sizeof(db->db_path) - 1] = '\0';

	db->capacity = 64;
	db->count = 0;
	db->entries = (struct sdb_entry **)calloc(db->capacity, sizeof(struct sdb_entry *));
	if (!db->entries)
		return SDB_ERR_MALLOC;

	FILE *fp = fopen(path, "rb");
	if (!fp)
		return SDB_OK;

	unsigned int magic;
	unsigned int version;
	if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != SDB_MAGIC)
		goto corrupt_cleanup;
	if (fread(&version, sizeof(version), 1, fp) != 1 || version != SDB_VERSION)
		goto corrupt_cleanup;

	int file_count;
	if (fread(&file_count, sizeof(file_count), 1, fp) != 1)
		goto corrupt_cleanup;
	if (file_count < 0 || file_count > 1000000)
		goto corrupt_cleanup;

	int i;
	for (i = 0; i < file_count; i++)
	{
		int key_len;
		int value_len;
		if (fread(&key_len, sizeof(key_len), 1, fp) != 1)
			break;
		if (fread(&value_len, sizeof(value_len), 1, fp) != 1)
			break;
		if ((key_len <= 0) || (key_len >= SDB_MAX_KEY))
			break;
		if ((value_len < 0) || (value_len > 100 * 1024 * 1024))
			break;

		char *tmp_key = (char *)malloc(key_len + 1);
		char *tmp_val = (char *)malloc(value_len + 1);
		if (!tmp_key || !tmp_val)
		{
			free(tmp_key);
			free(tmp_val);
			break;
		}

		if (fread(tmp_key, 1, key_len, fp) != (int)key_len)
		{
			free(tmp_key);
			free(tmp_val);
			break;
		}
		if (fread(tmp_val, 1, value_len, fp) != (int)value_len)
		{
			free(tmp_key);
			free(tmp_val);
			break;
		}
		tmp_key[key_len] = '\0';

		unsigned int stored_crc;
		if (fread(&stored_crc, sizeof(stored_crc), 1, fp) != 1)
		{
			free(tmp_key);
			free(tmp_val);
			break;
		}

		int crc_data_len = key_len + value_len + sizeof(int);
		unsigned char *crc_buf = (unsigned char *)malloc(crc_data_len);
		if (!crc_buf)
		{
			free(tmp_key);
			free(tmp_val);
			break;
		}
		int offset = 0;
		memcpy(crc_buf + offset, tmp_key, key_len);
		offset += key_len;
		memcpy(crc_buf + offset, &value_len, sizeof(int));
		offset += sizeof(int);
		memcpy(crc_buf + offset, tmp_val, value_len);
		unsigned int calc_crc = crc32_simple(crc_buf, crc_data_len);
		free(crc_buf);

		if (calc_crc != stored_crc)
		{
			free(tmp_key);
			free(tmp_val);
			continue;
		}

		if (db->count >= db->capacity)
		{
			if (db->capacity > 0x7FFFFFFF)
			{
				free(tmp_key);
				free(tmp_val);
				break;
			}
			db->capacity *= 2;
			struct sdb_entry **new_entries = (struct sdb_entry **)realloc(db->entries, db->capacity * sizeof(struct sdb_entry *));
			if (!new_entries)
			{
				free(tmp_key);
				free(tmp_val);
				break;
			}
			db->entries = new_entries;
		}

		struct sdb_entry *entry = (struct sdb_entry *)malloc(sizeof(struct sdb_entry));
		if (!entry)
		{
			free(tmp_key);
			free(tmp_val);
			break;
		}
		memcpy(entry->key, tmp_key, key_len + 1);
		entry->value_len = value_len;
		entry->value = tmp_val;
		entry->crc = stored_crc;
		free(tmp_key);

		db->entries[db->count++] = entry;
	}

	fclose(fp);
	if (i < file_count)
	{
		sdb_cleanup_entries(db, db->count);
		free(db->entries);
		memset(db, 0, sizeof(*db));
		return SDB_ERR_CORRUPT;
	}
	return SDB_OK;

corrupt_cleanup:
	if (fp)
		fclose(fp);
	sdb_cleanup_entries(db, db->count);
	free(db->entries);
	memset(db, 0, sizeof(*db));
	return SDB_ERR_CORRUPT;
}

void sdb_close(struct sdb *db)
{
	int i;
	for (i = 0; i < db->count; i++)
	{
		entry_free(db->entries[i]);
	}
	free(db->entries);
	memset(db, 0, sizeof(*db));
}

enum sdb_err sdb_put(struct sdb *db, char *key, char *value, int value_len)
{
	if (!key || !value)
		return SDB_ERR_MALLOC;
	if (strlen(key) >= SDB_MAX_KEY)
		return SDB_ERR_KEY_TOO_LONG;
	if (value_len < 0)
		return SDB_ERR_VALUE_TOO_LARGE;

	int i;
	for (i = 0; i < db->count; i++)
	{
		if (strcmp(db->entries[i]->key, key) == 0)
		{
			struct sdb_entry *new_entry = entry_create(key, value, value_len);
			if (!new_entry)
				return SDB_ERR_MALLOC;
			entry_free(db->entries[i]);
			db->entries[i] = new_entry;
			return SDB_OK;
		}
	}

	if (db->count >= db->capacity)
	{
		if (db->capacity > 0x7FFFFFFF)
			return SDB_ERR_MALLOC;
		int new_capacity = db->capacity * 2;
		struct sdb_entry **new_entries = (struct sdb_entry **)realloc(db->entries, new_capacity * sizeof(struct sdb_entry *));
		if (!new_entries)
			return SDB_ERR_MALLOC; /* capacity must only grow with entries: a doubled capacity over the old array makes the NEXT put write out of bounds */
		db->capacity = new_capacity;
		db->entries = new_entries;
	}

	db->entries[db->count] = entry_create(key, value, value_len);
	if (!db->entries[db->count])
		return SDB_ERR_MALLOC;
	db->count++;
	return SDB_OK;
}

enum sdb_err sdb_get(struct sdb *db, char *key, char **out_value, int *out_len)
{
	if (!key)
		return SDB_ERR_NOT_FOUND;
	int i;
	for (i = 0; i < db->count; i++)
	{
		if (strcmp(db->entries[i]->key, key) == 0)
		{
			if (out_len)
				*out_len = db->entries[i]->value_len;
			if (out_value)
			{
				*out_value = (char *)malloc(db->entries[i]->value_len + 1);
				if (!*out_value)
					return SDB_ERR_MALLOC;
				memcpy(*out_value, db->entries[i]->value, db->entries[i]->value_len);
				(*out_value)[db->entries[i]->value_len] = '\0';
			}
			return SDB_OK;
		}
	}
	return SDB_ERR_NOT_FOUND;
}

enum sdb_err sdb_remove(struct sdb *db, char *key)
{
	int i;
	for (i = 0; i < db->count; i++)
	{
		if (strcmp(db->entries[i]->key, key) == 0)
		{
			entry_free(db->entries[i]);
			db->count--;
			if (i < db->count)
			{
				memmove(&db->entries[i], &db->entries[i + 1], (db->count - i) * sizeof(struct sdb_entry *));
			}
			return SDB_OK;
		}
	}
	return SDB_ERR_NOT_FOUND;
}

enum sdb_err sdb_has(struct sdb *db, char *key, int *exists)
{
	int i;
	for (i = 0; i < db->count; i++)
	{
		if (strcmp(db->entries[i]->key, key) == 0)
		{
			*exists = 1;
			return SDB_OK;
		}
	}
	*exists = 0;
	return SDB_OK;
}

enum sdb_err sdb_list(struct sdb *db, char ***keys, int *count)
{
	if (db->count == 0)
	{
		*keys = 0;
		*count = 0;
		return SDB_OK;
	}
	*keys = (char **)calloc(db->count, sizeof(char *));
	if (!*keys)
		return SDB_ERR_MALLOC;

	int i;
	for (i = 0; i < db->count; i++)
	{
		(*keys)[i] = strdup(db->entries[i]->key);
	}
	*count = db->count;
	return SDB_OK;
}

enum sdb_err sdb_save(struct sdb *db)
{
	FILE *fp = fopen(db->db_path, "wb");
	if (!fp)
		return SDB_ERR_IO;

	unsigned int magic = SDB_MAGIC;
	unsigned int version = SDB_VERSION;
	if (fwrite(&magic, sizeof(magic), 1, fp) != 1)
	{
		fclose(fp);
		return SDB_ERR_IO;
	}
	if (fwrite(&version, sizeof(version), 1, fp) != 1)
	{
		fclose(fp);
		return SDB_ERR_IO;
	}
	if (fwrite(&db->count, sizeof(db->count), 1, fp) != 1)
	{
		fclose(fp);
		return SDB_ERR_IO;
	}

	int i;
	for (i = 0; i < db->count; i++)
	{
		struct sdb_entry *e = db->entries[i];
		int key_len = (int)strlen(e->key);
		if (fwrite(&key_len, sizeof(key_len), 1, fp) != 1)
		{
			fclose(fp);
			return SDB_ERR_IO;
		}
		if (fwrite(&e->value_len, sizeof(e->value_len), 1, fp) != 1)
		{
			fclose(fp);
			return SDB_ERR_IO;
		}
		if (fwrite(e->key, 1, key_len, fp) != (int)key_len)
		{
			fclose(fp);
			return SDB_ERR_IO;
		}
		if (fwrite(e->value, 1, e->value_len, fp) != (int)e->value_len)
		{
			fclose(fp);
			return SDB_ERR_IO;
		}
		if (fwrite(&e->crc, sizeof(e->crc), 1, fp) != 1)
		{
			fclose(fp);
			return SDB_ERR_IO;
		}
	}

	fclose(fp);
	return SDB_OK;
}
