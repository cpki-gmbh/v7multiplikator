// v7 - part of the v7 inference engine
// Copyright (C) 2026 Gorg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* transplant_ini.c -- transplant INI shape settings into a cycle-tuner
 * population file's tail slots.
 *
 * Reads a standalone INI (one [NxD] section, the winner-genome archive
 * format: 22 dispatch params + pf_dist_0..7 + pf_type_0..7) and overwrites
 * the requested number of TAIL genome slots (127 downward) of a
 * v7_tune_param_<n>_<d> file with that configuration. The population's
 * other slots are untouched; the n/d slot headers are stamped from the
 * file name's <n>_<d> when missing.
 *
 * Layout of one 512-byte genome slot (the engine's shape block):
 *   +0    dword n, +4 dword d
 *   +8    param_set, 24 dwords (PARAM_OFFSET_* order)
 *   +104  pf_dist[0..40]
 *   +300  pf_type[0..40]
 *
 * Usage:
 *   transplant_ini <settings.ini> <v7_tune_param_file> <count>
 *   transplant_ini --all <ini-dir>        transplants every
 *       <n>x<d>.ini-shaped INI in the dir into v7_tune_param_<n>_<d>
 *       next to it (or in the current directory), count = 16 per default
 *       in --all mode.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define GENOME_BYTES 512
#define SLOTS 128
#define PARAM_OFF 8        /* param_set dword offset inside the slot */
#define PF_DIST_OFF 104    /* pf_dist[0..40] dword offset */
#define PF_TYPE_OFF 300    /* pf_type[0..40] dword offset */
#define PF_SLOTS 41

#define DEFAULT_TRANSPLANT_COUNT 16

/* The 22 dispatch params in the archive INI key order, as param_set dword
 * indices (the PARAM_OFFSET_* numbering / 4). Mirrors
 * population_to_inis.py exactly. */
static const int param_key_index[22] = {
    0,  /* num_chunks */
    2,  /* batch_small */
    3,  /* batch_medium */
    4,  /* batch_large */
    5,  /* batch_huge */
    6,  /* batch_med_thresh */
    7,  /* batch_large_thresh */
    8,  /* batch_huge_thresh */
    9,  /* chunk_multiplier */
    10, /* min_chunk_cols */
    13, /* caller_batch_small */
    14, /* caller_batch_medium */
    15, /* caller_batch_large */
    16, /* caller_batch_huge */
    17, /* caller_batch_med_thresh */
    18, /* caller_batch_large_thresh */
    19, /* caller_batch_huge_thresh */
    20, /* chunk_split_pct */
    21, /* chunk_group_1_count */
    22, /* chunk_group_2_count */
    1,  /* taper_shift */
    12, /* quantize_chunk_groups */
};
static const char* param_key_names[22] = {
    "num_chunks", "batch_small", "batch_medium", "batch_large",
    "batch_huge", "batch_med_thresh", "batch_large_thresh",
    "batch_huge_thresh", "chunk_multiplier", "min_chunk_cols",
    "caller_batch_small", "caller_batch_medium", "caller_batch_large",
    "caller_batch_huge", "caller_batch_med_thresh",
    "caller_batch_large_thresh", "caller_batch_huge_thresh",
    "chunk_split_pct", "chunk_group_1_count", "chunk_group_2_count",
    "taper_shift", "quantize_chunk_groups",
};

/* one parsed INI genome: everything the slot fields get */
typedef struct ini_genome
{
    int n;
    int d;
    unsigned int params[24];
    int pf_dist[PF_SLOTS];
    unsigned int pf_type[PF_SLOTS];
} ini_genome;

/* case-insensitive key=value lookup inside an INI text body */
static int ini_lookup(const char* text, const char* key, int* out_value)
{
    size_t key_len = strlen(key);
    const char* p = text;
    while ((p = strstr(p, key)) != NULL)
    {
        /* word boundary on the left: start of line or non-key char */
        if (p != text && p[-1] != '\n' && p[-1] != ' ' && p[-1] != '\t')
        {
            p += key_len;
            continue;
        }
        const char* q = p + key_len;
        while (*q == ' ' || *q == '\t')
        {
            q++;
        }
        if (*q != '=')
        {
            p += key_len;
            continue;
        }
        q++;
        *out_value = atoi(q);
        return 1;
    }
    return 0;
}

/* parse a standalone winner-genome INI into a genome struct */
static int parse_ini_genome(const char* path, ini_genome* genome)
{
    FILE* f = fopen(path, "rb");
    if (!f)
    {
        printf("cannot open %s\n", path);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* text = (char*)malloc(size + 1);
    if (!text)
    {
        printf("out of memory reading %s\n", path);
        fclose(f);
        return 0;
    }
    size_t read = fread(text, 1, size, f);
    text[read] = 0;
    fclose(f);

    int i;
    for (i = 0; i < 22; i++)
    {
        if (!ini_lookup(text, param_key_names[i], (int*)&genome->params[param_key_index[i]]))
        {
            genome->params[param_key_index[i]] = 0;
        }
    }
    for (i = 0; i < PF_SLOTS; i++)
    {
        char key[32];
        sprintf(key, "pf_dist_%d", i);
        if (!ini_lookup(text, key, &genome->pf_dist[i]))
        {
            genome->pf_dist[i] = 0;
        }
        sprintf(key, "pf_type_%d", i);
        if (!ini_lookup(text, key, (int*)&genome->pf_type[i]))
        {
            genome->pf_type[i] = 0;
        }
    }
    /* n/d: [NxD] section header or filename fallback */
    genome->n = 0;
    genome->d = 0;
    {
        const char* brace = strchr(text, '[');
        if (brace)
        {
            int nn = 0;
            int dd = 0;
            if (sscanf(brace, "[%dx%d]", &nn, &dd) == 2)
            {
                genome->n = nn;
                genome->d = dd;
            }
        }
    }
    if (genome->n == 0 || genome->d == 0)
    {
        /* filename fallback: the INI name is <n>x<d>[_tag].ini */
        const char* base = strrchr(path, '/');
        base = base ? base + 1 : strrchr(path, '\\');
        base = base ? base + 1 : path;
        int nn = 0;
        int dd = 0;
        if (sscanf(base, "%dx%d", &nn, &dd) == 2)
        {
            genome->n = nn;
            genome->d = dd;
        }
    }
    free(text);
    if (genome->n == 0 || genome->d == 0)
    {
        printf("%s: cannot derive n/d (neither [NxD] nor filename)\n", path);
        return 0;
    }
    return 1;
}

/* overwrite count tail slots (127 downward) of the population file */
static int transplant(const char* param_path, const ini_genome* genome, int count)
{
    FILE* f = fopen(param_path, "r+b");
    if (!f)
    {
        printf("cannot open %s\n", param_path);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size != GENOME_BYTES * SLOTS)
    {
        printf("%s: %ld bytes, expected %d -- not a population file\n",
               param_path, size, GENOME_BYTES * SLOTS);
        fclose(f);
        return 0;
    }

    int written = 0;
    int i;
    for (i = 0; i < count; i++)
    {
        int slot = SLOTS - 1 - i; /* tail: 127, 126, ... */
        long base = (long)slot * GENOME_BYTES;
        if (fseek(f, base, SEEK_SET) != 0)
        {
            printf("%s: seek to slot %d failed\n", param_path, slot);
            break;
        }
        unsigned int slot_buf[GENOME_BYTES / 4];
        if (fread(slot_buf, 4, GENOME_BYTES / 4, f) != GENOME_BYTES / 4)
        {
            printf("%s: read slot %d failed\n", param_path, slot);
            break;
        }
        slot_buf[0] = (unsigned int)genome->n;
        slot_buf[1] = (unsigned int)genome->d;
        memcpy(&slot_buf[PARAM_OFF / 4], genome->params, sizeof(genome->params));
        memcpy(&slot_buf[PF_DIST_OFF / 4], genome->pf_dist, sizeof(genome->pf_dist));
        memcpy(&slot_buf[PF_TYPE_OFF / 4], genome->pf_type, sizeof(genome->pf_type));
        if (fseek(f, base, SEEK_SET) != 0)
        {
            printf("%s: seek back to slot %d failed\n", param_path, slot);
            break;
        }
        if (fwrite(slot_buf, 4, GENOME_BYTES / 4, f) != GENOME_BYTES / 4)
        {
            printf("%s: write slot %d failed\n", param_path, slot);
            break;
        }
        written++;
    }
    fclose(f);
    return written;
}

int main(int argc, char** argv)
{
    if (argc == 4 && strcmp(argv[1], "--all") == 0)
    {
        /* --all <ini-dir> [count] : transplant every <n>x<d>*.ini into the
         * matching v7_tune_param_<n>_<d> in the current directory */
        int count = argc == 4 ? atoi(argv[3]) : DEFAULT_TRANSPLANT_COUNT;
        char filter[MAX_PATH];
        sprintf(filter, "%s\\*.ini", argv[2]);
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA(filter, &fd);
        if (h == INVALID_HANDLE_VALUE)
        {
            printf("no INIs in %s\n", argv[2]);
            return 1;
        }
        int total = 0;
        do
        {
            int nn = 0;
            int dd = 0;
            if (sscanf(fd.cFileName, "%dx%d", &nn, &dd) != 2)
            {
                continue;
            }
            char ini_path[MAX_PATH];
            sprintf(ini_path, "%s\\%s", argv[2], fd.cFileName);
            char param_path[MAX_PATH];
            sprintf(param_path, "v7_tune_param_%d_%d", nn, dd);
            ini_genome genome;
            if (!parse_ini_genome(ini_path, &genome))
            {
                continue;
            }
            int w = transplant(param_path, &genome, count);
            printf("%s -> %s: %d tail slots\n", fd.cFileName, param_path, w);
            total += w;
        } while (FindNextFileA(h, &fd));
        FindClose(h);
        printf("total: %d tail slots transplanted\n", total);
        return total > 0 ? 0 : 1;
    }

    if (argc != 4)
    {
        printf("usage: transplant_ini <settings.ini> <v7_tune_param_file> <count>\n");
        printf("       transplant_ini --all <ini-dir> [count]\n");
        return 1;
    }
    int count = atoi(argv[3]);
    if (count < 0)
    {
        count = 0;
    }
    if (count > SLOTS)
    {
        count = SLOTS;
    }
    ini_genome genome;
    if (!parse_ini_genome(argv[1], &genome))
    {
        return 1;
    }
    int w = transplant(argv[2], &genome, count);
    printf("%s [%dx%d] -> %s: %d tail slots\n", argv[1], genome.n, genome.d, argv[2], w);
    return w > 0 ? 0 : 1;
}
