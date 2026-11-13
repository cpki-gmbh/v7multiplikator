// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * bm25_sectorlogo_benchmark.c -- per-phase profiling of the pure-Logo BM25
 * pipeline via HOST FFI slot timers.
 *
 * The host EXE exports two dllexport'd slot functions:
 *   bm25_slot_begin(int slot_id)  -- start the timer for slot_id
 *   bm25_slot_end(int slot_id)    -- stop the timer, accumulate elapsed
 *
 * The Logo script calls them via CallLibrary [ "HOST" "bm25_slot_begin" "void" [ "integer" ] ] 25
 * (the FFI resolver finds the host EXE's exports via the module handle
 * registered by v7s_set_caller_hmodule -- no DLL, no ASM, pure C callbacks).
 *
 * The BM25 Logo script is loaded from an EXTERNAL template file at runtime
 * (examples\sectorlogo\bm25_bench.logo.template or bm25_micro.logo.template).
 * The host finds the %REPS% marker in the file and splices in the rep count
 * from the command line before execution. The script is put into the microdb
 * as "logo:sidecar" by this benchmark, then executed directly via
 * v7s_sectorlogo_interp (standalone mode -- hook variables are NIL,
 * guarded IFELSE branches are no-ops, the BM25 search runs at top level).
 *
 * Full-bench slots (bm25_bench.logo.template, :sofs design):
 *   NEW run (STR-TOKENS tokenize, :sofs = 0):
 *     1  tokenize corpus (per-chunk tokenize loop, all chunks)
 *     2  chunk corpus (EXPLODE paragraph/sentence split)
 *     3  query tokenize + hash map build (tokenize + SORT + dedup walk)
 *     4  tf/df single pass (per-chunk term counting via qhash/LFIND/LSET)
 *     5  BM25 scoring loop (float math per chunk)
 *     6  rank + floor + pick (REVERSE SORT + dual floor + top-3)
 *     25 whole-query end-to-end
 *   OLD run (per-char tokenize_old, :sofs = 29): slots 30-35 + e2e slot 54
 *
 * Micro-bench slots (bm25_micro.logo.template):
 *   9  calibration (empty sbegin/send pair -- FFI round-trip cost)
 *   10 whole per-char REPEAT loop over one chunk
 *   11 SUBSTR char extract, 12 EQUALP end test, 13 FIND alnum classify,
 *   14 _ word concat, 15 stopword FIND, 16 FPUT unigram append,
 *   17 counter update, 18 EXPLODE space split, 19 per-word stopword scan,
 *   20 Lowercase whole string, 22 bigram build loop, 23 bigram build NGrams
 *
 * Usage: bm25_sectorlogo_benchmark.exe <v7_threading.dll> <lz4ht.db>
 *                                   <template.logo> [reps] [micro]
 *   template.logo = path to the .logo.template file (no default, Sec.106)
 *   reps          = rep count spliced over %REPS% (default 5)
 *   micro         = literal "micro" -- use the tokenize micro profile
 *                   (the caller passes bm25_micro.logo.template; this flag
 *                   only switches the REPORT layout)
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define MAX_SLOT 64
#define NANOSEC_PER_SEC 1000000000.0
#define TEMPLATE_MAX_BYTES (1024 * 1024)

/* ---- slot timer state (HOST side, read by the report at the end) ---- */

double slot_elapsed[MAX_SLOT];
double slot_started[MAX_SLOT];
int slot_calls[MAX_SLOT];
LARGE_INTEGER freq;

__declspec(dllexport) void bm25_slot_begin(int slot_id)
{
	LARGE_INTEGER now;

	if ((slot_id < 0) || (slot_id >= MAX_SLOT))
		return;
	QueryPerformanceCounter(&now);
	slot_started[slot_id] = (double)now.QuadPart / (double)freq.QuadPart;
}

__declspec(dllexport) void bm25_slot_end(int slot_id)
{
	LARGE_INTEGER now;
	double t;

	if ((slot_id < 0) || (slot_id >= MAX_SLOT))
		return;
	QueryPerformanceCounter(&now);
	t = (double)now.QuadPart / (double)freq.QuadPart;
	slot_elapsed[slot_id] += t - slot_started[slot_id];
	slot_calls[slot_id]++;
}

/* ------------------------------------------------------------------ */
/* Template loader: read the whole file, splice the rep count over     */
/* every %REPS% marker.                                                */
/* ------------------------------------------------------------------ */

static char *load_template(const char *path, int reps)
{
	FILE *file_ptr = fopen(path, "rb");
	if (!file_ptr)
	{
		printf("Fatal: cannot open template %s\n", path);
		return 0;
	}

	fseek(file_ptr, 0, SEEK_END);
	long file_len = ftell(file_ptr);
	fseek(file_ptr, 0, SEEK_SET);
	if ((file_len < 1) || (file_len > TEMPLATE_MAX_BYTES))
	{
		printf("Fatal: template %s has bad size %ld\n", path, file_len);
		fclose(file_ptr);
		return 0;
	}

	char *text = (char *)malloc((size_t)file_len + 1);
	if (!text)
	{
		printf("Fatal: malloc failed\n");
		fclose(file_ptr);
		return 0;
	}

	if (fread(text, 1, (size_t)file_len, file_ptr) != (size_t)file_len)
	{
		printf("Fatal: short read on %s\n", path);
		free(text);
		fclose(file_ptr);
		return 0;
	}
	fclose(file_ptr);
	text[file_len] = 0;

	/* splice the rep count over every %REPS% marker */
	char rep_text[32];
	sprintf(rep_text, "%d", reps);
	size_t rep_len = strlen(rep_text);
	size_t marker_len = strlen("%REPS%");

	/* count markers first, then build the spliced buffer in one pass */
	size_t marker_count = 0;
	for (char *scan = text; (scan = strstr(scan, "%REPS%")) != 0; scan += marker_len)
		marker_count++;
	if (marker_count == 0)
	{
		printf("Fatal: template %s has no %%REPS%% marker\n", path);
		free(text);
		return 0;
	}

	size_t spliced_len = (size_t)file_len + marker_count * (rep_len - marker_len) + 1;
	char *script = (char *)malloc(spliced_len);
	if (!script)
	{
		printf("Fatal: malloc failed\n");
		free(text);
		return 0;
	}

	char *src_cursor = text;
	char *dst_cursor = script;
	while (*src_cursor)
	{
		if ((src_cursor[0] == '%') && (strncmp(src_cursor, "%REPS%", marker_len) == 0))
		{
			memcpy(dst_cursor, rep_text, rep_len);
			dst_cursor += rep_len;
			src_cursor += marker_len;
		}
		else
		{
			*dst_cursor++ = *src_cursor++;
		}
	}
	*dst_cursor = 0;
	free(text);
	return script;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	if (argc < 4)
	{
		printf("Usage: %s <v7_threading.dll> <lz4ht.db> <template.logo> [reps] [micro]\n", argv[0]);
		printf("       template.logo = bm25_bench.logo.template or bm25_micro.logo.template\n");
		printf("       micro = tokenize-path statement-level profile (slots 9-22)\n");
		return 1;
	}

	char *dll_path = argv[1];
	char *db_path = argv[2];
	char *template_path = argv[3];
	int reps = 5;
	int micro_mode = 0;

	if (argc > 4)
		reps = atoi(argv[4]);
	if ((reps < 1) || (reps > 100000))
	{
		printf("Fatal: reps must be 1..100000 (got %d)\n", reps);
		return 1;
	}
	if ((argc > 5) && (strcmp(argv[5], "micro") == 0))
		micro_mode = 1;

	QueryPerformanceFrequency(&freq);

	struct v7_lib lib;
	if (v7_load(dll_path, &lib) != 0)
	{
		printf("Fatal: cannot load %s\n", dll_path);
		return 1;
	}

	/* register THIS module so the Logo script's C-CALL [ "HOST" ... ] finds
	 * bm25_slot_begin / bm25_slot_end via GetProcAddress on our HMODULE */
	lib.v7s_set_caller_hmodule(GetModuleHandleA(0));

	if (!lib.v7_init(db_path))
	{
		if (lib.v7_engine_is_dead())
			printf("Fatal: V7_init died: %s\n", lib.v7_get_death_reason());
		else
			printf("Fatal: V7_init failed for %s\n", db_path);
		v7_unload(&lib);
		return 1;
	}

	/* load the external template + splice the rep count over %REPS% */
	char *script = load_template(template_path, reps);
	if (!script)
		return 1;

	printf("[bm25-bench] template %s (%u bytes, reps=%d)\n", template_path, (unsigned)strlen(script), reps);

	/* put the script into the DB as the sidecar key (the interp path reads
	 * it from there via lg_init) -- this is the "put into the db" step */
	if (!lib.v7_db_add("logo:sidecar", (void *)script, (int)strlen(script) + 1, 0))
	{
		printf("Fatal: v7_db_add('logo:sidecar') failed\n");
		free(script);
		v7_unload(&lib);
		return 1;
	}
	printf("[bm25-bench] script stored as logo:sidecar in microdb\n");

	for (int slot = 0; slot < MAX_SLOT; slot++)
	{
		slot_elapsed[slot] = 0.0;
		slot_calls[slot] = 0;
	}

	LARGE_INTEGER t0;
	LARGE_INTEGER t1;
	QueryPerformanceCounter(&t0);
	int rc = lib.v7s_sectorlogo_interp(script);
	QueryPerformanceCounter(&t1);
	double total_ms = ((double)t1.QuadPart - (double)t0.QuadPart) * 1000.0 / (double)freq.QuadPart;

	/* ---- report ---- */
	if (micro_mode)
	{
		printf("\n.................. tokenize micro profile (sum over %d reps, 383-char chunk) ..................\n", reps);
		printf("whole interp call               %10.2f ms  (host wall clock)\n", total_ms);
		static const char *micro_names[] = {
			"CALIBRATION empty pair (9)",         /* 9 */
			"per-char loop total (10)",           /* 10 */
			"  Part char extract (11)",           /* 11 */
			"  Equal end test (12)",              /* 12 */
			"  Search alnum classify (13)",       /* 13 */
			"  _ word concat (14)",               /* 14 */
			"  stopword Search (15)",             /* 15 */
			"  Prepend unigram append (16)",      /* 16 */
			"  i=i+1 counter (17)",               /* 17 */
			"Split space split (18)",             /* 18 */
			"per-word stopword scan (19)",        /* 19 */
			"Lowercase whole string (20)",        /* 20 */
			"bigram build loop (22)",             /* 22 */
			"bigram build NGrams (23)",           /* 23 */
		};
		static const int micro_ids[] = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23};
		for (int k = 0; k < 14; k++)
		{
			int id = micro_ids[k];
			printf("slot %-2d %-32s %10.2f ms  (%d calls)\n", id, micro_names[k], slot_elapsed[id] * 1000.0, slot_calls[id]);
		}
	}
	else
	{
		printf("\n.................. bm25 slot profile (sum over %d reps) ..................\n", reps);
		printf("whole interp call               %10.2f ms  (host wall clock)\n", total_ms);

		static const char *names[] = {
			"tokenize corpus (isolated)",   /* 1 / 30 */
			"chunk corpus (EXPLODE)",       /* 2 / 31 */
			"query tokenize + qmap",        /* 3 / 32 */
			"tf/df pass",                   /* 4 / 33 */
			"BM25 scoring loop",            /* 5 / 34 */
			"rank+floor+pick",              /* 6 / 35 */
			"END-TO-END bm25_search",       /* 25 / 54 */
		};
		static const int ids[] = {1, 2, 3, 4, 5, 6, 25};
		static const int old_ids[] = {30, 31, 32, 33, 34, 35, 54};

		printf("\n.... per-phase old vs new (the whole point) ....\n");
		printf("%-28s %12s %12s %8s\n", "phase", "new(ms)", "old(ms)", "speedup");
		for (int k = 0; k < 7; k++)
		{
			double new_ms = slot_elapsed[ids[k]] * 1000.0;
			double old_ms = slot_elapsed[old_ids[k]] * 1000.0;
			double ratio = (new_ms > 0.0000001) ? (old_ms / new_ms) : 0.0;
			printf("%-28s %12.2f %12.2f %7.1fx\n", names[k], new_ms, old_ms, ratio);
		}

		printf("\n.... raw slot dump ....\n");
		for (int k = 0; k < 7; k++)
		{
			printf("slot %-2d %-28s %10.2f ms  (%d calls)\n", ids[k], names[k], slot_elapsed[ids[k]] * 1000.0, slot_calls[ids[k]]);
		}
		for (int k = 0; k < 7; k++)
		{
			printf("slot %-2d %-28s %10.2f ms  (%d calls)\n", old_ids[k], names[k], slot_elapsed[old_ids[k]] * 1000.0, slot_calls[old_ids[k]]);
		}
	}

	printf("interp rc = %d\n", rc);
	/* Sec.172: v7s_sectorlogo_interp is a GATED entry -- a script FATAL
	 * unwinds with the engine parked while rc reads 0 like a success. The
	 * exit code signals it (0 = ran, 1 = script FATAL); the tagged rc is
	 * informational only. */
	if (lib.v7_engine_is_dead())
	{
		printf("Fatal: script FATAL: %s\n", lib.v7_get_death_reason());
		lib.v7_shutdown();
		free(script);
		return 1;
	}
	lib.v7_shutdown();
	free(script);
	return 0;
}
