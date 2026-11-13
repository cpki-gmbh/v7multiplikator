// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * test_ardmark_rag.c -- 3-turn ardmark RAG regression driver
 *
 * Loads the ardmark sidecar (examples/research/ardmark/ardmark_rag.logo)
 * into the microdb, enables the hook + token dump, then runs three chat
 * turns the way the server does: reset=1 on turn 1, reset=0 afterwards.
 * The sidecar's later-turn gate reads "__server_chat__turn" from the DB,
 * exactly like the server writes it, so turn 2 exercises the LATER-TURN
 * handoff + EOS-rescue paths that the server battery reproduces.
 *
 * build:  cl /std:c++17 /EHsc /Zi -Isrc test_ardmark_rag.c /link bin\v7_threading.lib
 * usage:  test_ardmark_rag.exe <v7_threading.dll> <lz4ht.db>
 *
 * All engine traces ([TOK]/[samp]/RW_) print to stdout via the exported
 * enable_token_dump BSS gate. The sidecar logs its own decisions to
 * sectorlogo.log in the working directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#define MAX_OUT 20000
#define SIDEKICK_PATH_ARDMARK "examples/ardmark/ardmark_rag.logo"
#define REFERENCE_PATH "examples/ardmark/reference_transcript.txt"

/* ---- reference transcript helpers (demo-intactness check) ----
 * The driver samples greedy (it never writes the sampler), so the three
 * turns are a deterministic function of (binary, DB, quantize mode). The
 * reference transcript pins that function: any engine, sidecar, or stream-text
 * change that moves the demo's output shows up as a compare failure instead
 * of passing silently. Regenerate with -record after reviewing the diff.
 * The mode line in the file pins the quantize mode the reference was recorded
 * with; comparing an F32 run against a Q8 reference is refused, not failed. */

/* Read the whole reference transcript into a malloc buffer (NUL-terminated).
 * Returns 0 and sets *out_len on success, prints the reason and returns 1
 * on failure. */
int read_whole_reference(char **out_buf, long *out_len)
{
	FILE *reference_file = fopen(REFERENCE_PATH, "rb");
	if (!reference_file)
		return 1;
	fseek(reference_file, 0, SEEK_END);
	long reference_len = ftell(reference_file);
	fseek(reference_file, 0, SEEK_SET);
	char *reference_buf = (char *)malloc((size_t)reference_len + 1);
	if (!reference_buf)
	{
		printf("[FATAL] malloc failed for the reference transcript\n");
		fclose(reference_file);
		exit(1);
	}
	size_t got = fread(reference_buf, 1, (size_t)reference_len, reference_file);
	fclose(reference_file);
	reference_buf[got] = 0;
	*out_buf = reference_buf;
	*out_len = (long)got;
	return 0;
}

/* Extract the "=== TURN n ===" section (up to the next section marker or
 * EOF) into out. Returns the section length, or -1 when the marker or the
 * terminator is missing. */
int reference_extract(char *reference_buf, long reference_len, int turn, char *out, int out_cap)
{
	char marker[64];
	sprintf(marker, "=== TURN %d ===\n", turn);
	long mark_len = (long)strlen(marker);
	long at = -1;
	for (long i = 0; i + mark_len <= reference_len; i++)
	{
		if (memcmp(reference_buf + i, marker, (size_t)mark_len) == 0)
		{
			at = i + mark_len;
			break;
		}
	}
	if (at < 0)
		return -1;
	long end = reference_len;
	char next_marker[64];
	sprintf(next_marker, "\n=== TURN %d ===", turn + 1);
	long next_len = (long)strlen(next_marker);
	for (long i = at; i + next_len <= reference_len; i++)
	{
		if (memcmp(reference_buf + i, next_marker, (size_t)next_len) == 0)
		{
			end = i + 1; /* keep the trailing newline the writer emitted */
			break;
		}
	}
	long sect_len = end - at;
	if (sect_len >= out_cap)
		sect_len = out_cap - 1;
	memcpy(out, reference_buf + at, (size_t)sect_len);
	out[sect_len] = 0;
	return (int)sect_len;
}

/* Append "=== TURN n ===\n<output>\n" to the transcript buffer. */
void transcript_append(char *buf, int buf_cap, int *at, int turn, const char *output)
{
	int left = buf_cap - *at;
	if (left <= 0)
		return;
	int need = sprintf(buf + *at, "=== TURN %d ===\n%s\n", turn, output);
	if (need < 0)
		return;
	if (need > left)
		*at = buf_cap;
	else
		*at += need;
}

 /* Compare this turn's output against the reference section. Returns 0 = match,
 * 1 = mismatch (both texts printed from the first divergence). */
int reference_check_turn(int turn, const char *output, char *reference_buf, long reference_len)
{
	char expected[MAX_OUT + 4];
	if (reference_extract(reference_buf, reference_len, turn, expected, (int)sizeof(expected)) < 0)
	{
		printf("[FAIL] reference transcript has no === TURN %d === section\n", turn);
		return 1;
	}
	char actual[MAX_OUT + 4];
	sprintf(actual, "%s\n", output);
	if (strcmp(actual, expected) == 0)
	{
		printf("[test] reference transcript: TURN %d identical\n", turn);
		return 0;
	}
	printf("[FAIL] reference transcript: TURN %d differs\n", turn);
	unsigned __int64 common = 0;
	while (actual[common] && expected[common] && actual[common] == expected[common])
		common++;
 printf("       first divergence at byte %llu (actual %llu bytes, reference %llu bytes)\n",
	       (unsigned long long)common, (unsigned long long)strlen(actual),
	       (unsigned long long)strlen(expected));
	printf("       actual  from there: [%.200s]\n", actual + common);
 printf("       reference  from there: [%.200s]\n", expected + common);
	return 1;
}

/* run one chat turn on the given prompt; returns 0 on success, 1 if the
 * engine died (reason printed). reset=1 for the first turn, 0 afterwards. */
int run_turn_check(int argc_fake, struct v7_lib *dll, const char *prompt, char *output,
                   int out_len, int reset)
{
	memset(output, 0, (size_t)out_len);
	dll->v7_chat(prompt, 0, output, out_len, reset);
	if (dll->v7_engine_is_dead())
	{
		printf("[FATAL] engine died on prompt '%s': %s\n", prompt, dll->v7_get_death_reason());
		return 1;
	}
	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Usage: %s <v7_threading.dll> <lz4ht.db> [q8] [tokdump] [-record]\n", argv[0]);
 printf("       -record (re)writes examples/ardmark/reference_transcript.txt;\n");
		printf("       the default compares the run against it (demo-intactness).\n");
		return 1;
	}
	const char *dll_path = argv[1];
	const char *db_path = argv[2];
	int q8 = (argc > 3) ? atoi(argv[3]) : 1; /* default Q8, pass 0 for F32 */
	int tokdump = (argc > 4) ? atoi(argv[4]) : 0; /* pass 1 to enable engine traces */

	/* load DLL */
	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("[FATAL] v7_load failed: %s\n", dll_path);
		return 1;
	}

	/* engine traces ([TOK]/[samp]/RW_) are opt-in via argv[4]=1; default runs
	 * stay readable -- the sidecar's own sectorlogo.log carries the decisions */
	if (tokdump)
	{
		*dll.enable_token_dump = 1;
		printf("[test] token dump + sampler duel + rewrite traces enabled\n");
	}

	/* init DB */
	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] V7_init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] V7_init failed: %s\n", db_path);
		v7_unload(&dll);
		return 1;
	}

	/* load the ardmark sidecar BEFORE inference create (lg_init reads the
	 * logo:sidecar key lazily on the first hook call) */
	FILE *sidecar_file = fopen(SIDEKICK_PATH_ARDMARK, "rb");
	if (!sidecar_file)
	{
		printf("[FATAL] cannot open sidecar '%s'\n", SIDEKICK_PATH_ARDMARK);
		v7_unload(&dll);
		return 1;
	}
	fseek(sidecar_file, 0, SEEK_END);
	long sidecar_len = ftell(sidecar_file);
	fseek(sidecar_file, 0, SEEK_SET);
	char *sidecar_buf = (char *)malloc((size_t)sidecar_len + 1);
	if (!sidecar_buf)
	{
		printf("[FATAL] malloc failed for sidecar buffer\n");
		fclose(sidecar_file);
		v7_unload(&dll);
		return 1;
	}
	int sidecar_read = (int)fread(sidecar_buf, 1, (int)sidecar_len, sidecar_file);
	fclose(sidecar_file);
	sidecar_buf[sidecar_read] = 0; /* NUL-terminate for the logo reader */
	if (!dll.v7_db_add("logo:sidecar", sidecar_buf, sidecar_read + 1, 0))
	{
		printf("[FATAL] v7_db_add('logo:sidecar') failed\n");
		free(sidecar_buf);
		v7_unload(&dll);
		return 1;
	}
	printf("[test] sidecar loaded: %s (%d bytes)\n", SIDEKICK_PATH_ARDMARK, sidecar_read);
	free(sidecar_buf);

	/* The tuning journal is NEVER deleted (Sec.209 rule 9): it is the
	 * cumulative tuning memory. If it exists it is applied at the first
	 * chat's end and may shift dispatch chunking -- the golden transcript is
	 * recorded WITH the journal in place, and the lifetime verdict counters
	 * printed at destroy prove the tuner ran either way. */

	/* the hook is DATA-DRIVEN: the sidecar is in the DB (ht_add above),
	 * so it self-arms on the first chat */

	/* reference transcript mode: -record (re)writes the reference, the default
	 * compares against it. A missing reference is a FAILURE, never a pass --
	 * a check that silently skips when its reference is gone proves nothing. */
	int record_mode = 0;
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-record") == 0)
			record_mode = 1;
	}
	char *reference_buf = 0;
	long reference_len = 0;
	if (!record_mode)
	{
		if (read_whole_reference(&reference_buf, &reference_len))
		{
			printf("[FAIL] reference transcript missing at %s\n", REFERENCE_PATH);
			printf("       create it with: bin\\test_ardmark_rag.exe bin\\v7_threading.dll weights\\databases\\qwen3_0.6b\\lz4ht.db 1 -record\n");
			printf("       then eyeball all three turns before committing it\n");
			v7_unload(&dll);
			return 1;
		}
		char expected_mode[32];
		sprintf(expected_mode, "mode: %s\n", q8 ? "Q8" : "F32");
		if (strncmp(reference_buf, expected_mode, strlen(expected_mode)) != 0)
		{
			printf("[FAIL] reference transcript was recorded in a different quantize mode\n");
 printf("       reference starts with: [%.24s]\n", reference_buf);
			printf("       this run is:        [%s]", expected_mode);
			free(reference_buf);
			v7_unload(&dll);
			return 1;
		}
	}

	/* the run's own transcript: recorded verbatim in -record mode, and the
	 * per-turn compare in check mode works off the same sections */
	char transcript[3 * (MAX_OUT + 64)];
	int transcript_at = 0;

	/* initialize inference (context hidden in BSS) */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("[FATAL] inference init died: %s\n", dll.v7_get_death_reason());
		else
			printf("[FATAL] v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}
	dll.v7_set_quantize(q8);
	dll.v7_set_max_gen(10000);
	printf("[test] quantize=%d (%s), max_gen=10000\n", q8, q8 ? "Q8" : "F32");

	/* thinking mode ON (the template carries the empty think block) */
	dll.v7_enable_thinking(1);
	printf("[test] thinking enabled\n");

	char output[MAX_OUT];
	char turnbuf[16];
	const char *turn_prompts[3];
	turn_prompts[0] = "are there cities in ardmark?";
	turn_prompts[1] = "list some of these cities!";
	turn_prompts[2] = "who are you?";

	int failed = 0;

	/* === TURN 1: fresh session (reset=1), turn marker "0" === */
	printf("\n... TURN 1: \"%s\" (reset=1) ...\n", turn_prompts[0]);
	sprintf(turnbuf, "%d", 0);
	if (!dll.v7_db_add("__server_chat__turn", turnbuf, (int)strlen(turnbuf) + 1, 0))
	{
		printf("[FATAL] v7_db_add('__server_chat__turn') failed\n");
		failed = 1;
	}
	else
	{
		if (run_turn_check(0, &dll, turn_prompts[0], output, (int)sizeof(output), 1))
		{
			failed = 1;
		}
		else
		{
			printf("[TURN1] pos_after=%d output (%u chars):\n", dll.v7_get_position(),
			       (unsigned)strlen(output));
			printf("%s\n\n", output);
			transcript_append(transcript, (int)sizeof(transcript), &transcript_at, 1, output);
			if (!record_mode && reference_check_turn(1, output, reference_buf, reference_len))
				failed = 1;
		}
	}

	/* === TURN 2: continue session (reset=0), turn marker "1" === */
	if (!failed)
	{
		printf("... TURN 2: \"%s\" (reset=0) ...\n", turn_prompts[1]);
		dll.v7_set_max_gen(10000);
		sprintf(turnbuf, "%d", 1);
		if (!dll.v7_db_add("__server_chat__turn", turnbuf, (int)strlen(turnbuf) + 1, 0))
		{
			printf("[FATAL] v7_db_add('__server_chat__turn') failed\n");
			failed = 1;
		}
		else
		{
			if (run_turn_check(0, &dll, turn_prompts[1], output, (int)sizeof(output), 0))
			{
				failed = 1;
			}
			else
			{
				printf("[TURN2] pos_after=%d output (%u chars):\n", dll.v7_get_position(),
				       (unsigned)strlen(output));
				printf("%s\n\n", output);
				transcript_append(transcript, (int)sizeof(transcript), &transcript_at, 2, output);
				if (!record_mode && reference_check_turn(2, output, reference_buf, reference_len))
					failed = 1;
			}
		}
	}

	/* === TURN 3: continue session (reset=0), turn marker "2" === */
	if (!failed)
	{
		printf("... TURN 3: \"%s\" (reset=0) ...\n", turn_prompts[2]);
		dll.v7_set_max_gen(10000);
		sprintf(turnbuf, "%d", 2);
		if (!dll.v7_db_add("__server_chat__turn", turnbuf, (int)strlen(turnbuf) + 1, 0))
		{
			printf("[FATAL] v7_db_add('__server_chat__turn') failed\n");
			failed = 1;
		}
		else
		{
			if (run_turn_check(0, &dll, turn_prompts[2], output, (int)sizeof(output), 0))
			{
				failed = 1;
			}
			else
			{
				printf("[TURN3] pos_after=%d output (%u chars):\n", dll.v7_get_position(),
				       (unsigned)strlen(output));
				printf("%s\n\n", output);
				transcript_append(transcript, (int)sizeof(transcript), &transcript_at, 3, output);
				if (!record_mode && reference_check_turn(3, output, reference_buf, reference_len))
					failed = 1;
			}
		}
	}

	/* reference decision: record mode writes the reference only when every turn
	 * ran (a partial demo must never become the reference); check mode reports
	 * the aggregate line the gate greps for. */
	if (record_mode)
	{
		if (failed)
		{
			printf("[test] a turn failed -- reference transcript NOT written\n");
		}
		else
		{
			FILE *reference_file = fopen(REFERENCE_PATH, "wb");
			if (!reference_file)
			{
				printf("[FATAL] cannot write %s\n", REFERENCE_PATH);
				v7_unload(&dll);
				return 1;
			}
			fprintf(reference_file, "mode: %s\n", q8 ? "Q8" : "F32");
			fprintf(reference_file, "; reference transcript for test_ardmark_rag -- the demo-intactness reference.\n");
			fprintf(reference_file, "; Greedy sampling, deterministic per (binary, DB, mode). Regenerate with\n");
			fprintf(reference_file, "; bin\\test_ardmark_rag.exe bin\\v7_threading.dll weights\\databases\\qwen3_0.6b\\lz4ht.db 1 -record\n");
			fwrite(transcript, 1, (size_t)transcript_at, reference_file);
			fclose(reference_file);
			printf("[test] reference transcript written to %s (%d bytes) -- eyeball all three turns before committing\n",
			       REFERENCE_PATH, transcript_at);
		}
	}
	else
	{
		free(reference_buf);
		if (!failed)
			printf("[test] reference transcript: all 3 turns identical\n");
	}

	/* cleanup */
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	if (failed)
	{
		printf("[test] RESULT: FAILED\n");
		return 1;
	}
	printf("[test] RESULT: OK\n");
	return 0;
}
