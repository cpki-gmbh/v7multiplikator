// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v_generate_validate.c -- v7s_generate end-to-end correctness validator
 *
 * Exercises the GENERATE path (merged KV block via mdb_alloc_mem with the
 * file-backed OOM fallback, template-free single-turn session) end-to-end:
 * create context, run v7s_generate on the canonical paris prompt, check the
 * output contains "paris". Runs BOTH quantization modes (F32 and Q8) in one
 * invocation -- each mode must answer with paris (Sec.131 both-paths rule).
 *
 * This validator is the direct detector for generate-path memory regressions:
 * a broken allocator FATAL-crashes (or previously returned NULL) before the
 * first token; verify.bat's chat-path tests never touch v7s_generate.
 *
 * Usage: v_generate_validate.exe <dll_path> <db_path>
 *   dll_path -- path to v7_threading.dll (REQUIRED, no default per Sec.106)
 *   db_path  -- path to lz4ht.db (REQUIRED, no default)
 * Exit code: 0 = both modes passed, 1 = any mode failed.
 */

#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define OUTPUT_BUF_SIZE 32768

/* run one generate round in the given quantization mode; returns 1 when the
 * output contains "paris", 0 otherwise. the prompt is the canonical one the
 * chat-path inference tests use, so F32/Q8 parity is directly comparable. */
int run_generate_mode(struct v7_lib *dll_ptr, const char *db_path, int q8_mode, const char *prompt)
{
	printf("db_load: file = %s\n", db_path);
	if (!dll_ptr->v7_init(db_path))
	{
		printf("[FATAL] v7_init failed for '%s'\n", db_path);
		return 0;
	}

	dll_ptr->v7_initialize_inference();
	dll_ptr->v7_set_quantize(q8_mode);
	dll_ptr->v7_set_max_gen(200);
	printf("generate mode: %s\n", q8_mode ? "Q8" : "F32");

	char output[OUTPUT_BUF_SIZE];
	output[0] = 0;

	dll_ptr->v7_generate(prompt, output, OUTPUT_BUF_SIZE, 0);

	printf("%s\n", output);

	/* case-insensitive containment check -- the model capitalizes freely */
	char lower[OUTPUT_BUF_SIZE];
	int idx = 0;
	for (char *src = output; ((src) && (*src) && (idx + 1 < OUTPUT_BUF_SIZE)); src++)
	{
		char c = *src;
		if ((c >= 'A') && (c <= 'Z'))
			c = (char)(c - 'A' + 'a');
		lower[idx++] = c;
	}
	lower[idx] = 0;

	int found = strstr(lower, "paris") != 0;
	if (found)
	{
		printf(".");
		fflush(stdout);
	}
	else
	{
		printf("  FAIL: v7_generate %s - output does NOT contain paris\n", q8_mode ? "Q8" : "F32");
	}

	dll_ptr->v7_shutdown();
	dll_ptr->v7_free_inference();
	return found;
}

int main(int argc, char **argv)
{
	ZydisInit();
	setvbuf(stdout, 0, _IONBF, 0);

	if (argc < 3)
	{
		printf("Usage: %s <dll_path> <db_path>\n", argv[0]);
		printf("  dll_path is REQUIRED -- path to v7_threading.dll (no default)\n");
		printf("  db_path  is REQUIRED -- path to lz4ht.db (no default)\n");
		return 1;
	}

	struct v7_lib dll;
	if (v7_load(argv[1], &dll) != 0)
	{
		printf("Failed to load '%s'\n", argv[1]);
		return 1;
	}

	int pass_count = 0;
	int fail_count = 0;

	if (run_generate_mode(&dll, argv[2], 0, "What is the capital of france?"))
		pass_count++;
	else
		fail_count++;

	if (run_generate_mode(&dll, argv[2], 1, "What is the capital of france?"))
		pass_count++;
	else
		fail_count++;

	v7_unload(&dll);

	printf("\ngenerate_validate: %d passed, %d failed\n", pass_count, fail_count);
	if (fail_count == 0)
	{
		printf(" PASS: v_generate_validate\n");
		return 0;
	}
	printf(" FAIL: v_generate_validate\n");
	return 1;
}
