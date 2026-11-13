// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v7_pipe.c — A swiss-army-knife text processor using v7_threading.dll
 *
 * modes:
 *   summarize   summarize text from stdin
 *   explain     explain text from stdin in simple terms
 *   ask         ask a question (single-turn, no template)
 *   chat        chat with template (multi-turn)
 *   qa          answer a question about a document from stdin
 *
 * usage:
 *   echo "long text..." | v7_pipe summarize
 *   cat README.md | v7_pipe explain
 *   v7_pipe ask "What is the capital of france?"
 *   v7_pipe chat "Tell me about cats"
 *   v7_pipe qa "Who built the tower?" < doc.txt
 *
 * build:
 *   cl /nologo /O2 /arch:AVX2 /MD /std:c17 /D_CRT_SECURE_NO_WARNINGS v7_pipe.c /fe"v7_pipe.exe" /I"src"
 *
 * license: GPL v3
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OUTPUT_SIZE 32768
#define INPUT_SIZE 65536

void print_usage(char *prog)
{
	printf("v7_pipe — swiss-army-knife text processor using v7_threading.dll\n\n"
	       "Usage:\n"
	       "  echo \"text\" | %s summarize    summarize piped text\n"
	       "  cat file.md | %s explain        explain piped text\n"
	       "  %s ask \"question\"              ask a question (single-turn)\n"
	       "  %s chat \"message\"              chat (multi-turn, templated)\n"
	       "  %s qa \"question\" < doc.txt     answer question about document\n\n"
	       "Environment:\n"
	       "  V7_DLL   path to v7_threading.dll (required)\n"
	       "  V7_DB    path to lz4ht.db (required)\n",
	       prog, prog, prog, prog, prog);
}

char *read_stdin(int *out_len)
{
	int cap = 4096;
	int len = 0;
	char *buf = (char *)malloc(cap);
	if (!buf)
		return 0;

	int ch;
	while ((ch = fgetc(stdin)) != EOF)
	{
		if (len + 1 >= cap)
		{
			cap *= 2;
			char *tmp = (char *)realloc(buf, cap);
			if (!tmp)
			{
				free(buf);
				return 0;
			}
			buf = tmp;
		}
		buf[len++] = (char)ch;
	}
	buf[len] = '\0';
	*out_len = len;
	return buf;
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		print_usage(argv[0]);
		return 1;
	}

	char *mode = argv[1];
	char *dll_path = getenv("V7_DLL");
	if (!dll_path || !*dll_path)
	{
		printf("Error: V7_DLL environment variable must be set (path to v7_threading.dll)\n");
		print_usage(argv[0]);
		return 1;
	}
	char *db_path = getenv("V7_DB");
	if (!db_path || !*db_path)
	{
		printf("Error: V7_DB environment variable must be set (path to lz4ht.db)\n");
		print_usage(argv[0]);
		return 1;
	}

	/* --- LLM modes: load DLL + DB + create context --- */
	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("Error: failed to load %s\n", dll_path);
		return 1;
	}

	if (dll.v7_init(db_path) == 0)
	{
		if (dll.v7_engine_is_dead())
			printf("Error: V7_init(\"%s\") died: %s\n", db_path, dll.v7_get_death_reason());
		else
			printf("Error: V7_init(\"%s\") failed — file not found or corrupt\n", db_path);
		v7_unload(&dll);
		return 1;
	}

	/* Sec.172: a gated entry -- a caught FATAL also returns 0, so the real return
	 * AND the death flag must both be checked. */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("Error: inference init died: %s\n", dll.v7_get_death_reason());
		else
			printf("Error: v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}

	char output[OUTPUT_SIZE];
	int rc = 0;

/* run one gated call and stop on a caught FATAL: the engine is parked and
 * every further call would silently answer empty */
#define RUN_AND_CHECK(call)         \
	do                              \
	{                               \
		call;                       \
		if (dll.v7_engine_is_dead())\
		{                           \
			printf("Error: engine died: %s\n", dll.v7_get_death_reason()); \
			rc = 1;                 \
			goto done;              \
		}                           \
	} while (0)

	if (strcmp(mode, "summarize") == 0)
	{
		int in_len = 0;
		char *input = read_stdin(&in_len);
		if (!input || in_len == 0)
		{
			printf("Error: no input on stdin (pipe text to summarize)\n");
			rc = 1;
		}
		else
		{
			char prompt[INPUT_SIZE];
			snprintf(prompt, sizeof(prompt), "Summarize the following text in 3-5 sentences:\n\n%s", input);
			memset(output, 0, sizeof(output));
			RUN_AND_CHECK(dll.v7_chat(prompt, 0, output, sizeof(output), 1));
			printf("%s\n", output);
		}
		free(input);
	}
	else if (strcmp(mode, "explain") == 0)
	{
		int in_len = 0;
		char *input = read_stdin(&in_len);
		if (!input || in_len == 0)
		{
			printf("Error: no input on stdin (pipe text to explain)\n");
			rc = 1;
		}
		else
		{
			char prompt[INPUT_SIZE];
			snprintf(prompt, sizeof(prompt), "Explain the following text in simple terms:\n\n%s", input);
			memset(output, 0, sizeof(output));
			RUN_AND_CHECK(dll.v7_chat(prompt, 0, output, sizeof(output), 1));
			printf("%s\n", output);
		}
		free(input);
	}
	else if (strcmp(mode, "ask") == 0)
	{
		if (argc < 3)
		{
			printf("Error: ask mode requires a question argument\n");
			print_usage(argv[0]);
			rc = 1;
		}
		else
		{
			memset(output, 0, sizeof(output));
			RUN_AND_CHECK(dll.v7_generate(argv[2], output, sizeof(output), 0));
			printf("%s\n", output);
		}
	}
	else if (strcmp(mode, "chat") == 0)
	{
		if (argc < 3)
		{
			printf("Error: chat mode requires a message argument\n");
			print_usage(argv[0]);
			rc = 1;
		}
		else
		{
			memset(output, 0, sizeof(output));
			RUN_AND_CHECK(dll.v7_chat(argv[2], 0, output, sizeof(output), 1));
			printf("%s\n", output);
		}
	}
	else if (strcmp(mode, "qa") == 0)
	{
		if (argc < 3)
		{
			printf("Error: qa mode requires a question argument\n");
			print_usage(argv[0]);
			rc = 1;
		}
		else
		{
			int in_len = 0;
			char *input = read_stdin(&in_len);
			if (!input || in_len == 0)
			{
				printf("Error: no document on stdin (pipe a document)\n");
				rc = 1;
			}
			else
			{
				char prompt[INPUT_SIZE];
				snprintf(prompt, sizeof(prompt),
				         "Answer the question based on the following document.\n\n"
				         "Document:\n%s\n\nQuestion: %s",
				         input, argv[2]);
				memset(output, 0, sizeof(output));
				RUN_AND_CHECK(dll.v7_chat(prompt, 0, output, sizeof(output), 1));
				printf("%s\n", output);
			}
			free(input);
		}
	}
	else
	{
		printf("Error: unknown mode '%s'\n\n", mode);
		print_usage(argv[0]);
		rc = 1;
	}

done:
	/* v7_shutdown joins the workers BEFORE the heap teardown + FreeLibrary
	 * (the wrong order leaves workers executing inside the DLL while its
	 * pages are unmapped -- AGENTS.md 105). */
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	return rc;
}
