// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * v_encode_validate.cpp -- Standalone BPE tokenizer encode correctness validator
 *
 * Tests the BPE encode function using a synthetic vocab with known merge rules.
 * Needs ht_init/ht_add/ht_find/encode exports from the DLL.
 * Usage: v_encode_validate.exe <dll_path>
 */
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <windows.h> /* Linux tool adapters: unsigned __int64 */
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NUM_VOCAB_TOKENS 25

typedef struct encode_ref_tokenizer
{
	char **vocab;
	float *merge_scores;
	int vocab_size;
	unsigned int max_token_length;
	unsigned int bos;
	unsigned int eos;
	char pt[1024];
	char spt[1024];
	int is_sentencepiece;

} encode_ref_tokenizer;

/* Reference BPE encode */
static void encode_ref(void *tokenizer_ptr, char *text, int *tokens, int *n_tokens, int max_tokens, ht_find_fn ht_find_fn_ptr)
{
	encode_ref_tokenizer *tok = (encode_ref_tokenizer *)tokenizer_ptr;
	if (max_tokens < 1)
	{
		*n_tokens = 0;
		return;
	}
	int str_buf_size = tok->max_token_length * 2 + 8;
	char stack_buf[4096];
	char *str_buffer;
	if (str_buf_size <= 4096)
		str_buffer = stack_buf;
	else
		str_buffer = (char *)malloc(str_buf_size);
	char *dyn = 0;
	if (str_buffer != stack_buf)
		dyn = str_buffer;
	char special[65];
	char one_byte[2];
	*n_tokens = 0;
	int str_len = 0;
	/* SentencePiece dummy prefix (reference checks text != "" -- the empty-text
	   no-prefix behavior under test here) */
	if (tok->is_sentencepiece && text[0] != 0)
	{
		unsigned __int64 len;
		void *data = ht_find_fn_ptr("\xE2\x96\x81", &len);
		if (((data) && (len == (int)sizeof(int))) && (*n_tokens < max_tokens))
			tokens[(*n_tokens)++] = *(int *)data;
	}
	int sp = tok->is_sentencepiece;
	for (char *c = text; *c; c++)
	{
		int id = -1;
		int found_special = 0;
		/* UTF-8 class check (reference run.c:539-543): a leading byte or ASCII
		   starts a new codepoint and resets the cursor; a continuation byte
		   (0b10xxxxxx) keeps accumulating */
		if (((*c & 0xC0) != 0x80))
			str_len = 0;
		if ((*c == 0x20) && sp)
		{
			/* SentencePiece space rewrite: the space becomes U+2581 as a COMPLETE
			   3-byte codepoint (str_len = 3, no accumulation, no peek) */
			str_buffer[0] = (char)0xE2;
			str_buffer[1] = (char)0x96;
			str_buffer[2] = (char)0x81;
			str_buffer[3] = 0;
			str_len = 3;
		}
		else
		{
			str_buffer[str_len] = *c;
			str_len++;
			str_buffer[str_len] = 0;
		}
		if (*c == '<')
		{
			int end = -1;
			for (int k = 0; ((c[k]) && (k < 64)); k++)
			{
				if (c[k] == '>')
				{
					end = k;
					break;
				}
			}
			if (end != -1)
			{
				strncpy(special, c, end + 1);
				special[end + 1] = 0;
				unsigned __int64 len;
				void *data = ht_find_fn_ptr(special, &len);
				if (((data) && (len == (int)sizeof(int))))
				{
					id = *(int *)data;
					c += end;
					found_special = 1;
					str_len = 0;
				}
			}
		}
		if (!found_special)
		{
			/* codepoint-completeness peek (reference run.c:551-555): while the
			   NEXT input byte is a continuation byte and the cursor is below the
			   4-byte cap, keep accumulating without a lookup */
			if (!(((c[1] & 0xC0) == 0x80) && (str_len < 4)))
			{
				unsigned __int64 len;
				void *data = ht_find_fn_ptr(str_buffer, &len);
				id = -1;
				if (((data) && (len == (int)sizeof(int))))
					id = *(int *)data;
				if (id == -1)
				{
					/* byte fallback (reference run.c byte-fallback branch): encode
					   each accumulated byte through its own key */
					for (int i = 0; i < str_len; i++)
					{
						one_byte[0] = str_buffer[i];
						one_byte[1] = 0;
						data = ht_find_fn_ptr(one_byte, &len);
						if (((data) && (len == (int)sizeof(int))))
						{
							int byte_id = *(int *)data;
							if ((byte_id != -1) && (*n_tokens < max_tokens))
								tokens[(*n_tokens)++] = byte_id;
						}
					}
					str_len = 0;
					continue;
				}
			}
		}
		if (((id != -1) && (*n_tokens < max_tokens)))
		{
			tokens[(*n_tokens)++] = id;
			str_len = 0;
		}
	}
	while (1)
	{
		float best_score = -1e10f;
		int best_id = -1;
		int best_idx = -1;
		for (int i = 0; i < (*n_tokens - 1); i++)
		{
			sprintf(str_buffer, "%s%s", tok->vocab[tokens[i]], tok->vocab[tokens[i + 1]]);
			unsigned __int64 len;
			void *data = ht_find_fn_ptr(str_buffer, &len);
			if (((data) && (len == (int)sizeof(int))))
			{
				int id = *(int *)data;
				if (((id != -1) && (tok->merge_scores[id] > best_score)))
				{
					best_score = tok->merge_scores[id];
					best_id = id;
					best_idx = i;
				}
			}
		}
		if (best_idx == -1)
			break;
		tokens[best_idx] = best_id;
		for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
			tokens[i] = tokens[i + 1];
		(*n_tokens)--;
	}
	if (dyn)
		free(dyn);
}

static int compare_token_arrays(int *left_tokens, int left_count, int *right_tokens, int right_count)
{
	if (left_count != right_count)
		return 0;
	for (int i = 0; i < left_count; i++)
		if (left_tokens[i] != right_tokens[i])
			return 0;
	return 1;
}

typedef struct encode_test
{
	const char *text;
	int max_tokens;
	int expected[16];
	int expected_n;
	const char *desc;

} encode_test;

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		printf("usage: v_encode_validate.exe <dll_path>\n");
		return 1;
	}
	ZydisInit();
	printf("[bisect] ZydisInit done\n");
	fflush(stdout);

	v7_lib lib;
	printf("[bisect] calling v7_load\n");
	fflush(stdout);
	if ((v7_load(argv[1], &lib) != 0))
	{
		printf("v7_load failed\n");
		return 1;
	}
	printf("[bisect] v7_load done\n");
	fflush(stdout);
	/* V7_init deliberately skipped -- test creates a synthetic vocab via ht_init/ht_add */
	printf("... encode_validate: %s ...\n", argv[1]);
	fflush(stdout);

	/* Create synthetic vocab (20 tokens) */
	char *vocab[NUM_VOCAB_TOKENS];
	float merge_scores[NUM_VOCAB_TOKENS];
	for (int i = 0; i < NUM_VOCAB_TOKENS; i++)
	{
		vocab[i] = (char *)malloc(32);
		merge_scores[i] = -1e10f;
	}

	/* Every index must hold a DISTINCT string. Uninitialized/duplicate keys
	   (e.g. several empty strings) are invalid hash-table input: ht_add treats
	   two identical keys as the same entry (an update), collapsing them and
	   corrupting later lookups. Give unused indices unique single chars. */
	sprintf(vocab[0], "a");
	sprintf(vocab[1], "b");
	sprintf(vocab[2], "c");
	sprintf(vocab[3], "d");
	sprintf(vocab[4], "e");
	sprintf(vocab[5], "f");
	sprintf(vocab[6], "g");
	sprintf(vocab[7], "h");
	sprintf(vocab[8], "i");
	sprintf(vocab[9], "j");

	sprintf(vocab[10], "ab");
	merge_scores[10] = 0.5f;
	sprintf(vocab[11], "bc");
	merge_scores[11] = 0.3f;
	sprintf(vocab[12], "abc");
	merge_scores[12] = 0.9f;
	sprintf(vocab[13], "cd");
	merge_scores[13] = 0.4f;
	sprintf(vocab[14], "ef");
	merge_scores[14] = 0.6f;
	sprintf(vocab[15], "def");
	merge_scores[15] = 0.8f;
	sprintf(vocab[16], "<test>");
	sprintf(vocab[17], "q");
	sprintf(vocab[18], "r");
	sprintf(vocab[19], "s");

	/* UTF-8 test keys: a 2-byte codepoint (a-umlaut), its raw lead/continuation
	   bytes as fallback aliases, a 3-byte codepoint (U+2581, the SP prefix
	   shape), and a 4-byte codepoint (emoji). Byte aliases let the
	   byte-fallback branch tests miss the codepoint on purpose. The UTF-8 keys
	   are memcpy'd, NOT sprintf'd: raw non-ASCII bytes in a sprintf FORMAT
	   string send the CRT's format parser through its multibyte path (system
	   codepage dependent) -- observed crashing with a garbage write cursor. */
	memcpy(vocab[20], "\xC3\xA4", 3);
	memcpy(vocab[21], "\xC3", 2);
	memcpy(vocab[22], "\xA9", 2);
	memcpy(vocab[23], "\xE2\x96\x81", 4);
	memcpy(vocab[24], "\xF0\x9F\x98\x80", 5);

	/* Build tokenizer struct (same layout the DLL's encode expects) */
	encode_ref_tokenizer tok;
	tok.vocab = vocab;
	tok.merge_scores = merge_scores;
	tok.vocab_size = NUM_VOCAB_TOKENS;
	tok.max_token_length = 8;
	tok.bos = 0;
	tok.eos = 0;
	/* is_sentencepiece: struct padding is intentionally zeroed so the DLL's
	   TOK_IS_SP read (offset 2080, inf_structs.inc) sees a defined 0. The
	   synthetic vocab here is not SentencePiece -- no U+2581 prefix lookup. */
	tok.is_sentencepiece = 0;
	memset(tok.pt, 0, 1024);
	memset(tok.spt, 0, 1024);

	/* Populate MicroDB -- destroy DB-loaded data first, then init fresh */
	printf("[bisect] ht_destroy\n");
	fflush(stdout);
	lib.ht_destroy();
	printf("[bisect] ht_init\n");
	fflush(stdout);
	lib.ht_init();
	for (int i = 0; i < NUM_VOCAB_TOKENS; i++)
	{
		printf("  ht_add '%s' (%d bytes)\n", vocab[i], (int)strlen(vocab[i]));
		fflush(stdout);
		int add_ok = lib.ht_add(vocab[i], &i, sizeof(int), 0);
		if (!add_ok)
		{
			printf("  warning: ht_add returned 0 for key='%s'\n", vocab[i]);
			fflush(stdout);
		}
	}
	printf("  setup: %d keys added\n", NUM_VOCAB_TOKENS);
	fflush(stdout);

	/* Test cases (max_tokens >= 1 always -- the ASM encode contract is >= 1;
	   max_tokens=0 is a contract violation that FATALs by design, not a valid
	   inference scenario, so it is not tested here) */
	encode_test tests[] = {
		{ "abc", 256, { 12 }, 1, "encode('abc') -> [12] (abc merge)" },
		{ "abcd", 256, { 0 }, 0, "encode('abcd') -> DLL matches ref" },
		{ "abcdef", 256, { 12, 15 }, 2, "encode('abcdef') -> [12,15] (abc + def)" },
		{ "<test>", 256, { 16 }, 1, "encode('<test>') -> [16] (special token)" },
		{ "a<test>b", 256, { 0, 16, 1 }, 3, "encode('a<test>b') -> [0,16,1]" },
		{ "ab<test>cd", 256, { 10, 16, 13 }, 3, "encode('ab<test>cd') -> [10,16,13]" },
		{ "abcdef", 6, { 12, 15 }, 2, "encode('abcdef', max=6) -> [12,15] (exact Phase-1 fit: 6 singles stored, merges only shrink the count)" },
		{ "a\xC3\xA4" "b", 256, { 0, 20, 1 }, 3, "encode('a\\xC3\\xA4b') -> [0,20,1] (2-byte codepoint assembled and looked up as one token)" },
		{ "\xC3\xA4", 256, { 20 }, 1, "encode('\\xC3\\xA4') -> [20] (2-byte codepoint at text start)" },
		{ "c\xC3\xA9" "d", 256, { 2, 21, 22, 3 }, 4, "encode('c\\xC3\\xA9d') -> [2,21,22,3] (codepoint miss falls back to raw-byte tokens)" },
		{ "a\xE2\x96\x81" "b", 256, { 0, 23, 1 }, 3, "encode('a\\xE2\\x96\\x81b') -> [0,23,1] (3-byte codepoint assembled)" },
		{ "a\xF0\x9F\x98\x80" "b", 256, { 0, 24, 1 }, 3, "encode('a\\xF0\\x9F\\x98\\x80b') -> [0,24,1] (4-byte codepoint assembled at the cursor cap)" },
	};
	int n_tests = (int)(sizeof(tests) / sizeof(tests[0]));

	int pass = 0;
	int total = 0;
	for (int ti = 0; ti < n_tests; ti++)
	{
		encode_test *t = &tests[ti];
		int dll_tokens[256] = {};
		int dll_token_count = 0;
		int ref_tokens[256] = {};
		int ref_token_count = 0;

		/* Run DLL encode */
		printf("  [dll about to encode: %s]\n", t->desc);
		fflush(stdout);
		lib.encode((void *)&tok, (char *)t->text, dll_tokens, &dll_token_count, t->max_tokens);
		printf("  [dll done: %s]\n", t->desc);
		fflush(stdout);

		/* Run reference encode */
		printf("  [ref about to encode: %s]\n", t->desc);
		fflush(stdout);
		encode_ref((void *)&tok, (char *)t->text, ref_tokens, &ref_token_count, t->max_tokens, lib.ht_find);
		printf("  [ref done: %s]\n", t->desc);
		fflush(stdout);

		/* Compare */
		int ok;
		if (t->expected_n > 0)
		{
			ok = compare_token_arrays(dll_tokens, dll_token_count, (int *)t->expected, t->expected_n);
		}
		else
		{
			ok = compare_token_arrays(dll_tokens, dll_token_count, ref_tokens, ref_token_count);
		}
		if (ok)
		{
			++pass;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("\n  %s: Fail", t->desc);
			printf("\n    expected: ");
			for (int i = 0; i < t->expected_n; i++)
				printf("%d ", t->expected[i]);
			printf("\n    DLL:      ");
			for (int i = 0; i < dll_token_count; i++)
				printf("%d ", dll_tokens[i]);
			printf("\n    ref:      ");
			for (int i = 0; i < ref_token_count; i++)
				printf("%d ", ref_tokens[i]);
			printf("\n");
		}
		++total;
	}

	/* ---- SentencePiece group: flip the SP flag and re-run the border cases.
	   The U+2581 key (vocab[23]) must resolve for the prefix emit; empty text
	   must produce ZERO tokens (no dummy prefix, reference checks text != ""). */
	printf("[bisect] SP group\n");
	fflush(stdout);
	tok.is_sentencepiece = 1;
	encode_test sp_tests[] = {
		{ "", 256, { 0 }, 0, "encode('', SP) -> [] (empty text: no dummy prefix)" },
		{ "ab", 256, { 0 }, 0, "encode('ab', SP) -> DLL matches ref (prefix + singles)" },
	};
	for (int ti = 0; ti < (int)(sizeof(sp_tests) / sizeof(sp_tests[0])); ti++)
	{
		encode_test *t = &sp_tests[ti];
		int dll_tokens[256] = {};
		int dll_token_count = 0;
		int ref_tokens[256] = {};
		int ref_token_count = 0;

		lib.encode((void *)&tok, (char *)t->text, dll_tokens, &dll_token_count, t->max_tokens);
		encode_ref((void *)&tok, (char *)t->text, ref_tokens, &ref_token_count, t->max_tokens, lib.ht_find);

		int ok;
		if (t->expected_n > 0)
			ok = compare_token_arrays(dll_tokens, dll_token_count, (int *)t->expected, t->expected_n);
		else
			ok = compare_token_arrays(dll_tokens, dll_token_count, ref_tokens, ref_token_count);
		/* the empty-text case is additionally pinned to a zero count: both sides
		   computing the same wrong nonzero count would otherwise pass together */
		if (t->expected_n == 0 && t->text[0] == 0 && dll_token_count != 0)
			ok = 0;
		if (ok)
		{
			++pass;
			printf(".");
			fflush(stdout);
		}
		else
			printf("\n  %s: Fail\n", t->desc);
		++total;
	}
	tok.is_sentencepiece = 0;

	/* Cleanup */
	lib.ht_destroy();
	for (int i = 0; i < NUM_VOCAB_TOKENS; i++)
		free(vocab[i]);

	printf("\n%d/%d passed\n", pass, total);
	v7_unload(&lib);
	if (pass == total)
		return 0;
	return 1;
}
