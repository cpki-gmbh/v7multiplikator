// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

// v_win32_quote_validate.exe -- validate the command-line arg escaping used by
// v7_execute / v7_execute_fmt (.append_quoted_arg in src/asm/execute.inc).
//
// The DLL exports no quoting helper (the escaper is an internal append helper),
// so this validator proves the RULES two ways:
//   1. reference escaper  -- the canonical backslash-style algorithm (a run of
//      N backslashes terminated by '"' emits 2N+1 backslashes + the quote;
//      a run terminated by end-of-arg emits 2N; an interior run emits N).
//   2. asm mirror escaper -- the exact mixed style .append_quoted_arg emits
//      (lone quotes doubled as "" per MSVCRT pair-of-quotes rule; backslash
//      runs handled identically to the reference).
// Both must round-trip through the MSVCRT argv tokenizer:
//   crt_parse(escape(arg)) == arg   for every possible arg.
//
// MSVCRT rules (learn.microsoft.com "Parsing C++ command-line arguments"):
//   - whitespace splits arguments outside quotes
//   - within a quoted string, a PAIR of quotes ("") is one escaped literal quote
//   - even run of backslashes + " : half are literal, the quote TOGGLES quoting
//   - odd run of backslashes + "  : (N-1)/2 literal backslashes + a literal "
//   - unclosed quote: rest of string belongs to the argument
//
// Usage: v_win32_quote_validate.exe [dll_path] [--db db_path]
//   (dll_path/--db accepted but unused -- the escaper is DLL-internal; args
//    exist so verify.bat can call it with the same uniform command line.)

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "zydis_debug_header.h"

// ---------------------------------------------------------------------------
// crt_parse_one: MSVCRT argv tokenizer for ONE argument. src holds a single
// escaped argument (leading space allowed, ignored; parsing stops at the
// first unquoted whitespace or end of string). out receives the decoded
// bytes. Returns decoded length, or -1 on overflow.
// ---------------------------------------------------------------------------
static int crt_parse_one(const char* src, char* out, int out_max)
{
	int out_pos = 0;
	int slashes = 0;
	int quoted = 0;
	int done = 0;
	int i = 0;

	while (!done)
	{
		char c = src[i];

		if (c == 0)
		{
			// end of string: pending slashes are literal, unclosed quote is fine
			while (slashes > 0)
			{
				if (out_pos >= out_max) return -1;
				out[out_pos++] = '\\';
				slashes--;
			}
			done = 1;
		}
		else if (c == '\\')
		{
			slashes++;
			i++;
		}
		else if (c == '"')
		{
			if ((quoted != 0) && (src[i + 1] == '"') && ((slashes % 2) == 0))
			{
				// doubled quote inside a quoted string: one escaped literal "
				while (slashes > 0)
				{
					if (out_pos >= out_max) return -1;
					out[out_pos++] = '\\';
					slashes--;
				}
				if (out_pos >= out_max) return -1;
				out[out_pos++] = '"';
				i += 2;
			}
			else if ((slashes % 2) == 0)
			{
				// even run: half the backslashes are literal, quote TOGGLES quoting
				int emit = slashes / 2;
				while (emit > 0)
				{
					if (out_pos >= out_max) return -1;
					out[out_pos++] = '\\';
					emit--;
				}
				quoted = !quoted;
				slashes = 0;
				i++;
			}
			else
			{
				// odd run: (N-1)/2 literal backslashes + a literal "
				int emit = (slashes - 1) / 2;
				while (emit > 0)
				{
					if (out_pos >= out_max) return -1;
					out[out_pos++] = '\\';
					emit--;
				}
				if (out_pos >= out_max) return -1;
				out[out_pos++] = '"';
				slashes = 0;
				i++;
			}
		}
		else
		{
			// normal character: pending slashes are literal
			while (slashes > 0)
			{
				if (out_pos >= out_max) return -1;
				out[out_pos++] = '\\';
				slashes--;
			}
			if ((quoted == 0) && ((c == ' ') || (c == '\t')))
			{
				// unquoted whitespace ENDS the argument
				done = 1;
			}
			else
			{
				if (out_pos >= out_max) return -1;
				out[out_pos++] = c;
				i++;
			}
		}
	}

	if (out_pos >= out_max) return -1;
	out[out_pos] = 0;
	return out_pos;
}

// ---------------------------------------------------------------------------
// win32_escape_arg: reference escaper (canonical backslash style). Returns a
// malloc'd string (caller frees), 0 on allocation failure.
// ---------------------------------------------------------------------------
static char* win32_escape_arg(const char* arg)
{
	int len = (int)strlen(arg);
	int extra = 2;              // opening + closing quote
	int backslashes = 0;
	int i;

	// pass 1: compute the exact required size
	for (i = 0; i < len; i++)
	{
		if (arg[i] == '\\')
		{
			backslashes++;
		}
		else
		{
			if (arg[i] == '"')
				extra += backslashes + 1;      // each pending backslash doubles, plus the escape
			backslashes = 0;
		}
	}
	extra += backslashes;       // trailing run doubles before the closing quote

	char* out = (char*)malloc(len + extra + 1);
	if (!out)
		return 0;

	char* p = out;
	*p++ = '"';

	backslashes = 0;
	for (i = 0; i < len; i++)
	{
		char c = arg[i];

		if (c == '\\')
		{
			backslashes++;
			continue;
		}

		if (c == '"')
		{
			// every pending '\' doubles, one extra '\', then the escaped quote
			int j;
			for (j = 0; j < (backslashes * 2) + 1; j++)
				*p++ = '\\';
			*p++ = '"';
		}
		else
		{
			// normal character: emit pending backslashes unchanged
			while (backslashes > 0)
			{
				*p++ = '\\';
				backslashes--;
			}
			*p++ = c;
		}
		backslashes = 0;
	}

	// trailing run doubles because it precedes the closing quote
	while (backslashes > 0)
	{
		*p++ = '\\';
		*p++ = '\\';
		backslashes--;
	}

	*p++ = '"';
	*p = 0;
	return out;
}

// ---------------------------------------------------------------------------
// asm_escape_arg: exact mirror of .append_quoted_arg (src/asm/execute.inc).
// Mixed style: lone quotes are DOUBLED as "" (MSVCRT pair-of-quotes rule),
// a backslash run terminated by '"' emits 2N+1 backslashes + the quote,
// a run terminated by NUL (arg end) emits 2N backslashes, an interior run
// emits N verbatim. Returns malloc'd string, 0 on allocation failure.
// ---------------------------------------------------------------------------
static char* asm_escape_arg(const char* arg)
{
	int len = (int)strlen(arg);
	char* out = (char*)malloc((len * 4) + 6);
	int src_pos = 0;
	int dst_pos = 0;

	if (!out)
		return 0;

	out[dst_pos++] = '"';

	while (src_pos < len)
	{
		char c = arg[src_pos];

		if (c == '"')
		{
			// lone quote (no backslash run pending): doubled ""
			out[dst_pos++] = '"';
			out[dst_pos++] = '"';
			src_pos++;
			continue;
		}

		if (c != '\\')
		{
			out[dst_pos++] = c;
			src_pos++;
			continue;
		}

		// backslash: count the run, classify by its terminator
		int run = 1;
		while ((src_pos + run) < len && arg[src_pos + run] == '\\')
			run++;

		char term = ((src_pos + run) < len) ? arg[src_pos + run] : 0;
		src_pos += run;

		if (term == '"')
		{
			// quote-terminated run: 2N+1 backslashes, then the quote (consumed)
			int emit;
			for (emit = 0; emit < (run * 2) + 1; emit++)
				out[dst_pos++] = '\\';
			out[dst_pos++] = '"';
			src_pos++;
		}
		else if (term == 0)
		{
			// arg-end run: 2N backslashes (closing quote follows)
			int emit;
			for (emit = 0; emit < run * 2; emit++)
				out[dst_pos++] = '\\';
		}
		else
		{
			// interior run: N backslashes verbatim
			int emit;
			for (emit = 0; emit < run; emit++)
				out[dst_pos++] = '\\';
		}
	}

	out[dst_pos++] = '"';
	out[dst_pos] = 0;
	return out;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------
typedef struct quote_test_case
{
	const char* input;
	const char* expected_ref;      // expected win32_escape_arg output, 0 = round-trip only
	const char* expected_asm;      // expected asm_escape_arg output, 0 = round-trip only
} quote_test_case;

static const quote_test_case test_cases[] =
{
	{ "", "\"\"", "\"\"" },
	{ "plain", "\"plain\"", "\"plain\"" },
	{ "with space", "\"with space\"", "\"with space\"" },
	{ "C:\\Foo\\", "\"C:\\Foo\\\\\"", "\"C:\\Foo\\\\\"" },
	{ "foo\\\\", "\"foo\\\\\\\\\"", "\"foo\\\\\\\\\"" },
	{ "say \"hi\"", "\"say \\\"hi\\\"\"", "\"say \"\"hi\"\"\"" },
	{ "a\\\"b", "\"a\\\\\\\"b\"", "\"a\\\\\\\"b\"" },
	{ "a\\b", "\"a\\b\"", "\"a\\b\"" },
	{ "\\", "\"\\\\\"", "\"\\\\\"" },
	{ "\\\\", "\"\\\\\\\\\"", "\"\\\\\\\\\"" },
	{ "tab\there", "\"tab\there\"", "\"tab\there\"" },
	{ "\"", "\"\\\"\"", "\"\"\"\"" },
	{ "a\\\\\"b", 0, 0 },                    // 2 bs + quote: round-trip only (5 bs + ")
	{ "a\\\\\\\"b", 0, 0 },                  // 3 bs + quote: round-trip only (7 bs + ")
	{ "a\\\\\\\\\"b", 0, 0 },                // 4 bs + quote: round-trip only (9 bs + ")
	{ "x\"y\\\\\"", 0, 0 },                  // quote, chars, 2 bs, quote: round-trip only
	{ "trailing bs then space \\\\ ", 0, 0 }, // interior run before space: round-trip only
};

static const int num_test_cases = (int)(sizeof(test_cases) / sizeof(test_cases[0]));

// round-trip one (escaper, input) pair; returns 0 on success, 1 on failure
static int check_round_trip(const char* style, const char* input, char* escaped, char* scratch, int scratch_size)
{
	int parsed_len = crt_parse_one(escaped, scratch, scratch_size);

	if (parsed_len < 0)
	{
		printf("  %s round-trip: CRT parse failed: FAIL\n", style);
		return 1;
	}
	if (strcmp(scratch, input) != 0)
	{
		printf("  %s round-trip got '%s' != in '%s': FAIL\n", style, scratch, input);
		return 1;
	}
	return 0;
}

int main(int argc, char** argv)
{
	int passing_tests_counter = 0;
	int failing_tests_counter = 0;
	int case_index;
	char parsed[8192];

	ZydisInit();

	// dll_path / --db args accepted but unused (escaper is DLL-internal; the
	// uniform verify.bat command line must not break this tool)
	(void)argc;
	(void)argv;

	printf("win32 arg-quoting validator\n");
	printf("reference + asm-mirror escapers vs MSVCRT argv tokenizer\n");
	printf("\n..................\n");

	for (case_index = 0; case_index < num_test_cases; case_index++)
	{
		const quote_test_case* tc = &test_cases[case_index];
		char* ref_out = win32_escape_arg(tc->input);
		char* asm_out = asm_escape_arg(tc->input);
		int failed = 0;

		if (!ref_out || !asm_out)
		{
			printf("case %d: allocation failed: FAIL\n", case_index);
			failing_tests_counter++;
			if (ref_out) free(ref_out);
			if (asm_out) free(asm_out);
			continue;
		}

		printf("case %d: in='%s'\n", case_index, tc->input);
		printf("  ref: %s\n", ref_out);
		printf("  asm: %s\n", asm_out);

		if ((tc->expected_ref != 0) && (strcmp(ref_out, tc->expected_ref) != 0))
		{
			printf("  ref expected '%s' != got '%s': FAIL\n", tc->expected_ref, ref_out);
			failed = 1;
		}
		if ((tc->expected_asm != 0) && (strcmp(asm_out, tc->expected_asm) != 0))
		{
			printf("  asm expected '%s' != got '%s': FAIL\n", tc->expected_asm, asm_out);
			failed = 1;
		}

		failed += check_round_trip("ref", tc->input, ref_out, parsed, (int)sizeof(parsed));
		failed += check_round_trip("asm", tc->input, asm_out, parsed, (int)sizeof(parsed));

		if (failed == 0)
		{
			printf(".");
			fflush(stdout);
			passing_tests_counter++;
		}
		else
		{
			failing_tests_counter++;
		}
		free(ref_out);
		free(asm_out);
	}

	// exhaustive sweep: every string over the alphabet {'a', '\', '"'} up to
	// length 8, round-tripped through BOTH escapers (9841 inputs x 2 styles).
	// This covers every backslash/quote interaction shape that can exist.
	{
		char input[16];
		int len;
		long total = 0;
		long bad = 0;

		printf("\n..................\n");
		printf("exhaustive sweep: alphabet {a, backslash, quote} lengths 0..8, both styles\n");

		for (len = 0; len <= 8; len++)
		{
			long count = 1;
			long index;
			int i;

			for (i = 0; i < len; i++) count *= 3;

			for (index = 0; index < count; index++)
			{
				long v = index;
				char* ref_out;
				char* asm_out;

				for (i = len - 1; i >= 0; i--)
				{
					static const char alphabet[3] = { 'a', '\\', '"' };
					input[i] = alphabet[v % 3];
					v /= 3;
				}
				input[len] = 0;

				ref_out = win32_escape_arg(input);
				asm_out = asm_escape_arg(input);
				if (!ref_out || !asm_out)
				{
					printf("sweep alloc failed at len %d\n", len);
					bad++;
					if (ref_out) free(ref_out);
					if (asm_out) free(asm_out);
					continue;
				}

				if ((crt_parse_one(ref_out, parsed, (int)sizeof(parsed)) < 0) || (strcmp(parsed, input) != 0))
				{
					printf("sweep ref mismatch at '%s'\n", input);
					bad++;
				}
				if ((crt_parse_one(asm_out, parsed, (int)sizeof(parsed)) < 0) || (strcmp(parsed, input) != 0))
				{
					printf("sweep asm mismatch at '%s'\n", input);
					bad++;
				}
				free(ref_out);
				free(asm_out);
				total++;
			}
		}

		printf("sweep total: %ld inputs x2 styles, %ld failures\n", total, bad);
		if (bad == 0)
		{
			printf("sweep: pass\n");
			passing_tests_counter++;
		}
		else
		{
			printf("sweep: FAIL\n");
			failing_tests_counter++;
		}
	}

	// MS docs ground truth: argv[1] decoding of raw command-line fragments
	// straight from the "Parsing C++ command-line arguments" table.
	{
		typedef struct docs_case
		{
			const char* cmdline_fragment;
			const char* expected_argv1;
		} docs_case;

		static const docs_case docs_cases[] =
		{
			{ "\"a b c\" d e", "a b c" },
			{ "\"ab\\\"c\" \"\\\\\" d", "ab\"c" },
			{ "a\\\\\\b d\"e f\"g h", "a\\\\\\b" },          // 3 bs before 'b': all literal
			{ "a\\\\\\\"b c d", "a\\\"b" },                  // 3 bs + ": odd run -> 1 bs + literal "
			{ "a\\\\\\\\\"b c\" d e", "a\\\\b c" },          // 4 bs + ": even run -> 2 bs, toggle on
			{ "a\"b\"\" c d", "ab\" c d" },                  // doubled "" inside quotes -> literal "
		};
		int docs_count = (int)(sizeof(docs_cases) / sizeof(docs_cases[0]));
		int d;
		int docs_bad = 0;

		printf("\n..................\n");
		printf("MS docs table ground truth\n");

		for (d = 0; d < docs_count; d++)
		{
			int parsed_len = crt_parse_one(docs_cases[d].cmdline_fragment, parsed, (int)sizeof(parsed));

			if ((parsed_len < 0) || (strcmp(parsed, docs_cases[d].expected_argv1) != 0))
			{
				printf("docs case %d: got '%s' != expected '%s': FAIL\n", d, (parsed_len >= 0) ? parsed : "(parse fail)", docs_cases[d].expected_argv1);
				docs_bad++;
			}
		}

		if (docs_bad == 0)
		{
			printf("docs cases: pass\n");
			passing_tests_counter++;
		}
		else
		{
			printf("docs cases: FAIL (%d)\n", docs_bad);
			failing_tests_counter++;
		}
	}

	printf("\n..................\n");
	printf("passed: %d, failed: %d\n", passing_tests_counter, failing_tests_counter);
	if (failing_tests_counter == 0)
	{
		printf("Test result: all passed\n");
		return 0;
	}
	printf("Test result: failures detected\n");
	return 1;
}
