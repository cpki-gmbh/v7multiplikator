// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_sprintf_validate.cpp -- v7sprintf bounded formatted-output validator
 *
 * Tests v7sprintf (ASM implementation in sprintf.inc) against expected
 * outputs for all supported format specifiers: %d, %s, %c, %f, %%
 * and unknown specifiers.
 *
 * Usage: v_sprintf_validate.exe <dll_path>
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdarg>

int passing_tests_counter = 0;
int failing_tests_counter = 0;
v7_lib *inference_dll_ptr = 0;

void compare_string_result(const char *test_label, const char *reference_result, char *actual_value)
{
	if (strcmp(reference_result, actual_value) == 0)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		failing_tests_counter++;
		printf("  FAIL: %s -> expected \"%s\", got \"%s\"\n", test_label, reference_result, actual_value);
	}
}

void test_fmt(const char *desc, const char *expected, const char *fmt, ...)
{
	char buf[256];
	va_list ap;
	va_start(ap, fmt);
	va_end(ap);
	buf[0] = '\0';
	(void)desc;
	(void)expected;
	(void)fmt;
}

void test_int_format(const char *test_label, int value, const char *reference_result)
{
	char buf[256];
	int ret_len = inference_dll_ptr->v7sprintf(buf, (int)sizeof(buf), "%d", value);
	compare_string_result(test_label, reference_result, buf);
	if (ret_len != (int)strlen(reference_result))
	{
		failing_tests_counter++;
		printf("  FAIL: %s return value: expected %d, got %d\n", test_label, (int)strlen(reference_result), ret_len);
	}
}

void test_string_format(const char *test_label, const char *value, const char *reference_result)
{
	char buf[256];
	int ret_len = inference_dll_ptr->v7sprintf(buf, (int)sizeof(buf), "%s", value);
	compare_string_result(test_label, reference_result, buf);
	if (ret_len != (int)strlen(reference_result))
	{
		failing_tests_counter++;
		printf("  FAIL: %s return value: expected %d, got %d\n", test_label, (int)strlen(reference_result), ret_len);
	}
}

void test_char_format(const char *test_label, char value, const char *reference_result)
{
	char buf[256];
	int ret_len = inference_dll_ptr->v7sprintf(buf, (int)sizeof(buf), "%c", (int)value);
	compare_string_result(test_label, reference_result, buf);
	if (ret_len != (int)strlen(reference_result))
	{
		failing_tests_counter++;
		printf("  FAIL: %s return value: expected %d, got %d\n", test_label, (int)strlen(reference_result), ret_len);
	}
}

void test_float_format(const char *test_label, float value, const char *reference_result)
{
	char buf[256];
	unsigned int fbits;
	memcpy(&fbits, &value, 4);
	int ret_len = inference_dll_ptr->v7sprintf(buf, (int)sizeof(buf), "%f", (int)fbits);
	compare_string_result(test_label, reference_result, buf);
	if (ret_len != (int)strlen(reference_result))
	{
		failing_tests_counter++;
		printf("  FAIL: %s return value: expected %d, got %d\n", test_label, (int)strlen(reference_result), ret_len);
	}
}

void test_percent_format(const char *test_label, const char *fmt, const char *reference_result)
{
	char buf[256];
	int ret_len = inference_dll_ptr->v7sprintf(buf, (int)sizeof(buf), fmt);
	compare_string_result(test_label, reference_result, buf);
	if (ret_len != (int)strlen(reference_result))
	{
		failing_tests_counter++;
		printf("  FAIL: %s return value: expected %d, got %d\n", test_label, (int)strlen(reference_result), ret_len);
	}
}

void test_combo_format(const char *test_label, const char *reference_result, const char *fmt, int dval, char *sval)
{
	char buf[256];
	int ret_len = inference_dll_ptr->v7sprintf(buf, (int)sizeof(buf), fmt, dval, sval);
	compare_string_result(test_label, reference_result, buf);
	(void)ret_len;
}

/* Six %d conversions push 5 varargs past r9d into the STACK slots, where Win64
   leaves the upper half of each 8-byte slot undefined for a 32-bit argument.
   The old qword vararg fetch fed that residue into the 64-bit reciprocal
   multiply for positive values -- garbage digits with no crash. The narrowing
   32-bit fetch (SPR_FETCH_NEXT_ARG32) discards the residue. */
void test_decimal_many_varargs()
{
	char buf[128];
	int ret_len = inference_dll_ptr->v7sprintf(buf, (int)sizeof(buf), "%d %d %d %d %d %d", 1, -22, 333, -4444, 55555, -6);
	compare_string_result("decimal many varargs", "1 -22 333 -4444 55555 -6", buf);
	if (ret_len != (int)strlen("1 -22 333 -4444 55555 -6"))
	{
		failing_tests_counter++;
		printf("  FAIL: decimal many varargs return value: expected %d, got %d\n", (int)strlen("1 -22 333 -4444 55555 -6"), ret_len);
	}
}

void test_buffer_size(const char *test_label, int buf_size, const char *fmt, const char *reference_result, int dval)
{
	char *buf = (char *)malloc(buf_size + 1);
	memset(buf, 0xCC, buf_size + 1);
	int ret_len = inference_dll_ptr->v7sprintf(buf, buf_size, fmt, dval);
	if ((ret_len >= 0) && (ret_len <= buf_size - 1) && (buf[ret_len] == '\0') && (strcmp(buf, reference_result) == 0))
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		failing_tests_counter++;
		printf("  FAIL: %s (buf_size=%d) -> expected \"%s\", got \"%s\" (ret=%d)\n", test_label, buf_size, reference_result, buf, ret_len);
	}
	free(buf);
}

int main(int argc, char **argv)
{
	ZydisInit();

	if (argc < 2)
	{
		printf("Usage: %s <dll_path>\n", argv[0]);
		return 1;
	}

	struct v7_lib lib;
	if (v7_load(argv[1], &lib) != 0)
	{
		printf("Fatal: cannot load %s\n", argv[1]);
		return 1;
	}

	inference_dll_ptr = &lib;

	printf("... v7sprintf validator ...\n\n");

	/* ---- %d tests ---- */
	printf("--- %%d tests ---\n");
	test_int_format("zero", 0, "0");
	test_int_format("positive single", 5, "5");
	test_int_format("positive small", 42, "42");
	test_int_format("positive large", 12345, "12345");
	test_int_format("negative single", -1, "-1");
	test_int_format("negative small", -99, "-99");
	test_int_format("negative large", -12345, "-12345");
	test_int_format("INT_MIN", -2147483647 - 1, "-2147483648");
	test_int_format("INT_MAX", 2147483647, "2147483647");

	/* ---- %d with many stack-passed varargs (Sec.170 regression) ---- */
	printf("\n--- %%d many varargs ---\n");
	test_decimal_many_varargs();

	/* ---- %s tests ---- */
	printf("\n--- %%s tests ---\n");
	test_string_format("simple string", "hello", "hello");
	test_string_format("empty string", "", "");
	test_string_format("single char", "x", "x");
	test_string_format("long string", "abcdefghij", "abcdefghij");
	/* ("0 pointer -> (null)" test deleted -- NULL %%s is a contract violation and FATALs per Sec.57d; the "(null)" fabrication was removed) */

	/* ---- %c tests ---- */
	printf("\n--- %%c tests ---\n");
	test_char_format("letter A", 'A', "A");
	test_char_format("digit 0", '0', "0");
	test_char_format("space", ' ', " ");
	test_char_format("newline", '\n', "\n");

	/* ---- %f tests ---- */
	printf("\n--- %%f tests ---\n");
	test_float_format("zero", 0.0f, "0");
	test_float_format("negative zero", -0.0f, "0");
	test_float_format("0.5", 0.5f, "0.5");
	test_float_format("3.5", 3.5f, "3.5");
	test_float_format("3.0 (no dot)", 3.0f, "3");
	test_float_format("-0.25", -0.25f, "-0.25");
	test_float_format("1.0", 1.0f, "1");
	test_float_format("10.0", 10.0f, "10");
	test_float_format("100.0", 100.0f, "100");
	test_float_format("0.1", 0.1f, "0.1");
	test_float_format("0.123456", 0.123456f, "0.123456");
	test_float_format("0.000001", 0.000001f, "0.000001");
	test_float_format("-1.5", -1.5f, "-1.5");
	test_float_format("12.34", 12.34f, "12.34");
	test_float_format("99.999", 99.999f, "99.999001");
	test_float_format("0.123", 0.123f, "0.123");
	test_float_format("large integer", 12345.0f, "12345");
	test_float_format("large negative", -9999.0f, "-9999");

	/* ---- %f edge cases: rounding carry ---- */
	printf("\n--- %%f rounding carry ---\n");
	test_float_format("rounding carry to 1.0", 0.9999995f, "1");

	/* ---- %f special values ---- */
	printf("\n--- %%f special values ---\n");
	test_float_format("NaN", NAN, "nan");
	test_float_format("positive Inf", INFINITY, "inf");
	test_float_format("negative Inf", -INFINITY, "-inf");

	/* ---- %% tests ---- */
	printf("\n--- %% tests ---\n");
	test_percent_format("single %%", "%%", "%");
	test_percent_format("%% at end", "100%%", "100%");

	/* ---- unknown specifier ---- */
	printf("\n--- unknown specifier ---\n");
	test_percent_format("unknown %x", "%x", "%x");
	test_percent_format("unknown %q", "%q", "%q");

	/* ---- combined format ---- */
	printf("\n--- combined format ---\n");
	{
		char buf[256];
		int ret_len = inference_dll_ptr->v7sprintf(buf, (int)sizeof(buf), "hello %d world %s", 42, "test");
		compare_string_result("combined %d+%s", "hello 42 world test", buf);
		(void)ret_len;
	}

	/* ---- plain text (no specifiers) ---- */
	printf("\n--- plain text ---\n");
	test_percent_format("no specifiers", "plain text", "plain text");
	test_percent_format("empty format", "", "");

	/* ---- buffer bounds ---- */
	printf("\n--- buffer bounds ---\n");
	test_buffer_size("buf_size=1 (nothing fits)", 1, "%d", "", 42);
	test_buffer_size("buf_size=2 (1 char)", 2, "%d", "4", 42);
	test_buffer_size("buf_size=3 (2 chars)", 3, "%d", "42", 42);
	test_buffer_size("buf_size=5 (truncated)", 5, "%d", "1234", 12345);
	test_buffer_size("exact fit", 6, "%d", "12345", 12345);

	/* ---- dangling dot trim on truncation ---- */
	printf("\n--- dangling dot trim on truncation ---\n");
	{
		char buf[5];
		unsigned int fbits;
		float fv = 12.3f;
		memcpy(&fbits, &fv, 4);
		int ret_len = inference_dll_ptr->v7sprintf(buf, 4, "%f", (int)fbits);
		if ((buf[ret_len] == '\0') && (ret_len >= 0))
		{
			int has_dangling_dot = (ret_len > 0) && (buf[ret_len - 1] == '.');
			if ((!has_dangling_dot) && (ret_len <= 3))
			{
				passing_tests_counter++;
				printf(".");
				fflush(stdout);
			}
			else
			{
				failing_tests_counter++;
				printf("  FAIL: dangling dot trim (buf_size=4) -> \"%s\" (ret=%d) dangling_dot=%d\n", buf, ret_len, has_dangling_dot);
			}
		}
		else
		{
			failing_tests_counter++;
			printf("  FAIL: dangling dot trim (buf_size=4) -> not null-terminated or negative ret ret_len=%d\n", ret_len);
		}
	}

	/* ---- prior specifier's '.' must survive a zero-byte %f truncation ---- */
	/* Repro of the BUG-1 class: "hi.%f" with buf_size=4 -- the literal dot is
	   the last writable char, %f writes zero bytes, and the old guard (which
	   compared against the whole-call buffer start) ate the dot and returned
	   "hi" instead of "hi.". The r10 per-specifier snapshot fixed it. */
	printf("\n--- prior dot survives zero-byte %%f ---\n");
	{
		char buf[8];
		unsigned int fbits;
		float fv = 1.5f;
		memcpy(&fbits, &fv, 4);
		int ret_len = inference_dll_ptr->v7sprintf(buf, 4, "hi.%f", (int)fbits);
		if ((ret_len == 3) && (strcmp(buf, "hi.") == 0))
		{
			passing_tests_counter++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			failing_tests_counter++;
			printf("  FAIL: literal dot kept -> \"%s\" (ret=%d)\n", buf, ret_len);
		}
		ret_len = inference_dll_ptr->v7sprintf(buf, 4, "%s%f", "hi.", (int)fbits);
		if ((ret_len == 3) && (strcmp(buf, "hi.") == 0))
		{
			passing_tests_counter++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			failing_tests_counter++;
			printf("  FAIL: %%s dot kept (buf_size=4) -> \"%s\" (ret=%d)\n", buf, ret_len);
		}
		/* buf_size=5: one byte of %f fits -- "hi.1" is the correct truncation of "hi.1.5" */
		ret_len = inference_dll_ptr->v7sprintf(buf, 5, "%s%f", "hi.", (int)fbits);
		if ((ret_len == 4) && (strcmp(buf, "hi.1") == 0))
		{
			passing_tests_counter++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			failing_tests_counter++;
			printf("  FAIL: %%s dot kept, 1 frac byte (buf_size=5) -> \"%s\" (ret=%d)\n", buf, ret_len);
		}
	}

	/* ---- null termination ---- */
	printf("\n--- null termination ---\n");
	{
		char buf[10];
		memset(buf, 'X', sizeof(buf));
		inference_dll_ptr->v7sprintf(buf, (int)sizeof(buf), "%d", 7);
		int ok = (buf[1] == '\0');
		if (ok)
		{
			passing_tests_counter++;
		}
		else
		{
			failing_tests_counter++;
		}
		printf("  %s: null terminator after single-digit output\n", ok ? "Pass" : "Fail");
	}

	/* ---- summary ---- */
	printf("\n... v7sprintf summary: %d passed, %d failed ...\n", passing_tests_counter, failing_tests_counter);

	v7_unload(&lib);
	if (failing_tests_counter)
		return 1;
	else
		return 0;
}
