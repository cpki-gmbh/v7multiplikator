// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_sectorlogo_validate.cpp -- SectorLogo64 interpreter comprehensive validator
 *
 * Tests v7s_sectorlogo_interp() end-to-end, covering all language features.
 * Takes <dll_path> as first positional arg.
 * Uses Zydis for crash disassembly (via zydis_debug_header.h).
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <float.h> /* _control87: x87 control-word leak check */
#include <nmmintrin.h> /* _mm_crc32_u8: same CRC32C instruction as sl_hash_float */
#include <windows.h> /* QueryPerformanceCounter: per-suite timing log */

/* Platform-shaped paths inside generated SectorLogo scripts: the validator
 * spawns the echo_args helper and names the FFI test module and the engine
 * module. Linux lays both under bin-linux with ELF names, and dlopen only
 * resolves relative paths that carry a directory component (Windows
 * LoadLibraryA also searches the working directory). */
#ifdef __linux__
#define SLFFI_TEST_MODULE "\"bin-linux/slffi_test.so\""
#define SLFFI_GC_M1_STEM "bin-linux/slffi_test"
#define SLFFI_TEST_EXT ".so"
#define SLFFI_ECHO_ARGS_ARG "\"bin-linux/echo_args\""
#define SLFFI_GC_M2_STEM "bin-linux/libv7"
#define SLFFI_GC_M2_EXT "_threading.so"
#else
#define SLFFI_TEST_MODULE "\"slffi_test.dll\""
#define SLFFI_GC_M1_STEM "slffi_test"
#define SLFFI_TEST_EXT ".dll"
#define SLFFI_ECHO_ARGS_ARG "\"bin\\\\echo_args.exe\""
#define SLFFI_GC_M2_STEM "bin\\\\v7"
#define SLFFI_GC_M2_EXT "_threading.dll"
#endif

#define SL_CONS_TAG 0x80000000
#define SL_NUMBER_TAG 0x40000000
#define SL_FLOAT_TAG 0xC0000000
#define SL_TAG_MASK 0xC0000000
#define SL_PAYLOAD_MASK 0x3FFFFFFF

int num_val(int tagged_value)
{
	int payload = tagged_value & SL_PAYLOAD_MASK;
	if (payload & 0x20000000)
		return payload | 0xC0000000;
	return payload;
}
int is_num(int tagged_value)
{
	return (tagged_value & SL_TAG_MASK) == SL_NUMBER_TAG;
}
int is_float(int tagged_value)
{
	return (tagged_value & SL_TAG_MASK) == SL_FLOAT_TAG;
}
int is_nil(int tagged_value)
{
	return tagged_value == 0;
}

/* Unpack a SectorLogo FLOAT tagged value to IEEE binary32.
   The 30-bit payload (top 30 bits of a binary32, low 2 bits zero)
   is shifted left by 2 to reruct the original float bits. */
float float_val(int tagged_value)
{
	int payload = tagged_value & SL_PAYLOAD_MASK;
	int bits = payload << 2;
	float result;
	memcpy(&result, &bits, 4);
	return result;
}

int passing_tests_counter = 0;
int failing_tests_counter = 0;
int test_number = 0;
int start_from = 0;
v7_lib *inference_dll_ptr = 0;

int last_run_engine_died;      /* set by run(): the script died a caught death */
char last_death_reason[512];   /* captured before the reset (v7_free_inference clears it) */

/* autoexec stdlib seeding: after v7_free_inference wipes the hash table
   (ht_destroy + ht_init -- NOTHING survives), the __sectorlogo__autoexec
   key must go back in before the next lg_init, or the stdlib falls back to
   the file tier and every death-reload pays a disk read. The script text
   is read once at startup and re-added here. */
char *autoexec_db_text = 0;
long autoexec_db_size = 0;

/* the validator's --db argument, kept for suites that must RELOAD the whole
   table from disk (the live-engine suite: earlier FATAL-catch revives wipe
   the model keys, and only the autoexec key is re-seeded) */
const char *validator_db_path = 0;

void seed_autoexec_db(void)
{
	if (!autoexec_db_text)
		return;
	if (!inference_dll_ptr->ht_add("__sectorlogo__autoexec", autoexec_db_text, (unsigned __int64)autoexec_db_size, 1))
		printf("ht_add('__sectorlogo__autoexec') re-seed failed\n");
}

int run(const char *script)
{
	int result = inference_dll_ptr->v7s_sectorlogo_interp((char *)script);
	last_run_engine_died = inference_dll_ptr->v7_engine_is_dead();
	if (last_run_engine_died)
	{
		const char *reason = inference_dll_ptr->v7_get_death_reason();
		snprintf(last_death_reason, sizeof(last_death_reason), "%s", reason ? reason : "?");
		inference_dll_ptr->v7_free_inference(); /* revive: one FATAL must not blind the rest of the suite */
		seed_autoexec_db();                     /* the revive wiped the table: put the stdlib key back */
	}
	else
	{
		last_death_reason[0] = 0;
	}
	return result;
}

/* Fail the current test when its script died a caught death: the value
   assertions would otherwise compare against the meaningless death-return 0
   and every LATER test would silently return NIL (the dead-engine cascade). */
int run_died(const char *desc)
{
	if (!last_run_engine_died)
		return 0;
	failing_tests_counter++;
	printf("  FAIL: %s -> script died: %s\n", desc, last_death_reason);
	return 1;
}

/* RUN_SUITE gates test-suite calls on -startfrom N.
   Each suite gets a sequential number. Suites before N are skipped.
   Every executed suite is timed (QueryPerformanceCounter) and the
   elapsed milliseconds are printed AND appended to the timing log
   (build_logs/v_sectorlogo_validate_timing.log) so slow suites are
   identifiable from one run.
   Slow suites (dominated by ~1s engine-recovery joins per caught FATAL,
   console-poll sleeps, and one outright hang) are tagged with
   RUN_SLOW_SUITE and skipped unless -with-slow or -only-slow is given. */
FILE *suite_timing_log = 0;
int include_slow_suites = 0;
int only_slow_suites = 0;

#define RUN_SUITE_TIMED(n, call, slow_gate) \
	do \
	{ \
		test_number = n; \
		if (test_number < start_from) break; \
		if (slow_gate) \
		{ \
			if (!(include_slow_suites) && !(only_slow_suites)) \
			{ \
				printf("suite %d skipped (slow -- -with-slow includes it, -only-slow runs it alone)\n", test_number); \
				break; \
			} \
		} \
		else \
		{ \
			if (only_slow_suites) break; \
		} \
		{ \
			LARGE_INTEGER suite_timer_start, suite_timer_end, suite_timer_freq; \
			QueryPerformanceCounter(&suite_timer_start); \
			call; \
			QueryPerformanceCounter(&suite_timer_end); \
			QueryPerformanceFrequency(&suite_timer_freq); \
			double suite_elapsed_ms = (double)(suite_timer_end.QuadPart - suite_timer_start.QuadPart) * 1000.0 / (double)suite_timer_freq.QuadPart; \
			printf("suite %d finished in %.1f ms\n", test_number, suite_elapsed_ms); \
			if (suite_timing_log) \
			{ \
				fprintf(suite_timing_log, "suite %4d  %10.1f ms\n", test_number, suite_elapsed_ms); \
				fflush(suite_timing_log); \
			} \
		} \
	} while (0)

#define RUN_SUITE(n, call) RUN_SUITE_TIMED(n, call, 0)
#define RUN_SLOW_SUITE(n, call) RUN_SUITE_TIMED(n, call, 1)

void test_num(const char *desc, const char *script, int expected)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	if (is_num(interp_result) && num_val(interp_result) == expected)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		if (is_num(interp_result))
			printf("  FAIL: %s -> got %d (0x%08X), expected %d\n", desc, num_val(interp_result), (unsigned)interp_result, expected);
		else
			printf("  FAIL: %s -> got non-NUMBER 0x%08X, expected %d\n", desc, (unsigned)interp_result, expected);
	}
}

void test_true(const char *desc, const char *script)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	if (interp_result != 0)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> got NIL, expected TRUE\n", desc);
	}
}

/* assert a substring in a live-engine output buffer (chat-turn suites) */
void test_contains(const char *test_label, const char *actual_text, const char *expected_substring)
{
	if (strstr(actual_text, expected_substring) != 0)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		failing_tests_counter++;
		printf("  FAIL: %s (substring '%s' missing in output)\n", test_label, expected_substring);
	}
}

void test_nil(const char *desc, const char *script)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	if (is_nil(interp_result))
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> got 0x%08X, expected NIL\n", desc, (unsigned)interp_result);
	}
}

/* Test that result is a NUMBER with a nonzero value (for FFI tests where
   the exact value is non-deterministic, e.g. GetTickCount, num_workers). */
void test_num_nonzero(const char *desc, const char *script)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	if (is_num(interp_result) && num_val(interp_result) != 0)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> 0x%08X, expected nonzero NUMBER\n", desc, (unsigned)interp_result);
	}
}

void test_float_val(const char *desc, const char *script, float expected)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	if (is_float(interp_result))
	{
		float got = float_val(interp_result);
		float diff = got - expected;
		if (diff < 0)
			diff = -diff;
		float tol;
		if (expected != 0.0f)
		{
			float abs_expected = expected;
			if (abs_expected < 0)
				abs_expected = -abs_expected;
			tol = abs_expected * 1e-5f;
		}
		else
		{
			tol = 1e-6f;
		}
		if (diff <= tol)
		{
			passing_tests_counter++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("  TEST: %s  [%s]\n", desc, script);
			failing_tests_counter++;
			printf("  FAIL: %s -> got %.8g, expected %.8g (diff %.8g)\n", desc, (double)got, (double)expected, (double)diff);
		}
	}
	else if (is_num(interp_result))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> got NUMBER %d, expected FLOAT %.8g\n", desc, num_val(interp_result), (double)expected);
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> got 0x%08X, expected FLOAT %.8g\n", desc, (unsigned)interp_result, (double)expected);
	}
}

/* Assert a float literal parses to the EXACT correctly-rounded binary32 bits.
   expected_double is a C double literal; the (float) cast below is C's own
   correctly-rounded double-to-float step. The FLOAT tag carries only 30 bits
   (the 2 low mantissa bits are dropped by the tag format on EVERY float, so
   the reference is masked with ~3u before the compare) -- a double-rounding
   (binary64 build, binary32 accumulate, round again) still shows up as a
   multi-ULP bit mismatch that the relative-tolerance test_float_val misses. */
void test_float_bits(const char *desc, const char *script, double expected_double)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	if (is_float(interp_result))
	{
		float reference_float = (float)expected_double;
		unsigned reference_bits;
		memcpy(&reference_bits, &reference_float, 4);
		reference_bits &= ~3u; /* the 30-bit FLOAT tag drops the 2 low mantissa bits */
		unsigned got_bits = ((unsigned)(interp_result & SL_PAYLOAD_MASK)) << 2;
		if (got_bits == reference_bits)
		{
			passing_tests_counter++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("  TEST: %s  [%s]\n", desc, script);
			failing_tests_counter++;
			float got_float;
			memcpy(&got_float, &got_bits, 4);
			printf("  FAIL: %s -> got %.9g (0x%08X), expected %.9g (0x%08X)\n", desc, (double)got_float, got_bits, (double)reference_float, reference_bits);
		}
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> got 0x%08X, expected FLOAT with bits of %.17g\n", desc, (unsigned)interp_result, expected_double);
	}
}

/* Test that a script returns a FLOAT that is NaN (float_val comparison is
   useless for NaN -- diff <= tolerance is always false -- so check isnan). */
void test_float_nan(const char *desc, const char *script)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	if (is_float(interp_result))
	{
		float got = float_val(interp_result);
		if (got != got) /* NaN is the only value not equal to itself */
		{
			passing_tests_counter++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("  TEST: %s  [%s]\n", desc, script);
			failing_tests_counter++;
			printf("  FAIL: %s -> %.8g, expected NaN\n", desc, (double)got);
		}
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> got 0x%08X, expected FLOAT NaN\n", desc, (unsigned)interp_result);
	}
}

/* Test that a script returns a FLOAT infinity of the given sign (+1 / -1).
   test_float_val cannot check infinities: inf - inf is NaN, so diff <= tol is
   always false, and the same holds for any NaN expectation. */
void test_float_inf(const char *desc, const char *script, int sign)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	if (is_float(interp_result))
	{
		float got = float_val(interp_result);
		int got_sign = 0;
		if (got > 0.0f && got == got * 2.0f)
			got_sign = 1;   /* larger than every finite float and still finite-scaled: +inf */
		else if (got < 0.0f && got == got * 2.0f)
			got_sign = -1;  /* same test on the negative side: -inf */
		if (got_sign == sign)
		{
			passing_tests_counter++;
			printf(".");
			fflush(stdout);
		}
		else
		{
			printf("  TEST: %s  [%s]\n", desc, script);
			failing_tests_counter++;
			printf("  FAIL: %s -> %.8g, expected %sinf\n", desc, (double)got, sign > 0 ? "+" : "-");
		}
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> got 0x%08X, expected FLOAT %sinf\n", desc, (unsigned)interp_result, sign > 0 ? "+" : "-");
	}
}

/* Test that a script returns a FLOAT (any value, just check the tag). */
void test_is_float(const char *desc, const char *script)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	if (is_float(interp_result))
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> got 0x%08X, expected FLOAT\n", desc, (unsigned)interp_result);
	}
}

void test_atom(const char *desc, const char *script)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	int tag = interp_result & SL_TAG_MASK;
	if (interp_result != 0 && tag == 0)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> got 0x%08X, expected atom\n", desc, (unsigned)interp_result);
	}
}

void test_cons(const char *desc, const char *script)
{
	int interp_result = run(script);
	if (run_died(desc))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		return;
	}
	if (interp_result & SL_CONS_TAG)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> got 0x%08X, expected cons\n", desc, (unsigned)interp_result);
	}
}

/* Caught-FATAL case: the script must die a CAUGHT
   death. Asserts interp result 0, v7_engine_is_dead()==1, and the death
   reason containing the expected text (Sec.173: the reason names the
   offender). v7_free_inference() resets the death state afterwards so the
   engine is alive again for the next test -- several FATALs per process. */
void test_fatal_catch_case(const char *desc, const char *script, const char *expected_reason_part)
{
	int interp_result = run(script);
	int engine_dead = last_run_engine_died;      /* captured in run() before the reset */
	const char *death_reason = last_death_reason;
	if ((interp_result != 0) || (engine_dead != 1))
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> result 0x%08X dead=%d, expected 0 + dead=1\n", desc, (unsigned)interp_result, engine_dead);
	}
	else if (strstr(death_reason, expected_reason_part) == 0)
	{
		printf("  TEST: %s  [%s]\n", desc, script);
		failing_tests_counter++;
		printf("  FAIL: %s -> reason '%s' missing '%s'\n", desc, death_reason, expected_reason_part);
	}
	else
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
}

/* ====== TEST SUITES ====== */

void test_integer_literals()
{
	printf("\n--- Integer literals ---\n");
	test_num("0", "Output 0", 0);
	test_num("1", "Output 1", 1);
	test_num("42", "Output 42", 42);
	test_num("12345", "Output 12345", 12345);
	test_num("-1", "Output -1", -1);
	test_num("-42", "Output -42", -42);
	test_num("999999", "Output 999999", 999999);
}

void test_arith_basics()
{
	printf("\n--- Infix arithmetic ---\n");
	test_num("2 + 3", "Output (2 + 3)", 5);
	test_num("-5 + 10", "Output (-5 + 10)", 5);
	test_num("10 - 3", "Output (10 - 3)", 7);
	test_num("3 - 10", "Output (3 - 10)", -7);
	test_num("6 * 7", "Output (6 * 7)", 42);
	test_num("-3 * 8", "Output (-3 * 8)", -24);
	test_num("42 / 6", "Output (42 / 6)", 7);
	test_num("7 / 2", "Output (7 / 2)", 3);
	test_num("7 % 2", "Output (7 % 2)", 1);
	test_num("Power 2 10", "Output Power 2 10", 1024);
	test_num("Power 3 3", "Output Power 3 3", 27);
	test_num("Power 5 0", "Output Power 5 0", 1);
	test_num("SquareRoot 49", "Output SquareRoot 49", 7);
	test_num("SquareRoot 100", "Output SquareRoot 100", 10);
	test_num("SquareRoot 0", "Output SquareRoot 0", 0);
	test_num("AbsoluteValue 42", "Output AbsoluteValue 42", 42);
	test_num("AbsoluteValue -42", "Output AbsoluteValue -42", 42);
	test_num("Negate 5", "Output Negate 5", -5);
	test_num("Negate -3", "Output Negate -3", 3);
	test_num("Maximum 3 7", "Output Maximum 3 7", 7);
	test_num("Maximum 7 3", "Output Maximum 7 3", 7);
	test_num("Minimum 3 7", "Output Minimum 3 7", 3);
	test_num("Minimum 7 3", "Output Minimum 7 3", 3);
	/* 30-bit payload SATURATING arithmetic (owner directive: math clamps,
	   never FATALs). MakeNum's mask cannot DETECT an out-of-range result, so
	   an unchecked positive sum would come back NEGATIVE (400000000 +
	   400000000 -> -273741824) with no diagnostic at all. The integer
	   operators saturate at the payload boundaries instead of dying, so every
	   result a script can read is the correct magnitude up to the clamp. */
	test_num("at the payload ceiling", "Output (536870911 + 0)", 536870911);
	test_num("just under the ceiling", "Output (536870910 + 1)", 536870911);
	test_num("at the payload floor", "Output (-536870911 + 0)", -536870911);
	test_num("Multiply at the payload floor", "Output (-536870911 * 1)", -536870911);
	test_num("ShiftLeft just inside the ceiling", "Output (1 << 28)", 268435456);
	test_num("Add past the payload ceiling saturates", "Output (400000000 + 400000000)", 536870911);
	test_num("Subtract past the payload floor saturates", "Output (-536870911 - 536870911)", -536870912);
	test_num("Multiply past the payload range saturates", "Output (1000000 * 1000000)", 536870911);
	/* RUN in expression position rejects every non-CONS, non-NIL value. The bare
	   FATAL it used to carry named neither the type nor the value, so a script
	   author had nothing to act on. */
	test_fatal_catch_case("Run on a bare atom names the offender",
	                      "Output Run \"hello\"",
	                      ", got word '");
	test_fatal_catch_case("Run on a NUMBER names the offender",
	                      "Output Run 42",
	                      ", got number '");
	/* a 64-bit-exact product is required: 65536*65536 = 2^32 truncates to 0 in
	   32 bits, so a 32-bit-only clamp decision would pass this through as
	   NUMBER 0 */
	test_num("Multiply wrapping past 32 bits saturates", "Output (65536 * 65536)", 536870911);
	test_num("ShiftLeft past the payload ceiling saturates", "Output (1 << 29)", 536870911);
	/* Divide: -2^29 / -1 = +2^29, one past the ceiling; MakeNum's mask would
	   silently truncate it back to -2^29 (the exact negation of the answer).
	   It saturates instead. */
	test_num("Divide at the payload ceiling", "Output (536870911 / 1)", 536870911);
	test_num("Divide past the payload ceiling saturates", "Output (-536870912 / -1)", 536870911);
	/* the INFIX spellings share the NUM_RANGE_CHECK macro with the prefix
	   forms, so infix saturates identically -- one arithmetic boundary, one
	   behaviour, whichever spelling the script uses. */
	test_num("Infix Multiply wrapping past 32 bits saturates", "Output (65536 * 65536)", 536870911);
	test_num("Infix Add past the payload ceiling saturates", "Output (400000000 + 400000000)", 536870911);
	test_num("Infix Subtract past the payload floor saturates", "Output (-536870911 - 536870911)", -536870912);
	test_num("Infix ShiftLeft past the payload ceiling saturates", "Output (1 << 29)", 536870911);
	test_num("Infix ShiftLeft by 31 with an in-range result is legal",
	         "Output 0 << 31", 0);
	test_num("Infix Divide past the payload ceiling saturates", "Output -536870912 / -1", 536870911);
	test_num("Infix Divide at the payload ceiling", "Output -536870911 / -1", 536870911);
	test_num("Infix Modulo keeps the signed remainder", "Output -7 % 3", -1);
	/* shift COUNT. x86 masks a 32-bit shift's cl to 5
	   bits in hardware, so a negative or >= 32 count silently answered a
	   DIFFERENT question than the script asked: "Output 8 >> -1" returned 0,
	   "Output 8 >> 32" returned 8, and "Output 8 << 33" returned 16 -- no
	   diagnostic at all. The count is now range-checked in both spellings, and
	   the message is identical either way so one authoring mistake reads the
	   same regardless of which form the script used. */
	test_fatal_catch_case("Infix ShiftRight with a negative count dies",
	                      "Output 8 >> -1",
	                      "shift count must be a NUMBER in the range 0..31");
	test_fatal_catch_case("Infix ShiftLeft with a negative count names the count, not the result",
	                      "Output 8 << -1",
	                      "shift count must be a NUMBER in the range 0..31");
	test_fatal_catch_case("Infix ShiftRight with a count over 31 dies",
	                      "Output 8 >> 32",
	                      "shift count must be a NUMBER in the range 0..31");
	test_fatal_catch_case("Infix ShiftLeft with a count over 31 dies",
	                      "Output 8 << 33",
	                      "shift count must be a NUMBER in the range 0..31");
	test_num("Infix ShiftRight by 0 is the identity", "Output 8 >> 0", 8);
	test_num("Infix ShiftRight keeps the sign", "Output -8 >> 1", -4);
	/* the exponentiation-by-squaring intermediates saturate too: 2^40
	   overflows the running BASE square long before the accumulator leaves
	   range; with both clamped the visible result is the ceiling */
	test_num("Power with an out-of-range intermediate saturates",
	         "Output Power 2 40",
	         536870911);
	/* F2I saturation path (owner-directed exception to the range-FATAL law).
	   A finite float whose truncated value lands outside the 30-bit NUMBER
	   payload SATURATES to the nearest representable NUMBER; NaN alone keeps
	   the loud FATAL (it has no nearest value). The float-domain compares run
	   before conversion, so even beyond-int32 magnitudes clamp instead of
	   hitting an integer-indefinite pattern. */
	test_num("AsInteger of +inf saturates at the payload max",
	         "Output AsInteger (1.0 / 0.0)",
	         536870911);
	test_fatal_catch_case("AsInteger of NaN dies",
	                      "Output AsInteger (0.0 / 0.0)",
	                      "AsInteger: value is NaN, no clamped value exists");
	test_num("AsInteger of a huge float saturates at the payload max",
	         "Output AsInteger (100000.0 * 100000.0)",
	         536870911);
	test_num("AsInteger of a huge negative float saturates at the payload min",
	         "Output AsInteger (-100000.0 * 100000.0)",
	         -536870912);
	test_num("AsInteger of an in-range float still converts",
	         "Output AsInteger (9.0 / 2.0)",
	         4);
	/* 30-bit payload saturation (owner-directed). MakeNum is
	   `and eax, SL_PAYLOAD_MASK` -- it TRUNCATES silently, it cannot detect an
	   out-of-range input. AsInteger clamps in the float domain before
	   converting, so a past-payload float saturates at +/-2^29 instead of
	   wrapping silently:
	     AsInteger 1e9        -> 536870911   (saturated, the old wrap read -463129088)
	     AsInteger 2^30       -> 536870911   (saturated, the old wrap read 0)
	     AsInteger -1e9       -> -536870912  (saturated)
	   The integer operators saturate identically (NUM_RANGE_CHECK), so
	   conversions and arithmetic agree at the boundaries. */
	test_num("AsInteger of 1e9 saturates instead of reading back negative",
	         "Output AsInteger 1000000000.0",
	         536870911);
	test_num("AsInteger of 2^30 saturates instead of reading back zero",
	         "Output AsInteger 1073741824.0",
	         536870911);
	test_num("AsInteger of -1e9 saturates at the payload min",
	         "Output AsInteger -1000000000.0",
	         -536870912);
	test_num("AbsoluteValue of the most negative NUMBER saturates",
	         "Output AbsoluteValue -536870912",
	         536870911);
	test_num("Negate of the most negative NUMBER saturates",
	         "Output Negate -536870912",
	         536870911);
	test_num("unary Subtract of the most negative NUMBER saturates",
	         "Output - -536870912",
	         536870911);
	/* missing neg in .p_diff_unary_int: the block
	   documented an "and/neg pair" but emitted only the `and`, so unary
	   subtract was the identity -- `- 5` returned 5. */
	test_num("unary minus negates an integer", "Output - 5", -5);
	test_float_val("unary minus negates a float", "Output - 5.0", -5.0f);
	test_num("unary minus negates a negative NUMBER", "Output - -3", 3);
	test_num("unary minus handles the largest legal NUMBER", "Output - 536870911", -536870911);
	test_num("prefix minus symbol negates", "Output - 5", -5);
	/* A unary/binary `-` followed by a builtin that heads an EXPRESSION call
	   must consume that whole call as its second operand -- it is a value-start.
	   Routing every registered builtin to the unary arm (a plausible-looking fix for
	   `Make "x" -5` + a following statement keyword) broke exactly this, because
	   sl_builtin_bitmap cannot tell an operand-taking builtin from a no-operand
	   statement keyword. This is the regression net for that. */
	test_num("prefix minus consumes a following expression call as its 2nd operand",
	         "Output (0 - Power 2 4)", -16);
	/* OPT-2: this was the only bare FATAL reachable from .p_diff -- every other
	   type rejection in the file names the value, and Sec.173 requires it. */
	test_fatal_catch_case("unary subtract of a non-number names the offender",
	                      "Output (- [1 2])",
	                      "Subtract: unary negation requires a number");
	/* Sec.173: a negative SQRT argument is a domain violation. The signed jle
	   used to sweep every n < 0 into the same block as n == 0, so `SQRT -5`
	   answered 0 while `SQRT -5.0` answered NaN. */
	test_num("SQRT 0 is 0", "Output SquareRoot 0", 0);
	test_num("SQRT 16 is 4", "Output SquareRoot 16", 4);
	test_fatal_catch_case("SQRT of a negative NUMBER names the offender",
	                      "Output SquareRoot -5",
	                      "SquareRoot: argument must be >= 0");
	/* the off-by-one guard in the OTHER direction: the new checks must not have
	   become an off-by-one that rejects the largest representable magnitude.
	   536870784.0, not 536870911.0. A SectorLogo FLOAT is a 30-bit TRUNCATED
	   binary32 (SL_FLOAT_SHIFT drops the low 2 mantissa bits), so the spacing at
	   [2^28, 2^29) is 128, not 32: 536870911.0 is not representable and the
	   literal rounds UP to 2^29, which is exactly the value the new check
	   rejects -- correctly so. 536870784.0 is the largest FLOAT below 2^29. */
	test_num("AbsoluteValue of the most positive NUMBER still works",
	         "Output AbsoluteValue 536870911",
	         536870911);
	test_num("AbsoluteValue of -536870911 still works",
	         "Output AbsoluteValue -536870911",
	         536870911);
	test_num("Negate of the most positive NUMBER still works",
	         "Output Negate 536870911",
	         -536870911);
	test_num("AsInteger of the largest in-range float still converts",
	         "Output AsInteger 536870784.0",
	         536870784);
}

void test_infix_arithmetic()
{
	printf("\n--- Infix arithmetic ---\n");
	test_num("2 + 3", "Output 2 + 3", 5);
	test_num("10 - 3", "Output 10 - 3", 7);
	test_num("6 * 7", "Output 6 * 7", 42);
	test_num("42 / 6", "Output 42 / 6", 7);
	test_num("7 / 2", "Output 7 / 2", 3);
	test_num("10 pct 3", "Output 10 % 3", 1);
	test_num("1 + 2 + 3", "Output 1 + 2 + 3", 6);
}

void test_math_precedence()
{
	printf("\n--- Math operator precedence (*/%% before +/-) ---\n");
	test_num("2 + 3 * 4", "Output 2 + 3 * 4", 14);
	test_num("10 - 2 * 3", "Output 10 - 2 * 3", 4);
	test_num("2 * 3 + 4", "Output 2 * 3 + 4", 10);
	test_num("2 * 3 + 4 * 5", "Output 2 * 3 + 4 * 5", 26);
	test_num("100 / 10 + 3", "Output 100 / 10 + 3", 13);
	test_num("3 + 100 / 10", "Output 3 + 100 / 10", 13);
	test_num("20 - 15 % 4", "Output 20 - 15 % 4", 17);
	test_num("2 + 3 * 4 + 5", "Output 2 + 3 * 4 + 5", 19);
	test_num("1 + 2 * 3 + 4 * 5 + 6", "Output 1 + 2 * 3 + 4 * 5 + 6", 33);
	test_num("2 * (3 + 4)", "Output 2 * (3 + 4)", 14);
	test_num("(2 + 3) * 4", "Output (2 + 3) * 4", 20);
	test_num("(10 - 2) * (3 + 1)", "Output (10 - 2) * (3 + 1)", 32);
}

void test_divmod_edge()
{
	printf("\n--- Division/modulo edge cases (non-zero) ---\n");
	test_num("0 / 5", "Output 0 / 5", 0);
	test_num("0 % 5", "Output 0 % 5", 0);
}

void test_infix_bitwise()
{
	printf("\n--- Infix bitwise ---\n");
	test_num("12 & 10", "Output (12 & 10)", 8);
	test_num("12 | 10", "Output (12 | 10)", 14);
	test_num("12 ^ 10", "Output (12 ^ 10)", 6);
	test_num("~0", "Output ~0", ~0);
	test_num("1 << 4", "Output (1 << 4)", 16);
	test_num("16 >> 2", "Output (16 >> 2)", 4);
	test_num("7 << 3", "Output (7 << 3)", 56);
	test_num("12 band 10", "Output (12 & 10)", 8);
	test_num("12 bor 10", "Output (12 | 10)", 14);
	test_num("12 bxor 10", "Output (12 ^ 10)", 6);
}

void test_infix_relational()
{
	printf("\n--- Infix relational ---\n");
	test_true("3 eq 3", "Output (3 = 3)");
	test_nil("3 eq 4", "Output (3 = 4)");
	/* Sec.170 coverage: NIL is the value 0 and 0 & SL_TAG_MASK is 0, so it is
	   indistinguishable from a plain atom by tag alone. Without the two guards at
	   .atom_or_cons, the text compare read heap[1] (NIL's NUL content) against the
	   empty-string atom's NUL and reported NIL == "". The last two pin the
	   behaviour that must NOT change. */
	test_nil("NIL vs empty string is false", "Output ((First []) = \"\")");
	test_nil("empty string vs NIL is false", "Output (\"\" = (First []))");
	test_true("NIL vs NIL is still true", "Output ((First []) = (First []))");
	test_true("empty vs empty is still true", "Output (\"\" = \"\")");
	test_true("3 ne 4", "Output (3 != 4)");
	test_nil("3 ne 3", "Output (3 != 3)");
	test_true("5 gt 3", "Output (5 > 3)");
	test_nil("3 gt 5", "Output (3 > 5)");
	test_true("3 lt 5", "Output (3 < 5)");
	test_nil("5 lt 3", "Output (5 < 3)");
	test_true("5 ge 5", "Output (5 >= 5)");
	test_nil("3 ge 5", "Output (3 >= 5)");
	test_true("5 le 5", "Output (5 <= 5)");
	test_nil("5 le 3", "Output (5 <= 3)");
	/* INFIX_COMPARE entry guard (eval_infix BUG-2). A
	   chained comparison feeds the previous result back in as the lhs, and that
	   boolean reached sl_promote_pair -> NumVal, which died with "a NUMBER was
	   required (arithmetic, index, count, or key operand)" -- four kinds of operand,
	   none of them the comparison the author wrote. The guard names it instead. */
	test_fatal_catch_case("chained comparison names the mistake",
	                      "Output (1 < 2 < 3)",
	                      "chained comparison like 1 < 2 < 3");
	test_fatal_catch_case("chained comparison on the false result names it too",
	                      "Output (5 < 3 < 9)",
	                      "chained comparison like 1 < 2 < 3");
	test_fatal_catch_case("chained ge comparison names the mistake",
	                      "Output (3 >= 2 >= 1)",
	                      "chained comparison like 1 < 2 < 3");
	/* a NON-numeric lhs from a source that is not a comparison (NIL) must still
	   die, and with a message that is true of that case too */
	test_fatal_catch_case("comparison against NIL still needs a number on the left",
	                      "Output ((First []) < 3)",
	                      "comparison needs a NUMBER on the left");
	/* the continuation that does NOT route through INFIX_COMPARE must keep
	   working: = is structural, so it needs no numeric lhs */
	test_nil("boolean then = still works", "Output ((1 < 2) = 3)");
	test_true("plain comparison still works", "Output (2 < 3)");
	test_true("arithmetic lhs still works", "Output ((1 + 1) < 3)");
	test_true("parenthesised comparison still works", "Output (1 < 2)");
}

void test_boolean()
{
	printf("\n--- Boolean operators ---\n");
	test_true("1 && 1", "Output ((1 = 1) && (1 = 1))");
	test_nil("1 && 0", "Output ((1 = 1) && (1 = 0))");
	test_nil("0 && 1", "Output ((1 = 0) && (1 = 1))");
	test_nil("0 && 0", "Output ((1 = 0) && (1 = 0))");
	test_true("1 || 0", "Output ((1 = 1) || (1 = 0))");
	test_true("0 || 1", "Output ((1 = 0) || (1 = 1))");
	test_nil("0 || 0", "Output ((1 = 0) || (1 = 0))");
	test_nil("NIL || NIL", "Output ([] || [])");
	test_true("1 || NIL", "Output ((1 = 1) || [])");
	test_true("NIL || 1", "Output ([] || (1 = 1))");
	test_true("! 0", "Output (! 0)");
	test_nil("! 1", "Output (! 1)");
	/* && and || are INFIX ONLY (user directive, Oct 2026): the expression table
	   has no entries for their atoms, so a prefix spelling is a dead name that
	   self-evaluates -- exactly the removed-AND/OR/NOT doctrine (Sec.146 fact
	   38). IsWordAtom pins the self-evaluated WORD atom: not NIL, not a FATAL,
	   not a dispatched prefix call. */
	test_true("prefix && spelling is a dead word atom", "Output IsWordAtom (&&)");
	test_true("prefix || spelling is a dead word atom", "Output IsWordAtom (||)");
	test_nil("prefix && spelling is not a string atom", "Output IsStringAtom (&&)");
	test_true("IsZero 0", "Output IsZero 0");
	test_nil("IsZero 1", "Output IsZero 1");
	test_true("IsNumber 5", "Output IsNumber 5");
	test_true("IsNumber 0", "Output IsNumber 0");
	test_nil("IsNumber on string atom", "Output IsNumber \"42\"");
	/* Equal coerces a plain ATOM rhs numerically (sl_atom_num) so that a DB-GET
	   result spelling "1" compares equal to the literal 1. That coercion has a
	   signed bound: a negative atom whose magnitude exceeds 2^31 must be rejected
	   as non-numeric, not truncated to a NUMBER. The bound constant is loaded with
	   a 32-bit destination so it zero-extends; a 64-bit destination sign-extends
	   0x80000000 to 0xFFFFFFFF80000000, which puts the comparison out of reach and
	   makes every out-of-range negative atom coerce to 0 -- so this Equal answered
	   TRUE. */
	test_nil("Equal rejects an atom magnitude past int32 instead of coercing it", "Output (0 = -4294967296)");
	/* IsStringAtom/IsWordAtom: expose the atom-header type byte (the real
	   primitive pair behind the historical phantom "IsWord" claim) */
	test_true("IsStringAtom on string literal", "Output IsStringAtom \"abc\"");
	/* .expr_fallback reading the type byte directly instead of
	   calling lg_ensure_word and comparing the returned offset. The contract being
	   pinned is that a quoted literal SELF-EVALUATES -- it is data, and must never
	   dispatch as the builtin whose name it spells. Both operands here are quoted
	   builtin names, so a regression that let them dispatch would consume the next
	   form as an argument instead of returning the six-byte string. */
	test_num("Quoted builtin name self-evaluates, does not dispatch", "Output TextLength \"output\"", 6);
	test_num("Quoted operator literal self-evaluates, does not dispatch", "Output TextLength \"+\"", 1);
	test_true("Quoted operator literal is a STRING atom", "Output IsStringAtom \"+\"");
	/* The discriminator between the two arms: a BARE word of the same text is a
	   WORD atom and takes the ':var'/numeric-literal path, so it must NOT report as
	   a string atom -- that is what makes the single type-byte test sufficient. */
	test_nil("Bare word of the same text is not a STRING atom", "Output IsStringAtom output");
	test_num("Bare numeric literal still parses as a number", "Output (2 + 3)", 5);
	test_nil("Quoted numeric literal is NOT a number", "Output IsNumber \"2\"");
	test_nil("IsStringAtom on AsWord result", "Output IsStringAtom AsWord \"abc\"");
	test_nil("IsStringAtom on bare word", "Output IsStringAtom hello");
	test_nil("IsStringAtom on number", "Output IsStringAtom 42");
	test_nil("IsStringAtom on float", "Output IsStringAtom 1.5");
	test_nil("IsStringAtom on list", "Output IsStringAtom [ 1 ]");
	test_nil("IsStringAtom on NIL", "Output IsStringAtom []");
	test_true("IsWordAtom on AsWord result", "Output IsWordAtom AsWord \"abc\"");
	test_nil("IsWordAtom on string literal", "Output IsWordAtom \"abc\"");
	test_true("IsWordAtom on bare word", "Output IsWordAtom hello");
	test_nil("IsWordAtom on number", "Output IsWordAtom 42");
	test_nil("IsWordAtom on list", "Output IsWordAtom [ 1 ]");
	test_nil("IsWordAtom on NIL", "Output IsWordAtom []");
	/* the pair is exhaustive over plain atoms: exactly one of the two answers true */
	test_num("IsStringAtom/IsWordAtom split strings", "Output ((IfElse IsStringAtom \"s\" [1] [0]) + (IfElse IsWordAtom \"s\" [1] [0]))", 1);
	test_num("IsStringAtom/IsWordAtom split words", "Output ((IfElse IsStringAtom w [1] [0]) + (IfElse IsWordAtom w [1] [0]))", 1);
}

void test_list_ops()
{
	printf("\n--- List operations ---\n");
	test_cons("Prepend 1 NIL", "Output Prepend 1 0");
	test_num("First of Prepend", "Output First Prepend 5 0", 5);
	test_nil("ButFirst of Prepend", "Output ButFirst Prepend 5 []");
}

void test_make_and_vars()
{
	printf("\n--- Variables (Make / :var) ---\n");
	test_num("Make X=42", "Make \"X\" 42 Output :X", 42);
	test_num("Make A=10 B=20", "Make \"A\" 10 Make \"B\" 20 Output (:A + :B)", 30);
	test_num("Make VAL=-5", "Make \"VAL\" -5 Output AbsoluteValue :VAL", 5);
}

void test_if_ifelse()
{
	printf("\n--- If / IfElse ---\n");
	test_num("If true", "If 1 [ Make \"R\" 99 ] Output :R", 99);
	test_num("If false", "Make \"R\" 0 If 0 [ Make \"R\" 99 ] Output :R", 0);
	test_num("IfElse true", "IfElse 1 [ Make \"R\" 10 ] [ Make \"R\" 20 ] Output :R", 10);
	test_num("IfElse false", "IfElse 0 [ Make \"R\" 10 ] [ Make \"R\" 20 ] Output :R", 20);
	/* value-form block contract (eval_primary): the branches
	   arrive RAW from PopCons, so a non-list branch was silently evaluated as NIL --
	   indistinguishable from the legitimate empty-block result. RUN already FATALs
	   the same mistake, so the two block-consuming forms must agree. The empty-block
	   path and both branch selections must stay working. */
	test_nil("IfElse with an empty then-branch is NIL", "Output IfElse 1 [] [Output 7]");
	test_num("IfElse takes the else branch", "Output IfElse 0 [Output 7] [Output 9]", 9);
	test_fatal_catch_case("IfElse on a non-list branch names the branch",
	                      "Output IfElse 1 5 6",
	                      "IFELSE requires a list");
}

void test_repeat()
{
	printf("\n--- Repeat ---\n");
	test_num("Repeat 5", "Make \"C\" 0 Repeat 5 [ Make \"C\" :C + 1 ] Output :C", 5);
	test_num("Repeat 3 add10", "Make \"S\" 0 Repeat 3 [ Make \"S\" :S + 10 ] Output :S", 30);
	test_num("Repeat 0", "Make \"S\" 0 Repeat 0 [ Make \"S\" 99 ] Output :S", 0);
	test_num("Repeat -1", "Make \"S\" 0 Repeat -1 [ Make \"S\" 99 ] Output :S", 0);
}

void test_stop()
{
	printf("\n--- Stop (break from Repeat) ---\n");
	test_nil("Stop prevents later Make", "Make \"X\" 1 Stop Make \"X\" 99 Output :X");
	/* STOP inside REPEAT acts like C break: exits loop, continues after it */
	test_num("Stop breaks Repeat early", "Make \"S\" 0 Repeat 100 [ Make \"S\" :S + 1 IfElse (:S = 5) [ Stop ] [] ] Output :S", 5);
	/* STOP only breaks the current REPEAT, not the whole program */
	test_num("Stop then continue after Repeat", "Make \"A\" 0 Make \"B\" 0 Repeat 10 [ Make \"A\" :A + 1 IfElse (:A = 3) [ Stop ] [] ] Make \"B\" 99 Output (:A + :B)", 102);
}

void test_continue()
{
	printf("\n--- Continue (skip to next Repeat iteration) ---\n");
	/* CONTINUE skips odd numbers: I is explicitly counted 1..5 */
	test_num("Continue skips odd numbers", "Make \"S\" 0 Make \"I\" 0 Repeat 5 [ Make \"I\" :I + 1 IfElse ((:I % 2) = 0) [] [ Continue ] Make \"S\" :S + :I ] Output :S", 6);
	/* CONTINUE in first iteration: body after CONTINUE is skipped */
	test_num("Continue at loop start", "Make \"S\" 0 Repeat 3 [ Continue Make \"S\" 99 ] Output :S", 0);
	/* CONTINUE does not break the loop: iterations still run */
	test_num("Continue preserves iteration count", "Make \"C\" 0 Repeat 5 [ Make \"C\" :C + 1 Continue Make \"C\" 99 ] Output :C", 5);
}

void test_terminate()
{
	printf("\n--- Exit (exit entire program) ---\n");
	/* TERMINATE exits the whole program immediately (old STOP behavior) */
	test_nil("Exit stops all execution", "Make \"X\" 1 Exit Make \"X\" 99 Output :X");
	/* TERMINATE inside REPEAT exits the program, not just the loop */
	test_nil("Exit in Repeat", "Make \"S\" 0 Repeat 100 [ Make \"S\" :S + 1 Exit ] Output :S");
}

void test_stop_continue_propagation()
{
	printf("\n--- Stop/Continue propagation ---\n");
	/* STOP inside IF inside REPEAT: propagates up to REPEAT */
	test_num("Stop in If propagates to Repeat", "Make \"S\" 0 Repeat 100 [ Make \"S\" :S + 1 If (:S = 7) [ Stop ] ] Output :S", 7);
	/* STOP inside function called from REPEAT: STOP is consumed at the
	   function boundary (standard Logo: STOP returns from the current
	   PROCEDURE) -- the caller's REPEAT keeps running. The old leaking
	   behavior (STOP crossing the fn boundary and breaking the caller's
	   loop) was cross-function break, inconsistent with standard Logo;
	   the fn-boundary consumption is the Sep 2026 ground truth. */
	test_num("Stop in fn returns from fn, loop continues", "Define \"check\" [:n] [ IfElse (:n = 5) [ Stop ] [] ] Make \"S\" 0 Repeat 100 [ Make \"S\" :S + 1 check :S ] Output :S", 100);
	/* CONTINUE inside IF inside REPEAT: propagates up to REPEAT */
	test_num("Continue in If propagates to Repeat", "Make \"S\" 0 Make \"I\" 0 Repeat 5 [ Make \"I\" :I + 1 If (:I = 3) [ Continue ] Make \"S\" :S + :I ] Output :S", 12);
	/* STOP consumed at the fn boundary (standard Logo procedure return) */
	test_num("Stop in fn returns, script continues", "Define \"f\" [] [ Stop Make \"X\" 99 ] Make \"X\" 1 f Output :X", 1);
	test_true("Stop in fn: fn output is NIL", "Define \"f\" [] [ Stop ] Make \"R\" f Output (:R = [])");
	test_num("nested fn Stop consumes innermost only", "Define \"inner\" [] [ Stop ] Define \"outer\" [] [ inner Output 42 ] Output outer", 42);
	/* STOP with NO REPEAT around it unwinds every stream out to the top and ends
	   the script body: .exec_stop raises the request and jumps to .stream_done, and
	   at top level there is no loop left to catch it (Sec.146 fact 31). The later
	   Repeat is therefore never reached and its Output never runs. */
	test_nil("Stop outside any Repeat terminates the top-level stream",
	          "IfElse 1 [ Stop ] [] Make \"trips\" 0 Repeat 3 [ Make \"trips\" :trips + 1 ] Output :trips");
	test_nil("Continue outside any Repeat terminates the top-level stream",
	          "IfElse 1 [ Continue ] [] Make \"trips\" 0 Repeat 3 [ Make \"trips\" :trips + 1 ] Output :trips");
	/* same through the endless entry -- both REPEAT modes converge on the shared
	   body-fetch path, and neither is ever entered here */
	test_nil("Stop outside any Repeat terminates the stream before an endless Repeat",
	          "IfElse 1 [ Stop ] [] Make \"trips\" 0 Repeat [ Make \"trips\" :trips + 1 IfElse (:trips = 4) [ Stop ] [] ] Output :trips");
	/* REGRESSION GUARD for the second half of the contract: a REPEAT that IS
	   entered drops a request left over from before its own body (both flags are
	   cleared at .repeat entry), so no stale request can shorten a later loop. The
	   trip is therefore spent inside a Define, which returns before the loop sees
	   anything, and all three trips run. */
	test_num("Stale Stop raised inside a fn does not truncate the caller's Repeat",
	         "Define \"raiser\" [] [ Stop ] Make \"trips\" 0 Repeat 3 [ Make \"trips\" :trips + 1 raiser ] Output :trips", 3);
	test_num("Stale Continue raised inside a fn does not truncate the caller's Repeat",
	         "Define \"raiser\" [] [ Continue ] Make \"trips\" 0 Repeat 3 [ Make \"trips\" :trips + 1 raiser ] Output :trips", 3);
	/* REGRESSION GUARD: a STOP raised by the REPEAT's OWN body, through a
	   nested IfElse block, must still break that loop on its first trip. A
	   naive fix that cleared the flags at the nested block's return (or at
	   .stream_loop) would instead run all three trips. */
	test_num("Stop in a nested block still breaks the enclosing Repeat",
	         "Make \"trips\" 0 Repeat 3 [ Make \"trips\" :trips + 1 IfElse 1 [ Stop ] [] ] Output :trips", 1);
	/* innermost-REPEAT rule: the inner loop consumes the request itself, so the
	   outer loop's post-body check sees a cleared flag and both its trips run.
	   The inner loop therefore trips once per outer trip (2), not three times. */
	test_num("Stop in a nested Repeat breaks only the inner loop (outer trips)",
	         "Make \"outer\" 0 Make \"inner\" 0 Repeat 2 [ Make \"outer\" :outer + 1 "
	         "Repeat 3 [ Make \"inner\" :inner + 1 Stop ] ] Output :outer", 2);
	test_num("Stop in a nested Repeat breaks only the inner loop (inner trips)",
	         "Make \"outer\" 0 Make \"inner\" 0 Repeat 2 [ Make \"outer\" :outer + 1 "
	         "Repeat 3 [ Make \"inner\" :inner + 1 Stop ] ] Output :inner", 2);
	test_num("Continue in a nested block still skips exactly one trip",
	         "Make \"trips\" 0 Repeat 3 [ Make \"trips\" :trips + 1 IfElse 1 [ Continue ] [] ] Output :trips", 3);
}

void test_repeat_endless()
{
	printf("\n--- Repeat endless (count omitted) ---\n");
	/* STOP is the canonical exit: endless loop counting to 5 */
	test_num("endless + Stop", "Make \"C\" 0 Repeat [ Make \"C\" :C + 1 IfElse (:C = 5) [ Stop ] [] ] Output :C", 5);
	/* execution continues after the broken endless loop */
	test_num("endless + Stop then after-loop code", "Make \"A\" 0 Repeat [ Make \"A\" :A + 1 IfElse (:A = 3) [ Stop ] [] ] Make \"B\" 99 Output (:A + :B)", 102);
	/* CONTINUE skips the rest of an iteration; loop runs until STOP */
	test_num("endless + Continue + Stop", "Make \"S\" 0 Make \"I\" 0 Repeat [ Make \"I\" :I + 1 IfElse (:I = 2) [ Continue ] [] IfElse (:I = 5) [ Stop ] [] Make \"S\" :S + :I ] Output :S", 8);
	/* TERMINATE inside an endless loop exits the whole program */
	test_nil("endless + Exit", "Make \"X\" 1 Repeat [ Exit ] Make \"X\" 99 Output :X");
	/* OUTPUT escapes an endless loop from inside a function */
	test_num("endless + Output in fn", "Define \"f\" [] [ Repeat [ Output 7 ] ] Output f", 7);
	/* STOP in a fn body is consumed at the fn boundary (standard Logo
	   procedure return -- see the propagation tests); the caller's endless
	   REPEAT runs on and terminates via its own STOP. */
	test_num("endless + Stop in fn returns from fn", "Define \"check\" [:n] [ IfElse (:n = 4) [ Stop ] [] ] Make \"S\" 0 Repeat [ Make \"S\" :S + 1 check :S IfElse (:S = 7) [ Stop ] [] ] Output :S", 7);
	/* regression guard: counted REPEAT still works via RUN-built stream
	   (FPUT produces a pre-tagged NUMBER in the count slot -- the peek must
	   see a NON-CONS head and route to the counted path). Note this test
	   cannot distinguish full-mask vs bit31-only compare: NUMBER has bit31
	   clear, so both compares correctly report "not CONS". The FLOAT-count
	   protection of the full-mask compare is code-inspection-only (a float
	   count NumVal-FATALs before any behavioral difference can be observed
	   in a live test). */
	test_num("counted via Run-built stream", "Make \"X\" 0 Run Prepend \"repeat\" Prepend 3 Prepend [ Make \"X\" :X + 1 ] [] Output :X", 3);
	/* endless over a block held in a variable -- the ONLY var-based endless
	   form (a bare :VAR in the count slot still FATALs via NumVal) */
	test_num("endless + Run-held body", "Make \"B\" [ Make \"X\" :X + 1 IfElse (:X = 5) [ Stop ] [] ] Make \"X\" 0 Repeat [ Run :B ] Output :X", 5);
	/* endless REPEAT as a WHILE body: RUN of an OUTPUT-ing condition block
	   (validator scripts do not load autoexec, so WHILE itself is not tested) */
	test_num("endless as While body", "Make \"I\" 0 Repeat [ IfElse Run [ Output (:I < 5) ] [ Make \"I\" :I + 1 ] [ Stop ] ] Output :I", 5);
}

void test_string_concat()
{
	printf("\n--- String concatenation (_ operator) ---\n");
	/* Concatenate two string atoms */
	test_atom("atom _ atom", "Output \"hello\" _ \"world\"");
	/* Concatenate a string and a number */
	test_atom("atom _ number", "Output \"val\" _ 42");
	/* Concatenate two numbers: both converted to decimal strings */
	test_atom("number _ number", "Output 12 _ 34");
	/* Chained concatenation */
	test_atom("chained _ _", "Output \"a\" _ \"b\" _ \"c\"");
	/* Concatenation with infix arithmetic on each side */
	test_atom("_ with arithmetic", "Output 1 + 1 _ 3 * 3");
	/* Concatenate NIL: produces empty string (atom from empty intern) */
	test_atom("atom _ NIL", "Output \"x\" _ 0 * 0");
	/* Concatenation in a variable assignment */
	test_atom("Make with _", "Make \"S\" \"foo\" _ \"bar\" Output :S");
	/* A numeric rhs must contribute its DECIMAL TEXT and nothing else: the result is
	   the concatenation, never the number (alone or re-floated). Each case below pairs
	   an exact-text Equal with an IsStringAtom / TextLength check, because Equal alone
	   cannot tell "concatenated to a STRING" from "coerced to a NUMBER that happens to
	   compare equal" -- and a dropped lhs is exactly the bug these guard. */
	test_true("paren: string _ NUMBER", "Output ((\"x\" _ 2) = \"x2\")");
	test_true("paren: string _ NUMBER is a STRING atom", "Output IsStringAtom ( \"x\" _ 2 )");
	test_true("paren: string _ leading-zero NUMBER", "Output ((\"x\" _ 02) = \"x2\")");
	test_true("paren: string _ numeric STRING", "Output ((\"x\" _ \"2\") = \"x2\")");
	test_true("paren: string _ zero-padded STRING keeps the padding",
	          "Output ((\"x\" _ \"0002\") = \"x0002\")");
	test_num("paren: string _ zero-padded STRING length",
	         "Output TextLength ( \"x\" _ \"0002\" )", 5);
	test_true("paren: string _ fractional STRING", "Output ((\"x\" _ \"2.0\") = \"x2.0\")");
	/* A FLOAT rhs renders through the normal float formatter (Print of the bare 2.0
	   literal is "2.0"), so the concatenation is "x" + that same rendering. */
	test_true("paren: string _ FLOAT", "Output ((\"x\" _ 2.0) = \"x2.0\")");
	test_true("paren: string _ FLOAT is a STRING atom", "Output IsStringAtom ( \"x\" _ 2.0 )");
	test_true("paren: string _ negative FLOAT", "Output ((\"x\" _ -1.0) = \"x-1.0\")");
	test_true("paren: string _ zero FLOAT", "Output ((\"x\" _ 0.0) = \"x0.0\")");
	/* Three-way chain: every operand contributes, so a dropped middle or last operand
	   shows up as a text mismatch rather than as a shorter prefix. */
	test_true("paren: chained string _ NUMBER _ NUMBER",
	          "Output ((\"x\" _ 2 _ 3) = \"x23\")");
	/* A quoted numeric literal keeps its SL_TYPE_STRING type byte, so Make stores the
	   TEXT "0002" and the variable reads back with the padding intact -- it is not
	   re-parsed into NUMBER 2 on either the store or the read. */
	test_true("Make keeps a quoted numeric literal as text",
	          "Make \"x\" \"0002\" Output (:x = \"0002\")");
	test_num("Make keeps a quoted numeric literal length",
	         "Make \"x\" \"0002\" Output TextLength :x", 4);
	test_true("Make keeps a quoted numeric literal as a STRING atom",
	          "Make \"x\" \"0002\" Output IsStringAtom :x");
	/* .saf_not_float / .sef_no flags contract. Both leaves
	   used `mov edx, 1`, which is the same 5 bytes as the xor/inc pair but does NOT
	   write flags -- so they left the ZF=1 of the rejecting test/jz above, and the
	   caller's bare `jnz` (which reads ZF, never edx) read failure as success. Every
	   unconvertible lhs then compared equal to ANY float, whatever the float. */
	test_num("Equal non-numeric atom vs a matching-looking FLOAT is false",
	         "Output IfElse (\"x2\" = 2.0) [1] [0]", 0);
	test_num("Equal non-numeric atom vs an unrelated FLOAT is false",
	         "Output IfElse (\"x2\" = 99.0) [1] [0]", 0);
	test_num("Equal NIL vs FLOAT is false", "Output IfElse ([] = 2.0) [1] [0]", 0);
	test_num("Equal list vs FLOAT is false", "Output IfElse ([1 2] = 2.0) [1] [0]", 0);
	test_num("Equal non-numeric atom vs FLOAT is false (reversed order)",
	         "Output IfElse (2.0 = \"x2\") [1] [0]", 0);
	/* The control: a genuinely numeric text still equals the float it spells, so the
	   fix did not turn the coercion path off. */
	test_true("Equal numeric STRING vs FLOAT is still true", "Output (\"2.5\" = 2.5)");
}

void test_parens()
{
	printf("\n--- Parenthesized expressions ---\n");
	test_num("( 2 + 3 )", "Output ( 2 + 3 )", 5);
	test_num("( 10 + 20 )", "Output ( 10 + 20 )", 30);
}

void test_quoted_atoms()
{
	printf("\n--- Quoted atoms (double-quoted string syntax) ---\n");
	test_atom("hello", "Output \"hello\"");
	test_atom("cat", "Output \"cat\"");
	test_atom("dog", "Output \"dog\"");
}

void test_equalp_atoms()
{
	printf("\n--- Equal on atoms ---\n");
	test_true("Equal cat cat", "Output (\"cat\" = \"cat\")");
	test_nil("Equal cat dog", "Output (\"cat\" = \"dog\")");
	/* Equal atom-text SWAR compare. Equal is ZF-ONLY
	   (its header says so), so an equal exit that forgets to set ZF reports FALSE
	   for the most common case of all: two texts that fit in one 16-byte window.
	   Two DISTINCT atoms with the SAME text are what reaches this arm at all --
	   equal text interns to one atom and takes the offset fast path. AsWord is
	   how a script produces that pair: a STRING atom and the WORD atom of the
	   same text are different offsets (Substr cannot be used here: it has no
	   expression form, so `(Substr ...)` self-evaluates to the name). */
	test_true("Equal 30-byte texts, distinct atoms, window 2 boundary",
	          "Make \"s\" \"0123456789abcdefghijklmnopqrst\" Make \"w\" AsWord :s Output (:s = :w)");
	test_true("Equal 8-byte texts inside one window",
	          "Make \"s\" \"abcdefgh\" Make \"w\" AsWord :s Output (:s = :w)");
	test_nil("Equal 30-byte texts differing at the last byte",
	         "Make \"s\" \"0123456789abcdefghijklmnopqrst\" Output (:s = \"0123456789abcdefghijklmnopqrsu\")");
	test_nil("Equal 30-byte texts differing mid-window",
	         "Make \"s\" \"0123456789abcdefghijklmnopqrst\" Output (:s = \"0123456789abcdefghijklmnopXrst\")");
	/* the rhs ends BEFORE the lhs: its NUL sits at an earlier lane, so bt finds
	   nothing at the lhs's terminator lane */
	test_nil("Equal 30-byte text against its 24-byte prefix",
	         "Make \"s\" \"0123456789abcdefghijklmnopqrst\" Output (:s = \"0123456789abcdefghijklmnop\")");
	/* the mirror: the rhs runs PAST the lhs (its NUL is one lane later) */
	test_nil("Equal 30-byte text against a 31-byte extension",
	         "Make \"s\" \"0123456789abcdefghijklmnopqrst\" Output (:s = \"0123456789abcdefghijklmnopqrstt\")");
}

/* NUMBER-vs-string coercion in Equal: a numeric literal parses as NUMBER,
   but the same text from V7-DB-GET (or any string source) is a plain atom.
   Equal now parses the STRING side numerically, so 1 = "1" is TRUE while
   1 = "s" stays NIL. Guards the DB-flag bug class (AGENTS 143, historical). */
void test_equalp_num_string_coercion()
{
	printf("\n--- Equal NUMBER vs string coercion ---\n");
	test_true("1 eq \"1\"", "Output (1 = \"1\")");
	test_true("\"1\" eq 1", "Output (\"1\" = 1)");
	test_true("42 eq \"42\"", "Output (42 = \"42\")");
	test_true("-7 eq \"-7\"", "Output (-7 = \"-7\")");
	test_nil("1 eq \"s\"", "Output (1 = \"s\")");
	test_nil("1 eq \"2\"", "Output (1 = \"2\")");
	test_nil("1 eq \"11\"", "Output (1 = \"11\")");
	test_nil("1 eq \"\" ", "Output (1 = \"\")");
	test_nil("0 eq \"s\"", "Output (0 = \"s\")");
	test_true("0 eq \"0\"", "Output (0 = \"0\")");
	/* infix = and != route through the same Equal */
	test_true("1 = \"1\" infix", "Output (1 = \"1\")");
	test_nil("1 != \"1\" infix", "Output (1 != \"1\")");
	test_true("1 != \"2\" infix", "Output (1 != \"2\")");
	/* NUMBER vs NUMBER unchanged */
	test_true("5 eq 5", "Output (5 = 5)");
	test_nil("5 eq 6", "Output (5 = 6)");
}

/* DB round-trip: a value stored via V7-DB-PUT and read back via V7-DB-GET
   must compare EQUALP-equal to the identical string literal. This is the
   exact bug class from AGENTS.md section 143 (sidecar pass-2 flags): a
   DB-returned atom silently comparing unequal to its literal broke the
   memory-RAG sidecar into an INFINITE rewrite loop. */
void test_v7_db_roundtrip()
{
	printf("\n--- Database round-trip vs literals (AGENTS 143) ---\n");
	/* store "s1" under a key, read it back, compare against the literal */
	test_true("DB-GET round-trip Equal literal", "DatabaseSet \"sl_t1\" \"s1\" Output Equal DatabaseGet \"sl_t1\" \"s1\"");
	/* the exact historical failing case: flag value \"1\" vs literal \"1\" */
	test_true("DB-GET flag 1 Equal literal 1", "DatabaseSet \"sl_t2\" \"1\" Output ((DatabaseGet \"sl_t2\") = \"1\")");
	/* multi-char value round-trip */
	test_true("DB-GET multi-char Equal literal", "DatabaseSet \"sl_t3\" \"hello\" Output ((DatabaseGet \"sl_t3\") = \"hello\")");
	/* non-equal value must NOT compare equal (guard against always-true) */
	test_nil("DB-GET s1 NotEqual literal s2", "DatabaseSet \"sl_t4\" \"s1\" Output ((DatabaseGet \"sl_t4\") = \"s2\")");
	/* overwrite semantics: later PUT wins, still EQUALP-clean */
	test_true("DB-PUT overwrite round-trip", "DatabaseSet \"sl_t5\" \"aa\" DatabaseSet \"sl_t5\" \"bb\" Output ((DatabaseGet \"sl_t5\") = \"bb\")");
	/* missing key returns NIL (baseline behavior) */
	test_nil("DB-GET missing key is NIL", "Output DatabaseGet \"sl_no_such_key_at_all\"");
	/* CROSS-CALL round-trip: PUT in one interp run, EQUALP in a LATER run --
	   this is the exact sidecar shape (flag set at one on_token, tested at a
	   later on_token). If the atom table or DB view resets between runs, the
	   single-call tests above pass while this one fails. */
	run("DatabaseSet \"sl_x1\" \"s\" Output 1");                                                /* earlier call stores */
	test_true("DB-GET cross-call Equal literal", "Output ((DatabaseGet \"sl_x1\") = \"s\")"); /* later call compares */
	run("DatabaseSet \"sl_x2\" \"1\" Output 1");
	test_true("DB-GET cross-call flag 1 Equal literal 1", "Output ((DatabaseGet \"sl_x2\") = \"1\")");
	/* FIND-sentinel baseline must keep working (the shipped workaround) */
	run("DatabaseSet \"sl_x3\" \"s\" Output 1");
	test_true("DB-GET cross-call Search sentinel", "Output IsNumber Search \"s\" DatabaseGet \"sl_x3\"");
}

void test_strings()
{
	printf("\n--- Double-quoted string literals ---\n");
	/* string atoms with closing quote: "hello" interns as atom hello (quotes stripped) */
	test_atom("string hello", "Output \"hello\"");
	test_atom("string cat", "Output \"cat\"");
	/* string Equal: quotes stripped, so "cat" == cat atom */
	test_true("Equal string string", "Output (\"cat\" = \"cat\")");
	test_nil("Equal string diff", "Output (\"cat\" = \"dog\")");
	/* strings with spaces preserved inside quotes */
	test_atom("string with spaces", "Output \"hello world\"");
	/* Make with string variable name: "R" (closing quote) */
	test_num("Make string var", "Make \"R\" 42 Output :R", 42);
	/* multiple MAKE with string var names */
	test_num("Make string vars", "Make \"X\" 10 Make \"Y\" 20 Output (:X + :Y)", 30);
	/* Define with string name */
	test_num("Define string name", "Define \"inc\" [:x] [ Output :x + 1 ] Output inc 5", 6);
	/* escape sequences: \" inside string */
	test_atom("escaped quote", "Output \"a\\\"b\"");
	/* "a\"b" must be exactly 3 bytes: a, ", b -- index 3 is past the end,
	   so Part returns "" only when the escape collapsed to ONE byte */
	test_true("escaped quote is ONE byte", "Output ((Part \"a\\\"b\" 3 1) = \"\")");
	/* escape sequences: \\ inside string */
	test_atom("escaped backslash", "Output \"a\\\\b\"");
	/* "a\\b" must be exactly 3 bytes: a, \, b */
	test_true("escaped backslash is ONE byte", "Output ((Part \"a\\\\b\" 3 1) = \"\")");
	/* escape sequences: \n becomes a real LF byte.
	   Ground truth: a RAW LF byte embedded directly in the script source --
	   the lexer stores raw bytes inside strings verbatim (no escape
	   machinery), so EQUALP succeeds only if \n produced that exact byte. */
	test_true("escaped newline == raw LF byte", "Output (\"a\\nb\" = \"a\nb\")");
	/* byte 1 of "a\nb" must be the raw LF byte */
	test_true("Part byte 1 of a\\nb is LF", "Output ((Part \"a\\nb\" 1 1) = \"\n\")");
	/* FIND with a raw-LF needle must locate index 1 */
	test_num("Search raw-LF in a\\nb -> 1", "Output Search \"\n\" \"a\\nb\"", 1);
	/* negative checks: \n must NOT survive as two literal chars or drop to 'n' */
	test_nil("\\n Not literal backslash+n", "Output (\"a\\nb\" = \"a\\\\nb\")");
	test_nil("\\n Not dropped to plain n", "Output (\"a\\nb\" = \"anb\")");
	/* needle in an EMPTY haystack: nothing to find -> NIL -> IsNumber is false
	   (the flag parser reads NIL as falsy, not as the NUMBER 0 falsely-truthy
	   found-index, and even '0' at position 0 is IsNumber-true with a value of
	   0, so the empty-haystack case must return NIL and stay false) */
	test_nil("Search 1 in empty string -> NIL", "Output Search \"1\" \"\"");
	test_num("IsNumber Search 1 in empty string -> false", "Output IfElse IsNumber (Search \"1\" \"\") [1] [0]", 0);
}

void test_def_fn()
{
	printf("\n--- Define / TO / Output ---\n");
	test_num("Define add1", "Define \"add1\" [:x] [ Output :x + 1 ] Output add1 5", 6);
	test_num("Define double", "Define \"double\" [:n] [ Output :n * 2 ] Output double 7", 14);
	test_num("Define square", "Define \"square\" [:x] [ Output Power :x 2 ] Output square 9", 81);
	test_num("Define via TO keyword", "Define \"triple\" [:x] [ Output :x * 3 ] Output triple 4", 12);
}

/* Global Define binding (DEF_BIND_TAIL): a Define executed inside a user
   function must register the new procedure in the shared base env so it is
   callable after the installer returns. The pre-sentinel engine bound it
   into the caller's frame env, which died at the fn boundary -- the nested
   Define self-evaluated as a dead name afterwards (the wrapper-installer
   pattern was silently broken). Also covers update-in-place on redefine. */
void test_define_global_bind()
{
	printf("\n--- Global Define binding (nested Define) ---\n");
	/* the motivating case: installer fn Defines a nested fn, which must be
	   callable after the installer returned */
	test_num("nested Define survives the installer", "Define \"install\" [] [ Define \"wrapped\" [] [ Output 42 ] ] install Output wrapped", 42);
	/* nested Define with parameters */
	test_num("nested Define with params", "Define \"mk\" [] [ Define \"add1\" [:x] [ Output :x + 1 ] ] mk Output add1 5", 6);
	/* top-level Define is still callable immediately */
	test_num("top-level Define callable", "Define \"top\" [] [ Output 7 ] Output top", 7);
	/* redefine updates the existing binding instead of shadowing */
	test_num("redefine updates in place", "Define \"f\" [] [ Output 1 ] Define \"f\" [] [ Output 2 ] Output f", 2);
	/* Define inside a REPEAT: the last definition wins, chain does not grow */
	test_num("Define inside Repeat", "Repeat 2 [ Define \"loopy\" [] [ Output 9 ] ] Output loopy", 9);
	/* two installers, two nested fns: both visible afterwards */
	test_num("two installers both bind", "Define \"i1\" [] [ Define \"a1\" [] [ Output 10 ] ] Define \"i2\" [] [ Define \"b1\" [] [ Output 20 ] ] i1 i2 Output (a1 + b1)", 30);
}

/* RAG API stdlib (autoexec buckets). The validator does not load autoexec,
   so every script installs the bucket it needs first -- that is itself part
   of the contract (DefineRagFunctions / DefinePureLogoBM25 are plain Defines,
   callable from any script). Suite 97 covers the pure-Logo BM25 pipeline
   end-to-end (init, ranking, floor suppression, MaxLines) and the rag exe
   wrappers (RagSearch vs a real corpus file, RagRemember + RagSearch
   roundtrip on a scratch corpus). */
void test_rag_api()
{
	/* Make inside a fn cannot CREATE a global (the binding dies with the fn
	   frame; only an existing outer binding is updated in place) -- the
	   persistent BM25 config must exist at the script's top level before
	   BM25Init/BM25Set* run. autoexec creates these at its install tail;
	   validator scripts create them here, because they do not load autoexec. */
	const char *init_bm25 =
		"Make \"bm25corpus\" \"\" Make \"bm25sw\" \" \" Make \"bm25k1\" 1.2 "
		"Make \"bm25b\" 0.75 Make \"bm25dens\" 0.026 Make \"bm25topmin\" 3.0 "
		"Make \"bm25maxlines\" 3 DefinePureLogoBM25 ";
	const char *corpus = "Cats eat grass. Cats are strict herbivores that graze all day.\\n\\nDogs chase balls. Dogs are playful animals that fetch sticks.";

	printf("\n--- RAG API (autoexec wrapper buckets) ---\n");
	/* BM25Init stages a corpus and resets the calibrated defaults */
	{
		char script[768];
		snprintf(script, sizeof(script), "%sOutput BM25Init \"%s\"", init_bm25, corpus);
		test_num("BM25Init stages the corpus", script, 1);
	}
	/* ranking: the herbivore query hits the cats sentence */
	{
		char script[896];
		snprintf(script, sizeof(script),
		         "%sMake \"c\" \"%s\" BM25Init :c Output IfElse IsNumber Search \"herbivores\" ( BM25Search \"what do herbivorous cats eat\" ) [1] [0]",
		         init_bm25, corpus);
		test_num("BM25Search ranks the matching sentence", script, 1);
	}
	/* dual floor: a query sharing no terms with the corpus suppresses.
	   Equal returns the truthy ATOM, not NUMBER 1 (Sec.146 fact 38) -- wrap
	   in IfElse for test_num. */
	{
		char script[896];
		snprintf(script, sizeof(script),
		         "%sMake \"c\" \"%s\" BM25Init :c Output IfElse ((BM25Search \"qqzzxx vvbbnn\") = \"\") [1] [0]",
		         init_bm25, corpus);
		test_num("BM25Search floor suppresses an unrelated query", script, 1);
	}
	/* BM25SetMaxLines: a query hitting BOTH paragraphs returns one sentence */
	{
		char script[1024];
		snprintf(script, sizeof(script),
		         "%sMake \"c\" \"%s\" BM25Init :c BM25SetMaxLines 1 Output IfElse IsNumber Search \"\\n\" ( BM25Search \"cats and dogs\" ) [0] [1]",
		         init_bm25, corpus);
		test_num("BM25SetMaxLines caps the answer", script, 1);
	}
	/* DefineRagFunctions: RagSearch e2e against a real corpus file (spawns
	   bm25_rag.exe via the wrapper; on POSIX the spawn path strips the .exe
	   suffix, where the helper builds as examples/rag/bm25_rag) */
	test_num("RagSearch e2e finds a stored fact",
	         "DefineRagFunctions Output IfElse IsNumber Search \"ruges\" ( RagSearch \"bruges\" \"base=examples/rag/memory/user_facts.txt;maxlines=1\" ) [1] [0]", 1);
	/* RagRemember + RagSearch roundtrip on a scratch corpus under build_logs */
	test_num("RagRemember stores, RagSearch finds",
	         "DefineRagFunctions Make \"w\" RagRemember \"validator temp fact zqj\" \"base=build_logs/validator_rag_corpus.txt\" Output IfElse IsNumber Search \"zqj\" ( RagSearch \"zqj\" \"base=build_logs/validator_rag_corpus.txt;recent=1\" ) [1] [0]", 1);
}

/* LOCAL declares a function-scope variable: MAKE updates it in-place
   (O(1) in loops, no dead binding pile-up), outer bindings are restored
   on function return, non-LOCAL MAKE still shadows via prepend. */
void test_local()
{
	printf("\n--- Local (function-scope variables) ---\n");
	// repeated MAKE on a LOCAL var updates in-place -> final value wins
	test_num("Local repeated Make", "Define \"t1\" [] [ Local \"x\" Make \"x\" 1 Make \"x\" 2 Make \"x\" 3 Output :x ] Output t1", 3);
	// LOCAL var does not leak: outer y unchanged after fn sets local y=20
	test_num("Local no leak to outer", "Make \"y\" 10 Define \"t2\" [] [ Local \"y\" Make \"y\" 20 ] t2 Output :y", 10);
	// LOCAL shadows outer inside fn; outer restored after return (200+100)
	test_num("Local shadows outer", "Make \"z\" 100 Define \"t3\" [] [ Local \"z\" Make \"z\" 200 Output :z ] Output (t3 + :z)", 300);
	// the motivating pattern: loop counter via LOCAL + repeated MAKE
	test_num("Local counter pattern", "Define \"counter\" [:n] [ Local \"total\" Make \"total\" 0 Repeat :n [ Make \"total\" :total + 1 ] Output :total ] Output counter 100", 100);
	// LOCAL of a never-existing outer variable works standalone
	test_num("Local without outer binding", "Define \"t4\" [] [ Local \"w\" Make \"w\" 42 Output :w ] Output t4", 42);
	// non-LOCAL MAKE keeps classic shadowing semantics
	test_num("non-Local Make still prepends", "Make \"a\" 1 Make \"a\" 2 Output :a", 2);
	/* a bare [] statement head is inert data (NIL form): it must not resolve a
	   tombstoned LOCAL binding (Assoc key 0 matches the tombstone car 0) and
	   must not misdispatch -- the script simply continues. Before the
	   .stmt_fallback guard, the [] head fell into dispatch slot 0 and reached
	   Assoc with key 0, resurrecting the newest tombstoned LOCAL's value. */
	test_num("bare [] statement is inert", "[] Output 5", 5);
	test_num("bare [] after a tombstoned local stays inert",
	         "Make \"y\" 10 Define \"t2\" [] [ Local \"y\" Make \"y\" 20 ] t2 [] Output 5", 5);
	// multi-name Local declares EVERY name in the list (bm25 bug: only the first was declared)
	test_num("Local multi-name declares all", "Define \"t5\" [] [ Local \"p\" \"q\" \"r\" Make \"p\" 1 Make \"q\" 2 Make \"r\" 3 Output ((:p + :q) + :r) ] Output t5", 6);
	// multi-name Local: none of the names leak to the caller
	test_num("Local multi-name no leak", "Make \"q\" 10 Make \"s\" 4 Define \"t6\" [] [ Local \"q\" \"s\" Make \"q\" 20 Make \"s\" 5 ] t6 Output (:q + :s)", 14);
	// nested-fn clobber scenario (the herbivorous_bm25 crash): inner fn Local-declares
	// the same name the outer fn uses; inner writes must NOT touch outer's binding
	test_num("Local nested same-name clobber", "Define \"inner\" [] [ Local \"n\" Make \"n\" 999 Output :n ] Define \"outer\" [] [ Local \"n\" Make \"n\" 42 Make \"dummy\" inner Output :n ] Output outer", 42);
	// The count-restore side of sl_tombstone_locals: before the fix the restoring
	// store read ecx, which the tombstone walk had walked up to the current count,
	// so it was a no-op and the LOCAL set grew by one per block evaluation. Wrapped
	// in DEF_FN this always passed, because a fn return restores the count itself
	// (sectorlogo_fn.inc) -- only a bare top-level block exercises the macro's own
	// restore. Make is a registered builtin, so the name-list peek ends the list there.
	// NOTE: IFELSE takes TWO blocks; a one-armed call silently consumes the next
	// stream form as the missing else, which reads like a binding bug but is not one.
	// Uses IF (one-armed) for the leak case so no else-block can be miscounted.
	test_num("Local in a top-level block does not leak",
	         "Make \"x\" 1 If True [ Local \"x\" Make \"x\" 99 ] Output IfElse (:x = 1) [1] [0]", 1);
	// the write side: after the block a top-level MAKE must still reach the live
	// outer binding, not a dead inner cell
	test_num("Make after a top-level Local block updates the outer binding",
	         "Make \"y\" 1 If True [ Local \"y\" Make \"y\" 99 ] Make \"y\" 7 Output :y", 7);
	// an UNRELATED binding must survive the block too -- the strongest form, since a
	// shadowing bug on the redeclared name would not explain losing this one
	test_num("Unrelated binding survives a top-level Local block",
	         "Make \"p\" 5 Make \"q\" 6 If True [ Local \"q\" 1 ] Output :p", 5);
	test_num("Repeated top-level Local blocks do not accumulate",
	         "Make \"z\" 5 Repeat 200 [ IfElse True [ Local \"z\" Make \"z\" 1 ] ] Output :z", 5);
	// infix operator atoms END the name list (B25): Local "x" _ ... declares x
	// only, then the "_" concat expression runs as the next statement -- before
	// the op-bitmap check, the _ was eaten as a NAME and .lg_validate_name FATAL'd
	// (Equal alone returns the truthy atom, so wrap in IfElse for test_num)
	test_num("Local name list ends at infix _", "Define \"t7\" [] [ Local \"x\" _ \"a\" \"b\" Make \"x\" \"ok\" Output :x ] Output IfElse ((t7) = \"ok\") [1] [0]", 1);
	// multi-name Local followed by a list statement: the peek must stop at the [ body ]
	test_num("Local multi-name then body", "Define \"t7\" [] [ Local \"u\" \"v\" Make \"u\" 7 Make \"v\" 8 Repeat 1 [ Make \"u\" :u + :v ] Output :u ] Output t7", 15);
	// Bug #9 fix: Make inside fn without LOCAL updates outer-scope binding
	test_num("Make in fn updates outer", "Make \"x\" 10 Define \"bump\" [] [ Make \"x\" 20 ] bump Output :x", 20);
	// nested: inner fn updates a global that the outer fn also sees
	test_num("Make in inner fn updates global", "Make \"g\" 1 Define \"inner\" [] [ Make \"g\" 99 ] Define \"outer\" [] [ inner Output :g ] Output outer", 99);
	// Make in fn creates new binding when no outer exists
	test_num("Make in fn creates if no outer", "Define \"creator\" [] [ Make \"fresh\" 7 Output :fresh ] Output creator", 7);
	// Make in fn with LOCAL still shadows (LOCAL path takes priority)
	test_num("Local shadows even with env walk", "Make \"v\" 10 Define \"t8\" [] [ Local \"v\" Make \"v\" 30 Output :v ] Output (t8 + :v)", 40);
	// counter pattern without LOCAL: Make should update outer accumulator
	test_num("Make in fn accumulates outer", "Make \"acc\" 0 Define \"add\" [:n] [ Make \"acc\" :acc + :n ] add 5 add 3 add 2 Output :acc", 10);
}

/* ====== SORT / REVERSE / HASH-FLOAT / LENGTH / LSET / LFIND TESTS ====== */

void test_sort_reverse()
{
	printf("\n--- Sort / Reverse ---\n");
	/* .accum_done lookahead hoist: sl_g_look is now
	   published ONCE per token at .done instead of once per consumed character.
	   A broken lookahead has a distinctive signature -- the same token re-lexed
	   until sl_parse_depth trips the "parser nesting too deep" FATAL -- so these
	   three cases exercise the word-token path end to end. */
	test_num("Lexer lookahead: multi-character word token",
	         "Make \"L\" Prepend \"hello\" Prepend \"x\" [] Output Length :L", 2);
	test_num("Lexer lookahead: single-character word token",
	         "Make \"L\" Prepend \"x\" Prepend \"yy\" [] Output Length :L", 2);
	/* sign-glued token: covers .check_sign_digit, the second of the two edx
	   conversions in .accum_done and the one whose dl reuse is least obvious */
	test_num("Lexer lookahead: sign-glued token in a list",
	         "Output Item [-3 5 -10 0] 1", 5);
	/* Item's sign pre-filter is STRICTER than Sort's coerce grammar by design:
	   a sign-led atom without a digit stays the raw atom (unlike [.5], which
	   the no-sign path coerces). Pin both facts. */
	test_float_val("Item bare-dot float [0]", "Output Item [.5] 0", 0.5f);
	test_true("Item minus-dot atom stays the raw atom",
	          "Make \"x\" Item [-.5] 0 Output (Equal \"-.5\" :x)");
	test_true("Item plus-dot atom stays the raw atom",
	          "Make \"x\" Item [+.5] 0 Output (Equal \"+.5\" :x)");
	test_true("Item bare minus-dot still rejects to the atom",
	          "Make \"x\" Item [-.] 0 Output (Equal \"-.\" :x)");
	// integers, incl. duplicates: probe all 8 positions
	test_num("Sort dup ints [0]", "Output Item Sort [3 1 4 1 5 9 2 6] 0", 1);
	test_num("Sort dup ints [1]", "Output Item Sort [3 1 4 1 5 9 2 6] 1", 1);
	test_num("Sort dup ints [2]", "Output Item Sort [3 1 4 1 5 9 2 6] 2", 2);
	test_num("Sort dup ints [7]", "Output Item Sort [3 1 4 1 5 9 2 6] 7", 9);
	// negatives exercise the sign-bit key transform
	test_num("Sort negatives [0]", "Output Item Sort [-3 5 -10 0] 0", -10);
	test_num("Sort negatives [3]", "Output Item Sort [-3 5 -10 0] 3", 5);
	// floats
	test_float_val("Sort floats [0]", "Output Item Sort [3.5 1.25 2.5] 0", 1.25f);
	test_float_val("Sort floats [2]", "Output Item Sort [3.5 1.25 2.5] 2", 3.5f);
	// mixed NUMBER/FLOAT: order promoted, NUMBER elements stay NUMBER
	test_float_val("Sort mixed [0]", "Output Item Sort [3 1.5 2] 0", 1.5f);
	test_num("Sort mixed [2]", "Output Item Sort [3 1.5 2] 2", 3);
	// an over-32-bit trailing-dot atom ("N." stays an atom in the reader
	// but sl_sort_coerce_num's own trailing-dot rule accepts it) must yield the
	// correctly-rounded float -- before the fix MakeFloat read the low 32 bits of
	// a binary64 (denormal near zero). f32 rounds 5368709134 to the nearest
	// representable multiple of 512 at this magnitude: 5368709120.
	test_float_val("Sort over-32-bit trailing-dot atom [1]",
	               "Output Item Sort [5368709134. 1] 1", 5368709120.0f);
	// degenerate shapes
	test_num("Sort singleton", "Output Item Sort [42] 0", 42);
	test_nil("Sort empty", "Output Sort []");
	// sortedness via pairwise comparisons over neighbors (REPEAT walk)
	test_true("Sort fully ordered", "Make \"L\" Sort [9 2 8 1 7 3 6 4 5] "
	                                "Make \"OK\" 1 "
	                                "Repeat 8 [ IfElse ((First :L) < (First ButFirst :L)) [] [ Make \"OK\" 0 ] "
	                                "  Make \"L\" ButFirst :L ] "
	                                "Output :OK");
	// REVERSE
	test_num("Reverse [0]", "Output Item Reverse [1 2 3] 0", 3);
	test_num("Reverse [2]", "Output Item Reverse [1 2 3] 2", 1);
	test_nil("Reverse empty", "Output Reverse []");
	// Reverse must stop at a non-CONS tail exactly as Length and Sort do.
	// SetCdr ButFirst :L 9 builds the improper list (1 2 . 9): the cdr of the
	// SECOND cell is the one that gets replaced. Before the fix, Reverse spliced
	// a fabricated NIL and returned (NIL 2 1), so element 0 was NIL, not 2.
	test_num("Reverse stops at improper tail, element 0",
	         "Make \"L\" Prepend 1 Prepend 2 [] SetCdr ButFirst :L 9 Output Item Reverse :L 0",
	         2);
	test_num("Reverse stops at improper tail, length",
	         "Make \"L\" Prepend 1 Prepend 2 [] SetCdr ButFirst :L 9 Output Length Reverse :L",
	         2);
	// descending composition
	test_num("Reverse Sort [0]", "Output Item Reverse Sort [1 2 3] 0", 3);
	// GC stress: 300 Prepend cells (forces GC cycles), then sort + probe min
	test_num("Sort after 300 Prepend (GC)",
	         "Make \"L\" 0 "
	         "Make \"I\" 0 "
	         "Repeat 300 [ Make \"I\" :I + 1 Make \"L\" Prepend :I :L ] "
	         "Make \"L\" Sort :L "
	         "Output Item :L 0",
	         1);
	// sub-lists sorted by first element (key-value pairs)
	test_num("Sort key-value [0]", "Output TextToNumber First Item Sort [[3 \"b\"] [1 \"a\"] [2 \"c\"]] 0", 1);
	test_num("Sort key-value [2]", "Output TextToNumber First Item Sort [[3 \"b\"] [1 \"a\"] [2 \"c\"]] 2", 3);
	/* RS_INSERT_MAX dispatch (stable insertion path for
	   N <= 64 vs radix above): stability with duplicate sub-list keys, float and
	   negative keys through the insertion path, and both sides of the N=64/65
	   boundary -- built by Prepend so the input order is exactly the given one. */
	test_true("Sort stable duplicate keys keep input order",
	          "Output ((Item Sort [[1 111] [0 5] [1 222] [0 6]] 2) = [1 111])");
	test_true("Sort stable duplicate keys second occurrence",
	          "Output ((Item Sort [[1 111] [0 5] [1 222] [0 6]] 3) = [1 222])");
	test_true("Sort insertion float keys ordered",
	          "Make \"L\" Sort [2.5 0.5 1.5] "
	          "Make \"OK\" 1 "
	          "Repeat 2 [ IfElse ((First :L) < (First ButFirst :L)) [] [ Make \"OK\" 0 ] Make \"L\" ButFirst :L ] "
	          "Output :OK");
	test_float_val("Sort insertion negative key", "Output Item Sort [-3.5 2.5 -1.0] 0", -3.5f);
	test_num("Sort N=64 boundary min (insertion path)",
	         "Make \"L\" 0 "
	         "Make \"I\" 65 "
	         "Repeat 64 [ Make \"I\" :I - 1 Make \"L\" Prepend :I :L ] "
	         "Output Item Sort :L 0",
	         1);
	test_num("Sort N=65 boundary min (radix path)",
	         "Make \"L\" 0 "
	         "Make \"I\" 66 "
	         "Repeat 65 [ Make \"I\" :I - 1 Make \"L\" Prepend :I :L ] "
	         "Output Item Sort :L 0",
	         1);
	test_num("Sort N=64 boundary max (insertion path)",
	         "Make \"L\" 0 "
	         "Make \"I\" 65 "
	         "Repeat 64 [ Make \"I\" :I - 1 Make \"L\" Prepend :I :L ] "
	         "Output Item Sort :L 63",
	         64);
	test_num("Sort N=2 smallest insertion case", "Output Item Sort [2 1] 0", 1);
	// mixed bare numbers + sub-lists: bare numbers use themselves as key
	test_num("Sort mixed bare+sublist [0]", "Output TextToNumber First Item Sort [2 [3 23 11] [1 \"hello\"] [4]] 0", 1);
	// bare number (key=2) lands between the 1-key and 3-key sub-lists
	test_num("Sort mixed bare+sublist [1]", "Output Item Sort [2 [3 23 11] [1 \"hello\"] [4]] 1", 2);
	// sub-list retained: inner [3 23 11] survives the sort (value rides the record)
	test_num("Sort sub-list intact",
	         "Make \"L\" Sort [2 [3 23 11] 1] "
	         "Output Item Item :L 2 2",
	         11);
	// FATAL cases (covered by test_fatal_catch):
	//   Output Sort ["hello" 3]     -> FATAL non-numeric, non-list element
	//   Output Sort [["bad" 1] [3]] -> FATAL sub-list first element not numeric
	// 30-bit payload boundary regressions (parser threshold at 53687091):
	// atom-coerced ints >= 2^29 promote to FLOAT instead of silently wrapping
	// negative in MakeNum's payload mask. SORT must order them as ~1e9 (largest),
	// not as negative. The atoms enter the list via Prepend (args ARE evaluated);
	// list literals are data -- [:S ...] would store the unparseable atom ":S".
	test_float_val("Sort 1e9 atom largest",
	         "Make \"S\" \"1000000000\" "
	         "Output Item Sort Prepend :S Prepend -5 Prepend 3 [] 2",
	         1000000000.0f);
	test_num("Sort 1e9 not negative",
	         "Make \"S\" \"1000000000\" "
	         "Output Item Sort Prepend :S Prepend -5 Prepend 3 [] 0",
	         -5);
	// overflow-fraction regression (Sep 2026): the old overflow path multiplied
	// by 10 per FRACTION digit -- "1000000000.5" parsed as 1e10 (10x too large).
	// The fixed path shares the integer-accurate fraction parser: 1000000000.5.
	test_float_val("Sort big-int.frac exact",
	         "Make \"S\" \"1000000000.5\" "
	         "Output Item Sort Prepend :S Prepend -5 Prepend 3 [] 2",
	         1000000000.5f);
	// same bug class, ordering probe: 1000000000.5 must sit ABOVE 999999999
	// (the element itself is a FLOAT -- big-int.frac parses to a tagged FLOAT)
	test_float_val("Sort big-int.frac above 999999999",
	         "Make \"S\" \"1000000000.5\" "
	         "Make \"T\" \"999999999\" "
	         "Output Item Sort Prepend :T Prepend :S [] 1",
	         1000000000.5f);
	// 2^63 overflow wall (Sep 2026): the guard cmp rax, 922337203685477579
	// cannot encode as imm32 -- NASM truncated it ("signed dword exceeds
	// bounds" warning), the exact-accumulator wall never fired, and digits
	// past 2^63 wrapped negative. A 23-digit atom must coerce to FLOAT
	// ~1.23e22 and sort ABOVE 1e18 (which is still inside the exact range).
	test_float_val("Sort 23-digit atom above 1e18",
	         "Make \"BIG\" \"12345678901234567890123\" "
	         "Make \"S\" \"1000000000000000000\" "
	         "Output Item Sort Prepend :S Prepend :BIG [] 1",
	         1.2345679e22f);
	// bit-exact: the >2^63 digit tail must accumulate at binary64 width and narrow
	// once, so the result is the NEAREST float32 to the true 23-digit value. The
	// binary32 per-digit chain it replaced landed 1-2 ulp away, which the 1e-5
	// relative tolerance of test_float_val cannot see.
	test_float_bits("Sort 23-digit atom is the nearest binary32",
	         "Make \"BIG\" \"12345678901234567890123\" "
	         "Make \"S\" \"1000000000000000000\" "
	         "Output Item Sort Prepend :S Prepend :BIG [] 1",
	         12345678901234567890123.0);
	// boundary: 536870911 (2^29-1) is the last exact NUMBER; via atom coercion it
	// must still round-trip (promotes to float 536870912.0f, +1 ulp, the only
	// representation the 30-bit type system offers) and order above 1
	test_num("FindFirstAtLeast 2^29 boundary",
	         "Make \"S\" \"536870911\" "
	         "Output FindFirstAtLeast Sort Prepend 1 Prepend :S [] 0 2",
	         1);
	// large atom must not compare below small ints (the wrap-to-negative bug)
	test_num("FindFirstAtLeast 1e9 above 1",
	         "Make \"S\" \"1000000000\" "
	         "Output FindFirstAtLeast Sort Prepend 1 Prepend :S [] 0 2",
	         1);
	//   OUTPUT SORT 42              -> FATAL not a list (test_fatal_catch)
	//   OUTPUT REVERSE 42           -> FATAL not a list (test_fatal_catch)
}

void test_hash_float()
{
	printf("\n--- HashNumber ---\n");
	// exact value: CRC32C with the same instruction, then the ASM's exact packing:
	// (crc >> 8) as a 24-bit integer (exact in f32) multiplied by 2^-24
	unsigned crc = 0;
	for (const char *p = "hello"; *p; ++p)
		crc = _mm_crc32_u8(crc, (unsigned char)*p);
	float expect = (float)(crc >> 8) / 16777216.0f;
	test_float_val("HashNumber hello exact", "Output HashNumber \"hello\"", expect);
	// determinism: two calls on same string produce identical results
	test_true("HashNumber same string", "Make \"A\" HashNumber \"test\" Make \"B\" HashNumber \"test\" Output IfElse (:A = :B) [1] [0]");
	// range: result is in [0, 1)
	test_true("HashNumber in [0,1)", "Make \"H\" HashNumber \"anything\" Output IfElse (:H < 1.0) [1] [0]");
	// empty string: CRC32 of "" = 0, so hash = 0.0
	test_float_val("HashNumber empty", "Output HashNumber \"\"", 0.0f);
	// long-atom coverage (>=16 chars crosses the 16-byte SSE chunk boundary: the
	// length scan must add the chunk base to the within-chunk NUL offset -- the
	// tzcnt chunk-base bug hashed only a 0-15 byte prefix for >=16-char atoms)
	unsigned crc20 = 0;
	for (const char *p = "abcdefghijklmnopqrst"; *p; ++p)
		crc20 = _mm_crc32_u8(crc20, (unsigned char)*p);
	float expect20 = (float)(crc20 >> 8) / 16777216.0f;
	test_float_val("HashNumber 20-char exact", "Output HashNumber \"abcdefghijklmnopqrst\"", expect20);
	// 16-char atom: NUL lands at chunk 1 offset 0 -- the old bug hashed ZERO bytes
	unsigned crc16 = 0;
	for (const char *p = "0123456789abcdef"; *p; ++p)
		crc16 = _mm_crc32_u8(crc16, (unsigned char)*p);
	float expect16 = (float)(crc16 >> 8) / 16777216.0f;
	test_float_val("HashNumber 16-char exact", "Output HashNumber \"0123456789abcdef\"", expect16);
	// 16-char vs 20-char must hash differently (both were 0.0 under the bug)
	test_true("HashNumber 16 != 20 chars",
	          "Make \"A\" HashNumber \"0123456789abcdef\" Make \"B\" HashNumber \"abcdefghijklmnopqrst\" Output IfElse (:A = :B) [0] [1]");
	// FATAL cases (covered by test_fatal_catch):
	//   OUTPUT HASH-FLOAT 42        -> FATAL not an atom
	//   OUTPUT HASH-FLOAT [1 2 3]   -> FATAL not an atom
	// A bare WORD atom carries neither the NUMBER (bit30) nor the CONS/FLOAT (bit31) tag,
	// so it is NOT rejected by the guards and is hashed exactly like a string -- the FATAL
	// text states that, and this pins it (bare unquoted word, not a quoted literal).
	test_true("HashNumber accepts a bare WORD atom and agrees with the string",
	          "Make \"W\" HashNumber cat Make \"S\" HashNumber \"cat\" Output Equal :W :S");
	test_true("HashNumber on a WORD atom equals the string form",
	          "Make \"W\" HashNumber AsWord \"cat\" Output Equal :W HashNumber \"cat\"");
}

void test_length()
{
	printf("\n--- Length ---\n");
	test_num("Length [1 2 3]", "Output Length [1 2 3]", 3);
	test_num("Length []", "Output Length []", 0);
	test_num("Length [5]", "Output Length [5]", 1);
	test_num("Length long", "Make \"L\" [1 2 3 4 5 6 7 8 9 10] Output Length :L", 10);
	// LENGTH composes with SORT and EXPLODE
	test_num("Length Sort", "Output Length Sort [3 1 2]", 3);
	test_num("Length Split", "Output Length Split \"a,b,c\" \",\"", 3);
	// FATAL cases (manual):
	//   OUTPUT LENGTH 42            -> FATAL not a list
	//   OUTPUT LENGTH \"hello\"      -> FATAL not a list
}

void test_lset()
{
	printf("\n--- SetItem ---\n");
	// basic replacement at each position
	test_num("SetItem [0]", "Output Item SetItem [1 2 3] 0 99 0", 99);
	test_num("SetItem [1]", "Output Item SetItem [1 2 3] 1 99 1", 99);
	test_num("SetItem [2]", "Output Item SetItem [1 2 3] 2 99 2", 99);
	// non-replaced elements survive
	test_num("SetItem non-target [0]", "Output Item SetItem [1 2 3] 2 99 0", 1);
	test_num("SetItem non-target [1]", "Output Item SetItem [1 2 3] 2 99 1", 2);
	// returns a NEW list: original unchanged
	test_num("SetItem original intact", "Make \"L\" [1 2 3] Make \"L2\" SetItem :L 0 99 Output Item :L 0", 1);
	// sub-list replacement (hash-table bucket update)
	test_true("SetItem sub-list key-value", "Make \"H\" [[0.3 \"A\"] [0.7 \"B\"]] "
	                                     "Make \"H\" SetItem :H 1 [0.7 \"C\"] "
	                                     "Output ((Item Item :H 1 1) = \"C\")");
	// FATAL cases (manual):
	//   OUTPUT LSET [1 2 3] 3 99    -> FATAL index out of bounds
	//   OUTPUT LSET [1 2 3] -1 99   -> FATAL negative index
	//   OUTPUT LSET 42 0 99         -> FATAL not a list
	//   OUTPUT LSET [] 0 99         -> FATAL index out of bounds
	// improper tail. SetCdr on the SECOND cell splices a non-CONS tail
	// onto (1 2), giving (1 2 . 3) -- SetCdr on the HEAD would replace the whole
	// tail and give the unrelated (1 . 3). The walk must stop at any non-CONS
	// cursor, exactly like sl_sort / sl_reverse / sl_length do -- before the fix
	// it tested NIL only, so Car's NIL-on-non-CONS was appended as a real third
	// element and the script's tail value 3 was silently discarded.
	test_num("SetItem stops at an improper non-CONS tail",
	         "Make \"l\" [1 2] SetCdr ButFirst :l 3 Make \"r\" SetItem :l 0 9 Output Length :r", 2);
	test_num("SetItem improper tail keeps the head replacement",
	         "Make \"l\" [1 2] SetCdr ButFirst :l 3 Make \"r\" SetItem :l 0 9 Output Item :r 0", 9);
	test_num("SetItem improper tail keeps the last real element",
	         "Make \"l\" [1 2] SetCdr ButFirst :l 3 Make \"r\" SetItem :l 0 9 Output Item ButFirst :r 0", 2);
	// index past the last real cell is out of bounds (the list's length is 2),
	// not a substitution into a fabricated element
	test_fatal_catch_case("SetItem index past an improper tail is out of bounds",
	                      "Make \"l\" [1 2] SetCdr ButFirst :l 3 Output SetItem :l 2 9",
	                      "index out of bounds");
	// the sibling walkers already agreed on this list, so the fix makes SetItem
	// agree with them rather than inventing a third answer
	test_num("Length agrees on the improper tail",
	         "Make \"l\" [1 2] SetCdr ButFirst :l 3 Output Length :l", 2);
}

void test_setcar_setcdr()
{
	printf("\n--- SetCar / SetCdr ---\n");
	// basic destructive car write, observed through the SAME holder
	test_num("SetCar basic", "Make \"L\" Prepend 10 Prepend 20 Prepend 30 [] SetCar :L 99 Output Item :L 0", 99);
	// non-target cells untouched
	test_num("SetCar leaves siblings", "Make \"L\" Prepend 10 Prepend 20 Prepend 30 [] SetCar :L 99 Output Item :L 1", 20);
	// destructive cdr write truncates the list (cdr -> NIL)
	test_num("SetCdr truncate", "Make \"L\" Prepend 1 Prepend 2 Prepend 3 [] SetCdr :L [] Output Length :L", 1);
	// sibling observability: the tail is SHARED -- a write through the alias is visible through the original
	test_num("SetCar shared tail", "Make \"L\" Prepend 1 Prepend 2 [] Make \"T\" ButFirst :L SetCar :T 42 Output Item :L 1", 42);
	// SetCdr splice: ButFirst :L is the SECOND cell -- writing its cdr APPENDS a new tail
	test_num("SetCdr splice appends", "Make \"L\" Prepend 1 Prepend 2 [] SetCdr ButFirst :L Prepend 9 [] Output Length :L", 3);
	test_num("SetCdr splice tail value", "Make \"L\" Prepend 1 Prepend 2 [] SetCdr ButFirst :L Prepend 9 [] Output Item :L 2", 9);
	// SetCdr on the HEAD cell replaces the whole tail
	test_num("SetCdr replace tail", "Make \"L\" Prepend 1 Prepend 2 [] SetCdr :L Prepend 9 [] Output Length :L", 2);
	test_num("SetCdr replace tail value", "Make \"L\" Prepend 1 Prepend 2 [] SetCdr :L Prepend 9 [] Output Item :L 1", 9);
	// the new value may be any tagged value, including a list
	test_num("SetCar list value", "Make \"L\" Prepend 1 Prepend 2 [] SetCar ButFirst :L Prepend 7 [] Output Item Item :L 1 0", 7);
	// the value evaluation allocates (GC runs) while the cell is staged -- must stay rooted
	test_num("SetCar across allocation", "Make \"C\" Prepend 1 [] SetCar :C Prepend 2 Prepend 3 [] Output Item Item :C 0 0", 2);
	// return value is NIL (Scheme unspecified)
	test_nil("SetCar returns NIL", "Output SetCar Prepend 1 [] 5");
	test_nil("SetCdr returns NIL", "Output SetCdr Prepend 1 [] 5");
	// case-folded lowercase spelling resolves to the same builtin
	test_num("setcar case fold", "Make \"L\" Prepend 1 [] setcar :L 9 Output Item :L 0", 9);
	// SetCdr can build a cycle; fixed-index reads prove it without walking it
	test_num("SetCdr cycle item 2", "Make \"L\" Prepend 1 Prepend 2 [] SetCdr ButFirst :L :L Output Item :L 2", 1);
	test_num("SetCdr cycle item 3", "Make \"L\" Prepend 1 Prepend 2 [] SetCdr ButFirst :L :L Output Item :L 3", 2);
	// FATAL paths (Sec.173: the reason names type + value of the offender)
	test_fatal_catch_case("SetCar NUMBER offender", "Output SetCar 5 1", "SetCar requires a pair");
	test_fatal_catch_case("SetCar offender named", "Output SetCar 5 1", "got number '5'");
	test_fatal_catch_case("SetCdr NIL offender", "Output SetCdr [] 1", "SetCdr requires a pair");
	test_fatal_catch_case("SetCdr FLOAT offender", "Output SetCdr 1.5 1", "SetCdr requires a pair");
	test_fatal_catch_case("SetCdr word offender named", "Output SetCdr \"abc\" 1", "got word 'abc'");
}

/* lg_value_to_str dotted-tail fix: an improper
   (dotted) list must render its tail after " . " -- the pre-fix code let
   PopCons turn the non-cons rest into NIL, so the tail value silently
   vanished and rendered as an empty nested list ("[1 []]" instead of
   "[1 . 5]"). Brackets cannot appear inside SectorLogo string literals
   (Sec.146 fact 24), so the assertions probe the rendering via Search
   and TextLength on an "x"-prefixed concat instead of a full compare. */
void test_dotted_list_rendering()
{
	printf("\n--- Dotted (improper) list rendering ---\n");

	// NUMBER tail: "x[1 . 5]" -- the " . " separator sits at 0-based index 3
	test_num("dotted NUMBER tail renders dot separator",
	         "Make \"L\" Prepend 1 [] SetCdr :L 5 Make \"S\" \"x\" _ :L Output Search \" . \" :S",
	         3);
	// full width: "x[1 . 5]" = 8 bytes (the pre-fix "[1 []]" form is 7 with the prefix)
	test_num("dotted NUMBER tail text length",
	         "Make \"L\" Prepend 1 [] SetCdr :L 5 Make \"S\" \"x\" _ :L Output TextLength :S",
	         8);
	// word tail: "x[1 . tail]" -- same separator position
	test_num("dotted word tail renders dot separator",
	         "Make \"L\" Prepend 1 [] SetCdr :L \"tail\" Make \"S\" \"x\" _ :L Output Search \" . \" :S",
	         3);
	// dotted chain flattens like standard Lisp: (1 2 . 3) -> "x[1 2 . 3]", separator at index 5
	test_num("dotted chain renders flat",
	         "Make \"L\" Prepend 1 Prepend 2 [] SetCdr ButFirst :L 3 Make \"S\" \"x\" _ :L Output Search \" . \" :S",
	         5);
	// FLOAT tails take the dotted path too
	test_true("dotted FLOAT tail keeps separator",
	          "Make \"L\" Prepend 1 [] SetCdr :L 2.5 Make \"S\" \"x\" _ :L Output IsNumber ( Search \" . \" :S )");
}

void test_samevalue()
{
	printf("\n--- SameValue (tagged-word identity) ---\n");

	// same-tag basics: payload identity within a tag
	test_true("SameValue numbers", "Output SameValue 5 5");
	test_nil("SameValue numbers differ", "Output SameValue 5 6");
	test_true("SameValue nil nil", "Output SameValue [] []");
	test_nil("SameValue number 0 vs nil", "Output SameValue 0 []");
	test_true("SameValue float same", "Output SameValue 1.5 1.5");
	test_nil("SameValue float vs number", "Output SameValue 1.0 1");

	// interning: bare word tokens fold A-Z to a-z, identical text -> same atom
	test_true("SameValue folded words", "Output SameValue ABC abc");
	test_nil("SameValue different words", "Output SameValue cat dog");

	// THE discrimination test: structurally equal but DISTINCT lists
	test_nil("SameValue distinct equal lists", "Output SameValue Prepend 1 Prepend 2 [] Prepend 1 Prepend 2 []");
	test_true("Equal agrees they match", "Output ((Prepend 1 (Prepend 2 [])) = (Prepend 1 (Prepend 2 [])))");

	// identity through sharing and mutation (cells never move)
	test_true("SameValue self", "Make \"L\" Prepend 1 Prepend 2 [] Output SameValue :L :L");
	test_true("SameValue shared tail", "Make \"L\" Prepend 1 Prepend 2 [] Output SameValue ButFirst :L ButFirst :L");
	test_nil("SameValue sibling tails distinct", "Make \"L\" Prepend 1 Prepend 2 [] Make \"M\" Prepend 1 Prepend 2 [] Output SameValue ButFirst :L ButFirst :M");
	test_true("SameValue survives SetCar", "Make \"L\" Prepend 1 [] Make \"M\" :L SetCar :L 9 Output SameValue :L :M");

	// both dispatch positions (the SetCar regression class: statement position must consume both args)
	test_num("SameValue statement consumed", "Make \"a\" 5 Make \"b\" 5 SameValue :a :b Output 1", 1);
	test_num("SameValue in IfElse", "Output IfElse SameValue 5 5 [7] [8]", 7);
	test_num("SameValue false branch", "Output IfElse SameValue 5 6 [7] [8]", 8);

	// primitive name case-folds like every builtin
	test_true("samevalue case fold", "Output samevalue 4 4");
}

/* Suite 82: Run-built statement blocks with STRING heads must execute like
   parsed literals (asm_bug_Report.md). Root cause: lg_ensure_word re-interned
   the STRING text case-preserved, so "Make" became WORD "Make" (above the
   dispatch highwater) instead of the builtin "make" -- the statement silently
   skipped. Fixed: the STRING->WORD conversion folds A-Z, and a converted
   non-builtin WORD falls through to the user-fn lookup instead of a no-op. */
void test_run_string_dispatch()
{
	printf("\n--- Run-built blocks: STRING heads dispatch (bug report fix) ---\n");

	// controls: parsed literal + lowercase string head worked before the fix
	test_num("parsed literal Make block (repro A)", "Run [ Make \"c1\" 5 ] Output :c1", 5);
	test_num("built block lowercase string head", "Make \"b\" [] Make \"b\" Prepend 5 :b Make \"b\" Prepend \"b1\" :b Make \"b\" Prepend \"make\" :b Run :b Output :b1", 5);

	// THE reported bug: mixed-case STRING head must fold to the builtin
	test_num("built block mixed-case string head", "Make \"b\" [] Make \"b\" Prepend 5 :b Make \"b\" Prepend \"b2\" :b Make \"b\" Prepend \"Make\" :b Run :b Output :b2", 5);
	test_num("built block ALL-CAPS repeat head", "Make \"y\" 0 Make \"b\" Prepend [ Make \"y\" :y + 1 ] [] Make \"b\" Prepend 3 :b Make \"b\" Prepend \"REPEAT\" :b Run :b Output :y", 3);

	// runtime-string USER-FN head: .stmt_word_not_builtin falls through to the fn lookup
	// (the fn body updates a PRE-BOUND outer var: Make inside a fn with no outer
	// binding creates it in the fn frame -- lost on return, not observable)
	test_num("built block mixed-case user-fn head", "Make \"g7\" 0 Define \"MyFn\" [] [ Make \"g7\" 7 ] Make \"b\" Prepend \"MYFN\" [] Run :b Output :g7", 7);
	test_num("built block bare user-fn head control", "Make \"g8\" 0 Define \"MyFn2\" [] [ Make \"g8\" 8 ] Make \"b\" Prepend Item [ MyFn2 ] 0 [] Run :b Output :g8", 8);

	// word heads captured from parsed list literals (report repro D/E heads)
	test_num("built block word head word name", "Make \"b\" [] Make \"b\" Prepend 5 :b Make \"b\" Prepend Item [ c3 ] 0 :b Make \"b\" Prepend Item [ Make ] 0 :b Run :b Output :c3", 5);
	test_num("built block word head string name", "Make \"b\" [] Make \"b\" Prepend 5 :b Make \"b\" Prepend \"c4\" :b Make \"b\" Prepend Item [ Make ] 0 :b Run :b Output :c4", 5);

	// mixed-case STRING name: Make folds it (lg_atom_lowercase) -> :var read works
	test_num("mixed-case string name binds folded key", "Make \"b\" [] Make \"b\" Prepend 5 :b Make \"b\" Prepend \"C5\" :b Make \"b\" Prepend \"make\" :b Run :b Output :c5", 5);

	// expression position keeps the design rule: quoted builtin text is DATA
	test_true("quoted builtin text in value slot is data", "Output ((Item [ \"Make\" ] 0) = \"Make\")");

	// greedy infix absorption is correct Logo semantics; the FATAL now renders
	// the control byte READABLY: Character absorbs (9 _ :cr) -> atom "9\r" -> \x0D
test_fatal_catch_case("control byte in FATAL offender renders escaped",
	               "Make \"cr\" Character 13 Make \"t\" Character 9 _ :cr", "9\\x0D");
	/* SL_CHAR_N1_MAX (254) is load-bearing, not a convenient round number: raising it
	   to 255 would admit Character 256, whose ax = 0x0100 stores a NUL as the
	   character and silently interns the empty-string atom. Pin both ends. */
	test_true("Character 255 is the last accepted code", "Output TextLength Character 255");
	test_fatal_catch_case("Character 256 is rejected, not silently empty",
	                      "Output Character 256", "range 1..255");
	test_fatal_catch_case("Character 0 is rejected", "Output Character 0", "range 1..255");
}

/* Suite 83: AsWord + runtime-built paren groups (second bug-report blocker).
   In expression position STRING atoms are DATA by design (the quoted-"+" rule);
   control tokens (parens, call heads, infix operators) must be WORD atoms.
   AsWord exposes the STRING->folded-WORD bridge so generated code (the Scheme
   compiler) can produce them from runtime-built name strings. */
void test_asword()
{
	printf("\n--- AsWord + built-block paren groups (bug report fix 2) ---\n");

	// conversion identity: AsWord of a string IS the parsed WORD spelling
	test_true("AsWord of string is the word atom", "Output SameValue AsWord \"Make\" Item [ Make ] 0");
	test_true("AsWord folds case", "Output SameValue AsWord \"MYFN\" Item [ myfn ] 0");
	test_true("AsWord of word is itself", "Output SameValue AsWord Item [ hello ] 0 Item [ hello ] 0");
	test_true("AsWord of operator token", "Output SameValue AsWord \"_\" Item [ _ ] 0");

	// the generated-call mechanic: runtime-string head + paren group in a built block
	test_num("built block AsWord paren group", "Define \"fx\" [:a :b] [ Output :a + :b ] Make \"g\" 0 Define \"setter\" [:v] [ Make \"g\" :v ] Make \"lp\" AsWord \"(\" Make \"rp\" AsWord \")\" Make \"fw\" AsWord \"FX\" Make \"blk\" [] Make \"blk\" Prepend :rp :blk Make \"blk\" Prepend 22 :blk Make \"blk\" Prepend 20 :blk Make \"blk\" Prepend :fw :blk Make \"blk\" Prepend :lp :blk Make \"blk\" Prepend \"setter\" :blk Run :blk Output :g", 42);
	// the word-literal spelling of the same shape (probe_wp2 mechanic)
	test_num("built block word-atom paren group", "Define \"fy\" [:a :b] [ Output :a + :b ] Make \"g2\" 0 Define \"setter2\" [:v] [ Make \"g2\" :v ] Make \"blk\" [] Make \"blk\" Prepend Item [ ) ] 0 :blk Make \"blk\" Prepend 22 :blk Make \"blk\" Prepend 20 :blk Make \"blk\" Prepend Item [ fy ] 0 :blk Make \"blk\" Prepend Item [ ( ] 0 :blk Make \"blk\" Prepend \"setter2\" :blk Run :blk Output :g2", 42);

	// data preservation: paren/operator STRINGS remain data in expression position
	test_true("paren string stays data", "Output (\"(\" = \"(\")");
	test_num("concat paren strings", "Output TextLength \"(\" _ \")\"", 2);
	test_true("plus string stays data", "Output (\"+\" = \"+\")");

	// Sec.173: non-atom offenders FATAL with type + value named
	test_fatal_catch_case("AsWord NUMBER offender", "Output AsWord 42", "got number '42'");
	test_fatal_catch_case("AsWord list offender", "Output AsWord [ 1 2 ]", "got 'list'");

	// Run of a fn-calling block propagates the fn's Output (dynamic apply --
	// the sl_stmt_fn_retval channel; the OUTPUT flag itself is consumed at
	// the fn boundary, so this is the ONLY path the value reaches Run on)
	test_num("Run applies fn with Output body", "Define \"outvar\" [:x] [ Output :x ] Output Run Prepend \"outvar\" Prepend 7 []", 7);
	test_num("Run applies built fn-call block", "Define \"adder\" [:a :b] [ Output :a + :b ] Make \"blk\" [] Make \"blk\" Prepend 22 :blk Make \"blk\" Prepend 20 :blk Make \"blk\" Prepend \"adder\" :blk Output Run :blk", 42);
	test_num("Run fn-call with runtime-string head", "Define \"outvar2\" [:x] [ Output :x * 2 ] Output Run Prepend \"OUTVAR2\" Prepend 21 []", 42);
	test_nil("Run without fn call still NIL", "Output Run [ Make \"q\" 1 ]");
	// the channel never leaks across Runs: a fn-less Run after a fn-Run reads fresh state
	test_num("Run channel does not leak", "Define \"outvar3\" [:x] [ Output :x ] Make \"a\" Run Prepend \"outvar3\" Prepend 9 [] Make \"b\" Run [ Make \"q\" 1 ] Output :a", 9);

	printf("\n--- NIL element preservation (empty lists as list items) ---\n");

	// the data model: Prepend stores a NIL item as a real one-element list
	test_num("Prepend preserves NIL element (length 1)", "Output Length Prepend [] []", 1);
	test_num("literal nested empty list length", "Output Length [ [] ]", 1);
	test_true("built NIL-element list equals literal", "Output ((Prepend [] []) = [ [] ])");
	test_nil("First of NIL-element list is NIL", "Output First Prepend [] []");
	test_num("NIL element mid-list keeps length", "Output Length [ 1 [] 2 ]", 3);

	// the renderers: PrintLogoList and lg_value_to_str must show '[]' for a
	// NIL element ([1  2] with a hole and a bare [] for (NIL) hid the element)
	test_num("concat renders one-element NIL list", "Output TextLength \"x\" _ Prepend [] []", 5);
	test_true("concat renders nested empty list", "Output (\"[[]]\" = ( \"\" _ [ [] ] ))");
	test_true("concat renders NIL mid-list", "Output (\"[1 [] 2]\" = ( \"\" _ [ 1 [] 2 ] ))");
	test_true("concat renders built NIL mid-list", "Output (\"[1 [] 2]\" = ( \"\" _ (Prepend 1 (Prepend [] (Prepend 2 []))) ))");
}

void test_intern_boundary_lengths()
{
	printf("\n--- Intern boundary lengths (dual-hash h2 exact-copy paths) ---\n");
	// Two identical quoted strings at every copy-path boundary: the second
	// intern must agree with the first (bloom + slot cache + linear scan all
	// key on h2). Lengths 0/1 (tail only), 7/8/15/16 (chunk boundaries),
	// 31/32/63/64 (block boundary), 65 (clamped long string).
	int lengths[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65};
	char script[256];

	for (int i = 0; i < 11; i++)
	{
		int n = lengths[i];
		char *cursor = script;
		cursor += sprintf(cursor, "Output Equal \"");
		for (int j = 0; j < n; j++)
		{
			*cursor++ = (char)('a' + (j % 26));
		}
		cursor += sprintf(cursor, "\" \"");
		for (int j = 0; j < n; j++)
		{
			*cursor++ = (char)('a' + (j % 26));
		}
		cursor += sprintf(cursor, "\"");
		*cursor = 0;
		char desc[64];
		sprintf(desc, "intern equal len %d", n);
		test_true(desc, script);
	}
}

void test_lfind_family()
{
	printf("\n--- FindItem / FindFirstAtLeast / FindFirstAtMost ---\n");
	// LFIND exact match
	test_num("FindItem bare", "Output FindItem [1 3 5 7] 0 3", 1);
	test_num("FindItem sub-list", "Output FindItem [[2 \"a\"] [4 \"b\"]] 0 4", 1);
	test_nil("FindItem not found", "Output FindItem [1 3 5 7] 0 99");
	test_nil("FindItem startpos past end", "Output FindItem [1 2 3] 5 1");
	// LFIND with startpos (skips the bare 4? no: sub-list [4] at position 3)
	test_num("FindItem startpos", "Output FindItem [1 3 [2 \"hallo\"] [4]] 2 4", 3);
	// LFINDGE on sorted list: lower bound
	test_num("FindFirstAtLeast sorted", "Output FindFirstAtLeast Sort [5 1 3 7] 0 4", 2);
	test_num("FindFirstAtLeast exact", "Output FindFirstAtLeast Sort [5 1 3 7] 0 3", 1);
	test_nil("FindFirstAtLeast none", "Output FindFirstAtLeast Sort [5 1 3 7] 0 9");
	// LFINDLE on sorted list: upper bound
	test_num("FindFirstAtMost sorted", "Output FindFirstAtMost Sort [7 3 5 1] 0 4", 1);
	test_num("FindFirstAtMost exact", "Output FindFirstAtMost Sort [7 3 5 1] 0 5", 2);
	test_nil("FindFirstAtMost none", "Output FindFirstAtMost Sort [7 3 5 1] 0 0");
	// float keys: mixed NUMBER/FLOAT promotion in the comparison
	test_num("FindFirstAtLeast float key", "Output FindFirstAtLeast Sort [1 2 3 4] 0 2.5", 2);
	// coerced ATOM fraction keys must bit-match the literal FLOAT of the same
	// text: sl_sort_coerce_num shares the literal parser's forward fraction
	// scheme (one rounding, SF_ATOM_DIGIT_MAX cap), so .lf_flt_eq finds them
	test_num("FindItem coerced atom frac equals literal", "Output FindItem [\"0.999\" 1.5] 0 0.999", 0);
	test_num("FindItem coerced atom frac 9 digits", "Output FindItem [\"0.123456789\" 1.5] 0 0.123456789", 0);
	test_num("FindFirstAtLeast coerced atom frac", "Output FindFirstAtLeast Sort [\"0.25\" 1] 0 0.1", 0);
	// over the fraction digit cap the coercion rejects with the atom named --
	// the same domain rule the literal parser enforces (SF_ATOM_DIGIT_MAX)
	test_fatal_catch_case("FATAL FindItem atom frac over digit cap", "Output FindItem [\"0.1234567890\"] 0 1", "a numeric value was required");
	// incremental scan: find first, then find next
	test_num("FindItem then next",
	         "Make \"P\" FindItem [1 2 3 2 4] 0 2 "
	         "Output FindItem [1 2 3 2 4] (:P + 1) 2",
	         3);
	// FATAL cases (manual):
	//   OUTPUT LFIND 42 0 3         -> FATAL not a list
	//   OUTPUT LFINDGE [1 2 3] -1 3 -> FATAL negative startpos
	//   OUTPUT LFINDLE [1 2 3] 0 \"x\" -> FATAL key must be NUMBER or FLOAT
}

void test_run_eval()
{
	printf("\n--- Run / EVAL ---\n");
	test_num("Run block", "Make \"X\" 10 Run [ Make \"X\" 20 ] Output :X", 20);
	/* RUN plain-atom arm (eval_stmt Sec.173 fix): a word
	   handed to RUN used to be skipped SILENTLY while NUMBER/FLOAT died naming
	   the offender -- now every non-list, non-NIL value aborts uniformly. */
	test_fatal_catch_case("Run of a NUMBER dies",
	                      "Output Run 5", "RUN requires a list or NIL");
	test_fatal_catch_case("Run of a word dies (was a silent skip)",
	                      "Output Run \"someword\"", "RUN requires a list or NIL");
}

void test_nested_control()
{
	printf("\n--- Nested control flow ---\n");
	test_num("IfElse+Repeat", "IfElse 1 [ Make \"A\" 0 Repeat 3 [ Make \"A\" :A + 5 ] ] [ Make \"A\" -1 ] Output :A", 15);
}

void test_complex()
{
	printf("\n--- Complex expressions ---\n");
	test_num("nested calls", "Output ((3 * 4) + (20 - 8))", 24);
	test_num("sequential Make", "Make \"A\" 1 Make \"A\" :A + 2 Make \"A\" :A + 3 Output :A", 6);
}

void test_negative_numbers()
{
	printf("\n--- Negative numbers ---\n");
	test_num("literal -1", "Output -1", -1);
	test_num("literal -100", "Output -100", -100);
	test_num("negative plus positive", "Output (-5 + 10)", 5);
	test_num("negative times negative", "Output (-3 * -7)", 21);
	test_num("AbsoluteValue -999", "Output AbsoluteValue -999", 999);
	/* The signed 30-bit payload range is [-2^29, 2^29-1] = [-536870912, 536870911].
	   The integer-literal parser checked the POSITIVE limit before the sign, so
	   -536870912 (magnitude 2^29, final digit 2) was wrongly promoted to FLOAT. */
	test_num("literal -536870912 (minimum NUMBER)", "Output -536870912", -536870912);
	test_num("literal -536870911", "Output -536870911", -536870911);
	test_float_val("literal 536870912 stays FLOAT (above the payload)", "Output 536870912", 536870912.0f);
	test_float_val("literal -536870913 stays FLOAT (below the payload)", "Output -536870913", -536870913.0f);
}

void test_string_operations()
{
	printf("\n--- String operations (Part, Search, Lowercase, Uppercase) ---\n");

	/* SUBSTR: basic extraction */
	test_true("Part basic", "Output ((Part \"hello\" 1 3) = \"ell\")");
	test_true("Part from start", "Output ((Part \"hello\" 0 2) = \"he\")");
	test_true("Part to end", "Output ((Part \"hello\" 3 2) = \"lo\")");
	test_true("Part single char", "Output ((Part \"abc\" 1 1) = \"b\")");
	test_true("Part zero len", "Output ((Part \"abc\" 0 0) = \"\")");
	test_true("Part past end", "Output ((Part \"abc\" 5 2) = \"\")");
	test_true("Part partial past end", "Output ((Part \"abc\" 2 5) = \"c\")");
	test_true("Part start 0", "Output ((Part \"xyz\" 0 3) = \"xyz\")");

	/* FIND: basic substring search */
	test_num("Search found", "Output Search \"ell\" \"hello\"", 1);
	test_num("Search start", "Output Search \"hel\" \"hello\"", 0);
	test_num("Search end", "Output Search \"lo\" \"hello\"", 3);
	test_nil("Search not found", "Output Search \"xyz\" \"hello\"");
	test_num("Search empty needle", "Output Search \"\" \"hello\"", 0);
	test_true("Search empty needle with NIL haystack", "Make \"H\" First [] Make \"P\" Search \"\" :H Output IsNumber :P");
	test_num("Search single char", "Output Search \"e\" \"hello\"", 1);
	test_nil("Search empty haystack", "Output Search \"x\" \"\"");

	/* Phase 5 scan regression net (docs/plans/sectorlogo_faster_strings_plan.md 7.7.1-4).
	   These sit BEFORE the STR_LOWER block; wuf/kyi appear nowhere else in the
	   validator, so the atom-intern-order trick holds: the atom area is a bump
	   allocator, interning "wuf" first then "kyi" places "kyi"'s bytes right
	   after "wuf"'s NUL -- the past-the-NUL adversarial layout the window scan
	   must reject via the bzhi NUL-ordering guard. */
	test_nil("Search first-char past NUL", "Make \"H\" \"wuf\" Make \"X\" \"kyi\" Output Search \"kyi\" :H");
	test_num("Search after failed candidate", "Output Search \"cd\" \"cXcd\"", 2);

	/* Window-boundary trios: needle at offsets 15/16/17 of a 31-char string
	   (straddles the first 16-byte window edge). Infix _ binds tighter than
	   prefix args (Sec.146 fact 19), so the prefix/rest splits go through
	   temporaries -- a bare Part :S 0 15 _ "q" would fold 15 _ "q" into
	   SUBSTR's len arg and NumVal-FATAL. */
	test_num("Search window edge 15", "Make \"S\" \"\" Repeat 31 [ Make \"S\" :S _ \"z\" ] Make \"P\" Part :S 0 15 Make \"R\" Part :S 16 15 Make \"S\" :P _ \"q\" _ :R Output Search \"q\" :S", 15);
	test_num("Search window edge 16", "Make \"S\" \"\" Repeat 31 [ Make \"S\" :S _ \"z\" ] Make \"P\" Part :S 0 16 Make \"R\" Part :S 17 14 Make \"S\" :P _ \"q\" _ :R Output Search \"q\" :S", 16);
	test_num("Search window edge 17", "Make \"S\" \"\" Repeat 31 [ Make \"S\" :S _ \"z\" ] Make \"P\" Part :S 0 17 Make \"R\" Part :S 18 13 Make \"S\" :P _ \"q\" _ :R Output Search \"q\" :S", 17);

	/* Replace same-boundary trio: old at 15/16/17, replaced by "Q". */
	test_true("Replace window edge 15", "Make \"S\" \"\" Repeat 31 [ Make \"S\" :S _ \"z\" ] Make \"P\" Part :S 0 15 Make \"R\" Part :S 16 15 Make \"S\" :P _ \"q\" _ :R Output Equal Replace :S \"q\" \"Q\" :P _ \"Q\" _ :R");
	test_true("Replace window edge 16", "Make \"S\" \"\" Repeat 31 [ Make \"S\" :S _ \"z\" ] Make \"P\" Part :S 0 16 Make \"R\" Part :S 17 14 Make \"S\" :P _ \"q\" _ :R Output Equal Replace :S \"q\" \"Q\" :P _ \"Q\" _ :R");
	test_true("Replace window edge 17", "Make \"S\" \"\" Repeat 31 [ Make \"S\" :S _ \"z\" ] Make \"P\" Part :S 0 17 Make \"R\" Part :S 18 13 Make \"S\" :P _ \"q\" _ :R Output Equal Replace :S \"q\" \"Q\" :P _ \"Q\" _ :R");

	/* STRLOWER: uppercase to lowercase */
	test_true("Lowercase basic", "Output ((Lowercase \"HELLO\") = \"hello\")");
	test_true("Lowercase mixed", "Output ((Lowercase \"HeLLo\") = \"hello\")");
	test_true("Lowercase already lower", "Output ((Lowercase \"hello\") = \"hello\")");
	test_true("Lowercase with numbers", "Output ((Lowercase \"ABC123\") = \"abc123\")");
	test_true("Lowercase empty", "Output ((Lowercase \"\") = \"\")");

	/* STRUPPER: lowercase to uppercase */
	test_true("Uppercase basic", "Output ((Uppercase \"hello\") = \"HELLO\")");
	test_true("Uppercase mixed", "Output ((Uppercase \"HeLLo\") = \"HELLO\")");
	test_true("Uppercase already upper", "Output ((Uppercase \"HELLO\") = \"HELLO\")");
	test_true("Uppercase with numbers", "Output ((Uppercase \"abc123\") = \"ABC123\")");
	test_true("Uppercase empty", "Output ((Uppercase \"\") = \"\")");

	/* Empty-result tails of Part/Lowercase/Uppercase: a NIL source normalises to the
	   interned empty string atom, and that path returns through call+ret (not a tail
	   jmp), so the Win64 non-volatile rsi reaches the caller intact and evaluation
	   continues normally. This is the regression net for the rsi clobber. */
	test_true("Lowercase NIL returns a STRING atom", "Make \"n\" First [] Make \"r\" Lowercase :n Output IsStringAtom :r");
	test_true("Uppercase NIL returns a STRING atom", "Make \"n\" First [] Make \"r\" Uppercase :n Output IsStringAtom :r");
	test_true("Part of NIL returns a STRING atom", "Make \"n\" First [] Make \"r\" Part :n 0 5 Output IsStringAtom :r");
	test_num("Part of NIL is empty", "Make \"n\" First [] Make \"r\" Part :n 0 5 Output TextLength :r", 0);

	/* Combined string operations */
	test_true("Lowercase + Part", "Output ((Part (Lowercase \"HELLO\") 0 2) = \"he\")");
	test_true("Uppercase + Part", "Output ((Part (Uppercase \"hello\") 3 2) = \"LO\")");
	test_true("Search after Lowercase", "Output ((Search \"hello\" (Lowercase \"HELLO\")) = 0)");

	/* STRLEN: byte length (NUL excluded) */
	test_num("TextLength basic", "Output TextLength \"hello\"", 5);
	test_num("TextLength empty", "Output TextLength \"\"", 0);
	test_num("TextLength NIL", "Output TextLength []", 0);
	test_num("TextLength single", "Output TextLength \"x\"", 1);
	test_num("TextLength with space", "Output TextLength \"a b\"", 3);
	/* SWAR window edges: 7/8/9-byte strings straddle the qword boundary */
	test_num("TextLength 7 (window edge)", "Output TextLength \"abcdefg\"", 7);
	test_num("TextLength 8 (full window)", "Output TextLength \"abcdefgh\"", 8);
	test_num("TextLength 9 (window+1)", "Output TextLength \"abcdefghi\"", 9);
	/* long string: forces multiple SWAR windows + re-gate cycles */
	test_num("TextLength long (multi-window)", "Make \"S\" \"\" Repeat 100 [ Make \"S\" :S _ \"12345678\" ] Output TextLength :S", 800);
	/* roundtrip: STRLEN of a known-length concat result */
	test_num("TextLength concat", "Output TextLength \"abc\" _ \"de\"", 5);
	/* STRLEN == 0 sentinel replaced (Sec.146 #8): the old SUBSTR-clamp idiom */
	test_true("TextLength replaces Part sentinel", "Output ((TextLength \"w\") = 1)");

	/* STR_REPLACE tests */
	test_true("Replace basic", "Output ((Replace \"hello world\" \"world\" \"there\") = \"hello there\")");
	test_true("Replace not found", "Output ((Replace \"hello\" \"xyz\" \"abc\") = \"hello\")");
	test_true("Replace cat->dog", "Output ((Replace \"cat\" \"cat\" \"dog\") = \"dog\")");
	test_true("Replace empty old", "Make \"E\" \"\" Output ((Replace \"hello\" :E \"X\") = \"Xhello\")");
	test_atom("Replace empty old atom", "Make \"E\" \"\" Output Replace \"hello\" :E \"X\"");
	test_true("Replace empty new (delete)", "Make \"E\" \"\" Output ((Replace \"hello world\" \"world\" :E) = \"hello \")");
	test_true("Replace first only", "Output ((Replace \"aXaXa\" \"X\" \"Y\") = \"aYaXa\")");
	test_true("Replace at start", "Output ((Replace \"caturday\" \"cat\" \"sat\") = \"saturday\")");
	test_true("Replace at end", "Output ((Replace \"hellocat\" \"cat\" \"dog\") = \"hellodog\")");
	test_true("Replace whole string", "Output ((Replace \"abc\" \"abc\" \"xyz\") = \"xyz\")");
	test_true("Replace same old new", "Output ((Replace \"hello\" \"ll\" \"ll\") = \"hello\")");
	test_nil("Replace NIL source", "Output Replace [] \"x\" \"y\"");
	test_true("Replace in long string", "Make \"S\" \"A\" Repeat 999 [ Make \"S\" :S _ \"A\" ] "
	                                        "Make \"S\" :S _ \"XYZ\" "
	                                        "Output ((Search \"abc\" (Replace :S \"XYZ\" \"abc\")) = 1000)");

	/* Sec.173: every type-error FATAL in sectorlogo_string.inc must NAME the
	   offending value. Each of the eight sites carries its own operation-specific
	   message (they must not be unified) plus the rendered offender, so a script
	   author sees both WHICH argument was wrong and WHAT it was. Before FATAL_VALUE
	   all eight printed a fixed string with no offender at all. */
	test_fatal_catch_case("Part type error names the operation", "Output Part 42 0 1", "Part requires a string atom argument");
	test_fatal_catch_case("Part type error names the offender", "Output Part 42 0 1", "got number '42'");
	test_fatal_catch_case("Search needle type error names the operation", "Output Search 42 \"hello\"", "Search requires string atom arguments");
	test_fatal_catch_case("Search needle type error names the offender", "Output Search 42 \"hello\"", "got number '42'");
	test_fatal_catch_case("Search haystack type error names the offender", "Output Search \"x\" 42", "got number '42'");
	test_fatal_catch_case("Lowercase type error names the operation", "Output Lowercase 42", "Lowercase requires a string atom argument");
	test_fatal_catch_case("Lowercase type error names the offender", "Output Lowercase 42", "got number '42'");
	test_fatal_catch_case("Uppercase type error names the operation", "Output Uppercase 42", "Uppercase requires a string atom argument");
	test_fatal_catch_case("Uppercase type error names the offender", "Output Uppercase 42", "got number '42'");
	test_fatal_catch_case("TextLength type error names the operation", "Output TextLength 42", "TextLength requires a string atom argument");
	test_fatal_catch_case("TextLength type error names the offender", "Output TextLength 42", "got number '42'");
	test_fatal_catch_case("Replace source type error names the operation", "Output Replace 42 \"a\" \"b\"", "Replace requires string atom arguments");
	test_fatal_catch_case("Replace source type error names the offender", "Output Replace 42 \"a\" \"b\"", "got number '42'");
	test_fatal_catch_case("Replace old type error names the offender", "Output Replace \"abc\" 42 \"b\"", "got number '42'");
	test_fatal_catch_case("Replace new type error names the offender", "Output Replace \"abc\" \"a\" 42", "got number '42'");
	test_fatal_catch_case("TextToNumber type error names the operation", "Output TextToNumber [ 1 2 ]", "TextToNumber requires a string atom argument");
	test_fatal_catch_case("TextToNumber type error names the offender", "Output TextToNumber [ 1 2 ]", "got 'list'");
	test_fatal_catch_case("Split source type error names the operation", "Output Split 42 \",\"", "Split requires string atom arguments");
	test_fatal_catch_case("Split source type error names the offender", "Output Split 42 \",\"", "got number '42'");
	test_fatal_catch_case("Split delimiter type error names the offender", "Output Split \"a,b\" 42", "got number '42'");
}

void test_ffi()
{
	printf("\n--- FFI (CallLibrary native function calls) ---\n");

	/* === Mini DLL (slffi_test.dll) tests: deterministic endpoints === */
	/* All quoted atoms use closing \" per SectorLogo string syntax. */

	/* INT return, 0 args */
	test_num("FFI noargs -> 42",
	         "Make \"b1\" [" SLFFI_TEST_MODULE " \"slffi_noargs\" \"integer\" []] "
	         "Output CallLibrary :b1",
	         42);

	/* INT return, 1 INT arg */
	test_num("FFI double 21 -> 42",
	         "Make \"b2\" [" SLFFI_TEST_MODULE " \"slffi_double\" \"integer\" [\"integer\"]] "
	         "Output CallLibrary :b2 21",
	         42);

	/* INT return, 2 INT args */
	test_num("FFI add 17 25 -> 42",
	         "Make \"b3\" [" SLFFI_TEST_MODULE " \"slffi_add\" \"integer\" [\"integer\" \"integer\"]] "
	         "Output CallLibrary :b3 17 25",
	         42);
	test_num("FFI mul 6 7 -> 42",
	         "Make \"b4\" [" SLFFI_TEST_MODULE " \"slffi_mul\" \"integer\" [\"integer\" \"integer\"]] "
	         "Output CallLibrary :b4 6 7",
	         42);
	test_num("FFI sub 50 8 -> 42",
	         "Make \"b5\" [" SLFFI_TEST_MODULE " \"slffi_sub\" \"integer\" [\"integer\" \"integer\"]] "
	         "Output CallLibrary :b5 50 8",
	         42);

	/* INT return, 3 INT args (rcx, rdx, r8) */
	test_num("FFI add3 10 20 12 -> 42",
	         "Make \"b6\" [" SLFFI_TEST_MODULE " \"slffi_add3\" \"integer\" [\"integer\" \"integer\" \"integer\"]] "
	         "Output CallLibrary :b6 10 20 12",
	         42);

	/* INT return, 5 INT args (args 5+ on stack per Win64 ABI) */
	test_num("FFI add5 1 2 3 4 32 -> 42",
	         "Make \"b7\" [" SLFFI_TEST_MODULE " \"slffi_add5\" \"integer\" [\"integer\" \"integer\" \"integer\" \"integer\" \"integer\"]] "
	         "Output CallLibrary :b7 1 2 3 4 32",
	         42);

	/* INT return, 1 PTR arg (strlen) */
	test_num("FFI strlen hello -> 5",
	         "Make \"b8\" [" SLFFI_TEST_MODULE " \"slffi_strlen\" \"integer\" [\"pointer\"]] "
	         "Output CallLibrary :b8 \"hello\"",
	         5);
	test_num("FFI strlen empty -> 0",
	         "Make \"b9\" [" SLFFI_TEST_MODULE " \"slffi_strlen\" \"integer\" [\"pointer\"]] "
	         "Output CallLibrary :b9 \"\"",
	         0);
	test_num("FFI strlen abcdef -> 6",
	         "Make \"b10\" [" SLFFI_TEST_MODULE " \"slffi_strlen\" \"integer\" [\"pointer\"]] "
	         "Output CallLibrary :b10 \"abcdef\"",
	         6);

	/* STR return, 1 PTR arg (identity: returns same pointer) */
	test_true("FFI identity returns same string", "Make \"b11\" [" SLFFI_TEST_MODULE " \"slffi_identity\" \"text\" [\"pointer\"]] "
	                                              "Output ((CallLibrary :b11 \"test\") = \"test\")");

	/* VOID return + INT arg, then INT return with no args (global get/set) */
	test_nil("FFI set_global VOID -> NIL", "Make \"b12a\" [" SLFFI_TEST_MODULE " \"slffi_set_global\" \"void\" [\"integer\"]] "
	                                       "CallLibrary :b12a 99");
	test_num("FFI get_global after set -> 99",
	         "Make \"b12b\" [" SLFFI_TEST_MODULE " \"slffi_get_global\" \"integer\" []] "
	         "Output CallLibrary :b12b",
	         99);

	/* PTR + INT args */
	test_num("FFI starts_with hello 3 -> 1",
	         "Make \"b13\" [" SLFFI_TEST_MODULE " \"slffi_starts_with\" \"integer\" [\"pointer\" \"integer\"]] "
	         "Output CallLibrary :b13 \"hello\" 3",
	         1);
	test_num("FFI starts_with hi 5 -> 0 (too short)",
	         "Make \"b14\" [" SLFFI_TEST_MODULE " \"slffi_starts_with\" \"integer\" [\"pointer\" \"integer\"]] "
	         "Output CallLibrary :b14 \"hi\" 5",
	         0);

	/* === kernel32.dll: lstrlenA (system DLL) === */
	test_num("FFI lstrlenA hello -> 5",
	         "Make \"b15\" [\"kernel32.dll\" \"lstrlenA\" \"integer\" [\"pointer\"]] "
	         "Output CallLibrary :b15 \"hello\"",
	         5);
	/* STR ("text") arg type: marshals as char*, same as PTR. Regression (Sep 2026):
	   the "text" arm of the arg-type dispatch had no jump -- the "jne" left a
	   matched STR falling straight into the INT marshal, where NumVal rejects a
	   tagless string atom with "a NUMBER was required" -- an arithmetic error
	   for a valid, documented argument type. Every other string-arg case here
	   declares "pointer", which is exactly why this went unseen. Note "text" must
	   be lowercase: the type atoms are interned lowercase. */
	test_num("FFI text arg marshals as char* strlen hello -> 5",
	         "Make \"b18\" [" SLFFI_TEST_MODULE " \"slffi_strlen\" \"integer\" [\"text\"]] "
	         "Output CallLibrary :b18 \"hello\"",
	         5);

	/* === SELF: call v7's own exported functions === */
	test_num("FFI SELF num_workers",
	         "Make \"b16\" [\"SELF\" \"v7s_get_num_workers\" \"integer\" []] "
	         "Output CallLibrary :b16",
	         0);
	test_nil("FFI SELF set_spin VOID -> NIL", "Make \"b17\" [\"SELF\" \"v7s_set_spin\" \"void\" [\"integer\"]] "
	                                          "CallLibrary :b17 1");

	/* FFI INT return (ffi BUG-1). A NUMBER is a 30-bit
	   payload, so an INT return of 0x40000000 used to be masked into 0 and come
	   back to the script as a plain wrong number with no diagnostic anywhere --
	   and INT is the only integer return type the FFI offers, so the value had no
	   correct spelling. It saturates at the payload ceiling now, same
	   boundary behaviour as the arithmetic operators. */
	test_num("FFI INT return outside the 30-bit payload saturates instead of wrapping",
	         "Make \"b19\" [" SLFFI_TEST_MODULE " \"slffi_too_big\" \"integer\" []] "
	         "Output CallLibrary :b19",
	         536870911);
	/* PTR/STR tag order (ffi BUG-2). A FLOAT carries BOTH
	   tag bits (SL_FLOAT_TAG = 0xC0000000), so testing SL_NUMBER_TAG first routed a
	   float through NumVal and it died with NumVal's "a NUMBER was required
	   (arithmetic, index, count, or key operand)" -- naming arithmetic for an
	   argument the script declared "pointer", and never mentioning C-CALL. The
	   FFI's own message must be the one that fires. */
	test_fatal_catch_case("FFI float declared as pointer dies with the C-CALL type error",
	                      "Make \"b20\" [" SLFFI_TEST_MODULE " \"slffi_strlen\" \"integer\" [\"pointer\" 1.5]] "
	                      "Output CallLibrary :b20",
	                      "CallLibrary type mismatch");
	/* argument-side type dispatch moving AHEAD of
	   EvalNextArg. The ret_type arm has always validated before running anything;
	   the argument arm used to evaluate the argument first and reject afterwards,
	   so a bad declared type applied the argument's side effects (a nested
	   C-CALL, a SetStream, a Reject) before reporting the bad descriptor. The
	   observable half of that contract is that the FFI's own type error must be
	   what fires for an unknown declared ARGUMENT type -- not whatever the
	   argument expression happens to do when it is evaluated. Here the argument
	   is a nested C-CALL that would raise a resolvable module, so if it ran first
	   the script would either succeed or die with the inner module's message
	   instead of naming the descriptor's bad type. */
	test_fatal_catch_case("FFI unknown argument type dies with the C-CALL type error",
	                      "Make \"b21\" [" SLFFI_TEST_MODULE " \"slffi_strlen\" \"integer\" [\"nosuchtype\" 0]] "
	                      "Output CallLibrary :b21",
	                      "CallLibrary type mismatch");
	test_fatal_catch_case("FFI unknown argument type does not run the argument first",
	                      "Make \"b22\" [" SLFFI_TEST_MODULE " \"slffi_strlen\" \"integer\" [\"nosuchtype\" "
	                      "CallLibrary [" SLFFI_TEST_MODULE " \"slffi_strlen\" \"integer\" [\"integer\" \"abc\"]]]] "
	                      "Output CallLibrary :b22",
	                      "CallLibrary type mismatch");
	/* The control: every valid declared type must still marshal, so hoisting the
	   dispatch did not strand one of the three arms. Each descriptor declares
	   exactly as many argument types as there are arguments passed after it --
	   under-declaring leaves the loop with no type to marshal the next argument
	   with, which is a different failure entirely. */
	test_num("FFI INT argument still marshals after the reorder",
	         "Make \"b23\" [" SLFFI_TEST_MODULE " \"slffi_double\" \"integer\" [\"integer\"]] "
	         "Output CallLibrary :b23 21", 42);
	test_num("FFI PTR argument still marshals after the reorder",
	         "Make \"b24\" [" SLFFI_TEST_MODULE " \"slffi_strlen\" \"integer\" [\"pointer\"]] "
	         "Output CallLibrary :b24 \"abcd\"", 4);
	test_num("FFI STR argument still marshals after the reorder",
	         "Make \"b25\" [" SLFFI_TEST_MODULE " \"slffi_strlen\" \"integer\" [\"text\"]] "
	         "Output CallLibrary :b25 \"abcd\"", 4);
	test_num("FFI two INT arguments still marshal after the reorder",
	         "Make \"b26\" [" SLFFI_TEST_MODULE " \"slffi_add\" \"integer\" [\"integer\" \"integer\"]] "
	         "Output CallLibrary :b26 17 25", 42);
	/* arg_types cursor advancing by exactly ONE list
	   level. A cursor that lands one level too deep reads the argtypes LIST
	   itself as the next declared type, and every argument then dies naming a
	   list where a type atom belongs. */
	test_num("FFI arg_types cursor advances one level per argument",
	         "Make \"b27\" [" SLFFI_TEST_MODULE " \"slffi_add3\" \"integer\" [\"integer\" \"integer\" \"integer\"]] "
	         "Output CallLibrary :b27 10 20 12", 42);
}

/* DefineCommand (DEF_EXE_FN): spawn an external process, capture its
   stdout+stderr, intern the capture, return it as a string atom.
   Regression (Sep 2026): the epilogue restored the escape-arena cursor
   through rax, destroying the Intern return atom -- every call returned a
   garbage NUMBER (arena-pointer low bits). Every test below fails under
   that bug. echo_args.exe prints "argc=N" plus its argv -- needles probe
   the capture text. NOTE (Sec.146.24): string literals must NOT contain
   brackets -- probe argc counts and bare values, never "arg1=[...]" text. */
void test_def_exe_fn()
{
	printf("\n--- DefineCommand (external process capture) ---\n");
	/* the capture is a plain string atom, not a NUMBER (the garbage bug) */
	test_atom("DefExeFn return is a string atom",
	          "DefineCommand \"vexe\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"%s\" "
	          "Make \"cap\" vexe \"hello\" "
	          "Output :cap");
	/* %s substitutes raw: the format string owns quoting (documented contract
	   -- a space-bearing value arrives as SEPARATE child argv entries, so
	   the child counts argc=3: argv0 + hello + world). IsNumber returns the
	   truthy atom, not NUMBER 1 -- test_true. */
	test_true("DefExeFn raw %s splits on space",
	          "DefineCommand \"vraw\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"%s\" "
	          "Make \"cap\" vraw \"hello world\" "
	          "Output IsNumber Search \"argc=3\" :cap");
	/* quoted %s keeps a space-bearing value as ONE child argv entry (argc=2) */
	test_true("DefExeFn quoted %s keeps one arg",
	          "DefineCommand \"vq\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\"\" "
	          "Make \"cap\" vq \"hello world\" "
	          "Output IsNumber Search \"argc=2\" :cap");
	/* %d formats a NUMBER argument (the child receives the digits as ONE argv
	   entry -- argc=2). NOTE: the needle cannot be the bare digits: a quoted
	   numeric literal is parsed as a NUMBER (Sec.146 fact 39), and Search
	   FATALs on a non-string needle */
	test_true("DefExeFn %d number arg",
	          "DefineCommand \"vnum\" [:n] " SLFFI_ECHO_ARGS_ARG " \"%d\" "
	          "Make \"cap\" vnum 42 "
	          "Output IsNumber Search \"argc=2\" :cap");
	/* two params, one format specifier each (alpha/beta: collision-free
	   literals -- Sec.146.29, first/second would invoke the builtins) */
	test_true("DefExeFn two args",
	          "DefineCommand \"vtwo\" [:a :b] " SLFFI_ECHO_ARGS_ARG " \"%s %s\" "
	          "Make \"cap\" vtwo \"alpha\" \"beta\" "
	          "Output IsNumber Search \"beta\" :cap");
	/* Ordinal desync regression: %i is emitted LITERALLY by v7sprintf and
	   fetches no vararg, so only the two %s consume slots. sl_xfn_count_conversions
	   returns 2 and the arity gate passes with two declared params -- but a probe
	   that counted %i as a conversion would hand ordinal 1 to the %i and report -2
	   for the second %s slot, storing a raw integer where a char* is read. */
	test_true("DefExeFn unsupported spec does not shift the ordinal",
	          "DefineCommand \"vlit\" [:a :b] " SLFFI_ECHO_ARGS_ARG " \"%s %i %s\" "
	          "Make \"cap\" vlit \"alpha\" \"beta\" "
	          "Output IsNumber Search \"beta\" :cap");
	/* Sec.173 slot-mismatch coverage: a string atom against a %d format and a
	   number against a %s format are both legal VALUES in illegal SLOTS -- the
	   abort must name the slot conflict, not deny the value's own type. */
	test_fatal_catch_case("DefExeFn string into %d names the slot",
	                      "DefineCommand \"vsd\" [:n] " SLFFI_ECHO_ARGS_ARG " \"%d\" "
	                      "Output vsd \"alpha\"",
	                      "format slot does not accept this argument type");
	test_fatal_catch_case("DefExeFn number into %s names the slot",
	                      "DefineCommand \"vns\" [:s] " SLFFI_ECHO_ARGS_ARG " \"%s\" "
	                      "Output vns 42",
	                      "format slot does not accept this argument type");
	/* BUG-1 regression (Sep 2026): the escape-arena cursor and saved-cursor
	   stack slots sat at offsets 88/96, which are exactly argv[4] and argv[5]
	   (argv base 56 + 4*8 and + 5*8). A call with 5+ arguments therefore let
	   .xfn_store_arg overwrite the cursor with an arena pointer, so the NEXT
	   string argument escaped into a heap address, and argv[5] clobbered the
	   saved cursor the epilogue restores. Six string arguments trigger both. */
	test_true("DefExeFn six args keeps arena cursor independent of argv",
	          "DefineCommand \"vsix\" [:p1 :p2 :p3 :p4 :p5 :p6] " SLFFI_ECHO_ARGS_ARG " "
	          "\"\\\"%s\\\" \\\"%s\\\" \\\"%s\\\" \\\"%s\\\" \\\"%s\\\" \\\"%s\\\"\" "
	          "Make \"cap\" vsix \"one\" \"two\" \"three\" \"four\" \"five\" \"six\" "
	          "Output IsNumber Search \"argc=7\" :cap");
	test_true("DefExeFn six args: last value survives the arena intact",
	          "DefineCommand \"vsix2\" [:p1 :p2 :p3 :p4 :p5 :p6] " SLFFI_ECHO_ARGS_ARG " "
	          "\"\\\"%s\\\" \\\"%s\\\" \\\"%s\\\" \\\"%s\\\" \\\"%s\\\" \\\"%s\\\"\" "
	          "Make \"cap2\" vsix2 \"one\" \"two\" \"three\" \"four\" \"five\" \"six\" "
	          "Output IsNumber Search \"six\" :cap2");
	/* zero params: the child sees only its own argv[0] */
	test_true("DefExeFn zero args",
	          "DefineCommand \"vzero\" [] " SLFFI_ECHO_ARGS_ARG " \"\" "
	          "Make \"cap\" vzero "
	          "Output IsNumber Search \"argc=1\" :cap");
	/* empty capture interns as the empty atom (cmd /c rem prints nothing);
	   Equal returns a truthy value when both sides are the empty string */
	test_true("DefExeFn empty capture is empty string",
	          "DefineCommand \"vstill\" [] \"C:\\\\Windows\\\\System32\\\\cmd.exe\" \"/c rem\" "
	          "Make \"cap\" vstill "
	          "Output (:cap = \"\")");
	/* Trailing-backslash regression (BUG-1, Sep 2026). A value ending in a
	   backslash must be doubled ONLY when the format places its closing '"'
	   directly after that %s: the final backslash would otherwise escape the
	   format-owned quote and the CRT would never terminate the argument.
	   Both modes are covered, and BOTH directions are asserted -- the value
	   must arrive exactly (one backslash) and must NOT be over-doubled (two).
	   The needle needs the "[" that echo_args prints after "arg1=", and
	   Sec.146.24 forbids a literal "[" inside a SectorLogo string (it would
	   open a list block and swallow the rest of the script), so it is built
	   from Character 91 and concatenated with the proven 3-term `Make` infix
	   pattern. The leading "arg1=" makes the match unambiguous: the only other
	   lines carrying argv text are prefixed "arg0=" and "cmdline=", so a hit
	   can only come from the DECODED argv line, never from the raw command
	   line (which necessarily holds escaped bytes).
	   The first case is a CONTROL with a backslash-free value that must MATCH
	   the identical needle shape -- without it a mistyped needle would make
	   the "not doubled" assertions pass vacuously, and a negative test that
	   cannot fail proves nothing. */
	test_true("DefExeFn arg1 needle matches a plain value (control)",
	          "DefineCommand \"vbctl\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\"\" "
	          "Make \"cap\" vbctl \"hi\" "
	          "Make \"ob\" Character 91 Make \"n\" \"arg1=\" _ :ob _ \"hi\" "
	          "Output IsNumber Search :n :cap");
	test_true("DefExeFn quoted %s trailing backslash keeps one arg",
	          "DefineCommand \"vbq\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\"\" "
	          "Make \"cap\" vbq \"hello\\\\\" "
	          "Output IsNumber Search \"argc=2\" :cap");
	test_true("DefExeFn quoted %s trailing backslash decodes exactly",
	          "DefineCommand \"vbq2\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\"\" "
	          "Make \"cap\" vbq2 \"hello\\\\\" "
	          "Make \"ob\" Character 91 Make \"n\" \"arg1=\" _ :ob _ \"hello\\\\\" "
	          "Output IsNumber Search :n :cap");
	test_nil("DefExeFn quoted %s trailing backslash not doubled",
	         "DefineCommand \"vbq3\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\"\" "
	         "Make \"cap\" vbq3 \"hello\\\\\" "
	         "Make \"ob\" Character 91 Make \"n\" \"arg1=\" _ :ob _ \"hello\\\\\\\\\" "
	         "Output IsNumber Search :n :cap");
	/* raw %s: the format owns no closing quote, so the run must stay
	   single-width -- doubling here would corrupt the value (the mirror image
	   of the bug above) */
	test_true("DefExeFn raw %s trailing backslash keeps one arg",
	          "DefineCommand \"vbr\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"%s\" "
	          "Make \"cap\" vbr \"hello\\\\\" "
	          "Output IsNumber Search \"argc=2\" :cap");
	test_true("DefExeFn raw %s trailing backslash decodes exactly",
	          "DefineCommand \"vbr2\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"%s\" "
	          "Make \"cap\" vbr2 \"hello\\\\\" "
	          "Make \"ob\" Character 91 Make \"n\" \"arg1=\" _ :ob _ \"hello\\\\\" "
	          "Output IsNumber Search :n :cap");
	test_nil("DefExeFn raw %s trailing backslash not doubled",
	         "DefineCommand \"vbr3\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"%s\" "
	         "Make \"cap\" vbr3 \"hello\\\\\" "
	         "Make \"ob\" Character 91 Make \"n\" \"arg1=\" _ :ob _ \"hello\\\\\\\\\" "
	         "Output IsNumber Search :n :cap");

	/* Three-state quote protocol (sl_xfn_quote_follows returns -1 or M, and
	   win32_escape_inner emits 2N - M). A 1-bit flag cannot tell "no quote
	   follows" from "a quote follows M >= 1 format backslashes", and getting
	   that wrong silently produces an ODD run: the CRT then treats the '"' as
	   a literal, the quoted section never closes, and the whole rest of the
	   command line is swallowed into argv[1]. Every M >= 1 row below must
	   therefore (a) keep argc at 2 and (b) deliver the value byte-exactly --
	   values are built with Character 92 so no source-level escaping ambiguity
	   can creep in (Sec.146 fact 3 decodes \\ to one backslash).
	   M = 1 (format bytes after %s: \") */
	test_true("DefExeFn M=1 quote context keeps one arg",
	          "DefineCommand \"vm1\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\\\\\"\\\"\" "
	          "Make \"v\" \"cd\" _ Character 92 Make \"cap\" vm1 :v "
	          "Output IsNumber Search \"argc=2\" :cap");
	test_true("DefExeFn M=1 quote context decodes K=1 exactly",
	          "DefineCommand \"vm1b\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\\\\\"\\\"\" "
	          "Make \"v\" \"cd\" _ Character 92 Make \"cap\" vm1b :v "
	          "Make \"ob\" Character 91 Make \"n\" \"arg1=\" _ :ob _ :v "
	          "Output IsNumber Search :n :cap");
	test_true("DefExeFn M=1 quote context decodes K=2 exactly",
	          "DefineCommand \"vm1c\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\\\\\"\\\"\" "
	          "Make \"v\" \"ab\" _ (Character 92) _ (Character 92) Make \"cap\" vm1c :v "
	          "Make \"ob\" Character 91 Make \"n\" \"arg1=\" _ :ob _ :v "
	          "Output IsNumber Search :n :cap");
	test_true("DefExeFn M=1 quote context decodes K=4 exactly",
	          "DefineCommand \"vm1d\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\\\\\"\\\"\" "
	          "Make \"v\" \"a\" _ (Character 92) _ (Character 92) _ (Character 92) _ (Character 92) Make \"cap\" vm1d :v "
	          "Make \"ob\" Character 91 Make \"n\" \"arg1=\" _ :ob _ :v "
	          "Output IsNumber Search :n :cap");
	/* M = 2 (format bytes after %s: \\") -- the K=1/M=2 case is the
	   zero-length emit: 2*1 - 2 = 0, so the escaper writes NO backslash and
	   the format's own two supply the whole run. */
	test_true("DefExeFn M=2 quote context decodes K=1 exactly",
	          "DefineCommand \"vm2\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\\\\\\\\\"\\\"\" "
	          "Make \"v\" \"cd\" _ Character 92 Make \"cap\" vm2 :v "
	          "Make \"ob\" Character 91 Make \"n\" \"arg1=\" _ :ob _ :v "
	          "Output IsNumber Search :n :cap");
	/* M = 3 with N = 1 is the one unrepresentable shape: the format alone
	   places more backslashes than 2N can carry, so no emission can close the
	   section. That is a named FATAL, not a silently mis-parsed command line
	   (Sec.57d). */
	test_fatal_catch_case("FATAL DefExeFn format backslash run exceeds 2N",
	                      "DefineCommand \"vm3\" [:msg] " SLFFI_ECHO_ARGS_ARG " \"\\\"%s\\\\\\\\\\\\\\\"\\\"\" "
	                      "Make \"v\" \"cd\" _ Character 92 "
	                      "Output vm3 :v",
	                      "the format string places more backslashes before its closing quote");
	/* Format/param arity mismatch: v7sprintf fetches varargs with no bound,
	   so a format with MORE conversions than declared params would hand the
	   child uninitialised stack bytes (and %s would dereference them as a
	   char*). The engine must reject it at spawn time, naming the invariant. */
	test_fatal_catch_case("FATAL DefExeFn surplus conversions rejected",
	                      "DefineCommand \"varity\" [:a] " SLFFI_ECHO_ARGS_ARG " \"%s %s\" "
	                      "Output varity \"one\"",
	                      "DefineCommand format has a different number of %-conversions");
	/* Typed value in the wrong specifier slot: a NUMBER aimed at a %s slot is
	   dereferenced as a char* by v7sprintf (address 42 -> AV) -- the engine
	   must abort naming the argument instead of crashing. */
	test_fatal_catch_case("FATAL DefExeFn number into %s slot rejected",
	                      "DefineCommand \"vnumstr\" [:n] " SLFFI_ECHO_ARGS_ARG " \"%s\" "
	                      "Output vnumstr 42",
	                      "format slot does not accept this argument type");
}

/* Caught-FATAL suite (Sec.172 + Sec.173): v7s_sectorlogo_interp is gated,
   so a script FATAL unwinds to this C caller as 0 with the dead flag set
   and the reason recorded -- FATALs are testable in-process now. */
void test_fatal_catch()
{
	printf("\n--- Caught FATAL ---\n");
	test_fatal_catch_case("FATAL First of word names it",
	                      "Output First \"word\"",
	                      "FIRST requires a list, got word 'word'");
	test_fatal_catch_case("FATAL ButFirst of number names it",
	                      "Output ButFirst 42",
	                      "BUTFIRST requires a list, got number '42'");
	test_fatal_catch_case("FATAL Reverse of number names it",
	                      "Output Reverse 42",
	                      "REVERSE requires a list, got number '42'");
	test_fatal_catch_case("FATAL DefExeFn list arg names it",
	                      "DefineCommand \"vbad\" [:x] " SLFFI_ECHO_ARGS_ARG " \"%s\" "
	                      "Output vbad [1 2]",
	                      "DefineCommand argument must be atom, number, or float, got 'list'");
	test_fatal_catch_case("FATAL Sort of number",
	                      "Output Sort 42",
	                      "SORT requires a list of numbers or sub-lists");
	test_fatal_catch_case("FATAL Sort non-numeric element",
	                      "Output Sort [3 \"hello\" 5]",
	                      "a numeric value was required");
	test_fatal_catch_case("FATAL Sort empty sub-list element",
	                      "Output Sort [1 [] 3]",
	                      "numbers or sub-lists");
	/* Shared coercion FATAL is operation-neutral: NearestMatch routes
	   malformed vector/key/component through the same sl_sort_coerce_num path */
	test_fatal_catch_case("FATAL NearestMatch non-numeric vector element",
	                      "Output NearestMatch False [\"bad\"] [[1 0.0]]",
	                      "a numeric value was required");
	test_fatal_catch_case("FATAL NearestMatch non-numeric codebook key",
	                      "Output NearestMatch False [0.0] [[\"bad\" 0.0]]",
	                      "a numeric value was required");
	test_fatal_catch_case("FATAL NearestMatch non-numeric codebook component",
	                      "Output NearestMatch False [0.0] [[1 \"bad\"]]",
	                      "a numeric value was required");
	/* splitting .vq_bad_row. Two structurally different faults
	   shared one message: a codebook cell that is not a list at all, and a NIL key
	   inside a row that IS a list. "has no components" is accurate only for the
	   second -- a non-list row is not missing components, it is not a row. The
	   reproducer must be [10 20 30] (a codebook whose first CELL is a NUMBER); with
	   [7] the codebook guard above fires first and reports "requires rows" instead. */
	test_fatal_catch_case("FATAL NearestMatch non-list codebook row names the row",
	                      "Output NearestMatch True [1 2 3] [10 20 30]",
	                      "codebook rows must be lists");
	/* The NIL-KEY path needs a row that IS a list but whose key slot is NIL. With
	   [[] 1 2 3] the ROW itself is NIL, so the non-list guard fires first and the
	   key is never examined -- that is the split this pair of tests pins. */
	test_fatal_catch_case("FATAL NearestMatch NIL key still reports no components",
	                      "Output NearestMatch True [1 2 3] [[[] 1 2 3]]",
	                      "codebook row has no components");
	test_fatal_catch_case("FATAL HashNumber of number",
	                      "Output HashNumber 42",
	                      "HASH-FLOAT requires an atom (string or word)");
	/* Sec.146 fact 40: bare SELF lowercases to "self" (lexer A-Z fold) and
	   resolves nothing -- the quoting gotcha is part of the documented
	   contract; the improved FATAL names it */
	test_fatal_catch_case("FATAL C-CALL bare SELF fails",
	                      "Output CallLibrary [SELF \"v7s_get_num_workers\" \"integer\" []]",
	                      "failed to resolve function");
	/* Sec.146 fact 41: a bare PROC name containing underscores SPLITS at the
	   delimiter (lexer single-char delimiters) -- descriptor[1] becomes the
	   atom "v7s" and GetProcAddress fails. Procs must be quoted. */
	test_fatal_catch_case("FATAL C-CALL bare underscore proc fails",
	                      "Output CallLibrary [\"SELF\" v7s_get_num_workers \"integer\" []]",
	                      "failed to resolve function");
	/* Sec.173: the unknown-arg-type FATAL must name the TYPE atom, not the value
	   EvalNextArg left in eax. Type names are lowercase (data.inc:222-225) and the
	   ret-type field is validated FIRST, so an "Integer" here would be reported as a
	   RETURN-type error and never reach the arg-type site. */
	test_fatal_catch_case("FATAL C-CALL unknown arg type names the type atom",
	                      "Output CallLibrary [\"SELF\" \"v7s_get_num_workers\" \"integer\" [ \"bogus\" ]]",
	                      "got word 'bogus'");
	/* reader Sec.173: a stray ']' at top level used to intern as an atom and
	   silently evaluate to NIL -- it must FATAL naming the offender */
	test_fatal_catch_case("FATAL stray close-bracket at top level",
	                      "Output 1 ]",
	                      "stray ']'");
	/* Sec.173 type-word rendering: NumVal's FATAL now names the offender AND
	   its type -- atom '7' and NUMBER 7 render identical text, the type word
	   is the discriminator (this exact ambiguity cost a full debugging
	   session in the Lisp-interpreter example) */
	test_fatal_catch_case("FATAL Multiply of list-element word names type",
	                      "Make \"x\" First [ 7 8 ] Output (:x * 2)",
	                      "a NUMBER was required (arithmetic, index, count, or key operand), got word '7'");
	/* arithmetic builtins take FLOAT operands fine -- the float-NumVal case is
	   a REPEAT count (Sec.146: float count FATALs via NumVal) */
	test_fatal_catch_case("FATAL float REPEAT count names type",
	                      "Repeat 2.5 [ Make \"q\" 1 ]",
	                      "a NUMBER was required (arithmetic, index, count, or key operand), got float '");
	/* Sec.173 coverage for the statement handlers' offender-naming FATALs
	   (FATAL_VALUE form; each asserts the operation, the requirement AND the
	   offending value/type) */
	test_fatal_catch_case("FATAL Run of a number names it",
	                      "Run 5",
	                      "RUN requires a list or NIL, got number '5'");
	test_fatal_catch_case("FATAL Make with a list name names it",
	                      "Make [1 2] 5 Output 1",
	                      "MAKE requires a string atom as variable name, got 'list'");
	test_fatal_catch_case("FATAL Define with a list name names it",
	                      "Define [1 2] [:x] [ Output 1 ] Output 1",
	                      "Define requires a string atom as function name, got 'list'");
	test_fatal_catch_case("FATAL Local with a list name names it",
	                      "Local [1 2] Output 1",
	                      "LOCAL requires a string atom variable name, got 'list'");
	test_fatal_catch_case("FATAL DefExeFn list name names it",
	                      "DefineCommand [1 2] [:x] " SLFFI_ECHO_ARGS_ARG " \"%s\" Output 1",
	                      "DefineCommand requires a string atom as function name, got 'list'");
	test_fatal_catch_case("FATAL DefExeFn list exe path names it",
	                      "DefineCommand \"fx\" [:x] [1 2] \"%s\" Output 1",
	                      "DefineCommand requires a string atom as exe path, got 'list'");
	test_fatal_catch_case("FATAL DefExeFn list format names it",
	                      "DefineCommand \"fy\" [:x] " SLFFI_ECHO_ARGS_ARG " [1 2] Output 1",
	                      "DefineCommand requires a string atom as format string, got 'list'");
	/* Sec.173: the LOCAL-limit guardrail names the var being declared --
	   "limit exceeded" with no name gave no hint WHERE the scope stack blew */
	/* REMOVED: "FATAL LOCAL overflow names the var" (was Repeat 999999 inside a
	   DEF_FN). It only ever passed BECAUSE sl_tombstone_locals failed to restore
	   sl_local_count -- every block evaluation leaked one entry, so the count
	   climbed to the 65536 limit. Now that the restore works, a Repeat body's
	   count is correctly saved and restored per iteration, so this can never reach
	   the limit: 	   the tripwire fires only on genuine deep recursion, which overflows
	   the native stack first. The guard itself stays (it is the correct reaction to
	   an impossible-by-contract state, and Sec.57d allows a FATAL for that) -- it is
	   simply unreachable from a script, so it has no test. */
	/* recovery: after the death reset the interpreter is fully usable */
	test_num("Recovery after caught FATALs",
	         "Make \"X\" 20 "
	         "Output (:X + 22)",
	         42);
	/* sl_lg_initialized latch reset (main ITEM 1).
	   The death reset must re-arm the init latch, so the NEXT script re-runs
	   lg_init on a fresh arena instead of dispatching against the world the
	   FATAL left half-mutated. The observable is the atom path: a stale START
	   bit sends the Intern scan into uninitialised bytes and yields the wrong
	   atom for a name, so a name that has to be interned and dispatched after
	   the FATAL is the check. It also fails loudly if the re-init itself is
	   broken (a fresh arena with unwritten intern caches), which is what the
	   cache wipes in intern_builtins exist for. */
	test_fatal_catch_case("caught FATAL leaves the next script's atom interning intact",
	                      "Output (6 * \"notanumber\")",
	                      "NUMBER");
	test_num("interned name still dispatches after a caught FATAL",
	         "Make \"probe\" \"someuniqueinternedname\" "
	         "Output TextLength :probe",
	         22);
}

/* GC stress tests: allocate many cons cells via Prepend and MAKE in REPEAT loops,
   forcing the garbage collector to run, then verify correctness of the result. */
void test_gc_stress()
{
	printf("\n--- GC stress (cons cell allocation) ---\n");

	/* 1. Build a list of 500 elements via REPEAT + FPUT, then read FIRST.
	   Each Prepend allocates 1 cons cell. 500 is well beyond the initial
	   free-set; with env bindings per iteration, GC must fire. */
	test_num("GC: 500 Prepend list",
	         "Make \"L\" [] "
	         "Make \"I\" 0 "
	         "Repeat 500 [ "
	         "  Make \"L\" Prepend :I :L "
	         "  Make \"I\" :I + 1 "
	         "] "
	         "Output First :L", /* last prepended = 499 */
	         499);

	/* 2. Build list, verify length by walking with REPEAT + FIRST/BUTFIRST */
	test_num("GC: list length 200",
	         "Make \"L\" [] "
	         "Make \"I\" 0 "
	         "Repeat 200 [ "
	         "  Make \"L\" Prepend :I :L "
	         "  Make \"I\" :I + 1 "
	         "] "
	         "Make \"LEN\" 0 "
	         "Make \"CUR\" :L "
	         "Repeat 200 [ "
	         "  Make \"LEN\" :LEN + 1 "
	         "  Make \"CUR\" ButFirst :CUR "
	         "] "
	         "Output :LEN",
	         200);

	/* 3. Many MAKE bindings in a loop (each MAKE = 2 cons cells for env).
	   After the loop, verify the variable still holds the correct value. */
	test_num("GC: 300 Make bindings",
	         "Make \"ACC\" 0 "
	         "Repeat 300 [ "
	         "  Make \"ACC\" :ACC + 1 "
	         "] "
	         "Output :ACC",
	         300);

	/* 4. Nested REPEAT: allocate in inner loop, discard in outer.
	   Each iteration of the outer loop creates 100 cons cells that go
	   out of scope when the next outer iteration replaces :L. */
	test_num("GC: nested Repeat discard",
	         "Make \"RESULT\" 0 "
	         "Repeat 10 [ "
	         "  Make \"L\" [] "
	         "  Repeat 100 [ "
	         "    Make \"L\" Prepend 42 :L "
	         "  ] "
	         "  Make \"RESULT\" :RESULT + (First :L) "
	         "] "
	         "Output :RESULT",
	         420);

	/* 5. Function calls in a loop: each call allocates cons cells for
	   the function descriptor + env bindings. Verify return value. */
	test_num("GC: function call 200x",
	         "Define \"inc\" [:x] [ Output :x + 1 ] "
	         "Make \"V\" 0 "
	         "Repeat 200 [ "
	         "  Make \"V\" inc :V "
	         "] "
	         "Output :V",
	         200);

	/* 6. Deep list ruction: build a list of 1000 elements, then
	   verify the 500th element by walking with BUTFIRST. */
	test_num("GC: deep list 1000 elements",
	         "Make \"L\" [] "
	         "Make \"I\" 999 "
	         "Repeat 1000 [ "
	         "  Make \"L\" Prepend :I :L "
	         "  Make \"I\" :I - 1 "
	         "] "
	         "Make \"C\" :L "
	         "Repeat 500 [ "
	         "  Make \"C\" ButFirst :C "
	         "] "
	         "Output First :C",
	         500);
}

/* Complex multi-step Logo programs: real-world-style programs with
   multiple functions, loops, conditionals, and state. */
/* Regression: sl_stmt_fn_retval must be an exact GC root.
   The statement evaluator publishes a user/xfn return value into that .bss slot
   (sectorlogo_eval.inc:185/192) and the value-position Run reads it only AFTER
   the whole block has finished (sectorlogo_eval_primary.inc:343). The slot is
   outside the conservative stack scan range, so without an explicit root a
   GC_Collect inside the block swept it: the caller read a cell with car=0 and
   cdr=freelist link, and the next Cons handed the same cell to an unrelated
   owner. The block below publishes a 3-element list and then allocates hard
   enough to exhaust the 1M-cell cons heap, so GC fires between publish and read. */
void test_gc_stmt_fn_retval_root()
{
	char script[512];

	printf("\n--- GC: statement-fn return value is a GC root ---\n");

	/* mk3 returns a freshly consed [1 2 3]; the block then allocates ~1.2M cons
	   cells so GC_Collect runs while the published value is the ONLY reference
	   to those three cells. Each iteration rebinds J to a FRESH 3-cell list, so
	   the churn is garbage the collector can actually reclaim -- a self-growing
	   ":J" would instead pin every cell it made and simply exhaust the 1M-cell
	   heap ("Cons heap overflow") without ever exercising a sweep. */
	sprintf(script,
	        "Define \"mk3\" [] [ Output Prepend 3 Prepend 2 Prepend 1 [] ] "
	        "Make \"R\" Run [ mk3 Make \"J\" 0 Repeat 1200000 [ Make \"J\" Prepend 1 [1 2] ] ] "
	        "Output Item :R 1");
	test_num("statement-fn return list survives a GC inside the same block", script, 2);

	/* The head element too -- a swept list can leave the car intact while the
	   cdr walks into the freelist, so probe element 0 as well. mk3 returns
	   Prepend 3 (Prepend 2 (Prepend 1 [])), i.e. [3 2 1], so the head is 3. */
	sprintf(script,
	        "Define \"mk3\" [] [ Output Prepend 3 Prepend 2 Prepend 1 [] ] "
	        "Make \"R\" Run [ mk3 Make \"J\" 0 Repeat 1200000 [ Make \"J\" Prepend 1 [1 2] ] ] "
	        "Output Item :R 0");
	test_num("statement-fn return list head survives a GC inside the same block", script, 3);

	/* Length is the sharpest probe: it walks the whole cdr chain, so any
	   freelist link anywhere in it changes the answer. */
	sprintf(script,
	        "Define \"mk3\" [] [ Output Prepend 3 Prepend 2 Prepend 1 [] ] "
	        "Make \"R\" Run [ mk3 Make \"J\" 0 Repeat 1200000 [ Make \"J\" Prepend 1 [1 2] ] ] "
	        "Output Length :R");
	test_num("statement-fn return list length survives a GC inside the same block", script, 3);
}

/* sl_atom_gc granule-units fix: the sweep cursor and
   the span advances must both be GRANULES. The pre-fix bug advanced the cursor
   in BYTES (8x the required value), so the very first span advance overshot
   the bound comparison, the trailing-gap path declared everything above the
   false stop dead, rewound sl_atom_free over LIVE atoms, and later interning
   silently overwrote their text. The script below interns ~61 MiB of distinct
   long names so the atom cursor crosses SL_ATOM_GC_SOFT_OFF (48 MiB) and the
   gated atom sweep actually runs. :MK is interned at ~7.7 MiB -- above the
   ~6.3 MiB mark where the byte-units bug falsely ended the sweep -- and stays
   bound (live) across the collect. The post-collect churn then reuses any
   wrongly-handed-back space, so a corrupted sweep overwrites :MK's text and
   the final Equal fails; a correct sweep keeps :MK byte-identical and the
   re-intern of the same name dedups to the same atom. */
void test_atom_gc_survival()
{
	printf("\n--- Atom GC sweep survival (soft-limit collect) ---\n");

	int prefix_len = 3500;
	int churn_marker = 2200;      /* marker lands ~7.7 MiB, above the false-stop line */
	int churn_main = 11800;       /* carries the cursor past the 48 MiB soft limit */
	int churn_post = 3500;        /* overwrites the marker region if it was wrongly freed */
	int script_len = prefix_len + 256;

	char *script = (char *)malloc(script_len);
	if (!script)
	{
		failing_tests_counter++;
		printf("  FAIL: atom GC survival (malloc %d bytes failed)\n", script_len);
		return;
	}

	char *cursor = script;
	cursor += sprintf(cursor, "Make \"P\" \"");
	for (int i = 0; i < prefix_len; i++)
	{
		*cursor++ = 'A';
	}
	cursor += sprintf(cursor, "\" Make \"I\" 0 ");
	cursor += sprintf(cursor, "Repeat %d [ Make \"T\" :P _ :I Make \"I\" :I + 1 ] ", churn_marker);
	cursor += sprintf(cursor, "Make \"MK\" :P _ \"mk\" ");
	cursor += sprintf(cursor, "Repeat %d [ Make \"T\" :P _ :I Make \"I\" :I + 1 ] ", churn_main);
	cursor += sprintf(cursor, "Repeat %d [ Make \"T\" :P _ :I Make \"I\" :I + 1 ] ", churn_post);
	cursor += sprintf(cursor, "Make \"EXP\" :P _ \"mk\" ");
	cursor += sprintf(cursor, "Output IfElse (:MK = :EXP) [ 1 ] [ 0 ]");
	*cursor = 0;
	test_num("live atom survives the soft-limit atom sweep", script, 1);
	free(script);
}

void test_complex_programs()
{
	printf("\n--- Complex programs ---\n");

	/* 1. Fibonacci: iterative, using two accumulators */
	test_num("fib(10)",
	         "Make \"A\" 0 "
	         "Make \"B\" 1 "
	         "Repeat 10 [ "
	         "  Make \"T\" :B "
	         "  Make \"B\" :A + :B "
	         "  Make \"A\" :T "
	         "] "
	         "Output :A",
	         55);

	/* 2. Factorial: iterative multiply */
	test_num("factorial(6)",
	         "Make \"R\" 1 "
	         "Make \"I\" 1 "
	         "Repeat 6 [ "
	         "  Make \"R\" :R * :I "
	         "  Make \"I\" :I + 1 "
	         "] "
	         "Output :R",
	         720);

	/* 3. Sum of 1..N using REPEAT */
	test_num("sum 1..100",
	         "Make \"S\" 0 "
	         "Make \"I\" 1 "
	         "Repeat 100 [ "
	         "  Make \"S\" :S + :I "
	         "  Make \"I\" :I + 1 "
	         "] "
	         "Output :S",
	         5050);

	/* 4. GCD using Euclidean algorithm (loop with modulus) */
	test_num("GCD(48,18)",
	         "Make \"A\" 48 "
	         "Make \"B\" 18 "
	         "Repeat 20 [ "
	         "  IfElse (:B > 0) [ "
	         "    Make \"T\" :A % :B "
	         "    Make \"A\" :B "
	         "    Make \"B\" :T "
	         "  ] [ Stop ] "
	         "] "
	         "Output :A",
	         6);

	/* 5. Power function: compute 2^10 via REPEAT */
	test_num("2^10 via Repeat",
	         "Make \"R\" 1 "
	         "Repeat 10 [ "
	         "  Make \"R\" :R * 2 "
	         "] "
	         "Output :R",
	         1024);

	/* 6. Collatz sequence: count steps from n=27 to 1 */
	test_num("Collatz(27) steps",
	         "Make \"N\" 27 "
	         "Make \"C\" 0 "
	         "Repeat 200 [ "
	         "  IfElse (:N = 1) [ Stop ] [ "
	         "    IfElse ((:N % 2) = 0) [ "
	         "      Make \"N\" :N / 2 "
	         "    ] [ "
	         "      Make \"N\" (:N * 3) + 1 "
	         "    ] "
	         "    Make \"C\" :C + 1 "
	         "  ] "
	         "] "
	         "Output :C",
	         111);

	/* 7. Recursive-style with Define: factorial as a function */
	test_num("Define factorial",
	         "Define \"fact\" [:n] [ "
	         "  IfElse (:n < 2) [ Output 1 ] [ "
	         "    Output :n * (fact (:n - 1)) "
	         "  ] "
	         "] "
	         "Output fact 7",
	         5040);

	/* 8. String atoms in list: build list of quoted atoms, verify FIRST is truthy */
	test_true("list of string atoms", "Make \"L\" [] "
	                                  "Make \"L\" Prepend \"world\" :L "
	                                  "Make \"L\" Prepend \"hello\" :L "
	                                  "Output ((First :L) = (First :L))");

	/* 9. Multiple function definitions and mutual calls */
	test_num("double then add1",
	         "Define \"dbl\" [:x] [ Output :x * 2 ] "
	         "Define \"add1\" [:x] [ Output :x + 1 ] "
	         "Output add1 dbl 5",
	         11);

	/* 10. Nested IFELSE with arithmetic */
	test_num("classify number",
	         "Make \"X\" 42 "
	         "IfElse (:X < 10) [ "
	         "  Make \"C\" 1 "
	         "] [ IfElse (:X < 50) [ "
	         "  Make \"C\" 2 "
	         "] [ "
	         "  Make \"C\" 3 "
	         "] ] "
	         "Output :C",
	         2);

	/* 11. Digit sum: sum digits of 12345 */
	test_num("digit sum 12345",
	         "Make \"N\" 12345 "
	         "Make \"S\" 0 "
	         "Repeat 5 [ "
	         "  Make \"S\" :S + (:N % 10) "
	         "  Make \"N\" :N / 10 "
	         "] "
	         "Output :S",
	         15);

	/* 12. Prime check: is 17 prime? (trial division) */
	test_num("is 17 prime",
	         "Make \"N\" 17 "
	         "Make \"P\" 1 "
	         "Make \"D\" 2 "
	         "Repeat 15 [ "
	         "  IfElse ((:D * :D) > :N) [ Stop ] [ "
	         "    IfElse ((:N % :D) = 0) [ "
	         "      Make \"P\" 0 "
	         "      Stop "
	         "    ] [ "
	         "      Make \"D\" :D + 1 "
	         "    ] "
	         "  ] "
	         "] "
	         "Output :P",
	         1);
}

/* Long string tests: verify operations work with strings > 512 bytes
   (the old sl_concat_buf limit) and > 64 bytes (the old sl_tok_buf limit).
   Each test builds a long string via REPEAT + _ concat, then operates on it. */
void test_long_strings()
{
	printf("\n--- Long strings (> 512 bytes) ---\n");

	/* Build a 1000-byte string of "A"s via concat, verify it's a valid atom.
	   999 iterations of _ "A" on a 1-byte start produces 1000 bytes.
	   Before the fix, this would overflow the 512-byte sl_concat_buf. */
	test_atom("1000-byte concat atom", "Make \"S\" \"A\" "
	                                   "Repeat 999 [ Make \"S\" :S _ \"A\" ] "
	                                   "Output :S");

	/* SUBSTR on a 1000-byte string: extract 10 bytes starting at position 500.
	   Result should equal "AAAAAAAAAA" (10 A's). */
	test_true("Part on 1000-byte string", "Make \"S\" \"A\" "
	                                        "Repeat 999 [ Make \"S\" :S _ \"A\" ] "
	                                        "Output ((Part :S 500 10) = \"AAAAAAAAAA\")");

	/* Search on a 1000-byte string: find "AAA" starting at position 0.
	   Should return NUMBER(0). */
	test_num("Search on 1000-byte string",
	         "Make \"S\" \"A\" "
	         "Repeat 999 [ Make \"S\" :S _ \"A\" ] "
	         "Output Search \"AAA\" :S",
	         0);

	/* FIND on a 1000-byte string: find a substring near the end.
	   Insert a marker "XYZ" at position 990 by splitting the build. */
	test_num("Search marker near end of long string",
	         "Make \"S\" \"A\" "
	         "Repeat 989 [ Make \"S\" :S _ \"A\" ] "
	         "Make \"S\" :S _ \"XYZ\" "
	         "Repeat 7 [ Make \"S\" :S _ \"A\" ] "
	         "Output Search \"XYZ\" :S",
	         990);

	/* STRLOWER on a 1000-byte string of uppercase letters.
	   Result should be 1000 lowercase 'a's. Verify first 10 match "aaaaaaaaaa". */
	test_true("Lowercase on 1000-byte string", "Make \"S\" \"A\" "
	                                          "Repeat 999 [ Make \"S\" :S _ \"A\" ] "
	                                          "Output ((Part (Lowercase :S) 0 10) = \"aaaaaaaaaa\")");

	/* STRUPPER on a 1000-byte string of lowercase letters.
	   Result should be 1000 uppercase 'A's. Verify first 10 match "AAAAAAAAAA". */
	test_true("Uppercase on 1000-byte string", "Make \"S\" \"a\" "
	                                          "Repeat 999 [ Make \"S\" :S _ \"a\" ] "
	                                          "Output ((Part (Uppercase :S) 0 10) = \"AAAAAAAAAA\")");

	/* Concatenation of two > 512 byte strings: total > 1024 bytes.
	   Build two 600-byte strings, concat them, verify result is an atom. */
	test_atom("concat two 600-byte strings", "Make \"A\" \"X\" "
	                                         "Repeat 599 [ Make \"A\" :A _ \"X\" ] "
	                                         "Make \"B\" \"Y\" "
	                                         "Repeat 599 [ Make \"B\" :B _ \"Y\" ] "
	                                         "Output :A _ :B");

	/* SUBSTR from near the end of a 1000-byte string.
	   Extract 5 bytes starting at position 995 (last 5 bytes). */
	test_true("Part near end of long string", "Make \"S\" \"A\" "
	                                            "Repeat 999 [ Make \"S\" :S _ \"A\" ] "
	                                            "Output ((Part :S 995 5) = \"AAAAA\")");

	/* Verify long string content at multiple positions via SUBSTR + EQUALP.
	   Build a string of alternating "AB" pairs (1000 bytes = 500 AB pairs).
	   Check position 0 = "AB", position 500 = "AB", position 998 = "AB". */
	test_true("long string content at positions", "Make \"S\" \"AB\" "
	                                              "Repeat 499 [ Make \"S\" :S _ \"AB\" ] "
	                                              "Output ((Part :S 0 2) = \"AB\")");

	test_true("long string content at middle", "Make \"S\" \"AB\" "
	                                           "Repeat 499 [ Make \"S\" :S _ \"AB\" ] "
	                                           "Output ((Part :S 500 2) = \"AB\")");

	test_true("long string content at end", "Make \"S\" \"AB\" "
	                                        "Repeat 499 [ Make \"S\" :S _ \"AB\" ] "
	                                        "Output ((Part :S 998 2) = \"AB\")");

	/* Full-string h2 property (aes_string_hash): two 100-byte strings sharing
	   a 64-byte prefix but differing at byte 80 must intern to DISTINCT atoms
	   that each round-trip their own content. Under the old 64-byte-prefix
	   hash these two had IDENTICAL h2; the linear scan masked it, but this
	   test now proves the hash itself separates them. */
	test_true("full-string h2: 100-char twins stay distinct", "Make \"A\" \"\" "
	                                        "Repeat 100 [ Make \"A\" :A _ \"A\" ] "
	                                        "Make \"B\" \"\" "
	                                        "Repeat 79 [ Make \"B\" :B _ \"A\" ] "
	                                        "Make \"B\" :B _ \"B\" "
	                                        "Repeat 20 [ Make \"B\" :B _ \"A\" ] "
	                                        "Output ((! (:A = :B)) && ((Part :B 79 1) = \"B\"))");
	test_true("full-string h2: twin A round-trips its own byte 80", "Make \"A\" \"\" "
	                                        "Repeat 100 [ Make \"A\" :A _ \"A\" ] "
	                                        "Make \"B\" \"\" "
	                                        "Repeat 79 [ Make \"B\" :B _ \"A\" ] "
	                                        "Make \"B\" :B _ \"B\" "
	                                        "Repeat 20 [ Make \"B\" :B _ \"A\" ] "
	                                        "Output ((Part :A 79 1) = \"A\")");
}

/* ====== AUTOEXEC.LOGO TESTS ====== */

void test_autoexec()
{
	printf("\n--- AUTOEXEC.LOGO ---\n");
	/* The repo-root autoexec.logo IS the stdlib under test (created by
	   the console feature, plan Sec.5). The validator seeds it into the
	   database BEFORE the first interp call (see main), so lg_init takes
	   the __sectorlogo__autoexec path -- the same path production uses --
	   and every caught-FATAL reload is a hash lookup, not a disk read.
	   The assertions below prove the DB-backed stdlib built the shared
	   base env (Sec.3.9). */

	/* The very first v7s_sectorlogo_interp call triggers lg_init, which
	   loads the stdlib and evaluates it once into sl_lg_base_env. */
	test_num("stdlib PRINT def available (via ansi)", "Output ForegroundCode 3", 33);

	/* Second call: base env persists (lg_init runs once, Sec.3.9). */
	test_num("stdlib fn persists across calls", "Output ForegroundCode 12", 94);

	/* stdlib fn works inside infix expressions (arg evaluation) */
	test_num("stdlib fn inside +", "Output ((ForegroundCode 5) + 1)", 36);
}

/* ====== CONSOLE PRIMITIVES TESTS (plan Sec.9) ====== */

void test_console()
{
	printf("\n--- Console Primitives ---\n");
	/* ---- pure primitives ---- */
	test_num("Number", "Output Number \"A\"", 65);
	test_num("Number empty", "Output Number \"\"", 0);
	test_num("Number NIL", "Output Number []", 0);
	test_num("Character roundtrip", "Output Number Character 200", 200);
	test_true("Character atom eq", "Output ((Character 65) = \"A\")");
	/* Word atoms must FATAL: before the SL_TYPE_STRING check, a bare
	   word silently returned its first text byte (Number hello -> 'h' = 104) */
	test_fatal_catch_case("FATAL Number on word atom", "Output Number hello", "requires a string atom");
	test_fatal_catch_case("FATAL Number on NUMBER arg", "Output Number 42", "requires a string atom");
	/* Sec.173: the CHAR-TO-INT type FATAL must NAME the offending value, not
	   just state the requirement. A type error that renders identically for
	   word 'hello' and NUMBER 42 sends the reader hunting for the wrong bug. */
	test_fatal_catch_case("FATAL Number on word names the offender", "Output Number hello", "got word 'hello'");
	test_fatal_catch_case("FATAL Number on NUMBER names the offender", "Output Number 42", "got number '42'");
	/* Sec.173: the INT-TO-CHAR range FATAL must name the rejected code point.
	   The tagged value is preserved across NumVal into r10d, so the render
	   shows the actual argument the script passed. */
	test_fatal_catch_case("FATAL Character 256 names the offender", "Output Character 256", "got number '256'");
	test_fatal_catch_case("FATAL Character range text", "Output Character 256", "Character requires a NUMBER in the range 1..255");
	test_fatal_catch_case("FATAL Character -1 names the offender", "Output Character -1", "got number '-1'");
	/* Character 0 is rejected, not accepted: NUL-terminated storage cannot
	   hold a NUL, so it would return an atom bit-identical to the empty-string
	   literal and the script could not tell them apart. */
	test_fatal_catch_case("FATAL Character 0 names the offender", "Output Character 0", "got number '0'");
	test_num("Character 1 is a one-char string", "Output TextLength Character 65", 1);
	test_num("Character 255 is the top of the range", "Output TextLength Character 255", 1);
	/* ---- console-dependent: smoke only (must not FATAL / block) ----
	   IFELSE is a STATEMENT in this evaluator, not an expression, so the
	   probe runs as a statement and a separate OUTPUT reports success. */
	test_num("EnableColors smoke", "EnableColors Output 1", 1);
	test_num("ReadKey smoke", "ReadKey Output 1", 1);
	/* The CONSOLE arm of ReadKey is unreachable from here: this process's stdin is
	   never a real console (suite 75 redirects it to a pipe, and every other run
	   inherits a file), so no script can drive PeekConsoleInputA. Its coverage is
	   therefore the assembly-time drift gates in sectorlogo_const.inc, which pin
	   SL_IR_SIZE to sizeof(INPUT_RECORD) and SL_IR_KEY_ASCII_OFF to the absolute
	   offset of KEY_EVENT_RECORD.uChar -- the two layout facts the arm's buffer
	   size and its key-acceptance test depend on. Adding a script test here could
	   never fail, which would be false coverage (Sec.186 rule 6). */
	/* ---- PRINT via the stdlib function (statement position, user-fn path) ---- */
	test_num("PRINT via stdlib", "PRINT \"x\" Output 1", 1);
	/* ---- nearest16 (palette math, hand-derived expectations) ---- */
	test_num("NearestColor exact red", "Output NearestColor 255 0 0", 12);
	test_num("NearestColor exact black", "Output NearestColor 0 0 0", 0);
	test_num("NearestColor off-white", "Output NearestColor 230 230 230", 15);
	test_num("NearestColor orange", "Output NearestColor 255 165 0", 14);
	/* ---- ansi code helpers ---- */
	test_num("ForegroundCode low", "Output ForegroundCode 3", 33);
	test_num("ForegroundCode high", "Output ForegroundCode 12", 94);
	test_num("BackgroundCode low", "Output BackgroundCode 2", 42);
	test_num("BackgroundCode high", "Output BackgroundCode 9", 101);
	/* ---- NearestMatch (pure primitive -- no stdlib needed) ----
	   Flag literals MUST be uppercase TRUE/FALSE: bare words self-evaluate
	   to their interned atoms; the engine compares atom_sl_true/atom_sl_false
	   directly (plan Sec.3.8) and lowercase words intern differently. */
	test_num("NearestMatch min key", "Output First NearestMatch False [0.2 0.3 0.4] [[666 0.2 0.3 0.4] [1 1.0 1.0 1.0]]", 666);
	test_num("NearestMatch min key swap", "Output First NearestMatch False [1.0 1.0 1.0] [[666 0.2 0.3 0.4] [1 1.0 1.0 1.0]]", 1);
	test_true("NearestMatch dist zero", "Output ((First ButFirst (NearestMatch False [0.2 0.3 0.4] [[666 0.2 0.3 0.4] [1 1.0 1.0 1.0]])) = 0.0)");
	test_num("NearestMatch full first key", "Output First Item NearestMatch True [0.2 0.3 0.4] [[666 0.2 0.3 0.4] [1 1.0 1.0 1.0]] 0", 666);
	test_num("NearestMatch full second key", "Output First Item NearestMatch True [0.2 0.3 0.4] [[666 0.2 0.3 0.4] [1 1.0 1.0 1.0]] 1", 1);
	test_num("NearestMatch int vector", "Output First NearestMatch False [255 0 0] [[12 255 0 0] [0 0 0 0]]", 12);
	/* n=9 regression: main-loop bound must be a BYTE bound ((n&~7)*4), not the
	   float count -- with the bug, float 8 of every row is never read and row 1
	   (true dist 100^2) falsely wins over row 2 (true dist 1). */
	test_num("NearestMatch n9 main-loop byte bound", "Output First NearestMatch False [0 0 0 0 0 0 0 0 0] [[42 0 0 0 0 0 0 0 0 100] [7 1 0 0 0 0 0 0 0 0]]", 7);
	/* n=12 regression: tail must start at byte (n&~7)*4 and read the remaining
	   n&7 floats -- with the bug the tail re-scored floats 2..5 (already read)
	   and never read floats 6..11. Vector = 12 zeros; row1 = [42, 0*11, 100]:
	   correct dist = 10000; row2 = [7, 1, 0*11]: correct dist = 1 -> key 7 wins. */
	test_num("NearestMatch n12 tail byte offset", "Output First NearestMatch False [0 0 0 0 0 0 0 0 0 0 0 0] [[42 0 0 0 0 0 0 0 0 0 0 0 100] [7 1 0 0 0 0 0 0 0 0 0 0 0]]", 7);
	/* .vq_dist_hsum reduction. Both assert the DISTANCE for
	   n >= 8, which is the only way to catch a reduction defect -- the key-only
	   assertions cannot, because the corruption is additive and never changes the argmin
	   for separated rows.
	   The broken reduction returned 2*((x0 + x4) + row4): vmovhlps left the stale row low
	   half of xmm1 in place (so lanes 0:1 got row4/row5 instead of lanes 2:3) and
	   vmovshdup duplicated lane 0 instead of lane 1.
	   Test 1 isolates defect 2 (vmovshdup duplicates lane 0): row4 = 0, x0 = 1, so the
	   only error is the doubling -> 2.0 with the bug, 1.0 after.
	   Test 2 isolates defect 1 (vmovhlps leaves the stale row low half in place): the
	   non-zero row component must be at index 4 or 5, which is what the broken shuffle
	   reads -- putting it at 6 or 7 changes nothing. Here x4 = 1 and row4 = 1 -> 4.0 with
	   the bug (2.0 with defect 1 alone, 1.0 after both).
	   Both rows carry the key plus all n = 9 components: a short row is a documented FATAL
	   (.vq_bad_rowlen), not a distance.
	   `First ButFirst`, not `ButFirst`: the non-full winner is the IMPROPER pair
	   (key . (dist)) built at .vq_nonfull_rebuild, so its cdr is the one-element list
	   (dist) -- the same shape every pre-existing distance probe uses. */
	test_float_val("NearestMatch n9 distance (hsum lane-0 doubling)",
	               "Output First ButFirst NearestMatch False [0 0 0 0 0 0 0 0 0] [[7 1 0 0 0 0 0 0 0 0]]",
	               1.0f);
	test_float_val("NearestMatch n9 distance (hsum stale row low half)",
	               "Output First ButFirst NearestMatch False [0 0 0 0 0 0 0 0 0] [[7 0 0 0 0 1 0 0 0 0]]",
	               1.0f);
	/* exact binary32 winner compare: farow dist = (1+2^-23)^2 rounds to
	   0x3F800002, nearow dist = 1.0 = 0x3F800000 -- 2 ULP apart, BOTH pack
	   to the same 30-bit FLOAT (0x3F800002>>2 == 0x3F800000>>2), so the old
	   tagged compare saw a tie and kept the first row (key 2); the raw
	   VQ_BESTF binary32 slot must let the truly nearer row (key 1) win */
	test_num("NearestMatch exact-distance tie-break",
	         "Make \"eps\" 0.0 - (Power 2.0 -23.0) "
	         "Make \"farow\" Prepend 2 Prepend :eps [] "
	         "Make \"nearow\" Prepend 1 Prepend 0.0 [] "
	         "Make \"codebook\" Prepend :farow Prepend :nearow [] "
	         "Output First NearestMatch False [1.0] :codebook",
	         1);
}

/* ====== STR-TO-NUM / EXPLODE / LINDEX TESTS ====== */

void test_str_to_num()
{
	printf("\n--- TextToNumber ---\n");
	// atom-path parse (function results bypass the literal auto-parse)
	test_num("TextToNumber atom single digit", "Output TextToNumber Part \"7x\" 0 1", 7);
	test_num("TextToNumber atom multi digit", "Output TextToNumber Part \"42x\" 0 2", 42);
	// concat-with-NUMBER content (regression: lg_value_to_str r8 bounds bug
	// silently dropped NUMBER operands from _ concat)
	test_true("concat pair NUMBER content", "Make \"S\" \"4\" _ \"2\" Output Equal Part :S 1 1 \"2\"");
	test_true("concat chain NUMBER content", "Make \"S\" \"1\" _ \"2\" _ \"3\" Output Equal Part :S 2 1 \"3\"");
	test_num("TextToNumber on concat+substr", "Make \"S\" \"4\" _ \"2\" Output TextToNumber Part :S 0 2", 42);
	test_num("TextToNumber integer", "Make \"S\" \"4\" _ \"2\" Output TextToNumber :S", 42);
	test_num("TextToNumber zero", "Output TextToNumber \"0\"", 0);
	test_num("TextToNumber large", "Make \"S\" \"9\" _ \"99999\" Output TextToNumber :S", 999999);
	test_nil("TextToNumber empty", "Output TextToNumber \"\"");
	test_nil("TextToNumber non-numeric", "Output TextToNumber \"abc\"");
	test_nil("TextToNumber mixed", "Make \"S\" \"1\" _ \"2abc\" Output TextToNumber :S");
	test_num("TextToNumber from concat", "Make \"S\" \"1\" _ \"2\" _ \"3\" Output TextToNumber :S", 123);
	test_float_val("TextToNumber float", "Make \"S\" \"3\" _ \".14\" Output TextToNumber :S", 3.14f);
	test_float_val("TextToNumber float negative", "Make \"D\" Part \"x-y\" 1 1 Make \"S\" :D _ \"1.5\" Output TextToNumber :S", -1.5f);
	test_float_val("TextToNumber 0.5", "Make \"S\" \"0\" _ \".5\" Output TextToNumber :S", 0.5f);
	test_nil("TextToNumber trailing dot", "Make \"S\" \"3\" _ \".\" Output TextToNumber :S");
	test_nil("TextToNumber dot alone", "Output TextToNumber \".\"");
}

void test_explode()
{
	printf("\n--- Split ---\n");
	test_true("Split space split", "Output ((First Split \"hello world\" \" \") = \"hello\")");
	test_true("Split comma split", "Output ((First Split \"a,b,c\" \",\") = \"a\")");
	test_true("Split second element", "Output ((First ButFirst Split \"a,b,c\" \",\") = \"b\")");
	test_true("Split third element", "Output ((First ButFirst ButFirst Split \"a,b,c\" \",\") = \"c\")");
	test_true("Split chars empty delim", "Output ((First Split \"abc\" \"\") = \"a\")");
	test_true("Split chars 2nd", "Output ((First ButFirst Split \"abc\" \"\") = \"b\")");
	test_true("Split chars 3rd", "Output ((First ButFirst ButFirst Split \"abc\" \"\") = \"c\")");
	test_true("Split empty source first is empty", "Output ((First Split \"\" \",\") = \"\")");
	/* %%adv NUL-mask test in SL_FIRSTCHAR_SCAN. %%more has
	   two predecessors: a NUL-free window, where advancing 16 is right, and the
	   %%zn path, where bzhi has already masked every candidate at/after a PROVED
	   NUL. %%adv advanced 16 for both, so draining a NUL-bearing window resumed the
	   scan inside the FOLLOWING atom -- where a same-first-char/second-char pair
	   matched and Search/Split/Replace reported an offset past the haystack's
	   length (and Replace then computed a wrapped suffix_len and died with a bogus
	   2 MiB concat-buffer overflow). Every case below is a needle that shares its
	   first character with the haystack but cannot match inside it. */
	test_nil("Search past-NUL: non-matching 2-char needle in a 2-char haystack",
	         "Output Search \"ac\" \"aa\"");
	test_num("Split past-NUL: yields exactly one element, the whole haystack",
	         "Output Length Split \"aa\" \"ac\"", 1);
	test_true("Split past-NUL: that element is the haystack, unmangled",
	          "Output ((First Split \"aa\" \"ac\") = \"aa\")");
	test_true("Replace past-NUL: leaves the haystack unchanged",
	          "Output ((Replace \"aa\" \"ac\" \"Z\") = \"aa\")");
	/* The control: a genuine in-string match with the same first character still
	   works, so the new NUL-mask test did not stop the scan one window early. */
	test_num("Search still finds a real match past the first candidate",
	         "Output Search \"ac\" \"aac\"", 1);
	test_num("Split still splits on a real delimiter",
	         "Output Length Split \"aaXbb\" \"X\"", 2);
}

void test_str_tokens()
{
	printf("\n--- Words ---\n");
	test_num("Words word count", "Output Length Words \"cats chew grain\"", 3);
	test_true("Words first word", "Output ((First Words \"cats chew grain\") = \"cats\")");
	test_true("Words fold keeps every token", "Output ((Item Words \"The cat is on the mat\" 1) = \"cat\")");
	test_num("Words single chars kept", "Output Length Words \"a i 1\"", 3);
	test_num("Words single-char token count", "Output Length Words \"x\"", 1);
	test_true("Words single char in text order", "Output ((Item Words \"a bb\" 0) = \"a\")");
	test_true("Words single char last in text order", "Output ((Item Words \"bb a\" 1) = \"a\")");
	test_num("Words empty string count", "Output Length Words \"\"", 0);
	test_nil("Words empty string", "Output Words \"\"");
	test_nil("Words NIL arg", "Output Words []");
	test_num("Words punctuation boundaries", "Output Length Words \"cats, chew! grain?\"", 3);
	test_num("Words digits are token chars", "Output Length Words \"q8 123 4x\"", 3);
	test_true("Words fold inside digit word", "Output ((First Words \"Q8\") = \"q8\")");
	test_num("Words fold + intern identity", "Output Length Words \"Cats CATS cats\"", 3);
	test_true("Words repeat-call determinism", "Make \"A\" Words \"one two\" Make \"B\" Words \"one two\" Output IfElse ((First :A) = (First :B)) [1] [0]");
	test_num("Words mixed text", "Output Length Words \"Hello, World! 123\"", 3);
	test_true("Words text-order output", "Output ((Item Words \"ab cd\" 1) = \"cd\")");
	/* staged-scan parity cases (single-pass rework): output must be identical */
	test_num("Words repeated separators", "Output Length Words \"cats    chew\"", 2);
	test_true("Words trailing run emitted", "Output ((Item Words \"the cat\" 1) = \"cat\")");
	test_true("Words digit-only token", "Output ((Item Words \"ab 42\" 1) = \"42\")");
	test_true("Words mixed-case fold output", "Output ((First Words \"CaT\") = \"cat\")");
	/* GC stress: build tokens, force 300 FPUTs (GC), re-walk -- frame-slot GC roots must keep the list intact */
	test_true("Words survives GC", "Make \"A\" Words \"alpha beta gamma\" Make \"L\" [] Repeat 300 [ Make \"L\" Prepend 7 :L ] Output ((Item :A 2) = \"gamma\")");
	/* FATAL contract tests. Since the Sec.172 landing zone these are catchable
	   in-process, so they run for real instead of being commented out.
	   The message used to name "STR-TOKENS requires a string atom" -- a dead name
	   (the primitive is Words) asserting a contract that does not exist
	   (ASSERT_ATOM_OR_NIL accepts any plain atom, so `Words hello` works). */
	test_fatal_catch_case("FATAL Words on NUMBER names the offender", "Output Words 42", "got number '42'");
	test_fatal_catch_case("FATAL Words message names the live primitive", "Output Words 42", "Words requires a string or word atom argument");
	test_fatal_catch_case("FATAL Words on a list names the offender", "Output Words [ 1 2 ]", "got 'list'");
	/* a word atom is legal input (the type byte is not checked), so the message
	   must not imply otherwise -- this is the call the old text called an error */
	test_num("Words on a word atom is legal input", "Output Length Words hello", 1);
}

void test_split_ngrams()
{
	printf("\n--- NGrams ---\n");
	// string path: n-char byte windows, text order
	test_num("NGrams bigram count", "Output Length NGrams \"abcd\" 2", 3);
	test_true("NGrams first bigram", "Output ((First NGrams \"abcd\" 2) = \"ab\")");
	test_true("NGrams last bigram text order", "Output ((Item NGrams \"abcd\" 2 2) = \"cd\")");
	test_num("NGrams hello count", "Output Length NGrams \"hello\" 2", 4);
	test_num("NGrams unigram count", "Output Length NGrams \"abc\" 1", 3);
	test_true("NGrams unigram first", "Output ((First NGrams \"abc\" 1) = \"a\")");
	test_num("NGrams n equals len single window", "Output Length NGrams \"abc\" 3", 1);
	test_nil("NGrams n greater than len NIL", "Output NGrams \"ab\" 3");
	test_nil("NGrams empty string NIL", "Output NGrams \"\" 2");
	test_nil("NGrams NIL source NIL", "Output NGrams [] 2");
	test_true("NGrams no case fold verbatim", "Output ((First NGrams \"AbC\" 3) = \"AbC\")");
	test_num("NGrams huge n zero windows", "Output Length NGrams \"a\" 1000000", 0);
	// list path: n-element sublists, elements verbatim
	test_num("NGrams list bigram count", "Output Length NGrams [10 20 30 40] 2", 3);
	test_true("NGrams list first window", "Output ((First NGrams [10 20 30] 2) = [10 20])");
	test_true("NGrams list last window text order", "Output ((Item NGrams [10 20 30 40] 2 2) = [30 40])");
	test_true("NGrams list atom elements", "Output ((First NGrams [cats chew grain] 2) = [cats chew])");
	test_true("NGrams nested sublists as elements", "Output ((First NGrams [[a b] [c d] [e f]] 2) = [[a b] [c d]])");
	test_nil("NGrams list n greater than length NIL", "Output NGrams [10 20] 3");
	test_num("NGrams list unigram count", "Output Length NGrams [a b c] 1", 3);
	/* GC stress: build windows, force 300 FPUTs (GC), re-walk -- WINDOW+RESULT
	   frame-slot roots + the r8d-rider mechanism must keep the list intact */
	test_true("NGrams survives GC", "Make \"A\" NGrams [aa bb cc dd] 2 Make \"L\" [] Repeat 300 [ Make \"L\" Prepend 7 :L ] Output ((Item :A 2) = [cc dd])");
	/* FATAL contract tests (commented, matching house style):
	   SPLIT-NGRAMS 42 2       -> FATAL NUMBER source
	   SPLIT-NGRAMS 1.5 2     -> FATAL FLOAT source
	   NGrams "abc" 0   -> FATAL n < 1
	   NGrams "abc" -1  -> FATAL n < 1
	   NGrams "abc" 1.5 -> FATAL FLOAT n (NumVal)
	   (uncomment individually to verify; a FATAL terminates the process) */
}

void test_lindex()
{
	printf("\n--- Item ---\n");
	// list literal elements are raw ATOMS (reader interns verbatim) ÔÇö use
	// EQUALP coercion or STR-TO-NUM composition for numeric expectations
	test_true("Item first", "Output ((Item [10 20 30] 0) = 10)");
	test_true("Item second", "Output ((Item [10 20 30] 1) = 20)");
	test_true("Item third", "Output ((Item [10 20 30] 2) = 30)");
	test_num("Item first via TextToNumber", "Output TextToNumber Item [10 20 30] 0", 10);
	test_num("Item second via TextToNumber", "Output TextToNumber Item [10 20 30] 1", 20);
	/* out-of-range and non-list first operands FATAL (the silent-NIL return was
	   the bug: it made index bugs invisible). Sec.173 names the offender. */
	test_fatal_catch_case("FATAL Item index past the end", "Output Item [10 20 30] 5", "out of bounds");
	test_fatal_catch_case("FATAL Item negative index", "Output Item [1 2] -1", "out of bounds");
	test_fatal_catch_case("FATAL Item empty list", "Output Item [] 0", "requires a non-empty list");
	test_fatal_catch_case("FATAL Item on a number", "Output Item 5 0", "requires a non-empty list");
	test_fatal_catch_case("FATAL Item on a plain atom", "Output Item hello 0", "requires a non-empty list");
	test_fatal_catch_case("FATAL Item index names the offender", "Output Item [10 20 30] 5", "got number '5'");
	/* an in-range item may still BE NIL -- that path is untouched */
	test_nil("Item in-range NIL element", "Output Item [ [] 1 ] 0");
	test_true("Item string list", "Output ((Item [\"a\" \"b\" \"c\"] 1) = \"b\")");
	test_true("Item on Split result", "Output ((Item Split \"x,y,z\" \",\" 1) = \"y\")");
}

/* ====== FLOAT TEST SUITES ====== */

void test_float_literals()
{
	printf("\n--- Float literals ---\n");
	test_float_val("1.0", "Output 1.0", 1.0f);
	test_float_val("2.5", "Output 2.5", 2.5f);
	test_float_val("0.5", "Output 0.5", 0.5f);
	test_float_val("-1.5", "Output -1.5", -1.5f);
	test_float_val("3.14159", "Output 3.14159", 3.14159f);
	test_float_val("0.1", "Output 0.1", 0.1f);
	test_float_val("100.0", "Output 100.0", 100.0f);
	test_float_val("-0.25", "Output -0.25", -0.25f);
	test_float_val("0.0 (pos zero)", "Output 0.0", 0.0f);
	test_float_val("99.9", "Output 99.9", 99.9f);
	test_float_val("0.001", "Output 0.001", 0.001f);
}

/* Literal parsing must produce the EXACT correctly-rounded binary32 of the
   written decimal: the parser accumulates in binary64 and narrows once at the
   end. Each expected value is a C double literal -- the (float) cast inside
   test_float_bits is the same correctly-rounded narrowing. */
void test_literal_precision()
{
	printf("\n--- Literal precision (exact bits) ---\n");
	test_float_bits("1.4999999 nearest is not 1.5", "Output 1.4999999", 1.4999999);
	test_float_bits("8388609.4999999 integer part above 2^23", "Output 8388609.4999999", 8388609.4999999);
	test_float_bits("14159265.589793238 eight int digits", "Output 14159265.589793238", 14159265.589793238);
	test_float_bits("3.14159265", "Output 3.14159265", 3.14159265);
	test_float_bits("0.1", "Output 0.1", 0.1);
	test_float_bits("-2.5 sign flip after narrowing", "Output -2.5", -2.5);
	test_float_bits("1234567.89", "Output 1234567.89", 1234567.89);
	test_float_bits("huge integer 12345678901234", "Output 12345678901234", 12345678901234.0);
}

/* A caught FATAL re-runs lg_init on the next interp call (v7_free_inference
   clears sl_lg_initialized), which allocates a FRESH arena -- the eval_script
   prog caches are cons pointers into the OLD arena, so they must be
   cleared at re-init or the next eval_script walks unrelated bytes as cons
   cells. Regression: populate a cache, take a caught FATAL, then eval_script
   the same key again -- it must re-parse and succeed. Uses the one routed
   script key, "on_stream_evict". */
void test_gc_root_reinit()
{
	int rc;
	printf("\n--- GC roots cleared on caught-FATAL lg_init re-run ---\n");
	/* ht_find in eval_script reads the global hash table v7_init loaded --
	 * no gate call needed, the engine DB is the only database */
	/* store the script text the eval_script cache will parse (the
	   trailing Output gives the statement a return value to assert on) */
	test_true("store on_stream_evict script", "DatabaseSet \"on_stream_evict\" \"Make \\\"gcrp\\\" 1 Output 1\" Output 1");
	/* populate the cache: parse + evaluate (returns 0 on success) */
	rc = inference_dll_ptr->v7s_sectorlogo_eval_script("on_stream_evict");
	if (rc == 0)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		failing_tests_counter++;
		printf("  FAIL: first eval_script(on_stream_evict) -> %d, expected 0\n", rc);
	}
	/* trigger a caught FATAL; run() revives the engine via v7_free_inference,
	   which clears sl_lg_initialized so lg_init re-runs on the NEXT call and
	   also re-inits the hash table (temp keys are gone -- DatabaseSet stores
	   persist=0). Any FATAL works; integer arithmetic saturates now, so the
	   oldest probe (a payload overflow) no longer dies -- division by zero
	   still does. */
	run("Output (1 / 0)");
	if (!last_run_engine_died)
	{
		failing_tests_counter++;
		printf("  FAIL: division-by-zero script did not die (test precondition)\n");
	}
	/* re-store the script text (the revive wiped the temp DB entry), then
	   re-eval: the stale cache slot must have been cleared by lg_init */
	test_true("re-store on_stream_evict script after revive", "DatabaseSet \"on_stream_evict\" \"Make \\\"gcrp\\\" 1 Output 1\" Output 1");
	rc = inference_dll_ptr->v7s_sectorlogo_eval_script("on_stream_evict");
	if (rc == 0 && !inference_dll_ptr->v7_engine_is_dead())
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		failing_tests_counter++;
		printf("  FAIL: eval_script(on_stream_evict) after caught FATAL -> %d (dead=%d)\n",
		       rc, inference_dll_ptr->v7_engine_is_dead());
	}
}

/* .es_streq key dispatch in v7s_sectorlogo_eval_script.
   eval_script routes ONLY its wired script keys. The old A/B tuner's keys
   (tune_init / tune_accept / tune_reject) were removed from the routing
   table together with the scripts -- a script stored under such a key must
   route to not-found (-1), never to some other key's program.

   The one live wired key beyond the hook is "on_stream_evict" (the
   StreamingLLM eviction callback). Its script stores its own marker, so a
   misroute is observable: whichever script actually ran is the only marker
   that appears. The markers are probed for EXISTENCE, not for a value:
   DatabaseGet returns the stored bytes as a freshly interned STRING atom
   (sectorlogo_v7prim.inc:131-137) and NIL when the key is absent, so
   `Equal [] DatabaseGet "key"` is the only representation-independent
   test. A value probe would read the atom offset. */
void test_eval_script_key_dispatch()
{
	int rc;

	printf("\n--- eval_script routes each key to its own script ---\n");

	test_true("store on_stream_evict marker script",
	          "DatabaseSet \"on_stream_evict\" \"DatabaseSet \\\"es_ran_evict\\\" 1\" Output 1");

	rc = inference_dll_ptr->v7s_sectorlogo_eval_script("on_stream_evict");
	if (rc == 0)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		failing_tests_counter++;
		printf("  FAIL: eval_script(on_stream_evict) -> %d, expected 0\n", rc);
	}

	test_num("eviction-callback key ran the eviction script",
	         "Output IfElse ([] = (DatabaseGet \"es_ran_evict\")) [0] [1]", 1);

	/* the removed A/B tuner keys must not route: they answer not-found,
	   even when a script text sits under the key in the database */
	test_true("store a script under the removed tune_init key",
	          "DatabaseSet \"tune_init\" \"DatabaseSet \\\"es_ran_tune\\\" 1\" Output 1");
	rc = inference_dll_ptr->v7s_sectorlogo_eval_script("tune_init");
	if (rc == -1)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		failing_tests_counter++;
		printf("  FAIL: eval_script(tune_init, unrouted) -> %d, expected -1\n", rc);
	}
	test_num("the unrouted tuner key ran nothing",
	         "Output IfElse ([] = (DatabaseGet \"es_ran_tune\")) [0] [1]", 0);

	rc = inference_dll_ptr->v7s_sectorlogo_eval_script("nosuchkey_zzz");
	if (rc == -1)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		failing_tests_counter++;
		printf("  FAIL: eval_script(nosuchkey_zzz) -> %d, expected -1\n", rc);
	}
}

void test_float_prefix_arithmetic()
{
	printf("\n--- Float arithmetic ---\n");
	test_float_val("Add 1.5 2.5", "Output (1.5 + 2.5)", 4.0f);
	test_float_val("Add -1.5 3.0", "Output (-1.5 + 3.0)", 1.5f);
	test_float_val("Subtract 5.5 2.5", "Output (5.5 - 2.5)", 3.0f);
	test_float_val("Subtract 2.5 5.5", "Output (2.5 - 5.5)", -3.0f);
	test_float_val("Multiply 2.5 4.0", "Output (2.5 * 4.0)", 10.0f);
	test_float_val("Multiply -3.0 2.5", "Output (-3.0 * 2.5)", -7.5f);
	test_float_val("Divide 7.5 2.5", "Output (7.5 / 2.5)", 3.0f);
	test_float_val("Divide 10.0 4.0", "Output (10.0 / 4.0)", 2.5f);
	test_float_val("Remainder 7.5 2.5", "Output (7.5 % 2.5)", 0.0f);
	test_float_val("Remainder 7.0 3.0", "Output (7.0 % 3.0)", 1.0f);
	test_float_val("Power 2.0 3.0", "Output Power 2.0 3.0", 8.0f);
	test_float_val("Power 4.0 0.5", "Output Power 4.0 0.5", 2.0f);
	test_float_val("Power -2.0 3.0", "Output Power -2.0 3.0", -8.0f);
	test_float_val("Power -2.0 2.0", "Output Power -2.0 2.0", 4.0f);
	test_float_val("Power -0.5 3.0", "Output Power -0.5 3.0", -0.125f);
	test_num("Power -2 3", "Output Power -2 3", -8);
	test_num("Power -2 2", "Output Power -2 2", 4);
	test_float_val("SquareRoot 2.25", "Output SquareRoot 2.25", 1.5f);
	test_float_val("SquareRoot 9.0", "Output SquareRoot 9.0", 3.0f);
	test_float_val("AbsoluteValue -3.5", "Output AbsoluteValue -3.5", 3.5f);
	test_float_val("AbsoluteValue 3.5", "Output AbsoluteValue 3.5", 3.5f);
	test_float_val("Negate 2.5", "Output Negate 2.5", -2.5f);
	test_float_val("Negate 1.5", "Output Negate 1.5", -1.5f);
	test_float_val("Maximum 2.5 3.5", "Output Maximum 2.5 3.5", 3.5f);
	test_float_val("Minimum 2.5 3.5", "Output Minimum 2.5 3.5", 2.5f);
}

void test_float_infix_arithmetic()
{
	printf("\n--- Float infix arithmetic ---\n");
	test_float_val("1.5 + 2.5", "Output (1.5 + 2.5)", 4.0f);
	test_float_val("5.5 - 2.5", "Output (5.5 - 2.5)", 3.0f);
	test_float_val("2.5 * 4.0", "Output (2.5 * 4.0)", 10.0f);
	test_float_val("7.5 / 2.5", "Output (7.5 / 2.5)", 3.0f);
	test_float_val("7.5 pct 2.5", "Output (7.5 % 2.5)", 0.0f);
	test_float_val("10.5 pct 3.0", "Output (10.5 % 3.0)", 1.5f);
}

void test_float_promotion()
{
	printf("\n--- Float promotion (int+float => float) ---\n");
	test_float_val("Add 1 2.5 (int+float)", "Output (1 + 2.5)", 3.5f);
	test_float_val("Add 2.5 1 (float+int)", "Output (2.5 + 1)", 3.5f);
	test_float_val("Subtract 10 2.5", "Output (10 - 2.5)", 7.5f);
	test_float_val("Multiply 3 2.5", "Output (3 * 2.5)", 7.5f);
	test_float_val("Divide 10 4.0", "Output (10 / 4.0)", 2.5f);
	test_float_val("Remainder 7 2.5", "Output (7 % 2.5)", 2.0f);
	test_float_val("Power 4 0.5", "Output Power 4 0.5", 2.0f);
	test_float_val("3 + 2.5 (infix)", "Output (3 + 2.5)", 5.5f);
	test_float_val("2.5 * 4 (infix)", "Output (2.5 * 4)", 10.0f);
	test_float_val("10 / 4.0 (infix)", "Output (10 / 4.0)", 2.5f);

	/* int x int must remain NUMBER (no promotion) */
	test_num("Add 2 3 stays int", "Output (2 + 3)", 5);
	test_num("Multiply 6 7 stays int", "Output (6 * 7)", 42);
	test_num("3 + 4 stays int", "Output (3 + 4)", 7);
}

void test_float_comparisons()
{
	printf("\n--- Float comparisons ---\n");
	test_true("1.5 eq 1.5", "Output (1.5 = 1.5)");
	test_nil("1.5 eq 2.5", "Output (1.5 = 2.5)");
	test_true("2.5 gt 1.5", "Output (2.5 > 1.5)");
	test_nil("1.5 gt 2.5", "Output (1.5 > 2.5)");
	test_true("1.5 lt 2.5", "Output (1.5 < 2.5)");
	test_nil("2.5 lt 1.5", "Output (2.5 < 1.5)");
	test_true("2.5 ge 2.5", "Output (2.5 >= 2.5)");
	test_true("2.5 le 2.5", "Output (2.5 <= 2.5)");

	/* Float infix comparisons */
	test_true("1.5 = 1.5 infix", "Output (1.5 = 1.5)");
	test_nil("1.5 = 2.5 infix", "Output (1.5 = 2.5)");
	test_true("2.5 > 1.5 infix", "Output (2.5 > 1.5)");
	test_true("1.5 < 2.5 infix", "Output (1.5 < 2.5)");
	test_true("2.5 >= 2.5 infix", "Output (2.5 >= 2.5)");
	test_true("1.5 <= 2.5 infix", "Output (1.5 <= 2.5)");

	/* Cross-type comparisons (int vs float) */
	test_true("1 eq 1.0 cross", "Output (1 = 1.0)");
	test_nil("1 eq 1.5 cross", "Output (1 = 1.5)");
	test_true("1.0 eq 1 cross", "Output (1.0 = 1)");
	test_true("2 gt 1.5 cross", "Output (2 > 1.5)");
	test_nil("1 gt 1.5 cross", "Output (1 > 1.5)");
	test_true("1.5 lt 2 cross", "Output (1.5 < 2)");
}

void test_float_truthiness()
{
	printf("\n--- Float truthiness ---\n");
	test_true("1.5 is truthy", "If 1.5 [ Output 1 ] Output 0");
	test_num("0.0 is falsy (If skips)", "If 0.0 [ Output 1 ] Output 0", 0);
	test_true("-1.5 is truthy", "If -1.5 [ Output 1 ] Output 0");
	test_true("0.001 is truthy", "If 0.001 [ Output 1 ] Output 0");
	test_true("&& with float truthy", "Output ((1 = 1) && 1.5)");
	test_nil("&& with float falsy", "Output ((1 = 1) && 0.0)");
	test_true("|| with float falsy lhs", "Output ((1 = 0) || 1.5)");
	test_true("! float falsy", "Output (! 0.0)");
	test_nil("! float truthy", "Output (! 1.5)");
}

void test_float_type_predicates()
{
	printf("\n--- Float type predicates ---\n");
	test_nil("IsNumber 1.5", "Output IsNumber 1.5");
	test_true("IsFloat 1.5", "Output IsFloat 1.5");
	test_nil("IsFloat 1", "Output IsFloat 1");
	test_nil("IsFloat 0", "Output IsFloat 0");
	test_true("IsFloat 0.0", "Output IsFloat 0.0");
	test_true("IsFloat -2.5", "Output IsFloat -2.5");
	test_nil("IsZero 1.5", "Output IsZero 1.5");
	test_true("IsZero 0.0", "Output IsZero 0.0");
}

void test_float_conversions()
{
	printf("\n--- Float conversions ---\n");
	test_float_val("AsFloat 5", "Output AsFloat 5", 5.0f);
	test_float_val("AsFloat -3", "Output AsFloat -3", -3.0f);
	test_float_val("AsFloat 0", "Output AsFloat 0", 0.0f);
	/* Leading-dot magnitudes across all three sign spellings. AsFloat does NOT take
	   string atoms (sl_to_float goes through NumVal), so the parser is reached the
	   way production reaches it: through sl_sort_coerce_num, i.e. a Sort/LFIND key.
	   The minus case is the Sec.170 regression: ".5" and "+.5" always parsed, while
	   "-.5" hard-FATALed before the .scn_minus magnitude peek was removed.
	   Sorted ascending: -3, -0.5, -0.25. LE key -0.4 admits -3 and -0.5 and returns
	   the LAST of them (index 1) -- which also proves "-.5" sorted into the right
	   place, not merely that the run survived. */
	test_num("leading-dot minus parses (sort key path)",
	         "Output FindFirstAtMost Sort [\"-3\" \"-.5\" \"-.25\"] 0 -0.4", 1);
	test_num("leading-dot plus parses (sort key path)",
	         "Output FindFirstAtMost Sort [\"-3\" \"+.5\" \".25\"] 0 0.3", 1);
	test_fatal_catch_case("bare minus still dies", "Output Sort [\"-\"]", "a numeric value was required");
	test_fatal_catch_case("minus trailing dot still dies", "Output Sort [\"-.\"]", "a numeric value was required");
	test_fatal_catch_case("minus dot garbage still dies", "Output Sort [\"-.junk\"]", "a numeric value was required");
	test_num("AsInteger 2.0", "Output AsInteger 2.0", 2);
	test_num("AsInteger -1.0", "Output AsInteger -1.0", -1);
	test_num("AsInteger 2.7 (truncates)", "Output AsInteger 2.7", 2);
	test_num("AsInteger -2.7 (truncates)", "Output AsInteger -2.7", -2);
	test_float_val("Round 2.5 (bankers)", "Output Round 2.5", 2.0f);
	test_float_val("Round 2.4", "Output Round 2.4", 2.0f);
	test_float_val("Round -2.5 (bankers)", "Output Round -2.5", -2.0f);
	test_float_val("Round 2.6", "Output Round 2.6", 3.0f);
	test_float_val("Round 3.5 (bankers)", "Output Round 3.5", 4.0f);
	test_float_val("Floor 2.9", "Output Floor 2.9", 2.0f);
	test_float_val("Floor -2.1", "Output Floor -2.1", -3.0f);
	test_float_val("Floor 2.0", "Output Floor 2.0", 2.0f);
	/* overflow-path fraction: the integer part (>= 2^29) must NOT be pre-rounded to
	 * binary32 before the fraction is added -- 536870944 is an exact binary32
	 * midpoint, so a premature rounding loses the +0.5 tie-breaker (1 ULP = 64) */
	test_float_val("AsFloat 536870944.5 (overflow-path fraction rounds up)", "Output AsFloat 536870944.5", 536870976.0f);
	test_float_val("AsFloat 1073741888.25 keeps its fraction", "Output AsFloat 1073741888.25", 1073741952.0f);
}

void test_float_transcendentals()
{
	printf("\n--- Float transcendentals ---\n");
	test_float_val("LogBaseTwo 2.0", "Output LogBaseTwo 2.0", 1.0f);
	test_float_val("LogBaseTwo 4.0", "Output LogBaseTwo 4.0", 2.0f);
	test_float_val("LogBaseTwo 8.0", "Output LogBaseTwo 8.0", 3.0f);
	test_float_val("LogBaseTwo 1.0", "Output LogBaseTwo 1.0", 0.0f);
	test_float_val("PowerOfTwo 1.0", "Output PowerOfTwo 1.0", 2.0f);
	test_float_val("PowerOfTwo 2.0", "Output PowerOfTwo 2.0", 4.0f);
	test_float_val("PowerOfTwo 0.0", "Output PowerOfTwo 0.0", 1.0f);
	test_float_val("NaturalLog e (2.71828)", "Output NaturalLog 2.71828", 1.0f);
	test_float_val("NaturalLog 1.0", "Output NaturalLog 1.0", 0.0f);
	test_float_val("FloatRemainder 7.5 2.5", "Output FloatRemainder 7.5 2.5", 0.0f);
	test_float_val("FloatRemainder 7.0 3.0", "Output FloatRemainder 7.0 3.0", 1.0f);
	test_float_val("FloatRemainder 10.0 3.0", "Output FloatRemainder 10.0 3.0", 1.0f);
test_float_val("FloatRemainder -7.0 3.0", "Output FloatRemainder -7.0 3.0", -1.0f);
test_float_val("FloatRemainder 7.0 -3.0", "Output FloatRemainder 7.0 -3.0", 1.0f);
/* exact integer remainder (eval_ops BUG-5). The old
   path converted both operands to binary32 first, so any |operand| > 2^24 was
   rounded BEFORE the remainder was taken and the float remainder of the
   rounded operands is not the integer remainder. All three answered 0.0, 0.0
   and 2.0 respectively before the fix. */
test_float_val("FloatRemainder int operands above 2^24 (exact integer remainder)",
	               "Output FloatRemainder 16777217 2", 1.0f);
test_float_val("FloatRemainder int operands at the payload maximum",
	               "Output FloatRemainder 536870911 2", 1.0f);
test_float_val("FloatRemainder int operands above 2^24, non-power-of-two divisor",
	               "Output FloatRemainder 33554434 5", 4.0f);
	test_float_val("Power 2.0 10.0", "Output Power 2.0 10.0", 1024.0f);
	test_float_val("Power 10.0 0.5", "Output Power 10.0 0.5", 3.16227766f);
	test_float_val("Power 2.0 -1.0", "Output Power 2.0 -1.0", 0.5f);
	test_float_val("SquareRoot 2.0", "Output SquareRoot 2.0", 1.41421356f);
	test_float_val("SquareRoot 0.25", "Output SquareRoot 0.25", 0.5f);
}

void test_float_transcendentals_ext()
{
	printf("\n--- Float math functions (extensive) ---\n");
	/* FLOG2 across magnitudes: powers of 2 exact, non-powers to 1 ulp */
	test_float_val("LogBaseTwo 0.5", "Output LogBaseTwo 0.5", -1.0f);
	test_float_val("LogBaseTwo 0.25", "Output LogBaseTwo 0.25", -2.0f);
	test_float_val("LogBaseTwo 3.0", "Output LogBaseTwo 3.0", 1.58496250f);
	test_float_val("LogBaseTwo 5.0", "Output LogBaseTwo 5.0", 2.32192809f);
	test_float_val("LogBaseTwo 6.0", "Output LogBaseTwo 6.0", 2.58496250f);
	test_float_val("LogBaseTwo 1024.0", "Output LogBaseTwo 1024.0", 10.0f);
	test_float_val("LogBaseTwo 1000000.0", "Output LogBaseTwo 1000000.0", 19.9315686f);
	test_float_val("LogBaseTwo 8 (int arg)", "Output LogBaseTwo 8", 3.0f);
	test_float_val("LogBaseTwo 1.5", "Output LogBaseTwo 1.5", 0.58496250f);
	/* FEXP2 across magnitudes incl. negatives and fractional exponents */
	test_float_val("PowerOfTwo -1.0", "Output PowerOfTwo -1.0", 0.5f);
	test_float_val("PowerOfTwo -10.0", "Output PowerOfTwo -10.0", 0.0009765625f);
	test_float_val("PowerOfTwo 3.5", "Output PowerOfTwo 3.5", 11.3137085f);
	test_float_val("PowerOfTwo 0.5", "Output PowerOfTwo 0.5", 1.41421356f);
	test_float_val("PowerOfTwo 10.0", "Output PowerOfTwo 10.0", 1024.0f);
	test_float_val("PowerOfTwo 20.0", "Output PowerOfTwo 20.0", 1048576.0f);
	test_float_val("PowerOfTwo 3 (int arg)", "Output PowerOfTwo 3", 8.0f);
	test_float_val("PowerOfTwo -0.5", "Output PowerOfTwo -0.5", 0.70710678f);
	/* FLN across magnitudes: ln reduces large args, negates on <1 */
	test_float_val("NaturalLog 10.0", "Output NaturalLog 10.0", 2.30258509f);
	test_float_val("NaturalLog 0.5", "Output NaturalLog 0.5", -0.69314718f);
	test_float_val("NaturalLog 100.0", "Output NaturalLog 100.0", 4.60517019f);
	test_float_val("NaturalLog 7.389056", "Output NaturalLog 7.389056", 2.0f);
	test_float_val("NaturalLog 20.085537", "Output NaturalLog 20.085537", 3.0f);
	test_float_val("NaturalLog 1000000.0", "Output NaturalLog 1000000.0", 13.8155111f);
	test_float_val("NaturalLog 1 (int arg)", "Output NaturalLog 1", 0.0f);
	/* FMOD sign rules + accumulation-grade fractions */
	test_float_val("FloatRemainder 5.5 2.5", "Output FloatRemainder 5.5 2.5", 0.5f);
	test_float_val("FloatRemainder -5.5 2.5", "Output FloatRemainder -5.5 2.5", -0.5f);
	test_float_val("FloatRemainder 5.5 -2.5", "Output FloatRemainder 5.5 -2.5", 0.5f);
	test_float_val("FloatRemainder 0.7 0.2", "Output FloatRemainder 0.7 0.2", 0.0999999f);
	test_float_val("FloatRemainder 100.25 7.0", "Output FloatRemainder 100.25 7.0", 2.25f);
	test_float_val("FloatRemainder 0.3 0.1", "Output FloatRemainder 0.3 0.1", 0.0999999f);
	test_float_val("FloatRemainder 7 3 (int args)", "Output FloatRemainder 7 3", 1.0f);
	test_float_val("FloatRemainder 100.0 100.0", "Output FloatRemainder 100.0 100.0", 0.0f);
	/* fprem loop hang: the loop tested the x87 status
	   bit via fstsw+sahf+jp, and SAHF loads only AH, so jp tested PF = status
	   bit 10 = ZE (the zero flag) instead of C2 (the "reduction incomplete"
	   condition code, which lives in AL). FPREM sets ZE on a ZERO result and
	   never clears it, so an exact-zero remainder re-entered fprem forever.
	   These are the COMMON cases, not exotic ones -- every power-of-two ratio
	   produces a zero remainder. Before the fix the first of these HANGS the
	   whole validator, so if this suite ever "passes" suspiciously fast, check
	   that the assertion actually executed. */
	test_float_val("FloatRemainder 100 10 (exact zero, was an infinite loop)",
	               "Output FloatRemainder 100 10", 0.0f);
	test_float_val("FloatRemainder 8 4 (exact zero, was an infinite loop)",
	               "Output FloatRemainder 8 4", 0.0f);
	/* A ratio past 2^66 forces FPREM to iterate (C2 set on the first pass),
	   so this covers the loop actually looping and then terminating. The
	   quotient is 2^100 / 2^-100 = 2^200, comfortably past the x87's 66-bit
	   partial reduction, and the exact remainder is 0. */
	test_float_val("FloatRemainder 2^100 2^-100 (quotient past 2^66)",
	               "Output FloatRemainder PowerOfTwo 100.0 PowerOfTwo -100.0", 0.0f);
	/* Large quotient with a dividend far larger than the divisor: 2^100 / 1.0 needs
	   many FPREM reductions, so the loop must actually iterate to convergence.
	   Reading AL bit 2 instead of AH bit 2 tests ZE, which FPREM never sets, so the
	   loop exits after one reduction and returns an unscaled partial remainder
	   instead of 0.0. (Distinct from the 2^100 / 2^-100 case above, where the
	   divisor is the tiny one and the quotient is likewise 2^200.) */
	test_float_val("FloatRemainder 2^100 1.0 (large quotient, C2 test must iterate)",
	               "Output FloatRemainder PowerOfTwo 100.0 1.0", 0.0f);
	/* Exact-zero remainder as the control: C2 is clear, so the loop must exit
	   immediately and return 0.0. A wrong C2 test that left C2 set would spin. */
	test_float_val("FloatRemainder 7 7 (exact zero, C2 clear, must exit at once)",
	               "Output FloatRemainder 7 7", 0.0f);
	/* POWER: reciprocal, integer, negative exponents (base>0 contract) */
	test_float_val("Power 0.5 -2.0", "Output Power 0.5 -2.0", 4.0f);
	test_float_val("Power 2.5 2.0", "Output Power 2.5 2.0", 6.25f);
	test_float_val("Power 9.0 0.5", "Output Power 9.0 0.5", 3.0f);
	test_float_val("Power 10.0 -3.0", "Output Power 10.0 -3.0", 0.001f);
	test_float_val("Power 100.0 2.0", "Output Power 100.0 2.0", 10000.0f);
	test_float_val("Power 1.5 4.0", "Output Power 1.5 4.0", 5.0625f);
	test_float_val("Power 2.0 16.0", "Output Power 2.0 16.0", 65536.0f);
	/* SQRT: perfect squares exact, fractions, tiny and large */
	test_float_val("SquareRoot 16.0", "Output SquareRoot 16.0", 4.0f);
	test_float_val("SquareRoot 25.0", "Output SquareRoot 25.0", 5.0f);
	test_float_val("SquareRoot 0.01", "Output SquareRoot 0.01", 0.1f);
	test_float_val("SquareRoot 0.0", "Output SquareRoot 0.0", 0.0f);
	test_float_val("SquareRoot 1000000.0", "Output SquareRoot 1000000.0", 1000.0f);
	test_float_val("SquareRoot 10.0", "Output SquareRoot 10.0", 3.16227766f);
	test_float_val("SquareRoot 152.52", "Output SquareRoot 152.52", 12.3498987f);
}

void test_float_math_chained()
{
	printf("\n--- Float math identities (function composition) ---\n");
	/* Inverse identities must round-trip within 1 ulp */
	test_float_val("PowerOfTwo LogBaseTwo 7.0", "Output PowerOfTwo LogBaseTwo 7.0", 7.0f);
	test_float_val("LogBaseTwo PowerOfTwo 5.0", "Output LogBaseTwo PowerOfTwo 5.0", 5.0f);
	test_float_val("PowerOfTwo LogBaseTwo 13.0", "Output PowerOfTwo LogBaseTwo 13.0", 13.0f);
	test_float_val("NaturalLog PowerOfTwo 3.0 = 3*ln2", "Output NaturalLog PowerOfTwo 3.0", 2.07944154f);
	test_float_val("PowerOfTwo NaturalLog 13.0 = 2^ln13", "Output PowerOfTwo NaturalLog 13.0", 5.91734124f);
	test_float_val("SquareRoot SquareRoot 81.0", "Output SquareRoot SquareRoot 81.0", 3.0f);
	test_float_val("SquareRoot Multiply 7.0 7.0", "Output SquareRoot (7.0 * 7.0)", 7.0f);
	test_float_val("LogBaseTwo Power 2.0 6.0", "Output LogBaseTwo Power 2.0 6.0", 6.0f);
	test_float_val("Power SquareRoot 3.0 2.0", "Output Power SquareRoot 3.0 2.0", 3.0f);
	/* FMOD within larger expressions */
	test_float_val("FloatRemainder Add 10.0 0.5 Add 2.0 1.0", "Output FloatRemainder (10.0 + 0.5) (2.0 + 1.0)", 1.5f);
	test_float_val("Add FloatRemainder 10.0 3.0 FloatRemainder 20.0 7.0", "Output ((FloatRemainder 10.0 3.0) + (FloatRemainder 20.0 7.0))", 7.0f);
	/* BM25-flavored: log(1+tf), k1*(1-b+b*dl/avgdl), idf = ln((N-n+0.5)/(n+0.5)) */
	test_float_val("BM25 ln(1+tf) tf=7", "Output NaturalLog (1.0 + 7.0)", 2.07944154f);
	test_float_val("BM25 idf N=100 n=10",
	               "Make \"NUM\" ((100.0 - 10.0) + 0.5) "
	               "Make \"DEN\" (10.0 + 0.5) "
	               "Output NaturalLog (:NUM / :DEN)",
	               2.15397459f);
	test_float_val("BM25 norm dl=3 avg=2.3", "Output (1.2 * (3.0 / 2.3))", 1.56521739f);
	test_float_val("BM25 log2 idf variant", "Output LogBaseTwo (99.0 / 7.0)", 3.82200153f);
	/* Comparisons feeding from math results */
	test_true("PowerOfTwo 10.0 > 1000.0", "Output ((PowerOfTwo 10.0) > 1000.0)");
	test_nil("SquareRoot 2.0 > 1.5", "Output ((SquareRoot 2.0) > 1.5)");
	test_true("LogBaseTwo 8.0 = 3.0 (cross)", "Output ((LogBaseTwo 8.0) = 3.0)");
	test_true("NaturalLog 1.0 is falsy (0.0)", "If NaturalLog 1.0 [ Output 0 ] Output 1");
	test_float_val("max of two sqrt", "Output Maximum SquareRoot 2.0 SquareRoot 3.0", 1.73205081f);
	test_float_val("min of two exp", "Output Minimum PowerOfTwo 3.0 PowerOfTwo 4.0", 8.0f);
}

void test_float_conversions_ext()
{
	printf("\n--- Float conversions (extensive) ---\n");
	/* FROUND banker's rounding table (round-to-nearest-even) */
	test_float_val("Round 0.5 (bankers)", "Output Round 0.5", 0.0f);
	test_float_val("Round 1.5 (bankers)", "Output Round 1.5", 2.0f);
	test_float_val("Round -1.5 (bankers)", "Output Round -1.5", -2.0f);
	test_float_val("Round 2.6", "Output Round 2.6", 3.0f);
	test_float_val("Round -2.6", "Output Round -2.6", -3.0f);
	test_float_val("Round 2.5 - stays even 2", "Output Round 2.5", 2.0f);
	test_float_val("Round 3.5 - goes even 4", "Output Round 3.5", 4.0f);
	test_float_val("Round 100.49", "Output Round 100.49", 100.0f);
	test_float_val("Round 100.51", "Output Round 100.51", 101.0f);
	/* FLOOR strictly down */
	test_float_val("Floor 0.0", "Output Floor 0.0", 0.0f);
	test_float_val("Floor -0.5", "Output Floor -0.5", -1.0f);
	test_float_val("Floor 0.001", "Output Floor 0.001", 0.0f);
	test_float_val("Floor 99.999", "Output Floor 99.999", 99.0f);
	/* F2I truncation toward zero at boundaries */
	test_num("AsInteger 0.99", "Output AsInteger 0.99", 0);
	test_num("AsInteger -0.99", "Output AsInteger -0.99", 0);
	test_num("AsInteger 1000000.0", "Output AsInteger 1000000.0", 1000000);
	test_num("AsInteger -1000000.0", "Output AsInteger -1000000.0", -1000000);
	test_num("AsInteger 999.5", "Output AsInteger 999.5", 999);
	test_num("AsInteger PowerOfTwo 20.0", "Output AsInteger PowerOfTwo 20.0", 1048576);
	/* I2F round-trip within exact int range */
	test_float_val("AsFloat 1000000", "Output AsFloat 1000000", 1000000.0f);
	test_float_val("AsFloat 536870911 (2^29-1)", "Output AsFloat 536870911", 536870912.0f);
	test_num("AsInteger AsFloat 123", "Output AsInteger AsFloat 123", 123);
	/* promotion arithmetic on converted values */
	test_float_val("Add AsFloat 40 2.5", "Output ((AsFloat 40) + 2.5)", 42.5f);
	test_num("AsInteger Multiply 3.3 3.0", "Output AsInteger (3.3 * 3.0)", 9);
}

void test_float_formatter()
{
	printf("\n--- Float formatter (DB round-trip via Equal) ---\n");
	/* V7-DB-PUT formats the float to decimal text; V7-DB-GET returns an atom;
	   EQUALP compares atom text vs literal. The 30-bit truncation means both
	   paths drop the same low bits, so they must match exactly. */
	test_true("DB-PUT 1.5 round-trip", "DatabaseSet \"sl_ft1\" 1.5 "
	                                   "Output ((DatabaseGet \"sl_ft1\") = 1.5)");
	test_true("DB-PUT -2.5 round-trip", "DatabaseSet \"sl_ft2\" -2.5 "
	                                    "Output ((DatabaseGet \"sl_ft2\") = -2.5)");
	test_true("DB-PUT 0.5 round-trip", "DatabaseSet \"sl_ft3\" 0.5 "
	                                   "Output ((DatabaseGet \"sl_ft3\") = 0.5)");
	test_true("DB-PUT 100.0 round-trip", "DatabaseSet \"sl_ft4\" 100.0 "
	                                     "Output ((DatabaseGet \"sl_ft4\") = 100.0)");
	test_true("DB-PUT 0.0 round-trip", "DatabaseSet \"sl_ft5\" 0.0 "
	                                   "Output ((DatabaseGet \"sl_ft5\") = 0.0)");
	test_true("DB-PUT 3.14159 round-trip", "DatabaseSet \"sl_ft6\" 3.14159 "
	                                       "Output ((DatabaseGet \"sl_ft6\") = 3.14159)");
	/* exponent-notation up-loop in sl_float_fmt. That
	   loop is bottom-tested, so its success exit falls out of the body; with no
	   jump after the loop test it landed in the FTZ-flush block, and EVERY
	   float with 0 < |x| < 1e-3 printed "0.0" with the computed exponent
	   discarded. Reachable from PrintFloat, lg_value_to_str's FLOAT arm and
	   the v7 FLOAT-to-text conversion. Assert the TEXT, not the value: the
	   numbers were always right, so a test_float_val check cannot see this --
	   and Search returns 0 for a match at position 0, which is falsy, so
	   found-ness has to be tested with IsNumber. */
	test_true("sub-1e-3 formats with an exponent, not 0.0",
	          "DatabaseSet \"sl_ft7\" 0.0001 "
	          "Make \"hit\" Search \"e-4\" DatabaseGet \"sl_ft7\" "
	          "Output IfElse IsNumber :hit [ 1 ] [ 0 ]");
	test_true("sub-1e-3 at a different exponent still formats",
	          "DatabaseSet \"sl_ft8\" 0.0005 "
	          "Make \"hit\" Search \"e-4\" DatabaseGet \"sl_ft8\" "
	          "Output IfElse IsNumber :hit [ 1 ] [ 0 ]");
}

void test_float_complex()
{
	printf("\n--- Float complex expressions ---\n");
	test_float_val("Pythagorean 3 4", "Output SquareRoot ((3.0 * 3.0) + (4.0 * 4.0))", 5.0f);
	test_float_val("Celsius to Fahrenheit", "Output (((9.0 / 5.0) * 100.0) + 32.0)", 212.0f);
	test_float_val("1.5 + 2.5 * 3.0 (precedence)", "Output (1.5 + (2.5 * 3.0))", 9.0f);
	test_float_val("(1.5 + 2.5) * 3.0 (parens)", "Output ((1.5 + 2.5) * 3.0)", 12.0f);
	test_float_val("Repeat sum floats", "Make \"S\" 0.0 Repeat 5 [ Make \"S\" :S + 0.5 ] Output :S", 2.5f);
	test_float_val("Define with float", "Define \"half\" [:x] [ Output :x / 2.0 ] Output half 7.0", 3.5f);
	test_float_val("Heron's formula triangle 3-4-5",
	               "Make \"A\" 3.0 Make \"B\" 4.0 Make \"C\" 5.0 "
	               "Make \"S\" ((:A + :B) + :C) / 2.0 "
	               "Output SquareRoot (:S * ((:S - :A) * ((:S - :B) * (:S - :C))))",
	               6.0f);
}

void test_float_edge_cases()
{
	printf("\n--- Float edge cases ---\n");
	/* Very small float */
	test_is_float("0.001 is FLOAT", "Output 0.001");
	/* Negative zero */
	test_float_val("NEG 0.0 = -0.0", "Output Negate 0.0", -0.0f);
	/* Float result stored in variable */
	test_float_val("Make var float", "Make \"X\" 2.5 Output :X", 2.5f);
	/* Float from expression stored in variable */
	test_float_val("Make var float expr", "Make \"X\" 1.0 + 2.0 Output :X", 3.0f);
	/* Comparison of promoted result */
	test_true("3 * 1.5 eq 4.5", "Output ((3 * 1.5) = 4.5)");
	/* Nested float arithmetic */
	test_float_val("nested float ops", "Output ((2.0 * 3.0) + (10.0 / 4.0))", 8.5f);
	/* FloatRemainder special values: FPREM never clears C2 for inf/NaN
	   operands -- without the up-front classification these hang forever */
	test_float_nan("FloatRemainder infinity dividend",
	               "Output FloatRemainder PowerOfTwo 1000.0 2.0");
	test_float_val("FloatRemainder infinity divisor",
	               "Output FloatRemainder 5.0 PowerOfTwo 1000.0", 5.0f);
	test_float_nan("FloatRemainder NaN divisor",
	               "Output FloatRemainder 5.0 (0.0 / 0.0)");
	/* POWER with an infinite or NaN base (Sep 2026). These used to route into the
	   log2/exp2 composition, and sl_flog2 FATALs on an all-ones exponent -- so
	   (+Inf)^2 and NaN^3 died instead of returning the IEEE 754 result. The base
	   is now classified up front. PowerOfTwo 1000.0 = 2^1000 = +Inf is the
	   established idiom for constructing an infinity in this language. */
	test_float_inf("Power +Inf ^ 2.0", "Output Power PowerOfTwo 1000.0 2.0", 1);
	test_float_val("Power +Inf ^ -1.0", "Output Power PowerOfTwo 1000.0 -1.0", 0.0f);
	test_float_val("Power +Inf ^ 0.0", "Output Power PowerOfTwo 1000.0 0.0", 1.0f);
	test_float_nan("Power NaN ^ 3.0", "Output Power (0.0 / 0.0) 3.0");
	test_float_val("Power NaN ^ 0.0 is 1.0 (C99: pow(x, +-0) = 1 for any x)",
	               "Output Power (0.0 / 0.0) 0.0", 1.0f);
	test_float_nan("Power +Inf ^ NaN", "Output Power PowerOfTwo 1000.0 (0.0 / 0.0)");
	/* zero-base NaN-exponent branch (IEEE 754-2019
	   9.2.1 / C99 F.10.4.4: pow(x, NaN) is NaN unless x is +-1, and +-0 is not
	   +-1). PowerOfTwo 1000000 overflows the x87 fscale to +inf; FloatRemainder
	   of an infinite dividend returns the canonical qNaN. pow(0, NaN) answered
	   0.0 before the .fpow_zero_e -> .fpow_nan redirect. */
	test_float_nan("Power 0.0 ^ NaN is NaN, not 0",
	               "Output Power 0.0 (FloatRemainder (PowerOfTwo 1000000) 1.0)");
	test_float_inf("Power -Inf ^ 2.0 (even -> +Inf)",
	               "Output Power Negate PowerOfTwo 1000.0 2.0", 1);
	test_float_inf("Power -Inf ^ 3.0 (odd -> -Inf)",
	               "Output Power Negate PowerOfTwo 1000.0 3.0", -1);
	/* zero-base exponent classification (float BUG-1).
	   C99 F.10.4.4 and IEEE-754 both require pow(0, -inf) = +HUGE_VAL; the
	   zero-base fast path returned 0.0 for EVERY non-zero exponent, so this
	   answered 0.0. The other zero-base cells are pinned by the lines above. */
	test_float_inf("Power 0 ^ -Inf is +Inf (C99 pow(0, -inf) = +HUGE_VAL)",
	               "Output Power 0.0 Negate PowerOfTwo 1000.0", 1);
	/* zero-base finite negative exponents (C99 F.10.4.4 row: pow(+-0, y<0)
	   = +-HUGE_VAL -- -Inf only for (-0, odd y)). The old code returned +0
	   for every finite negative exponent. */
	test_float_inf("Power 0 ^ -2 is +Inf (C99 pow(0, y<0) = +HUGE_VAL)",
	               "Output Power 0.0 -2.0", 1);
	test_float_inf("Power 0 ^ -3 is +Inf (odd y on a +0 base)",
	               "Output Power 0.0 -3.0", 1);
	test_float_inf("Power -0 ^ -3 is -Inf (odd y keeps the -0 base sign)",
	               "Output Power (-1.0 * 0.0) -3.0", -1);
	test_float_inf("Power -0 ^ -2 is +Inf (even y drops the base sign)",
	               "Output Power (-1.0 * 0.0) -2.0", 1);
	/* inf-base parity fix: pow(+inf, y) = +inf for
	   EVERY y > 0 (IEEE 754-2019 9.2.1 / C99 F.10.4.4) -- the sign of the BASE
	   decides, not the exponent parity. The old odd-parity arm flipped a +Inf
	   base to -Inf, and a positive fractional exponent on +Inf died in the
	   integrality test instead of answering +Inf. */
	test_float_inf("Power +Inf ^ 3.0 (odd -> +Inf)",
	               "Output Power PowerOfTwo 1000.0 3.0", 1);
	test_float_inf("Power +Inf ^ 0.5 (fractional -> +Inf)",
	               "Output Power PowerOfTwo 1000.0 0.5", 1);
	test_fatal_catch_case("Power -Inf ^ 0.5 dies (no real result)",
	                      "Output Power Negate PowerOfTwo 1000.0 0.5",
	                      "POWER with negative base needs an integral exponent");
	/* huge-|e| negative-base fix (Sec.173 class: the
	   old .fpow_big_e parked r9d=0 as a PARITY placeholder and then fed that
	   zero to the magnitude as the exponent, so every finite negative base
	   with |e| >= 2^24 answered 1.0. PowerOfTwo 25 = 2^25 is the smallest
	   such exponent (even, so no sign flip) -- the true results are the
	   overflow/underflow limits, not 1. */
	test_float_inf("Power -2 ^ 2^25 overflows to +Inf (even exponent)",
	               "Output Power -2.0 PowerOfTwo 25.0", 1);
	test_float_val("Power -0.5 ^ 2^25 underflows to +0.0 (even exponent)",
	               "Output Power -0.5 PowerOfTwo 25.0", 0.0f);
	test_float_val("Power -2 ^ -2^25 underflows to +0.0 (even exponent)",
	               "Output Power -2.0 Negate PowerOfTwo 25.0", 0.0f);
	/* regression: the small-integral negative-base path is unchanged */
	test_float_val("Power -2.0 ^ 10.0 still 1024.0",
	               "Output Power -2.0 10.0", 1024.0f);
	test_fatal_catch_case("Power -2 ^ 0.5 still dies (fractional negative base)",
	                      "Output Power -2.0 0.5",
	                      "POWER with negative base needs an integral exponent");
	/* .fpow_big_e classification. The magnitude test was
	   `jae` against the +Inf pattern, so it swallowed BOTH +-Inf and every NaN
	   (magnitudes sort above +Inf) and FATALed on all four defined cases.
	   IEEE 754-2019 9.2.1 / C99 F.10.4.4: pow(x,NaN) = NaN for finite nonzero x;
	   pow(x,+Inf) = +Inf when |x| > 1, +0 when |x| < 1; pow(x,-Inf) = +0 when
	   |x| > 1, +Inf when |x| < 1; and pow(-1,y) = 1 for ANY y -- the |x| = 1 row
	   is 1, which is why .fpow_neg_base_inf_e needs its own b == -1 arm.
	   PowerOfTwo 1000.0 = +Inf and Divide 0.0 0.0 = NaN are the established
	   one-token idioms (see the +Inf-BASE block just above); -Inf is
	   `Negate PowerOfTwo 1000.0` -- PowerOfTwo of a NEGATIVE exponent underflows to
	   +0 (2^-1000 is far below the smallest denormal), it does not yield -Inf. */
	test_float_inf("Power -2 ^ +Inf is +Inf (|x| > 1)",
	               "Output Power -2.0 PowerOfTwo 1000.0", 1);
	test_float_val("Power -0.5 ^ +Inf is +0 (|x| < 1)",
	               "Output Power -0.5 PowerOfTwo 1000.0", 0.0f);
	test_float_val("Power -2 ^ -Inf is +0 (pow(x,-Inf) mirrors the +Inf row for |x| > 1)",
	               "Output Power -2.0 Negate PowerOfTwo 1000.0", 0.0f);
	test_float_inf("Power -0.5 ^ -Inf is +Inf (pow(x,-Inf) mirrors the +Inf row for |x| < 1)",
	               "Output Power -0.5 Negate PowerOfTwo 1000.0", 1);
	test_float_val("Power -1 ^ +Inf is 1.0 (the |x| = 1 row)",
	               "Output Power -1.0 PowerOfTwo 1000.0", 1.0f);
	test_float_nan("Power -2 ^ NaN is NaN, not a FATAL",
	               "Output Power -2.0 (0.0 / 0.0)");
	/* pow(-0, y>0) must return -0, not +0 (IEEE 754-2019 9.2.1). The reciprocal
	   makes the sign numerically observable: 1/-0 = -inf (LessThan -> true)
	   where 1/+0 = +inf. */
	test_true("Power of a negative zero keeps the sign",
	          "Output ((1 / (Power -0.0 3.0)) < 0)");
	/* e>0 zero-base arm of sl_fpow. IEEE 754-2019 9.2.1 gives
	   pow(-0, y>0) = -0 only for an ODD INTEGER y. -0.0 and +0.0 compare equal, so a
	   test_float_val check is blind to it: assert on the FORMATTED TEXT, which is
	   where the sign surfaces (sl_float_fmt prints "-0.0"). Search returns 0 -- falsy --
	   for a match at position 0, so found-ness goes through IsNumber. */
	test_true("pow(-0, even exponent) has no minus sign",
	          "DatabaseSet \"sl_fp1\" Power (-1.0 * 0.0) 2.0 "
	          "Make \"hit\" Search \"-\" DatabaseGet \"sl_fp1\" "
	          "Output IfElse IsNumber :hit [ 0 ] [ 1 ]");
	test_true("pow(-0, non-integer exponent) has no minus sign",
	          "DatabaseSet \"sl_fp2\" Power (-1.0 * 0.0) 0.5 "
	          "Make \"hit\" Search \"-\" DatabaseGet \"sl_fp2\" "
	          "Output IfElse IsNumber :hit [ 0 ] [ 1 ]");
	test_true("pow(-0, odd exponent) keeps the minus sign",
	          "DatabaseSet \"sl_fp3\" Power (-1.0 * 0.0) 3.0 "
	          "Make \"hit\" Search \"-\" DatabaseGet \"sl_fp3\" "
	          "Output IfElse IsNumber :hit [ 1 ] [ 0 ]");
	test_true("pow(+0, even exponent) stays positive",
	          "DatabaseSet \"sl_fp4\" Power 0.0 2.0 "
	          "Make \"hit\" Search \"-\" DatabaseGet \"sl_fp4\" "
	          "Output IfElse IsNumber :hit [ 0 ] [ 1 ]");
	/* pow(x, NaN) is 1 iff x is +-1 (IEEE 754-2019 9.2.1); the square root of a negative is NaN.
	   Both operands are FLOATs here, so Power answers a FLOAT 1.0 -- hence test_float_val,
	   not test_num (a NUMBER result would mean the integer path had run). */
	test_float_val("Power 1 of a NaN exponent is 1",
	               "Output Power 1.0 (SquareRoot -1.0)", 1.0f);
	/* .fpow_pos b == 1 row. pow(1, y) = 1 for EVERY y, and the
	   log2/exp2 composition cannot produce it for a non-finite y: log2(1) is exactly 0,
	   and 0 * NaN = NaN and 0 * Inf = NaN, so both of these answered NaN. The infinity
	   row is the one a positive-base script can actually hit (Divide 1.0 0.0 is the
	   only way to write an infinity here). */
	test_float_val("Power 1 of +inf is 1",
	               "Output Power 1.0 (1.0 / 0.0)", 1.0f);
	test_float_val("Power 1 of -inf is 1",
	               "Output Power 1.0 (-1.0 / 0.0)", 1.0f);
	test_float_val("Power 1 of a finite exponent is still 1",
	               "Output Power 1.0 7.0", 1.0f);
	/* b == 1 must not have swallowed the ordinary positive-base path. */
	test_float_val("Power of an ordinary positive base is unchanged",
	               "Output Power 2.0 10.0", 1024.0f);
	/* Regression guard for the .fpow_nan fix: a NaN BASE must stay NaN. A bare
	   `vucomiss xmm0, 1.0 / je` would answer NaN^NaN = 1 (VUCOMISS sets ZF=1 when
	   unordered), so this is the test that distinguishes the correct fix from the
	   tempting one. */
	test_true("Power of a NaN base stays NaN, not 1",
	          "Output ((Power (SquareRoot -1.0) (SquareRoot -1.0)) != 1.0)");
}

/* ====== NEW TOKENIZER TESTS (Part E of better_tokenizer_plan) ====== */

void test_tokenizer_delimiter_splitting()
{
	printf("\n--- Tokenizer: delimiter-based splitting (no spaces needed) ---\n");
	/* Operators and brackets always split tokens, even without whitespace.
	   The new delimiter table makes spaces optional between most tokens. */
	/* Names are alnum runs: a digit suffix is PART of the name (test2/add2 are
	   legal names), so the tokenizer never splits digits off a word. */
	test_num("digit-suffixed name stays whole", "Define \"add2\" [:a :b] [ Output :a + :b ] Output add2 3 4", 7);
	/* Parenthesized expression without spaces */
	test_num("parens no spaces", "Output (2 + 3)", 5);
	/* Square brackets split: [1 2 3] works even without inner spaces in some cases */
	test_num("bracket split", "Output Item[1 2 3]0", 1);
	/* Multiple tokens concatenated: precedence with no spaces */
	test_num("mixed ops no spaces", "Output ((2 * 3) + 1)", 7);
	/* ':' is a word char (var tokens are ":x"): a name followed by a colon var
	   needs a space -- Add:x would be one word, not Add + :x. */
	test_num("colon var spaced after name", "Make \"x\" 5 Output (:x + 3)", 8);
}

/* Regression tests for the sl_delim_table / sl_tok_class disagreement: '\' , '{',
   '}' and DEL were marked word-token boundaries in the table while the class
   table and the table's own summary treat them as word characters. As boundaries
   they terminated a preceding word AND produced a standalone dead atom that
   self-evaluates and desyncs the argument stream (Sec.146 fact 38). */
void test_tokenizer_word_char_boundaries()
{
	printf("\n--- Tokenizer: word-char boundary table agrees with the class table ---\n");
	/* A backslash in script CODE is a word char: the whole path is one token. The
	   expected string is written with doubled backslashes because the string-literal
	   path decodes '\\' to a single backslash. */
	test_true("backslash inside a word is not a boundary",
	          "Make \"p\" c:\\temp\\file.txt Output Equal :p \"c:\\\\temp\\\\file.txt\"");
	/* Standalone-token half of the same bug: a backslash could not appear in
	   expression position at all, because the dead atom desynced the stream. */
	test_num("backslash word token usable as an argument",
	         "Make \"p\" c:\\x Output TextLength :p", 4);
	/* The 0x50-0x5F row also carries '[' , ']' , '^' and '_' -- the fix must
	   leave every one of them a boundary. '12^10' with NO spaces is the probe:
	   if '^' stopped being a boundary the token would read as one word "12^10". */
	test_num("caret is still a boundary (no spaces around it)", "Output 12^10", 6);
	test_num("bitwise and is still a boundary (no spaces around it)", "Output 12&10", 8);
	test_num("bitwise or is still a boundary (no spaces around it)", "Output 12|10", 14);
	test_num("underscore concat is still a boundary (no spaces)", "Output TextLength \"ab\"_\"cd\"", 4);
	/* Braces were boundaries mid-word but word-accum at token start, so the same
	   character behaved differently by position. */
	test_true("braces inside a word are not boundaries",
	          "Make \"b\" a{b}c Output Equal :b \"a{b}c\"");
	test_num("brace word token usable as an argument",
	         "Make \"b\" a{b}c Output TextLength :b", 5);
}

void test_tokenizer_case_folding()
{
	printf("\n--- Tokenizer: case folding (case-insensitive names) ---\n");
	/* All A-Z are folded to a-z at token store time. Builtins are all lowercase.
	   So Make/make/MAKE all intern to "make" and match the builtin. */
	test_num("make lowercase", "make \"x\" 42 Output :x", 42);
	test_num("MAKE uppercase", "MAKE \"y\" 99 Output :y", 99);
	test_num("MaKe mixed case", "MaKe \"z\" 7 Output :z", 7);
	test_num("output lowercase", "Make \"a\" 3 output :a", 3);
	test_num("REPEAT uppercase", "Make \"c\" 0 REPEAT 3 [ Make \"c\" :c + 1 ] Output :c", 3);
	test_num("ifelse mixed", "IfElse 1 [ Make \"d\" 10 ] [ Make \"d\" 20 ] Output :d", 10);
	test_num("define mixed case", "DeFiNe \"inc\" [:x] [ Output :x + 1 ] Output inc 5", 6);
}

void test_tokenizer_unary_minus_gluing()
{
	printf("\n--- Tokenizer: unary minus/plus gluing ---\n");
	/* When '-' appears in operand position (after an operator, after '(', or at start),
	   it glues to the following number token. So "-5 + 10" produces (-5) + 10 = 5. */
	test_num("unary minus at start", "Output (-5 + 10)", 5);
	test_num("unary minus after paren", "Output ((-3) + 8)", 5);
	test_num("double unary minus", "Output Negate -3", 3);
	test_num("unary plus", "Output (+5 + +3)", 8);
	/* Unary minus before a variable: - :x is Subtract(0, :x) via the unary path */
	test_num("unary minus before var", "Make \"v\" 5 Output (0 - :v)", -5);
	/* Comment separates tokens exactly like whitespace: the '-' after
	   "1;comment<LF>" must glue as a UNARY sign (5 + (-2) = 3), not infix 5-2=3
	   -- before the reader set the leading-ws flag on the comment path, the
	   sign glued as binary and subtract consumed the 5 */
	test_num("comment acts as token separator for sign glue", "Output (1 + ;comment\n-2)", -1);
	test_num("comment CRLF acts as token separator for sign glue", "Output (1 + ;comment\r\n-2)", -1);
	/* 30-bit signed payload is [-2^29, 2^29-1] = [-536870912, 536870911]. In
	   sl_sort_coerce_num the guard fired at acc == 53687091 unconditionally, so
	   magnitudes of 536870910 / 536870911 and the negative -536870912 were promoted
	   to FLOAT instead. Float spacing near 2^29 is 64, so any two such magnitudes
	   round to the SAME float and TIE -- the ordering is what these probes detect.
	   (Literal numbers in script source go through the reader's own parser, not
	   sl_sort_coerce_num, so the probe must be Sort, not IsNumber.) */
	test_true("Sort: 536870848 precedes 536870911 (they tie as rounded FLOATs when the boundary bug is present)",
	          "Output ((Item Sort [ \"536870911\" \"536870848\" ] 0) = \"536870848\")");
	test_true("Sort: -536870912 precedes -536870848 (negative payload boundary, same tie)",
	          "Output ((Item Sort [ \"-536870848\" \"-536870912\" ] 0) = \"-536870912\")");
}

void test_name_validation()
{
	printf("\n--- Name validation (reject invalid names) ---\n");
	/* Names with hyphens, spaces, or special chars are rejected by FATAL.
	   We cannot test FATALs directly (they terminate the process), but we
	   can test that valid names with digits and letters work. */
	/* Valid names: letters and digits only */
	test_num("valid name with digits", "Define \"test2\" [] [ Output 42 ] Output test2", 42);
	test_num("valid name all letters", "Define \"hello\" [] [ Output 7 ] Output hello", 7);
	/* The old hyphenated names are gone: V7-DB-PUT is now DatabaseSet.
	   Hyphens in user-defined names are FATAL (tested manually):
	     Define "bad-name" [] [ Output 1 ]  -> FATAL invalid name
	*/
}

/* Suite 71 (Sep 2026, opt_and_bug batch B): TERMINATE dedicated flag, LOCAL
   function-name guard, param-list case fold, nested DO_VQ stack staging,
   AVX2 case-fold windows, FFI ret-type pre-validation. */
void test_batch_b_regressions()
{
	printf("\n--- Batch B regressions (TERMINATE / LOCAL guard / param fold / DO_VQ) ---\n");
	/* TERMINATE (live name: Exit -- "terminate" is a DEAD name after the Sec.169
	   rename) inside a user fn must exit the WHOLE script: the fn-boundary
	   state restore swallows the old shared has_returned flag; the dedicated
	   sl_terminate_requested flag survives it. If the Make after the call ran,
	   the script returns NUMBER 1; a terminated script returns NIL (raw 0). */
	test_nil("Exit inside a fn exits the script",
	         "Define \"g\" [] [ Exit ] g Make \"x\" 1 Output :x");
	/* same via a REPEAT body: Exit must break out of every nesting level */
	test_nil("Exit in a fn inside REPEAT exits the script",
	         "Define \"g\" [] [ Exit ] Repeat 2 [ g Make \"x\" 1 ] Output :x");
	/* top-level Exit right before a side effect */
	test_nil("Exit at top level exits the script",
	         "Make \"x\" 0 Exit Make \"x\" 1 Output :x");
	/* LOCAL must stop its name list at a user-fn name: h stays callable.
	   With the swallow bug, Local binds h as a variable and Output h reads NIL. */
	test_num("LOCAL name list stops at a user function name",
	         "Define \"h\" [] [ Output 99 ] Local \"a\" h Output h",
	         99);
	/* quoted uppercase param names fold to the lexer-folded :VAR lookups.
	   The string literal keeps its case; the body's :x is folded to x. */
	test_num("Quoted uppercase param folds to lowercase",
	         "Define \"f\" [\"X\"] [ Output :x ] Output f 5",
	         5);
	/* nested DO_VQ (live name: NearestMatch -- "dovq" is a DEAD name after the
	   Sec.169 rename): the inner full-mode call runs while the outer vector is
	   staged -- the outer query now lives on the stack, so the inner staging
	   cannot clobber it. Outer dist = (0-0)^2 = 0; with the clobber bug the
	   buffer held the inner vector [5 5] and dist was 25. */
	test_float_val("nested NearestMatch keeps the outer vector",
	               "Make \"r\" NearestMatch False [0] NearestMatch True [5 5] [[7 5 5] [8 4 4]] "
	               "Output First ButFirst :r",
	               0.0f);
	/* AVX2 case-fold windows (>16 chars exercises the vector path; the
	   high-bit bytes must pass through unfolded) */
	test_true("Lowercase 24-char vector window",
	          "Output ((Lowercase \"ABCDEFGHIJKLMNOPQRSTUVWX\") = \"abcdefghijklmnopqrstuvwx\")");
	test_true("Uppercase 24-char vector window",
	          "Output ((Uppercase \"abcdefghijklmnopqrstuvwx\") = \"ABCDEFGHIJKLMNOPQRSTUVWX\")");
	test_true("Lowercase passes high-bit bytes through",
	          "Output ((Lowercase \"ABC\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9" "DEF\") = \"abc\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9" "def\"");
	/* FFI: an unknown return type must FATAL BEFORE the native call -- the
	   message comes from the pre-call validation site */
	test_fatal_catch_case("FATAL C-CALL unknown ret type pre-call",
	                      "Output CallLibrary [\"SELF\" \"v7s_get_num_workers\" \"wrongtype\" []]",
	                      "return type must be");
	/* a FLOAT in the parameter-list position is a malformed LIST, not a
	   malformed parameter: the FLOAT-band guard routes it to the list-level
	   FATAL, which names the offender (a misdiagnosis here said
	   "parameter must be a name") */
	test_fatal_catch_case("FATAL Define FLOAT params list names the list",
	                      "Define \"f\" 1.5 [ ]",
	                      "parameter list must be a list");
	/* EnableColors (CONSOLE-24BIT) must return the TRUE atom or NIL -- never
	   the stale probe value 2 (tagged NUMBER) the cached-OFF path used to
	   leak in eax. Piped stdout (validator under verify.bat) takes the OFF
	   path where the bug fired; (= 2 2) would make this Output 1. */
	test_num("EnableColors returns TRUE or NIL, never stale 2",
	         "IfElse ((EnableColors) = 2) [Output 1] [Output 0]",
	         0);
	/* EncodeWord with a NIL word must name the WORD as the offender, not
	   claim a missing hook context (the context is fine; the argument is NIL) */
	test_fatal_catch_case("FATAL EncodeWord NIL word names the offender",
	                      "Output EncodeWord :nosuchvar",
	                      "requires a word argument");
	/* max_tokens arg5 bug: sectorlogo_v7prim built
	   max_tokens = len+2 in edx, then line 60 (mov rdx, rsi) claimed edx for
	   arg2, so the arg5 slot received the low 32 bits of the text pointer
	   instead. The low 32 bits of a user-mode heap address are normally a large
	   positive number, so the bound was silently DISABLED rather than too tight
	   -- address-layout dependent, and no short-word test could have caught it.

	   The FIX is verified by inspection, not from here: reaching the encode()
	   call needs a live sl_hook_ctx, and this pure-interp harness never
	   establishes one (on_token stores it; nothing in the validator runs the
	   hook), so every positive EncodeWord path here dies at the ctx FATAL on
	   line 40 before the arg5 slot is ever written. A harness that creates an
	   inference context and then calls EncodeWord would be a NEW tool, which
	   needs user consent (Sec.179), so it is not being added unilaterally.

	   What IS observable from this harness, and is pinned below, is the guard
	   ORDER in sl_v7_encode_word: the word-atom guard must run BEFORE the ctx
	   check. A valid WORD therefore names the missing context, and only a
	   non-atom argument names the word -- which is what makes the
	   "requires a word argument" message above reachable at all. */
	test_fatal_catch_case("FATAL EncodeWord on a valid word names the missing host context",
	                      "Output EncodeWord \"hello\"",
	                      "requires host hook context");
	test_fatal_catch_case("FATAL EncodeWord on a NUMBER names the word offender",
	                      "Output EncodeWord 42",
	                      "requires a word or string argument, got number '42'");
	/* v7prim diagnostic rewrite: every host-primitive
	   FATAL named a DEAD name (V7-DB-GET / V7-DB-PUT / V7-SET-STREAM), and a dead
	   name is a name a script author cannot type. Each guard is also flags-only
	   (rejects a tag bit), so a plain WORD atom is legal input -- "requires a
	   string" stated a contract the code never enforced. */
	test_fatal_catch_case("FATAL DatabaseGet NUMBER key names the offender",
	                      "Output DatabaseGet 42",
	                      "got number '42'");
	test_fatal_catch_case("FATAL DatabaseGet message names the live primitive",
	                      "Output DatabaseGet 42",
	                      "DatabaseGet requires a word or string key");
	/* DatabaseSet and SetStream are STMT_ENTRY-only (no EXPR_ENTRY), so in
	   expression position the name is an unbound atom that SELF-EVALUATES
	   (Sec.146 fact 43) and the FATAL is never reached -- these must be bare
	   statements, or the test would pass for the wrong reason. */
	test_fatal_catch_case("FATAL DatabaseSet NUMBER key names the offender",
	                      "DatabaseSet 42 \"v\"",
	                      "got number '42'");
	test_fatal_catch_case("FATAL DatabaseSet list value names the offender",
	                      "DatabaseSet \"k\" [ 1 2 ]",
	                      "got 'list'");
	test_fatal_catch_case("FATAL SetStream NUMBER argument names the offender",
	                      "SetStream 42 Output 1",
	                      "got number '42'");
	/* SetOutputToken is STMT_ENTRY-only like SetStream: in expression position the
	   name would self-evaluate and never reach the primitive. The guard order
	   is arg-type first (NumVal), then the hook-context check -- a standalone
	   interp has no engine generation running, so a VALID id names the missing
	   token hook. */
	test_fatal_catch_case("FATAL SetOutputToken NUMBER argument outside the hook names the missing token hook",
	                      "SetOutputToken 5 Output 1",
	                      "SetOutputToken requires the token hook");
	test_fatal_catch_case("FATAL SetOutputToken non-number argument names the offender",
	                      "SetOutputToken \"x\" Output 1",
	                      "a NUMBER was required");
	/* a plain word atom IS legal for every one of these guards -- the case the
	   old "requires a string" wording called an error */
	test_true("DatabaseGet on a word key is legal input", "DatabaseSet hello \"v\" Output Equal DatabaseGet hello \"v\"");
	/* lexer state leak regression: a stray-']' FATAL mid-parse used to leave
	   sl_list_depth at -1 in BSS; the NEXT script's first ']' then re-FATALed
	   "stray ']'" instantly (full-run-only cascade -- standalone suites passed).
	   Run the fatal script first, then a bracketed script MUST evaluate fine. */
	test_fatal_catch_case("FATAL stray bracket (lexer-leak setup)",
	                      "Output 1 ]]",
	                      "stray ']'");
	test_num("lexer state resets after a stray-']' FATAL",
	         "Define \"h\" [] [ Output 42 ] Output h",
	         42);
}

/* V7_PCT_STMT register-clobber bug: the 64-bit
   pointer load `mov rax, [rel sl_hook_temp_ptr]` used to run BEFORE the
   vcvtsi2ss, destroying the eax that NumVal had just returned -- so
   SetTemperature 80 stored float(low32(&temperature))/100.0 into the
   sampler instead of 0.80 (silent: the live hook publishes non-NULL
   pointers, so the NULL early-out never fired).

   The live hook is the only observer (standalone interp has NULL hook
   pointers and skips the write), so this suite arms a real sidecar, re-arms
   lg_init via v7_free_inference (the destroy path resets the init latch and
   wipes the table -- the sidecar key must be added AFTER the revival), and
   drives v7s_sectorlogo_on_token directly with a minimal context that fakes
   the decode chain (ctx+8 tokenizer -> tokenizer+0 vocab -> vocab[42]). */
void test_decode_byte_fallback_reject()
{
	/* lg_decode_token_real byte-fallback REJECTION path.

	   This path is unreachable from the plain-eval suites: with no sidecar armed
	   the hook takes .tok_no_callback, which passes the token through with no
	   decode, no intern and no eval (sectorlogo_main.inc). So the test has to arm
	   a sidecar and drive on_token with the decode chain faked, exactly as
	   test_v7prim_pct_hook does. It runs BEFORE that suite (which re-arms for
	   itself and must stay last).

	   The bug: at the lo-nibble load the decoder used eax as the holding
	   register, so on the non-hex-lo rejection branch `al` no longer held the '<'
	   that lodsb had consumed -- stosb stored a NUL instead, and the whole token
	   truncated to empty. The fix restores '<' on that branch only, leaving the
	   valid <0xXX> decode (which needs the lo nibble in al for the final
	   `or al, dl`) untouched. Both halves are checked here: the valid escape must
	   still collapse to one character, and the rejection must keep its six. */

	static char fake_ctx[64];
	static char fake_tokenizer[2096];
	static char *fake_vocab[43];
	static char token_text[] = "<0x1z>";
	static char valid_text[] = "<0x41>";
	float temperature;
	float topp;
	int out_token;
	int action_code;
	const char *sidecar_script = "SetTemperature TextLength :Piece";

	printf("\n--- byte-fallback decode: non-hex lo char must keep the '<' ---\n");

	inference_dll_ptr->v7_free_inference();
	if (!inference_dll_ptr->ht_add("logo:sidecar", (void *)sidecar_script, (unsigned __int64)(strlen(sidecar_script) + 1), 0))
	{
		printf("  FAIL: ht_add('logo:sidecar') failed\n");
		failing_tests_counter++;
		return;
	}

	char **vocab_base = fake_vocab;
	char *tokenizer_base = fake_tokenizer;
	fake_vocab[42] = token_text;
	memcpy(fake_tokenizer, &vocab_base, sizeof(char *));
	memcpy(fake_ctx + 8, &tokenizer_base, sizeof(char *));

	temperature = -1.0f;
	topp = -1.0f;
	out_token = 0;
	action_code = inference_dll_ptr->v7s_sectorlogo_on_token((inference_context *)fake_ctx, 42, 3, &out_token, &temperature, &topp);
	if (action_code != 0)
	{
		printf("  FAIL: on_token action code %d (expected 0 = ACCEPT)\n", action_code);
		failing_tests_counter++;
	}
	/* SetTemperature divides its argument by 100, so 6/100 is the exact value the
	   six-character token must produce. With the bug the atom was empty (TextLength
	   0) and temperature came back 0.0. */
	if (temperature != 0.06f)
	{
		printf("  FAIL: '<0x1z>' decoded to %.9g, expected 0.06 (a NUL was stored where the '<' belongs, truncating the atom)\n", temperature);
		failing_tests_counter++;
	}
	else
	{
		printf("  pass: non-hex lo char keeps the '<' -> token length 6\n");
		passing_tests_counter++;
	}

	/* Same harness, valid escape: '<0x41>' must still collapse to the single
	   character 'A'. This is the half the fix must NOT disturb -- the lo nibble
	   has to reach al for the final `or al, dl`. */
	fake_vocab[42] = valid_text;
	temperature = -1.0f;
	topp = -1.0f;
	out_token = 0;
	action_code = inference_dll_ptr->v7s_sectorlogo_on_token((inference_context *)fake_ctx, 42, 3, &out_token, &temperature, &topp);
	if (action_code != 0)
	{
		printf("  FAIL: on_token action code %d (expected 0 = ACCEPT)\n", action_code);
		failing_tests_counter++;
	}
	if (temperature != 0.01f)
	{
		printf("  FAIL: '<0x41>' decoded to %.9g, expected 0.01 (a valid escape must still collapse to one character)\n", temperature);
		failing_tests_counter++;
	}
	else
	{
		printf("  pass: valid escape still collapses -> token length 1\n");
		passing_tests_counter++;
	}
}

void test_v7prim_pct_hook()
{
	/* The hook decodes the token id before running the script
	   (lg_decode_token_real: ctx->tokenizer->vocab[token]), so the minimal
	   context must fake that exact chain -- offsets per sectorlogo_const.inc:
	   SL_CTX_TOKENIZER = 8 (qword Tokenizer*), SL_TOK_VOCAB = 0 (qword char**),
	   SL_CTX_OUT_TOKENS = 32 (qword int*, NULL is fine: this script calls no
	   GetToken). The vocab array needs one entry at the token id used below. */
	static char fake_ctx[64];
	static char fake_tokenizer[2096];          /* TOK_IS_SP (2080, dword) stays 0 -- unused by SetTemperature */
	static char *fake_vocab[43];               /* token id 42 indexes the last entry */
	static char token_text[] = "hello";
	float temperature;
	float topp;
	int out_token;
	int action_code;
	const char *sidecar_script = "SetTemperature 80 SetNucleus 25";

	printf("\n--- V7_PCT_STMT live-hook percentage conversion ---\n");
	/* Re-arm lg_init FIRST: the destroy path also wipes and re-inits the
	   microdb table (inf_cleanup steps 2-3), so the sidecar key must be
	   added AFTER the revival -- adding it before would be destroyed. The
	   validator's earlier suites already consumed the one-time init WITHOUT
	   a sidecar (the qwen3 DB ships none); clearing sl_lg_initialized
	   (inf_cleanup step 4) makes the next hook call re-run lg_init and
	   parse the script added below. This suite runs LAST: the table stays
	   wiped (qwen3 keys gone) afterwards, and nothing runs after it. */
	inference_dll_ptr->v7_free_inference();
	/* ht_find in lg_init/on_token always reads the global hash table v7_init
	   loaded -- the sidecar key present in that table IS the switch */
	if (!inference_dll_ptr->ht_add("logo:sidecar", (void *)sidecar_script, (unsigned __int64)(strlen(sidecar_script) + 1), 0))
	{
		printf("  FAIL: ht_add('logo:sidecar') failed\n");
		failing_tests_counter++;
		return;
	}

	/* Wire the fake tokenizer chain the hook's decode path walks: the struct
	   fields must hold the ADDRESSES of the next level (content copies would
	   store whatever the first 8 bytes of the array hold -- NULL here). */
	char **vocab_base = fake_vocab;            /* its VALUE is the char** array address */
	char *tokenizer_base = fake_tokenizer;     /* its VALUE is the fake Tokenizer address */
	fake_vocab[42] = token_text;
	memcpy(fake_tokenizer, &vocab_base, sizeof(char *));    /* SL_TOK_VOCAB = 0: char** vocab */
	memcpy(fake_ctx + 8, &tokenizer_base, sizeof(char *));  /* SL_CTX_TOKENIZER = 8 */
	/* fake_ctx[32..39] (SL_CTX_OUT_TOKENS) stays NULL */

	temperature = -1.0f;
	topp = -1.0f;
	out_token = 0;

	printf("  [dbg] ctx=%p ctx+8=%p tok=%p vocab[42]=%p -> calling on_token\n",
	       (void *)fake_ctx, *(void **)(fake_ctx + 8), (void *)fake_tokenizer, (void *)fake_vocab[42]);
	fflush(stdout);

	action_code = inference_dll_ptr->v7s_sectorlogo_on_token((inference_context *)fake_ctx, 42, 3, &out_token, &temperature, &topp);
	if (action_code != 0)
	{
		printf("  FAIL: on_token action code %d (expected 0 = ACCEPT)\n", action_code);
		failing_tests_counter++;
	}

	/* Both results are exact: 80/100 and 25/100 are correctly-rounded IEEE
	   divides of exact operands, bit-identical to the C literals 0.8f/0.25f. */
	if (temperature != 0.8f)
	{
		printf("  FAIL: SetTemperature 80 stored %.9g, expected 0.8 (the clobber bug writes float(low32(ptr))/100 here)\n", temperature);
		failing_tests_counter++;
	}
	else
	{
		printf("  pass: SetTemperature 80 -> temperature = 0.8\n");
		passing_tests_counter++;
	}
	if (topp != 0.25f)
	{
		printf("  FAIL: SetNucleus 25 stored %.9g, expected 0.25\n", topp);
		failing_tests_counter++;
	}
	else
	{
		printf("  pass: SetNucleus 25 -> topp = 0.25\n");
		passing_tests_counter++;
	}
}

void test_batch_d_regressions()
{
	printf("\n--- Batch D regressions (Equal r9 clobber / DO_VQ NaN-best / radix binary64 / parser / reader operand) ---\n");
	/* Equal loses its second operand when the FIRST is a numeric text atom:
	   sl_eq_floatize -> sl_atom_float uses r9d as its fraction accumulator,
	   so the un-saved rhs tag became garbage. Runtime-built atoms ("2" _ "5")
	   keep the operand an ATOM (a quoted numeric literal parses to NUMBER). */
	test_true("Equal runtime atom int vs FLOAT",
	          "Output ((\"2\" _ \"5\") = 25.0)");
	test_true("Equal FLOAT vs runtime atom int (reversed order)",
	          "Output (25.0 = (\"2\" _ \"5\"))");
	test_true("Equal runtime atom fraction vs FLOAT",
	          "Output ((\"2\" _ \".5\") = 2.5)");
	test_num("Equal runtime atom vs different FLOAT",
	         "Output IfElse ((\"2\" _ \".5\") = 2.6) [1] [0]",
	         0);
	test_num("Equal non-numeric atom vs FLOAT",
	         "Output IfElse ((\"x\" _ \"2\") = 2.0) [1] [0]",
	         0);
	/* sl_atom_float accumulation width. It summed in
	   binary32 and rounded again after the add, while the FLOAT LITERAL parser
	   accumulates in binary64 and narrows exactly once -- so a 5-digit integer part
	   plus 4 fraction digits could land on different binary32 values on the two
	   paths, and `Equal` then answered NIL for a value the script can plainly write.
	   This text atom is 9 significant characters, so the binary32 int conversion
	   (> 2^24) rounds on both the int and the add.
	   The negative probe below uses a value more than one ulp away: near 12345 the
	   binary32 ulp is 2^-10 (~9.77e-4), so 12345.6790 and 12345.6789 are the SAME
	   binary32 and `Equal` is right to call them equal. */
	test_true("Equal text atom vs FLOAT: 12345.6789",
	          "Output ((\"12345\" _ \".6789\") = 12345.6789)");
	test_true("Equal text atom vs FLOAT: reversed operand order",
	          "Output (12345.6789 = (\"12345\" _ \".6789\"))");
	test_true("Equal text atom vs FLOAT: 9 int digits",
	          "Output ((\"123456789\" _ \".25\") = 123456789.25)");
	test_num("Equal text atom still differs from a different FLOAT",
	         "Output IfElse ((\"12345\" _ \".6789\") = 12345.7) [1] [0]",
	         0);
	/* DO_VQ: a NaN FIRST codebook row used to win forever (vucomiss unordered
	   skipped every finite challenger). Fixed: the first finite row replaces a
	   NaN best. List literals are DATA (components staged via raw Car, no
	   element evaluation) -- the NaN rows are built with Prepend so that
	   SquareRoot -4.0 is evaluated as an expression (vsqrtss -> QNaN). */
	test_num("NearestMatch NaN first row replaced by finite",
	         "Make \"n\" SquareRoot -4.0 "
	         "Make \"r1\" Prepend 1 Prepend :n [] "
	         "Make \"r2\" Prepend 2 Prepend 3.0 [] "
	         "Make \"cb\" Prepend :r1 Prepend :r2 [] "
	         "Output First NearestMatch False [0.0] :cb",
	         2);
	test_num("NearestMatch NaN row after finite winner ignored",
	         "Make \"n\" SquareRoot -4.0 "
	         "Make \"r1\" Prepend 1 Prepend 3.0 [] "
	         "Make \"r2\" Prepend 2 Prepend :n [] "
	         "Make \"cb\" Prepend :r1 Prepend :r2 [] "
	         "Output First NearestMatch False [0.0] :cb",
	         1);
	test_num("NearestMatch all-NaN rows keep the first",
	         "Make \"n\" SquareRoot -4.0 "
	         "Make \"r1\" Prepend 5 Prepend :n [] "
	         "Make \"r2\" Prepend 6 Prepend :n [] "
	         "Make \"cb\" Prepend :r1 Prepend :r2 [] "
	         "Output First NearestMatch False [0.0] :cb",
	         5);
	/* Radix sort binary64 keys: distinct NUMBERs above 2^24 used to collide
	   on one binary32 key (stable sort kept the descending input order). */
	test_num("Sort orders adjacent NUMBERs at 2^24",
	         "Output First Sort [16777217 16777216]",
	         16777216);
	test_num("Sort orders adjacent NUMBERs at 2^26",
	         "Output First Sort [67108865 67108864]",
	         67108864);
	test_num("Sort negative numbers",
	         "Output First Sort [-5 3 -10 0]",
	         -10);
	/* NaN sort keys keep their IEEE totalOrder slot, no longer normalized to
	   +0.0. SquareRoot of a negative yields x86's NEGATIVE default QNaN
	   (0xFFC00000): totalOrder places it BELOW -inf. Negate flips the sign
	   bit -> POSITIVE NaN, which sorts above +inf: [0.0 +NaN 1.0] -> 0.0,
	   1.0, +NaN. NaN built at runtime (list literals do not evaluate their
	   elements); Equal can never match a NaN (IEEE) so positions are probed
	   via the FINITE neighbours. */
	test_num("Sort positive NaN key above +inf",
	         "Make \"p\" Negate SquareRoot -4.0 "
	         "Make \"L\" Prepend 0.0 Prepend :p Prepend 1.0 [] "
	         "Output IfElse ((First ButFirst (Sort :L)) = 0.0) [1] [0]",
	         0);
	test_num("Sort negative NaN key below -inf",
	         "Make \"n\" SquareRoot -4.0 "
	         "Make \"L\" Prepend 0.0 Prepend :n Prepend 1.0 [] "
	         "Output IfElse ((Item (Sort :L) 1) = 0.0) [1] [0]",
	         1);
	/* Fraction parser: the 7-digit forward cap truncated leading-zero
	   fractions to +0.0 ("0.00000001" -> 0). Backward binary64 accumulation
	   keeps it nonzero. The atom is runtime-built so the sort-coercion parser
	   (.scn) handles it, not the literal lexer. */
	test_num("leading-zero fraction survives sort coercion",
	         "Make \"t\" (\"0.0000000\" _ \"1\") "
	         "Make \"L\" Prepend 1.0 Prepend :t Prepend 2.0 [] "
	         "Output IfElse ((First (Sort :L)) = 0.0) [1] [0]",
	         0);
	/* Reader: a glued signed token (-5) was classified non-operand, so the
	   compact -3 after it ALSO glued (desync) instead of binary subtraction. */
	test_num("compact signed-literal infix subtracts",
	         "Output -5-3",
	         -8);
	test_num("compact signed-literal infix adds",
	         "Output -5+3",
	         -2);
}

void test_batch_d_regressions_ifelse_value()
{
	printf("\n--- IfElse value form + statement-in-expression FATAL ---\n");
	/* IfElse in EXPRESSION position used to have no expr dispatch entry: the
	   atom silently self-evaluated to its own name (truthy garbage) and the
	   remaining forms executed as top-level statements (stream desync). Now a
	   single-form chosen block yields its expression value (the documented
	   IfElse cond [expr] [expr] selector idiom). */
	test_num("IfElse value form true branch",
	         "Output IfElse IsNumber 1 [11] [22]",
	         11);
	test_num("IfElse value form false branch",
	         "Output IfElse 0 [11] [22]",
	         22);
	test_num("IfElse value form expression block",
	         "Output IfElse 1 [(2 + 3)] [9]",
	         5);
	test_num("IfElse value form nested in arithmetic",
	         "Output (100 + (IfElse IsNumber 7 [1] [2]))",
	         101);
	test_num("IfElse value form bound via Make",
	         "Make \"x\" IfElse IsNumber 1 [7] [8] Output :x",
	         7);
}

void test_enabletrace()
{	printf("\n--- EnableTrace ---\n");
	/* EnableTrace as statement: level 0 disables, no crash */
	test_num("EnableTrace 0 no crash", "EnableTrace 0 Output 1", 1);
	/* EnableTrace 1 then 0: no crash, normal execution resumes */
	test_num("EnableTrace 1 then 0", "EnableTrace 1 Make \"x\" 42 EnableTrace 0 Output :x", 42);
	/* EnableTrace 2 then 0: expression results traced but script still correct */
	test_num("EnableTrace 2 then 0", "EnableTrace 2 Make \"x\" 3 + 4 EnableTrace 0 Output :x", 7);
	/* EnableTrace expression form returns old level */
	test_num("EnableTrace expr returns old level 0", "Output EnableTrace 0", 0);
	/* EnableTrace 2 returns old level 0, then EnableTrace :old restores */
	test_num("EnableTrace roundtrip save/restore",
	         "Make \"old\" EnableTrace 2 Make \"x\" 99 EnableTrace :old Output :x",
	         99);
	/* Verify old level was 0 (before we set it to 2) */
	test_num("EnableTrace old level was 0",
	         "Make \"old\" EnableTrace 2 EnableTrace 0 Output :old",
	         0);
	/* EnableTrace 3: max level, no crash */
	test_num("EnableTrace 3 no crash", "EnableTrace 3 Make \"x\" 1 EnableTrace 0 Output :x", 1);
	/* EnableTrace does not affect computation results */
	test_num("EnableTrace computation unaffected",
	         "EnableTrace 2 Make \"a\" 10 + 20 EnableTrace 0 Make \"b\" :a * 2 Output :b",
	         60);
	/* Out-of-range levels must die, not silently clamp (Sec.57d / Sec.154).
	   The expression form reports the offender; the statement form has its own
	   handler, so both are covered. */
	test_fatal_catch_case("EnableTrace above max names the offender",
	                      "Output EnableTrace 99",
	                      "EnableTrace level must be 0..3");
	test_fatal_catch_case("EnableTrace above max names the tagged value",
	                      "Output EnableTrace 99",
	                      "got number '99'");
	test_fatal_catch_case("EnableTrace negative names the offender",
	                      "Output EnableTrace (0 - 1)",
	                      "EnableTrace level must be 0..3");
	test_fatal_catch_case("EnableTrace statement form above max also dies",
	                      "EnableTrace 42 Output 1",
	                      "EnableTrace level must be 0..3");
	/* single-WriteFile indent: sl_trace_print_indent now
	   emits depth*2 spaces in ONE PrintRaw instead of one PutChar per space, so the
	   multi-depth and clamp paths must still run and leave the result untouched.
	   Every script here restores the level with EnableTrace 0 BEFORE the final Output
	   on purpose: a top-level Output ends the script immediately, so a trailing
	   EnableTrace 0 would never run (and sl_trace_level is now reset per interp call
	   anyway -- but the tests must not rely on that to stay quiet). */
	test_num("EnableTrace nested block indent does not disturb the result",
	         "Make \"acc\" 0 EnableTrace 2 Repeat 3 [ Make \"acc\" :acc + 5 ] EnableTrace 0 Output :acc",
	         15);
	test_num("EnableTrace nested block indent at max level",
	         "Make \"acc\" 0 EnableTrace 3 Repeat 2 [ Repeat 2 [ Make \"acc\" :acc + 1 ] ] EnableTrace 0 Output :acc",
	         4);
	/* sl_trace_level reset: a script that ends while tracing
	   is on (here via top-level Output, which never returns) must not leave tracing
	   enabled for the NEXT interp call. The second script would emit trace output --
	   and on a cyclic list that is an unbounded-output hang. */
	test_num("script leaving trace on does not poison the next script",
	         "EnableTrace 3 Output 1",
	         1);
	test_num("next script after a trace-leaving script is still quiet and correct",
	         "Make \"l\" Prepend 1 Prepend 2 [] SetCdr ButFirst :l :l Output Item :l 0",
	         1);
}

/* EnableStrictNames (A1): the opt-in gate that turns "a dead name in
   expression position silently self-evaluates" into a FATAL naming it. Default
   is OFF, so bare-word data values keep working; ON arms exactly ONE edge of
   .ret_bare_atom (the Assoc miss), never the self-evaluating literal tags, an
   unbound LOCAL's NIL binding, or a bound non-fn value. */
void test_enablestrictnames()
{
	printf("\n--- EnableStrictNames ---\n");
	/* default OFF: the dead name self-evaluates to a WORD atom (the historic
	   behaviour scripts -- and the Scheme compiler's symbol constants -- rely on) */
	test_true("strict off by default: dead name self-evaluates",
	          "Make \"dn\" NumberP Output IsWordAtom :dn");
	test_true("strict off: dead name in an IfElse condition is truthy",
	          "Output IfElse NumberP [1] [0]");
	/* flag round-trips through both forms */
	test_num("EnableStrictNames expression form returns old flag 0", "Output EnableStrictNames 0", 0);
	test_num("EnableStrictNames statement form arms and disarms",
	          "EnableStrictNames 1 Make \"x\" 42 EnableStrictNames 0 Output :x", 42);
	test_num("EnableStrictNames expression form returns the PREVIOUS flag",
	          "EnableStrictNames 1 Make \"wason\" EnableStrictNames 0 Output :wason", 1);
	/* strict ON: ordinary computation and data are untouched */
	test_num("strict on leaves arithmetic alone", "EnableStrictNames 1 Output (2 + 3)", 5);
	test_num("strict on leaves strings alone", "EnableStrictNames 1 Output TextLength \"abc\"", 3);
	test_true("strict on leaves predicates alone",
	          "EnableStrictNames 1 Output IsStringAtom \"abc\"");
	/* strict ON: a dead name dies naming itself */
	test_fatal_catch_case("strict on: dead name in expression position FATALs",
	                      "EnableStrictNames 1 Output NumberP",
	                      "unbound name in expression position");
	test_fatal_catch_case("strict on: the FATAL names the dead name",
	                      "EnableStrictNames 1 Output NumberP",
	                      "got word 'numberp'");
	test_fatal_catch_case("strict on: expression form arms it too",
	                      "Output ((EnableStrictNames 1) + NumberP)",
	                      "unbound name in expression position");
	/* the range check: 0/1 only, never clamped (Sec.57d) */
	test_fatal_catch_case("EnableStrictNames above range names the offender",
	                      "EnableStrictNames 2 Output 1",
	                      "EnableStrictNames flag must be 0 or 1");
	test_fatal_catch_case("EnableStrictNames negative names the offender",
	                      "EnableStrictNames (0 - 1) Output 1",
	                      "EnableStrictNames flag must be 0 or 1");
	/* a caught FATAL must not leave the flag armed for the next interp call
	   (SL_RESET_CALL_STATE): the script below would FATAL on its own dead name
	   if the previous test's arming had leaked. */
	test_true("strict mode does not leak across calls after a caught FATAL",
	          "Make \"dn2\" NumberP Output IsWordAtom :dn2");
	/* RealRandom: same 0/1-only contract as EnableStrictNames, same FATAL shape
	   (domain check; the flag persists across calls --
	   NOT in SL_RESET_CALL_STATE -- so arm then disarm to leave it at 0). */
	test_num("RealRandom arms and returns the previous flag",
	          "Make \"rrwas\" RealRandom 1 Output :rrwas", 0);
	test_num("RealRandom disarms and returns the previous flag",
	          "Make \"rroff\" RealRandom 0 Output :rroff", 1);
	test_fatal_catch_case("RealRandom above range names the offender",
	                      "Output RealRandom 2",
	                      "RealRandom flag must be 0 or 1");
	test_fatal_catch_case("RealRandom negative names the offender",
	                      "Output RealRandom (0 - 1)",
	                      "RealRandom flag must be 0 or 1");
	/* statement form (the dispatch-table pair rule): a bare `RealRandom 1`
	   statement must ARM the flag, not fall back to the skipped-word warning.
	   The proof reads the flag back through the expression form: with the
	   statement row missing the arm is a no-op and the read-back returns 0. */
	test_num("RealRandom statement form arms (expression form reads it back)",
	         "RealRandom 1 Make \"was\" RealRandom 0 Output :was", 1);
	test_num("RealRandom statement form disarms too",
	         "RealRandom 1 RealRandom 0 Make \"was\" RealRandom 1 Output :was", 0);
	test_true("Random under an armed statement-form flag stays numeric and in range",
	          "RealRandom 1 Make \"r\" Random 10 RealRandom 0 Output IsNumber :r");
	test_fatal_catch_case("RealRandom statement form above range names the offender",
	                      "RealRandom 2 Output 1",
	                      "RealRandom flag must be 0 or 1");
}

/* lg_init stdlib-eval reset: lg_init's one-shot autoexec
   eval (sectorlogo_main.inc:475) is an EVALUATOR ENTRY, so it must run
   SL_RESET_CALL_STATE exactly like on_token / interp / eval_script do. Without it
   a flag left set by a caught FATAL reaches EvalLogoStream, whose jnz .stream_done
   (sectorlogo_eval.inc:124) abandons the stream BEFORE its first form, and :477
   then stores NIL as sl_lg_base_env -- a silently empty base env that makes every
   later script FATAL on its first unbound stdlib name.

   The three steps mirror the real reachability chain: arm a flag that survives a
   FATAL, let v7_free_inference clear sl_lg_initialized (inf_cleanup.inc:286), then
   drive a second interp that must rebuild the Logo world from autoexec. */
void test_lg_init_call_state_reset()
{
	printf("\n--- lg_init stdlib-eval call-state reset ---\n");

	/* 1. arm sl_strict_names, then FATAL before any later form can clear it.
	      With strict on, a dead name in expression position is the FATAL (it
	      names the offender -- Sec.173). The caught FATAL unwinds to C with the
	      flag still set; run() calls v7_free_inference afterwards, which is
	      exactly step 2. */
	test_fatal_catch_case("armed strict flag is left set by a caught FATAL",
	                      "EnableStrictNames 1 Output NumberP",
	                      "unbound name in expression position");

	/* 2. v7_free_inference ran inside run() above: it cleared sl_lg_initialized
	      (inf_cleanup.inc:286), so the next interp takes .tok_first_init and runs
	      lg_init again -- the window that had no SL_RESET_CALL_STATE.

	/* 3. the rebuilt stdlib must answer a stdlib name. Before the fix the leaked
	      flag made EvalLogoStream abandon autoexec on its first form, so the base
	      env came back NIL and ForegroundCode resolved to nothing -- a bare
	      unbound name, which self-evaluates instead of erroring (Sec.146 #38), so
	      it would be REPORTED as something other than the number below.
	      ForegroundCode 3 = 33 (palette index < 8 adds 30) -- a value only the
	      autoexec Define can produce. */
	test_num("lg_init rebuilds the stdlib after a caught FATAL left a flag armed",
	         "Output ForegroundCode 3", 33);
	// a second, INDEPENDENT stdlib entry point: its Define body calls Prepend,
	// NearestMatch, First and OUTPUTs the result, and it reads the autoexec
	// global `pal` -- so it passes only if the whole base env came back, not just
	// one Define binding. 255 0 0 is palette row 12.
	test_num("second stdlib Define body runs after the leaked-flag re-init",
	         "Output NearestColor 255 0 0", 12);
}

/* getchar pipe-path fix: PeekNamedPipe arg5/arg6 must be
   staged by API_CALL inside its own frame (they were previously hand-written to the
   wrong RSP and PeekNamedPipe wrote its output through garbage pointers). Redirect
   stdin to a pipe holding the bytes "AB", then ReadKey must intern and return the
   string atom "AB" (before the fix the pipe peek failed and ReadKey returned NIL). */
void test_getchar_pipe()
{
	printf("suite 75: getchar pipe path\n");
	fflush(stdout);

	HANDLE pipe_read = 0;
	HANDLE pipe_write = 0;
	if (!CreatePipe(&pipe_read, &pipe_write, 0, 0))
	{
		printf("  FAIL: ReadKey pipe setup: CreatePipe failed\n");
		failing_tests_counter++;
		return;
	}

	DWORD bytes_written = 0;
	WriteFile(pipe_write, "AB", 2, &bytes_written, 0);
	CloseHandle(pipe_write);

	HANDLE saved_stdin = GetStdHandle(STD_INPUT_HANDLE);
	SetStdHandle(STD_INPUT_HANDLE, pipe_read);
	test_num("ReadKey reads piped bytes back", "Output IfElse ((ReadKey) = \"AB\") [1] [0]", 1);
	SetStdHandle(STD_INPUT_HANDLE, saved_stdin);
	CloseHandle(pipe_read);
}

/* parser recursion guard on the private Logo stack
   (SL_MAX_PARSE_DEPTH = 262144 is unreachable through the 64KB script buffer --
   max literal nesting ~65535, so the depth FATAL can no longer fire from interp
   input; the guard stays as defense): 3000 nested '[]' brackets FATAL'd at the
   old 2048 cap -- on the private stack they must PARSE and run. The runaway
   coverage moves to the eval call-depth cap (SL_MAX_CALL_DEPTH = 65536):
   recursion past it must die as a caught FATAL, not an AV. Scripts are built
   at runtime (too long to print in the test header). */
void test_parser_depth_guard()
{
	printf("suite 76: parser depth on the private stack\n");
	fflush(stdout);

	char deep_script[6030];
	int offset = 0;

	offset += sprintf(deep_script + offset, "Make \"L\" ");
	for (int i = 0; i < 3000; i++)
	{
		deep_script[offset++] = '[';
	}
	for (int i = 0; i < 3000; i++)
	{
		deep_script[offset++] = ']';
	}
	offset += sprintf(deep_script + offset, " Output 1");
	deep_script[offset] = 0;

	test_num("3000-deep bracket nesting parses (FATAL at the old 2048 cap)", deep_script, 1);

	test_fatal_catch_case("countdown 70000 dies at the 65536 call-depth cap (caught, not an AV)",
		(char *)"Define \"countdown\" [:n] [ IfElse ( :n > 0 ) [ Output countdown :n - 1 ] [ Output :n ] ] Output countdown 70000",
		(char *)"recursion depth exceeded");
}

/* sl_lindex partial-numeric FATAL fix (lisp BUG-1):
   atoms whose FIRST byte passes the old gate but whose text does not fully
   parse ("5x", "-foo", "-.5") must return UNCHANGED -- not kill the script via
   sl_sort_coerce_num's FATAL. Fully-numeric atoms still coerce to NUMBER.
   NOTE: the live builtin name is Item (str_sl_lindex) -- LINDEX is the doc-name
   and self-evaluates as a dead atom (Sec.146 fact 38). */
void test_lindex_partial_numeric()
{
	printf("suite 77: lindex partial-numeric atoms\n");
	fflush(stdout);

	test_num("Item [\"5x\" hello] 0 returns the atom unchanged", "Output IfElse ((Item [ \"5x\" hello ] 0) = \"5x\") [1] [0]", 1);
	test_num("Item [\"-foo\"] 0 returns the atom unchanged", "Output IfElse ((Item [ \"-foo\" ] 0) = \"-foo\") [1] [0]", 1);
	test_num("Item [\"-.5\" x] 0 returns the atom unchanged", "Output IfElse ((Item [ \"-.5\" x ] 0) = \"-.5\") [1] [0]", 1);
	test_num("Item [3 4] 1 still coerces to NUMBER 4", "Output Item [ 3 4 ] 1", 4);
	test_num("Item [\"+-5\" x] 0 returns the atom unchanged (double sign)", "Output IfElse ((Item [ \"+-5\" x ] 0) = \"+-5\") [1] [0]", 1);
	test_num("Item [\"+-\" x] 0 returns the atom unchanged", "Output IfElse ((Item [ \"+-\" x ] 0) = \"+-\") [1] [0]", 1);
	test_num("Item [\"+5\" x] 0 still coerces to NUMBER 5", "Output Item [ \"+5\" x ] 0", 5);
}

/* private Logo stack (sectorlogo_seperate_stack.md):
   deep recursion in every dimension that the old 1 MiB host stack capped at
   2048 must now run to completion on the 64 MiB block, the stack switch must
   survive a caught FATAL and a same-thread nested entry, and the watermark
   sites must never turn legal input into an AV. Scripts are built at runtime
   (too long to print in the test header). */
void test_private_stack()
{
	printf("suite 78: private Logo stack\n");
	fflush(stdout);

	/* eval recursion: 10000 user-fn call levels (FATAL at the old 2048 cap).
	   The fn must OUTPUT its value -- a fn body without Output yields NIL
	   (Sec.146 fact 30), so the base case Output is load-bearing here. */
	test_num("deep user-fn recursion 10000 levels runs",
		(char *)"Define \"countdown\" [:n] [ IfElse ( :n > 0 ) [ Output countdown :n - 1 ] [ Output :n ] ] Output countdown 10000", 0);

	/* parser recursion per ELEMENT: a ~30000-element flat list literal (60000
	   chars -- near the 64KB script buffer limit) parses, Length returns it */
	char flat_script[61000];
	int flat_offset = sprintf(flat_script, "Make \"L\" [");
	for (int i = 0; i < 30000; i++)
	{
		flat_script[flat_offset++] = '1';
		flat_script[flat_offset++] = ' ';
	}
	flat_offset += sprintf(flat_script + flat_offset, "] Output Length :L");
	flat_script[flat_offset] = 0;
	test_num("flat list literal 30000 elements parses (one parser frame each)", flat_script, 30000);

	/* expression-nesting recursion through the EvalPrimary funnel: 30000
	   nested parens parse AND evaluate (both capped at 2048 before). COMPACT
	   parens (no spaces): 7 + 30000 + 1 + 30000 + NUL = 60009 bytes -- fits the
	   64KB script buffer AND the C staging buffer with margin. */
	char paren_script[61000];
	int paren_offset = sprintf(paren_script, "Output ");
	for (int i = 0; i < 30000; i++)
	{
		paren_script[paren_offset++] = '(';
	}
	paren_script[paren_offset++] = '7';
	for (int i = 0; i < 30000; i++)
	{
		paren_script[paren_offset++] = ')';
	}
	paren_script[paren_offset] = 0;
	test_num("30000-deep paren nesting parses and evaluates", paren_script, 7);

	/* runtime-built trees: Prepend inside REPEAT nests 30000 deep x2, then =
	   descends both via CAR recursion -- exercises the Equal/GC_Mark watermark
	   shapes with no parser involvement at all */
	test_num("programmatic 30000-deep Prepend nesting + deep Equal",
		(char *)"Make \"A\" [] Repeat 30000 [ Make \"A\" Prepend :A [] ] Make \"B\" [] Repeat 30000 [ Make \"B\" Prepend :B [] ] Output IfElse (:A = :B) [1] [0]", 1);

	/* NOTE: the plan's same-thread nested-entry test (CallLibrary ->
	   v7s_sectorlogo_interp from mid-eval) is BLOCKED by a pre-existing FFI
	   bug (bug_log.md 2026-09-24: FFI calls into ANY SectorLogo entry FATAL
	   "expected NUMBER" -- A/B-verified present before the private stack).
	   Add the nested-entry assertion when that bug is fixed. The stale-owner
	   reuse test below still exercises a fresh switch after an abandoned one. */

	/* death-abandoned switch reuse: the countdown-70000 FATAL in suite 76 died
	   mid-recursion on the private stack; v7_free_inference revived the engine
	   and THIS script must run on a fresh top (stale owner == my TID reuse) */
	test_num("next script after a deep caught FATAL reuses the stack cleanly",
		(char *)"Output 41", 41);
}

/* The script source cap is SL_SCRIPT_BUF_SIZE (1 MiB, sectorlogo_const.inc).
   These two cases pin BOTH sides of it:
     - a script well past the OLD 64 KB cap must still run, proving the raise
       took effect (under the old constant this died a caught FATAL),
     - a script past 1 MiB must die a CAUGHT FATAL that names the buffer --
       never a silent clamp, because a truncated script is a corrupted script
       (Sec.165).
   The body is comment filler on purpose: the copy gate is what is under test,
   not the parser. A ';' comment costs one GetChar per byte and allocates no
   cons cell, so both cases stay fast and do not depend on the 1,000,000-cell
   arena. test_num/test_fatal_catch_case cannot be used here because they echo
   the whole script, which would print megabytes of filler. */
void test_script_cap()
{
	printf("\n--- Script size cap (1 MiB) ---\n");

	int filler_len = 300000;                 /* ~4.6x the old 64 KB cap */
	char *big_script = (char *)malloc(filler_len + 32);
	if (!big_script)
	{
		failing_tests_counter++;
		printf("  FAIL: script cap past-side (malloc %d bytes failed)\n", filler_len + 32);
		return;
	}
	memset(big_script, ';', filler_len);
	big_script[filler_len] = '\n';
	strcpy(big_script + filler_len + 1, "Output 4242");
	int big_len = (int)strlen(big_script);
	int big_result = run(big_script);
	/* is_num/num_val: the interpreter returns the tagged value, and a top-level
	   OUTPUT arrives as SL_OUTPUT_TAG|4242 (Sec.146 #27) -- comparing the raw
	   int against 4242 would fail on a correct run. */
	if (is_num(big_result) && num_val(big_result) == 4242 && !last_run_engine_died)
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		failing_tests_counter++;
		printf("  FAIL: %d-byte script -> 0x%08X dead=%d reason '%s', expected NUMBER 4242 alive\n",
			big_len, (unsigned)big_result, last_run_engine_died, last_death_reason);
	}
	free(big_script);

	int oversize_len = 1100000;              /* past 1 MiB = 1048576 */
	char *oversize_script = (char *)malloc(oversize_len + 2); /* len + newline + NUL */
	if (!oversize_script)
	{
		failing_tests_counter++;
		printf("  FAIL: script cap oversize-side (malloc %d bytes failed)\n", oversize_len + 2);
		return;
	}
	memset(oversize_script, ';', oversize_len);
	oversize_script[oversize_len] = '\n';
	oversize_script[oversize_len + 1] = 0;
	int oversize_result = run(oversize_script);
	if ((oversize_result == 0) && (last_run_engine_died) &&
		(strstr(last_death_reason, "exceeds the 1 MiB script buffer") != 0))
	{
		passing_tests_counter++;
		printf(".");
		fflush(stdout);
	}
	else
	{
		failing_tests_counter++;
		printf("  FAIL: %d-byte script -> 0x%08X dead=%d reason '%s', expected a caught oversize FATAL\n",
			oversize_len, (unsigned)oversize_result, last_run_engine_died, last_death_reason);
	}
	free(oversize_script);
}

/* B1 pinning (sectorlogo_found_bugs_to_analyze_and_fix.md): a STRING atom whose
   text spells a number is STABLE on every path the Scheme compiler uses. The
   old "SetCdr loses \"5\"" claim does not reproduce -- Intern keys on the type
   byte and every store is verbatim. All eight probe paths from the report
   (E2a-E2d, P2-P5, P10, E3), asserted via the A2 type predicates. */
void test_string_atom_stability()
{
	printf("\n--- STRING atom stability pinning (B1) ---\n");
	/* E2a: the literal itself */
	test_true("string literal \"5\" is a STRING atom", "Output IsStringAtom \"5\"");
	test_nil("string literal \"5\" is not a NUMBER", "Output IsNumber \"5\"");
	/* P2/E2b: cons storage round-trip (Prepend) */
	test_true("Prepend keeps \"5\" a STRING atom", "Output IsStringAtom First Prepend \"5\" []");
	/* E2c: same, compared against the literal */
	test_true("Prepend \"5\" equals the literal", "Output ((First Prepend \"5\" []) = \"5\")");
	/* E2d: SetCdr with a literal */
	test_true("SetCdr keeps literal \"5\" a STRING", "Make \"pr\" Prepend 1 [] SetCdr :pr \"5\" Output IsStringAtom ButFirst :pr");
	/* P3: SetCdr with a variable */
	test_true("SetCdr keeps variable-held \"5\" a STRING", "Make \"sv\" \"5\" Make \"p3\" Prepend 1 [] SetCdr :p3 :sv Output IsStringAtom ButFirst :p3");
	/* P4/P5: Run re-read -- the block element is re-parsed as source. Variable
	   names are alphanumeric only: the engine rejects '_' in a name. */
	test_true("Run-bound \"5\" stays a STRING", "Run [ Make \"stabout\" \"5\" ] Output IsStringAtom :stabout");
	/* P10: nested twice through First */
	test_true("double-nested \"5\" stays a STRING", "Output IsStringAtom First First Prepend Prepend \"5\" [] []");
	/* E3: a STRING atom element re-read as SOURCE in expression position --
	   the compiled-code shape (built via AsWord/Prepend, evaluated by Run) */
	test_true("source-re-read \"5\" lands as STRING in a value slot",
		"Make \"src\" Prepend AsWord \"Make\" Prepend AsWord \"stabout2\" Prepend AsWord \"IsStringAtom\" Prepend \"5\" [] Run :src Output :stabout2");
	/* the bare word 5 is and stays a NUMBER-side atom: the two never dedup */
	test_nil("bare 5 is not a STRING atom", "Output IsStringAtom 5");
	test_true("bare 5 stays a NUMBER", "Output IsNumber 5");
}

/* test_unknown_word_warning -- the runtime typo-trap warn must be pure
 * OBSERVATION: the unknown word still self-evaluates in a value slot (the
 * warn printed before the value is returned) and an unknown statement head
 * still skips (the warn printed, the statement dropped, the script ran on).
 * The warn itself prints to stdout -- NOT asserted here, since a once-per-
 * -word-per-process cache makes output matching order-dependent; what the
 * validator pins is that the SEMANTICS are unchanged for all callers and
 * both position classes, and that the process state stays clean. */
void test_unknown_word_warning()
{
	/* value slot: an unknown bare word self-evaluates as a WORD atom (the
	   evaluator never fatals on it -- `ybbbbbbb > 3`-style use stays legal) */
	test_true("unknown word in a value slot is still a WORD atom",
		"Output IsWordAtom ybbbbbbb");
	test_nil("unknown word in a value slot is not a STRING atom",
		"Output IsStringAtom ybbbbbbb");

	/* statement position: an unknown statement head is skipped and the script
	   continues (the warn fires, `ybbbbbbb 1 2` drops, Output 9 still runs) */
	test_num("unknown statement head is skipped, the script continues",
		"ybbbbbbb 1 2\nOutput 9", 9);

	/* TAGGED statement heads (the ardmark crash, fixed Sep 2026): a bare list
	   literal, a bare number, a bare float and a bare empty list are all
	   legitimate statement heads that no dispatch claims. The warn helper used
	   to read their type byte at [r13 + rax] with the tag bits still set --
	   r13 + 0x800000E8 dereferenced off the atom area and the process died
	   with 0xC0000005. They now take the silent-skip edge like any other
	   non-warnable head and the script continues. */
	test_num("bare list statement head is skipped silently", "[1 2 3]\nOutput 9", 9);
	test_num("bare number statement head is skipped silently", "42\nOutput 9", 9);
	test_num("bare float statement head is skipped silently", "1.5\nOutput 9", 9);
	test_num("bare NIL statement head is skipped silently", "[]\nOutput 9", 9);

	/* quoted data in a value slot: the STRING literal form never fatals and
	   evaluates to itself (the warn's WORD-type filter sees no WORD atom here) */
	test_true("quoted unknown text stays STRING literal data",
		"Output IsStringAtom \"ybbbbbbb\"");

	/* define-bound names must NOT be captured by the warn path: a bound
	   non-fn value keeps returning the ATOM itself (Sec.146 fact 29, the
	   legitimate self-evaluation edges stay silent) */
	test_true("a bound non-fn value keeps returning the atom itself",
		"Make \"ybbbbbbb2\" \"hello\" Output IsWordAtom ybbbbbbb2");

	/* dead name under strict mode still FATALs (the warn sits on the NON-strict
	   edge only -- strict behavior untouched) */
	test_fatal_catch_case("strict mode dead name still fatals, message names the atom",
		"EnableStrictNames 1 Make \"z\" ybbbbbbb3",
		"unbound name in expression position");

	/* position-class pairing stress (the class parks on the stack across the
	   print ops -- an unbalanced push/pop pair would corrupt the frame and
	   crash on return): fire the warning in alternating positions and inside
	   a loop, then require the script to finish numerically correct */
	test_num("alternating expression/statement warnings keep the stack paired",
		"ybbbbbbb4\nMake \"r\" ybbbbbbb5\nybbbbbbb6\nOutput (2 + 3)", 5);
	test_num("repeated warnings inside a loop stay paired",
		"Make \"n\" 0\nRepeat [ Make \"r\" ybbbbbbb7 Make \"n\" :n + 1 If :n = 100 [ Stop ] ]\nOutput :n", 100);
}

/* ==========================================================================
   Suite 99: cons-GC and atom-GC gate variants (integrated from the retired
   t_sectorlogo_gc tool -- one validator per module). Every gate is a
   self-contained script with a fixed expected NUMBER.

   Cons gates: the allocation shapes force a cons collect DURING evaluation
   (live env growth, shadowed bindings, in-place locals + dead bindings,
   fib recursion then a loop) and the run must stay numerically correct.
   NUMBER payloads are 30-bit two's-complement, so the triangular sums are
   kept below 536870911.

   Atom gates: the 12000-iteration growing-string churn crosses the atom
   area's 48 MiB soft limit and forces atom collects mid-run. A duplicated
   atom (missed scan), a reclaimed live atom, a broken LOCAL name binding,
   a corrupted sort record or a lost FFI memo entry shows up as a wrong
   answer through script-visible behaviour only.
   ========================================================================== */
char gc_gate_script[8192];

const char *gc_gate_fib_script =
	"Define \"fib\" [:n] [ IFELSE (:n < 2) [ OUTPUT :n ] [ OUTPUT (fib (:n - 1)) + (fib (:n - 2)) ] ] OUTPUT fib 24";

void test_gc_gate_variants(void)
{
	/* cons gates: collect during live-env growth / shadowed bindings / locals */
	sprintf(gc_gate_script,
		"MAKE \"S\" 0 MAKE \"I\" 1 REPEAT 3000 [ MAKE \"S\" :S + :I MAKE \"I\" :I + 1 ] OUTPUT :S");
	test_num("gc gate env-grow 3000 (collect during live-env growth)", gc_gate_script, 4501500);
	sprintf(gc_gate_script,
		"MAKE \"G\" 0 REPEAT 200000 [ MAKE \"G\" :G + 1 ] OUTPUT :G");
	test_num("gc gate env-garbage 200000 (shadowed bindings, live env O(1))", gc_gate_script, 200000);
	sprintf(gc_gate_script,
		"LOCAL \"S\" LOCAL \"I\" MAKE \"S\" 0 MAKE \"I\" 1 REPEAT 3000 [ MAKE \"S\" :S + :I MAKE \"I\" :I + 1 MAKE \"G\" :I + 1 MAKE \"G\" :I + 2 ] OUTPUT :S");
	test_num("gc gate env-local 3000 (in-place locals plus dead bindings)", gc_gate_script, 4501500);

	/* bench chronology: fib GCs first, then a loop that must not see them;
	   the extra fib runs reproduce the fibN-seq gate's repetition */
	test_num("gc gate fib(24) run 1 = 46368", (char *)gc_gate_fib_script, 46368);
	test_num("gc gate fib(24) run 2 after prior GCs", (char *)gc_gate_fib_script, 46368);
	sprintf(gc_gate_script,
		"MAKE \"S\" 0 MAKE \"I\" 1 REPEAT 20000 [ MAKE \"S\" :S + :I MAKE \"I\" :I + 1 ] OUTPUT :S");
	test_num("gc gate loop-add 20000 after two fib runs", gc_gate_script, 200010000);
	test_num("gc gate fib(24) run 3", (char *)gc_gate_fib_script, 46368);
	test_num("gc gate loop-add 20000 after three fib runs", gc_gate_script, 200010000);

	/* atom gates: collect mid-run, survival visible through behaviour */
	test_num("gc gate atom-churn (12000 growing strings, final length)",
		"Define \"churn\" [:n] [ Make \"S\" \"\" Repeat :n [ Make \"S\" :S _ \"x\" ] Output TextLength :S ] Output churn 12000",
		12000);
	test_num("gc gate atom-large (dead atom adjacent to live, coalescing)",
		"Make \"S\" \"\" Repeat 4096 [ Make \"S\" :S _ \"0123456789abcdef0123456789abcdef\" ] Repeat 400 [ Make \"S\" Part :S 0 (TextLength :S) - 1 ] Output TextLength :S",
		131072 - 400);
	test_num("gc gate atom-ident (equal runtime strings dedup to one atom)",
		"Make \"A\" Part \"alphabeta\" 0 5 Make \"B\" Part \"alphabeta--zz\" 0 5 IfElse SameValue :A :B [ Output 1 ] [ Output 0 ]",
		1);
	test_num("gc gate atom-empty (empty-string interning stays consistent)",
		"Make \"E\" \"\" Make \"F\" \"\" _ \"\" IfElse SameValue :E :F [ Output 1 ] [ Output 0 ]",
		1);
	test_num("gc gate atom-thrash (live keys survive reclaim, text intact)",
		"Make \"KEYS\" [] Make \"I\" 0 Repeat 8 [ Make \"K\" \"livemarker\" _ TextLength :I _ \"\" Make \"KEYS\" Prepend :K :KEYS Make \"I\" :I + 1 ] Make \"CH\" \"\" Repeat 12000 [ Make \"CH\" :CH _ \"x\" ] Output TextLength Item :KEYS 7",
		11);
	test_num("gc gate atom-local (LOCAL name atom survives a collect)",
		"Define \"lt\" [:n] [ Local \"q1\" Make \"q1\" :n Make \"CH\" \"\" Repeat 12000 [ Make \"CH\" :CH _ \"x\" ] Output :q1 ] Output lt 7",
		7);
	test_num("gc gate atom-sort (sort records stay whole across a collect)",
		"Make \"L\" [] Make \"I\" 0 Repeat 8 [ Make \"K\" \"srt\" _ TextLength :I _ \"\" Make \"L\" Prepend Prepend HashNumber :K Prepend :K [] :L Make \"I\" :I + 1 ] Make \"CH\" \"\" Repeat 12000 [ Make \"CH\" :CH _ \"x\" ] IfElse ((Length (Sort :L)) = 8) [ Output 1 ] [ Output 0 ]",
		1);
	test_num("gc gate atom-exe-args (runtime-built spawn argument survives)",
		"DefineCommand \"vx\" [:m] " SLFFI_ECHO_ARGS_ARG " \"%s\" Make \"CH\" \"\" Repeat 12000 [ Make \"CH\" :CH _ \"x\" ] Make \"CAP\" vx \"hel\" _ \"lo\" IfElse IsNumber Search \"argc=2\" :CAP [ Output 1 ] [ Output 0 ]",
		1);
	test_num("gc gate atom-ffi-memo (stale module memo entry is wiped)",
		"Make \"M1\" \"" SLFFI_GC_M1_STEM "\" _ \"" SLFFI_TEST_EXT "\" Make \"D1\" Prepend :M1 Prepend \"slffi_add\" Prepend \"integer\" Prepend [\"integer\" \"integer\"] [] Make \"R1\" CallLibrary :D1 20 22 Make \"D1\" [] Make \"CH\" \"\" Repeat 12000 [ Make \"CH\" :CH _ \"x\" ] Make \"M2\" \"" SLFFI_GC_M2_STEM "\" _ \"" SLFFI_GC_M2_EXT "\" Make \"D2\" Prepend :M2 Prepend \"v7s_get_num_workers\" Prepend \"integer\" Prepend [] [] Make \"R2\" CallLibrary :D2 IfElse (:R1 > 0) [ IfElse IsNumber :R2 [ Output 1 ] [ Output 0 ] ] [ Output 0 ]",
		1);
	test_num("gc gate atom-nocells (collect with a near-empty cons heap)",
		"Define \"solo\" [:n] [ Make \"S\" \"\" Repeat :n [ Make \"S\" :S _ \"y\" ] Output TextLength :S ] Output solo 12000",
		12000);
}

/* ==========================================================================
   Suite 100: SetOutputToken whole-state switch (integrated from the retired
   v_settoken_validate tool -- one validator per module). A sidecar ladder on
   the end token proves the engine POV: SetOutputToken switches the WHOLE
   state (no hook refire; the engine decodes the replacement exactly as if
   it had been sampled; the script's own :Piece and :TokenNumber resolve to
   the switched state from the statement onward; the LAST switch wins).
   The ladder runs on turn 1's first end token (Paris, then Berlin, flag
   latches), passively resets on turn 1's second end token so the turn ends,
   and re-arms for turn 2. Thinking OFF: the ladder works on the plain
   end-of-turn token, not on a think-close inside reasoning.
   ========================================================================== */
void test_settoken_whole_state(void)
{
	/* Earlier FATAL-catch suites revive via v7_free_inference, which wipes the
	 * whole hash table; only the autoexec key is re-seeded, so the model keys
	 * this suite's engine init needs are gone. Reload the table from disk. */
	inference_dll_ptr->v7_free_inference();
	if (!inference_dll_ptr->v7_init(validator_db_path))
	{
		failing_tests_counter++;
		printf("  FAIL: settoken suite could not reload the database: %s\n",
			inference_dll_ptr->v7_engine_is_dead() ? inference_dll_ptr->v7_get_death_reason() : "init failed");
		return;
	}
	seed_autoexec_db();
	if (!inference_dll_ptr->v7_initialize_inference())
	{
		failing_tests_counter++;
		printf("  FAIL: settoken suite needs a live engine: %s\n",
			inference_dll_ptr->v7_engine_is_dead() ? inference_dll_ptr->v7_get_death_reason() : "initialize failed");
		return;
	}
	inference_dll_ptr->v7_enable_thinking(0);

	const char *sidecar_script =
		"IfElse (:TokenNumber = EndToken)\n"
		"[\n"
		"  IfElse ((DatabaseGet \"test_paris_flag\") = \"1\")\n"
		"  [\n"
		"    DatabaseSet \"test_paris_flag\" \"0\"\n"
		"  ]\n"
		"  [\n"
		"    SetOutputToken (EncodeWord \"Paris\")\n"
		"    Print \"tok1\"\n"
		"    Print :TokenNumber\n"
		"    Print \"piece1\"\n"
		"    Print :Piece\n"
		"    SetOutputToken (EncodeWord \"Berlin\")\n"
		"    Print \"tok2\"\n"
		"    Print :TokenNumber\n"
		"    Print \"piece2\"\n"
		"    Print :Piece\n"
		"    DatabaseSet \"test_paris_flag\" \"1\"\n"
		"  ]\n"
		"]\n"
		"[]";
	if (!inference_dll_ptr->ht_add("logo:sidecar", (void *)sidecar_script, (unsigned __int64)(strlen(sidecar_script) + 1), 0))
	{
		failing_tests_counter++;
		printf("  FAIL: ht_add('logo:sidecar') rejected the settoken ladder\n");
		return;
	}

	char turn_output[65536];

	printf("  turn 1: SetOutputToken replaces the end token (Paris -> Berlin ladder)\n");
	memset(turn_output, 0, sizeof(turn_output));
	inference_dll_ptr->v7_chat("What is 1+1?", 0, turn_output, sizeof(turn_output), 1);
	if (inference_dll_ptr->v7_engine_is_dead())
	{
		failing_tests_counter++;
		printf("  FAIL: settoken turn 1 died: %s\n", inference_dll_ptr->v7_get_death_reason());
		inference_dll_ptr->v7_free_inference(); /* revive for the rest of the suite */
		seed_autoexec_db();
		return;
	}
	/* the model's own answer must survive: a swallowed or empty answer fails
	   the test, not just a missing replacement piece */
	test_contains("settoken turn 1 keeps the model's own answer in the stream", turn_output, "1 + 1 = 2");
	/* the engine decoded the FINAL replacement id, so Berlin (the last
	   switch) must be in the visible stream */
	test_contains("settoken turn 1 stream carries the final replacement piece (Berlin)", turn_output, "Berlin");

	printf("  turn 2: flag reset -- the ladder fires fresh on the next end token\n");
	memset(turn_output, 0, sizeof(turn_output));
	inference_dll_ptr->v7_chat("What is 2+2?", 0, turn_output, sizeof(turn_output), 0);
	if (inference_dll_ptr->v7_engine_is_dead())
	{
		failing_tests_counter++;
		printf("  FAIL: settoken turn 2 died: %s\n", inference_dll_ptr->v7_get_death_reason());
		inference_dll_ptr->v7_free_inference();
		seed_autoexec_db();
		return;
	}
	/* an unreset flag would have skipped the ladder; a never-set one would
	   emit no Berlin -- both asserts together prove the reset-and-rearm */
	test_contains("settoken turn 2 keeps its own answer (session intact)", turn_output, "4");
	test_contains("settoken turn 2 ladder re-armed (Berlin again)", turn_output, "Berlin");

	/* disarm the hook, then tear the live engine down so the validator's final
	 * unload is clean; the table wipe is repaired with the usual re-seed */
	inference_dll_ptr->ht_add("logo:sidecar", (void *)"", (unsigned __int64)1, 0);
	inference_dll_ptr->v7_shutdown();
	inference_dll_ptr->v7_free_inference();
	seed_autoexec_db();
}

int main(int argc, char **argv)
{
	ZydisInit();
	if (argc < 4)
	{
		printf("Usage: %s <dll_path> --db <db_path> [-startfrom N] [-with-slow] [-only-slow]\n", argv[0]);
		return 1;
	}
	if (strcmp(argv[2], "--db") != 0)
	{
		printf("Usage: %s <dll_path> --db <db_path> [-startfrom N] [-with-slow] [-only-slow]\n", argv[0]);
		return 1;
	}

	for (int i = 4; i < argc; i++)
	{
		if (strcmp(argv[i], "-startfrom") == 0 && i + 1 < argc)
		{
			start_from = atoi(argv[i + 1]);
			i++;
		}
		else if (strcmp(argv[i], "-with-slow") == 0)
		{
			include_slow_suites = 1;
		}
		else if (strcmp(argv[i], "-only-slow") == 0)
		{
			only_slow_suites = 1;
		}
	}

	/* x87 control word at CRT baseline, BEFORE any v7 code runs: the rope
	   builder's bare finit used to leak 0x037F (extended PC) into every later
	   x87 op on this thread. The save/restore fix must keep the word at this
	   captured value across the entire validator workload. */
	unsigned int control_word_start = _control87(0, 0);

	v7_lib lib;
	inference_dll_ptr = &lib;
	validator_db_path = argv[3];

	suite_timing_log = fopen("build_logs/v_sectorlogo_validate_timing.log", "w");
	if (!suite_timing_log)
		printf("warning: cannot open build_logs/v_sectorlogo_validate_timing.log -- suite timings print to stdout only\n");
	else
		fprintf(suite_timing_log, "v_sectorlogo_validate suite timings (ms)\n");

	LARGE_INTEGER wall_start, wall_end, wall_freq;
	QueryPerformanceCounter(&wall_start);

	if (v7_load(argv[1], &lib) != 0)
	{
		printf("Cannot load %s\n", argv[1]);
		return 1;
	}

	LARGE_INTEGER db_load_start, db_load_end, db_load_freq;
	QueryPerformanceCounter(&db_load_start);
	if (!lib.v7_init(argv[3]))
	{
		if (lib.v7_engine_is_dead())
			printf("v7_init('%s') died: %s\n", argv[3], lib.v7_get_death_reason());
		else
			printf("v7_init('%s') failed\n", argv[3]);
		v7_unload(&lib);
		return 1;
	}
	QueryPerformanceCounter(&db_load_end);
	QueryPerformanceFrequency(&db_load_freq);
	double db_load_ms = (double)(db_load_end.QuadPart - db_load_start.QuadPart) * 1000.0 / (double)db_load_freq.QuadPart;
	printf("v7_init (DB load): %.1f ms\n", db_load_ms);
	if (suite_timing_log)
	{
		fprintf(suite_timing_log, "v7_init (DB load)  %10.1f ms\n", db_load_ms);
		fflush(suite_timing_log);
	}

	/* Seed the autoexec stdlib into the database ONCE: lg_init re-runs after
	   every caught FATAL (v7_free_inference clears the init latch), and a
	   file-backed stdlib re-reads autoexec.logo from disk on every one of
	   those reloads. With the DB key present, lg_init takes the
	   __sectorlogo__autoexec path and the reload is a hash-table lookup.
	   v7_free_inference wipes the whole table (ht_destroy + ht_init), so
	   run() re-seeds after every caught FATAL from the cached text. */
	{
		FILE *autoexec_file = fopen("autoexec.logo", "rb");
		if (!autoexec_file)
		{
			printf("autoexec.logo not found in the working directory\n");
			v7_unload(&lib);
			return 1;
		}
		fseek(autoexec_file, 0, SEEK_END);
		autoexec_db_size = ftell(autoexec_file);
		fseek(autoexec_file, 0, SEEK_SET);
		autoexec_db_text = (char *)malloc((size_t)autoexec_db_size + 1);
		if (!autoexec_db_text)
		{
			printf("out of memory reading autoexec.logo\n");
			fclose(autoexec_file);
			v7_unload(&lib);
			return 1;
		}
		if (fread(autoexec_db_text, 1, (size_t)autoexec_db_size, autoexec_file) != (size_t)autoexec_db_size)
		{
			printf("short read on autoexec.logo\n");
			free(autoexec_db_text);
			fclose(autoexec_file);
			v7_unload(&lib);
			return 1;
		}
		autoexec_db_text[autoexec_db_size] = 0;
		fclose(autoexec_file);
		seed_autoexec_db();
		/* lg_init always queries the "__sectorlogo__autoexec" DB key first
		   (the gate check was removed -- the engine DB is the only database),
		   so the seeded tier-1 stdlib is picked up without any host call */
		printf("autoexec stdlib seeded into the database (%ld bytes)\n", autoexec_db_size);
	}

	printf("... SectorLogo64 Comprehensive Validator ...\n");

	/* autoexec.logo test MUST run first: lg_init loads the repo-root stdlib
	   on the very first v7s_sectorlogo_interp call (plan Sec.11 step 1) */
	RUN_SUITE(1,  test_autoexec());

	RUN_SLOW_SUITE(2,  test_console());

	RUN_SUITE(3,  test_integer_literals());
	RUN_SLOW_SUITE(4,  test_arith_basics());
	RUN_SUITE(5,  test_infix_arithmetic());
	RUN_SUITE(6,  test_math_precedence());
	RUN_SUITE(7,  test_divmod_edge());
	RUN_SUITE(8,  test_infix_bitwise());
	RUN_SUITE(9,  test_infix_bitwise());
	RUN_SLOW_SUITE(10, test_infix_relational());
	RUN_SLOW_SUITE(11, test_infix_relational());
	RUN_SUITE(12, test_boolean());
	RUN_SUITE(13, test_list_ops());
	RUN_SUITE(14, test_make_and_vars());
	RUN_SUITE(15, test_if_ifelse());
	RUN_SUITE(16, test_repeat());
	RUN_SUITE(17, test_stop());
	RUN_SUITE(18, test_continue());
	RUN_SUITE(19, test_terminate());
	RUN_SUITE(20, test_stop_continue_propagation());
	RUN_SUITE(21, test_repeat_endless());
	RUN_SUITE(22, test_string_concat());
	RUN_SLOW_SUITE(23, test_string_operations());
	RUN_SUITE(24, test_long_strings());
	RUN_SUITE(25, test_str_to_num());
	RUN_SUITE(26, test_explode());
	RUN_SLOW_SUITE(27, test_str_tokens());
	RUN_SUITE(28, test_split_ngrams());
	RUN_SLOW_SUITE(29, test_lindex());
	RUN_SLOW_SUITE(30, test_ffi());
	RUN_SUITE(31, test_parens());
	RUN_SUITE(32, test_quoted_atoms());
	RUN_SUITE(33, test_equalp_atoms());
	RUN_SUITE(34, test_equalp_num_string_coercion());
	RUN_SUITE(35, test_v7_db_roundtrip());
	RUN_SUITE(36, test_strings());
	RUN_SUITE(37, test_def_fn());
	RUN_SUITE(38, test_local());
	RUN_SUITE(39, test_sort_reverse());
	RUN_SUITE(40, test_hash_float());
	RUN_SUITE(41, test_length());
	RUN_SUITE(42, test_lset());
	RUN_SUITE(43, test_lfind_family());
	RUN_SLOW_SUITE(44, test_run_eval());
	RUN_SUITE(45, test_nested_control());
	RUN_SUITE(46, test_complex());
	RUN_SUITE(47, test_negative_numbers());
	RUN_SUITE(48, test_gc_stress());
	RUN_SUITE(49, test_complex_programs());

	/* Float test suites */
	RUN_SUITE(50, test_float_literals());
	RUN_SUITE(51, test_float_prefix_arithmetic());
	RUN_SUITE(52, test_float_infix_arithmetic());
	RUN_SUITE(53, test_float_promotion());
	RUN_SUITE(54, test_float_comparisons());
	RUN_SUITE(55, test_float_truthiness());
	RUN_SUITE(56, test_float_type_predicates());
	RUN_SLOW_SUITE(57, test_float_conversions());
	RUN_SUITE(58, test_float_transcendentals());
	RUN_SUITE(59, test_float_transcendentals_ext());
	RUN_SUITE(60, test_float_math_chained());
	RUN_SUITE(61, test_float_conversions_ext());
	RUN_SUITE(62, test_float_formatter());
	RUN_SUITE(63, test_float_complex());
	RUN_SLOW_SUITE(64, test_float_edge_cases());

	/* New tokenizer tests (Part E) */
	RUN_SUITE(65, test_tokenizer_delimiter_splitting());
	RUN_SUITE(66, test_tokenizer_case_folding());
	RUN_SUITE(67, test_tokenizer_unary_minus_gluing());
	RUN_SUITE(68, test_name_validation());

	/* DefineCommand + caught-FATAL tests (Sep 2026) */
	RUN_SLOW_SUITE(69, test_def_exe_fn());
	RUN_SLOW_SUITE(70, test_fatal_catch());

	/* Batch B regression suite (opt_and_bug): TERMINATE flag, LOCAL fn guard,
	   param case fold, nested DO_VQ, AVX2 case-fold windows */
	RUN_SLOW_SUITE(71, test_batch_b_regressions());

	/* Batch D regression suite (opt_and_bug revised): Equal r9 clobber,
	   DO_VQ NaN-best, radix binary64 keys, fraction parser, reader operand
	   classification, x87 control-word leak (checked in main) */
	RUN_SUITE(72, test_batch_d_regressions());
	RUN_SUITE(73, test_batch_d_regressions_ifelse_value());

	/* EnableTrace runtime evaluator tracing */
	RUN_SLOW_SUITE(74, test_enabletrace());
 	RUN_SUITE(75, test_getchar_pipe());

 	RUN_SLOW_SUITE(76, test_parser_depth_guard());
  	RUN_SUITE(77, test_lindex_partial_numeric());
  	RUN_SUITE(78, test_private_stack());
	RUN_SLOW_SUITE(79, test_setcar_setcdr());
	RUN_SUITE(80, test_intern_boundary_lengths());
	RUN_SUITE(81, test_samevalue());
	RUN_SLOW_SUITE(82, test_run_string_dispatch());
	RUN_SLOW_SUITE(83, test_asword());
	RUN_SUITE(84, test_script_cap());

	/* Reader boundary table agrees with the class table: '\' '{' '}' DEL are
	   word chars, while '^' '|' '&' '_' '[' ']' stay operator boundaries */
	RUN_SUITE(85, test_tokenizer_word_char_boundaries());
	RUN_SUITE(86, test_gc_stmt_fn_retval_root());

	/* Dotted-list rendering (lg_value_to_str " . " tail) and atom-sweep
	   survival (sl_atom_gc granule-units fix) -- Sec.170 coverage */
	RUN_SUITE(87, test_dotted_list_rendering());
	RUN_SLOW_SUITE(88, test_atom_gc_survival());
	RUN_SUITE(89, test_literal_precision());
	/* .es_streq key dispatch: the routed key must run its OWN script, removed
	   tuner keys must route to not-found -- runs BEFORE the GC-reinit suite
	   because eval_script caches its parsed program per process: the marker
	   eval needs the cache slot still empty (the GC suite populates it) */
	RUN_SUITE(90, test_eval_script_key_dispatch());
	RUN_SUITE(90, test_gc_root_reinit());

	/* STRING atom stability pinning (B1): the numeric-text instability premise
	   does not reproduce -- all eight probe paths asserted via A2 predicates */
	RUN_SUITE(91, test_string_atom_stability());

	/* EnableStrictNames: the opt-in gate for the dead-name self-evaluation trap
	   (A1) -- strict off/on, both primitive forms, the range FATAL, and the
	   per-call reset that keeps a caught FATAL from arming the next script */
	RUN_SUITE(92, test_enablestrictnames());

	/* lg_init's stdlib eval is a fourth evaluator entry and must reset the
	   per-call state too: without it a caught FATAL's leftover flag silently
	   truncates sl_lg_base_env to NIL on the next re-init.
	   Numbered after 93 so -startfrom N keeps registration order == execution
	   order; runs after the pct hook because it re-inits the Logo world itself. */
	RUN_SUITE(94, test_lg_init_call_state_reset());

	/* byte-fallback decode rejection: arms a sidecar and drives on_token
	   with the decode chain faked. Runs before the pct-hook suite below, which
	   re-arms for itself and must stay last. */
	RUN_SUITE(95, test_decode_byte_fallback_reject());

	/* V7_PCT_STMT live-hook percentage conversion: arms a sidecar,
	   re-arms lg_init, and drives on_token directly -- leaves the sidecar armed,
	   so it runs before the re-init suite above rather than last */
	RUN_SUITE(93, test_v7prim_pct_hook());

	/* Global Define binding: a Define inside a user fn registers in
	   the shared base env (the sentinel insert) -- the wrapper-installer
	   pattern; runs anywhere (no sidecar arming, no re-init) */
	RUN_SUITE(96, test_define_global_bind());

	/* RAG API stdlib: DefinePureLogoBM25 pipeline end-to-end plus
	   the RagSearch/RagRemember exe wrappers -- spawns bm25_rag.exe */
	RUN_SUITE(97, test_rag_api());

	/* Unknown-word warning: the runtime typo-trap warn must never
	   change the semantics it observes -- unknown words still self-evaluate
	   in value slots and still skip as statements */
	RUN_SUITE(98, test_unknown_word_warning());
	RUN_SUITE(99, test_gc_gate_variants());
	RUN_SUITE(100, test_settoken_whole_state());


	/* x87 control-word leak check: after the full workload (rope build inside
	   inference-create + every SectorLogo float suite + the live-engine
	   settoken turn), the x87 HALF of the combined word must equal the CRT
	   baseline captured at main start. A bare finit leak would have flipped
	   PC to extended (and changed rounding for later float ops). The SSE-side
	   high half is owned by the engine: every dispatch entry loads the
	   captured FTZ/DAZ/PM image (SET_FTZ_DAZ, the determinism requirement that
	   keeps a greedy run bit-identical across processes) and Win64 does not
	   preserve MXCSR across calls, so the engine-owned bits are masked out of
	   the comparison. */
	unsigned int control_word_end = _control87(0, 0);
	unsigned int x87_mask = 0x0000FFFFu; /* low half: x87 CW incl. the PC bits the finit leak class flips */
	if ((control_word_end & x87_mask) != (control_word_start & x87_mask))
	{
		printf("FAIL: x87 control word changed across the workload (start 0x%X, end 0x%X)\n", control_word_start, control_word_end);
		failing_tests_counter++;
	}
	else
	{
		printf("pass: x87 control word unchanged across the workload\n");
		passing_tests_counter++;
	}

	QueryPerformanceCounter(&wall_end);
	QueryPerformanceFrequency(&wall_freq);
	double wall_ms = (double)(wall_end.QuadPart - wall_start.QuadPart) * 1000.0 / (double)wall_freq.QuadPart;
	printf("total wall time: %.1f ms\n", wall_ms);
	if (suite_timing_log)
	{
		fprintf(suite_timing_log, "total            %10.1f ms\n", wall_ms);
		fclose(suite_timing_log);
		suite_timing_log = 0;
#ifdef _WIN32
		printf("suite timings written to build_logs\\v_sectorlogo_validate_timing.log\n");
#else
		printf("suite timings written to build_logs/v_sectorlogo_validate_timing.log\n");
#endif
	}

	v7_unload(&lib);
	inference_dll_ptr = 0;

	printf("\n... SectorLogo64 Validator: %d passed, %d failed ...\n", passing_tests_counter, failing_tests_counter);
	if (failing_tests_counter)
		return 1;
	else
		return 0;
}
