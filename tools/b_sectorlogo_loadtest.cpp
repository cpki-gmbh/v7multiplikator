// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * b_sectorlogo_loadtest.cpp -- SectorLogo interpreter load benchmark vs C reference
 *
 * Benchmarks the v7s_sectorlogo_interp tree-walking evaluator against the
 * SAME algorithms hand-written in C (stdlib only), and reports the slowdown
 * ratio. Reference ratios for dispatch-heavy workloads on comparable
 * hardware: Lua 5.x ~15-40x vs C, CPython ~60-150x vs C (classical
 * recursive fib benchmarks; programming-language-benchmarks data).
 *
 * Math tests:   recursive fib(n) via Define (dispatch-heavy -- the classic
 *               interpreter-overhead probe) and an integer loop-add
 *               (loop + variable-update, no function calls).
 * String tests: iterative string building via _ concat (atom interning +
 *               heap append per step) and a replace-all loop (Search +
 *               Replace per occurrence) -- the canonical sidecar idiom.
 * Atom tests:  an atom-churn workload that appends NOVEL atoms every rep, so
 *               a long run actually crosses the soft limit and the atom
 *               collector runs. It is ASM-only: the C++ port's atom area is
 *               append-only with no collector, so it cannot run this shape.
 *
 * C references: recursive fib + loop-add (plain int ops), string building
 *               via realloc growable buffer (stdlib only), replace-all via
 *               strstr + memcpy (stdlib only).
 *
 * Methodology: warmup + correctness gate per workload, then N repetitions
 * INTERLEAVED (one Logo rep, then one C rep, alternating) so thermal drift
 * cancels (AGENTS.md Sec.54: 10-25% run-to-run variation on this laptop);
 * the reported number is the MEDIAN of the reps. Results are consumed via
 * the expected-value check each rep so the optimizer cannot eliminate work.
 *
 * Every workload is verified against the known answer before timing
 * (a wrong result invalidates the benchmark).
 *
 * THREE implementations are timed per workload, not two: the ASM tree-walker
 * behind the DLL, the readable C++ port (sectorlogo_c\), and the hand-written C
 * reference. The C++ port is here because it is the SectorLogo semantics in a
 * form a human reads, and the interesting number is the RATIO between the two
 * ports -- that is what says how much of the ASM's speed is the algorithm and
 * how much is the hand-assembly.
 *
 * The two interpreters provably run the SAME script bytes: every logo_*_work()
 * sprintf's the workload into script_buf and then runs script_buf, so whatever
 * is in script_buf after an ASM call is exactly what the ASM just ran. The C++
 * port is handed that same buffer, so there is no second copy of any workload
 * string to drift out of step. The C++ result is gated against the expected
 * value exactly like the other two, so a port that computes something else
 * fails the benchmark rather than posting a fast wrong number.
 *
 * Usage: b_sectorlogo_loadtest.exe <dll_path> --db <db_path> [--reps N] [--quick]
 *   --reps N   repetitions per workload (default 5, max 64)
 *   --quick    fib(16) instead of fib(24), 2k instead of 20k loop iterations
 */

#include "v7_lib.h"
#include "zydis_debug_header.h"
#include "kt_utils.h"
#include "../sectorlogo_c/sectorlogo_rewrite.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define SL_CONS_TAG     0x80000000
#define SL_NUMBER_TAG   0x40000000
#define SL_FLOAT_TAG    0xC0000000
#define SL_TAG_MASK     0xC0000000
#define SL_PAYLOAD_MASK 0x3FFFFFFF

#define DEFAULT_REPS 5
#define MAX_REPS     64

v7_lib *inference_dll_ptr = 0;

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

int fib_n = 24;
int loop_n = 20000;
int strbuild_n = 2000;
int replace_n = 60;

int reps = DEFAULT_REPS;

int passing_checks = 0;
int failing_checks = 0;

int run_logo(const char *script)
{
	return inference_dll_ptr->v7s_sectorlogo_interp(script);
}

/* ------------------------------------------------------------------ */
/* C references (stdlib only) -- same algorithms                       */
/* ------------------------------------------------------------------ */

int c_fib(int n)
{
	if (n < 2)
		return n;
	return c_fib(n - 1) + c_fib(n - 2);
}

long long c_loop_add(int iterations)
{
	long long sum = 0;
	int counter = 1;

	for (int i = 0; i < iterations; i++)
	{
		sum += counter;
		counter += 1;
	}
	return sum;
}

/* Growable-buffer append, 1 byte per step -- mirrors the Logo concat
 * workload's allocation behavior (heap append per iteration). */
char *c_strbuild(int iterations)
{
	int cap = 64;
	int len = 1;
	char *buf = (char *)malloc((size_t)cap);

	if (!buf)
		return 0;
	buf[0] = 'x';
	for (int i = 0; i < iterations; i++)
	{
		if (len + 2 > cap)
		{
			cap *= 2;
			char *grown = (char *)realloc(buf, (size_t)cap);
			if (!grown)
			{
				free(buf);
				return 0;
			}
			buf = grown;
		}
		buf[len] = 'y';
		buf[len + 1] = 0;
		len += 1;
	}
	return buf;
}

/* Replace-all via strstr + memcpy: :N occurrences of "cat " -> "dog "
 * inside a 4*N-char haystack. Same result as the Logo version. */
char *c_replace_all(int occurrences)
{
	size_t total = (size_t)occurrences * 4;
	char *hay = (char *)malloc(total + 1);
	char *writer;
	char *cursor;
	int count;

	if (!hay)
		return 0;
	writer = hay;
	for (int i = 0; i < occurrences; i++)
	{
		memcpy(writer, "cat ", 4);
		writer += 4;
	}
	*writer = 0;
	cursor = hay;
	count = 0;
	while ((cursor = (char *)strstr(cursor, "cat ")) != 0)
	{
		memcpy(cursor, "dog ", 4);
		cursor += 4;
		count++;
	}
	if (count != occurrences)
	{
		free(hay);
		return 0;
	}
	return hay;
}

/* ------------------------------------------------------------------ */
/* Timing helpers: interleaved reps, median                            */
/* ------------------------------------------------------------------ */

int cmp_double(const void *left, const void *right)
{
	double first = *(const double *)left;
	double second = *(const double *)right;

	if (first < second) return -1;
	if (first > second) return 1;
	return 0;
}

double median_of(double *samples, int count)
{
	qsort(samples, (size_t)count, sizeof(double), cmp_double);
	if (count % 2 == 1)
		return samples[count / 2];
	return (samples[count / 2 - 1] + samples[count / 2]) / 2.0;
}

/* Three columns: the two interpreters and the C reference. The two ratios are
 * both worth having and they say different things -- asm/c is how far the
 * hand-written interpreter is from plain C, and cpp/c is the same question for
 * the readable port, so the pair localises how much of the gap is the algorithm
 * and how much is the assembly.
 *
 * cpp_ms < 0 means the C++ port does not run this workload (bench_asm_only),
 * so its column and ratio print as n/a rather than as a zero that would read
 * as "the port was infinitely fast". */
void print_row(const char *label, double asm_ms, double cpp_ms, double c_ms)
{
	if (cpp_ms < 0.0)
	{
		printf("  %-26s %10.3f ms      %-10s %8.3f ms   %6.1fx   %6s\n",
		       label, asm_ms, "n/a", c_ms, asm_ms / c_ms, "n/a");
		return;
	}
	printf("  %-26s %10.3f ms  %10.3f ms  %8.3f ms   %6.1fx %6.1fx\n",
	       label, asm_ms, cpp_ms, c_ms, asm_ms / c_ms, cpp_ms / c_ms);
}

/* ------------------------------------------------------------------ */
/* Work functions: Logo side runs a fresh script each call (interpreter */
/* state does not persist between interp calls); C side runs the same  */
/* algorithm. Each returns its checkable result as double.              */
/* ------------------------------------------------------------------ */

char script_buf[8192];

/* The readable C++ port's interpreter, created once and reused: its Machine
 * constructor allocates and clears a 72 MiB heap, and doing that per rep would
 * put the allocation in the measurement instead of the workload. */
sectorlogo::Machine *cpp_port_machine = 0;

/* Run whatever is currently in script_buf through the C++ port.
 *
 * This is the whole fairness argument in one function. Every logo_*_work()
 * sprintf's its workload into script_buf and then calls run_logo(script_buf), so
 * on return script_buf holds the exact bytes the ASM just ran. Re-running that
 * same buffer through the second interpreter means there is no second copy of
 * any workload string, and therefore no way for the two sides to drift apart
 * through a later edit to one of them.
 *
 * num_val is the SAME converter the ASM side uses, so the three columns are
 * compared as like with like rather than through two different decoders. */
double run_cpp_port()
{
	return (double)num_val(cpp_port_machine->interp(script_buf));
}

/* Recursive fib via Define -- validated pattern (validator "Define
 * factorial": OUTPUT Multiply :n fact Subtract :n 1). fib(24) = 46368:
 * ~150k recursive calls, enough work to time reliably. */
double logo_fib_work()
{
	sprintf(script_buf,
	        "Define \"fib\" [:n] [ "
	        "  IFELSE (:n < 2) [ OUTPUT :n ] [ "
	        "    OUTPUT (fib (:n - 1)) + (fib (:n - 2)) "
	        "  ] "
	        "] "
	        "OUTPUT fib %d",
	        fib_n);
	return (double)num_val(run_logo(script_buf));
}

double c_fib_work()
{
	return (double)c_fib(fib_n);
}

/* Integer loop-add: REPEAT + variable read/modify. Same shape as the
 * validator's "sum 1..100" test but at benchmark scale. */
/* Integer loop-add: REPEAT + variable read/modify. Idiomatic Logo: the
 * loop variables are LOCAL (update-in-place MAKE -- no env alist growth,
 * the same O(1) per-iteration variable cost Python locals / JS let have). */
double logo_loop_add_work()
{
	sprintf(script_buf,
	        "Define \"loopadd\" [] [ "
	        "  LOCAL \"S\" "
	        "  LOCAL \"I\" "
	        "  MAKE \"S\" 0 "
	        "  MAKE \"I\" 1 "
	        "  REPEAT %d [ "
	        "    MAKE \"S\" :S + :I "
	        "    MAKE \"I\" :I + 1 "
	        "  ] "
	        "  OUTPUT :S "
	        "] "
	        "OUTPUT loopadd",
	        loop_n);
	return (double)num_val(run_logo(script_buf));
}

double c_loop_add_work()
{
	return (double)c_loop_add(loop_n);
}

/* String building: one _ concat per iteration (validator's long-string
 * pattern). Idiomatic Logo: accumulator LOCAL (update-in-place MAKE, no
 * env growth). The interpreter itself counts the 'y' chars via a Part
 * walk (validator pattern: Part :s start 1 == "" past the end), so the
 * returned value is a checkable number. */
double logo_strbuild_work()
{
	sprintf(script_buf,
	        "Define \"strbuild\" [] [ "
	        "  LOCAL \"S\" "
	        "  LOCAL \"C\" "
	        "  LOCAL \"K\" "
	        "  LOCAL \"W\" "
	        "  MAKE \"S\" \"x\" "
	        "  REPEAT %d [ MAKE \"S\" :S _ \"y\" ] "
	        "  MAKE \"C\" 0 "
	        "  MAKE \"K\" 0 "
	        "  REPEAT 99999 [ "
	        "    MAKE \"W\" Part :S :K 1 "
	        "    IFELSE (:W = \"\") [ STOP ] [ "
	        "      IFELSE (:W = \"y\") [ MAKE \"C\" :C + 1 ] [] "
	        "      MAKE \"K\" :K + 1 "
	        "    ] "
	        "  ] "
	        "  OUTPUT :C "
	        "] "
	        "OUTPUT strbuild",
	        strbuild_n);
	return (double)num_val(run_logo(script_buf));
}

double c_strbuild_work()
{
	static char *strbuild_buf = 0;

	if (strbuild_buf)
		free(strbuild_buf);
	strbuild_buf = c_strbuild(strbuild_n);
	if (!strbuild_buf)
		return -1.0;
	return (double)strlen(strbuild_buf) - 1.0; /* 'y' count = len - 1 */
}

/* Concat-only variant of the string-build workload: identical build loop,
 * then a single TextLength instead of the 99,999-step Part walk. Timing this
 * and the full workload separately yields the walk share by subtraction
 * (walk = full - concat-only; noise on the difference is ~2x a single
 * median's noise, AGENTS.md Sec.54). */
double logo_strbuild_concat_only_work()
{
	sprintf(script_buf,
	        "Define \"strbuild\" [] [ "
	        "  LOCAL \"S\" "
	        "  MAKE \"S\" \"x\" "
	        "  REPEAT %d [ MAKE \"S\" :S _ \"y\" ] "
	        "  OUTPUT TextLength :S "
	        "] "
	        "OUTPUT strbuild",
	        strbuild_n);
	return (double)num_val(run_logo(script_buf));
}

double c_strbuild_concat_only_work()
{
	char *buf = c_strbuild(strbuild_n);

	if (!buf)
		return -1.0;
	free(buf);
	return (double)strbuild_n + 1.0; /* strlen = n + leading 'x' */
}

/* Atom-churn workload: the collect-path gate.
 *
 * The two append-path workloads above (string build, concat only) build
 * "x", "xy", "xyy", ... -- the SAME 2000 strings on every rep. Rep 1 appends
 * them; reps 2..N re-intern all 2000 and append NOTHING. That is the correct
 * shape for measuring steady-state append cost (which is why those two remain
 * the append gates), but it means a long rep count never reaches the soft
 * limit and the collector never runs.
 *
 * The fix is a per-rep seed that makes every intermediate string a NOVEL
 * atom. `((Random 90000) + 10000)` is a five-digit value, so the seed text is
 * always 5 bytes wide and TextLength stays an exact expected value
 * (CHURN_SEED_BYTES + churn_n) while the atom TEXT differs on every rep. Each
 * rep therefore appends the full sum of lengths (~2.0 MB at churn_n = 2000),
 * and the soft limit is crossed every ~24 reps.
 *
 * ASM-ONLY: the C++ port's atom area is append-only with no collector
 * (sectorlogo_rewrite.cpp:454 FATALs on overflow), so a workload that churns
 * 2.0 MB per rep would abort the port on its second rep. The C reference is
 * still timed -- it is the same realloc-growable-buffer append -- so the row
 * keeps its asm/c column.
 */
#define CHURN_SEED_BYTES 5

double logo_atom_churn_work()
{
	sprintf(script_buf,
	        "Define \"churn\" [] [ "
	        "  LOCAL \"S\" "
	        "  MAKE \"S\" ((Random 90000) + 10000) "
	        "  REPEAT %d [ MAKE \"S\" :S _ \"y\" ] "
	        "  OUTPUT TextLength :S "
	        "] "
	        "OUTPUT churn",
	        strbuild_n);
	return (double)num_val(run_logo(script_buf));
}

double c_atom_churn_work()
{
	char *buf = c_strbuild(strbuild_n);

	if (!buf)
		return -1.0;
	free(buf);
	return (double)(CHURN_SEED_BYTES + strbuild_n);
}

/* Repeat-intern workload: re-intern the SAME built string 2000 times.
 * This is the production V7-STREAM shape (a sidecar reads the accumulated
 * stream once per token) -- bloom hit, slot hit, then a full length verify
 * every iteration. Gates the Intern hash + verify-path optimizations. */
double logo_repeat_intern_work()
{
	sprintf(script_buf,
	        "Define \"repintern\" [] [ "
	        "  LOCAL \"S\" "
	        "  LOCAL \"t\" "
	        "  LOCAL \"C\" "
	        "  MAKE \"S\" \"x\" "
	        "  REPEAT %d [ MAKE \"S\" :S _ \"y\" ] "
	        "  MAKE \"C\" 0 "
	        "  REPEAT 2000 [ "
	        "    MAKE \"t\" :S _ \"\" "
	        "    IFELSE (:t = :S) [ MAKE \"C\" :C + 1 ] [] "
	        "  ] "
	        "  OUTPUT :C "
	        "] "
	        "OUTPUT repintern",
	        strbuild_n);
	return (double)num_val(run_logo(script_buf));
}

double c_repeat_intern_work()
{
	char *buf = c_strbuild(strbuild_n);
	int count = 0;

	if (!buf)
		return -1.0;
	/* C reference: strcmp the buffer against itself 2000 times */
	for (int i = 0; i < 2000; i++)
	{
		if (strcmp(buf, buf) == 0)
			count++;
	}
	free(buf);
	return (double)count;
}

/* Replace-all loop: the canonical sidecar idiom (AGENTS.md Sec.139) --
 * REPEAT + Search + Replace, first-occurrence replace driven to
 * completion. The haystack has :N marked words; the interpreter counts
 * the resulting 'd' chars via the same Part walk. */
/* Replace-all loop (Search + Replace per occurrence) -- the canonical
 * sidecar idiom (sidecar_cats_to_dogs.logo). Idiomatic Logo: all loop
 * variables LOCAL (update-in-place MAKE, no env alist growth). */
double logo_replace_work()
{
	sprintf(script_buf,
	        "Define \"replall\" [] [ "
	        "  LOCAL \"T\" "
	        "  LOCAL \"F\" "
	        "  LOCAL \"C\" "
	        "  LOCAL \"K\" "
	        "  LOCAL \"W\" "
	        "  MAKE \"T\" \"\" "
	        "  REPEAT %d [ MAKE \"T\" :T _ \"cat \" ] "
	        "  REPEAT 99999 [ "
	        "    MAKE \"F\" Search \"cat \" :T "
	        "    IFELSE IsNumber :F [ "
	        "      MAKE \"T\" Replace :T \"cat \" \"dog \" "
	        "    ] [ STOP ] "
	        "  ] "
	        "  MAKE \"C\" 0 "
	        "  MAKE \"K\" 0 "
	        "  REPEAT 99999 [ "
	        "    MAKE \"W\" Part :T :K 1 "
	        "    IFELSE (:W = \"\") [ STOP ] [ "
	        "      IFELSE (:W = \"d\") [ MAKE \"C\" :C + 1 ] [] "
	        "      MAKE \"K\" :K + 1 "
	        "    ] "
	        "  ] "
	        "  OUTPUT :C "
	        "] "
	        "OUTPUT replall",
	        replace_n);
	return (double)num_val(run_logo(script_buf));
}

double c_replace_work()
{
	static char *replace_buf = 0;

	if (replace_buf)
		free(replace_buf);
	replace_buf = c_replace_all(replace_n);
	if (!replace_buf)
		return -1.0;
	return (double)replace_n; /* c_replace_all verified count == occurrences */
}

/* Find-scan only: absent-needle pure scan -- REPEAT 9999 [ Search "zzz" :T ]
 * over the same cat-word haystack. Every Search walks the full haystack to
 * the NUL without a single first-char candidate hit -- exactly the
 * O(haystack) scan term Phase 5 replaces (replace-all minus this row
 * isolates the build share). Counts NIL outcomes, expected 9999. */
double logo_find_scan_work()
{
	sprintf(script_buf,
	        "Define \"findscan\" [] [ "
	        "  LOCAL \"T\" "
	        "  LOCAL \"F\" "
	        "  LOCAL \"C\" "
	        "  MAKE \"T\" \"\" "
	        "  REPEAT %d [ MAKE \"T\" :T _ \"cat \" ] "
	        "  MAKE \"C\" 0 "
	        "  REPEAT 9999 [ "
	        "    MAKE \"F\" Search \"zzz\" :T "
	        "    IFELSE IsNumber :F [] [ MAKE \"C\" :C + 1 ] "
	        "  ] "
	        "  OUTPUT :C "
	        "] "
	        "OUTPUT findscan",
	        replace_n);
	return (double)num_val(run_logo(script_buf));
}

double c_find_scan_work()
{
	static char *find_scan_buf = 0;

	if (find_scan_buf)
		free(find_scan_buf);
	/* identical haystack to the Logo side: replace_n words of "cat " */
	find_scan_buf = (char *)malloc((size_t)replace_n * 4 + 1);
	if (!find_scan_buf)
		return -1.0;
	char *writer = find_scan_buf;
	for (int i = 0; i < replace_n; i++)
	{
		memcpy(writer, "cat ", 4);
		writer += 4;
	}
	*writer = 0;
	int misses = 0;
	for (int rep = 0; rep < 9999; rep++)
	{
		if (strstr(find_scan_buf, "zzz") == 0)
			misses++;
	}
	return (double)misses;
}

/* ------------------------------------------------------------------ */
/* bench_pair: warmup + correctness gate, then interleaved reps        */
/*                                                                     */
/* The name is left alone deliberately: it is the name of the SHAPE     */
/* (one Logo workload against one C reference), and the C++ port was    */
/* added as a third leg of that same shape rather than as a new one.    */
/* ------------------------------------------------------------------ */

void bench_pair(const char *name, double (*logo_fn)(), double (*c_fn)(), double expected)
{
	double asm_samples[MAX_REPS];
	double cpp_samples[MAX_REPS];
	double c_samples[MAX_REPS];
	double asm_result = logo_fn(); /* warmup + correctness gate */
	/* script_buf now holds the exact script the ASM just ran, so the C++
	 * port is gated on the same bytes rather than on a copy. */
	double cpp_result = run_cpp_port();
	double c_result = c_fn();

	if (asm_result != expected)
	{
		printf("  %-26s FAIL: asm result %.0f != %.0f -- benchmark invalid\n", name, asm_result, expected);
		failing_checks++;
		return;
	}
	if (cpp_result != expected)
	{
		printf("  %-26s FAIL: c++ port result %.0f != %.0f -- benchmark invalid\n", name, cpp_result, expected);
		failing_checks++;
		return;
	}
	if (c_result != expected)
	{
		printf("  %-26s FAIL: c result %.0f != %.0f -- benchmark invalid\n", name, c_result, expected);
		failing_checks++;
		return;
	}
	passing_checks++;

	for (int rep = 0; rep < reps; rep++)
	{
		double stamp0 = kt_get_seconds();
		double ar = logo_fn();
		double stamp1 = kt_get_seconds();
		double cr1 = run_cpp_port();
		double stamp2 = kt_get_seconds();
		double cr = c_fn();
		double stamp3 = kt_get_seconds();

		asm_samples[rep] = stamp1 - stamp0;
		cpp_samples[rep] = stamp2 - stamp1;
		c_samples[rep] = stamp3 - stamp2;
		/* consume results: any drift invalidates the run */
		if (ar != expected)
		{
			printf("  %-26s FAIL: asm result %.0f != %.0f drifted at rep %d\n", name, ar, expected, rep);
			failing_checks++;
			return;
		}
		if (cr1 != expected)
		{
			printf("  %-26s FAIL: c++ port result %.0f != %.0f drifted at rep %d\n", name, cr1, expected, rep);
			failing_checks++;
			return;
		}
		if (cr != expected)
		{
			printf("  %-26s FAIL: c result %.0f != %.0f drifted at rep %d\n", name, cr, expected, rep);
			failing_checks++;
			return;
		}
		/* Per-rep asm time: a median over a long run is dominated by the
		 * collector-free reps, so a collect that fires every eighth rep is
		 * invisible in the median. The per-rep line is where that shows up.
		 * The C and C++ legs stay median-only: neither touches the
		 * interpreter's atom area, so their spread is pure noise. */
		printf("  %-26s rep %2d  asm %8.3f ms\n", name, rep, asm_samples[rep] * 1000.0);
	}
	print_row(name,
	          median_of(asm_samples, reps) * 1000.0,
	          median_of(cpp_samples, reps) * 1000.0,
	          median_of(c_samples, reps) * 1000.0);
}

/* bench_asm_only: the same gate and the same interleaved reps as bench_pair,
 * minus the C++ port. A workload whose atom churn outruns the port's
 * append-only atom area has no third leg -- the port would FATAL on its
 * second rep and take the whole run with it. The per-rep print is what makes
 * the collect visible: it fires every ~24 reps, and a median over 64 reps
 * would bury it under the 40+ collector-free reps. */
void bench_asm_only(const char *name, double (*logo_fn)(), double (*c_fn)(), double expected)
{
	double asm_samples[MAX_REPS];
	double c_samples[MAX_REPS];
	double asm_result = logo_fn(); /* warmup + correctness gate */
	double c_result = c_fn();

	if (asm_result != expected)
	{
		printf("  %-26s FAIL: asm result %.0f != %.0f -- benchmark invalid\n", name, asm_result, expected);
		failing_checks++;
		return;
	}
	if (c_result != expected)
	{
		printf("  %-26s FAIL: c result %.0f != %.0f -- benchmark invalid\n", name, c_result, expected);
		failing_checks++;
		return;
	}
	passing_checks++;

	for (int rep = 0; rep < reps; rep++)
	{
		double stamp0 = kt_get_seconds();
		double ar = logo_fn();
		double stamp1 = kt_get_seconds();
		double cr = c_fn();
		double stamp2 = kt_get_seconds();

		asm_samples[rep] = stamp1 - stamp0;
		c_samples[rep] = stamp2 - stamp1;
		/* consume results: any drift invalidates the run */
		if (ar != expected)
		{
			printf("  %-26s FAIL: asm result %.0f != %.0f drifted at rep %d\n", name, ar, expected, rep);
			failing_checks++;
			return;
		}
		if (cr != expected)
		{
			printf("  %-26s FAIL: c result %.0f != %.0f drifted at rep %d\n", name, cr, expected, rep);
			failing_checks++;
			return;
		}
		/* Per-rep asm time: a median over a long run is dominated by the
		 * collector-free reps, so a collect that fires every 24th rep is
		 * invisible in the median. The per-rep line is where that shows up.
		 * The C leg stays median-only: it never touches the interpreter's
		 * atom area, so its spread is pure noise. */
		printf("  %-26s rep %2d  asm %8.3f ms\n", name, rep, asm_samples[rep] * 1000.0);
	}
	print_row(name,
	          median_of(asm_samples, reps) * 1000.0,
	          -1.0,
	          median_of(c_samples, reps) * 1000.0);
}

int run_fib = 1;
int run_loop = 1;
int run_strbuild = 1;
int run_concat_only = 1;
int run_repeat_intern = 1;
int run_replace = 1;
int run_find_scan = 1;
int run_atom_churn = 1;

int expected_fib()
{
	int table[] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368};

	if ((fib_n >= 0) && (fib_n <= 24))
		return table[fib_n];
	return -1;
}

int main(int argc, char **argv)
{
	v7_lib lib;

	if ((argc < 4) || (strcmp(argv[2], "--db") != 0))
	{
		printf("Usage: %s <dll_path> --db <db_path> [--reps N] [--quick] [--only fib|loop|str|concat|intern|replace|findscan|churn|math|strings|all] [--reverse]\n", argv[0]);
		return 1;
	}
	ZydisInit();

	inference_dll_ptr = &lib;
	if (v7_load(argv[1], &lib) != 0)
	{
		printf("Cannot load %s\n", argv[1]);
		return 1;
	}

	int reverse_order = 0;
	for (int i = 3; i < argc; i++)
	{
		if ((strcmp(argv[i], "--reps") == 0) && ((i + 1) < argc))
			reps = atoi(argv[++i]);
		else if (strcmp(argv[i], "--quick") == 0)
		{
			fib_n = 16;
			loop_n = 2000;
			strbuild_n = 400;
			replace_n = 15;
		}
		else if ((strcmp(argv[i], "--only") == 0) && ((i + 1) < argc))
		{
			const char *which = argv[++i];

		run_fib = 0;
		run_loop = 0;
		run_strbuild = 0;
		run_concat_only = 0;
		run_repeat_intern = 0;
		run_replace = 0;
		run_find_scan = 0;
		run_atom_churn = 0;
		if (strcmp(which, "fib") == 0) run_fib = 1;
		else if (strcmp(which, "loop") == 0) run_loop = 1;
		else if (strcmp(which, "str") == 0) run_strbuild = 1;
		else if (strcmp(which, "concat") == 0) run_concat_only = 1;
		else if (strcmp(which, "intern") == 0) run_repeat_intern = 1;
		else if (strcmp(which, "replace") == 0) run_replace = 1;
		else if (strcmp(which, "findscan") == 0) run_find_scan = 1;
		else if (strcmp(which, "churn") == 0) run_atom_churn = 1;
		else if (strcmp(which, "math") == 0) { run_fib = 1; run_loop = 1; }
		else if (strcmp(which, "strings") == 0) { run_strbuild = 1; run_concat_only = 1; run_repeat_intern = 1; run_replace = 1; run_find_scan = 1; }
		else if (strcmp(which, "all") == 0) { run_fib = 1; run_loop = 1; run_strbuild = 1; run_concat_only = 1; run_repeat_intern = 1; run_replace = 1; run_find_scan = 1; run_atom_churn = 1; }
		}
		else if (strcmp(argv[i], "--reverse") == 0)
			reverse_order = 1;
	}
	if (reps < 1) reps = 1;
	if (reps > MAX_REPS) reps = MAX_REPS;

	lib.v7_init(argv[3]);

	/* The C++ port's Machine, created before the first timed rep: its
	 * constructor allocates and clears a 72 MiB heap and loads the stdlib, and
	 * that one-time cost has no business inside a measurement. It is created
	 * here rather than per workload so the cost is paid once for the whole run,
	 * the same way the DLL is loaded once. */
	cpp_port_machine = new sectorlogo::Machine();

	printf("... SectorLogo interpreter load test (ASM + C++ port vs C reference) ...\n");
	printf("fib(n)=%d  loop-add iters=%d  strbuild steps=%d  replace words=%d  reps=%d (interleaved, median)\n\n",
	       fib_n, loop_n, strbuild_n, replace_n, reps);

	printf("%-30s %12s %12s %10s %7s %7s\n", "workload", "asm ms", "c++ ms", "c ms", "asm/c", "cpp/c");

	if (reverse_order)
	{
		/* reverse order: strings first, then math -- isolates cross-workload
		 * state pollution (the loop-add-after-fib failure) */
		printf("  strings\n");
		if (run_strbuild)
			bench_pair("string build (concat)", logo_strbuild_work, c_strbuild_work, (double)strbuild_n);
		if (run_concat_only)
			bench_pair("concat only (no walk)", logo_strbuild_concat_only_work, c_strbuild_concat_only_work, (double)strbuild_n + 1.0);
		if (run_repeat_intern)
			bench_pair("repeat intern (V7-STREAM)", logo_repeat_intern_work, c_repeat_intern_work, 2000.0);
		if (run_replace)
			bench_pair("replace all (Search+Replace)", logo_replace_work, c_replace_work, (double)replace_n);
		if (run_find_scan)
			bench_pair("find scan (absent needle)", logo_find_scan_work, c_find_scan_work, 9999.0);
		if (run_atom_churn)
			bench_asm_only("atom churn (collect path)", logo_atom_churn_work, c_atom_churn_work, (double)(CHURN_SEED_BYTES + strbuild_n));
		printf("  math\n");
		if (run_fib)
			bench_pair("recursive fib", logo_fib_work, c_fib_work, (double)expected_fib());
		if (run_loop)
			bench_pair("loop add", logo_loop_add_work, c_loop_add_work, (double)((long long)loop_n * (loop_n + 1) / 2));
	}
	else
	{
		printf("  math\n");
		if (run_fib)
			bench_pair("recursive fib", logo_fib_work, c_fib_work, (double)expected_fib());
		if (run_loop)
			bench_pair("loop add", logo_loop_add_work, c_loop_add_work, (double)((long long)loop_n * (loop_n + 1) / 2));
		printf("  strings\n");
		if (run_strbuild)
			bench_pair("string build (concat)", logo_strbuild_work, c_strbuild_work, (double)strbuild_n);
		if (run_concat_only)
			bench_pair("concat only (no walk)", logo_strbuild_concat_only_work, c_strbuild_concat_only_work, (double)strbuild_n + 1.0);
		if (run_repeat_intern)
			bench_pair("repeat intern (V7-STREAM)", logo_repeat_intern_work, c_repeat_intern_work, 2000.0);
		if (run_replace)
			bench_pair("replace all (Search+Replace)", logo_replace_work, c_replace_work, (double)replace_n);
		if (run_find_scan)
			bench_pair("find scan (absent needle)", logo_find_scan_work, c_find_scan_work, 9999.0);
		if (run_atom_churn)
			bench_asm_only("atom churn (collect path)", logo_atom_churn_work, c_atom_churn_work, (double)(CHURN_SEED_BYTES + strbuild_n));
	}
	printf("\n");
	if (failing_checks > 0)
	{
		printf("Fail: %d workload result mismatches\n", failing_checks);
		v7_unload(&lib);
		return 1;
	}
	printf("Pass: %d workloads verified, timings above\n", passing_checks);
	v7_unload(&lib);
	return 0;
}
