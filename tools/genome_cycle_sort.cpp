// v7 - part of the v7 inference engine
// Copyright (C) 2026 Gorg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * genome_cycle_sort.cpp -- experimental fixed-list cycle sort over fully
 * random probe genomes, the climb test for the in-process sorting tuner
 * design (see in_process_sorting_tuner_report.md). It answers one
 * question: how much does a bare A/B/A stochastic sort improve a fully
 * random pool in n rounds, for a given set of genetic parameters -- the
 * tool exists to compare --mutate-percentage and --sex-influence settings
 * against each other on identical starting conditions.
 *
 * All randomness is deterministic: the PRNG is seeded once with the fixed
 * seed 42, so every run draws the identical benchmark tensors, the
 * identical reference genome and the identical probe list. Timing noise
 * still varies the measurements themselves, but the searched CONFIG SPACE
 * is the same every run -- that is what makes the parameter comparison
 * fair.
 *
 * The tool holds a STATIC list of 127 genomes (a fixed pool, no grow, no
 * shrink, no archive). The list is filled at startup with fully random
 * probe genomes drawn and inserted UNMEASURED -- no baseline election, no
 * A/B/A during the fill. A reference genome is drawn at startup from the
 * same seed and is NEVER touched afterwards: no sorting, no mutation, no
 * list membership. It is the untouched ruler the final benchmark reads.
 *
 * The sort itself is a rotating probabilistic pair sweep with the bare
 * A/B/A compare ONLY: for slot1 walking 0..126, slot2 is drawn by the
 * --partner formula (default pow2: frand()^2 * slot1, a partner below
 * the rotating position, biased toward the front), then the tool
 * dispatches the genome at the LOWER
 * index (a), the genome at the UPPER index (b -- the one that would be
 * exchanged to move up the ranking when it wins), and the genome at the
 * LOWER index again (a). No warmup rounds, no accumulation, no min-diff,
 * no caps, no deadlines: one timed dispatch per member, and b wins the
 * exchange exactly when b < min(a_first, a_second) -- the faster genome
 * ends at the lower index.
 *
 * Every timed dispatch of a cycle updates the cycle's best-rate record;
 * when a round over the list finishes, the tool prints the GB/s of that
 * recorded fastest runtime. The climb is visible in the per-cycle prints
 * ratcheting upward toward the pool's ceiling.
 *
 * At the end the first 10 list genomes (the ranking front) are
 * benchmarked against the UNCHANGED reference genome -- the reference is
 * dispatched around every genome and its fastest reading is the
 * comparator -- and the tool prints how much better (or slower) the best
 * genome is, in percent. That number per run is the parameter-effect
 * measurement: same seed, same probes, same reference, so differences
 * between runs come from the genetic parameters alone.
 *
 * Options:
 *   --db <path>            REQUIRED database path (no default)
 *   --shape <NxM>          REQUIRED probe draw shape (no archive exists)
 *   --rounds <n>           how many rounds the sort cycles over the fixed
 *                          list (default 10, >= 1)
 *   --partner <mode>       the comparison partner's draw formula below the
 *                          rotating position: powA = slot1 * frand()^A
 *                          (power-law rank selection), tourK = the minimum
 *                          of K uniform draws (deterministic tournament of
 *                          size K), expA = a truncated exponential on
 *                          [0,1) with decay A (exponential rank selection),
 *                          half = the deterministic middle partner
 *                          slot1 >> 1 (constant 0.5, no randomness).
 *                          exp3 is the default, the 3-pass sweep winner;
 *                          pow2 is the historical control. Legal: pow1 pow2
 *                          pow3 pow4 pow6 tour2 tour3 tour4 exp2 exp3 half.
 *                          Only the COMPARISON partner rides this formula;
 *                          the mutation parents stay frand()^2-drawn.
 *   --mutate-percentage <p>  at every sweep end, replace the tail p% of
 *                          the list with mutants of frand()^2-drawn parents
 *                          (default 0 = no mutation; the tail slots are
 *                          overwritten in place -- the displaced genomes
 *                          are gone, the list stays fixed at 127)
 *   --mutations-max <n>    each mutant changes between 1 and n fields
 *                          (uniform draw, default 10)
 *   --sex-influence <pct>  maximum percentage of parameter slots a second
 *                          parent (drawn frand()^2 through the current
 *                          list ranking) donates to a mutant, same slot to
 *                          same slot (default 0 = asexual)
 *   --sorter-stdout-file <path>  when set, every sort print goes to the
 *                          FILE ONLY (flushed per line)
 *   --q8                   Q8 matmul (default)
 *   --f32                  F32 matmul
 */
#include <windows.h>
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <errno.h>     /* strtol/strtod range reporting */
#include <math.h>      /* exp/log for the expA partner draw */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>      /* genome identity fields */
#include <cstdarg>     /* va_start/va_list for the stdout-tee printf */
#include <cstdio>      /* _vsnprintf_s, fputs for the stdout-tee printf */

/* ======================================================================
 * Shape-block and prefetch-table layout (must match v7_data.inc)
 * ====================================================================== */

#define PF_SLOTS 41           /* pf_dist_0..40, pf_type_0..40 */
#define PF_TYPE_MIN 0         /* 0=NTA 1=T0 2=T1 3=T2 4=PREFETCHW 5=PREFETCHWT1 6=CLDEMOTE */
#define PF_TYPE_MAX 6
#define SB_PF_DIST_OFF 104    /* pf_dist[41] inside a 512-byte shape block */
#define SB_PF_TYPE_OFF 300    /* pf_type[41] */
#define SB_SIZE 512           /* bytes per shape block */
#define SB_PARAM_OFF 8        /* param_set dwords start here (after n,d header) */
#define SB_MAX_BLOCKS 256     /* block 0 = active, block 1 = documentation, 2+ = per-shape */
#define DISPATCH_PARAM_COUNT 22

#define CYCLE_LIST_SIZE 127   /* the fixed sort list: 127 genome slots, never resized */
#define RNG_SEED 42           /* the fixed PRNG seed: identical tensors, probes and
                               * reference genome on every run */

/* param_set DWORD index for each dispatch param index (the C-side twin of
 * v7_data.inc's PARAM_OFFSET_* byte constants divided by 4). The layout is
 * not contiguous by dispatch index: chunk_multiplier sits at dword 9 but the
 * caller batch group starts at dword 13; taper_shift (idx 20) and
 * quantize_chunk_groups (idx 21) live in slots 1 and 12. */
static const int sb_param_off_for_idx[DISPATCH_PARAM_COUNT] = {
	0,  2,  3,  4,  5,  6,  7,  8,  9,  10, /* idx 0..9  */
	13, 14, 15, 16, 17, 18, 19, 20, 21, 22, /* idx 10..19 */
	1,  12                                  /* idx 20=taper_shift, 21=quantize_chunk_groups */
};

/* Dispatch param names in dispatch-index order, matching the INI key names
 * of the archive format and the genome struct field order exactly. */
static const char *dispatch_param_names[DISPATCH_PARAM_COUNT] = {
	"num_chunks",          "batch_small",           "batch_medium",           "batch_large",
	"batch_huge",          "batch_med_thresh",      "batch_large_thresh",     "batch_huge_thresh",
	"chunk_multiplier",    "min_chunk_cols",        "caller_batch_small",     "caller_batch_medium",
	"caller_batch_large",  "caller_batch_huge",     "caller_batch_med_thresh", "caller_batch_large_thresh",
	"caller_batch_huge_thresh", "chunk_split_pct",  "chunk_group_1_count",    "chunk_group_2_count",
	"taper_shift",         "quantize_chunk_groups"
};

/* ======================================================================
 * Genome layer (C++)
 * ====================================================================== */

/* All parameters of one genome in the fixed list. The 22 dispatch params
 * ride in dispatch-index order (the same order dispatch_param_names
 * labels), the pf tables carry all 41 slots (the probe draw writes the
 * tuned kernel's owned slots; unowned slots stay 0). The result fields
 * stay 0 until a measurement step fills them. */
struct genome
{
	int shape_n;              /* matmul rows */
	int shape_d;              /* matmul cols */
	int counter;              /* identity: the probe's list slot + 1 */
	std::string path;         /* "(probe)" -- no archive exists in this tool */
	std::string section;      /* the shape token, kept for diagnostics */
	double best_gbs;          /* unused bookkeeping slot, stays 0 */
	int num_chunks;
	int batch_small;
	int batch_medium;
	int batch_large;
	int batch_huge;
	int batch_med_thresh;
	int batch_large_thresh;
	int batch_huge_thresh;
	int chunk_multiplier;
	int min_chunk_cols;
	int caller_batch_small;
	int caller_batch_medium;
	int caller_batch_large;
	int caller_batch_huge;
	int caller_batch_med_thresh;
	int caller_batch_large_thresh;
	int caller_batch_huge_thresh;
	int chunk_split_pct;
	int chunk_group_1_count;
	int chunk_group_2_count;
	int taper_shift;
	int quantize_chunk_groups;
	int pf_dist[PF_SLOTS];
	int pf_type[PF_SLOTS];
	int was_modified;         /* 1 = a mutation produced this genome */
	double shape_a_gbs;       /* last-measured rate, 0 = unmeasured */
	double shape_b_gbs;
	double shape_c_gbs;
};

/* Pointer-to-member for every dispatch param, in dispatch-index order --
 * reading a genome's full parameter vector is then one loop with one array
 * (the struct field order matches the table exactly). */
int genome::*genome_param_ptr[DISPATCH_PARAM_COUNT] = {
	&genome::num_chunks, &genome::batch_small, &genome::batch_medium, &genome::batch_large,
	&genome::batch_huge, &genome::batch_med_thresh, &genome::batch_large_thresh, &genome::batch_huge_thresh,
	&genome::chunk_multiplier, &genome::min_chunk_cols, &genome::caller_batch_small, &genome::caller_batch_medium,
	&genome::caller_batch_large, &genome::caller_batch_huge, &genome::caller_batch_med_thresh, &genome::caller_batch_large_thresh,
	&genome::caller_batch_huge_thresh, &genome::chunk_split_pct, &genome::chunk_group_1_count, &genome::chunk_group_2_count,
	&genome::taper_shift, &genome::quantize_chunk_groups
};

/* The fixed sort list and the reference genome. The list IS the ranking:
 * index 0 is the fastest slot, the front region collects the fast genomes
 * as the exchanges run. The reference genome is the untouched random
 * ruler drawn at startup from the same seed -- it never enters the list,
 * is never sorted or mutated, and is only measured in the final
 * benchmark. */
genome cycle_list[CYCLE_LIST_SIZE];
genome reference_genome;

int sorter_out(const char *fmt, ...); /* forward: defined near main, used by the sort prints */

/* ======================================================================
 * Benchmark tensor descriptors and fill
 * ====================================================================== */

typedef struct
{
	char *input_q;
	float *input_s;
	char *weight_q;
	float *weight_s;
} tensor_q8;

typedef struct
{
	float *input_f;
	float *input_s;
	char *weight_q;
	float *weight_s;
} tensor_f32;

struct v7_lib inference_dll_ptr;

tensor_q8 g_tsq;  /* Q8 benchmark tensors (use_q8_mode path) */
tensor_f32 g_tsf; /* F32 benchmark tensors */

int use_q8_mode;      /* 1 = Q8 dispatch, 0 = F32 dispatch */
const char *kernel_label;
int dispatch_n;       /* matmul rows */
int dispatch_d;       /* matmul cols */

unsigned char *shape_blocks_ptr;  /* base of the exported shape_blocks array */
unsigned char *matched_block_ptr; /* per-shape block for (n,d), block 1 excluded */

int pf_owned[PF_SLOTS];      /* 1 = this slot drives the benchmarked kernel */
int pf_owned_list[PF_SLOTS]; /* owned slot ids */
int pf_owned_count;

/* The config a measurement pushes into the dispatcher: 22 dispatch params
 * + the benchmarked kernel's owned pf values (from the genome struct). */
int loaded_params[DISPATCH_PARAM_COUNT];
int loaded_pf_dist[PF_SLOTS];
int loaded_pf_type[PF_SLOTS];

float *xout_buffer;          /* matmul output row */
double shape_bytes_per_call; /* n*d*5: the established per-call byte count */

int pf_dist_lo;              /* legal distance window: -(weight-matrix bytes / 2) */
int pf_dist_hi;              /* legal distance window: +(weight-matrix bytes / 2) */

int cycle_rounds = 10;       /* --rounds: how many rounds the sort cycles over the list */
double sex_influence_percent = 0.0; /* --sex-influence: the MAXIMUM percentage of parameter
 * slots a second parent donates to a mutant (0 = asexual) */
int mutate_percentage = 0;   /* --mutate-percentage: tail slots replaced by mutants at
 * every sweep end (default 0 = no mutation) */
int mutations_max = 10;      /* --mutations-max: each mutant changes between 1 and
 * mutations_max fields (uniform draw; default 10) */
char partner_mode[16] = "exp3"; /* --partner: the comparison partner's draw formula
 * below the rotating position. powA = slot1 * frand()^A, tourK = the
 * minimum of K uniform draws, expA = a truncated exponential with decay A
 * (exp3 is the default, the 3-pass partner-sweep winner; pow2 is the
 * historical f*f control). Validated against the legal-mode list in
 * parse_cli. */
char cycle_shape[64] = "";   /* --shape NxM: the probe draw shape, REQUIRED (no
 * archive exists to inherit the shape from) */
char sorter_stdout_path[2 * MAX_PATH] = ""; /* --sorter-stdout-file: when set,
 * stdout tees into this file */
FILE *sorter_stdout_capture; /* the tee target, 0 = disabled */

/* ======================================================================
 * Timing (QPC-based)
 * ====================================================================== */

#include "qpc_seconds.h"

/* ======================================================================
 * Tensor alloc/fill (benchmark filler data)
 * All draws come from a xorshift64* PRNG seeded once with the FIXED seed
 * 42, so every run draws the identical tensors, the identical reference
 * genome and the identical probe list -- the parameter-effect comparison
 * depends on that determinism.
 * ====================================================================== */

unsigned __int64 rng_state = RNG_SEED; /* xorshift64* state (never zero) */

unsigned rng_u32(void)
{
	rng_state ^= rng_state >> 12;
	rng_state ^= rng_state << 25;
	rng_state ^= rng_state >> 27;
	return (unsigned)((rng_state * 0x2545F4914F6CDD1DULL) >> 32);
}

float frand(void)
{
	unsigned int bits = 0x3F800000u | (rng_u32() & 0x007FFFFFu);
	float value;
	memcpy(&value, &bits, sizeof(value));
	return value - 1.0f;
}

void init_random_int8(char *data, __int64 count)
{
	for (__int64 i = 0; i < count; i++)
		data[i] = (char)((int)(rng_u32() % 201u) - 100);
}

void init_random_float(float *data, __int64 count)
{
	for (__int64 i = 0; i < count; i++)
		data[i] = frand();
}

void tensor_q8_alloc(tensor_q8 *ts, int matmul_rows, int matmul_cols)
{
	__int64 weight_count = (__int64)matmul_cols * matmul_rows;
	__int64 scale_count = weight_count / 64 + 1;
	ts->input_q = (char *)malloc(matmul_rows);
	ts->input_s = (float *)malloc((matmul_rows / 64 + 1) * sizeof(float));
	ts->weight_q = (char *)malloc(weight_count);
	ts->weight_s = (float *)malloc(scale_count * sizeof(float));
	if (!ts->input_q || !ts->input_s || !ts->weight_q || !ts->weight_s)
	{
		printf("[FATAL] tensor_q8_alloc: out of memory for %dx%d tensor "
		       "(~%lld bytes).\n",
		       matmul_rows, matmul_cols, (long long)weight_count);
		exit(EXIT_FAILURE);
	}
}

void tensor_q8_fill(tensor_q8 *ts, int matmul_rows, int matmul_cols)
{
	__int64 weight_count = (__int64)matmul_cols * matmul_rows;
	init_random_int8(ts->input_q, matmul_rows);
	init_random_float(ts->input_s, matmul_rows / 64 + 1);
	init_random_int8(ts->weight_q, weight_count);
	init_random_float(ts->weight_s, weight_count / 64 + 1);
}

void tensor_q8_free(tensor_q8 *ts)
{
	free(ts->input_q);
	free(ts->input_s);
	free(ts->weight_q);
	free(ts->weight_s);
}

void tensor_f32_alloc(tensor_f32 *ts, int matmul_rows, int matmul_cols)
{
	__int64 weight_count = (__int64)matmul_cols * matmul_rows;
	__int64 scale_count = weight_count / 64 + 1;
	ts->input_f = (float *)malloc(matmul_rows * sizeof(float));
	ts->input_s = (float *)malloc((matmul_rows / 64 + 1) * sizeof(float));
	ts->weight_q = (char *)malloc(weight_count);
	ts->weight_s = (float *)malloc(scale_count * sizeof(float));
	if (!ts->input_f || !ts->input_s || !ts->weight_q || !ts->weight_s)
	{
		printf("[FATAL] tensor_f32_alloc: out of memory for %dx%d tensor "
		       "(~%lld bytes).\n",
		       matmul_rows, matmul_cols, (long long)weight_count);
		exit(EXIT_FAILURE);
	}
}

void tensor_f32_fill(tensor_f32 *ts, int matmul_rows, int matmul_cols)
{
	__int64 weight_count = (__int64)matmul_cols * matmul_rows;
	init_random_float(ts->input_f, matmul_rows);
	init_random_float(ts->input_s, matmul_rows / 64 + 1);
	init_random_int8(ts->weight_q, weight_count);
	init_random_float(ts->weight_s, weight_count / 64 + 1);
}

void tensor_f32_free(tensor_f32 *ts)
{
	free(ts->input_f);
	free(ts->input_s);
	free(ts->weight_q);
	free(ts->weight_s);
}

/* ======================================================================
 * Dispatched matmul (one call, Q8 or F32)
 * ====================================================================== */

void dispatch_one(void)
{
	if (use_q8_mode)
	{
		v7_q8_tensor x;
		v7_q8_tensor w;
		x.q = g_tsq.input_q;
		x.s = g_tsq.input_s;
		w.q = g_tsq.weight_q;
		w.s = g_tsq.weight_s;
		inference_dll_ptr.v7d_q8(xout_buffer, &x, &w, dispatch_n, dispatch_d);
	}
	else
	{
		v7_f32_tensor xf;
		v7_q8_tensor w;
		xf.data = g_tsf.input_f;
		xf.scales = g_tsf.input_s;
		w.q = g_tsf.weight_q;
		w.s = g_tsf.weight_s;
		inference_dll_ptr.v7d_f32(xout_buffer, &xf, &w, dispatch_n, dispatch_d);
	}
}

/* Scan shape_blocks for the block whose (n,d) header matches the experiment
 * shape. Blocks 2..255 are per-shape configs; the first block with a zero
 * n ends the list. A missing match is a FATAL: without the per-shape block
 * there is nowhere to write any genome's config. */
unsigned char *find_matched_block(void)
{
	for (int block_index = 2; block_index < SB_MAX_BLOCKS; block_index++)
	{
		int *header = (int *)(shape_blocks_ptr + block_index * SB_SIZE);
		if (header[0] == 0)
			break; /* first empty block = end of the per-shape list */
		if (header[0] == dispatch_n && header[1] == dispatch_d)
			return (unsigned char *)header;
	}
	printf("[FATAL] find_matched_block: no per-shape block for %dx%d in the "
	       "loaded DB -- rebuild the DB from an INI that has a [%dx%d] section.\n",
	       dispatch_n, dispatch_d, dispatch_n, dispatch_d);
	exit(EXIT_FAILURE);
}

/* ======================================================================
 * Genome mutation engine (transplanted from autotune_dispatch.c's
 * hill-climb phase -- the same draw, legality and sanitize machinery,
 * operating on the genome struct instead of param_snapshot)
 * ====================================================================== */

#define MAX_JOBS 4096 /* matches ASM MAX_JOBS: num_chunks and split g1+g2 cap */

/* Legal windows for the 22 dispatch params, in dispatch-index order
 * (num_chunks, batch counts and so on through taper_shift and
 * quantize_chunk_groups) -- the autotuner's tuning table verbatim. */
int tune_min[DISPATCH_PARAM_COUNT] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1 };
int tune_max[DISPATCH_PARAM_COUNT] = { 4096, 64, 64, 64, 64, 4096, 4096, 4096, 128, 64, 64, 64, 64, 64, 4096, 4096, 4096, 99, 256, 256, 112, 64 };

/* Prefetch-type support (CPUID-probed at startup): type ids the CPU cannot
 * execute must never reach the SMC patcher, so the sanitizer remaps them to
 * a random supported type. */
int pf_type_ok[7];            /* 1 = this cpu executes the type's instruction */
int pf_type_supported[7];     /* the supported type ids (the mutation draw pool) */
int pf_type_supported_count;  /* how many of the 7 types are supported */

/* Uniform integer draw in [lo,hi] (inclusive). The window spans up to 2^32
 * values (half of the weight-matrix bytes on huge shapes): frand() carries
 * only 23 random mantissa bits, so a double-scaled draw would quantize to
 * ~512-step granularity and most legal distances would be UNREACHABLE. Two
 * PRNG words give a full 64-bit draw; the modulo bias is negligible. */
int rng_int(int lo, int hi)
{
	__int64 span = (__int64)hi - (__int64)lo + 1;
	if (span <= 0)
		return lo;
	unsigned __int64 r = ((unsigned __int64)rng_u32() << 32) | (unsigned __int64)rng_u32();
	return (int)((__int64)lo + (__int64)(r % (unsigned __int64)span));
}

void detect_pf_support(void)
{
	int regs[4];
	pf_type_ok[0] = 1; /* prefetchnta: SSE */
	pf_type_ok[1] = 1; /* prefetcht0:  SSE */
	pf_type_ok[2] = 1; /* prefetcht1:  SSE */
	pf_type_ok[3] = 1; /* prefetcht2:  SSE */
	pf_type_ok[4] = 0;
	pf_type_ok[5] = 0;
	pf_type_ok[6] = 0;
	__cpuid(regs, 0);
	int max_basic = regs[0]; /* highest basic leaf from CPUID.00H:EAX */
	__cpuid(regs, 0x80000000);
	unsigned int max_ext = (unsigned int)regs[0]; /* highest extended leaf */
	if (max_basic >= 7)
	{
		int ext[4];
		__cpuidex(ext, 7, 0);
		pf_type_ok[5] = (ext[2] >> 0) & 1;   /* prefetchwt1: CPUID.7.ECX[0] */
		pf_type_ok[6] = (ext[2] >> 25) & 1;  /* cldemote:    CPUID.7.ECX[25] */
	}
	if (max_ext >= 0x80000001u)
	{
		int ext[4];
		__cpuidex(ext, 0x80000001, 0);
		pf_type_ok[4] = (ext[2] >> 8) & 1;   /* prefetchw: CPUID.80000001H:ECX[8] */
	}
	pf_type_supported_count = 0;
	for (int t = 0; t < 7; t++)
		if (pf_type_ok[t])
			pf_type_supported[pf_type_supported_count++] = t;
}

/* One random SUPPORTED prefetch type (the sanitizer's replacement draw).
 * The SSE family is always supported, so the pool is never empty. */
int pf_type_draw(void)
{
	return pf_type_supported[rng_int(0, pf_type_supported_count - 1)];
}

/* Sort three values ascending: the batch/caller threshold triples require
 * strictly ascending tier boundaries. */
void sort3(int *first, int *second, int *third)
{
	if (*first > *second)
	{
		int tmp = *first;
		*first = *second;
		*second = tmp;
	}
	if (*second > *third)
	{
		int tmp = *second;
		*second = *third;
		*third = tmp;
	}
	if (*first > *second)
	{
		int tmp = *first;
		*first = *second;
		*second = tmp;
	}
}

/* Re-sort the two ascending-threshold triples (batch thresholds at param
 * indexes 5-7, caller thresholds at 14-16) after independent mutation draws. */
void sort_threshold_triples(genome *g)
{
	sort3(&g->batch_med_thresh, &g->batch_large_thresh, &g->batch_huge_thresh);
	sort3(&g->caller_batch_med_thresh, &g->caller_batch_large_thresh, &g->caller_batch_huge_thresh);
}

/* Enforce strictly ascending lo < mid < hi on one threshold triple, TOTAL
 * over every input: sort, raise the floor inward, push the ceiling down if
 * the top ran past hi_max, re-check the floor. Mutated genomes must never
 * abort the session, so the normalizer always produces a valid triple; the
 * FATAL fires only if [lo_min, hi_max] itself cannot hold three distinct
 * values -- a bounds-table misconfiguration. */
void fix_threshold_triple(int *lo, int *mid, int *hi, int hi_max, int lo_min)
{
	sort3(lo, mid, hi);
	if (*lo < lo_min)
		*lo = lo_min;
	if (*mid < *lo + 1)
		*mid = *lo + 1;
	if (*hi < *mid + 1)
		*hi = *mid + 1;
	if (*hi > hi_max)
	{
		*hi = hi_max;
		if (*mid > *hi - 1)
			*mid = *hi - 1;
		if (*lo > *mid - 1)
			*lo = *mid - 1;
	}
	if (!(*lo < *mid && *mid < *hi) || *lo < lo_min)
	{
		printf("[FATAL] fix_threshold_triple: threshold range [%d,%d] cannot "
		       "hold three strictly ascending boundaries (%d,%d,%d).\n",
		       lo_min, hi_max, *lo, *mid, *hi);
		exit(EXIT_FAILURE);
	}
}

/* The ONLY legal way to mutate a prefetch distance. The window is
 * +/- (d * w / 2) -- half of the weight matrix (int8 layout); the engine
 * interprets values in [0..31] as prefetch-disabled and negative values as
 * backward prefetches, so no special-casing happens here. Mutation type is
 * a uniform rand(0..4):
 *   0: new distance = uniform random in [-half, +half]
 *   1: distance = distance * 2
 *   2: distance = distance / 2
 *   3: distance = distance + (random in [-half,+half]) / (random in [-half,+half])
 *   4: distance = distance +/- (random in [1,half]) / (random in [1,half])
 * (the ratio draw redraws until the denominator is non-zero). An unchanged
 * result steps off the current value so the caller never re-measures the
 * same config. */
int mutate_distance(int rows, int cols, int current_dist)
{
	__int64 wide = (__int64)rows * cols / 2;
	if (wide > 2147483647)
		wide = 2147483647; /* keep the whole window inside the signed disp32 budget */
	int half = (int)wide;
	if (half < 1)
		half = 1;
	int window_low = -half;
	int window_high = half;
	int mutation_kind = rng_int(0, 4);
	int new_distance;
	switch (mutation_kind)
	{
	case 0:
		new_distance = rng_int(window_low, window_high);
		break;
	case 1:
	{
		/* x2 computed in 64-bit with BOTH bounds clamped before the int
		 * cast: a current_dist near +half doubles past window_high, one near
		 * -half doubles past window_low -- either cast would overflow. */
		__int64 doubled = (__int64)current_dist * 2;
		if (doubled > window_high)
			doubled = window_high;
		if (doubled < window_low)
			doubled = window_low;
		new_distance = (int)doubled;
	}
		break;
	case 2:
		new_distance = current_dist / 2;
		break;
	default:
	{
		/* Signed ratio: numerator and denominator are each drawn over the
		 * FULL SIGNED window, so the ratio lands on either side of zero and
		 * the mutation can walk a positive distance back toward 0 and below.
		 * Double math: half exceeds 2^24 for large matrices, where float
		 * stops resolving integers. The denominator redraws until non-zero,
		 * and the ratio is clamped into the legal window in double space
		 * before the int cast. The SUM is computed in 64-bit: current +
		 * ratio + jitter overflows a signed int for huge legal windows. */
		double numerator = (frand() * 2.0 - 1.0) * (double)half;
		double denominator;
		do
		{
			denominator = (frand() * 2.0 - 1.0) * (double)half;
		} while (denominator == 0.0);
		double ratio = numerator / denominator;
		if (ratio > (double)window_high)
			ratio = (double)window_high;
		if (ratio < (double)window_low)
			ratio = (double)window_low;
		__int64 wide_sum = (__int64)current_dist + (__int64)ratio + rng_int(-32, 32);
		if (wide_sum > window_high)
			wide_sum = window_high;
		if (wide_sum < window_low)
			wide_sum = window_low;
		new_distance = (int)wide_sum;
	}
		break;
	case 4:
	{
		/* Positive-ratio step: numerator and denominator are drawn over
		 * 1..half (strictly positive -- no zero denominator, no redraw
		 * loop), the integer quotient gets a random sign, and the signed
		 * step shifts the current distance. A quotient of 1 nudges by one
		 * line, a quotient of half jumps straight toward the window edge.
		 * The sum is computed in 64-bit and clamped before the int cast:
		 * the current distance can sit at the window edge and a max
		 * quotient can push the sum past +2^31 for huge legal windows. */
		int numerator = rng_int(1, half);
		int denominator = rng_int(1, half);
		int step = numerator / denominator;
		if (rng_int(0, 1))
			step = -step;
		__int64 sum = (__int64)current_dist + (__int64)step;
		if (sum > window_high)
			sum = window_high;
		if (sum < window_low)
			sum = window_low;
		new_distance = (int)sum;
	}
		break;
	}
	if (new_distance < window_low)
		new_distance = window_low;
	if (new_distance > window_high)
		new_distance = window_high;
	/* An unchanged result steps off the current value so the caller never
	 * re-measures the same config: cases 1/2 on a distance of 0 or 1 return
	 * the current value, and a donor identical to the baseline would then
	 * put an exact clone into the pool. One step toward the window's open
	 * side always changes the distance (the window holds >= 2 values). */
	if (new_distance == current_dist)
	{
		if (current_dist > window_low)
			new_distance = current_dist - 1;
		else
			new_distance = current_dist + 1;
	}
	return new_distance;
}

/* Legal bounds for dispatch param idx on the CURRENT experiment shape; the
 * split group counts (idx 18/19) are additionally bounded by the columns
 * their group receives at the trial's chunk_split_pct. */
void dispatch_field_bounds(int idx, const genome &g, int *lo, int *hi)
{
	*lo = tune_min[idx];
	*hi = tune_max[idx];
	if (idx == 18 && g.chunk_split_pct > 0)
	{
		int g1_cols = dispatch_d * g.chunk_split_pct / 100;
		if (g1_cols < 1)
			g1_cols = 1;
		if (g1_cols < *hi)
			*hi = g1_cols; /* each group-1 chunk >= 1 column */
	}
	else if (idx == 19 && g.chunk_split_pct > 0)
	{
		int remaining = dispatch_d - dispatch_d * g.chunk_split_pct / 100;
		if (remaining < 1)
			remaining = 1;
		if (remaining < *hi)
			*hi = remaining; /* each group-2 chunk >= 1 column */
	}
}

/* Enforce the engine's config invariants on a MUTATED genome: range-clamp
 * every dispatch param, ascending threshold triples, taper requires
 * min_chunk_cols >= 4, split group counts within their column budgets,
 * unsupported prefetch types remapped to a supported one (OWNED slots
 * only -- another kernel's tuned slots are preserved exactly). */
void sanitize_candidate(genome *g)
{
	int *params[DISPATCH_PARAM_COUNT] = {
		&g->num_chunks, &g->batch_small, &g->batch_medium, &g->batch_large,
		&g->batch_huge, &g->batch_med_thresh, &g->batch_large_thresh, &g->batch_huge_thresh,
		&g->chunk_multiplier, &g->min_chunk_cols, &g->caller_batch_small, &g->caller_batch_medium,
		&g->caller_batch_large, &g->caller_batch_huge, &g->caller_batch_med_thresh, &g->caller_batch_large_thresh,
		&g->caller_batch_huge_thresh, &g->chunk_split_pct, &g->chunk_group_1_count, &g->chunk_group_2_count,
		&g->taper_shift, &g->quantize_chunk_groups
	};
	for (int i = 0; i < DISPATCH_PARAM_COUNT; i++)
	{
		if (*params[i] < tune_min[i])
			*params[i] = tune_min[i];
		if (*params[i] > tune_max[i])
			*params[i] = tune_max[i];
	}
	fix_threshold_triple(&g->batch_med_thresh, &g->batch_large_thresh, &g->batch_huge_thresh,
	                     tune_max[7], tune_min[5]);
	fix_threshold_triple(&g->caller_batch_med_thresh, &g->caller_batch_large_thresh, &g->caller_batch_huge_thresh,
	                     tune_max[16], tune_min[14]);
	/* min_chunk_cols: a non-pow2 W_min takes the div/imul AlignDown path and
	 * aligns exactly, so no power-of-two normalization -- but the taper's
	 * 4-row kernel block needs >= 4, so a taper-on candidate normalizes UP. */
	if (g->taper_shift > 0 && g->min_chunk_cols < 4)
		g->min_chunk_cols = 4;
	/* Split fill group counts must stay within their column budgets or the
	 * split path fails on a zero-column chunk. */
	if (g->chunk_split_pct > 0)
	{
		int pct = g->chunk_split_pct;
		int g1 = g->chunk_group_1_count;
		int g2 = g->chunk_group_2_count;
		int g1_cols = dispatch_d * pct / 100;
		int min_pct = (100 + dispatch_d - 1) / dispatch_d;
		if (g1_cols < 1 && min_pct <= 99)
		{
			pct = min_pct; /* smallest pct that assigns group 1 at least one column */
			g1_cols = dispatch_d * pct / 100;
			g->chunk_split_pct = pct;
		}
		if (g1_cols < 1 || dispatch_d - g1_cols < 1)
		{
			g->chunk_split_pct = 0; /* degenerate shapes admit no valid split */
		}
		else
		{
			int remaining = dispatch_d - g1_cols;
			if (g1 < 1)
				g1 = 1;
			if (g1 > g1_cols)
				g1 = g1_cols; /* each g1 chunk >= 1 col */
			if (g2 < 1)
				g2 = 1;
			if (g2 > remaining)
				g2 = remaining; /* each g2 chunk >= 1 col */
			g->chunk_group_1_count = g1;
			g->chunk_group_2_count = g2;
		}
	}
	for (int slot = 0; slot < PF_SLOTS; slot++)
	{
		if (!pf_owned[slot])
			continue;
		int t = g->pf_type[slot];
		if (t < PF_TYPE_MIN || t > PF_TYPE_MAX || !pf_type_ok[t])
			g->pf_type[slot] = pf_type_draw();
	}
}

/* Get one mutated genome: a COPY of the baseline with a random number of
 * 1..mutations_max mutations (the field picks may repeat -- a repeated
 * pick re-mutates the same field, so the EXPECTED distinct-field count is
 * slightly below the drawn count), then the legality pass every generated
 * genome goes through. Every active
 * dispatch param, every OWNED pf distance and every OWNED pf type carries
 * equal weight in the field draw, each mutation is a uniform redraw over
 * the field's legal window (distances through mutate_distance's five
 * kinds), the threshold triples are re-sorted and the result sanitized.
 * The mutant carries NO speed record of its own (shape_a_gbs/best_gbs
 * reset to 0 -- it has not been measured yet). Transplanted from
 * autotune_dispatch.c's hill-climb round construction; measuring and
 * acceptance are NOT part of this -- the cycle sort wires those.
 *
 * Sexual crossover draws the donor through the CURRENT LIST RANKING
 * (cycle_list[0..donor_count), the front region the sort has established):
 * f*f biases the donor toward the ranking front where the fast genomes
 * collect. The donor donates a random number of parameter slots to the
 * child -- same slot to same slot, straight transfer, double transfers of
 * one slot stay possible and harmless. The donated count is a uniform
 * frand() * (total slots * sex_influence/100): at 100% the child can
 * receive any fraction of ALL slots, at 0 nothing is transferred
 * (asexual). The legality pass runs again over the mixed genome so
 * threshold triples that crossed parents stay strictly ascending. */
genome get_mutated_genome(const genome &baseline, int donor_count)
{
	genome trial = baseline;
	trial.shape_a_gbs = 0; /* the mutant has no measurement of its own yet */
	trial.best_gbs = 0;
	int active_params[DISPATCH_PARAM_COUNT];
	int num_active_params = 0;
	for (int param_index = 0; param_index < DISPATCH_PARAM_COUNT; param_index++)
		active_params[num_active_params++] = param_index;
	int mutations = rng_int(1, mutations_max); /* fields changed this trial */
	for (int m = 0; m < mutations; m++)
	{
		int field_pick = rng_int(0, num_active_params + 2 * pf_owned_count - 1);
		if (field_pick < num_active_params)
		{
			int param_slot = active_params[field_pick];
			int lowest_legal;
			int highest_legal;
			dispatch_field_bounds(param_slot, trial, &lowest_legal, &highest_legal);
			if (rng_int(0, 3) < 3)
			{
				/* 75%: NUDGE the current value by a sign-flipped integer
				 * ratio of the field's span instead of redrawing it --
				 * the quotient of two uniform 1..span draws is usually 1,
				 * so most mutations walk +-1 line from the incumbent and a
				 * few jump far, keeping the search near good genomes
				 * while never losing reach. Clamped into the legal window
				 * (a nudge at the window edge would otherwise exit it). */
				int span = 1 + (highest_legal - lowest_legal > 0 ? highest_legal - lowest_legal : lowest_legal - highest_legal);
				int step = rng_int(1, span) / rng_int(1, span);
				if (rng_int(0, 1))
					step = -step;
				int nudged = trial.*genome_param_ptr[param_slot] + step;
				if (nudged < lowest_legal)
					nudged = lowest_legal;
				if (nudged > highest_legal)
					nudged = highest_legal;
				trial.*genome_param_ptr[param_slot] = nudged;
			}
			else
			{
				/* 25%: the full uniform redraw over the legal window. */
				trial.*genome_param_ptr[param_slot] = rng_int(lowest_legal, highest_legal);
			}
		}
		else if (field_pick < num_active_params + pf_owned_count)
		{
			int pf_slot = pf_owned_list[field_pick - num_active_params];
			trial.pf_dist[pf_slot] = mutate_distance(dispatch_n, dispatch_d, trial.pf_dist[pf_slot]);
		}
		else
		{
			int pf_slot = pf_owned_list[field_pick - num_active_params - pf_owned_count];
			int new_type = pf_type_draw();
			trial.pf_type[pf_slot] = new_type;
		}
	}
	sort_threshold_triples(&trial);
	sanitize_candidate(&trial);

	/* Sexual crossover: the donor is drawn through the CURRENT LIST RANKING
	 * (cycle_list[0..donor_count)), NOT an archive order. f*f biases the
	 * donor toward the ranking front where the fast genomes collect. The
	 * donor donates a random number of parameter slots to the child --
	 * same slot to same slot, straight transfer, double transfers of one
	 * slot stay possible and harmless. The donated count is a uniform
	 * frand() * (total slots * sex_influence/100): at 100% the child can
	 * receive any fraction of ALL slots, at 0 nothing is transferred
	 * (asexual). The legality pass runs again over the mixed genome so
	 * threshold triples that crossed parents stay strictly ascending. */
	if (sex_influence_percent > 0.0 && donor_count > 0)
	{
		int total_slots = DISPATCH_PARAM_COUNT + 2 * pf_owned_count;
		int transfer_count = (int)(frand() * (double)total_slots * sex_influence_percent / 100.0);
		if (transfer_count > total_slots)
			transfer_count = total_slots;
		float f = frand();
		f = f * f;
		int pick = (int)((double)donor_count * f);
		if (pick >= donor_count)
			pick = donor_count - 1;
		const genome &donor = cycle_list[pick];
		for (int transfer = 0; transfer < transfer_count; transfer++)
		{
			int slot = rng_int(0, total_slots - 1);
			if (slot < DISPATCH_PARAM_COUNT)
			{
				int param_index = slot;
				trial.*genome_param_ptr[param_index] = donor.*genome_param_ptr[param_index];
			}
			else if (slot < DISPATCH_PARAM_COUNT + pf_owned_count)
			{
				int pf_slot = pf_owned_list[slot - DISPATCH_PARAM_COUNT];
				trial.pf_dist[pf_slot] = donor.pf_dist[pf_slot];
			}
			else
			{
				int pf_slot = pf_owned_list[slot - DISPATCH_PARAM_COUNT - pf_owned_count];
				trial.pf_type[pf_slot] = donor.pf_type[pf_slot];
			}
		}
		sort_threshold_triples(&trial);
		sanitize_candidate(&trial);
	}
	return trial;
}

/* ======================================================================
 * Genome config apply and per-genome measurement
 * ====================================================================== */

/* Push one genome's config into the dispatcher. Dispatch params go into the
 * same dword slot of BOTH blocks: block 0 is the live param_set the
 * dispatcher reads immediately, and the matched block is the shadow copy a
 * shape-switch config reload restores from. The pf arrays of the
 * benchmarked kernel's OWNED slots go into both blocks the same way, and
 * every owned slot whose value differs from the stored state is ALSO
 * patched live through the exported SMC entries, distance first then type:
 * the dispatcher re-applies pf from the blocks only on a shape switch, and
 * the experiment dispatches one shape for its whole run. Unowned slots
 * belong to other kernels: they are neither written nor patched here.
 * No console output: this runs per-measure step and the sort's own prints
 * carry the flow. */
void apply_genome_config(const genome &g)
{
	for (int param_index = 0; param_index < DISPATCH_PARAM_COUNT; param_index++)
		loaded_params[param_index] = g.*genome_param_ptr[param_index];

	int *param_block0 = (int *)(shape_blocks_ptr + SB_PARAM_OFF);
	int *param_matched = (int *)(matched_block_ptr + SB_PARAM_OFF);
	for (int param_index = 0; param_index < DISPATCH_PARAM_COUNT; param_index++)
	{
		int slot = sb_param_off_for_idx[param_index];
		param_block0[slot] = loaded_params[param_index];
		param_matched[slot] = loaded_params[param_index];
	}

	int *dist_block0 = (int *)(shape_blocks_ptr + SB_PF_DIST_OFF);
	int *type_block0 = (int *)(shape_blocks_ptr + SB_PF_TYPE_OFF);
	int *dist_matched = (int *)(matched_block_ptr + SB_PF_DIST_OFF);
	int *type_matched = (int *)(matched_block_ptr + SB_PF_TYPE_OFF);
	for (int owned_position = 0; owned_position < pf_owned_count; owned_position++)
	{
		int slot = pf_owned_list[owned_position];
		int dist = g.pf_dist[slot];
		int type = g.pf_type[slot];
		int dist_changed = (dist != dist_block0[slot]) || (dist != dist_matched[slot]);
		int type_changed = (type != type_block0[slot]) || (type != type_matched[slot]);
		dist_block0[slot] = dist;
		dist_matched[slot] = dist;
		type_block0[slot] = type;
		type_matched[slot] = type;
		if (dist_changed)
		{
			/* Distance first: the distance patcher restores the slot's backup
			 * bytes (re-enabling a disabled slot), while the type patcher
			 * skips a disabled slot -- the type patch after it re-establishes
			 * the genome's type even on a previously disabled slot. */
			inference_dll_ptr.set_prefetch_distance(slot, dist);
			inference_dll_ptr.set_prefetch_type(slot, type);
		}
		else if (type_changed)
		{
			inference_dll_ptr.set_prefetch_type(slot, type);
		}
	}
}

/* ======================================================================
 * Random probe draw (the probe fill's genome generator)
 * ====================================================================== */

/* Parse one NxM / N:M shape token into rows/cols (1..1<<20). Returns 0 on
 * success, 1 on a malformed or out-of-range token. */
int parse_shape_token(char *token, int *rows, int *cols)
{
	char tok[64];
	snprintf(tok, sizeof(tok), "%s", token);
	char *sep = strchr(tok, 'x');
	if (!sep)
		sep = strchr(tok, ':');
	if (!sep || sep == tok)
		return 1;
	*sep = '\0';
	char *n_end = 0;
	char *d_end = 0;
	errno = 0;
	long r = strtol(tok, &n_end, 10);
	long c = strtol(sep + 1, &d_end, 10);
	if (n_end == tok || *n_end || !d_end || *(sep + 1) == '\0' || *d_end ||
	    r < 1 || r > (1 << 20) || c < 1 || c > (1 << 20))
		return 1;
	*rows = (int)r;
	*cols = (int)c;
	return 0;
}

/* Fill g with a FULLY RANDOM probe genome: all 22 dispatch params uniform
 * over their legal windows, all OWNED pf distances uniform over the whole
 * legal window, all OWNED pf types uniform over the full type range (the
 * legality pass remaps unsupported ones). NOTHING is inherited from any
 * prior state. Unowned slots stay 0 -- the apply touches owned slots only.
 * Threshold triples are sorted so the draw is a legal config. */
void draw_random_probe_genome(genome *g)
{
	for (int param_index = 0; param_index < DISPATCH_PARAM_COUNT; param_index++)
		g->*genome_param_ptr[param_index] = rng_int(tune_min[param_index], tune_max[param_index]);
	for (int owned_position = 0; owned_position < pf_owned_count; owned_position++)
	{
		int slot = pf_owned_list[owned_position];
		g->pf_dist[slot] = rng_int(pf_dist_lo, pf_dist_hi);
		g->pf_type[slot] = rng_int(PF_TYPE_MIN, PF_TYPE_MAX);
	}
	sort_threshold_triples(g);
	sanitize_candidate(g);
}

/* ======================================================================
 * The bare A/B/A measurement, the probe fill and the cycle sort
 * ====================================================================== */

/* One bare timed dispatch of the given genome: the config apply sits
 * OUTSIDE the timed region, exactly one dispatch is measured. No warmup
 * rounds, no accumulation, no min-diff, no caps -- the cycle sort's
 * single measurement primitive, shared by the probe fill and the sweep.
 * The genome's last-measured rate is recorded on it. Returns the elapsed
 * seconds; the rate in GB/s rides back through gbs_out. */
double measure_genome_once(genome &g, double *gbs_out)
{
	apply_genome_config(g);
	double stamp_a = get_seconds();
	dispatch_one();
	double stamp_b = get_seconds();
	double secs = stamp_b - stamp_a;
	double rate = shape_bytes_per_call / secs / 1e9;
	g.shape_a_gbs = rate;
	if (gbs_out)
		*gbs_out = rate;
	return secs;
}

/* Draw the reference genome: one fully random draw from the same seeded
 * RNG, made at startup BEFORE the probe fill. It is never sorted, never
 * mutated, never written into the list -- the untouched ruler the final
 * benchmark compares the sorted ranking against. The same seed reproduces
 * the identical reference on every run. */
void draw_reference_genome(void)
{
	char shape_text[32];
	snprintf(shape_text, sizeof(shape_text), "%dx%d", dispatch_n, dispatch_d);

	reference_genome.shape_n = dispatch_n;
	reference_genome.shape_d = dispatch_d;
	reference_genome.counter = 0;
	reference_genome.path = "(reference)";
	reference_genome.section = shape_text;
	reference_genome.was_modified = 0;
	reference_genome.best_gbs = 0;
	reference_genome.shape_a_gbs = 0;
	draw_random_probe_genome(&reference_genome);
	sorter_out("reference genome drawn (seeded, never changed)\n");
}

/* Fill the fixed 127-slot list with fully random probe genomes. NO
 * measurement happens here and nothing is elected: every probe is drawn
 * from the seeded RNG and written into its list slot, nothing more. The
 * same seed produces the identical probe list on every run -- the sort's
 * climb from this shared starting point is what the parameter-effect
 * comparison reads. */
void fill_list_with_probes(void)
{
	char shape_text[32];
	snprintf(shape_text, sizeof(shape_text), "%dx%d", dispatch_n, dispatch_d);

	for (int slot = 0; slot < CYCLE_LIST_SIZE; slot++)
	{
		genome *probe = &cycle_list[slot];
		probe->shape_n = dispatch_n;
		probe->shape_d = dispatch_d;
		probe->counter = slot + 1;
		probe->path = "(probe)";
		probe->section = shape_text;
		probe->was_modified = 0;
		probe->best_gbs = 0;
		probe->shape_a_gbs = 0;
		draw_random_probe_genome(probe);
	}
	sorter_out("fill complete: %d random probes inserted (seeded, unmeasured)\n", CYCLE_LIST_SIZE);
}

/* The partner fraction in [0,1) for the rotating position -- the formula
 * --partner selects. powA: frand()^A (A=2 is the historical default, a
 * Beta(1/2,1) law whose median lands a quarter of the way down the range).
 * tourK: the minimum of K uniform draws (deterministic tournament of size
 * K, CDF 1-(1-x)^K). expA: the truncated exponential on [0,1) with decay
 * A, drawn by inverse CDF -- the exponential-rank shape whose front
 * density stays finite while every slot stays reachable. half: the
 * constant 0.5, the deterministic middle partner slot1 >> 1, consumes no
 * RNG at all. The stochastic formulas front-bias the partner; half is
 * purely geometric. */
double draw_partner_fraction(void)
{
	if (partner_mode[0] == 'h')
		return 0.5; /* half: slot2 = slot1 >> 1, the deterministic middle partner */
	if (partner_mode[0] == 'p')
	{
		double f = frand();
		if (partner_mode[3] == '1')
			return f;
		if (partner_mode[3] == '2')
			return f * f;
		if (partner_mode[3] == '3')
			return f * f * f;
		if (partner_mode[3] == '4')
		{
			double f2 = f * f;
			return f2 * f2;
		}
		/* pow6 */
		double f3 = f * f * f;
		return f3 * f3;
	}
	if (partner_mode[0] == 't')
	{
		int draws = atoi(&partner_mode[4]);
		double best = frand();
		for (int draw = 1; draw < draws; draw++)
		{
			double candidate = frand();
			if (candidate < best)
				best = candidate;
		}
		return best;
	}
	/* expA: x = -ln(1 - u * (1 - e^-A)) / A maps u in [0,1) onto [0,1) */
	double lambda = (partner_mode[3] == '2') ? 2.0 : 3.0;
	double span = 1.0 - exp(-lambda);
	return -log(1.0 - frand() * span) / lambda;
}

/* The cycle sort: --rounds rounds over the fixed list. One round = one
 * sweep: slot1 walks 0..126, slot2 is drawn by the --partner formula
 * (default pow2: frand()^2 * slot1, a partner below the rotating
 * position, biased toward the front), and the
 * pair is compared with the bare A/B/A ONLY -- lower index dispatched,
 * upper index dispatched, lower index dispatched again, no warmup. The
 * upper genome (b) wins the exchange exactly when its single time beats
 * min(a_first, a_second), and the exchange puts the faster genome at the
 * lower index. Every timed dispatch updates the cycle's best-rate
 * record; when the round over the list finishes, the recorded fastest
 * runtime prints in GB/s. With --mutate-percentage > 0, every sweep end
 * replaces the tail slots with mutants of frand()^2-drawn parents (the
 * displaced genomes are gone -- the list stays fixed at 127). */
void cycle_sort(int rounds)
{
	long long pair_steps = 0;
	double overall_best_gbs = 0.0; /* the best runtime over ALL cycles */

	for (int round = 1; round <= rounds; round++)
	{
		double cycle_best_gbs = 0.0; /* the fastest timed dispatch of THIS cycle */
		double gbs = 0.0;

		for (int slot1 = 0; slot1 < CYCLE_LIST_SIZE; slot1++)
		{
			int slot2 = (int)((double)slot1 * draw_partner_fraction()); /* partner below the rotating position, --partner selects the formula */
			if (slot2 >= slot1)
				continue; /* the draw hit the rotating slot itself (slot1 == 0) */

			/* The bare A/B/A: lower index is a (dispatched first and
			 * last), upper index is b (dispatched once). */
			double a_first = measure_genome_once(cycle_list[slot2], &gbs);
			if (gbs > cycle_best_gbs)
				cycle_best_gbs = gbs;
			double b_time = measure_genome_once(cycle_list[slot1], &gbs);
			if (gbs > cycle_best_gbs)
				cycle_best_gbs = gbs;
			double a_second = measure_genome_once(cycle_list[slot2], &gbs);
			if (gbs > cycle_best_gbs)
				cycle_best_gbs = gbs;
			pair_steps++;

			double a_min = a_first < a_second ? a_first : a_second;
			if (b_time < a_min)
			{
				/* b won the bracket: exchange so the faster genome ends
				 * at the lower index -- it moved up the ranking. */
				genome tmp = cycle_list[slot1];
				cycle_list[slot1] = cycle_list[slot2];
				cycle_list[slot2] = tmp;
			}
		}

		/* Sweep end: the mutant tail refresh, --mutate-percentage only.
		 * Both parents are frand()^2 draws through the whole list
		 * ranking, so the fast front is the likeliest parent while every
		 * genome stays reachable. The tail slots are overwritten in
		 * place -- the list stays fixed at 127 slots. */
		if (mutate_percentage > 0)
		{
			int mutate_count = CYCLE_LIST_SIZE * mutate_percentage / 100;
			if (mutate_count > CYCLE_LIST_SIZE / 2)
				mutate_count = CYCLE_LIST_SIZE / 2; /* the tail must not eat the population */
			if (mutate_count < 1)
				mutate_count = 1;
			for (int m = 0; m < mutate_count; m++)
			{
				int tail_slot = CYCLE_LIST_SIZE - mutate_count + m;
				float f = frand();
				f = f * f;
				int parent_pick = (int)((double)CYCLE_LIST_SIZE * f);
				if (parent_pick >= CYCLE_LIST_SIZE)
					parent_pick = CYCLE_LIST_SIZE - 1;
				genome mutant = get_mutated_genome(cycle_list[parent_pick], CYCLE_LIST_SIZE);
				mutant.was_modified = 1;
				cycle_list[tail_slot] = mutant;
			}
			sorter_out("cycle %d: %d fresh mutants in the tail\n", round, mutate_count);
		}

		/* The round over the list is finished: print the GB/s of the
		 * recorded fastest runtime of this cycle. */
		if (cycle_best_gbs > overall_best_gbs)
			overall_best_gbs = cycle_best_gbs;
		sorter_out("cycle %d/%d done (%lld compares): fastest runtime this cycle %.1f GB/s\n",
		       round, rounds, pair_steps, cycle_best_gbs);
		fflush(stdout);
	}

	/* Final report: the top-13 of the fixed list (fastest first -- the
	 * list IS the ranking) and the overall best rate. */
	sorter_out("FINAL top 13 of the fixed list (fastest first, last-measured rates):\n");
	int show = 13;
	if (show > CYCLE_LIST_SIZE)
		show = CYCLE_LIST_SIZE;
	for (int slot = 0; slot < show; slot++)
		sorter_out("  %2d: probe %d  %.1f GB/s\n", slot, cycle_list[slot].counter, cycle_list[slot].shape_a_gbs);
	sorter_out("best runtime over all %d cycle(s): %.1f GB/s\n", rounds, overall_best_gbs);
	sorter_out("cycle sort done\n");
}

/* The final benchmark: the sorted list's first 10 genomes against the
 * UNCHANGED reference genome (the random startup draw -- never sorted,
 * never mutated, never in the list). The reference is dispatched once
 * before the sweep, once after every genome measurement, and its FASTEST
 * reading (the min time, the same standard the sort's a_min applies) is
 * the comparator: a genome only counts as better when it beats the
 * reference's best observed dispatch. Both sides ride the same seed, so
 * this comparison is repeatable across runs and across
 * --mutate-percentage / --sex-influence settings -- the printed
 * improvement percent is the parameter-effect measurement. */
void benchmark_top_against_reference(void)
{
	const int show = 10;
	double gbs = 0.0;
	double ref_best_rate = 0.0;  /* max GB/s over all reference dispatches (= min time) */
	double best_genome_rate = 0.0;
	int best_slot = -1;
	double genome_rates[show];

	measure_genome_once(reference_genome, &gbs);
	ref_best_rate = gbs;

	for (int slot = 0; slot < show; slot++)
	{
		measure_genome_once(cycle_list[slot], &gbs);
		genome_rates[slot] = gbs;
		if (gbs > best_genome_rate)
		{
			best_genome_rate = gbs;
			best_slot = slot;
		}
		measure_genome_once(reference_genome, &gbs);
		if (gbs > ref_best_rate)
			ref_best_rate = gbs;
	}

	sorter_out("final benchmark: first %d list genomes vs the unchanged reference\n", show);
	for (int slot = 0; slot < show; slot++)
	{
		double vs_reference = genome_rates[slot] / ref_best_rate - 1.0;
		sorter_out("  %2d: probe %3d  %7.1f GB/s  %+6.1f%% vs reference\n",
		       slot, cycle_list[slot].counter, genome_rates[slot], vs_reference * 100.0);
	}

	double improvement = best_genome_rate / ref_best_rate - 1.0;
	sorter_out("reference (unchanged startup draw): %.1f GB/s\n", ref_best_rate);
	sorter_out("best genome: slot %d (probe %d) at %.1f GB/s -- %.1f%% %s than the reference\n",
	       best_slot, cycle_list[best_slot].counter, best_genome_rate,
	       improvement * 100.0,
	       improvement >= 0.0 ? "faster" : "slower");
}

/* ======================================================================
 * CLI parsing
 * ====================================================================== */

int parse_cli(int argc, char **argv, char **db_path)
{
	for (int i = 2; i < argc; i++)
	{
		if (strcmp(argv[i], "--db") == 0 && i + 1 < argc)
			*db_path = argv[++i];
		else if (strcmp(argv[i], "--shape") == 0 && i + 1 < argc)
		{
			snprintf(cycle_shape, sizeof(cycle_shape), "%s", argv[++i]);
		}
		else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			long value = strtol(argv[++i], &end, 10);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value < 1)
			{
				printf("[FATAL] --rounds: malformed or too-small count '%s' (>= 1)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			cycle_rounds = (int)value;
		}
		else if (strcmp(argv[i], "--sex-influence") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			double value = strtod(argv[++i], &end);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value < 0.0 || value > 100.0)
			{
				printf("[FATAL] --sex-influence: malformed or out-of-range percent '%s' (0..100)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			sex_influence_percent = value;
		}
		else if (strcmp(argv[i], "--mutate-percentage") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			long value = strtol(argv[++i], &end, 10);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value < 0 || value > 100)
			{
				printf("[FATAL] --mutate-percentage: malformed or out-of-range percent '%s' (0..100)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			mutate_percentage = (int)value;
		}
		else if (strcmp(argv[i], "--mutations-max") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			long value = strtol(argv[++i], &end, 10);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value < 1)
			{
				printf("[FATAL] --mutations-max: malformed or too-small count '%s' (>= 1)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			mutations_max = (int)value;
		}
		else if (strcmp(argv[i], "--partner") == 0 && i + 1 < argc)
		{
			const char *legal[] = { "pow1", "pow2", "pow3", "pow4", "pow6", "tour2", "tour3", "tour4", "exp2", "exp3", "half", 0 };
			int legal_index = 0;
			int is_legal = 0;
			++i;
			while (legal[legal_index])
			{
				if (strcmp(argv[i], legal[legal_index]) == 0)
					is_legal = 1;
				legal_index++;
			}
			if (!is_legal)
			{
				printf("[FATAL] --partner: unknown formula '%s' (legal: pow1 pow2 pow3 pow4 pow6 tour2 tour3 tour4 exp2 exp3 half)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			snprintf(partner_mode, sizeof(partner_mode), "%s", argv[i]);
		}
		else if (strcmp(argv[i], "--sorter-stdout-file") == 0 && i + 1 < argc)
		{
			char abs[2 * MAX_PATH];
			DWORD path_len = GetFullPathNameA(argv[++i], (DWORD)sizeof(abs), abs, 0);
			if (path_len == 0 || path_len >= sizeof(abs))
			{
				printf("[FATAL] --sorter-stdout-file: path invalid or too long: %s\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			snprintf(sorter_stdout_path, sizeof(sorter_stdout_path), "%s", abs);
		}
		else if (strcmp(argv[i], "--q8") == 0)
			use_q8_mode = 1;
		else if (strcmp(argv[i], "--f32") == 0)
			use_q8_mode = 0;
		else
		{
			printf("[FATAL] unrecognized argument '%s' (or it is missing its value)\n", argv[i]);
			return 1;
		}
	}
	if (!*db_path)
	{
		printf("Fail: --db <db_path> is required\n");
		return 1;
	}
	if (!cycle_shape[0])
	{
		printf("Fail: --shape <NxM> is required (the fixed list is filled by probe draws, no archive exists)\n");
		return 1;
	}
	return 0;
}

/* stdout-to-file capture: when --sorter-stdout-file is given, every sort
 * print goes to the FILE ONLY (full detail, flushed per line). Without
 * the option every print goes to the console as before. */
int sorter_out(const char *fmt, ...)
{
	char text[2048];
	va_list args;
	va_start(args, fmt);
	int written = _vsnprintf_s(text, sizeof(text), _TRUNCATE, fmt, args);
	va_end(args);
	if (sorter_stdout_capture)
	{
		if (written > 0)
			fputs(text, sorter_stdout_capture);
		fflush(sorter_stdout_capture); /* the capture file shows the current state at any moment */
		return written;
	}
	return fputs(text, stdout);
}

int main(int argc, char **argv)
{
	ZydisInit();

	/* Above-normal process priority keeps the measurement cadence steady when
	 * other user processes compete for CPU. Failure is a warning, not a
	 * FATAL: measuring at normal priority is valid, just noisier. */
	if (!SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS))
		printf("(warning: SetPriorityClass(ABOVE_NORMAL_PRIORITY_CLASS) failed, error %lu -- continuing at normal priority)\n", GetLastError());

	if (argc < 2)
	{
		printf("usage: %s <v7_threading.dll> --db <db_path> --shape <NxM> [--rounds <n>] [--mutations-max <n>] [--partner <mode>] [--q8|--f32]\n", argv[0]);
		return 1;
	}
	char *dll_path = argv[1];
	char *db_path = 0;
	use_q8_mode = 1; /* default; --q8 / --f32 override */
	kernel_label = "q8";
	if (parse_cli(argc, argv, &db_path))
		return 1;

	/* stdout capture first: everything below (engine init, the fill, the
	 * whole sort) tees into the capture file. */
	if (sorter_stdout_path[0])
	{
		sorter_stdout_capture = fopen(sorter_stdout_path, "w");
		if (!sorter_stdout_capture)
		{
			printf("[FATAL] --sorter-stdout-file: cannot open %s\n", sorter_stdout_path);
			return 1;
		}
	}

	/* CPUID-probe the prefetch-type support ONCE: the sanitizer's unsupported
	 * type remap and the mutation type draws read pf_type_ok -- without this
	 * call the table stays zeroed and every type silently collapses to NTA. */
	detect_pf_support();

	/* Kernel selection BEFORE the probe draw: the draw writes the OWNED pf
	 * slots only, and pf_owned/pf_owned_list/pf_owned_count depend solely on
	 * --q8/--f32. */
	if (use_q8_mode)
	{
		kernel_label = "q8";
		pf_owned_count = 8;
		for (int s = 0; s < 8; s++)
		{
			pf_owned[s] = 1;
			pf_owned_list[s] = s;
		}
	}
	else
	{
		kernel_label = "f32";
		pf_owned_count = 3;
		for (int s = 0; s < 3; s++)
		{
			pf_owned[8 + s] = 1;
			pf_owned_list[s] = 8 + s;
		}
	}

	/* The probe draw shape: the fixed list holds no archive genomes, so the
	 * shape comes from --shape and nothing else. */
	int probe_rows;
	int probe_cols;
	if (parse_shape_token(cycle_shape, &probe_rows, &probe_cols))
	{
		printf("[FATAL] --shape: malformed shape token '%s' (expected NxM)\n", cycle_shape);
		return 1;
	}
	dispatch_n = probe_rows;
	dispatch_d = probe_cols;

	/* The pf distance window from the shape: +/- (weight-matrix bytes/2),
	 * clamped into the signed disp32 budget. */
	{
		__int64 matrix_bytes = (__int64)dispatch_n * dispatch_d;
		__int64 half = matrix_bytes / 2;
		if (half > 2147483647)
			half = 2147483647;
		if (half < 1)
			half = 1;
		pf_dist_hi = (int)half;
		pf_dist_lo = -pf_dist_hi;
	}
	shape_bytes_per_call = (__int64)dispatch_n * dispatch_d * 5;

	/* Load DLL and initialize the engine from the database. */
	if (v7_load(dll_path, &inference_dll_ptr) != 0)
		return 1;
	if (!inference_dll_ptr.v7_init(db_path) || inference_dll_ptr.v7_engine_is_dead())
	{
		printf("[FATAL] v7_init failed for %s (engine dead: %d)\n", db_path, inference_dll_ptr.v7_engine_is_dead());
		v7_unload(&inference_dll_ptr);
		return 1;
	}

	shape_blocks_ptr = (unsigned char *)GetProcAddress((HMODULE)inference_dll_ptr.module, "shape_blocks");
	if (!shape_blocks_ptr)
	{
		printf("[FATAL] shape_blocks export not found in %s -- the dispatcher "
		       "config cannot be reached, the experiment cannot run.\n",
		       dll_path);
		v7_unload(&inference_dll_ptr);
		return 1;
	}

	printf("genome_cycle_sort: %dx%d %s, fixed list %d genomes, %d round(s), partner %s\n",
	       dispatch_n, dispatch_d, kernel_label, CYCLE_LIST_SIZE, cycle_rounds, partner_mode);

	/* Benchmark tensors for the experiment kernel. */
	if (use_q8_mode)
	{
		tensor_q8_alloc(&g_tsq, dispatch_n, dispatch_d);
		tensor_q8_fill(&g_tsq, dispatch_n, dispatch_d);
	}
	else
	{
		tensor_f32_alloc(&g_tsf, dispatch_n, dispatch_d);
		tensor_f32_fill(&g_tsf, dispatch_n, dispatch_d);
	}
	xout_buffer = (float *)malloc(dispatch_d * sizeof(float));
	if (!xout_buffer)
	{
		printf("[FATAL] main: out of memory for the %d-float output row.\n", dispatch_d);
		v7_unload(&inference_dll_ptr);
		return 1;
	}

	/* Spin mode: workers park on pause-loops instead of sleeping, so the
	 * dispatched benchmark sees steady latency (production inference runs
	 * the same way). */
	inference_dll_ptr.v7s_set_spin(1);

	/* Shape activation: one untimed dispatch materializes the per-shape
	 * block for (n,d) so every genome apply has a matched block to write. */
	dispatch_one();
	matched_block_ptr = find_matched_block();

	/* The reference draw (seeded, untouched ruler), the unmeasured probe
	 * fill, the cycle sort, then the top-10 vs reference benchmark. */
	draw_reference_genome();
	fill_list_with_probes();
	cycle_sort(cycle_rounds);
	benchmark_top_against_reference();

	/* Cleanup. */
	free(xout_buffer);
	if (use_q8_mode)
		tensor_q8_free(&g_tsq);
	else
		tensor_f32_free(&g_tsf);
	v7_unload(&inference_dll_ptr);
	return 0;
}
