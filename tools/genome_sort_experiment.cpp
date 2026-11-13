// v7 - part of the v7 inference engine
// Copyright (C) 2026 Gorg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * genome_sort_experiment.cpp -- C++ genome-analysis copy of
 * test_shape_speed.c, running a noisy-sort experiment over the archived
 * winner genomes.
 *
 * Derived from test_shape_speed.c (same dispatcher apply machinery, same
 * the shape_a metric); the measurement machinery is specialized for the
 * experiment:
 *
 *   - the per-comparison measurement keeps ONLY the dispatched-matmul part
 *     of the round (the float-sum CPU probe and the copy walks are gone:
 *     the float-sum at its 1-2 ms cost would dominate a 5 ms budget, and
 *     the walks are irrelevant to an ordering of matmul speed);
 *   - struct genome holds ALL parameters of one archived winner genome:
 *     the 22 dispatch params, both pf tables (41 slots), the identity
 *     (shape, archive counter, path, section), the archive's best_gbs
 *     record, and result fields;
 *   - a global std::vector<genome> holds the stride-selected subset
 *     (counter 1 + every GENOME_LOAD_STRIDE-th counter) of
 *     winner_genomes\<shape>_<counter>.ini, sorted by (shape, counter).
 *
 * The sort itself orders the genomes descending by matmul speed (fastest
 * first). Its comparison primitive is an accumulated-difference sequential
 * test: the two genomes' matmul dispatches alternate (20 warmup rounds
 * first, pure matmul -- no copy walks, no float-sum probe), each genome's
 * matmul time accumulates, and the compare resolves the moment the two
 * accumulators differ by more than the min-difference time (--min-diff, a
 * time in seconds). A real speed gap grows the difference linearly with
 * the round count while per-round noise sums as a random walk, so the test
 * is stable for near-ties and fast for clear gaps; the winner is the
 * genome with the smaller accumulated time. The sort itself is
 * probabilistic with a rotating anchor: index 1 cycles over every genome
 * (1, 2, ..., count, wrapping), index 2 is drawn as frand() squared times
 * index1 -- a partner below the rotating position, biased toward the front
 * -- and the pair exchanges so the faster ends at the lower index. Every
 * genome gets equal attention per sweep while the front region receives
 * the densest comparisons.
 *
 * N-stage cascade: stage 1 sorts every genome at --min-diff; between
 * stages the population is cut to --top-percent percent of the previous
 * stage and the accumulated-time requirement grows tenfold. The --runtime
 * budget divides equally among the stages. Example: 100 genomes, 3 stages,
 * 50% -> the stages sort 100, 50 and 25 genomes.
 *
 * Options:
 *   --db <path>            REQUIRED database path (no default)
 *   --stages <n>           number of cascade stages, 1..20 (default 2)
 *   --runtime <minutes>    total cascade budget, divided equally among the
 *                          stages (default 0 = unlimited)
 *   --top-percent <p>      per-stage survival fraction in percent, 1..99 (default 10)
 *   --min-diff <seconds>   stage-1 accumulated-time difference that resolves a compare
 *                          (default 0.005; each later stage runs at ten times this)
 *   --compare-max <s>      per-compare runtime cap in seconds (default 5): when a compare
 *                          reaches the cap (or the stage deadline) it decides by whichever
 *                          genome accumulated less time -- a compare is NEVER undecided
 *   --load-stride <n>      load genome 1 plus every Nth archive counter (default 1 =
 *                          the whole archive loads)
 *   --genome-dir <path>    archive directory to load the genomes from
 *                          (default: winner_genomes)
 *   --mutate-percentage <p>  at every sweep restart, copy the top p% of the
 *                          current order into the tail and replace each copy
 *                          with a hill-climb mutation of itself (default 0 =
 *                          no mutation). Each mutant can win real comparisons
 *                          and pull the selection forward.
 *   --target-pool-size <n>  when given, the sort runs as ONE stage (the phase
 *                          cuts are bypassed) and the pool is under TIME
 *                          CONTROL: at every sweep end the pool moves
 *                          linearly from its starting size toward <n>,
 *                          proportional to the elapsed budget -- shrinking
 *                          culls the worst-ranked genomes, extension fills
 *                          the new slots with mutated children
 *   --random-probes <n>    start the pool through ACCEPTANCE-FILTERED random
 *                          search: the first random probe becomes the
 *                          baseline (20-round self-warmup), every further
 *                          probe is measured against the current baseline
 *                          with the standard accumulated-difference compare
 *                          -- winners join the pool and become the baseline,
 *                          losers are discarded -- until n probes have been
 *                          added. No source measurements (archive) needed;
 *                          pair with --shape
 *   --shape <NxM>          the probe draw shape, REQUIRED with
 *                          --random-probes (no archive exists to inherit the
 *                          shape from); ignored otherwise
 *   --sex-influence <pct>  maximum percentage of parameter slots a SECOND
 *                          parent (drawn from the pool, frand()^2) donates
 *                          to a mutated genome, same slot to same slot
 *                          (default 0 = asexual; 100 = up to every slot
 *                          can be transferred). The legality pass reruns
 *                          over the mixed genome.
 *   --genome-output-dir <path>  when set, the reduced + mutated population's
 *                          modified genomes are written as
 *                          <dir>\<shape>_<n>.ini files (the archive format
 *                          the other tools load again) and top lists print
 *                          while the stage runs
 *   --q8                   Q8 matmul (default)
 *   --f32                  F32 matmul
 */
#include <windows.h>
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <immintrin.h> /* _rdrand32_step for the benchmark filler data */
#include <errno.h>     /* strtol range reporting */
#include <limits.h>    /* INT_MIN/INT_MAX */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>      /* the genome vector */
#include <string>      /* genome identity fields */
#include <algorithm>   /* std::sort for the load order */
#include <cstdarg>     /* va_start/va_list for the stdout-tee printf */
#include <cstdio>      /* _vsnprintf_s, fputs for the stdout-tee printf */
#include <cctype>      /* isdigit for diagnostic classification */

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

#define COMPARE_MIN_DIFF_DEFAULT 0.005 /* default accumulated-time difference (seconds) that resolves a compare */
#define COMPARE_MAX_SECONDS_DEFAULT 5.0 /* default per-compare runtime cap (seconds) */
#define WARMUP_ROUNDS 20        /* unmeasured warmup dispatches per genome before the timed interleave */
/* The compare measurement is pure dispatched matmul: 20 warmup rounds, then
 * genome A and genome B dispatch alternately, accumulating each genome's
 * matmul time, until the two accumulators differ by more than the
 * min-difference time. The copy walks and the float-sum CPU probe of the
 * drift-probe canon are not part of it. */
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
 * of the archive and the genome struct field order exactly. */
static const char *dispatch_param_names[DISPATCH_PARAM_COUNT] = {
	"num_chunks",          "batch_small",           "batch_medium",           "batch_large",
	"batch_huge",          "batch_med_thresh",      "batch_large_thresh",     "batch_huge_thresh",
	"chunk_multiplier",    "min_chunk_cols",        "caller_batch_small",     "caller_batch_medium",
	"caller_batch_large",  "caller_batch_huge",     "caller_batch_med_thresh", "caller_batch_large_thresh",
	"caller_batch_huge_thresh", "chunk_split_pct",  "chunk_group_1_count",    "chunk_group_2_count",
	"taper_shift",         "quantize_chunk_groups"
};

/* ======================================================================
 * Genome archive layer (C++)
 * ====================================================================== */

/* All parameters of one archived winner genome. The 22 dispatch params ride
 * in dispatch-index order (the same order dispatch_param_names labels), the
 * pf tables carry all 41 slots (the archive writes the tuned kernel's owned
 * slots; unowned slots load as 0). The result fields stay 0 until a
 * measurement step fills them. */
struct genome
{
	int shape_n;              /* matmul rows, parsed from the file name */
	int shape_d;              /* matmul cols */
	int counter;              /* archive counter from the file name */
	std::string path;         /* winner_genomes\<shape>_<counter>.ini */
	std::string section;      /* INI section name == "<shape>" */
	double best_gbs;          /* the archive's best_gbs record */
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
	int was_modified;         /* 1 = a mutation changed the config since the last
	                          * write_populations call (0 = archive-clean) */
	double shape_a_gbs;       /* experiment result fields, 0 = unmeasured */
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

std::vector<genome> genomes; /* the stride-selected subset of the archive */

int sorter_out(const char *fmt, ...); /* forward: defined near main, used by the sort prints */

/* Stage-heartbeat: the ONLY stderr output of the sort (declared here so the
 * stage loop can call it). One carriage-returned line per decile, so
 * watching the sorter shows one cycling percentage instead of a flood. The
 * stage number distinguishes a between-stage pause (the line sits at 100%
 * while the next stage's first decile is up to a tenth of its budget away)
 * from the end-of-sort population writer. */
void stage_heartbeat(int percent, int stage, int stages);

/* --load-stride: load genome 1 plus every Nth archive counter; 1 = the
 * whole archive loads (default). Lives here because load_genomes filters
 * with it. */
int genome_load_stride = 1;
char genome_dir[2 * MAX_PATH] = "winner_genomes"; /* --genome-dir: the archive the
                         * vector loads from (default: the original winner_genomes) */

/* forward: load_genomes reads the archive files before the INI section */
int ini_section_try_get_int(const char *ini_file, const char *section, const char *key, int *out);
double ini_section_try_get_double(const char *ini_file, const char *section, const char *key);

bool genome_order_less(const genome &a, const genome &b)
{
	if (a.shape_n != b.shape_n)
		return a.shape_n < b.shape_n;
	if (a.shape_d != b.shape_d)
		return a.shape_d < b.shape_d;
	return a.counter < b.counter;
}

void load_genomes(void)
{
	char search[3 * MAX_PATH];
	snprintf(search, sizeof(search), "%s\\*.ini", genome_dir);
	WIN32_FIND_DATAA fd;
	HANDLE find = FindFirstFileA(search, &fd);
	if (find == INVALID_HANDLE_VALUE)
	{
		printf("genomes: %s not found -- the genome vector stays empty\n", search);
		return;
	}
	int load_failures = 0;
	do
	{
		char path[3 * MAX_PATH];
		snprintf(path, sizeof(path), "%s\\%s", genome_dir, fd.cFileName);
		/* File-name identity: <shape>_<counter>.ini -- split at the LAST
		 * underscore so a shape token containing none is still unambiguous. */
		std::string file_name = fd.cFileName;
		std::string stem = file_name.substr(0, file_name.size() - 4); /* strip ".ini" */
		size_t underscore = stem.rfind('_');
		if (underscore == std::string::npos || underscore == 0)
		{
			printf("genomes: skipping '%s' -- the name does not match <shape>_<counter>.ini\n", fd.cFileName);
			continue;
		}
		std::string shape_text = stem.substr(0, underscore);
		std::string counter_text = stem.substr(underscore + 1);
		size_t cross = shape_text.find('x');
		if (cross == std::string::npos || counter_text.find_first_not_of("0123456789") != std::string::npos)
		{
			printf("genomes: skipping '%s' -- shape/counter unparsable\n", fd.cFileName);
			continue;
		}
		genome g;
		g.was_modified = 0;
		g.path = path;
		g.section = shape_text;
		g.shape_n = atoi(shape_text.substr(0, cross).c_str());
		g.shape_d = atoi(shape_text.substr(cross + 1).c_str());
		g.counter = atoi(counter_text.c_str());
		/* Stride filter first, before any INI read: with stride 1 the whole
		 * archive loads; with stride N, genome 1 plus every Nth counter. */
		if (!(g.counter == 1 || (g.counter % genome_load_stride) == 0))
			continue;
		int *param_fields[DISPATCH_PARAM_COUNT] = {
			&g.num_chunks, &g.batch_small, &g.batch_medium, &g.batch_large,
			&g.batch_huge, &g.batch_med_thresh, &g.batch_large_thresh, &g.batch_huge_thresh,
			&g.chunk_multiplier, &g.min_chunk_cols, &g.caller_batch_small, &g.caller_batch_medium,
			&g.caller_batch_large, &g.caller_batch_huge, &g.caller_batch_med_thresh, &g.caller_batch_large_thresh,
			&g.caller_batch_huge_thresh, &g.chunk_split_pct, &g.chunk_group_1_count, &g.chunk_group_2_count,
			&g.taper_shift, &g.quantize_chunk_groups
		};
		int missing = 0;
		for (int param_index = 0; param_index < DISPATCH_PARAM_COUNT; param_index++)
		{
			if (!ini_section_try_get_int(path, g.section.c_str(), dispatch_param_names[param_index], param_fields[param_index]))
			{
				printf("genomes: '%s' has no key '%s' -- incomplete archive entry\n",
				       fd.cFileName, dispatch_param_names[param_index]);
				missing = 1;
			}
		}
		for (int slot = 0; slot < PF_SLOTS; slot++)
		{
			char keyname[32];
			snprintf(keyname, sizeof(keyname), "pf_dist_%d", slot);
			if (!ini_section_try_get_int(path, g.section.c_str(), keyname, &g.pf_dist[slot]))
				g.pf_dist[slot] = 0; /* unowned slots are absent from the archive by design */
			snprintf(keyname, sizeof(keyname), "pf_type_%d", slot);
			if (!ini_section_try_get_int(path, g.section.c_str(), keyname, &g.pf_type[slot]))
				g.pf_type[slot] = 0;
		}
		if (missing)
		{
			load_failures++;
			continue; /* an incomplete genome never enters the vector */
		}
		g.best_gbs = ini_section_try_get_double(path, g.section.c_str(), "best_gbs");
		genomes.push_back(g);
	} while (FindNextFileA(find, &fd));
	FindClose(find);
	std::sort(genomes.begin(), genomes.end(), genome_order_less);
	printf("genomes: loaded %d genome(s) from %s (load stride %d: %s; %d skipped as incomplete)\n",
	       (int)genomes.size(), genome_dir, genome_load_stride,
	       genome_load_stride == 1 ? "whole archive" : "counter 1 + every Nth",
	       load_failures);
}

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

long experiment_comparisons; /* resolved confident compares so far */
long experiment_measurements; /* individual genome measurements so far (2 per round) */
double sort_runtime_minutes; /* --runtime: total cascade budget in minutes, divided
                         * equally among the stages; 0 = unlimited */
double sex_influence_percent = 0.0; /* --sex-influence: the MAXIMUM percentage of parameter
                         * slots a second parent donates to a mutated genome (0 = asexual) */
int top_percent = 10;        /* --top-percent: per-stage survival fraction (default 10) */
int sort_stages = 2;         /* --stages: number of cascade stages (default 2) */
int target_pool_size = 0;    /* --target-pool-size: when set, one stage runs and the
                         * pool shrinks/extends linearly toward this size over the
                         * runtime; the phase cuts are bypassed */
int random_probe_count = 0;  /* --random-probes: start the pool with N fully random
                         * probe genomes instead of loading an archive */
char random_probe_shape[64] = ""; /* --shape NxM: the probe draw shape (required
                         * with --random-probes: no archive exists to inherit one) */
int pf_dist_lo;              /* legal distance window: -(weight-matrix bytes / 2) */
int pf_dist_hi;              /* legal distance window: +(weight-matrix bytes / 2) */
int mutate_percentage = 0;   /* --mutate-percentage: top genomes copied+mutated into the
                         * tail at every sweep restart (default 0 = no mutation) */
char genome_output_dir[2 * MAX_PATH] = ""; /* --genome-output-dir: when set, the final
                         * reduced population is written as <dir>\<shape>_<n>.ini files */
int g_counter_write;         /* the file counter for the population writer */
char sorter_stdout_path[2 * MAX_PATH] = ""; /* --sorter-stdout-file: when set,
                         * stdout tees into this file (full detail, plus the
                         * stderr stage heartbeat on the console) */
FILE *sorter_stdout_capture; /* the tee target, 0 = disabled */
double compare_min_difference = COMPARE_MIN_DIFF_DEFAULT; /* --min-diff: stage-1 accumulated-time
                         * difference (seconds) that resolves a compare; stage 2 runs at
                         * ten times this value; the default lives here because parse_cli
                         * only overwrites it when the flag is given */
double compare_max_seconds = COMPARE_MAX_SECONDS_DEFAULT; /* --compare-max: upper bound on one
                         * compare's runtime; at the cap the compare DECIDES by whichever
                         * genome accumulated less time -- the accumulated times are the
                         * evidence */

/* ======================================================================
 * Timing (QPC-based)
 * ====================================================================== */

#include "qpc_seconds.h"

/* ======================================================================
 * Tensor alloc/fill (benchmark filler data)
 * Random draws use the x86 RDRAND instruction, no seeding.
 * ====================================================================== */

unsigned rng_u32(void)
{
	for (int tries = 0; tries < 10; tries++)
	{
		unsigned value = 0;
		if (_rdrand32_step(&value))
			return value;
	}
	printf("[FATAL] rng_u32: RDRAND failed 10 consecutive draws.\n");
	exit(EXIT_FAILURE);
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
 * RDRAND words give a full 64-bit draw; the modulo bias is negligible. */
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
 * a uniform rand(0..3):
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
 * 1..10 mutations on DISTINCT fields (a repeated field pick redraws), then
 * the legality pass every generated genome goes through. Every active
 * dispatch param, every OWNED pf distance and every OWNED pf type carries
 * equal weight in the field draw, each mutation is a uniform redraw over
 * the field's legal window (distances through mutate_distance's four kinds),
 * the threshold triples are re-sorted and the result sanitized. The mutant
 * carries NO speed record of its own (shape_a_gbs/best_gbs reset to 0 --
 * it has not been measured yet). Transplanted from autotune_dispatch.c's
 * hill-climb round construction; measuring and acceptance are NOT part of
 * this -- the sorter's experiment wires those.
 *
 * Sexual crossover draws BOTH parents as frand()^2 through the CURRENT
 * RANKING (order[0..donor_count)): parent 1 (the mutation baseline) and
 * parent 2 (the crossover donor) each land front-biased toward the fast
 * genomes, with every ranked genome reachable as a parent. The donor
 * donates a random
 * number of parameter slots to the child -- same slot to same slot,
 * straight transfer, double transfers of one slot stay possible and
 * harmless. The donated count is a uniform frand() * (total slots *
 * sex_influence/100): at 100% the child can receive any fraction of ALL
 * slots, at 0 nothing is transferred (assexual). The legality pass runs
 * again over the mixed genome so threshold triples that crossed parents
 * stay strictly ascending. */
genome get_mutated_genome(const genome &baseline, const int *order, int donor_count)
{
	genome trial = baseline;
	trial.shape_a_gbs = 0; /* the mutant has no measurement of its own yet */
	trial.best_gbs = 0;
	int active_params[DISPATCH_PARAM_COUNT];
	int num_active_params = 0;
	for (int param_index = 0; param_index < DISPATCH_PARAM_COUNT; param_index++)
		active_params[num_active_params++] = param_index;
	int mutations = rng_int(1, 10); /* fields changed this trial */
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

	/* Sexual crossover: a second parent is drawn through the CURRENT RANKING
	 * (order[0..donor_count), the front region the sort has established --
	 * NOT the pool's archive order, where the lowest counters would dominate
	 * the draw regardless of speed). f*f biases the donor toward the ranking
	 * front where the fast genomes collect. The donor donates a random
	 * number of parameter slots to the child -- same slot to same slot,
	 * straight transfer, double transfers of one slot stay possible and
	 * harmless. The donated count is a uniform frand() * (total slots *
	 * sex_influence/100): at 100% the child can receive any fraction of ALL
	 * slots, at 0 nothing is transferred (assexual). The legality pass runs
	 * again over the mixed genome so threshold triples that crossed parents
	 * stay strictly ascending. */
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
		const genome &donor = genomes[order[pick]];
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
 * INI read (the genome archive files)
 * ====================================================================== */

/* Read one int key inside an INI section via the Windows INI API.
 * GetPrivateProfileIntA cannot distinguish "absent" from "invalid", so the
 * key is read with GetPrivateProfileStringA using a sentinel default: the
 * buffer equals the sentinel iff the key is absent. A missing key is not an
 * error here -- the caller decides what absence means (dispatch params are
 * required; pf slots of other kernels legitimately are absent). A key that
 * IS present but malformed stays a FATAL: a typo must not become a
 * DIFFERENT config. */
int ini_section_try_get_int(const char *ini_file, const char *section, const char *key, int *out)
{
	/* 512 bytes: archive INI lines are short but the pool stays uniform. */
	char buf[512];
	char absent[] = "__v7_ini_key_absent__";
	DWORD n = GetPrivateProfileStringA(section, key, absent, buf, sizeof(buf), ini_file);
	if (n == strlen(absent) && strcmp(buf, absent) == 0)
		return 0;
	/* Strip a trailing inline comment (';' or '#') and surrounding
	 * whitespace so a strict strtol parse accepts the '752 ; comment' form. */
	char *cut = strpbrk(buf, ";#");
	if (cut)
		*cut = '\0';
	char *end_ptr = buf + strlen(buf);
	while (end_ptr > buf && (end_ptr[-1] == ' ' || end_ptr[-1] == '\t'))
		*--end_ptr = '\0';
	char *parse_end = 0;
	errno = 0;
	long value = strtol(buf, &parse_end, 10);
	if (parse_end == buf || *parse_end != '\0' || errno == ERANGE || value < INT_MIN || value > INT_MAX)
	{
		printf("[FATAL] ini_section_try_get_int: key '%s' in [%s] (%s) has "
		       "malformed or out-of-range value '%s'.\n",
		       key, section, ini_file, buf);
		exit(EXIT_FAILURE);
	}
	*out = (int)value;
	return 1;
}

/* Read one double key (the best_gbs record). Absent -> 0. */
double ini_section_try_get_double(const char *ini_file, const char *section, const char *key)
{
	char buf[512];
	char absent[] = "__v7_ini_key_absent__";
	GetPrivateProfileStringA(section, key, absent, buf, sizeof(buf), ini_file);
	if (strcmp(buf, absent) == 0)
		return 0.0;
	char *cut = strpbrk(buf, ";#");
	if (cut)
		*cut = '\0';
	return strtod(buf, 0);
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
 * No console output: this runs per-compare round and the sort's own prints
 * carry the flow. */
void apply_genome_config(int genome_index)
{
	const genome &g = genomes[genome_index];
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
 * Random probe draw (transplanted from autotune_dispatch.c's
 * draw_random_snapshot -- the probe phase's genome generator)
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
 * Threshold triples are sorted so the draw is a legal config. Transplanted
 * from autotune_dispatch.c's draw_random_snapshot (probe-phase generator);
 * the probe phase's paired measurement is the sorter's experiment instead. */
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

/* Compare two genomes by ACCUMULATED time difference. The interleaved
 * phase alternates the two genomes' matmul dispatches, accumulating each
 * genome's matmul time (the config apply between the two dispatches sits
 * OUTSIDE both accumulators, and the order is FIXED -- A first, then B --
 * so the fresh-apply treatment is symmetric between them). The compare
 * ALWAYS returns a winner -- it is never undecided:
 *   - the moment |acc_a - acc_b| exceeds the stage's accumulated-time
 *     requirement, or
 *   - at the per-compare cap (--compare-max), or
 *   - at the stage deadline,
 * whichever comes first: after that much measuring the accumulated times
 * are fairly certain even when the difference is small, so the decision is
 * simply whichever genome accumulated LESS time. Both rates are recorded
 * for the sort's bookkeeping. */

/* Register a genome in the pool: always APPEND, never compare and never
 * dedup -- every mutant is a new vector entry and the pool grows with the
 * experiment. */
int buy_genome(const genome &g)
{
	genomes.push_back(g);
	return (int)genomes.size() - 1;
}

int confident_compare(int a_index, int b_index, double deadline, double stage_min_diff)
{
	apply_genome_config(a_index);
	for (int warm = 0; warm < WARMUP_ROUNDS; warm++)
		dispatch_one();
	apply_genome_config(b_index);
	for (int warm = 0; warm < WARMUP_ROUNDS; warm++)
		dispatch_one();

	double acc_a = 0;
	double acc_b = 0;
	int rounds = 0;
	double measure_start = get_seconds();
	for (;;)
	{
		double stamp_a;
		double stamp_b;

		apply_genome_config(a_index); /* untimed */
		stamp_a = get_seconds();
		dispatch_one();
		stamp_b = get_seconds();
		acc_a += stamp_b - stamp_a;

		apply_genome_config(b_index); /* untimed */
		stamp_a = get_seconds();
		dispatch_one();
		stamp_b = get_seconds();
		acc_b += stamp_b - stamp_a;

		rounds++;
		experiment_measurements += 2;
		if (fabs(acc_a - acc_b) > stage_min_diff)
			break;
		double now = get_seconds();
		if (now >= deadline)
			break; /* stage deadline: decide by what has accumulated */
		if (now - measure_start >= compare_max_seconds)
			break; /* compare cap: decide by what has accumulated */
	}

	experiment_comparisons++;
	genomes[a_index].shape_a_gbs = (double)rounds * shape_bytes_per_call / acc_a / 1e9;
	genomes[b_index].shape_a_gbs = (double)rounds * shape_bytes_per_call / acc_b / 1e9;
	if (acc_a < acc_b)
		return a_index;
	return b_index;
}

/* Challenge a pool genome with one trial genome: the trial is appended to
 * the pool (the compare interface works on pool indices), measured against
 * the baseline with the sorter's STANDARD accumulated-difference compare --
 * the same growing-value logic every genome speed comparison uses -- and
 * popped back off when it loses. Returns the trial's pool index when it
 * WON (it stays in the pool), -1 when it lost (the pool is unchanged). */
int challenge_baseline(int baseline_index, const genome &trial_genome)
{
	genome copy = trial_genome;
	int trial_index = buy_genome(copy);
	int winner = confident_compare(baseline_index, trial_index, 1.0e308, compare_min_difference);
	if (winner == trial_index)
		return trial_index;
	genomes.pop_back(); /* the loser is discarded */
	return -1;
}

/* ======================================================================
 * Probe phase (the autotuner's probe-phase protocol, sorter edition)
 * ====================================================================== */

/* Fill the pool through ACCEPTANCE-FILTERED random search: the first random
 * probe becomes the baseline and gets a 20-round self-warmup; every further
 * random probe is measured against the CURRENT baseline via
 * challenge_baseline (the standard confident compare) -- a winner joins the
 * pool AND becomes the new baseline, a loser is discarded. The baseline
 * only improves, so accepts get rarer and the attempt count is unbounded;
 * every compare is bounded by the per-compare cap. Runs until probe_count
 * probes have been ADDED. */
void probe_phase(int probe_count)
{
	char shape_text[32];
	snprintf(shape_text, sizeof(shape_text), "%dx%d", dispatch_n, dispatch_d);

	genome first;
	first.was_modified = 0;
	first.shape_n = dispatch_n;
	first.shape_d = dispatch_d;
	first.counter = 1;
	first.path = "(random probe)";
	first.section = shape_text;
	first.best_gbs = 0;
	first.shape_a_gbs = 0;
	draw_random_probe_genome(&first);
	int baseline_index = buy_genome(first);

	/* 20-round self-warmup: settle the machine on the baseline's config. */
	apply_genome_config(baseline_index);
	for (int warm = 0; warm < 20; warm++)
		dispatch_one();

	long long attempts = 0;
	int added = 1;
	while (added < probe_count)
	{
		attempts++;
		genome trial;
		trial.was_modified = 0;
		trial.shape_n = dispatch_n;
		trial.shape_d = dispatch_d;
		trial.counter = (int)(attempts + 1);
		trial.path = "(random probe)";
		trial.section = shape_text;
		trial.best_gbs = 0;
		trial.shape_a_gbs = 0;
		draw_random_probe_genome(&trial);
		int winner_index = challenge_baseline(baseline_index, trial);
		if (winner_index >= 0)
		{
			baseline_index = winner_index; /* the winner is the new baseline */
			added++;
			sorter_out("probe %d accepted after %lld attempts (pool %d)\n",
			       added, attempts, (int)genomes.size());
		}
	}
	sorter_out("probe phase done: %d probes added after %lld attempts\n", added, attempts);
}

/* ======================================================================
 * Sort experiment
 * ====================================================================== */

void print_order(const int *order, int order_count)
{
	sorter_out("   ");
	for (int slot = 0; slot < order_count; slot++)
		sorter_out(" %d", genomes[order[slot]].counter);
	sorter_out("\n");
	fflush(stdout);
}

/* Print the current top-X (default 13) of the order. Pure console/file
 * output, never touches the genome pool. */
void print_top_genomes(int *order, int order_count, int top_x)
{
	int show = top_x > 0 ? top_x : 13;
	if (show > order_count)
		show = order_count;
	sorter_out("top %d of %d (fastest first):", show, order_count);
	for (int slot = 0; slot < show; slot++)
		sorter_out(" %d", genomes[order[slot]].counter);
	sorter_out("\n");
	fflush(stdout);
}

/* Write the WHOLE genome pool (the population the last stage worked on --
 * possibly reduced by the phase cuts) as INI files. This runs ONCE at the
 * end of the sort and ONLY when --genome-output-dir was given: during the
 * run the genomes live in RAM, the output directory is the single reason
 * any INI gets written. Every pool genome becomes
 * <dir>\<shape>_<file counter>.ini in the archive format the measurement
 * script and later sort runs read back. Plain stdio writes one whole file
 * per genome -- the Windows profile API re-reads and rewrites the whole
 * file PER KEY, which made a 762-genome pool take minutes under real-time
 * antivirus scanning (the apparent hang after the last heartbeat). */
void write_final_population(int *order, int order_count)
{
	if (!genome_output_dir[0])
		return; /* no output directory given: no INI files, the pool stays in RAM */

	fprintf(stderr, "\rwriting %d genomes ...        \n", order_count);
	print_top_genomes(order, order_count, 13);
	CreateDirectoryA(genome_output_dir, NULL); /* exists -> ERROR_ALREADY_EXISTS, ignored */
	int written_count = 0;
	for (int slot = 0; slot < order_count; slot++)
	{
		genome *g = &genomes[order[slot]];
		char path[3 * MAX_PATH];
		snprintf(path, sizeof(path), "%s\\%s_%d.ini", genome_output_dir, g->section.c_str(), ++g_counter_write);
		FILE *ini = fopen(path, "w");
		if (!ini)
		{
			printf("(warning: cannot open %s for writing)\n", path);
			continue;
		}
		fprintf(ini, "[%s]\n", g->section.c_str());
		for (int param_index = 0; param_index < DISPATCH_PARAM_COUNT; param_index++)
			fprintf(ini, "%s=%d\n", dispatch_param_names[param_index], g->*genome_param_ptr[param_index]);
		for (int owned = 0; owned < PF_SLOTS; owned++)
		{
			if (!pf_owned[owned])
				continue;
			fprintf(ini, "pf_dist_%d=%d\n", owned, g->pf_dist[owned]);
			fprintf(ini, "pf_type_%d=%d\n", owned, g->pf_type[owned]);
		}
		fprintf(ini, "best_gbs=%.1f\n", g->shape_a_gbs);
		fclose(ini);
		g->was_modified = false;
		written_count++;
	}
	sorter_out("population written: %d genome file(s) to %s\n", written_count, genome_output_dir);
}

/* One sort stage: rotating probabilistic pair exchanges. Index 1 rotates
 * over every genome, sweep after sweep (1, 2, ..., count, 1, 2, ...), so
 * every genome gets equal attention. Index 2 is drawn per step as
 * f = frand(); f = f*f; index2 = f*f*index1 -- a partner BELOW the rotating
 * position, biased toward the front, where the fast genomes collect. The
 * confident compare decides, and the pair exchanges so the FASTER genome
 * ends at the LOWER index: the invariant stays fastest first. The stage
 * reruns sweeps until its deadline. With --mutate-percentage P > 0,
 * reruns sweeps until its deadline. With --mutate-percentage P > 0, at
 * every sweep end the tail is refreshed with fresh mutants of f*f-drawn
 * parents. With --target-pool-size T > 0 the pool itself is under TIME
 * CONTROL: at every sweep end (before the mutant tail is written in) the
 * pool moves linearly from its starting size toward T, proportional to the
 * elapsed budget -- shrinking culls the worst-ranked genomes, extension
 * fills the new slots with mutated children. The mutant tail region
 * recomputes from the CURRENT pool size every sweep. */
long long run_sort_stage(int *order, int order_count, double stage_deadline, double stage_min_diff, int stage_number, int stage_count, int *final_pool_size)
{
	long long pair_steps = 0;
	long long sweeps_done = 0; /* completed sweeps (restarts fired) */
	int sweep_position = 0;    /* rotating position inside the current sweep */
	double stage_start = get_seconds();
	double stage_duration = stage_deadline - stage_start;
	int next_decile = 1; /* the array prints when the stage passes 10%, 20%, ..., 100% */
	int start_population = order_count; /* the pool time control interpolates from here */
	int unlimited_stage = stage_duration > 1.0e6;

	while (get_seconds() < stage_deadline)
	{
		/* ---- Sweep end: pool time control, then the mutant tail ---- */
		if (sweep_position >= order_count)
		{
			sweep_position = 0;
			sweeps_done++;

			/* Pool time control: linear move toward --target-pool-size,
			 * proportional to the elapsed budget. Culling removes the
			 * worst-ranked genomes (the ranking tail); extension fills the
			 * new slots with mutated children of f*f-drawn parents. Runs
			 * BEFORE the mutant tail write-in, exactly once per sweep. */
			if (target_pool_size > 0 && !unlimited_stage)
			{
				double budget_used = (get_seconds() - stage_start) / stage_duration;
				if (budget_used < 0.0)
					budget_used = 0.0;
				if (budget_used > 1.0)
					budget_used = 1.0;
				int desired = start_population + (int)((double)(target_pool_size - start_population) * budget_used);
				if (desired < 2)
					desired = 2;
				if (desired < order_count)
				{
					sorter_out("sweep %lld: pool culled %d -> %d (%.0f%% of budget used)\n",
					       sweeps_done, order_count, desired, budget_used * 100.0);
					order_count = desired;
				}
				else if (desired > order_count)
				{
					for (int slot = order_count; slot < desired; slot++)
					{
						float f = frand();
						f = f * f;
						int parent_pick = (int)((double)order_count * f);
						genome mutant = get_mutated_genome(genomes[order[parent_pick]], order, order_count);
						mutant.was_modified = 1;
						order[slot] = buy_genome(mutant);
					}
					sorter_out("sweep %lld: pool extended %d -> %d with mutated children (%.0f%% of budget used)\n",
					       sweeps_done, order_count, desired, budget_used * 100.0);
					order_count = desired;
				}
			}

			/* Refresh the mutant tail from the current ranking: BOTH parents
			 * are frand()^2 draws through the whole ranking -- parent 1 (the
			 * mutation baseline) and parent 2 (the crossover donor) each land
			 * front-biased -- concentrated near the fast genomes -- while
			 * every ranked genome stays reachable as a parent. No INI writes
			 * here -- the genomes stay in RAM for the whole sort; the output
			 * directory (if given) receives the whole pool ONCE at the end. */
			if (mutate_percentage > 0)
			{
				int mutate_count = order_count * mutate_percentage / 100;
				if (mutate_count > order_count / 2)
					mutate_count = order_count / 2; /* the copied tail must not eat the population */
				if (mutate_count < 1)
					mutate_count = 1;
				for (int m = 0; m < mutate_count; m++)
				{
					int tail_slot = order_count - mutate_count + m;
					float f = frand();
					f = f * f;
					int parent_pick = (int)((double)order_count * f);
					genome mutant = get_mutated_genome(genomes[order[parent_pick]], order, order_count);
					mutant.was_modified = 1;
					order[tail_slot] = buy_genome(mutant);
				}
				sorter_out("sweep %lld: %d fresh mutants in the tail (pool now %d)\n",
				       sweeps_done, mutate_count, order_count);
			}
		}

		int slot1 = sweep_position;
		float f = frand();
		f = f * f;
		int slot2 = (int)((double)slot1 * f); /* partner below the current rotating position */
		sweep_position++;
		if (slot2 >= slot1)
			continue; /* the draw hit the rotating slot itself (slot1 == 0) */
		int winner = confident_compare(order[slot1], order[slot2], stage_deadline, stage_min_diff);
		/* faster must sit at the lower index */
		int faster_slot;
		int slower_slot;
		if (winner == order[slot1])
		{
			faster_slot = slot1;
			slower_slot = slot2;
		}
		else
		{
			faster_slot = slot2;
			slower_slot = slot1;
		}
		if (faster_slot > slower_slot)
		{
			int tmp = order[slot1];
			order[slot1] = order[slot2];
			order[slot2] = tmp;
		}
		pair_steps++;
		/* Decile progress prints: the array as it stands at 10%, 20%, ...,
		 * 100% of the stage's time budget. Unlimited stages (no real
		 * duration) skip them and only print at the stage's end. */
		if (stage_duration > 1.0e6)
			continue;
		double progress = (get_seconds() - stage_start) / stage_duration;
		while (next_decile <= 10 && progress * 10.0 >= (double)next_decile)
		{
			sorter_out("array at %d0%% of the stage (%d comparisons so far):\n",
			           next_decile, experiment_comparisons);
			print_order(order, order_count);
			stage_heartbeat(next_decile * 10, stage_number, stage_count);
			next_decile++;
		}
	}
	sorter_out("order at the stage's last iteration (%lld pair steps, %lld sweeps):\n",
	       pair_steps, sweeps_done);
	print_order(order, order_count);
	*final_pool_size = order_count; /* the pool time control may have shrunk/extended it */
	return sweeps_done; /* the completed sweeps over the pool */
}

/* The sort experiment: an N-stage cascade. Stage 1 sorts every loaded
 * genome; between stages the population is cut to --top-percent percent of
 * the previous stage and the accumulated-time requirement grows tenfold.
 * The --runtime budget divides equally among the stages. Example: 100
 * genomes, 3 stages, 50% -> the stages sort 100, 50 and 25 genomes. */
void experiment_sort(void)
{
	/* The probe phase fills the pool first when --random-probes is given:
	 * acceptance-filtered random search (winners only join), no archive
	 * involved. */
	if (random_probe_count > 0)
		probe_phase(random_probe_count);

	int order_count = (int)genomes.size();
	if (order_count < 2)
	{
		printf("sort: needs at least 2 genomes, got %d\n", order_count);
		return;
	}
	for (int slot = 0; slot < order_count; slot++)
	{
		if (genomes[slot].shape_n != dispatch_n || genomes[slot].shape_d != dispatch_d)
		{
			printf("[FATAL] experiment_sort: genome %s is %dx%d but the experiment runs %dx%d -- "
			       "mixed-shape archives are not ordered by one shape's measurement.\n",
			       genomes[slot].path.c_str(), genomes[slot].shape_n, genomes[slot].shape_d,
			       dispatch_n, dispatch_d);
			v7_unload(&inference_dll_ptr);
			exit(EXIT_FAILURE);
		}
	}

	/* The pool time control can EXTEND the pool beyond its entry size (up to
	 * --target-pool-size): the order array must hold the maximum, or the
	 * extension writes past the allocation and corrupts the heap. */
	int order_capacity = order_count;
	if (target_pool_size > order_capacity)
		order_capacity = target_pool_size;
	int *order = (int *)malloc(order_capacity * sizeof(int));
	if (!order)
	{
		printf("[FATAL] experiment_sort: out of memory for the order array\n");
		v7_unload(&inference_dll_ptr);
		exit(EXIT_FAILURE);
	}
	/* Start in archive order (ascending counter): the sort reorders this. */
	for (int slot = 0; slot < order_count; slot++)
		order[slot] = slot;

	/* With --target-pool-size the sort runs as ONE stage (the pool time
 * control replaces the phase cuts); the phase machinery stays available
 * for runs without the option. */
	int stage_count = sort_stages;
	if (target_pool_size > 0)
		stage_count = 1;
	double stage_minutes = sort_runtime_minutes / (double)stage_count;
	double stage_min_diff = compare_min_difference;
	int population = order_count;
	long long total_sweeps = 0; /* sweeps completed over the pool, all stages */
	long long cascade_comparisons_start = experiment_comparisons;
	long cascade_measurements_start = experiment_measurements;

	for (int stage = 1; stage <= stage_count; stage++)
	{
		sorter_out("sort: stage %d of %d: %d genomes, min-diff %.5f s, budget %.3f minute(s)%s%s\n",
		       stage, stage_count, population, stage_min_diff,
		       stage_minutes, stage_minutes > 0.0 ? "" : " (0 = unlimited)",
		       target_pool_size > 0 ? ", pool time control toward target" : "");
		double stage_deadline = get_seconds() + stage_minutes * 60.0;
		if (stage_minutes <= 0.0)
			stage_deadline = 1.0e308; /* unlimited: the deadline never fires */

		long stage_comparisons_start = experiment_comparisons;
		long stage_measurements_start = experiment_measurements;
		int stage_pool_size = population; /* the stage reports its final pool size back
		 * (the --target-pool-size control may have shrunk/extended it) */
		total_sweeps += run_sort_stage(order, population, stage_deadline, stage_min_diff, stage, stage_count, &stage_pool_size);
		population = stage_pool_size;
		sorter_out("stage %d done: %d comparisons, %d measurements\n",
		       stage,
		       (int)(experiment_comparisons - stage_comparisons_start),
		       (int)(experiment_measurements - stage_measurements_start));

		if (stage < stage_count)
		{
			/* Stage cut: keep the top top_percent% of this stage's order and
			 * re-sort the survivors at ten times the accumulated-time
			 * requirement -- the survivors' internal order was decided with
			 * weak evidence, the cut concentrates the remaining budget on
			 * them. */
			int keep = population * top_percent / 100;
			if (keep < 2)
				keep = 2;
			if (keep > population)
				keep = population;
			population = keep;
			stage_min_diff *= 10.0;
			sorter_out("sort: cut to top %d%%: %d genomes advance to stage %d (min-diff %.5f s)\n",
			       top_percent, population, stage + 1, stage_min_diff);
		}
	}

	printf("FINAL top-%d array after %d stages (%d total comparisons, %d measurements; fastest first):\n",
	       population, stage_count,
	       (int)(experiment_comparisons - cascade_comparisons_start),
	       (int)(experiment_measurements - cascade_measurements_start));
	print_order(order, population);
	fprintf(stderr, "\rsort finished: %lld sweeps over the genome pool\n", total_sweeps);

	/* Write the WHOLE final population (the last stage's reduced pool, which
	 * includes every mutant created during the run) when an output dir is
	 * given: this is the only INI writing of the whole sort. */
	write_final_population(order, population);
	sorter_out("sort done\n");
	free(order);
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
		else if (strcmp(argv[i], "--min-diff") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			double value = strtod(argv[++i], &end);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value <= 0.0)
			{
				printf("[FATAL] --min-diff: malformed or non-positive seconds '%s'\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			compare_min_difference = value;
		}
		else if (strcmp(argv[i], "--runtime") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			double value = strtod(argv[++i], &end);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value < 0.0)
			{
				printf("[FATAL] --runtime: malformed or negative minutes '%s'\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			sort_runtime_minutes = value;
		}
		else if (strcmp(argv[i], "--random-probes") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			long value = strtol(argv[++i], &end, 10);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value < 1)
			{
				printf("[FATAL] --random-probes: malformed or too-small count '%s' (>= 1)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			random_probe_count = (int)value;
		}
		else if (strcmp(argv[i], "--shape") == 0 && i + 1 < argc)
		{
			snprintf(random_probe_shape, sizeof(random_probe_shape), "%s", argv[++i]);
		}
		else if (strcmp(argv[i], "--target-pool-size") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			long value = strtol(argv[++i], &end, 10);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value < 2)
			{
				printf("[FATAL] --target-pool-size: malformed or too-small size '%s' (>= 2)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			target_pool_size = (int)value;
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
		else if (strcmp(argv[i], "--genome-output-dir") == 0 && i + 1 < argc)
		{
			/* Normalize to absolute: the ANSI profile APIs resolve relative
			 * paths against the Windows directory, not the process CWD. */
			char abs[2 * MAX_PATH];
			DWORD dir_len = GetFullPathNameA(argv[++i], (DWORD)sizeof(abs), abs, 0);
			if (dir_len == 0 || dir_len >= sizeof(abs))
			{
				printf("[FATAL] --genome-output-dir: path invalid or too long: %s\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			snprintf(genome_output_dir, sizeof(genome_output_dir), "%s", abs);
		}
		else if (strcmp(argv[i], "--genome-dir") == 0 && i + 1 < argc)
		{
			/* Normalize to absolute: the ANSI profile APIs resolve relative
			 * paths against the Windows directory, not the process CWD. */
			char abs[2 * MAX_PATH];
			DWORD dir_len = GetFullPathNameA(argv[++i], (DWORD)sizeof(abs), abs, 0);
			if (dir_len == 0 || dir_len >= sizeof(abs))
			{
				printf("[FATAL] --genome-dir: path invalid or too long: %s\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			snprintf(genome_dir, sizeof(genome_dir), "%s", abs);
		}
		else if (strcmp(argv[i], "--load-stride") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			long value = strtol(argv[++i], &end, 10);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value < 1)
			{
				printf("[FATAL] --load-stride: malformed or non-positive stride '%s' (>= 1)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			genome_load_stride = (int)value;
		}
		else if (strcmp(argv[i], "--compare-max") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			double value = strtod(argv[++i], &end);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value <= 0.0)
			{
				printf("[FATAL] --compare-max: malformed or non-positive seconds '%s'\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			compare_max_seconds = value;
		}
		else if (strcmp(argv[i], "--stages") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			long value = strtol(argv[++i], &end, 10);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value < 1 || value > 20)
			{
				printf("[FATAL] --stages: malformed or out-of-range stage count '%s' (1..20)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			sort_stages = (int)value;
		}
		else if (strcmp(argv[i], "--top-percent") == 0 && i + 1 < argc)
		{
			char *end = 0;
			errno = 0;
			long value = strtol(argv[++i], &end, 10);
			if (end == argv[i] || (end && *end != '\0') || errno == ERANGE || value < 1 || value > 99)
			{
				printf("[FATAL] --top-percent: malformed or out-of-range percent '%s' (1..99)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			top_percent = (int)value;
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
	return 0;
}

/* stdout-to-file capture: when --sorter-stdout-file is given, every sort
 * print goes to the FILE ONLY (full detail, flushed per line) and the
 * console carries just the stderr stage heartbeat -- a long run can be
 * watched without spam and read back complete afterwards. Without the
 * option every print goes to the console as before. */
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

/* Stage-heartbeat: the ONLY stderr output. One carriage-returned line per
 * stage-progress decile, so watching the sorter shows one cycling
 * percentage instead of a print flood. */
void stage_heartbeat(int percent, int stage, int stages)
{
	fprintf(stderr, "\rstage %d/%d sorting ... %3d%%", stage, stages, percent);
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
		printf("usage: %s <v7_threading.dll> --db <db_path> [--q8|--f32]\n", argv[0]);
		return 1;
	}
	char *dll_path = argv[1];
	char *db_path = 0;
	use_q8_mode = 1; /* default; --q8 / --f32 override */
	kernel_label = "q8";
	if (parse_cli(argc, argv, &db_path))
		return 1;

	/* stdout capture first: everything below (genomes, v7_init, the whole
	 * sort) tees into the capture file, the console shows only the stderr
	 * heartbeats. */
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
	 * --q8/--f32. (The draw ran before this block once and the pool carried
	 * uninitialized garbage pf distances -- the patcher happily applied them
	 * and the written INIs failed the window check.) */
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

	/* The genome pool: N fully random probes (no archive needed), or the
	 * stride-selected archive subset. */
	if (random_probe_count > 0)
	{
		/* The shape must be given explicitly: the probes are drawn before any
		 * archive exists to inherit one from. */
		int probe_rows;
		int probe_cols;
		if (parse_shape_token(random_probe_shape, &probe_rows, &probe_cols))
		{
			printf("[FATAL] --random-probes requires --shape <NxM> (the probes are drawn before any archive exists; got '%s')\n", random_probe_shape);
			v7_unload(&inference_dll_ptr);
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
		/* The probes themselves are drawn in the probe phase (inside
		 * experiment_sort): they are MEASURED against a running baseline
		 * there, after the engine is up -- only the shape and window are
		 * prepared here. */
		printf("random probes: shape %dx%d, %d probes to add via acceptance filtering\n",
		       dispatch_n, dispatch_d, random_probe_count);
	}
	else
	{
		load_genomes();
		if (genomes.empty())
		{
			printf("[FATAL] no genomes loaded -- nothing to sort.\n");
			v7_unload(&inference_dll_ptr);
			return 1;
		}
		/* The experiment dispatches ONE shape: the archive subset must be of a
		 * single shape or the measured ordering mixes incompatible kernels. */
		dispatch_n = genomes[0].shape_n;
		dispatch_d = genomes[0].shape_d;
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

	/* Kernel selection and its prefetch slot ownership (slot map: q8 0-7,
	 * f32 8-10 -- the only slots a genome pf apply touches). */
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

	printf("genome_sort_experiment: %dx%d %s\n", dispatch_n, dispatch_d, kernel_label);

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

	/* The sort experiment. */
	experiment_sort();

	/* Cleanup. */
	free(xout_buffer);
	if (use_q8_mode)
		tensor_q8_free(&g_tsq);
	else
		tensor_f32_free(&g_tsf);
	v7_unload(&inference_dll_ptr);
	return 0;
}