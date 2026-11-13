// v7 - part of the v7 inference engine
// Copyright (C) 2026 Gorg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/*
 * test_shape_speed.c -- benchmark one matmul shape under the INI-committed
 * dispatch config
 *
 * Loads the shape's [NxD] section from threading_proxy.ini and applies it to
 * the live dispatcher, then measures four rates interleaved, one QPC timing
 * per step:
 *
 *   per round:
 *     1. one dispatched matmul          -> shape_time_a
 *     2. 16KB-chunk copies small->large -> small_to_large
 *     3. one dispatched matmul          -> shape_time_b
 *     4. 16KB-chunk copies large->small -> large_to_small
 *     5. one dispatched matmul          -> shape_time_c
 *     6. dispatched memcpy kernel, CLI block onto itself -> memcpy_kernel
 *     7. 256K float sin steps on a carried state = one virtual MB -> float_sum
 *
 * 20 warmup rounds run first (identical round procedure, timings discarded);
 * measurement rounds then run until the runtime budget (seconds, double) has
 * passed, accumulating the four step times. The final report converts each
 * accumulator to GB/s: the matmul steps use the established per-call byte
 * count (n*d*5 = input + int8 weight block + output, the same accounting the
 * loadtest and the autotuner print), the copy loops use the walk's total
 * byte volume (megabytes * 1MB per loop).
 *
 * The INI keys are applied through the dispatcher's own surfaces so every
 * measured dispatch really executes them:
 *   - the 22 dispatch params are written into block 0 (the live param_set
 *     the dispatcher reads on the hot path) AND into the matched per-shape
 *     block (the shadow copy a shape-switch config reload would restore);
 *   - the pf_dist/pf_type values of the benchmarked kernel's slots are
 *     written the same way AND live-patched through the exported
 *     set_prefetch_distance/set_prefetch_type whenever they differ from the
 *     loaded values. This direct patch is mandatory, not cosmetic: the
 *     dispatcher's prefetch config only re-applies from the blocks on a
 *     shape switch, and this tool exercises a single shape for its whole
 *     run -- without the patch the loaded distances would never reach the
 *     SMC and the benchmark would measure stale prefetch code. Distance is
 *     patched first, then type: the distance patch restores a slot's backup
 *     bytes (re-enabling a disabled slot), while the type patch skips a
 *     disabled slot -- that ordering is what makes a type change on a
 *     previously disabled slot stick.
 *
 * Config strictness mirrors the dispatcher's own contracts: every dispatch
 * param and every owned pf key is REQUIRED (an absent key is a config gap,
 * not a default), an owned pf distance outside [+/- weight-matrix bytes / 2]
 * cannot be encoded in the patch slot's disp32 and is a fix-the-INI FATAL,
 * and a pf type outside 0..6 is a fix-the-INI FATAL.
 *
 * Usage:
 *   test_shape_speed.exe <dll> --db <db> --ini <ini> [options] <NxM> <megabytes> <runtime_seconds>
 *
 * Options:
 *   --db <path>    REQUIRED database path (no default)
 *   --ini <path>   REQUIRED threading_proxy.ini path (no default)
 *   --q8           Q8 matmul (default)
 *   --f32          F32 matmul
 */
#include <windows.h>
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <immintrin.h> /* _rdrand32_step for the benchmark filler data */
#include <errno.h>     /* strtol range reporting */
#include <limits.h>    /* INT_MIN/INT_MAX */
#include <math.h>      /* isfinite for the runtime parse */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================
 * Shape-block and prefetch-table layout (must match v7_data.inc)
 * ====================================================================== */

#define PF_SLOTS 41           /* pf_dist_0..40, pf_type_0..40 */
#define SB_PF_DIST_OFF 104    /* pf_dist[41] inside a 512-byte shape block */
#define SB_PF_TYPE_OFF 300    /* pf_type[41] */
#define PF_TYPE_MIN 0         /* 0=NTA 1=T0 2=T1 3=T2 4=PREFETCHW 5=PREFETCHWT1 6=CLDEMOTE */
#define PF_TYPE_MAX 6
#define SB_SIZE 512           /* bytes per shape block */
#define SB_PARAM_OFF 8        /* param_set dwords start here (after n,d header) */
#define SB_MAX_BLOCKS 256     /* block 0 = active, block 1 = documentation, 2+ = per-shape */
#define DISPATCH_PARAM_COUNT 22

#define MEMCPY_CHUNK_BYTES 16384 /* the 16KB chunk the copy loops step through */
#define WARMUP_ROUNDS 20
#define FLOAT_SUM_ITERATIONS 262144 /* 256K sinf steps per round = one virtual megabyte */

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

/* Dispatch param names in dispatch-index order, for apply messages and the
 * absence FATALs. */
static const char *dispatch_param_names[DISPATCH_PARAM_COUNT] = {
	"num_chunks",          "batch_small",           "batch_medium",           "batch_large",
	"batch_huge",          "batch_med_thresh",      "batch_large_thresh",     "batch_huge_thresh",
	"chunk_multiplier",    "min_chunk_cols",        "caller_batch_small",     "caller_batch_medium",
	"caller_batch_large",  "caller_batch_huge",     "caller_batch_med_thresh", "caller_batch_large_thresh",
	"caller_batch_huge_thresh", "chunk_split_pct",  "chunk_group_1_count",    "chunk_group_2_count",
	"taper_shift",         "quantize_chunk_groups"
};

static const char *pf_type_names[7] = { "NTA", "T0", "T1", "T2", "PREFETCHW", "PREFETCHWT1", "CLDEMOTE" };

/* ======================================================================
 * Benchmark tensor descriptors
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

/* ======================================================================
 * Benchmark session state
 * ====================================================================== */

struct v7_lib inference_dll_ptr;

tensor_q8 g_tsq;  /* Q8 benchmark tensors (use_q8_mode path) */
tensor_f32 g_tsf; /* F32 benchmark tensors */

int use_q8_mode;      /* 1 = Q8 dispatch, 0 = F32 dispatch */
const char *kernel_label;
int dispatch_n;       /* matmul rows */
int dispatch_d;       /* matmul cols */
int pf_dist_lo;       /* legal distance window: -(weight-matrix bytes / 2) */
int pf_dist_hi;       /* legal distance window: +(weight-matrix bytes / 2) */

unsigned char *shape_blocks_ptr;  /* base of the exported shape_blocks array */
unsigned char *matched_block_ptr; /* per-shape block for (n,d), block 1 excluded */

int pf_owned[PF_SLOTS];      /* 1 = this slot drives the benchmarked kernel */
int pf_owned_list[PF_SLOTS]; /* owned slot ids */
int pf_owned_count;

/* The config the INI commits for this shape: 22 dispatch params + pf values
 * for every slot (owned slots come from the INI, unowned keep the loaded
 * values so the apply never rewrites another kernel's tuned config). */
int loaded_params[DISPATCH_PARAM_COUNT];
int loaded_pf_dist[PF_SLOTS];
int loaded_pf_type[PF_SLOTS];

char *ini_path;
char section_name[32];

char *small_buffer;         /* 16KB, copied through every chunk of large_buffer */
char *large_buffer;         /* megabytes * 1MB, walked in 16KB chunks */
float *xout_buffer;         /* matmul output row */
__int64 large_buffer_bytes;
double runtime_seconds;     /* measurement budget from the CLI (seconds) */

/* Timing accumulators (seconds). Warmup rounds leave them untouched; every
 * measurement round adds its six step times. */
double shape_time_a;
double shape_time_b;
double shape_time_c;
double small_to_large_time;
double large_to_small_time;
double memcpy_kernel_time;
double float_sum_time;   /* CPU-speed probe: total seconds over all rounds */

volatile float float_sum_state; /* the sinf chain's carried value: starts at
                          * 1.234567, each round iterates FLOAT_SUM_ITERATIONS
                          * steps on it, and it is printed at the end so the
                          * loop is live code. volatile forces every iteration
                          * through a real load/store round-trip so the loop
                          * cannot be register-cached or otherwise optimized
                          * away -- an honest fixed-work CPU probe. */

/* ======================================================================
 * Timing (pure C, QPC-based)
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

/* ======================================================================
 * Dispatcher shape-block access
 * ====================================================================== */

/* Scan shape_blocks for the block whose (n,d) header matches the benchmark
 * shape. Blocks 2..255 are per-shape configs; the first block with a zero
 * n ends the list. A missing match is a FATAL: without the per-shape block
 * there is nowhere to write the INI config so a later shape-switch config
 * reload could not restore it. */
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
 * INI read (threading_proxy.ini)
 * ====================================================================== */

/* Read one int key inside an INI section via the Windows INI API.
 * GetPrivateProfileIntA cannot distinguish "absent" from "invalid", so the
 * key is read with GetPrivateProfileStringA using a sentinel default: the
 * buffer equals the sentinel iff the key is absent. A missing key is not an
 * error here -- the caller decides what absence means (in this tool every
 * required key absent is a config gap FATAL). A key that IS present but
 * malformed stays a FATAL: a typo must not become a DIFFERENT config. */
int ini_section_try_get_int(char *ini_file, char *section, char *key, int *out)
{
	/* 512 bytes: committed INI lines carry long inline comments after each
	 * value, and GetPrivateProfileStringA returns the whole line. The number
	 * always leads the line, so truncating the comment tail is harmless. */
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

/* Read the shape's committed config from the INI: all 22 dispatch params
 * plus the benchmarked kernel's pf keys. Every key is REQUIRED -- an absent
 * key is a config gap (the dispatcher would refuse an incomplete shape
 * config too; this tool reports it before anything is applied). Only the
 * benchmarked kernel's OWNED pf slots are filled here; the other slots are
 * left zero and are never applied or patched below -- another kernel's
 * tuned distances and types are not this run's business. Block 0's pf
 * arrays are NOT a read source for any pf value: v7_init fills block 1 and
 * the per-shape blocks only, block 0's pf area is dispatcher scratch. */
void load_ini_config(void)
{
	for (int param_index = 0; param_index < DISPATCH_PARAM_COUNT; param_index++)
	{
		if (!ini_section_try_get_int(ini_path, section_name, (char *)dispatch_param_names[param_index], &loaded_params[param_index]))
		{
			printf("[FATAL] load_ini_config: [%s] has no key '%s' in %s -- "
			       "the shape's committed config is incomplete; every dispatch "
			       "param is required.\n",
			       section_name, dispatch_param_names[param_index], ini_path);
			exit(EXIT_FAILURE);
		}
	}

	for (int owned_position = 0; owned_position < pf_owned_count; owned_position++)
	{
		int slot = pf_owned_list[owned_position];
		char keyname[32];
		int value;
		snprintf(keyname, sizeof(keyname), "pf_dist_%d", slot);
		if (!ini_section_try_get_int(ini_path, section_name, keyname, &value))
		{
			printf("[FATAL] load_ini_config: [%s] has no key '%s' in %s -- "
			       "the shape's committed prefetch config is incomplete; the "
			       "pf distances are required.\n",
			       section_name, keyname, ini_path);
			exit(EXIT_FAILURE);
		}
		if (value < pf_dist_lo || value > pf_dist_hi)
		{
			printf("[FATAL] load_ini_config: [%s] %s=%d outside [%d..%d] -- "
			       "the legal distance window is +/- (weight-matrix bytes / 2) "
			       "for this shape; fix the committed INI.\n",
			       section_name, keyname, value, pf_dist_lo, pf_dist_hi);
			exit(EXIT_FAILURE);
		}
		loaded_pf_dist[slot] = value;
		snprintf(keyname, sizeof(keyname), "pf_type_%d", slot);
		if (!ini_section_try_get_int(ini_path, section_name, keyname, &value))
		{
			printf("[FATAL] load_ini_config: [%s] has no key '%s' in %s -- "
			       "the shape's committed prefetch config is incomplete; the "
			       "pf types are required.\n",
			       section_name, keyname, ini_path);
			exit(EXIT_FAILURE);
		}
		if (value < PF_TYPE_MIN || value > PF_TYPE_MAX)
		{
			printf("[FATAL] load_ini_config: [%s] %s=%d outside [%d..%d] -- "
			       "unknown prefetch type; fix the committed INI.\n",
			       section_name, keyname, value, PF_TYPE_MIN, PF_TYPE_MAX);
			exit(EXIT_FAILURE);
		}
		loaded_pf_type[slot] = value;
	}
}

/* Apply the loaded config to the dispatcher. Dispatch params go into the
 * same dword slot of BOTH blocks: block 0 is the live param_set the
 * dispatcher reads immediately, and the matched block is the shadow copy a
 * shape-switch config reload restores from. The pf arrays of the
 * benchmarked kernel's OWNED slots go into both blocks the same way, and
 * every owned slot whose value differs from the stored state is ALSO
 * patched live through the exported SMC entries, distance first then type:
 * the dispatcher re-applies pf from the blocks only on a shape switch, and
 * this tool dispatches one shape for its whole run, so the direct patch is
 * the only way the INI's distances reach the SMC. Unowned slots belong to
 * other kernels: they are neither written nor patched here. */
void apply_ini_config(void)
{
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

	char patched_desc[512];
	int patched_len = 0;
	int patched_count = 0;
	patched_desc[0] = '\0';
	for (int owned_position = 0; owned_position < pf_owned_count; owned_position++)
	{
		int slot = pf_owned_list[owned_position];
		int dist = loaded_pf_dist[slot];
		int type = loaded_pf_type[slot];
		int dist_changed = (dist != dist_block0[slot]) || (dist != dist_matched[slot]);
		int type_changed = (type != type_block0[slot]) || (type != type_matched[slot]);

		dist_block0[slot] = dist;
		dist_matched[slot] = dist;
		type_block0[slot] = type;
		type_matched[slot] = type;

		if (!dist_changed && !type_changed)
			continue;
		if (dist_changed)
		{
			/* Distance first: the distance patcher restores the slot's backup
			 * bytes (re-enabling a disabled slot), while the type patcher
			 * skips a disabled slot -- so the type patch after it re-establishes
			 * the loaded type on a previously disabled slot. Skipped by the
			 * patcher itself when the new distance disables the slot. */
			inference_dll_ptr.set_prefetch_distance(slot, dist);
			inference_dll_ptr.set_prefetch_type(slot, type);
		}
		else if (type_changed)
		{
			inference_dll_ptr.set_prefetch_type(slot, type);
		}
		patched_count++;
		/* Bounded append: stop listing when the descriptor nears capacity. */
		if (patched_len < (int)sizeof(patched_desc) - 32)
			patched_len += snprintf(patched_desc + patched_len, sizeof(patched_desc) - (size_t)patched_len,
			                        " pf%02d d=%d t=%d", slot, dist, type);
	}

	printf("ini: [%s] loaded and applied (%d dispatch params + this kernel's pf keys)\n",
	       section_name, DISPATCH_PARAM_COUNT);
	if (patched_count == 0)
		printf("pf: no live patches (INI matches the loaded config)\n");
	else
		printf("pf: live-patched %d slot(s):%s\n", patched_count, patched_desc);
}

/* ======================================================================
 * Measurement round: dispatch, copy small->large, dispatch, copy large->small
 * ====================================================================== */

void run_round(int accumulate)
{
	double stamp_a;
	double stamp_b;

	stamp_a = get_seconds();
	dispatch_one();
	stamp_b = get_seconds();
	if (accumulate)
		shape_time_a += stamp_b - stamp_a;

	/* Walk the large buffer in 16KB chunks, writing the small buffer's
	 * content into every chunk (small -> large). */
	stamp_a = get_seconds();
	for (__int64 offset = 0; offset < large_buffer_bytes; offset += MEMCPY_CHUNK_BYTES)
		memcpy(large_buffer + offset, small_buffer, MEMCPY_CHUNK_BYTES);
	stamp_b = get_seconds();
	if (accumulate)
		small_to_large_time += stamp_b - stamp_a;

	stamp_a = get_seconds();
	dispatch_one();
	stamp_b = get_seconds();
	if (accumulate)
		shape_time_b += stamp_b - stamp_a;

	/* Walk the large buffer again, pulling every chunk back into the small
	 * buffer (large -> small). */
	stamp_a = get_seconds();
	for (__int64 offset = 0; offset < large_buffer_bytes; offset += MEMCPY_CHUNK_BYTES)
		memcpy(small_buffer, large_buffer + offset, MEMCPY_CHUNK_BYTES);
	stamp_b = get_seconds();
	if (accumulate)
		large_to_small_time += stamp_b - stamp_a;

	stamp_a = get_seconds();
	dispatch_one();
	stamp_b = get_seconds();
	if (accumulate)
		shape_time_c += stamp_b - stamp_a;

	/* Dispatched memcpy kernel: block-copy the CLI buffer onto itself. The
	 * v7d_memcpy entry decides ERMS single-thread vs worker-pool dispatch by
	 * its own 64KB threshold, so this measures the engine's real memcpy path
	 * at the CLI block size. */
	stamp_a = get_seconds();
	inference_dll_ptr.v7d_memcpy(large_buffer, large_buffer, (int)large_buffer_bytes);
	stamp_b = get_seconds();
	if (accumulate)
		memcpy_kernel_time += stamp_b - stamp_a;

	/* CPU-speed probe: 256K float sin iterations on the carried state = one
	 * virtual megabyte. Pure compute, zero memory traffic, so its rate tracks
	 * the core clock -- the counterpart to the memory-bound memcpy steps. One
	 * round of this counts as one virtual megabyte when reporting MB/s. */
	stamp_a = get_seconds();
	for (int k = 0; k < FLOAT_SUM_ITERATIONS; k++)
		float_sum_state = sinf(float_sum_state);
	stamp_b = get_seconds();
	if (accumulate)
		float_sum_time += stamp_b - stamp_a;
}

/* ======================================================================
 * CLI parsing
 * ====================================================================== */

int parse_cli_int(char *token, char *flag)
{
	char *end = 0;
	errno = 0;
	long value = strtol(token, &end, 10);
	if (token == end || (end && *end != '\0') || errno == ERANGE || value < INT_MIN || value > INT_MAX)
	{
		printf("[FATAL] %s: malformed or out-of-range integer '%s'\n", flag, token);
		exit(EXIT_FAILURE);
	}
	return (int)value;
}

double parse_cli_double(char *token, char *flag)
{
	char *end = 0;
	errno = 0;
	double value = strtod(token, &end);
	if (token == end || (end && *end != '\0') || !isfinite(value))
	{
		printf("[FATAL] %s: malformed or non-finite number '%s'\n", flag, token);
		exit(EXIT_FAILURE);
	}
	return value;
}

/* Parse one position of the required positional trio. Position 0 is the
 * shape token (NxM or N:M, both dimensions 1..1<<20), position 1 is the
 * megabyte size (>= 1), position 2 is the runtime budget in seconds (> 0). */
void parse_positional(int position, char *token)
{
	if (position == 0)
	{
		char tok[64];
		snprintf(tok, sizeof(tok), "%s", token);
		char *sep = strchr(tok, 'x');
		if (!sep)
			sep = strchr(tok, ':');
		if (!sep || sep == tok)
		{
			printf("[FATAL] unrecognized shape token '%s' (expected NxM or N:M)\n", token);
			exit(EXIT_FAILURE);
		}
		*sep = '\0';
		char *n_end = 0;
		char *d_end = 0;
		errno = 0;
		long rows = strtol(tok, &n_end, 10);
		long cols = strtol(sep + 1, &d_end, 10);
		int ok = (n_end != tok && *n_end == '\0' &&
		          d_end && *(sep + 1) != '\0' && *d_end == '\0' &&
		          rows > 0 && rows <= (1 << 20) && cols > 0 && cols <= (1 << 20));
		if (!ok)
		{
			printf("[FATAL] unrecognized or out-of-range shape token '%s' (each dimension must be 1..%d)\n", token, 1 << 20);
			exit(EXIT_FAILURE);
		}
		dispatch_n = (int)rows;
		dispatch_d = (int)cols;
		return;
	}
	if (position == 1)
	{
		int megabytes = parse_cli_int(token, "<megabytes>");
		if (megabytes < 1)
		{
			printf("[FATAL] <megabytes> must be >= 1 (got %d)\n", megabytes);
			exit(EXIT_FAILURE);
		}
		if (megabytes > 2047)
		{
			printf("[FATAL] <megabytes> must be <= 2047: the engine's memcpy "
			       "kernel takes a 32-bit byte count (got %d)\n", megabytes);
			exit(EXIT_FAILURE);
		}
		large_buffer_bytes = (__int64)megabytes * 1024 * 1024;
		return;
	}
	runtime_seconds = parse_cli_double(token, "<runtime_seconds>");
	if (runtime_seconds <= 0.0)
	{
		printf("[FATAL] <runtime_seconds> must be > 0 (got %g)\n", runtime_seconds);
		exit(EXIT_FAILURE);
	}
}

int parse_cli(int argc, char **argv, char **db_path, char **ini_path_arg)
{
	int positional_seen = 0;

	for (int i = 2; i < argc; i++)
	{
		if (strcmp(argv[i], "--db") == 0 && i + 1 < argc)
			*db_path = argv[++i];
		else if (strcmp(argv[i], "--ini") == 0 && i + 1 < argc)
			*ini_path_arg = argv[++i];
		else if (strcmp(argv[i], "--q8") == 0)
			use_q8_mode = 1;
		else if (strcmp(argv[i], "--f32") == 0)
			use_q8_mode = 0;
		else if (argv[i][0] == '-')
		{
			printf("[FATAL] unrecognized option '%s' (or it is missing its value)\n", argv[i]);
			return 1;
		}
		else
		{
			if (positional_seen >= 3)
			{
				printf("[FATAL] extra positional argument '%s' (expected shape, megabytes, runtime seconds)\n", argv[i]);
				return 1;
			}
			parse_positional(positional_seen, argv[i]);
			positional_seen++;
		}
	}

	if (!*db_path)
	{
		printf("Fail: --db <db_path> is required\n");
		return 1;
	}
	if (!*ini_path_arg)
	{
		printf("Fail: --ini <ini_path> is required (the shape's committed config is read from it)\n");
		return 1;
	}
	if (positional_seen < 3)
	{
		printf("Fail: <shape NxM> <megabytes> <runtime_seconds> are required\n");
		return 1;
	}
	return 0;
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
		printf("usage: %s <v7_threading.dll> --db <db_path> --ini <ini_path> [--q8|--f32] <NxM> <megabytes> <runtime_seconds>\n", argv[0]);
		return 1;
	}
	char *dll_path = argv[1];
	char *db_path = 0;
	char *ini_path_arg = 0;
	use_q8_mode = 1; /* default; --q8 / --f32 override */
	kernel_label = "q8";
	if (parse_cli(argc, argv, &db_path, &ini_path_arg))
		return 1;

	/* Normalize the INI path to absolute: the ANSI profile APIs resolve a
	 * relative lpFileName against the Windows directory, not the process
	 * CWD, so a relative --ini would silently read nothing (every key reads
	 * as absent). GetFullPathNameA resolves against the process CWD. */
	char ini_abs[MAX_PATH];
	DWORD ini_path_len = GetFullPathNameA(ini_path_arg, (DWORD)sizeof(ini_abs), ini_abs, 0);
	if (ini_path_len == 0 || ini_path_len >= sizeof(ini_abs))
	{
		printf("[FATAL] main: --ini path invalid or too long: %s\n", ini_path_arg);
		v7_unload(&inference_dll_ptr);
		return 1;
	}
	ini_path = strdup(ini_abs);

	printf("step: after parse\n");
	fflush(stdout);

	/* Load DLL and initialize the engine from the database. */
	if (v7_load(dll_path, &inference_dll_ptr) != 0)
		return 1;
	printf("step: after v7_load\n");
	fflush(stdout);
	if (!inference_dll_ptr.v7_init(db_path) || inference_dll_ptr.v7_engine_is_dead())
	{
		printf("[FATAL] v7_init failed for %s (engine dead: %d)\n", db_path, inference_dll_ptr.v7_engine_is_dead());
		v7_unload(&inference_dll_ptr);
		return 1;
	}
	printf("step: after v7_init\n");
	fflush(stdout);

	shape_blocks_ptr = (unsigned char *)GetProcAddress((HMODULE)inference_dll_ptr.module, "shape_blocks");
	if (!shape_blocks_ptr)
	{
		printf("[FATAL] shape_blocks export not found in %s -- the dispatcher "
		       "config cannot be reached, the benchmark cannot run.\n",
		       dll_path);
		v7_unload(&inference_dll_ptr);
		return 1;
	}
	printf("step: after GetProcAddress\n");
	fflush(stdout);

	/* Kernel selection and its prefetch slot ownership (slot map: q8 0-7,
	 * f32 8-10 -- the only slots this run's INI read touches). */
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

	snprintf(section_name, sizeof(section_name), "%dx%d", dispatch_n, dispatch_d);

	/* Legal prefetch distance window from the shape: +/- (weight-matrix
	 * bytes / 2), computed in 64-bit and clamped into the signed disp32
	 * budget. */
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

	printf("test_shape_speed: %dx%d %s, %lld MB, %.3f s\n",
	       dispatch_n, dispatch_d, kernel_label, (long long)(large_buffer_bytes / (1024 * 1024)), runtime_seconds);
	fflush(stdout);

	/* Benchmark tensors for the tuned kernel. */
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
	printf("step: tensors filled\n");
	fflush(stdout);
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
	printf("step: set_spin done\n");
	fflush(stdout);

	/* Shape activation: the dispatcher creates the shape's per-shape block
	 * LAZILY from the DB keys on the FIRST dispatch at that shape. One
	 * untimed dispatch here materializes the block for (n,d) so the INI
	 * apply below has a matched block to write. A missing per-shape config
	 * FATALs inside the DLL here -- the DB must carry the shape. */
	dispatch_one();
	printf("step: materializing dispatch done\n");
	fflush(stdout);

	matched_block_ptr = find_matched_block();
	{
		int block_index = (int)((matched_block_ptr - shape_blocks_ptr) / SB_SIZE);
		printf("shape_blocks: matched block index = %d (0=active,1=documentation,2+=per-shape) for %dx%d\n",
		       block_index, dispatch_n, dispatch_d);
	}
	fflush(stdout);

	/* The INI is the config source for this benchmark: read and apply the
	 * shape's committed params + pf values before anything is timed. */
	load_ini_config();
	printf("step: ini loaded\n");
	fflush(stdout);
	apply_ini_config();
	printf("step: ini applied\n");
	fflush(stdout);

	/* Copy buffers: one 16KB source and one megabytes-sized walk target. The
	 * small buffer is written once so every memcpy reads defined data. The
	 * large buffer's pages commit on the first warmup round's small->large
	 * pass, so every measured round sees fully committed memory. */
	small_buffer = (char *)malloc(MEMCPY_CHUNK_BYTES);
	large_buffer = (char *)malloc(large_buffer_bytes);
	if (!small_buffer || !large_buffer)
	{
		printf("[FATAL] main: out of memory for the copy buffers "
		       "(16KB + %lld MB).\n",
		       (long long)(large_buffer_bytes / (1024 * 1024)));
		v7_unload(&inference_dll_ptr);
		return 1;
	}
	memset(small_buffer, 0xAB, MEMCPY_CHUNK_BYTES);
	float_sum_state = 1.234567f; /* the sinf chain's starting value */

	/* Warmup: 20 identical rounds, timings discarded. This settles worker
	 * spin state, branch predictors and page commitment before the measured
	 * rounds begin. */
	printf("warming up: %d rounds\n", WARMUP_ROUNDS);
	for (int round = 0; round < WARMUP_ROUNDS; round++)
		run_round(0);

	/* Measurement: run rounds until the runtime budget has passed. Each
	 * round accumulates its four step times into the four accumulators. */
	printf("measuring: %.3f s budget\n", runtime_seconds);
	fflush(stdout);
	double measure_start = get_seconds();
	int measured_rounds = 0;
	while (get_seconds() - measure_start < runtime_seconds)
	{
		run_round(1);
		measured_rounds++;
	}
	double measure_elapsed = get_seconds() - measure_start;

	/* Convert each accumulator to GB/s. The matmul steps move n*d*5 bytes
	 * per call (input row + int8 weight block + output row, the same byte
	 * accounting the loadtest and autotuner report); each copy loop moves
	 * exactly the large buffer's byte volume. */
	__int64 shape_bytes_per_call = (__int64)dispatch_n * dispatch_d * 5;
	double shape_a_gbs = (double)measured_rounds * (double)shape_bytes_per_call / shape_time_a / 1e9;
	double shape_b_gbs = (double)measured_rounds * (double)shape_bytes_per_call / shape_time_b / 1e9;
	double shape_c_gbs = (double)measured_rounds * (double)shape_bytes_per_call / shape_time_c / 1e9;
	double small_to_large_gbs = (double)measured_rounds * (double)large_buffer_bytes / small_to_large_time / 1e9;
	double large_to_small_gbs = (double)measured_rounds * (double)large_buffer_bytes / large_to_small_time / 1e9;
	double memcpy_kernel_gbs = (double)measured_rounds * (double)large_buffer_bytes / memcpy_kernel_time / 1e9;
	/* One round of the sinf chain = one virtual megabyte. */
	double float_sum_mbps = (double)measured_rounds / float_sum_time;

	printf("\nshape %dx%d %s: %d rounds in %.3f s\n",
	       dispatch_n, dispatch_d, kernel_label, measured_rounds, measure_elapsed);
	printf("shape_time_a    %8.1f GB/s\n", shape_a_gbs);
	printf("small_to_large  %8.1f GB/s\n", small_to_large_gbs);
	printf("shape_time_b    %8.1f GB/s\n", shape_b_gbs);
	printf("large_to_small  %8.1f GB/s\n", large_to_small_gbs);
	printf("shape_time_c    %8.1f GB/s\n", shape_c_gbs);
	printf("memcpy_kernel   %8.1f GB/s\n", memcpy_kernel_gbs);
	printf("float_sum       %8.1f MB/s (1 round = 1 virtual MB)\n", float_sum_mbps);
	printf("float_sum state %.7g\n", float_sum_state);
	/* Derived drift-correction value: clock probe over memory reference.
	 * At >= 16MB the dispatched memcpy is DRAM-bound and clock-insensitive,
	 * so this ratio isolates the core-clock state: compare it against a
	 * reference run's value at the SAME buffer size and divide the matmul
	 * rates by the factor to correct power-state drift. At 1MB the ratio is
	 * degenerate (both probes are clock-bound there). */
	printf("float/memcpy    %8.3f (clock-state factor)\n", float_sum_mbps / memcpy_kernel_gbs);
	/* Drift probe: ratios of the step times to each other. Both sides of
	 * each ratio run on the same clock, so every ratio should stay constant
	 * across power-state changes -- a drifting ratio points at a step whose
	 * speed moved independently of the machine's overall rate. */
	printf("ratios shape/copy time\n");
	printf("a/small_to_large  %10.4f\n", shape_time_a / small_to_large_time);
	printf("a/large_to_small  %10.4f\n", shape_time_a / large_to_small_time);
	printf("b/small_to_large  %10.4f\n", shape_time_b / small_to_large_time);
	printf("b/large_to_small  %10.4f\n", shape_time_b / large_to_small_time);
	printf("c/small_to_large  %10.4f\n", shape_time_c / small_to_large_time);
	printf("c/large_to_small  %10.4f\n", shape_time_c / large_to_small_time);
	printf("a/memcpy_kernel   %10.4f\n", shape_time_a / memcpy_kernel_time);
	printf("b/memcpy_kernel   %10.4f\n", shape_time_b / memcpy_kernel_time);
	printf("c/memcpy_kernel   %10.4f\n", shape_time_c / memcpy_kernel_time);
	printf("a/float_sum       %10.4f\n", shape_time_a / float_sum_time);
	printf("b/float_sum       %10.4f\n", shape_time_b / float_sum_time);
	printf("c/float_sum       %10.4f\n", shape_time_c / float_sum_time);
	printf("memcpy_kernel/float_sum %10.4f\n", memcpy_kernel_time / float_sum_time);
	fflush(stdout);

	/* Cleanup. */
	free(small_buffer);
	free(large_buffer);
	free(xout_buffer);
	if (use_q8_mode)
		tensor_q8_free(&g_tsq);
	else
		tensor_f32_free(&g_tsf);
	v7_unload(&inference_dll_ptr);
	return 0;
}