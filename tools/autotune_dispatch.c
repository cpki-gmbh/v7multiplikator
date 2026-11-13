// v7 - part of the v7 inference engine
// Copyright (C) 2026 Gorg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)
#include <windows.h> 

void gotoxy(int x, int y)
{
  COORD coord;
  coord.X = x;
  coord.Y = y;
  SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);
}

/*
 * autotune_dispatch.c -- Hill-climb autotuner for dispatcher parameters
 *
 * Continuously A/B tests candidate dispatch parameter settings against
 * the ruler (the committed INI [NxD] config; a frozen random genome R when
 * the INI has no complete section) for a single matmul shape. One cooldown
 * sleep runs before the one-time warmup; there are no sleeps between tests
 * -- the interleaved A/B alternation is the fairness mechanism.
 *
 * Usage: autotune_dispatch.exe <dll> --db <db> --ini <ini> <shape N:M> [options]
 *
 * Options:
 *   --db <path>    REQUIRED database path (no default)
 *   --ini <path>   REQUIRED threading_proxy.ini path (no default; the FILE
 *                  itself may not exist yet). When the file, the [NxD] section,
 *                  or any tunable key is missing, the ruler is a frozen random
 *                  genome R. The INI is a READ-ONLY ruler: this tool NEVER
 *                  writes it -- every adopted config is archived as a
 *                  standalone genome INI via --winner-dir instead.
 *   --q8           Q8 matmul (default)
 *   --f32          F32 matmul
 *   --time <s>     accepted for compatibility, IGNORED (no calibration loop)
 *   --iterations N accepted for compatibility, IGNORED (no calibration loop)
 *   --cooldown N   Ms of one-time settle sleep before the warmup (default 20)
 *   --rounds N     Number of A/B rounds (default 60)
 *   --runtime N    Total tuning budget in minutes (default 0 = unlimited).
 *                  Checked before EVERY parameter change (probe, hill-climb
 *                  round, refinement trial): when expired, the run ends
 *                  keeping the current best (archived via --winner-dir when
 *                  given). A run with --winner-dir ALWAYS leaves at least
 *                  one genome behind: when nothing was adopted, the final
 *                  best (the ruler when no probe beat it) is archived.
 *   --probe N      Random-probe genomes drawn before the A/B rounds
 *                  (default 30; must be >= 1). Probe #0 is always the ruler
 *                  itself (committed INI config, or random R), measured with
 *                  the same sample count as every probe. With --runtime the
 *                  probe phase consumes at most a QUARTER of the budget --
 *                  the remaining time belongs to the hill-climb and refine
 *                  phases that actually collect genome wins; without
 *                  --runtime the probes run through the full count.
 *   --median N     Interleaved samples per config per round (default 3).
 *                  Base and candidate are measured N times alternately; the
 *                  BEST (max) of each side is used for the ratio. Higher N
 *                  reduces thermal/noise chasing at the cost of runtime.
 *
 * All 22 dispatch params are tuned (indices 0-21). Params are read/written
 * DIRECTLY (no DLL API -- the dispatch_param_table accessor exports are gone):
 * ALL 22 live in param_set slots, written via the exported shape_blocks symbol
 * into block 0 (live) and the matched per-shape block (shadow) -- taper_shift
 * in slot 1, quantize_chunk_groups in slot 12; there are no standalone BSS
 * globals for dispatch params. Cross-param invariants are enforced via
 * sanitize_candidate(): worker/caller batch thresholds ascending. Random draws
 * use the x86
 * RDRAND instruction (_rdrand32_step, no seeding).
 *
 * The 82 prefetch params (pf_dist_0..40, pf_type_0..40) are tuned across the
 * WHOLE parameter space -- all 41 slots of the shape block's pf arrays
 * (+104 / +300) are probe-drawn, mutated, and refined, with no artificial
 * window beyond the engine's 41-slot table (the INI's pf_dist_41..48 lines
 * belong to the removed fused-Q8 kernel and are dead keys the engine never
 * reads; they are left untouched). All 82 keys are read from the INI (a
 * missing file, section, or key means no committed config: probe #0 is
 * skipped and the search seeds from the uniform baseline) and carried
 * through every snapshot, so every archived genome stays fully populated. The legal distance window is
 * data-driven: every distance from -(weight-matrix bytes / 2) to
 * +(weight-matrix bytes / 2) is legal, where the weight matrix is the
 * benchmark's NxD int8 weight block. Negative distances are REAL backward
 * prefetches (the SMC patcher's unsigned < 32 check passes them through to
 * the signed disp32 write); the unsigned [0..31] band is the engine's
 * disabled-slot regime -- a legitimate config the INI itself uses
 * (pf_dist_8=0 disables the F32 activation prefetch).
 * The tuner never rewrites committed distances: any in-window value is
 * accepted as-is, and only a value outside the window is a fix-the-INI
 * FATAL. Types are tuned in [0..6] (NTA/T0/T1/T2/PREFETCHW/PREFETCHWT1/
 * CLDEMOTE -- the engine FATALs on anything else, a real contract).
 * Because the dispatcher's level-2 prefetch cache only re-patches on a
 * shape switch, apply_snapshot() calls the exported
 * set_prefetch_distance/set_prefetch_type directly for every slot whose
 * value changed (distance first, then type -- the distance patch restores a
 * NOPped slot's backup bytes, so the type re-patch afterwards re-establishes
 * the tuned type instead of silently leaving the backup opcode live).
 *
 * taper_shift is in fixed-point units of 1/16 (k = value/16), so e.g.
 * 16 = k=1.0, 20 = k=1.25, 48 = k=3.0, 112 = k=7.0 (max). 0 = uniform.
 */
#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#include "zydis_debug_header.h"
#include <immintrin.h> /* _rdrand32_step */
#include <errno.h>     /* strtol range reporting */
#include <limits.h>    /* INT_MIN/INT_MAX */
#include <math.h>      /* exp/log for log-uniform probe draws */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

/* ======================================================================
 * Core data definitions (used throughout the file)
 * ====================================================================== */

#define TUNE_MAX_PARAMS 22

/* === Prefetch tuning space (must match v7_data.inc / smc_macros.inc) === */
#define PF_SLOTS 41          /* PF_DIST_TABLE_ENTRIES: pf_dist_0..40, pf_type_0..40 */
#define SB_PF_DIST_OFF 104   /* SHAPE_BLOCK_PF_DIST_OFF: pf_dist[41] inside a 512-byte shape block */
#define SB_PF_TYPE_OFF 300   /* SHAPE_BLOCK_PF_TYPE_OFF: pf_type[41] */
#define PF_TYPE_MIN 0        /* 0=NTA 1=T0 2=T1 3=T2 4=PREFETCHW 5=PREFETCHWT1 6=CLDEMOTE */
#define PF_TYPE_MAX 6
#define MEASURE_MAX_SAMPLES 64 /* widest sample array: measure_ratio_best's per-side buffers */

/* Legal distance window, computed in main() from the tuned shape: every
 * distance in [-(weight-matrix bytes / 2), +(weight-matrix bytes / 2)] is
 * legal. Negative values are real backward prefetches (the SMC patcher
 * writes the signed disp32 whenever the unsigned value is >= 32); the
 * unsigned [0..31] band disables the slot (the cache line is already being
 * fetched), which is a legitimate configuration, not an error. */
int pf_dist_lo; /* -(matrix bytes / 2) */
int pf_dist_hi; /* +(matrix bytes / 2) */

typedef struct
{
	int values[TUNE_MAX_PARAMS];
	int count;
	int pf_dist[PF_SLOTS]; /* per-shape prefetch distances (bytes ahead) */
	int pf_type[PF_SLOTS]; /* per-shape prefetch types (0..6) */
} param_snapshot;

/* Benchmark tensors. Each tensor's buffers are OOM-FATAL'd at the alloc site
 * (see tensor_*_alloc) -- a 0 pointer here is a process-ending bug. */
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

/* Dual-block write state.
 * shape_blocks = exported base of the SHAPE_BLOCKS array (block 0 = active).
 * matched_block = pointer to the per-shape block whose header (n,d) equals the
 *   autotuner's target shape (found via find_matched_block, which FATALs if no
 *   such block exists -- there is no default fallback for the shadow target).
 *   Writing to *this* block's param area keeps the candidate alive across the
 *   dispatcher's .config_change cold path, which copies matched_block -> block 0.
 */
unsigned char *g_shape_blocks = 0;  /* base of shape_blocks export (block 0) */
unsigned char *g_matched_block = 0; /* per-shape block for (n,d), never block 1 */

/* param_set DWORD index for each dispatch param index (the C-side twin of
 * v7_data.inc's PARAM_OFFSET_* byte constants divided by 4). The param_set
 * layout is NOT contiguous by dispatch index: chunk_multiplier sits at dword
 * 9 but the caller batch group starts at dword 13. taper_shift (idx 20) and
 * quantize_chunk_groups (idx 21) live in param_set slots 1 and 12 -- ALL 22
 * params are param_set-backed; there are no standalone BSS globals. */
int sb_param_off_for_idx[TUNE_MAX_PARAMS] = {
	0,  2,  3,  4,  5,  6,  7,  8,  9,  10, /* idx 0..9  */
	13, 14, 15, 16, 17, 18, 19, 20, 21, 22, /* idx 10..19 */
	1,  12                                  /* idx 20=taper_shift(slot 1), 21=quantize_chunk_groups(slot 12) */
};

#define DISPATCH_PARAM_COUNT 22 /* hardcoded: matches v7_data.inc dispatch table size */

/* Hardcoded dispatch param names in EXACT dispatch-index order (the old
 * dispatch_param_table order, preserved from git history). With the accessor
 * exports gone there is no DLL query for these -- every display/INI path uses
 * this table. Indices 20/21 are param_set slots 1 and 12. */
char *dispatch_param_names[TUNE_MAX_PARAMS] = {
	"num_chunks",          "batch_small",         "batch_medium",        "batch_large",          "batch_huge",        "batch_med_thresh",        "batch_large_thresh",        "batch_huge_thresh",        "chunk_multiplier",
	"min_chunk_cols",      "caller_batch_small",  "caller_batch_medium", "caller_batch_large",   "caller_batch_huge", "caller_batch_med_thresh", "caller_batch_large_thresh", "caller_batch_huge_thresh", "chunk_split_pct",
	"chunk_group_1_count", "chunk_group_2_count", "taper_shift",         "quantize_chunk_groups"
};

/* All 22 dispatch params are tuned. tune_indices is identity (slot w == dispatch
 * index w), so tune_min[w]/tune_max[w] bound dispatch param w directly. */
int tune_indices[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };

/* Min/max bounds per dispatch index (engine contract: fill-mode rules).
 *   num_chunks           1..4096  (capped to MAX_JOBS)
 *   batch_small..huge    1..64
 *   batch_*_thresh       1..4096
 *   chunk_multiplier     1..128
 *   min_chunk_cols       4..64   (>= 4 with taper -- Q8 4-row block; no pow2
 *                                 requirement: the engine div-aligns non-pow2 W_min)
 *   caller_batch_*       1..64
 *   caller_*_thresh      1..4096sp
 *   chunk_split_pct      0..99
 *   chunk_group_1/2      1..256   (g1+g2 <= MAX_JOBS when split active)
 *   taper_shift          0..112  (fixed-point 1/16: 112 = k=7.0 = TAPER_SHIFT_MAX)
 *   quantize_chunk_groups 1..64  (0 or negative is an ENGINE FATAL in
 *                         v7d_quantize -- there is no "0 = auto" fallback)
 */
int tune_min[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1 };
int tune_max[] = { 4096, 64, 64, 64, 64, 4096, 4096, 4096, 128, 64, 64, 64, 64, 64, 4096, 4096, 4096, 99, 256, 256, 112, 64 };

/* ======================================================================
 * Tuning session state (globals instead of multi-hundred-byte prototypes)
 * ====================================================================== */

/* The entire tuning session shares ONE set of globals, set up once in main()
 * and read/written by every phase. This eliminates the old 16-18 parameter
 * function prototypes (probe_phase, hill_climb, refine_candidate, ...) where
 * every caller had to thread the same state by hand and one wrong order was a
 * silent miscompile. Each phase function now reads/writes this state directly.
 *
 * All snapshot globals are plain value copies of param_snapshot; the probe and
 * hill-climb phases mutate candidate_snapshot/best_snapshot in place, and the
 * winner archive writes best_snapshot. */
struct v7_lib inference_dll_ptr; /* loaded DLL + all resolved function ptrs */

int use_q8_mode;            /* 1 = Q8 dispatch, 0 = F32 dispatch */
int dispatch_n, dispatch_d; /* matmul shape being tuned (N x D) */
int g_cooldown_ms;          /* ms of the per-comparison settle sleep (A-B-A protocol) */
double screen_best_gbs;     /* fastest throughput seen in this session */
int dispatch_nparams;       /* dispatch params in a snapshot (DISPATCH_PARAM_COUNT) */
int g_probe_n;              /* random-probe samples before hill-climb */
int g_max_rounds;           /* A/B hill-climb rounds */
int g_runtime_minutes;      /* --runtime: tuning budget in minutes (0 = unlimited) */
double g_deadline;          /* absolute get_seconds() time when the budget expires */
long baseline_update_count; /* how often the best genome was overwritten across
                         * all phases: probe adoption, hill-climb ratchet, refine wins */
int g_winner_counter;       /* the archive counter for --winner-dir recording */
char g_winner_dir[MAX_PATH] = ""; /* --winner-dir: when set, every baseline win is
                         * archived as <dir>\<shape>_<counter>.ini; empty = count only */

void save_winner_genome(double win_gbs); /* forward: refinement runs before the definition */

tensor_q8 g_tsq;  /* Q8 benchmark tensors (use_q8_mode path) */
tensor_f32 g_tsf; /* F32 benchmark tensors */
float *g_xout;    /* matmul output buffer (d floats) */

char *g_ini_path;   /* threading_proxy.ini (normalized absolute) */
char g_section[32]; /* "NxD" section name for read/write */
int g_have_ini_config; /* 1 when the INI carried a complete [NxD] section (probe #0); 0 = seed from the uniform baseline */

/* Snapshots -- the live tuning state all phases operate on.
 *   g_original     -- params as loaded from the DLL (restored before exit)
 *   bigram_base_snap    -- current live baseline; ratcheted to winners
 *   g_uniform_snap -- immutable ruler reference (INI config or random R):
 *                     refine and final confirmation always report against
 *                     the TRUE ruler, never against a ratcheted winner
 *   g_ini_snap     -- committed INI [NxD] config (ALWAYS probe #0)
 *   candidate_snapshot    -- current hill-climb candidate
 *   best_snapshot    -- best-known candidate in the uniform frame
 *   bigram_best_ratio / bigram_best_cand_gbs / bigram_best_base_gbs -- its score */
param_snapshot g_original;
param_snapshot bigram_base_snap;
param_snapshot g_uniform_snap;
param_snapshot g_ini_snap;
param_snapshot candidate_snapshot;
param_snapshot best_snapshot;
double bigram_best_ratio = 0.0;
double bigram_best_cand_gbs = 0.0;
double bigram_best_base_gbs = 0.0;

/* === Shape block layout (must match v7_data.inc) ===
 * Block layout: block_n(4) + block_d(4) + param_set(96 = 24 dwords) + pf_dist[49] + pf_type[49] + pad
 *   SHAPE_BLOCK_SIZE     = 512 bytes total per block
 *   SHAPE_BLOCK_PARAM_OFF = 8  (param_set dwords start here)
 * Block 0 = the live param_set the dispatcher reads at dispatch time.
 * Block 1 = default fallback (dispatcher uses it when no per-shape block matches;
 *           this tool never writes it -- find_matched_block FATALs instead).
 * Blocks 2..255 = per-shape configs (header n,d nonzero); scan stops at block with n==0.
 *
 * Critical: the dispatcher's .config_change path (cold path) copies the matched
 * per-shape block's params -> block 0, overwriting whatever a writer put in
 * block 0 alone. To survive this, apply_snapshot() writes the candidate to BOTH
 * block 0 (immediate effect) AND the matched per-shape block (preserved through
 * subsequent .config_change copies). Since the benchmark only ever dispatches one
 * shape, .config_change fires only on the first call after init/warmup; subsequent
 * calls hit the cached_shape_n/d hot path. But the first dispatch after a write
 * could otherwise trigger .config_change (if the previous shape differs -- e.g.
 * prior probe shape), so writing the matched block fixes the race.
 */
#define SB_SIZE 512
#define SB_PARAM_OFF 8
#define SB_MAX_BLOCKS 256
#define SB_PARAMS_DWORDS 24

/* Refinement (coordinate descent) limits. REFINE_MAX_PASSES bounds re-scans of the
 * full param list within one descent iteration; REFINE_MAX_ITERS is the hard cap on
 * whole-iteration re-runs. */
#define REFINE_MAX_PASSES 3
#define REFINE_MAX_ITERS 5

/* ======================================================================
 * Timing (pure C, QPC-based)
 * ====================================================================== */

#include "qpc_seconds.h"

/* 1 when the --runtime budget is exhausted. The clock starts when the first
 * tuning phase begins (DLL load, tensor fill, and calibration are setup, not
 * tuning). Every phase checks BEFORE each parameter change (probe draw,
 * hill-climb round, refinement trial) and ends the run on expiry. */
int runtime_expired(void)
{
	if (g_runtime_minutes <= 0)
		return 0; /* no --runtime given: unlimited, never expires */
	return get_seconds() >= g_deadline;
}

/* ======================================================================
 * Tensor alloc/fill (pure C, adapted from kt_utils.h)
 * Random draws use the x86 RDRAND instruction (_rdrand32_step, no seeding).
 * ====================================================================== */

/* One 32-bit hardware-random draw. RDRAND can report transient failure
 * (zero return): retry a bounded number of times, then FATAL -- a failed draw
 * silently becoming 0 would pin frand()/rng_int() and degenerate every probe
 * and mutation. Hardware presence is already guaranteed by the DLL's CPUID
 * baseline gate (RDRAND is in the required set), so no separate CPUID probe. */
unsigned rng_u32(void)
{
	for (int t = 0; t < 10; t++)
	{
		unsigned v = 0;
		if (_rdrand32_step(&v))
			return v;
	}
	printf("[FATAL] rng_u32: RDRAND failed 10 consecutive draws.\n");
	exit(EXIT_FAILURE);
}

/* Uniform float in [0.0, 1.0) with 23 random mantissa bits (the implicit bit
 * is fixed at 1). RDRAND
 * gives 32 bits; mask 23 stored mantissa bits and set the exponent field to
 * 0 (bias 127), producing a value in [1.0, 2.0) whose significand is fully
 * random. Subtracting 1.0f maps it to
 * [0.0, 1.0) -- not the coarse 100-step grid the old v%100 formula produced,
 * which quantized the scale/activation space and hid distribution effects. */
float frand(void)
{
	unsigned int bits = 0x3F800000u | (rng_u32() & 0x007FFFFFu);
	float f;
	memcpy(&f, &bits, sizeof(f));
	return f - 1.0f;
}

void init_random_int8(char *data, __int64 count)
{
	/* One RDRAND per element, mapped to [-100, 100]. Plain and per element --
	 * this is benchmark filler data, there is no performance problem here. */
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
	__int64 wc = (__int64)matmul_cols * matmul_rows;
	__int64 sc = wc / 64 + 1; /* scale count in __int64: (int) truncation wrapped for huge accepted shapes */
	ts->input_q = (char *)malloc(matmul_rows);
	ts->input_s = (float *)malloc((matmul_rows / 64 + 1) * sizeof(float));
	ts->weight_q = (char *)malloc(wc);
	ts->weight_s = (float *)malloc(sc * sizeof(float));
	/* Out-of-memory is a FATAL, not a recoverable condition: a 0 tensor
	 * pointer would crash (or silently mis-measure) far from the allocation
	 * site, and there is no meaningful fallback for a missing benchmark
	 * tensor. Fail loudly at the source of the failure. */
	if (!ts->input_q || !ts->input_s || !ts->weight_q || !ts->weight_s)
	{
		printf("[FATAL] tensor_q8_alloc: out of memory for %dx%d tensor "
		       "(~%lld bytes).\n",
		       matmul_rows, matmul_cols, (long long)(wc + matmul_rows));
		exit(EXIT_FAILURE);
	}
}

void tensor_q8_free(tensor_q8 *ts)
{
	free(ts->input_q);
	free(ts->input_s);
	free(ts->weight_q);
	free(ts->weight_s);
}

void tensor_q8_fill(tensor_q8 *ts, int matmul_rows, int matmul_cols)
{
	__int64 wc = (__int64)matmul_cols * matmul_rows;
	init_random_int8(ts->input_q, matmul_rows);
	init_random_float(ts->input_s, matmul_rows / 64 + 1);
	init_random_int8(ts->weight_q, wc);
	init_random_float(ts->weight_s, wc / 64 + 1);
}

void tensor_f32_alloc(tensor_f32 *ts, int matmul_rows, int matmul_cols)
{
	__int64 wc = (__int64)matmul_cols * matmul_rows;
	__int64 sc = wc / 64 + 1; /* __int64: same truncation hazard as the Q8 scales */
	ts->input_f = (float *)malloc(matmul_rows * sizeof(float));
	ts->input_s = (float *)malloc((matmul_rows / 64 + 1) * sizeof(float));
	ts->weight_q = (char *)malloc(wc);
	ts->weight_s = (float *)malloc(sc * sizeof(float));
	/* Out-of-memory is a FATAL, not a recoverable condition: a 0 tensor
	 * pointer would crash (or silently mis-measure) far from the allocation
	 * site, and there is no meaningful fallback for a missing benchmark
	 * tensor. Fail loudly at the source of the failure. */
	if (!ts->input_f || !ts->input_s || !ts->weight_q || !ts->weight_s)
	{
		printf("[FATAL] tensor_f32_alloc: out of memory for %dx%d tensor "
		       "(~%lld bytes).\n",
		       matmul_rows, matmul_cols, (long long)(wc + matmul_rows));
		exit(EXIT_FAILURE);
	}
}

void tensor_f32_free(tensor_f32 *ts)
{
	free(ts->input_f);
	free(ts->input_s);
	free(ts->weight_q);
	free(ts->weight_s);
}

void tensor_f32_fill(tensor_f32 *ts, int matmul_rows, int matmul_cols)
{
	__int64 wc = (__int64)matmul_cols * matmul_rows;
	init_random_float(ts->input_f, matmul_rows);
	init_random_float(ts->input_s, matmul_rows / 64 + 1);
	init_random_int8(ts->weight_q, wc);
	init_random_float(ts->weight_s, wc / 64 + 1);
}

/* ======================================================================
 * Dispatch matmul (one iteration, Q8 or F32)
 *
 * The Q8 crash root cause was an ASM split-fill routing bug: matmul_dispatch.inc
 * .split_fill_prep stored the RAW chunk_split_pct in mm_chunk_split_pct, but the
 * fill-loop routing compares that slot against the FILL_SPLIT sentinel (== 1).
 * For pct > 1 the first split job routed back into the UNIFORM loop with
 * mm_base = g1_cols and mm_chunk_count = g1+g2, producing a runaway
 * column_offset and garbage job pointers (observed crash col = 15362 for
 * d=1024, g1_cols=256). FIXED in ASM: .split_fill_prep now stores the
 * FILL_SPLIT sentinel, so split fill routes correctly for any pct. The per-call
 * tensor structs live on the stack here; the DLL reads the weight pointer
 * (heap) out of the job struct, so caller-side storage location is irrelevant.
 * ====================================================================== */

/* === Tunable kernel scope ===
 * The benchmark measures ONE dispatched entry, and every prefetch slot
 * belongs to exactly one kernel (the slot map: q8 0-7, f32 8-10, fused f32
 * 11-12, swiglu f32 13-26, swiglu q8 27-38, attention 39-40). Slots of OTHER
 * kernels are never drawn, mutated, or archived: a genome file carries only
 * the prefetch keys of the kernel the run really measured. */
#define KERN_Q8 0
#define KERN_F32 1
#define KERN_FUSED 2
#define KERN_SWIGLU_F32 3
#define KERN_SWIGLU_Q8 4
int g_kernel = KERN_Q8;
const char *g_kernel_label = "q8";
int g_pf_owned[PF_SLOTS];       /* 1 = this slot drives the benchmarked kernel */
int g_pf_owned_list[PF_SLOTS];  /* owned slot ids (the mutation pick pool) */
int g_pf_owned_count = 0;

/* Per-kernel scratch tensors (allocated in main for non-q8/f32 kernels). */
float *g_kx_f32;                /* fused / swiglu_f32 activation [n x n] */
char *g_kx_q8;                  /* swiglu_q8 activation [n x n] */
float *g_kx_q8_s;               /* swiglu_q8 activation scales */
float *g_normw;                 /* fused: norm weights [n] */
tensor_q8 g_kernw;              /* fused q8 weight [d x n] */
tensor_q8 g_gatew;              /* swiglu gate weights [d x n] */
tensor_q8 g_upw;                /* swiglu up weights [d x n] */
float *g_kout;                  /* per-kernel output [n x d] */

void dispatch_one(void)
{
	switch (g_kernel)
	{
	case KERN_Q8:
	{
		v7_q8_tensor x, w;
		x.q = g_tsq.input_q;
		x.s = g_tsq.input_s;
		w.q = g_tsq.weight_q;
		w.s = g_tsq.weight_s;
		inference_dll_ptr.v7d_q8(g_xout, &x, &w, dispatch_n, dispatch_d);
		break;
	}
	case KERN_F32:
	{
		v7_f32_tensor xf;
		v7_q8_tensor w;
		xf.data = g_tsf.input_f;
		xf.scales = g_tsf.input_s;
		w.q = g_tsf.weight_q;
		w.s = g_tsf.weight_s;
		inference_dll_ptr.v7d_f32(g_xout, &xf, &w, dispatch_n, dispatch_d);
		break;
	}
	case KERN_FUSED:
	{
		v7_f32_tensor xf;
		v7_q8_tensor w;
		xf.data = g_kx_f32;
		xf.scales = 0;
		w.q = g_kernw.weight_q;
		w.s = g_kernw.weight_s;
		inference_dll_ptr.v7d_fused_f32(g_kout, &xf, &w, dispatch_n, dispatch_d, g_normw, 0);
		break;
	}
	case KERN_SWIGLU_F32:
	{
		v7_f32_tensor xf;
		v7_q8_tensor gate, up;
		xf.data = g_kx_f32;
		xf.scales = 0;
		gate.q = g_gatew.weight_q;
		gate.s = g_gatew.weight_s;
		up.q = g_upw.weight_q;
		up.s = g_upw.weight_s;
		inference_dll_ptr.v7d_swiglu(g_kout, &xf, &gate, dispatch_n, dispatch_d, &up);
		break;
	}
	case KERN_SWIGLU_Q8:
	{
		v7_q8_tensor x, gate, up;
		x.q = g_kx_q8;
		x.s = g_kx_q8_s;
		gate.q = g_gatew.weight_q;
		gate.s = g_gatew.weight_s;
		up.q = g_upw.weight_q;
		up.s = g_upw.weight_s;
		inference_dll_ptr.v7d_swiglu_q8(g_kout, &x, &gate, dispatch_n, dispatch_d, &up);
		break;
	}
	}
}

/* ======================================================================
 * Measure GB/s for the current params (one timed sample, no warmup)
 * ====================================================================== */

double measure_gbs(void)
{
	/* ONE dispatched call, timed once. The A-B-A structure is the noise
	 * mechanism -- four groups, best value per position -- so there is no
	 * calibrated iteration loop and no averaging here. */
	double t0 = get_seconds();
	dispatch_one();
	double t1 = get_seconds();
	double elapsed = t1 - t0;
	/* bytes moved per call, per kernel: activations + weights + output */
	__int64 nn = (__int64)dispatch_n * dispatch_n;
	__int64 nd = (__int64)dispatch_n * dispatch_d;
	__int64 bytes_per_call;
	switch (g_kernel)
	{
	case KERN_Q8:
	case KERN_F32:
		bytes_per_call = nd * 5; /* x(n) + w(n*d) + out(n*d) reads, int8 dominated */
		break;
	case KERN_FUSED:
		bytes_per_call = nd * 5 + dispatch_n * 4; /* + the norm weights */
		break;
	case KERN_SWIGLU_F32:
		bytes_per_call = nn * 4 + 2 * nd + nd * 4; /* f32 x + two int8 weights + f32 out */
		break;
	default: /* KERN_SWIGLU_Q8 */
		bytes_per_call = nn + 2 * nd + nd * 4; /* int8 x + two int8 weights + f32 out */
		break;
	}
	return (double)bytes_per_call / elapsed / 1e9;
}

/* Scan shape_blocks for the block whose (n,d) header matches (target_n,target_d).
 * The autotuner writes candidates into the matched per-shape block so the
 * dispatcher's .config_change copy preserves them; without that block there is
 * nowhere to shadow the candidate and the values are silently lost on the next
 * shape switch. A missing match is therefore a FATAL -- the DB/INI the DLL
 * loaded does not carry a config for the shape being tuned. (g_shape_blocks
 * itself is validated by the caller in main() right after GetProcAddress; this
 * function only scans, it does not re-check the base pointer.) */
unsigned char *find_matched_block(void)
{
	/* Scan blocks 2..SB_MAX_BLOCKS-1 until first block with n==0 (end of list). */
	for (int i = 2; i < SB_MAX_BLOCKS; i++)
	{
		int *header = (int *)(g_shape_blocks + i * SB_SIZE); /* block header: [0] = n, [1] = d */
		if (header[0] == 0)
			break; /* first empty block = end of per-shape list */
		if (header[0] == dispatch_n && header[1] == dispatch_d)
			return (unsigned char *)header; /* exact match */
	}
	printf("[FATAL] find_matched_block: no per-shape block for %dx%d "
	       "(scanned blocks 2..%d). The DB the DLL loaded has no config "
	       "for this shape, so tuned values could not survive a shape "
	       "switch. Rebuild the DB from an INI that has a [%dx%d] section.\n",
	       dispatch_n, dispatch_d, SB_MAX_BLOCKS - 1, dispatch_n, dispatch_d);
	exit(EXIT_FAILURE);
}

/* Apply a snapshot to BOTH block 0 (param_set_0 = immediate effect) and the
 * matched per-shape block (so .config_change copy preserves the values).
 * One mechanism per action:
 *   - param_set-backed params (indices 0-19): stored into the same dword slot
 *     of BOTH blocks through the shared offset map below.
 * All 22 dispatch params now live in param_set slots (taper_shift in
 * slot 1 at offset 4, quantize_chunk_groups in slot 12 at offset 48).
 * There are no more standalone BSS globals for dispatch params. Both
 * g_shape_blocks and g_matched_block are resolved and validated by the
 * caller (main): the base pointer is FATAL-checked right after GetProcAddress
 * and find_matched_block FATALs when no per-shape block exists -- so this
 * function never sees a 0 block pointer.
 *
 * snap->count is DISPATCH_PARAM_COUNT (22), set by snapshot_params and copied
 * by every candidate snapshot -- it never exceeds the array width, so no
 * count bound is re-checked here. */
void apply_snapshot(param_snapshot *snap)
{
	/* The param_set area of both blocks as the int array it is.
	 * sb_param_off_for_idx[i] is a DWORD index into that array (the
	 * PARAM_OFFSET_* byte constants in v7_data.inc divided by 4). */
	int *blk0 = (int *)(g_shape_blocks + SB_PARAM_OFF);
	int *mblk = (int *)(g_matched_block + SB_PARAM_OFF);
	for (int i = 0; i < snap->count; i++)
	{
		int slot = sb_param_off_for_idx[i];
		/* Param_set-backed param: write the same dword into BOTH blocks. Block 0
		 * is the live param_set_0 (immediate effect); the matched block is the
		 * shadow copy the dispatcher's .config_change path copies into block 0,
		 * so the candidate survives any later shape switch. */
		blk0[slot] = snap->values[i];
		mblk[slot] = snap->values[i];
	}
	/* Prefetch arrays: same dual-block write. The dispatcher reads pf_dist/
	 * pf_type ONLY on a level-2 cache miss (shape switch), which never fires
	 * while this tool hammers one shape -- so every changed slot is ALSO
	 * patched live through the exported SMC entries. Distance patches before
	 * type: the distance patcher restores the slot's backup bytes (re-enabling
	 * a NOPped slot), and set_prefetch_type skips a NOPped slot, so the order
	 * is what makes a type change on a previously-disabled slot stick. */
	int *dist0 = (int *)(g_shape_blocks + SB_PF_DIST_OFF);
	int *type0 = (int *)(g_shape_blocks + SB_PF_TYPE_OFF);
	int *distm = (int *)(g_matched_block + SB_PF_DIST_OFF);
	int *typem = (int *)(g_matched_block + SB_PF_TYPE_OFF);
	for (int s = 0; s < PF_SLOTS; s++)
	{
		int dist = snap->pf_dist[s];
		int type = snap->pf_type[s];
		int dist_changed = (dist != dist0[s]) || (dist != distm[s]);
		int type_changed = (type != type0[s]) || (type != typem[s]);
		dist0[s] = dist;
		distm[s] = dist;
		type0[s] = type;
		typem[s] = type;
		if (!dist_changed && !type_changed)
			continue; /* steady-state A/B alternation: most slots match, skip the serializing patch */
		if (dist_changed)
		{
			inference_dll_ptr.set_prefetch_distance(s, dist);
			/* Always re-establish the type after a distance patch: the
			 * distance patcher restores the slot's BACKUP bytes (original
			 * opcode) before writing the disp, so a dist-only change would
			 * otherwise leave the backup type live while the snapshot says
			 * otherwise. Skipped by the patcher itself when the new distance
			 * NOPped the slot (unsigned < 32) -- a disabled slot has no type. */
			inference_dll_ptr.set_prefetch_type(s, type);
		}
		else if (type_changed)
		{
			inference_dll_ptr.set_prefetch_type(s, type);
		}
	}
}

/* Read the CURRENT prefetch arrays from block 0 into snap. Called wherever a
 * snapshot is constructed by hand (the INI snapshot) so the
 * pf arrays always hold real loaded values -- an uninitialized pf array would
 * be applied verbatim by apply_snapshot and patch garbage distances. */
void snapshot_pf(param_snapshot *snap)
{
	int *dist0 = (int *)(g_shape_blocks + SB_PF_DIST_OFF);
	int *type0 = (int *)(g_shape_blocks + SB_PF_TYPE_OFF);
	for (int s = 0; s < PF_SLOTS; s++)
	{
		snap->pf_dist[s] = dist0[s];
		snap->pf_type[s] = type0[s];
	}
}

void snapshot_params(param_snapshot *snap)
{
	/* The dispatch param count is DISPATCH_PARAM_COUNT = 22, hardcoded to match
	 * v7_data.inc's dispatch table. That define is the ONE place carrying the
	 * contract: a DLL with a different param count is a rebuild mismatch that
	 * no client-side check can detect here (count is a compile-time constant),
	 * so the count and the sb_param_off_for_idx map must be updated together
	 * whenever v7_data.inc changes. */
	snap->count = DISPATCH_PARAM_COUNT;
	/* Read the LIVE values: all params from block 0's param_set array.
	 * Block 0 is the active param_set_0 the dispatcher reads. */
	int *param0 = (int *)(g_shape_blocks + SB_PARAM_OFF);
	for (int i = 0; i < snap->count; i++)
	{
		int slot = sb_param_off_for_idx[i];
		snap->values[i] = param0[slot];
	}
	snapshot_pf(snap); /* pf arrays ride in every snapshot so copies stay complete */
}

/* ======================================================================
 * Primary dispatch params to tune (indices match dispatch_param_names)
 * ====================================================================== */

/* === Cross-param invariant enforcement === */

#define MAX_JOBS 4096 /* matches ASM MAX_JOBS: num_chunks and split g1+g2 cap */

/* Enforce strictly ascending `min <= lo < mid < hi <= hi_max` on one threshold
 * triple. Sort first, then clamp DOWNWARD from the top (never raises anything,
 * single pass, cannot oscillate -- the old pairwise +1 cascade could terminate
 * with equal neighbours, e.g. (4095,4096,4096) -> (4095,4095,4096)). An
 * unsatisfiable triple (all three pinned at the floor) is a FATAL: silently
 * shipping a collapsed tier would defeat the invariant this function exists
 * to enforce. */
void fix_threshold_triple(int *lo, int *mid, int *hi, int hi_max, int lo_min);

/* === CPU prefetch-type support (CPUID), for the sanitizer's type map === */

int g_pf_type_ok[7];            /* 1 = this cpu executes the type's instruction */
int g_pf_type_supported[7];     /* the supported type ids (the sanitizer's draw pool) */
int g_pf_type_supported_count;  /* how many of the 7 types are supported */

int rng_int(int lo, int hi);    /* defined below; the sanitizer's draw needs it */
int pf_type_draw(void);         /* defined below (needs rng_int) */

/* Probe the prefetch features once at startup with CPUID. NTA/T0/T1/T2 are
 * SSE instructions and run on every x64 cpu. PREFETCHW is
 * CPUID.80000001H:ECX[8] (leaf 1 ECX[8] is THERMAL MONITOR 2, not a prefetch
 * flag). PREFETCHWT1 is CPUID.7.ECX[0] and CLDEMOTE is CPUID.7.ECX[25]. */
void detect_pf_support(void)
{
	int regs[4];
	g_pf_type_ok[0] = 1; /* prefetchnta: SSE */
	g_pf_type_ok[1] = 1; /* prefetcht0:  SSE */
	g_pf_type_ok[2] = 1; /* prefetcht1:  SSE */
	g_pf_type_ok[3] = 1; /* prefetcht2:  SSE */
	g_pf_type_ok[4] = 0;
	g_pf_type_ok[5] = 0;
	g_pf_type_ok[6] = 0;
	__cpuid(regs, 0);
	int max_basic = regs[0]; /* highest basic leaf from CPUID.00H:EAX */
	__cpuid(regs, 0x80000000);
	unsigned int max_ext = (unsigned int)regs[0]; /* highest extended leaf */
	if (max_basic >= 7)
	{
		int ext[4];
		__cpuidex(ext, 7, 0);
		g_pf_type_ok[5] = (ext[2] >> 0) & 1;  /* prefetchwt1: CPUID.7.ECX[0] */
		g_pf_type_ok[6] = (ext[2] >> 25) & 1; /* cldemote:    CPUID.7.ECX[25] */
	}
	if (max_ext >= 0x80000001u)
	{
		int ext[4];
		__cpuidex(ext, 0x80000001, 0);
		g_pf_type_ok[4] = (ext[2] >> 8) & 1; /* prefetchw: CPUID.80000001H:ECX[8] */
	}
	g_pf_type_supported_count = 0;
	for (int t = 0; t < 7; t++)
		if (g_pf_type_ok[t])
			g_pf_type_supported[g_pf_type_supported_count++] = t;
}

/* Enforce invariants on a CANDIDATE config that would otherwise FATAL or
 * silently misbehave in ASM. Only candidate configs pass through this --
 * probes, mutations, and --cfg. Baseline/original snapshots come straight from
 * the DLL and are already valid, so they must NOT be fed back through here.
 * Rules:
 *  - worker/caller batch thresholds must be strictly ascending (SELECT_BATCH_COMMON
 *    checks huge, then large, then medium, then small; descending order would
 *    make lower tiers unreachable)
 *  - num_chunks <= MAX_JOBS (ASM clamps too, but keep probes honest)
 *  - chunk split g1+g2 <= MAX_JOBS (ASM FATALs above this)
 *  - chunk split group counts must not exceed the number of columns in their
 *    group (ASM computes chunk_dim = floor(group_cols / count); if count >
 *    group_cols, some chunks get chunk_dim == 0 and the split path FATALs)
 *  - prefetch types the current cpu does not support (CPUID-probed at
 *    startup, detect_pf_support) become a random SUPPORTED type -- the
 *    engine's set_prefetch_type would patch an opcode the cpu raises #UD on
 *  - taper mode (taper_shift > 0) requires min_chunk_cols >= 4: the Q8
 *    4-row kernel block FATALs below that. A candidate drawn with
 *    taper on and min_chunk_cols 1-3 normalizes UP to 4 (nearest-legal);
 *    the explore space keeps min_chunk_cols 1-3 for taper_shift == 0
 *
 * snap->count is contractually <= TUNE_MAX_PARAMS (snapshot_params FATALs
 * otherwise), so no count bound is re-checked here -- the range loops below are
 * bounded by the real param count. */
void sanitize_candidate(param_snapshot *snap)
{
	/* Range-clamp every param that will actually be applied to its documented
	 * bounds. This is the single source of truth for out-of-range values coming
	 * from --cfg, INI, or random draws. Slots at/above count are never applied,
	 * so they are left untouched. */
	for (int i = 0; i < snap->count; i++)
	{
		if (snap->values[i] < tune_min[i])
			snap->values[i] = tune_min[i];
		if (snap->values[i] > tune_max[i])
			snap->values[i] = tune_max[i];
	}
	/* worker/caller batch thresholds must be strictly ascending (SELECT_BATCH_COMMON
	 * checks huge, then large, then medium, then small; descending order would
	 * make lower tiers unreachable). The whole-triple normalizer sorts and
	 * clamps downward from the top, guaranteeing lo < mid < hi in range for
	 * ANY input -- the old one-shot pairwise cascade could finish with equal
	 * neighbours (e.g. (4095,4096,4096) -> (4095,4095,4096)). count is always
	 * the full 22, so the triples are always fully present. */
	fix_threshold_triple(&snap->values[5], &snap->values[6], &snap->values[7], tune_max[7], tune_min[5]);
	fix_threshold_triple(&snap->values[14], &snap->values[15], &snap->values[16], tune_max[16], tune_min[14]);
	/* min_chunk_cols (idx 9) needs NO power-of-two normalization: the engine
	 * picks mask-vs-div AlignDown at runtime off a (W_min-1) & W_min test
	 * (matmul_dispatch.inc .taper_fill_common) -- a non-pow2 W_min takes the
	 * div/imul path and aligns exactly. Contracts: >= 1 always (0 is a
	 * creation FATAL), and >= 4 whenever the taper runs -- the ASM FATALs on
	 * a taper + W_min < 4 config, so a candidate with taper on normalizes
	 * min_chunk_cols UP to 4 here (nearest-legal; with taper off, 1-3 stay
	 * explorable). */
	if (snap->values[20] > 0 && snap->values[9] < 4)
		snap->values[9] = 4;
	/* Split fill (chunk_split_pct > 0) is legal now that the ASM routing bug is
	 * fixed, but the group counts must stay within their column budgets or the
	 * split path FATALs on chunk_dim <= 0 (matmul_dispatch.inc). Clamp both
	 * counts to the number of columns the percentage actually assigns. */
	if (snap->values[17] > 0)
	{
		int pct = snap->values[17];
		int g1 = snap->values[18];
		int g2 = snap->values[19];
		int g1_cols = dispatch_d * pct / 100;
		/* A pct that assigns group 1 ZERO columns (small d, tiny pct) cannot
		 * produce a legal split: every g1 chunk would get chunk_dim 0 and the
		 * dispatcher FATALs. Normalize the percentage UP to the smallest
		 * value that yields >= 1 column instead of keeping an illegal
		 * geometry (nearest-legal config, never a rejected trial). */
		int min_pct = (100 + dispatch_d - 1) / dispatch_d;
		if (g1_cols < 1 && min_pct <= 99)
		{
			pct = min_pct;
			g1_cols = dispatch_d * pct / 100;
			snap->values[17] = pct;
		}
		if (g1_cols < 1 || dispatch_d - g1_cols < 1)
		{
			/* Degenerate shapes (d < 2) admit NO valid split at any pct:
			 * the nearest-legal config is split OFF (the dispatcher never
			 * reads the group counts then). */
			snap->values[17] = 0;
		}
		else
		{
			if (g1 < 1)
				g1 = 1;
			if (g1 > g1_cols)
				g1 = g1_cols; /* each g1 chunk >= 1 col */
			/* pct capped at 99 above, so remaining = d - g1_cols >= 1 always. */
			int remaining = dispatch_d - g1_cols;
			if (g2 < 1)
				g2 = 1;
			if (g2 > remaining)
				g2 = remaining; /* each g2 chunk >= 1 col */
			/* g1+g2 needs no MAX_JOBS clamp: tune_max[18]/[19] cap both at
			 * 256, so the sum is <= 512 < 4096 by construction. */
			snap->values[18] = g1;
			snap->values[19] = g2;
		}
	}
	/* Prefetch types the current cpu cannot execute map to a random SUPPORTED
	 * type: the engine's set_prefetch_type would patch an opcode the cpu
	 * raises #UD on, killing the run at the first dispatch of the patched
	 * kernel. The supported set is probed once at startup (detect_pf_support).
	 * OWNED slots only: another kernel's slots are preserved EXACTLY -- this
	 * benchmark cannot evaluate them, and apply_snapshot patches all 41 into
	 * the live DLL, so remapping an unrelated slot would corrupt that
	 * kernel's tuned config. */
	for (int s = 0; s < PF_SLOTS; s++)
	{
		if (!g_pf_owned[s])
			continue;
		int t = snap->pf_type[s];
		if (t < PF_TYPE_MIN || t > PF_TYPE_MAX || !g_pf_type_ok[t])
			snap->pf_type[s] = pf_type_draw();
	}
}

/* === CPU prefetch-type support (CPUID), for the sanitizer's type map === */

int g_pf_type_ok[7];            /* 1 = this cpu executes the type's instruction */
int g_pf_type_supported[7];     /* the supported type ids (the sanitizer's draw pool) */
int g_pf_type_supported_count;  /* how many of the 7 types are supported */

/* One random SUPPORTED prefetch type (the sanitizer's replacement draw). The
 * SSE family is always supported, so the pool is never empty. */
int pf_type_draw(void)
{
	return g_pf_type_supported[rng_int(0, g_pf_type_supported_count - 1)];
}

/* Sort three values ascending. Used for the batch/caller threshold triples whose
 * SELECT_BATCH_COMMON semantics require strictly ascending tier boundaries. */
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

/* Enforce strictly ascending `lo_min <= lo < mid < hi <= hi_max` on one
 * threshold triple, TOTAL over every input: sort, raise the floor inward
 * (mid >= lo+1, hi >= mid+1), then push the ceiling down if the top ran past
 * hi_max, then re-check the floor (a single clamp pass in one direction only
 * could terminate invalid -- e.g. (1,1,1) or (1,2,2) hit the floor FATAL even
 * though (1,2,3) fits the [1,4096] range). Generated candidates (probes,
 * mutations, refine) MUST never abort the session, so the normalizer always
 * produces a valid triple; a committed INI that normalization would change is
 * still caught by the raw-vs-normalized equality check on probe #0. The final
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
		printf("[FATAL] sanitize_candidate: threshold range [%d,%d] cannot "
		       "hold three strictly ascending boundaries (%d,%d,%d).\n",
		       lo_min, hi_max, *lo, *mid, *hi);
		exit(EXIT_FAILURE);
	}
}

/* Re-sort the two ascending-threshold triples (batch_*_thresh idx 5-7 and
 * caller_*_thresh idx 14-16) in a snapshot after independent random draws or a
 * single-param mutation. Sorting at generation time preserves genuine spacing
 * between the three tiers -- unlike sanitize_candidate's additive +1 nudge,
 * which collapses out-of-order draws onto consecutive integers (large_thresh =
 * med_thresh + 1 squeezes the "large" tier to a single reachable column count,
 * so the tier-boundary logic goes untested). With generation-time sorting, the
 * sanitize cascade almost never fires. Slots beyond count are never applied,
 * so they are left untouched. */
void sort_threshold_triples(param_snapshot *snap)
{
	if (snap->count > 7)
		sort3(&snap->values[5], &snap->values[6], &snap->values[7]);
	if (snap->count > 16)
		sort3(&snap->values[14], &snap->values[15], &snap->values[16]);
}

/* Uniform integer draw in [lo,hi] (inclusive). The window spans up to 2^32
 * values (half of the weight-matrix bytes on huge shapes): frand() carries
 * only 23 random mantissa bits, so a double-scaled draw would quantize to
 * ~512-step granularity and most legal distances would be UNREACHABLE. Two
 * RDRAND words give a full 64-bit draw; the modulo bias is below 2^-32 of a
 * step and irrelevant next to RDRAND noise. */
int rng_int(int lo, int hi)
{
	__int64 span = (__int64)hi - (__int64)lo + 1;
	if (span <= 0)
		return lo;
	unsigned __int64 r = ((unsigned __int64)rng_u32() << 32) | (unsigned __int64)rng_u32();
	return (int)((__int64)lo + (__int64)(r % (unsigned __int64)span));
}

/* Probe draws are PLAIN UNIFORM (rng_int / the pf mix below):
 * no index-specific remapping, no anneal schedule, no delta-scale heuristics
 * -- random draws are never skewed. A mutation is a
 * fresh uniform draw over the field's legal window; local structure is the
 * refinement's job. */

/* Fill t with a FULLY RANDOM genome: all 22 dispatch params uniform over
 * their legal windows, all 41 pf distances uniform over the whole legal
 * window, all 41 pf types uniform. NOTHING is inherited from any prior
 * state -- in particular not from the DB-loaded block-0 values (DB values
 * never flow into the search, and no field is ever
 * seeded to 0). Threshold triples are sorted so the draw is a legal config;
 * inactive group-count fields are canonicalized to their floor so generated
 * candidates don't differ only in dead dimensions. */
void draw_random_snapshot(param_snapshot *t)
{
	t->count = DISPATCH_PARAM_COUNT;
	for (int param_index = 0; param_index < dispatch_nparams; param_index++)
	{
		int param_slot = tune_indices[param_index];
		t->values[param_slot] = rng_int(tune_min[param_slot], tune_max[param_slot]);
	}
	for (int pf_slot = 0; pf_slot < PF_SLOTS; pf_slot++)
	{
		if (!g_pf_owned[pf_slot])
		{
			/* Another kernel's slot: this benchmark cannot measure it, and
			 * apply_snapshot patches ALL slots into the live DLL -- so the
			 * trial must carry the COMMITTED value, never (0, 0) (which would
			 * disable another kernel's prefetches inside the running DLL). */
			t->pf_dist[pf_slot] = g_original.pf_dist[pf_slot];
			t->pf_type[pf_slot] = g_original.pf_type[pf_slot];
			continue;
		}
		t->pf_dist[pf_slot] = rng_int(pf_dist_lo, pf_dist_hi);
		t->pf_type[pf_slot] = rng_int(PF_TYPE_MIN, PF_TYPE_MAX);
	}
	sort_threshold_triples(t);
	sanitize_candidate(t);
}

/* Append a printf-formatted chunk to a row buffer; stops cleanly at capacity. */
void row_add(char *buf, int cap, int *len, const char *fmt, ...)
{
	va_list args;
	if (*len >= cap - 1)
		return;
	va_start(args, fmt);
	int n = vsnprintf(buf + *len, cap - *len, fmt, args);
	va_end(args);
	if (n < 0)
		return;
	if (*len + n >= cap - 1)
		*len = cap - 1; /* further appends see a full buffer and stop */
	else
		*len += n;
}

/* === Output: plain scrolling lines, one code path, no mode logic === */

void genome_block(const param_snapshot *g, int columns, char *out, int cap);
void print_genome(const param_snapshot *g, const char *label);
void screen_row(const char *fmt, ...);
#define screen_paint screen_row     /* budget messages ride the status line */

/* The 1-per-second print restriction: the tuner's console output is allowed
 * at most ONE print event per second total (the panes repaint quietly in
 * place on a real console; through pipes the throttle keeps the flood down
 * to one event per second). The simple check: time() != static var. */
int print_time_throttled(void)
{
	static int last_second = 0;
	int now = (int)time(0);
	if (now == last_second)
		return 1; /* the same second: this print is skipped */
	last_second = now;
	return 0;
}

/* One status/progress line. */
void screen_row(const char *fmt, ...)
{
	if (print_time_throttled())
		return;
	static char t[16384];
	va_list a; va_start(a, fmt); vsnprintf(t, sizeof(t), fmt, a); va_end(a);
	fputs(t, stdout);
	fflush(stdout);
}

/* ======================================================================
 * print_genome: 3-column dump of a full genome
 * ======================================================================
 * Builds "GENOME <label>" plus genome_block and prints it. Every phase
 * prints the BASELINE and the CANDIDATE genome through this function with
 * the measured GB/s in the label, so all phases read identically. */
void print_genome(const param_snapshot *g, const char *label)
{
	
	char row[8192];
	int row_len = 0;
	row_add(row, sizeof(row), &row_len, "GENOME %s\n", label);
	/* genome_block does not advance row_len: measure what it wrote with
	 * strlen, or the output would show the heading WITHOUT the genome. */
	int genome_len = row_len;
	genome_block(g, 3, row + genome_len, (int)sizeof(row) - genome_len);
	row_len = genome_len + (int)strlen(row + genome_len);
	fwrite(row, 1, row_len, stdout);
	fflush(stdout);
}

/* The per-comparison report EVERY phase prints: the measured baseline
 * genome, the measured candidate genome (measured GB/s in each label), then
 * the caller's verdict line. One shape of output everywhere. The whole
 * event -- BEST line, both genome dumps -- is ONE print event under the
 * 1-per-second restriction. */
void print_pair_gbs(const param_snapshot *base, double base_gbs,
                    const param_snapshot *trial, double trial_gbs,
                    const char *phase)
{
	if (print_time_throttled())
		return;
	gotoxy(0,50)
		;printf("BEST:%F   ",screen_best_gbs);
	char lb[128];
	gotoxy(0,4);
	snprintf(lb, sizeof(lb), "%s BASE %.1f GB/s", phase, base_gbs);
	print_genome(base, lb);
	snprintf(lb, sizeof(lb), "%s CANDIDATE %.1f GB/s", phase, trial_gbs);
	gotoxy(0,20);
	print_genome(trial, lb);
}

/* Render a genome into a compact text block in `columns` side-by-side
 * columns: the 22 dispatch params as "name = value", the 41 prefetch slots as
 * "pfNN dist=X type=T (NAME)", fixed-width cells. 3 columns is ~111 chars
 * wide, 2 columns ~74, 1 column ~37 -- paint() picks the widest mode that
 * fits the console without wrapping (a wrapped row breaks the fixed-height
 * pane). print_genome's redirected output uses 3. */
void genome_block(const param_snapshot *g, int columns, char *out, int cap)
{
	static const char *pf_type_names[7] = {
		"NTA", "T0", "T1", "T2", "PREFETCHW", "PREFETCHWT1", "CLDEMOTE"
	};
	int len = 0;
	/* dispatch params, `columns` per row. Fixed %9d value width: a huge value
	 * pads instead of shifting the columns that follow. */
	for (int i = 0; i < TUNE_MAX_PARAMS; i += columns)
	{
		for (int c = 0; c < columns && i + c < TUNE_MAX_PARAMS; c++)
			row_add(out, cap, &len, "%-25s = %9d  ",
				dispatch_param_names[i + c], g->values[tune_indices[i + c]]);
		row_add(out, cap, &len, "\n");
	}
	/* prefetch slots, `columns` per row -- DISPLAY only the selected kernel's
	 * OWNED slots (the mutation pool's scope); the real slot identifiers are
	 * kept (pf00, pf07, ...), no renumbering. Unowned slots stay in the
	 * snapshots and the INI, they are just not this run's business. Fixed
	 * %11d distance width (covers -2147483647) keeps t= and the type name in
	 * their columns for any legal distance. */
	for (int pos = 0; pos < g_pf_owned_count; pos += columns)
	{
		for (int c = 0; c < columns && pos + c < g_pf_owned_count; c++)
		{
			int s = g_pf_owned_list[pos + c];
			int t = g->pf_type[s];
			const char *tname = (t >= PF_TYPE_MIN && t <= PF_TYPE_MAX) ? pf_type_names[t] : "?";
			row_add(out, cap, &len, "pf%02d d=%11d t=%d %-11s  ",
				s, g->pf_dist[s], t, tname);
		}
		row_add(out, cap, &len, "\n");
	}
}

/* === Prefetch param draws (all 41 slots -- the whole parameter space) === */

/* The distance draw is a PLAIN UNIFORM rng_int over the whole legal window
 * [-(matrix bytes/2), +(matrix bytes/2)]. Nothing is special-cased here --
 * the disabled band [0..31] is the ENGINE's interpretation of the passed
 * value (set_prefetch_distance NOPs the slot), never the tuner's. */

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
int mutate_distance(int d, int w, int current_dist)
{
	__int64 wide = (__int64)d * w / 2;
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
	case 3:
	default:
	{
		/* Signed ratio: numerator and denominator are each drawn over the
		 * FULL SIGNED window, so the ratio lands on either side of zero and
		 * the mutation can walk a positive distance back toward 0 and below,
		 * not only grow it (the old draw was rnd=half*frand()/rnd2=half*frand()
		 * -- both factors positive, ratio always positive, every application
		 * pushed the distance UP until the window clamp). Double math: half
		 * exceeds 2^24 for large matrices, where float stops resolving
		 * integers. The denominator redraws until non-zero (a zero
		 * denominator makes the ratio infinite, not a ratio), and the ratio
		 * is clamped into the legal window in double space before the int
		 * cast so an enormous ratio cannot overflow the cast. The SUM is
		 * computed in 64-bit: current + ratio + jitter overflows a signed
		 * int for huge legal windows before the clamps below could run. */
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
		__int64 wide = (__int64)current_dist + (__int64)ratio + rng_int(-32, 32);
		if (wide > window_high)
			wide = window_high;
		if (wide < window_low)
			wide = window_low;
		new_distance = (int)wide;
	}
		break;
	case 4:
	{
		/* Positive-ratio step: numerator and denominator are drawn over
		 * 1..half (strictly positive -- no zero denominator, no redraw
		 * loop), the integer quotient gets a random sign, and the signed
		 * step shifts the current distance. A quotient of 1 nudges by one
		 * line, a quotient of half jumps straight toward the window edge,
		 * and the sign flip lets every size step land on either side. The
		 * sum is computed in 64-bit and clamped before the int cast: the
		 * current distance can sit at the window edge and a max quotient
		 * can push the sum past +2^31 for huge legal windows. */
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
	return new_distance;
}

/* === INI read/write (threading_proxy.ini) === */

/* Try to read one int key inside an INI section via the Windows INI API.
 * GetPrivateProfileIntA cannot distinguish "absent" from "invalid", and a 0
 * default would be indistinguishable from a legit 0 value, so the key is read
 * with GetPrivateProfileStringA using a sentinel default: the buffer equals the
 * sentinel iff the key is absent (the API also does the case-insensitive key
 * match for us). Returns 1 and fills *out on success; returns 0 when the key
 * is absent (a missing file or section reads as every key absent). A missing
 * key is NOT an error here -- the caller decides what absence means (the
 * tuner treats an incomplete [NxD] section as "no committed config" and seeds
 * from the uniform baseline). A key that IS
 * present but malformed stays a FATAL: a typo must not become a DIFFERENT
 * config that silently replaces the ruler. */
int ini_section_try_get_int(char *ini_path, char *section, char *key, int *out)
{
	/* 512 bytes: the committed INI carries LONG inline comments after each
	 * value ("num_chunks=752 ; Matmul chunk count ..."), and
	 * GetPrivateProfileStringA returns the whole line -- a 64-byte buffer
	 * truncated every such value at 63 chars. The number itself always leads
	 * the line, so after the comment strip below, truncation of the comment
	 * tail is harmless and needs no guard. */
	char buf[512];
	char absent[] = "__v7_ini_key_absent__";
	DWORD n = GetPrivateProfileStringA(section, key, absent, buf, sizeof(buf), ini_path);
	/* n == strlen(absent) and buf == sentinel => key not found. */
	if (n == strlen(absent) && strcmp(buf, absent) == 0)
		return 0;
	/* Strip a trailing inline comment (';' or '#') and surrounding whitespace:
	 * GetPrivateProfileStringA hands both through verbatim, and a strict
	 * strtol parse would otherwise reject the valid '752 ; comment' form. */
	char *cut = strpbrk(buf, ";#");
	if (cut)
		*cut = '\0';
	char *e = buf + strlen(buf);
	while (e > buf && (e[-1] == ' ' || e[-1] == '\t'))
		*--e = '\0';
	/* Strict decimal parse: strtol with full-consumption + range checks. atoi()
	 * silently maps 'abc' to 0, accepts '12junk' as 12, and wraps overflow --
	 * a typo would become a DIFFERENT config that silently replaces the ruler. */
	char *end = 0;
	errno = 0;
	long v = strtol(buf, &end, 10);
	if (end == buf || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX)
	{
		printf("[FATAL] ini_section_try_get_int: key '%s' in [%s] (%s) has "
		       "malformed or out-of-range value '%s'.\n",
		       key, section, ini_path, buf);
		exit(EXIT_FAILURE);
	}
	*out = (int)v;
	return 1;
}

/* Compare base/candidate with four uninterrupted A-B-A groups.
 * One settle and one baseline warmup happen before all measurements.
 * Each A/B/C position calls measure_gbs() exactly ONCE.
 * A, B, C are the respective BEST throughputs over the four groups.
 * A candidate wins only when B exceeds BOTH baseline bests.
	 *
	 * The ONE cooldown sleep runs here, before the warmup; there is no other
	 * sleep, warmup, logging, or screen work anywhere between the 12 timed
	 * measurements (applying a snapshot to select the config is the only
	 * permitted work between positions). Shared measurement primitive of every
	 * tuning phase (probe ranking, hill-climb scoring, refinement trials,
	 * final confirmation). */
double measure_ratio_best(param_snapshot *base, param_snapshot *trial,
                          double *base_out, double *cand_out)
{
	/* One settle and one little untimed warmup per comparison. */
	Sleep(g_cooldown_ms);
	apply_snapshot(base);
	dispatch_one();
	dispatch_one();

	double a = 0.0;
	double b = 0.0;
	double c = 0.0;

	for (int i = 0; i < 4; i++)
	{
		apply_snapshot(base);
		double ai = measure_gbs();

		apply_snapshot(trial);
		double bi = measure_gbs();

		apply_snapshot(base);
		double ci = measure_gbs();

		if (ai > a) a = ai;
		if (bi > b) b = bi;
		if (ci > c) c = ci;
	}

	double baseline_gbs = a > c ? a : c;

	if (baseline_gbs > screen_best_gbs)
		screen_best_gbs = baseline_gbs;
	if (b > screen_best_gbs)
		screen_best_gbs = b;



	*base_out = baseline_gbs;
	*cand_out = b;
	return b / baseline_gbs;
}

/* ======================================================================
 * Refinement: local coordinate search around the best candidate
 * ====================================================================== */

/* Legal window for ONE dispatch field at the CURRENT config: the tune-table
 * bounds, plus the per-column group budget while split mode is active.
 * Threshold members (5-7, 14-16) draw from their FULL documented range: the
 * triples are legal in any order after sort_threshold_triples, so restricting
 * a member between its current neighbours would fence the search inside the
 * incumbent's ordering and block valid re-orderings in a single step. */
void dispatch_field_bounds(int idx, const param_snapshot *snap, int *lo, int *hi)
{
	*lo = tune_min[idx];
	*hi = tune_max[idx];
	if (idx == 18 && snap->values[17] > 0)
	{
		int g1_cols = dispatch_d * snap->values[17] / 100;
		if (g1_cols < 1)
			g1_cols = 1; /* pct >= 1 on any d >= 1 assigns at least one column */
		if (g1_cols < *hi)
			*hi = g1_cols; /* each group-1 chunk >= 1 column */
	}
	else if (idx == 19 && snap->values[17] > 0)
	{
		int remaining = dispatch_d - dispatch_d * snap->values[17] / 100;
		if (remaining < 1)
			remaining = 1; /* pct <= 99 leaves group 2 at least d/100 >= 1 columns */
		if (remaining < *hi)
			*hi = remaining; /* each group-2 chunk >= 1 column */
	}
}

/* Coordinate-descent refinement over the tuned params starting from best_snapshot.
 * bigram_best_ratio/bigram_best_base_gbs/bigram_best_cand_gbs are in/out. The ruler
 * seed set in main() always leaves a populated best_snapshot, so the incumbent
 * re-measure below always scores a real config. */
void refine_candidate(void)
{
	/* Re-measure the incumbent once so the header names the baseline the
	 * trials below are paired against (same protocol as every trial). The
	 * comparison is a pair (ruler vs incumbent) and prints as one. */
	double incumbent_med = measure_ratio_best(&g_uniform_snap, &best_snapshot, &bigram_best_base_gbs, &bigram_best_cand_gbs);
	print_pair_gbs(&g_uniform_snap, bigram_best_base_gbs, &best_snapshot, bigram_best_cand_gbs, "REFINE");
	int improved = 1;
	int refine_iters = 0;
	gotoxy(0, 0);
	screen_row("REFINEMENT (local search, adopt any measured gain > 1.0; incumbent %.3f)", incumbent_med);
	/* Coordinate descent around the best candidate. The inner pass loop used to
	 * be a fixed 3-pass re-run even after the descent converged -- that tripled
	 * runtime for nothing on a near-optimal config. Now a pass that adopts a
	 * gain immediately re-explores from the improved state (next while-iter),
	 * and a full pass with no gains exits immediately. REFINE_MAX_ITERS is a
	 * hard cap so thermal noise hovering near the +2% adopt threshold can never
	 * churn the search forever. */
	int budget_out = 0;
	while (improved && !budget_out)
	{
		improved = 0;
		for (int pass = 0; pass < REFINE_MAX_PASSES && !budget_out; pass++)
		{
			int pass_improved = 0;
			for (int param_index = 0; param_index < dispatch_nparams && !budget_out; param_index++)
			{
				/* --runtime budget check before each coordinate trial: every
				 * candidate below is a parameter change, so the per-param
				 * head is the regular checkpoint. Expiry stops the descent;
				 * best_snapshot already holds the current best. */
				if (runtime_expired())
				{
					screen_paint("  -- runtime budget exhausted -- refinement stopped, keeping current best --\n");
					budget_out = 1;
					break;
				}
				int param_slot = tune_indices[param_index];
				/* Split group counts participate ALWAYS: a trial can enable
				 * split AND set its group counts in the same round (the
				 * sanitizer normalizes the geometry), and with split off a
				 * count-only trial measures an identical config and simply
				 * loses -- excluding the fields would make a better split
				 * combination unreachable. */
				int current_value = best_snapshot.values[param_slot];
				char *param_name = dispatch_param_names[param_slot];
				int lowest_legal;
				int highest_legal;
				dispatch_field_bounds(param_slot, &best_snapshot, &lowest_legal, &highest_legal);
				int candidate_values[8], num_candidates = 0;
				{
					/* Nearest-neighbour candidates +-2..+2 including the
					 * zero-offset trial: no range-derived step scale, no
					 * magic divisor, no exclusions. */
					for (int neighbour_step = -2; neighbour_step <= 2; neighbour_step++)
					{
						int new_value = current_value + neighbour_step;
						if (new_value < lowest_legal)
							new_value = lowest_legal;
						if (new_value > highest_legal)
							new_value = highest_legal;
						candidate_values[num_candidates++] = new_value;
					}
					/* Geometric neighbours: half and double of the current
					 * value, clamped into the legal window. A +-1/+-2 step can
					 * never cross a wide field (num_chunks 3970 -> 12 is two
					 * thousand additive steps, but nine halvings), so the
					 * local search also walks multiplicatively -- the same
					 * shape mutate_distance uses for pf distances. */
					for (int geometric_step = 0; geometric_step < 2; geometric_step++)
					{
						int new_value = geometric_step == 0 ? current_value / 2 : current_value * 2;
						if (new_value < lowest_legal)
							new_value = lowest_legal;
						if (new_value > highest_legal)
							new_value = highest_legal;
						candidate_values[num_candidates++] = new_value;
					}
				}
				for (int candidate_index = 0; candidate_index < num_candidates; candidate_index++)
				{
					/* Budget check before EVERY trial measurement (one param
					 * can carry up to six candidates): the documented
					 * before-every-parameter-change contract. */
					if (runtime_expired())
					{
						screen_paint("  -- runtime budget exhausted -- refinement stopped, keeping current best --\n");
						budget_out = 1;
						break;
					}
					int new_value = candidate_values[candidate_index];
					param_snapshot trial = best_snapshot;
					trial.values[param_slot] = new_value; /* in-bounds by construction: no normalization, no silent rewrite */
					/* The same legality pass the mutation phase applies: a
					 * pct step re-scales the dependent group counts, a
					 * threshold step can reorder the triple -- the trial is
					 * normalized BEFORE it reaches the DLL. */
					sort_threshold_triples(&trial);
					sanitize_candidate(&trial);
					double base_gbs;
					double trial_gbs;
					double trial_ratio = measure_ratio_best(&best_snapshot, &trial, &base_gbs, &trial_gbs);
					print_pair_gbs(&best_snapshot, base_gbs, &trial, trial_gbs, "REFINE");
					if (trial_ratio > 1.0)
					{
						int old_value = best_snapshot.values[param_slot]; /* capture BEFORE the adoption: the message must name the real old->new change */
						bigram_best_cand_gbs = trial_gbs;
						bigram_best_base_gbs = base_gbs;
						best_snapshot = trial;
						baseline_update_count++;
						save_winner_genome(trial_gbs);
						improved = 1;
						pass_improved = 1;
						/* ONE verdict line per trial, WITH the measured GB/s. */
						gotoxy(0, 33);
						screen_row("REFINE %s %d->%d  base %.1f  trial %.1f  ratio %.3f  ACCEPT",
						           param_name, old_value, new_value, base_gbs, trial_gbs, trial_ratio);
						break; /* leave this param's candidate scan: the remaining offsets were derived from the pre-adoption value; the pass loop re-scans from the updated best */
					}
					gotoxy(0, 33);
					screen_row("REFINE %s %d->%d  base %.1f  trial %.1f  ratio %.3f  REJECT",
					           param_name, best_snapshot.values[param_slot], new_value, base_gbs, trial_gbs, trial_ratio);
				}
			}
			/* Prefetch coordinate pass: OWNED slots only (another kernel's
			 * slots cannot influence this benchmark). Distance coordinate
			 * candidates are uniform draws over the whole legal window -- NO
			 * step size, NO granularity -- and type coordinates step +-1;
			 * adopt rule and paired protocol identical. */
			for (int pf_slot = 0; pf_slot < PF_SLOTS && !budget_out; pf_slot++)
			{
				if (!g_pf_owned[pf_slot])
					continue;
				if (runtime_expired())
				{
					screen_paint("  -- runtime budget exhausted -- refinement stopped, keeping current best --\n");
					budget_out = 1;
					break;
				}
				int current_distance = best_snapshot.pf_dist[pf_slot];
				int distance_candidates[4], num_candidates = 0;
				for (int draw = 0; draw < 4; draw++)
				{
					distance_candidates[num_candidates++] = mutate_distance(dispatch_n, dispatch_d, current_distance); /* the ONLY legal distance mutation */
				}
				for (int candidate_index = 0; candidate_index < num_candidates; candidate_index++)
				{
					if (runtime_expired())
					{
						screen_paint("  -- runtime budget exhausted -- refinement stopped, keeping current best --\n");
						budget_out = 1;
						break;
					}
					param_snapshot trial = best_snapshot;
					trial.pf_dist[pf_slot] = distance_candidates[candidate_index];
					sort_threshold_triples(&trial);
					sanitize_candidate(&trial); /* owned pf types stay supported; dispatch geometry stays legal */
					double base_gbs;
					double trial_gbs;
					double trial_ratio = measure_ratio_best(&best_snapshot, &trial, &base_gbs, &trial_gbs);
					print_pair_gbs(&best_snapshot, base_gbs, &trial, trial_gbs, "REFINE");
					if (trial_ratio > 1.0)
					{
						int old_distance = best_snapshot.pf_dist[pf_slot]; /* capture BEFORE the adoption: the message names the real old->new */
						bigram_best_cand_gbs = trial_gbs;
						bigram_best_base_gbs = base_gbs;
						best_snapshot = trial;
						baseline_update_count++;
						save_winner_genome(trial_gbs);
						improved = 1;
						pass_improved = 1;
						gotoxy(0, 33);
						screen_row("REFINE pf_dist_%d %d->%d  base %.1f  trial %.1f  ratio %.3f  ACCEPT",
						           pf_slot, old_distance, distance_candidates[candidate_index], base_gbs, trial_gbs, trial_ratio);
						break;
					}
					gotoxy(0, 33);
					screen_row("REFINE pf_dist_%d %d->%d  base %.1f  trial %.1f  ratio %.3f  REJECT",
					           pf_slot, best_snapshot.pf_dist[pf_slot], distance_candidates[candidate_index], base_gbs, trial_gbs, trial_ratio);
				}
				/* Prefetch TYPES are categorical instruction choices, not an
				 * ordered axis: numeric adjacency (current +- 1) cannot step
				 * over an unsupported type. Each trial draws a random
				 * CPU-SUPPORTED type instead -- every supported type stays
				 * reachable from any current value. */
				for (int type_trial = 0; type_trial < 2; type_trial++)
				{
					if (runtime_expired())
					{
						screen_paint("  -- runtime budget exhausted -- refinement stopped, keeping current best --\n");
						budget_out = 1;
						break;
					}
					int new_type = pf_type_draw();
					param_snapshot trial = best_snapshot;
					trial.pf_type[pf_slot] = new_type;
					sort_threshold_triples(&trial);
					sanitize_candidate(&trial); /* owned pf types stay supported; dispatch geometry stays legal */
					double base_gbs;
					double trial_gbs;
					double trial_ratio = measure_ratio_best(&best_snapshot, &trial, &base_gbs, &trial_gbs);
					print_pair_gbs(&best_snapshot, base_gbs, &trial, trial_gbs, "REFINE");
					if (trial_ratio > 1.0)
					{
						int old_type = best_snapshot.pf_type[pf_slot]; /* capture BEFORE the adoption */
						bigram_best_cand_gbs = trial_gbs;
						bigram_best_base_gbs = base_gbs;
						best_snapshot = trial;
						baseline_update_count++;
						save_winner_genome(trial_gbs);
						improved = 1;
						pass_improved = 1;
						gotoxy(0, 33);
						screen_row("REFINE pf_type_%d %d->%d  base %.1f  trial %.1f  ratio %.3f  ACCEPT",
						           pf_slot, old_type, new_type, base_gbs, trial_gbs, trial_ratio);
						break;
					}
					gotoxy(0, 33);
					screen_row("REFINE pf_type_%d %d->%d  base %.1f  trial %.1f  ratio %.3f  REJECT",
					           pf_slot, best_snapshot.pf_type[pf_slot], new_type, base_gbs, trial_gbs, trial_ratio);
				}
			}
			if (!pass_improved)
				break; /* full pass with no gains: converged */
		}
		if (++refine_iters >= REFINE_MAX_ITERS)
		{
			screen_paint("  (refinement iteration cap reached: %d)\n", REFINE_MAX_ITERS);
			break;
		}
	}
	/* bigram_best_ratio is NOT written here: the fresh final confirmation
	 * overwrites it after this phase, and the compounded intermediate value
	 * is search bookkeeping, not a report. */
}

/* ======================================================================
 * Final confirmation (best-of-7 interleaved)
 * ====================================================================== */

/* INI key names for the genome archive: [0, 22) are the dispatch params,
 * [22, 22+41) the pf_dist_N keys, [22+41, 22+82) the pf_type_N keys. The
 * genome files carry the 22 dispatch keys PLUS the prefetch keys of the
 * kernel the run actually benchmarked (slot map: q8 0-7, f32 8-10, fused
 * 11-12, swiglu f32 13-26, swiglu q8 27-38) -- the OTHER kernels' prefetch
 * keys are not this run's business. The best_gbs record key carries the
 * winning GB/s for cross-run analysis. */
#define INI_KEY_COUNT (TUNE_MAX_PARAMS + 2 * PF_SLOTS)

/* Fill out with the INI key name for flat key index i (see INI_KEY_COUNT). */
void ini_key_name(int i, char *out)
{
	if (i < TUNE_MAX_PARAMS)
	{
		strcpy(out, dispatch_param_names[i]);
		return;
	}
	i -= TUNE_MAX_PARAMS;
	if (i < PF_SLOTS)
		sprintf(out, "pf_dist_%d", i);
	else
		sprintf(out, "pf_type_%d", i - PF_SLOTS);
}

/* ======================================================================
 * Winner-genome archive
 * ====================================================================== */

/* Save the just-adopted winning genome (best_snapshot) to
 * <winner-dir>/<shape>_<counter>.ini: a complete [NxD] section with all 22
 * dispatch params plus this kernel's OWNED pf keys and the winning GB/s as
 * the best_gbs record. The file is a standalone INI that test_shape_speed
 * measures directly via --ini <file> -- no DB rebuild involved. The counter
 * makes every winner a separate file, so the archive keeps the full winning
 * lineage of a run, not just the last one. Recording is gated by
 * --winner-dir: without it the function returns and the run only counts
 * baseline updates. Plain stdio writes the whole file in ONE pass: the
 * Windows profile API re-reads and rewrites the entire file PER KEY, which
 * cannot keep up with hundreds of winners per tune (each save would cost
 * ~39 full-file rewrites, usually re-scanned by real-time antivirus). The
 * written format is exactly what the INI readers load back. */
void save_winner_genome(double win_gbs)
{
	if (g_winner_dir[0] == '\0')
		return; /* no --winner-dir given: this run counts only */
	static int counter_seeded = 0;
	if (!counter_seeded)
	{
		counter_seeded = 1;
		/* Continue the archive numbering: the files are contiguous
		 * <section>_1.ini .. <section>_N.ini by construction, so scanning to
		 * the first missing index finds the highest existing winner. A
		 * second collection into the same directory then appends new files
		 * instead of overwriting the earlier run's genomes. */
		for (int n = 1;; n++)
		{
			char probe_path[2 * MAX_PATH];
			snprintf(probe_path, sizeof(probe_path), "%s\\%s_%d.ini", g_winner_dir, g_section, n);
			if (GetFileAttributesA(probe_path) == INVALID_FILE_ATTRIBUTES)
				break;
			g_winner_counter = n;
		}
	}
	CreateDirectoryA(g_winner_dir, NULL); /* already exists -> ERROR_ALREADY_EXISTS, ignored */
	char path[2 * MAX_PATH];
	snprintf(path, sizeof(path), "%s\\%s_%d.ini", g_winner_dir, g_section, ++g_winner_counter);
	FILE *ini = fopen(path, "w");
	if (!ini)
	{
		printf("  (warning: winner genome %s cannot be opened for writing, error %lu)\n", path, GetLastError());
		return;
	}
	fprintf(ini, "[%s]\n", g_section);
	char keyname[32];
	char val[16];
	for (int i = 0; i < TUNE_MAX_PARAMS; i++)
	{
		ini_key_name(i, keyname);
		snprintf(val, sizeof(val), "%d", best_snapshot.values[i]);
		fprintf(ini, "%s=%s\n", keyname, val);
	}
	for (int slot = 0; slot < PF_SLOTS; slot++)
	{
		if (!g_pf_owned[slot])
			continue; /* another kernel's prefetch slots are not this run's business */
		fprintf(ini, "pf_dist_%d=%d\n", slot, best_snapshot.pf_dist[slot]);
		fprintf(ini, "pf_type_%d=%d\n", slot, best_snapshot.pf_type[slot]);
	}
	fprintf(ini, "best_gbs=%.1f\n", win_gbs);
	fclose(ini);
	printf("  WINNER GENOME saved: %s (best %.1f GB/s)\n", path, win_gbs);
	fflush(stdout);
}

/* ======================================================================
 * Probe phase: random search over the parameter space before hill-climbing
 * ====================================================================== */

/* Random search over the tuned params. Every probe is paired against the
 * CURRENT BEST with the tool's one measurement protocol (four A-B-A groups,
 * one decision): probe 1 pairs the INI ruler (or the frozen random genome R)
 * against a random probe, later probes pair the running best. A probe
 * becomes the seed only by winning its paired comparison. The best probe
 * seeds both the hill-climb candidate and the refinement coordinate search,
 * so hill-climbing starts inside a promising region.
 *
 * Outputs (all in the g_* globals, candidate_snapshot is purely out):
 *   candidate_snapshot -- hill-climb start: the best probe sample (or the
 *                  ruler when every probe measured slower).
 *   best_snapshot    -- the winning config for the phases that follow. */
void probe_phase(void)
{
	gotoxy(0, 0);
	screen_row("RANDOM PROBE (%d random genomes, baseline = %s)", g_probe_n, g_have_ini_config ? "INI config" : "frozen random genome R");
	best_snapshot = bigram_base_snap;
	int best_is_ruler = 1;
	/* The probes only SEED the search: at most a quarter of the --runtime
	 * budget goes to probing, so the hill-climb -- the phase that actually
	 * collects genome wins -- always keeps its share. Unlimited runs (no
	 * --runtime) probe through the full probe count. */
	double probe_deadline = (g_runtime_minutes > 0)
	                        ? g_deadline - 0.75 * (double)g_runtime_minutes * 60.0
	                        : 0.0;
	for (int p = 0; p < g_probe_n; p++)
	{
		/* probe budget check BEFORE each probe draw: a huge --probe count
		 * must not stretch the run past the probe share of the budget.
		 * Expiry abandons the remaining probes; the measured seed from the
		 * probes run so far is what the next phases start from. */
		if (probe_deadline > 0.0 && get_seconds() >= probe_deadline)
		{
			gotoxy(0, 33);
			screen_row("  -- probe share of the runtime budget spent after %d of %d probes --\n", p, g_probe_n);
			break;
		}
		param_snapshot trial;
		draw_random_snapshot(&trial); /* all 22 dispatch params + owned pf values drawn, nothing inherited */
		/* Paired A/B against the CURRENT BEST, remeasured on the base side
		 * of every probe: the one acceptance rule of the whole tool (the
		 * candidate must beat the freshly remeasured baseline). Probe 1's
		 * baseline IS the INI ruler -- same protocol as every other probe. */
		double bs;
		double ts;
		double ratio = measure_ratio_best(&best_snapshot, &trial, &bs, &ts);
		print_pair_gbs(&best_snapshot, bs, &trial, ts, "PROBE");
		gotoxy(0, 33);
		char row[16384];
		int row_len = 0;
		row_add(row, sizeof(row), &row_len, "PROBE %-4d base %8.1f  trial %8.1f  ratio %.3f  %s",
		        p + 1, bs, ts, ratio, ts > bs ? "ACCEPT" : "REJECT");
		if (ts > bs)
		{
			best_snapshot = trial;
			best_is_ruler = 0;
			baseline_update_count++;
			save_winner_genome(ts);
			screen_row("%s\n", row);
		}
		else
			screen_row("%s\n", row);
	}
	/* Seed adaptive rounds + refinement from the best probe sample. The ruler
	 * seed from main() guarantees best_snapshot is populated even when every
	 * probe loses, so the hill-climb is never seeded from an unmeasured
	 * config. */
	candidate_snapshot = best_snapshot;
	gotoxy(0, 35);
	screen_row("  best probe: %s", best_is_ruler ? "the ruler (no probe measured faster)" : "random probe");
}

/* ======================================================================
 * Main A/B loop: best-of-N interleaved hill-climb
 * ====================================================================== */

/* Hill-climb: every round mutates ONE uniformly-picked field and measures it
 * against the live baseline. A measured win (ratio > 1.0) ratchets the
 * candidate in as the new live baseline; a measured loss changes NOTHING --
 * there is NO revert, because a single measurement error must never kick the
 * walk or the proven best config anywhere. best_snapshot tracks the
 * best-ever MEASURED config by absolute best-of-N throughput -- every
 * round's candidate side is a fresh measurement in the same units, so the
 * fastest config actually seen wins regardless of which baseline it was
 * paired against (the old compounded-ratio tracker multiplied noisy ratios
 * into a running product and picked the luckiest estimate, not the fastest
 * config). best_snapshot is what the winner archive saves. Every ratchet
 * prints the winner's genome with print_genome (multi-line, described).
 * bigram_base_snap is in/out (ratcheted on wins); best_snapshot/
 * bigram_best_cand_gbs/bigram_best_base_gbs are in/out, seeded by the probe
 * phase and re-anchored fresh at climb start. */
void hill_climb(void)
{
	/* The climb's ACCEPTED BASELINE is the best probe, not the original
	 * ruler: phase one adopted best_snapshot on absolute throughput, so
	 * starting the walk from the ruler would let a config that is worse than
	 * the best probe get accepted merely by beating the ruler. There is no
	 * separate "incumbent re-measure" here: the first round measures the
	 * baseline as the A/B base side, exactly like every other round. */
	bigram_base_snap = best_snapshot;
	gotoxy(0, 0);
	screen_row("HILL-CLIMB");
	for (int round = 0; round < g_max_rounds; round++)
	{
		/* --runtime budget check BEFORE the mutation below: each round changes
		 * one field, so the round head is exactly the "before every parameter
		 * change" checkpoint. Expiry abandons the climb; best_snapshot already
		 * holds the current best for the archive. */
		if (runtime_expired())
		{
			screen_paint("  -- runtime budget exhausted before round %d -- keeping current best --\n", round + 1);
			break;
		}

		/* MUTATE FIRST, measure second: round 0 would otherwise re-measure the
		 * unmutated seed (a full round of cooldowns spent on a config already
		 * scored). */

		/* 1..10 MUTATIONS per round, on DISTINCT fields of a COPY OF THE
		 * CURRENT ACCEPTED BASELINE: every trial is built from
		 * bigram_base_snap, never from a cumulative carrier -- rejected
		 * rounds must not accumulate, or later rounds would test compounded
		 * changes instead of fresh mutations of the incumbent. The field
		 * count is honest: a repeated pick is redrawn, so the promised 1..10
		 * is what actually changes. Every active dispatch param, every pf
		 * distance and every pf type carries equal weight; each mutation is
		 * a PLAIN UNIFORM redraw over the field's legal window, then the same
		 * legality pass every generated genome goes through. */
		param_snapshot trial = bigram_base_snap;
		char mut_desc[1024];                   /* the round's mutation list (status line) */
		int md = 0;
		mut_desc[0] = '\0';
		int active_params[TUNE_MAX_PARAMS];
		int num_active_params = 0;
		for (int param_index = 0; param_index < dispatch_nparams; param_index++)
		{
			/* ALL dispatch params are pickable, including the split group
			 * counts while split is off: a trial can enable split AND set its
			 * group counts in one round -- excluding 18/19 would make better
			 * split combinations unreachable through an accepted intermediate
			 * state. With split off a count-only trial measures an identical
			 * config and simply loses. */
			active_params[num_active_params++] = tune_indices[param_index];
		}
		int mutations = rng_int(1, 10);    /* fields changed this round */
		for (int m = 0; m < mutations; m++)
		{
			char fname[32];
			int old_v = 0;
			int new_v = 0;
			/* The field is a PLAIN UNIFORM draw over the full space: 22
			 * dispatch params, then one id per OWNED pf distance, then one id
			 * per OWNED pf type -- so a round can change pf_dist_3 AND
			 * pf_type_3 together. No distinctiveness machinery: the same
			 * field may be drawn twice and that is a legal mutation. */
			int field_pick = rng_int(0, num_active_params + 2 * g_pf_owned_count - 1);
			if (field_pick < num_active_params)
			{
				int param_slot = active_params[field_pick];
				int lowest_legal;
				int highest_legal;
				dispatch_field_bounds(param_slot, &trial, &lowest_legal, &highest_legal);
				int new_value;
				if (rng_int(0, 3) < 3)
				{
					/* 75%: NUDGE the current value by a sign-flipped integer
					 * ratio of the field's span instead of redrawing it --
					 * the quotient of two uniform 1..span draws is usually 1,
					 * so most mutations walk +-1 line from the incumbent and
					 * a few jump far, keeping the probe near the ruler while
					 * never losing reach. Clamped into the legal window (a
					 * nudge at the window edge would otherwise exit it). */
					int span = 1 + (highest_legal - lowest_legal > 0 ? highest_legal - lowest_legal : lowest_legal - highest_legal);
					int step = rng_int(1, span) / rng_int(1, span);
					if (rng_int(0, 1))
						step = -step;
					new_value = trial.values[param_slot] + step;
					if (new_value < lowest_legal)
						new_value = lowest_legal;
					if (new_value > highest_legal)
						new_value = highest_legal;
				}
				else
				{
					/* 25%: the full uniform redraw over the legal window. */
					new_value = rng_int(lowest_legal, highest_legal);
				}
				snprintf(fname, sizeof(fname), "%s", dispatch_param_names[param_slot]);
				old_v = trial.values[param_slot];
				trial.values[param_slot] = new_value;
				new_v = new_value;
			}
			else if (field_pick < num_active_params + g_pf_owned_count)
			{
				int pf_slot = g_pf_owned_list[field_pick - num_active_params];
				snprintf(fname, sizeof(fname), "pf_dist_%d", pf_slot);
				old_v = trial.pf_dist[pf_slot];
				int new_distance = mutate_distance(dispatch_n, dispatch_d, trial.pf_dist[pf_slot]);
				trial.pf_dist[pf_slot] = new_distance;
				new_v = new_distance;
			}
			else
			{
				int pf_slot = g_pf_owned_list[field_pick - num_active_params - g_pf_owned_count];
				/* Draw directly from the CPU-SUPPORTED set: an unsupported
				 * type cannot reach the DLL (the sanitizer remaps it), and
				 * drawing from the supported pool keeps every mutation
				 * executable. */
				int new_type = pf_type_draw();
				snprintf(fname, sizeof(fname), "pf_type_%d", pf_slot);
				old_v = trial.pf_type[pf_slot];
				trial.pf_type[pf_slot] = new_type;
				new_v = new_type;
			}
			md += snprintf(mut_desc + md, sizeof(mut_desc) - md, "%s%s %d->%d",
			               md ? ", " : "", fname, old_v, new_v);
			if (md >= (int)sizeof(mut_desc) - 2)
				break; /* the status line's mutation list is full; the mutations applied stand */
		}
		sort_threshold_triples(&trial); /* a threshold mutation can unorder the triple */
		sanitize_candidate(&trial);     /* keep cross-parameter constraints legal */

		/* Interleaved A/B: the candidate is measured AND the baseline is
		 * remeasured, best (max) of N samples per side. */
		double base_gbs;
		double cand_gbs;
		double ratio = measure_ratio_best(&bigram_base_snap, &trial, &base_gbs, &cand_gbs);
		print_pair_gbs(&bigram_base_snap, base_gbs, &trial, cand_gbs, "HILL");

		/* The top-line number is the MEASURED MAXIMUM -- a NUMBER only. It
		 * never pins a genome: speed drift makes an old high number
		 * meaningless for the current walk. */
		if (base_gbs > screen_best_gbs)
			screen_best_gbs = base_gbs;
		if (cand_gbs > screen_best_gbs)
			screen_best_gbs = cand_gbs;

		/* Status row: WHICH fields changed, old -> new, and the verdict. */
		gotoxy(0, 33);
		char row[16384];
		int row_len = 0;
		row_add(row, sizeof(row), &row_len, "HILL %-4d (%d) %s  base %8.1f  trial %8.1f  ratio %.3f  %s",
		        round + 1, mutations, mut_desc, base_gbs, cand_gbs, ratio,
		        cand_gbs > base_gbs ? "ACCEPT" : "REJECT");

		/* THE ONLY new-best / new-baseline rule in the whole code: the
		 * candidate was measured, the baseline was remeasured, and the
		 * candidate beat the remeasured baseline. No session-max comparison,
		 * no additional logic of any kind -- the absolute maximum above is a
		 * number, never a genome pin. */
		if (cand_gbs > base_gbs)
		{
			bigram_base_snap = trial; /* becomes the new baseline ... */
			best_snapshot = trial;    /* ... and the new best */
			baseline_update_count++;
			save_winner_genome(cand_gbs);
			row_add(row, sizeof(row), &row_len, "  +BEST+RATCHET\n");
			screen_row("%s", row);
			continue; /* the accepted row is painted; the pair above shows both genomes */
		}
		screen_row("%s\n", row);
	}
}

/* ======================================================================
 * Main
 * ====================================================================== */

/* Parsed CLI options, filled by parse_cli() and consumed by main(). */
typedef struct
{
	char *db_path;      /* --db (required) */
	char *ini_path;     /* --ini, normalized to absolute (no default) */
	char kernel[32];    /* --kernel: q8|f32|fused|swiglu_f32|swiglu_q8 ("" = --q8/--f32) */
	int precision;      /* 1=Q8 (default), 0=F32 */
	int shape_n;        /* matmul rows (default 1024) */
	int shape_d;        /* matmul cols (default 1024) */
	double target_time; /* --time (accepted for CLI compatibility, ignored: no calibration loop exists) */
	int fixed_iters;    /* --iterations (accepted for CLI compatibility, ignored) */
	int cooldown_ms;    /* --cooldown: ms between measurements (clamped >=0) */
	int max_rounds;     /* --rounds */
	int runtime_minutes; /* --runtime: budget in minutes (0 = unlimited) */
	int probe_n;        /* --probe */
	int median_n;       /* --median (accepted for CLI compatibility, ignored: the A-B-A protocol fixes its own structure) */
} cli_options;

/* Parse CLI options into o. Returns 0 on success; 1 if a specific error was
 * already printed (missing --db, invalid --ini path). Invalid numeric ranges
 * (--cooldown < 0, --probe < 1) are FATAL exits, never clamped. */

/* Strict signed-decimal int parse for a CLI token: the WHOLE token must
 * consume, no overflow. atoi() silently mapped '--rounds abc' to 0 -- which
 * then skipped the hill-climb and still wrote the INI back. */
int parse_cli_int(char *tok, char *flag)
{
	char *end = 0;
	errno = 0;
	long v = strtol(tok, &end, 10);
	if (tok == end || (end && *end != '\0') || errno == ERANGE || v < INT_MIN || v > INT_MAX)
	{
		printf("[FATAL] %s: malformed or out-of-range integer '%s'\n", flag, tok);
		exit(EXIT_FAILURE);
	}
	return (int)v;
}

/* Strict double parse for a CLI token: whole token consumed, result finite. */
double parse_cli_double(char *tok, char *flag)
{
	char *end = 0;
	errno = 0;
	double v = strtod(tok, &end);
	if (tok == end || (end && *end != '\0') || !isfinite(v))
	{
		printf("[FATAL] %s: malformed or non-finite number '%s'\n", flag, tok);
		exit(EXIT_FAILURE);
	}
	return v;
}

int parse_cli(int argc, char **argv, cli_options *o)
{
	o->db_path = 0;
	o->ini_path = 0;
	o->kernel[0] = '\0';
	o->precision = 1;
	o->shape_n = 1024;
	o->shape_d = 1024;
	o->target_time = 0.1;
	o->fixed_iters = 0;	o->cooldown_ms = 20;
	o->max_rounds = 60;
	o->runtime_minutes = 0;
	o->probe_n = 30;
	o->median_n = 3;
	int shape_seen = 0;

	for (int i = 2; i < argc; i++)
	{
		if ((strcmp(argv[i], "--winner-dir") == 0) && (i + 1 < argc))
		{
			/* Normalize to absolute: the ANSI profile APIs resolve relative
			 * paths against the Windows directory, not the process CWD. */
			char abs[MAX_PATH];
			DWORD dir_len = GetFullPathNameA(argv[++i], (DWORD)sizeof(abs), abs, 0);
			if (dir_len == 0 || dir_len >= sizeof(abs))
			{
				printf("[FATAL] --winner-dir: path invalid or too long: %s\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			snprintf(g_winner_dir, sizeof(g_winner_dir), "%s", abs);
			CreateDirectoryA(g_winner_dir, NULL); /* exists -> ERROR_ALREADY_EXISTS, ignored: the archive dir exists from startup on */
		}
		else if ((strcmp(argv[i], "--db") == 0) && (i + 1 < argc))
			o->db_path = argv[++i];
		else if ((strcmp(argv[i], "--q8") == 0))
			o->precision = 1;
		else if ((strcmp(argv[i], "--f32") == 0))
			o->precision = 0;
		else if ((strcmp(argv[i], "--kernel") == 0) && (i + 1 < argc))
		{
			char *k = argv[++i];
			if (strlen(k) >= sizeof(o->kernel))
			{
				printf("[FATAL] --kernel name too long: %s\n", k);
				exit(EXIT_FAILURE);
			}
			strcpy(o->kernel, k);
		}
		else if ((strcmp(argv[i], "--time") == 0) && (i + 1 < argc))
			o->target_time = parse_cli_double(argv[++i], "--time");
		else if ((strcmp(argv[i], "--iterations") == 0) && (i + 1 < argc))
			o->fixed_iters = parse_cli_int(argv[++i], "--iterations");
		else if ((strcmp(argv[i], "--cooldown") == 0) && (i + 1 < argc))
			o->cooldown_ms = parse_cli_int(argv[++i], "--cooldown");
		else if ((strcmp(argv[i], "--rounds") == 0) && (i + 1 < argc))
			o->max_rounds = parse_cli_int(argv[++i], "--rounds");
		else if ((strcmp(argv[i], "--runtime") == 0) && (i + 1 < argc))
			o->runtime_minutes = parse_cli_int(argv[++i], "--runtime");
		else if ((strcmp(argv[i], "--probe") == 0) && (i + 1 < argc))
			o->probe_n = parse_cli_int(argv[++i], "--probe");
		else if ((strcmp(argv[i], "--median") == 0) && (i + 1 < argc))
			o->median_n = parse_cli_int(argv[++i], "--median");
		else if ((strcmp(argv[i], "--ini") == 0) && (i + 1 < argc))
			o->ini_path = argv[++i];
		else
		{
			/* Shape token: N:M or NxM, parsed with strtol so an overflowing
			 * %d cannot silently retarget the run (e.g. 4294967297x1024),
			 * requiring the WHOLE token to be the shape, and rejecting a
			 * second shape token. Upper sanity bound fails fast before the
			 * tensor allocation. A token starting with '-' is a flag typo or
			 * a flag missing its value -- named as such, not as a shape. */
			char tok[64];
			if (argv[i][0] == '-')
			{
				printf("[FATAL] unrecognized option '%s' (or it is missing its value)\n", argv[i]);
				exit(EXIT_FAILURE);
			}
			snprintf(tok, sizeof(tok), "%s", argv[i]); /* parse a copy: the shape path never mutates argv, so error paths cite the original token (--cfg does write NULs into its token, which is safe: nothing re-reads argv after parse_cli) */
			char *sep = strchr(tok, 'x');
			if (!sep)
				sep = strchr(tok, ':');
			int pn = 0;
			int pd = 0;
			if (sep && sep != tok)
			{
				*sep = '\0';
				char *n_end = 0;
				char *d_end = 0;
				errno = 0;
				long nv = strtol(tok, &n_end, 10);
				long dv = strtol(sep + 1, &d_end, 10);
				/* The explicit <= 1<<20 range check also catches ERANGE
				 * overflow (LONG_MAX >> 20), so errno need not be re-tested
				 * after the second parse overwrote it. */
				int ok = (n_end != tok && *n_end == '\0' &&
				          d_end && *(sep + 1) != '\0' && *d_end == '\0' &&
				          nv > 0 && nv <= (1 << 20) && dv > 0 && dv <= (1 << 20));
				if (ok)
				{
					pn = (int)nv;
					pd = (int)dv;
				}
				if (pn > 0 && pd > 0)
				{
					if (shape_seen)
					{
						printf("[FATAL] multiple shape tokens ('%s' after %dx%d)\n", argv[i], o->shape_n, o->shape_d);
						exit(EXIT_FAILURE);
					}
					o->shape_n = pn;
					o->shape_d = pd;
					shape_seen = 1;
				}
				else
				{
					printf("[FATAL] unrecognized or out-of-range shape token '%s' (each dimension must be 1..%d)\n", argv[i], 1 << 20);
					exit(EXIT_FAILURE);
				}
			}
			else
			{
				printf("[FATAL] unrecognized shape token '%s'\n", argv[i]);
				exit(EXIT_FAILURE);
			}
		}
	}

	/* The shape is the tuning target -- a silent 1024x1024 default would tune
	 * (and write back a config for) the wrong shape with no diagnostic. */
	if (!shape_seen)
	{
		printf("Fail: <shape N:M> is required\n");
		return 1;
	}

	/* A negative --cooldown would wrap to 0xFFFFFFFF in the DWORD Sleep() arg,
	 * i.e. INFINITE -- a hard hang. Invalid input must FATAL, not clamp. */
	if (o->cooldown_ms < 0)
	{
		printf("[FATAL] --cooldown must be >= 0 (got %d)\n", o->cooldown_ms);
		exit(EXIT_FAILURE);
	}

	/* --iterations/--time are accepted for CLI compatibility and ignored:
	 * measure_gbs() times exactly ONE dispatched call (no calibration loop
	 * exists). Both keep their old range checks so legacy command lines
	 * fail loudly on garbage values. */
	if (o->fixed_iters < 0)
	{
		printf("[FATAL] --iterations must be >= 0 (got %d)\n", o->fixed_iters);
		exit(EXIT_FAILURE);
	}

	/* A --rounds of 0 (or any value that atoi() mapped to 0, like 'abc')
	 * silently skips the hill-climb while refine/confirm still run -- the
	 * tool would archive an untuned genome as if it were a tuning result.
	 * The tuner needs at least one adaptive round. */
	if (o->max_rounds < 1)
	{
		printf("[FATAL] --rounds must be >= 1 (got %d)\n", o->max_rounds);
		exit(EXIT_FAILURE);
	}

	/* A negative --runtime would put the deadline before the first probe and
	 * end every run before tuning began. 0 stays the unlimited sentinel. */
	if (o->runtime_minutes < 0)
	{
		printf("[FATAL] --runtime must be >= 0 minutes (0 = unlimited; got %d)\n", o->runtime_minutes);
		exit(EXIT_FAILURE);
	}

	/* --time is ignored (no calibration loop); range-checked so legacy
	 * command lines fail loudly on garbage values. */
	if (!(o->target_time > 0.0) || o->target_time > 3600.0)
	{
		printf("[FATAL] --time must be finite and in (0, 3600] seconds (got %g)\n", o->target_time);
		exit(EXIT_FAILURE);
	}

	/* The probe phase always runs and probe #0 is always measured (the ruler
	 * itself, against which every drawn probe competes by absolute
	 * throughput), so --probe < 1 has no meaning -- it would draw no random
	 * genome and leave the hill-climb with only the ruler seed. */
	if (o->probe_n < 1)
	{
		printf("[FATAL] --probe must be >= 1 (got %d)\n", o->probe_n);
		exit(EXIT_FAILURE);
	}

	/* --median is accepted for CLI compatibility and IGNORED: every
	 * comparison runs the fixed four-group A-B-A protocol (one measure_gbs()
	 * per position), so there is no configurable sample count left. */
	if (o->median_n < 1 || o->median_n > MEASURE_MAX_SAMPLES)
	{
		printf("[FATAL] --median must be in [1,%d] (got %d)\n", MEASURE_MAX_SAMPLES, o->median_n);
		exit(EXIT_FAILURE);
	}

	if (!o->db_path)
	{
		printf("Fail: --db <db_path> is required\n");
		return 1;
	}

	/* --ini is REQUIRED (no committed config to pair against and no [NxD]
	 * section to write back to without one). */
	if (!o->ini_path)
	{
		printf("Fail: --ini <path> is required for tuning (no default)\n");
		return 1;
	}

	if (o->ini_path)
	{
		/* Normalize the INI path to absolute: GetPrivateProfileStringA
		 * resolves relative paths against the Windows directory, not the
		 * process CWD, so the read side needs a full path. The buffer MUST
		 * be heap-owned: g_ini_path outlives parse_cli, and a stack array
		 * would dangle into every later use. The FILE may not exist -- an
		 * absent INI just means the ruler is the frozen random genome R,
		 * which is not a parse error. */
		char ini_abs[_MAX_PATH];
		DWORD r = GetFullPathNameA(o->ini_path, (DWORD)sizeof(ini_abs), ini_abs, 0);
		if (r == 0 || r >= sizeof(ini_abs))
		{
			printf("Fail: --ini path invalid or too long: %s\n", o->ini_path);
			return 1;
		}
		o->ini_path = strdup(ini_abs);
		if (!o->ini_path)
		{
			printf("Fail: out of memory duplicating the --ini path.\n");
			return 1;
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	system("cls");	
	ZydisInit();
	setvbuf(stdout, NULL, _IOFBF, 1 << 16); /* full buffering: frames go out as single writes, no per-line console churn */
	detect_pf_support(); /* CPUID probe: which prefetch types this cpu executes (the sanitizer maps the rest away) */

	/* Above-normal process priority keeps the measurement cadence steady when
	 * other user processes compete for CPU. Process-level only -- the DLL's
	 * worker-priority handling is untouched. Failure is a warning, not a
	 * FATAL: tuning at normal priority is valid, just noisier. */
	if (!SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS))
		printf("(warning: SetPriorityClass(ABOVE_NORMAL_PRIORITY_CLASS) failed, error %lu -- continuing at normal priority)\n", GetLastError());

	/* argv[1] is the DLL path -- must exist before parse_cli touches argv[2..].
	 * Without this check, running the tool with no args passes 0 to v7_load
	 * instead of printing a clean usage message. */
	if (argc < 2)
	{
		char *prog_name = argv[0];
		if (!prog_name)
			prog_name = "autotune_dispatch";
		printf("usage: %s <v7_threading.dll> --db <db_path> --ini <ini_path> [options] <NxM>\n", prog_name);
		return 1;
	}

	cli_options opt;
	int parse_rc = parse_cli(argc, argv, &opt);
	if (parse_rc)
		return 1; /* parse_cli already printed the specific error */

	char *dll_path = argv[1];

	/* Load DLL (v7_load zeroes the struct itself) */
	if (v7_load(dll_path, &inference_dll_ptr) != 0)
		return 1;
	if (!inference_dll_ptr.v7_init(opt.db_path) || inference_dll_ptr.v7_engine_is_dead())
	{
		/* v7_init returns 0 and/or parks the engine dead on a DB failure --
		 * tuning on from there would measure empty gate returns and commit
		 * that garbage to the INI. */
		printf("[FATAL] v7_init failed for %s (engine dead: %d)\n", opt.db_path, inference_dll_ptr.v7_engine_is_dead());
		v7_unload(&inference_dll_ptr);
		return 1;
	}

	/* The engine's own A/B/A tuner must never fire during a tuning run: its
	 * delay comes from the DB key tuner:delaydefault (read by the tune_init
	 * script at chat time). Push the key to 99999 in the loaded table and
	 * persist it when the stored value differs -- a tuner that armed would
	 * face an unreachable match count, and a tuning run never chats, so the
	 * in-process tuner never initializes here at all. */
	{
		unsigned __int64 delay_len = 0;
		const char *stored_delay = (const char *)inference_dll_ptr.v7_db_find("tuner:delaydefault", &delay_len);
		int delay_ok = stored_delay && delay_len == 6 && memcmp(stored_delay, "99999", 6) == 0;
		if (!delay_ok)
		{
			if (!inference_dll_ptr.v7_db_add("tuner:delaydefault", (void *)"99999", 6, 1))
				printf("  (warning: v7_db_add tuner:delaydefault failed -- the in-memory delay stays at the stored value)\n");
			else if (!inference_dll_ptr.v7_db_save(opt.db_path))
				printf("  (warning: v7_db_save failed -- tuner:delaydefault set for this process only)\n");
			else
				printf("  tuner:delaydefault = 99999 persisted to %s (the in-process tuner cannot interfere)\n", opt.db_path);
		}
		fflush(stdout);
	}

	/* Resolve shape_blocks export so we can write the matched block too
	 * (keeps autotuner candidates alive across dispatcher .config_change copies).
	 * Missing export is a FATAL: apply_snapshot() hard-requires the matched block
	 * for its dual-block write, so tuning cannot proceed without it. */
	g_shape_blocks = (unsigned char *)GetProcAddress((HMODULE)inference_dll_ptr.module, "shape_blocks");
	if (!g_shape_blocks)
	{
		printf("[FATAL] shape_blocks export not found in %s; dual-block "
		       "write impossible, autotuner cannot run.\n",
		       dll_path);
		v7_unload(&inference_dll_ptr);
		return 1;
	}

	use_q8_mode = (opt.precision == 1);
	/* Resolve the tuned kernel: --kernel names it explicitly; without it the
	 * legacy --q8/--f32 flags select q8/f32. Each kernel owns exactly its
	 * slot range (docs/tuner_slots.md slot map); only owned slots are drawn,
	 * mutated, or archived. */
	{
		const char *k = opt.kernel;
		if (k[0] == '\0')
			g_kernel = use_q8_mode ? KERN_Q8 : KERN_F32;
		else if (strcmp(k, "q8") == 0)
			g_kernel = KERN_Q8;
		else if (strcmp(k, "f32") == 0)
			g_kernel = KERN_F32;
		else if (strcmp(k, "fused") == 0)
			g_kernel = KERN_FUSED;
		else if (strcmp(k, "swiglu_f32") == 0)
			g_kernel = KERN_SWIGLU_F32;
		else if (strcmp(k, "swiglu_q8") == 0)
			g_kernel = KERN_SWIGLU_Q8;
		else
		{
			printf("[FATAL] --kernel '%s' is not one of q8|f32|fused|swiglu_f32|swiglu_q8 "
			       "(attention is not tunable yet: it needs a RunState harness).\n", k);
			v7_unload(&inference_dll_ptr);
			return 1;
		}
		switch (g_kernel)
		{
		case KERN_Q8:
			g_kernel_label = "q8";
			g_pf_owned_count = 8;
			for (int s = 0; s < 8; s++)
			{
				g_pf_owned[s] = 1;
				g_pf_owned_list[s] = s;
			}
			break;
		case KERN_F32:
			g_kernel_label = "f32";
			g_pf_owned_count = 3;
			for (int s = 0; s < 3; s++)
			{
				g_pf_owned[8 + s] = 1;
				g_pf_owned_list[s] = 8 + s;
			}
			break;
		case KERN_FUSED:
			g_kernel_label = "fused";
			g_pf_owned_count = 2;
			for (int s = 0; s < 2; s++)
			{
				g_pf_owned[11 + s] = 1;
				g_pf_owned_list[s] = 11 + s;
			}
			break;
		case KERN_SWIGLU_F32:
			g_kernel_label = "swiglu_f32";
			g_pf_owned_count = 14;
			for (int s = 0; s < 14; s++)
			{
				g_pf_owned[13 + s] = 1;
				g_pf_owned_list[s] = 13 + s;
			}
			break;
		default: /* KERN_SWIGLU_Q8 */
			g_kernel_label = "swiglu_q8";
			g_pf_owned_count = 12;
			for (int s = 0; s < 12; s++)
			{
				g_pf_owned[27 + s] = 1;
				g_pf_owned_list[s] = 27 + s;
			}
			break;
		}
		use_q8_mode = (g_kernel == KERN_Q8 || g_kernel == KERN_SWIGLU_Q8);
	}
	dispatch_n = opt.shape_n;
	dispatch_d = opt.shape_d;
	/* Legal distance window from the tuned shape: +/-(weight-matrix bytes/2).
	 * The weight matrix is the benchmark's NxD int8 block (n*d bytes); the
	 * __int64 product guards a pathological 1M x 1M CLI shape, and the clamp
	 * keeps the window inside the signed disp32 budget. */
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
	g_cooldown_ms = opt.cooldown_ms;
	/* --median is accepted for CLI compatibility and IGNORED (see parse_cli). */
	g_probe_n = opt.probe_n;
	g_max_rounds = opt.max_rounds;
	g_runtime_minutes = opt.runtime_minutes;
	g_ini_path = opt.ini_path;
	snprintf(g_section, sizeof(g_section), "%dx%d", dispatch_n, dispatch_d);

	/* Allocate ONLY the primary pool the tuned kernel actually reads
	 * (fused/swiglu modes use their own g_kx and g_kernw/g_gatew/g_upw
	 * buffers below): populating an NxD tensor costs one RDRAND draw per
	 * element. */
	if (g_kernel == KERN_Q8)
	{
		tensor_q8_alloc(&g_tsq, dispatch_n, dispatch_d);
		tensor_q8_fill(&g_tsq, dispatch_n, dispatch_d);
	}
	if (g_kernel == KERN_F32)
	{
		tensor_f32_alloc(&g_tsf, dispatch_n, dispatch_d);
		tensor_f32_fill(&g_tsf, dispatch_n, dispatch_d);
	}
	g_xout = (float *)malloc(dispatch_d * sizeof(float));
	if (!g_xout)
	{
		printf("[FATAL] main: out of memory for the %d-float output row.\n", dispatch_d);
		v7_unload(&inference_dll_ptr);
		return 1;
	}

	/* Per-kernel scratch tensors. The activations are [n x n] (token count n,
	 * input feature dim n -- the shape token's N names BOTH for the FFN
	 * kernels), the swiglu/fused weights are [d x n], the outputs [n x d].
	 * Over-allocation is harmless; under-allocation would let the kernel read
	 * past the buffer. OOM is a FATAL, same as the pools above. */
	__int64 kx_elems = (__int64)dispatch_n * dispatch_n;
	__int64 kw_elems = (__int64)dispatch_n * dispatch_d;
	if (g_kernel == KERN_FUSED || g_kernel == KERN_SWIGLU_F32 || g_kernel == KERN_SWIGLU_Q8)
	{
		g_kout = (float *)malloc(kw_elems * sizeof(float));
		if (!g_kout)
		{
			printf("[FATAL] main: out of memory for the kernel output (%lld floats).\n", (long long)kw_elems);
			v7_unload(&inference_dll_ptr);
			return 1;
		}
	}
	if (g_kernel == KERN_FUSED || g_kernel == KERN_SWIGLU_F32)
	{
		g_kx_f32 = (float *)malloc(kx_elems * sizeof(float));
		if (!g_kx_f32)
		{
			printf("[FATAL] main: out of memory for the f32 activation (%lld floats).\n", (long long)kx_elems);
			v7_unload(&inference_dll_ptr);
			return 1;
		}
		init_random_float(g_kx_f32, kx_elems);
	}
	if (g_kernel == KERN_SWIGLU_Q8)
	{
		g_kx_q8 = (char *)malloc(kx_elems);
		g_kx_q8_s = (float *)malloc((kx_elems / 64 + 1) * sizeof(float));
		if (!g_kx_q8 || !g_kx_q8_s)
		{
			printf("[FATAL] main: out of memory for the q8 activation (%lld bytes).\n", (long long)kx_elems);
			v7_unload(&inference_dll_ptr);
			return 1;
		}
		init_random_int8(g_kx_q8, kx_elems);
		for (__int64 i = 0; i <= kx_elems / 64; i++)
			g_kx_q8_s[i] = 1.0f;
	}
	if (g_kernel == KERN_FUSED)
	{
		tensor_q8_alloc(&g_kernw, dispatch_n, dispatch_d);
		tensor_q8_fill(&g_kernw, dispatch_n, dispatch_d);
		g_normw = (float *)malloc(dispatch_n * sizeof(float));
		if (!g_normw)
		{
			printf("[FATAL] main: out of memory for the fused norm weights (%d floats).\n", dispatch_n);
			v7_unload(&inference_dll_ptr);
			return 1;
		}
		init_random_float(g_normw, dispatch_n);
	}
	if (g_kernel == KERN_SWIGLU_F32 || g_kernel == KERN_SWIGLU_Q8)
	{
		tensor_q8_alloc(&g_gatew, dispatch_n, dispatch_d);
		tensor_q8_fill(&g_gatew, dispatch_n, dispatch_d);
		tensor_q8_alloc(&g_upw, dispatch_n, dispatch_d);
		tensor_q8_fill(&g_upw, dispatch_n, dispatch_d);
	}

	/* Spin mode: workers park on pause-loops instead of sleeping, so the
	 * dispatched benchmarks see steady latency (production inference runs
	 * the same way). This is scheduling, not measurement -- the A-B-A
	 * protocol settles clocks itself before each comparison. */
	inference_dll_ptr.v7s_set_spin(1);

	/* Shape activation: the dispatcher creates a shape's per-shape block
	 * LAZILY from the DB's __ini__ keys on the FIRST dispatch at that shape.
	 * One untimed dispatch here materializes the block for (n,d) so
	 * find_matched_block below can pin it. This is not a warmup (timed
	 * comparisons warm themselves inside the A-B-A protocol) and not a
	 * measurement -- it is the dispatcher's own block-creation contract. */
	dispatch_one();

	/* The per-shape block now exists and block 0 is settled to the matched
	 * config for (n,d). Pin g_matched_block so apply_snapshot() shadows both.
	 * g_shape_blocks is guaranteed non-0 here (FATAL'd above), and
	 * find_matched_block FATALs if the DB lacks a per-shape block for (n,d). */
	g_matched_block = find_matched_block();
	{
		int blk_idx = (int)((g_matched_block - g_shape_blocks) / SB_SIZE);
		printf("shape_blocks: tuned block index = %d (0=active,1=default,2+=per-shape) for %dx%d\n", blk_idx, dispatch_n, dispatch_d);
	}

	/* Save original params (restore at end) */
	snapshot_params(&g_original);
	/* Number of dispatch params. Every probe/mutation/display loop is bounded by
	 * this count (== DISPATCH_PARAM_COUNT == 22), never by the fixed array width
	 * -- slots at or above count would hold uninitialized data and must never be
	 * written by apply_snapshot. */
	dispatch_nparams = g_original.count;

	/* INI probe #0: read the current [NxD] section from the INI. The FILE, the
	 * section, or individual keys may not exist -- any absence marks the
	 * committed config missing (the ruler is then the frozen random genome
	 * R). Keys that ARE present must be valid: an
	 * out-of-window prefetch value or an invariant-violating dispatch value
	 * stays a fix-the-INI FATAL. */
	g_have_ini_config = 1;
	{
		g_ini_snap = g_original; /* the FULL original snapshot: unowned pf slots keep the committed values this kernel never touches */
		for (int i = 0; i < g_ini_snap.count && g_have_ini_config; i++)
		{
			int v;
			if (!ini_section_try_get_int(g_ini_path, g_section, dispatch_param_names[i], &v))
				g_have_ini_config = 0; /* file, section, or this key does not exist yet */
			else
				g_ini_snap.values[i] = v;
		}
		/* Prefetch keys: only the OWNED slots are required and read. The
		 * scoped writer saves just these (a q8 run never touches f32/swiglu
		 * prefetch keys), so requiring all 82 would call every second run
		 * incomplete and throw away the committed config. Unowned slots keep
		 * whatever the ORIGINAL snapshot carries -- another kernel's values,
		 * preserved verbatim. Distances accept anything in
		 * [pf_dist_lo..pf_dist_hi] verbatim -- disabled slots (unsigned 0..31)
		 * and negative (backward-prefetch) values are legal committed states,
		 * never rewritten. Only a PRESENT owned value outside the window is a
		 * fix-the-INI FATAL. */
		for (int s = 0; s < PF_SLOTS && g_have_ini_config; s++)
		{
			if (!g_pf_owned[s])
				continue;
			char keyname[32];
			int v;
			snprintf(keyname, sizeof(keyname), "pf_dist_%d", s);
			if (!ini_section_try_get_int(g_ini_path, g_section, keyname, &v))
			{
				g_have_ini_config = 0;
				break;
			}
			if (v < pf_dist_lo || v > pf_dist_hi)
			{
				printf("[FATAL] INI [%s] %s=%d outside [%d..%d] -- the legal "
				       "distance window is +/-(weight-matrix bytes / 2) for this "
				       "shape; fix the committed INI.\n",
				       g_section, keyname, v, pf_dist_lo, pf_dist_hi);
				exit(EXIT_FAILURE);
			}
			g_ini_snap.pf_dist[s] = v;
			snprintf(keyname, sizeof(keyname), "pf_type_%d", s);
			if (!ini_section_try_get_int(g_ini_path, g_section, keyname, &v))
			{
				g_have_ini_config = 0;
				break;
			}
			if (v < PF_TYPE_MIN || v > PF_TYPE_MAX)
			{
				printf("[FATAL] INI [%s] %s=%d outside [%d..%d] -- unknown prefetch type; fix the committed INI.\n",
				       g_section, keyname, v, PF_TYPE_MIN, PF_TYPE_MAX);
				exit(EXIT_FAILURE);
			}
			g_ini_snap.pf_type[s] = v;
		}
		if (g_have_ini_config)
		{
			/* The INI [NxD] config is the ruler and must be the config the
			 * file actually commits: if normalization would change ANY
			 * value, the committed INI violates the documented invariants
			 * -- fix the file instead of silently measuring a different
			 * config. EXCEPTIONS: with chunk_split_pct == 0 the split group
			 * counts are dead (the dispatcher never reads them; sanitize
			 * pins them to their floor), and an unsupported prefetch TYPE
			 * is remapped by the sanitizer to a random supported one -- both
			 * are reported, not fatal. */
			param_snapshot raw_ini = g_ini_snap;
			sort_threshold_triples(&g_ini_snap);
			sanitize_candidate(&g_ini_snap);
			for (int s = 0; s < PF_SLOTS; s++)
				if (raw_ini.pf_type[s] != g_ini_snap.pf_type[s])
					printf("ini: [%s] pf_type_%d = %d is not supported on this cpu -- remapped to %d\n",
					       g_section, s, raw_ini.pf_type[s], g_ini_snap.pf_type[s]);
			for (int i = 0; i < g_ini_snap.count; i++)
			{
				if (raw_ini.values[i] == g_ini_snap.values[i])
					continue;
				if ((i == 18 || i == 19) && raw_ini.values[17] == 0)
					continue;
				printf("[FATAL] INI [%s] %s=%d violates the documented invariants "
				       "(normalized value would be %d) -- fix the committed INI; "
				       "probe #0 must measure the config as committed.\n",
				       g_section, dispatch_param_names[i], raw_ini.values[i], g_ini_snap.values[i]);
				exit(EXIT_FAILURE);
			}
			printf("ini: %s section [%s] loaded from INI (the ruler)\n", g_ini_path, g_section);
		}
		else
		{
			printf("ini: %s section [%s] missing or incomplete -- the ruler is a frozen random genome R\n", g_ini_path, g_section);
		}
	}

	/* The ruler (A/B yardstick): NEVER a nulled config -- a zeroed reference
	 * sanitizes the search's anchor. It is either the
	 * committed INI config (tuner-written, never zero-init) or a frozen
	 * random genome R drawn from the full legal space. No nulled ruler of
	 * any kind exists: not taper_shift=0, not zeroed pf arrays. */
	if (g_have_ini_config)
	{
		bigram_base_snap = g_ini_snap;
	}
	else
	{
		draw_random_snapshot(&bigram_base_snap); /* R, frozen for the whole run */
	}
	best_snapshot = bigram_base_snap;
	bigram_best_ratio = 1.0; /* the ruler against itself, honestly */
	/* Immutable reference for refinement + final confirmation. base_snap may
	 * be ratcheted to a winning candidate; g_uniform_snap never changes, so
	 * the reported final ratio is always honest against the ruler. */
	g_uniform_snap = bigram_base_snap;
	candidate_snapshot = bigram_base_snap;

	printf("autotune %s: %dx%d cooldown=%dms rounds=%d runtime=%dmin\n", g_kernel_label, dispatch_n, dispatch_d, g_cooldown_ms, g_max_rounds, g_runtime_minutes);
	/* Setup facts are in the 64 KB stdout buffer: flush them NOW so a hard
	 * death inside a phase (DLL FATAL -> TerminateProcess) cannot drop them. */
	fflush(stdout);

	/* Start the --runtime clock when tuning begins: DLL load, tensor fill, and
	 * setup are not tuning. The probe phase runs to completion
	 * regardless (it seeds the search); hill-climb and refinement check
	 * runtime_expired() before every parameter change. */
	g_deadline = get_seconds() + (double)g_runtime_minutes * 60.0;

	/* === Random probe phase: sample the parameter space before hill-climbing === */
	probe_phase();

	/* === Main A/B loop (best-of-N interleaved hill-climb) === */
	hill_climb();

	/* === Refinement: local coordinate search around best candidate === */
	refine_candidate();


	/* ALWAYS leave a genome behind: a run where nothing beat the ruler
	 * (strong INI champion, tiny --runtime budget) would otherwise archive
	 * nothing and the sorter would have nothing to load. bigram_best_cand_gbs
	 * is the measured throughput of best_snapshot from the refine entry
	 * measurement, so the record number is real even when no adoption
	 * happened in the whole run (best_snapshot is then the ruler itself). */
	if (g_winner_dir[0] != '\0' && g_winner_counter == 0)
		save_winner_genome(bigram_best_cand_gbs);

	/* The threading_proxy.ini is NEVER written by this tool: it is the
	 * read-only ruler the search measures against. The winner-genome
	 * archive (--winner-dir) holds every adopted config as a standalone
	 * INI; transplanting a champion into the proxy INI is update_ini_
	 * champions.py's job. */
	printf("\n  baseline genome updated %ld time(s) across all phases (probe + hill-climb + refine)\n",
	       baseline_update_count);

/* Cleanup */
	free(g_xout);
	if (use_q8_mode)
		tensor_q8_free(&g_tsq);
	else
		tensor_f32_free(&g_tsf);
	v7_unload(&inference_dll_ptr);
	return 0;
}
