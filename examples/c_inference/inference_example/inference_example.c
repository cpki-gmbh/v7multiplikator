// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* inference_example.c -- minimal v7 inference example with performance statistics
 *
 * usage:  inference_example.exe <dll_path> <db_path> [-q8] [-i "prompt"] [-n <tokens>]
 *                [-r <rounds>] [-think <0|1>] [-rewrite] [-script <path>]
 *   dll_path -- path to v7_threading.dll (REQUIRED, no default)
 *   db_path  -- path to lz4ht.db (REQUIRED, no default)
 *   -q8      -- use Q8 quantized activation mode (default: F32)
 *   -i       -- prompt text (default: "What is the capital of france?")
 *   -n       -- generation token budget (default: 0 = unlimited, run until EOS)
 *   -r       -- number of rounds to run (default: 1). each round resets the
 *               statistics block and prints a fresh [perf] report, so extra
 *               rounds act as burn-in: later rounds show steady-state numbers
 *               free of cold-cache / OS scheduler effects.
 *   -think   -- thinking-template selector (default: 1). 0 renders turns with
 *               the base (non-thinking) templates via v7s_enable_thinking(0);
 *               no-op for models whose DB stores no @think keys.
 *   -rewrite -- enable sector_logo sidecar hook (on-token rewriting).
 *               calls v7s_sectorlogo_set_db() before context creation.
 *   -script  -- load a sector_logo sidecar script from <path>. the file
 *               contents are stored into the microdb as "logo:sidecar"
 *               via ht_add BEFORE v7s_inference_create. must be used
 *               with -rewrite (otherwise the hook is not enabled).
 *
 * build:  cl /nologo /O2 /std:c11 inference_example.c /fe:inference_example.exe
 *
 * requires:  v7_threading.dll path passed as first CLI argument.
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"
#ifdef __linux__
#include <windows.h> /* Linux tool adapters */
#endif
#include <intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OUTPUT_BUF_SIZE 32768
#define PERF_SLOTS 10 /* uint64 statistics block: see v7_lib.h layout */

#ifdef V7_MEASURE
/* measurement-only instrumentation (paper P1-5): load latency around v7_init
 * and process memory at the lifecycle points. Compiled out of normal builds
 * (build with /DV7_MEASURE); nothing in the engine's paths is touched. */
#include <windows.h>
#include <psapi.h>

static void measure_mem(const char *label)
{
	PROCESS_MEMORY_COUNTERS pmc;
	pmc.cb = sizeof(pmc);
	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
	printf("[measure] mem %s: working set %.1f MB, private (pagefile) %.1f MB\n",
		label, (double)pmc.WorkingSetSize / 1048576.0, (double)pmc.PagefileUsage / 1048576.0);
}
#endif

/* calibrate the TSC frequency once via QueryPerformanceCounter + Sleep.
 * returns cycles per second as double (invariant TSC assumed -- any x64 CPU
 * from the last decade qualifies). used to print milliseconds next to the
 * raw RDTSC cycle deltas the DLL recorded. A disturbed calibration (CPU
 * frequency transition during the Sleep window, scheduler anomaly) makes
 * sec huge and the result a denormal -- one retry, then a fallback, so the
 * [perf] report never prints 1e48 ms. */
double tsc_hz(void)
{
	LARGE_INTEGER freq, c0, c1;
	unsigned long long t0, t1;
	QueryPerformanceFrequency(&freq);
	for (int attempt = 0; attempt < 2; attempt++)
	{
		t0 = __rdtsc();
		QueryPerformanceCounter(&c0);
		Sleep(200);
		t1 = __rdtsc();
		QueryPerformanceCounter(&c1);
		double sec = (double)(c1.QuadPart - c0.QuadPart) / (double)freq.QuadPart;
		double result = (double)(t1 - t0) / sec;
		/* a modern x64 TSC runs at 1 MHz to 10 GHz; outside that range the
		 * calibration window was disturbed -- retry once before falling back */
		if (result > 1e6 && result < 1e13)
			return result;
	}
	printf("warning: TSC calibration insane after retry -- using QPC freq for ms conversion\n");
	return (double)freq.QuadPart;
}

/* print one [perf] line: phase name, cycle delta, milliseconds, optional suffix */
void perf_line(char *name, unsigned long long cycles, double hz, char *suffix)
{
	printf("  %-13s: %12llu cycles  %9.3f ms  %s\n", name, cycles, (double)cycles / hz * 1000.0, suffix);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	if (argc < 3)
	{
		printf("Usage: %s <dll_path> <db_path> [-q8] [-i prompt] [-n tokens] [-r rounds] [-think 0|1] [-rewrite] [-script path] [-tokdump] [-ctxdump] [-prio n]\n", argv[0]);
		printf("  dll_path is REQUIRED -- path to v7_threading.dll (no default)\n");
		printf("  db_path  is REQUIRED -- path to lz4ht.db (no default)\n");
		printf("  -prio n  sets the Win32 priority of ALL worker threads after v7_init\n");
		printf("           (-15 idle, -2 lowest, -1 below normal, 0 normal, 1 above normal,\n");
		printf("            2 highest, 15 time critical; INI default is 15)\n");
		return 1;
	}
	char *dll_path = argv[1];
	char *db_path = argv[2];
	char *prompt = "What is the capital of france?";
	char *set_template = 0; /* -set-template "<fmt>": override __sys_template in-memory (test hook) */
	char *script_path = 0;  /* -script <path>: load sector_logo sidecar script from file */
	int q8 = 0;
	int max_gen = 0; /* generation cap from -n; 0 = unlimited (run until EOS, Sec.154) */
	int rounds = 1;    /* number of rounds for burn-in testing */
	int thinking = 1;  /* thinking-template selector (engine default: ON) */
	int rewrite = 0;   /* enable sector_logo sidecar hook */
	int tokdump = 0;   /* enable token stream dump */
	int ctxdump = 0;   /* enable inference context dump */
	int worker_prio = 0;  /* -prio value (Win32 SetThreadPriority constant) */
	int prio_given = 0;   /* 1 once -prio was given on the command line */

	/* parse flags: -q8, -i "prompt", -n <tokens>, -set-template "<fmt>", -r <rounds>, -think <0|1>,
	 *              -rewrite, -script <path>, -tokdump (argv[1]=dll, argv[2]=db are positional) */
	for (int i = 3; i < argc; i++)
	{
		if (strcmp(argv[i], "-q8") == 0)
			q8 = 1;
		else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
			prompt = argv[++i];
		else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
			max_gen = atoi(argv[++i]);
		else if (strcmp(argv[i], "-set-template") == 0 && i + 1 < argc)
			set_template = argv[++i];
		else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc)
			rounds = atoi(argv[++i]);
		else if (strcmp(argv[i], "-think") == 0 && i + 1 < argc)
			thinking = atoi(argv[++i]);
		else if (strcmp(argv[i], "-rewrite") == 0)
			rewrite = 1;
		else if (strcmp(argv[i], "-script") == 0 && i + 1 < argc)
			script_path = argv[++i];
		else if (strcmp(argv[i], "-tokdump") == 0)
			tokdump = 1;
		else if (strcmp(argv[i], "-ctxdump") == 0)
			ctxdump = 1;
		else if (strcmp(argv[i], "-prio") == 0 && i + 1 < argc)
		{
			worker_prio = atoi(argv[++i]);
			prio_given = 1;
		}
	}
	if (rounds < 1)
	{
		printf("Invalid rounds %d (must be >= 1)\n", rounds);
		return 1;
	}

	/* step 1: load the DLL (path from CLI -- no default) */
	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("Failed to load '%s'\n", dll_path);
		return 1;
	}
	printf("Loaded %s\n", dll_path);

	/* enable per-token stream dump if -tokdump was given (exported BSS gate:
	 * chat.inc calls dump_token_stream for every generated token) */
	if (tokdump)
	{
		*dll.enable_token_dump = 1;
		printf("token stream dump enabled\n");
	}

	/* enable the InferenceContext dump if -ctxdump was given (exported BSS gate:
	 * dump_inference_context prints the context at the gated chat/generate sites) */
	if (ctxdump)
	{
		*dll.enable_dump = 1;
		printf("inference context dump enabled\n");
	}

	/* step 2: initialize database (loads model weights, tokenizer, config) */
#ifdef V7_MEASURE
	{
		LARGE_INTEGER q0, q1, qf;
		QueryPerformanceFrequency(&qf);
		measure_mem("before v7_init");
		QueryPerformanceCounter(&q0);
#endif
		if (!dll.v7_init(db_path))
		{
			if (dll.v7_engine_is_dead())
				printf("V7_init died for '%s': %s\n", db_path, dll.v7_get_death_reason());
			else
				printf("V7_init failed for '%s'\n", db_path);
			v7_unload(&dll);
			return 1;
		}
#ifdef V7_MEASURE
		QueryPerformanceCounter(&q1);
		printf("[measure] v7_init_ms = %.1f\n", (double)(q1.QuadPart - q0.QuadPart) * 1000.0 / (double)qf.QuadPart);
		measure_mem("after v7_init");
	}
#endif
	printf("Database loaded: %s\n", db_path);

	/* runtime worker-priority override: the worker pool is created LAZILY at
	 * the first dispatched operation (initialize_once CAS, Sec.92), so a
	 * direct v7_set_worker_priority call here would FATAL. Instead, hand the
	 * value to the engine through the v7_override_prio environment variable:
	 * the engine reads it at pool creation and applies it through the same
	 * setter, so the FIRST dispatch already runs at this priority. Win32
	 * constants: -15 idle .. 15 time critical; INI default is 15. */
	if (prio_given)
	{
		char prio_buf[16];
		sprintf(prio_buf, "%d", worker_prio);
		SetEnvironmentVariableA("v7_override_prio", prio_buf);
		printf("Worker priority override set to %d\n", worker_prio);
	}

	/* load sidecar script from file if -script was given.
	 * must happen BEFORE v7s_inference_create because lg_init (inside
	 * on_token's lazy init) reads the "logo:sidecar" key from the
	 * microdb hash table. we inject it via ht_add now. */
	if (script_path)
	{
		FILE *sf = fopen(script_path, "rb");
		if (!sf)
		{
			printf("[FATAL] cannot open script file '%s'\n", script_path);
			v7_unload(&dll);
			return 1;
		}
		fseek(sf, 0, SEEK_END);
		long script_len = ftell(sf);
		fseek(sf, 0, SEEK_SET);
		/* the engine's sl_lg_script_buf needs one byte per script byte for the
		 * NUL, so it accepts at most SL_SCRIPT_BUF_SIZE-1 text bytes. match that
		 * exactly here so an oversize script is reported by the host instead of
		 * FATALing deep inside lg_init after the model is already loaded. */
		if (script_len <= 0 || script_len > 1024 * 1024 - 1)
		{
			printf("[FATAL] script file '%s' is empty or too large (%ld bytes, max 1048575)\n", script_path, script_len);
			fclose(sf);
			v7_unload(&dll);
			return 1;
		}
		char *script_buf = (char *)malloc((int)script_len + 1);
		if (!script_buf)
		{
			printf("[FATAL] malloc failed for script buffer\n");
			fclose(sf);
			v7_unload(&dll);
			return 1;
		}
		int nread = fread(script_buf, 1, (int)script_len, sf);
		fclose(sf);
		script_buf[nread] = 0; /* NUL-terminate for the logo reader */
		if (!dll.v7_db_add("logo:sidecar", script_buf, nread + 1, 0))
		{
			printf("[FATAL] v7_db_add('logo:sidecar') failed\n");
			free(script_buf);
			v7_unload(&dll);
			return 1;
		}
		printf("Sidecar script loaded: %s (%ld bytes)\n", script_path, (long)nread);
		free(script_buf);
	}

	/* the sector_logo hook is DATA-DRIVEN: with -script the sidecar is
	 * ht_add-ed into the DB above, and every chat refreshes the hook from
	 * the "logo:sidecar" key -- present = hook on, changed = re-parsed. */
	if (rewrite)
		printf("SectorLogo hook armed from the DB sidecar key\n");

	/* test hook: override the prompt template in the loaded hash table BEFORE
	 * v7s_inference_create copies it into the tokenizer struct. lets us try
	 * corrected template formats without rebuilding the multi-GB DB.
	 * escape decoding: \n -> newline (same convention as dbg_image_recog). */
	if (set_template)
	{
		char decoded[1024];
		int dl = 0;
		for (char *p = set_template; *p && dl + 1 < sizeof(decoded); p++)
		{
			if (p[0] == '\\' && p[1] == 'n')
			{
				decoded[dl++] = '\n';
				p++;
			}
			else
				decoded[dl++] = *p;
		}
		decoded[dl] = 0;
		if (!dll.v7_db_add("__sys_template", decoded, dl + 1, 0))
		{
			printf("[FATAL] v7_db_add('__sys_template') failed\n");
			v7_unload(&dll);
			return 1;
		}
		printf("Template overridden: %s\n", decoded);
	}

	/* step 3: initialize inference from microdb (context hidden in BSS; FATALs internally on failure) */
	dll.v7_db_add("__server_chat__turn", "0", 2, 0); /* fake turn counter for sidecars: a run.bat-style host serves turn 0 only */
	/* pseudo-exception check: v7_initialize_inference is a GATED entry, so a
	 * create-time FATAL unwinds to here with the engine parked AND returns 0.
	 * Check the return AND the death flag -- either one alone is insufficient.
	 * The canonical caught-FATAL lifecycle ends in unload, per
	 * docs/plans/fatal_to_pseudo_exception_plan.md. */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			printf("engine died during init: %s\n", dll.v7_get_death_reason());
		else
			printf("v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}
	dll.v7_set_quantize(q8);
	dll.v7_set_max_gen(max_gen); /* generation cap from -n; 0 = unlimited (run until EOS, Sec.154) */
	printf("Inference context created (%s mode)\n", q8 ? "Q8" : "F32");
#ifdef V7_MEASURE
	measure_mem("after v7_initialize_inference");
#endif

	/* thinking-template selector: AFTER inference_create (which resets the
	 * engine default to ON per context). takes effect at the next template
	 * render inside v7s_chat (no-op for models without @think keys). */
	dll.v7_enable_thinking(thinking);

	/* statistics block pointer + one-time TSC calibration for cycle->ms print */
	unsigned long long *s = (unsigned long long *)dll.v7_get_generation_statistic();
	double hz = tsc_hz();
	printf("Tsc calibrated: %.4f GHz\n", hz / 1e9);

	/* step 4: run inference. always use v7s_chat so multi-turn and system
	 * prompts work. v7s_generate also renders the turn-1 template (it calls
	 * v7s_chat internally), but it runs on a separate KV cache and loops
	 * hybrid-thinking models like mini_c_p_m5 on the bare thinking token —
	 * so it is bypassed here. */
	char output[OUTPUT_BUF_SIZE];

	for (int round = 1; round <= rounds; round++)
	{
		memset(output, 0, sizeof(output));
		for (int i = 0; i < PERF_SLOTS; i++)
			s[i] = 0; /* reset statistics block */

		printf("\n... round %d/%d ...\n", round, rounds);
		printf("Prompt: %s\n", prompt);
		printf("---\n");

		dll.v7_chat((char *)prompt, 0, output, OUTPUT_BUF_SIZE, 1);

		/* pseudo-exception check: a mid-generation FATAL unwinds to here with
		 * the engine parked (reason queryable, workers quiesced) -- report,
		 * destroy, unload, exit nonzero */
		if (dll.v7_engine_is_dead())
		{
			printf("engine died: %s\n", dll.v7_get_death_reason());
			dll.v7_free_inference();
			v7_unload(&dll);
			return 1;
		}

		printf("%s\n", output);
#ifdef V7_MEASURE
		if (round == 1)
			measure_mem("after first turn");
#endif

		/* derived phase deltas from the flat statistics block */
		unsigned long long c_tok = s[1] - s[0];
		unsigned long long c_pre = s[2] - s[1];
		unsigned long long c_gen = s[3] - s[2];
		unsigned long long c_all = s[3] - s[0];
		unsigned long long n_pre = s[4];
		unsigned long long n_gen = s[5];
		double gen_sec = (double)c_gen / hz;
		double pre_tok_ms = n_pre ? (double)c_pre / hz * 1000.0 / (double)n_pre : 0.0;
		char pre_info[64], gen_info[64];
		snprintf(pre_info, sizeof(pre_info), "(%llu tok, %.3f ms/tok)", n_pre, pre_tok_ms);
		snprintf(gen_info, sizeof(gen_info), "(%llu tok, %.2f tok/s)", n_gen, gen_sec > 0.0 ? (double)n_gen / gen_sec : 0.0);

		printf("[perf] round %d\n", round);
		perf_line("tokenization", c_tok, hz, "");
		perf_line("prefill", c_pre, hz, pre_info);
		perf_line("generation", c_gen, hz, gen_info);
		perf_line("total", c_all, hz, "");
		if (s[7])
		{ /* vision encode ran (image turn) */
			char vis_info[64];
			snprintf(vis_info, sizeof(vis_info), "(%llu vis prefill tok, %llu vis gen tok)", (unsigned long long)s[8], (unsigned long long)s[9]);
			perf_line("vision encode", s[7] - s[6], hz, vis_info);
		}

		/* cycle-tuner pool fitness: per-shape average c measurement of the
		 * elite front (list slots 0..9) and the whole non-tail pool; a 0
		 * means the shape has not completed a full population sweep yet */
		int tuner_shape_count = dll.v7_get_active_shape_count();
		unsigned long long *tuner_best = dll.v7_get_best_fitness();
		unsigned long long *tuner_pool = dll.v7_get_pool_fitness();
		printf("[perf] tuner fitness: %d shape(s)\n", tuner_shape_count);
		for (int shape_index = 0; shape_index < tuner_shape_count; shape_index++)
		{
			printf("[perf] tuner shape %d: best %llu cycles, pool %llu cycles\n", shape_index,
			       (unsigned long long)tuner_best[shape_index], (unsigned long long)tuner_pool[shape_index]);
		}
	}

	/* step 5: clean up. v7s_shutdown MUST run first -- it joins the worker
	 * threads so no worker can still be executing inside the DLL when its
	 * heap is freed (v7s_inference_destroy) and its code pages are unmapped
	 * (v7_unload -> FreeLibrary -> DllMain DETACH -> ht_destroy). without
	 * this, a worker still running produces the intermittent execute-fault
	 * at a BSS address after the answer is printed (AGENTS.md 105). */
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
