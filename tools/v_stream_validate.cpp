// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v_stream_validate.cpp -- monolithic StreamingLLM validator.
 *
 * One validator owns ALL StreamingLLM coverage (Sec.11a one-validator-per-
 * module). Small-window wrap tests on the reference model: a tiny ring
 * (S=4, W=8, C=12) forces wrapping within the first dozen tokens, so the
 * slot resolver, the three-phase attention kernel, the eviction ring and the
 * eviction callback all run inside a fast validator turn.
 *
 * Sections:
 *   s1  setter pre-init FATAL (caught; gate lands the unwind)
 *   s2  setter rejects + gate pairing (v7_get_position must succeed after
 *       every rejected call -- a "concurrent" FATAL there means a leaked
 *       zone claim)
 *   s3  T1 legacy bit-equivalence, F32: enable=1 with C far above the turn
 *       length must produce byte-identical output to the legacy run
 *   s4  T1 legacy bit-equivalence, Q8
 *   s5  small-window wrap (S=4, W=8, C=12) + on_stream_evict callback:
 *       long prompt + generation crosses C repeatedly; the Logo probe
 *       script delivers the evicted text into "ltm_probe"; the batch meta
 *       opens at first=4 (= S = C-W); two consecutive wrapped turns prove
 *       the engine stays alive and the callback keeps delivering
 *   s6  disable at pos > 0 is legal and leaves the session usable
 *   s7  tier-1 REWRITE under streaming: enable with C far above the turn,
 *       a sidecar SetStream fires on the first generated token batch --
 *       legacy rollback path must run (alive, replacement text in output,
 *       latch set, next turn works)
 *   s8  tier-2 REWRITE under streaming: small C = 12 + the s5 long prompt
 *       forces pos > C; the sidecar SetStream fires mid-wrap -- full restart
 *       rebuild must run (alive, rebuilt stream in output, latch set, the
 *       pending eviction batch flushed BEFORE the rebuild with its meta
 *       intact, eviction delivery still exactly-once on the next turn, and
 *       the RECORDED PROMPT REPLAYED by the rebuild -- proven arithmetically
 *       by the next turn's eviction batch first= position)
 *
 * usage:  v_stream_validate.exe <dll_path> <db_path>
 * exit 0 = all sections passed; nonzero = failure.
 */

#include "zydis_debug_header.h"
#include "stream_ref.h"
#include "v7_lib.h"

#include <windows.h> /* Linux tool adapters: unsigned __int64, sscanf_s */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OUT_BUF_SIZE 8192

static int passing_tests_counter = 0;
static int failing_tests_counter = 0;

static void report_check(const char *test_label, int ok)
{
	if (ok)
	{
		printf("  pass: %s\n", test_label);
		passing_tests_counter++;
	}
	else
	{
		printf("  FAIL: %s\n", test_label);
		failing_tests_counter++;
	}
}

static void run_fresh_context(v7_lib *dll, const char *db_path)
{
	/* the destroy path wipes the microdb (ht_destroy + ht_init -- "the caller
	 * will db_load before next inference_create"), so a fresh context is:
	 * free -> v7_init(db) -> initialize_inference. */
	dll->v7_free_inference();
	if (!dll->v7_init(db_path))
	{
		printf("FAIL: v7_init('%s') died: %s\n", db_path, dll->v7_get_death_reason());
		exit(1);
	}
	if ((!dll->v7_initialize_inference()) || dll->v7_engine_is_dead())
	{
		printf("FAIL: v7_initialize_inference died: %s\n", dll->v7_get_death_reason());
		exit(1);
	}
}

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);
	ZydisInit();

	if (argc < 4 || strcmp(argv[2], "--db") != 0)
	{
		printf("Usage: %s <dll_path> --db <db_path>\n", argv[0]);
		printf("  both paths are REQUIRED (no defaults)\n");
		return 1;
	}
	char *dll_path = argv[1];
	char *db_path = argv[3];

	printf("... v_stream_validate: StreamingLLM small-window wrap + eviction callback ...\n");

	v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		printf("FAIL: cannot load '%s'\n", dll_path);
		return 1;
	}

	/* ---- s1: the setter is a gated public entry -- pre-init it FATALs and the
	 * ---- catch machinery parks the engine; free_inference resets the state. */
	printf(" [s1] setter pre-init must FATAL with a named reason ...\n");
	dll.v7_set_streaming(1, 4, 8);
	{
		/* the reason pointer is only valid while the engine is dead; guard it
		 * so a missing reason cannot dereference NULL in strstr/%s. */
		const char *preinit_reason = dll.v7_get_death_reason();
		if ((dll.v7_engine_is_dead() != 1) || (preinit_reason == 0) || (strstr(preinit_reason, "v7_set_streaming") == 0))
		{
			printf("FAIL: pre-init v7_set_streaming must FATAL naming the entry, got: %s\n",
			       preinit_reason ? preinit_reason : "(no reason)");
			v7_unload(&dll);
			return 1;
		}
		printf("  caught: %s\n", preinit_reason);
	}
	dll.v7_free_inference();                   /* reset the death state -- the rest of the suite runs on */
	report_check("pre-init setter FATAL caught with named reason", 1);

	/* ---- init once for the rest of the suite ---- */
	if (!dll.v7_init(db_path))
	{
		printf("FAIL: v7_init('%s') died: %s\n", db_path, dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	run_fresh_context(&dll, db_path);

	/* ---- s2: rejects return -1 and must NOT leak the engine zone ---- */
	printf(" [s2] setter rejects + gate pairing ...\n");
	int reject_cases[7][3] = {
		{2, 1, 1},                             /* enable out of range */
		{1, 0, 1},                             /* zero sinks */
		{1, 1, 0},                             /* zero window */
		{1, 2000000000, 2000000000},           /* S + W overflows int32 */
		{1, -2, 3},                            /* negative S, C wraps to a small positive (carry rejects) */
		{1, 3, -1},                            /* negative W, C wraps (carry rejects) */
		{1, -2000000000, 1999999999},          /* mixed signs, no carry: C = 2^32-1 is huge, capacity rejects */
	};
	int rejects_ok = 1;
	for (int case_index = 0; case_index < 7; case_index++)
	{
		int rc = dll.v7_set_streaming(reject_cases[case_index][0], reject_cases[case_index][1], reject_cases[case_index][2]);
		int pos_after_reject = dll.v7_get_position();
		if (rc != -1)
		{
			printf("  FAIL: reject case %d returned %d, expected -1\n", case_index, rc);
			rejects_ok = 0;
		}
		if (dll.v7_engine_is_dead() != 0)
		{
			printf("  FAIL: reject case %d parked the engine (reason: %s)\n", case_index, dll.v7_get_death_reason());
			rejects_ok = 0;
			break;
		}
		if (pos_after_reject < 0)
		{
			printf("  FAIL: v7_get_position after reject case %d failed -- leaked zone claim\n", case_index);
			rejects_ok = 0;
		}
	}
	report_check("all setter rejects return -1 with the gate pair intact", rejects_ok);

	/* ---- s3: T1 legacy bit-equivalence, F32 ---- */
	printf(" [s3] T1 legacy equivalence, F32 (enable=1, C=8188 >> turn length) ...\n");
	char out_streaming[OUT_BUF_SIZE];
	char out_legacy[OUT_BUF_SIZE];
	const char *capital_prompt = "What is the capital of France?";
	dll.v7_set_quantize(0);
	dll.v7_set_max_gen(48);
	int enable_rc = dll.v7_set_streaming(1, 4, 8184);
	if (enable_rc != 0)
	{
		printf("FAIL: v7_set_streaming(1, 4, 8184) returned %d (dead=%d, reason: %s)\n",
		       enable_rc, dll.v7_engine_is_dead(), dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat(capital_prompt, 0, out_streaming, OUT_BUF_SIZE, 1);
	if (dll.v7_engine_is_dead())
	{
		printf("FAIL: streaming F32 turn died: %s\n", dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	run_fresh_context(&dll, db_path);          /* destroy resets g_stream_enable to 0 -- fresh legacy run */
	dll.v7_set_quantize(0);                    /* re-apply: a fresh context starts at engine defaults */
	dll.v7_set_max_gen(48);
	memset(out_legacy, 0, sizeof(out_legacy));
	dll.v7_chat(capital_prompt, 0, out_legacy, OUT_BUF_SIZE, 1);
	/* a bit-identity pass on two EMPTY outputs would prove nothing: require
	 * both turns to have produced text before comparing. */
	int f32_identical = ((out_streaming[0] != 0) && (out_legacy[0] != 0) && (strcmp(out_streaming, out_legacy) == 0));
	if (!f32_identical)
	{
		printf("  streaming: [%.80s]\n", out_streaming);
		printf("  legacy:    [%.80s]\n", out_legacy);
	}
	report_check("T1 F32: wrapped-capable enable with no wrap is bit-identical to legacy", f32_identical);

	/* ---- s4: T1 legacy bit-equivalence, Q8 ---- */
	printf(" [s4] T1 legacy equivalence, Q8 ...\n");
	run_fresh_context(&dll, db_path);          /* the s3 legacy turn left pos > 0 -- enable needs pos == 0 */
	dll.v7_set_quantize(1);
	dll.v7_set_max_gen(48);
	enable_rc = dll.v7_set_streaming(1, 4, 8184);
	if (enable_rc != 0)
	{
		printf("FAIL: v7_set_streaming(1, 4, 8184) returned %d (Q8; dead=%d, reason: %s)\n",
		       enable_rc, dll.v7_engine_is_dead(), dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat(capital_prompt, 0, out_streaming, OUT_BUF_SIZE, 1);
	if (dll.v7_engine_is_dead())
	{
		printf("FAIL: streaming Q8 turn died: %s\n", dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	run_fresh_context(&dll, db_path);
	dll.v7_set_quantize(1);
	dll.v7_set_max_gen(48);
	memset(out_legacy, 0, sizeof(out_legacy));
	dll.v7_chat(capital_prompt, 0, out_legacy, OUT_BUF_SIZE, 1);
	/* same empty-output guard as s3: an empty==empty "pass" proves nothing. */
	int q8_identical = ((out_streaming[0] != 0) && (out_legacy[0] != 0) && (strcmp(out_streaming, out_legacy) == 0));
	if (!q8_identical)
	{
		printf("  streaming: [%.80s]\n", out_streaming);
		printf("  legacy:    [%.80s]\n", out_legacy);
	}
	report_check("T1 Q8: wrapped-capable enable with no wrap is bit-identical to legacy", q8_identical);

	/* ---- s5: small-window wrap + eviction callback ---- */
	printf(" [s5] wrap test: S=4, W=8, C=12 + on_stream_evict probe ...\n");
	run_fresh_context(&dll, db_path);
	dll.v7_set_quantize(1);
	dll.v7_set_max_gen(80);
	/* the probe script: copy the delivered batch text + meta into probe keys
	 * the validator reads back through v7_db_find */
	const char *evict_script =
		"DatabaseSet \"ltm_probe\" DatabaseGet \"stream_remove_text\"\r\n"
		"DatabaseSet \"ltm_probe_meta\" DatabaseGet \"stream_remove_meta\"\r\n";
	dll.v7_db_add("on_stream_evict", (void *)evict_script, (int)strlen(evict_script), 0);
	enable_rc = dll.v7_set_streaming(1, 4, 8);
	if (enable_rc != 0)
	{
		printf("FAIL: v7_set_streaming(1, 4, 8) returned %d (dead=%d, reason: %s)\n",
		       enable_rc, dll.v7_engine_is_dead(), dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	const char *long_prompt =
		"Name the colors red, green, blue, yellow, orange, purple, pink, brown, "
		"black, white, gray and violet, and for every color in that list write "
		"one short sentence describing where the color appears in nature.";
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat(long_prompt, 0, out_streaming, OUT_BUF_SIZE, 1);
	int wrap_alive = (dll.v7_engine_is_dead() == 0);
	report_check("wrapped turn (pos far past C=12) runs without FATAL or crash", wrap_alive);
	printf("  diag: session pos = %d, output bytes = %llu\n", dll.v7_get_position(), (unsigned long long)strlen(out_streaming));
	printf("  diag: output tail = [%.120s]\n", (strlen(out_streaming) > 120) ? out_streaming + strlen(out_streaming) - 120 : out_streaming);
	if (!wrap_alive)
	{
		printf("  death reason: %s\n", dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	/* in-turn delivery assertions come FIRST: the diagnostic direct eval_script
	 * below also sets ltm_probe, so asserting after it could pass for the
	 * wrong reason (out-of-turn delivery masking a broken in-turn path) */
	unsigned __int64 probe_len = 0;
	const char *probe_text = (const char *)dll.v7_db_find("ltm_probe", &probe_len);
	report_check("eviction callback delivered batch text into ltm_probe", (probe_text != 0));
	unsigned __int64 meta_len = 0;
	const char *meta_text = (const char *)dll.v7_db_find("ltm_probe_meta", &meta_len);
	/* the meta blob is (ptr,len) with NO NUL guarantee (ht_find raw blob), and a
	 * bare "first=4" prefix test would also match "first=40": copy a bounded
	 * segment into a NUL-terminated stack buffer and parse the number exactly. */
	int meta_first = -1;
	if ((meta_text != 0) && (meta_len > 0))
	{
		char meta_buf[64];
		unsigned __int64 copy_len = meta_len < (sizeof(meta_buf) - 1) ? meta_len : (sizeof(meta_buf) - 1);
		memcpy(meta_buf, meta_text, (size_t)copy_len);
		meta_buf[copy_len] = 0;
		sscanf_s(meta_buf, "first=%d", &meta_first);
	}
	report_check("batch meta opens at first=4 (S = C-W)", (meta_first == 4));
	if (probe_text != 0 && probe_len > 0)
		printf("  ltm_probe (%llu bytes): [%.60s]\n", (unsigned long long)probe_len, probe_text);
	if (meta_text != 0)
		printf("  ltm_probe_meta: [%.*s]\n", (int)meta_len, meta_text);
	/* diagnostics: split engine-side (flush wrote its keys) from Logo-side (probe script ran) */
	unsigned __int64 etext_len = 0;
	const char *etext = (const char *)dll.v7_db_find("stream_remove_text", &etext_len);
	unsigned __int64 emeta_len = 0;
	const char *emeta = (const char *)dll.v7_db_find("stream_remove_meta", &emeta_len);
	unsigned __int64 script_len = 0;
	const char *script_text = (const char *)dll.v7_db_find("on_stream_evict", &script_len);
	printf("  diag: engine text=%s(%llu) meta=%s(%llu) script=%s(%llu)\n",
	       etext ? "yes" : "NO", etext ? (unsigned long long)etext_len : 0,
	       emeta ? "yes" : "NO", emeta ? (unsigned long long)emeta_len : 0,
	       script_text ? "yes" : "NO", script_text ? (unsigned long long)script_len : 0);
	int eval_rc = dll.v7s_sectorlogo_eval_script("on_stream_evict");
	printf("  diag: direct eval_script rc=%d dead=%d ltm_probe=%s\n", eval_rc,
	       dll.v7_engine_is_dead(), dll.v7_db_find("ltm_probe", &probe_len) ? "SET" : "absent");
	if (dll.v7_engine_is_dead())
	{
		printf("  diag: death reason: %s\n", dll.v7_get_death_reason());
		dll.v7_free_inference();
		run_fresh_context(&dll, db_path);
		dll.v7_set_quantize(1);
		dll.v7_set_max_gen(80);
		dll.v7_db_add("on_stream_evict", (void *)evict_script, (int)strlen(evict_script), 0);
		dll.v7_set_streaming(1, 4, 8);
	}
	/* second wrapped turn: the callback keeps delivering, engine stays alive.
	 * Sentinel first: without it, turn 1's probe would make this pass even if
	 * turn 2 delivered nothing. */
	dll.v7_db_add("ltm_probe", (void *)"<awaiting>", 11, 0);
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat("Now name two more colors the same way.", 0, out_streaming, OUT_BUF_SIZE, 0);
	report_check("second wrapped turn (reset=0) stays alive", dll.v7_engine_is_dead() == 0);
	unsigned __int64 probe2_len = 0;
	const char *probe2_text = (const char *)dll.v7_db_find("ltm_probe", &probe2_len);
	report_check("eviction callback delivered again on turn 2",
	             (probe2_text != 0) && (probe2_len > 11) && (strncmp(probe2_text, "<awaiting>", 10) != 0));

	/* ---- s5b: nested v7_generate mid-session (P1-3 regression) ----
	 * v7_generate swaps streaming off and runs a reset=1 nested chat; the
	 * guarded entry path must preserve the session's eviction state (no
	 * pending-batch drop, no warm-latch wipe), and the next turn still
	 * delivers. */
	printf(" [s5b] nested v7_generate mid-streaming-session ...\n");
	char gen_out[4096];
	memset(gen_out, 0, sizeof(gen_out));
	dll.v7_generate("The capital of France is", gen_out, (int)sizeof(gen_out), 1);
	report_check("v7_generate mid-session stays alive", dll.v7_engine_is_dead() == 0);
	dll.v7_db_add("ltm_probe", (void *)"<awaiting>", 11, 0);   /* sentinel: a stale delivery must not pass this turn */
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat("Name one more color.", 0, out_streaming, OUT_BUF_SIZE, 0);
	report_check("chat turn after nested generate stays alive", dll.v7_engine_is_dead() == 0);
	unsigned __int64 probe3_len = 0;
	const char *probe3_text = (const char *)dll.v7_db_find("ltm_probe", &probe3_len);
	report_check("eviction callback still delivering after nested generate",
	             (probe3_text != 0) && (probe3_len > 11) && (strncmp(probe3_text, "<awaiting>", 10) != 0));

	/* ---- s5c: multi-batch turn (many eviction batches in ONE turn) ----
	 * A long wrapped turn crosses the EVICT_BATCH drain inside the ingest /
	 * decode loops; the accumulating probe collects every batch. The batch
	 * metas must form a contiguous chain: a lost or duplicated batch leaves
	 * a gap or overlap in the chain, independent of how long the model's
	 * generated text happens to be. */
	printf(" [s5c] multi-batch turn (accumulating probe, contiguous batch chain) ...\n");
	run_fresh_context(&dll, db_path);
	dll.v7_set_quantize(1);
	dll.v7_set_max_gen(400);
	const char *acc_script =
		"DatabaseSet \"ltm_acc\" (DatabaseGet \"ltm_acc\") _ (DatabaseGet \"stream_remove_text\")\r\n"
		"DatabaseSet \"ltm_acc_meta_all\" (DatabaseGet \"ltm_acc_meta_all\") _ \"|\" _ (DatabaseGet \"stream_remove_meta\")\r\n"
		"DatabaseSet \"ltm_acc_meta\" DatabaseGet \"stream_remove_meta\"\r\n";
	dll.v7_db_add("on_stream_evict", (void *)acc_script, (int)strlen(acc_script), 0);
	enable_rc = dll.v7_set_streaming(1, 4, 8);
	if (enable_rc != 0)
	{
		printf("FAIL: s5c v7_set_streaming returned %d\n", enable_rc);
		v7_unload(&dll);
		return 1;
	}
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat(long_prompt, 0, out_streaming, OUT_BUF_SIZE, 1);
	report_check("multi-batch wrapped turn stays alive", dll.v7_engine_is_dead() == 0);
	unsigned __int64 acc_len = 0;
	const char *acc_text = (const char *)dll.v7_db_find("ltm_acc", &acc_len);
	printf("  diag: ltm_acc = %llu bytes\n", acc_text ? (unsigned long long)acc_len : 0);
	report_check("eviction batches delivered text", (acc_text != 0 && acc_len > 0));
	unsigned __int64 accmeta_len = 0;
	const char *accmeta = (const char *)dll.v7_db_find("ltm_acc_meta", &accmeta_len);
	report_check("last batch meta present", (accmeta != 0 && accmeta_len > 0));
	if (accmeta != 0)
		printf("  last batch meta: [%.*s]\n", (int)accmeta_len, accmeta);
	/* Machinery check, content-independent: walk the accumulated batch metas
	 * and require first + tokens of every batch to equal the next batch's
	 * first. A dropped or duplicated batch delivery breaks the chain. */
	unsigned __int64 metas_len = 0;
	const char *metas_text = (const char *)dll.v7_db_find("ltm_acc_meta_all", &metas_len);
	int batch_count = 0;
	int chain_ok = (metas_text != 0 && metas_len > 0);
	int prev_first = -1;
	int prev_tokens = -1;
	const char *scan = metas_text;
	const char *scan_end = metas_text + metas_len;
	while (chain_ok && scan < scan_end)
	{
		int first_pos = 0;
		int batch_tokens = 0;
		const char *seg = (const char *)memchr(scan, '|', (size_t)(scan_end - scan));
		size_t seg_len = seg ? (size_t)(seg - scan) : (size_t)(scan_end - scan);
		if (seg_len == 0)
		{
			scan = seg ? seg + 1 : scan_end;
			continue;
		}
		if (sscanf_s(scan, "first=%d tokens=%d", &first_pos, &batch_tokens) != 2)
			chain_ok = 0;
		if (chain_ok && batch_count > 0 && first_pos != prev_first + prev_tokens)
			chain_ok = 0;
		if (chain_ok)
		{
			prev_first = first_pos;
			prev_tokens = batch_tokens;
			batch_count++;
		}
		scan = seg ? seg + 1 : scan_end;
	}
	printf("  diag: eviction batches = %d, chain_ok = %d\n", batch_count, chain_ok);
	report_check("eviction batches contiguous (no dropped or duplicated batch)", chain_ok);
	report_check("turn spans multiple eviction batches", batch_count >= 2);

	/* ---- s5d: wrapped KV save / load round-trip ---- */
	printf(" [s5d] wrapped KV save / load round-trip ...\n");
	run_fresh_context(&dll, db_path);
	dll.v7_set_quantize(1);
	dll.v7_set_max_gen(60);
	dll.v7_db_add("on_stream_evict", (void *)evict_script, (int)strlen(evict_script), 0);
	enable_rc = dll.v7_set_streaming(1, 4, 8);
	if (enable_rc != 0)
	{
		printf("FAIL: s5d enable returned %d\n", enable_rc);
		v7_unload(&dll);
		return 1;
	}
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat(long_prompt, 0, out_streaming, OUT_BUF_SIZE, 1);
	report_check("pre-save wrapped turn alive", dll.v7_engine_is_dead() == 0);
	int save_rc = dll.save_kv_cache("build_logs\\_stream_roundtrip.kv");
	report_check("save_kv_cache on a wrapped session returns 1", save_rc == 1);
	run_fresh_context(&dll, db_path);
	dll.v7_set_quantize(1);
	dll.v7_set_max_gen(60);
	enable_rc = dll.v7_set_streaming(1, 4, 8);   /* enable at pos=0 BEFORE the load */
	if (enable_rc != 0)
	{
		printf("FAIL: s5d re-enable returned %d\n", enable_rc);
		v7_unload(&dll);
		return 1;
	}
	int load_rc = dll.load_kv_cache("build_logs\\_stream_roundtrip.kv");
	report_check("load_kv_cache into the enabled session returns 1", load_rc == 1);
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat("Continue: name one more color.", 0, out_streaming, OUT_BUF_SIZE, 0);
	report_check("continuation after wrapped reload stays alive", dll.v7_engine_is_dead() == 0);

	/* ---- s5e: enable / disable / enable cycle reallocates the ring ---- */
	printf(" [s5e] enable/disable/enable cycle at pos = 0 ...\n");
	run_fresh_context(&dll, db_path);
	dll.v7_set_quantize(1);
	dll.v7_set_max_gen(60);
	int cyc1 = dll.v7_set_streaming(1, 2, 6);
	int cyc2 = dll.v7_set_streaming(0, 1, 1);
	int cyc3 = dll.v7_set_streaming(1, 4, 8);
	report_check("enable(2,6) / disable / enable(4,8) cycle all accepted",
	             (cyc1 == 0 && cyc2 == 0 && cyc3 == 0 && dll.v7_engine_is_dead() == 0));
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat(long_prompt, 0, out_streaming, OUT_BUF_SIZE, 1);
	report_check("wrapped turn after the realloc cycle stays alive", dll.v7_engine_is_dead() == 0);

	/* ---- s7: tier-1 rewrite (streaming on, pos <= C -- legacy rollback path) ---- */
	printf(" [s7] tier-1 rewrite: streaming on, linear regime ...\n");
	run_fresh_context(&dll, db_path);      /* s5 left pos > 0 and C = 12 -- a fresh context resets pos and drops the old sidecar keys */
	dll.v7_set_quantize(1);
	dll.v7_set_max_gen(96);
	/* the rewrite script: fires ONCE, the first time the generated stream
	 * contains "Paris" (Search returns NIL until then; IsNumber NIL is
	 * false). The latch keeps a later "Paris" from re-triggering. The
	 * replacement does not contain the trigger word, so the rewritten
	 * stream cannot immediately re-fire. */
	const char *rw_script_t1 =
		"IfElse IsNumber Search \"Paris\" GetStream [\r\n"
		"  IfElse IsNumber DatabaseGet \"rw1_latch\" [] [\r\n"
		"    SetStream \"The capital of France is Berlin.\"\r\n"
		"    DatabaseSet \"rw1_latch\" 1\r\n"
		"  ]\r\n"
		"] []\r\n";
	dll.v7_db_add("logo:sidecar", (void *)rw_script_t1, (int)strlen(rw_script_t1), 0);
	enable_rc = dll.v7_set_streaming(1, 4, 8184);   /* C = 8188 >> turn length: the rewrite must land in the LINEAR tier */
	if (enable_rc != 0)
	{
		printf("FAIL: s7 v7_set_streaming(1, 4, 8184) returned %d (dead=%d, reason: %s)\n",
		       enable_rc, dll.v7_engine_is_dead(), dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat(capital_prompt, 0, out_streaming, OUT_BUF_SIZE, 1);
	int t1_alive = (dll.v7_engine_is_dead() == 0);
	if (!t1_alive)
		printf("  s7 death reason: %s\n", dll.v7_get_death_reason());
	report_check("tier-1 rewrite turn stays alive", t1_alive);
	unsigned __int64 latch1_len = 0;
	report_check("tier-1 rewrite actually fired (latch set)",
	             t1_alive && dll.v7_db_find("rw1_latch", &latch1_len) != 0);
	report_check("tier-1 rewrite output contains the replacement text",
	             t1_alive && strstr(out_streaming, "Berlin") != 0);
	if (t1_alive)
		printf("  tier-1 output: [%.80s]\n", out_streaming);
	/* the engine must remain fully usable after the rollback: next turn works */
	char out_after_t1[OUT_BUF_SIZE];
	memset(out_after_t1, 0, sizeof(out_after_t1));
	dll.v7_chat("What river flows through Paris?", 0, out_after_t1, OUT_BUF_SIZE, 0);
	report_check("turn after tier-1 rewrite works (alive, non-empty output)",
	             dll.v7_engine_is_dead() == 0 && out_after_t1[0] != 0);

	/* ---- s8: tier-2 rewrite (streaming on, pos > C -- full restart rebuild) ---- */
	printf(" [s8] tier-2 rewrite: streaming on, wrapped regime (pos > C) ...\n");
	run_fresh_context(&dll, db_path);      /* s7 left pos > 0 -- the enable below needs pos == 0 */
	dll.v7_set_quantize(1);
	dll.v7_set_max_gen(24);                /* 24, not more: caps replacement + continuation so the replay
	                                        * discriminator at the bottom of this section is pure arithmetic
	                                        * -- a replacement-only rebuild cannot push the session position
	                                        * past 24, the replayed prompt does */
	/* fires ONCE, on the model's first generated token: greedy generation
	 * opens every think block with "Okay", so Search hits immediately while
	 * the s5 long prompt has already wrapped the ring hard during ingest
	 * (pos >> C = 12) -- the rewrite must take the RESTART tier. The
	 * replacement contains no "Okay", and the latch keeps any later one
	 * from re-triggering. */
	const char *rw_script_t2 =
		"IfElse IsNumber Search \"Okay\" GetStream [\r\n"
		"  IfElse IsNumber DatabaseGet \"rw2_latch\" [] [\r\n"
		"    SetStream \"Colors listed: crimson, scarlet and ruby.\"\r\n"
		"    DatabaseSet \"rw2_latch\" 1\r\n"
		"  ]\r\n"
		"] []\r\n";
	dll.v7_db_add("logo:sidecar", (void *)rw_script_t2, (int)strlen(rw_script_t2), 0);
	dll.v7_db_add("on_stream_evict", (void *)evict_script, (int)strlen(evict_script), 0);
	enable_rc = dll.v7_set_streaming(1, 4, 8);      /* C = 12: the long prompt wraps during ingest */
	if (enable_rc != 0)
	{
		printf("FAIL: s8 v7_set_streaming(1, 4, 8) returned %d (dead=%d, reason: %s)\n",
		       enable_rc, dll.v7_engine_is_dead(), dll.v7_get_death_reason());
		v7_unload(&dll);
		return 1;
	}
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat(long_prompt, 0, out_streaming, OUT_BUF_SIZE, 1);
	int t2_alive = (dll.v7_engine_is_dead() == 0);
	if (!t2_alive)
		printf("  s8 death reason: %s\n", dll.v7_get_death_reason());
	report_check("tier-2 rewrite turn (pos far past C) stays alive", t2_alive);
	unsigned __int64 latch2_len = 0;
	report_check("tier-2 rewrite actually fired (latch set)",
	             t2_alive && dll.v7_db_find("rw2_latch", &latch2_len) != 0);
	report_check("tier-2 rebuild output contains the replacement text",
	             t2_alive && strstr(out_streaming, "crimson") != 0);
	if (t2_alive)
		printf("  tier-2 output: [%.80s]\n", out_streaming);
	/* exactly-once still holds after the rebuild: the prompt's ingest
	 * evictions (long prompt vs C = 12, well below the EVICT_BATCH drain of
	 * 256) are pending when the rewrite fires -- the engine must deliver
	 * them (rewrite-entry flush), which runs the probe script; the next
	 * wrapped turn must deliver again through the normal drain sites */
	unsigned __int64 probe_t2_len = 0;
	const char *probe_t2 = (const char *)dll.v7_db_find("ltm_probe", &probe_t2_len);
	report_check("pending evictions delivered across the tier-2 rebuild (probe written)",
	             t2_alive && probe_t2 != 0 && probe_t2_len > 0);
	if (probe_t2)
		printf("  pre-rewrite batch (%llu bytes): [%.60s]\n", (unsigned long long)probe_t2_len, probe_t2);
	dll.v7_db_add("ltm_probe", (void *)"<awaiting>", 11, 0);   /* sentinel: the rebuild's probe must not pass the next turn */
	memset(out_streaming, 0, sizeof(out_streaming));
	dll.v7_chat("Now name two more colors the same way.", 0, out_streaming, OUT_BUF_SIZE, 0);
	report_check("wrapped turn after the tier-2 rebuild stays alive", dll.v7_engine_is_dead() == 0);
	unsigned __int64 probe4_len = 0;
	const char *probe4_text = (const char *)dll.v7_db_find("ltm_probe", &probe4_len);
	report_check("eviction callback still delivers after a tier-2 rebuild (exactly-once intact)",
	             (dll.v7_engine_is_dead() == 0) && (probe4_text != 0) && (probe4_len > 11) && (strncmp(probe4_text, "<awaiting>", 10) != 0));
	/* V5-1 discriminator: the rebuild must REPLAY the recorded prompt
	 * history. Turn 2's eviction batch opens at first = (turn-2 start
	 * pos) - W, and turn-2's start pos carries the replayed prompt.
	 * With the replay: first >= 30 (conservative prompt-token floor for
	 * the long prompt) + 1 (replacement) - 8 (W) = 23.
	 * With a replacement-only rebuild: the whole post-rewrite timeline is
	 * <= max_gen = 24 tokens, so first <= 24 - 8 = 16. The ranges do not
	 * overlap; the check is pure arithmetic and does not depend on model
	 * behavior. */
	unsigned __int64 meta3_len = 0;
	const char *meta3 = (const char *)dll.v7_db_find("ltm_probe_meta", &meta3_len);
	int first_pos = -1;
	if (meta3)
		sscanf_s(meta3, "first=%d", &first_pos);
	report_check("tier-2 rebuild replayed the recorded prompt (turn-2 batch first >= 20)",
	             dll.v7_engine_is_dead() == 0 && first_pos >= 20);
	if (meta3)
		printf("  turn-2 batch meta: [%.*s]\n", (int)meta3_len, meta3);
	/* NOTE: ltm_probe_meta holds the LAST flush's batch -- turn 2's -- so
	 * the first= value asserted above is turn 2's opening evicted position,
	 * not the pre-rewrite ingest batch's (s5's first=4 assertion covers
	 * the meta format). The bound, not an exact value, is asserted
	 * because the post-rebuild continuation length is model behavior. */

	/* ---- s6: disable contract ---- */
	printf(" [s6] disable contract ...\n");
	int disable_rc = dll.v7_set_streaming(0, 1, 1);
	report_check("disable refused (-1) at wrapped pos > C", disable_rc == -1 && dll.v7_engine_is_dead() == 0);
	run_fresh_context(&dll, db_path);          /* fresh session: pos = 0, streaming off */
	disable_rc = dll.v7_set_streaming(0, 1, 1);
	report_check("disable at pos = 0 returns 0 (legal)", disable_rc == 0 && dll.v7_engine_is_dead() == 0);
	int pos_after_disable = dll.v7_get_position();
	report_check("session position queryable after disable", pos_after_disable >= 0);

	/* ---- oracle sanity: the plan's reference model agrees with itself on the
	 * ---- boundary shapes the engine test exercises (h == C-1 and pos = C) */
	int oracle_ok = 1;
	oracle_ok &= (stream_slot(11, 4, 8, 12) == 11);   /* last linear token: slot == pos */
	oracle_ok &= (stream_slot(12, 4, 8, 12) == 4);    /* first wrapped: S + (C-S) mod W = S + 0 */
	oracle_ok &= (stream_slot(20, 4, 8, 12) == 4);    /* S + (20-4) % 8 = 4 + 0 */
	oracle_ok &= (stream_slot(21, 4, 8, 12) == 5);    /* S + (21-4) % 8 = 4 + 1 */
	oracle_ok &= (stream_expected_distance(12, 0, 4, 8, 12) == 11); /* sink: C-1 - 0 */
	oracle_ok &= (stream_expected_distance(12, 4, 4, 8, 12) == 0);  /* self pair: fresh row at h */
	oracle_ok &= (stream_expected_distance(12, 11, 4, 8, 12) == 1); /* stale row: (h+W) - o = 12-11 */
	report_check("stream_ref oracle boundary shapes", oracle_ok);

	printf("\n... v_stream_validate summary: %d passed, %d failed ...\n", passing_tests_counter, failing_tests_counter);
	v7_unload(&dll);
	if (failing_tests_counter > 0)
		return 1;
	printf("v_stream_validate: all passed\n");
	return 0;
}
