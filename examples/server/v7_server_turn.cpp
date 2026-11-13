// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_turn.cpp -- engine thinking mode, session reset, the dead-engine
 * reporter, one chat/image turn with the missing-think-tag rescue, the
 * reasoning/answer split and the reset-flag take (relocated verbatim from
 * v7_server_main.cpp). */

#include "v7_server_common.h"

/* Set the engine's thinking mode AND record it. Every handler declares its
 * desired mode explicitly per turn (the web checkbox, the API default ON);
 * the think-tag rescue restores the previous mode instead of a hardcoded 1,
 * so a web turn with the box unchecked can never silently re-enable
 * thinking for the API path.
 *
 * The switch also re-arms the Qwen3 model-card sampling pair for the mode
 * (the researched best-practice values):
 *   thinking   : temperature 0.6, top_p 0.95 (greedy DEGRADES Qwen3 --
 *                endless repetitions),
 *   non-thinking: temperature 0.7, top_p 0.8;
 * The think-tag rescue saves g_sampler_temp before its ladder and restores
 * through it, so the rescue always hands back the CURRENT mode's pair. */
void set_engine_thinking(int on)
{
	server_dll.v7_enable_thinking(on);
	g_engine_thinking = on;
	g_sampler_temp = (on) ? 0.6f : 0.7f;
	server_dll.v7_set_sampler_temperature(g_sampler_temp);
	server_dll.v7_set_sampler_topp((on) ? 0.95f : 0.8f);
}

/* Sidecar-set chat background: a SectorLogo script writes the image FILENAME
 * into the microdb under __server_html_background_image (DatabaseSet). The
 * page render and GET /background read it here; the file itself is served
 * from backgrounds\<name> relative to the server's working directory.
 * Returns 1 with the name copied to out when a non-empty value is set. */


/* One full session reset: fresh KV on the next turn AND a cleared web
 * history. Both the API and the web reset paths go through THIS —
 * resetting only one side left the page showing an old conversation over
 * a fresh KV (or the reverse). */
void session_reset(void)
{
	reset_next_turn.store(1);
	chat_history_clear(); /* the entries own their malloc'd text -- free, don't just zero the count */
	g_think_counter = 0; /* history is empty — restart div ids from zero */
}

/* A call hit a dead engine: tell the client, mark the server for restart.
 * The restart loop (server.bat) only stops on the clean codes (42 quit, 43
 * port busy) — any other code brings the process back with a fresh engine
 * instead of serving empty 200s forever. Returns 1 so handlers can
 * `return handle_engine_death(fd);`. */
int handle_engine_death(SOCKET fd)
{
	/* the reason stays in the CONSOLE log only: it may carry internal
	 * paths -- the HTTP client gets a generic message */
	printf("[v7_server] engine died: %s\n", server_dll.v7_get_death_reason());
	char err[256];
	int elen = build_error_response(err, sizeof(err), (char *)"internal engine error -- the server will restart", "engine_error");
	send_response(fd, 500, "application/json", err, elen);
	server_exit_code.store(1); /* NOT 42: the bat loop restarts us */
	server_running.store(0);   /* accept loop exits after the current handler */
	return 1;
}

/* answer_has_printable -- true when the text AFTER the closing think tag
 * (think turn) or the whole output (thinking-off / no-tag turn) carries at
 * least one printable, non-whitespace character. A turn that ended right at
 * (or just after) the close tag without saying anything has NO printable
 * answer -- that is the "closed but empty" cut-off, and it must trigger the
 * same rescue ladder as a fully missing tag. Only printable ASCII/UTF-8
 * lead bytes count (+ U+00A0 handled as whitespace); CR/LF/HT/SP and NULs
 * are the junk a close-tag tail leaves behind. */
static const char *closing_think_tag = "</think>";
int answer_has_printable(char *out, int thinking)
{
	/* find the answer start: after the closing think tag when one exists */
	char *answer = strstr(out, closing_think_tag);
	if (answer)
		answer += (int)strlen(closing_think_tag);
	else
		answer = out; /* no closed block: the answer is the entire output */

	for (char *c = answer; *c; c++)
	{
		unsigned char u = (unsigned char)*c;
		if (u <= ' ')
			continue; /* NUL..space: control chars, CR/LF/HT/SP -- not printable content */
		return 1; /* any byte above space (ASCII letter/digit/punctuation or a
			   * UTF-8 lead byte 0xC2..0xF4 which is also above space) prints */
	}
	return 0;
}

/* One chat turn (text or image) with the missing-think-tag rescue.
 *
 * prompt/sys   — the user turn (sys is sent on every turn by the API handler;
 * img/w/h      — 0 for a text turn, else a raw 24bpp RGB image
 * out/out_cap  — output buffer (caller-owned)
 * thinking     — the mode THIS handler wants (recorded, restored)
 * reset        — 1 = fresh session, 0 = continue the KV
 *
 * Rescue contract (only when thinking is ON and the turn produced NO
 * closing think tag — i.e. the limit/EOS cut the reasoning):
 *   - reset=1: the retry is ALSO reset=1 with NO set_position. The old
 *     pos_before belongs to the PREVIOUS session; rewinding a fresh KV to
 *     it would continue at a stale offset over mixed rows.
 *   - reset=0: the retry rewinds to pos_before (pre-turn position), then
 *     re-runs the same turn at reset=0.
 *   - the retries run WITHOUT changing the generation budget: the caller
 *     has already armed the budget this turn must honor (a client
 *     max_tokens / the -maxgen server default, 0 = unlimited). A rerun at
 *     a WIDER budget would answer a max_tokens-bounded request with a 4x
 *     longer rescue answer, so the standing budget applies to the retries.
 *   - thinking AND the sampler temperature are restored to the PREVIOUS
 *     mode/value afterwards, never to a hardcoded 1 / 0.0.
 *   - a dead engine aborts the turn (out empty) and flags the restart:
 *     a turn ran on it pays its one FATAL, and no retry or fallback
 *     follows -- every retry already rewound the position, and calling
 *     v7_chat on a dead engine does nothing except look like progress.
 * Returns 0 on success, 1 when the engine died (caller: handle_engine_death).
 * A sidecar mid-turn flag carries into the retry, so a hook-injected turn
 * is not injected twice — an accepted degradation for the rescue path. */
int run_turn(char *prompt, char *sys_ptr, unsigned char *img, int img_w, int img_h,
             char *out, int out_cap, int thinking, int reset)
{
	int prev_thinking = g_engine_thinking;
	float prev_temp = g_sampler_temp; /* what WE last set through v7_set_sampler_temperature (0.0 = the engine's greedy default) */
	int died = 0;

	set_engine_thinking(thinking);

	/* Position works for image turns too — the session position is where
	 * THIS turn starts regardless of modality (0 would rewind a
	 * continuation image turn to the SESSION start and corrupt the KV). */
	int pos_before = server_dll.v7_get_position();
	memset(out, 0, out_cap);
	if (img)
		server_dll.v7_chat_image(prompt, sys_ptr, img, img_w, img_h, out, out_cap, reset);
	else
		server_dll.v7_chat(prompt, sys_ptr, out, out_cap, reset);
	if (hook_enabled)
		hook_executed = 1; /* sidecar hook (if on) has now initialized */

	/* Trim trailing whitespace, then back to a UTF-8 boundary (a byte-limited
	 * output buffer can cut inside a multi-byte sequence). */
	int out_len = trim_tail_utf8(out);

	if (server_dll.v7_engine_is_dead())
	{
		printf("[v7_server] engine died during turn: %s\n", server_dll.v7_get_death_reason());
		out[0] = 0;
		set_engine_thinking(prev_thinking);
		reset_next_turn.store(1); /* the KV is undefined after a death — the NEXT turn must start fresh */
		return 1;
	}

	/* Rescue: only a THINKING turn can be missing its closing think tag, or
	 * carry one with an EMPTY / non-printable answer after it (the model
	 * closed the book and said nothing -- the same cut-off class, checked
	 * with answer_has_printable below). The retry ladder (user directive):
	 * three retries at DIFFERENT temperatures -- a cut-off think block is a
	 * sampling artifact, so the retry rerolls the same turn at t=0.7 / 1.0 /
	 * 1.3 -- before the no-thinking fallback. Each retry keeps the SAME
	 * reset posture (continue rewinds to pos_before, fresh-session stays
	 * fresh) and the SAME generation budget the caller armed (the client's
	 * max_tokens / -maxgen; a wider rescue budget would answer a bounded
	 * request with a 4x longer answer), and each must re-rollback the
	 * position: the previous retry CONSUMED KV rows, so retry n>1 rewinds
	 * too. A retry that closes the book AND says something printable is
	 * KEPT -- the no-think fallback never overwrites a good answer. The
	 * ladder restores the PREV thinking mode AND temperature afterwards,
	 * never a hardcoded 1 / 0.0. A dead engine aborts (below). */
	if (thinking && (strstr(out, closing_think_tag) == 0 || !answer_has_printable(out, thinking)))
	{
		int ladder_fixed = 0; /* a retry closed the block AND produced a printable answer */
		for (int retry = 0; retry < 3; retry++)
		{
			static const double retry_temp[3] = { 0.7, 1.0, 1.3 };
			int retry_reset = reset;
			if (reset == 0)
				server_dll.v7_set_position(pos_before); /* continue-path: rewind over THIS turn (and any prior retries) */
			else
				retry_reset = 1; /* fresh session: restart fresh, NEVER rewind into the old session's rows */

			printf("[v7_server] no closing think tag -- retry %d/3 at t=%.2f (%s)\n",
				retry + 1, retry_temp[retry], reset == 0 ? "KV rollback" : "fresh-session");
			g_sampler_temp = (float)retry_temp[retry];
			server_dll.v7_set_sampler_temperature(g_sampler_temp); /* the standing budget is NOT touched: the caller armed it for this turn */
			memset(out, 0, out_cap);
			if (img)
				server_dll.v7_chat_image(prompt, sys_ptr, img, img_w, img_h, out, out_cap, retry_reset);
			else
				server_dll.v7_chat(prompt, sys_ptr, out, out_cap, retry_reset);
			if (hook_enabled)
				hook_executed = 1;

			/* Trim, then judge: this retry either CLOSED the book AND said
			 * something printable (keep it: no further tries, no fallback)
			 * or it spent its budget and the next temperature follows. The
			 * trim is shared with the ORIGINAL turn's tail. */
			out_len = trim_tail_utf8(out);

			if (server_dll.v7_engine_is_dead())
			{
				printf("[v7_server] engine died during retry: %s\n", server_dll.v7_get_death_reason());
				out[0] = 0;
				died = 1;
				break; /* the KV is undefined: the fallback still runs ON A DEAD ENGINE -- skipped below */
			}
			if (out_len == 0)
				printf("[v7_server] think retry %d produced an empty output\n", retry + 1);
			else
				printf("[v7_server] think retry %d:\n%s\n", retry + 1, out);

			if (strstr(out, closing_think_tag) != 0 && answer_has_printable(out, thinking))
			{
				ladder_fixed = 1; /* the retry produced a usable answer -- never overwrite it with the fallback turn */
				break;
			}
		}

		/* The no-think fallback is the FOURTH and last try and runs ONLY
		 * when the ladder came up empty at all three temperatures. After a
		 * mid-ladder engine death the turn ends here (every retry already
		 * rewound the position; v7_chat on a dead engine would only look
		 * like progress), and a ladder success is precious n>1 output that
		 * the fallback MUST NOT replace. */
		if (!died && !ladder_fixed)
		{
			int fallback_reset = reset;
			if (reset == 0)
				server_dll.v7_set_position(pos_before); /* continue-path: rewind over everything above */
			else
				fallback_reset = 1;

			printf("[v7_server] no closing think tag -- no-think fallback\n");
			set_engine_thinking(0);
			g_sampler_temp = 0.0f;
			server_dll.v7_set_sampler_temperature(g_sampler_temp); /* deterministic fallback: the no-think retrial doesn't need a sampler roll */
			memset(out, 0, out_cap);
			if (img)
				server_dll.v7_chat_image(prompt, sys_ptr, img, img_w, img_h, out, out_cap, fallback_reset);
			else
				server_dll.v7_chat(prompt, sys_ptr, out, out_cap, fallback_reset);
			if (hook_enabled)
				hook_executed = 1;

			out_len = trim_tail_utf8(out);

			if (server_dll.v7_engine_is_dead())
			{
				printf("[v7_server] engine died during rescue: %s\n", server_dll.v7_get_death_reason());
				out[0] = 0;
				died = 1;
				reset_next_turn.store(1); /* undefined KV — next turn fresh */
			}
			else if (out_len == 0)
			{
				printf("[v7_server] no-think fallback produced an empty answer\n");
			}
			else
			{
				printf("[v7_server] no-think fallback:\n%s\n", out);
			}
		}
		else if (died)
		{
			reset_next_turn.store(1); /* undefined KV — next turn fresh (the death happened inside the ladder) */
		}
	}

	/* Restore the thinking mode AND the sampler temperature the turn
	 * started with: both were changed for the rescue. A dead engine takes
	 * the process down (the bat restarts it fresh), so its restore step
	 * would just call into a corpse. */
	set_engine_thinking(prev_thinking);
	if (!died)
	{
		g_sampler_temp = prev_temp;
		server_dll.v7_set_sampler_temperature(prev_temp);
	}
	return died;
}

/* Split a finished turn into reasoning + answer at the closing think tag.
 * thinking-on turns carry "reasoning\nanswer"; the OpenAI wire format
 * wants the reasoning in reasoning_content and ONLY the answer in content
 * (clients render content as the message body — raw reasoning in there
 * reads as the answer). On a thinking-OFF turn the whole output is the
 * answer: a no-think answer that merely MENTIONS the tag (the model
 * explaining the format) must never be split. Returns the answer start
 * offset; *reasoning_len gets the reasoning byte count (0 when the turn
 * has no think block).
 * The answer string is the ALREADY-NUL-TERMINATED tail starting at the
 * returned offset — nothing is written past it (terminating at answer_start
 * would NUL the first answer character and empty every content). */
int split_think(char *out, int out_len, int thinking, int *reasoning_len)
{
	const char *close_tag = "</think>";
	char *hit = thinking ? strstr(out, close_tag) : 0;
	if (!hit)
	{
		*reasoning_len = 0;
		return 0; /* no think block (thinking-off or raw generate): all answer */
	}
	int close_off = (int)(hit - out); /* close-tag offset in the ORIGINAL layout */
	int reasoning_bytes = close_off;
	int answer_at = close_off + (int)strlen(close_tag);
	/* strip a model-emitted open tag from the reasoning head: when the model
	 * generates its own open tag (rather than using the prompt-embedded
	 * one), it lands at offset 0 and would pollute reasoning_content AND
	 * shift the close tag. Move the reasoning text down so
	 * out[0..reasoning_bytes) is EXACTLY the reasoning text. */
	int skip = 0;
	if (reasoning_bytes >= 7 && memcmp(out, "<think>", 7) == 0)
		skip = 7;
	else if (reasoning_bytes >= 10 && (unsigned char)out[0] == 0xEF && (unsigned char)out[1] == 0xBB && (unsigned char)out[2] == 0xBF &&
	         memcmp(out + 3, "<think>", 7) == 0)
		skip = 10; /* BOM + open tag */
	if (skip)
	{
		reasoning_bytes -= skip;
		memmove(out, out + skip, (size_t)reasoning_bytes);
	}
	/* strip leading newlines from the answer */
	while (answer_at < out_len && (out[answer_at] == '\n' || out[answer_at] == '\r'))
		answer_at++;
	/* trim the reasoning's trailing whitespace/newlines */
	while (reasoning_bytes > 0 && (out[reasoning_bytes - 1] == '\n' || out[reasoning_bytes - 1] == '\r' || out[reasoning_bytes - 1] == ' '))
		reasoning_bytes--;
	int split_len = utf8_trim_to_boundary(out, reasoning_bytes);
	out[split_len] = '\0'; /* reasoning becomes a standalone NUL-terminated string (this byte is at or before the close tag — the answer starts later and is untouched) */
	*reasoning_len = split_len;
	return answer_at;
}

/* Take-and-clear the session-reset flag: the caller passes the returned
 * value as v7_chat's reset parameter. First turn after /v1/chat/reset
 * (or server start) resets the position to 0; later turns continue the KV. */
int session_take_reset(void)
{
	int reset_flag = reset_next_turn.load();
	reset_next_turn.store(0);
	return reset_flag;
}
