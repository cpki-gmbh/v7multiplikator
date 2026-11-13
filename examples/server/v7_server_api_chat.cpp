// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_api_chat.cpp -- the inference endpoints: /v1/chat/completions,
 * /v1/decide and /v1/completions (relocated verbatim from
 * v7_server_main.cpp). */

#include "v7_server_common.h"

/* POST /v1/chat/completions — multi-turn chat inference */
void handle_chat_completions(SOCKET fd, char *body, int body_len, char *model_id)
{
	chat_request req;

	request_counter.fetch_add(1); /* per-request turn id for logs + response ids */

	if (parse_chat_request(body, body_len, &req) != 0 || !req.user_prompt || req.user_len == 0)
	{
		chat_request_free(&req);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "messages must contain at least one user message", "invalid_request_error");
		send_response(fd, 400, "application/json", err, elen);
		return;
	}

	/* temp/top_p are parsed but NOT applied -- the server owns the Qwen3
	 * model-card sampler defaults (mode-aware re-arming in
	 * set_engine_thinking) and a request-level override is deliberately
	 * not honored. Say so ONCE per request that carries either field, so a
	 * client debugging sampling behavior is not left guessing. */
	if (req.sampler_fields_present)
		printf("[v7_server] temperature/top_p from the request are not applied (sampler is engine-owned)\n");

	/* Stateless clients: this server keeps ONE KV session and normally
	 * consumes ONLY the last user message (clients send just the new turn).
	 * A client that sends FULL history every call (any assistant message or
	 * more than one user message in the array) would double every turn into
	 * the KV — those requests are STATELESS and force a fresh session so
	 * the request's own history IS the context. The counts come from the
	 * PARSER (struct fields), never from raw-body substring matching: a
	 * user prompt that quotes JSON containing "role":"assistant" must not
	 * flip the session mode. */
	int stateless = 0;
	if (req.assistant_count > 0 || req.user_count > 1)
	{
		stateless = 1; /* full-history client: the request carries its own context */
		printf("[v7_server] request carries its own history -- server session reset (stateless client)\n");
	}

	/* Generation budget: the client's max_tokens when > 0, else the server
	 * default (-maxgen, 0 = unlimited — only EOS and the output buffer bound
	 * the turn, Sec.154). */
	if (req.max_tokens <= 0)
		req.max_tokens = g_max_gen;
	server_dll.v7_set_max_gen(req.max_tokens);

	/* Call v7_chat — first turn after reset uses reset=1 (clears KV cache),
	 * all subsequent turns use reset=0 (preserves KV cache for multi-turn).
	 * A stateless request ALWAYS resets: the KV would otherwise continue a
	 * conversation the client never meant to join. The system prompt rides
	 * along on every turn (the DLL renders it into the turn template). */
	int reset = session_take_reset();
	/* Stateless full-history clients: the TRANSCRIPT (all messages, in
	 * order) becomes the turn prompt — the request's own history IS the
	 * context, exactly as the reset promises. The chat template still
	 * wraps the turn; system is already embedded in the transcript. */
	char *turn_prompt = (char *)req.user_prompt;
	char *sys_ptr = 0;
	if (req.system_prompt)
		sys_ptr = (char *)req.system_prompt;
	if (stateless && req.transcript)
	{
		turn_prompt = req.transcript;
		sys_ptr = 0;
	}
	if (stateless)
		{
			reset = 1;
			/* the stateless transcript JUST replaced the KV -- the web page must */
			/* not keep showing the old conversation over the fresh session, and */
			/* the think-div numbering restarts with it (same as session_reset) */
			chat_history_clear(); /* the entries own their malloc'd text */
			g_think_counter = 0;
		}
	/* Thinking toggle, API side: a sidecar hooks the think tag and does
	 * NOTHING on a no-think turn -- with a sidecar loaded, thinking stays
	 * ON and we SAY SO (same contract as the web path; an explicit
	 * enable_thinking:false from the client is overridden while the hook
	 * is armed). */
	if (hook_enabled && !req.thinking)
	{
		req.thinking = 1;
		printf("[v7_server] sidecar active: thinking forced ON\n");
	}

	/* Sidecar turn counter (same contract as the web path): "__server_chat__turn"
	 * is 0 on the reset turn, +1 per continuation — written to the microdb
	 * before the turn so a SectorLogo sidecar can read it mid-generation. */
	if (reset == 1)
		g_server_chat_turn = 0;
	else
		g_server_chat_turn++;
	char turn_str[16];
	int turn_len = snprintf(turn_str, sizeof(turn_str), "%d", g_server_chat_turn);
	if (!server_dll.v7_db_add("__server_chat__turn", turn_str, turn_len + 1, 0))
		printf("[v7_server] WARNING: v7_db_add('__server_chat__turn') failed -- sidecars gate on a stale turn number\n");
	/* usage counts come from the engine's statistics block (slots 4/5 hold
	 * the prefill and generation token counts of the turn) */
	unsigned __int64 *turn_stats = (unsigned __int64 *)server_dll.v7_get_generation_statistic();
	for (int i = 0; i < 10; i++)
		turn_stats[i] = 0;
	/* the turn output buffer is GROWN to fit the turn: a longer answer
	 * doubles it instead of being cut at a fixed bound */
	static char *turn_out = 0;
	static int turn_out_cap = 0;
	if (!scratch_grow(&turn_out, &turn_out_cap, MAX_INFERENCE_OUT))
	{
		chat_request_free(&req);
		send_error(fd, 500, "out of memory allocating the inference output", "internal_error");
		return;
	}
	if (run_turn(turn_prompt, sys_ptr, 0, 0, 0, turn_out, turn_out_cap, req.thinking, reset))
	{
		chat_request_free(&req);
		handle_engine_death(fd);
		return;
	}
	int prompt_tokens = (int)turn_stats[4];
	int completion_tokens = (int)turn_stats[5];

	/* finish_reason: the engine trims trailing whitespace before returning,
	 * so a turn cut by the output BUFFER lands within a few bytes of the
	 * cap -- the check must use the WHOLE turn length read BEFORE
	 * split_think (with thinking on, a buffer-cut turn carries a long
	 * reasoning prefix, so the ANSWER length alone never approaches the
	 * cap and "length" would never be reported). */
	int total_len = (int)strlen(turn_out);

	/* Split reasoning from answer: on thinking turns the rescue guarantees a
	 * closed think block, so content alone would carry raw reasoning to
	 * OpenAI-compatible clients. reasoning_content holds the think block;
	 * content holds ONLY the answer. A thinking-off turn is never split. */
	int reasoning_len = 0;
	int answer_start = split_think(turn_out, total_len, req.thinking, &reasoning_len);
	char *answer = turn_out + answer_start;
	char *reasoning = reasoning_len > 0 ? turn_out : 0;
	int out_len = (int)strlen(answer);

	/* finish_reason: the output-BUFFER cut (whole turn near the cap) AND a
	 * client max_tokens cut (the turn generated up to the budget) both
	 * report "length" -- OpenAI clients retry/continue on "length" and
	 * must never see a budget cut disguised as "stop". */
	const char *finish_reason = (total_len >= turn_out_cap - 64 || (req.max_tokens > 0 && completion_tokens >= req.max_tokens)) ? "length" : "stop";

	/* Full request + full response in ONE block AFTER the inference call.
	 * answer and reasoning are logged SEPARATELY: split_think NUL-terminates
	 * the buffer at the reasoning end, so printing turn_out alone would show
	 * only the reasoning and lose the answer. */
	printf("[v7_server] turn %d (chat): max_tokens=%d reset=%d thinking=%d\n", request_counter.load(), req.max_tokens, reset, req.thinking);
	if (req.system_prompt)
	{
		printf("[v7_server] system: \"");
		print_sanitized(req.system_prompt);
		printf("\"\n");
	}
	printf("[v7_server] prompt: \"");
	print_sanitized(turn_prompt);
	printf("\"\n");
	if (reasoning_len > 0)
	{
		printf("[v7_server] reasoning:\n");
		print_sanitized(reasoning);
		printf("\n");
	}
	printf("[v7_server] answer:\n");
	print_sanitized(answer);
	printf("\n");

	/* Build OpenAI-format JSON response. NOTE: stream:true emits the full
	 * SSE chunk SEQUENCE as one block AFTER the whole turn -- real
	 * token-level streaming would need a per-token callback from the DLL.
	 * The body buffer is grown to the exact worst case (every content,
	 * reasoning and model byte can escape to \u00xx), so a response can
	 * never silently DROP its content. */
	char *model_str = model_id;
	if (req.model && req.model[0])
		model_str = req.model;
	static char *resp_body = 0;
	static int resp_body_cap = 0;
	int resp_need = 6 * out_len + 6 * reasoning_len + 6 * (int)strlen(model_str) + 4096;
	if (!scratch_grow(&resp_body, &resp_body_cap, resp_need))
	{
		chat_request_free(&req);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "out of memory building the response", "server_error");
		send_response(fd, 500, "application/json", err, elen);
		return;
	}
	int resp_len = build_chat_response_ex(resp_body, resp_body_cap, answer, out_len, reasoning, reasoning_len, model_str, request_counter.load(), req.stream, finish_reason, prompt_tokens, completion_tokens);
	if (resp_len <= 0)
	{
		chat_request_free(&req);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "out of memory building the response", "server_error");
		send_response(fd, 500, "application/json", err, elen);
		return;
	}
	send_response(fd, 200, req.stream ? (const char *)"text/event-stream" : (const char *)"application/json", resp_body, resp_len);
	chat_request_free(&req);
}

/* POST /v1/decide — typed decision (docs/typed_decisions.md): build the raw
 * decision prompt, one one-token-slot raw generate, probabilities per option
 * label via v7_get_token_probs. Public API only (§149); the generate runs
 * under a LOCAL greedy fence (temp forced 0 around it — a sampled decision
 * would waver between labels across identical requests — then the session's
 * Qwen3 model-card value is restored). v7_generate is
 * session-transparent (private KV; pos/rng/history saved+restored inside
 * generate.inc), so the chat session needs neither set_position nor a forced
 * reset — same precedent as the /v1/completions handler. */
void handle_decide(SOCKET fd, char *body, int body_len)
{
	decide_request req;

	request_counter.fetch_add(1); /* per-request turn id for logs + response ids */

	if (parse_decide_request(body, body_len, &req) != 0 || !req.state || !req.question || req.num_options < 2 || req.num_options > MAX_DECIDE_OPTIONS)
	{
		decide_request_free(&req);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "decide needs kind, state, question and 2..26 options", "invalid_request_error");
		send_response(fd, 400, "application/json", err, elen);
		return;
	}

	/* kind validation: the kind is REQUIRED (a missing kind is a client
	 * bug, not a default), and the parser only accepts the three
	 * documented kinds -- so the mapping below is total. */
	if (!req.kind[0])
	{
		send_error(fd, 400, "decide needs kind, state, question and 2..26 options", "invalid_request_error");
		decide_request_free(&req);
		return;
	}
	int expected_options = 0;
	if (strcmp(req.kind, "noul") == 0)
		expected_options = 2;
	else if (strcmp(req.kind, "score") == 0)
		expected_options = 6;
	else if (strcmp(req.kind, "choice") == 0)
		expected_options = -1; /* 2..MAX */
	if (expected_options > 0 && req.num_options != expected_options)
	{
		char err[512];
		char msg[256];
		snprintf(msg, sizeof(msg), "kind '%s' needs exactly %d options, got %d", req.kind, expected_options, req.num_options);
		int elen = build_error_response(err, sizeof(err), msg, "invalid_request_error");
		send_response(fd, 400, "application/json", err, elen);
		decide_request_free(&req);
		return;
	}

	/* verbalizer ids: every option label must encode to exactly one token.
	 * Duplicates are REJECTED BY TOKEN ID: two different strings that encode
	 * to the same id (or two identical labels) produce identical
	 * probabilities and the argmax silently picks the first. */
	int label_ids[MAX_DECIDE_OPTIONS];
	for (int i = 0; i < req.num_options; i++)
	{
		int label_tokens[8];
		int num_label_tokens = 0;
		server_dll.v7_encode(req.options[i], label_tokens, &num_label_tokens, 8); /* encode() bounds the write by max_tokens (encode.inc) */
		if (num_label_tokens != 1)
		{
			char err[512];
			char msg[256];
			snprintf(msg, sizeof(msg), "option label '%s' encodes to %d tokens -- labels must be single-token", req.options[i], num_label_tokens);
			int elen = build_error_response(err, sizeof(err), msg, "invalid_option_label");
			send_response(fd, 400, "application/json", err, elen);
			decide_request_free(&req);
			return;
		}
		label_ids[i] = label_tokens[0];
		for (int j = 0; j < i; j++)
		{
			if (label_ids[i] == label_ids[j])
			{
				char err[512];
				char msg[256];
				snprintf(msg, sizeof(msg), "duplicate option label '%s' -- options must be distinct tokens", req.options[i]);
				int elen = build_error_response(err, sizeof(err), msg, "invalid_option_label");
				send_response(fd, 400, "application/json", err, elen);
				decide_request_free(&req);
				return;
			}
		}
	}

	/* raw decision prompt — no chat template. Malloc'd to the exact size of
	 * every part (the sizes are computable from the request), so the old
	 * fixed 16 KB buffer and its "budget exceeded" 400 are gone -- that
	 * error told the client to shorten text the server itself had
	 * truncated. */
	int prompt_need = 7 + (int)strlen(req.kind) + 1   /* "[kind] " + kind + "\n" */
	                  + 8 + (int)strlen(req.state) + 1    /* "[state] " + state + "\n" */
	                  + 11 + (int)strlen(req.question) + 1 /* "[question] " + question + "\n" */
	                  + 10;                                /* "[options]\n" */
	for (int i = 0; i < req.num_options; i++)
		prompt_need += 4 + (int)strlen(req.options[i]); /* "%c) " + label + "\n" */
	prompt_need += 12 + 1;                              /* "\n[decision]:" + NUL */
	char *prompt = (char *)malloc((size_t)prompt_need);
	if (!prompt)
	{
		decide_request_free(&req);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "out of memory building the decision prompt", "server_error");
		send_response(fd, 500, "application/json", err, elen);
		return;
	}
	int plen = sprintf(prompt, "[kind] %s\n[state] %s\n[question] %s\n[options]\n", req.kind, req.state, req.question);
	for (int i = 0; i < req.num_options; i++)
		plen += sprintf(prompt + plen, "%c) %s\n", 'A' + i, req.options[i]);
	sprintf(prompt + plen, "\n[decision]:");

	server_dll.v7_set_max_gen(1); /* decision generate: EXACTLY one token -- the logits of the decision position stay in place for get_token_probs (the 8-byte output buffer alone also bounds it, but the budget makes the one-token stop explicit) */
	/* the ONE-token decision must be DETERMINISTIC: the session sampler now
	 * runs the Qwen3 model-card defaults (temp 0.6/0.7), which would sample
	 * the label and waver across identical requests -- fence greedy around
	 * the raw generate and restore the session value right after. With
	 * temp 0 the top_p/top_k filters cannot change the argmax. */
	float decide_saved_temp = g_sampler_temp;
	g_sampler_temp = 0.0f;
	server_dll.v7_set_sampler_temperature(0.0f);
	char gen_out[8]; /* one token slot: generate stops after the decision token (8 bytes cover a 4-byte UTF-8 sequence + NUL) */
	server_dll.v7_generate(prompt, gen_out, (int)sizeof(gen_out), 1);
	free(prompt); /* the decision prompt is transient: the generate consumed it */
	g_sampler_temp = decide_saved_temp;
	server_dll.v7_set_sampler_temperature(decide_saved_temp);
	server_dll.v7_set_max_gen(g_max_gen); /* restore the server budget immediately: the one-token fence must not leak a max_gen of 1 into later turns */
	if (hook_enabled)
		hook_executed = 1; /* generate runs the token hook too — the sidecar may have initialized here */
	if (server_dll.v7_engine_is_dead())
	{
		handle_engine_death(fd);
		decide_request_free(&req);
		return;
	}

	float probs[MAX_DECIDE_OPTIONS] = { 0 }; /* zero-filled: a get_token_probs that skips an option must not argmax over stack garbage */
	server_dll.v7_get_token_probs(label_ids, req.num_options, probs);
	if (server_dll.v7_engine_is_dead())
	{
		handle_engine_death(fd);
		decide_request_free(&req);
		return;
	}

	int best_index = 0;
	for (int i = 1; i < req.num_options; i++)
	{
		if (probs[i] > probs[best_index])
			best_index = i;
	}

	printf("[v7_server] decision %d (%s): \"", request_counter.load(), req.kind);
	print_sanitized(req.question);
	printf("\"\n");
	printf("[v7_server] probs:");
	for (int i = 0; i < req.num_options; i++)
	{
		printf(" ");
		print_sanitized(req.options[i]);
		printf("=%.4f", probs[i]);
	}
	printf("  -> ");
	print_sanitized(req.options[best_index]);
	printf("\n");

	/* Response body grown to the exact worst case: every option label can
	 * appear twice (options[] + choice), every byte escaping to \u00xx,
	 * plus the probabilities array and chrome. */
	static char *resp_body = 0;
	static int resp_body_cap = 0;
	int resp_need = 1024;
	for (int i = 0; i < req.num_options; i++)
		resp_need += 12 * (int)strlen(req.options[i]) + 32;
	if (!scratch_grow(&resp_body, &resp_body_cap, resp_need))
	{
		decide_request_free(&req);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "out of memory building the response", "server_error");
		send_response(fd, 500, "application/json", err, elen);
		return;
	}
	int resp_len = build_decide_response(resp_body, resp_body_cap, &req, probs, best_index);
	if (resp_len <= 0)
	{
		decide_request_free(&req);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "out of memory building the response", "server_error");
		send_response(fd, 500, "application/json", err, elen);
		return;
	}
	send_response(fd, 200, "application/json", resp_body, resp_len);
	decide_request_free(&req);
}

/* POST /v1/completions — single-turn generate */
void handle_completions(SOCKET fd, char *body, int body_len, char *model_id)
{
	completion_request req;

	request_counter.fetch_add(1); /* per-request turn id for logs + response ids */

	if (parse_completion_request(body, body_len, &req) != 0 || !req.prompt || req.prompt_len == 0)
	{
		send_error(fd, 400, "prompt must be provided", "invalid_request_error");
		completion_request_free(&req);
		return;
	}

	/* temp/top_p are parsed but NOT applied -- the server owns the Qwen3
	 * model-card sampler defaults; a request-level override is
	 * deliberately not honored. */
	if (req.sampler_fields_present)
		printf("[v7_server] temperature/top_p from the request are not applied (sampler is engine-owned)\n");

	/* Generation budget: same rule as /v1/chat/completions -- client value
	 * when > 0, else the server default (-maxgen, 0 = unlimited). */
	if (req.max_tokens <= 0)
		req.max_tokens = g_max_gen;
	server_dll.v7_set_max_gen(req.max_tokens);

	/* v7_generate creates its own independent KV cache internally.
	   The position is saved/restored by generate.inc — no manual reset needed.
	   The chat fallback (v7_chat with reset=1) is FORBIDDEN: it destroys the
	   shared session's KV cache, silently breaking subsequent /chat/completions
	   turns. */
	/* usage counts come from the engine's statistics block (slots 4/5 hold
	 * the prefill and generation token counts of the turn) */
	unsigned __int64 *turn_stats = (unsigned __int64 *)server_dll.v7_get_generation_statistic();
	for (int i = 0; i < 10; i++)
		turn_stats[i] = 0;
	/* the output buffer is GROWN to fit the response: a longer answer
	 * doubles it instead of being cut at a fixed bound */
	static char *inference_out = 0;
	static int inference_out_cap = 0;
	if (!scratch_grow(&inference_out, &inference_out_cap, MAX_INFERENCE_OUT))
	{
		completion_request_free(&req);
		send_error(fd, 500, "out of memory allocating the inference output", "internal_error");
		return;
	}
	server_dll.v7_generate((char *)req.prompt, inference_out, inference_out_cap, 0);
	if (hook_enabled)
		hook_executed = 1; /* sidecar hook (if on) has now initialized */
	if (server_dll.v7_engine_is_dead())
	{
		handle_engine_death(fd);
		completion_request_free(&req);
		return;
	}
	int prompt_tokens = (int)turn_stats[4];
	int completion_tokens = (int)turn_stats[5];

	/* Trim trailing whitespace, then back to a UTF-8 boundary (a byte-limited
	 * output buffer can cut inside a multi-byte sequence). */
	int out_len = trim_tail_utf8(inference_out);

	/* finish_reason: an output-BUFFER cut OR a max_tokens cut reports
	 * "length" (natural EOS reports "stop") -- same contract as the chat
	 * endpoint. */
	const char *finish_reason = (out_len >= inference_out_cap - 64 || (req.max_tokens > 0 && completion_tokens >= req.max_tokens)) ? "length" : "stop";

	/* Full request + full response in ONE block AFTER the inference call. */
	printf("[v7_server] turn %d (completions): max_tokens=%d\n", request_counter.load(), req.max_tokens);
	printf("[v7_server] prompt: \"");
	print_sanitized(req.prompt);
	printf("\"\n");
	printf("[v7_server] response:\n");
	print_sanitized(inference_out);
	printf("\n");

	/* Build OpenAI-format JSON response (SSE sequence when stream:true).
	 * Body grown to the exact worst case -- no content can be dropped. */
	char *model_str = model_id;
	if (req.model && req.model[0])
		model_str = req.model;
	static char *resp_body = 0;
	static int resp_body_cap = 0;
	if (!scratch_grow(&resp_body, &resp_body_cap, 6 * out_len + 6 * (int)strlen(model_str) + 4096))
	{
		completion_request_free(&req);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "out of memory building the response", "server_error");
		send_response(fd, 500, "application/json", err, elen);
		return;
	}
	int resp_len = build_completion_response(resp_body, resp_body_cap, inference_out, out_len, model_str, request_counter.load(), finish_reason, req.stream, prompt_tokens, completion_tokens);
	if (resp_len <= 0)
	{
		completion_request_free(&req);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "out of memory building the response", "server_error");
		send_response(fd, 500, "application/json", err, elen);
		return;
	}
	send_response(fd, 200, req.stream ? (const char *)"text/event-stream" : (const char *)"application/json", resp_body, resp_len);
	completion_request_free(&req);
}

