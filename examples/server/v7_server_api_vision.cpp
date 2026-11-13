// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_api_vision.cpp -- POST /v1/vision/chat: an image turn on the
 * chat session (relocated verbatim from v7_server_main.cpp). */

#include "v7_server_common.h"

/* ========================================================================
 * POST /v1/vision/chat — image turn on the chat session
 * ======================================================================== */

void handle_vision_chat(SOCKET fd, char *body, int body_len, char *model_id)
{
	vision_request req;

	request_counter.fetch_add(1); /* per-request turn id for logs + response ids */

	if (parse_vision_request(body, body_len, &req) != 0 || !req.image_b64 || req.image_b64_len == 0)
	{
		vision_request_free(&req); /* parse failed partway -- the free releases any allocated field */
		send_error(fd, 400, "image_b64 (base64 PNG/JPEG) is required", "invalid_request_error");
		return;
	}

	/* base64 -> bytes -> GDI+ decode -> raw RGB. Data URLs
	 * ("data:image/png;base64,....") are stripped to the payload; the
	 * decoder itself skips whitespace. */
	char *b64_payload = req.image_b64;
	if (b64_payload && strncmp(b64_payload, "data:", 5) == 0)
	{
		char *comma = strchr(b64_payload, ',');
		if (comma)
		{
			b64_payload = comma + 1;
			req.image_b64_len -= (int)(b64_payload - req.image_b64);
		}
	}
	int cap = (req.image_b64_len / 4 + 1) * 3;
	unsigned char *img_bytes = (unsigned char *)malloc((int)cap);
	if (!img_bytes)
	{
		vision_request_free(&req);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "out of memory decoding image", "internal_error");
		send_response(fd, 500, "application/json", err, elen);
		return;
	}
	int img_len = b64_decode(b64_payload, req.image_b64_len, img_bytes, cap);
	free(req.image_b64);
	req.image_b64 = 0; /* freed — a death path below must not free it twice */
	if (img_len <= 0)
	{
		free(img_bytes);
		vision_request_free(&req);
		send_error(fd, 400, "image_b64 is not valid base64", "invalid_request_error");
		return;
	}

	decoded_image img;
	if (image_decode(img_bytes, img_len, &img) != 0)
	{
		free(img_bytes);
		vision_request_free(&req);
		send_error(fd, 400, "image could not be decoded (PNG/JPEG/BMP/GIF supported)", "invalid_request_error");
		return;
	}
	free(img_bytes);

	/* temp/top_p are parsed but NOT applied -- the server owns the Qwen3
	 * model-card sampler defaults (mode-aware re-arming in
	 * set_engine_thinking); a request-level override is deliberately
	 * not honored. */
	if (req.sampler_fields_present)
		printf("[v7_server] temperature/top_p from the request are not applied (sampler is engine-owned)\n");

	/* An empty prompt is a client bug, not a hidden default: answer 400. */
	if (!req.prompt || !req.prompt[0])
	{
		vision_request_free(&req);
		send_error(fd, 400, "prompt is required for a vision turn", "invalid_request_error");
		return;
	}

	/* Generation budget: the client's max_tokens when > 0 (restored to the
	 * server default right after the turn), else the server default
	 * (-maxgen, 0 = unlimited). */
	server_dll.v7_set_max_gen(req.max_tokens > 0 ? req.max_tokens : g_max_gen);

	char *vision_prompt = req.prompt;
	int reset = session_take_reset();
	/* Sidecar turn counter (same contract as the text paths): an image turn
	 * IS a chat turn for SectorLogo scripts -- without this they desync
	 * after the first image. */
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
	static char *inference_out = 0;
	static int inference_out_cap = 0;
	if (!scratch_grow(&inference_out, &inference_out_cap, MAX_INFERENCE_OUT))
	{
		free(img.rgb);
		vision_request_free(&req);
		send_error(fd, 500, "out of memory allocating the inference output", "internal_error");
		return;
	}
	if (run_turn(vision_prompt, 0, img.rgb, img.width, img.height, inference_out, inference_out_cap, 1, reset))
	{
		free(img.rgb);
		vision_request_free(&req);
		handle_engine_death(fd);
		return;
	}
	free(img.rgb);
	server_dll.v7_set_max_gen(g_max_gen); /* the turn is over: the server default governs the next turn */
	int prompt_tokens = (int)turn_stats[4];
	int completion_tokens = (int)turn_stats[5];

	/* finish_reason: only the output-BUFFER cut can report "length" here
	 * (a max_tokens cut is reported by the chat/completions endpoints; on
	 * the vision path the turn ran at the budget above). Same
	 * whole-turn-length contract as the chat endpoint (the reasoning
	 * prefix would hide the cut from an answer-length check). */
	int total_len = (int)strlen(inference_out);

	/* Split reasoning from answer (same wire format as the text chat): the
	 * vision turn runs thinking=1, so the raw think markup must not leak
	 * into the OpenAI content field. */
	int reasoning_len = 0;
	int answer_start = split_think(inference_out, total_len, 1, &reasoning_len);
	char *answer = inference_out + answer_start;
	char *reasoning = reasoning_len > 0 ? inference_out : 0;
	int out_len = (int)strlen(answer);
	const char *finish_reason = (total_len >= inference_out_cap - 64) ? "length" : "stop";

	/* Full request + full response in ONE block AFTER the inference call, so
	 * it never interleaves with the DLL's own output during generation.
	 * answer and reasoning are logged SEPARATELY: split_think NUL-terminates
	 * the buffer at the reasoning end, so printing inference_out alone would
	 * show only the reasoning and lose the answer. */
	printf("[v7_server] turn %d (vision): image %dx%d reset=%d\n", request_counter.load(), img.width, img.height, reset);
	printf("[v7_server] prompt: \"");
	print_sanitized(vision_prompt);
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

	char *model_str = model_id;
	if (req.model && req.model[0])
		model_str = req.model;
	/* Response body grown to the exact worst case -- no content can drop. */
	static char *resp_body = 0;
	static int resp_body_cap = 0;
	if (!scratch_grow(&resp_body, &resp_body_cap, 6 * out_len + 6 * reasoning_len + 6 * (int)strlen(model_str) + 4096))
	{
		vision_request_free(&req);
		send_error(fd, 500, "out of memory building the response", "server_error");
		return;
	}
	int resp_len = build_chat_response_ex(resp_body, resp_body_cap, answer, out_len, reasoning, reasoning_len, model_str, request_counter.load(), 0, finish_reason, prompt_tokens, completion_tokens);
	if (resp_len <= 0)
	{
		vision_request_free(&req);
		send_error(fd, 500, "out of memory building the response", "server_error");
		return;
	}
	send_response(fd, 200, "application/json", resp_body, resp_len);
	vision_request_free(&req);
}

