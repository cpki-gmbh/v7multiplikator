// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_web_handlers.cpp -- GET / (page), GET /background, and the
 * form endpoints /chat/submit + /chat/reset. */

#include "v7_server_common.h"

/* GET /background -- serve the sidecar-selected chat background image.
 * The filename comes from the __server_html_background_image microdb key
 * (set by a SectorLogo script via DatabaseSet); the file is read from
 * backgrounds\<name> relative to the server's working directory. The
 * content type is sniffed from the magic bytes, so the stored extension
 * does not have to match the real format. */

/* serve_backgrounds_file -- shared body of /background and /avatar: validate
 * the DB-sourced filename (plain name only, always inside backgrounds\),
 * read the file, sniff the format from the magic bytes and send it. Sent
 * with `Cache-Control: max-age=31536000, immutable`: every page fetch uses
 * a unique ?t= cache buster (the crossfade depends on the browser serving
 * the PREVIOUS turn's URL from cache and the current one from the net). */
static void serve_backgrounds_file(SOCKET fd, const char *image_name)
{
	char path[512];
	/* the name comes from a sidecar-written DB value: reject every path
	 * separator, parent reference and drive colon so it can only ever name
	 * a file INSIDE backgrounds\ */
	for (const char *scan = image_name; *scan; scan++)
	{
		if (*scan == '\\' || *scan == '/' || *scan == ':' || *scan == '?')
		{
			send_error(fd, 400, "image name must be a plain filename", "invalid_request_error");
			return;
		}
	}
	if (strncmp(image_name, "..", 2) == 0)
	{
		send_error(fd, 400, "image name must be a plain filename", "invalid_request_error");
		return;
	}
	snprintf(path, sizeof(path), "backgrounds\\%s", image_name);
	FILE *image_file = fopen(path, "rb");
	if (!image_file)
	{
		send_error(fd, 404, "image file missing", "not_found_error");
		return;
	}
	fseek(image_file, 0, SEEK_END);
	long image_size = ftell(image_file);
	rewind(image_file);
	if (image_size < 4)
	{
		fclose(image_file);
		send_error(fd, 400, "image is empty", "invalid_request_error");
		return;
	}
	char *image = (char *)malloc((size_t)image_size); /* exact-size read: no cap on a background */
	if (!image)
	{
		fclose(image_file);
		send_error(fd, 500, "out of memory reading the image", "internal_error");
		return;
	}
	int image_len = (int)fread(image, 1, image_size, image_file);
	fclose(image_file);
	const char *mime = "application/octet-stream";
	if ((unsigned char)image[0] == 0x89 && (unsigned char)image[1] == 0x50)
		mime = "image/png";
	else if ((unsigned char)image[0] == 0xFF && (unsigned char)image[1] == 0xD8)
		mime = "image/jpeg";
	else if (image[0] == 'G' && image[1] == 'I' && image[2] == 'F')
		mime = "image/gif";
	else if (image[0] == 'B' && image[1] == 'M')
		mime = "image/bmp";
	{
		char header[512];
		int hlen = sprintf(header,
		                   "HTTP/1.1 200 OK\r\n"
		                   "Content-Type: %s\r\n"
		                   "Content-Length: %d\r\n"
		                   "Cache-Control: max-age=31536000, immutable\r\n"
		                   "X-Content-Type-Options: nosniff\r\n"
		                   "Connection: close\r\n\r\n",
		                   mime, image_len);
		send_all(fd, header, hlen);
		send_all(fd, image, image_len);
	}
	free(image);
}

void handle_background(SOCKET fd)
{
	char background_name[2048];
	if (!background_image_name(background_name, sizeof(background_name)))
	{
		send_error(fd, 404, "no background image set", "not_found_error");
		return;
	}
	serve_backgrounds_file(fd, background_name);
}

/* GET /avatar -- the sidecar-selected persona picture shown on the bot's
 * messages; the filename comes from __server__chat_avatar_image. */
void handle_avatar(SOCKET fd)
{
	char avatar_name[2048];
	if (!avatar_image_name(avatar_name, sizeof(avatar_name)))
	{
		send_error(fd, 404, "no avatar image set", "not_found_error");
		return;
	}
	serve_backgrounds_file(fd, avatar_name);
}

void handle_index(SOCKET fd)
{
	int len = 0;
	char *page = build_chat_page(&len); /* grown inside web_render: ALL turns render, nothing is dropped */
	if (!page)
	{
		send_error(fd, 500, "out of memory building the page", "internal_error");
		return;
	}
	/* no-store: a bfcache restore would show the FROZEN page (locked inputs,
	 * fake bubbles) without re-running onload — never cache the live chat.
	 * X-Frame-Options keeps any other site from framing this page. */
	char header[512];
	int hlen = sprintf(header,
	                   "HTTP/1.1 200 OK\r\n"
	                   "Content-Type: text/html; charset=utf-8\r\n"
	                   "Content-Length: %d\r\n"
	                   "Cache-Control: no-store\r\n"
	                   "X-Content-Type-Options: nosniff\r\n"
	                   "X-Frame-Options: DENY\r\n"
	                   "Connection: close\r\n\r\n",
	                   len);
	send_all(fd, header, hlen);
	send_all(fd, page, len);
}

/* ========================================================================
 * POST /chat/submit — process a user message, call inference, redirect to /
 * ======================================================================== */

/* Decode one form-body value (the + convention and %XX escapes) into a
 * malloc'd exact-size buffer. Sets *saw_nul when a %00 decodes to an
 * embedded NUL (an invalid message: the caller answers 400). Returns the
 * buffer and its length via out_len, or 0 on allocation failure. */
static char *form_decode_value(char *value, int *out_len, int *saw_nul)
{
	int span = 0;
	while (value[span] && value[span] != '&')
		span++;
	char *decoded = (char *)malloc((size_t)span + 1);
	if (!decoded)
		return 0;
	int len = 0;
	char *qp = value;
	while (*qp && *qp != '&')
	{
		if (*qp == '+')
		{
			decoded[len++] = ' '; /* form-body convention */
			qp++;
		}
		else if (*qp == '%' && qp[1] && qp[2])
		{
			char h1 = qp[1], h2 = qp[2];
			int hi1 = (h1 >= '0' && h1 <= '9') || (h1 >= 'a' && h1 <= 'f') || (h1 >= 'A' && h1 <= 'F');
			int hi2 = (h2 >= '0' && h2 <= '9') || (h2 >= 'a' && h2 <= 'f') || (h2 >= 'A' && h2 <= 'F');
			if (hi1 && hi2)
			{
				char hex[3] = { h1, h2, 0 };
				char byte = (char)strtol(hex, 0, 16);
				if (byte == 0)
				{
					/* %00: an embedded NUL can never be part of a form
					 * message -- the C-string chain (inference prompt,
					 * logs) would truncate there while the length stayed
					 * larger. Report it; the caller rejects the message. */
					*saw_nul = 1;
					qp += 3;
					continue;
				}
				decoded[len++] = byte; /* the decoded BYTE: a %XX percent-encode of UTF-8 survives verbatim */
				qp += 3;
			}
			else
			{
				/* "%zz" is NOT an escape: strtol would silently produce 0,
				 * inserting a NUL that truncates the message and its
				 * display. Copy the '%' literally instead. */
				decoded[len++] = *qp++;
			}
		}
		else
		{
			decoded[len++] = *qp++;
		}
	}
	decoded[len] = 0;
	*out_len = len;
	return decoded;
}

void handle_chat_submit(SOCKET fd, char *body, int body_len)
{
	/* Extract the q= field from the form-encoded body. Only accept q= at the
	 * start of the body or right after a '&' — never a "q=" inside another
	 * field's value. The decoded message is malloc'd to the field's raw
	 * span (the decoded text can never exceed it), so a pasted document of
	 * any size reaches the model uncapped. */
	int saw_nul = 0;
	int user_len = 0;

	char *qp = find_field(body, "q=");
	char *user_msg = 0;
	if (qp)
		user_msg = form_decode_value(qp, &user_len, &saw_nul);
	if (!user_msg)
	{
		if (!qp)
		{
			/* no q= field at all: a raw POST with no message is a client
			 * bug -- answer 400, never a silent redirect */
			send_error(fd, 400, "the message field q= is required", "invalid_request_error");
			return;
		}
		send_error(fd, 500, "out of memory", "internal_error");
		return;
	}
	if (saw_nul)
	{
		/* a %00 in the decoded message: an invalid message, never a
		 * silently shortened one */
		send_error(fd, 400, "the message must not contain a NUL byte (%00)", "invalid_request_error");
		free(user_msg);
		return;
	}

	/* textarea newlines arrive as \r\n: normalize to \n (the model, the
	 * history and the transcript all expect plain \n) */
	{
		int clean_len = 0;
		for (int i = 0; i < user_len; i++)
			if (user_msg[i] != '\r')
				user_msg[clean_len++] = user_msg[i];
		user_len = clean_len;
		user_msg[user_len] = 0;
	}
	/* UTF-8 validation runs on the DECODED bytes BEFORE any boundary trim:
	 * the validator must see exactly what the client sent, and the trim
	 * (a cut can land inside a multi-byte sequence) only shortens a text
	 * that is already well-formed */
	if (!utf8_wellformed(user_msg, user_len))
	{
		printf("[v7_server] web form rejected: message is not valid UTF-8\n");
		send_error(fd, 400, "message must be valid UTF-8", "invalid_request_error");
		free(user_msg);
		return;
	}
	user_len = utf8_trim_to_boundary(user_msg, user_len);
	user_msg[user_len] = 0;

	/* Whitespace-only text is an empty turn: the JS checks trim(), but a
	 * raw POST bypasses the browser -- answer 400 instead of spending a
	 * whole inference (and a KV turn) on nothing. */
	int has_visible = 0;
	for (int i = 0; i < user_len; i++)
	{
		if (user_msg[i] != ' ' && user_msg[i] != '\t' && user_msg[i] != '\n')
		{
			has_visible = 1;
			break;
		}
	}
	if (user_len == 0 || !has_visible)
	{
		send_error(fd, 400, "the message is empty", "invalid_request_error");
		free(user_msg);
		return;
	}

	/* Call inference into a grown output buffer: a turn longer than the
	 * starting bound doubles the buffer instead of being cut. */
	static char *inference_out = 0;
	static int inference_out_cap = 0;
	if (!scratch_grow(&inference_out, &inference_out_cap, MAX_INFERENCE_OUT))
	{
		send_error(fd, 500, "out of memory allocating the inference output", "internal_error");
		free(user_msg);
		return;
	}
	request_counter.fetch_add(1); /* per-request turn id for logs + response ids */
	server_dll.v7_set_max_gen(g_max_gen); /* server default: unlimited (0) unless -maxgen set */
	/* Sampler untouched — the server never writes v7-internal state */

	/* Thinking toggle: an unchecked HTML checkbox is not submitted at all, so the
	 * field's presence IS the enabled state. run_turn sets the mode through
	 * set_engine_thinking for THIS turn only — the previous engine mode is
	 * restored afterwards, so the API path is never flipped by a web turn. */
	int thinking_on = 0;
	if (find_field(body, "think="))
		thinking_on = 1;
	/* A sidecar that hooks the think tag (RAG injection, memory) does
	 * NOTHING on a no-think turn -- there is no </think> piece to fire on.
	 * With a sidecar loaded, thinking stays ON; say so instead of silently
	 * running a dead hook. */
	if (hook_enabled && !thinking_on)
	{
		thinking_on = 1;
		printf("[v7_server] sidecar active: thinking forced ON\n");
	}
	thinking_enabled = thinking_on; /* echoed back by build_chat_page: the page reloads after every turn, so an uncheck would otherwise appear to revert */

	int reset = session_take_reset();
	/* Sidecar turn counter: "__server_chat__turn" is 0 on the reset turn and
	 * +1 per continuation. Written BEFORE the turn so a SectorLogo sidecar
	 * can read it mid-generation (DatabaseGet). */
	if (reset == 1)
		g_server_chat_turn = 0;
	else
		g_server_chat_turn++;
	char turn_str[16];
	int turn_len = snprintf(turn_str, sizeof(turn_str), "%d", g_server_chat_turn);
	if (!server_dll.v7_db_add("__server_chat__turn", turn_str, turn_len + 1, 0))
		printf("[v7_server] WARNING: v7_db_add('__server_chat__turn') failed -- sidecars gate on a stale turn number\n");
	if (run_turn(user_msg, 0, 0, 0, 0, inference_out, inference_out_cap, thinking_on, reset))
	{
		/* Dead engine: NOTHING was appended to the history (the appends
		 * happen below, after a successful turn) -- a question that never
		 * got an answer never enters the page. */
		handle_engine_death(fd);
		free(user_msg);
		return;
	}

	int out_len = (int)strlen(inference_out);

	/* Full request + full response in ONE block AFTER the inference call. */
	printf("[v7_server] turn %d (web): reset=%d thinking=%d\n", request_counter.load(), reset, thinking_on);
	printf("[v7_server] prompt: \"");
	print_sanitized(user_msg);
	printf("\"\n");
	if (out_len == 0)
		printf("[v7_server] empty answer\n");
	else
	{
		printf("[v7_server] answer:\n");
		print_sanitized(inference_out);
		printf("\n");
	}

	/* History is written ONLY after a successful turn that produced an
	 * answer, as a COMPLETE user+bot pair. An EMPTY answer appends nothing:
	 * the page shows only turns the model actually answered (fabricated
	 * marker text the model never produced is never stored). */
	if (out_len > 0)
	{
		chat_history_append('u', 0, user_msg, user_len);
		chat_history_append('b', (char)thinking_on, inference_out, out_len);
	}

	/* Sidecar background hook: checked after EVERY turn -- a script that
	 * wrote __server_html_background_image mid-turn changes the background
	 * on the page the following redirect renders. */
	char background_name[2048];
	if (background_image_name(background_name, sizeof(background_name)))
		printf("[v7_server] chat background set by sidecar: %s\n", background_name);

	/* Redirect back to the chat page */
	send_redirect(fd);
	free(user_msg);
}

/* ========================================================================
 * POST /chat/reset — clear conversation history + KV cache, redirect to /
 * ======================================================================== */

void handle_chat_reset_page(SOCKET fd, char *body)
{
	(void)body;
	session_reset(); /* same reset as the API path — the page and the KV move together */
	/* A new session starts VIRGIN: clear the sidecar-set background AND the
	 * chat avatar so the page comes back on the plain theme with the plain
	 * v7 label, exactly like a freshly started server (empty values make
	 * background_image_name / avatar_image_name report "not set"). */
	char empty = 0;
	server_dll.v7_db_add("__server_html_background_image", &empty, 1, 0);
	server_dll.v7_db_add("__server__chat_avatar_image", &empty, 1, 0);
	printf("[v7_server] chat reset via web UI\n");
	send_redirect(fd);
}
