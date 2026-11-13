// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_api_meta.cpp -- state-free API endpoints: /v1/health,
 * /v1/models, CORS preflight, /v1/chat/reset and /quit (relocated verbatim
 * from v7_server_main.cpp). */

#include "v7_server_common.h"

/* ========================================================================
 * Route handlers
 * ======================================================================== */

/* GET /v1/health — health check. The status matches the HEAD answer and
 * the health-thread branch: a DEAD engine must never report ok (a
 * watchdog probing a restarting server would otherwise kill it for
 * "responding" with 200). */
void handle_health(SOCKET fd)
{
	if (server_dll.v7_engine_is_dead())
		send_response(fd, 503, "application/json", (char *)"{\"status\":\"engine_dead\"}", (int)strlen("{\"status\":\"engine_dead\"}"));
	else
		send_response(fd, 200, "application/json", (char *)"{\"status\":\"ok\"}", 15);
}

/* build the /v1/models body (shared by GET and HEAD so the HEAD response's
 * Content-Length always matches what GET would send). Returns the malloc'd
 * body via out and its length (0 with *out == 0 on allocation failure).
 * Malloc'd, not static: the handler runs on the health thread AND the main
 * thread, and the model id (request-supplied) has no length bound to lean
 * on -- the escape buffer is sized to the exact worst case. */
int build_models_body(char **out, char *model_id)
{
	*out = 0;
	int mlen = (int)strlen(model_id);
	char *esc_model = (char *)malloc((size_t)mlen * 6 + 16);
	char *body = (char *)malloc((size_t)mlen * 6 + 256);
	if (!esc_model || !body)
	{
		free(esc_model);
		free(body);
		return 0;
	}
	json_escape(model_id, mlen, esc_model, mlen * 6 + 16);
	int blen = snprintf(body, (size_t)(mlen * 6 + 256),
	                    "{\"object\":\"list\","
	                    "\"data\":[{"
	                    "\"id\":\"%s\","
	                    "\"object\":\"model\","
	                    "\"created\":0,"
	                    "\"owned_by\":\"v7\""
	                    "}]}\n",
	                    esc_model);
	free(esc_model);
	if (blen < 0)
		blen = 0;
	*out = body;
	return blen;
}

/* GET /v1/models — return a canned model list. */
void handle_models(SOCKET fd, char *model_id)
{
	char *body = 0;
	int blen = build_models_body(&body, model_id);
	if (!body)
	{
		char err[256];
		int elen = build_error_response(err, sizeof(err), "out of memory", "internal_error");
		send_response(fd, 500, "application/json", err, elen);
		return;
	}
	send_response(fd, 200, "application/json", body, blen);
	free(body);
}

/* OPTIONS * — CORS preflight, restricted to the /v1/ API paths (state-free
 * reads for browser SDKs). Everywhere else the 403 answer carries NO CORS
 * headers: a browser preflight fails, so the browser never sends the actual
 * request — this keeps a cross-site page from preflight-then-post the
 * state-changing endpoints.
 * The requested header list is reflected as sent. */
void handle_options(SOCKET fd, char *req_buf, int total, char *path)
{
	if (strncmp(path, "/v1/", 4) != 0)
	{
		char err[256];
		int elen = build_error_response(err, sizeof(err), "preflight is only supported for /v1/ API endpoints", "forbidden");
		char header[512];
		int hlen = sprintf(header,
		                   "HTTP/1.1 403 Forbidden\r\n"
		                   "Content-Type: application/json\r\n"
		                   "Content-Length: %d\r\n"
		                   "X-Content-Type-Options: nosniff\r\n"
		                   "X-Frame-Options: DENY\r\n"
		                   "Connection: close\r\n\r\n",
		                   elen);
		send_all(fd, header, hlen);
		send_all(fd, err, elen);
		return;
	}

	char req_headers[512] = "";
	int h = header_find(req_buf, total, "access-control-request-headers");
	if (h >= 0)
	{
		/* copy the requested header names, comma-separated. The cursor
		 * ALWAYS advances: separators are skipped at the top of each
		 * iteration, and after a copied token p already sits past it. */
		char *p = req_buf + h;
		int w = 0;
		while (*p && *p != '\r' && *p != '\n' && w < (int)sizeof(req_headers) - 12)
		{
			while (*p == ' ' || *p == '\t' || *p == ',')
				p++; /* separators + padding before the token */
			if (*p == 0 || *p == '\r' || *p == '\n')
				break;
			char *tok = p;
			while (*p && *p != ',' && *p != '\r' && *p != '\n')
				p++; /* p now ends the token */
			int tok_len = (int)(p - tok);
			if (w + 1 + tok_len >= (int)sizeof(req_headers))
				break; /* the token does not fit: stop BEFORE writing a separator, so no dangling comma is left */
			if (w > 0)
				req_headers[w++] = ',';
			memcpy(req_headers + w, tok, tok_len);
			w += tok_len;
			req_headers[w] = 0;
		}
	}
	char header[1024];
	int hlen = sprintf(header,
	                   "HTTP/1.1 204 No Content\r\n"
	                   "Access-Control-Allow-Origin: *\r\n"
	                   "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
	                   "Access-Control-Allow-Headers: %s\r\n"
	                   "Access-Control-Max-Age: 86400\r\n"
	                   "Content-Length: 0\r\n"
	                   "X-Content-Type-Options: nosniff\r\n"
	                   "Connection: close\r\n\r\n",
	                   req_headers[0] ? req_headers : "Content-Type, Authorization");
	send_all(fd, header, hlen);
}


/* ========================================================================
 * POST /v1/chat/reset — start a fresh session on the next turn
 * ======================================================================== */

void handle_chat_reset(SOCKET fd)
{
	session_reset(); /* fresh KV AND cleared web history — one session, one truth */
	/* A new session starts VIRGIN: clear the sidecar-set background AND the
	 * chat avatar so the page comes back on the plain theme with the plain
	 * v7 label, exactly like a freshly started server. */
	char empty = 0;
	server_dll.v7_db_add("__server_html_background_image", &empty, 1, 0);
	server_dll.v7_db_add("__server__chat_avatar_image", &empty, 1, 0);
	printf("[v7_server] session reset requested\n");
	send_response(fd, 200, "application/json", (char *)"{\"status\":\"reset\"}", 18);
}

/* ========================================================================
 * GET/POST /quit?code=N — terminate the server, exiting with code N.
 * Default code is 42 ("clean quit" — the start bat stops looping on it).
 * Any other code lets the bat branch (restart, reindex, ...). A code field
 * that is not a number or outside 0..255 answers 400: a silently corrected
 * code can make the supervisor RESTART when the caller meant STOP.
 * ======================================================================== */

void handle_quit(SOCKET fd, char *query)
{
	int code = 42;
	char *cp = find_field(query, "code=");
	if (cp)
	{
		/* Strict digits: "/quit?code=abc" answers 400 rather than atoi's
		 * silent 0 (which would make the restart loop RESTART instead of
		 * stopping). */
		char *p = cp;
		while (*p >= '0' && *p <= '9')
			p++;
		if (!(p > cp && (*p == 0 || *p == '&' || *p == ' ')))
		{
			send_error(fd, 400, "the /quit code must be a number in 0..255", "invalid_request_error");
			return;
		}
		int v = 0;
		for (char *d = cp; d < p; d++)
		{
			v = v * 10 + (*d - '0');
			if (v > 255)
			{
				send_error(fd, 400, "the /quit code must be a number in 0..255", "invalid_request_error");
				return;
			}
		}
		code = v;
	}

	char body[128];
	int blen = sprintf(body, "{\"status\":\"quitting\",\"code\":%d}", code);
	send_response(fd, 200, "application/json", body, blen);

	server_exit_code.store(code);
	server_running.store(0); /* accept loop exits within its 1s select timeout */
	printf("[v7_server] quit (code %d)\n", code);
}

