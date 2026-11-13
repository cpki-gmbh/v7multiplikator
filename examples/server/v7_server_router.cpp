// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_router.cpp -- serve_client: request read, route, answer, close
 * for ONE connection. */

#include "v7_server_common.h"

/* Serve ONE connection: read the request, route, answer, close. Used by the
 * main accept loop for connections the health thread handed over.
 * (External linkage: called from main() in v7_server_main.cpp --
 * v7_server_common.h declares it.) */
void serve_client(SOCKET client, char *model_id)
{
	static char req_buf[MAX_HEADER_BUF + MAX_BODY_BUF]; /* static: 9.4 MB — on the 1 MB thread stack this overflows instantly */

	/* Receive/send timeouts: a client that connects and then stalls (or
	 * never finishes its request) must not block this single-threaded
	 * server forever inside a kernel recv()/send() wait. */
	DWORD io_timeout_ms = 10000; /* 10 s per recv/send call */
	server_set_timeout(client, SO_RCVTIMEO, io_timeout_ms);
	server_set_timeout(client, SO_SNDTIMEO, io_timeout_ms);

	/* Read HTTP request */
	char *body = 0;
	int body_len = 0;
	int total = read_http_request(client, req_buf, sizeof(req_buf) - 1, &body, &body_len);
	if (total == -2)
	{
		char err[256];
		int elen = build_error_response(err, sizeof(err), "Transfer-Encoding other than identity is not supported — send Content-Length", "not_implemented");
		send_response(client, 501, "application/json", err, elen);
		graceful_close(client);
		return;
	}
	if (total == -3)
	{
		char err[256];
		int elen = build_error_response(err, sizeof(err), "request body too large or truncated", "invalid_request_error");
		send_response(client, 413, "application/json", err, elen);
		graceful_close(client);
		return;
	}
	if (total == -4)
	{
		/* timeout before a complete request: answer 408 so the client sees
		 * WHY instead of a bare TCP close */
		char err[256];
		int elen = build_error_response(err, sizeof(err), "request timed out before it was complete", "invalid_request_error");
		send_response(client, 408, "application/json", err, elen);
		graceful_close(client);
		return;
	}
	if (total == -5)
	{
		char err[256];
		int elen = build_error_response(err, sizeof(err), "request header fields too large", "invalid_request_error");
		send_response(client, 431, "application/json", err, elen);
		graceful_close(client);
		return;
	}
	if (total == -7)
	{
		char err[256];
		int elen = build_error_response(err, sizeof(err), "Content-Length is required for a POST body", "invalid_request_error");
		send_response(client, 411, "application/json", err, elen);
		graceful_close(client);
		return;
	}
	if (total <= 0)
	{
		/* the peer is gone before anything readable arrived: there is no
		 * client left to answer, just release the socket */
		closesocket(client);
		return;
	}

	/* A request with truncated headers never set body — hand the
	 * handlers an EMPTY body, never a NULL (a NULL derefs inside the
	 * JSON parser). */
	if (!body)
	{
		body = req_buf + total;
		body[0] = 0;
		body_len = 0;
	}

	/* Extract method, path, and query string. The query IS the sidecar
	 * script path on /sidecar: a cut here would truncate into a wrong
	 * fopen with a misleading "cannot open" — overflow answers 414. */
	char method[16];
	char path[256];
	char query[8192];
	get_http_method(req_buf, method, sizeof(method));
	if (get_http_path(req_buf, path, sizeof(path)) < 0 || get_http_query(req_buf, query, sizeof(query)) < 0)
	{
		char err[256];
		int elen = build_error_response(err, sizeof(err), "request URI too long", "invalid_request_error");
		send_response(client, 414, "application/json", err, elen);
		graceful_close(client);
		return;
	}

	/* Normalize a trailing slash ("/v1/models/" == "/v1/models"): several
	 * clients append one and the old router answered 404. */
	int plen2 = (int)strlen(path);
	if (plen2 > 1 && path[plen2 - 1] == '/')
		path[plen2 - 1] = 0;

	sanitize_for_log(method);
	sanitize_for_log(path);
	/* the QUERY is logged from a COPY: sanitize_for_log mutates in place,
	 * and /sidecar reads the raw query afterwards (a 0x7F in a script path
	 * must reach the file open as-is, only the log line is flattened) */
	char log_query[8192];
	strncpy(log_query, query, sizeof(log_query) - 1);
	log_query[sizeof(log_query) - 1] = 0;
	sanitize_for_log(log_query);
	printf("[v7_server] %s %s%s\n", method, path, query[0] ? "?" : "");
	if (query[0])
		printf("%s\n", log_query); /* full query on its own line */

	/* The search covers ONLY the header block: `total` includes the body,
	 * and a body line would otherwise read as a header. */
	int header_block_len = body ? (int)(body - req_buf) : total;

	int is_head = (strcmp(method, "HEAD") == 0);

	/* Route */
	if (strcmp(method, "OPTIONS") == 0)
	{
		handle_options(client, req_buf, header_block_len, path); /* header block only: a body line must never forge a CORS header echo */
	}
	else if ((strcmp(method, "GET") == 0 || is_head) && strcmp(path, "/") == 0)
	{
		if (is_head)
		{
			int page_len = 0;
			char *head_page = build_chat_page(&page_len); /* grown inside web_render -- HEAD reports the REAL page length */
			if (!head_page)
			{
				send_error(client, 500, "out of memory building the page", "internal_error");
			}
			else
				send_response(client, 200, "text/html; charset=utf-8", head_page, -page_len); /* negative length: headers only (HEAD) */
		}
		else
		{
			handle_index(client);
		}
	}
	else if (strcmp(path, "/") == 0)
	{
		send_error(client, 405, "Only GET is allowed for /", "invalid_request_error");
	}
	else if (strcmp(method, "GET") == 0 && strcmp(path, "/background") == 0)
	{
		handle_background(client);
	}
	else if (strcmp(path, "/background") == 0)
	{
		send_error(client, 405, "Only GET is allowed for /background", "invalid_request_error");
	}
	else if (strcmp(method, "GET") == 0 && strcmp(path, "/avatar") == 0)
	{
		handle_avatar(client);
	}
	else if (strcmp(path, "/avatar") == 0)
	{
		send_error(client, 405, "Only GET is allowed for /avatar", "invalid_request_error");
	}
	else if (strcmp(method, "POST") == 0 && strcmp(path, "/chat/submit") == 0)
	{
		handle_chat_submit(client, body, body_len);
	}
	else if (strcmp(path, "/chat/submit") == 0)
	{
		send_error(client, 405, "Only POST is allowed for /chat/submit", "invalid_request_error");
	}
	else if (strcmp(method, "POST") == 0 && strcmp(path, "/chat/reset") == 0)
	{
		handle_chat_reset_page(client, body);
	}
	else if (strcmp(path, "/chat/reset") == 0)
	{
		/* POST-only: a cross-site GET (an <img src> or link a visited
		 * page fires) must never reset the user's session. */
		send_error(client, 405, "Only POST is allowed for /chat/reset", "invalid_request_error");
	}
	else if ((strcmp(method, "GET") == 0 || is_head) && strcmp(path, "/v1/health") == 0)
	{
		if (is_head)
		{
			/* the HEAD answer carries the SAME status the GET health check
			 * would: an always-200 HEAD lied to watchdogs probing the dead
			 * engine while GET clients saw the 503. */
			if (server_dll.v7_engine_is_dead())
				send_response(client, 503, "application/json", (char *)"", 0 - (int)strlen("{\"status\":\"engine_dead\"}"));
			else
				send_response(client, 200, "application/json", (char *)"", 0 - 15); /* HEAD: same headers, no body */
		}
		else
			handle_health(client);
	}
	else if (strcmp(path, "/v1/health") == 0)
	{
		send_error(client, 405, "Only GET is allowed for /v1/health", "invalid_request_error");
	}
	else if ((strcmp(method, "GET") == 0 || is_head) &&
	         (strcmp(path, "/v1/models") == 0 || strncmp(path, "/v1/models/", 11) == 0))
	{
		/* "/v1/models/<id>" is what several SDK clients probe; this server
		 * carries ONE model, so any id under /v1/models/ serves it. */
		if (is_head)
		{
			char *head_body = 0;
			int hblen = build_models_body(&head_body, model_id);
			if (!head_body)
			{
				send_error(client, 500, "out of memory", "internal_error");
			}
			else
			{
				send_response(client, 200, "application/json", head_body, -hblen); /* HEAD: the REAL length, headers only */
				free(head_body);
			}
		}
		else
			handle_models(client, model_id);
	}
	else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/completions") == 0)
	{
		handle_chat_completions(client, body, body_len, model_id);
	}
	else if (strcmp(path, "/v1/chat/completions") == 0)
	{
		send_error(client, 405, "Only POST is allowed for /v1/chat/completions", "invalid_request_error");
	}
	else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/completions") == 0)
	{
		handle_completions(client, body, body_len, model_id);
	}
	else if (strcmp(path, "/v1/completions") == 0)
	{
		send_error(client, 405, "Only POST is allowed for /v1/completions", "invalid_request_error");
	}
	else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/decide") == 0)
	{
		handle_decide(client, body, body_len);
	}
	else if (strcmp(path, "/v1/decide") == 0)
	{
		send_error(client, 405, "Only POST is allowed for /v1/decide", "invalid_request_error");
	}
	else if (strcmp(path, "/quit") == 0)
	{
		handle_quit(client, query);
	}
	else if (strcmp(path, "/sidecar") == 0)
	{
		handle_sidecar(client, query);
	}
	else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/vision/chat") == 0)
	{
		handle_vision_chat(client, body, body_len, model_id);
	}
	else if (strcmp(path, "/v1/vision/chat") == 0)
	{
		send_error(client, 405, "Only POST is allowed for /v1/vision/chat", "invalid_request_error");
	}
	else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/reset") == 0)
	{
		handle_chat_reset(client);
	}
	else
	{
		send_error(client, 404, "Not found", "not_found_error");
	}

	graceful_close(client);
}
