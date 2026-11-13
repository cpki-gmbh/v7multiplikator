// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_http.cpp -- socket send helpers, HTTP request framing and the
 * response sender (relocated verbatim from v7_server_main.cpp). */

#include "v7_server_common.h"

#ifdef __linux__
#define SERVER_SEND_FLAGS MSG_NOSIGNAL
#else
#define SERVER_SEND_FLAGS 0
#endif

/* ========================================================================
 * HTTP helpers
 * ======================================================================== */

/* Send entire buffer to the socket. Returns the number of bytes actually
 * sent (== len on success), or -1 when the first send failed -- callers
 * treat ret != len as a failed delivery. */
int send_all(SOCKET fd, char *buf, int len)
{
	int sent = 0;
	while (sent < len)
	{
		int bytes_sent = send(fd, buf + sent, len - sent, SERVER_SEND_FLAGS);
		if (bytes_sent <= 0)
			return sent > 0 ? sent : -1;
		sent += bytes_sent;
	}
	return sent;
}

/* Close gracefully: a bare closesocket after unread request bytes sends a
 * TCP RST on Windows, which can DROP the response the client is still
 * reading. shutdown(SD_SEND) + a bounded drain keeps the FIN clean; a
 * client that still streams megabytes past the response gets the RST — an
 * accepted residual. The drain is bounded by a WALL-CLOCK deadline (not by
 * an iteration count): per-recv timeouts only cap ONE call, so a client
 * trickling 1 byte per 200 ms would otherwise hold the single server
 * thread for minutes. A well-behaved client that closes after reading the
 * response makes the first recv return <= 0 immediately -- the drain costs
 * nothing in the common case. */
void graceful_close(SOCKET fd)
{
	shutdown(fd, SD_SEND);
	DWORD drain_timeout_ms = 100;
	server_set_timeout(fd, SO_RCVTIMEO, drain_timeout_ms);
	char drain[512];
	unsigned long long deadline_ms = GetTickCount64() + 1000;
	int guard = 0;
	while (guard++ < 2048 && GetTickCount64() < deadline_ms && recv(fd, drain, sizeof(drain), 0) > 0)
		;
	closesocket(fd);
}

/* Read HTTP request (headers + body) from socket.
   Chunked reads (RECV_CHUNK at a time), a Content-Length clamp, and a hard
   30s whole-request deadline — a client that connects and stalls (or drip-
   feeds bytes) must NEVER hang the single-threaded server in a blocking
   recv(). SO_RCVTIMEO on the socket caps each individual recv() call.
   Header lookups are CASE-INSENSITIVE ("Content-length:" is legal HTTP).
   Returns total bytes in req_buf, or:
     -1  connection error / empty
     -2  Transfer-Encoding: chunked (unsupported — caller answers 400)
     -3  declared body exceeds MAX_BODY_BUF (caller answers 413) */
int header_find(char *hdr, int hdr_len, const char *lower_name)
{
	/* Case-insensitive line-header search: walk line starts, compare the
	 * name up to ':' lowercased. Returns the offset of the value (past the
	 * colon and any spaces), or -1. */
	int i = 0;
	int name_len = (int)strlen(lower_name);
	while (i < hdr_len)
	{
		int line_start = i;
		while (i < hdr_len && hdr[i] != '\n')
			i++;
		int line_end = i; /* at '\n' or hdr_len */
		int j = line_start;
		int k = 0;
		while (j < line_end && k < name_len)
		{
			char c = hdr[j];
			if (c >= 'A' && c <= 'Z')
				c = (char)(c - 'A' + 'a');
			if (c != lower_name[k])
				break;
			k++;
			j++;
		}
		if (k == name_len && j < line_end && hdr[j] == ':')
		{
			j++;
			while (j < line_end && (hdr[j] == ' ' || hdr[j] == '\t'))
				j++;
			return j;
		}
		i++;
	}
	return -1;
}

int read_http_request(SOCKET fd, char *req_buf, int buf_size, char **body_out, int *body_len_out)
{
	int total = 0;
	int header_end = -1;
	int content_length = -1; /* -1 = no Content-Length seen (GET) */
	int expect_continue = 0;
	int answered_continue = 0;
	/* WALL-clock deadline (GetTickCount64): clock() measures CPU time, which
	 * does not advance while recv() blocks — the old deadline never fired. */
	unsigned long long deadline_ms = GetTickCount64() + 30000;

	*body_out = 0;
	*body_len_out = 0;

	while (total < buf_size - 1)
	{
		if (GetTickCount64() > deadline_ms)
			return -4; /* timed out before the FULL request arrived (headers or body): never dispatch a partial request */
		int chunk = buf_size - 1 - total;
		if (chunk > RECV_CHUNK)
			chunk = RECV_CHUNK;
		int bytes_read = recv(fd, req_buf + total, chunk, 0);
		if (bytes_read <= 0)
		{
			req_buf[total] = 0;
			if (bytes_read < 0 && WSAGetLastError() == WSAETIMEDOUT)
				return -4; /* per-recv timeout: the connection is alive but stalled -- the caller answers 408, not a silent close */
			return -1; /* the peer is gone (empty connect, preconnect close, or vanished mid-request): there is nobody left to answer */
		}
		total += bytes_read;
		req_buf[total] = 0;

		/* Header-size check BEFORE the terminator: a client that never */
		/* sends the terminator rides out the 30s deadline with ~9 MB. */
		if (header_end < 0 && total > MAX_HEADER_BUF)
			return -5; /* caller answers 431 */

		if (header_end < 0)
		{
			/* search only the NEW tail (full-buffer rescan is O(n^2)) */
			char *scan_from = req_buf + (total > bytes_read + 3 ? total - bytes_read - 3 : 0);
			char *hdr_end = strstr(scan_from, "\r\n\r\n");
			if (hdr_end)
			{
				header_end = (int)(hdr_end - req_buf) + 4;
				if (header_end > MAX_HEADER_BUF)
					return -5; /* caller answers 431 — headers exceeded their own limit */
				int cl = header_find(req_buf, header_end, "content-length");
				if (cl >= 0)
				{
					/* wide parse first: on 64-bit Linux a long is 64 bits, */
					/* so narrowing before the range check can wrap huge    */
					/* lengths negative. Malformed and negative values are  */
					/* rejected outright -- never accepted as a zero body.  */
					char *cl_end = 0;
					long long cl_value = strtoll(req_buf + cl, &cl_end, 10);
					if (cl_end == req_buf + cl || (*cl_end && *cl_end != ' ' && *cl_end != '\t' && *cl_end != '\r' && *cl_end != '\n'))
						return -2; /* malformed Content-Length: reject, never misparse */
					if (cl_value < 0)
						return -2; /* negative length is not a header-only request */
					if (cl_value >= MAX_BODY_BUF)
						return -3; /* caller answers 413 — no silent clamp */
					content_length = (int)cl_value;
				}
			int te = header_find(req_buf, header_end, "transfer-encoding");
			if (te >= 0)
			{
				/* exact token compare: "identityXYZ" must not pass as the
				 * legal no-encoding value. The value ends at the header
				 * line's CR/LF; trailing spaces are trimmed. */
				int vlen = 0;
				while (req_buf[te + vlen] && req_buf[te + vlen] != '\r' && req_buf[te + vlen] != '\n')
					vlen++;
				while (vlen > 0 && (req_buf[te + vlen - 1] == ' ' || req_buf[te + vlen - 1] == '\t'))
					vlen--;
				if (vlen != 8 || _strnicmp(req_buf + te, "identity", 8) != 0)
					return -2; /* chunked (or any unknown encoding): not supported -- reject, never misparse */
			}
				int ex = header_find(req_buf, header_end, "expect");
				if (ex >= 0 && _strnicmp(req_buf + ex, "100-continue", 12) == 0)
					expect_continue = 1;
				/* A slow link must still be able to upload a big body: the
				 * total deadline grows past the base 30 s at a 64 KB/s floor
				 * for the declared body size. */
				if (content_length > 0)
					deadline_ms += (unsigned long long)content_length / 64;
			}
		}
		/* Answer Expect: 100-continue the moment the headers are in, so curl
		 * does not stall ~1 s waiting before sending a large vision body. */
		if (expect_continue && !answered_continue && header_end >= 0)
		{
			char *cont = (char *)"HTTP/1.1 100 Continue\r\n\r\n";
			send_all(fd, cont, (int)strlen(cont));
			answered_continue = 1;
		}
		/* Done: headers complete AND (no body expected OR full body read) */
		if (header_end >= 0 && (content_length < 0 || total >= header_end + content_length))
			break;
	}
	req_buf[total] = 0;

	/* The loop also exits when the buffer filled BEFORE the body was
	 * complete: headers + an exactly-at-cap body never fit. That is a
	 * size violation, not a short read — answer 413 instead of parsing a
	 * silently truncated body. */
	if (header_end < 0)
		return -4;
	if (content_length >= 0 && total < header_end + content_length)
		return -3;

	/* HTTP/1.1 requires Content-Length (or chunked, rejected above) for a
	 * request body: a POST without one is a framing error, not a parse
	 * error -- the caller answers 411 instead of a misleading field 400. */
	if (content_length < 0 && header_end >= 5 && memcmp(req_buf, "POST ", 5) == 0)
		return -7;

	/* FRAMING: expose exactly the declared body length, never the raw byte
	 * count — a pipelined next request coalesced into the same recv belongs
	 * to the NEXT read, not to this body. NUL-terminating at the body end
	 * keeps every C-string consumer (JSON parsers, form parsing) from
	 * reading into those pipelined bytes. Without Content-Length (GET),
	 * there IS no body. */
	*body_out = req_buf + header_end;
	*body_len_out = (content_length >= 0) ? content_length : 0;
	if (content_length >= 0)
		req_buf[header_end + content_length] = 0;
	return total;
}

/* ========================================================================
 * Send HTTP response
 * ======================================================================== */

void send_response(SOCKET fd, int status, const char *content_type, char *body, int body_len)
{
	char *status_text = (char *)"OK";
	if (status == 201)
		status_text = (char *)"Created";
	else if (status == 204)
		status_text = (char *)"No Content";
	else if (status == 400)
		status_text = (char *)"Bad Request";
	else if (status == 401)
		status_text = (char *)"Unauthorized";
	else if (status == 403)
		status_text = (char *)"Forbidden";
	else if (status == 408)
		status_text = (char *)"Request Timeout";
	else if (status == 411)
		status_text = (char *)"Length Required";
	else if (status == 413)
		status_text = (char *)"Content Too Large";
	else if (status == 431)
		status_text = (char *)"Request Header Fields Too Large";
	else if (status == 404)
		status_text = (char *)"Not Found";
	else if (status == 405)
		status_text = (char *)"Method Not Allowed";
	else if (status == 501)
		status_text = (char *)"Not Implemented";
	else if (status == 503)
		status_text = (char *)"Service Unavailable";
	else if (status == 500)
		status_text = (char *)"Internal Server Error";

	char header[768];
	int hlen = sprintf(header,
	                   "HTTP/1.1 %d %s\r\n"
	                   "Content-Type: %s\r\n"
	                   "Content-Length: %d\r\n"
	                   "X-Content-Type-Options: nosniff\r\n"
	                   "X-Frame-Options: DENY\r\n"
	                   "Access-Control-Allow-Origin: *\r\n"
	                   "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
	                   "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
	                   "Connection: close\r\n\r\n",
	                   status, status_text, content_type, body_len < 0 ? -body_len : body_len);

	/* Send failures are LOGGED, never silent: the answer was fully produced
	 * (and is in the console log) -- an operator must SEE that the client
	 * walked away before it was delivered, or the loss is undiagnosable.
	 * A NEGATIVE body_len is the HEAD convention: the header carries the
	 * real Content-Length but no body follows. */
	int header_sent = send_all(fd, header, hlen);
	if (header_sent != hlen)
		printf("[v7_server] response send failed (header): client gone -- %d %s, %d body bytes never delivered\n",
		       status, content_type, body_len < 0 ? 0 : body_len);
	else if (body_len > 0)
	{
		int body_sent = send_all(fd, body, body_len);
		if (body_sent != body_len)
			printf("[v7_server] response send failed (body): client gone after the header -- %d of %d body bytes lost\n",
			       body_sent < 0 ? 0 : body_sent, body_len);
	}
}
