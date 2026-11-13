// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_health.cpp -- the handoff queue and the health listener thread,
 * the SINGLE acceptor: /v1/health + /v1/models stay live during long
 * inferences, every other connection is handed to the main thread
 * (relocated verbatim from v7_server_main.cpp; the handoff state and
 * thread entry drop `static` -- v7_server_common.h declares them). */

#include "v7_server_common.h"

/* ========================================================================
 * Main — startup + accept loop
 * ======================================================================== */

/* ========================================================================
 * Health listener thread — liveness during long inferences
 *
 * Inference blocks the single server thread for 30-60 s; without this
 * thread, /v1/health and /v1/models die during a turn and watchdogs kill
 * a HEALTHY server. Two acceptors on one listening socket is legal: each
 * connection goes to exactly one thread. This thread serves ONLY
 * state-free endpoints (health, models) — it never touches the engine,
 * so no locking is needed. Silent: no printf (interleaved CRT writes
 * would garble the main thread's turn logs).
 * ======================================================================== */

const char *g_health_model_id = 0; /* set by main before the health thread starts */

/* ---- handoff queue: connections the health thread peeks but does not
 * serve are passed to the main thread untouched (NOTHING consumed — the
 * peek is MSG_PEEK, so the main thread's read_http_request reads the
 * request from scratch). An auto-reset event wakes the main thread the
 * moment a connection enters the queue -- without it the main thread only
 * noticed the queue once per select timeout. (HANDOFF_CAP is in
 * v7_server_common.h.) ---- */
SOCKET g_handoff[HANDOFF_CAP];
int g_handoff_head = 0;
int g_handoff_tail = 0;
CRITICAL_SECTION g_handoff_lock;
#ifdef __linux__
server_event g_handoff_event = 0;
#else
HANDLE g_handoff_event = 0;
#endif

void handoff_push(SOCKET client)
{
	EnterCriticalSection(&g_handoff_lock);
	int next = (g_handoff_tail + 1) % HANDOFF_CAP;
	if (next != g_handoff_head)
	{
		g_handoff[g_handoff_tail] = client;
		g_handoff_tail = next;
#ifdef __linux__
		server_signal_event(g_handoff_event);
#else
		SetEvent(g_handoff_event); /* wake the main thread: a queued inference request must not wait out the poll interval */
#endif
	}
	else
	{
		/* queue full: answer 503 with Retry-After instead of a bare close,
		 * so a client sees WHY and how long to wait */
		LeaveCriticalSection(&g_handoff_lock);
		char err[256];
		int elen = build_error_response(err, sizeof(err), "server busy: the inference handoff queue is full -- retry shortly", "server_error");
		char header[512];
		int hlen = sprintf(header,
		                   "HTTP/1.1 503 Service Unavailable\r\n"
		                   "Content-Type: application/json\r\n"
		                   "Content-Length: %d\r\n"
		                   "Retry-After: 1\r\n"
		                   "Access-Control-Allow-Origin: *\r\n"
		                   "Connection: close\r\n\r\n",
		                   elen);
		send_all(client, header, hlen);
		send_all(client, err, elen);
		graceful_close(client);
		return;
	}
	LeaveCriticalSection(&g_handoff_lock);
}

SOCKET handoff_pop(void)
{
	SOCKET client = INVALID_SOCKET;
	EnterCriticalSection(&g_handoff_lock);
	if (g_handoff_head != g_handoff_tail)
	{
		client = g_handoff[g_handoff_head];
		g_handoff_head = (g_handoff_head + 1) % HANDOFF_CAP;
	}
	LeaveCriticalSection(&g_handoff_lock);
	return client;
}

/* Read the health/models request headers after the peek matched. Returns 1
 * when a complete header block arrived, 0 when the peer stalled past its
 * buffer (the caller closes instead of answering a half-read request and
 * then RST-ing the answer away). */
static int health_read_headers(SOCKET client, char *req, int req_cap)
{
	/* 3 s per recv: a stalled health client must not hold the ONLY
	 * acceptor anywhere near a watchdog window (health requests are tiny;
	 * anything slower than 3 s per recv is not a health check). The read
	 * headers are left in the caller's buffer so the models branch can run
	 * the Host/API-key gates on them. */
	DWORD read_timeout_ms = 3000;
	server_set_timeout(client, SO_RCVTIMEO, read_timeout_ms);
	int total = 0;
	while (total < req_cap - 1)
	{
		int n = recv(client, req + total, req_cap - 1 - total, 0);
		if (n <= 0)
			return 0;
		total += n;
		req[total] = 0;
		if (strstr(req, "\r\n\r\n"))
			return total;
	}
	return 0;
}

DWORD WINAPI health_thread(void *param)
{
	SOCKET listen_fd = (SOCKET)(uintptr_t)param;
	while (server_running.load())
	{
#ifdef __linux__
		SOCKET client = accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
#else
		SOCKET client = accept(listen_fd, 0, 0);
#endif
		if (client == INVALID_SOCKET)
		{
			if (!server_running.load())
				break; /* the listening socket was closed for shutdown: the normal exit */
			/* A listening socket that cannot accept is a dead server: the
			 * only acceptor would loop here forever while the process looks
			 * alive. Report and stop -- the supervisor bat relaunches. */
			printf("[v7_server] accept() failed (%d) on the only listening socket -- stopping\n", WSAGetLastError());
			server_exit_code.store(1);
			server_running.store(0);
			break;
		}
		DWORD io_timeout_ms = 10000;
		server_set_timeout(client, SO_RCVTIMEO, io_timeout_ms);
		server_set_timeout(client, SO_SNDTIMEO, io_timeout_ms);

		/* PEEK at the request line (never consume): the path decides who
		 * serves. Health/models stay live during a 60 s inference; every
		 * OTHER connection is handed to the main thread untouched.
		 * WALL-clock deadline (GetTickCount64): clock() measures CPU time,
		 * which does not advance while recv blocks. n == 0 means the peer
		 * closed — stop peeking.
		 * MSG_PEEK does NOT consume: every recv re-copies the SAME front
		 * bytes, so the destination is ALWAYS peek+0 (writing at peek+seen
		 * would accumulate duplicate prefixes and break the strncmp). */
		char peek[64];
		int seen = 0;
		unsigned long long deadline_ms = GetTickCount64() + 2000;
		/* Short per-recv timeout for the PEEK phase: the request-line bytes
		 * arrive in the first segment or not at all, and the socket's normal
		 * 10 s SO_RCVTIMEO would let one connect-and-wait client stall the
		 * health thread for 10 s while the 2 s deadline sits uncheckable
		 * INSIDE the blocking recv. */
		DWORD peek_timeout_ms = 500;
		server_set_timeout(client, SO_RCVTIMEO, peek_timeout_ms);
		/* The deadline is checked on EVERY iteration (also n > 0): MSG_PEEK
		 * never consumes, so a client that sent a short prefix keeps
		 * returning that same byte forever — an unchecked n > 0 branch
		 * would spin this thread at 100% CPU indefinitely. */
		while (seen < 16)
		{
			int n = recv(client, peek, sizeof(peek) - 1, MSG_PEEK);
			if (n > 0)
			{
				if (n > seen)
					seen = n; /* high-water mark: MSG_PEEK re-copies the same front bytes */
				if (seen >= 16)
					break;
			}
			else if (n == 0)
				break;
			if (GetTickCount64() > deadline_ms)
				break;
			Sleep(10);
		}
		peek[seen] = 0;
		/* serving phase: back to the regular 10 s per-recv cap */
		server_set_timeout(client, SO_RCVTIMEO, io_timeout_ms);

		if (seen == 0)
		{
			/* A connect that sent NOTHING (preconnect / TCP probe / port
			 * scanner) is closed right here: handing it to the main thread
			 * made it block read_http_request for its whole timeout, and
			 * the health thread is the primary acceptor -- everything
			 * waits behind it. */
			closesocket(client);
		}
		else if (seen >= 15 && strncmp(peek, "GET /v1/health", 14) == 0 && (peek[14] == ' ' || peek[14] == '?'))
		{
			/* consume + answer */
			static char hdr[MAX_HEADER_BUF + 1]; /* static: 1 MB cannot live on the 1 MB thread stack; one health thread, sequential use */
			if (health_read_headers(client, hdr, (int)sizeof(hdr)))
			{
				if (server_dll.v7_engine_is_dead())
					send_response(client, 503, "application/json", (char *)"{\"status\":\"engine_dead\"}", (int)strlen("{\"status\":\"engine_dead\"}"));
				else
					send_response(client, 200, "application/json", (char *)"{\"status\":\"ok\"}", 15);
				graceful_close(client); /* the RST of a bare closesocket can drop the answer the watchdog is still reading */
			}
			else
			{
				closesocket(client);
			}
		}
		else if (seen >= 15 && strncmp(peek, "GET /v1/models", 14) == 0 && (peek[14] == ' ' || peek[14] == '?' || peek[14] == '/'))
		{
			/* "/v1/models/<id>" also stays live: the trailing id is the 15th
			 * peek byte, and anything under /v1/models/ is still a models
			 * probe -- handing it to the main thread made an SDK model
			 * check wait out a whole inference. */
			static char hdr[MAX_HEADER_BUF + 1]; /* static: same stack rationale as the health branch above */
			int hdr_len = health_read_headers(client, hdr, (int)sizeof(hdr));
			if (hdr_len)
			{
				/* g_health_model_id is process-lifetime argv — read-only here is safe */
				handle_models(client, (char *)g_health_model_id);
				graceful_close(client);
			}
			else
			{
				closesocket(client);
			}
		}
		else
		{
			/* A HALF-OPEN connection (sent its request line, then stalls
			 * before the headers) handed to the main thread would block
			 * the ONLY inference thread in read_http_request for up to
			 * 30 s. MSG_PEEK-wait here (never consuming) until the full
			 * header block is in the socket buffer or the deadline fires:
			 * the main thread still reads the request from scratch. */
			unsigned long long header_deadline = GetTickCount64() + 10000;
			int headers_ready = 0;
			static char hdr_scan[MAX_HEADER_BUF + 1]; /* static: same stack rationale as the health branches above */
			while (server_running.load())
			{
				int n = recv(client, hdr_scan, sizeof(hdr_scan) - 1, MSG_PEEK);
				if (n <= 0)
				{
					/* n == 0 = the peer CLOSED while we waited: the
					 * connection can never become complete, so no
					 * Time-Sleep spin may outlive it to the deadline --
					 * that used to park the ONLY acceptor for 10 s per
					 * connected-and-vanished client. */
					if (n == 0 || WSAGetLastError() != WSAETIMEDOUT)
						break; /* peer gone, reset or error: close below */
				}
				else
				{
					hdr_scan[n] = 0;
					if (strstr(hdr_scan, "\r\n\r\n"))
					{
						headers_ready = 1;
						break;
					}
				}
				if (GetTickCount64() > header_deadline)
					break;
				Sleep(10);
			}
			if (headers_ready)
				handoff_push(client); /* inference conn — the main thread serves it */
			else
				closesocket(client); /* stalled or vanished: never reach the inference thread */
		}
	}
	return 0;
}
