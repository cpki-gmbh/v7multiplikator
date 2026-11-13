/* v7 - part of the v7 inference engine
 * Copyright (C) 2026 Görg Pflug & CPKI GmbH
 * License: GPL v3 (see LICENSE)
 */

#pragma once

/*
 * v7_server_common.h -- shared constants, state and declarations for the v7
 * OpenAI-compatible server. Every server translation unit includes THIS and
 * nothing else from the server, with one exception:
 * v7_server_json.cpp additionally includes v7_server_json.h, which is the
 * sole owner of the JSON function DEFINITIONS.
 */

#ifdef __linux__
#include "server_linux.h"
#else
#ifndef _WINSOCK2API_
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <windows.h>
#include <objbase.h>
#include <gdiplus.h>
#endif
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "ole32.lib")
#endif

/* v7_lib.h is header-only and its four implementation functions
 * (v7_load, v7_unload, v7_lib_reset_stats, v7_lib_print_stats) are all
 * `static`, so including it from every translation unit is link-safe --
 * each TU gets its own private copy, and the compiler discards the copies
 * nobody calls. Only v7_server_main.cpp actually calls v7_load/v7_unload. */
#include "src/v7_lib.h"

#include "v7_server_json_types.h"

/* ---- constants ---- */
/* header framing bound: answers 431 before an unbounded buffer. 1 MB is
 * pathological-request territory; real headers sit in the low KBs. */
#define MAX_HEADER_BUF (1 << 20)
/* request body framing bound: answers 413 before an unbounded buffer. The
 * model context window rejects oversized prompts first; this only frames
 * the transport. */
#define MAX_BODY_BUF (256 << 20)
/* inference output bound: the STARTING capacity of the per-handler grown
 * output buffer (scratch_grow doubles past it, so a turn longer than 8 MB
 * is served, never cut). EOS / max_gen / the caller's max_tokens are the
 * real stops. The handler buffers are function-static (grown buffers
 * cannot live on the 1 MB thread stack; all inference runs on the single
 * main thread). */
#define MAX_INFERENCE_OUT (8 << 20)
#define DEFAULT_PORT 8080
#define RECV_CHUNK 4096
/* exit code for "the port is owned by somebody else": the supervisor stops on
 * it instead of relaunching into the same conflict */
#define SERVER_EXIT_PORT_BUSY 43
/* bytes the collapsible thinking block spends on markup; format_bot_text
 * reserves this from dst_cap before writing the header */
#define THINK_BLOCK_CHROME_BYTES 1024
/* handoff queue capacity: connections the health thread passes to main */
#define HANDOFF_CAP 64

/* ---- shared types ---- */
typedef struct
{
	char *text; /* malloc'd full turn text (never truncated) */
	int len;
	char role;  /* 'u' = user, 'b' = bot */
	char think; /* 1 = this bot turn ran with thinking ON (its reasoning is behind the collapsible block) */
} chat_msg;

typedef struct
{
	unsigned char *rgb; /* malloc'd width*height*3, caller frees */
	int width;
	int height;
} decoded_image;

/* ---- shared state (definitions: v7_server_main.cpp / v7_server_health.cpp) ---- */
extern struct v7_lib server_dll;
extern std::atomic<int> server_running;
extern std::atomic<int> request_counter;
extern std::atomic<int> reset_next_turn;
extern std::atomic<int> server_exit_code;
extern int hook_enabled;       /* sidecar hook enabled via /sidecar */
extern int hook_executed;      /* an inference turn completed with hook on */
extern int g_server_chat_turn; /* sidecar turn counter (0 on the reset turn) */
extern int g_max_gen;          /* server-wide default generation budget from -maxgen; 0 = UNLIMITED */
extern int g_engine_thinking;  /* what WE last set via set_engine_thinking */
extern float g_sampler_temp;   /* what WE last set via v7_set_sampler_temperature;
                                  0.0 = the engine's greedy default. Only the
                                  think-tag rescue writes it; run_turn saves and
                                  restores it around the retry ladder */
extern chat_msg *g_chat_history; /* server-side conversation history, malloc'd
                                    -history entries (evict-oldest-pair when full) */
extern int g_chat_history_cap;   /* history entry capacity from -history */
extern int g_chat_count;
extern int g_think_counter; /* unique <div id=tk%d> ids WITHIN one rendered page */
extern int thinking_enabled; /* last submitted state of the thinking checkbox */

/* health thread's read-only argv copy -- defined in v7_server_health.cpp */
extern const char *g_health_model_id;

/* handoff queue (definitions in v7_server_health.cpp; main() initialises and
 * drains the queue, the health thread pushes to it) */
extern SOCKET g_handoff[HANDOFF_CAP];
extern int g_handoff_head;
extern int g_handoff_tail;
extern CRITICAL_SECTION g_handoff_lock;
#ifdef __linux__
extern server_event g_handoff_event;
#else
extern HANDLE g_handoff_event;
#endif

/* ========================================================================
 * JSON entry points
 *
 * THE DEFINITIONS OF THESE LIVE INSIDE extern "C" IN v7_server_json.h.
 * Every declaration here must therefore also carry C linkage: a plain C++
 * declaration would ask the linker for the mangled name and fail with an
 * unresolved external. This is the single easiest way to break the split.
 * ======================================================================== */

extern "C"
{
	int build_chat_response_ex(char *output, int output_size, char *content, int content_len,
	                           char *reasoning, int reasoning_len, const char *model, int req_id,
	                           int stream, const char *finish_reason, int prompt_tokens, int completion_tokens);
	int build_completion_response(char *output, int output_size, char *content, int content_len,
	                              const char *model, int req_id, const char *finish_reason, int stream,
	                              int prompt_tokens, int completion_tokens);
	int build_decide_response(char *dst, int dst_cap, decide_request *req, float *probs, int best_index);
	int build_error_response(char *output, int output_size, const char *message, const char *error_type);
	int parse_chat_request(char *body, int body_len, chat_request *req);
	int parse_completion_request(char *body, int body_len, completion_request *req);
	int parse_vision_request(char *body, int body_len, vision_request *req);
	int parse_decide_request(char *body, int body_len, decide_request *req);
	/* exact-size string parse: mallocs the raw span + 1 (the decoded text is
	 * never longer than its raw JSON span), replaces *dst, frees the old
	 * buffer. Returns the new parse position, or 0 on allocation failure. */
	char *json_parse_string_alloc(char *p, char **dst, int *out_len);
	/* ownership: release every allocation a request holds; safe to call
	 * twice (all pointers are zeroed after the free) */
	void chat_request_free(chat_request *req);
	void completion_request_free(completion_request *req);
	void vision_request_free(vision_request *req);
	void decide_request_free(decide_request *req);
	int json_escape(char *src, int src_len, char *dst, int dst_size);
	int b64_decode(char *src, int src_len, unsigned char *dst, int dst_cap);
}

/* ========================================================================
 * Server functions -- C++ linkage. These were plain C++ symbols before the
 * split, so they are NOT wrapped in extern "C".
 * ======================================================================== */

/* util */
void sanitize_for_log(char *s);
int utf8_trim_to_boundary(char *buf, int len);
void print_sanitized(const char *s);
int utf8_wellformed(const char *s, int len);
void get_http_method(char *req, char *method, int method_size);
int get_http_path(char *req, char *path, int path_size);
int get_http_query(char *req, char *query, int query_size);
char *find_field(char *query_string, const char *key);
void url_decode_query(char *s);
void send_error(SOCKET fd, int status, const char *message, const char *error_type);
void send_redirect(SOCKET fd);
int trim_tail_utf8(char *out);

/* http */
static inline void server_set_timeout(SOCKET socket, int option, unsigned milliseconds)
{
#ifdef __linux__
	struct timeval timeout = { (time_t)(milliseconds / 1000), (suseconds_t)(milliseconds % 1000) * 1000 };
	setsockopt(socket, SOL_SOCKET, option, &timeout, sizeof(timeout));
#else
	DWORD timeout = milliseconds;
	setsockopt(socket, SOL_SOCKET, option, (char *)&timeout, sizeof(timeout));
#endif
}

int send_all(SOCKET fd, char *buf, int len);
void graceful_close(SOCKET fd);
int header_find(char *hdr, int hdr_len, const char *lower_name);
int read_http_request(SOCKET fd, char *req_buf, int buf_size, char **body_out, int *body_len_out);
void send_response(SOCKET fd, int status, const char *content_type, char *body, int body_len);

/* turn */
void set_engine_thinking(int on);
void session_reset(void);
int handle_engine_death(SOCKET fd);
int run_turn(char *prompt, char *sys_ptr, unsigned char *img, int img_w, int img_h,
             char *out, int out_cap, int thinking, int reset);
int split_think(char *out, int out_len, int thinking, int *reasoning_len);
int session_take_reset(void);

/* image */
int image_decode(unsigned char *data, int data_len, decoded_image *out);

/* api meta */
void handle_health(SOCKET fd);
int build_models_body(char **out, char *model_id);
void handle_models(SOCKET fd, char *model_id);
void handle_options(SOCKET fd, char *req_buf, int total, char *path);
void handle_chat_reset(SOCKET fd);
void handle_quit(SOCKET fd, char *query);

/* api chat + vision */
void handle_chat_completions(SOCKET fd, char *body, int body_len, char *model_id);
void handle_decide(SOCKET fd, char *body, int body_len);
void handle_completions(SOCKET fd, char *body, int body_len, char *model_id);
void handle_vision_chat(SOCKET fd, char *body, int body_len, char *model_id);

/* sidecar */
void handle_sidecar(SOCKET fd, char *query);

/* web render */
int background_image_name(char *out, int out_cap);
int avatar_image_name(char *out, int out_cap);
void chat_history_append(char role, char think, char *text, int len);
void chat_history_clear(void);
int html_escape_buf(char *src, int src_len, char *dst, int dst_cap);
int find_tag(char *src, int src_len, const char *tag, int tag_len);
int format_bot_text(char *src, int src_len, char *dst, int dst_cap, int think_on);
/* renders the FULL history into a grown page buffer (never dropped turns);
 * returns the buffer and its length via out_len, or 0 on allocation failure */
char *build_chat_page(int *out_len);

/* web handlers */
void handle_background(SOCKET fd);
void handle_avatar(SOCKET fd);
void handle_index(SOCKET fd);
void handle_chat_submit(SOCKET fd, char *body, int body_len);
void handle_chat_reset_page(SOCKET fd, char *body);

/* router */
void serve_client(SOCKET client, char *model_id);

/* health (the acceptor thread; its queue helpers stay internal) */
DWORD WINAPI health_thread(void *param);
SOCKET handoff_pop(void);
