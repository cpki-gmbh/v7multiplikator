// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#pragma once

#include <stdlib.h> /* realloc/free (scratch_grow) */

/*
 * v7_server_json_types.h -- the request STRUCTS shared between the JSON
 * parser unit (v7_server_json.h, included by exactly one .cpp) and the
 * server handlers (several .cpp files).
 *
 * TYPEDEFS ONLY: no function bodies, no linkage -- safe to include from
 * every server translation unit. The function declarations the handlers
 * call live in v7_server_common.h inside an extern "C" block (the
 * definitions are C-linkage in v7_server_json.h).
 *
 * OWNERSHIP CONTRACT: every text field is a malloc'd exact-size copy owned
 * by the struct (fixed stores truncated oversized requests at the old
 * 8-16 KB bounds). The matching *_request_free() releases every field, so
 * a handler frees the whole request on EVERY exit path -- including parse
 * errors, where a partial parse already holds allocations.
 */

/* native decision labels A-Z for /v1/decide (docs/typed_decisions.md).
 * 26 is the verbalizer's letter range ('A' + i) -- a protocol bound, not a
 * defensive cap: the engine names a decision by a single letter. */
#define MAX_DECIDE_OPTIONS 26

/* ---- grow-on-demand scratch ----
 * Ensures *cap >= need by realloc-doubling and returns the buffer, or 0 on
 * allocation failure (callers answer 500 -- never a truncated document).
 * static inline: linkage-free, safe in every including translation unit.
 * All server users run on the single main thread, so grown statics are
 * race-free by construction. */
static inline char *scratch_grow(char **buf, int *cap, int need)
{
	if (*buf && *cap >= need)
		return *buf;
	int new_cap = *cap ? *cap : 256;
	while (new_cap < need)
		new_cap *= 2;
	char *grown = (char *)realloc(*buf, (size_t)new_cap);
	if (!grown)
		return 0;
	*buf = grown;
	*cap = new_cap;
	return grown;
}

/* request structure — filled by parse_chat_request() */
typedef struct
{
	char *system_prompt; /* malloc'd concatenation of system/developer messages, or 0 */
	int system_len;
	int system_cap;
	char *user_prompt; /* malloc'd copy of the last user message, or 0 */
	int user_len;
	char *transcript; /* ALL messages as "Role: text" lines —
	                     stateless full-history prompt (0 when the
	                     request carried a single user turn) */
	int transcript_len;
	int transcript_cap;
	float temperature; /* default 0.0 */
	float top_p;       /* default 0.9 */
	int sampler_fields_present; /* 1 when the request carried temperature or top_p: the server never applies them, and says so once per request */
	int max_tokens;    /* 0 = unlimited (server default applies) */
	int stream;        /* 0 = plain JSON body, 1 = SSE chunk envelope */
	int thinking;      /* 1 = reasoning before the answer (default), 0 = the
	                       request disabled it ("enable_thinking": false or
	                       "reasoning_effort": "none") */
	char *model;       /* malloc'd model string from request, or 0 */
	int user_count;      /* how many user messages the request carried —
	                        the stateless detector counts THESE, never raw
	                        body substrings (a prompt quoting JSON with
	                        "role":"assistant" in it must not flip the mode) */
	int assistant_count;
	char *content_buf; /* parse-time message-content scratch (grown to the
	                      exact raw span of each string, since the decoded
	                      text is never longer than its raw JSON span) */
	int content_buf_cap;
} chat_request;

/* completion request structure — for /v1/completions endpoint */
typedef struct
{
	char *prompt; /* malloc'd prompt (string form, or array elements
	                 joined with newlines), or 0 */
	int prompt_len;
	int prompt_cap;
	float temperature; /* default 0.0 */
	float top_p;       /* default 0.9 */
	int sampler_fields_present; /* 1 when the request carried temperature or top_p */
	int max_tokens;    /* 0 = unlimited (server default applies) -- the only legal cap is the buffer */
	int stream;        /* 0 = plain JSON body, 1 = SSE chunk envelope */
	char *model;       /* malloc'd model string from request, or 0 */
} completion_request;

/* vision request — POST /v1/vision/chat
 *   { "prompt": "...", "image_b64": "<base64 PNG/JPEG>", "temperature",
 *     "top_p", "max_tokens", "model" } */
typedef struct
{
	char *prompt; /* malloc'd copy of the prompt, or 0 */
	int prompt_len;
	char *image_b64; /* malloc'd NUL-terminated base64, exact raw span */
	int image_b64_len;
	float temperature;
	float top_p;
	int sampler_fields_present; /* 1 when the request carried temperature or top_p */
	int max_tokens;
	char *model; /* malloc'd model string from request, or 0 */
} vision_request;

/* decide request — POST /v1/decide body (docs/typed_decisions.md):
 *   {"kind":"choice", "state":"...", "question":"...", "options":["a","b"]}
 * kind: "noul" (2 options) | "score" (6 options) | "choice" (2..26 options) */
typedef struct
{
	char kind[16];                   /* printed into the prompt verbatim */
	char *state;                     /* malloc'd copy, or 0 */
	char *question;                  /* malloc'd copy, or 0 */
	char *options[MAX_DECIDE_OPTIONS]; /* malloc'd option label texts */
	int num_options;
} decide_request;
