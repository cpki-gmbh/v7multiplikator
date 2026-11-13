// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

#pragma once

/*
 * v7_server_json.h — minimal JSON parser/builder for OpenAI-compatible
 * chat completions.
 *
 * inbound: parse messages[], temperature, top_p, max_tokens from POST body.
 * outbound: build OpenAI-format chat.completion JSON response.
 *
 * no allocations — all output goes into caller-provided buffers.
 * no third-party JSON library — hand-rolled.
 *
 * SINGLE-TU HEADER: the function definitions in here are intentionally
 * non-static — this header is included exactly once (v7_server_json.cpp).
 * Including it from a second translation unit would duplicate every
 * definition and fail the link. The request STRUCTS live in
 * v7_server_json_types.h (linkage-free, include-anywhere); every other
 * translation unit sees the function declarations in v7_server_common.h.
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "v7_server_json_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

	/* ========================================================================
	 * request structure — filled by parse_chat_request()
	 * (chat_request lives in v7_server_json_types.h)
	 * ======================================================================== */



	void completion_request_init(completion_request *r)
	{
		memset(r, 0, sizeof(*r));
		r->temperature = 0.0f;
		r->top_p = 0.9f;
		r->max_tokens = 0; /* 0 = unlimited: only EOS and the output buffer bound the turn */
		r->stream = 0;
	}

	void chat_request_init(chat_request *r)
	{
		memset(r, 0, sizeof(*r));
		r->temperature = 0.0f;
		r->top_p = 0.9f;
		r->max_tokens = 0; /* 0 = unlimited: only EOS and the output buffer bound the turn */
		r->stream = 0;
		r->thinking = 1; /* engine default: reasoning on unless the request disables it */
		r->user_count = 0;
		r->assistant_count = 0;
	}	/* ========================================================================
	 * minimal JSON helpers
	 * ======================================================================== */

	/* set by json_parse_string on a malformed \uXX escape: the enclosing
	 * parse_*_request checks it and fails the whole request (a malformed
	 * escape used to be silently dropped from the text). Single main thread
	 * does all body parsing, so a plain file-scope int is race-free. */
	static int json_string_had_error = 0;

	char *json_skip_ws(char *p)
	{
		while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')
			p++;
		return p;
	}

	char *json_expect(char *p, char c)
	{
		p = json_skip_ws(p);
		if (*p == c)
			return p + 1;
		return 0;
	}

	char *json_parse_string(char *p, char *dst, int dst_size, int *out_len)
	{
		int len = 0;
		if (out_len)
			*out_len = 0;
		while (*p && *p != '"')
		{
			if (*p == '\\')
			{
				p++;
				if (!*p)
					break; /* trailing backslash at end of input: stop — p++ past the NUL would walk off the buffer */
				switch (*p)
				{
				case '"':
					if (len < dst_size - 1)
						dst[len++] = '"';
					break;
				case '\\':
					if (len < dst_size - 1)
						dst[len++] = '\\';
					break;
				case '/':
					if (len < dst_size - 1)
						dst[len++] = '/';
					break;
				case 'b':
					if (len < dst_size - 1)
						dst[len++] = '\b';
					break;
				case 'f':
					if (len < dst_size - 1)
						dst[len++] = '\f';
					break;
				case 'n':
					if (len < dst_size - 1)
						dst[len++] = '\n';
					break;
				case 'r':
					if (len < dst_size - 1)
						dst[len++] = '\r';
					break;
				case 't':
					if (len < dst_size - 1)
						dst[len++] = '\t';
					break;
				case 'u': {
					/* \uXXXX -> UTF-8 (browser chat sends non-ASCII this way).
					 * surrogate pairs D800-DBFF + DC00-DFFF combine to U+10000+.
					 * A LONE surrogate is invalid UTF-8 (RFC 3629) — replaced
					 * with U+FFFD so the output never contains surrogates. */
					unsigned cp = 0;
					int hex_read = 0;
					for (int i = 0; i < 4 && p[1 + i]; i++)
					{
						char h = p[1 + i];
						unsigned d;
						if (h >= '0' && h <= '9')
							d = (unsigned)(h - '0');
						else if (h >= 'a' && h <= 'f')
							d = (unsigned)(h - 'a' + 10);
						else if (h >= 'A' && h <= 'F')
							d = (unsigned)(h - 'A' + 10);
						else
							break; /* non-hex digit: NOT consumed -- the char is processed normally next iteration (e.g. the closing quote ends the string) */
						cp = (cp << 4) | d;
						hex_read++;
					}
					p += hex_read; /* advance ONLY past digits that exist -- p += 4 would step over a NUL on truncated input */
					if (hex_read < 4)
					{
						json_string_had_error = 1; /* malformed escape ("\u12"): the parse fails, the handler answers 400 */
						break;
					}
					if (cp >= 0xD800 && cp <= 0xDBFF && p[1] == '\\' && p[2] == 'u')
					{
						unsigned lo = 0;
						int lo_read = 0;
						for (int i = 0; i < 4 && p[3 + i]; i++)
						{
							char h = p[3 + i];
							unsigned d;
							if (h >= '0' && h <= '9')
								d = (unsigned)(h - '0');
							else if (h >= 'a' && h <= 'f')
								d = (unsigned)(h - 'a' + 10);
							else if (h >= 'A' && h <= 'F')
								d = (unsigned)(h - 'A' + 10);
							else
								break;
							lo = (lo << 4) | d;
							lo_read++;
						}
						if (lo_read == 4 && lo >= 0xDC00 && lo <= 0xDFFF)
						{
							cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
							p += 2 + lo_read; /* past "\u" + the digits that exist */
						}
					}
					if (cp >= 0xD800 && cp <= 0xDFFF)
						cp = 0xFFFD; /* lone surrogate: not representable in UTF-8 — U+FFFD replacement */
					if (len < dst_size - 1)
					{
						if (cp == 0)
						{
							/* \u0000: emitting a raw NUL would silently
							 * truncate every C-string consumer — drop it. */
						}
						else if (cp < 0x80)
						{
							dst[len++] = (char)cp;
						}
						else if (cp < 0x800)
						{
							if (len < dst_size - 2)
							{
								dst[len++] = (char)(0xC0 | (cp >> 6));
								dst[len++] = (char)(0x80 | (cp & 0x3F));
							}
						}
						else if (cp < 0x10000)
						{
							if (len < dst_size - 3)
							{
								dst[len++] = (char)(0xE0 | (cp >> 12));
								dst[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
								dst[len++] = (char)(0x80 | (cp & 0x3F));
							}
						}
						else
						{
							if (len < dst_size - 4)
							{
								dst[len++] = (char)(0xF0 | (cp >> 18));
								dst[len++] = (char)(0x80 | ((cp >> 12) & 0x3F));
								dst[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
								dst[len++] = (char)(0x80 | (cp & 0x3F));
							}
						}
					}
					break;
				}
				default:
					if (len < dst_size - 1)
						dst[len++] = *p;
					break;
				}
			}
			else
			{
				if (len < dst_size - 1)
					dst[len++] = *p;
			}
			p++;
		}
		/* A buffer-full cut can land inside a multi-byte UTF-8 sequence: back
		 * the length up to the last sequence boundary so the stored string is
		 * always VALID UTF-8. On a complete tail this is a no-op (the trailing
		 * continuations match their lead). */
		{
			int k = 0;
			while (k < len && ((unsigned char)dst[len - 1 - k] & 0xC0) == 0x80)
				k++;
			if (k < len)
			{
				int lead = (int)(unsigned char)dst[len - 1 - k];
				int expect = (lead & 0xF0) == 0xF0 ? 4 : (lead & 0xE0) == 0xE0 ? 3 : (lead & 0xC0) == 0xC0 ? 2 : 1;
				expect--; /* continuation count the lead byte needs (an ASCII lead needs none) */
				if (k < expect)
					len -= k + 1; /* incomplete sequence (incl. a lone lead at the cut): drop lead + what exists of it */
				else
					len -= (k - expect); /* complete sequence plus stray extra continuations: drop ONLY the strays (an ASCII lead keeps its character) */
			}
			else if (k > 0)
				len = 0; /* only continuation bytes: no lead byte exists -- keep nothing */
		}
		dst[len] = 0;
		if (out_len)
			*out_len = len;
		if (*p == '"')
			p++;
		return p;
	}

	/* Raw span of the JSON string whose CONTENT starts at p (p[0] is the
	 * first character after the opening quote): scans to the closing quote
	 * honoring backslash escapes. Returns the byte count between the
	 * quotes, or -1 when the body ends first (no closing quote). The
	 * decoded text is never LONGER than this span (every escape shrinks:
	 * \n 2->1, \uXXXX 6->3, surrogate pair 12->4), so span+1 is an exact
	 * allocation bound and the parse below can never truncate. */
	int json_string_span(char *p)
	{
		char *s = p;
		while (*p && *p != '"')
		{
			if (*p == '\\')
			{
				p++;
				if (!*p)
					break; /* trailing backslash at end of input */
			}
			p++;
		}
		if (*p != '"')
			return -1;
		return (int)(p - s);
	}

	/* Parse a JSON string into an exact-size heap buffer. *dst is replaced
	 * (any prior buffer freed), so repeated parses of the same field never
	 * leak. Returns the new parse position, or 0 on allocation failure --
	 * callers answer 500, never a truncated field. */
	char *json_parse_string_alloc(char *p, char **dst, int *out_len)
	{
		int span = json_string_span(p);
		if (span < 0)
			span = (int)strlen(p); /* unterminated: the parse below stops at the body NUL */
		char *buf = (char *)malloc((size_t)span + 1);
		if (!buf)
			return 0;
		free(*dst);
		*dst = buf;
		return json_parse_string(p, buf, span + 1, out_len);
	}

	/* ---- request ownership: one free per struct, safe to call twice ---- */

	void chat_request_free(chat_request *r)
	{
		free(r->system_prompt);
		free(r->user_prompt);
		free(r->transcript);
		free(r->model);
		free(r->content_buf);
		memset(r, 0, sizeof(*r)); /* all pointers 0 after: a second call is a no-op */
	}

	void completion_request_free(completion_request *r)
	{
		free(r->prompt);
		free(r->model);
		memset(r, 0, sizeof(*r));
	}

	void vision_request_free(vision_request *r)
	{
		free(r->prompt);
		free(r->image_b64);
		free(r->model);
		memset(r, 0, sizeof(*r));
	}

	void decide_request_free(decide_request *r)
	{
		free(r->state);
		free(r->question);
		for (int i = 0; i < MAX_DECIDE_OPTIONS; i++)
			free(r->options[i]);
		memset(r, 0, sizeof(*r));
	}

	/* Convert a parsed JSON number token to an int budget: atof handles the
	 * exponential and float forms ("1e5" is 100000, not atoi's 1), and the
	 * clamp makes overflow impossible (atoi on a huge value is UB). Negative
	 * or non-numeric results mean "no budget" (0 = server default). */
	int json_int_safe(char *numbuf)
	{
		double d = atof(numbuf);
		if (!(d > 0.0)) /* NaN, negative and zero all mean unlimited */
			return 0;
		if (d > 2147483647.0)
			d = 2147483647.0;
		return (int)d;
	}

	char *json_parse_number(char *p, float *out_f, int *out_i, int *is_int)
	{
		char buf[64];
		char *start = p;
		int len = 0;
		if (*p == '-')
		{
			buf[len++] = *p++;
		}
		/* digits (also the no-progress guard: a leading character that is
		 * neither '-' nor a digit -- or a lone '-' with no digits, e.g. the
		 * '-' of "->" inside a skipped value -- leaves no NUMBER: the caller
		 * detects p == start and json_skip_value's the value instead, so
		 * "null" or a string never wedge the message loop) */
		while (*p >= '0' && *p <= '9')
		{
			if (len < 62)
				buf[len++] = *p;
			p++;
		}
		if (len == 0 || (len == 1 && buf[0] == '-'))
			return start; /* not a number: no progress */
		if (*p == '.')
		{
			if (len < 62)
				buf[len++] = *p;
			p++;
			while (*p >= '0' && *p <= '9')
			{
				if (len < 62)
					buf[len++] = *p;
				p++;
			}
		}
		if (*p == 'e' || *p == 'E')
		{
			if (len < 62)
				buf[len++] = *p;
			p++; /* ALWAYS advance: with a full buffer the parser must not strand on the exponent char */
			if (*p == '+' || *p == '-')
			{
				if (len < 62)
					buf[len++] = *p;
				p++;
			}
			while (*p >= '0' && *p <= '9')
			{
				if (len < 62)
					buf[len++] = *p;
				p++;
			}
		}
		buf[len] = 0;
		if (is_int)
			*is_int = (strchr(buf, '.') == 0 && strchr(buf, 'e') == 0 && strchr(buf, 'E') == 0);
		if (out_f)
			*out_f = (float)atof(buf);
		if (out_i)
			*out_i = json_int_safe(buf);
		return p;
	}

	char *json_skip_value(char *p)
	{
		p = json_skip_ws(p);
		if (*p == '"')
		{
			p++;
			while (*p && *p != '"')
			{
				if (*p == '\\')
				{
					p++;
					if (!*p)
						break; /* trailing backslash at end of input */
				}
				p++;
			}
			if (*p == '"')
				p++;
		}
		else if (*p == '{')
		{
			int depth = 1;
			p++;
			while (*p && depth > 0)
			{
				if (*p == '{')
					depth++;
				else if (*p == '}')
					depth--;
				else if (*p == '"')
				{
					p++;
					while (*p && *p != '"')
					{
						if (*p == '\\')
						{
							p++;
							if (!*p)
								break; /* trailing backslash at end of input */
						}
						p++;
					}
					if (*p == '"')
						p++;
					continue;
				}
				p++;
			}
		}
		else if (*p == '[')
		{
			int depth = 1;
			p++;
			while (*p && depth > 0)
			{
				if (*p == '[')
					depth++;
				else if (*p == ']')
					depth--;
				else if (*p == '"')
				{
					p++;
					while (*p && *p != '"')
					{
						if (*p == '\\')
						{
							p++;
							if (!*p)
								break; /* trailing backslash at end of input */
						}
						p++;
					}
					if (*p == '"')
						p++;
					continue;
				}
				p++;
			}
		}
		else if (*p == '-' || (*p >= '0' && *p <= '9'))
		{
			if (*p == '-')
				p++;
			while (*p >= '0' && *p <= '9')
				p++;
			if (*p == '.')
			{
				p++;
				while (*p >= '0' && *p <= '9')
					p++;
			}
			if (*p == 'e' || *p == 'E')
			{
				p++;
				if (*p == '+' || *p == '-')
					p++;
				while (*p >= '0' && *p <= '9')
					p++;
			}
		}
		else if (strncmp(p, "true", 4) == 0)
		{
			p += 4;
		}
		else if (strncmp(p, "false", 5) == 0)
		{
			p += 5;
		}
		else if (strncmp(p, "null", 4) == 0)
		{
			p += 4;
		}
		return p;
	}

	/* ========================================================================
	 * parse_chat_request — extract fields from an OpenAI-format POST body
	 *
	 * body is modified in-place (strings point into it).
	 * returns 0 on success, -1 on parse error.
	 * ======================================================================== */

	int parse_chat_request(char *body, int body_len, chat_request *req)
	{
		chat_request_init(req);
		json_string_had_error = 0; /* fresh state per request: the flag belongs to THIS body only */

		char role_buf[64];
		int role_len = 0, content_len = 0;

		char *p = body;
		/* A UTF-8 BOM (EF BB BF) before the object is legal for Windows-first
		 * clients (PowerShell 5.1 Out-File -Encoding utf8): skip it instead of
		 * answering a misleading "no user message" 400. */
		if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF)
			p += 3;
		p = json_expect(p, '{');
		if (!p)
			return -1;

		while (*p)
		{
			p = json_skip_ws(p);
			if (*p == '}')
			{
				p++; /* consume and require the top-level closing brace */
				if (json_string_had_error)
					return -1; /* a malformed \uXX escape appeared in any string of the request */
				return 0;
			}
			if (*p == 0)
				return -1; /* EOF inside the top-level object: truncated request */
			if (*p == ',')
			{
				p++;
				continue;
			}

			p = json_expect(p, '"');
			if (!p)
				return -1;
			char key[128];
			p = json_parse_string(p, key, sizeof(key), 0);
			p = json_expect(p, ':');
			if (!p)
				return -1;
			p = json_skip_ws(p);

			if (strcmp(key, "messages") == 0)
			{
				p = json_expect(p, '[');
				if (!p)
					return -1;
				/* A duplicate "messages" key REPLACES the previous one (JSON
				 * objects are unordered maps): reset everything it fills, or
				 * the transcript grows twice and the counts double-count.
				 * Allocations are kept (reset lengths, keep capacity). */
				req->transcript_len = 0;
				if (req->transcript)
					req->transcript[0] = 0;
				req->user_count = 0;
				req->assistant_count = 0;
				free(req->user_prompt);
				req->user_prompt = 0;
				req->user_len = 0;
				free(req->system_prompt);
				req->system_prompt = 0;
				req->system_len = 0;
				while (*p)
				{
					p = json_skip_ws(p);
					if (*p == 0)
						return -1; /* EOF inside the messages array: truncated request */
					if (*p == ']')
					{
						p++;
						break;
					}
					if (*p == ',')
					{
						p++;
						continue;
					}

					p = json_expect(p, '{');
					if (!p)
						return -1;

					char *msg_role = 0;
					int msg_role_len = 0;
					const char *msg_content = 0;
					int msg_content_len = 0;

					while (*p)
					{
						p = json_skip_ws(p);
						if (*p == 0)
							return -1; /* EOF inside a message object: truncated request */
						if (*p == '}')
						{
							p++;
							break;
						}
						if (*p == ',')
						{
							p++;
							continue;
						}

						p = json_expect(p, '"');
						if (!p)
							return -1;
						char mkey[128];
						p = json_parse_string(p, mkey, sizeof(mkey), 0);
						p = json_expect(p, ':');
						if (!p)
							return -1;
						p = json_skip_ws(p);

							if (strcmp(mkey, "role") == 0)
						{
							p = json_expect(p, '"');
							if (!p)
								return -1; /* "role": null / numeric -- NULL deref otherwise */
							p = json_parse_string(p, role_buf, sizeof(role_buf), &role_len);
							msg_role = role_buf;
							msg_role_len = role_len;
						}
						else if (strcmp(mkey, "content") == 0)
						{
							if (*p == '"')
							{
								p++;
								int span = json_string_span(p);
								if (span < 0)
									span = (int)strlen(p);
								if (!scratch_grow(&req->content_buf, &req->content_buf_cap, span + 1))
									return -1;
								p = json_parse_string(p, req->content_buf, req->content_buf_cap, &content_len);
								msg_content = req->content_buf;
								msg_content_len = content_len;
							}
							else if (*p == '[')
							{
								/* multimodal form: content is an array of parts,
								 * e.g. [{"type":"text","text":"..."}]. Concatenate
								 * every text part (and bare string elements) into
								 * content_buf, separated by newlines. Non-text
								 * parts (image_url etc.) are skipped -- this
								 * server takes images only via /v1/vision/chat. */
								p++;
								int clen = 0;
								int first_part = 1;
								while (*p && *p != ']')
								{
									p = json_skip_ws(p);
									if (*p == ',' || *p == ']')
									{
										if (*p == ',')
											p++;
										continue;
									}
									if (*p == '{')
									{
										p++;
										while (*p && *p != '}')
										{
											p = json_skip_ws(p);
											if (*p == '}' || *p == ',')
											{
												if (*p == ',')
													p++;
												continue;
											}
											p = json_expect(p, '"');
											if (!p)
												return -1; /* malformed part key: break falls through to the *p == '}' check below with p NULL */
											char pkey[64];
											p = json_parse_string(p, pkey, sizeof(pkey), 0);
											p = json_expect(p, ':');
											if (!p)
												return -1; /* missing ':' after part key: break falls through to the *p == '}' check below with p NULL */
											p = json_skip_ws(p);
											if (strcmp(pkey, "text") == 0 && *p == '"')
											{
												p++;
												int span = json_string_span(p);
												if (span < 0)
													span = (int)strlen(p);
												if (!scratch_grow(&req->content_buf, &req->content_buf_cap, clen + span + 2))
													return -1;
												if (!first_part)
													req->content_buf[clen++] = '\n';
												int seg = 0;
												p = json_parse_string(p, req->content_buf + clen, req->content_buf_cap - clen, &seg);
												clen += seg;
												first_part = 0;
											}
											else
											{
												char *q = json_skip_value(p);
												if (q == p)
													return -1; /* bare unrecognized token: no progress -- this loop would spin forever */
												p = q;
											}
										}
										if (*p == '}')
											p++;
									}
									else if (*p == '"')
									{
										p++;
										int span = json_string_span(p);
										if (span < 0)
											span = (int)strlen(p);
										if (!scratch_grow(&req->content_buf, &req->content_buf_cap, clen + span + 2))
											return -1;
										if (!first_part)
											req->content_buf[clen++] = '\n';
										int seg = 0;
										p = json_parse_string(p, req->content_buf + clen, req->content_buf_cap - clen, &seg);
										clen += seg;
										first_part = 0;
									}
									else
									{
										char *q = json_skip_value(p);
										if (q == p)
											break; /* malformed element: bail out of the array */
										p = q;
									}
								}
								if (*p == ']')
									p++;
								req->content_buf[clen] = 0;
								if (clen > 0)
								{
									msg_content = req->content_buf;
									msg_content_len = clen;
								}
							}
							else
							{
								p = json_skip_value(p); /* content: null / number -- role still counts below */
							}
						}
						else
						{
							p = json_skip_value(p);
						}
					}

				/* Stateless detection counts ROLES, not contents: an assistant
				 * tool-call turn (content: null) or an array-content message
				 * still proves the client sends full history. */
				if (msg_role)
				{
					if (strcmp(msg_role, "user") == 0)
						req->user_count++;
					else if (strcmp(msg_role, "assistant") == 0)
						req->assistant_count++;
					if (!msg_content)
					{
						/* a content-less turn (assistant tool call, content:
						 * null) still goes INTO the transcript: dropping it
						 * rewrote the model's visible history -- it saw
						 * "User: q / Tool: r" with no assistant turn in
						 * between and the call that asked for the tool
						 * result vanished. The faithful (unparseable into
						 * speech anyway) tool_calls structure becomes a
						 * compact marker line. */
						msg_content = "(no content)";
						msg_content_len = (int)strlen(msg_content);
					}
				}

				if (msg_role && msg_content)
				{
					/* transcript: EVERY message in order, as labelled lines.
					 * A stateless full-history client gets this as the turn
					 * prompt, so the request's own history IS the context.
					 * system and developer both label as System (developer is
					 * OpenAI's renamed system role); tool turns label as Tool
					 * so they never read as the user speaking. */
					const char *role_label = "Other"; /* unknown roles never read as the user speaking */
					int role_is_system = 0;
					if (strcmp(msg_role, "system") == 0 || strcmp(msg_role, "developer") == 0)
					{
						role_label = "System";
						role_is_system = 1;
					}
					else if (strcmp(msg_role, "assistant") == 0)
						role_label = "Assistant";
					else if (strcmp(msg_role, "tool") == 0 || strcmp(msg_role, "function") == 0)
						role_label = "Tool";
					else if (strcmp(msg_role, "user") == 0)
						role_label = "User";

					/* Append "Label: content\n\n" to a transcript grown to the
					 * exact need: the request's own history is NEVER dropped
					 * (the old fixed store deleted the oldest lines when full,
					 * silently rewriting the context the client sent). */
					int tlen = req->transcript_len;
					int line_len = (int)strlen(role_label) + 2 + msg_content_len + 2;
					if (!scratch_grow(&req->transcript, &req->transcript_cap, tlen + line_len + 1))
						return -1;

					/* Write "Label: content\n\n". Every newline RUN inside the
					 * content collapses to "\n " (newline + space): a user
					 * message (or a document it quotes) can then never forge a
					 * "\n\nSystem:" / "\n\nAssistant:" turn boundary -- the
					 * only real boundaries are the ones THIS loop writes. */
					int tcap = req->transcript_cap;
					int wrote = snprintf(req->transcript + tlen, tcap - tlen, "%s: ", role_label);
					if (wrote < 0)
						wrote = 0;
					if (wrote > tcap - tlen)
						wrote = tcap - tlen;
					for (int ci = 0; ci < msg_content_len && tlen + wrote < tcap - 2; ci++)
					{
					char c = msg_content[ci];
					if (c == '\r')
						continue; /* CR is dropped: "\r\n\r\n" would otherwise survive the newline-run collapse and forge a turn boundary */
					req->transcript[tlen + wrote++] = c;
					if (c == '\n' && ci + 1 < msg_content_len && (msg_content[ci + 1] == '\n' || msg_content[ci + 1] == '\r'))
					{
						if (tlen + wrote >= tcap - 2)
							break;
							req->transcript[tlen + wrote++] = ' ';
							ci++; /* the run collapses: the second newline is swallowed */
						}
					}
					req->transcript[tlen + wrote++] = '\n';
					req->transcript[tlen + wrote++] = '\n';
					req->transcript[tlen + wrote] = 0;
					req->transcript_len = tlen + wrote;

					if (role_is_system)
					{
						/* Multiple system/developer messages CONCATENATE (the
						 * engine takes one system prompt; dropping earlier
						 * instructions would silently lose them). Grown to the
						 * exact need -- nothing is clipped. */
						int slen = req->system_len;
						int sep = (slen > 0) ? 2 : 0;
						if (!scratch_grow(&req->system_prompt, &req->system_cap, slen + sep + msg_content_len + 1))
							return -1;
						if (sep)
						{
							req->system_prompt[slen++] = '\n';
							req->system_prompt[slen++] = '\n';
						}
						memcpy(req->system_prompt + slen, msg_content, (size_t)msg_content_len);
						slen += msg_content_len;
						req->system_prompt[slen] = 0;
						req->system_len = slen;
					}
					else if (strcmp(msg_role, "user") == 0)
					{
						char *copy = (char *)malloc((size_t)msg_content_len + 1);
						if (!copy)
							return -1;
						memcpy(copy, msg_content, (size_t)msg_content_len + 1); /* content_buf is NUL-terminated at msg_content_len */
						free(req->user_prompt);
						req->user_prompt = copy;
						req->user_len = msg_content_len;
					}
				}
				}
			}
			else if (strcmp(key, "temperature") == 0)
			{
				req->sampler_fields_present = 1; /* present-but-unapplied: the handler logs it */
				int is_int = 0;
				char *q = json_parse_number(p, &req->temperature, 0, &is_int);
				if (q == p)                    /* "temperature": null or a string: no progress -- skip the value token */
					p = json_skip_value(p);
				else
					p = q;
			}
			else if (strcmp(key, "top_p") == 0)
			{
				req->sampler_fields_present = 1; /* present-but-unapplied: the handler logs it */
				int is_int = 0;
				char *q = json_parse_number(p, &req->top_p, 0, &is_int);
				if (q == p)                    /* null / string: no progress -- skip */
					p = json_skip_value(p);
				else
					p = q;
			}
			else if (strcmp(key, "max_tokens") == 0 || strcmp(key, "max_completion_tokens") == 0)
			{
				/* number form only. A quoted value ("max_tokens": "100") is
				 * a malformed budget, not a silent fallback to the server
				 * default -- the parse fails and the handler answers 400.
				 * null -> no progress -> skip, and the server default
				 * stands (documented: null means absent). */
				int is_int = 0;
				float f = 0;
				char *q = json_parse_number(p, &f, &req->max_tokens, &is_int);
				if (q == p)
				{
					if (*p == '"')
						return -1; /* a budget as a JSON string: malformed request */
					p = json_skip_value(p);
				}
				else
					p = q;
			}
			else if (strcmp(key, "enable_thinking") == 0)
			{
				/* non-OpenAI extension: let API clients turn reasoning off
				 * (the engine default is ON, which burns budget on reasoning
				 * and can trigger the no-close-tag rescue). */
				if (strncmp(p, "true", 4) == 0)
				{
					req->thinking = 1;
					p += 4;
				}
				else if (strncmp(p, "false", 5) == 0)
				{
					req->thinking = 0;
					p += 5;
				}
				else
					p = json_skip_value(p);
			}
			else if (strcmp(key, "reasoning_effort") == 0)
			{
				/* OpenAI field: only the "none" level maps onto this engine
				 * (reasoning off); every other level keeps the default ON. */
				if (*p == '"')
				{
					char effort[32];
					p++;
					char *after = json_parse_string(p, effort, sizeof(effort), 0);
					if (strcmp(effort, "none") == 0)
						req->thinking = 0;
					p = after;
				}
				else
					p = json_skip_value(p);
			}
			else if (strcmp(key, "stream") == 0)
			{
				if (strncmp(p, "true", 4) == 0)
				{
					req->stream = 1;
					p += 4;
				}
				else if (strncmp(p, "false", 5) == 0)
				{
					req->stream = 0;
					p += 5;
				}
				else
					p = json_skip_value(p);
			}
			else if (strcmp(key, "model") == 0)
			{
				if (strncmp(p, "null", 4) == 0)
				{
					p += 4; /* "model": null is legal JSON and common from SDK wrappers: skip it, keep the server default */
				}
				else
				{
					p = json_expect(p, '"');
					if (!p)
						return -1; /* non-string, non-null model -- NULL deref otherwise */
					char *after = json_parse_string_alloc(p, &req->model, 0);
					if (!after)
						return -1;
					p = after;
				}
			}
			else if (strcmp(key, "chat_template_kwargs") == 0 && *p == '{')
			{
				/* vLLM/Qwen3 convention: {"chat_template_kwargs":{"enable_thinking":false}}.
				 * Bounded brace-walk looking for the one key we understand;
				 * everything else inside is skipped. */
				p++;
				int depth = 1;
				while (*p && depth > 0)
				{
					if (*p == '{')
					{
						depth++;
						p++;
					}
					else if (*p == '}')
					{
						depth--;
						p++;
					}
					else if (strncmp(p, "\"enable_thinking\"", 17) == 0)
					{
						p += 17;
						p = json_skip_ws(p);
						if (*p == ':')
						{
							p = json_skip_ws(p + 1);
							if (strncmp(p, "false", 5) == 0)
							{
								req->thinking = 0;
								p += 5;
							}
							else if (strncmp(p, "true", 4) == 0)
							{
								req->thinking = 1;
								p += 4;
							}
							else
								p = json_skip_value(p);
						}
					}
					else if (*p == '"')
					{
						p++; /* skip a quoted key or value wholesale */
						while (*p && *p != '"')
						{
							if (*p == '\\' && p[1])
								p++;
							p++;
						}
						if (*p)
							p++;
					}
					else
						p++;
				}
			}
			else
			{
				p = json_skip_value(p);
			}
		}
		return -1; /* ran off the body without the top-level '}': truncated request */
	}

	/* ========================================================================
	 * parse_completion_request — extract fields from /v1/completions POST body
	 *
	 * returns 0 on success, -1 on parse error.
	 * ======================================================================== */

	int parse_completion_request(char *body, int body_len, completion_request *req)
	{
		completion_request_init(req);
		json_string_had_error = 0; /* fresh state per request: the flag belongs to THIS body only */

		char *p = body;
		/* UTF-8 BOM tolerance (same reasoning as parse_chat_request). */
		if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF)
			p += 3;
		p = json_expect(p, '{');
		if (!p)
			return -1;

		while (*p)
		{
			p = json_skip_ws(p);
			if (*p == '}')
			{
				p++;
				if (json_string_had_error)
					return -1; /* a malformed \uXX escape appeared in any string of the request */
				return 0;
			}
			if (*p == 0)
				return -1; /* EOF inside the top-level object: truncated request */
			if (*p == ',')
			{
				p++;
				continue;
			}

			p = json_expect(p, '"');
			if (!p)
				return -1;
			char key[128];
			p = json_parse_string(p, key, sizeof(key), 0);
			p = json_expect(p, ':');
			if (!p)
				return -1;
			p = json_skip_ws(p);

			if (strcmp(key, "prompt") == 0)
			{
				if (*p == '"')
				{
					p++;
					/* exact-size parse: the decoded text is never longer
					 * than its raw span, so the malloc never truncates */
					char *after = json_parse_string_alloc(p, &req->prompt, &req->prompt_len);
					if (!after)
						return -1;
					req->prompt_cap = req->prompt_len + 1;
					p = after;
				}
				else if (*p == '[')
				{
					/* Array form: OpenAI returns one choice PER element; this
					 * server produces one continuation, so the elements are
					 * joined with newlines in order (a single prompt made of
					 * the parts). Grown to the exact need per segment. */
					p++;
					int plen2 = 0;
					int first_seg = 1;
					while (*p && *p != ']')
					{
						p = json_skip_ws(p);
						if (*p == 0)
							return -1; /* EOF inside the prompt array */
						if (*p == ',' || *p == ']')
						{
							if (*p == ',')
								p++;
							continue;
						}
						if (*p == '"')
						{
							p++;
							int span = json_string_span(p);
							if (span < 0)
								span = (int)strlen(p);
							if (!scratch_grow(&req->prompt, &req->prompt_cap, plen2 + span + 2))
								return -1;
							if (!first_seg)
								req->prompt[plen2++] = '\n';
							int seg = 0;
							p = json_parse_string(p, req->prompt + plen2, req->prompt_cap - plen2, &seg);
							plen2 += seg;
							first_seg = 0;
						}
						else
						{
							char *q = json_skip_value(p);
							if (q == p)
								return -1; /* malformed element: not recoverable */
							p = q;
						}
					}
					if (*p == ']')
						p++;
					if (plen2 > 0)
					{
						req->prompt[plen2] = 0;
						req->prompt_len = plen2;
					}
				}
				else
				{
					p = json_skip_value(p);
				}
			}
			else if (strcmp(key, "stream") == 0)
			{
				if (strncmp(p, "true", 4) == 0)
				{
					req->stream = 1;
					p += 4;
				}
				else if (strncmp(p, "false", 5) == 0)
				{
					req->stream = 0;
					p += 5;
				}
				else
					p = json_skip_value(p);
			}
			else if (strcmp(key, "temperature") == 0)
			{
				req->sampler_fields_present = 1; /* present-but-unapplied: the handler logs it */
				int is_int = 0;
				char *q = json_parse_number(p, &req->temperature, 0, &is_int);
				if (q == p)                    /* null / string: skip, default stands */
					p = json_skip_value(p);
				else
					p = q;
			}
			else if (strcmp(key, "top_p") == 0)
			{
				req->sampler_fields_present = 1; /* present-but-unapplied: the handler logs it */
				int is_int = 0;
				char *q = json_parse_number(p, &req->top_p, 0, &is_int);
				if (q == p)
					p = json_skip_value(p);
				else
					p = q;
			}
			else if (strcmp(key, "max_tokens") == 0 || strcmp(key, "max_completion_tokens") == 0)
			{
				/* number form only: a quoted budget is malformed -- parse
				 * error, the handler answers 400. null -> default stands. */
				int is_int = 0;
				float f = 0;
				char *q = json_parse_number(p, &f, &req->max_tokens, &is_int);
				if (q == p)
				{
					if (*p == '"')
						return -1;
					p = json_skip_value(p);
				}
				else
					p = q;
			}
			else if (strcmp(key, "model") == 0)
			{
				if (strncmp(p, "null", 4) == 0)
				{
					p += 4; /* "model": null is legal JSON and common from SDK wrappers: skip it, keep the server default */
				}
				else
				{
					p = json_expect(p, '"');
					if (!p)
						return -1; /* non-string, non-null model */
					char *after = json_parse_string_alloc(p, &req->model, 0);
					if (!after)
						return -1;
					p = after;
				}
			}
			else
			{
				p = json_skip_value(p);
			}
		}
		return -1; /* ran off the body without the top-level '}': truncated request */
	}

	/* ========================================================================
	 * JSON string escaper — writes escaped content into dst.
	 * returns number of chars written (excluding null terminator).
	 * dst_size is the buffer capacity.
	 * Invalid UTF-8 (a broken multi-byte sequence) is replaced with the
	 * U+FFFD replacement character: raw garbage bytes would make the whole
	 * JSON response invalid for strict parsers.
	 * ======================================================================== */

	int json_escape(char *src, int src_len, char *dst, int dst_size)
	{
		int w = 0;
		for (int i = 0; i < src_len && w < dst_size - 8; )
		{
			unsigned char c = (unsigned char)src[i];
			if (c == '"')
			{
				if (w + 2 >= dst_size)
					break;
				dst[w++] = '\\';
				dst[w++] = '"';
				i++;
			}
			else if (c == '\\')
			{
				if (w + 2 >= dst_size)
					break;
				dst[w++] = '\\';
				dst[w++] = '\\';
				i++;
			}
			else if (c == '\n')
			{
				if (w + 2 >= dst_size)
					break;
				dst[w++] = '\\';
				dst[w++] = 'n';
				i++;
			}
			else if (c == '\r')
			{
				if (w + 2 >= dst_size)
					break;
				dst[w++] = '\\';
				dst[w++] = 'r';
				i++;
			}
			else if (c == '\t')
			{
				if (w + 2 >= dst_size)
					break;
				dst[w++] = '\\';
				dst[w++] = 't';
				i++;
			}
			else if (c < 0x20)
			{
				if (w + 6 >= dst_size)
					break;
				w += sprintf(dst + w, "\\u%04x", c);
				i++;
			}
			else if (c < 0x80)
			{
				dst[w++] = (char)c;
				i++;
			}
			else
			{
				/* multi-byte UTF-8: copy the sequence only when it is BOTH
				 * structurally complete (lead + the right number of
				 * continuation bytes inside the source) AND value-correct.
				 * The value checks are the utf8_wellformed bounds: an
				 * overlong encoding (C0/C1/E0 80..9F/F0 80..8F), a UTF-16
				 * surrogate (ED A0..BF) or a value above U+10FFFF (F4
				 * 90..BF) is invalid JSON just like a broken sequence --
				 * passing any of them through unchanged makes the whole
				 * response body fail strict parsers. */
				int seq = (c >= 0xC2 && c <= 0xDF) ? 2 : (c >= 0xE0 && c <= 0xEF) ? 3 : (c >= 0xF0 && c <= 0xF4) ? 4 : 0;
				int first_min = 0x80, first_max = 0xBF; /* bounds on the FIRST continuation */
				if (c == 0xE0)
					first_min = 0xA0; /* E0 80..9F = overlong */
				else if (c == 0xED)
					first_max = 0x9F; /* ED A0..BF = UTF-16 surrogate */
				else if (c == 0xF0)
					first_min = 0x90; /* F0 80..8F = overlong */
				else if (c == 0xF4)
					first_max = 0x8F; /* F4 90..BF = beyond U+10FFFF */
				int valid = seq > 0 && i + seq <= src_len;
				for (int j = 1; valid && j < seq; j++)
				{
					unsigned char cc = (unsigned char)src[i + j];
					int cmin = (j == 1) ? first_min : 0x80;
					int cmax = (j == 1) ? first_max : 0xBF;
					if ((cc & 0xC0) != 0x80 || cc < cmin || cc > cmax)
						valid = 0;
				}
				if (valid)
				{
					if (w + seq >= dst_size)
						break;
					memcpy(dst + w, src + i, (size_t)seq);
					w += seq;
					i += seq;
				}
				else
				{
					if (w + 3 >= dst_size)
						break;
					dst[w++] = (char)0xEF; /* U+FFFD in UTF-8 */
					dst[w++] = (char)0xBB;
					dst[w++] = (char)0xBD;
					i++; /* skip only the bad lead byte; the bytes after it are re-examined */
				}
			}
		}
		dst[w] = 0;
		return w;
	}

	/* ========================================================================
	 * Response builders
	 *
	 * Escape scratch: grown to the exact worst case (every byte escapes to
	 * \u00xx = 6x the source length), so a response can never silently DROP
	 * its content and no content ever clips. The statics are BSS, not the
	 * 1 MB thread stack -- the builders run on the single main thread only.
	 * ======================================================================== */

	char *json_escape_scratch(char **buf, int *cap, char *src, int src_len, int *out_len)
	{
		if (!scratch_grow(buf, cap, src_len * 6 + 16))
			return 0;
		*out_len = json_escape(src, src_len, *buf, *cap);
		return *buf;
	}

	/* ========================================================================
	 * build_completion_response — construct OpenAI /v1/completions response
	 * ======================================================================== */

	int build_completion_response(char *output, int output_size, char *content, int content_len, const char *model, int req_id, const char *finish_reason, int stream, int prompt_tokens, int completion_tokens)
	{
		int w = 0;
		time_t now = time(0);

		static char *escaped = 0;
		static int escaped_cap = 0;
		int elen = 0;
		char *escp = json_escape_scratch(&escaped, &escaped_cap, content, content_len, &elen);
		if (!escp)
			return 0; /* allocation failure: the handler answers 500 -- never a dropped body */
		static char *esc_model = 0;
		static int esc_model_cap = 0;
		int mlen = 0;
		char *model_esc = json_escape_scratch(&esc_model, &esc_model_cap, (char *)model, (int)strlen(model), &mlen);
		if (!model_esc)
			return 0;

		if (stream)
		{
			/* SSE sequence for strict streaming parsers: one content chunk
			 * with finish_reason null, one final chunk with an empty text and
			 * the finish_reason, then [DONE]. */
			w += sprintf(output + w,
			             "data: {"
			             "\"id\":\"cmpl-%d\","
			             "\"object\":\"text_completion\","
			             "\"created\":%lld,"
			             "\"model\":\"%s\","
			             "\"choices\":[{"
			             "\"index\":0,"
			             "\"text\":\"",
			             req_id, (long long)now, model_esc);
			if (w + elen < output_size - 256)
			{
				memcpy(output + w, escp, elen);
				w += elen;
			}
			w += sprintf(output + w, "\",\"finish_reason\":null}]}\n\n");
			w += sprintf(output + w,
			             "data: {"
			             "\"id\":\"cmpl-%d\","
			             "\"object\":\"text_completion\","
			             "\"created\":%lld,"
			             "\"model\":\"%s\","
			             "\"choices\":[{"
			             "\"index\":0,"
			             "\"text\":\"\","
			             "\"finish_reason\":\"%s\"}]}\n\n"
			             "data: [DONE]\n\n",
			             req_id, (long long)now, model_esc, finish_reason);
			return w;
		}

		w += sprintf(output + w,
		             "{"
		             "\"id\":\"cmpl-%d\","
		             "\"object\":\"text_completion\","
		             "\"created\":%lld,"
		             "\"model\":\"%s\","
		             "\"choices\":[{"
		             "\"index\":0,"
		             "\"text\":\"",
		             req_id, (long long)now, model_esc);

		if (w + elen < output_size - 256)
		{
			memcpy(output + w, escp, elen);
			w += elen;
		}

		w += sprintf(output + w, "\","
		                         "\"finish_reason\":\"%s\""
		                         "}],"
		                         "\"usage\":{"
		                         "\"prompt_tokens\":%d,"
		                         "\"completion_tokens\":%d,"
		                         "\"total_tokens\":%d}"
		                         "}\n",
		                         finish_reason, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);

		return w;
	}

	/* ========================================================================
	 * build_chat_response_ex — construct OpenAI /v1/chat/completions response
	 *
	 * output: buffer for the JSON body (not the HTTP headers)
	 * output_size: capacity of output buffer
	 * content: assistant's reply text (raw, may contain control chars)
	 * content_len: length of content
	 * model: model ID string
	 * req_id: incrementing request counter
	 *
	 * returns number of bytes written to output.
	 * ======================================================================== */

	int build_chat_response_ex(char *output, int output_size, char *content, int content_len,
	                           char *reasoning, int reasoning_len, const char *model, int req_id, int stream, const char *finish_reason, int prompt_tokens, int completion_tokens)
	{
		int w = 0;
		time_t now = time(0);

		static char *escaped = 0;
		static int escaped_cap = 0;
		int elen = 0;
		char *escp = json_escape_scratch(&escaped, &escaped_cap, content, content_len, &elen);
		if (!escp)
			return 0; /* allocation failure: the handler answers 500 -- never a dropped body */
		static char *esc_reasoning = 0;
		static int esc_reasoning_cap = 0;
		int rlen = 0;
		char *reasoning_esc = 0;
		if (reasoning)
		{
			reasoning_esc = json_escape_scratch(&esc_reasoning, &esc_reasoning_cap, reasoning, reasoning_len, &rlen);
			if (!reasoning_esc)
				return 0;
		}
		static char *esc_model = 0;
		static int esc_model_cap = 0;
		int mlen = 0;
		char *model_esc = json_escape_scratch(&esc_model, &esc_model_cap, (char *)model, (int)strlen(model), &mlen);
		if (!model_esc)
			return 0;

		if (stream)
		{
			/* chunk 1: role only (strict parsers expect the role chunk first) */
			w += sprintf(output + w,
			             "data: {"
			             "\"id\":\"chatcmpl-%d\","
			             "\"object\":\"chat.completion.chunk\","
			             "\"created\":%lld,"
			             "\"model\":\"%s\","
			             "\"choices\":[{"
			             "\"index\":0,"
			             "\"delta\":{\"role\":\"assistant\"},"
			             "\"finish_reason\":null}]}\n\n",
			             req_id, (long long)now, model_esc);

			/* chunk 2: reasoning (only on thinking turns) */
			if (reasoning && rlen > 0)
			{
				w += sprintf(output + w,
				             "data: {"
				             "\"id\":\"chatcmpl-%d\","
				             "\"object\":\"chat.completion.chunk\","
				             "\"created\":%lld,"
				             "\"model\":\"%s\","
				             "\"choices\":[{"
				             "\"index\":0,"
				             "\"delta\":{\"reasoning_content\":\"",
				             req_id, (long long)now, model_esc);
				if (w + rlen < output_size - 256)
				{
					memcpy(output + w, reasoning_esc, rlen);
					w += rlen;
				}
				w += sprintf(output + w, "\"},\"finish_reason\":null}]}\n\n");
			}

			/* chunk 3: content */
			w += sprintf(output + w,
			             "data: {"
			             "\"id\":\"chatcmpl-%d\","
			             "\"object\":\"chat.completion.chunk\","
			             "\"created\":%lld,"
			             "\"model\":\"%s\","
			             "\"choices\":[{"
			             "\"index\":0,"
			             "\"delta\":{\"content\":\"",
			             req_id, (long long)now, model_esc);
			if (w + elen < output_size - 256)
			{
				memcpy(output + w, escp, elen);
				w += elen;
			}
			w += sprintf(output + w, "\"},\"finish_reason\":null}]}\n\n");

			/* chunk 4: final empty delta + finish_reason, then [DONE] */
			w += sprintf(output + w,
			             "data: {"
			             "\"id\":\"chatcmpl-%d\","
			             "\"object\":\"chat.completion.chunk\","
			             "\"created\":%lld,"
			             "\"model\":\"%s\","
			             "\"choices\":[{"
			             "\"index\":0,"
			             "\"delta\":{},"
			             "\"finish_reason\":\"%s\"}]}\n\n"
			             "data: [DONE]\n\n",
			             req_id, (long long)now, model_esc, finish_reason);
			return w;
		}

		w += sprintf(output + w,
		             "{"
		             "\"id\":\"chatcmpl-%d\","
		             "\"object\":\"chat.completion\","
		             "\"created\":%lld,"
		             "\"model\":\"%s\","
		             "\"choices\":[{"
		             "\"index\":0,"
		             "\"message\":{"
		             "\"role\":\"assistant\",",
		             req_id, (long long)now, model_esc);

		if (reasoning && rlen > 0 && w + rlen < output_size - 256)
		{
			w += sprintf(output + w, "\"reasoning_content\":\"");
			memcpy(output + w, reasoning_esc, rlen);
			w += rlen;
			w += sprintf(output + w, "\",");
		}

		w += sprintf(output + w, "\"content\":\"");

		if (w + elen < output_size - 256)
		{
			memcpy(output + w, escp, elen);
			w += elen;
		}

	w += sprintf(output + w, "\"},"
	                         "\"finish_reason\":\"%s\""
	                         "}],"
	                         "\"usage\":{"
	                         "\"prompt_tokens\":%d,"
	                         "\"completion_tokens\":%d,"
	                         "\"total_tokens\":%d}"
	                         "}\n",
	                         finish_reason, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);

		return w;
	}

	/* ========================================================================
	 * build_error_response — construct OpenAI error JSON
	 * ======================================================================== */

	int build_error_response(char *output, int output_size, const char *message, const char *error_type)
	{
		/* JSON-escape the message: windows paths contain backslashes and user
		 * input may contain quotes — raw embedding produces invalid JSON.
		 * The escape buffer is grown to the exact worst case (6x the message
		 * length): a smaller esc truncates mid-escape-pair and the JSON tail
		 * gets cut. */
		static char *esc = 0;
		static int esc_cap = 0;
		int msg_len = (int)strlen(message);
		if (!scratch_grow(&esc, &esc_cap, msg_len * 6 + 16))
			return 0;
		json_escape((char *)message, msg_len, esc, esc_cap);
		int r = _snprintf(output, output_size,
		                  "{\"error\":{"
		                  "\"message\":\"%s\","
		                  "\"type\":\"%s\","
		                  "\"param\":null,"
		                  "\"code\":null"
		                  "}}\n",
		                  esc, error_type);
		if (r < 0 || r >= output_size)
		{
			/* _snprintf returns -1 on truncation and may not NUL-terminate —
			 * returning -1 as a body length produces "Content-Length: -1" */
			output[output_size - 1] = 0;
			return output_size - 1;
		}
		return r;
	}

	/* ========================================================================
	 * vision request — POST /v1/vision/chat (vision_request lives in
	 * v7_server_json_types.h)
	 * ======================================================================== */

	void vision_request_init(vision_request *r)
	{
		memset(r, 0, sizeof(*r));
		r->temperature = 0.0f;
		r->top_p = 0.9f;
		r->max_tokens = 0; /* 0 = unlimited: the vision handler forces the server default anyway */
	}

	int parse_vision_request(char *body, int body_len, vision_request *req)
	{
		vision_request_init(req);
		json_string_had_error = 0; /* fresh state per request: the flag belongs to THIS body only */

		/* NO fixed buffers anywhere: the base64 is parsed into an
		 * exact-span malloc (the decoded image data is derived from the
		 * base64 length alone, so the handler sizes its decode output the
		 * same way) and the prompt into an exact malloc. */
		char *p = body;
		/* UTF-8 BOM tolerance (same reasoning as parse_chat_request). */
		if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF)
			p += 3;
		p = json_expect(p, '{');
		if (!p)
			return -1;

		while (*p)
		{
			p = json_skip_ws(p);
			if (*p == '}')
			{
				p++;
				if (json_string_had_error)
					return -1; /* a malformed \uXX escape appeared in any string of the request */
				return 0;
			}
			if (*p == 0)
				return -1; /* EOF inside the top-level object: truncated request */
			if (*p == ',')
			{
				p++;
				continue;
			}

			p = json_expect(p, '"');
			if (!p)
				return -1;
			char key[128];
			p = json_parse_string(p, key, sizeof(key), 0);
			p = json_expect(p, ':');
			if (!p)
				return -1;
			p = json_skip_ws(p);

			if (strcmp(key, "prompt") == 0)
			{
				if (*p == '"')
				{
					p++;
					/* exact-size parse: no prompt truncation */
					char *after = json_parse_string_alloc(p, &req->prompt, &req->prompt_len);
					if (!after)
						return -1;
					p = after;
				}
				else
				{
					p = json_skip_value(p);
				}
			}
			else if (strcmp(key, "image_b64") == 0 || strcmp(key, "image") == 0)
			{
				if (*p == '"')
				{
					p++;
					/* exact-span parse of the base64: malloc = raw span + 1,
					 * so an image is limited only by the body framing bound
					 * (the old fixed 8 MB cap rejected large scans with 413
					 * and pre-allocated the worst case for every request) */
					char *after = json_parse_string_alloc(p, &req->image_b64, &req->image_b64_len);
					if (!after)
						return -1;
					p = after;
				}
				else
				{
					p = json_skip_value(p);
				}
			}
			else if (strcmp(key, "temperature") == 0)
			{
				req->sampler_fields_present = 1; /* present-but-unapplied: the handler logs it */
				int is_int = 0;
				char *q = json_parse_number(p, &req->temperature, 0, &is_int);
				if (q == p)                    /* null / string: skip, default stands */
					p = json_skip_value(p);
				else
					p = q;
			}
			else if (strcmp(key, "top_p") == 0)
			{
				req->sampler_fields_present = 1; /* present-but-unapplied: the handler logs it */
				int is_int = 0;
				char *q = json_parse_number(p, &req->top_p, 0, &is_int);
				if (q == p)
					p = json_skip_value(p);
				else
					p = q;
			}
			else if (strcmp(key, "max_tokens") == 0 || strcmp(key, "max_completion_tokens") == 0)
			{
				/* number form only: a quoted budget is malformed -- parse
				 * error, the handler answers 400. null -> default stands. */
				int is_int = 0;
				float f = 0;
				char *q = json_parse_number(p, &f, &req->max_tokens, &is_int);
				if (q == p)
				{
					if (*p == '"')
						return -1;
					p = json_skip_value(p);
				}
				else
					p = q;
			}
			else if (strcmp(key, "model") == 0)
			{
				if (strncmp(p, "null", 4) == 0)
				{
					p += 4; /* "model": null is legal JSON and common from SDK wrappers: skip it, keep the server default */
				}
				else
				{
					p = json_expect(p, '"');
					if (!p)
						return -1; /* non-string, non-null model */
					char *after = json_parse_string_alloc(p, &req->model, 0);
					if (!after)
						return -1;
					p = after;
				}
			}
			else
			{
				p = json_skip_value(p);
			}
		}
		return -1; /* ran off the body without the top-level '}': truncated request (the handler frees the request on every error) */
	}

	/* ========================================================================
	 * base64 decoder — standard AND URL-safe alphabets, ignores whitespace.
	 * returns decoded byte count, or -1 on invalid input / insufficient dst.
	 * ======================================================================== */

	int b64_value(char c)
	{
		if (c >= 'A' && c <= 'Z')
			return c - 'A';
		if (c >= 'a' && c <= 'z')
			return c - 'a' + 26;
		if (c >= '0' && c <= '9')
			return c - '0' + 52;
		if (c == '+' || c == '-')
			return 62; /* '-' is the URL-safe twin of '+' */
		if (c == '/' || c == '_')
			return 63; /* '_' is the URL-safe twin of '/' */
		return -1;
	}

	int b64_decode(char *src, int src_len, unsigned char *dst, int dst_cap)
	{
		unsigned int acc = 0;
		int nbits = 0, w = 0;
		for (int i = 0; i < src_len; i++)
		{
			char c = src[i];
			if (c == '\n' || c == '\r' || c == ' ' || c == '\t')
				continue;
			if (c == '=')
				break; /* padding ends the data */
			int v = b64_value(c);
			if (v < 0)
				return -1;
			/* unsigned shift: signed overflow would be UB -- acc tops out at
			 * 24 bits (3 written bytes), far below the 32-bit wrap. */
			acc = (acc << 6) | (unsigned int)v;
			nbits += 6;
			if (nbits >= 8)
			{
				nbits -= 8;
				if (w >= dst_cap)
					return -1;
				dst[w++] = (unsigned char)((acc >> nbits) & 0xFF);
			}
		}
		return w;
	}

	/* ========================================================================
	 * parse_decide_request — extract fields from a /v1/decide POST body.
	 * Loop shape cloned from parse_completion_request; the options array
	 * iterates like the messages array in parse_chat_request.
	 * returns 0 on success, -1 on parse error.
	 * (decide_request + MAX_DECIDE_OPTIONS live in v7_server_json_types.h)
	 * ======================================================================== */

	int parse_decide_request(char *body, int body_len, decide_request *req)
	{
		memset(req, 0, sizeof(*req)); /* kind stays EMPTY when omitted -- a missing kind is a client bug, the handler answers 400 */
		json_string_had_error = 0; /* fresh state per request: the flag belongs to THIS body only */

		char *p = body;
		/* UTF-8 BOM tolerance (same reasoning as parse_chat_request). */
		if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF)
			p += 3;
		p = json_expect(p, '{');
		if (!p)
			return -1;

		while (*p)
		{
			p = json_skip_ws(p);
			if (*p == '}')
			{
				p++;
				if (json_string_had_error)
					return -1; /* a malformed \uXX escape appeared in any string of the request */
				return 0;
			}
			if (*p == 0)
				return -1; /* EOF inside the top-level object: truncated request */
			if (*p == ',')
			{
				p++;
				continue;
			}

			p = json_expect(p, '"');
			if (!p)
				return -1;
			char key[128];
			p = json_parse_string(p, key, sizeof(key), 0);
			p = json_expect(p, ':');
			if (!p)
				return -1;
			p = json_skip_ws(p);

			if (strcmp(key, "kind") == 0 && *p == '"')
			{
				/* the three documented kinds, compared directly: the kind
				 * is never copied through a truncating buffer, so a wrong
				 * or overlong kind fails HERE as a parse error instead of
				 * surfacing as a misleading option-count complaint later */
				char *value = p + 1;
				if (strncmp(value, "noul", 4) == 0 && value[4] == '"')
				{
					strcpy(req->kind, "noul");
					p = value + 5;
				}
				else if (strncmp(value, "score", 5) == 0 && value[5] == '"')
				{
					strcpy(req->kind, "score");
					p = value + 6;
				}
				else if (strncmp(value, "choice", 6) == 0 && value[6] == '"')
				{
					strcpy(req->kind, "choice");
					p = value + 7;
				}
				else
					return -1; /* unknown kind: malformed request */
			}
			else if (strcmp(key, "state") == 0 && *p == '"')
			{
				p++;
				char *after = json_parse_string_alloc(p, &req->state, 0);
				if (!after)
					return -1;
				p = after;
			}
			else if (strcmp(key, "question") == 0 && *p == '"')
			{
				p++;
				char *after = json_parse_string_alloc(p, &req->question, 0);
				if (!after)
					return -1;
				p = after;
			}
		else if (strcmp(key, "options") == 0 && *p == '[')
		{
			for (int oi = 0; oi < MAX_DECIDE_OPTIONS; oi++)
			{
				free(req->options[oi]); /* a duplicate "options" key REPLACES the list (same contract as duplicate "messages"): labels must not accumulate or leak */
				req->options[oi] = 0;
			}
			req->num_options = 0;
			p++;
				p = json_skip_ws(p);
				while (*p && *p != ']')
				{
					if (*p == 0)
						return -1; /* EOF inside the options array: truncated request */
					if (*p == ',')
					{
						p++;
						p = json_skip_ws(p);
						continue;
					}
					if (*p != '"')
					return -1; /* a non-string element ([abc], 123, null) is a malformed options array, not a skippable value */
					p++;
					if (req->num_options < MAX_DECIDE_OPTIONS)
					{
						/* exact-size label: the engine's own single-token
						 * rule is the real label bound -- the old fixed
						 * 128-byte optbuf truncated a long label FIRST and
						 * the verifier then rejected the mangled text */
						char *after = json_parse_string_alloc(p, &req->options[req->num_options], 0);
						if (!after)
							return -1;
						p = after;
					}
					else
					{
						/* beyond the protocol letter range: skip the string,
						 * keep the count -- the handler's > MAX check must
						 * still see the extras and answer 400 */
						int span = json_string_span(p);
						if (span < 0)
							span = (int)strlen(p);
						p += span;
						if (*p == '"')
							p++;
					}
					req->num_options++;
					p = json_skip_ws(p);
				}
				if (*p == ']')
					p++; /* consume the closer; on EOF the top-level check below reports the truncation */
			}
			else
			{
				p = json_skip_value(p);
			}
		}
		return -1; /* ran off the body without the top-level '}': truncated request */
	}

	/* ========================================================================
	 * build_decide_response — one JSON line:
	 *   {"options":[...],"probabilities":[...],"choice_index":N,"choice":"..."}
	 * ======================================================================== */

	int build_decide_response(char *dst, int dst_cap, decide_request *req, float *probs, int best_index)
	{
		/* label escape scratch, grown to the exact worst case (labels are
		 * malloc'd exact-size now -- no 128-byte bound exists to lean on) */
		static char *esc = 0;
		static int esc_cap = 0;

		/* snprintf returns the WOULD-BE length: added to w unbounded, w can
		 * pass dst_cap — and `dst + w` would then form a POINTER past the
		 * buffer end (UB, sanitizer flag) even with a zero size. w is
		 * clamped to dst_cap - 1 after every append, and the final return
		 * is the clamped written length. */
		int w = snprintf(dst, dst_cap, "{\"options\":[");
		if (w < 0)
			w = 0;
		if (w > dst_cap - 1)
			w = dst_cap - 1;
		for (int i = 0; i < req->num_options; i++)
		{
			int label_len = (int)strlen(req->options[i]);
			if (!scratch_grow(&esc, &esc_cap, label_len * 6 + 16))
				return 0;
			int elen = json_escape(req->options[i], label_len, esc, esc_cap);
			w += snprintf(dst + w, dst_cap - w, "%s\"%.*s\"", i ? "," : "", elen, esc);
			if (w < 0)
				w = 0;
			if (w > dst_cap - 1)
				w = dst_cap - 1;
		}
		w += snprintf(dst + w, dst_cap - w, "],\"probabilities\":[");
		if (w > dst_cap - 1)
			w = dst_cap - 1;
		for (int i = 0; i < req->num_options; i++)
		{
			/* NaN/inf can never come from a live softmax, but printing one
			 * would corrupt the whole JSON line -- a non-finite probability
			 * is reported as 0. */
			float p = probs[i];
			if (!(p >= 0.0f) || p > 1.0f)
				p = 0.0f;
			w += snprintf(dst + w, dst_cap - w, "%s%.4f", i ? "," : "", p);
			if (w < 0)
				w = 0;
			if (w > dst_cap - 1)
				w = dst_cap - 1;
		}
		int choice_len = (int)strlen(req->options[best_index]);
		if (!scratch_grow(&esc, &esc_cap, choice_len * 6 + 16))
			return 0;
		int elen = json_escape(req->options[best_index], choice_len, esc, esc_cap);
		w += snprintf(dst + w, dst_cap - w, "],\"choice_index\":%d,\"choice\":\"%.*s\"}", best_index, elen, esc);
		if (w < 0)
			w = 0;
		if (w > dst_cap - 1)
			w = dst_cap - 1;
		return w;
	}

#ifdef __cplusplus
}
#endif
