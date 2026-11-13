// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_util.cpp -- console-log sanitization, UTF-8 boundary trimming,
 * well-formedness checks, HTTP method/path/query extraction, form-field
 * lookup and percent-decoding (relocated verbatim from v7_server_main.cpp). */

#include "v7_server_common.h"

/* Replace control characters with '?' in place: method/path/query/prompt
 * strings reach the console log raw, and embedded ESC (ANSI) or CR/LF can
 * forge log lines or manipulate the terminal. Newlines/tabs survive so
 * prompt logs stay readable. */
void sanitize_for_log(char *s)
{
	for (; *s; s++)
	{
		unsigned char c = (unsigned char)*s;
		if (c < 0x20 && c != '\n' && c != '\r' && c != '\t')
			*s = '?';
		else if (c == 0x7F)
			*s = '?';
	}
}

/* Backtrack a byte length to the last complete UTF-8 sequence boundary.
 * k counts trailing continuation bytes (10xxxxxx); their lead byte tells
 * the EXPECTED sequence length. Only the truly excess bytes are dropped:
 * an incomplete sequence loses lead + what exists of it, stray extra
 * continuations after a COMPLETE sequence (or after an ASCII char) lose
 * only the strays — a buffer ending on a complete non-ASCII character
 * keeps it (an unconditional lead-byte drop would eat the final
 * 。/emoji off every CJK answer). */
int utf8_trim_to_boundary(char *buf, int len)
{
	int k = 0;
	while (k < len && ((unsigned char)buf[len - 1 - k] & 0xC0) == 0x80)
		k++;
	if (k >= len)
		return 0; /* all continuation bytes: no lead byte exists — drop everything (a buf[-1] read / negative return would corrupt the caller) */
	int lead = (unsigned char)buf[len - 1 - k];
	int expect = (lead & 0xF0) == 0xF0 ? 4 : (lead & 0xE0) == 0xE0 ? 3 : (lead & 0xC0) == 0xC0 ? 2 : 1;
	int need = expect - 1; /* continuations the lead byte needs (an ASCII lead needs none) */
	if (k < need)
		return len - 1 - k; /* incomplete sequence: drop lead + what exists of it */
	return len - (k - need); /* complete sequence plus stray extras: drop ONLY the strays */
}

/* print a string to the console log with control characters (except
 * \n \r \t) flattened to '?': raw ESC/CR bytes in prompts or model output
 * could otherwise forge log lines or manipulate the terminal. Streamed in
 * fixed chunks: the log carries the FULL text at any length (a capped
 * buffer once silently cut long reasoning/answers from exactly the
 * diagnostics an operator needs when a turn went wrong). */
void print_sanitized(const char *s)
{
	char chunk[4096];
	int w = 0;
	for (; *s; s++)
	{
		unsigned char c = (unsigned char)*s;
		chunk[w++] = ((c < 0x20 && c != '\n' && c != '\r' && c != '\t') || c == 0x7F) ? '?' : (char)c;
		if (w == (int)sizeof(chunk))
		{
			printf("%.*s", w, chunk);
			w = 0;
		}
	}
	if (w > 0)
		printf("%.*s", w, chunk);
}

/* basic UTF-8 well-formedness (structure only): rejects the stray
 * continuation bytes and truncated sequences a hand-crafted POST can put
 * into the prompt, the history and the rendered page. */
int utf8_wellformed(const char *s, int len)
{
	int i = 0;
	while (i < len)
	{
		unsigned char c = (unsigned char)s[i];
		int extra;
		int first_min = 0x80, first_max = 0xBF; /* bounds on the FIRST continuation (overlong/surrogate rejection) */
		if (c < 0x80)
			extra = 0;
		else if (c >= 0xC2 && c <= 0xDF)
			extra = 1;
		else if (c == 0xE0)
		{
			extra = 2;
			first_min = 0xA0; /* E0 80..9F would be an overlong 2-byte encoding */
		}
		else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF)
			extra = 2;
		else if (c == 0xED)
		{
			extra = 2;
			first_max = 0x9F; /* ED A0..BF encodes UTF-16 surrogates */
		}
		else if (c == 0xF0)
		{
			extra = 3;
			first_min = 0x90; /* F0 80..8F would be an overlong 3-byte encoding */
		}
		else if (c >= 0xF1 && c <= 0xF3)
			extra = 3;
		else if (c == 0xF4)
		{
			extra = 3;
			first_max = 0x8F; /* beyond U+10FFFF */
		}
		else
			return 0; /* C0, C1, F5-FF: never valid leads */
		if (i + extra > len - 1)
			return 0; /* truncated sequence at the end */
		/* continuation-byte checks: ONLY for multi-byte sequences. For ASCII
		 * (extra == 0) there is no continuation to validate -- checking s[i+1]
		 * here rejected every plain-ASCII message, because the byte after an
		 * ASCII character is almost never in 0x80..0xBF. */
		for (int j = 1; j <= extra; j++)
		{
			int cmin = (j == 1) ? first_min : 0x80;  /* the FIRST continuation carries the sequence-specific bound */
			int cmax = (j == 1) ? first_max : 0xBF;
			unsigned char cc = (unsigned char)s[i + j];
			if ((cc & 0xC0) != 0x80 || cc < cmin || cc > cmax)
				return 0;
		}
		i += extra + 1;
	}
	return 1;
}

/* Extract the HTTP method from the request buffer (first word of first line). */
void get_http_method(char *req, char *method, int method_size)
{
	int i = 0;
	while (i < method_size - 1 && req[i] && req[i] != ' ' && req[i] != '\r' && req[i] != '\n')
	{
		method[i] = req[i];
		i++;
	}
	method[i] = 0;
}

/* Extract the URL path from the request buffer (second word of first line).
 * Returns the written length, or -1 when the path did not fit (the caller
 * answers 414 -- a truncated path would route to the WRONG handler or open
 * the WRONG sidecar file). */
int get_http_path(char *req, char *path, int path_size)
{
	char *p = req;
	while (*p && *p != ' ')
		p++; /* skip method */
	while (*p == ' ')
		p++; /* skip spaces */
	int i = 0;
	while (*p && *p != ' ' && *p != '?' && *p != '\r' && *p != '\n')
	{
		if (i >= path_size - 1)
			return -1;
		path[i++] = *p++;
	}
	path[i] = 0;
	return i;
}

/* Extract the query string (everything after '?' in the request line).
 * Returns the written length, or -1 when the query did not fit (the caller
 * answers 414 -- the /sidecar query IS the file path, a cut would open the
 * WRONG file). */
int get_http_query(char *req, char *query, int query_size)
{
	char *p = req;
	while (*p && *p != ' ')
		p++; /* skip method */
	while (*p == ' ')
		p++; /* skip spaces */
	while (*p && *p != ' ' && *p != '?' && *p != '\r' && *p != '\n')
		p++; /* skip path */
	int i = 0;
	if (*p == '?')
	{
		p++;
		while (*p && *p != ' ' && *p != '\r' && *p != '\n')
		{
			if (i >= query_size - 1)
				return -1;
			query[i++] = *p++;
		}
	}
	query[i] = 0;
	return i;
}

/* Find a key in a query string/form body, matching only at field boundaries
 * (start of string or after '&') — never inside another field's value.
 * Returns the value pointer (past "key="), or 0 if absent. */
char *find_field(char *query_string, const char *key)
{
	if (!query_string)
		return 0;
	int key_len = (int)strlen(key);
	char *scan_pos = query_string;
	while ((scan_pos = strstr(scan_pos, key)) != 0)
	{
		int at_start = (scan_pos == query_string);
		int after_amp = (scan_pos > query_string && *(scan_pos - 1) == '&');
		if (at_start || after_amp)
			return scan_pos + key_len;
		scan_pos++;
	}
	return 0;
}

/* Decode %XX escapes in place (the form-body decoder's rules: an invalid
 * escape copies the '%' literally). Keeps /sidecar working for clients
 * that URL-encode the path (spaces, umlauts). '+' is NOT decoded to a
 * space here: that convention belongs to form bodies, and paths can
 * legitimately contain '+' characters. */
void url_decode_query(char *s)
{
	char *w = s;
	while (*s)
	{
		if (*s == '%' && s[1] && s[2])
		{
			char h1 = s[1], h2 = s[2];
			int hi1 = (h1 >= '0' && h1 <= '9') || (h1 >= 'a' && h1 <= 'f') || (h1 >= 'A' && h1 <= 'F');
			int hi2 = (h2 >= '0' && h2 <= '9') || (h2 >= 'a' && h2 <= 'f') || (h2 >= 'A' && h2 <= 'F');
			if (hi1 && hi2)
			{
				char hex[3] = { h1, h2, 0 };
				*w++ = (char)strtol(hex, 0, 16);
				s += 3;
				continue;
			}
		}
		*w++ = *s++;
	}
	*w = 0;
}

/* Build a JSON error body and send it: the one shape every error answer in
 * the server takes (dozens of hand-rolled three-line blocks collapsed into
 * this call). */
void send_error(SOCKET fd, int status, const char *message, const char *error_type)
{
	char err[256];
	int elen = build_error_response(err, sizeof(err), (char *)message, (char *)error_type);
	send_response(fd, status, "application/json", err, elen);
}

/* 303 back to the chat page: the answer every web form endpoint returns
 * after its work (the redirect lands on a page that renders the turn). */
void send_redirect(SOCKET fd)
{
	char *redir = (char *)"HTTP/1.1 303 See Other\r\nLocation: /\r\nContent-Length: 0\r\n\r\n";
	send_all(fd, redir, (int)strlen(redir));
}

/* Trim trailing whitespace, then back to a UTF-8 boundary (a byte-limited
 * output buffer can cut inside a multi-byte sequence). Shared by every
 * inference-output consumer. Returns the new length. */
int trim_tail_utf8(char *out)
{
	int out_len = (int)strlen(out);
	while (out_len > 0 && (out[out_len - 1] == '\n' || out[out_len - 1] == '\r' || out[out_len - 1] == ' '))
		out[--out_len] = 0;
	out_len = utf8_trim_to_boundary(out, out_len);
	out[out_len] = 0;
	return out_len;
}
