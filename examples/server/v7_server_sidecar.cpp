// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_sidecar.cpp -- GET/POST /sidecar?<path>: runtime load of a
 * SectorLogo sidecar script. */

#include "v7_server_common.h"

/* ========================================================================
 * GET/POST /sidecar?<path> — load a SectorLogo sidecar script at runtime.
 * The query string IS the file path (URL percent-decoding applied, so
 * clients that encode spaces or umlauts work too).
 *
 * The script file is read from disk and stored into the microdb hash table
 * under "logo:sidecar" via ht_add (which COPIES the bytes into its pool --
 * freeing the read buffer right after is safe), then the hook is enabled.
 * The hook lazy-inits on the first generated token and reads the key -- so:
 *   - no turn ran with the hook on yet  -> new script is ACTIVE next turn
 *   - a turn already ran with hook on   -> new script is STAGED only; the
 *     parsed program is baked in DLL state, restart (/quit + -loop bat)
 *     to activate it.
 * ======================================================================== */

void handle_sidecar(SOCKET fd, char *query)
{
	char path[8192] = { 0 };
	if (!query || !query[0])
	{
		send_error(fd, 400, "sidecar path missing -- use /sidecar?C:/dir/x.logo", "invalid_request_error");
		return;
	}
	strncpy(path, query, sizeof(path) - 1);
	url_decode_query(path);

	FILE *f = fopen(path, "rb");
	if (!f)
	{
		/* Message + escape buffer sized from the ACTUAL path: a fixed
		 * buffer would cut the JSON mid-string on a long path. */
		int msg_len = 26 + (int)strlen(path);
		char *msg = (char *)malloc((size_t)msg_len + 1);
		if (!msg)
		{
			send_error(fd, 500, "out of memory", "internal_error");
			return;
		}
		snprintf(msg, (size_t)msg_len + 1, "cannot open sidecar file: %s", path);
		char *err = (char *)malloc((size_t)msg_len * 6 + 128);
		if (!err)
		{
			free(msg);
			send_error(fd, 500, "out of memory", "internal_error");
			return;
		}
		int elen = build_error_response(err, msg_len * 6 + 128, msg, "invalid_request_error");
		send_response(fd, 400, "application/json", err, elen);
		free(msg);
		free(err);
		return;
	}
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	rewind(f);
	if (fsize < 1)
	{
		fclose(f);
		send_error(fd, 400, "sidecar script file is empty", "invalid_request_error");
		return;
	}
	char *script = (char *)malloc((size_t)fsize + 1);
	if (!script)
	{
		fclose(f);
		send_error(fd, 500, "out of memory loading sidecar", "internal_error");
		return;
	}
	int nread = (int)fread(script, 1, (size_t)fsize, f);
	fclose(f);
	/* A short read would silently store a TRUNCATED script under the key:
	 * the next turn would run half a program. Fail the load instead. */
	if (nread != fsize)
	{
		free(script);
		send_error(fd, 500, "sidecar file read came back short -- the file changed under us", "internal_error");
		return;
	}
	script[nread] = 0;

	if (!server_dll.v7_db_add("logo:sidecar", script, nread + 1, 0))
	{
		free(script);
		send_error(fd, 500, "v7_db_add('logo:sidecar') failed", "internal_error");
		return;
	}
	/* the hook is DATA-DRIVEN: the next chat refreshes the sidecar from the
	 * DB key (add/overwrite/switch), no host call arms it */
	hook_enabled = 1;
	free(script);

	/* JSON-escape the path for the response: paths contain backslashes
	 * (\ -> \\), quotes, and can contain control bytes -- the shared
	 * escaper handles all of them, so the JSON always stays valid. Escape
	 * buffer + response grown to the exact worst case (6x per byte), so a
	 * long path can never cut the response mid-string. */
	int path_len = (int)strlen(path);
	char *escaped_path = (char *)malloc((size_t)path_len * 6 + 16);
	if (!escaped_path)
	{
		send_error(fd, 500, "out of memory", "internal_error");
		return;
	}
	json_escape(path, path_len, escaped_path, path_len * 6 + 16);

	const char *status = "active";
	if (hook_executed)
		status = "staged";
	char *resp = (char *)malloc((size_t)path_len * 6 + 1024);
	if (!resp)
	{
		free(escaped_path);
		send_error(fd, 500, "out of memory", "internal_error");
		return;
	}
	int resp_cap = path_len * 6 + 1024;
	int rlen = snprintf(resp, (size_t)resp_cap,
	                   "{\"status\":\"%s\",\"file\":\"%s\",\"bytes\":%u,"
	                   "\"note\":\"%s\"}\n",
	                   status, escaped_path, (unsigned)nread, hook_executed ? "hook already initialized -- restart the server (/quit + -loop bat) to activate" : "loaded -- active from the next chat turn");
	send_response(fd, 200, "application/json", resp, rlen);
	free(escaped_path);
	free(resp);

	printf("[v7_server] sidecar %s (%u bytes): %s\n", path, (unsigned)nread, status);
}

/* ========================================================================
 * Server-side conversation history for the HTML chat interface
 * (the chat_msg type lives in v7_server_common.h; the array is malloc'd
 * to the -history capacity in v7_server_main.cpp)
 * ======================================================================== */

int g_chat_count = 0;
int g_think_counter = 0; /* unique <div id=tk%d> ids WITHIN one rendered page (reset at every page build) */
int thinking_enabled = 1; /* last submitted state of the thinking checkbox; the page re-renders after every turn, so it must be echoed back or an uncheck appears to revert */
