// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/* v7_server_web_render.cpp -- the browser chat page: server-side history,
 * HTML escaping, the thinking-block splitter, the page template and the
 * sidecar background lookup (relocated verbatim from v7_server_main.cpp). */

#include "v7_server_common.h"

/* Append a message to the chat history. Each entry owns a malloc'd exact
 * copy of the text -- nothing is ever clipped. When full, evicts the OLDEST
 * pair (freeing its text) -- new turns must never silently vanish from the
 * page while inference runs. */
void chat_history_append(char role, char think, char *text, int len)
{
	if (g_chat_count >= g_chat_history_cap)
	{
		int drop = 2;
		if (g_chat_history_cap < 2)
			drop = 1;
		for (int i = 0; i < drop; i++)
		{
			free(g_chat_history[i].text);
			g_chat_history[i].text = 0;
		}
		memmove(&g_chat_history[0], &g_chat_history[drop], sizeof(chat_msg) * (g_chat_count - drop));
		g_chat_count -= drop;
	}
	char *copy = (char *)malloc((size_t)len + 1);
	if (!copy)
	{
		printf("[v7_server] chat history: out of memory -- the turn is not stored\n");
		return; /* a failed allocation drops the WHOLE append: the page shows fewer turns, never a truncated one */
	}
	memcpy(copy, text, len);
	copy[len] = 0;
	g_chat_history[g_chat_count].role = role;
	g_chat_history[g_chat_count].think = think;
	g_chat_history[g_chat_count].text = copy;
	g_chat_history[g_chat_count].len = len;
	g_chat_count++;
}

/* Release every history entry (session reset / fresh stateless session):
 * the entries own their malloc'd text. */
void chat_history_clear(void)
{
	for (int i = 0; i < g_chat_count; i++)
	{
		free(g_chat_history[i].text);
		g_chat_history[i].text = 0;
	}
	g_chat_count = 0;
}

/* HTML-escape a string into dst. Returns bytes written. A cap-cut never
 * lands inside a multi-byte UTF-8 sequence (the cut is backed up to the
 * last sequence boundary — a partial sequence would be invalid UTF-8 all
 * the way into the browser). */
int html_escape_buf(char *src, int src_len, char *dst, int dst_cap)
{
	int write_pos = 0;
	for (int i = 0; i < src_len && write_pos < dst_cap - 6; i++)
	{
		unsigned char ch = (unsigned char)src[i];
		if (ch == '&')
		{
			dst[write_pos++] = '&';
			dst[write_pos++] = 'a';
			dst[write_pos++] = 'm';
			dst[write_pos++] = 'p';
			dst[write_pos++] = ';';
		}
		else if (ch == '<')
		{
			dst[write_pos++] = '&';
			dst[write_pos++] = 'l';
			dst[write_pos++] = 't';
			dst[write_pos++] = ';';
		}
		else if (ch == '>')
		{
			dst[write_pos++] = '&';
			dst[write_pos++] = 'g';
			dst[write_pos++] = 't';
			dst[write_pos++] = ';';
		}
		else if (ch == '"')
		{
			dst[write_pos++] = '&';
			dst[write_pos++] = 'q';
			dst[write_pos++] = 'u';
			dst[write_pos++] = 'o';
			dst[write_pos++] = 't';
			dst[write_pos++] = ';';
		}
		else if (ch == '\n')
		{
			dst[write_pos++] = '<';
			dst[write_pos++] = 'b';
			dst[write_pos++] = 'r';
			dst[write_pos++] = '>';
		}
		else if (ch == '\r')
		{
			/* skip: CRLF text would otherwise render "\r<br>" as an extra
			 * blank line under white-space:pre-wrap (the LF already became
			 * the line break) */
		}
		else
		{
			dst[write_pos++] = (char)ch;
		}
	}
	write_pos = utf8_trim_to_boundary(dst, write_pos); /* a cap cut can end mid-sequence */
	dst[write_pos] = 0;
	return write_pos;
}

/* Return the index of the first occurrence of tag in the first src_len bytes of
 * src, or -1 when the tag does not occur (including when the remaining bytes
 * are shorter than the tag). */
int find_tag(char *src, int src_len, const char *tag, int tag_len)
{
	for (int i = 0; i + tag_len <= src_len; i++)
	{
		if (memcmp(src + i, tag, tag_len) == 0)
		{
			return i;
		}
	}
	return -1;
}

/* Split bot output into prefix / thinking block / suffix and emit the thinking
 * block as a clickable collapsible div, so the user clicks to read it.
 *
 * think_on gates the whole detection: a turn that ran WITHOUT thinking is
 * plain answer text, even when it MERELY MENTIONS the tag (the model
 * explaining the format must not be hidden behind a reasoning header).
 *
 * The OPENING tag lives in the PROMPT — the chat template opens the block — so
 * the model normally emits ONLY the closing tag: every byte before it is
 * thinking content and must be hidden behind the "thinking..." header. An
 * opening tag inside the output (a model that generates its own) just moves
 * the block start past it. No tags at all => the whole turn is answer text.
 *
 * The tag names are split to avoid disturbing agent software that processes
 * this source file. */
int format_bot_text(char *src, int src_len, char *dst, int dst_cap, int think_on)
{
	const char *open_tag = "<thi"
	                 "nk>";
	const char *close_tag = "</thi"
	                  "nk>";
	int open_len = (int)strlen(open_tag);
	int close_len = (int)strlen(close_tag);

	/* Offsets of the first opening/closing tag in the turn, or -1 when absent.
	 * "<think>" and "</think>" differ at byte 1 ('<' vs '/'), so a single hit
	 * is never both. */
	int open_at = think_on ? find_tag(src, src_len, open_tag, open_len) : -1;
	int close_at = think_on ? find_tag(src, src_len, close_tag, close_len) : -1;

	/* Byte range of the thinking block and where answer text resumes after it.
	 * block_start < 0 means the turn carries no thinking block at all. */
	int block_start = -1;
	int block_end = src_len;
	int suffix_start = src_len;
	int write_pos = 0;

	if (open_at >= 0 && (close_at < 0 || open_at < close_at))
	{
		block_start = open_at + open_len;
	}
	else if (close_at >= 0)
	{
		block_start = 0;
	}

	/* No tags: the whole turn is answer text. */
	if (block_start < 0)
	{
		return html_escape_buf(src, src_len, dst, dst_cap);
	}

	/* A closing tag at or after the block start closes it, and answer text
	 * resumes right after that tag. */
	if (close_at >= block_start)
	{
		block_end = close_at;
		suffix_start = close_at + close_len;
	}

	/* Text before the block is answer text and renders inline. When the
	 * block started at a model-generated OPEN tag, the prefix ends at the
	 * tag itself -- including it would render a literal "<think>" above
	 * the reasoning box. Reserving the chrome bytes keeps the header write
	 * below in bounds after a long prefix. */
	int prefix_len = block_start;
	if (open_at >= 0 && block_start == open_at + open_len)
		prefix_len = open_at;
	if (prefix_len > 0)
	{
		write_pos = html_escape_buf(src, prefix_len, dst, dst_cap - THINK_BLOCK_CHROME_BYTES);
	}

	/* Clickable header + hidden content div. build_chat_page sizes dst to
	 * the per-turn worst case (6x the text + chrome), and the prefix above
	 * never wrote past dst_cap - THINK_BLOCK_CHROME_BYTES, so the ~700-byte
	 * header always fits. The toggle also flips the marker glyph so the
	 * header reads as expanded or collapsed. */
	write_pos += sprintf(dst + write_pos,
	             "<div style=\"margin:2px 0 12px 0;padding:12px 16px;border-radius:var(--radius);"
	             "border:1px solid rgba(242,18,235,0.28);background:rgba(14,6,30,0.90)\">"
	             "<div onclick=\"var e=document.getElementById('tk%d');"
	             "e.style.display=e.style.display==='none'?'block':'none';"
	             "this.firstChild.nodeValue=e.style.display==='none'?'&#9654; reasoning':'&#9662; reasoning';\" "
	             "style=\"cursor:pointer;font-family:var(--mono);font-size:12px;letter-spacing:2px;"
	             "text-transform:uppercase;color:var(--acc2)\">&#9654; reasoning</div>"
	             "<div id=\"tk%d\" style=\"display:none;margin-top:12px;padding:14px 16px;"
	             "border-radius:var(--radius);border:1px solid var(--line);background:rgba(0,3,29,0.85);"
	             "font-family:var(--mono);font-size:13px;line-height:1.65;color:var(--faint);"
	             "white-space:pre-wrap;word-wrap:break-word\">",
	             g_think_counter, g_think_counter);
	g_think_counter++;

	/* HTML-escape the think content with the FULL remaining capacity: the
	 * caller sizes the destination to the worst case, so reasoning and
	 * answer always both fit -- the old half-split starved exactly the
	 * reasoning the user clicked to read. The -128 still covers the closing
	 * tags. */
	write_pos += html_escape_buf(src + block_start, block_end - block_start, dst + write_pos, dst_cap - write_pos - 128);
	write_pos += sprintf(dst + write_pos, "</div></div>");

	/* Answer text after the block renders inline, escaped like any model text. */
	write_pos += html_escape_buf(src + suffix_start, src_len - suffix_start, dst + write_pos, dst_cap - write_pos);
	return write_pos;
}

/* ========================================================================
 * GET / — browser chat UI (template-based, form-submit, reload-on-send)
 *
 * The conversation history is stored server-side (g_chat_history[]).
 * GET / renders the full conversation with an input form.
 * POST /chat/submit appends a user message, calls inference, redirects to /.
 * POST /chat/reset clears history and KV cache, redirects to /.
 *
 * The model's thinking content is wrapped in a clickable collapsible div so the
 * user clicks to read it. The chat template opens the thinking block in the
 * prompt, so the model usually emits only the closing "thi"+"nk" tag and
 * everything before it is the thinking block.
 * ======================================================================== */

char *build_chat_page(int *out_len)
{
	/* The page buffer is grown to hold EVERY turn, the RAG panel and the
	 * tail form -- nothing is dropped (the old fixed 1 MB page stopped
	 * rendering at the newest turns that fit). Static + grown: the page is
	 * built only on the single main thread. */
	static char *page = 0;
	static int page_cap = 0;
	if (!scratch_grow(&page, &page_cap, 16384))
		return 0;
	char *buf = page;
	int buf_cap = page_cap;
	g_think_counter = 0; /* ids only need uniqueness WITHIN one rendered page: a per-boot counter would grow without bound */
	int write_pos = 0;
	write_pos += sprintf(buf + write_pos, "<!DOCTYPE html>\n"
	                      "<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
	                      "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
	                      "<title>v7 chat &middot; CPKI GmbH</title><style>\n"
	                      /* Local font stacks only: a third-party @font-face would leak
	                       * every page view (IP + Referer http://127.0.0.1:8080/) to a
	                       * remote server and break offline. */
	                      ":root{--bg:#00031D;--bg2:#060A25;--panel:#0a1030;--panel2:#060A25;"
	                      "--line:rgba(227,231,240,0.11);--txt:#f2f4f7;--dim:#a4b0d4;--faint:#727f9f;"
	                      "--acc:#4feef7;--acc2:#F212EB;--radius:0.75rem;"
	                      "--mono:\"Cascadia Code\",\"Consolas\",\"JetBrains Mono\",monospace}\n"
	                      "*{margin:0;padding:0;box-sizing:border-box}\n"
	                      "html,body{height:100%%}\n"
	                      "body{font-family:\"Cascadia Code\",\"Segoe UI\",system-ui,sans-serif;font-size:16px;"
	                      "line-height:1.5;letter-spacing:0.04em;background:var(--bg);color:var(--txt);"
	                      "display:flex;flex-direction:column;overflow:hidden}\n"
	                      "body::before{content:\"\";position:fixed;inset:0;z-index:0;pointer-events:none;"
	                      "background-image:linear-gradient(rgba(79,238,247,0.028) 1px,transparent 1px),"
	                      "linear-gradient(90deg,rgba(79,238,247,0.028) 1px,transparent 1px),"
	                      "radial-gradient(1200px 600px at 85%% -10%%,rgba(38,76,171,0.28),transparent 60%%),"
	                      "radial-gradient(900px 500px at -10%% 110%%,rgba(242,18,235,0.10),transparent 60%%);"
	                      "background-size:44px 44px,44px 44px,auto,auto}\n"
	                      "#topbar{position:fixed;top:0;left:0;right:0;height:6px;z-index:30;"
	                      "background:linear-gradient(90deg,var(--acc),var(--acc2))}\n"
	                      "#head{position:relative;z-index:2;display:flex;align-items:center;gap:16px;"
	                      "padding:22px 28px 14px 28px;font-family:var(--mono);font-size:13px;"
	                      "letter-spacing:2px;text-transform:uppercase;color:var(--faint)}\n"
	                      "#head .logo{font-size:21px;letter-spacing:1px;text-transform:none;color:var(--acc);"
	                      "font-weight:700;border:2px solid var(--acc);border-radius:var(--radius);"
	                      "padding:5px 14px;background:rgba(79,238,247,0.06)}\n"
	                      "#head .grow{flex:1}\n"
	                      "#log{position:relative;z-index:1;flex:1;overflow-y:auto;padding:6px 28px 26px 28px;"
	                      "display:flex;flex-direction:column;gap:16px}\n"
	                      "#log::-webkit-scrollbar{width:10px}\n"
	                      "#log::-webkit-scrollbar-thumb{background:rgba(227,231,240,0.16);border-radius:5px}\n"
                      ".msg{max-width:78%%;padding:14px 20px;border-radius:var(--radius);"
                      "border:1px solid rgba(227,231,240,0.16);box-shadow:0 8px 22px rgba(0,0,0,.22);"
                      "background:rgba(6,10,34,0.90);backdrop-filter:blur(8px);"
                      "white-space:pre-wrap;word-wrap:break-word;line-height:1.55}\n"
                      ".msg .who{display:block;font-family:var(--mono);font-size:11.5px;letter-spacing:2px;"
                      "text-transform:uppercase;color:var(--faint);margin-bottom:9px}\n"
                      ".msg.user{align-self:flex-end;border-color:rgba(79,238,247,0.30);"
                      "background:linear-gradient(160deg,rgba(24,46,72,0.92),rgba(8,12,40,0.92))}\n"
	                      ".msg.user .who{color:var(--acc)}\n"
	                      ".msg.bot{align-self:flex-start}\n"
	                      "#bar{position:relative;z-index:2;display:flex;gap:12px;align-items:center;"
	                      "padding:16px 28px 22px 28px;border-top:1px solid rgba(227,231,240,0.08);"
	                      "background:linear-gradient(180deg,rgba(0,3,29,0),rgba(0,3,29,0.86) 45%%,#00031D)}\n"
	                      "#bar form{display:flex;gap:12px;flex:1}\n"
                      "#txt{flex:1;background:rgba(6,10,37,0.9);color:var(--txt);border:1px solid var(--line);"
                      "border-radius:var(--radius);padding:13px 16px;font-family:inherit;font-size:15px;"
                      "letter-spacing:0.04em;resize:none;height:48px;overflow-y:auto;"
                      "outline:none;transition:all .2s ease}\n"
	                      "#txt:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(79,238,247,0.12)}\n"
	                      "#txt::placeholder{color:var(--faint)}\n"
	                      "a.btn{font-family:var(--mono);font-size:12.5px;letter-spacing:2px;color:var(--dim);"
	                      "text-decoration:none;background:transparent;border:2px solid rgba(227,231,240,0.25);"
	                      "border-radius:var(--radius);padding:9px 18px;cursor:pointer;transition:all .2s ease}\n"
	                      "a.btn:hover{color:var(--acc);border-color:var(--acc)}\n"
	                      "#send{font-family:var(--mono);font-size:12.5px;letter-spacing:2px;font-weight:700;"
	                      "color:var(--acc2);background:transparent;border:2px solid var(--acc2);"
	                      "border-radius:var(--radius);padding:0 26px;height:48px;cursor:pointer;"
	                      "transition:all .2s ease}\n"
                      "#send:hover:enabled{background:rgba(242,18,235,0.12);color:var(--txt)}\n"
                      "#send:disabled{opacity:0.45;cursor:default}\n"
                      ".chk{display:flex;align-items:center;gap:9px;font-family:var(--mono);font-size:12.5px;\n"
                      "letter-spacing:2px;text-transform:uppercase;color:var(--dim);cursor:pointer;\n"
                      "border:2px solid rgba(227,231,240,0.25);border-radius:var(--radius);padding:0 16px;\n"
                      "height:48px;user-select:none;white-space:nowrap;transition:all .2s ease}\n"
                      ".chk:hover{border-color:var(--acc);color:var(--acc)}\n"
                      ".chk input{width:15px;height:15px;accent-color:#4feef7;cursor:pointer;margin:0}\n"
                      ".chk.on{border-color:var(--acc);color:var(--acc)}\n"
                      ".dots{display:inline-block;margin-left:2px;white-space:nowrap}\n"
                      ".dots i{display:inline-block;width:.42em;height:.42em;border-radius:50%%;\n"
                      "background:var(--acc);margin-right:.16em;vertical-align:middle;\n"
                      "animation:pulse 1.3s ease-in-out infinite}\n"
                      ".dots i:nth-child(2){animation-delay:.17s}\n"
                      ".dots i:nth-child(3){animation-delay:.34s}\n"
                       "@keyframes pulse{0%%,100%%{opacity:.22;transform:translateY(0)}50%%{opacity:1;transform:translateY(-3px)}}\n");

	/* Sidecar-set background image: a SectorLogo script writes the image
	 * FILENAME into the microdb under __server_html_background_image;
	 * GET /background serves the file and this override paints it. No key
	 * (or an empty value) leaves the plain dark theme.
	 *
	 * Crossfade: the body itself stays plain. Two fixed layers paint the
	 * image -- #bgold from the BROWSER-CACHED /background (the previous
	 * turn's picture), #bgnew from a cache-busted fetch (the current one).
	 * The script fades #bgnew in over 4 s, so a background change dissolves
	 * instead of hard-cutting on the page reload. A session reset clears the
	 * key server-side, and the page comes back without either layer. */
	char background_name[2048];
	if (background_image_name(background_name, sizeof(background_name)))
		write_pos += sprintf(buf + write_pos,
		                     "#bgold,#bgnew{position:fixed;inset:0;z-index:-1;pointer-events:none;"
		                     "background-size:cover;background-position:center}\n"
		                     "#bgnew{opacity:0;transition:opacity 4s ease}\n"
		                     ".bhead{display:flex;align-items:center;gap:9px;margin-bottom:9px}\n"
		                     ".bhead .who{margin-bottom:0}\n"
		                     ".ava{width:32px;height:32px;border-radius:50%%;object-fit:cover;"
		                     "border:1px solid var(--line);flex:0 0 auto}\n");

	write_pos += sprintf(buf + write_pos,
	                     "</style></head><body>\n"
	                     "<div id=\"topbar\"></div>\n"
	                     "<div id=\"head\"><span class=\"logo\">v7</span>"
	                     "<span>Software Intelligence</span><span class=\"grow\"></span>"
	                     "<form action=\"/chat/reset\" method=\"post\" style=\"display:inline\">"
	                     "<button type=\"submit\" class=\"btn\" style=\"font-family:var(--mono);font-size:12.5px;"
	                     "letter-spacing:2px;color:var(--dim);background:transparent;border:2px solid rgba(227,231,240,0.25);"
	                     "border-radius:var(--radius);padding:9px 18px;cursor:pointer\">New chat</button></form></div>\n"
	                     "<div id=\"log\">\n");

	/* The crossfade layers, only when a background is set (same condition as
	 * the CSS above): #bgold paints the cached previous image, #bgnew is
	 * filled by the script and faded in over 4 s. */
	if (background_image_name(background_name, sizeof(background_name)))
		write_pos += sprintf(buf + write_pos, "<div id=\"bgold\"></div><div id=\"bgnew\"></div>\n");

	/* Render the FULL conversation history, oldest first. Pass 1 computes
	 * the exact worst-case rendered size of every turn (html_escape_buf can
	 * emit 6 bytes per source char, plus the markup chrome); pass 2 renders
	 * sequentially into that buffer -- no budget, no eviction, no dropped
	 * turns. */
	{
		size_t hist_need = 1; /* + the NUL */
		for (int i = 0; i < g_chat_count; i++)
			hist_need += (size_t)g_chat_history[i].len * 6 + 2048; /* 6x escape worst case + markup/think-block chrome */
		char *hist_buf = (char *)malloc(hist_need);
		if (!hist_buf)
		{
			printf("[v7_server] chat page: out of memory rendering the history\n");
			return 0;
		}
		int hw = 0;
		char avatar_name[2048];
		int has_avatar = avatar_image_name(avatar_name, sizeof(avatar_name)); /* constant for the whole page -- read once, used per bot turn */
		for (int i = 0; i < g_chat_count; i++)
		{
			chat_msg *turn = &g_chat_history[i];
			if (turn->role == 'u')
			{
				hw += sprintf(hist_buf + hw, "<div class=\"msg user\"><span class=\"who\">you</span>");
				hw += html_escape_buf(turn->text, turn->len, hist_buf + hw, (int)hist_need - hw);
				hw += sprintf(hist_buf + hw, "</div>\n");
			}
			else
			{
				/* When a sidecar set __server__chat_avatar_image, the bot's
				 * messages carry that persona picture next to the v7 label. */
				if (has_avatar)
					hw += sprintf(hist_buf + hw, "<div class=\"msg bot\"><div class=\"bhead\">"
					                              "<img class=\"ava\" src=\"/avatar\" alt=\"\">"
					                              "<span class=\"who\">v7</span></div>");
				else
					hw += sprintf(hist_buf + hw, "<div class=\"msg bot\"><span class=\"who\">v7</span>");
				hw += format_bot_text(turn->text, turn->len, hist_buf + hw, (int)hist_need - hw, turn->think);
				hw += sprintf(hist_buf + hw, "</div>\n");
			}
		}
		hist_buf[hw] = 0;
		if (!scratch_grow(&page, &page_cap, write_pos + hw + 1))
		{
			free(hist_buf);
			printf("[v7_server] chat page: out of memory rendering the history\n");
			return 0;
		}
		buf = page;
		buf_cap = page_cap;
		if (hw > 0)
		{
			memcpy(buf + write_pos, hist_buf, hw);
			write_pos += hw;
		}
		free(hist_buf);
	}

	/* RAG pipeline panel: a sidecar that plays the retrieval game publishes
	 * __server_rag_query / __server_rag_answer / __server_rag_injected per
	 * turn; this panel surfaces exactly what the gate saw and decided on the
	 * latest turn -- the QUERY / CORPUS ANSWER / INJECTED banners of the
	 * server window, in the browser, UNTRUNCATED (the full retrieval input
	 * and the full corpus answer -- a truncated display hides exactly the
	 * evidence a RAG debugging session needs). The page is grown to hold
	 * the full escaped values. No __server_rag_injected key = the loaded
	 * sidecar does not play the RAG game = no panel. An allocation failure
	 * DROPS THE PANEL and keeps the page: the turns are already rendered,
	 * and the panel is auxiliary debug surface. */
	{
		unsigned __int64 injected_len = 0;
		const char *injected_raw = (const char *)server_dll.v7_db_find("__server_rag_injected", &injected_len);
		if (injected_raw && injected_len > 0)
		{
			unsigned __int64 query_len_raw = 0;
			unsigned __int64 answer_len_raw = 0;
			const char *query_raw = (const char *)server_dll.v7_db_find("__server_rag_query", &query_len_raw);
			const char *answer_raw = (const char *)server_dll.v7_db_find("__server_rag_answer", &answer_len_raw);
			/* html_escape_buf emits at most 6 bytes per source char (&amp;);
			 * the escape buffers are sized EXACTLY for the full values so the
			 * panel can never truncate -- a fixed cap here once hid a bm25
			 * input-truncation bug behind a cut display. Malloc'd, not
			 * static: values are unbounded by policy. */
			char *rag_query_esc = (char *)malloc((size_t)query_len_raw * 6 + 16);
			char *rag_answer_esc = (char *)malloc((size_t)answer_len_raw * 6 + 16);
			int panel_grown = 0;
			if (rag_query_esc && rag_answer_esc)
			{
				html_escape_buf((char *)query_raw, query_raw ? (int)query_len_raw : 0,
				                rag_query_esc, (int)((size_t)query_len_raw * 6 + 16));
				html_escape_buf((char *)answer_raw, answer_raw ? (int)answer_len_raw : 0,
				                rag_answer_esc, (int)((size_t)answer_len_raw * 6 + 16));
				const char *decision = "gate not reached yet this turn";
				const char *decision_color = "var(--dim)";
				if (injected_raw[0] == '1')
				{
					decision = "facts injected into the thinking";
					decision_color = "var(--acc)";
				}
				else if (injected_raw[0] == '0')
				{
					decision = "no records matched; answered from the role's own knowledge";
				}
				int panel_need = write_pos + (int)strlen(rag_query_esc) + (int)strlen(rag_answer_esc) + 2048;
				panel_grown = scratch_grow(&page, &page_cap, panel_need) != 0;
				if (panel_grown)
				{
					buf = page;
					buf_cap = page_cap;
					write_pos += sprintf(buf + write_pos,
					                     "<div style=\"position:relative;z-index:2;margin:0 28px 10px 28px;padding:10px 16px;"
					                     "border-radius:var(--radius);border:1px solid rgba(242,18,235,0.28);background:rgba(14,6,30,0.90)\">"
					                     "<div onclick=\"var e=document.getElementById('ragbody');"
					                     "e.style.display=e.style.display==='none'?'block':'none';"
					                     "this.firstChild.nodeValue=e.style.display==='none'?'&#9654; rag pipeline':'&#9662; rag pipeline';\" "
					                     "style=\"cursor:pointer;font-family:var(--mono);font-size:12px;letter-spacing:2px;"
					                     "text-transform:uppercase;color:var(--acc2)\">&#9654; rag pipeline</div>"
					                     "<div id=\"ragbody\" style=\"display:none;margin-top:10px;font-family:var(--mono);font-size:12.5px;"
					                     "line-height:1.6;color:var(--dim);white-space:pre-wrap;word-wrap:break-word\">"
					                     "<span style=\"color:var(--faint);letter-spacing:2px\">decision:</span> "
					                     "<span style=\"color:%s\">%s</span>\n"
					                     "<span style=\"color:var(--faint);letter-spacing:2px\">query:</span>\n%s\n"
					                     "<span style=\"color:var(--faint);letter-spacing:2px\">corpus answer:</span>\n%s"
					                     "</div></div>\n",
					                     decision_color, decision, rag_query_esc, rag_answer_esc);
				}
			}
			if (!rag_query_esc || !rag_answer_esc || !panel_grown)
				printf("[v7_server] rag panel: out of memory -- the panel is skipped, the page keeps every turn\n");
			free(rag_query_esc);
			free(rag_answer_esc);
		}
	}

	if (!scratch_grow(&page, &page_cap, write_pos + 8192))
	{
		printf("[v7_server] chat page: out of memory writing the tail\n");
		return 0;
	}
	buf = page;
	buf_cap = page_cap;
	write_pos += sprintf(buf + write_pos, "</div>\n"
	                      "<div id=\"bar\">\n"
	                      "<form action=\"/chat/submit\" method=\"post\" onsubmit=\"return submitting()\">\n"
	                      "<label class=\"chk%s\" id=\"chklab\"><input type=\"checkbox\" id=\"think\" name=\"think\" value=\"1\"%s><span>thinking</span></label>\n"
	                      "<textarea id=\"txt\" name=\"q\" placeholder=\"ask anything...\" "
	                      "autocomplete=\"off\"></textarea>\n"
	                      "<button type=\"submit\" id=\"send\">Send</button>\n"
	                      "</form>\n"
	                      "</div>\n"
	                      "<script>\n"
	                      "var submitted=false;\n"
                      /* Auto-grow input: the box grows with its content instead of showing an
                       * inner scrollbar; past half the window height the box scrolls (a true
                       * paste-wall). */
                      "var box=document.getElementById('txt');\n"
                      "function growbox(){box.style.height='auto';box.style.height=Math.min(box.scrollHeight,Math.floor(window.innerHeight*0.5))+'px';}\n"
                      "box.addEventListener('input',growbox);\n"
	                      /* Optimistic submit: inference takes 30-60 s — without instant feedback
	                       * the page looks frozen and Enter appears dead. Echo the user bubble,
	                       * lock the inputs, flip the button, THEN let the form post normally;
	                       * the 303 redirect lands on a fresh page with the real turn. */
	                      "function submitting(){\n"
	                      "  if(submitted)return false;\n" /* a second Enter/click must not queue a duplicate inference */
	                      "  var t=document.getElementById('txt');\n"
	                      "  if(!t.value||!t.value.trim())return false;\n"
	                      "  submitted=true;\n"
	                      "  var l=document.getElementById('log');\n"
	                      "  var d=document.createElement('div');\n"
	                      "  d.className='msg user';\n"
	                      "  var w=document.createElement('span');\n" /* author label, matching the server-rendered bubbles */
	                      "  w.className='who';\n"
	                      "  w.textContent='you';\n"
	                      "  d.appendChild(w);\n"
	                      "  d.appendChild(document.createTextNode(t.value));\n" /* a text node, never innerHTML — model/user text can't inject markup */
	                      "  l.appendChild(d);\n"
	                      "  l.scrollTop=l.scrollHeight;\n"
	                      /* Inference takes 15-40 s with no output at all, so the log shows an
	                       * animated v7 bubble: the user sees the engine is alive and working
	                       * rather than staring at a frozen page. The label matches the
	                       * thinking checkbox -- an unchecked box means the turn is an ANSWER,
	                       * and calling it "reasoning" would be wrong. */
	                      "  var w2=document.createElement('div');\n"
	                      "  w2.className='msg bot';\n"
	                      "  var lab=document.createElement('span');\n"
	                      "  lab.className='who';\n"
	                      "  lab.textContent=document.getElementById('think').checked?'reasoning':'v7';\n"
	                      "  w2.appendChild(lab);\n"
	                      "  var sp=document.createElement('span');\n"
	                      "  sp.className='dots';\n"
	                      "  sp.innerHTML='<i></i><i></i><i></i>';\n" /* static markup only -- no user text reaches innerHTML */
	                      "  w2.appendChild(sp);\n"
	                      "  l.appendChild(w2);\n"
	                      "  l.scrollTop=l.scrollHeight;\n"
	                      "  var b=document.getElementById('send');\n"
                      "  b.disabled=true;b.textContent='thinking...';\n"
                      "  t.readOnly=true;\n"
                      /* Clear the box once the form data is serialized: the submit event handler
                       * runs BEFORE the browser builds the posted entry list, so the clear is
                       * deferred past the submission with a zero timeout. */
                      "  setTimeout(function(){t.value='';t.style.height='48px';},0);\n"
                      "  return true;\n"
	                      "}\n"
	                      "window.onload=function(){\n" /* after the 303 back to /: show the newest turn and focus the box */
	                      "  submitted=false;\n"
	                      "  var l=document.getElementById('log');l.scrollTop=l.scrollHeight;\n"
	                      "  var t=document.getElementById('txt');\n"
	                      "  growbox();\n" /* size the box for its initial (empty) content */
	                      "  t.focus();\n"
	                      "  var c=document.getElementById('think');\n"
	                      "  document.getElementById('chklab').className=c.checked?'chk on':'chk';\n"
	                      /* Background crossfade: /background is served immutable, every fetch
	                       * carries a unique ?t= buster, and the PREVIOUS turn's buster lives in
	                       * sessionStorage. bgold therefore paints the cached PREVIOUS image
	                       * instantly while bgnew fetches the current one and fades it in over
	                       * 4 s. On the very first backgrounded turn there is no previous buster,
	                       * so bgold stays empty and the image fades in over the plain theme.
	                       * The double rAF lets the opacity:0 start state commit before the
	                       * transition runs, so the fade actually plays. */
	                      "  var bo=document.getElementById('bgold');\n"
	                      "  var bn=document.getElementById('bgnew');\n"
	                      "  if(bo&&bn){var prev=sessionStorage.getItem('v7_bg_t');\n"
	                      "    var fresh=String(Date.now());\n"
	                      "    if(prev)bo.style.backgroundImage='url(\"/background?t='+prev+'\")';\n"
	                      "    bn.style.backgroundImage='url(\"/background?t='+fresh+'\")';\n"
	                      "    requestAnimationFrame(function(){requestAnimationFrame(function(){bn.style.opacity='1';});});\n"
	                      "    sessionStorage.setItem('v7_bg_t',fresh);\n"
	                      "  }\n"
	                      "};\n"
	                      /* bfcache: back-navigation restores the FROZEN page (readOnly inputs,
	                       * "thinking..." button, fake bubbles) without re-running onload.
	                       * Reload on restore so the page reflects the real server state. */
	                      "window.addEventListener('pageshow',function(e){\n"
	                      "  if(e.persisted)location.reload();\n"
	                      "});\n"
	                      "document.getElementById('think').onchange=function(){\n"
	                      "  document.getElementById('chklab').className=this.checked?'chk on':'chk';\n"
	                      "};\n"
	                      "document.getElementById('txt').addEventListener('keydown',function(e){\n" /* textarea: Enter sends, Shift+Enter makes a newline; isComposing keeps CJK IME confirmations from submitting half-typed text */
	                      "  if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();this.form.requestSubmit();}\n"
	                      "});\n"
	                      "</script>\n"
	                      "</body></html>\n",
	                      thinking_enabled ? " on" : "", thinking_enabled ? " checked" : "");
	buf[write_pos] = 0;
	if (out_len)
		*out_len = write_pos;
	return buf;
}

