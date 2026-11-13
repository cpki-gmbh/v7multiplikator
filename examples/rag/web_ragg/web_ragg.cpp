// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/**
 * @file web_ragg.cpp
 * @brief internet RAG: web search -> fetch first N pages -> v7 distillation
 *        into one ~100-word paragraph. augments the agent's memories with
 *        current web knowledge.
 *
 * CLI (v7_execute-compatible: max 3 args after the exe):
 *   web_ragg.exe search "<query>" ["pages=5;words=100;maxgen=250;fresh"]
 *       duck_duck_go search -> fetch first N result pages -> MMR extractive
 *       compression per page -> v7 distills ONE paragraph (~100 words).
 *       prints ONLY the paragraph to stdout.
 *
 *   web_ragg.exe makequery "<think and memories text>"
 *       distills the think-tag contents (+ appended memory-RAG lines) into
 *       ONE short web search query (3-8 words). prints ONLY the query.
 *
 * options (search, semicolon-separated, bm25_rag style):
 *   pages=N    pages to grab (1-10, default 5)
 *   words=N    target paragraph length in words (default 100)
 *   maxgen=N   max LLM generation tokens (10-500, default 250)
 *   fresh      bypass the page cache, re-download
 *
 * v7 DLL and model database paths come from web_ragg.ini next to the exe:
 *   [v7]
 *   dll=bin\v7_threading.dll
 *   db=weights\databases\qwen3_0.6b\lz4ht.db
 * no silent defaults: missing INI or missing key is a usage error.
 *
 * distillation prompt overrides for tuning (also from the INI):
 *   [search]  prepend=...  append=...   (files: prepend_file= / append_file=)
 *   [makequery] prepend_file=...
 *
 * diagnostics go to web_ragg.log next to the exe -- NEVER to stderr/stdout:
 * when called from a sector_logo sidecar via C-CALL v7_execute, stderr is
 * merged into the captured buffer and would poison the LLM stream.
 */

#define V7_LIB_IMPLEMENTATION
#include "v7_lib.h"

#include "webtool.h"
#include "search.h"
#include "http_client.h"
#include "text_shortener.h"
#include "simpledb.h"

#include <cstdio>
#include <cstdarg>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <io.h>
#include <fcntl.h>
#include <string>
#include <vector>
#include <windows.h>

/* ---- logging (file only, see header comment) ---- */

FILE *log_file = 0;

void log_line(char *fmt, ...)
{
	if (!log_file)
		return;
	va_list args;
	va_start(args, fmt);
	vfprintf(log_file, fmt, args);
	va_end(args);
	fputc('\n', log_file);
	fflush(log_file);
}

/* ---- file I/O ---- */

bool ReadFile(std::string &path, std::string &out)
{
	FILE *f = fopen(path.c_str(), "rb");
	if (!f)
		return false;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	out.resize(n);
	if (n > 0 && fread(&out[0], 1, n, f) != (int)n)
	{
		fclose(f);
		return false;
	}
	fclose(f);
	return true;
}

/* ---- LLM output post-processing ---- */

/** strip a residual think block, surrounding quotes, and whitespace. */
std::string clean_llm_output(std::string &s)
{
	std::string r = s;
	int close = r.find("</think>");
	if (close != std::string::npos)
		r = r.substr(close + 8);
	int open = r.find("<think>");
	if (open != std::string::npos)
		r.erase(open, 7);
	int b = r.find_first_not_of(" \t\r\n\"'");
	int e = r.find_last_not_of(" \t\r\n\"'");
	if (b == std::string::npos)
		return "";
	std::string t = r.substr(b, e - b + 1);
	/* single trailing newline max */
	for (int i = 0; i + 1 < t.size(); i++)
		if (t[i] == '\n' && t[i + 1] == '\n')
			t.erase(i + 1, 1), i = 0;
	return t;
}

/* ---- v7 distillation ---- */

typedef struct distill_ctx
{
	struct 	v7_lib dll;
} distill_ctx;

bool distill_init(distill_ctx &d, char *dll_path, char *db_path, int maxgen)
{
	memset(&d, 0, sizeof(d));
	if (v7_load(dll_path, &d.dll) != 0)
	{
		log_line("[FATAL] cannot load '%s'", dll_path);
		return false;
	}
	if (!d.dll.v7_init(db_path))
	{
		if (d.dll.v7_engine_is_dead())
			log_line("[FATAL] v7_init died for '%s': %s", db_path, d.dll.v7_get_death_reason());
		else
			log_line("[FATAL] v7_init failed for '%s'", db_path);
		v7_unload(&d.dll);
		return false;
	}
	/* Sec.172: a gated entry -- a caught FATAL also returns 0, so the real return
	 * AND the death flag must both be checked. */
	if ((!d.dll.v7_initialize_inference()) || d.dll.v7_engine_is_dead())
	{
		if (d.dll.v7_engine_is_dead())
			log_line("[FATAL] inference init died: %s", d.dll.v7_get_death_reason());
		else
			log_line("[FATAL] v7_initialize_inference failed");
		v7_unload(&d.dll);
		return false;
	}
	d.dll.v7_set_quantize(1);
	d.dll.v7_set_max_gen(maxgen);
	d.dll.v7_enable_thinking(0);
	return true;
}

void distill_shutdown(distill_ctx &d)
{
	d.dll.v7_shutdown();     /* join workers BEFORE the heap teardown + FreeLibrary */
	d.dll.v7_free_inference();
	v7_unload(&d.dll);
}

/** run one distillation: prompt in, cleaned text out. empty string = failure. */
std::string distill_run(distill_ctx &d, std::string &prompt)
{
	char output[32768];
	memset(output, 0, sizeof(output));
	d.dll.v7_chat((char *)prompt.c_str(), 0, output, sizeof(output), 1);
	if (d.dll.v7_engine_is_dead())
	{
		log_line("[FATAL] engine died during distillation: %s", d.dll.v7_get_death_reason());
		return "";
	}
	std::string output_str = output;
	return clean_llm_output(output_str);
}

/* ---- page cache (simpledb, next to the exe) ---- */

struct sdb database_ptr;
bool database_is_open = false;

std::string exe_dir()
{
	char exe_path[MAX_PATH];
	GetModuleFileNameA(0, exe_path, sizeof(exe_path));
	std::string d = exe_path;
	int pos = d.rfind('\\');
	if (pos != std::string::npos)
		d = d.substr(0, pos);
	return d;
}

bool page_cache_get(std::string &key, std::string &out)
{
	if (!database_is_open)
	{
		std::string db_path = exe_dir() + "\\web_ragg.db";
		sdb_open(&database_ptr, (char *)db_path.c_str());
		database_is_open = true;
	}
	char *val = 0;
	int len = 0;
	if (sdb_get(&database_ptr, (char *)key.c_str(), &val, &len) == SDB_OK && val)
	{
		out.assign(val, val + len);
		free(val);
		return true;
	}
	return false;
}

void page_cache_put(std::string &key, std::string &val)
{
	if (!database_is_open)
	{
		std::string db_path = exe_dir() + "\\web_ragg.db";
		sdb_open(&database_ptr, (char *)db_path.c_str());
		database_is_open = true;
	}
	sdb_put(&database_ptr, (char *)key.c_str(), (char *)val.c_str(), (int)val.size());
	sdb_save(&database_ptr);
}

std::string url_to_key(std::string &url)
{
	std::string k;
	for (int i = 0; i < url.size() && k.size() < 120; i++)
	{
		char c = url[i];
		if (isalnum((unsigned char)c))
			k += c;
		else if (c != 0)
			k += '_';
	}
	return "page:" + k;
}

/* ---- search command ---- */

/* returns the distilled paragraph in *out ("" = nothing retrieved).
 * does NOT print to stdout: main() prints after stdout is restored, so the
 * DLL's internal prints (db_load etc.) can never leak into the result.
 * opts = semicolon-separated "key=value" list, bm25_rag style. */
int cmd_search(char *dll_path, char *db_path, std::string &query, std::string &opts, std::string *out)
{
	int pages = 5, words = 100, maxgen = 250, fresh = 0;
	std::string fprepend, fappend, ini = exe_dir() + "\\web_ragg.ini";
	char buf[512];

	/* options string: pages=N;words=N;maxgen=N;fresh */
	std::string rest = opts;
	while (!rest.empty())
	{
		int semi = rest.find(';');
		std::string one = (semi == std::string::npos) ? rest : rest.substr(0, semi);
		rest = (semi == std::string::npos) ? "" : rest.substr(semi + 1);
		if (one.rfind("pages=", 0) == 0)
			pages = atoi(one.c_str() + 6);
		else if (one.rfind("words=", 0) == 0)
			words = atoi(one.c_str() + 6);
		else if (one.rfind("maxgen=", 0) == 0)
			maxgen = atoi(one.c_str() + 7);
		else if (one == "fresh")
			fresh = 1;
	}
	/* prompt overrides from the INI (tuning hooks) */
	if (GetPrivateProfileStringA("search", "PrependFile", "", buf, sizeof(buf), ini.c_str()) && buf[0])
	{
		std::string path = buf;
		if (!ReadFile(path, fprepend))
		{
			log_line("[FATAL] cannot read prepend_file %s", buf);
			return 1;
		}
	}
	if (GetPrivateProfileStringA("search", "AppendFile", "", buf, sizeof(buf), ini.c_str()) && buf[0])
	{
		std::string path = buf;
		if (!ReadFile(path, fappend))
		{
			log_line("[FATAL] cannot read append_file %s", buf);
			return 1;
		}
	}
	if (GetPrivateProfileStringA("search", "Prepend", "", buf, sizeof(buf), ini.c_str()) && buf[0])
		fprepend = buf;
	if (GetPrivateProfileStringA("search", "Append", "", buf, sizeof(buf), ini.c_str()) && buf[0])
		fappend = buf;

	if (pages < 1)
		pages = 1;
	if (pages > 10)
		pages = 10;
	if (words < 30)
		words = 30;
	if (words > 300)
		words = 300;
	if (maxgen < 10)
		maxgen = 10;
	if (maxgen > 500)
		maxgen = 500;

	log_line("search q=\"%s\" pages=%d words=%d maxgen=%d fresh=%d", query.c_str(), pages, words, maxgen, fresh);

	/* 1. web search — with a generic recency prior: the current month/year
	 * is appended to the query so results skew toward fresh pages. this is
	 * content-agnostic (no intent classification, works for any topic). */
	SYSTEMTIME st;
	GetLocalTime(&st);
	char *MON[] = { "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" };
	std::string dated_query = query + " " + MON[(st.wMonth >= 1 && st.wMonth <= 12) ? st.wMonth - 1 : 0] + " " + std::to_string(st.wYear);
	std::vector<search_result> results = search(dated_query, pages * 2);
	log_line("search dated q=\"%s\" results=%d", dated_query.c_str(), (int)results.size());
	if (results.empty())
	{
		log_line("search no results");
		return 0;
	}

	/* 2. fetch first N usable pages, extract main content */
	std::vector<std::string> sources;
	for (int i = 0; i < results.size() && (int)sources.size() < pages; i++)
	{
		std::string &url = results[i].url;
		if (url.find("http") != 0)
			continue;

		std::string html;
		if (!fresh && page_cache_get(url_to_key(url), html))
		{
			log_line("page %d cached %s", (int)i, url.c_str());
		}
		else
		{
			http_response resp = curlGet(url, 20);
			html = resp.body;
			log_line("page %d fetch status=%d bytes=%d url=%s", (int)i, resp.statusCode, (int)html.size(), url.c_str());
			if (html.empty())
				continue;
			page_cache_put(url_to_key(url), html);
		}
		if (html.empty())
			continue;

		page_info info = parsePage(html, url);
		if (info.content.size() < 200)
		{
			log_line("page %d too thin (%d chars), skipped", (int)i, (int)info.content.size());
			continue;
		}
		/* MMR extractive compression: query-relevant passages per page */
		std::string excerpt = shortenTextQuery(info.content, 1500, query);
		sources.push_back(excerpt);
	}
	log_line("usable sources=%d", (int)sources.size());
	if (sources.empty())
	{
		log_line("search no usable pages");
		return 0;
	}

	/* 3. build distillation prompt (no meta-language: the model must never
	 * mention pages, excerpts, sources, queries, or the task itself) */
	std::string prepend = fprepend.empty() ? "You are a research assistant. the texts below are from web pages about one topic. "
	                                         "Write ONE paragraph of about 100 words summarizing the current situation around the topic. "
	                                         "Keep concrete facts: numbers, names, dates, places. write only the paragraph itself - "
	                                         "no introduction, no heading, no list, no quotation marks, and never mention pages, "
	                                         "texts, sources, or this task.\n\nTopic: " +
	                                             query + "\n\n"
	                                       : fprepend;
	std::string append = fappend.empty() ? "\n\nWrite the paragraph now:" : fappend;

	std::string body;
	for (int i = 0; i < sources.size(); i++)
	{
		body += "[source ";
		char num[8];
		_itoa_s((int)(i + 1), num, 10);
		body += num;
		body += "]\n";
		body += sources[i];
		body += "\n\n";
	}

	/* 4. v7 distillation */
	distill_ctx d;
	if (!distill_init(d, dll_path, db_path, maxgen))
		return 1;
	std::string para = distill_run(d, prepend + body + append);
	distill_shutdown(d);

	log_line("distilled %d chars: %.200s", (int)para.size(), para.c_str());
	if (para.empty())
	{
		log_line("distillation empty");
		return 0;
	}
	*out = para;
	return 0;
}

/* ---- makequery command ---- */

int cmd_makequery(char *dll_path, char *db_path, std::string &think, std::string *out)
{
	if (think.size() < 40)
	{
		log_line("makequery input too short (%d chars)", (int)think.size());
		return 0;
	}
	/* cap the think text: the query needs the TOPIC, not the whole reasoning */
	std::string input = think.size() > 4000 ? think.substr(0, 4000) : think;

	std::string prompt = "You write web search queries. the text below is a language model's reasoning about a question, "
	                     "possibly with some reference notes appended. identify the main factual topic it needs current "
	                     "information about, and write ONE short web search query (3 to 8 words) that would find up-to-date "
	                     "information about that topic. output ONLY the query words - no quotes, no explanation, no punctuation "
	                     "at the end.\n\nText: >>>\n" +
	                     input + "\n<<<\n\nThe search query:";
	distill_ctx d;
	if (!distill_init(d, dll_path, db_path, 60))
		return 1;
	std::string q = distill_run(d, prompt);
	distill_shutdown(d);

	/* collapse to single line, drop query punctuation residue */
	std::string clean;
	for (int i = 0; i < q.size(); i++)
	{
		char c = q[i];
		if (c == '\n' || c == '\r' || c == '\t')
			c = ' ';
		if (strchr(".,;:!?", c))
			continue;
		if (clean.size() && clean[0] == ' ' && isspace((unsigned char)c))
			continue;
		clean += c;
	}
	int b = clean.find_first_not_of(' ');
	int e = clean.find_last_not_of(' ');
	if (b == std::string::npos)
	{
		log_line("makequery empty");
		return 0;
	}
	clean = clean.substr(b, e - b + 1);
	/* hard cap: first 12 words */
	int wc = 0;
	for (int i = 0; i < clean.size(); i++)
		if (clean[i] == ' ' && ++wc >= 12)
		{
			clean = clean.substr(0, i);
			break;
		}

	log_line("makequery -> \"%s\"", clean.c_str());
	*out = clean;
	return 0;
}

/* ---- main ---- */

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);
	log_file = fopen((exe_dir() + "\\web_ragg.log").c_str(), "a");

	if (argc < 3)
	{
		printf("usage: web_ragg.exe search \"<query>\" [\"pages=5;words=100;maxgen=250;fresh\"]\n"
		       "       web_ragg.exe makequery \"<think and memories text>\"\n"
		       "v7 paths come from web_ragg.ini next to the exe ([v7] dll= / db=)\n");
		if (log_file)
			fclose(log_file);
		return 1;
	}

	/* v7 paths from the INI -- no silent defaults (missing = usage error) */
	char dll_buf[MAX_PATH], db_buf[MAX_PATH];
	std::string ini = exe_dir() + "\\web_ragg.ini";
	if (!GetPrivateProfileStringA("v7", "Dll", "", dll_buf, sizeof(dll_buf), ini.c_str()) || !dll_buf[0])
	{
		printf("[FATAL] web_ragg.ini missing or lacks [v7] dll= key\n");
		return 1;
	}
	if (!GetPrivateProfileStringA("v7", "Db", "", db_buf, sizeof(db_buf), ini.c_str()) || !db_buf[0])
	{
		printf("[FATAL] web_ragg.ini missing or lacks [v7] db= key\n");
		if (log_file)
			fclose(log_file);
		return 1;
	}

	std::string cmd = argv[1];
	std::string input = argv[2];
	std::string opts = (argc >= 4) ? argv[3] : "";

	/* redirect stdout to the log during ALL work: the DLL prints internal
	 * diagnostics (db_load etc.) to stdout, which must never leak into the
	 * captured sidecar output. the result string is printed after restore.
	 * The redirect happens REGARDLESS of whether the log fopen succeeded:
	 * without it, a failed log open would let the DLL's prints poison the
	 * sidecar capture -- exactly what this block exists to prevent. */
	std::string result;
	int saved_stdout = _dup(1);
	if (saved_stdout >= 0)
	{
		if (log_file)
			fclose(log_file);
		freopen((exe_dir() + "\\web_ragg.log").c_str(), "a", stdout);
		log_file = stdout;
	}

	int rc;
	if (cmd == "makequery")
		rc = cmd_makequery(dll_buf, db_buf, input, &result);
	else if (cmd == "search")
		rc = cmd_search(dll_buf, db_buf, input, opts, &result);
	else
	{
		printf("unknown command '%s' (search|makequery)\n", cmd.c_str());
		rc = 1;
	}

	/* restore the real stdout, print ONLY the result */
	if (saved_stdout >= 0)
	{
		_dup2(saved_stdout, 1);
		_close(saved_stdout);
		log_file = 0; /* log handle was stdout's; do not close it twice */
	}
	if (rc == 0 && !result.empty())
		printf("%s", result.c_str());

	if (database_is_open)
		sdb_close(&database_ptr);
	return rc;
}
