// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/**
 * @file llm_text_processor.cpp
 * @brief v7-LLM text reformulation tool (BM25 retrieval + LLM rewriting).
 *
 * modes:
 *   (default)     BM25 search → LLM reformulation
 *   -direct       skip BM25, use -finputtext as the source text → LLM
 *   -only-bm25    skip LLM, print BM25 results plain (no formatting)
 *                 with -direct, print prepend+inputtext+append plain
 *   -distill      memory-distillation mode (replaces the old mem_bridge.exe):
 *                 -direct with stdout discipline -- ALL diagnostics go to
 *                 stderr, stdout carries ONLY the LLM output. this is the
 *                 output contract a SectorLogo DEF_EXE_FN caller needs:
 *                 v7_execute_fmt captures stdout+stderr into the LLM
 *                 thought stream, so a stray diagnostic line on stdout
 *                 would poison the stream. use with -finputtext (the
 *                 sidecar writes the turn text there first via
 *                 v7s_writefile) and -fprepend/-fappend (the distillation
 *                 prompt files).
 *
 * usage:
 *   llm_text_processor.exe <dll_path> <db_path> <corpus_path> "<query>" [options]
 *
 * positional:
 *   dll_path      v7_threading.dll
 *   db_path       lz4ht.db
 *   corpus_path   text corpus for BM25 (unused with -direct+no BM25)
 *   query         BM25 search query (unused with -direct)
 *
 * options:
 *   -maxlines N   BM25 chunks to retrieve (1-20, default 5)
 *   -k1 F         BM25 k1 term frequency saturation (default 1.2)
 *   -b F          BM25 b length normalization (default 0.75)
 *   -topic F      topic-paragraph boost (default 1.15)
 *   -cov F        coverage weight base (default 0.3)
 *   -maxgen N     max LLM generation tokens (10-500, default 200)
 *   -bigrams 0|1  use bigram features (default 1)
 *   -prompt S     built-in prompt style: default, strict, simple
 *   -direct       skip BM25, use -finputtext as source text
 *   -only-bm25    skip LLM, print raw BM25/plain text to stdout
 *   -distill      stdout discipline for DEF_EXE_FN callers (see above)
 *   -sweep        print machine-readable tab-separated sweep line
 *
 * file overrides (when given, these override built-in prompt / corpus behavior):
 *   -finputtext F  read source text from file F (with -direct: used instead of BM25)
 *   -fprepend F    read front prompt from file F (overrides -prompt)
 *   -fappend F    read back prompt from file F (overrides -prompt)
 *   -foutputfile F write thoughts to file F (in addition to stdout)
 */

#define V7_LIB_IMPLEMENTATION
#include "../../src/v7_lib.h"

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <windows.h>

typedef std::vector<std::string> str_vec;

/* ---- tunable parameters (set from CLI) ---- */
double bigram_k1 = 1.2;
double bigram_b = 0.75;
double topic_threshold = 1.15;
double coverage_threshold = 0.3;
int use_bigrams = 1;
int max_generation_tokens = 200;
int sweep_mode = 0;
int direct_mode = 0;
int g_only_bm25 = 0;
int g_distill_mode = 0;
char *g_prompt_style = "default";
std::string g_finputtext;
std::string g_fprepend;
std::string g_fappend;
std::string g_foutputfile;

/* ---- diagnostics ----
 * -distill output contract: v7_execute_fmt captures stdout AND stderr into
 * the LLM thought stream, so in distill mode a diagnostic may go to NEITHER
 * stream -- it goes to llm_text_processor.log next to the exe instead
 * (same contract as bm25_rag's ragg.log). in every other mode diag() is a
 * plain printf. */
void diag(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	if (g_distill_mode)
	{
		char exe_path[MAX_PATH];
		GetModuleFileNameA(0, exe_path, sizeof exe_path);
		std::string dir(exe_path);
		int slash = dir.find_last_of("\\/");
		FILE *f = fopen(((slash < 0) ? std::string("llm_text_processor.log") : dir.substr(0, slash + 1) + "llm_text_processor.log").c_str(), "a");
		if (f)
		{
			vfprintf(f, fmt, args);
			fclose(f);
		}
	}
	else
	{
		vprintf(fmt, args);
	}
	va_end(args);
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

bool WriteFile(std::string &path, std::string &data)
{
	FILE *f = fopen(path.c_str(), "wb");
	if (!f)
		return false;
	if (!data.empty() && fwrite(data.c_str(), 1, data.size(), f) != data.size())
	{
		fclose(f);
		return false;
	}
	fclose(f);
	return true;
}

std::string snippet(std::string &text, int max_chars)
{
	std::string t = text.substr(0, max_chars);
	for (int i = 0; i < t.size(); i++)
		if ((unsigned char)t[i] < ' ')
			t[i] = ' ';
	return t;
}

/* ---- BM25 tokenizer ---- */

char *STOP[] = {
	"the",  "a",    "an",    "and",    "or",      "of",     "to",    "in",    "is",   "are",  "was",  "were", "it",    "its",     "that", "this",  "with",  "for",  "as",    "on",    "at",    "by",     "be",      "been",  "they",    "their",
	"them", "he",   "she",   "his",    "her",     "we",     "you",   "i",     "not",  "but",  "from", "has",  "have",  "had",     "do",   "does",  "did",   "will", "would", "can",   "could", "should", "than",    "then",  "so",      "such",
	"into", "over", "after", "before", "between", "during", "above", "below", "up",   "down", "out",  "off",  "again", "further", "once", "here",  "there", "when", "where", "why",   "how",   "all",    "any",     "both",  "each",    "few",
	"more", "most", "other", "some",   "no",      "nor",    "only",  "own",   "same", "too",  "very", "just", "also",  "about",   "what", "which", "who",   "whom", "these", "those", "am",    "if",     "because", "while", "against", "under",
};

bool is_stop(std::string &w)
{
	for (int i = 0; i < sizeof STOP / sizeof STOP[0]; i++)
		if (w == STOP[i])
			return true;
	return false;
}

str_vec tokenize(std::string &text)
{
	str_vec words, out;
	std::string cur;
	for (int i = 0; i <= text.size(); i++)
	{
		unsigned char c = (i < text.size()) ? text[i] : '\0';
		if (isalnum(c))
			cur += (char)tolower(c);
		else if (!cur.empty())
		{
			if (cur.size() > 1 && !is_stop(cur))
				words.push_back(cur);
			cur.clear();
		}
	}
	for (int i = 0; i < words.size(); i++)
	{
		out.push_back(words[i]);
		if (use_bigrams && i + 1 < words.size())
			out.push_back(words[i] + "_" + words[i + 1]);
	}
	return out;
}

/* ---- chunking ---- */

#define MAXSENT 280
#define MINSENT 20

typedef struct chunk
{
	std::string text;
	str_vec terms;
	double score;
	bool topic;
} chunk;

void push_sentence(std::vector<chunk> &chunks, std::string s)
{
	int b = s.find_first_not_of(" \t\r\n");
	if (b == std::string::npos)
		return;
	int e = s.find_last_not_of(" \t\r\n");
	s = s.substr(b, e - b + 1);

	if (!chunks.empty() && s.size() < MINSENT)
	{
		chunks.back().text += " " + s;
		return;
	}
	while (s.size() > MAXSENT)
	{
		int cut = s.rfind(' ', MAXSENT);
		if (cut == std::string::npos || cut < MINSENT)
			cut = MAXSENT;
		chunk c;
		c.text = s.substr(0, cut);
		c.score = 0;
		c.topic = false;
		chunks.push_back(c);
		s = s.substr(cut);
		int nb = s.find_first_not_of(" \t\r\n");
		if (nb == std::string::npos)
			return;
		s = s.substr(nb);
	}
	chunk c;
	c.text = s;
	c.score = 0;
	c.topic = false;
	chunks.push_back(c);
}

std::vector<chunk> chunk_text(std::string &text)
{
	std::vector<chunk> chunks;
	int pos = 0;
	while (pos < text.size())
	{
		int nl = text.find("\n\n", pos);
		int end = (nl == std::string::npos) ? text.size() : nl;
		std::string para = text.substr(pos, end - pos);
		pos = (nl == std::string::npos) ? text.size() : nl + 2;

		int before = chunks.size();
		std::string cur;
		for (int i = 0; i < para.size(); i++)
		{
			char c = para[i];
			if (c == '\n' || c == '\r')
				c = ' ';
			cur += c;
			if ((c == '.' || c == '!' || c == '?') && (i + 1 >= para.size() || isspace((unsigned char)para[i + 1])))
			{
				push_sentence(chunks, cur);
				cur.clear();
			}
		}
		if (!cur.empty())
			push_sentence(chunks, cur);
		if (chunks.size() > before)
			chunks[before].topic = true;
	}
	for (int i = 0; i < chunks.size(); i++)
		chunks[i].terms = tokenize(chunks[i].text);
	return chunks;
}

/* ---- BM25 ranking (uses global bigram_k1, bigram_b, topic_threshold, coverage_threshold) ---- */

typedef struct ranked
{
	int idx;
	double score;
	bool operator<(ranked &o)
	{
		return score > o.score;
	}
} ranked;

std::vector<ranked> rank_chunks(std::vector<chunk> &chunks, str_vec &qterms)
{
	int num_chunks = chunks.size();
	std::map<std::string, int> df;
	for (int i = 0; i < num_chunks; i++)
	{
		std::map<std::string, char> seen;
		for (int j = 0; j < chunks[i].terms.size(); j++)
			seen[chunks[i].terms[j]] = 1;
		for (std::map<std::string, char>::iterator it = seen.begin(); it != seen.end(); ++it)
			df[it->first]++;
	}
	double avglen = 0;
	for (int i = 0; i < num_chunks; i++)
		avglen += (double)chunks[i].terms.size();
	if (num_chunks > 0)
		avglen /= (double)num_chunks;

	std::map<std::string, char> quni;
	for (int q = 0; q < qterms.size(); q++)
		if (qterms[q].find('_') == std::string::npos)
			quni[qterms[q]] = 1;

	for (int i = 0; i < num_chunks; i++)
	{
		std::map<std::string, int> tf;
		for (int j = 0; j < chunks[i].terms.size(); j++)
			tf[chunks[i].terms[j]]++;
		std::map<std::string, char> hits;
		double s = 0;
		for (int q = 0; q < qterms.size(); q++)
		{
			std::map<std::string, int>::iterator it = tf.find(qterms[q]);
			if (it == tf.end())
				continue;
			int doc_freq = df[qterms[q]];
			double idf = log(1.0 + ((double)num_chunks - (double)doc_freq + 0.5) / ((double)doc_freq + 0.5));
			double lenn = (avglen > 0) ? (double)chunks[i].terms.size() / avglen : 1.0;
			double tfc = (double)it->second * (bigram_k1 + 1.0) / ((double)it->second + bigram_k1 * (1.0 - bigram_b + bigram_b * lenn));
			s += idf * tfc;
			if (qterms[q].find('_') == std::string::npos)
				hits[qterms[q]] = 1;
		}
		double cov = quni.empty() ? 1.0 : (double)hits.size() / (double)quni.size();
		s *= coverage_threshold + (1.0 - coverage_threshold) * cov;
		if (chunks[i].topic)
			s *= topic_threshold;
		chunks[i].score = s;
	}
	std::vector<ranked> r(num_chunks);
	for (int i = 0; i < num_chunks; i++)
	{
		r[i].idx = i;
		r[i].score = chunks[i].score;
	}
	std::sort(r.begin(), r.end());
	return r;
}

/* ---- BM25 search: return top-N chunk text ---- */

std::string bm25_search(std::string &corpus_text, std::string &query, int maxlines)
{
	std::vector<chunk> chunks = chunk_text(corpus_text);
	if (chunks.empty())
		return "";

	str_vec qterms = tokenize(query);
	if (qterms.empty())
		return "";

	std::vector<ranked> ranked = rank_chunks(chunks, qterms);

	std::vector<int> pool;
	for (int i = 0; i < ranked.size() && pool.size() < 20; i++)
		if (ranked[i].score > 0)
			pool.push_back(ranked[i].idx);
	if (pool.empty())
		return "";

	int want = (int)maxlines;
	if (want > pool.size())
		want = pool.size();

	std::string out;
	for (int i = 0; i < want; i++)
	{
		if (!out.empty())
			out += "\n";
		out += chunks[pool[i]].text;
	}
	diag("[bm25] top score=%.2f, %d chunks retrieved for \"%s\"\n", ranked[0].score, want, snippet(query, 60).c_str());
	return out;
}

/* ---- prompt builders ---- */

std::string get_prepend()
{
	if (!g_fprepend.empty())
		return g_fprepend;

	if (strcmp(g_prompt_style, "strict") == 0)
		return "You must reformulate the following text into exactly 3 simple sentences. "
		       "Write ONLY the 3 sentences separated by periods. no introduction, no list, no extra text.\n\n"
		       "Text: >>>";

	if (strcmp(g_prompt_style, "simple") == 0)
		return "Summarize this into 3 short sentences: >>>";

	return "reformulate the following text into 3 logical simple thoughts, "
	       "write these as sentences seperated by \".\" do not write a list, "
	       "just normal text:>>>";
}

std::string get_append()
{
	if (!g_fappend.empty())
		return g_fappend;

	if (strcmp(g_prompt_style, "strict") == 0)
		return "<<<";

	if (strcmp(g_prompt_style, "simple") == 0)
		return "<<<";

	return "<<<:reformulate the text into 3 logical simple thoughts, "
	       "write these as sentences seperated by \".\" do not write a list, "
	       "just normal text";
}

std::string build_prompt(std::string &source_text)
{
	return get_prepend() + source_text + get_append();
}

/* ---- CLI parser ---- */

int find_flag(int argc, char **argv, char *flag)
{
	for (int i = 1; i < argc; i++)
		if (strcmp(argv[i], flag) == 0)
			return i;
	return 0;
}

char *find_flag_str(int argc, char **argv, char *flag, char *def)
{
	int idx = find_flag(argc, argv, flag);
	if (idx > 0 && idx + 1 < argc)
		return argv[idx + 1];
	return def;
}

double find_flag_double(int argc, char **argv, char *flag, double def)
{
	int idx = find_flag(argc, argv, flag);
	if (idx > 0 && idx + 1 < argc)
		return atof(argv[idx + 1]);
	return def;
}

int find_flag_int(int argc, char **argv, char *flag, int def)
{
	int idx = find_flag(argc, argv, flag);
	if (idx > 0 && idx + 1 < argc)
		return atoi(argv[idx + 1]);
	return def;
}

std::string find_flag_file(int argc, char **argv, char *flag)
{
	char *path = find_flag_str(argc, argv, flag, 0);
	if (!path)
		return "";
	std::string content;
	std::string path_str(path);
	if (!ReadFile(path_str, content))
	{
		diag("[FATAL] cannot read file for %s: '%s'\n", flag, path);
		exit(1);
	}
	diag("[%s] loaded %u bytes from %s\n", flag, (unsigned)content.size(), path);
	return content;
}

/* ---- main ---- */

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	if (argc < 3)
	{
		printf("llm_text_processor.exe -- v7-LLM text reformulation (BM25 retrieval + LLM rewriting)\n"
		       "\n"
		       "usage: llm_text_processor.exe <dll_path> <db_path> [corpus_path] [query] [options]\n"
		       "\n"
		       "Positional:\n"
		       "  dll_path        v7_threading.dll\n"
		       "  db_path         lz4ht.db\n"
		       "  corpus_path     text corpus for BM25 (optional with -direct)\n"
		       "  query           BM25 search query (optional with -direct)\n"
		       "\n"
		       "BM25 tuning:\n"
		       "  -maxlines N     BM25 chunks to retrieve (1-20, default 5)\n"
		       "  -k1 F           BM25 k1 term frequency saturation (default 1.2)\n"
		       "  -b F            BM25 b length normalization (default 0.75)\n"
		       "  -topic F        topic-paragraph boost (default 1.15)\n"
		       "  -cov F          coverage weight base (default 0.3)\n"
		       "  -bigrams 0|1    use bigram features (default 1)\n"
		       "\n"
		       "LLM tuning:\n"
		       "  -maxgen N       max LLM generation tokens (10-500, default 200)\n"
		       "  -prompt S       built-in prompt style: default, strict, simple\n"
		       "\n"
		       "Mode:\n"
		       "  -direct         skip BM25, use -finputtext as source text\n"
		       "  -only-bm25      skip LLM, print BM25/plain text to stdout\n"
		       "  -distill        memory-distillation mode: -direct + TOTAL diagnostic\n"
		       "                  silence (stdout AND stderr carry nothing but the\n"
		       "                  LLM output -- v7_execute_fmt captures both streams\n"
		       "                  into the LLM thought stream, so a single stray\n"
		       "                  diagnostic byte would poison the stream)\n"
		       "  -sweep          print machine-readable tab-separated sweep line\n"
		       "\n"
		       "File overrides (override built-in prompts / output):\n"
		       "  -finputtext F   read source text from file F\n"
		       "  -fprepend F     read front prompt from file F\n"
		       "  -fappend F      read back prompt from file F\n"
		       "  -foutputfile F  write thoughts to file F\n");
		return 1;
	}

	char *dll_path = argv[1];
	char *db_path = argv[2];
	char *corpus_path = (argc >= 4) ? argv[3] : "";
	char *query = (argc >= 5) ? argv[4] : "";

	int maxlines = find_flag_int(argc, argv, "-maxlines", 5);
	if (maxlines < 1)
		maxlines = 1;
	if (maxlines > 20)
		maxlines = 20;

	bigram_k1 = find_flag_double(argc, argv, "-k1", 1.2);
	bigram_b = find_flag_double(argc, argv, "-b", 0.75);
	topic_threshold = find_flag_double(argc, argv, "-topic", 1.15);
	coverage_threshold = find_flag_double(argc, argv, "-cov", 0.3);
	max_generation_tokens = find_flag_int(argc, argv, "-maxgen", 200);
	use_bigrams = find_flag_int(argc, argv, "-bigrams", 1);
	sweep_mode = find_flag(argc, argv, "-sweep") ? 1 : 0;
	direct_mode = find_flag(argc, argv, "-direct") ? 1 : 0;
	g_only_bm25 = find_flag(argc, argv, "-only-bm25") ? 1 : 0;
	g_distill_mode = find_flag(argc, argv, "-distill") ? 1 : 0;
	g_prompt_style = find_flag_str(argc, argv, "-prompt", "default");

	/* -distill implies -direct (memory distillation never does BM25) */
	if (g_distill_mode)
		direct_mode = 1;

	if (max_generation_tokens < 10)
		max_generation_tokens = 10;
	if (max_generation_tokens > 500)
		max_generation_tokens = 500;

	/* load file overrides */
	g_finputtext = find_flag_file(argc, argv, "-finputtext");
	g_fprepend = find_flag_file(argc, argv, "-fprepend");
	g_fappend = find_flag_file(argc, argv, "-fappend");
	char *outpath = find_flag_str(argc, argv, "-foutputfile", 0);
	if (outpath)
		g_foutputfile = outpath;

	/* ---- step 1: get the source text ---- */

	std::string source_text;

	if (direct_mode)
	{
		/* -direct: use -finputtext as source text, no BM25 */
		if (g_finputtext.empty())
		{
			diag("[FATAL] -direct requires -finputtext\n");
			return 1;
		}
		source_text = g_finputtext;
		diag("[direct] using input text (%u bytes)\n", (unsigned)source_text.size());
	}
	else
	{
		/* normal: BM25 search on corpus */
		if (corpus_path[0] == '\0' || query[0] == '\0')
		{
			diag("[FATAL] need corpus_path and query (or use -direct -finputtext)\n");
			return 1;
		}
		std::string corpus_path_str(corpus_path);
		std::string corpus_text;
		if (!ReadFile(corpus_path_str, corpus_text))
		{
			diag("[FATAL] cannot read corpus '%s'\n", corpus_path);
			return 1;
		}
		diag("[corpus] %s: %u bytes\n", corpus_path, (unsigned)corpus_text.size());

		std::string query_str(query);
		source_text = bm25_search(corpus_text, query_str, maxlines);
		if (source_text.empty())
		{
			diag("[bm25] no results for query \"%s\"\n", query);
			return 0;
		}
		diag("[bm25] raw output (%u bytes):\n%s\n", (unsigned)source_text.size(), snippet(source_text, 500).c_str());
	}

	/* ---- step 2: if -only-bm25, print plain and exit ---- */

	if (g_only_bm25)
	{
		if (direct_mode)
		{
			/* with -direct: print prepend+inputtext+append as plain text */
			std::string plain = get_prepend() + source_text + get_append();
			printf("%s", plain.c_str());
		}
		else
		{
			/* print BM25 results with zero formatting */
			printf("%s", source_text.c_str());
		}
		if (!g_foutputfile.empty())
		{
			if (direct_mode)
				WriteFile(g_foutputfile, get_prepend() + source_text + get_append());
			else
				WriteFile(g_foutputfile, source_text);
		}
		return 0;
	}

	/* ---- step 3: build the full prompt ---- */

	std::string prompt = build_prompt(source_text);

	/* ---- step 4: load v7 and run inference ---- */

	struct v7_lib dll;
	if (v7_load(dll_path, &dll) != 0)
	{
		diag("[FATAL] cannot load '%s'\n", dll_path);
		return 1;
	}
	diag("[v7] loaded %s\n", dll_path);

	if (!dll.v7_init(db_path))
	{
		if (dll.v7_engine_is_dead())
			diag("[FATAL] v7_init died for '%s': %s\n", db_path, dll.v7_get_death_reason());
		else
			diag("[FATAL] v7_init failed for '%s'\n", db_path);
		v7_unload(&dll);
		return 1;
	}
	diag("[v7] database loaded: %s\n", db_path);

	/* Sec.172: a gated entry -- a caught FATAL also returns 0, so the real return
	 * AND the death flag must both be checked. */
	if ((!dll.v7_initialize_inference()) || dll.v7_engine_is_dead())
	{
		if (dll.v7_engine_is_dead())
			diag("[FATAL] inference init died: %s\n", dll.v7_get_death_reason());
		else
			diag("[FATAL] v7_initialize_inference failed\n");
		v7_unload(&dll);
		return 1;
	}
	dll.v7_set_quantize(1);
	dll.v7_set_max_gen(max_generation_tokens);
	diag("[v7] inference context created (Q8, maxgen=%d)\n", max_generation_tokens);

	dll.v7_enable_thinking(0);

	/* step 5: chat with the reformulation prompt */
	char output[32768];
	memset(output, 0, sizeof(output));

	diag("[v7] prompt style=%s, %u chars: %s\n", g_prompt_style, (unsigned)prompt.size(), snippet(prompt, 200).c_str());
	diag("[v7] running v7s_chat...\n");

	dll.v7_chat((char *)prompt.c_str(), 0, output, sizeof(output), 1);
	if (dll.v7_engine_is_dead())
	{
		diag("[FATAL] engine died during chat: %s\n", dll.v7_get_death_reason());
		dll.v7_shutdown();
		dll.v7_free_inference();
		v7_unload(&dll);
		return 1;
	}

	/* step 6: print result. -distill: stdout AND stderr are captured by
	 * v7_execute_fmt into the LLM thought stream, so the ONLY thing this
	 * process may emit is the LLM output itself; diagnostics went to the
	 * log file via diag(). */
	if (g_distill_mode)
	{
		diag("[distill] prompt %u chars, output %u chars\n", (unsigned)prompt.size(), (unsigned)strlen(output));
		printf("%s", output);
	}
	else if (sweep_mode)
	{
		printf("Sweep\tmaxlines=%d\tk1=%.2f\tb=%.2f\ttopic=%.2f\tcov=%.2f\tmaxgen=%d\tbigrams=%d\tprompt=%s\tdirect=%d\toutput=%s\n", maxlines, bigram_k1, bigram_b, topic_threshold, coverage_threshold, max_generation_tokens, use_bigrams,
		       g_prompt_style, direct_mode, output);
	}
	else
	{
		printf("... query: %s ...\n", query);
		printf("... params: maxlines=%d k1=%.2f b=%.2f topic=%.2f cov=%.2f maxgen=%d bigrams=%d prompt=%s direct=%d ...\n", maxlines, bigram_k1, bigram_b, topic_threshold, coverage_threshold, max_generation_tokens, use_bigrams, g_prompt_style,
		       direct_mode);
		printf("... source ...\n%s\n", snippet(source_text, 800).c_str());
		printf("... thoughts ...\n%s\n", output);
	}

	/* write to output file if requested */
	if (!g_foutputfile.empty())
	{
		std::string output_str(output);
		if (WriteFile(g_foutputfile, output_str))
			diag("[output] wrote %u bytes to %s\n", (unsigned)strlen(output), g_foutputfile.c_str());
		else
			diag("[FATAL] cannot write to '%s'\n", g_foutputfile.c_str());
	}

	/* cleanup */
	dll.v7_shutdown();
	dll.v7_free_inference();
	v7_unload(&dll);
	return 0;
}
