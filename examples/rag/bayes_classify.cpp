// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/**
 * @file bayes_classify.cpp -- multinomial naive bayes two-class text classifier.
 *
 * built for the bayesian-augmented RAG experiments: classify a model think
 * text against two example corpora (e.g. "confused" vs "sure" reasoning)
 * with ONE word on stdout -- the class name. sidecar-friendly:
 *
 *   bayes_classify.exe -build <case0_file> <case1_file> <state_file>
 *       count words per class, write the model state.
 *   bayes_classify.exe -classify <state_file> <text>
 *       print "class0" or "class1" (argmax; tie -> class0).
 *
 * commands also work without the dash (v7_execute passes exe + 3 params:
 * classify is exactly 3, so the hot path fits the FFI directly).
 *
 * output contract (v7_execute capture boundary): classify stdout carries
 * ONLY the class word -- anything else would land inside the LLM think
 * stream. errors go to stderr with a nonzero exit (sidecar sees empty
 * stdout and skips). usage/help goes to stdout for interactive users.
 *
 * math: multinomial NB with laplace (+1) smoothing, log space, uniform
 * priors (the caller controls class balance by sizing the corpora).
 * out-of-vocabulary tokens at classify time are skipped (standard).
 *
 * state file format (plain text, human-inspectable):
 *   bayesnb v1
 *   total0 <sum of class0 counts>   total1 <sum of class1 counts>
 *   vocab <V>
 *   <word> <count0> <count1>        (one per line)
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cstdarg>
#include <cmath>
#include <string>
#include <map>
#include <windows.h>

typedef std::map<std::string, __int64> word_map;

/* classify-path diagnostics: v7_execute captures stdout AND stderr into the
 * LLM think stream, so a diagnostic may go to NEITHER stream -- it goes to
 * bayes_classify.log next to the exe (same contract as bm25_rag's ragg.log).
 * -build is interactive and keeps its printf diagnostics. */
void bayes_diag(const char *fmt, ...)
{
	char exe_path[MAX_PATH];
	GetModuleFileNameA(0, exe_path, sizeof exe_path);
	std::string dir(exe_path);
	int slash = (int)dir.find_last_of("\\/");
	std::string log_path = (slash < 0) ? std::string("bayes_classify.log") : dir.substr(0, slash + 1) + "bayes_classify.log";
	FILE *f = fopen(log_path.c_str(), "a");
	if (!f)
		return;
	va_list args;
	va_start(args, fmt);
	vfprintf(f, fmt, args);
	va_end(args);
	fclose(f);
}

void tokenize(const char *text, word_map &out)
{
	std::string cur;

	for (const char *p = text;; p++)
	{
		unsigned char c = (unsigned char)*p;
		if (isalnum(c))
		{
			cur += (char)tolower(c);
			continue;
		}
		if (!cur.empty())
		{
			if (cur.size() >= 2)
				out[cur]++;
			cur.clear();
		}
		if (!c)
			break;
	}
}

bool ReadFile(char *path, std::string &out)
{
	FILE *f = fopen(path, "rb");
	if (!f)
		return false;

	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (n < 0)
	{
		fclose(f);
		return false;
	}
	out.resize((int)n);
	if ((n > 0) && (fread(&out[0], 1, (int)n, f) != (int)n))
	{
		fclose(f);
		return false;
	}
	fclose(f);
	return true;
}

int cmd_build(char *case0_path, char *case1_path, char *state_path)
{
	std::string text0;
	std::string text1;

	if (!ReadFile(case0_path, text0))
	{
		printf("[bayes] cannot read case0 '%s'\n", case0_path);
		return 1;
	}
	if (!ReadFile(case1_path, text1))
	{
		printf("[bayes] cannot read case1 '%s'\n", case1_path);
		return 1;
	}

	word_map counts0;
	word_map counts1;

	tokenize(text0.c_str(), counts0);
	tokenize(text1.c_str(), counts1);

	word_map vocab;

	for (word_map::iterator it = counts0.begin(); it != counts0.end(); ++it)
		vocab[it->first] = 0;
	for (word_map::iterator it = counts1.begin(); it != counts1.end(); ++it)
		vocab[it->first] = 0;

	__int64 total0 = 0;
	__int64 total1 = 0;

	for (word_map::iterator it = counts0.begin(); it != counts0.end(); ++it)
		total0 += it->second;
	for (word_map::iterator it = counts1.begin(); it != counts1.end(); ++it)
		total1 += it->second;

	FILE *f = fopen(state_path, "wb");
	if (!f)
	{
		printf("[bayes] cannot write state '%s'\n", state_path);
		return 1;
	}
	fprintf(f, "bayesnb v1\n");
	fprintf(f, "total0 %lld\ntotal1 %lld\n", total0, total1);
	fprintf(f, "vocab %u\n", (unsigned)vocab.size());

	for (word_map::iterator it = vocab.begin(); it != vocab.end(); ++it)
	{
		word_map::iterator a = counts0.find(it->first);
		word_map::iterator b = counts1.find(it->first);
		fprintf(f, "%s %lld %lld\n", it->first.c_str(), (a != counts0.end()) ? a->second : 0, (b != counts1.end()) ? b->second : 0);
	}
	fclose(f);
	printf("[bayes] built '%s': vocab=%u total0=%lld total1=%lld\n", state_path, (unsigned)vocab.size(), total0, total1);
	return 0;
}

int cmd_classify(char *state_path, char *text)
{
	std::string s;

	if (!ReadFile(state_path, s))
	{
		bayes_diag("[bayes] cannot read state '%s'\n", state_path);
		return 1;
	}

	__int64 total0 = 0;
	__int64 total1 = 0;
	__int64 vocab_count = 0;
	word_map w0;
	word_map w1;

	{
		FILE *f = fopen(state_path, "rb");
		if (!f)
		{
			bayes_diag("[bayes] cannot open state '%s'\n", state_path);
			return 1;
		}
		char line[512];

		if ((!fgets(line, sizeof line, f)) || (strncmp(line, "bayesnb", 7) != 0))
		{
			fclose(f);
			bayes_diag("[bayes] '%s' is not a bayesnb state file\n", state_path);
			return 1;
		}
		char dummy[32];
		__int64 v = 0;

		if (fscanf(f, "%31s %lld", dummy, &total0) != 2)
		{
			fclose(f);
			bayes_diag("[bayes] state '%s' is truncated (total0)\n", state_path);
			return 1;
		}
		if (fscanf(f, "%31s %lld", dummy, &total1) != 2)
		{
			fclose(f);
			bayes_diag("[bayes] state '%s' is truncated (total1)\n", state_path);
			return 1;
		}
		if (fscanf(f, "%31s %lld", dummy, &v) != 2)
		{
			fclose(f);
			bayes_diag("[bayes] state '%s' is truncated (vocab)\n", state_path);
			return 1;
		}
		vocab_count = v;

		char word[480];
		__int64 a;
		__int64 b;

		while (fscanf(f, "%479s %lld %lld", word, &a, &b) == 3)
		{
			if (a > 0)
				w0[word] = a;
			if (b > 0)
				w1[word] = b;
		}
		fclose(f);
	}
	if (vocab_count <= 0)
	{
		printf("[bayes] state has empty vocabulary\n");
		return 1;
	}

	word_map text_counts;

	tokenize(text, text_counts);

	double den0 = (double)total0 + (double)vocab_count + 1.0;
	double den1 = (double)total1 + (double)vocab_count + 1.0;
	double score0 = 0.0;
	double score1 = 0.0;
	__int64 matched_tokens = 0;

	for (word_map::iterator it = text_counts.begin(); it != text_counts.end(); ++it)
	{
		word_map::iterator a = w0.find(it->first);
		word_map::iterator b = w1.find(it->first);
		if ((a == w0.end()) && (b == w1.end()))
			continue;
		matched_tokens += it->second;
		double p0 = (a != w0.end() ? (double)a->second : 0.0) + 1.0;
		double p1 = (b != w1.end() ? (double)b->second : 0.0) + 1.0;
		score0 += (double)it->second * log(p0 / den0);
		score1 += (double)it->second * log(p1 / den1);
	}
	if (matched_tokens == 0)
	{
		printf("class0\n");
		bayes_diag("[bayes] no vocabulary overlap -- tie -> class0 (exit 2)\n");
		return 2;
	}

	if (score1 > score0)
		printf("class1\n");
	else
		printf("class0\n");
	bayes_diag("[bayes] s0=%.3f s1=%.3f matched=%lld -> %s\n", score0, score1, matched_tokens, (score1 > score0) ? "class1" : "class0");
	return 0;
}

void print_help(void)
{
	printf("bayes_classify.exe -- two-class naive bayes text classifier\n"
	       "\n"
	       "USAGE\n"
	       "  bayes_classify.exe -build <case0_file> <case1_file> <state_file>\n"
	       "  bayes_classify.exe -classify <state_file> <text>\n"
	       "  (commands also work without the leading dash)\n"
	       "\n"
	       "COMMANDS\n"
	       "  -build    read two plain-text example corpora (case0 = class 0\n"
	       "            examples, case1 = class 1 examples), count words per\n"
	       "            class, and write the model to <state_file> (plain text,\n"
	       "            human-inspectable: totals, vocabulary, per-word counts).\n"
	       "  -classify tokenize <text>, score it against both classes with\n"
	       "            laplace-smoothed multinomial naive bayes (log space,\n"
	       "            uniform priors), and print the winning class name:\n"
	       "            \"class0\" or \"class1\" (tie -> class0) on stdout.\n"
	       "\n"
	       "OUTPUT CONTRACT\n"
	       "  stdout carries ONLY the class word (sidecar capture boundary: this\n"
	       "  tool's stdout may be spliced into an LLM think stream). build\n"
	       "  progress and score details go to stderr. exit codes: 0 = decided,\n"
	       "  2 = decided but NO vocabulary overlap (no signal), 1 = error.\n"
	       "\n"
	       "MODEL\n"
	       "  P(w|c) = (count(w,c) + 1) / (total(c) + V + 1)   (laplace +1)\n"
	       "  score(c) = sum over text tokens found in vocab of\n"
	       "             count(w,text) * ln P(w|c)\n"
	       "  out-of-vocabulary tokens are skipped. class balance is controlled\n"
	       "  by the relative sizes of the two example files.\n"
	       "\n"
	       "EXAMPLES\n"
	       "  bayes_classify.exe -build confused.txt sure.txt bayes_state.txt\n"
	       "  bayes_classify.exe -classify bayes_state.txt \"hmm not sure, maybe?\"\n"
	       "\n");
}

int main(int argc, char **argv)
{
	setvbuf(stdout, 0, _IONBF, 0);

	char *cmd = (argc >= 2) ? argv[1] : 0;

	if (cmd && (strcmp(cmd, "-build") == 0))
	{
		if (argc != 5)
		{
			print_help();
			return 1;
		}
		return cmd_build(argv[2], argv[3], argv[4]);
	}
	if (cmd && (strcmp(cmd, "build") == 0))
	{
		if (argc != 5)
		{
			print_help();
			return 1;
		}
		return cmd_build(argv[2], argv[3], argv[4]);
	}
	if (cmd && (strcmp(cmd, "-classify") == 0))
	{
		if (argc != 4)
		{
			print_help();
			return 1;
		}
		return cmd_classify(argv[2], argv[3]);
	}
	if (cmd && (strcmp(cmd, "classify") == 0))
	{
		if (argc != 4)
		{
			print_help();
			return 1;
		}
		return cmd_classify(argv[2], argv[3]);
	}
	print_help();
	return 1;
}
