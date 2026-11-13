// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

// text_shortener.cpp - extractive summarization via maximal marginal relevance
// MMR = lambda*sim(query,sentence) - (1-lambda)*max sim(sentence,picked)
// combines query relevance, lead/position bias, fact-density (numbers), and diversity.
#include "text_shortener.h"
#include "text_summarizer.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cmath>
#include <set>
#include <string>
#include <vector>

summarizer_context summarizer_instance = 0;
shortener_mode summarizer_mode = SHORTENER_SIMPLE;

void shortener_set_mode(shortener_mode mode)
{
	summarizer_mode = mode;
}
shortener_mode shortener_get_mode()
{
	return summarizer_mode;
}

int shortener_init(char *dll, char *ini)
{
	if (summarizer_instance)
	{
		summarizer_shutdown(summarizer_instance);
		summarizer_instance = 0;
	}
	summarizer_mode = SHORTENER_SIMPLE;
	(void)dll;
	(void)ini;
	return 0;
}
void shortener_shutdown()
{
	if (summarizer_instance)
	{
		summarizer_shutdown(summarizer_instance);
		summarizer_instance = 0;
	}
}

// ---- minimal helpers ----
std::vector<std::string> split_sentences(std::string &t)
{
	std::vector<std::string> out;
	std::string cur;
	for (int i = 0; i < t.size(); i++)
	{
		cur += t[i];
		char c = t[i];
		bool end = (c == '.' || c == '!' || c == '?') && (i + 1 >= t.size() || isspace((unsigned char)t[i + 1]) || t[i + 1] == '\n');
		bool nl2 = (c == '\n') && i + 1 < t.size() && t[i + 1] == '\n';
		if ((end || nl2) && cur.size() >= 25)
		{
			out.push_back(cur);
			cur.clear();
		}
	}
	if (cur.size() >= 20)
		out.push_back(cur);
	return out;
}

std::vector<std::string> words(std::string &t)
{
	std::vector<std::string> w;
	std::string c;
	for (char ch : t)
	{
		if (isalnum((unsigned char)ch))
			c += (char)tolower((unsigned char)ch);
		else if (!c.empty())
		{
			w.push_back(c);
			c.clear();
		}
	}
	if (!c.empty())
		w.push_back(c);
	return w;
}

bool stopword(std::string &w)
{
	char *sw[] = { "the",  "a",    "an",   "and",  "or",   "but",  "in",    "on",    "at",     "to",    "for",   "of",    "with", "by",    "from",    "as",     "is",    "are",     "was",  "were", "be",  "been", "being", "have",
		           "has",  "had",  "do",   "does", "did",  "will", "would", "could", "should", "may",   "might", "can",   "it",   "its",   "this",    "that",   "these", "those",   "i",    "you",  "he",  "she",  "they",  "we",
		           "not",  "no",   "so",   "if",   "than", "then", "also",  "very",  "just",   "about", "after", "again", "all",  "any",   "because", "before", "below", "between", "both", "each", "few", "more", "most",  "other",
		           "some", "such", "only", "own",  "same", "too",  "under", "up",    "while",  "which", "who",   "what",  "when", "where", "how",     "their",  "there", "here",    "them", "his",  "her", "our",  "your" };
	static std::set<std::string> s;              // built once (function-local persistent state -- called per word per sentence)
	if (s.empty())
		for (int i = 0; i < sizeof(sw) / sizeof(sw[0]); i++)
			s.insert(sw[i]);
	return s.count(w) > 0;
}

std::set<std::string> sig_words(std::string &t)
{
	std::set<std::string> r;
	for (auto &w : words(t))
		if (!stopword(w) && w.size() > 2)
			r.insert(w);
	return r;
}

double jaccard(std::set<std::string> &a, std::set<std::string> &b)
{
	if (a.empty() && b.empty())
		return 0.0;
	int inter = 0;
	for (auto &w : a)
		if (b.count(w))
			inter++;
	int uni = (int)a.size() + (int)b.size() - inter;
	if (uni)
		return (double)inter / uni;
	return 0.0;
}

bool has_number(std::string &s)
{
	for (char c : s)
		if (isdigit((unsigned char)c))
			return true;
	return false;
}

// MMR-based query-focused extractive selection
std::string mmr_select(std::string &content, int maxChars, std::string &query)
{
	auto sents = split_sentences(content);
	if (sents.empty())
		return content.substr(0, maxChars);

	auto qsig = sig_words(query);
	if (qsig.empty())
	{
		// no query signal: lead-bias (first sentences win)
		std::string r;
		for (auto &s : sents)
		{
			if ((int)(r.size() + s.size()) > maxChars)
				break;
			r += s;
		}
		return r;
	}

	int N = sents.size();
	std::vector<std::set<std::string>> ssig(N);
	std::vector<double> base(N, 0.0);
	for (int i = 0; i < N; i++)
	{
		ssig[i] = sig_words(sents[i]);
		// query relevance: fraction of query terms present
		int hit = 0;
		for (auto &w : qsig)
			if (ssig[i].count(w))
				hit++;
		double qrel = (double)hit / qsig.size();
		// position (lead) bonus: fades over first ~20 sentences
		double pos = std::max(0.0, 1.0 - (double)i / 20.0); // fact density: numbers/dates are fact-bearing
		double fact = has_number(sents[i]) ? 0.3 : 0.0;
		// length penalty: avoid boilerplate stubs and giant run-ons
		double len = (double)sents[i].size();
		double lpen = (len < 40) ? len / 40.0 : (len > 400 ? 400 / len : 1.0);
		base[i] = (1.0 * qrel + 0.25 * pos + fact) * lpen;
	}

	// MMR greedy selection
	double LAMBDA = 0.6;
	std::vector<bool> used(N, false);
	std::vector<int> picked;
	int total = 0;
	for (int round = 0; round < (int)N; round++)
	{
		int best = -1;
		double best_score = -1e9;
		for (int i = 0; i < N; i++)
		{
			if (used[i])
				continue;
			double sim_picked = 0.0;
			for (auto p : picked)
				sim_picked = std::max(sim_picked, jaccard(ssig[i], ssig[p]));
			double mmr = LAMBDA * base[i] - (1.0 - LAMBDA) * sim_picked;
			if (mmr > best_score)
			{
				best_score = mmr;
				best = (int)i;
			}
		}
		if (best < 0 || best_score <= 0.0)
			break;
		if (total + sents[best].size() > (int)maxChars)
			break;
		used[best] = true;
		picked.push_back(best);
		total += sents[best].size();
	}

	if (picked.empty())
	{
		// fallback: first sentences
		std::string r;
		for (auto &s : sents)
		{
			if ((int)(r.size() + s.size()) > maxChars)
				break;
			r += s;
		}
		return r;
	}

	// output in ORIGINAL document order (better reading flow for the model)
	std::sort(picked.begin(), picked.end());
	std::string r;
	for (auto i : picked)
	{
		r += sents[i];
		if (r.back() != '\n')
			r += " ";
	}
	if (content.size() > total + 50)
		r += "[..]";
	return r;
}

std::string shortenText(std::string &t, int m)
{
	if ((int)t.size() <= m)
		return t;
	std::string empty_query;
	return mmr_select(t, m, empty_query);
}
std::string shortenTextSmart(std::string &t, int m)
{
	return shortenText(t, m);
}
std::string shortenTextQuery(std::string &t, int m, std::string &q)
{
	if ((int)t.size() <= m)
		return t;
	return mmr_select(t, m, q);
}
