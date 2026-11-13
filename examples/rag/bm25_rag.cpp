// v7 - part of the v7 inference engine
// Copyright (C) 2026 Görg Pflug & CPKI GmbH
// License: GPL v3 (see LICENSE)

/**
 * @file bm25_rag.cpp
 * @brief production BM25 retrieval tool for qwen3 thought-stream injection.
 *
 * the production tool distilled from RAG research experiments 1-17
 * (see docs/research_logbook/rag_research_log.md and
 * docs/radispatch_development_research_journey.md):
 *
 *   bm25_rag.exe search "<query>" "<opts>"
 *       opts is ONE packed string (v7_execute passes exe + 3 params only):
 *       "rand=N;maxlines=N;minscore=S;topmin=S;base=PATH" -- any subset.
 *   bm25_rag.exe add "<text>"
 *
 * options:
 *   maxlines N   server-side line limit (default 3). exp 15: injecting more
 *                lines than the answer has slots invites slot-stuffing
 *                (unused RAG lines become hedged fake reasons).
 *   minscore S   answer-presence floor, part 1: score DENSITY (topscore /
 *                distinct query unigrams). protects small corpora from
 *                query-length inflation (quantum think vs cat KB: absolute
 *                topscore 2.6 but 0.021/term). default 0 = off.
 *   topmin S     answer-presence floor, part 2: absolute TOPSCORE. protects
 *                long thinks on large corpora where density collapses for
 *                true hits (david/goliath: density 0.45, topscore 34).
 *                injection passes if density>=minscore OR topscore>=topmin
 *                (exp 17, law 9: no single length-normalized metric
 *                separates hit from miss across query AND corpus regimes).
 *   base=PATH    knowledge base override -- read this file instead of
 *                ragbigram_base.txt next to the exe. removes the need to copy
 *                the exe per corpus.
 *
 * FLOOR CALIBRATION GUIDE (measured, exp T7/17): calibrate BOTH floors with
 * real think-length probes (500-2500 chars, not short questions): collect
 * density+topscore for 2-3 known hits and 2-3 known misses, set each floor
 * between its hit/miss cluster. cat KB: minscore=0.026, topmin off.
 * bible KB (31k chunks): minscore=6.0, topmin=25. known limit (law 10):
 * when the model prior is WRONG, its think text poisons the query -- the
 * expansion that helps correct thinkers retrieves confirming chunks for
 * wrong ones. first-sentence querying is immune but needs its own floor
 * regime (short queries scale differently).
 *
 * OUTPUT CONTRACT: stdout carries ONLY the result lines (or nothing = below
 * floor / no match). v7_execute merges stderr INTO the captured buffer, so
 * every diagnostic goes to ragg.log next to the exe -- a single stderr byte
 * printed during a v7_execute call lands inside the LLM thought stream
 * (measured: a score line caused a "catfish" hallucination, exp T7 run 1).
 *
 * inherited from ragg.cpp: BM25 unigram+bigram ranking, distinct-term
 * coverage multiplier, topic-sentence bonus, near-duplicate suppression,
 * rank-weighted random pick mode.
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <limits.h>
#include <time.h>
#endif

typedef std::vector<std::string> str_vec;

/** @brief directory of the exe, with trailing separator. all data files
 *         (ragbigram_base.txt, ragg_corpus.txt, ragg.log) resolve relative to
 *         this, so a copied exe serves its own experiment folder. the
 *         program is SINGLE-INSTANCE by usage (one server, sequential
 *         sidecar calls): the log is plain append, no interprocess locking. */
std::string exe_dir()
{
#ifdef _WIN32
	char buf[MAX_PATH];
	DWORD n = GetModuleFileNameA(0, buf, MAX_PATH);
	if (n == 0 || n >= MAX_PATH)
	{
		buf[0] = 0; /* unreadable or truncated: an empty dir lets every file open fail visibly instead of resolving into garbage */
		return "";
	}
	buf[MAX_PATH - 1] = 0;
#else
	char buf[PATH_MAX];
	ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
	if (n <= 0)
	{
		buf[0] = 0; /* unreadable: an empty dir lets every file open fail visibly instead of resolving into garbage */
		return "";
	}
	buf[n] = 0;
#endif
	std::string p(buf);
	std::string::size_type slash = p.find_last_of("\\/");
	if (slash == std::string::npos)
		return "";
	return p.substr(0, slash + 1);
}

/** @brief read a whole file into @p out. missing/short reads return false;
 *         an empty corpus is not an error (search uses what exists). */
bool read_whole_file(std::string path, std::string &out)
{
	FILE *f = fopen(path.c_str(), "rb");
	if (!f)
		return false;
	if (fseek(f, 0, SEEK_END) != 0)
	{
		fclose(f);
		return false;
	}
	long n = ftell(f);
	if (n < 0)
	{
		fclose(f);
		return false; /* ftell failed: resize(-1) would throw */
	}
	if (fseek(f, 0, SEEK_SET) != 0)
	{
		fclose(f);
		return false;
	}
	/* read into a TEMPORARY and swap only on complete success: a partial
	 * read must never leave half a corpus in @p out for a caller that
	 * ignores the return value */
	std::string tmp;
	tmp.resize(n);
	if (n > 0 && fread(&tmp[0], 1, n, f) != (size_t)n)
	{
		fclose(f);
		return false;
	}
	fclose(f);
	/* strip \r so the \n\n paragraph detection and the newline flush work
	 * no matter whether the file was written in text mode (add/remember
	 * write \n through a text-mode handle: the file holds \r\n) */
	std::string flat;
	flat.reserve(tmp.size());
	for (size_t i = 0; i < tmp.size(); i++)
		if (tmp[i] != '\r')
			flat += tmp[i];
	/* strip a UTF-8 BOM so the first token of a Notepad-saved corpus stays whole */
	if (flat.size() >= 3 && (unsigned char)flat[0] == 0xEF && (unsigned char)flat[1] == 0xBB && (unsigned char)flat[2] == 0xBF)
		flat.erase(0, 3);
	out.swap(flat);
	return true;
}

/** @brief append one timestamped entry to ragg.log next to the exe. */
void log_line(std::string dir, std::string msg)
{
	FILE *f = fopen((dir + "ragg.log").c_str(), "a");
	if (!f)
		return;
	time_t now = time(0);
	char stamp[32];
	stamp[0] = 0;
	const tm *lt = localtime(&now);
	if (lt)
		strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", lt);
	fprintf(f, "[%s] %s\n", stamp, msg.c_str());
	fclose(f);
}

/** @brief first @p n chars of @p s with control chars flattened to spaces. */
std::string snippet(std::string text, int max_chars)
{
	std::string t = text.substr(0, max_chars);
	for (int i = 0; i < t.size(); i++)
		if ((unsigned char)t[i] < ' ')
			t[i] = ' ';
	return t;
}

/* ---- tokenize: lowercase alnum runs, stopwords dropped --------------------
 * not/no/nor are deliberately NOT stopwords: "does not like" and "likes"
 * must stay distinguishable, or the memory NOOP check silently drops
 * corrections of stored facts. */
const char *STOP[] = {
	"the",  "a",    "an",    "and",    "or",      "of",     "to",    "in",    "is",   "are",  "was",  "were", "it",    "its",     "that", "this",  "with",  "for",  "as",    "on",    "at",    "by",     "be",      "been",  "they",    "their",
	"them", "he",   "she",   "his",    "her",     "we",     "you",   "i",     "but",  "from", "has",  "have",  "had",     "do",   "does",  "did",   "will", "would", "can",   "could", "should", "than",    "then",  "so",      "such",
	"into", "over", "after", "before", "between", "during", "above", "below", "up",   "down", "out",  "off",  "again", "further", "once", "here",  "there", "when", "where", "why",   "how",   "all",    "any",     "both",  "each",    "few",
	"more", "most", "other", "some",   "only",    "own",    "same",  "too",   "very", "just", "also",  "about",   "what", "which", "who",   "whom", "these", "those", "am",    "if",     "because", "while", "against", "under",
};

bool is_stop(std::string w)
{
	static std::set<std::string> stopset;
	if (stopset.empty())
		for (int i = 0; i < sizeof STOP / sizeof STOP[0]; i++)
			stopset.insert(STOP[i]);
	return stopset.count(w) > 0;
}

/** @brief expand English negation contractions BEFORE tokenization: don't ->
 *         do not, can't -> cannot, won't -> will not, and every other n't
 *         form. without this, "doesn't like" loses its negation ("does" is
 *         a stopword, the apostrophe splits the rest) and a corrected fact
 *         reads as a duplicate of the statement it reverses. */
std::string expand_contractions(std::string text);

/** @brief English plural folding for MATCHING only (the raw chunk text is
 *         never touched): a plural QUERY token must find its singular form in
 *         the corpus and vice versa ("dragons" <-> "dragon", "winters" ->
 *         "winter", "stories" -> "story", "churches" -> "church", "wolves" ->
 *         "wolf"). Rules are deliberately narrow so no real word is mangled:
 *         3-letter tokens and ss/us/is endings stay, and the special endings
 *         (-ies, -ves, -ches/-shes/-sses/-xes/-zes) are handled before the
 *         generic single-s strip. */
std::string singularize(const std::string& w)
{
	if (w.size() < 4)
		return w;
	const char c0 = w[w.size() - 1], c1 = w[w.size() - 2], c2 = w[w.size() - 3];
	if (c0 != 's')
		return w;
	if (c1 == 's')
		return w; /* glass, press, cross */
	if (c1 == 'e')
	{
		if (c2 == 'i')
			return w.substr(0, w.size() - 3) + "y"; /* stories -> story */
		if (c2 == 'v')
			return w.substr(0, w.size() - 3) + "f"; /* wolves -> wolf */
		if (c2 == 'c' || c2 == 's' || c2 == 'x' || c2 == 'z' || c2 == 'h')
			return w.substr(0, w.size() - 2); /* churches -> church, boxes -> boxe.. xes -> box handled by c2=='x' -> strip es */
		return w; /* ages, ties, notes stay (silent-e plurals match their base only when the base ends in e: "notes" vs "note" is covered by the generic strip below ONLY if not ending e+? -- keep 'Xes' non-chshxz plurals unstripped to avoid mangling "notes"->"note" -- actually strip here too? no: "notes" -> generic s-strip would give "note" which is RIGHT. falling through loses nothing. */
	}
	if (c1 == 'u' || c1 == 'i')
		return w; /* virus, iris, status */
	return w.substr(0, w.size() - 1); /* dragons -> dragon, winters -> winter */
}

 /** @brief tokenize into lowercase unigrams plus bigrams ("night vision" ->
  *         night, vision, night_vision). bigrams are the strongest precision
  *         signal for thought-text queries. */
str_vec tokenize(std::string text)
{
	text = expand_contractions(text); /* negations must survive tokenization: a corrected fact is not a duplicate of what it corrects */
	str_vec words, out;
	std::string cur;
	for (int i = 0; i <= text.size(); i++)
	{
		unsigned char c = (i < text.size()) ? text[i] : '\0';
		if (isalnum(c))
			cur += (char)tolower(c);
		else if (c == 0xC3 && i + 1 < text.size() && (unsigned char)text[i + 1] >= 0x80 && (unsigned char)text[i + 1] <= 0xBF)
		{
			/* UTF-8 umlaut pair: keep the lead byte; fold the uppercase
			 * continuation range to lowercase (C3 84 "�"" -> C3 A4 "��"), so
			 * sentence-initial "�ber" matches mid-sentence "Ǭber". Lowercase
			 * pairs pass through untouched; every other high byte (E2/C2
			 * punctuation, stray continuations) ends the token. */
			unsigned char n2 = (unsigned char)text[i + 1];
			if (n2 <= 0x9E && n2 != 0x97)
				n2 |= 0x20;
			cur += (char)c;
			cur += (char)n2;
			i++;
		}
		else if (!cur.empty())
		{
			if ((cur.size() > 1 || isdigit((unsigned char)cur[0])) && !is_stop(cur))
				words.push_back(singularize(cur)); /* plural folding: "dragons" and "dragon" are the same MATCHING term */
			cur.clear();
		}
	}
	for (int i = 0; i < words.size(); i++)
	{
		out.push_back(words[i]);
		if (i + 1 < words.size())
			out.push_back(words[i] + "_" + words[i + 1]);
	}
	return out;
}

/** @brief normalize text for better BM25 matching: lowercase, strip * and -,
 *         replace ! with ., collapse whitespace to single space, strip leading
 *         "- " / "* " / "** " list markers, strip **bold** markers.
 *         applied to both corpus text (before chunking) and query text so
 *         the BM25 index and queries share the same normalized space. */
std::string normalize(std::string text)
{
	std::string out;
	bool prev_space = true; /* true = skip leading spaces */
	for (int i = 0; i < text.size(); i++)
	{
		unsigned char c = (unsigned char)text[i]; /* UNSIGNED: high bytes (0xE2 smart quotes, 0xB4, 0xC3 umlauts) must compare equal to their byte values, not to negative numbers */
		/* strip ** (markdown bold marker) */
		if (c == '*' && i + 1 < text.size() && text[i + 1] == '*')
		{
			i++;
			continue;
		}
		/* strip * (markdown italic / bullet) */
		if (c == '*')
			continue;
		/* strip - only when NOT between word chars (preserve hyphenated
		 * words). a high byte counts as a word char here: UTF-8 umlauts
		 * surround hyphens too ("über-" ends a word, not a sentence). a
		 * dropped hyphen that would glue two words together ("fast--really")
		 * leaves a space behind. */
		if (c == '-')
		{
			bool left_alpha = (i > 0 && (isalnum((unsigned char)text[i - 1]) || (unsigned char)text[i - 1] >= 0x80));
			bool right_alpha = (i + 1 < text.size() && (isalnum((unsigned char)text[i + 1]) || (unsigned char)text[i + 1] >= 0x80));
			if (left_alpha && right_alpha)
			{
				out += c;
				prev_space = false;
			}
			else if (!prev_space)
			{
				out += ' ';
				prev_space = true;
			}
			continue;
		}
		/* normalize smart apostrophes and backticks to ASCII ': the model
		 * writes "don't" with U+2019 (E2 80 99) as often as with ', and the
		 * contraction expander below only knows the ASCII form. the
		 * spacing accent U+00B4 arrives as the UTF-8 pair C2 B4 -- handle
		 * the PAIR, a lone 0xB4 continuation byte must not turn into an
		 * apostrophe inside another character. */
		if (c == 0xE2 && i + 2 < text.size() && (unsigned char)text[i + 1] == 0x80 && ((unsigned char)text[i + 2] == 0x99 || (unsigned char)text[i + 2] == 0x98))
		{
			out += '\'';
			i += 2;
			prev_space = false;
			continue;
		}
		if (c == 0xC2 && i + 1 < text.size() && (unsigned char)text[i + 1] == 0xB4)
		{
			out += '\'';
			i++;
			prev_space = false;
			continue;
		}
		if (c == '`')
		{
			out += '\'';
			prev_space = false;
			continue;
		}
		/* replace ! with . */
		if (c == '!')
			c = '.';
		/* collapse whitespace */
		if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
		{
			if (!prev_space)
			{
				out += ' ';
				prev_space = true;
			}
			continue;
		}
		/* lowercase */
		out += (char)tolower((unsigned char)c);
		prev_space = false;
	}
	/* trim trailing space */
	while (!out.empty() && out.back() == ' ')
		out.pop_back();
	return out;
}

/** @brief expand English negation contractions BEFORE tokenization: don't ->
 *         do not, can't -> cannot, won't -> will not, and every other n't
 *         form. without this, "doesn't like" loses its negation ("does" is
 *         a stopword, the apostrophe splits the rest) and a corrected fact
 *         reads as a duplicate of the statement it reverses. */
std::string expand_contractions(std::string text)
{
	const char *words[][2] = {
		{"won't", "will not"},
		{"can't", "cannot"},
		{"don't", "do not"},
		{"doesn't", "does not"},
		{"didn't", "did not"},
		{"isn't", "is not"},
		{"aren't", "are not"},
		{"wasn't", "was not"},
		{"weren't", "were not"},
		{"hasn't", "has not"},
		{"haven't", "have not"},
		{"hadn't", "had not"},
		{"couldn't", "could not"},
		{"shouldn't", "should not"},
		{"wouldn't", "would not"},
		{"mustn't", "must not"},
		{"needn't", "need not"},
		{"mightn't", "might not"},
		{"shan't", "shall not"},
	};
	for (int w = 0; w < sizeof words / sizeof words[0]; w++)
	{
		std::string from = words[w][0];
		std::string to = words[w][1];
		size_t p = text.find(from);
		while (p != std::string::npos)
		{
			bool left_ok = (p == 0) || !isalnum((unsigned char)text[p - 1]);
			size_t after = p + from.size();
			bool right_ok = (after >= text.size()) || !isalnum((unsigned char)text[after]);
			if (left_ok && right_ok)
			{
				text = text.substr(0, p) + to + text.substr(after);
				after = p + to.size(); /* the replacement is longer than the contraction: the old offset would land INSIDE it */
			}
			else
				after = p + 1; /* part of a longer word: step past it */
			p = text.find(from, after);
		}
	}
	return text;
}

/* ---- chunking: text -> sentence list -------------------------------------- */
#define MAXSENT 280
#define MINSENT 20

typedef struct chunk
{
	std::string text;
	str_vec terms;
	double score;
	bool topic;
} chunk;

/** @brief push one sentence into the chunk list with stub-merging and
 *         hard-splitting of overlong sentences (identical to ragg.cpp).
 *         @p is_topic marks the paragraph's first sentence: the flag lands
 *         on the chunk at CREATION time, so a stub merged from the previous
 *         paragraph can never carry (or misplace) the topic bonus. */
void push_sentence(std::vector<chunk> &chunks, std::string s, bool is_topic)
{
	int b = s.find_first_not_of(" \t\r\n");
	if (b == std::string::npos)
		return;
	int e = s.find_last_not_of(" \t\r\n");
	s = s.substr(b, e - b + 1);

	if (!chunks.empty() && !is_topic && s.size() < MINSENT)
	{
		chunks.back().text += " " + s;
		return;
	}
	while (s.size() > MAXSENT)
	{
		int cut = s.rfind(' ', MAXSENT);
		if (cut == std::string::npos || cut < MINSENT)
			cut = MAXSENT;
		while (cut > 0 && cut < (int)s.size() && ((unsigned char)s[cut] & 0xC0) == 0x80)
			cut--; /* back up over UTF-8 continuation bytes: never split mid-character */
		if (cut < 1)
			cut = MAXSENT; /* the backup must never stall the split: progress beats a mid-character cut */
		chunk c;
		c.text = s.substr(0, cut);
		c.score = 0;
		c.topic = is_topic;
		chunks.push_back(c);
		is_topic = false; /* only the first piece of a topic sentence keeps the bonus */
		s = s.substr(cut);
		int nb = s.find_first_not_of(" \t\r\n");
		if (nb == std::string::npos)
			return;
		s = s.substr(nb);
	}
	chunk c;
	c.text = s;
	c.score = 0;
	c.topic = is_topic;
	chunks.push_back(c);
}

/** @brief paragraph-aware chunking; first sentence of each paragraph is the
 *         topic sentence (gets the scoring bonus). The text arrives RAW:
 *         chunking works on original case and paragraph structure, and only
 *         the term index is built from the normalized text. A bare newline
 *         ends a line-oriented entry (memory corpora append with "%s\n"). */
std::vector<chunk> chunk_text(std::string text)
{
	std::vector<chunk> chunks;
	/* normalize newlines FIRST: argv text arrives with CRLF, and the \n\n
	 * paragraph split below must see the same shape file content sees */
	{
		std::string flat;
		flat.reserve(text.size());
		for (int i = 0; i < text.size(); i++)
			if (text[i] != '\r')
				flat += text[i];
		text = flat;
	}
	int pos = 0;
	while (pos < text.size())
	{
		int nl = text.find("\n\n", pos);
		int end = (nl == std::string::npos) ? text.size() : nl;
		std::string para = text.substr(pos, end - pos);
		pos = (nl == std::string::npos) ? text.size() : nl + 2;

		bool para_topic = true;
		std::string cur;
		for (int i = 0; i < para.size(); i++)
		{
			char c = para[i];
			if (c == '\n' || c == '\r')
			{
				bool pushed = cur.find_first_not_of(" \t") != std::string::npos; /* blank lines between paragraphs do not spend the topic bonus */
				push_sentence(chunks, cur, para_topic); /* a bare newline ends a line-oriented entry */
				cur.clear();
				if (pushed)
					para_topic = false;
				continue;
			}
			cur += c;
			if ((c == '.' || c == '!' || c == '?') && (i + 1 >= para.size() || isspace((unsigned char)para[i + 1])))
			{
				/* a period right after one or two digits is a LIST MARKER
				 * ("1. The model was trained..."), not a sentence end: a
				 * split here would push "1." as its own chunk and steal the
				 * paragraph's topic bonus from the real first sentence */
				bool list_marker = (c == '.');
				if (list_marker)
				{
					int d = (int)cur.size() - 2; /* the dot was just appended: the digit run ends ONE BEFORE it */
					int digits = 0;
					while (d >= 0 && isdigit((unsigned char)cur[d]))
					{
						d--;
						digits++;
					}
					if (digits > 2 || digits == 0 || d >= 0)
						list_marker = false; /* more than two digits, no digits, or digits mid-word: a real sentence dot */
				}
				if (!list_marker)
				{
					push_sentence(chunks, cur, para_topic);
					cur.clear();
					para_topic = false;
				}
			}
		}
		if (!cur.empty())
			push_sentence(chunks, cur, para_topic);
	}
	for (int i = 0; i < chunks.size(); i++)
		chunks[i].terms = tokenize(normalize(chunks[i].text));
	return chunks;
}

/* ---- BM25 ranking ---------------------------------------------------------- */

typedef struct ranked
{
	int idx;
	double score;
	bool operator<(const ranked &o) const
	{
		if (score != o.score)
			return score > o.score;
		return idx < o.idx; /* deterministic tie-break: equal scores keep corpus order */
	}
} ranked;

/** @brief BM25 (k1=1.2, b=0.75) with coverage multiplier and topic bonus
 *         (identical scoring to ragg.cpp so scores are comparable).
 *         qweights (optional): per-query-term multipliers for the FOCUS
 *         ranking pass -- the sidecar's query shape is "own words \n
 *         question", and the question line carries the discriminative terms;
 *         boosting them re-ranks the pool without touching the floor scale. */
std::vector<ranked> rank_chunks(std::vector<chunk> &chunks, str_vec &qterms,
                                const std::map<std::string, int> *qweights = NULL)
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

	for (std::map<std::string, int>::iterator it = df.begin(); it != df.end(); ++it)
		if (num_chunks >= 10 && (double)it->second / (double)num_chunks > 0.4)
			it->second = (int)num_chunks; /* auto-stopword, same rule as relate: on corpora under 10 chunks the rule would clamp EVERY term */

	double k1 = 1.2;
	double b = 0.75;
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
			/* standard smoothed BM25 idf, ALWAYS applied: a clamped
			 * auto-stopword term keeps a sliver of weight (Lucene behavior),
			 * and a 1-chunk corpus still scores. the >=10-chunk clamp guard
			 * above already keeps tiny corpora out of the stopword rule. */
			double idf = log(1.0 + ((double)num_chunks - (double)doc_freq + 0.5) / ((double)doc_freq + 0.5));
			double lenn = (avglen > 0) ? (double)chunks[i].terms.size() / avglen : 1.0;
			double tfc = (double)it->second * (k1 + 1.0) / ((double)it->second + k1 * (1.0 - b + b * lenn));
			double w = 1.0;
			if (qweights)
			{
				std::map<std::string, int>::const_iterator wt = qweights->find(qterms[q]);
				if (wt != qweights->end())
					w = (double)wt->second;
			}
			s += w * idf * tfc;
			if (qterms[q].find('_') == std::string::npos)
				hits[qterms[q]] = 1;
		}
		double cov = quni.empty() ? 1.0 : (double)hits.size() / (double)quni.size();
		s *= 0.3 + 0.7 * cov;
		if (chunks[i].topic)
			s *= 1.15;
		chunks[i].score = s;
	}
	std::vector<ranked> r(num_chunks);
	for (int i = 0; i < num_chunks; i++)
	{
		r[i].idx = i;
		r[i].score = chunks[i].score;
	}
	std::stable_sort(r.begin(), r.end());
	return r;
}

/** @brief clean a single output line: strip leading "- ", "* ", "** ",
 *         markdown bold markers (**text** -> text), trailing spaces,
 *         and return empty string for blank lines.
 *         the LLM think stream should not carry markdown list markers
 *         or blank filler lines from the corpus. */
std::string clean_line(std::string s)
{
	int i = 0;
	while (i < s.size() && (s[i] == ' ' || s[i] == '\t'))
		i++;
	if (i >= s.size())
		return "";
	/* the leading numeric marker, its VALUE, and its dot position are
	 * captured here: the value decides the trailing-residue strip below
	 * (markdown concatenation "1. fact one. 2. fact two." chunks end with
	 * ". 2." -- the NEXT item's number). "1. His age was 25." ends with 25
	 * != 1+1, so the age survives. */
	int lead_num = -1;
	if (isdigit((unsigned char)s[i]))
	{
		int d = i;
		int v = 0;
		while (d < s.size() && isdigit((unsigned char)s[d]))
		{
			if (d - i >= 9)
				break; /* a ten-digit run is a timestamp or id, not a list marker: cap the accumulator before it can overflow */
			v = v * 10 + (s[d] - '0');
			d++;
		}
		if (d > i && d < s.size() && s[d] == '.')
			lead_num = v; /* "N." -- the dot is what makes it a marker, "1985 was cold" stays */
	}
	bool numbered_line = (lead_num >= 0) && (lead_num <= 99); /* one- or two-digit markers are list numbering */
	if (s.compare(i, 4, "- **") == 0 || s.compare(i, 3, "** ") == 0)
		i += (s[i] == '-') ? 4 : 3;
	else if (s.compare(i, 2, "- ") == 0 || s.compare(i, 2, "* ") == 0)
		i += 2; /* a list marker is the marker PLUS its space: "-5 degrees" keeps its sign */
	else if (numbered_line) /* ONLY one/two-digit markers: "1985. It was cold." keeps its year */
	{
		/* leading numeric list marker "3. text" / "17. text": strip the
		 * marker and its space. a bare number that starts a sentence
		 * ("1985 was cold") has no dot and survives. */
		int d = i;
		while (d < s.size() && isdigit((unsigned char)s[d]))
		{
			if (d - i >= 9)
				break; /* a ten-digit run is an id, not a list marker */
			d++;
		}
		if (d + 1 < s.size() && s[d] == '.' && (s[d + 1] == ' ' || s[d + 1] == '\t'))
			i = d + 2;
	}
	std::string r = s.substr(i);
	int rb = r.find_first_not_of(" \t");
	if (rb == std::string::npos)
		return ""; /* nothing but whitespace between the markers */
	if (rb > 0)
		r = r.substr(rb); /* "1.   text" keeps its extra spaces after the marker strip: drop them */
	/* strip markdown asterisks in every shape: **, *** and lone * */
	std::string c;
	for (int j = 0; j < r.size(); j++)
	{
		if (r[j] == '*')
			continue;
		c += r[j];
	}
	while (!c.empty() && (c.back() == ' ' || c.back() == '\t'))
		c.pop_back();
	/* strip trailing LIST-NUMBERING residue (the leading "1. " marker was
	 * already removed above); ONLY on a line that was itself numbered, and
	 * ONLY when the trailing number is the line's SEQUENTIAL successor
	 * (leading 1 -> trailing 2): that exact shape is markdown list
	 * concatenation. "1. His age was 25." keeps 25 (25 != 2); a trailing
	 * number on any other line is content -- "born in 1985.", "chapter
	 * 12.", German "am 14. Oktober" survive. */
	bool numbered = numbered_line;
	int trail_num = -1;
	while (numbered && c.size() >= 3)
	{
		int p = c.size() - 1;
		if (c[p] != '.')
			break;
		int q = p;
		while (q > 0 && c[q - 1] >= '0' && c[q - 1] <= '9')
			q--;
		if (q == p || q == 0 || c[q - 1] != ' ')
			break; /* no space-delimited number before the dot */
		int sp = q;
		while (sp > 0 && c[sp - 1] == ' ')
			sp--; /* walk over ALL spaces: "one.  2." is still concatenation residue */
		if (sp == 0 || (c[sp - 1] != '.' && c[sp - 1] != '!' && c[sp - 1] != '?'))
			break; /* the number must close a SENTENCE ("...one. 2.") to be list residue: "1. His age was 2." keeps its age */
		if (p - q > 2)
			break; /* three or more digits: a year, a count -- keep it */
		trail_num = atoi(c.substr(q, p - q).c_str());
		if (trail_num != lead_num + 1)
			break; /* not the next list item: the number is content */
		while (q > 0 && c[q - 1] == ' ')
			q--;
		c = c.substr(0, q);
		while (!c.empty() && (c.back() == ' ' || c.back() == '\t'))
			c.pop_back();
		break; /* one residue per line */
	}
	return c;
}

/* ---- relate ---------------------------------------------------------------- */

/** @brief handle relate: the vector-space relatedness between one text and a
 *         whole corpus (Salton's vector space model, 1975). The corpus is
 *         ONE document -- every line together -- and both sides become
 *         tf-idf vectors: term weight = term frequency times ln(lines/df),
 *         where df is the LINE-level document frequency, so a word that
 *         pervades the corpus counts little and a rare shared word counts
 *         much. Input terms the corpus never speak get the maximal
 *         distinctiveness ln(2*lines), so a text leaning on words the
 *         records never use grows its own norm and the cosine drops -- that
 *         is the discriminative signal, not the shared vocabulary alone.
 *
 *         OUTPUT CONTRACT: TWO lines on stdout. Line 1 is the raw score
 *         (0..1, four decimals). Line 2 is "relate above floor" or "relate
 *         below floor" -- the decision against the floor= option, printed as
 *         TEXT because the Logo side has no string-to-number conversion and
 *         reads the decision with a string search. The measure is
 *         query-to-corpus relevance, NOT symmetric: relate(A,B) compares
 *         A's term frequencies against B's whole-document weight, so
 *         swapping the arguments changes the score. The score is also
 *         corpus-shape dependent: chunking or tokenization changes move
 *         the calibrated floor.
 *
 *         Prints the score and the decision; the CALLER thresholds (and may
 *         vary the floor at random to steer how strongly the persona is
 *         pulled back to the records). */
int do_relate(std::string dir, std::string text, std::string basepath, double floor_score)
{
	std::string base;
	if (!basepath.empty())
	{
		/* the same fail-closed rule as search: an EXPLICIT base= that cannot
		 * be read must not silently judge relatedness against the
		 * remembered-fact store instead of the requested records */
		if (!read_whole_file(basepath, base) || base.empty())
		{
			log_line(dir, "relate base= '" + snippet(basepath, 80) + "' unreadable -> score 0");
			printf("0.0000\nrelate below floor\n"); /* BOTH contract lines, always: the Logo decision probe searches for the second one */
			return 0;
		}
	}
	else
		read_whole_file(dir + "ragbigram_base.txt", base);
	/* the remember/add store participates ONLY when the caller did not pin a
	 * domain -- the same isolation rule search follows: both commands must
	 * judge the SAME corpus */
	std::string corpus;
	if (basepath.empty())
		read_whole_file(dir + "ragg_corpus.txt", corpus);
	std::vector<chunk> chunks = chunk_text(base + (base.empty() || corpus.empty() ? "" : "\n\n") + corpus);
	if (chunks.empty())
	{
		log_line(dir, "relate -> (empty corpus)");
		printf("0.0000\nrelate below floor\n"); /* BOTH contract lines, always */
		return 0;
	}
	int lines = (int)chunks.size();

	/* line-level document frequency plus whole-corpus term frequency: the
	 * corpus vector has one dimension per distinct unigram */
	std::map<std::string, int> df;
	std::map<std::string, int> ctf;
	for (int i = 0; i < lines; i++)
	{
		std::map<std::string, char> seen;
		for (int j = 0; j < chunks[i].terms.size(); j++)
		{
			std::string t = chunks[i].terms[j];
			if (t.find('_') != std::string::npos)
				continue; /* unigrams only: the vector space of single words */
			seen[t] = 1;
			ctf[t]++;
		}
		for (std::map<std::string, char>::iterator it = seen.begin(); it != seen.end(); ++it)
			df[it->first]++;
	}
	for (std::map<std::string, int>::iterator it = df.begin(); it != df.end(); ++it)
		if (lines >= 10 && (double)it->second / (double)lines > 0.4)
			it->second = lines; /* auto-stopword, same rule AND same small-corpus guard as the ranker (relate's smoothed idf keeps a small weight for such terms by design: the relatedness cosine needs the dimension present, only down-weighted) */

	/* corpus vector: weight = tf * idf, and the norm over all dimensions.
	 * the idf is smoothed ln(1 + lines/df) so it never reaches zero: a term
	 * in every line (df == lines, or a single-line corpus) keeps a small
	 * nonzero weight instead of collapsing the whole vector to zero. */
	double corpus_norm = 0;
	std::map<std::string, double> corpus_w;
	for (std::map<std::string, int>::iterator it = ctf.begin(); it != ctf.end(); ++it)
	{
		double idf = log(1.0 + (double)lines / (double)df[it->first]);
		double w = (double)ctf[it->first] * idf;
		corpus_w[it->first] = w;
		corpus_norm += w * w;
	}
	corpus_norm = sqrt(corpus_norm);

	/* input vector over the same dimensions plus the corpus-absent ones */
	std::map<std::string, int> itf;
	str_vec terms = tokenize(normalize(text));
	for (int i = 0; i < terms.size(); i++)
		if (terms[i].find('_') == std::string::npos)
			itf[terms[i]]++;
	double absent_idf = log(1.0 + 2.0 * (double)lines);
	double input_norm = 0;
	double dot = 0;
	for (std::map<std::string, int>::iterator it = itf.begin(); it != itf.end(); ++it)
	{
		std::map<std::string, double>::iterator cw = corpus_w.find(it->first);
		double idf = absent_idf;
		if (cw != corpus_w.end())
			idf = log(1.0 + (double)lines / (double)df[it->first]); /* same smoothing as the corpus side */
		double w = (double)it->second * idf;
		input_norm += w * w;
		if (cw != corpus_w.end())
			dot += w * cw->second;
	}
	input_norm = sqrt(input_norm);

	double score = (corpus_norm > 0 && input_norm > 0) ? dot / (corpus_norm * input_norm) : 0.0;
	printf("%.4f\n", score);
	bool above = (score > 0.0) && (score >= floor_score); /* zero overlap is NEVER a match, even at floor 0 */
	printf("relate %s floor\n", above ? "above" : "below");
	log_line(dir, "relate score=" + std::to_string(score).substr(0, 6) + " floor=" + std::to_string(floor_score).substr(0, 6) + (above ? " -> above" : " -> below") + " text=\"" + snippet(text, 60) + "\"");
	return 0;
}

/* ---- search ---------------------------------------------------------------- */

/** @brief unigram-set jaccard similarity between two chunk texts (0..1).
 *         used for near-duplicate suppression: the memory distiller re-emits
 *         the same facts each turn with cosmetic drift (v7/V7, first/third
 *         person: enjoy/enjoys), which defeats exact-prefix comparison.
 *         bigrams are EXCLUDED (they skew the ratio low for subset-dups) and
 *         a naive trailing-'s' stem unifies person-shifted verbs. numbers
 *         count: "has 32 layers" vs "has 24 layers" is a CHANGED FACT, not
 *         a duplicate. */
double line_jaccard(chunk &a, chunk &b)
{
	std::map<std::string, char> sa, su;
	for (int i = 0; i < a.terms.size(); i++)
	{
		std::string t = a.terms[i];
		if (t.find('_') != std::string::npos)
			continue; /* unigrams only */
		sa[t.size() > 3 && t[t.size() - 1] == 's' ? t.substr(0, t.size() - 1) : t] = 1;
	}
	for (int i = 0; i < b.terms.size(); i++)
	{
		std::string t = b.terms[i];
		if (t.find('_') != std::string::npos)
			continue; /* unigrams only */
		su[t.size() > 3 && t[t.size() - 1] == 's' ? t.substr(0, t.size() - 1) : t] = 1;
	}
	/* a correction guard BEFORE the ratio: two facts that disagree in a
	 * negation word or in a number are CHANGED FACTS, however high their
	 * word overlap -- "does not like" vs "likes", "32 layers" vs "24
	 * layers" must never read as duplicates */
	int neg_a = 0;
	int neg_b = 0;
	int num_a = 0;
	int num_b = 0;
	for (std::map<std::string, char>::iterator it = sa.begin(); it != sa.end(); ++it)
	{
		if (it->first == "not" || it->first == "never" || it->first == "cannot")
			neg_a = 1;
		if (it->first.find_first_not_of("0123456789") == std::string::npos)
			num_a = 1;
	}
	for (std::map<std::string, char>::iterator it = su.begin(); it != su.end(); ++it)
	{
		if (it->first == "not" || it->first == "never" || it->first == "cannot")
			neg_b = 1;
		if (it->first.find_first_not_of("0123456789") == std::string::npos)
			num_b = 1;
	}
	if (neg_a != neg_b || num_a != num_b)
		return 0.0; /* one side negates or the numbers moved: different facts by definition */
	int inter = 0;
	for (std::map<std::string, char>::iterator it = sa.begin(); it != sa.end(); ++it)
		if (su.count(it->first))
			inter++;
	int uni = sa.size() + su.size() - inter;
	if (uni)
		return (double)inter / (double)uni;
	return 0.0; /* two term-less chunks share nothing: treating them as identical would NOOP new entries and let xbase drop chunks blindly */
}

/** @brief collect the newest MEMORY chunks (index >= base_chunks: never the
 *         static reference text) as the recent-memory fallback, near-dup
 *         suppressed. the emitted order is CHRONOLOGICAL: the oldest of the
 *         collected memories first, the newest last, so the text reads like
 *         a recollection building up to the present. returns "" when the
 *         memory range holds nothing usable. */
std::string recent_memory_tail(std::vector<chunk> &chunks, int base_chunks, int maxlines, double dedup)
{
	std::string out;
	std::vector<int> tail;
	int want2 = maxlines;
	for (int i = (int)chunks.size(); i-- > base_chunks && want2 > 0;)
	{
		std::string cl = clean_line(chunks[i].text);
		if (cl.empty())
			continue;
		bool dup = false;
		if (dedup > 0)
		{
			for (int k = 0; k < tail.size() && !dup; k++)
				if (line_jaccard(chunks[tail[k]], chunks[i]) >= dedup)
					dup = true;
		}
		tail.push_back(i);
		if (dup)
			continue;
		out = cl + "\n" + out;
		want2--;
	}
	return out;
}

/** @brief handle search: rank, apply the dual answer-presence floor
 *         (density minscore OR topscore topmin must pass; exp 17), pick
 *         maxlines chunks from the top-10 pool (rand=0: deterministic
 *         top-N; rand>0: rank-1 kept, rest rank-weighted random), near-dup
 *         suppression (jaccard when dedup>0, else legacy 60-char prefix),
 *         recency prior (recency=S, memory corpora), recent-memory
 *         fallback (recent=1: weak/empty match returns corpus tail),
 *         plain lines on stdout. empty stdout = below floor and no
 *         recent fallback. */
int do_search(std::string dir, std::string query, int rnd, int maxlines, double minscore, double topmin, std::string basepath, double dedup, double recency, int recent, std::string xbasepath, int qboost)
{
	std::string base, corpus;
	bool base_was_specified = !basepath.empty();
	if (base_was_specified)
	{
		/* an EXPLICIT base= that is empty, cannot be read, or holds nothing
		 * fails CLOSED with an empty answer: falling back to the
		 * remembered-fact store would silently change the search domain --
		 * wrong-lore memories in a lore answer */
		if (basepath.empty())
			basepath = "<explicitly empty>";
		if (!read_whole_file(basepath, base) || base.empty())
		{
			log_line(dir, "search base= '" + snippet(basepath, 80) + "' unreadable -> empty result");
			return 0;
		}
	}
	else
	{
		if (!read_whole_file(dir + "ragbigram_base.txt", base))
			log_line(dir, "search default ragbigram_base.txt missing -> base empty");
	}
	/* the remember/add store participates ONLY when the caller did not pin a
	 * domain: an explicit base= is a domain contract, and mixing global
	 * memories into it would let recency-promoted chatter outrank lore */
	std::vector<chunk> chunks = chunk_text(base);
	int base_chunks = (int)chunks.size();
	if (!base_was_specified)
	{
		read_whole_file(dir + "ragg_corpus.txt", corpus);
		if (corpus != base)
		{
			std::vector<chunk> cch = chunk_text(corpus);
			for (int i = 0; i < cch.size(); i++)
				chunks.push_back(cch[i]);
		}
	}
	else
		base_chunks = 0; /* an explicit base= IS the caller's store: recency and the recent fallback must see all of it, not treat every chunk as frozen reference text */
	if (chunks.empty())
	{
		log_line(dir, "search q=\"" + snippet(query, 80) + "\" -> (empty)");
		return 0;
	}

	str_vec qterms_all = tokenize(normalize(query));
	/* dedupe: a generated think repeats words ("dragon dragon dragon") and
	 * every repeat would multiply the BM25 contribution while the coverage
	 * multiplier counts the term once -- relevance must not inflate */
	str_vec qterms;
	{
		std::map<std::string, char> seen_q;
		for (int i = 0; i < qterms_all.size(); i++)
			if (!seen_q[qterms_all[i]])
			{
				seen_q[qterms_all[i]] = 1;
				qterms.push_back(qterms_all[i]);
			}
	}
	/* xbase= cross-corpus exclusion (exp 23): drop chunks that near-duplicate
	 * lines from the exclusion base (e.g. the user-profile corpus) BEFORE
	 * ranking, and before EVERY path that can emit chunks -- the recent
	 * fallback for a stopword-only query runs below and must respect the
	 * exclusion too. ranking happens ONCE, after this filter. */
	if (!xbasepath.empty() && dedup <= 0)
		log_line(dir, "search xbase= given but dedup=0 -- the exclusion is INERT without a dedup threshold"); /* a silently dead exclusion looks exactly like a working one */
	if (!xbasepath.empty() && dedup > 0)
	{
		std::string xbase;
		/* an EXPLICIT xbase= that cannot be read fails CLOSED like base=:
		 * searching without the exclusion would inject exactly the
		 * cross-store duplicates the caller asked to keep out */
		if (!read_whole_file(xbasepath, xbase))
		{
			log_line(dir, "search xbase= '" + snippet(xbasepath, 80) + "' unreadable -> empty result");
			return 0;
		}
		if (!xbase.empty())
		{
			std::vector<chunk> xch = chunk_text(xbase);
			std::vector<chunk> keep;
			int kept_base = 0; /* the static/memory boundary moves when reference chunks are excluded: the recency prior classifies by index */
			for (int i = 0; i < chunks.size(); i++)
			{
				bool dup = false;
				for (int k = 0; k < xch.size() && !dup; k++)
					if (line_jaccard(chunks[i], xch[k]) >= dedup)
						dup = true;
				if (!dup)
				{
					keep.push_back(chunks[i]);
					if (i < base_chunks)
						kept_base++;
				}
			}
			chunks = keep;
			base_chunks = kept_base;
			if (chunks.empty())
			{
				log_line(dir, "search xbase excluded everything q=\"" + snippet(query, 80) + "\" -> (empty)");
				return 0;
			}
		}
	}
	if (qterms.empty())
	{
		log_line(dir, "search q=\"" + snippet(query, 80) + "\" -> (no terms)");
		if (recent)
		{
			/* a stopword-only question ("what was that about?") still wants
			 * memory recall when recent=1 -- the fallback never sees this
			 * path otherwise, because BM25 has no terms to match */
			std::string out = recent_memory_tail(chunks, base_chunks, maxlines, dedup);
			if (!out.empty())
			{
				printf("%s", out.c_str());
				log_line(dir, "search q=\"" + snippet(query, 80) + "\" -> (recent fallback, no terms)");
				return 0;
			}
		}
		return 0; /* an empty answer, not an error: a stopword-only query is valid input */
	}
	std::vector<ranked> ranked = rank_chunks(chunks, qterms);

	/* focus-line boost (generic mechanism, no topic strings): the sidecar
	 * query shape is "own words \n question" -- the model's think block
	 * restates PREVIOUS context and its many common terms crowd out the rare
	 * question terms ("are there dragons" vs a founding-heavy think). The
	 * FINAL \n-separated line of the query is the question: its tokens are
	 * re-ranked with a weight multiplier so the discriminative terms win the
	 * POOL. The floor block above runs on the UNBOOSTED ranking -- the
	 * calibrated minscore/topmin thresholds keep their scale. Single-line
	 * queries get no boost (weighting everything equally changes nothing but
	 * the floor scale, which must stay calibrated). */
	std::map<std::string, int> qweights;
	std::vector<struct ranked> ranked_focus = ranked; /* default: identical to the floor ranking */
	{
		size_t last_nl = query.rfind('\n');
		if (last_nl != std::string::npos && qboost > 1)
		{
			std::string focus = query.substr(last_nl + 1);
			if (!focus.empty())
			{
				str_vec fterms = tokenize(normalize(focus));
				for (int i = 0; i < fterms.size(); i++)
					qweights[fterms[i]] = qboost;
				ranked_focus = rank_chunks(chunks, qterms, &qweights);
			}
		}
	}

	/* recency prior (H3, exp 20): later chunks are newer memories - a small
	 * position-proportional bonus makes corrections outrank the entries they
	 * supersede. applied BEFORE floor/pool so the floor sees final scores.
	 * the bonus belongs to the MEMORY store only: base chunks are static
	 * reference text, their position in the file is not newness. */
	if (recency > 0 && chunks.size() > 1)
	{
		for (int i = 0; i < ranked.size(); i++)
		{
			if (ranked[i].score <= 0)
				continue;
			int ci = ranked[i].idx;
			if (ci < base_chunks)
				continue; /* static reference text does not age */
			ranked[i].score += recency * (double)(ci - base_chunks) / (double)(chunks.size() - base_chunks - 1 > 0 ? chunks.size() - base_chunks - 1 : 1);
		}
		std::stable_sort(ranked.begin(), ranked.end());
	}

	/* dual answer-presence floor (exp 17, law 9): a floor suppresses only
	 * when EVERY ARMED floor missed the query -- an unarmed floor can never
	 * fail, so one-floor configs suppress on their single metric. density
	 * (topscore / distinct query unigrams) protects small corpora from
	 * query-length inflation (quantum think vs cat KB: topscore 2.6
	 * absolute but 0.021/term). topscore floor protects long thinks on
	 * large corpora where density collapses for true hits (david/goliath
	 * think 1440c vs 31k-verse bible: density fell to 0.45 --
	 * indistinguishable from misses -- while topscore stayed 34 vs miss
	 * 12-19). diagnostics go to ragg.log ONLY -- v7_execute merges stderr
	 * into the captured buffer, i.e. into the LLM stream (T7 run 1 bug). */
	double density = 0;
	{
		std::map<std::string, char> distinct;
		for (int q = 0; q < qterms.size(); q++)
			if (qterms[q].find('_') == std::string::npos)
				distinct[qterms[q]] = 1;
		if (!distinct.empty())
			density = ranked[0].score / (double)distinct.size();
	}
	bool density_armed = minscore > 0;
	bool topscore_armed = topmin > 0;
	bool floor_hit = (density_armed && density >= minscore) || (topscore_armed && ranked[0].score >= topmin);
	if ((density_armed || topscore_armed) && (!floor_hit))
	{
		log_line(dir, "search FLOOR density=" + std::to_string(density).substr(0, 8) + " < " + std::to_string(minscore).substr(0, 8) + " AND topscore=" + std::to_string(ranked[0].score).substr(0, 8) + " < " + std::to_string(topmin).substr(0, 8) +
		                  " q=\"" + snippet(query, 80) + "\" -> suppressed");
		if (recent)
		{
			/* recent=1 (exp 20, H2): memory corpus - a below-floor match on
			 * "what do you remember" style queries is a vocabulary-mismatch
			 * MISS, not an absence of memories. return the newest ones,
			 * near-dup suppressed so variants of one fact do not fill the
			 * whole recall window. */
			std::string out = recent_memory_tail(chunks, base_chunks, maxlines, dedup);
			if (!out.empty())
			{
				printf("%s", out.c_str());
				log_line(dir, "search q=\"" + snippet(query, 80) + "\" -> (recent fallback, below floor)");
				return 0;
			}
		}
		return 0;
	}
	log_line(dir, "search density=" + std::to_string(density).substr(0, 8) + " topscore=" + std::to_string(ranked[0].score).substr(0, 8));

	/* candidate pool = top 10 with nonzero score (or maxlines, whichever is
	 * larger: a maxlines=15 request must be able to emit 15 lines) */
	std::vector<int> pool;
	int pool_cap = (maxlines > 10) ? maxlines : 10;
	for (int i = 0; i < ranked_focus.size() && pool.size() < pool_cap; i++)
		if (ranked_focus[i].score > 0)
			pool.push_back(ranked_focus[i].idx);
	if (pool.empty())
	{
		if (recent)
		{
			/* recent-memory fallback: queries like "what do you remember about
			 * me?" share NO terms with the corpus -- BM25 cannot match. return
			 * the most recent memories (corpus tail) instead of nothing: that
			 * is what a person recalls when asked out of the blue. dedup'd so
			 * one fact's variants do not fill the window. without recent=1 an
			 * empty match stays an empty answer -- the corpus tail must never
			 * leak into a think stream that asked about something else. */
			std::string out = recent_memory_tail(chunks, base_chunks, maxlines, dedup);
			if (!out.empty())
			{
				printf("%s", out.c_str());
				log_line(dir, "search q=\"" + snippet(query, 80) + "\" -> (recent fallback)");
				return 0;
			}
		}
		log_line(dir, "search q=\"" + snippet(query, 80) + "\" -> (no match)");
		return 0;
	}

	if (rnd > 0)
	{
#ifdef _WIN32
		srand((unsigned)((time(0) ^ GetTickCount64()) ^ 0x5EED)); /* two calls in one second must still differ */
#else
		timespec mono{};
		clock_gettime(CLOCK_MONOTONIC, &mono);
		srand((unsigned)((time(0) ^ (mono.tv_sec * 1000 + mono.tv_nsec / 1000000)) ^ 0x5EED)); /* same: monotonic milliseconds */
#endif
	}

	/* pick up to maxlines output chunks from the pool:
	 *   rand = 0  -> top maxlines of the pool in score order (deterministic)
	 *   rand > 0  -> rank-1 kept, the rest drawn rank-weighted at random */
	int want = (rnd == 0 || maxlines <= 1) ? (int)maxlines : (int)(2 + rand() % (maxlines - 1));
	if (want > pool.size())
		want = pool.size();
	/* maxlines is an UPPER BOUND, not a guarantee: the loop stops when the
	 * pool is exhausted, so fewer lines can come out when duplicates and
	 * clean-to-empty chunks eat the candidates */
	std::vector<char> used(pool.size(), 0);
	std::vector<int> picked;
	std::vector<int> seen; /* EVERY evaluated slot feeds the dedup check, so a duplicate of a skipped duplicate is skipped too */
	used[0] = 1;
	seen.push_back(pool[0]);
	if (!clean_line(chunks[pool[0]].text).empty())
		picked.push_back(pool[0]); /* a chunk that cleans to nothing must not consume an output slot */
	while (picked.size() < want)
	{
		long choice = -1;
		if (rnd == 0)
		{
			for (int i = 0; i < pool.size(); i++)
				if (!used[i])
				{
					choice = (long)i;
					break;
				}
		}
		else
		{
			double total = 0;
			for (int i = 1; i < pool.size(); i++)
				if (!used[i])
					total += (double)(pool.size() - i);
			if (total <= 0)
				break;
			double roll = ((double)rand() / RAND_MAX) * total;
			for (int i = 1; i < pool.size(); i++)
			{
				if (used[i])
					continue;
				roll -= (double)(pool.size() - i);
				if (roll <= 0)
				{
					choice = (long)i;
					break;
				}
			}
			if (choice < 0)
				for (long i = (long)pool.size() - 1; i >= 1; i--)
					if (!used[i])
					{
						choice = i;
						break;
					}
		}
		if (choice < 0)
			break;
		used[choice] = 1;
		bool dup = false;
		if (dedup > 0)
		{
			/* jaccard near-dup suppression (exp 20, H1): the memory distiller
			 * re-emits the same facts with cosmetic drift; token-set overlap
			 * catches v7/V7 and person-shift variants a prefix compare misses. */
			for (int k = 0; k < seen.size() && !dup; k++)
				if (line_jaccard(chunks[seen[k]], chunks[pool[choice]]) >= dedup)
					dup = true;
		}
		else
		{
			/* legacy guard: exact 60-char prefix equality */
			std::string head = chunks[pool[choice]].text.substr(0, 60);
			for (int k = 0; k < seen.size(); k++)
				if (chunks[seen[k]].text.substr(0, 60) == head)
					dup = true;
		}
		seen.push_back(pool[choice]);
		if (dup)
			continue;
		if (clean_line(chunks[pool[choice]].text).empty())
			continue; /* cleans to nothing: not an output candidate, but it already fed the dedup check above */
		picked.push_back(pool[choice]);
	}

	/* emit picked chunks in score order -- stdout goes into the LLM stream.
	 * each line separated by \n\n (a BLANK line between records: the injected
	 * lines must read as separate paragraphs inside the model's thinking -- a
	 * single \n glues them into one wall of text and breaks the thinking
	 * style), one trailing \n after the whole block. no markdown markers. */
	std::string out;
	for (int i = 0; i < pool.size(); i++)
	{
		bool in = false;
		for (int k = 0; k < picked.size(); k++)
			if (picked[k] == pool[i])
				in = true;
		if (!in)
			continue;
		std::string cl = clean_line(chunks[pool[i]].text);
		if (cl.empty())
			continue;
		if (!out.empty())
			out += "\n\n";
		out += cl;
	}
	if (!out.empty())
		printf("%s\n", out.c_str());
	log_line(dir, "search rnd=" + std::to_string(rnd) + " n=" + std::to_string(maxlines) + " floor=" + std::to_string(minscore).substr(0, 6) + " q=\"" + snippet(query, 80) + "\" -> \"" + snippet(out, 80) + "\"");
	return 0;
}

/** @brief true when the file at @p path exists and does NOT end with a
 *         newline: appending to it directly would glue the new entry onto
 *         the last line (manually edited corpora need this guard). */
bool file_needs_newline(std::string path)
{
	FILE *f = fopen(path.c_str(), "rb");
	if (!f)
		return false;
	if (fseek(f, 0, SEEK_END) != 0)
	{
		fclose(f);
		return false;
	}
	long n = ftell(f);
	bool need = (n > 0);
	if (need)
	{
		char last = 0;
		if (fseek(f, n - 1, SEEK_SET) == 0 && fread(&last, 1, 1, f) == 1)
			need = (last != '\n' && last != '\r');
	}
	fclose(f);
	return need;
}

/** @brief handle add: append text to ragg_corpus.txt as its own paragraph.
 *         diagnostics go to ragg.log only -- NEVER stderr (v7_execute merges
 *         stderr into the captured return buffer, i.e. into the LLM stream). */
int do_add(std::string dir, std::string text)
{
	std::string path = dir + "ragg_corpus.txt";
	bool need_nl = file_needs_newline(path); /* measured BEFORE the append handle opens: two handles on one file is asking for locking trouble */
	FILE *f = fopen(path.c_str(), "a");
	if (!f)
	{
		log_line(dir, "add FAILED - cannot open ragg_corpus.txt");
		return 1;
	}
	if (need_nl)
		fprintf(f, "\n");
	fprintf(f, "%s\n\n", text.c_str());
	if (fclose(f) != 0)
	{
		log_line(dir, "add FAILED - write error (disk full?)");
		return 1;
	}
	log_line(dir, "add \"" + snippet(text, 80) + "\"");
	return 0;
}

/** @brief handle remember: mem0-style NOOP-aware append (exp 22). appends
 *         the line to the base= corpus ONLY when it is not a near-duplicate
 *         of an existing line (token-jaccard < dedup threshold). this is
 *         the ADD/NOOP half of mem0 conflict resolution; UPDATE is
 *         approximated at read time by the recency prior. returns 0 either
 *         way; stdout stays empty so nothing leaks into the LLM stream. */
int do_remember(std::string dir, std::string text, double dedup, std::string basepath)
{
	std::string path;
	bool base_was_specified = !basepath.empty();
	if (base_was_specified)
		path = basepath;
	else
		path = dir + "ragg_corpus.txt";
	if (base_was_specified && path.empty())
	{
		log_line(dir, "remember base= explicitly EMPTY -> no store, no write");
		return 1;
	}
	std::string base;
	read_whole_file(path, base);
	/* the candidate is chunked and validated REGARDLESS of the corpus state:
	 * a text whose chunks carry NO indexable terms (stopword-only,
	 * punctuation-only) is never stored -- such lines can never be
	 * retrieved by BM25, they would only feed the recent tail */
	std::vector<chunk> cand = chunk_text(text);
	bool any_terms = false;
	for (int k = 0; k < cand.size() && !any_terms; k++)
		any_terms = !cand[k].terms.empty();
	if (cand.empty() || !any_terms)
	{
		log_line(dir, "remember NOOP (no indexable terms) \"" + snippet(text, 60) + "\"");
		return 0;
	}
	std::string append_text = text;
	if (dedup > 0)
	{
		/* within-submission dedup runs on an EMPTY corpus too: one text
		 * carrying the same fact three times must store it once */
		std::vector<chunk> chunks = chunk_text(base);
		/* each part is checked on its own -- against the existing corpus AND
		 * against the parts already accepted from this same submission, so a
		 * fact repeated inside one text is stored once */
		std::vector<chunk> keep;
		for (int k = 0; k < cand.size(); k++)
		{
			bool dup = false;
			for (int i = 0; i < chunks.size() && !dup; i++)
				if (line_jaccard(chunks[i], cand[k]) >= dedup)
					dup = true;
			for (int i = 0; i < keep.size() && !dup; i++)
				if (line_jaccard(keep[i], cand[k]) >= dedup)
					dup = true;
			if (!dup)
				keep.push_back(cand[k]);
		}
		if (keep.empty())
		{
			log_line(dir, "remember NOOP (all parts duplicates) \"" + snippet(text, 60) + "\"");
			return 0;
		}
		if (keep.size() == cand.size())
		{
			/* nothing was filtered: the ORIGINAL text is stored verbatim -- a
			 * rebuilt join would shatter single-paragraph memories into
			 * pseudo-paragraphs and misplace topic sentences on the next read */
			append_text = text;
		}
		else
		{
			append_text = keep[0].text;
			for (int k = 1; k < keep.size(); k++)
				append_text += "\n" + keep[k].text; /* survivors of the same submission stay one paragraph: they were one thought */
		}
	}
	bool need_nl = file_needs_newline(path); /* measured BEFORE the append handle opens */
	FILE *f = fopen(path.c_str(), "a");
	if (!f)
	{
		log_line(dir, "remember FAILED - cannot open " + path);
		return 1;
	}
	if (need_nl)
		fprintf(f, "\n");
	fprintf(f, "%s\n\n", append_text.c_str()); /* a blank line keeps each remembered fact its own paragraph, so topic detection survives */
	if (fclose(f) != 0)
	{
		log_line(dir, "remember FAILED - write error (disk full?)");
		return 1;
	}
	log_line(dir, "remember ADD \"" + snippet(append_text, 80) + "\"");
	return 0;
}

/** @brief trim one option token: " minscore=0.5" (a space after the
 *         semicolon) must parse like "minscore=0.5"; one layer of wrapping
 *         quotes goes too ("base=\"my file.txt\"" is a path, not a path
 *         with quotation marks in it). */
std::string trim_opt(std::string one)
{
	size_t b = one.find_first_not_of(" \t\r\n");
	if (b == std::string::npos)
		return "";
	one = one.substr(b, one.find_last_not_of(" \t\r\n") - b + 1);
	if (one.size() >= 2 && ((one.front() == '"' && one.back() == '"') || (one.front() == '\'' && one.back() == '\'')))
		one = one.substr(1, one.size() - 2);
	return one;
}

/** @brief strip one layer of wrapping quotes from an option VALUE
 *         ("base=\"my file.txt\"" -> "my file.txt"). */
std::string strip_quote_pair(std::string v)
{
	if (v.size() >= 2 && ((v.front() == '"' && v.back() == '"') || (v.front() == '\'' && v.back() == '\'')))
		return v.substr(1, v.size() - 2);
	return v;
}

/** @brief true when @p v parses COMPLETELY as an integer -- digits only,
 *         optional sign, at least one digit, and short enough that atoi()
 *         cannot overflow (nine digits keep every value inside int range).
 *         "3.9" and "abc" are rejected: atoi would silently truncate or
 *         zero them. */
bool is_valid_int(std::string v)
{
	if (v.empty())
		return false;
	size_t s = (v[0] == '-' || v[0] == '+') ? 1 : 0;
	if (s >= v.size() || v.size() - s > 9)
		return false;
	for (size_t i = s; i < v.size(); i++)
		if (!isdigit((unsigned char)v[i]))
			return false;
	return true;
}

/** @brief true when @p v parses COMPLETELY as a number (atoi/atof would
 *         silently turn "abc" or "1x" into 0 or 1, disabling a floor or a
 *         dedup threshold without any trace). */
bool is_valid_number(std::string v)
{
	if (v.empty())
		return false;
	char *end = 0;
	double d = strtod(v.c_str(), &end);
	if (end != v.c_str() + v.size())
		return false;
	return d == d && d * 0.0 == 0.0; /* not NaN, not inf */
}

/** @brief entry point: search/add dispatch.
 *
 * v7_execute (the sector_logo C-CALL bridge) passes exe + 3 string params
 * only, so the option surface is packed into ONE options string:
 *
 *   bm25_rag.exe search "<query>" "<opt;opt;...>"
 *       opts: rand=N; maxlines=N; minscore=S   (all optional, any order)
 *   bm25_rag.exe add "<text>"
 */
int run_main(int argc, char **argv)
{
	if (argc < 3)
	{
		/* NO stdout here: v7_execute captures stdout into the LLM stream, so
		 * a dropped parameter must answer with an EMPTY result, not usage
		 * text. the usage goes to the log for the operator. */
		log_line(exe_dir(), "USAGE ERROR argc<3 -> empty stdout; see ragg.log");
		FILE *f = fopen((exe_dir() + "ragg.log").c_str(), "a");
		if (f)
		{
			fprintf(f, "usage: bm25_rag.exe search \"<query>\" \"[rand=N;][maxlines=N;][minscore=S;][topmin=S;][base=PATH;][dedup=S;][recency=S;][recent=N;][xbase=PATH]\"\n"
			           "       bm25_rag.exe relate \"<text>\" \"[base=PATH]\"   vector-space relatedness of text to the whole corpus, prints 0..1\n"
		       "       bm25_rag.exe remember \"<fact line>\" \"[dedup=S;][base=PATH]\"\n"
			           "       bm25_rag.exe add \"<text to remember>\"\n"
			           "answer-presence: inject when ANY armed floor passes: density>=minscore OR topscore>=topmin\n"
			           "dedup=S    near-dup suppression via token-jaccard (0=off legacy prefix, 0.5=memory corpora)\n"
			           "recency=S  position-proportional score bonus, later=newer (memory corpora)\n"
			           "recent=1   weak/empty match returns corpus tail (memory recall fallback)\n"
			           "xbase=S    exclude near-duplicates of this corpus; inert unless dedup>0\n");
			fclose(f);
		}
		return 1;
	}
	std::string dir = exe_dir();
	std::string cmd = argv[1];
	std::string arg = argv[2];
	if (cmd == "add")
		return do_add(dir, arg);
	if (cmd == "remember")
	{
		/* mem0-style NOOP-aware append: opts carry dedup= and base= only */
		double dedup = 0.0;
		std::string basepath;
		if (argc >= 4)
		{
			std::string opts = argv[3];
			int pos = 0;
			while (pos < opts.size())
			{
				int semi = opts.find(';', pos);
				if (semi == std::string::npos)
					semi = opts.size();
				std::string one = trim_opt(opts.substr(pos, semi - pos));
				if (strncmp(one.c_str(), "dedup=", 6) == 0)
					dedup = is_valid_number(one.substr(6)) ? atof(one.c_str() + 6) : dedup;
				else if (strncmp(one.c_str(), "base=", 5) == 0)
					basepath = strip_quote_pair(one.substr(5));
				else if (!one.empty())
					log_line(dir, "remember IGNORED unknown option '" + one + "'"); /* a dropped base= would write the fact into the WRONG store */
				pos = semi + 1;
			}
		}
		if (dedup < 0)
			dedup = 0;
		if (dedup > 1)
			dedup = 1;
		return do_remember(dir, arg, dedup, basepath);
	}
	if (cmd == "search")
	{
		int rnd = 0;
		int maxlines = 3;
		int recent = 0;
		double minscore = 0.0;
		double topmin = 0.0;
		double dedup = 0.0;
		double recency = 0.0;
		int qboost = 3; /* focus-line weight default: the question line outranks think-block restatements */
		std::string basepath, xbasepath;
		if (argc >= 4)
		{ /* single packed options string */
			std::string opts = argv[3];
			int pos = 0;
			while (pos < opts.size())
			{
				int semi = opts.find(';', pos);
				if (semi == std::string::npos)
					semi = opts.size();
				std::string one = trim_opt(opts.substr(pos, semi - pos));
				if (strncmp(one.c_str(), "rand=", 5) == 0)
				{
					if (!is_valid_int(one.substr(5)))
						log_line(dir, "search INVALID value for rand: '" + one + "' (kept " + std::to_string(rnd) + ")");
					else
						rnd = atoi(one.c_str() + 5);
				}
				else if (strncmp(one.c_str(), "maxlines=", 9) == 0)
				{
					if (!is_valid_int(one.substr(9)))
						log_line(dir, "search INVALID value for maxlines: '" + one + "' (kept " + std::to_string(maxlines) + ")");
					else
						maxlines = atoi(one.c_str() + 9);
				}
				else if (strncmp(one.c_str(), "minscore=", 9) == 0)
				{
					if (!is_valid_number(strip_quote_pair(one.substr(9))))
						log_line(dir, "search INVALID value for minscore: '" + one + "' (floor stays armed/unarmed as parsed so far)");
					else
						minscore = atof(one.c_str() + 9);
				}
				else if (strncmp(one.c_str(), "topmin=", 7) == 0)
				{
					if (!is_valid_number(strip_quote_pair(one.substr(7))))
						log_line(dir, "search INVALID value for topmin: '" + one + "'");
					else
						topmin = atof(one.c_str() + 7);
				}
				else if (strncmp(one.c_str(), "qboost=", 7) == 0)
				{
					if (!is_valid_int(one.substr(7)))
						log_line(dir, "search INVALID value for qboost: '" + one + "' (kept " + std::to_string(qboost) + ")");
					else
						qboost = atoi(one.c_str() + 7);
				}
				else if (strncmp(one.c_str(), "base=", 5) == 0)
				{
					basepath = strip_quote_pair(one.substr(5));
					if (basepath.empty())
						basepath = "<explicitly empty>"; /* an explicitly EMPTY base must fail closed, never fall back to the default store */
				}
				else if (strncmp(one.c_str(), "dedup=", 6) == 0)
				{
					if (!is_valid_number(strip_quote_pair(one.substr(6))))
						log_line(dir, "search INVALID value for dedup: '" + one + "'");
					else
						dedup = atof(one.c_str() + 6);
				}
				else if (strncmp(one.c_str(), "recency=", 8) == 0)
				{
					if (!is_valid_number(strip_quote_pair(one.substr(8))))
						log_line(dir, "search INVALID value for recency: '" + one + "'");
					else
						recency = atof(one.c_str() + 8);
				}
				else if (strncmp(one.c_str(), "recent=", 7) == 0)
				{
					if (!is_valid_int(one.substr(7)))
						log_line(dir, "search INVALID value for recent: '" + one + "'");
					else
						recent = atoi(one.c_str() + 7);
				}
				else if (strncmp(one.c_str(), "xbase=", 6) == 0)
				{
					xbasepath = strip_quote_pair(one.substr(6));
					if (xbasepath.empty())
						xbasepath = "<explicitly empty>";
				}
				else if (!one.empty())
					log_line(dir, "search IGNORED unknown option '" + one + "'"); /* a typo'd floor must never fail silently */
				pos = semi + 1;
			}
		}
		if (rnd < 0)
			rnd = 0;
		if (rnd > 100)
			rnd = 100;
		if (maxlines < 1)
			maxlines = 1;
		if (maxlines > 20)
			maxlines = 20;
		if (minscore < 0)
			minscore = 0;
		if (topmin < 0)
			topmin = 0;
		if (dedup < 0)
			dedup = 0;
		if (dedup > 1)
			dedup = 1;
		if (recency < 0)
			recency = 0;
		recent = (recent != 0) ? 1 : 0; /* the fallback is a boolean contract, not a number */
		return do_search(dir, arg, rnd, maxlines, minscore, topmin, basepath, dedup, recency, recent, xbasepath, qboost);
	}
	if (cmd == "relate")
	{
		std::string basepath;
		double floor_score = 0.0;
		if (argc >= 4)
		{ /* single packed options string: base= and floor= are meaningful here */
			std::string opts = argv[3];
			int pos = 0;
			while (pos < opts.size())
			{
				int semi = opts.find(';', pos);
				if (semi == std::string::npos)
					semi = opts.size();
				std::string one = trim_opt(opts.substr(pos, semi - pos));
				if (strncmp(one.c_str(), "base=", 5) == 0)
					basepath = strip_quote_pair(one.substr(5));
				else if (strncmp(one.c_str(), "floor=", 6) == 0)
				{
					if (!is_valid_number(one.substr(6)))
						log_line(dir, "relate INVALID value for floor: '" + one + "'");
					else
					{
						floor_score = atof(one.c_str() + 6);
						if (floor_score < 0.0 || floor_score > 1.0)
						{
							log_line(dir, "relate floor out of range [0,1]: '" + one + "' -> clamped");
							floor_score = (floor_score < 0.0) ? 0.0 : 1.0;
						}
					}
				}
				else if (!one.empty())
					log_line(dir, "relate IGNORED unknown option '" + one + "'");
				pos = semi + 1;
			}
		}
		return do_relate(dir, arg, basepath, floor_score);
	}
	log_line(dir, "UNKNOWN command '" + cmd + "' -> empty stdout"); /* a mis-call must never print into a captured LLM stream */
	return 1;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
	_setmode(_fileno(stdout), _O_BINARY); /* \n stays \n: no \r may land in the captured LLM stream (POSIX stdout is binary already) */
#endif
	try
	{
		return run_main(argc, argv); /* an uncaught exception would terminate with a stderr message INSIDE the captured LLM stream */
	}
	catch (...)
	{
		log_line(exe_dir(), "EXCEPTION - request aborted, empty stdout");
		return 1;
	}
}
