# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

#!/usr/bin/env python
# apply_review_fixes2.py -- final review-round fixes for bm25_rag.cpp:
# relate zero-score never passes the floor, jaccard correction guard
# (negation/number differences force 0), CRLF normalization in chunk_text,
# leading-whitespace trim + lone-asterisk strip in clean_line, four missing
# contraction forms, recent_memory_tail docstring aligned to the
# chronological emission. Run from the project ROOT.
import io

p = r'examples\rag\bm25_rag.cpp'
s = io.open(p, encoding='utf-8').read()

pairs = [
    # relate: a zero score never counts as above the floor (floor=0 must not turn zero overlap into a match)
    ('	bool above = score >= floor_score;',
     '	bool above = (score > 0.0) && (score >= floor_score); /* zero overlap is NEVER a match, even at floor 0 */'),
    # recent_memory_tail docstring: the emission is chronological, not newest-first
    (''' *         static reference text) as the recent-memory fallback, newest
 *         first, near-dup suppressed. returns "" when the memory range
 *         holds nothing usable. */''',
     ''' *         static reference text) as the recent-memory fallback, near-dup
 *         suppressed. the emitted order is CHRONOLOGICAL: the oldest of the
 *         collected memories first, the newest last, so the text reads like
 *         a recollection building up to the present. returns "" when the
 *         memory range holds nothing usable. */'''),
    # line_jaccard: corrections are never duplicates -- a negation or number
    # difference forces similarity to zero before the threshold speaks
    ('''	int inter = 0;
	for (std::map<std::string, char>::iterator it = sa.begin(); it != sa.end(); ++it)
		if (su.count(it->first))
			inter++;
	int uni = sa.size() + su.size() - inter;
	if (uni)
		return (double)inter / (double)uni;
	return 0.0; /* two term-less chunks share nothing: treating them as identical would NOOP new entries and let xbase drop chunks blindly */''',
     '''	/* a correction guard BEFORE the ratio: two facts that disagree in a
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
	return 0.0; /* two term-less chunks share nothing: treating them as identical would NOOP new entries and let xbase drop chunks blindly */'''),
    # chunk_text: normalize CRLF/CR to LF so paragraph detection works on argv text too
    ('''std::vector<chunk> chunk_text(std::string text)
{
	std::vector<chunk> chunks;
	int pos = 0;''',
     '''std::vector<chunk> chunk_text(std::string text)
{
	std::vector<chunk> chunks;
	/* normalize newlines FIRST: argv text arrives with CRLF, and the \\n\\n
	 * paragraph split below must see the same shape file content sees */
	{
		std::string flat;
		flat.reserve(text.size());
		for (int i = 0; i < text.size(); i++)
			if (text[i] != '\\r')
				flat += text[i];
		text = flat;
	}
	int pos = 0;'''),
    # clean_line: trim leading whitespace after the marker strip
    ('''	std::string r = s.substr(i);
	/* strip **bold** markers */''',
     '''	std::string r = s.substr(i);
	int rb = r.find_first_not_of(" \\t");
	if (rb == std::string::npos)
		return ""; /* nothing but whitespace between the markers */
	if (rb > 0)
		r = r.substr(rb); /* "1.   text" keeps its extra spaces after the marker strip: drop them */
	/* strip **bold** markers */'''),
    # clean_line: strip ALL asterisks (***bold-italic*** left a dangling *)
    ('''	/* strip **bold** markers */
	std::string c;
	for (int j = 0; j < r.size(); j++)
	{
		if (j + 1 < r.size() && r[j] == '*' && r[j + 1] == '*')
		{
			j++;
			continue;
		}
		c += r[j];
	}''',
     '''	/* strip markdown asterisks in every shape: **, *** and lone * */
	std::string c;
	for (int j = 0; j < r.size(); j++)
	{
		if (r[j] == '*')
			continue;
		c += r[j];
	}'''),
    # expand_contractions: the four missing negation forms
    ('''		{"couldn't", "could not"},
		{"shouldn't", "should not"},
		{"wouldn't", "would not"},
	};''',
     '''		{"couldn't", "could not"},
		{"shouldn't", "should not"},
		{"wouldn't", "would not"},
		{"mustn't", "must not"},
		{"needn't", "need not"},
		{"mightn't", "might not"},
		{"shan't", "shall not"},
	};'''),
]

for old, new in pairs:
    assert old in s, 'MISSING: ' + old[:70]
    s = s.replace(old, new)

io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('applied', len(pairs), 'edits')
