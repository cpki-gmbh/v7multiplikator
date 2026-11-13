# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

#!/usr/bin/env python
# apply_review_fixes.py -- batched mechanical edits for bm25_rag.cpp from the
# review round: quoted numeric option values, remember invalid-dedup logging,
# explicitly-empty base/xbase sentinels at the parse sites, NUL-byte strip.
# Run from the project ROOT.
import io

p = r'examples\rag\bm25_rag.cpp'
s = io.open(p, encoding='utf-8').read()

pairs = [
    # quoted numeric option values: strip one quote pair before validation
    ('minscore = is_valid_number(one.substr(9)) ? atof(one.c_str() + 9) : minscore;',
     'minscore = is_valid_number(strip_quote_pair(one.substr(9))) ? atof(one.c_str() + 9) : minscore;'),
    ('topmin = is_valid_number(one.substr(7)) ? atof(one.c_str() + 7) : topmin;',
     'topmin = is_valid_number(strip_quote_pair(one.substr(7))) ? atof(one.c_str() + 7) : topmin;'),
    ('dedup = is_valid_number(one.substr(6)) ? atof(one.c_str() + 6) : dedup;',
     'dedup = is_valid_number(strip_quote_pair(one.substr(6))) ? atof(one.c_str() + 6) : dedup;'),
    ('recency = is_valid_number(one.substr(8)) ? atof(one.c_str() + 8) : recency;',
     'recency = is_valid_number(strip_quote_pair(one.substr(8))) ? atof(one.c_str() + 8) : recency;'),
    # search: explicitly empty base sentinel
    ('''				else if (strncmp(one.c_str(), "base=", 5) == 0)
					basepath = strip_quote_pair(one.substr(5));
				else if (strncmp(one.c_str(), "dedup=", 6) == 0)
				{
					if (!is_valid_number(strip_quote_pair(one.substr(6))))''',
     '''				else if (strncmp(one.c_str(), "base=", 5) == 0)
				{
					basepath = strip_quote_pair(one.substr(5));
					if (basepath.empty())
						basepath = "<explicitly empty>"; /* an explicitly EMPTY base must fail closed, never fall back to the default store */
				}
				else if (strncmp(one.c_str(), "dedup=", 6) == 0)
				{
					if (!is_valid_number(strip_quote_pair(one.substr(6))))'''),
    # search: explicitly empty xbase sentinel
    ('''				else if (strncmp(one.c_str(), "xbase=", 6) == 0)
					xbasepath = strip_quote_pair(one.substr(6));''',
     '''				else if (strncmp(one.c_str(), "xbase=", 6) == 0)
				{
					xbasepath = strip_quote_pair(one.substr(6));
					if (xbasepath.empty())
						xbasepath = "<explicitly empty>";
				}'''),
    # remember: explicitly empty base rejects the write outright
    ('''			else if (strncmp(one.c_str(), "base=", 5) == 0)
					basepath = strip_quote_pair(one.substr(5));
				else if (!one.empty())
					log_line(dir, "remember IGNORED unknown option''',
     '''			else if (strncmp(one.c_str(), "base=", 5) == 0)
				{
					basepath = strip_quote_pair(one.substr(5));
					if (basepath.empty())
					{
						log_line(dir, "remember base= explicitly EMPTY -> no store, no write");
						return 1;
					}
				}
				else if (!one.empty())
					log_line(dir, "remember IGNORED unknown option'''),
    # remember: invalid dedup logged (silently disabling dedup is the unsafe default)
    ('''			if (strncmp(one.c_str(), "dedup=", 6) == 0)
				dedup = is_valid_number(one.substr(6)) ? atof(one.c_str() + 6) : dedup;''',
     '''			if (strncmp(one.c_str(), "dedup=", 6) == 0)
			{
				if (!is_valid_number(strip_quote_pair(one.substr(6))))
					log_line(dir, "remember INVALID value for dedup: '" + one + "' (dedup stays off -- duplicates will NOT be suppressed)");
				else
					dedup = atof(one.c_str() + 6);
			}'''),
    # NUL bytes stripped alongside \r: a NUL would truncate every printf(%s) emission downstream
    ('''	for (size_t i = 0; i < tmp.size(); i++)
		if (tmp[i] != '\\r')
			flat += tmp[i];''',
     '''	for (size_t i = 0; i < tmp.size(); i++)
		if (tmp[i] != '\\r' && tmp[i] != 0)
			flat += tmp[i];'''),
]

for old, new in pairs:
    assert old in s, 'MISSING: ' + old[:70]
    s = s.replace(old, new)

io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('applied', len(pairs), 'edits')
