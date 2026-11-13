# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

#!/usr/bin/env python
# apply_review_fixes3.py -- final parse-site fixes against the CURRENT text:
# quoted numeric values accepted, explicitly empty base=/xbase= sentinels.
# Run from the project ROOT.
import io

p = r'examples\rag\bm25_rag.cpp'
s = io.open(p, encoding='utf-8').read()

pairs = [
    ('''				else if (strncmp(one.c_str(), "minscore=", 9) == 0)
				{
					if (!is_valid_number(one.substr(9)))''',
     '''				else if (strncmp(one.c_str(), "minscore=", 9) == 0)
				{
					if (!is_valid_number(strip_quote_pair(one.substr(9))))'''),
    ('''				else if (strncmp(one.c_str(), "topmin=", 7) == 0)
				{
					if (!is_valid_number(one.substr(7)))''',
     '''				else if (strncmp(one.c_str(), "topmin=", 7) == 0)
				{
					if (!is_valid_number(strip_quote_pair(one.substr(7))))'''),
    ('''				else if (strncmp(one.c_str(), "base=", 5) == 0)
					basepath = strip_quote_pair(one.substr(5));
				else if (strncmp(one.c_str(), "dedup=", 6) == 0)
				{
					if (!is_valid_number(one.substr(6)))''',
     '''				else if (strncmp(one.c_str(), "base=", 5) == 0)
				{
					basepath = strip_quote_pair(one.substr(5));
					if (basepath.empty())
						basepath = "<explicitly empty>"; /* an explicitly EMPTY base must fail closed, never fall back to the default store */
				}
				else if (strncmp(one.c_str(), "dedup=", 6) == 0)
				{
					if (!is_valid_number(strip_quote_pair(one.substr(6))))'''),
    ('''				else if (strncmp(one.c_str(), "recency=", 8) == 0)
				{
					if (!is_valid_number(one.substr(8)))''',
     '''				else if (strncmp(one.c_str(), "recency=", 8) == 0)
				{
					if (!is_valid_number(strip_quote_pair(one.substr(8))))'''),
    ('''				else if (strncmp(one.c_str(), "xbase=", 6) == 0)
					xbasepath = strip_quote_pair(one.substr(6));''',
     '''				else if (strncmp(one.c_str(), "xbase=", 6) == 0)
				{
					xbasepath = strip_quote_pair(one.substr(6));
					if (xbasepath.empty())
						xbasepath = "<explicitly empty>";
				}'''),
]

for old, new in pairs:
    assert old in s, 'MISSING: ' + old[:80]
    s = s.replace(old, new)

io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('applied', len(pairs), 'edits')
