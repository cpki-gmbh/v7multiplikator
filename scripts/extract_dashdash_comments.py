# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

# Extracts every comment containing " -- " from the .inc files under src/
# into dashdash_comments.txt (read-only tool: the .inc files are never touched).
#
# Comment rules follow src/check_asm_comments.py: the ';' that starts a comment
# is the FIRST ';' outside string literals, and everything from there to the
# end of the line is the comment. Entries are written as
#     path|line|comment text
# so a reformulated comment can later be mapped back to its source line.
#
# usage:  python extract_dashdash_comments.py [output_txt]

import os
import glob
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, 'dashdash_comments.txt')
SOURCE_ROOTS = [os.path.join(SCRIPT_DIR, 'src')]


def mask_quotes(line):
	# Quote-aware mask: replace every char inside '...'/"..."/`...` with 'x',
	# preserving length and indices. Backquoted strings process backslash
	# escapes (NASM rule); a backslash inside ' or " strings is a literal byte.
	out = []
	i = 0
	n = len(line)
	while i < n:
		c = line[i]
		if c in ('"', "'", '`'):
			j = i + 1
			while j < n:
				if c == '`' and line[j] == '\\' and j + 1 < n:
					j += 2
					continue
				if line[j] == c:
					break
				j += 1
			if j >= n:
				out.append(c)      # unpaired quote: keep it visible, scan on
				i += 1
			else:
				out.append('x' * (j - i + 1))
				i = j + 1
		else:
			out.append(c)
			i += 1
	return ''.join(out)


def extract_comment(raw):
	masked = mask_quotes(raw)
	semi = masked.find(';')
	if semi < 0:
		return None
	return raw[semi + 1:].strip()


def main():
	inc_files = []
	for root in SOURCE_ROOTS:
		inc_files.extend(glob.glob(os.path.join(root, '**', '*.inc'), recursive=True))
	inc_files.sort()

	entries = []
	per_file_counts = {}

	for path in inc_files:
		rel = os.path.relpath(path, SCRIPT_DIR)
		count = 0
		with open(path, 'r', encoding='utf-8', errors='replace') as f:
			for lineno, raw in enumerate(f, start=1):
				comment = extract_comment(raw)
				if comment is None:
					continue
				if ' -- ' in comment:
					entries.append((rel, lineno, comment))
					count += 1
		if count:
			per_file_counts[rel] = count

	with open(OUT_PATH, 'w', encoding='utf-8', newline='\n') as out:
		out.write('# dashdash comments extracted from .inc files (comment text after \';\', line contains " -- ")\n')
		out.write('# format: path|line|comment\n')
		out.write('# total: %d\n' % len(entries))
		for rel, lineno, comment in entries:
			out.write('%s|%d|%s\n' % (rel, lineno, comment))

	print('scanned %d .inc files' % len(inc_files))
	print('found %d comments containing " -- "' % len(entries))
	for rel in sorted(per_file_counts, key=lambda r: (-per_file_counts[r], r)):
		print('  %-45s %d' % (rel, per_file_counts[rel]))
	print('written: %s' % OUT_PATH)


if __name__ == '__main__':
	main()
