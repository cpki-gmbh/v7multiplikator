# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

# Builds v7_pipe_reformulate_batch.bat: one flat "v7_pipe.exe ask" line per
# sampled comment from dashdash_comments.txt (dumb-bat rule: one action per
# line, no loops, no variables). Sampling is stratified across comment length
# so the experiment can report how quality degrades with comment size.
# Comments containing '%' or '"' are skipped: they cannot ride a cmd double
# quoted argument without breaking argv parsing.
#
# usage:  python gen_reformulate_bat.py [count_per_bucket]

import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(SCRIPT_DIR, 'dashdash_comments.txt')
OUT_PATH = os.path.join(SCRIPT_DIR, 'v7_pipe_reformulate_batch.bat')
PER_BUCKET = int(sys.argv[1]) if len(sys.argv) > 1 else 4

# long comments are truncated at capped_len: generation cost grows with the
# prompt and the report wants short, medium and long, not giant one-offs
CAP_LONG = 330

BUCKETS = [
	('short  x<60', lambda n: n < 60),
	('medium 60<=x<150', lambda n: 60 <= n < 150),
	('long  x>=150', lambda n: n >= 150),
]


def load_entries():
	entries = []
	with open(SRC_PATH, 'r', encoding='utf-8') as f:
		for raw in f:
			raw = raw.rstrip('\n')
			if raw.startswith('#'):
				continue
			parts = raw.split('|', 2)
			if len(parts) != 3:
				continue
			rel, lineno, comment = parts
			if '%' in comment:
				continue
			if '"' in comment:
				continue
			if '--' not in comment:
				continue
			entries.append((rel, lineno, comment))
	return entries


def prompt_for(comment):
	return ('reformulate this comment in a assembly code to only describe behaviour, '
	        'and to not contain \\"(\\" \\"--\\" \\")\\" now the comment:>>>' + comment + '<<<'
	        'reformulate this comment in a assembly code to only describe behaviour, '
	        'and to not contain \\"(\\" \\"--\\" \\")\\"')
	return entries


def main():
	entries = load_entries()
	print('eligible comments (no percent sign and no double quote, contains --): %d' % len(entries))

	picks = []
	for label, pred in BUCKETS:
		bucket = [e for e in entries if pred(len(e[2]))]
		if not bucket:
			print('bucket %s: no candidates' % label)
			continue
		# spread over the file: take from beginning, middle and end
		stride = max(1, len(bucket) // (PER_BUCKET * 3))
		chosen = []
		start_offsets = [0, len(bucket) // 3, (len(bucket) * 2) // 3]
		for start in start_offsets:
			idx = start
			while len(chosen) < PER_BUCKET and idx < len(bucket):
				cand = bucket[min(idx, len(bucket) - 1)]
				if cand not in chosen:
					chosen.append(cand)
				idx += stride
		for cand in chosen:
			if len(cand[2]) > CAP_LONG:
				rel, lineno, comment = cand
				cand = (rel, lineno, comment[:CAP_LONG].rstrip())
				print('truncated to %d: %s:%s' % (CAP_LONG, rel, lineno))
			picks.append((label, cand))
		print('bucket %-18s -> %d picked' % (label, len(chosen)))

	with open(OUT_PATH, 'w', encoding='utf-8', newline='\r\n') as out:
		out.write('@echo off\n')
		out.write('setlocal\n')
		out.write('set THIS_DIR=%~dp0\n')
		out.write('set V7_DLL=%THIS_DIR%bin\\v7_threading.dll\n')
		out.write('set V7_DB=%THIS_DIR%weights\\databases\\qwen3_0.6b\\lz4ht.db\n')
		out.write('\n')
		idx = 0
		for label, (rel, lineno, comment) in picks:
			idx = idx + 1
			# the echo headers stay cmd-safe: no pipe, no redirect, no bare comment
			# text (unquoted < > | & in an echo line are cmd operators)
			token = 'SUB%d' % idx
			out.write('echo === SAMPLE %s of 12 ===\n' % token)
			out.write('"%THIS_DIR%bin\\v7_pipe.exe" ask "' + prompt_for(comment) + '"\n')
			out.write('echo === END %s ===\n' % token)
			out.write('\n')
			print('%s | %s line %s | %s' % (token, rel, lineno, comment[:110]))

	print('written: %s (%d test lines)' % (OUT_PATH, len(picks)))


if __name__ == '__main__':
	main()
