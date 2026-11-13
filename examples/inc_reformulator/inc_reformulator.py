# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

# inc_reformulator.py -- comment reformulation via the v7 HTTP server.
#
# WHAT IT DOES (owner directive): scan a .inc file for comments containing
# " -- ", send every one through the v7 server (/v1/chat/completions) with the
# sandwich prompt, check the reformulated answer is CLEAN of " -- ", "(" and
# ")", and only then edit the comment IN PLACE. Unclean answers are skipped
# (the next comment gets processed). Every attempt is logged with the .inc
# file, line number, original comment, reformulated comment, verdict/reason
# and the model's full reasoning, so the batch can be assessed later and bad
# spots fixed.
#
# The script prints EVERY step it performs to stdout, unfiltered: which file
# it scans, which comment it found, when it talks to the server, the verdict,
# and the applied replacement.
#
# The server (start_server.bat, default port 8080) splits the model output
# server-side: reasoning_content carries the think block, content carries ONLY
# the answer. Each reformulation runs as an independent turn: /v1/chat/reset
# forces a fresh session first.
#
# usage:  python inc_reformulator.py [target.inc] [log.txt]
# default target: examples/inc_reformulator/test_reformulator.inc

import os
import sys
import urllib.request
import urllib.error
import json

# live progress output: stdout must flush per print, never buffer
sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

SERVER_URL = os.environ.get('V7_SERVER_URL', 'http://127.0.0.1:8080')
DEFAULT_TARGET = os.path.join(SCRIPT_DIR, 'test_reformulator.inc')
DEFAULT_LOG = os.path.join(SCRIPT_DIR, 'reformulate_log.txt')

MAX_TOKENS = 3000


def mask_quotes(line):
	# Quote-aware mask: replace every char inside '...'/"..."/`...` with 'x',
	# preserving length and indices (mirrors check_asm_comments.py). Backquoted
	# strings process backslash escapes; a backslash inside ' or " strings is
	# a literal byte and does not hide a quote.
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


def build_prompt(comment):
	# The owner's mandated sandwich shape: instruction before AND after the
	# comment, "now the comment:" marks the payload position, and the append
	# demands the bare reformulated comment as the answer.
	return ('Reformulate this comment from a assembly codebase without information loss to only describe what the code does, '
	        'the new comment MUST not contain "(" "--" ")", write proper english, now the comment:>>>' + comment + '<<<'
	        'Reformulate this comment from a assembly codebase to only describe what the code does, '
	        'the new comment MUST not contain "(" "--" ")", now output the reformulated comment, only output the comment, nothing else, make certain no information is lost, write proper english keep close to the original comment!')


HDR_LOCAL = {'X-V7-Local': '1'}


def server_reset():
	# /v1/chat/reset is state-changing and requires the X-V7-Local: 1 header
	urllib.request.urlopen(urllib.request.Request(SERVER_URL + '/v1/chat/reset', method='POST', data=b'', headers=HDR_LOCAL), timeout=30).read()


def server_chat(prompt):
	body = json.dumps({
		'messages': [{'role': 'user', 'content': prompt}],
		'max_tokens': MAX_TOKENS,
		'enable_thinking': True,
	}).encode('utf-8')
	req = urllib.request.Request(SERVER_URL + '/v1/chat/completions', method='POST',
	                             data=body, headers={'Content-Type': 'application/json'})
	with urllib.request.urlopen(req, timeout=600) as resp:
		payload = json.loads(resp.read().decode('utf-8'))
	message = payload['choices'][0]['message']
	return message.get('content') or '', message.get('reasoning_content') or ''


def strip_code_fences(answer):
	# Models often wrap a one-line answer in markdown code fences; the fences
	# are output formatting, never part of the comment text. Strip the outer
	# fence block and any leading fence tag before the cleanliness check.
	text = answer.strip()
	if text.startswith('```'):
		nl = text.find('\n')
		text = text[nl + 1:] if nl >= 0 else ''
		end = text.rfind('```')
		if end >= 0:
			text = text[:end]
	return text.strip()


def is_clean(answer, comment):
	# The comment must be free of the dashdash, the parentheses and the
	# sandwich echo markers, must be ONE line (a comment cannot span lines),
	# must not be an echo of the instruction itself, and must not shrink the
	# content away (information preservation).
	if not answer:
		return False, 'empty answer'
	if ' -- ' in answer:
		return False, 'contains " -- "'
	if '(' in answer or ')' in answer:
		return False, 'contains parentheses'
	if '>>>' in answer or '<<<' in answer:
		return False, 'echoes sandwich markers'
	if '\n' in answer or '\r' in answer:
		return False, 'answer spans lines'
	low = answer.lower()
	if 'reformulat' in low or 'must not contain' in low or 'assembly code' in low:
		return False, 'echoes the instruction'
	tokens = answer.split()
	if len(tokens) >= 3 and all(t == tokens[0] for t in tokens):
		return False, 'degenerate repetition (same token %d times)' % len(tokens)
	# content grounding: the reformulation must retain the original's content
	# words. A token of 2 chars or less carries no meaning; an answer that
	# shares fewer than two significant words with the comment is a
	# placeholder, not a reformulation.
	orig_words = [w.lower() for w in comment.split() if len(w) > 2]
	shared = [w for w in orig_words if w in low]
	if len(orig_words) >= 4 and len(shared) < 2:
		return False, 'not grounded in the original (%d shared words)' % len(shared)
	if len(answer) < 0.4 * len(comment):
		return False, 'information lost (answer too short)'
	if len(answer) > 3 * len(comment) + 60:
		return False, 'answer too long'
	return True, 'ok'


def log_entry(logf, target, lineno, verdict, original, reform, reason, reasoning):
	logf.write('=== %s : %s : line %d ===\n' % (verdict, target, lineno))
	logf.write('ORIGINAL : %s\n' % original)
	logf.write('REFORM   : %s\n' % reform)
	if reason:
		logf.write('REASON   : %s\n' % reason)
	if reasoning:
		logf.write('REASONING:\n%s\n' % reasoning)
	logf.write('\n')


def process_file(target, log_path):
	print('--- scanning %s ---' % target)
	try:
		with open(target, 'r', encoding='utf-8', errors='replace', newline='') as f:
			lines = f.readlines()
	except OSError as exc:
		print('cannot read %s: %s' % (target, exc))
		return 1

	found = 0
	applied = 0
	skipped = 0
	failed = 0
	with open(log_path, 'w', encoding='utf-8', newline='') as logf:
		logf.write('# inc reformulator log (server: %s)\n' % SERVER_URL)
		logf.write('# target: %s\n\n' % target)
		for lineno, raw in enumerate(lines, start=1):
			masked = mask_quotes(raw)
			semi = masked.find(';')
			if semi < 0:
				continue
			body = raw.rstrip('\r\n')
			tail = raw[len(body):]
			comment_zone = body[semi + 1:]
			zone_stripped = comment_zone.lstrip()
			if ' -- ' not in zone_stripped:
				continue
			comment = zone_stripped.strip()
			if not comment:
				continue
			found += 1
			print('[found] %s line %d: %s' % (os.path.basename(target), lineno, comment))
			print('[server] reset + chat turn (max_tokens=%d) ...' % MAX_TOKENS)
			verdict = 'APPLIED'
			reform = ''
			reason = ''
			reasoning = ''
			try:
				server_reset()
				print('[sent] ORIGINAL comment: %s' % comment)
				answer, reasoning = server_chat(build_prompt(comment))
				self_str = strip_code_fences(answer)
				reform = self_str
				print('[recv] REFORMULATED comment: %s' % reform)
				clean, reason = is_clean(reform, comment)
				if clean:
					lead = len(comment_zone) - len(zone_stripped)
					new_body = body[:semi + 1] + ' ' * lead + reform
					new_line = new_body + tail
					if lines[lineno - 1] != new_line:
						lines[lineno - 1] = new_line
						print('[applied] line %d rewritten' % lineno)
					else:
						verdict = 'SKIPPED'
						reason = 'answer identical to original'
					print('done: answer ok -> new comment: %s' % reform)
				else:
					verdict = 'SKIPPED'
					print('[skipped] line %d: %s (answer: %s)' % (lineno, reason, reform))
			except Exception as exc:
				verdict = 'ERROR'
				reason = '%s: %s' % (type(exc).__name__, exc)
				print('[error] line %d: %s' % (lineno, reason))
			if verdict == 'APPLIED':
				applied += 1
			elif verdict == 'SKIPPED':
				skipped += 1
			else:
				failed += 1
			log_entry(logf, target, lineno, verdict, comment, reform, reason, reasoning)

	if applied:
		try:
			with open(target, 'w', encoding='utf-8', newline='') as f:
				f.writelines(lines)
			print('--- wrote %d edited line(s) back to %s ---' % (applied, target))
		except OSError as exc:
			print('cannot write %s: %s' % (target, exc))
			return 1

	print('--- summary: %d comments, %d applied, %d skipped, %d errored ---' % (found, applied, skipped, failed))
	print('--- log written: %s ---' % log_path)
	return 0


def main():
	target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
	log_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_LOG
	print('inc reformulator -- server %s' % SERVER_URL)
	print('target: %s' % target)
	print('log:    %s' % log_path)
	if not os.path.isfile(target):
		print('target not found!')
		return 1
	if not os.path.isfile(os.path.join(ROOT_DIR, 'bin', 'v7_threading.dll')):
		print('not in the project tree!')
		return 1
	print('checking server reachability ...')
	try:
		server_reset()
	except (urllib.error.URLError, urllib.error.HTTPError, OSError):
		print('server unreachable at %s -- start it with start_server.bat' % SERVER_URL)
		return 1
	print('server is up.')
	return process_file(target, log_path)


if __name__ == '__main__':
	sys.exit(main())
