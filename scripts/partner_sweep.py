#!/usr/bin/env python3
# partner_sweep.py -- one experiment of the cycle tuner's partner-draw
# formula sweep. Runs tools/genome_cycle_sort.exe --reps times with the
# given --partner mode (seed-locked tool: every run sorts the identical
# probe pool against the identical reference; score differences come from
# the partner formula and timing noise alone), parses the improvement
# percent of the best sorted genome over the unchanged reference, and
# appends one CSV row: mode, rep scores, median, mean.
#
# The mutation context stays fixed (passed through verbatim) so the sweep
# isolates the partner formula: --mutate/--sex/--mmax default to the
# hill-climb baseline (40/80/40).
#
# Usage:
#   python partner_sweep.py --partner pow3 [--rounds 100] [--reps 3]
#       [--shape 1024x3072] [--mutate 40] [--sex 80] [--mmax 40]
#       [--log partner_sweep_results.csv]

import argparse
import os
import re
import statistics
import subprocess
import sys
import time

TOOL_PATH = os.path.join("tools", "genome_cycle_sort.exe")
DLL_PATH = os.path.join("bin", "v7_threading.dll")
DB_PATH = os.path.join("weights", "databases", "qwen3_0.6b", "lz4ht.db")

LEGAL_MODES = ["pow1", "pow2", "pow3", "pow4", "pow6", "tour2", "tour3", "tour4", "exp2", "exp3", "half"]

# "-- 120.8% faster than the reference" / "-- 3.4% slower than the reference"
IMPROVEMENT_RE = re.compile(r"--\s+(-?[0-9.]+)% (faster|slower) than the reference")
BEST_RATE_RE = re.compile(r"best genome: slot \d+ \(probe \d+\) at ([0-9.]+) GB/s")


def run_once(mode, rounds, shape, mutate, sex, mmax, tag):
	"""One genome_cycle_sort run; returns (improvement_percent, best_gbs)."""
	command = [
		TOOL_PATH,
		DLL_PATH,
		"--db", DB_PATH,
		"--shape", shape,
		"--rounds", str(rounds),
		"--partner", mode,
		"--mutate-percentage", str(mutate),
		"--sex-influence", str(sex),
		"--mutations-max", str(mmax),
	]
	started = time.time()
	process = subprocess.run(command, capture_output=True, text=True)
	elapsed = time.time() - started
	if process.returncode != 0:
		sys.exit("partner_sweep: tool run failed (exit %d) for partner %s\n%s" % (process.returncode, mode, process.stdout[-2000:]))

	improvement = None
	best_gbs = None
	for match in IMPROVEMENT_RE.finditer(process.stdout):
		improvement = float(match.group(1))
		if match.group(2) == "slower":
			improvement = -improvement
	rate_match = BEST_RATE_RE.search(process.stdout)
	if rate_match:
		best_gbs = float(rate_match.group(1))
	if improvement is None:
		sys.exit("partner_sweep: no improvement line in tool output for partner %s\n%s" % (mode, process.stdout[-2000:]))

	print("  %-3s partner %-5s -> %+.1f%%  (%.1f GB/s, %.0f s)"
	      % (tag, mode, improvement, best_gbs if best_gbs else 0.0, elapsed))
	sys.stdout.flush()
	return improvement, best_gbs if best_gbs else 0.0


def main():
	parser = argparse.ArgumentParser(description="one partner-formula experiment: reps runs, median improvement into the CSV log")
	parser.add_argument("--partner", required=True)
	parser.add_argument("--rounds", type=int, default=100)
	parser.add_argument("--reps", type=int, default=3)
	parser.add_argument("--shape", default="1024x3072")
	parser.add_argument("--mutate", type=int, default=40)
	parser.add_argument("--sex", type=int, default=80)
	parser.add_argument("--mmax", type=int, default=40)
	parser.add_argument("--log", default="partner_sweep_results.csv")
	args = parser.parse_args()

	if args.partner not in LEGAL_MODES:
		sys.exit("partner_sweep: unknown partner mode '%s' (legal: %s)" % (args.partner, " ".join(LEGAL_MODES)))
	if not os.path.exists(TOOL_PATH):
		sys.exit("partner_sweep: %s not found -- run build.bat first" % TOOL_PATH)

	scores = []
	for rep in range(args.reps):
		improvement, best_gbs = run_once(args.partner, args.rounds, args.shape, args.mutate, args.sex, args.mmax, "r%d" % (rep + 1))
		scores.append(improvement)

	median_score = statistics.median(scores)
	mean_score = statistics.mean(scores)
	with open(args.log, "a") as log_file:
		log_file.write("%s,%s,%.2f,%.2f\n"
			       % (args.partner, ";".join("%.2f" % s for s in scores), median_score, mean_score))
	print("partner %-5s done: median %+.2f%% mean %+.2f%% (%d reps x %d rounds, logged to %s)"
	      % (args.partner, median_score, mean_score, args.reps, args.rounds, args.log))


if __name__ == "__main__":
	main()
