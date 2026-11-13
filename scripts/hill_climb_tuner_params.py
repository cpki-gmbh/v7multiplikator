#!/usr/bin/env python3
# hill_climb_tuner_params.py -- stochastic hill climber over the genetic
# parameters of the C++ cycle tuner (tools/genome_cycle_sort.exe).
#
# One candidate is a (mutate-percentage, sex-influence, mutations-max)
# triple. Its score is the MEDIAN improvement percent of the best sorted
# genome over the unchanged reference, parsed from --reps independent
# tool runs (--rounds each). The tool is seed-locked (42), so every run
# sorts the identical probe pool against the identical reference -- score
# differences between candidates come from the parameters alone; the
# median over reps absorbs timing noise.
#
# The climb starts from the best measured combo so far
# (mutate 40, sex 80, mutations-max 40 -- see the sweep logs), proposes a
# random legal neighbor each step (one axis moved by +-5 or +-10), and
# accepts strict improvements only. After --stall-limit rejected
# proposals it jumps to a random legal point (seeded), so the climber
# escapes narrow peaks. Every evaluation is appended to the CSV log.
#
# Usage:
#   python hill_climb_tuner_params.py [--rounds 100] [--reps 2]
#       [--steps 20] [--stall-limit 6] [--shape 1024x3072]
#       [--start-mutate 40] [--start-sex 80] [--start-mmax 40]
#       [--seed 42] [--log hill_climb_results.csv]
#
# Legality windows (never clamped beyond these -- the feature space stays
# whole): mutate-percentage [0,100], sex-influence [0,100],
# mutations-max [1,104] (104 = the full 22+41+41 field space).

import argparse
import os
import random
import re
import statistics
import subprocess
import sys
import time

TOOL_PATH = os.path.join("tools", "genome_cycle_sort.exe")
DLL_PATH = os.path.join("bin", "v7_threading.dll")
DB_PATH = os.path.join("weights", "databases", "qwen3_0.6b", "lz4ht.db")

LEGAL_MUTATE = (0, 100)
LEGAL_SEX = (0, 100)
LEGAL_MMAX = (1, 104)

# "-- 120.8% faster than the reference" / "-- 3.4% slower than the reference"
IMPROVEMENT_RE = re.compile(r"--\s+(-?[0-9.]+)% (faster|slower) than the reference")
BEST_RATE_RE = re.compile(r"best genome: slot \d+ \(probe \d+\) at ([0-9.]+) GB/s")


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def run_once(params, rounds, shape, tag):
    """One genome_cycle_sort run; returns (improvement_percent, best_gbs)."""
    command = [
        TOOL_PATH,
        DLL_PATH,
        "--db", DB_PATH,
        "--shape", shape,
        "--rounds", str(rounds),
        "--mutate-percentage", str(params["mutate"]),
        "--sex-influence", str(params["sex"]),
        "--mutations-max", str(params["mmax"]),
    ]
    started = time.time()
    process = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.time() - started
    if process.returncode != 0:
        sys.exit("hill_climb: tool run failed (exit %d) for %r\n%s" % (process.returncode, params, process.stdout[-2000:]))

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
        sys.exit("hill_climb: no improvement line in tool output for %r\n%s" % (params, process.stdout[-2000:]))

    print("  %-4s mutate %3d sex %3d mmax %3d -> %+.1f%%  (%.1f GB/s, %.0f s)"
          % (tag, params["mutate"], params["sex"], params["mmax"], improvement, best_gbs if best_gbs else 0.0, elapsed))
    sys.stdout.flush()
    return improvement, best_gbs if best_gbs else 0.0


def evaluate(params, rounds, reps, shape, log_file):
    """Median improvement over --reps runs; appends one CSV row."""
    scores = []
    rates = []
    for rep in range(reps):
        improvement, best_gbs = run_once(params, rounds, shape, "r%d" % (rep + 1))
        scores.append(improvement)
        rates.append(best_gbs)
    median_score = statistics.median(scores)
    mean_score = statistics.mean(scores)
    log_file.write("%d,%d,%d,%s,%.2f,%.2f\n"
                   % (params["mutate"], params["sex"], params["mmax"],
                      ";".join("%.2f" % s for s in scores), median_score, mean_score))
    log_file.flush()
    return median_score, mean_score, rates


def propose_neighbor(params, rng):
    """One axis moved by a random legal step; the window stays whole."""
    axis = rng.choice(["mutate", "sex", "mmax"])
    step = rng.choice([-10, -5, 5, 10])
    candidate = dict(params)
    if axis == "mutate":
        candidate["mutate"] = clamp(candidate["mutate"] + step, LEGAL_MUTATE[0], LEGAL_MUTATE[1])
    elif axis == "sex":
        candidate["sex"] = clamp(candidate["sex"] + step, LEGAL_SEX[0], LEGAL_SEX[1])
    else:
        candidate["mmax"] = clamp(candidate["mmax"] + step, LEGAL_MMAX[0], LEGAL_MMAX[1])
    return candidate


def random_point(rng):
    """A fully random legal point for stall restarts."""
    return {
        "mutate": rng.randint(LEGAL_MUTATE[0], LEGAL_MUTATE[1]),
        "sex": rng.randint(LEGAL_SEX[0], LEGAL_SEX[1]),
        "mmax": rng.randint(LEGAL_MMAX[0], LEGAL_MMAX[1]),
    }


def params_text(params):
    return "mutate %3d sex %3d mmax %3d" % (params["mutate"], params["sex"], params["mmax"])


def save_state(state_path, best, best_score, best_mean, step_index, steps, stall, stall_limit, current, note):
    """Rewrite the state file so an aborted run still shows the best."""
    state = (
        "# hill climber state (rewritten after every iteration -- read anytime)\n"
        "updated: %s\n"
        "step: %d/%d\n"
        "stall: %d/%d\n"
        "best: %s\n"
        "best_median: %+.2f%%\n"
        "best_mean: %+.2f%%\n"
        "current: %s (%s)\n"
    ) % (time.strftime("%Y-%m-%d %H:%M:%S"), step_index, steps, stall, stall_limit,
         params_text(best), best_score, best_mean, params_text(current), note)
    with open(state_path, "w") as state_file:
        state_file.write(state)


def main():
    parser = argparse.ArgumentParser(description="hill climb the cycle tuner's genetic parameters")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--stall-limit", type=int, default=6)
    parser.add_argument("--shape", default="1024x3072")
    parser.add_argument("--start-mutate", type=int, default=40)
    parser.add_argument("--start-sex", type=int, default=80)
    parser.add_argument("--start-mmax", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", default="hill_climb_results.csv")
    parser.add_argument("--state", default="hill_climb_state.txt")
    args = parser.parse_args()

    if not os.path.exists(TOOL_PATH):
        sys.exit("hill_climb: %s not found -- run build.bat first" % TOOL_PATH)

    rng = random.Random(args.seed)
    log_file = open(args.log, "a")
    log_file.write("# hill climb start %s: rounds %d reps %d steps %d seed %d\n"
                   % (time.strftime("%Y-%m-%d %H:%M:%S"), args.rounds, args.reps, args.steps, args.seed))
    log_file.write("# mutate,sex,mmax,rep_scores,median,mean\n")

    best = {
        "mutate": clamp(args.start_mutate, LEGAL_MUTATE[0], LEGAL_MUTATE[1]),
        "sex": clamp(args.start_sex, LEGAL_SEX[0], LEGAL_SEX[1]),
        "mmax": clamp(args.start_mmax, LEGAL_MMAX[0], LEGAL_MMAX[1]),
    }

    print("hill climb start: %s, %d rounds, %d reps/eval, %d steps, stall limit %d"
          % (params_text(best), args.rounds, args.reps, args.steps, args.stall_limit))
    best_score, best_mean, _ = evaluate(best, args.rounds, args.reps, args.shape, log_file)
    print("start score: %+.2f%% median (%.2f%% mean)" % (best_score, best_mean))
    save_state(args.state, best, best_score, best_mean, 0, args.steps, 0, args.stall_limit, best, "baseline")

    stall = 0
    for step_index in range(1, args.steps + 1):
        candidate = propose_neighbor(best, rng)
        print("step %d/%d: proposing %s" % (step_index, args.steps, params_text(candidate)))
        score, mean, _ = evaluate(candidate, args.rounds, args.reps, args.shape, log_file)
        if score > best_score:
            print("  ACCEPT: %+.2f%% > %+.2f%% -- new best %s" % (score, best_score, params_text(candidate)))
            best = candidate
            best_score = score
            best_mean = mean
            stall = 0
            save_state(args.state, best, best_score, best_mean, step_index, args.steps, stall, args.stall_limit, candidate, "accepted")
        else:
            stall += 1
            print("  reject (%+.2f%% vs best %+.2f%%), stall %d/%d" % (score, best_score, stall, args.stall_limit))
            save_state(args.state, best, best_score, best_mean, step_index, args.steps, stall, args.stall_limit, candidate, "rejected")
            if stall >= args.stall_limit:
                best = random_point(rng)
                print("  stalled out -- jumping to random point %s" % params_text(best))
                best_score, best_mean, _ = evaluate(best, args.rounds, args.reps, args.shape, log_file)
                print("  restart score: %+.2f%% median (%.2f%% mean)" % (best_score, best_mean))
                stall = 0
                save_state(args.state, best, best_score, best_mean, step_index, args.steps, stall, args.stall_limit, best, "restart")

    log_file.write("# hill climb done: best %s at %+.2f%% median\n" % (params_text(best), best_score))
    log_file.close()
    save_state(args.state, best, best_score, best_mean, args.steps, args.steps, stall, args.stall_limit, best, "done")
    print("hill climb done: best %s at %+.2f%% median improvement" % (params_text(best), best_score))


if __name__ == "__main__":
    main()
