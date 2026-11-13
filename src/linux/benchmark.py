#!/usr/bin/env python3
"""Run fresh-process, alternating F32 and Q8 generation trials."""
import argparse
import os
from pathlib import Path
import re
import resource
import subprocess
import tempfile
import time

PROMPT = (
    "Write a poem with 80 numbered lines about a lighthouse. "
    "Each line must be at least eight words long. Do not add any commentary."
)
PHASE = r"\b{}\s*:\s*\d+ cycles\s+([0-9.]+) ms\s+\((\d+) tok,"


def child_cpu_time():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def run(inference, library, database, mode, tokens, directory):
    command = [
        str(inference), str(library), str(database), "-think", "0",
        "-n", str(tokens), "-i", PROMPT,
    ]
    if mode == "Q8":
        command.append("-q8")
    cpu_start = child_cpu_time()
    wall_start = time.monotonic()
    result = subprocess.run(
        command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", timeout=180, check=False,
    )
    wall = time.monotonic() - wall_start
    cpu_percent = 100 * (child_cpu_time() - cpu_start) / wall
    (directory / "console.log").write_text(result.stdout)
    prefill = re.search(PHASE.format("prefill"), result.stdout)
    generation = re.search(PHASE.format("generation"), result.stdout)
    if result.returncode or not prefill or not generation:
        raise SystemExit(f"{mode} failed, see {directory / 'console.log'}")
    prefill_ms, prefill_tokens = float(prefill[1]), int(prefill[2])
    decode_ms, decode_tokens = float(generation[1]), int(generation[2])
    if decode_tokens != tokens:
        raise SystemExit(f"{mode} generated {decode_tokens}/{tokens} tokens, see {directory / 'console.log'}")
    return prefill_tokens, prefill_ms, decode_tokens, decode_ms, cpu_percent, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--bin-dir", type=Path, default=Path("bin-linux"))
    parser.add_argument("--cpus", type=int, help="Use this many CPUs from the current affinity mask.")
    parser.add_argument("--trials", type=int, default=3, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--logs", type=Path, help="Create a new directory to retain the trial logs.")
    args = parser.parse_args()

    allowed = sorted(os.sched_getaffinity(0))
    if args.cpus is not None:
        if not 2 <= args.cpus <= len(allowed):
            parser.error("--cpus must be between 2 and the current affinity count")
        os.sched_setaffinity(0, allowed[:args.cpus])
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    print("# CPUs " + ",".join(map(str, sorted(os.sched_getaffinity(0)))), flush=True)
    print("trial,mode,prompt_tokens,prefill_ms,generated_tokens,decode_ms,decode_tokens_s,process_cpu_pct,process_wall_s", flush=True)

    def trials(root):
        executable = args.bin_dir.resolve() / "inference_example"
        library = args.bin_dir.resolve() / "libv7_threading.so"
        database = args.database.resolve(strict=True)
        for trial in range(1, args.trials + 1):
            for mode in ("F32", "Q8") if trial % 2 else ("Q8", "F32"):
                directory = root / f"trial-{trial}-{mode.lower()}"
                directory.mkdir()
                prompt, prefill, tokens, decode, cpu, wall = run(
                    executable, library, database, mode, args.tokens, directory
                )
                print(f"{trial},{mode},{prompt},{prefill:.3f},{tokens},{decode:.3f},"
                      f"{1000 * tokens / decode:.2f},{cpu:.1f},{wall:.3f}", flush=True)

    if args.logs:
        args.logs.mkdir()
        trials(args.logs)
    else:
        with tempfile.TemporaryDirectory(prefix="v7-bench-") as path:
            trials(Path(path))


if __name__ == "__main__":
    main()
