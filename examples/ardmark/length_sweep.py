# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

# length_sweep.py -- runs the 5 length probe jobs against the already-loaded
# Ardmark sidecar and prints the visible answer word count per job plus the
# FULL raw output (nothing filtered). One variant must be loaded into the
# server BEFORE calling this script (sidecar files are memory-only).
# Usage: python length_sweep.py <variant_name>
# Raw outputs are appended to build_logs/length_sweep_<variant>.log; the
# word counts are printed here as the summary line per job.

import json
import subprocess
import sys
import os

QUESTIONS = [
    "Tell me, Bumwolf, how a fishing town like Kalvstrand holds itself together through one whole year?",
    "What is it like in the mines below Bjornvatn in the middle of winter, and why does no one walk alone down there?",
    "Tell me of the day of the banners, old one - the flags, the barrels, the walk home.",
    "Bumwolf, what happens to a village street in the dead of winter, when the story winters come and the lamps are lit?",
    "What does the harbor look like on a June light night, and what do the fishers do until the sun rises again?",
]

REQUEST_BODY = {
    "messages": [],
    "max_tokens": 0,
    "reset": 1,
    "thinking": 1,
}


def run_job(question):
    body = dict(REQUEST_BODY)
    body["messages"] = [{"role": "user", "content": question}]
    request_data = json.dumps(body).encode("utf-8")

    import urllib.request
    request = urllib.request.Request(
        "http://localhost:8080/v1/chat/completions",
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        raw = response.read()
    data = json.loads(raw.decode("utf-8", errors="replace"))

    thinking = ""
    answer = ""
    for choice in data.get("choices", []):
        message = choice.get("message", {})
        thinking = message.get("reasoning_content", "")
        answer = message.get("content", "")

    words = len(answer.replace("\r\n", " ").replace("\n", " ").split())
    return question, thinking, answer, words


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    log_path = os.path.join("..", "..", "..", "build_logs", "length_sweep_" + variant + ".log")
    log_file = open(os.path.abspath(log_path), "w", encoding="utf-8")

    print("variant: " + variant)
    for question in QUESTIONS:
        question_text, thinking, answer, words = run_job(question)
        line = "words=%d ties=[%s]" % (words, question_text[:50])
        print(line)

        log_file.write("============ QUESTION ============\n")
        log_file.write(question_text + "\n")
        log_file.write("============ THINKING ============\n")
        log_file.write(thinking + "\n")
        log_file.write("============ ANSWER ============\n")
        log_file.write(answer + "\n")
        log_file.write("============ WORDS: " + str(words) + " ============\n\n")

    log_file.close()
    print("raw outputs appended: " + log_path)


main()
