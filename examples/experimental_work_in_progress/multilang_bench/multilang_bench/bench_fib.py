# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

import sys, time

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def loop_add(iterations):
    total = 0
    counter = 1
    for _ in range(iterations):
        total += counter
        counter += 1
    return total

def strbuild(steps):
    s = "x"
    for _ in range(steps):
        s += "y"
    return s

def replace_all(text, old, new):
    while True:
        pos = text.find(old)
        if pos == -1:
            return text
        text = text[:pos] + new + text[pos + len(old):]

def median(samples):
    samples = sorted(samples)
    n = len(samples)
    if n % 2 == 1:
        return samples[n // 2]
    return (samples[n // 2 - 1] + samples[n // 2]) / 2.0

def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    quick = len(sys.argv) > 2 and sys.argv[2] == "--quick"

    fib_n = 16 if quick else 24
    loop_n = 2000 if quick else 20000
    strbuild_n = 400 if quick else 2000
    replace_n = 15 if quick else 60

    # correctness gate
    assert fib(fib_n) == (987 if fib_n == 16 else 46368)
    assert loop_add(loop_n) == loop_n * (loop_n + 1) // 2
    assert strbuild(strbuild_n).count("y") == strbuild_n

    # replace workload: build "cat " * n then replace all -> count 'd'
    hay = ""
    for _ in range(replace_n):
        hay += "cat "
    result = replace_all(hay, "cat ", "dog ")
    assert result.count("d") == replace_n

    fib_samples = []
    loop_samples = []
    strb_samples = []
    repl_samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r1 = fib(fib_n)
        t1 = time.perf_counter()
        r2 = loop_add(loop_n)
        t2 = time.perf_counter()
        s = strbuild(strbuild_n)
        r3 = s.count("y")
        t3 = time.perf_counter()
        hay = ""
        for _ in range(replace_n):
            hay += "cat "
        r4 = replace_all(hay, "cat ", "dog ").count("d")
        t4 = time.perf_counter()

        fib_samples.append(t1 - t0)
        loop_samples.append(t2 - t1)
        strb_samples.append(t3 - t2)
        repl_samples.append(t4 - t3)

        assert r1 == (987 if fib_n == 16 else 46368)
        assert r2 == loop_n * (loop_n + 1) // 2
        assert r3 == strbuild_n
        assert r4 == replace_n

    print("recursive fib          %10.3f ms" % (median(fib_samples) * 1000))
    print("loop add               %10.3f ms" % (median(loop_samples) * 1000))
    print("string build (concat)  %10.3f ms" % (median(strb_samples) * 1000))
    print("replace all            %10.3f ms" % (median(repl_samples) * 1000))

if __name__ == "__main__":
    main()
