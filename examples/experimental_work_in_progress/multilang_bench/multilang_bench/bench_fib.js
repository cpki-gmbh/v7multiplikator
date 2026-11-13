const reps = parseInt(process.argv[2] || "5", 10);
const quick = process.argv[3] === "--quick";

const fibN = quick ? 16 : 24;
const loopN = quick ? 2000 : 20000;
const strbN = quick ? 400 : 2000;
const replN = quick ? 15 : 60;

function fib(n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

function loopAdd(iterations) {
    let total = 0;
    let counter = 1;
    for (let i = 0; i < iterations; i++) {
        total += counter;
        counter += 1;
    }
    return total;
}

function strbuild(steps) {
    let s = "x";
    for (let i = 0; i < steps; i++) {
        s += "y";
    }
    return s;
}

function replaceAll(text, oldStr, newStr) {
    for (;;) {
        const pos = text.indexOf(oldStr);
        if (pos === -1) return text;
        text = text.slice(0, pos) + newStr + text.slice(pos + oldStr.length);
    }
}

function median(samples) {
    const sorted = samples.slice().sort((a, b) => a - b);
    const n = sorted.length;
    if (n % 2 === 1) return sorted[(n - 1) / 2];
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

const fibExpected = fibN === 16 ? 987 : 46368;
if (fib(fibN) !== fibExpected) throw new Error("fib wrong");
if (loopAdd(loopN) !== (loopN * (loopN + 1)) / 2) throw new Error("loop wrong");
if (strbuild(strbN).split("y").length - 1 !== strbN) throw new Error("strb wrong");

const fibSamples = [];
const loopSamples = [];
const strbSamples = [];
const replSamples = [];

for (let rep = 0; rep < reps; rep++) {
    let t0 = process.hrtime.bigint();
    const r1 = fib(fibN);
    let t1 = process.hrtime.bigint();
    const r2 = loopAdd(loopN);
    let t2 = process.hrtime.bigint();
    const r3 = strbuild(strbN).split("y").length - 1;
    let t3 = process.hrtime.bigint();
    let hay = "";
    for (let i = 0; i < replN; i++) hay += "cat ";
    const r4 = replaceAll(hay, "cat ", "dog ").split("d").length - 1;
    let t4 = process.hrtime.bigint();

    if (r1 !== fibExpected) throw new Error("fib drifted");
    if (r2 !== (loopN * (loopN + 1)) / 2) throw new Error("loop drifted");
    if (r3 !== strbN) throw new Error("strb drifted");
    if (r4 !== replN) throw new Error("repl drifted");

    fibSamples.push(Number(t1 - t0) / 1e6);
    loopSamples.push(Number(t2 - t1) / 1e6);
    strbSamples.push(Number(t3 - t2) / 1e6);
    replSamples.push(Number(t4 - t3) / 1e6);
}

function row(label, samples) {
    process.stdout.write(label.padEnd(22) + median(samples).toFixed(3).padStart(10) + " ms\n");
}

row("recursive fib", fibSamples);
row("loop add", loopSamples);
row("string build (concat)", strbSamples);
row("replace all", replSamples);
