# Genome Sorter and Champion INI Update

The genome sorter (`tools\genome_sort_experiment.exe`) searches for faster matrix-multiplication
dispatch configurations by comparing a pool of genomes using actual timings.
The champion INI updater (`update_ini_champions.py`) transplants the best found
genome per shape into `threading_proxy.ini`.

## The sorter

A genome is one complete dispatch configuration: 22 dispatch params + 8 owned
pf_dist/pf_type slots (Q8 kernel) or 3 (F32). Genomes are standalone INI files
(`<shape>_<counter>.ini`).

### Compare primitive

Two genomes race interleaved (A dispatch, B dispatch, A dispatch, B dispatch, and so on), accumulating
each side's matmul time. The compare resolves when the accumulated difference
exceeds `--min-diff` seconds (default 0.005, stage N uses 10x the base), or at
the `--compare-max` cap (default 5 s), or at the stage deadline — whichever
comes first. The winner is the genome with the lower accumulated time. Every comparison produces a winner, including comparisons that reach the time cap;
this does not imply that the measured difference is statistically significant.

### Sort algorithm

Probabilistic pair exchanges with a rotating anchor: index 1 cycles over the
pool (1, 2, ..., count, wrapping), index 2 is drawn as `frand()^2 * index1` —
a partner below the anchor, biased toward the front. The timing comparison selects the observed winner, even when it ends at the time cap.
The pair is exchanged so the observed faster genome occupies the lower index. Because
timings are noisy and the comparisons are not exhaustive, the resulting order is an
empirical ranking rather than a guaranteed global fastest-first ordering.

### Mutation and crossover

At the end of each sweep, the tail of the pool is replaced with mutated genomes.
Its size is controlled by `--mutate-percentage` (a percentage of the pool; default 0). Each mutant gets 1-10 random field mutations, then a
crossover: a second parent (drawn `frand()^2` through the ranking) donates
`frand() * total_slots * sex_influence/100` slots (same slot to same slot).
The legality checks then run again on the resulting genome.

### Pool time control

With `--target-pool-size N`, sorting runs as a single stage, and the pool shrinks
(or extends) linearly toward N over the runtime, proportional to elapsed
budget. Shrinking culls the worst-ranked genomes; extension fills with mutated
children. This replaces the phase-cut machinery (which still exists via
`--stages`/`--top-percent`).

### Probe phase

With `--random-probes N` the pool starts through acceptance-filtered random
search (no archive needed): the first random probe becomes the baseline
(20-round self-warmup), each further probe is challenged against the current
baseline via the standard compare — winners join and become the baseline,
losers are discarded. Pair with `--shape NxM`.

### Typical invocation

    tools\genome_sort_experiment.exe bin\v7_threading.dll ^
      --db weights\databases\qwen3_0.6b\lz4ht.db --q8 ^
      --genome-dir opt_1024x3072\run_a ^
      --target-pool-size 100 --runtime 5 ^
      --compare-max 0.005 --min-diff 0.00005 ^
      --mutate-percentage 10 --sex-influence 40 ^
      --genome-output-dir opt_1024x3072\target_pool ^
      --sorter-stdout-file sorter_saved\run.txt

The console shows only progress messages on stderr (`stage N/M sorting ... NN%`).
Detailed output, including the ranked deciles, stage summaries and final top-N results,
is written to the file specified by `--sorter-stdout-file`.

## The champion INI updater

    python update_ini_champions.py

The updater reads each champion genome file listed in its configuration (seven shapes
at the time of writing), locates the matching `[NxM]` section in
`threading_proxy.ini`, and replaces that section's values with the champion
configuration. After updating, rebuild the DB:

    bin\build_db.exe weights\qwen3_0.6b.bin -o weights\databases\qwen3_0.6b\lz4ht.db

## The reference benchmark

    measure_winner_genomes.bat --genome-dir <dir> --mb 16 --seconds 10 ^
      --genomes 1,2,3,4,5 --repeat 3

This command measures the genomes using ten-second steady-state runs and the `shape_time_a` metric.
The recorded correction is `corrected = raw * (31.0 / factor)`, where `factor`
is `float_sum_MBps / memcpy_kernel_GBs` from the same run. The correction
is not valid for the DRAM-bound `lm_head` shape (1024x151936).

## Champion record

The committed champion genomes live in the `opt_*\` archive directories
(one best genome per shape with exact file paths); the config that ships is
transplanted into `threading_proxy.ini` by `scripts\update_ini_champions.py`,
whose champion list is hard-coded in the script.
