# inc_reformulator experiment report

Goal: replace every ` -- ` in .inc comments with a formulation that describes
behaviour only, no ` -- `, no `(` `)`. The engine (Qwen3-0.6B in the v7
server) reformulates single comments through the sandwich prompt; the script
edits them in place only when the answer passes all cleanliness gates.

## Pipeline

- server: `start_server.bat` (port 8080), `POST /v1/chat/completions` with
  `enable_thinking: false`, `max_tokens: 200`; every comment gets a fresh
  session via `POST /v1/chat/reset` (`X-V7-Local: 1`).
- prompt (the owner's mandated shape, in `build_prompt`):
  `... MUST not contain "(" "--" ")" now the comment:>>>COMMENT<<<
  ... now output the reformulated comment, only output the comment, nothing else!`
- gates before an in-place edit (`is_clean`): no ` -- `, no `(`/`)`,
  no `>>>`/`<<<`, single line, no instruction echo (`reformulat` /
  `must not contain` / `assembly code`), information-preservation floor
  (`len >= 0.4 * original`), and a growth ceiling (`3x + 60`).
- fences: leading markdown code fences are stripped (formatting, not content).
- in-place edit keeps the code side and the comment column exactly;
  unclean answers are skipped and the next comment is processed.
- log `reformulate_log.txt` (next to the script): verdict, line number,
  original comment, reformulated comment, reason, full model reasoning.

## Verbatim results (final run on test_reformulator.inc, 2026-10-27)

10 comments, 6 applied cumulative across runs, 4 remaining skipped:

| line | original | applied verdict / kept result |
|------|----------|-------------------------------|
| 3 | `code + inline-comment lines. Every case below contains " -- ".` | applied -> `reformulated-comment` (information lost) |
| 5 | `lowercase case note -- the tool must keep the code side of a line untouched` | applied -> `lowercase case note the tool must keep the code side of a line untouched` (GOOD) |
| 6 | `answer value -- always forty-two` | applied -> `answer value always forty-two` (GOOD) |
| 7 | `bump eax to 43 (the real answer -- 42 came first)` | applied -> `bump eax to 42` (WRONG: fact flipped) |
| 10 | `return to caller -- nothing else cleaned here` | applied -> `reformulated_comment: return to caller` (garbage prefix) |
| 11 | `AES-NI -- vaesenc runs the Intern hash and the sampler RNG rounds` | applied -> `AES-NI runs the Intern hash and the sampler RNG rounds.` (GOOD) |
| 8 | `doubled the answer (42*2=84 -- a new meaning)` | skipped: echoes the original with ` -- ` |
| 9 | `next stack slot (Q8 row scale -- 8 bytes per group)` | skipped: echoes the original with ` -- ` |
| 12 | `score is scaled once -- one hoisted multiply per cache block, ...` | skipped: echo truncated |

## Quality assessment (0.6B model)

- Accept rate with the current gates: 0/4 on the remaining hard comments;
  the model either echoes the original verbatim (the ` -- ` it was told to
  remove) or echoes the instruction.
- On earlier accepted comments the quality was mixed: 3 of 6 were good,
  1 fact-flipped (43 -> 42), 2 were prefix garbage. The gates now catch the
  garbage shapes (echo, too short), but the ACCEPTED set needs human review
  in the log before any real .inc file is touched.
- The tooling works end-to-end: scan, server round-trip, gates, in-place
  edit, log. The bottleneck is 0.6B reformulation quality, not the pipeline.

## Files

- `examples/inc_reformulator/inc_reformulator.py` - the reformulator
- `examples/inc_reformulator/run_inc_reformulator.bat` - driver (project root cwd)
- `examples/inc_reformulator/test_reformulator.inc` - test target (currently
  holds the 6 applied rewrites; restore from git before the next clean run)
- `examples/inc_reformulator/reformulate_log.txt` - per-comment log
