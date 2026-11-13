# Reporting Rules, Actionable Optimization & Bug Reports

These rules apply to actionable optimization and defect reports, not to ordinary documentation or
separately requested readability changes. Include only findings requiring a specific source-code
change.

---

## 1. Every Item Must Warrant Action or Be Removed

- **No findings without defects.** Exclude code that is already correct. A "verified correct" entry does not request a change.
- **No "skip" items.** Omit findings that do not require action.
- **No style-only items.** Naming preferences and cosmetic SF_START consistency are refactorings without a demonstrated benefit and should be excluded. An actual Win64 ABI violation is a correctness bug, not a style issue, and must be reported.
- **No verification-only items.** "Positive verification recorded" does not describe an actionable change and should be excluded.

## 2. Every Item Must Be Directly Executable

- **Exact line numbers.** Provide the exact source line, not an approximate location.
- **Exact edit instruction.** "Delete line 548", "Change `ymm4` -> `ymm6`", "Insert X between line 563 and 564". No ambiguity.
- **Apply order.** State whether to apply top-down or bottom-up (to preserve line numbers).
- **No vague proposals.** "Consider using X instead of Y" is not actionable. "Change line 543: `vsubps ymm0, ymm0, ymm4` -> `vsubps ymm0, ymm0, ymm6`" is actionable.

## 3. Performance Is the Goal

- **Small simple micro-optimizations are in scope.** A 1-instruction deletion in a per-token hot path saves thousands of uops per inference. Every cycle counts.
- **Bit-preserving changes are preferred.** If a change alters numerical output, it must be justified as a precision improvement, not just a speed hack.
- **Keep the report within its requested scope.** A report limited to instruction-level micro-optimizations should not include unimplemented algorithmic redesigns. An algorithmic-optimization report may include a redesign when it specifies a concrete, implementable edit and its expected benefit.
- **Marginal but demonstrably beneficial changes are in scope.** A one-operand change that removes an instruction or dependency can be worth including. Distinguish a measured gain from a plausible hypothesis.

## 4. No Senseless or Dangerous Items

- **No complex refactors without global impact.** Converting manual push/pop prologues to SF_START macros is a style change, not a performance win. Exclude.
- **No defensive code for impossible paths.** Exclude bounds checks, null guards and fallback tails for conditions that cannot occur in supported model configurations.
- **Preserve numerical correctness.** A dispatch or accumulation-order change can alter floating-point output. Accept it only if the results are bit-identical or if testing demonstrates that precision is maintained or improved without a performance regression. Do not claim bit identity from algebraic equivalence alone.
- **Do not claim unverified performance gains.** If you can't explain exactly why it's faster on the measured target (Ryzen 5 7430U, Zen 3, AVX2), it's speculation, not an optimization.

## 5. Bugs Only If Real

- **Only real bugs.** A comparison polarity that looks wrong but is actually correct is NOT a bug. Do not report it.
- **Distinguish observed defects from future risks.** Report an existing defect when its triggering inputs or execution path are supported, even if it is intermittent. Do not label a purely hypothetical future change as a current bug.
- **Real bugs get top priority.** If a real bug is found (wrong output, crash, incorrect arithmetic), it goes first in the report and must be fixed immediately.

## 6. Consolidate, Don't Enumerate

- **Group related changes.** If five lines need the same operand change, provide one item with a numbered edit list rather than five separate items.
- **No filler.** A 10-item report where 8 are "skip" and 2 are real should be a 2-item report.
- **Summary tables are not needed.** If you have 4 actionable items, just list the 4 items. A summary table restating what the items already say is filler.

## 7. The Report Is a Todo List, Not a Document

- **The reader should be able to execute every item without re-reading the source.** All context needed to make the edit should be in the item itself.
- **The report is a work order.** Once its changes have been applied, verified, and committed, delete the processed report according to the development handbook. Preserve separate historical measurement records.
