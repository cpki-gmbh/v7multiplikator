# Development Rules for AI Agents

**Never edit source code with scripts. Preserve this rule in the handbook.**

**FILES IN `opt_and_bug\` MUST NEVER BE BULK DELETED— THEY MAY ONLY BE DELETED AS SINGLE FILES.** No
wildcards, no multi-file `del` commands, no `del *_revised.md`, no `del a.md b.md c.md`, no directory
wipe, no script that empties the folder. Exactly ONE `del` command per file, issued immediately after
that one report is consumed. We have lost the contents of this folder too often— every report in it is
irreplaceable work by a review agent, and a bulk delete destroys all of it in one stroke.

**EDITING LOGO CODE WITH SCRIPTS IS ALWAYS FORBIDDEN!** Every edit to a `.logo` file— autoexec,
sidecars, stdlib, research scripts, any of them— MUST be done by hand with the Edit tool, no exceptions.
PowerShell one-liners, python scripts, sed, find/replace pipelines, or any other programmatic mechanism
on `.logo` files is NEVER allowed. **ANY script access to Logo source code, reading, writing, probing
bytes, ANYTHING, is forbidden as well: Logo source is examined ONLY through the Read tool / grep and
modified ONLY through the Edit tool.** A misplaced edit in a Logo script silently changes model-facing
behaviour, and the emission-mangling bug class (eaten closing-tag literals, corrupted string bodies) is
exactly what hand edits, with the documented split-form trick `("</thi" _ "nk>")`, are there to avoid.
No ".logo tool edits" exception exists.

**LOW-VOLUME EDITS MUST ALWAYS BE DONE BY HAND.** Any edit of low volume— a single comment line, a
handful of comment lines, a small block of comments, one or two report items, a few lines of a report
file— MUST be made BY HAND with the Edit tool. NEVER use scripts, shell one-liners,
`Set-Content`/`Get-Content` pipelines, `sed`, `awk`, `python -c`, regex bulk replace, or any other
programmatic mechanism to perform a low-volume edit. ALWAYS. A script is for high-volume, purely
mechanical, byte-identical work only. Hand-editing is what lets the agent SEE the exact text it is
changing, verify the old text actually exists, and notice when a replacement duplicates or deletes
something it did not intend. Scripts silently corrupt low-volume edits: they match the wrong occurrence,
they duplicate blocks, they mangle encoding, and they destroy report structure while appearing to
succeed. If an edit is worth doing, it is worth doing by hand. If a task requires changes that cannot safely be made mechanically, divide it into
reviewable hand edits rather than applying an unsafe script.

**WHEN THE USER GIVES YOU A RULE TO ADD TO AGENTS.MD, YOU ADD THE RULE— VERBATIM, IN FULL, NO
EXCEPTIONS, NO SOFTENING, NO "BUT WHAT IF" QUALIFIERS, NO "ONLY WHEN" CLAUSES.** The user's rule is the
rule. You do NOT add exceptions, carve-outs, or softening language of your own. If the user says "no X,
ever," you write "no X, ever"— NOT "no X, except when Y." If the user says "always Z," you write "always
Z"— NOT "always Z, unless W." Any exception the user wants, the user will write themselves. Your job is
to transcribe, not to editorialize.

SMC CODE WILL NEVER BE REMOVED AND MUST NOT BE DAMAGED

**EVERY BUG MUST BE FIXED— NO EXCEPTIONS, NO DEFERRALS, NO "WE'LL KEEP IT."** If a bug is found, it MUST
be fixed in the SAME SESSION— not added to a todo list "for later," not noted and moved on, but FIXED
RIGHT NOW. A known bug that is not fixed in the session it was found is a violation of this rule. There
is no "later in the session"— there is only NOW. ABI violations in exported functions are bugs with the
same severity as crashes.

**NOTE (Aug 2026): ALL `ud2` instructions have been REMOVED from every `.inc` file** (commit `9c104a9`,
99 dead `ud2` after FATAL). The FATAL macro already ends with `int3` and never returns. Any report/audit
mentioning `ud2` removal or "missing ud2 after FATAL" is SATISFIED— ignore such items.

---

### XXa. REPEATED INSTRUCTIONS MEAN EXECUTE NOW— No More Delays, No More Setup

---

### XXb. TEST-CASE ADAPTATION IN RAG/SIDECAR CODE IS ILLEGAL— Generic Mechanisms Only

**Guards in the RAG/memory pipeline must be GENERIC mechanisms, never enumerations of strings observed
in specific test runs.** Hand-blacklisting the exact hallucinations or phrasings a particular model
produced on particular prompts is overfitting: it "fixes" the test while the same failure class recurs
with any different wording, model, or language.

**Hard rules:**
1. **No string blacklists of observed artifacts.** Banning `"John Doe"`, `"user name"`, `"**"`, `"no other facts"` etc. because they appeared in one run is ILLEGAL— the next distiller hallucination will simply use different words.
2. **No keyword lists that classify user questions into ad-hoc intents** (e.g. routing on `remember`/`about me`/`who am i`). Question routing must come from generic signals (retrieval scores, floors, fallbacks), not enumerated phrasings.
3. **Language-drift detection via word lists of one specific language is also illegal**— it only guards the language observed during testing.
4. **The legal toolbox:** source grounding (a distilled fact is stored only if its content tokens actually occur in the turn text it was distilled from— rejects template echoes, placeholder hallucinations, meta-commentary, and foreign-language drift in ONE generic check), calibrated statistical floors (density/topscore), similarity-based dedup (Jaccard), recency priors, and empty-result fallbacks. These are content-agnostic and survive wording/model changes.
5. **When a new failure mode appears, add a MECHANISM, not a string.** Ask: "what generic property distinguishes bad output here?" (e.g. not grounded in the source, duplicate of existing, below floor) and encode THAT.

### XXc. MANDATORY GENERICITY SELF-CHECK Before ANY RAG Change

**Before you do ANYTHING in the RAG/memory/sidecar pipeline— every guard, every rule, every filter,
every routing branch, every threshold— you MUST first ask yourself:**

> **"Will this improve the generic behaviour or only my single case? If it is generic I should do it; if it is not I MUST NOT do it."**

**You MUST always check this box before you do something:**

```
it is generic [ ]
```

Only proceed when the box checks TRUE. If the change only fixes the one prompt/one model/one wording/one
language you happen to be looking at right now, STOP— do not apply it. This check is non-optional and
applies to every single RAG edit, no exceptions.

---

**When the user stops you and repeats the same instruction, that is the definitive signal to JUST DO IT—
immediately, in that exact message, with zero further preparation.** A repeated instruction means the
user has watched you delay, plan, re-check, or build prerequisite todo lists instead of acting, and is
now explicitly demanding action.

**Hard rules:**
1. **The second time an instruction arrives, it executes IMMEDIATELY.** No new todo lists. No "finish this other thing first." No re-verification of prior steps. No re-reading of rules. The very next action is the instruction itself.
2. **Repeated instruction = highest priority, everything else pauses.** Whatever was in progress (a build, a plan, a pending check) is suspended. The user's repeated instruction is the ONLY task until it is done.
3. **Never answer a repeated instruction with another preparatory action.** Responding to "do X NOW" with "let me first rebuild Y" is a violation. Respond with the first concrete step of X itself.
4. **This overrides caution rules.** The time for asking, hedging, and sequencing is over. Act, then report what was done.

**Example:** The user says "process the reports"— you start a build and make a todo list. The user
interrupts and says "process the reports NOW"— you immediately open the first report file and start
applying its edits. Nothing else runs first.

---


### XX. PRIORITY RULE— Inference First, Prompting Second

**Before ANY prompt tuning, sampling parameter tuning, or agent behavior changes: verify that the ASM
inference pipeline (v7_threading.dll) works correctly with the given parameters.** Run
`inference_example.exe` directly (bypassing the agent) with `-t <temperature>` and confirm it produces
reasonable output. Only once inference at the target temperature is verified do you proceed to tune
prompts or agent parameters. If inference hangs at a given temperature, the bug is in the ASM sampler or
generate path— NOT in prompts or agent code. Debug the ASM pipeline first, never work around it by
changing agent parameters.

### 0b. CRASHES MUST NEVER BE IGNORED— ABSOLUTE RULE, NO EXCEPTIONS

**Every crash— access violation, execute fault, stack corruption, FATAL, or any other unrecoverable
error— MUST be fixed immediately. No exceptions. No deferrals. No workarounds. No "we'll get to it
later." No ignoring it and moving on.**

This is the single most important rule in this entire document. A crash is the code screaming that
something is fundamentally broken. Ignoring a crash is ignoring a defect that WILL cause data
corruption, wrong output, silent memory corruption, or unpredictable behavior in production.

**Hard rules:**
1. **A crash encountered at ANY time— during development, testing, debugging, code review, reading source, or walking past it— MUST be investigated and fixed in the SAME session.** No exceptions. If you see a crash, the crash becomes the #1 priority until it is fixed. Everything else stops.
2. **"Preexisting" is NEVER a valid reason to ignore a crash.** If a crash exists in code you did not write, in code that has been broken for months, in code that "already worked" before— it is STILL your responsibility to fix it. Every crash you see is YOUR crash to fix. There is no "someone else's crash."
3. **"Low risk," "minor," "doesn't affect normal flow," and "only happens in edge cases" are NEVER valid reasons to defer a crash fix.** Every crash is a real defect. Edge-case crashes become normal-case crashes the moment the input changes. Minor crashes compound into major crashes. Low-risk crashes become high-risk crashes when conditions align.
4. **No workarounds.** Never add a fallback path, an SEH wrapper, a null-check guard, or a "graceful degradation" around a crash. Fix the ROOT CAUSE of the crash. If the root cause is hard to find, keep investigating— do not paper over it.
5. **If a crash blocks other work, FIX IT FIRST.** You do not have the option to "skip it for now and come back later." The crash is the blocker. Fix it. Then proceed.
6. **Every crash must be root-caused.** Never apply a fix without understanding WHY the crash happened. A fix without root-cause analysis is a workaround in disguise— it will resurface as a different crash later.
7. **Document the crash and fix in AGENTS.md.** After fixing a crash, add a section documenting the root cause, the fix, and the lesson learned. This is institutional memory that prevents the same class of crash from recurring.

**Zero tolerance:** If you encounter a crash and do NOT fix it in the same session, you have violated
this rule. There is no justification. There is no exception. There is no "but."

### 0ab. Safe Optimizations MUST Be Applied— No Discarding, No Nitpicking, No Excuses

**Every safe optimization from the `opt_and_bug/*.md` backlog MUST be applied.** The backlog exists
because each item was verified against the source and confirmed to be correct and beneficial. The
purpose of the backlog is to APPLY it, not to re-evaluate and second-guess every item. If it is safe and
saves even a single cycle, it goes in.

**Hard rules:**
1. **NEVER discard a safe optimization.** If an item is verified correct and semantically identical, it MUST be applied. "Saves only 1 instruction" is NOT a reason to skip— every cycle counts on hot paths (Sec.126 rule 8). "Not worth the effort" is NOT a reason— the review already did the work; you just need to apply the edit.
2. **If something small is in the way, MOVE IT, don't skip the optimization.** TRACE macros between instructions that prevent a load-fusion? Move the TRACE before or after the fused sequence. A non-volatile register copy that could be eliminated if a register was rearranged? Rearrange the registers. An optimization is "blocked" only if it requires a major structural change— minor code motion is NOT a blocker.
3. **Show motivation.** When applying an optimization, briefly note what it saves (instructions, bytes, cycles) and why it is safe. One sentence is enough: "Saves 3 instructions per bitmap loop iteration; btr/bts native memory operand handles offset+bit automatically, pattern already exists at line 898."
4. **"The TRACE macro is between the loads" is NOT a rejection reason.** TRACE macros compile to nothing in release builds (`-DTRACE_ENABLE` is off by default). In debug builds they expand to save/restore all 16 GPRs + RFLAGS— they are explicitly designed to be transparent (Sec.14d). If a TRACE macro blocks a code motion optimization, move the TRACE to just before or just after the optimized sequence. The TRACE observes program state; as long as it fires at the same logical point (same values visible), moving it is safe.
5. **Batch safe edits aggressively.** Apply multiple safe optimizations from different `opt_and_bug/*.md` files in one pass, then one build+verify+commit cycle. Do not run verify.bat per file— that wastes time. Batch everything that is: (a) precision-preserving output (bits may differ as long as numerical precision is maintained or improved), (b) no dispatch/accumulation-order changes, (c) no kernel-math reordering, (d) no precision-affecting changes. Per Sec.126 rule 11, these can be batched.
6. **Do NOT delete `opt_and_bug/*.md` files until EVERY safe item in them is applied.** A file is only "consumed" when all its safe items are in the source. If an item was rejected, document why in the file— don't silently drop it.
7. **Real bugs in these files are applied FIRST (Sec.126 rule 7).** Before any optimization, fix every real bug documented in the file.

### 0. Proven Performance Baselines

**This hardware (Ryzen 5 7430U, 12 logical cores) has PROVEN, MEASURED capability of >300 GB/s Q8
1024x1024 and >150 GB/s F32 1024x1024.** These are measured numbers the codebase actually hit, not
aspirational targets. Measured post-fix numbers: Q8 1024x1024 = 382-426 GB/s, F32 1024x1024 = 176-197
GB/s, all cores active (>1000% CPU). **Do NOT doubt these baselines.** A dispatch regression is a real
bug, not measurement noise. **Speeds below 300 GB/s for Q8 are NEVER acceptable.**

### 0a. Centralized Dispatch Architecture— Single Codepath for ALL Kernel Types

**This is a non-negotiable architectural goal. There is no discussion.**

**All dispatched work— memcpy, Q8 matmul, F32 matmul, fused F32, SwiGLU, quantize, attention,
RMSNorm+RoPE— MUST flow through ONE single codepath for job acquisition, work stealing, caller
participation, and completion tracking.** The only thing that differs between dispatch types is what the
microkernel does inside `process_job`. Everything else (publish `gen_counter`, wake decision, caller
steal loop, `remaining_jobs` tracking, wait, post-dispatch starvation analysis) is IDENTICAL and MUST
NOT be duplicated across dispatch files.

**No duplicate synchronization logic.** Every synchronization operation— job stealing (`lock xadd
next_job`), job claiming, `remaining_jobs` decrement, wake-by-address decision, starvation heuristic,
`gen_counter` publish, caller steal loop, pause-spin wait— exists in EXACTLY ONE place: the
`DISPATCH_EXECUTE` macro in `dispatch_execute.inc`. If you find the same `lock xadd` or `lock sub` or
`WakeByAddressAll` pattern in two different files, that is a bug that must be unified.

**No duplicate higher-level dispatch operations.** One dispatched operation = one codepath = one call to
the microkernel. No alternative paths, no conditional dispatch strategies, no per-shape dispatch
bypasses, no single-chunk direct-call fast paths that bypass the worker pool. **ALL dispatch operations
must go through the unified `DISPATCH_EXECUTE` codepath, even for a single chunk.**

**Job queue access must be centralized into macros.** All reads and writes to `jobs_arr[]` MUST go
through named macros in `dispatch_execute.inc` (or `dispatch_job.inc`): `JOB_ADDR reg, index` computes
the address; `JOB_SET_OUTPUT`, `JOB_SET_COLUMN`, `JOB_SET_MODE` write fields; `JOB_LOAD_OUTPUT`,
`JOB_LOAD_COLUMN`, `JOB_LOAD_MODE` fetch them. **No dispatch file may compute `jobs_arr` offsets with
raw arithmetic** (e.g., `shl eax, 6` / `lea rcx, [rel jobs_arr]` / `add rcx, rax`). Any existing raw
offset logic MUST be replaced with these macros immediately when encountered.

**The `v7d_*` entry points are ONLY responsible for:**
1. Fallback threshold check (different per kernel type)
2. SMC patching
   - The `ds` segment-override prefix is **mandatory** on all prefetch instructions (yields a 9-byte instruction; the extra byte is ignored by the CPU but keeps instruction lengths uniform for our patch system).
3. Per-shape config selection (only matmul)
4. `jobs_arr[]` filling (via centralized job-queue macros— NOT raw offset arithmetic)
5. Pre-dispatch globals setup (fused_norm_weights, swiglu_up_weight_q, etc.)

**`DISPATCH_EXECUTE` handles everything from "publish" to "all done":**
1. Early-wake sleeping workers (BEFORE job fill, overlapping kernel transition)
2. `remaining_jobs = count`, `num_jobs = count`, `next_job = 0`
3. `lock inc gen_counter` (full barrier)
4. Publish-wake sleeping workers (AFTER gen_counter++)
5. Caller steal loop (`CALLER_SELECT_BATCH`— `lock xadd`— `process_job`— repeat)
6. `remaining_jobs -= caller_count`
7. Pause-spin wait on `remaining_jobs == 0`

**Two-wakeup design (sleeping workers only):** When `last_spin == 0` (workers are sleeping via
WaitOnAddress), DISPATCH_EXECUTE fires WakeByAddressAll TWICE:
  1. **Early-wake** (step 1): Wakes workers, who re-check gen_counter— still old— enter a tight pause-spin loop polling gen_counter. They stay spinning until gen_counter++ fires a few cycles later (step 3)— zero kernel transitions.
  2. **Publish-wake** (step 4, AFTER gen_counter++): Backup wake for workers that re-entered WaitOnAddress between the early-wake and gen_counter++ (rare).

**After the first dispatch cycle with `set_spin(1)`:** Workers complete their jobs, loop back to
`.worker_outer`, read `always_spin=1`, atomically write `last_spin=1`, and commit to pure pause-spin.
For ALL subsequent dispatches, `last_spin != 0`— both WakeByAddressAll calls are skipped entirely.
**Zero syscalls on the hot path.**

**No `WakeByAddressAll` "just in case."** Wake sleeping workers (`last_spin == 0`). Do NOT wake spinning
workers (`last_spin != 0`). Production code must never fire unconditional wake-by-address; a targeted,
measured intervention is the only acceptable fix for slow gen_counter visibility.

**No `num_workers == 0` fallback.** If `num_workers == 0` after init, that is a `FATAL` condition—
thread creation failed. Fix the bug, do not silently degrade to single-threaded.

**No special paths for tools, validators, or test harnesses— EVER.** The dispatch code must have EXACTLY
ONE codepath: the multi-threaded worker-pool path used by LLM inference. Any `num_workers == 0` check,
`.fallback` label, `.direct_loop` label, `single_chunk_direct` bypass, or sequential per-head/per-chunk
loop that exists solely because a tool or validator runs without initializing workers is a **senseless
bug-hiding path** that MUST be removed. If a tool/validator doesn't initialize the worker pool, **fix
the tool**— do not add a fallback codepath in the dispatcher.

**Only `WaitOnAddress` and `WakeByAddressAll` are allowed as synchronization API calls in the
dispatcher.** No `Sleep`, `SleepConditionVariableSRW`, `SwitchToThread`, `YieldProcessor` (beyond the
`pause` hint), `CreateEvent`, `SetEvent`, `EnterCriticalSection`, `WaitForSingleObject`, or any other
Win32 thread synchronization primitive may appear anywhere in dispatch code (`dispatch_execute.inc`,
`*_dispatch.inc`, `v7_worker.inc`). Adding any other synchronization primitive is a bug.

**The dispatcher MUST retain the option for workers to keep spinning (`always_spin`), avoiding both
`WakeByAddressAll` and `WaitOnAddress` API calls entirely on the hot path.** Spinning workers observe
`gen_counter` via pure `pause`-spin— zero API calls, zero kernel transitions. This mode MUST be tested
and MUST work (both `always_spin == 1` pure-spin and `always_spin == 0` sleep/wake must produce correct
results at full speed).

**Kernel math and job processing must be in separate files.** Math goes in `*_kernel.inc`; dispatch
goes in `*_dispatch.inc`. No file may mix both concerns. If a file mixes math and dispatch, refactor
it immediately; this separation is non-negotiable.

**Dispatcher rewrite plan:** `dispatch_architecture_fix_plan.md`— the authoritative plan for eliminating
all deviations from the single-codepath architecture. **Special paths to remove:**
`special_paths_to_remove.md` (archived in `attic/plans/`)— the inventory of tool/validator-only dispatch
bypasses that must be removed.

### 1. Model & Reference
- **Model file is correct.** All crashes or wrong output are assembly bugs, not model-file issues. Never debug the model format.

### 2. General Conduct
- **Immediate action:** When the user asks you to do web research, run it right away.
- **No background work:** Benchmarks, validation tools, or any executable must run from a *.bat* file, one at a time. Never start server/background processes directly from the AI prompt (Start-Process, etc.)— always use a .bat file that starts the server, waits, runs the client test, then kills the server.
- **verify.bat runs foreground-only— never parallel tool calls:** When running `verify.bat`, make it the ONLY tool call in the message. No parallel reads, writes, greps, or any other tool calls. Nothing else may happen during verify.bat— no file reads, no file writes, no searches, no other tool invocations of any kind. Sending parallel tool calls while verify.bat runs corrupts benchmark results and makes LLM output unreliable. verify.bat is the definitive validation— it must always run solo, foreground, with full system attention.
- **Commit early & often:** After every successful build/run, create a small, descriptive commit.
- **Never silence errors:** Every failure must either `FATAL`-terminate or emit a visible warning.
- **Leave the tools alone:** When told to read/examine assembler sources (.inc, .asm), read the file and stop. Do not run build tools, benchmarks, or any other executable unless explicitly instructed in the current goal section. When the current goal requires building and validating, do it— no need to ask.
- **Work through plans:** When given or asked to execute a plan with multiple phases/steps, always work through the entire plan. Do not stop to ask— decide and proceed. Only stop if you hit a problem you genuinely cannot resolve alone.
- **Never stop until a crash is fixed:** A crash (AV, execute fault, stack corruption, etc.) is a hard blocker. Once a crash is identified, every session must continue investigating and fixing until the crash is resolved. Only stop if explicitly told to by the user or if the root cause is proven to be outside the codebase.
- **`dbg_inference.exe` is a NORMAL build artifact**— it may be rebuilt any time via its build script. Its source (`dbg_inference.asm`) MUST be kept fixed and adapted to the current v7 (Sec.161/162 memory law, current export surface). When the DLL's contract changes, update and rebuild the harness in the same session.

### 3. API Calls
- **Use `API_CALL` exclusively** for any Win32 API. No manual `sub rsp— call [__imp_*]` sequences. All `microdb.inc` calls must follow this rule.

#### API_CALL Mechanical Behavior (v7_core.inc:767-842)

**Syntax:** `API_CALL fn_ptr [, arg1 [, arg2 [, ...]]]`

`fn_ptr` is the call target itself: an import entry like `[__imp_FunctionName]` (assembled as an
indirect call through the IAT entry, `call qword [rip+rel32]`) or an internal label (direct `call
rel32`). The target is the direct operand of the call instruction, no staging register, no stack
round-trip.

**Stack effect:**
```nasm
sub rsp, %%alloc    ; enter
; ... call ...
add rsp, %%alloc    ; exit (no net RSP change)
```
Where `%%alloc` is computed as:
```
n_args      = total arguments - 1
n_stack     = max(0, n_args - 4)
raw         = 32 (shadow) + ((n_stack + 1) & ~1) * 8
%%alloc     = (raw + 15) & ~15   ; 16-byte aligned
```

**Internal flow (all through rax temp):**
1. Save ALL arg SOURCE values to shadow/stack slots (mov rax, source— mov [rsp+slot], rax)
2. Load arg registers from shadow space: rcx?slot0, rdx?slot1, r8?slot2, r9?slot3
3. `call fn_ptr` (direct target: IAT-indirect for imports, rel32 for labels)
4. `add rsp, %%alloc` (restore)

Arg slots use `APICALL_SHDW0..3` (shadow, offsets 0/8/16/24) and `APICALL_ARG5..` (stack args at offset 32+).

**ORDER FIX:** ALL arg sources are saved to shadow/stack BEFORE the final argument registers are loaded.
A reload writes one of rcx/rdx/r8/r9, which may be another argument's SOURCE register, saving every
source first means each read sees its original value. (The call target itself is staged last-or-never:
`call fn_ptr` clobbers nothing.)

**Register clobber (defined in v7_core.inc comment):**
- CLOBBERED (volatile): `rax`, `rcx`, `rdx`, `r8`, `r9`, `r10`, `r11`
- PRESERVED (non-volatile): `rbx`, `rsi`, `rdi`, `r12-r15`, `rbp`

Any value in a volatile register (or cached from memory into one) is garbage after `API_CALL`. Reload
from non-volatile storage (BSS, stack slot, non-volatile reg) after any `API_CALL`.

- **No API calls in hot paths.** Never add `API_CALL`, `WriteFile`, `WaitOnAddress`, `WakeByAddressAll`, `Sleep`, `SleepConditionVariableSRW`, or any other Win32 API call to the dispatch/worker/matmul hot path. API calls cost microseconds each— adding even one per iteration destroys throughput on small matrices. The ONLY API calls allowed on the hot path are the ones already there (the single `WakeByAddressAll` at publish time, the single `WaitOnAddress` when a worker goes to sleep). Do NOT add more.

### 4. Optimization Sweep— Completed (Jun 2026)

**The sweep across `opt_documentation/*.md` is complete; there is no ongoing sweep.** When new
optimization opportunities are discovered, add a `.md` to `opt_documentation/` and process it
individually. `asm_trick.md` remains the permanent trick reference. Documented in `optmitation.log`.

### 4a. `v7s_set_spin` Worker Spin/Sleep Contract (current implementation)

The prior handbook's four-forward-function `set_spin(1)`/`set_spin(0)` pairing
was superseded. Current `forward_q8.inc`, `forward_f32.inc`, `prefill_q8.inc`,
`prefill_f32.inc`, and `vit_forward.inc` **enable** worker spinning at entry,
but none makes the formerly described `set_spin(0)` call before its last
dispatch. `src/asm/chat_finish.inc` invokes `v7s_stop_spin` after the chat turn,
which writes `always_spin=0` unconditionally.

`v7s_set_spin(1)` writes `always_spin=1` and wakes registered sleepers.
`v7s_set_spin(0)` writes `always_spin=0`, letting idle workers take the
`WaitOnAddress` sleep path. Chat completion's `chat_finish.inc` boundary calls
the internal `v7s_stop_spin` (a one-store leaf that writes `always_spin=0`).
Spin/sleep transition and wake-up behavior remain in the worker dispatcher.
Review changes against `src/asm/util_export.inc`, `src/asm/v7_init.inc`,
`src/asm/chat_finish.inc`, and the actual forward files rather than
treating superseded line numbers or historical rules as current facts.

### 4b. Systematic Debugging— C vs DB State Comparison via runq CLI (`-compare-init`)

**When debugging ASM init vs C init discrepancies, do NOT chaotically edit ASM files.** Use the
`-compare-init 1` CLI flag on `runq.exe` (function `compare_init()` in `runq.c`). It builds the C
reference from FILES, then calls `v7s_inference_create` for the ASM DB-init state, and walks ALL data
structures field-by-field, printing every difference to stderr: Config fields, TransformerWeights
pointers + 256-byte data checksums, QuantizedTensor q/s pointers, RunState buffer pointers, Tokenizer
(vocab, bos/eos IDs, merge_scores, templates), Sampler, MemLayout, and the model-weights blob checksum.

**Rules:**
- The comparison function must be in `runq.c`, NOT in assembly; it must be additive; it must clean up both states and exit cleanly.
- **NEVER edit ASM files based on guesses.** Always run `-compare-init 1` first to find the exact field that differs. Then trace back to the ASM code that sets that field.
- **If the DB is stale** (created before a code change), the comparison is meaningless. Always recreate the DB before debugging.
- **When not to use this tool:** if the hang is before `v7s_inference_create` returns (crash in init itself), use WRITE_STR/TRACE_* macros inside `v7s_inference_create` instead.

### 5. Attention Kernels
- **Export only `inner_attention2`.** No extra functions.
- **Self-contained:** No external calls, allocations, or I/O.
- **Thread-safe:** No static/global state.

### 6. Build System— Three Scripts, Strict Separation of Concerns

**There are exactly THREE bat scripts in the project root for build/verify:**


| Script | Purpose | May run validators? | May run inference? | May build DBs? |
|--------|---------|--------------------|--------------------|----------------|
| `build.bat` | Build ALL binaries (DLL + tools + validators + examples + server). Steps 2--6 (C/C++) are incremental via the printed freshness gate; `build.bat /F` forces a full C/C++ rebuild. | NO | NO | NO |
| `verify.bat` | Validate: build + run all validators + run inference smoke tests | YES | YES | YES (Qwen3 only) |
| `verify_llms.bat` | Run ALL present text LLMs for "paris" in both F32 and Q8 | NO | YES | NO |
| `build_text_model_databases.bat` | Build DBs for ALL text models in `weights\` (Qwen3 mandatory) | NO | NO | YES |
| `build_vit_databases.bat` | Build DBs for ALL VLMs in `weights\` | NO | NO | YES |

**Hard rules— NO EXCEPTIONS:**

1. **`build.bat` ONLY BUILDS.** It assembles, compiles, and links. It does NOT run validators, inference, or build databases. It runs the strict ASM comment checker (`check_asm_comments.py`) which autofixes indent/align and reports all other violations (NOCOMMENT, DIRECTCALL, RBPFRAME, RSPFRAME, NUMOFFSET, INI_NOCOMMENT) that MUST be hand-fixed before commit. There is ONE checker, not two— `check_asm_comments_autofix.py` was deleted as a duplicate.

2. **`verify.bat` is the ONLY script that runs validators.** NO other script may run validators. Validators run directly from verify.bat— no indirection through another script. verify.bat calls build.bat first, builds the Qwen3-0.6B database (mandatory), then runs all validators and inference smoke tests.

3. **No redundant build scripts.** All `build_v7.bat`, `build_all.bat`, `build_tools.bat`, `build_validators.bat`, `build_examples.bat`, `build_v7_server.bat`, `build_bayes_classify.bat`, `build_test_multiturn.bat`, `build_test_multiturn_manual.bat`, `build_autotune.bat`, `build_llava_db.bat`, `build_vision_db.bat`, `build_llama3_ref.bat`, `build_llama_ref.bat` are DELETED. One build.bat replaces them all.

4. **No directory scanning for model files.** The ONLY supported directory structure is `weights\<model_name>\<model_name>.bin`. No `models\` fallback, no `weights\<model>.bin` at the root, no multi-path lookups.

5. **Build steps may skip ONLY via an explicit, printed freshness gate.** A build step that is skipped must prove it is unnecessary from file state (source/header timestamps versus output timestamps), must print what it skipped and why, and must be overridable by a force flag (`build.bat /F`). Skipping is FORBIDDEN for a missing prerequisite, a failed probe, or convenience, a missing required input still ABORTs with FATAL. The ASM comment check and the NASM assembly in `build.bat` step 1 always run; only the C/C++ steps (2--6) are gated, by `src\check_stale.py` against the `src\build_c_targets.txt` manifest. **The gate may only ever cost time**, every ambiguity (missing output, missing input, unparseable manifest, checker exception) resolves to "rebuild", never to "skip". The two model-database scripts are the one other exception: they skip models whose `.bin` has not been fetched (Qwen3-0.6B is mandatory and FATAL if missing).

6. **Database builds are separate from compilation.** `build.bat` does not build databases. `build_text_model_databases.bat` builds text model DBs. `build_vit_databases.bat` builds VLM DBs. `verify.bat` builds the Qwen3 DB as part of its pipeline.

7. **`verify_llms.bat` tests ALL present text LLMs.** Qwen3-0.6B is mandatory (FATAL if missing). All other models are skipped if their database does not exist. Tests both F32 and Q8 for every model.

8. **All builds and runs via .bat files only**, never from the PowerShell/CMD prompt directly. Calling `cl`, `nasm`, `link`, or any executable directly from the AI prompt is FORBIDDEN.

9. **INFERENCE TESTS MUST ALSO RUN VIA .BAT FILES— NEVER START AN INFERENCE EXE DIRECTLY.** Every inference run must be launched from a `.bat` file. If no suitable `.bat` exists, CREATE one first, then run it.

10. **Pre-commit verification:** Run `verify.bat` (builds + all validators + inference tests). Pass or don't commit.

11. **No parallel benchmark runs.** One process at a time.

11a. **The build system and verify.bat MUST provide proper output that allows us to follow what is
happening.** Redirecting JUST EVERYTHING to `>nul` is DISCOURAGED— AS ALL OUTPUT REDIRECTION IS
GENERALLY DISCOURAGED. Every build step, validator, and inference run must print its progress and
results visibly to the console so the user can follow the pipeline in real time. Output redirection
(`>nul`, `> file`, `2>&1`) is acceptable ONLY when there is a concrete need (e.g. capturing a tool's
output for post-processing by another tool in the same command line), and even then the captured content
must be shown to the user afterwards. NEVER silence build/verify output to "keep the console clean"— the
output IS the deliverable.

11b. **`verify.bat` must NEVER check-and-skip a phase— this is a CRITICAL BUG.** Every verify phase
either RUNS its gate, or the phase does not exist at all. A phase that prints "skipped" (or otherwise
silently omits its gate while keeping the phase stub, a renumbered-totals lie, or any other
check-and-skip shape) reports green while testing nothing— exactly the bug-hiding behavior Sec.130
exists to prevent. When a gate's script or tool is removed, the phase is removed WITH it in the same
change— never left behind as a skip stub, and the remaining phase numbering stays truthful.

11c. **Every actionable verification failure must be reported; a passing result must not conceal
failed checks.** The documented external-memory-pressure exception for `v_generate_validate` is described
in Section 130a. Every validator, every inference test, every gate: a failure MUST produce a visible FAIL report
and a non-zero exit. No failure may be swallowed, downgraded to a warning, printed without affecting the
exit code, or left out of the summary. If any phase detects a failure, `verify.bat` exits non-zero— a
green summary and exit code 0 mean everything genuinely passed, nothing else.

12. **Show output to the user— never redirect it away.** Run commands with stdout/stderr visible in the terminal so the user can observe results in real time. Only redirect output to files when there is a concrete need. Never silently discard or redirect output just to "keep things clean".

**Keep diagnostic output visible in the terminal.** Redirecting or filtering the output of build,
verification, inference, and Git commands can conceal failures or important context. Do not suppress
output with constructs such as `> nul`, `2> nul`, `git diff > file.txt`, or output filters. Read the
complete output before making decisions. Where captured output is explicitly needed for subsequent
processing, follow the documented exception and display the captured information to the user.


13. **Check DLL exports by reading `v7_threading.def` and source code— never use `dumpbin` or any external tool.** We have the source; inspecting it directly is always faster, more reliable, and doesn't require VS dev env setup.

14. **When told to update AGENTS.md, do it immediately.** This file is the source of truth for the AI's behavior.

15. **Never use subagents**: The subagent feature is broken and causes lock-out and very long wait times; avoid it entirely. Do NOT use the Task tool with subagent_type— it is permanently disabled and will hang indefinitely.

### 6a. C/C++ Code Style— Unbreakable Rules for Tools, Validators, Examples, Benchmarks

**THESE RULES ARE NOT SUGGESTIONS. THEY ARE MANDATORY. EVERY DEVIATION IS A BUG THAT MUST BE FIXED ON
SIGHT— NO EXCEPTIONS, NO "I'LL FIX IT LATER," NO "THIS ONE DOESN'T MATTER."** If you spot a violation
while reading, editing, reviewing, or building ANY C/C++ file, you fix it RIGHT THEN— in the same edit,
in the same session. A known violation that is not fixed immediately is itself a violation of this
document.

**Deviations from these rules are not cosmetic nitpicks— they are BUGS that damage software quality.**
A K&R brace hides scope errors. A `fprintf(stderr, ...)` fragments the output stream. A cryptic
variable name wastes every future reader's time. A `size_t` loop counter invites 32-bit truncation on
larger models. A `static` on a file-scope global in a single-file program is dead noise that
desensitizes the reader to real `static` uses. Every single deviation makes the codebase worse— less
readable, less maintainable, more error-prone; this is not opinion. This is engineering reality borne
out by decades of bug reports tracing back to exactly these classes of defects. Fixing them is not
optional busywork— it is mandatory bug eradication, same as fixing a crash or a wrong-output bug.

**CURRENT GOAL: Make the ENTIRE C/C++ codebase fully conformant with these rules.** Every `.h`, `.c`,
and `.cpp` file in `src/`, `tools/`, and `examples/` must be audited and fixed. Assembly files (`.inc`,
`.asm`) are NOT touched for this— only C/C++. Where a rule cannot technically be applied (e.g. a
third-party header we do not control, a Windows API signature), document the exception inline with a
comment. Otherwise, every violation found is fixed on sight.

1. **`printf` is the ONLY allowed output stream. PERIOD.** `fprintf(stdout, ...)` and `fprintf(stderr, ...)` are FORBIDDEN in ALL v7 C/C++ code— tools, validators, examples, benchmarks, servers, everything we write. All program output goes to stdout via `printf(...)`. There is exactly ONE exception: `fprintf` is allowed when writing to an actual disk file (i.e. `FILE*` returned by `fopen`). Every `fprintf(stderr, ...)` or `fprintf(stdout, ...)` encountered in our code must be converted to `printf(...)` on sight. **This rule has NO special cases and NO "but stderr is better for errors" excuses.** The Zydis library and other third-party code are not ours to change— they are exempt. Our code is not.

2. **Struct/typedef definitions go at the top of the file, after includes.** All struct, typedef, and enum definitions in a `.c`/`.cpp` file MUST appear immediately after the `#include` block, before any function definitions or global variables. No struct definition may appear in the middle of a file after code. When a struct is found out of place, move it to the top immediately.

3. **`static` must be warranted or removed.** File-scope `static` globals in single-file programs (tools, validators, examples, benchmarks) are almost never warranted— the linker cannot see them anyway since they're the only translation unit. `static` is warranted ONLY for: (a) hiding a name from other translation units in multi-file programs, (b) `static inline` functions, (c) `static const` lookup tables at file scope, (d) function-local `static` for persistent state. Senseless `static` on file-scope globals in single-file programs must be removed on sight.

4. **No cryptic if-constructions.** Write readable, obvious code— "workbook C/C++." A condition like `if (!g_use_q8)` that means "if F32 path is unavailable" must be written as what it actually checks, with a clear comment or a helper variable. The reader should understand the intent without tracing multiple globals.

5. **`#pragma once` is the ONLY allowed include guard.** All `#ifndef FOO_H` / `#define FOO_H` / `#endif` include guard patterns must be replaced with `#pragma once` on sight. Every header file uses `#pragma once`— no exceptions.

6. **No `has_*` capability checks.** The DLL always exports every function. Checking `if (dll.has_v7s_sample)` or `if (lib.v7s_sectorlogo_interp)` before calling is BUG-HIDING— a missing export is a build/link error, not a runtime fallback. Every `has_*` field in `v7_lib.h` and every runtime check for a function pointer being non-null before calling it must be removed. Call unconditionally. If the export is missing, let it crash— that's the bug.

7. **No senseless `memset` after zero-initialization.** `struct v7_lib lib = {0};` already zeroes the struct. `memset(&lib, 0, sizeof(lib))` after `= {0}` is redundant. Remove on sight.

8. **Allman brace style— mandatory for all new or changed C/C++ code.** A single-statement body may appear directly after `if`/`for`/`while` without braces, or it may use braces— both are acceptable. When braces are used, the opening `{` is ALWAYS on its own line. For multiple-statement bodies, braces are required and the `{` is ALWAYS on its own line.

   **`{` ON ITS OWN LINE IS ABSOLUTE AND NON-NEGOTIABLE.** This is not a style preference. This is not open for discussion. A `{` that shares a line with `if`, `for`, `while`, `else`, or any other statement is a BUG— it damages readability, hides errors, and degrades software quality. Every deviation from this rule is a defect that must be fixed on sight. No "just this once." No "it's only one line." No "it's more compact." The `{` stands alone. Period.

   ```c
   // WRONG (K&R brace— { shares line with if)
   if (x > 0) {
       do_something();
   }

   // WRONG (same-line brace block— { not on own line AND multiple statements crammed)
   if (fscanf(f, "%31s %lld", dummy, &v) != 2) { fclose(f); return 1; }

   // CORRECT (multiple statements— braces required, { on its own line)
   if (fscanf(f, "%31s %lld", dummy, &v) != 2)
   {
       fclose(f);
       return 1;
   }

   // CORRECT (single statement— no braces, same line)
   if (x > 0) return 1;

   // ALSO CORRECT (single statement— braces used, { on its own line)
   if (x > 0)
   {
       return 1;
   }
   ```

9. **Parenthesize every sub-expression inside `if`/`while`/`for` conditions when combined with `&&`, `||`, `&`, `|`.** Every comparison, negation, function call, or other sub-expression must be wrapped in its own parentheses when combined with logical/binary operators, so the outer operators are the only unparenthesized tokens. This eliminates precedence ambiguity and makes compound conditions scannable at a glance. Single comparisons with no logical operators do NOT need extra parens.

   ```c
   // WRONG
   if (a == b && b == c)
   if (argc < 2 || !argv[1] || !argv[1][0]) return 1;
   if ((x > 0))                           /* single comparison, no logical op— no extra parens needed */

   // CORRECT
   if ((a == b) && (b == c))
   if ((argc < 2) || (!argv[1]) || (!argv[1][0])) return 1;
   if (x > 0)                             /* single comparison is fine as-is */
   ```

9a. **No complex function-call decision chains in `if` conditions.** Each `if` condition must test ONE
thing. When multiple I/O calls, parses, or function results must be checked, write EACH check as its own
`if` statement— never chain `fread(...) != 1 || fread(...) != 1 || fread(...) != 1` or `sscanf(...) != 1 || sscanf(...) != 1` into a single compound condition. Function calls in `if` conditions must be
getter-like (return a value to test) or used for simple error checking (one call, one result). A chain
of function calls combined with `&&`/`||` is a complex decision chain that must be broken into separate
`if` statements. Null-pointer allocation checks (e.g. `if (!a || !b || !c)`) are acceptable when all
pointers come from the same allocation block and the error handling is identical— but I/O calls, parses,
and multi-step validations must be separate.

   ```c
   // WRONG: three fread calls chained with ||
   if (fread(&a, sizeof(a), 1, f) != 1 || fread(&b, sizeof(b), 1, f) != 1 || fread(&c, sizeof(c), 1, f) != 1)
   {
       printf("truncated header\n");
       exit(1);
   }

   // CORRECT: each fread checked separately
   if (fread(&a, sizeof(a), 1, f) != 1)
   {
       printf("truncated header\n");
       exit(1);
   }
   if (fread(&b, sizeof(b), 1, f) != 1)
   {
       printf("truncated header\n");
       exit(1);
   }
   if (fread(&c, sizeof(c), 1, f) != 1)
   {
       printf("truncated header\n");
       exit(1);
   }

   // ALSO WRONG: sscanf chain
   if (argc < 6 || sscanf(argv[2], "%d", &n) != 1 || sscanf(argv[3], "%d", &d) != 1)

   // CORRECT: separate checks
   if (argc < 6)
   {
       printf("usage: ...\n");
       return 1;
   }
   if (sscanf(argv[2], "%d", &n) != 1)
   {
       printf("usage: ...\n");
       return 1;
   }
   if (sscanf(argv[3], "%d", &d) != 1)
   {
       printf("usage: ...\n");
       return 1;
   }
   ```

10. **No senseless `#define` gating macros.** When including a library header always provides the implementation (e.g. `#define V7_LIB_IMPLEMENTATION` before `#include "v7_lib.h"`), the `#define` is pointless ceremony that must be removed. The header must unconditionally provide the implementation— no opt-in required. A gating macro is only acceptable if there is a genuine scenario where the implementation is NOT needed (e.g. a header used in both a consumer and a provider translation unit). If every includer needs the implementation, the gate is dead code— remove the `#define` and make the header unconditional.

11. **No default generation limits except the output buffer size.** Every hardcoded generation cap (`max_gen = 200`, `DEFAULT_MAX_TOKENS = 256`, etc.) that is not the physical buffer limit is an unwarranted engine-decides restriction (Sec.154). Remove on sight. The user controls generation length via `-n`; the default is unlimited (run until EOS). The only acceptable cap is the output buffer size— that is a hardware constraint, not an artificial limit.

12. **Do not touch the sampler unless the CLI explicitly provides sampling parameters.** Setting `sam->temperature`, `sam->topp`, `sam->rng_state`, or any other sampler field from hardcoded defaults in C code is FORBIDDEN— the sampler is initialized by `v7s_inference_create` and MUST be left at its defaults unless the user passes `-t`, `-p`, `-s`, etc. on the command line. The only valid sampler write is one driven by a CLI flag. Remove all hardcoded sampler assignments on sight.

12a. **Tools that have nothing to do with the sampler NEVER touch it— no debug_sampler structs, no
ctx->sampler access, no ctx->quantize writes.** A tool that runs inference via `v7s_chat` /
`v7s_chat_image_mem` calls the API and reads the output— it does NOT reach into the inference context to
tweak sampler state or quantization mode. The sampler belongs to the DLL; the quantization mode is set
by the DLL at init time. If a tool needs to control quantization, the DLL must expose a proper API for
it— the tool must NEVER write into `ctx->quantize` or any other internal field. Remove all
`debug_sampler` struct definitions, all `ctx->sampler` pointer dereferences, and all `ctx->quantize` /
`ctx->*` writes from tools that are not sampler-testing tools. The inference context will eventually
move to `.bss` and disappear from C entirely— any code that depends on reaching into it is already
wrong.

12b. **Any access to members of `ctx->` must be warranted by the tool's purpose.** The inference context
(`inference_context*`) is transitioning to an opaque handle— `v7s_inference_create` will eventually
return a simple success/fail with all state living in `.bss` inside the DLL. When that happens,
`ctx->anything` will not compile. Right now, every `ctx->member` read or write in every tool must
justify itself: does the tool's stated purpose REQUIRE touching that field? If not, remove the access.
Examples: a bbox evaluation tool has no business reading `ctx->sampler`; a sampler validator has every
business reading it. When in doubt, the tool should call the public API (`v7s_chat`, `v7s_generate`,
etc.) and treat the context as opaque— if the public API cannot do what the tool needs, the API must be
extended, not the context pried open. **Exception: `ctx->quantize` is the current mechanism for tools to
select F32 vs Q8 mode— this is warranted access for any tool that runs inference, until the DLL exposes
a proper API for it.**

13. **No checks whether the inference context was created.** `v7s_inference_create` either returns a valid context or the engine has already FATAL'd. Checking `if (!ctx)` and printing an error is dead code— the FATAL already terminated the process. Remove on sight. (In the future the context will move to `.bss` and the pointer will vanish entirely.)

14. **Always use `typedef struct`— never bare `struct`.**
    ```c
    // WRONG
    struct kt_tensor_q8 {
        int8_t* input_q;
    };

    // CORRECT
    typedef struct kt_tensor_q8
    {
        char* input_q;
    } kt_tensor_q8;
    ```

15. **Anonymous structs are ILLEGAL— every struct MUST be a named `typedef struct` at the top of the file.** No inline anonymous struct definitions inside function bodies, no `struct { ... } *ptr = ...` casts, no `decltype`-based struct pointer trickery. Every struct used in our code must be defined as a `typedef struct { ... } name;` at the top of the file (per rule 2), and all pointers to it must use the typedef name. Anonymous/inline struct definitions are bugs— fix on sight by extracting the struct to a named typedef at file scope.

16. **Never use `int8_t` / `uint8_t` / `int16_t` / `uint16_t` / `int32_t` / `uint32_t` / `int64_t` / `uint64_t` / `size_t` without a proper reason.** Use `char`, `unsigned char`, `short`, `unsigned short`, `int`, `unsigned int`, `__int64`, `unsigned __int64` instead. The `_t` typedefs from `<stdint.h>` are only acceptable when there is a genuine portability requirement (e.g. a wire-format struct with exact-width fields). In our codebase, running on MSVC x64, plain C types are clearer and more idiomatic. Replace `_t` types on sight unless there is a concrete reason. **`size_t` is the worst offender**— `for (size_t i = 0; i < n; i++)` is clutter when `n` and `i` always fit in an `int`. Check every `size_t` usage: if `int` is enough in reality, use `int`. **CAST AT THE INTERFACE BORDER, NEVER PROPAGATE `_t` TYPES INTO OUR CODE.** When an external API (Windows, Zydis, third-party library, or v7 DLL exported struct fields like `v7_q8_tensor.q` which is `char*`) requires a `_t` type at its boundary, our code uses our native types (`char*` not `int8_t*`) and casts only at the call site. The cast is `(int8_t*)our_char_ptr` when calling the external function— never the reverse (changing our `char*` to `int8_t*` to "match" the external type). The external type stays outside; our type stays inside.

16. **Global variables must be defined after struct/typedef definitions.** The file layout is: (1) `#include` block, (2) `#define` block, (3) struct/typedef/enum definitions, (4) global variables, (5) function definitions. No globals may appear before the struct definitions they depend on, and no globals may be interleaved with function code.

17. **Never guard `#define`s with `#ifndef`.** Writing `#ifndef STATUS_GUARD_PAGE_VIOLATION` / `#define STATUS_GUARD_PAGE_VIOLATION ...` / `#endif` is defensive clutter. Our codebase is self-contained— if the define is needed, define it. If it conflicts, fix the conflict. Remove `#ifndef` guards around `#define`s on sight.

18. **`#define`s must be at the top of the file in one contiguous block.** After `#include`s, before struct definitions. No `#define`s scattered in the middle of code. If there is a very good reason to place one elsewhere (e.g. a conditional that must be near its sole consumer), document it with a comment. Otherwise, move it to the top block.

19. **When converting `fprintf` to `printf`, remove the senseless `fflush`.** `fprintf(stderr, "text\n"); fflush(stderr);` becomes just `printf("text\n");`— stdout is line-buffered and `printf` does not need a flush. Every `fflush(stderr)` or `fflush(stdout)` left behind after a `fprintf`?`printf` conversion is dead code. Remove on sight.

20. **Variable definitions must be separated from the opening brace and from the code body by blank lines.**
    ```c
    // WRONG
    int parse_zydis_flags(int argc, char** argv) {
        int first_nonflag = 1;
        for (int i = 1; i < argc; i++) {

    // CORRECT
    int parse_zydis_flags(int argc, char** argv)
    {
        int first_nonflag = 1;

        for (int i = 1; i < argc; i++)
        {
    ```
    Empty line after the opening `{`, empty line after the last variable definition, before the first executable statement.

21. **Avoid the `?:` ternary operator when a plain `if`/`else` is clearer.** `return (g_fail > 0)— 1 : 0;` must be written as:
    ```c
    if (g_fail)
        return 1;
    else
        return 0;
    ```
    The ternary is acceptable ONLY when it genuinely simplifies (e.g. inside a macro, or a trivial initializer like `int x = cond— a : b;` where both branches are simple constants). When in doubt, use `if`/`else`.

22. **Single-statement `if` keeps the statement on the same line.** When an `if` body is a single simple statement (assignment, return, break, continue, function call), keep it on the same line as the condition— no braces needed. You MAY also use an Allman block for a single statement (braces with `{` on its own line per rule 8)— both are acceptable. When the body has multiple statements, use an Allman block (rule 8)— the `{` on its own line. **Only ONE statement may follow `if` on the same line.** If you need two or more, you MUST use an Allman block— never cram multiple statements into a same-line `{ }` block.

    ```c
    // WRONG (multiple statements on same line with K&R braces)
    if (fscanf(f, "%31s %lld", dummy, &v) != 2) { fclose(f); return 1; }

    // CORRECT (single statement— no braces, same line)
    if (g_fail) g_fail = 1;
    return g_fail;

    // ALSO CORRECT (single statement— braces used, { on own line)
    if (g_fail)
    {
        g_fail = 1;
    }
    return g_fail;

    // CORRECT (multiple statements— Allman block, { on own line)
    if (fscanf(f, "%31s %lld", dummy, &v) != 2)
    {
        fclose(f);
        return 1;
    }
    ```

23. **No `static` on file-scope globals.** Every `static` on a global variable in a single-file program (tools, validators, examples, benchmarks) is clutter— the linker cannot see them anyway. `static` is warranted ONLY for: (a) hiding a name from other translation units in multi-file programs, (b) `static inline` functions, (c) `static const` lookup tables at file scope, (d) function-local `static` for persistent state. Remove senseless `static` on file-scope globals in single-file programs on sight. Every remaining `static` must have a concrete reason the code would fail without it.

24. **Every variable on its own line.** Never declare multiple variables in one statement. `int g_pass = 0, g_fail = 0;` is WRONG— each variable gets its own line with its own type and initializer. This makes grepping, diffing, and debugging easier.

25. **Use `0`, not `nullptr` or `NULL`.** Our codebase is C and C89-compatible C++. `0` is idiomatic, shorter, and unambiguous. Replace `nullptr` and `NULL` on sight.

26. **Speaking variable names— no cryptic abbreviations, no `g_` prefixes, NO 2-3 letter non-speaking names, NO single-letter parameters, NO unreadable struct members.** Variables must be named so a reader understands their purpose on first sight. `g_lib` is almost devoid of information— it tells you it's global (obvious from context) but not what library or what it's for. `inference_dll_ptr` says everything. Remove `g_`/`m_`/`s_` prefixes on sight— a compact function makes it obvious which variables are global. The name itself must carry the meaning. **2-3 letter variable names like `ci`, `nc`, `nr`, `fp`, `rb`, `cb`, `dp` are BUGS— they tell the reader NOTHING about what the variable holds. Use `column_index`, `num_candidates`, `num_rows`, `file_ptr`, `read_buffer`, `callback`, `data_ptr` instead. The ONLY exceptions are loop counters like `i`, `j`, `k` in trivial scopes (1-5 lines) where the meaning is obvious from the loop body. Fix non-speaking short names on sight.** **Function parameters follow the same rule: `p` as a config pointer is unreadable— use `config_pointer` or `pointer_to_config`. `s` as a RunState pointer is meaningless— use `state_pointer` or `run_state_ptr`. Every parameter name must tell the reader what it holds.**

**THIS APPLIES TO STRUCT MEMBERS WITH THE SAME FORCE.** Struct members like `.v`, `.ms`, `.vs`,
`.mtl`, `.bos`, `.eos`, `.pt`, `.spt` are CRITICAL BUGS— they are even worse than short locals because
a struct member is read by people who do NOT have the surrounding context. A member called `.v` could
be anything— vocab? vector? value? version? A member called `.ms` could be merge_scores? milliseconds?
message_size? **Every struct member must be a speaking name: `.vocab`, `.merge_scores`, `.vocab_size`,
`.max_token_length`, `.bos_token_id`, `.eos_token_id`, `.prompt_template`,
`.system_prompt_template`.** When you encounter a non-speaking struct member— whether in code you are
writing, code you are editing, or code you are merely reading— it MUST be renamed immediately, in the
same session, with no deferral. This is not optional; this is not a style preference. Non-speaking
names are bugs that waste every future reader's time.

26a. **No `static_assert`, no `assert`— EVER.** `static_assert` validating field offsets or struct sizes
of types WE defined ourselves is senseless: we created the struct, we know its layout. Checking
`offsetof(v7_run_state, x) == 0` proves nothing— if the struct definition is wrong, the assert was
written from the same wrong definition. `assert()` at runtime is a parallel mechanism to our FATAL
system— it either duplicates a check FATAL already makes, or it checks something only in debug builds
(meaning the check is silently disabled in release, which is a bug-hiding path). **Every `static_assert`
and every `assert` must be removed on sight.** If the check is important, replace it with a proper
runtime check that always runs (and FATALs on failure). If the check validates our own struct layout,
just remove it— it proves nothing and clutters the code.

    ```c
    // WRONG
    static int g_pass = 0, g_fail = 0;
    static v7_lib* g_lib = nullptr;

    // CORRECT
    int passing_tests_counter = 0;
    int failing_tests_counter = 0;
    v7_lib* inference_dll_ptr = 0;
    ```

    **Function names follow the same rule.** `test_s` is meaningless— what does `_s` mean? What does it test? A reader should understand the function's purpose from its name alone. `compare_string_result` tells you exactly what it does. Parameters too: `desc` is vague (description of what?), `expected` is ambiguous (expected by whom?)— use `test_label` and `reference_result`.

    ```c
    // WRONG
    static void test_s(const char* desc, const char* value, const char* expected)

    // CORRECT
    void compare_string_result(const char* test_label, const char* actual_value, const char* reference_result)
    ```

27. **No `const` on function parameters.** `int v7_load(const char* dll_path, const v7_lib* lib)` is clutter— the compiler already knows the value won't change at the call site, and `const` on a by-value parameter is meaningless (the copy is ours regardless). `const` on pointer parameters that point to read-only data is acceptable when it documents a contract (e.g. `const float* weights` = "this function does not modify the weights"), but `const` on value parameters (`const int n`) is always dead noise and must be removed on sight. `const` on local variables (`const char* prompt = argv[1];`) is equally senseless— the variable is local, the compiler knows it won't change, and the `const` adds zero value. Remove on sight.

28. **No `long long` when `int` suffices.** We are on a 64-bit system where `int` is 32 bits. If the value fits in an `int`, use `int`. Never use `long long`, `long`, or other exotic types when `int` would do. Use `__int64` only when the value genuinely exceeds 32 bits. Same principle as rule 15— plain C types, no unnecessary width.

28a. **No `volatile`— use C++ atomics instead.** The `volatile` keyword is NOT a threading primitive and
provides no memory ordering guarantees. In C++ code, use `std::atomic<T>` with explicit
`.load()`/`.store()`/`.fetch_add()` for any variable shared across threads (e.g. server flags like
`server_running`, `request_counter`). In C code, use Windows `Interlocked*` functions. `volatile` exists
in our codebase ONLY as a historical artifact— remove it on sight and replace with the proper atomic
primitive. The only valid use of `volatile` is for hardware register access (MMIO), which our codebase
never does.

29. **No unnecessary initializers.** `char output[4096] = {0};` is wrong when the buffer is immediately overwritten by a sprintf-style write— the zero-initialization serves no functional purpose and wastes cycles. Only initialize when needed for correctness (e.g. a counter that must start at 0, a flag that must start false, a string that must start empty). Remove senseless `= {0}` on sight when the value is overwritten before it is ever read.

30. **No unnecessary casts.** `(char*)prompt` when `prompt` is already `char*` is noise. `(HMODULE)lib.module` when the API accepts `void*` is noise. Cast only when required for correctness (e.g. narrowing conversions, void* to typed pointer across an API boundary). When a cast IS needed, always use C-style casts (`(int)x`), never C++ named casts (`static_cast<int>(x)`, `reinterpret_cast`, etc.)— the C-style cast is shorter, more readable, and we are writing C/C89-compatible code.

31. **All lowercase for variable names, struct names, function names, and type names.** No uppercase characters anywhere in identifiers. `STOP`— `stopwords`, `V7_init`— `v7_init` (the DLL export name is a separate concern— our C code uses lowercase). `kt_tensor_q8` is correct; `CpuSnapshot`— `cpu_snapshot`. The only exceptions are macro/`#define` names, which follow `UPPER_SNAKE_CASE` by convention.

32. **No assignments inside `if`/`while`/`for` conditions.** Every instruction must be on its own line. A semicolon is ALWAYS followed by a newline. We try to avoid cluttering our namespace if possible— declare loop variables inside the loop body, not in the condition.

    ```c
    // WRONG
    int c; while ((c = fgetc(f)) != EOF) putchar(c);

    // CORRECT
    while (1)
    {
        int c = fgetc(f);
        if (c == EOF) break;
        putchar(c);
    }
    ```

33. **Never fail silently.** Every error must produce visible output. `if (!f) return 1;` swallows the error— the user has no idea what went wrong. Print a message before returning:

    ```c
    // WRONG
    FILE* f = fopen("path", "wb");
    if (!f) return 1;

    // CORRECT
    FILE* f = fopen("path", "wb");
    if (!f)
    {
        printf("file not found!\n");
        return 1;
    }
    ```

34. **Use blank lines to separate logical code blocks.** Group related statements together, then separate from the next logical block with a blank line. This makes control flow scannable.

    ```c
    // WRONG
    int a = 1;
    int b = 2;
    for (int i = 0; i < n; i++) { a += i; }
    int c = a + b;

    // CORRECT
    int a = 1;
    int b = 2;

    for (int i = 0; i < n; i++)
    {
        a += i;
    }

    int c = a + b;
    ```

35. **No `void*` when a proper type exists.** Using `void*` for a parameter or variable when the actual type is known (e.g. `void* file` instead of `FILE* file`) is type-erasure that loses information and requires casts at every use site. `void*` is only acceptable when there is a genuine reason (e.g. an opaque handle from an API, a generic buffer that holds arbitrary data). When the type is known, use it directly. Replace `void*` with the proper type on sight.

36. **No `&` (C++ references) unless warranted by: (1) avoiding copying large data >64KB, or (2) changing data in the caller as a secondary return.** Otherwise always pass raw type by value. `&` hides side effects. When a C++ function takes `std::string&` and does not modify it, use `const std::string&` (the one place `const` is allowed— without it the code won't compile when called with temporaries).

37. **Console output must be non-aggressive and agreeable but detailed.** No `=====` separator lines or `********` banners. Use `.....` dots instead: `printf("\n..................\n");`. Avoid ALL-CAPS labels, exclamation marks, and visual aggression in printf output. The tone is calm, informative, detailed. When you spot aggressive formatting, fix it. **ALL-CAPS in printf output is FORBIDDEN— zero exceptions, zero tolerance.** Write `"Test result:"` not `"TEST RESULT:"`, `"Turn #1"` not `"TURN #1"`, `"Done"` not `"DONE"`. Every label, heading, status word, or section marker in any printf/fprintf/write statement must use normal mixed-case— never all-caps. This applies to every program we write: tools, validators, examples, servers, benchmarks. The only exception is acronyms that are conventionally all-caps (e.g., `"Q8"`, `"F32"`, `"API"`, `"INI"`), but section headings, status labels, and result markers are NEVER all-caps. When you see all-caps output, fix it.

38. **No senseless `{}` zero-initialization of structs.** `v7_lib lib = {};` is forbidden— the zero-initialization is unnecessary ceremony. Use `v7_lib lib;` and let the code fill the fields. `= {0}` is only acceptable for arrays or when the zero state is the correct initial state that the code relies on before writing. Remove `= {}` on sight for struct instances.

39. **`printf` lines should NOT be split across lines unless they exceed ~256 chars.** Prefer single-line printf statements. Do not wrap printf format strings across multiple lines just for style.

40. **Use `#define` for compile-time constants** that replace what would be `const int` locals used for array sizing or similar (e.g. `#define NV 20` instead of `int NV = 20;` for VLA sizes).

41. **Indent with tabulators. One tab per indentation level.** No spaces for indentation. Tabs are the only acceptable indentation character. Set your editor tab width to whatever you prefer for display, but the source file MUST contain tab characters for indentation, not spaces.

42. **Lines may be up to 250 characters long.** Do not break lines just to fit an arbitrary 80/120 column limit. A 200-character line that is readable as a unit is better than 5 lines of wrapped fragments. Only break lines when they genuinely exceed 250 chars or when breaking improves readability.

43. **No `inline` unless genuinely needed for performance.** `inline` is a hint to the compiler, not a guarantee— modern compilers inline at their own discretion with or without it. Adding `inline` to every small function is noise that desensitizes the reader. Use `inline` ONLY on functions that are proven hot-path and where inlining makes a measurable difference. Everywhere else, let the compiler decide. Remove `inline` on sight unless there is a concrete performance reason it must be there.

44. **No unnecessary casts.** `(char*)prompt` when `prompt` is already `char*` is noise. `(HMODULE)lib.module` when the API accepts `void*` is noise. `(int)strlen(...)` when the result is assigned to an `int` variable is noise— the implicit conversion does the same thing. Cast only when required for correctness (e.g. narrowing conversions, `void*` to typed pointer across an API boundary). When a cast IS needed, always use C-style casts (`(int)x`), never C++ named casts (`static_cast<int>(x)`, `reinterpret_cast`, etc.)— the C-style cast is shorter, more readable, and we are writing C/C89-compatible code.

45. **New comments use `//` only, never `/* */`.** The `/* */` style hampers our ability to use it to comment out blocks of code during development. Existing `/* */` comments are not converted, but all new comments must use `//`. Every `/* */` in new code is a style violation.

### 7. File-System Restrictions
- Work only inside `C:\project\v7\`.
- Temporary logs go to `C:\project\v7\build_logs\`. `build_logs\` is NOT a tool-storage directory, it holds transient run output only and may be wiped at any time. `build.bat` wipes its contents on every build.
- `C:\project\v7\new_tools\` holds only the HF box-test reference outputs (`box_test_ref\`). The tool driver bats were removed in the project cleanup, vision/VLM testing runs `run_vision_smoke.bat` and `test_vlms.bat` from the project root (Sec.135), and the engine test programs live in `tools\` (see Sec.11a: tests are never examples and never bat files).

### 8. Stack-Frame & ABI
- **All new functions must use the `SF_*` macros** (from `macros/v7_apicall.inc`).
- Follow Windows?x64 ABI: 32-byte shadow space, 16-byte RSP alignment, preserve non-volatile registers, no red-zone usage.

### 9. Assembly Style
- **Every instruction needs a meaningful comment** (what the value means, not "load rax").
- **Indent = 5 spaces, comment = 7 spaces from code**.
- No autogenerated comments; write them by hand.
- **All numeric offsets must use `%define` constants**— never use raw numbers like `[rbp+48]` or `[r12d-1]` in instructions.
- **All `check_asm_comments.py` results MUST be fixed** before any build/commit. Zero warnings tolerated. These requirements must NEVER be relaxed, waived, or suppressed for any reason.
- **NEVER spend time manually aligning comments to a column.** Comment alignment is auto-fixed by the checker/tooling at build time. Do NOT hand-count spaces to line up `;` comments— just write the code and a meaningful comment, and move on. Manual column alignment is wasted effort.
- **Improve comments proactively**— when you spot weak, wrong, or purely mechanical comments while editing, fix them to be meaningful.
- **Never use worse instruction sequences out of convenience.** Always use the most efficient/idiomatic instruction sequence for the task.
- **Never use `shl` + scalar-add to avoid `*N` scale addressing.** The `*1,2,4,8` index scale is valid NASM and more efficient. All scalar constants in indexed addressing MUST use a named `%define` for `N` (e.g. `DWORD_SCALE`).
- **Never use the listing file for debugging.** NASM is a proven assembler and always produces correct output from correct source. Debug the source code, not the listing. If the machine code doesn't match expectations, the bug is in the source instructions, not the assembler. This applies to C compilers too— MSVC, clang, gcc all produce correct machine code. Find the source bug.

### 4c. Optimization Sweep— RMW Zeroing Rule
- **Never use `and [mem], 0` to zero memory**— always use `mov [mem], 0` (or `mov dword/qword [mem], 0`). The RMW instruction forces a load+and+store dependency chain, causing store-forwarding stalls. A pure `mov` write has zero memory dependency and is strictly faster.
- **Scan all `and [mem], 0` occurrences during ANY codepath edit and convert them.** This applies to both `dword` and `qword` forms, project-wide.

### 9a. Macro Design— Contracts Over Universality

**Macros define the contract; callers conform to it— never the reverse.** Do not try to design
"universal" macros that handle every possible register width, addressing mode, and calling convention.
This is impossible and leads to brittle code that silently does the wrong thing.

- **Design macros with a specific, documented interface**— which registers are inputs, which are outputs, which are clobbered. The interface is the contract.
- **Callers adjust their register usage to match the macro's interface.** If the macro expects a 64-bit index register in `rbx`, the caller uses `rbx` (not `ebx`) for the loop counter.
- **Well-designed macros support chaining**— one macro's output interface feeds the input interface of the next. For example, `JOB_ADDR`— `JOB_SET_*`— `JOB_LOAD_*` chain with no register shuffling between them.
- **Good "near-universal" macros** (e.g., `API_CALL`, `TRACE_*`) are fine because they have well-defined, stable interfaces that are easy to conform to.
- **For big abstractions** (job queue access, dispatch, memory allocation), it is EXPECTED that the macro has specific register requirements and a fixed interface. The complexity lives in the macro implementation, not at every call site.
- **Unify kernel and dispatch interfaces to minimize code.** All kernels should share the same calling convention (e.g., rcx=xout, rdx=&input_tensor, r8=&weight_tensor, r9d=n, stack=d, col_off) so that a single mode-dispatch in `process_job` handles every kernel type.
- **When a macro interface conflicts with existing caller register usage, CHANGE THE CALLER to match the macro**— do not add register-width branches, scratch-register fallbacks, or alternative entry points to the macro.

**Current state (verified):** The `JOB_ADDR` macro's 2-arg form requires a 64-bit index register. All
callers must use 64-bit registers (rbx, not ebx) for the index parameter, and adjust their loop
control code accordingly; this is the correct pattern going forward.

### 11. Build & Test
- **Always run `verify.bat`** after any change— it builds the DLL, runs ALL validators, and runs the inference tests. Read the full output to check every validator result and the LLM output. Kernel performance is checked by the standalone `bench_regression.bat` when dispatch/kernel code changed (it is deliberately NOT a verify phase, see Sec.54).
- **Model file for inference tests:** The reference checkpoint is always **`weights\qwen3_0.6b.bin`**— not the `qwen30.6b-base` variant.
- **CI reality (stated explicitly so nobody assumes otherwise):** `.gitlab-ci.yml` is a Linux lint-only job. NO Windows runner exists, so `build.bat` + `verify.bat`, the actual verification gate, run 100% MANUALLY on the dev machine. Nothing automated proves a commit green; the discipline of running verify.bat before every commit (Sec.130) IS the gate. If a Windows runner ever becomes available, add a scheduled (not per-MR) stage that runs build.bat + verify.bat on the smallest model and archives the log.

### 11a. Validation Commands (standalone validators)

#### Validator Philosophy
- The standalone validators (`v_*.exe` in `tools\`) validate **valid inference scenarios** only. They are not intended to test deliberately invalid or pathological inputs; such cases may legitimately trigger `FATAL` or undefined behaviour.
- A `FATAL` is appropriate when an invalid state is encountered that should never arise during correct inference. We avoid cluttering hot-path code with `FATAL` checks merely to satisfy the validators.
- When a bug causes a `FATAL` during normal inference, the fix must address the underlying issue, not simply add a defensive check.
- **Validators must NEVER call any v7 kernel or dispatch function with invalid/impossible values** (e.g., zero-byte memcpy, unaligned dimensions, uninitialized pointer structs). Any validator that needs to test edge-case behavior must do so entirely in C reference code, not by calling v7 with invalid arguments.

#### One Validator Per Module, Monolithic Tests, No Test-Disabling Parameters

- **Every module is tested by a SINGLE monolithic validator if possible.** We do not want several validators for the same module, one `v_<module>_validate.exe` owns ALL test coverage for that module (SectorLogo: `v_sectorlogo_validate`; each kernel family: exactly one validator). New test coverage for a module is INTEGRATED into that module's existing validator, never added as a second program. When coverage grows, the validator grows, it is never split "because it got big", and a second validator for the same module is never created. Redundant or overlapping validators are deleted once their coverage is integrated.
- **Validators normally do NOT take parameters that disable tests.** A validator tests everything it was built to test on every run, no `-skip`, no `-only`, no "fast mode" that silently drops coverage. The one documented exception is `v_sectorlogo_validate` (it carries so many tests that selective suites are needed while debugging), and even there a default run with no arguments must execute every suite.
- **Tests in our system are NEVER bat files.** `verify.bat` is the ONLY bat file that runs tests, and it must not call other bat files that run tests, test logic lives in validators (C code) and in verify.bat phases, never in standalone test scripts. The ONLY exceptions are the three long-running suites that exist as separate bat files because they take far longer than a verify run: `run_vision_smoke.bat` (vision smoke, single Qwen3-VL baseline compare), `test_vlms.bat` (all VLMs, F32+Q8, full answers), and `verify_llms.bat` (all text LLMs, F32+Q8). No other test bat may be created, and there is no "quick" variant of a long suite.

#### Validator Reference

**All validators are run automatically by `verify.bat`.** Do NOT start individual validators manually—
always run `verify.bat` and check its output.

| Command | Purpose |
|---------|---------|
| `v_fused_validate.exe` | Fused matmul module: F32 matmul+RMSNorm+RoPE + Q8 fused matmul correctness |
| `v_matmul_validate.exe` | Q8 matmul + F32 matmul correctness |
| `v_quantize_validate.exe` | Float— int8 quantize correctness (GS=64) |
| `v_rmsnorm_validate.exe` | Norm-kernel family: RMSNorm, RMSNorm+quantize, RMSNorm+RoPE, fused add+RMSNorm (4 sections, one validator) |
| `v_add_inplace_validate.exe` | Element-wise add-inplace correctness |
| `v_swiglu_validate.exe` | SwiGLU module: F32?Q8 + Q8?Q8 correctness (both modes, one validator) |
| `v_memcpy_validate.exe` | Dispatched memcpy correctness |
| `v_mdb_validate.exe` | MicroDB stress test (ht_add/find, vocab, LZ4 incl. model-weights-scale blobs, persistence) |
| `v_sampler_validate.exe` | ASM sampler correctness (greedy must match) |
| `v_attention_validate.exe` | Flash attention correctness |
| `mdb_info.exe` / `mdb_list.exe` / `mdb_query_key.exe` / `mdb_search.exe` / `mdb_dump.exe` | MicroDB health/check (non-pass/fail, for inspection) |

### 12. Documentation & Progress
- Keep `attention_progress.md`, `bug_log.md`, and `benchmark_results.md` up-to-date.
- Do not edit progress reports; they are the source of truth.
- **Always update `opt_documentation/asm_trick.md`** when you discover a new, useful, or cool assembly trick during optimization work. This includes: (a) shorter or more efficient instruction sequences that replace longer ones, (b) clever x64 idioms (zero-extension, LEA arithmetic, branchless patterns), (c) register flexibility tricks, (d) any non-obvious insight about instruction behavior, encoding, or uarch interaction. Add a new numbered section with a descriptive title, the before/after pattern, and where the trick applies.

### 13. Non-Negotiable Rules
- **SMC is immutable**— never disable or replace it.
- **Never remove or replace existing AGENTS.md content unless explicitly requested.**
- **Never add exports or change interfaces without explicit user consent**— except when the current goal's implementation plan explicitly authorizes the new exports/interface changes it describes.
- **Every bug must be fixed— no exceptions, no deferrals.** Minor bugs accumulate in assembly code and compound into crashes, data corruption, and subtle miscomputations. "Low risk" and "minor" are never valid reasons to skip a fix. If a bug is confirmed real, fix it immediately in the same session it is found.
- **EVERY PERFORMANCE ISSUE IS A BUG THAT MUST BE FIXED IMMEDIATELY.** Performance issues are not "opportunities for future optimization"— they are bugs with the same severity as crashes or wrong output. The only exception is a fix that would change make numerical precision worse.
- **Defensive programming is just bad error handling— use FATAL.** Silent fallbacks, quiet skips, graceful degradation, and "safe" defaults are all bugs. The correct response to an invalid state is to `FATAL` immediately so the bug is visible and gets fixed— NOT to silently continue with garbage data. The only exception is truly expected runtime variation (e.g., optional features that are legitimately absent).
- **NEVER replace missing/failed values with defaults— MUST FATAL instead.** Every lookup (config key, INI value, microdb key, ht_find, GetPrivateProfileIntA, etc.) that returns a sentinel, zero, default, or fallback value because the lookup failed is a BUG. Crashing NOW with a clear FATAL message is the only correct behavior. Every missing value is a contract violation that must be fatal.

### 14. Debugging Aids
- **`WRITE_STR`/`WRITE_STR_NL` are for BUILDING OUTPUT CODE (production console output of the program itself), NOT for debugging.** They clobber ALL volatile registers (rax, rcx, rdx, r8-r11) and any value cached in a volatile register at the insertion point is silently destroyed. Using them as debug traces inside functions is WRONG in most cases. **For debugging, ALWAYS use the non-intrusive `TRACE_*` macros** (`TRACE_STR`, `TRACE_REG`, `TRACE_HEX`, `TRACE_NL` from `macros/trace_macro.inc`)— they save/restore all 16 GPRs + RFLAGS and can be inserted ANYWHERE without changing program behavior. Build with `TRACE_DEBUG=1` to enable them (release builds compile them to nothing).
- INT3-magic debugging (`INT3_DBG_*`) must work; Zydis crash disassembly is always enabled.

#### 14d. TRACE_* Macros— Transparent Diagnostic Output (`macros/trace_macro.inc`)

**`TRACE_*` macros print diagnostic output to stderr WITHOUT corrupting any CPU state.** They can be
inserted ANYWHERE in ASM code and the surrounding code continues exactly as if the macro never ran.

**Available macros:**
| Macro | Purpose |
|-------|---------|
| `TRACE_STR "text"` | Print a literal string + newline to stderr |
| `TRACE_HEX reg_or_mem` | Print "0xVALUE\n" (64-bit hex value of operand) |
| `TRACE_REG "rbx", rbx` | Print "regname=0xVALUE\n" |
| `TRACE_NL` | Print a bare newline to stderr |
| `TRACE_REGS` | Dump all 16 GPRs to stderr |

**Transparency guarantee:**
- Saves ALL 16 GPRs (rax?r15) + RFLAGS (including DF) via `pushfq` + 16 pushes.
- After printing (via `debug_ensure_std` + `WriteFile`), restores everything in exact reverse order.
- Stack alignment is handled internally— works at any RSP alignment state.

**Usage example:**
```nasm
     ; Inside any function, at any point:
     TRACE_STR "entered merge loop"
     TRACE_REG "r12", r12        ; print Tokenizer* value
     TRACE_HEX [r14 + r8*4]     ; print tokens[i] value
```

**Requirements:**
- `v7_core.inc` must be included BEFORE this file (for `debug_ensure_std`, `stderr_handle`, `API_CALL`, `DBG_*` offsets).
- Included from `v7.asm` (or any top-level `.asm` file).
- Uses `trace_strlen`, `trace_write_hex`, `trace_write_hex_nl` helper functions (also defined in the file).

**When to use TRACE_* vs INT3_DBG_*:**
- `TRACE_*`: When you need to see values at a specific point WITHOUT stopping execution. Non-invasive, works in release builds. Output goes to stderr.
- `INT3_DBG_*`: When you need instruction-by-instruction tracing or register clobber detection. Requires debug build (`-DZYDIS_TRACE`) and VEH handler.

**Internal design:**
- `TRACE_SAVE_ALL`: `pushfq` + push rax,rcx,rdx,r8,r9,r10,r11,rbx,rsi,rdi,r12,r13,r14,r15,rbp (8 + 15*8 = 128 bytes) + all 16 YMM registers via `vmovdqu` into a `sub rsp, 512` area (640 bytes total). The YMM area exists because the macros call Win32 helpers whose callees clobber every volatile vector register per the Win64 ABI (XMM0-5 outright, the UPPER 128 bits of all YMM0-15), kernels hold live 256-bit values in any of them.
- `TRACE_RESTORE_ALL`: Reload all 16 YMM registers, `add rsp, 512`, then pop in exact reverse order + `popfq`.
- Stack adjusted by `sub rsp, 24` or `sub rsp, 40` for shadow space (16-aligned).
- `vzeroupper` is NOT called (not needed— WriteFile handles SSE/AVX transitions internally).

#### 14a. INT3 Magic Debug System (Zydis)

**Activation:** Build all `.asm` files with `-DZYDIS_TRACE` (see `build_v7_asm_debug.bat`). Without this
define, all `INT3_DBG_*` macros expand to nothing (zero overhead in release).

**VEH handler** is in `vea_handler.h` (`tools\`, shared by all validators); it intercepts `INT3 +
0xDEADBEEF + cmd_byte + payload` sequences at runtime.

**Quick-reference cheat sheet** (from `zydis_macros.inc`):
```
INT3_DBG_STR "text"       — print string to stderr  (cmd=0)
INT3_DBG_REG 3            — dump RBX as hex         (cmd=1)
INT3_DBG_F32 0            — dump XMM0 as 4 floats   (cmd=8)
INT3_DBG_NL               — newline                 (cmd=0)
INT3_DBG_REGSAVE          — save 16 GPR+XMM snapshot (cmd=4)
INT3_DBG_REGCHECK         — compare regs vs snapshot (cmd=5)
INT3_DBG_MAGIC_SET        — poison nonvolatiles      (cmd=6)
INT3_DBG_MAGIC_CHECK      — detect clobbers+restore  (cmd=7)
INT3_DBG_STEP             — enable instruction trace  (cmd=2)
INT3_DBG_NOSTEP           — disable instruction trace (cmd=3)
```

**Register numbers:** 0=RAX 1=RCX 2=RDX 3=RBX 4=RSP 5=RBP 6=RSI 7=RDI 8=R8 9=R9 10=R10 11=R11 12=R12
13=R13 14=R14 15=R15

**Practical debugging workflow:**
1. Add `INT3_DBG_STR` / `INT3_DBG_REG` / `INT3_DBG_F32` calls at key points in kernel code
2. Build with `-DZYDIS_TRACE` (debug build) and run `v_fused_validate.exe v7_threading.dll`
3. Read the `[MAGIC]` output on stderr— it interleaves with validation output
4. Use `INT3_DBG_MAGIC_SET`/`CHECK` around suspect sections to detect register clobbers
5. Use `INT3_DBG_STEP` for instruction-by-instruction trace (heavy output— use sparingly)

**Sanity:** All macros are safe in ABI_LEAF functions— they don't clobber non-volatile registers in ways
that cross ABI boundaries.

**Easy debug workflow reminder:** INT3_DBG macros work in ANY `.inc` file (kernel, dispatch, worker,
etc.)— not just kernel code. Build with `build_debug.bat` (defines ZYDIS_TRACE) and run via
`v_fused_validate.exe v7_threading.dll`. Everything is already wired up. Just add
`INT3_DBG_STR`/`INT3_DBG_REG`/`INT3_DBG_F32` calls and rebuild. No extra tooling needed.

#### 14b. INT3 VEH Handler— Proven Correct, Leave Alone

**The INT3 VEH handler in `vea_handler.h` has proven itself correct across hundreds of debug runs.** It
does NOT corrupt the stack, does NOT modify registers beyond RIP advancement, and is NOT the cause of
any crash.

**Hard rules:**
- The VEH handler is not to be modified, analyzed, or reviewed for crash investigations.
- If a crash occurs with INT3_DBG macros active, the macros are revealing the bug, not causing it.
- The only exception: if a NEW crash appears where the first INT3 hit itself faults, consider whether recent changes to `zydis_macros.inc` or the VEH handler broke something— this has never happened.

#### 14c. Zydis Permanence— NEVER Disable, NEVER Comment Out, NEVER #ifdef Out

**Zydis crash disassembly is a permanent, non-negotiable requirement across the entire project.** Every
tool (standalone validators via `vea_handler.h`, `inference_example.exe`, `dbg_inference.exe`,
`runq.exe`, etc.) MUST include a Zydis crash-handler header and call `ZydisInit()` at startup.

**Hard rules:**
- The `#include "zydis_debug_header.h"` line in ANY source file is PERMANENT. It must NEVER be commented out, removed, or `#ifdef`'d out.
- The `ZydisInit()` call in `main()` is PERMANENT. It must NEVER be removed or guarded by a conditional.
- Zydis libraries must remain linked in ALL build configurations. No `#ifdef NDEBUG` guards around Zydis.
- The shared header `zydis_debug_header.h` is the SINGLE SOURCE OF TRUTH for the VEH handler. NO source file may define its own copy of the handler functions. Any tool that needs VEH/crash support includes this header— it does NOT duplicate the code.

**Crash dumps are imperfect— don't fix Zydis if it doesn't need fixing:** During a crash, the process
is in an undefined state— stack may be corrupted, heap may be gone, CRT may be unavailable. Partial
output, truncated disassembly, or missing register lines are EXPECTED; these are NOT bugs in
`zydis_debug_header.h`. Do NOT refactor, rewrite, or "improve" the crash dump code unless the VEH
handler itself is broken (it rarely is). The handler works; the crash it's reporting is the problem.

**Sanity-check at review time:** When reviewing any `.c`, `.cpp`, or `.h` file that uses
`AddVectoredExceptionHandler`, Zydis, or crash handling, verify that it either includes
`zydis_debug_header.h` or has an explicit, justified reason not to.

### 15. Design Constraints
- **No restructuring of the v7 core** (worker loop, job dispatch, etc.).
- Use "fake tensors" for any new data flow needed.

### 16. ASM Sampler— Status (Done)
**Status:** Complete. `v7s_sample` export provides fast approximate sampling (AVX2 polynomial softmax,
top-256 pre-filter, AES-NI RNG, multinomial CDF walk). Greedy tests match C reference; non-greedy tests
expected to differ due to float precision. Guarded by `-asm-sampler <0|1>` in `runq.c` (`v7s_sample`
func ptr + `has_v7s_sample` flag in `v7_lib.h`).

### 17. Reference & Running

- **Reference batch file:** `tb2.bat` in project root— always use this to run inference. It shows the exact command lines used for testing.
- **Model location:** Models are in `weights\` subdirectory. Reference model is `weights\qwen3_0.6b.bin` (~633MB).
- **CLI usage:** `.\runq.exe <checkpoint> <v7_threading.dll> [options]`
- **CLI options (all with defaults):**
  | Flag | Default | Description |
  |------|---------|-------------|
  | `-t <float>` | 1.0 | temperature in [0,inf], 0=greedy deterministic |
  | `-p <float>` | 0.9 | top-p nucleus sampling in [0,1], 1.0=off |
  | `-s <int>` | time(NULL) | random seed |
  | `-m <string>` | chat | mode: generate\|chat |
  | `-i <string>` | (none) | input prompt |
  | `-y <string>` | (none) | system prompt in chat mode |
  | `-r <int>` | 1 | reasoning mode: 0=no thinking, 1=thinking |
  | `-T <float>` | 0.0 | wall-clock timeout in seconds, 0=no timeout |
  | `-n <int>` | 0 | max tokens to generate, 0=unlimited |
  | `-rope <int>` | 0 | rope mode: 0=precomputed table, 1=inline cosf/sinf/powf |
  | `-attention <int>` | 0 | attention: 0=v7 scheduler, 1=pure C reference |
  | `-prefill <int>` | 1 | prefill: 0=per-token forward, 1=fused prefill for prompt |
  | `-q8 <int>` | 0 | quantize: 0=f32 activation forward, 1=int8 q8 forward |
  | `-fallbackforward <int>` | 1 | forward: 0=DLL, 1=C reference fallback |
  | `-fallbackprefill <int>` | 1 | prefill: 0=DLL, 1=C reference fallback |
  | `-asm-sampler <int>` | 1 | sampler: 0=force C fallback, 1=use ASM v7s_sample |
  | `-minimal-gen <int>` | 0 | generate: 0=normal generate, 1=minimal generate (buffer output, no bench) |
- **Typical test command (from tb2.bat):**
  ```
  .\runq.exe weights\qwen3_0.6b.bin v7_threading.dll -i "what is a cat?" -t 0 -n 50 -q8 1
  .\runq.exe weights\qwen3_0.6b.bin v7_threading.dll -i "what is a cat?" -t 0 -n 50 -q8 0
  ```

### 18. ASM-Only Inference via MicroDB (Complete)

**Goal achieved:** A single DLL (`v7_threading.dll`) that C code uses via exactly 3 calls—
`v7s_inference_create()` (opaque context), `v7s_chat(ctx, "prompt", out, max)`
(tokenize?prefill?generate?decode?output), `v7s_inference_destroy(ctx)`. C code knows nothing about AI,
transformers, configs, tokenizers, or templates.

**Data source: Everything in microdb** (`lz4ht.db`): config, tokenizer, templates, model weights.
Two-phase lifecycle: (1) `build_db.exe` reads model+tokenizer+templates and persists to microdb; (2)
`inference_example.exe`/standalone calls `v7s_inference_create()` which loads from `lz4ht.db` and runs
fully in ASM.

**Key microdb keys:** `config`, `model_path`, `model_weights`, `weight_meta`, `tokenizer_hdr`,
`merge_scores`, `vocab_bulk`, 151936? reverse-lookup entries, `template`, `template_sys`, `mem_layout`.

**Design decisions (still operative):**
- RoPE tables: NOT loaded from microdb. ASM generates them via `v7s_rope_init(head_dim)`— `rope_build_internal_tables`.
- Model weights: directly from microdb heap (`ht_find("model_weights")`), no VirtualAlloc copy.
- Weight_meta: directly from microdb heap (`ht_find("weight_meta")`).
- QuantizedTensor descriptor arrays: one contiguous VirtualAlloc block, filled with computed q/s pointers at init.
- Token embedding: VirtualAlloc'd, dequantized in ASM with AVX2.
- KV caches (current engine): two separate FP16 allocations through `mdb_alloc_mem`, with a separate FP32 K/V staging block; the main working block stops at `ML_OFF_KEY_CACHE` (`src/asm/inf_state.inc`).
- Microdb locked to readonly after init via `db_set_readonly(1)`.

### 19. RoPE Table Lifecycle— Verified Correct

**Key fact: `rope_max_dim` (elements per table position) MUST equal `head_dim/2`.** The ASM table
builder (`fused_f32_kernel.inc:rope_build_internal_tables`) uses stride = `half` floats per position;
the Q8 path (`rmsnorm_rope_kernel.inc`) pre-offsets by `pos * rope_max_dim` elements;
`set_rope_tables(rope_cos, rope_sin, rope_max_dim)` must receive `head_dim/2`. A mismatch (passing
`head_dim` instead of `head_dim/2`) silently reads wrong table positions for pos = 1. This was a real
bug, fixed Jun 2026— do not regress it.

### 20. MicroDB Inspection— `mdb_*` DB tools

**Purpose:** Debug/inspect the `lz4ht.db` contents loaded into `v7_threading.dll` without needing C code
or special tooling.

**Usage:**
```
mdb_info.exe <dll_path> --db <db_path>
mdb_list.exe <dll_path> --db <db_path> [--all] [--max N]
mdb_query_key.exe <dll_path> <key> --db <db_path>
mdb_search.exe <dll_path> <pattern> --db <db_path> [--max N]
mdb_dump.exe <dll_path> --db <db_path> [--max N] [--bytes B]
```

**Tools:**

| Tool | Description |
|------|-------------|
| `mdb_info` | Hash table summary: capacity, entry count, persistent/temp breakdown, load factor, key pool usage |
| `mdb_list` | Show entries with slot index, P/T flag, data length, key name. Default shows non-vocab entries; `--all` includes vocab reverse-lookup (**very verbose**, 151k lines). Use `--max N` to limit. |
| `mdb_query_key` | Look up a single key and display type-aware output (MemLayout struct, weight_meta table, Config fields, hex dump with ASCII sidebar, string/float/int32 inference) |
| `mdb_search` | Iterate all entries and show those whose key contains the substring. Use `--max N` to limit matches |
| `mdb_dump` | Raw data dump of entries. Use `--max N` / `--bytes B` to limit |

**Global options:** `--db <path>` (required; no default per Sec.106), `--max N`, `--all`.

**Implementation notes:**
- Requires `g_ht` export from `v7_data.inc` + `ht_find` export from `microdb.inc`.
- Sets `mdb_readonly=1` before `FreeLibrary` to prevent DllMain DETACH from overwriting DB.
- `g_ht` layout constants (`MDB_HT_TABLE`, `MDB_HT_CAP`, `MDB_KP_BASE`, etc.) and `mdb_ht_read()` are defined in `tools/mdb_shared.h`.

### 21. save_db_all Debug Prints
`runq.c` `save_db_all()` uses `HT_ADD_CHECK(key, data, len, persist)` for all system keys, which
prints `[FATAL] ht_add('keyname') FAILED` if `ht_add` returns 0. The reverse-lookup loop counts
failures and reports a summary at the end; this catches silent hash-table-full failures.

### 22. BSS Global Tracking for Inference Cleanup
`v7s_inference_create` stores every VirtualAlloc'd pointer to its corresponding BSS global
(`g_inf_transformer`, `g_inf_qt_desc`, `g_inf_rs_block`, `g_inf_emb_block`, `g_inf_ctx_block`).
`inf_cleanup_alloc` reads these globals to free resources. `v7s_inference_destroy` has its own
`SF_START`/`SF_END` frame for Win64 ABI compliance.

### 23. Zero-File-Init: Everything from microdb (Complete)

**Hard requirement (achieved):** After `build_db.exe`, the init must load completely from `lz4ht.db`
with zero file access. `v7s_inference_create()` returns `InferenceContext*` containing Transformer +
Tokenizer + Sampler, all populated from microdb.

**Design (single VirtualAlloc block):**
```
[InferenceContext header: 3 ptrs (transformer, tokenizer, sampler)]
[Transformer struct— Config embedded (48) + Weights ptrs (120) + RunState ptrs (144)]
[QuantizedTensor descriptor block]
[RunState buffer block]
[Embedding dequant block]
[Tokenizer: metadata + merge_scores + vocab ptr array + flat strings + templates]
[Sampler: probindex buffer + metadata]
```
One `VirtualFree` cleans all.

**Flow:** `build_db.exe`— `save_db_all()` writes everything to microdb, then `db_save()` persists to
disk; the inference harness calls `dll.v7s_inference_create()` and skips
`build_transformer`/`build_tokenizer`/`build_sampler`— zero file I/O. Fallback: if
`v7s_inference_create()` returns NULL, falls back to file init (the existing code path).

### 23b. MicroDB is the ONLY non-stack allocator for all modules

**The microdb hash table key-value pool (`KP_BASE`) is the ONLY memory allocation mechanism available to
any module beyond the stack.** No module (LZ4, tokenizer, sampler, forward, attention, chat, generate,
inference, etc.) may call VirtualAlloc, HeapAlloc, malloc, or any other allocator directly. All
persistent or cross-call-lifetime memory must go through `ht_add` or `kp_alloc_grow`. The only
exceptions are:
- **`microdb_mem.inc` `mdb_alloc_mem` / `mdb_free_mem`**: the single plain committed-block allocator for every runtime buffer (inference state, embedding, descriptors, ViT scratch, SectorLogo pinned heap, encoder dynamic buffers).

**Rationale:** A single allocator means zero fragmentation, zero leak tracking, zero dangling-pointer
bugs from wrong deallocators, and zero decisions about which allocator to use. All pool memory is freed
in one shot by `ht_destroy`.

### 24. Stack Allocation Strategy (Primary)

- **Stack is our primary allocation pool.** All temporary buffers (LZ4 hash tables, workspace arrays, serialization scratch) go on the stack via `sub rsp, N`. This is faster, simpler, and avoids heap fragmentation. Reviews suggesting heap for "safety" are REJECTED. **Sec. 24a is the hard rule: never move stack data onto an allocator, and reject-on-sight any report that proposes it.**
- **Why stack by default:** (a) no alloc/free overhead, (b) automatic cleanup on return/crash, (c) no fragmentation, (d) cache-friendly locality, (e) simplest possible code path.
- **Stack budget:** Windows default thread stack is 1MB (reserve) with 4KB guard pages. Our call depth is LOW (< 10 frames). Even a 256KB `sub rsp` in a leaf function leaves >700KB for other frames. No `__chkstk` needed below 512KB.
- **Only HeapAlloc when:** (a) The buffer must outlive the allocating function, OR (b) the buffer exceeds 512KB AND cannot be split into smaller chunks, OR (c) the allocation count is dynamic/unbounded.

### 24a. THE STACK IS OUR PRIME DATA ALLOCATION SPACE— NEVER MOVE STACK DATA TO AN ALLOCATOR

**DO NOT MOVE STUFF FROM THE STACK TO ALLOCATORS. IT WORKS PERFECTLY.**

The stack is our prime data allocation space. It is the FIRST place data belongs, not a fallback, not a
compromise, not something to be migrated away from the moment someone notices a `sub rsp, N` in the
source. `sub rsp, N` is the correct, idiomatic, intended way this project allocates data.

**Hard rules:**

1. **NEVER move data that lives on the stack onto an allocator.** No
   `VirtualAlloc`, no `HeapAlloc`, no `malloc`, no `mdb_alloc_mem`, no
   `ht_add`/`kp_alloc_grow`, no `.bss` promotion, no "let me move this out of the
   frame" refactor. The direction stack— allocator is FORBIDDEN, always, with
   no exceptions and no carve-outs.
2. **IT WORKS PERFECTLY.** A stack buffer is not a compromise, not a risk, not a
   latent bug, and not a "safety" concern. It is correct, it is fast, it is
   cache-friendly, it is automatically reclaimed on return and on crash, and it
   fragments nothing. A stack allocation is never a defect to be fixed.
3. **REVIEWERS CAN FUCK OFF.** Any review, report, audit, `opt_and_bug\*.md`
   file, code comment, suggestion, or agent— ASM or C— that proposes moving
   data off the stack onto an allocator is WRONG and MUST BE REJECTED ON SIGHT.
   Do not "consider" it, do not "evaluate the trade-off", do not file it as an
   OPT, do not add it to a todo list, do not soften it into a question, and do
   not carry it forward as a "possible future improvement". The reaction is a
   flat rejection, recorded in one line, exactly as with the other
   reject-on-sight categories (Sec. 57b theoretical nits, Sec. 124 impossible-state
   checks, Sec. 129a danger-area optimizations).
4. **A `sub rsp, N` in ASM, or a fixed-size local array in C, is never a
   finding.** There is no stack size, no buffer lifetime, no alignment, no
   nesting depth, and no "this should be dynamic" circumstance that turns a
   stack allocation into something an allocator should own. If a report needs a
   `sub rsp` to be wrong, the report is wrong.
5. **Per Sec. 38: zero parallel paths.** The stack is not a fallback tier beneath
   "the real" allocator tier. There is one allocation space for temporary data
   and it is the stack.
6. **When applying a report, delete any item that asks for a stack— allocator
   move** and say so in one line of the revised report. Do not implement it, do
   not "try it and measure" it (Sec. 126's try-and-test doctrine does **not** apply
   to this class— there is nothing to test, it is simply forbidden), and do not
   leave it in the report for a future agent to pick up.

### 26. Database Model Weights— No mmap in ASM

- **`__sys_model_weights` is persisted to microdb** by `save_db_all()` from the mmap'd file data (past the 256-byte header), stored AS A BLOB in the hash table.
- **`v7s_inference_create` (ASM) does `ht_find("__sys_model_weights")`** to get a direct pointer into the hash table's memory. No mmap, no file I/O. The pointer is valid as long as the DLL is loaded.
- **Weight pointers inside TransformerWeights** are computed as `model_weights_base + weight_meta[i].file_offset`— these point directly into the microdb heap.
- **The C `read_checkpoint` path** still uses mmap for the fallback file-based init. The DB-init path skips mmap entirely.

### 26. Bug Report Rule

- **Every real bug from any bug report MUST be fixed.** If a bug report details a bug and the bug is confirmed as real (not an intentional trade-off documented elsewhere), it must be fixed— no exceptions. "Won't fix" is only acceptable if the bug is an explicit, documented design trade-off. When a bug is deemed "not a bug" (e.g., dead code, wrong analysis), explain clearly why. When it IS a real bug, fix it immediately— don't defer, don't triage to "later".

### 27. db_load Idempotency— Double-Load Guard (FIXED Jun 2026)
**`db_load` has an idempotency guard:** it checks `HT_COUNT > 0` before loading and skips if already
loaded; this prevents double-load corruption (`DllMain(ATTACH)` + `v7s_inference_create`). **Residual
(cosmetic):** `kp_base == table_ptr` is by design— the key pool is embedded after the hash table
entries in the linear serialization buffer.

### 28. Dispatch Performance Status (Resolved) + CPU% Diagnostic

**The dispatch performance regression is RESOLVED.** Q8 1024x1024 = 382-426 GB/s, F32 1024x1024 =
176-197 GB/s, all cores active. Current focus is the architectural cleanup described in Sec.0a (see
`dispatch_architecture_fix_plan.md`).

**Key diagnostic— CPU% tells you everything:**
- **100% CPU** = only 1 core working (the caller thread). Workers are sleeping, not started, or not receiving work.
- **1200% CPU** = all 12 logical cores active. This is healthy multi-threaded dispatch.
- **300-900% CPU** = partial parallelism (3-9 cores active). Some workers are participating but not all.
- **If a loadtest reports ~100% CPU, the dispatch is broken— workers are idle.**### 32. MicroDB— No Conditional Paths Based On Data Size

**`microdb.inc` MUST always compress its output.** There MUST be exactly ONE serialization path in
`db_save` and ONE deserialization path in `db_load`. Conditional paths that change behavior based on
data size are CRITICAL BUGS and must never exist.

**Rationale:** Parallel paths mean the codebase has two formats, two sets of edge cases, two failure
modes. The file format must be invariant— `lz4ht.db` is always LZ4-compressed, period.

**Fixes applied/enforced:**
1. `db_save` raw path— REMOVED. All data, regardless of size, goes through LZ4 compression.
2. `db_load` raw path (bit63-flag check)— REMOVED. Load always expects LZ4-compressed data.
3. `MDB_SKIP_COMPRESS` threshold— REMOVED. Compression is unconditional. The compressed-size return from `mdb_compress` is 64-bit (callers must not truncate to `eax`).
4. Chunked write (>4GB compressed output)— handled via existing chunked WriteFile code, same as any large I/O.
5. **No parallel alloc/free paths**— `mdb_free_mem` must handle ALL buffer types (both HeapAlloc and VirtualAlloc). Never two separate deallocation strategies.

### 33. MicroDB Compression Format— No External LZ4 Compatibility Required

**`mdb_compress`/`mdb_decompress` do NOT need to produce or consume standard LZ4 bitstreams.** The
`lz4ht.db` format is our own— there is no interop requirement with `liblz4` or any external tool.

**Hard constraints:**
1. **Must always compress**— no raw/uncompressed fallback path (per Sec.32).
2. **Must not sacrifice compression ratio.** Any replacement algorithm must match or exceed the current ratio.
3. **Must handle 64-bit block sizes.** The compressor must accept a 64-bit `src_len` and produce a 64-bit `compressed_size`.
4. **Self-contained in `microdb.inc`.** No external library dependencies.

### 34. Model Weights— MUST Be in MicroDB, No External Files

**Model weights MUST be stored inside microdb.** The `__sys_model_weights` entry is a single blob in the
hash table. There is NO external file fallback for model weights— not for the ASM path, not for the C
path. Period.

**If the model weights blob exceeds 4GB**, the fix is to enlarge the Entry struct (`E_DLEN`— qword,
`ENTRY_SZ`— 40) and use `VirtualAlloc` for the large data buffer inside the hash table (Sec.41). Adding
external file references is forbidden— the whole point of microdb is a single self-contained file.

### 36. MicroDB Minimalism— Remove Unneeded Code

**MicroDB must be kept minimal.** Every feature, flag, branch, and field must justify its existence.
When adding new functionality, always ask: "Is this needed? Can we do it with less?" If existing code no
longer serves a purpose, remove it.

**This applies to:** Flag bits in `E_FLAG`, stack frame fields, conditional branches that handle edge
cases that no longer exist, commented-out/dead code paths, and `%define` constants not actively used.

**Rationale:** Code that is not needed is not "harmless"— it is a maintenance burden, a source of
confusion, and a potential bug.

### 37. MicroDB— Single Growing Pool for Keys and Values

**One pool for both keys and values.** `KP_BASE`/`KP_USED`/`KP_CAP` track a single contiguous
VirtualAlloc'd region that holds ALL entry data (both key strings and value blobs). There is no separate
value pool.

**Design rules:**
- `E_KEY` and `E_DATA` are both dword offsets into the single pool
- On lookup: `pool_base + E_KEY` = key string, `pool_base + E_DATA` = value data
- On update (existing key): new key uses same slot (no change), new value appends to pool
- On tombstone: no pool reclamation (db_save writes only persistent entries, db_load rebuilds pool fresh)
- `E_DLEN` is qword; `HT_HEAP` removed (KP_BASE moved to +24; HT_SIZE=40)
- **Large models (>4GB weights blob):** The pool offset for `__sys_model_weights` must stay within 32-bit (dword). If the model weights blob alone exceeds 4GB, enlarge `E_DATA` to qword offset and increase `ENTRY_SZ` to 48 (Sec.41). The pool itself can handle any size— only the dword offset field is the bottleneck.

### 38. MicroDB— Flat Design, Zero Workarounds, Zero Parallel Paths

**The microdb design MUST be flat.** No conditional workarounds, no fallback paths, no format variants.
Every operation has exactly one way to happen.

**Hard rules:**
- One pool for keys + values (KP_BASE/KP_USED/KP_CAP). No separate value pool. No per-entry VirtualAlloc.
- **Only two allocations exist:** the hash table array (HT_TABLE) and the key+value pool (KP_BASE). db_save/db_load use temporary one-off VirtualAlloc'd buffers for serialization, freed immediately after use.
- One serialization format. Always compressed (LZ4-like). No raw/uncompressed fallback.
- One allocation strategy: VirtualAlloc for everything. No HeapAlloc anywhere. No alloc_type branching.
- One bitwise flag operation. No conditional set/clear branches— always AND-then-OR.
- One ht_add data flow. New entries and updates share the same allocation+copy+free code.
- Zero parallel code paths. Any if/else in microdb.inc is a design smell that must be eliminated.

### 39. MicroDB— Split Into Multiple Files (PLANNED)

**`microdb.inc` is too large for a single `.inc` file.** It should be split into focused sub-files
(proposed mapping):

| File | Contents |
|------|----------|
| `mdb_core.inc` | `g_ht` layout constants, entry struct `%define`s, `HT_SIZE`, entry field offsets |
| `mdb_pool.inc` | `KP_ALLOC_INLINE`, `kp_alloc_grow`— key+value pool growth and allocation |
| `mdb_ht.inc` | `ht_init`, `ht_destroy`, `ht_add`, `ht_find`, `ht_close` |
| `mdb_compress.inc` | `mdb_compress`, `mdb_decompress` |
| `mdb_persist.inc` | `db_save`, `db_load`, `db_set_readonly`, serialization format constants |
| `mdb_util.inc` | `mdb_alloc_mem_nz`, `mdb_free_mem`, error tracking |

**Rules:** Each file self-contained with its own header and `global`/`extern` declarations;
`mdb_core.inc` constants used by all others and included first; no circular dependencies; all comment
validations apply per file. **Status: NOT YET DONE** (no `mdb_*.inc` files exist yet— this is planned
refactoring).

### 40. MicroDB Compression— Pragmatic, NOT LZ4-Spec-Compliant

**The `mdb_compress`/`mdb_decompress` functions do NOT need to produce or consume standard LZ4
bitstreams.** Trying to stay LZ4-spec-compliant is a trap that wastes time and introduces bugs.

**What matters:**
- **It MUST work.** Compress + decompress round-trip must produce identical output for ANY input.
- **It MUST always compress.** No raw/uncompressed fallback path (Sec.32).
- **It MUST handle 64-bit sizes.** `mdb_compress` accepts 64-bit `src_len`, produces 64-bit `compressed_size`; `mdb_decompress` accepts 64-bit `src_len` and `max_dst`.
- **It MUST be self-contained.** No external library dependencies.

**Known LZ4 deviations (documented— WON'T FIX):**
1. **Stream validation (`mdb_decompress`), owner directive (Oct 2026):** the decompressor ENFORCES the caller's `max_dst` and validates every stream element before it is used: literal runs and match copies must end inside `[dst, dst + max_dst)`, tokens, extension bytes and the offset field must lie inside `[src, src + src_len)`, and the match offset must be >= 1 and <= bytes written so far (offset == written reads the first output byte and is legal). Any violation is a named FATAL (`mdb_decompress: corrupt LZ77 stream...`). This validation is init-time only (`db_load`, `load_kv_cache`)— never a hot-path cost.
2. **No literal length / match length limit:** LZ4-spec limits extensions to 255 bytes per extra byte. Our implementation handles arbitrary-length extensions via the shared `.read_ext` routine with overflow detection.
3. **Compressed size header:** Our format stores only `decompressed_size (qword) + compressed_size (qword)` before the compressed data. No magic number, no frame descriptors.
4. **Hash table:** Our compressor uses a 32K-slot hash table (`LZ4_HT_SLOTS=32768`). Affects ratio, not correctness.
5. **No block dependency/independence distinction:** Always block-dependent (continuation hash table across the entire input).

**If the compression code has a bug:** Fix it so the round-trip works. Do NOT try to make it match some
external LZ4 test vector. The only correctness requirement is: `compress(src)`— `decompress(comp)` ==
`src` for all possible `src`.

### 41. MicroDB— Proper 64-bit Pool/Entry Layout (REQUIRED for =4B models)

**The single-pool offsets MUST be 64-bit.** The dword offsets (`E_KEY`, `E_DATA`, `KP_USED`, `KP_CAP`)
silently truncate at 4GB. This is broken for any model whose weights blob pushes the pool past 4GB
total.

**The fix MUST be done properly— no workarounds, no shortcuts:**
1. `E_KEY` and `E_DATA` enlarged from dword to qword— pool offsets are 64-bit.
2. `KP_USED` and `KP_CAP` enlarged from dword to qword— pool size is 64-bit.
3. Every read/write of these fields uses 64-bit registers and qword moves.
4. No "if data_len < 4GB use dword, else use qword" branching— always qword (Sec.38).
5. No "split the model weights blob into multiple entries"— a single blob must be addressable as a single entry.

**New layout (`v7_core.inc`):**
```
g_ht struct (HT_SIZE = 48):
  HT_TABLE  0  qword  Entry[] pointer
  HT_CAP    8  dword  capacity (power of two)
  HT_MASK  12  dword  capacity - 1
  HT_COUNT 16  dword  entry count
  ; +20..+23 padding
  KP_BASE  24  qword  key+value pool base
  KP_USED  32  qword  bytes used in pool (was dword)
  KP_CAP   40  qword  pool capacity (was dword)

Entry struct (ENTRY_SZ = 40, unchanged):
  E_HASH   0  dword  hash (0=empty, 0xFFFFFFFF=tombstone)
  ; +4..+7 padding
  E_DLEN   8  qword  data byte count (already 64-bit)
  E_KLEN  16  dword  padded key byte count
  E_FLAG  20  dword  flag bits
  E_KEY   24  qword  pool offset of key string (was dword)
  E_DATA  32  qword  pool offset of value data (was dword)
```

**Affected sites (must all be 64-bit clean):** `KP_ALLOC_INLINE` macro, `ht_add`, `ht_find`,
`db_save`/`db_load` serialization, `mdb_query`/`mdb_list` tools, `inference.inc` (`pool_base + offset`
arithmetic).

**Validation:** 0.6B model must continue to round-trip; 4B/7B models must build, load, and produce
correct inference output. This is a one-time breaking format change— regenerate all `lz4ht.db` files
after applying.

### 42. 7B Model— Deferred Until 4B Is Fully Green
**Once the 4B model produces correct output under ALL conditions** (all validators pass, both `-q8 0`
and `-q8 1` match, all harnesses correct), the next target is the 7B model (Qwen3-7B/8B). The 64-bit
pool/entry layout (Sec.41) is the prerequisite. No parallel work on 7B until 4B is done.

### 45. 64-Bit Model Support— All Size/Offset Math Must Be 64-Bit

**All arithmetic that computes sizes, strides, or byte counts MUST use 64-bit registers and
instructions.** 32-bit arithmetic silently truncates values above ~2.1B. While current Qwen models fit
within 32-bit bounds, fixing these proactively avoids corruption when larger models are loaded.

**Policy:**
1. Any `imul`, `mul`, `shl`, `add`, `div` that computes a size or offset must operate on 64-bit registers.
2. Any `lea` that computes a byte count must use 64-bit destination.
3. `div r/m32` should become `div r64` when the dividend may exceed 32 bits.
4. Any `rep movs*` counter (`rcx`) must be set with 64-bit instructions.
5. Mixed-width register usage is acceptable ONLY when the value provably fits in 32 bits AND a zero-extension is guaranteed. When encountered during edits, convert to consistent 64-bit.

**No bug is cosmetic.** Every real bug (including register-width mismatches, mixed-width arithmetic,
fragile counter setup) MUST be fixed when discovered. Init-time code has no hot-path constraints— fix
everything there. Do NOT defer fixes.

### 47. dbg_inference.exe— LoadLibraryA in WinMain for Equal Conditions

**`dbg_inference.exe` (pure ASM harness) MUST load `v7_threading.dll` via
`LoadLibraryA`/`GetProcAddress`** just like `inference_example.exe` does. This ensures the DLL's
`DllMain` runs with the same loader state, `db_load` is called from `DllMain(ATTACH)` at the same point,
and all function pointers resolve identically.

**Current status (already correct):** `dbg_inference.asm` loads the DLL via `LoadLibraryA` and resolves
`v7s_inference_create`, `v7s_inference_destroy`, `v7s_chat`, and `encode` via `GetProcAddress`. No
static linking, no implicit import. **If this ever changes or a new harness is created**, it MUST follow
the same pattern.

### 48. Stale Database Procedure

**Stale `lz4ht.db` is the #1 cause of false-positive ASM init bugs.** Whenever debugging ASM init
failures, ALWAYS recreate the database before investigating:
```
del lz4ht.db
.\build_db.exe weights\qwen3_0.6b.bin -f
```

**When to recreate:** (1) after any change to `save_db_all()` in `runq.c`; (2) after any change to
microdb serialization format; (3) when switching model checkpoints; (4) when `-compare-init 1` reports
data checksum mismatches; (5) at the start of any new debugging session focused on ASM init.

**Signs of a stale DB:** model weights checksum differs between C file and microdb, tokenizer fields
show garbage, vocab entries truncated, config fields disagree with the checkpoint, ASM init produces
wrong output even though comparison matches.

### 51. Git Commits— Never Use `git add -A`; ALL Uncommitted Project Files Are Our Responsibility— NEVER Leave Project Files Behind

**Never use `git add -A` or `git add .` before committing.** These commands stage ALL changes
(including unrelated build artifacts, logs, temp files, and `.bat` outputs). Instead: stage only the
files you changed (`git add <specific_file1> <specific_file2>`), then commit; this ensures every
commit is minimal, focused, and reviewable.

**ALL uncommitted project files are ALWAYS the responsibility of the committing agent— in every session,
without exception.** When preparing ANY commit:

1. **Check `git status` for ALL modified tracked project files (`.inc`, `.asm`, `.c`, `.cpp`, `.h`, `.ini`, `.md` docs, `src/`, `tools/`, `examples/`)— not just the files from the current task.** Any modified tracked project file in the working tree is OUR work— from the current session or a prior session, from our report or someone else's— and it MUST be accounted for.
2. **NEVER commit just "our own" changes while leaving other modified project files uncommitted.** Every commit must be complete for the state of the project: all modified tracked project files that are part of applied/validated work go in (or are explicitly reverted if abandoned). A commit that leaves behind modified project source files is INCOMPLETE— it fragments the work state and hides applied-but-unrecorded changes.
3. **No "I'll commit those later."** If modified project files exist at commit time and they contain applied, verified work, they are committed— in the same commit or immediately adjacent commits in the same session. They do not wait for a "better moment."
4. **The one iron rule of staging:** list the modified tracked project files explicitly (`git status --short`, read it, then `git add <file1> <file2> ...`)— never `-A`, never `.`, and never silently skipping files that appear in the list.

### 52. Dispatch Rewrite— Pre-Rewrite Backup
**Baseline commit `41667b4`** was the last working commit before the dispatch rewrite. A full
pre-rewrite backup of all dispatch-related files exists at **`attic/pre_rewrite_backup/`**. When a
dispatch change breaks something, diff against these backups to find what changed.

### 53. TASK_STATUS Markers— Auto-Resume Plugin

**At the end of EVERY text response, include exactly one marker on its own line:**
```
TASK_STATUS: COMPLETE
TASK_STATUS: BLOCKED
TASK_STATUS: INCOMPLETE
```
- `COMPLETE`— task fully done and verified
- `BLOCKED`— hard blocker or unrecoverable error
- `INCOMPLETE`— platform-forced only when session terminates mid-task; never choose deliberately

**Rules:** The marker must be the last line of the response; no additional text after the marker; only
ONE marker per response.

### 54. Dispatch Regression Testing— Speed Regressions Are Critical Bugs

**Any speed regression >20% below the all-time best for a given kernel shape is a CRITICAL BUG that must
be fixed immediately.** Speed regressions are not cosmetic— they mean the dispatch system is not
utilizing workers correctly.

**Regression testing workflow:** The regression benchmark is the standalone `bench_regression.bat` (NOT
a verify.bat phase, adding it there would roughly double the verify runtime; run it by hand when
dispatch/kernel code changed). It takes an optional db_path (default
`weights\databases\qwen3_0.6b\lz4ht.db`) and gates every loadtest on its exit code. Compare the printed
GB/s numbers against the recorded all-time bests quoted in Sec.0/Sec.28 (the old `bench_regression.py` /
`build_logs/bench_history.json` comparison tooling was removed in the cleanup, the comparison is manual
now). Any kernel >20% below its historical best is flagged as **REGRESSION**.

**When a regression is detected:**
1. Check the git commit that introduced it (`git log` to find what changed)
2. Diff the dispatch files against the last-known-good commit
3. Fix the root cause— do NOT add workarounds or increase iteration counts
4. Re-run the benchmark to confirm the fix

**Laptop power/throttling variation:** This codebase runs on a laptop (Ryzen 5 7430U). Benchmarks show
10-25% variation between runs due to power throttling, thermal limits, AC vs battery power, and
background processes. **Mitigation: re-run the benchmark 2-3 times.** If the same kernel consistently
shows >20% below its best across multiple runs at the same commit, THEN investigate. If it appears in
only one of several runs, it's noise— ignore it.

**Never accept a speed regression as "close enough."** The all-time best is the floor, not the ceiling.

### 57. USER-PROVIDED REPORTS— EXAMINE AND ADDRESS EVERY BUG WITH HIGHEST PRIORITY

**When the user provides a detailed bug report or analysis report, EVERY bug listed in it must be
examined and addressed with highest priority.** The report is the user's deep analysis— treat it as
authoritative and act on it immediately. For each bug:
1. Read the relevant code to verify the bug is real
2. Fix the root cause (not a workaround)
3. Address it in the current session, not "later"
4. If the fix is risky or complex, explain the approach to the user before implementing

No bug from a user report may be skipped without explicit user approval. **WORKING ON PROVIDED REPORTS
IS UNCONDITIONAL AND NON-NEGOTIABLE.** When the user provides a report (bug analysis, optimization
audit, or any other detailed document), the work on it begins immediately— there is no continuation of
previous goals. The report's contents define the new current task. Strengthening AGENTS.md rules to
prevent recurrence is part of the work.

### 57a. NO CAST NITPICKING— Readable Code Over Pedantic Casts

**Do NOT add casts for the sake of satisfying pedantic type-purity rules unless the code is proven to be
actually broken on the target compiler (MSVC).** Casts like `static_cast<unsigned char>(c)` on every
`std::isalpha` call are counterproductive— they bloat code, reduce readability, and waste time.

**Hard rules:**
- If the bug report says "undefined behavior on non-ASCII characters" involving `<cctype>` functions: verify the compiler and platform (MSVC on x64). On MSVC, `std::isalpha` with signed char is benign. Add the cast only if the same file already has it, and keep it minimal.
- If a cast is needed, prefer C-style `(unsigned char)` over `static_cast<unsigned char>`.
- If adding a cast would change 10+ lines in a file, reject the fix entirely and document "WON'T FIX— no real crash on MSVC".
- Bug reports that spend paragraphs on signed/unsigned char `isalpha` UB without evidence of actual crashes are NOT real bugs and must be rejected outright.
- **Prefer C-style casts over C++ named casts** (`(int)x` not `static_cast<int>(x)`).

**This applies only to `<cctype>` signed char UB and similar theoretical type-purity bugs.** Real bugs
(buffer overflows, null dereferences, integer overflow, command injection, memory corruption) are NEVER
exempt.

### 57b. THEORETICAL SECURITY NITS FROM REPORTS— REJECT ON SIGHT

**Do NOT add bounds checks, overflow guards, sanitization, or any defensive code that addresses purely
theoretical scenarios that cannot manifest on our target platform (MSVC on x64 Windows).**

**Hard rules:**
- "What if someone passes a negative value here?"— On MSVC with our callers, this cannot happen. WON'T FIX.
- "What if the buffer is exactly 1 byte too small?"— If `snprintf` truncation cannot actually cause harm given the data sizes involved, WON'T FIX.
- "What if memory allocation fails and returns null?"— On a 12GB-RAM development machine with small allocations, null-from-malloc is impossible in practice. Do NOT add a check that doesn't exist in surrounding code already.
- "What if an integer overflows here?"— If the values are bounded by config/data limits far below overflow range, WON'T FIX.
- "What about TOCTOU, race conditions, or path traversal in local file operations?"— Single-user local tool, not a network service. WON'T FIX.
- **The only acceptable theoretical fixes are:** (a) buffer overflows that can actually be triggered by real model output or user input, (b) null pointer dereferences in code paths that are actually reachable, (c) integer overflows that can actually wrap given real data sizes, (d) command injection via user-provided strings that pass through to `system()` or `CreateProcess`.
- **Readability and compactness are NEVER sacrificed for theoretical safety.**

### 57c. DEFENSIVE CODE ONLY AT THE OUTERMOST LAYER

**Bounds checks, input validation, and defensive guards are only allowed at the outermost API boundary
of a module.** Internal functions— those called only by other internal functions— must NOT contain
defensive code; they trust their callers.

**Hard rules:**
- The public entry points of a module may validate their inputs at the top and bail cleanly.
- Every other function in the call chain assumes valid inputs. No null checks, no bounds checks, no `if (n < 0)` guards in helpers.
- "Defense in depth" is explicitly forbidden. If the outermost layer validated, inner layers do not re-validate.
- Exception: functions that process external strings (tokenizer decode, page text) may check for null terminator or valid UTF-8, because the data crosses a trust boundary at that point.

**Rationale:** Every defensive check in an internal function is dead code that will never fire. If a
caller passes garbage, fix the caller— not every callee in the chain.

### 57d. TWO VALID REACTIONS TO INVALID CONDITIONS— IGNORE OR FATAL

**When encountering a condition that should not happen, there are exactly TWO valid reactions:**
1. **IGNORE (preferred)**— Remove the check entirely. Trust the contract. Let bugs crash naturally— Zydis catches them. Fewer instructions = fewer bugs = faster code.
2. **FATAL (if we need to catch bugs)**— Add a `test`/`cmp` + `jz`— FATAL. Only when we are actively debugging a class of bugs and need the diagnostic. Not on hot paths.

**Everything else is FORBIDDEN:** silent defaults, fallback paths, early returns on invalid input, value
correction (`cmovg`/`cmovl` clamping, `max(x, 1)` flooring), tail paths for impossible conditions.

**Hard rules:**
- Never add a check (FATAL or otherwise) inside a kernel, dispatch hot loop, worker steal loop, per-token generation loop, per-layer loop, or any code path called O(n) times per forward pass.
- Init code (one-time setup, configuration loading, worker pool creation) is fair game for FATAL checks.
- The correct design is: **inputs must be valid by contract.** The caller is responsible for providing valid inputs. The callee trusts its callers. A FATAL check in hot-path code is a bug: it means the contract was violated at the caller level, and the fix belongs at the caller.
- **Every check for an "impossible" condition on the hot path must be removed.** `n_kv_heads == 0` checks, null pointer checks on freshly-allocated buffers, `byte_count == 0` early returns— all must go.
- Exception: a FATAL check that catches hardware-level corruption (e.g. NaN detection) may be justified IF it costs zero extra instructions.

**Clarifications (user directives, binding):**
- **"IGNORE" means do EXACTLY NOTHING for handling.** No print, no diagnostic, no return code, no cleanup— the check and its entire handling path are deleted. Doing anything other than nothing is a third reaction and is forbidden.
- **We do NOT add FATALs for paths we do not add handling.** We write performance software and our code does NOT call routines with bad parameters. Error checking on parameters is appropriate only when we are actively bug-hunting a class of bugs— otherwise it is DEAD CODE that eats cycles through senseless checks.
- **Inducing error checks must be WARRANTED for searching a bug.** A check may only be introduced when we are actively hunting a concrete bug class; unwarranted checks are dead code and are removed.
- **When a FATAL or error-checking path is removed because it is dead, we NEVER replace it.** HAVING UNDEFINED BEHAVIOUR FOR UNDEFINED INPUTS IS INTENDED— we do not pay the price of cluttering our performance software with checks for inputs that are outside the contract.

### 58. SCRIPT-GENERATED COMMENTS STRICTLY FORBIDDEN

**It is strictly forbidden to use scripts (Python, sed, awk, etc.) to auto-generate or auto-fill
comments in any file— INI, ASM, or otherwise.** All comments must be written by hand, individually, with
semantic meaning specific to each line. If you use a script to generate comments, the user WILL detect
it, strip ALL auto-generated comments, and then you will have to rewrite ALL comments by hand over the
entire project. The script approach NEVER saves work— it multiplies it. Do not attempt it. Ever.

### 59. SEH PERMANENCE— Zydis VEH is the ONLY Crash Handler

**`__try`/`__except`/`__finally` (SEH) is FORBIDDEN project-wide.** No exceptions. No fallback paths. No
silent error swallowing. Zydis VEH (`AddVectoredExceptionHandler`) is the ONLY allowed crash handler.

**Hard rules:**
- Zero `__try`/`__except`/`__finally` blocks anywhere in the codebase.
- `safe_copy()` must be a void function (no return value) using bare `memcpy()`. If a bad address is passed, it crashes— Zydis catches it and disassembles the fault site.
- No SEH-based fallback paths. If code could crash, let it crash— Zydis provides crash-site disassembly.
- VEH-based debugging features (guard-page stepping, single-stepping, register snapshot, MAGIC_SET/CHECK, INT3 magic) are preserved— they are part of the Zydis/VEH infrastructure, NOT SEH.
- Any new source file that needs crash handling must include `zydis_debug_header.h`— NOT add `__try`/`__except`.

**Files that must NEVER have SEH:** `zydis_debug_header.h`, `tools/vea_handler.h`, `runq.c`,
`runq_inference_test.c`, `ref_kernel/matmul_refactor_v7.cpp`.

### 60. C Code Must NEVER Compensate for ASM Bugs— That Is a Bug in Itself

**C code (fallbacks, SEH wrappers, null-pointer checks, defensive returns) that masks, works around, or
catches crashes from ASM code is itself a bug.** Every C-level compensation hides a real ASM bug that
would otherwise be caught and fixed.

**Hard rules:**
- No SEH (`__try/__except`) around any ASM function call.
- No `if (dll.fn) { dll.fn(); } else { /* C fallback */ }` patterns for ASM exports— if the ASM function doesn't exist, it's a build/link error, not a runtime fallback.
- No `-fallbackforward`, `-fallbackprefill`, or any CLI flag whose sole purpose is to bypass ASM and run C instead. If the ASM path has a bug, **fix the ASM path**— do not add a C bypass.
- No C code that catches, swallows, or recovers from crashes in ASM code. The Zydis VEH handler is the only crash handler, and its purpose is **diagnostic** (print crash dump) not **recovery** (fallback to C).
- The only acceptable C fallback is for features that legitimately do not exist in ASM (e.g., a new kernel not yet ported). This must be temporary and tracked for removal.

### 61. `rsi`/`rdi` Clobbered by String Instructions— Latent Clobber Bug (FIXED, Lesson Stands)

**Status: RESOLVED (Jun 2026).** `chat.inc:v7s_chat` terminated early after ~1 token ~50% of the time
because `rsi` was used to cache `seq_len` while `rep movsb` (token decode copy) clobbered `rsi`.

**Lesson:** **Any loop that caches a value in `rsi`/`esi` and also contains a `rep movsb` (or any string
instruction that modifies `rsi`) MUST reload the cached value after the copy.** `rsi` is both a
non-volatile callee-saved register AND the implicit source pointer for `lods`/`movs`/`scas`/`cmps`. The
same applies to `rdi`/`edi` (implicit destination for `stos`/`movs`/`scas`/`cmps`).

**Enforcement:** Any code review of a function containing `rep movsb`, `rep movsq`, `rep stos*`, `rep
lods*`, `rep scas*`, or `rep cmps*` must verify that `rsi` and `rdi` are either (a) not used to cache
state across the string instruction, or (b) reloaded from a non-volatile source (stack slot, BSS global,
or non-volatile register) after each string instruction. This is a hard requirement— no exceptions.

### 62. Save Lines of Code— Macro Reuse, Shorter Sequences, Clever x64

**Every change must aim to reduce total lines of code**, not just optimize for speed. Fewer lines means
fewer bugs, less maintenance, and faster understanding.

**Strategy A— Macro-driven deduplication:** Any repeated instruction pattern that appears 3+ times in
the codebase must become a macro. Do not hesitate— macros are the fundamental ASM code-reuse mechanism
and are strongly preferred over copy-paste.

**Strategy B— Shorter instruction sequences:** Prefer the instruction sequence that does the same work
in fewer bytes or fewer ?ops. Examples:
- `xor eax, eax` (2 bytes) instead of `mov eax, 0` (5 bytes)
- `add rsp, 8*N` instead of multiple `pop` instructions
- `lea rax, [rbx + rcx*4 + constant]` instead of `imul` + `add` + `add`
- `movzx ecx, al` (3 bytes, zero-latency) instead of `and eax, 0xFF` (5 bytes)
- `neg rax` instead of `xor eax, -1` + `inc rax`
- `cdq` / `cqo` for sign-extension instead of `mov edx, eax` + `sar edx, 31`
- `sub reg, 1` + `jnc`/`jns` instead of `cmp` + `jl` when the overflow/carry flag tells you what you need
- `xchg` (3 bytes) instead of `mov` + `mov` + scratch reg for register swaps

Always ask: "Is there a shorter way to write this that a human can still read?" If yes, use it.

**Strategy C— Clever x64 idioms:**
- **Zero-idiom `mov`:** `mov r32, r/m32` zero-extends to the upper 32 bits for free.
- **LEA for arithmetic:** `lea rax, [rcx + rdx*4 + 16]` does `rax = rcx + rdx*4 + 16` in one instruction.
- **`rep movsb` for small copies:** On modern Intel/AMD, `rep movsb` with ERMSB has <40 cycles overhead and then runs at ~16 bytes/cycle.
- **`xor reg, reg` + `cmp` chain** instead of `mov reg, 0` + multiple `or`: zero a reg then `or` with each value— single `jz` at the end.
- **`cmov` for branchless min/max/clamp:** Replace `cmp` + `jge`/`jl` + `mov` with `cmp` + `cmovg`/`cmovl`.
- **`setcc` + `movzx` instead of `jc` + `mov`:** For boolean results from flags, `setz al` / `movzx eax, al` is 5 bytes branchless.
- **Implicit `rdi`/`rsi` clobber awareness:** When `rsi` is used as a cached value AND a `rep movs*` clobbers it, reload from the stack cache after the copy (one `mov`)— cheaper than avoiding `rsi` or using a separate register.

**Enforcement:** Before any commit, run `git diff --stat HEAD` and check the net line delta. If positive
(more added than removed), write a brief justification in the commit message. The default expectation is
net-negative or neutral for refactoring changes.

### 63. DllMain(DETACH) Must NEVER Call db_save— MicroDB Saves Are Explicit Only

**`DllMain(DETACH)` must never call `db_save`.** The only two places where the microdb database is persisted are:
1. **Explicit C code**— `save_db_all()` in `runq.c` calls `db_save()` after populating the DB.
2. **`ht_close()`**— called before process exit, handles final persistence.

**Rationale:** DllMain runs under the Windows loader lock. Calling `db_save` from DllMain(DETACH) races
with active worker threads and with `v7s_inference_destroy`'s `ht_destroy()`/`ht_init()` reset (which
can leave an empty table, causing db_save to **delete** the DB file). **Enforcement:** No ASM code path
may call `db_save` autonomously. All DB saves must come from explicit C code. If you see a `call
db_save` in any `.asm`/`.inc` file that is NOT inside runq.c's C?ASM bridge, it is a bug.

### 65. FATAL Fix Policy— Fix the Bug First, Never Remove a FATAL Before Fixing the Root Cause

**If a FATAL catches something, fix the underlying bug even if the FATAL "should not really exist" in an
ideal design.** Never remove a FATAL (or any check that catches bugs) before fixing the root cause that
triggers it. The sequence is ALWAYS:
1. **Fix the bug**— Identify why the FATAL fires and fix the root cause.
2. **Then evaluate the FATAL**— Only after the underlying bug is fixed, decide if the FATAL should stay as a safety net, be downgraded, or be removed for hot-path performance per Sec.57d.

**Hard rules:**
- No FATAL may be removed, commented out, or bypassed while the underlying bug it catches is still present.
- "This FATAL shouldn't exist because the condition should never happen in correct code" is NEVER a valid reason to remove it— the fact that it IS firing proves the condition DOES happen.
- Temporarily disabling a FATAL to "see if other things work" is forbidden.
- Performance is never a valid excuse to skip fixing a bug that a FATAL catches.

### 66. No "Preexisting"— Everything Is Our Responsibility

**There is never "preexisting"— only responsibility.** Every bug, crash, regression, FATAL, wrong
output, or quality issue we encounter during our work is OURS to fix. Dismissing something as
"preexisting" or "not caused by our change" is an excuse to defer a fix.

**Hard rules:**
1. **If we see it, we own it.** Any bug discovered during testing, review, or exploration must be fixed in the same session.
2. **No "preexisting" escape hatch.** The only question is: is the bug real? If yes, fix it now.
3. **The codebase is a single evolving artifact.** There is no "our code" vs "their code." If we touch a file, we are responsible for every bug in it.
4. **Regression in someone else's area is still a regression.** We fix it— we don't blame.
5. **The only exception** was a bug explicitly called out as an accepted known limitation in project documentation, all such limitations are resolved (Sec.104); there are no accepted failures left.
6. **Preexisting crashes are the HIGHEST priority.** A preexisting crash is a preexisting defect, and per Sec.0b, every defect must be fixed immediately.
7. **No ignoring problems of any kind.** Wrong output, FATALs, performance regressions, crashes, corrupted data, silent failures— none may be ignored, deferred, or worked around.

### 85. NO BACKWARD COMPATIBILITY— Ever

**We NEVER implement backward compatibility with old formats, old databases, or old file versions.**
When we change a format, the old format is dead. We do NOT add version detection, migration paths, or
any "if version == 2 do X, else do Y" branching.

**Hard rules:**
1. **No v2/v3 branching in loaders.** When we ship format v3, db_load only reads v3. Any v2 file is stale and must be regenerated by running `build_db.exe` again.
2. **No dual-path code.** Every `if (version == 2) { ... } else { ... }` branch is dead code that will never be exercised after the first re-save.
3. **The fix for old files is always the same:** delete the old file, re-run `build_db.exe`.
4. **Format version bumps are FORWARD-ONLY.** The version number exists solely to detect and reject corrupt/wrong files— not to support reading them.
5. **Stale DB files are a non-issue.** They are regeneratable from the original model checkpoint.

### 86. Release v1.0— Reference Rules (release dir: `C:\project\v7_release\`)

**The release is a copy of the dev tree with relative-path build scripts.** The golden rule: every
`.bat` and build path in the release directory must use **relative paths only**— no `C:\project\v7\`
hardcodes. A fresh clone must build and run with zero path configuration.

**Release rules (non-negotiable):**
1. **No hardcoded DB path in the DLL.** The DB path must NOT be hardcoded anywhere in `v7_data.inc`. The path MUST be passed via `V7_init(db_path)`.
2. **No bragging in documentation.** Factual, technical, restrained language only. Forbidden: "fastest", "revolutionary", "world-class", "state-of-the-art", any superlative.
3. **Every document must match source code.** Cross-check `docs/api.md` against `v7_lib.h`, `docs/exports.md` against `.def`, `docs/tools.md` against actual tool `--help`.
4. **All fixes sync back to dev.** Every bug found and fixed during release testing must be copied back to `C:\project\v7\`.
5. **All validators must pass. Zero accepted failures.** (The former accepted-failure list, 2 matmul embedding shapes, sampler non-greedy mismatches, swiglu known limits, is fully resolved: the sampler and swiglu references are bit-exact ASM ports and the matmul claim was stale, see Sec.104.)
6. **Zero crashes.** Every program must run to completion without access violations on all supported model databases.
7. **verify.bat must test everything.** All validators, both inference paths (`inference_example.exe` + `dbg_inference.exe`), both quantization modes, and verify inference output contains "paris".

### 87. v7d_memcpy— Simple Threshold: <64KB ERMS, >=64KB DISPATCH

**The 4th parameter (`min_slice_bytes`) of `v7d_memcpy` is PERMANENTLY REMOVED.** The function signature
is exactly 3 parameters: `(void* dst, const void* src, int byte_count)`. No exceptions.

**No tables, no calibration, no lookup.** The dispatch decision is a single `cmp eax, 65536`:
- Size < 64KB: single-threaded ERMS `rep movsb` via `v7s_memcpy_chunk`
- Size >= 64KB: dispatched across worker pool as num_workers slices

**Hard rules:**
1. **No 4th parameter in the typedef.** `v7d_memcpy_fn` takes exactly 3 args.
2. **No 4th register setup at call sites.** No `xor r9, r9`, no 4th arg in `API_CALL v7d_memcpy, ...`.
3. **No `calib_table`.** The `calib_table`, `CALIB_SIZES`, `CALIB_PATH_ERMS`, `CALIB_PATH_DISPATCH`, and `calib_min_slice` are all removed. The `v7s_get_calib_table_ptr` and `v7s_get_calib_min_slice_ptr` exports are removed.
4. **The 1024 divisor caps the slice COUNT.** All dispatched sizes use `shr eax, 10` to compute `byte_count / 1024` as a slice-count ceiling, which is then clamped to `num_workers`; the slice size itself is `(byte_count / num_slices) & ~31`.
5. **Tools must adapt.** `b_memcpy_loadtest` and `v_memcpy_validate` call `v7d_memcpy(dst, src, byte_count)` like everyone else.

**Rationale:** Below 64KB, ERMS is always faster because dispatch overhead dominates. Above 64KB, worker
parallelism wins. A fixed divisor with the slice size derived from the actual job count removes the
per-size tuning table that preceded it.

### 88. Dispatched Benchmarks MUST Use set_spin(1)— No Exceptions

**Every benchmark that tests a dispatched `v7d_*` function MUST call `v7s_set_spin(1)` before any
dispatched work, and MUST call the `v7d_*` dispatch entry point— never the internal `v7s_*_chunk` kernel
directly.** Two hard rules:

**Rule 1: Always set_spin(1).** Running a dispatched benchmark without `v7s_set_spin(1)` is a BUG— it
tests sleeping workers instead of spinning workers. Required pattern:
1. `LoadLibraryA("v7_threading.dll")`
2. Resolve `V7_init`, `v7d_*`, `v7s_set_spin` via `GetProcAddress`
3. `V7_init(NULL)`— initialize worker pool and calibration
4. **`v7s_set_spin(1)`**— commit workers to spin-wait mode BEFORE any dispatched call
5. Pin thread to core 0, set HIGH_PRIORITY_CLASS
6. Benchmark loop

**Rule 2: Always use the v7d_* dispatch entry point.** Calling internal kernel functions directly (e.g.
`v7s_memcpy_chunk`, `v7s_q8_chunk`, `v7s_f32_chunk`) is a BUG— it bypasses the dispatch system entirely.
The internal kernel function is NOT EVEN EXPORTED from the DLL.

**Rationale:** `set_spin(1)` tells workers to busy-wait instead of sleeping, avoiding microsecond-level
API latency per dispatch. Production code always runs with `set_spin(1)` during inference. Benchmarks
must match.

### 89. v7d_memcpy ERMS Path MUST Wake Sleeping Workers— Performance vs Correctness

**`v7d_memcpy` is the FIRST dispatched call in the inference chain.** When the ERMS path executes AND
`last_spin == 0` (workers are sleeping via WaitOnAddress), the ERMS path MUST still wake sleeping
workers via `WakeByAddressAll`— even though the API call costs microseconds and hurts memcpy throughput.
This is a CORRECTNESS requirement, not a performance optimization.

**The rule:** The wake must happen BEFORE the ERMS `call v7s_memcpy_chunk` so that workers begin
transitioning from sleep to spin-wait while the ERMS copy runs.

**Why `last_spin`, NOT `always_spin`:** `always_spin` is the intent signal set by `set_spin(1)`.
`last_spin` is what workers have ACTUALLY committed to. When `last_spin != 0`, workers are already
spinning— waking them is pure waste. When `last_spin == 0`, workers are sleeping— they MUST be woken
explicitly or they will not see the next dispatch.

**Required ERMS-path wake sequence (inside `.dm_call_erms`):**
1. Save args (rcx, rdx, r8) to stack— API_CALL clobbers all volatile registers
2. Check `cmp dword [last_spin], 0`— workers sleeping?
3. `jne .erms_skip_wake`— no: workers already spinning, skip the expensive API call
4. Check `cmp dword [num_workers], 0`— workers exist?
5. `je .erms_skip_wake`— no: skip wake
6. **`API_CALL [__imp_WakeByAddressAll], remaining_jobs`**— wake all sleeping workers
7. `.erms_skip_wake:`— reload saved args from stack
8. `call v7s_memcpy_chunk`— ERMS kernel (args restored)
9. `jmp .dm_done`

**The performance cost is ACCEPTED.** Paying the wake cost early (during the cheap memcpy) so it
overlaps with the ERMS copy is a net win vs paying it later (during the expensive matmul prologue) where
it serializes with job setup. The memcpy throughput benchmark will show lower numbers for small ERMS
copies— that is EXPECTED and CORRECT.

**No fast-path bypass.** There is no `MEMCPY_FAST_THRESHOLD` check. ALL sizes go through the table
lookup. **Always use `API_CALL` for the `WakeByAddressAll`**— never hand-roll `call [__imp_*]`
sequences.

**THE WAKE-BY-ADDRESS DESIGN IS INTENTIONAL AND MUST NEITHER BE REPORTED NOR CHANGED.** Every
`WakeByAddressAll` placement in the codebase— DISPATCH_EXECUTE's early-wake and publish-wake, the
v7d_memcpy ERMS-path wake, the v7d_quantize pre-dispatch wake, the `.qa_direct_call` wake, and every
other `last_spin`-gated wake— is deliberate, documented here, and CORRECT as written. These wakes exist
so sleeping workers transition to spin-wait while preceding work (ERMS copy, job fill, chunk math) runs
and the NEXT dispatch finds them already spinning. Do NOT report wake placements as bugs, duplications,
Sec.0a violations, or optimization targets in reviews/reports; do NOT "unify", remove, relocate, or gate
them differently. A report claiming any wake site is redundant or misplaced is a FALSE POSITIVE— reject
it on sight.

### 90. Show Output to the User— Never Redirect Silently

**All commands that produce useful output MUST display it to the user.** Never silently redirect
stdout/stderr to files unless there is a concrete need (e.g., piping to a post-processor, or output is
too large for the terminal). The user must be able to observe tool output in real time.

**Hard rules— output the user has to see is NEVER redirected:**
1. **Never redirect output the user has to see.** Any command, test, benchmark, build, or tool run whose output is part of the deliverable (inference output, validator results, benchmark numbers, crash dumps, diff output) MUST run with stdout/stderr visible to the user in the terminal. Redirecting it to a file and only showing a summary hides the result from the user— that is forbidden.
2. **Never silently discard output.** No `> nul`, `2>&1` into a void, pipe to `Select-String`/`grep` filters that strip the result, or any other construct that hides what the user needs to see. If a script or .bat inside the repo redirects to a file, the user must still be shown the full file content afterwards— the file is not the deliverable, the user seeing the output is.
3. **If a concrete need forces a redirect** (post-processing, huge output), the command MUST be followed by displaying the full redirected content to the user. A redirect is a temporary transport, never a destination.
4. **When the user asks to see the full output of a test/run, show the COMPLETE raw content**— every line, no truncation, no abridging, no "I showed it above." Read the file and reproduce it in full.
5. **NEVER redirect stdout or stderr of any tool, test, validator, inference run, or benchmark— EVER.** Running commands like `bin\tool.exe args > file.txt 2>&1` cripples the output: the user cannot see it in real time, buffered stderr ordering is lost, and crash dumps may be swallowed. ALWAYS run tools with stdout/stderr flowing directly to the terminal. The ONLY exception is when a tool's output must be post-processed by another tool in the same command line (e.g. `| python script.py`), and even then the raw output must be visible. If you need to capture a timing number from a tool, add timing output INSIDE the tool's C code— do NOT redirect the tool's output to a file to scrape it later.

### 91. Confirmed Bugs MUST Be Fixed Immediately— No Deferral

**When a bug report is provided and the bug is confirmed as real against the current codebase, it MUST
be fixed in the same session— no exceptions.** The sequence is: (1) read the relevant code to verify the
bug exists, (2) fix the root cause, (3) build and test, (4) commit. No bug that has been confirmed real
may be deferred to "later."

### 90c. NEVER Reverse-Engineer Binaries We Have the Source For

**We have the source code. When debugging ANY behavior, read and trace the SOURCE— never scan,
disassemble, or pattern-match the built DLL/EXE bytes.** Reverse-engineering our own artifacts wastes
time and proves nothing; the assembler is proven correct (Sec.9), so any behavior discrepancy is a
source-level issue, findable by reading.

**Hard rules:**
1. **Debug the source, not the binary.** No byte-pattern scans of `bin\*.dll`/`*.exe`, no disassembly, no "verify the assembled bytes" steps— ever.
2. **A source trace that looks correct but behaves wrong means you missed a code path**— keep reading until you find the real path (second parser, preprocessor, different entry, stale build), do NOT pivot to binary inspection.
3. **The build output is current only after a successful rebuild.** Before concluding "the source is right but it behaves wrong," confirm the binary was built from the exact source you read.

### 90b. PRINT in SectorLogo Scripts Is GOOD— Use It Where We Can

**`PRINT` in SectorLogo sidecar scripts is a GOOD thing, not a bad thing. We use it WHERE WE CAN.**
Visibility into what a script does beats silent operation every time— this is the same principle as
Sec.90 (show output to the user), applied to Logo scripts.

**Hard rules:**
1. **Sidecar scripts print their decisions.** When a sidecar queries an external tool (like ragg), prints the query it asked, the answer it got, and the new stream it is about to SET— three PRINTs, full transparency. Debugging a silent script is guesswork; debugging a printing script is reading.
2. **PRINT what you ask, what you got, what you will SET.** The proven pattern from `sectorlogo_rag.logo`: print the query sent to the external tool, print the tool's answer, print the final value before `SetStream` (or any state-changing primitive).
3. **Never strip PRINTs to "clean up" output.** A PR that removes diagnostic PRINTs from a sidecar script is a regression. PRINT output goes to stderr/stdout of the host and is the user's window into the hook.
4. **New sidecar scripts start with PRINTs from day one.** Writing the script silent first and "adding prints later when it breaks" is backwards— the prints ARE the development tool that finds the break in the first place.
5. **Exception: per-token hot loops.** A script body that runs on EVERY token (`PRINT :PIECE` every call) is acceptable during development but should be gated behind the specific condition being debugged once stable, so normal runs are not flooded.

### 92. num_workers == 0 Is a Fatal Bug, Not a Runtime Check

**`num_workers == 0` is a FATAL bug that must never be checked or handled at runtime.** The worker pool
is created exactly once, during `initialize_once` (called from `v7_init.inc`). If thread creation fails,
`v7_init` must `FATAL` immediately. After initialization, `num_workers > 0` is a permanent invariant.

**Hard rules:**
1. **Zero `num_workers` checks in hot paths.** No `cmp dword [num_workers], 0` / `jz` / `jnz` in any dispatch function, worker loop, memcpy kernel, matmul kernel, or any per-token/per-chunk/per-iteration code.
2. **The ONLY check is at worker creation.** `initialize_once` creates the worker thread pool. If thread creation fails, the code must `FATAL` immediately.
3. **Dispatch functions trust the invariant.** They do not check `num_workers`. If it is somehow 0 at dispatch time, the crash is the correct diagnostic.
4. **"What if a tool runs without init?"**— The tool is broken. Fix the tool. Do not add a `num_workers == 0` fallback in the dispatcher.
5. **`num_workers` may be read for arithmetic.** Using it in `div`, `cmp`, or slice-count computation is fine— the prohibition is against conditional branches that change behavior based on `num_workers == 0`.

### 95. Supported Models— Instruct Variants Only (Reference Table)

**All supported models use Instruct (chat-tuned) variants, not base models.** The `export.py` template
system automatically detects the model's chat_template (Jinja2, ChatML, or bare `%s` for base models).

**Fetch scripts (run from project root):**

| Script | HF Repo | Params | build_db flags |
|--------|---------|--------|----------------|
| `fetch_qwen3_0.6b.bat` | `Qwen/Qwen3-0.6B` | 0.6B | (defaults) |
| `fetch_qwen3_1.7b.bat` | `Qwen/Qwen3-1.7B` | 1.7B | (defaults) |
| `fetch_llama_3.2_1b.bat` | `unsloth/Llama-3.2-1B-Instruct` | 1B | `-thinking 0` (no-op*) |
| `fetch_smollm2_135m.bat` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | `-thinking 1` (no-op*) `-rope-base 100000` |
| `fetch_smollm2_360m.bat` | `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | `-thinking 1` (no-op*) `-rope-base 100000` |
| `fetch_tinyllama_1.1b.bat` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | `-thinking 0` (no-op*) `-rope-base 10000` |
| `fetch_phi3_mini.bat` | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | `-thinking 0` (no-op*) `-rope-base 10000` |
| `fetch_minicpm5_1b.bat` | `openbmb/MiniCPM5-1B` | 1B | `-thinking 0` (no-op*) `-rope-base 5000000` |

\* `-thinking` in build_db.exe is a NO-OP (it only prints a note): every DB stores the
base (non-thinking) template pair plus optional `@think` variants when the model's chat template has
them; whether a model thinks is selected at RUNTIME via `v7s_enable_thinking()` in the DLL (engine
default: ON, reset per context).

**Key per-model differences:**

| Model | rope_theta | rms_eps | has_qk_norm | rope_scaling | tie_embeddings | F32 | Q8 |
|-------|-----------|---------|-------------|--------------|----------------|-----|-----|
| Qwen3-0.6B | 1000000 | 1e-6 | Yes | none | Yes | PASS | PASS |
| Qwen3-1.7B | 1000000 | 1e-6 | Yes | none | Yes | PASS | PASS |
| Llama-3.2-1B-Instruct | 500000 | 1e-5 | No | llama3 (factor=32) | Yes | PASS | PASS |
| SmolLM2-135M-Instruct | 100000 | 1e-5 | No | none | Yes | PASS | PASS |
| SmolLM2-360M-Instruct | 100000 | 1e-5 | No | none | Yes | PASS | PASS* |
| TinyLlama-1.1B-Chat | 10000 | 1e-5 | No | none | No | PASS | PASS |
| Phi-3-mini-4k-instruct | 10000 | 1e-5 | No | none | No | PASS | PASS |
| MiniCPM5-1B | 5000000 | 1e-6 | No | none | No | PASS | PASS |

*SmolLM2-360M "capital of France": at Q8 weight precision the first-token tie ('The' 16.20 vs 'I' 15.80,
margin 0.396 < maxad-1.2 floor) flips to a coherent refusal— adjudicated Q8-floor, NOT a v7 bug (Aug
2026, `inference_results_analysis.md` Sec.4.2).

**NOT YET SUPPORTED:** Nanbeige4.1-3B (missing `model.norm.weight` in HF checkpoint), MiniCPM4-0.5B
(incompatible architecture).

**Database directory:** All `lz4ht.db` files live in `weights\databases\<model>\lz4ht.db`. DBs are
per-model artifacts, NOT committed to git— generated from `.bin` files by `build_db.exe` (`-o
weights\databases\<model>\lz4ht.db`). All `.bin` files in `weights\` are generated exclusively by
`export_all.bat` / `fetch_*.bat`.

### 96. No Magic Numbers or Version Fields— Database is Flat, Zero Versioning

**The microdb database has NO versioning, NO magic numbers, and NO format detection.** Any field that
previously held a magic number or version is now a reserved padding field, always written as 0. No code
may check, validate, or compare these fields.

**Fields eliminated:**
- Config offset 0 (was `magic_number`)— now `reserved0`, always 0
- Config offset 4 (was `version`)— now `reserved1`, always 0
- MemLayout offset 0 (was `version`)— now `reserved0`, always 0

**Hard rules:**
1. Config offsets 0 and 4 are reserved padding— always 0, and named CFG_RESERVED0 / CFG_RESERVED1 in src/asm/inf_structs.inc:14-15 (they carry no references; they exist so the offset sequence reads 0, 4, 8, ... with the two holes visible). No code may check or validate them.
2. MemLayout offset 0 is reserved padding— always 0.
3. No `cmp dword [rax + ...]` for magic/version, no `if (config->magic_number != ...)`— the concept does not exist.
4. No FATAL for "bad magic", "unsupported version", or "version mismatch"— these cannot happen.
5. `build_db.exe` and `export.py` write 0 to these fields.
6. Old databases with non-zero magic/version still load fine (the fields are ignored).
7. The C struct field names are `reserved0` and `reserved1`— never `magic_number` or `version`.

### 97. Always View Full Model Output (stdout+stderr)— No Filtering

**When analyzing model inference output from bash, ALWAYS capture both stdout and stderr together using
`2>&1`, NEVER filter with `Select-String "^out:"` or similar.** Model responses can contain backtick
code blocks, special characters, and formatting that appears truncated or empty when filtered.

**Hard rules:**
1. **Use `2>&1`**— `.\runq.exe ... 2>&1` captures all output to a file.
2. **NO `Select-String "^out:"`**— Filtering strips code blocks, chat templates, and diagnostic output.
3. **When comparing outputs:** Always read the raw unfiltered output file.
4. **Ideal capture pattern:**
   ```
   .\runq.exe weights\qwen3_0.6b.bin v7_threading.dll -i "..." -t 0 -n 50 -q8 1 > out.txt 2>&1
   ```
   Then read `out.txt` directly.

### 98. Llama Reference— `llama3.c-master` Is the AUTHORITATIVE Pure-C Reference

**`C:\project\v7\llama3.c-master` (jameswdelancey/llama3.c) is the authoritative pure-C implementation
of Llama 3.2 inference for this project.** Do NOT modify any file inside it— it is read-only reference
material. `ref_kernel/llama3.c` is an earlier clone (secondary); `llama3.c-master` takes precedence.

**Reference files:** `run.c` (complete inference CLI), `export.py`, `model.py`, `tokenizer.py`,
`tokenizer.bin`, `win.c`, `build_msvc.bat`.

**Key reference facts (verified):**
- Config = 7 ints: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len.
- RoPE is PLAIN base=500000— NO llama3 scaled-RoPE implemented in the reference. It CANNOT validate our rope_type=1 path; use karpathy/llm.c `train_llama3.py` `apply_scaling()` as the scaled-RoPE ground truth.
- forward(): rmsnorm(eps=1e-5); qkv matmul; RoPE; per-head attention `score /= sqrtf(head_size)`; softmax; GQA; SwiGLU; final rmsnorm; classifier.
- KV cache layout (current): independent key/value FP16 buffers, `(layer, slot, kv_dim)` with 2 bytes per element; `slot` is ring-mapped when streaming is enabled.
- BOS=128000, EOS=128001. `generate()` prepends BOS. `chat()` manually inserts the full llama3 template token IDs.

**Verified NOT the bug (Llama investigation):** ASM qk_norm gating branches on `CFG_HAS_QK_NORM`; KV
per-layer offset math is 64-bit clean; KV init reserves/commits correctly. Llama3.2 1B KV storage scales as `2 * n_layers * seq_len * n_kv_heads * head_dim * 2` bytes in the current FP16 runtime
per cache (8GB key+value) at seq_len=131072— correct, do not treat as suspicious.

### 100. Llama Testing— MUST Use llama3.c Reference, NOT runq

**`runq.exe` is NOT suitable for testing Llama models.** The v7 pipeline produces garbage for Llama 3.2
1B— the bug is in the ASM forward compute, not in init or the tokenizer. **The ONLY valid comparison for
Llama inference is `llama3.c-master\run.c`.**

**To validate Llama output:**
1. **Build the reference:** `cd llama3.c-master && build_msvc.bat`— produces `run.exe`
2. **Export the model:** `python export.py <output.bin> <hf_dir>`
3. **Run the reference:** `run.exe <model.bin> -t 0 -n 50 -i "What is the capital of France?"`
4. **Compare:** v7 output must match reference output token-for-token at temperature=0

**Why runq is flawed for Llama:** runq implements Qwen3, not Llama. Its RoPE/attention/prefill paths
assume Qwen3 semantics; its chat template does NOT match Llama's training template; it does NOT prepend
BOS 128000. Every runq result for Llama must be treated as flawed.

### 101. Llama 3.2 1B Debug— Forward Compute Is the Bug (Jul 2026)

**Status: The tokenizer was DEFINITIVELY ruled out as the Llama failure cause. The bug is in the v7
forward compute path.** A hardcoded-token experiment injected the EXACT reference token stream into
`chat.inc`; v7 still produced token soup. Generated token IDs are all valid (< 128256) but semantically
incoherent.

**Conclusion:** The bug is downstream of the tokenizer: RoPE tables, weight_meta/weight layout, matmul,
attention, KV cache, prefill, or a config field. Remaining comparison targets were (1) Llama3 scaled
RoPE (PRIMARY SUSPECT— does ASM `rope_build_internal_tables` read `__sys_rope_type`/`__sys_rope_factor`
and apply llama3 scaling?), (2) weight_meta layout vs actual .bin layout, (3) rms_eps, (4) attention
scaling, (5) KV layout, (6) prefill vs per-token forward. (Open question— not yet resolved.)

**THE QWEN3-DB-IN-CWD TRAP (critical):** `runq.exe` and any tool that calls `V7_init(NULL)` loads
`lz4ht.db` from the **current working directory**, not from the model path. `C:\project\v7\bin\lz4ht.db`
is the QWEN3 DB (631MB). Running runq from `bin\` silently tests Qwen3 data even when a Llama checkpoint
path is passed. Always run Llama tests from `weights\databases\llama_3.2_1b\` or pass `--db` explicitly.
(This is also why Sec.106 exists.)

**ASM tooling lessons:**
- `WRITE_STR` requires a **data label**, NOT a string literal. Use `TRACE_STR "text"` or define a label + `db`.
- `TRACE_STR`/`TRACE_HEX`/`TRACE_REG`/`TRACE_NL` compile to nothing unless `-DTRACE_ENABLE` is passed to nasm.

### 102. PREEXISTING BUGS ARE FIXED IMMEDIATELY— VERIFY FAILURES ARE NEVER IGNORED

**Every preexisting bug discovered during verify.bat or any other test MUST be fixed in the same
session— no exceptions, no deferrals, no "that's not from my change."** If verify.bat shows FAIL for any
validator, that FAIL is a bug that must be fixed NOW.

**Hard rules:**
1. **A FAIL in verify.bat is ALWAYS a blocker.** It does not matter whether the FAIL is "preexisting" or "new."
2. **"Preexisting" is NEVER a valid reason to skip a fix.**
3. **No partial commits with known FAILs.** Every commit must have a clean verify.bat.
4. **This applies to ALL failures**— validators, benchmarks, inference tests, comment checker, anything that verify.bat runs.
5. **If fixing a preexisting bug is complex**, it still gets fixed. No "I'll come back to it."

### 103. NEW VALIDATOR FAILURES MUST BE INVESTIGATED IN DETAIL— NEVER WAVE OFF AS "PRECISION"

**When a validator fails, INVESTIGATE THE ROOT CAUSE. Never dismiss a failure as "Q8 precision limits"
or "tolerance too tight" without reading the actual error values, understanding the math, and verifying
the ASM against the C reference element-by-element.** Relaxing a tolerance without investigation is
hiding a bug, not fixing it.

**Hard rules:**
1. **Read the error output.** What are the actual values? Is the diff small (rounding) or large (wrong computation)?
2. **Check the tolerance.** Is the diff barely exceeding the tolerance (precision) or orders of magnitude beyond it (bug)?
3. **Compare element-by-element.** If diff=20 at idx=0 with ref=0, asm=0— that's a comparison bug. READ THE COMPARISON CODE.
4. **Never relax tolerances.** If the tolerance needs changing, you must explain exactly why with numbers.
5. **New failures in previously-passing tests are ALWAYS regressions.** Find what changed— do not adjust the test.
6. **SwiGLU Q8xQ8 is the one documented accepted failure.** Its C reference is a TRUE independent
   double-precision reference (owner directive): the earlier bit-exact port proved only that the
   kernel matches itself, so it was replaced. The validator therefore reports real float-vs-double
   drift on shapes with large near-cancelling group sums (AGENTS.md Sec.104 carries the shape table),
   and verify.bat counts the section as FAIL/expected.

### 104. Known Validator Precision Limits (current)

| Validator | Section | Status |
|-----------|---------|--------|
| `v_swiglu_validate` Q8xQ8 (large shapes) | float-vs-double reference drift on near-cancelling group sums | ACCEPTED LIMIT — do not relax; the independent reference measures the kernel's real precision boundary |
| `v_matmul_validate` Q8 | claimed 2 embedding-layer failures at n=151936 | Claim stale: the validator carries no n=151936 shapes and always exited 0 (Oct 2026) |
| `v_sampler_validate` non-greedy | "stochastic mismatches" vs approximate reference | Reference rewritten as a bit-exact ASM port (poly exp FMA chain, 4x8-lane sum order, heap nucleus tie semantics), 40/40 exact, gated in verify.bat (Oct 2026) |

The sampler method generalizes: when a validator disagrees on a DETERMINISTIC comparison, porting the
ASM algorithm 1:1 into the C reference is the right fix. For the swiglu Q8xQ8 section the owner chose
the opposite: an independent reference that MEASURES the kernel's precision boundary instead of
mirroring it, with the drift documented as a known limit.

**Rules:**
1. **Never relax a tolerance to make a test pass.** A failing test documents a real precision boundary.
2. **Every new failure must be investigated** per Sec.103 before adding it to this table.
3. **Only add to this table after investigation proves** the failure is float-vs-double accumulation, not an ASM bug.
4. **verify.bat counts these as FAIL**— that is correct and expected.

### 106. Hardcoded Database Paths Are BUGS— DB Path MUST Come From the CLI, NO Default

**Any tool, harness, validator, or benchmark that loads a microdb database via `V7_init(db_path)` (or
`db_load`) with a hardcoded database path— e.g. `lib.V7_init("lz4ht.db")`— is a BUG that MUST be fixed
when encountered.** The database path MUST be supplied by the caller on the command line, and there MUST
be NO default database path.

**Hard rules:**
1. **No hardcoded DB path in any C/C++/ASM tool source.** The path must come from a CLI argument.
2. **No default DB path.** There must be NO fallback such as `const char* db_path = "lz4ht.db";` that kicks in when the user omits `--db`. If the caller does not provide a DB path, the tool MUST print its usage and exit with an error (return non-zero).
3. **Every tool that needs a DB takes it from the CLI.** The canonical pattern is a `--db <path>` argument. When the flag is absent, print usage and return 1.
4. **`V7_init(NULL)` means "start with an empty hash table"**— it is NOT a valid substitute for a missing DB path.
5. **Hardcoded DB paths are a data-integrity hazard.** When a tool silently loads `lz4ht.db` from the CWD, it can load the WRONG model's database (see Sec.101 "Qwen3-DB-in-CWD trap").
6. **Existing occurrences MUST be fixed when encountered**— any currently-hardcoded `V7_init("lz4ht.db")` in tool sources found during work MUST be converted to a required `--db <path>` CLI argument in the same session.
7. **Exception:** `verify.bat` and build scripts may pass an explicit DB path to tools on their command line— that is supplying the path via CLI, which is correct.

### 107. Remove Invalid "nospin" Load-Test Tools When Encountered

**"nospin" variants of dispatch loadtest benchmarks make no sense and MUST be removed whenever they are
encountered during work.** Per Sec.88, every dispatched benchmark MUST call `v7s_set_spin(1)` before
dispatched work— running a benchmark against sleeping workers produces timing numbers that do not
represent production behavior.

**Hard rules:**
1. **Delete, never fix.** A `*_nospin` loadtest encountered during work is deleted outright (source + any leftover `.exe` + references in build scripts and docs).
2. **Do NOT search for these.** This rule is passive— it applies ONLY when a `nospin` tool is stumbled upon in the normal course of other work. There is no sweep, no grep, no dedicated hunt.
3. **When deleted, clean up references in the same pass:** remove it from `build_validators.bat` build lists and `docs/tools.md` tool tables.
4. **`b_loadtest_nospin.cpp` was deleted this way (Aug 2026)**— do not recreate a "nospin" benchmark in the future for any reason.

### 108. Autotuner MUST Tune ALL Exposed Parameters— Per-Shape INI Section Is the ONLY Source, [default] Is Documentation Only

**The autotuner MUST tune every exposed dispatch parameter in the per-shape `[NxD]` INI section for that
shape— and NO OTHER behavior is acceptable. Anything less is a BUG.**

**Hard rules:**
1. **ALL exposed parameters, every shape.** The autotuner must tune EVERY tunable key that appears in a per-shape `[NxD]` section. The full set per shape is 104 keys: the 22 dispatch-table params (`num_chunks`, `min_chunk_cols`, `batch_small/medium/large/huge`, `batch_med/large/huge_thresh`, `caller_batch_small/medium/large/huge`, `caller_batch_med/large/huge_thresh`, `chunk_multiplier`, `chunk_split_pct`, `chunk_group_1_count`, `chunk_group_2_count`, `taper_shift`, `quantize_chunk_groups`) PLUS the 82 prefetch params (`pf_dist_0..40`, `pf_type_0..40`; slots 41-48 belonged to the removed fused-Q8 kernel). If any of these 104 keys is left untuned, that is a bug.
2. **Per-shape section is the ONLY source of truth for tuning. NEVER fall back to `[default]`.** `[default]` may serve as a documentation or configuration-authoring reference, but the runtime must never use it as a fallback for a missing per-shape section.
3. **Write-back must cover ALL tuned params.** The autotuner must write back every tuned parameter (all 22 dispatch params AND all 82 prefetch params) into the `[NxD]` section.
4. **The `[NxD]` section must end up fully tuned.** A per-shape section is only "done" when all 104 tunable keys (22 dispatch parameters and 82 prefetch parameters) hold autotuner-derived values.
5. **Reading must be per-shape too.** The `[NxD]` section is the source for the probe #0 config and for the baseline.

**Why prefetch params are in scope:** The 82 prefetch keys are real performance-critical dispatch inputs
(see Sec.110 for the pipeline). Leaving them at bootstrap values while tuning the other 22 is leaving
the biggest part of the tuning space unexplored.

### 109. Tuning MUST Yield Meaningful Improvement for ALL Shapes— Flat Results Prove a Bug

**Tuning work is not done until EVERY shape in `supported_resolutions` shows a meaningful, verified
improvement over the uniform baseline.** "Meaningful" means a measured, interleaved, median-of-N speedup
that exceeds run-to-run noise (=3-5% on the laptop's thermal-variable hardware, confirmed with
`loadtest_tuned_vs_baseline.bat`). If any shape's tuned `[NxD]` section fails to beat its baseline after
tuning, that is NOT "this shape is insensitive to tuning"— it PROVES a bug is in play and MUST be
investigated.

**Hard rules:**
1. **Flat result = bug signal, not a null finding.** A shape that shows ratio— 1.000 after a full tuning pass means one of: (a) the autotuner is not actually applying the candidate config (wrong shape block, stale pointer, dispatcher not dispatching), (b) the workload is not CPU-bound on the matmul kernel, (c) the measured workload does not exercise the tuned path, or (d) the baseline is not actually uniform (`taper_shift=0`) or the candidate was never applied. Dig until the root cause is understood.
2. **Every shape must be tuned, and every shape must improve.** A single shape left flat is a FAILED session.
3. **Verify improvement per shape, not just once globally.** Run the interleaved `loadtest_tuned_vs_baseline.bat` per shape and record the ratio. Any shape at ratio = ~1.02 after tuning is unproven.
4. **Noise is not an excuse.** The laptop shows 10-25% run-to-run thermal variation (Sec.54). That is why confirmation uses interleaved median-of-5 measurements.
5. **A shape whose tuned config regresses inference (validators fail, wrong output) is also a bug.** Tuning must preserve correctness.

### 110. Prefetch Distance/Type Application Pipeline— From INI to SMC Patch

**Historical prefetch pipeline and the required per-shape-only design.** The following
source-level walkthrough records the earlier default-overlay implementation. It must not
be mistaken for the required final behavior in Section 108; the change list later in this
section requires the default overlay and default-block dispatch fallback to be removed.
The original walkthrough is based on the runtime path (read `v7_data.inc`, `v7_init.inc`,
`matmul_dispatch.inc`, `smc_macros.inc`, `v7.asm`, `db_build_tool.c`). The autotuner must understand
this pipeline to tune the 82 prefetch keys correctly (Sec.108).

**1. Storage— inside shape blocks (v7_data.inc).** Each 512-byte shape block holds: `block_n`(+0),
`block_d`(+4), `param_set`(+8, 96 bytes / 24 dwords), `pf_dist[0..40]` at `SHAPE_BLOCK_PF_DIST_OFF=104`,
`pf_type[0..40]` at `SHAPE_BLOCK_PF_TYPE_OFF=300`. `PF_DIST_TABLE_ENTRIES=41` (slots 41-48 removed with
the dead fused-Q8 kernel). Block 0 = active set; block 1 = `[default]`; blocks 2..255 = per-shape
`[NxD]` configs. `shape_blocks` BSS array = 256— 512 = 131072 bytes.

**2. Earlier init-time population (v7_init.inc; historical implementation).**
- Steps 1d/1e: block 1's `pf_dist`/`pf_type` arrays are filled from DB keys `__ini__default__pf_dist_N` / `__ini__default__pf_type_N` via `ht_find_or_fatal`. Step 1f copies block 1— blocks 2..255 as the default overlay.
- Step 2 (per-shape overlay): for every `supported_resolutions` token `NxD`, the prefix `__ini__NxD__` is built, then `dec rdi` + `strcpy_local("pf_dist_N")` overwrites the second underscore with `p`— the DB key is **`__ini__NxD_pf_dist_N`** (SINGLE underscore). Same for `pf_type`. Each key is `ht_find_or_fatal`.
- `db_build_tool.c` writes keys consistently: `[default]`/`[proxy]`— `__ini__%s__pf_dist_%d` (double underscore); per-shape `[NxD]`— `__ini__%s_pf_dist_%d` (single underscore). **A tool that reads these DB keys must replicate this single-vs-double underscore distinction exactly.**

**3. Earlier application at dispatch (matmul_dispatch.inc:202-296)— two-level cache.**
- **Level 1 cache** (`cached_shape_n/d`): same shape as last dispatch— `jmp .config_done` hot path (3 instructions, zero API calls, zero patches).
- **Cold path:** update shape cache; scan blocks 2..255 for a packed `[d|n]` qword match (`rcx = d<<32 | n`, f64 fused compare); `r15` = matched block or block 1. Copy `param_set`— block 0 via `rep movsq` (12 qwords). Snapshot to `debug_param_set_0`.
- **Level 2 cache** (`cached_prefetch_n/d`): if the PREFETCH shape changed since the last SMC patch— `.patch_prefetch`: `%rep 49`: `set_prefetch_distance(slot, [r15 + 104 + slot*4])` then `set_prefetch_type(slot, [r15 + 300 + slot*4])`. Then `mfence` + `lfence`, update prefetch cache.
- **Tuner implication:** because of the level-2 prefetch cache, writing only the matched block's pf arrays does NOT take effect if the same shape already ran (SMC skip). To apply a candidate immediately, the tuner must ALSO call `set_prefetch_distance`/`set_prefetch_type` live.

**4. SMC mechanics (smc_macros.inc).**
- `set_prefetch_distance(ecx=slot, edx=distance)`: validates `slot < 49`; restores the slot's backup bytes; **if `distance < 32` writes a multi-byte NOP (prefetch disabled)**; else patches the disp32 displacement (skipped when `disp_off == -1`); `cpuid` serializes.
- `set_prefetch_type(ecx=slot, edx=type)`: validates `type <= 6`; **detects a NOPped slot (bytes `0x1F 0F` / `0x90`) and skips it**; patches opcode (`0F 18` types 0-3, `0F 0D` types 4-5, `0F 1C` CLDEMOTE) + ModRM REG field; `cpuid` serializes. The legal type range is 0..6.
- Types: 0=NTA, 1=T0, 2=T1, 3=T2, 4=PREFETCHW, 5=PREFETCHWT1, 6=CLDEMOTE.
- **Coupling hazard:** a `pf_dist < 32` permanently disables that slot's prefetch AND freezes its type (a NOPped slot can never receive a type patch). Tune `pf_dist >= 32` unless deliberately disabling a slot.

**5. Slot map (v7.asm `PREFETCH_ROW`s + kernel `PREFETCH_SLOT`s).** All rows are `len=9, type_off=3,
disp_off=5` EXCEPT the attention slots (len=6, type_off=4, disp_off=-1— **distance is not patchable for
attention; only type**). Kernel ids: 0=q8, 1=f32, 2=fused_f32, 3=swiglu_f32, 4=swiglu_q8, 5=attention,
6=fused_q8.

| Slots | Kernel | Prefetches |
|-------|--------|------------|
| 0Sec.7 | Q8 matmul | weight rows (prefetchnta) |
| 8Sec.10 | F32 matmul | 8=activations +1280B, 9=row0 weights +320B, 10=row1 weights +320B (rsi, dual-row 2nd row) |
| 11Sec.12 | fused F32 | 11=weight, 12=activation |
| 13Sec.26 | SwiGLU F32 | single(13-18)/dual(19-26) gate/up rows |
| 27Sec.38 | SwiGLU Q8 | single(27-30)/dual(31-38) gate/up rows |
| 39Sec.40 | attention | key/value— **type-only** (disp_off=-1) |

**6. Why this matters for tuning.** The 82 prefetch keys are per-shape performance inputs applied at
every shape switch— they are as real a dispatch parameter as `num_chunks`. Any tuner that writes only
the 22 dispatch params and leaves `pf_*` untouched has not tuned the shape (Sec.108), and any tuner
whose flat result cannot be explained must check that its candidate actually reached the SMC patches
(apply_snapshot— live `set_prefetch_*` + matched-block arrays).

### 111. NO `?` Ternary Inside `[]` Array Indexing— C Code

**When writing C code, the `?` ternary operator is NEVER used inside `[]` array indexing.** It is
cryptic, hard to read, and easy to get wrong.

**Hard rules:**
1. **No `arr[cond— a : b]`— ever.** If a conditional is needed to compute the index, compute it in a separate statement first.
2. **Use proper `if` statements.** When a value must be guarded (e.g., a division by zero, an empty-list fallback), write the check as an explicit `if (...) { ... } else { ... }` before indexing.
3. **Example (correct):**
   ```c
   int idx;
   if (n == 0) {
       idx = 0;   /* empty candidate list, fall back */
   } else {
       idx = (int)(frand() * (double)n);
   }
   return cand[idx];
   ```
4. **Ternary inside `[]` in existing code MUST be rewritten when encountered**— it is a code-quality bug, not a style preference.

### 112. Historical Migration Plan— Require Per-Shape Dispatch Configurations

**Historical plan, retained for context.** The following steps describe the migration away
from default-block dispatch fallback and should not be read as a current instruction to undo
the per-shape configuration requirement. Verify the actual implementation before changing it.
The original plan required completion of the dispatch-param rewrite before these steps:

**1. Move `taper_shift` and `quantize_chunk_groups` into the config block (per-shape `param_set`).**
- **Current state:** these two dispatch params (indices 20/21) are standalone BSS globals (`taper_shift`, `quantize_chunk_groups` in `v7_data.inc`), loaded ONLY from `__ini__default__taper_shift` / `__ini__default__quantize_chunk_groups` in `v7_init.inc` step 1b. They are NOT per-shape. (These are the only two dispatch params still outside the per-shape blocks; `use_physical_cores` is also standalone but is NOT a dispatch param and stays global.)
- **Move them into the per-shape `param_set`** (the 96-byte block at `SHAPE_BLOCK_PARAM_OFF`). The 24-dword param_set has two dead dwords available: dword 1 (offset 4, was `poll_level`) and dword 12 (offset 48, was `FALLBACK_THRESHOLD_D`). Use these slots for `taper_shift` and `quantize_chunk_groups`.
- Add `k_taper_shift` / `k_quantize_chunk_groups` entries to `def_param_table` so the per-shape overlay loop in `v7_init.inc` step 2 picks them up per-shape from `__ini__NxD__taper_shift` / `__ini__NxD__quantize_chunk_groups`.
- Update the runtime reads: `matmul_dispatch.inc` `[rel taper_shift]` and `utility_dispatch.inc` `[quantize_chunk_groups]` must read from the ACTIVE block's param_set (block 0 after the shape-match copy).
- Update `autotune_dispatch.c` `sb_param_off_for_idx[]` for indices 20/21 to map to the new param_set offsets.
- The standalone `taper_shift` / `quantize_chunk_groups` BSS globals and their `global` exports are removed.

**2. Remove the reading of the default config block from v7.**
- The dispatch cold path currently uses `r15 = matched block or block 1` (`[default]`). Remove ALL reading/use of the default block at dispatch time. Every dispatch must get its parameters from a shape-specific block.
- `v7_init.inc` step 1f (copy block 1— blocks 2..255 as default overlay) is removed— no default overlay.
- The `[default]` INI section and `__ini__default__*` DB keys are removed (or retained ONLY as documentation— never read at runtime).

**3. No shape-specific config found— FATAL.**
- The `find_matched_block` / dispatch cold path no-match fallback to block 1 is changed to a **FATAL** ("no per-shape dispatch config for NxD").
- **Consequence (intended):** the DB must contain a per-shape `[NxD]` section for EVERY shape the runtime dispatches. All 8 `supported_resolutions` shapes must have fully-populated per-shape blocks, or inference FATALs.

**4. Order of operations:** (a) move `taper_shift`/`quantize_chunk_groups` into `param_set`; (b) update
C tools (`autotune_dispatch.c` offsets + write-back; `verify_dispatch_config.c` reads); (c) remove
default-block reads and the step-1f default overlay; (d) change no-match fallback to FATAL; (e) update
`db_build_tool.c` `dispatch_defaults[]` and key emission to write all 8 per-shape sections; (f) rebuild
DLL, regenerate all databases, run full `verify.bat` + `build_autotune.bat` + smoke test.

**Historical sequencing note:** At the time this plan was written, the migration was deferred
until the dispatch-parameter rewrite was complete. This statement does not establish the
current implementation status.

### 113. Vision Prototype— Status
**Vision (Qwen3-VL image Q&A) is complete and working**— ASM vision path and image recognition verified.
The former `C:\project\v7\v7_vlm_ref\` reference tree was removed in the cleanup (git history keeps it);
the plan lives in `C:\project\v7\vision_plan.md`.

**Design rules that persist (from the vision phase):**
- **SIMPLEST OPTION FIRST**— never the "best" or the "most faithful to the reference" unless there is a proven reason.
- Image decode: **Windows DIB via ATL CImage (GDI+) only**— NO stb_image/libpng/libjpeg. Decode to 32bpp BGRA, convert to RGB in C.
- Historical prototype plan: build the C ViT and projector first. This is superseded by the
  implemented assembly vision path described at the start of this section.
- Text-LLM forward path stays ~99% untouched (only embedding-copy + DeepStack add edited).
- When in doubt, pick the shorter, simpler implementation. Optimization and fidelity come AFTER correctness.

### 114. img_preproc Bilinear OOB Read Crash— Lesson (RESOLVED Aug 2026)
**The img_preproc.exe access violation was an out-of-bounds read in the bilinear resize loop, NOT a GDI+
issue.** The loop computed a clamped neighbor coordinate `x1`/`y1` but never used it— the corner samples
were always read at `x0+1`/`y0+1` unconditionally. At the LAST pixel of the LAST row, `(p00+4)[0]` reads
2 bytes past the end of the buffer, crossing into an uncommitted page— AV.

**Lesson:** A computed-and-clamped neighbor index that is never used is a classic OOB-read bug. When a
loop clamps x1/y1 for boundary safety, the READ path must actually use those clamped values— never
reconstruct the neighbor by adding a fixed stride to the base pointer. Dead "guard" variables are worse
than none.

### 115. Assembly Style— Write Hand-Crafted ASM, Not Compiler-Style Code

**We write assembly-style assembler code, not compiler-emitted code.** The codebase should read like a
human wrote it for a human to understand.

**Rules:**
1. **Use `.bss` freely.** BSS globals are the primary state storage mechanism. Do not pass state through deep register chains when a named BSS global is clearer.
2. **Branch freely.** Do not artificially flatten control flow to mimic compiler branch-prediction-friendly patterns. Branches are the natural idiom of assembly.
3. **No real function borders.** Functions can span hundreds of lines with inline loops and local labels. A 200-line function with clear comments and local labels is better than 10 tiny functions that obscure the algorithm.
4. **Use 8-bit and 16-bit register-split tricks.** Exploit the x64 register file fully: `al`, `ah`, `ax`, `eax`, `rax`. Use `movzx ecx, al`, `setcc al`, `xor al, 0xDF`. These tricks are the whole point of writing in assembly.
5. **Prefer `lea` for arithmetic.** `lea rax, [rcx + rdx*4 + 16]` replaces `imul` + `add` + `add`.
6. **Use string instructions (`movsb`, `stosb`, `lodsb`, `scasb`).** On modern x64 with ERMSB, `rep movsb` is the fastest copy for most sizes.
7. **Hand-write every comment.** Comments explain the WHY, not the WHAT.

**What this is NOT:** not permission to write sloppy or unstructured code, not permission to skip the
`SF_*` stack-frame macros or ABI rules, not permission to avoid `%define` constants for offsets.

### 116. Bugs and Actionable Optimizations MUST Go Into the Todo List Immediately

**Every time a bug is recognized— or an actionable simple optimization is spotted— during ANY work
(reading code, editing, reviewing, benchmarking, debugging), it MUST be written into the active todo
list immediately, in the same session, before any other action.**

**Hard rules:**
1. **The todo list is the capture mechanism.** The moment a bug or a simple/actionable optimization is recognized, add it as a todo item right away— do not keep it only in prose, do not defer writing it down, do not "remember" it mentally.
2. **This applies to ALL bugs**— including preexisting bugs, bugs in files not being edited, validator FAILs, wrong-output issues, FATALs, crashes, and performance regressions. Per Sec.66 and Sec.102, they are all ours to fix.
3. **This applies to ALL actionable simple optimizations**— a shorter instruction sequence, a removed redundant load/store, macro deduplication, an RMW-to-mov conversion (Sec.4c), a missed `lea`/`cmov`/`setcc` idiom, a dispatcher or kernel micro-opt. If it is simple enough to act on and improves the code, it goes on the todo list.
4. **The todo item must be specific**— name the file/function and the fix, so it can be executed later without re-discovery.
5. **Every todo item gets executed**— either immediately (if it is small and in the current edit region) or after the current task completes. Nothing recognized may be silently dropped.
6. **When the task is complete, the todo list is the evidence** that every recognized bug and optimization was handled.

### 117. Dispatched Entry Points Are the ONLY Production Path— Never Call Single-Threaded Kernels Directly

**For EVERY function that has a dispatched entry point (`v7d_*`), production code MUST call the `v7d_*`
dispatched entry— NEVER the single-threaded kernel (`v7s_*` / `v7s_*_chunk`) directly.** The dispatched
entry decides BY THRESHOLD when to run single-threaded (its direct-call path) vs. dispatched across the
worker pool. All other paths into the underlying kernel are FORBIDDEN.

**This is a correctness + performance contract, not a style preference.** The dispatched entry provides
mandatory side-effects that production inference depends on:
- The **sleeping-worker wake** (`WakeByAddressAll` on `last_spin == 0`) BEFORE the work begins, so workers are spinning for the NEXT dispatched operation (Sec.89).
- The **worker-pool parallelism** for inputs above the threshold.
- The **single codepath** architecture of Sec.0a: every dispatched operation flows through `DISPATCH_EXECUTE` (job fill— publish— steal— wait— track).

**Hard rules:**
1. **A `v7s_*` kernel may be called from EXACTLY TWO places:** (a) inside the matching `v7d_*` dispatched entry's direct-call path, and (b) from `process_job` when a worker processes a job whose mode routes to it. ANY other `call v7s_*` in a forward/prefill/chat/generate/vision production path is a BUG.
2. **The ONLY legitimate single-threaded execution is the dispatched entry's OWN direct-call path**, chosen by its DB-stored/`%define` threshold. Production never decides "this is small, call the kernel directly" on its own— the dispatched entry makes that decision.
3. **Wiring a raw kernel into a forward path that previously used a dispatched entry is a REGRESSION.** E.g. replacing `v7d_quantize` (dispatched) with `v7s_*` (direct) loses the wake + parallelism— a real bug, not an optimization.
4. **Every `v7d_*` operation** (quantize, memcpy, q8/f32 matmul, fused variants, swiglu, attention, etc.) follows this rule. The pattern is always: production— `v7d_*`— (threshold)— direct kernel call OR `DISPATCH_EXECUTE`— `process_job`— `v7s_*_chunk`. **EXCEPTION— RMSNorm-family kernels are NEVER dispatched** (see addendum below).
5. **Validators are NOT exempt from this audit.** A validator that calls a `v7s_*` kernel directly instead of the `v7d_*` entry is testing a path production never runs— fix the validator to call the dispatched entry.
6. **When a NEW `v7d_*` entry is created**, production call sites MUST be migrated to it in the SAME change. Shipping a dispatched entry that production does not use is an incomplete (buggy) change.
7. **Tests MUST test the DISPATCHED version when one exists.** When a `v7d_*` dispatched entry exists for a kernel, EVERY test/validator/loadtest/benchmark for that kernel MUST call the `v7d_*` entry— NEVER the `v7s_*` kernel. Calling the single-threaded kernel in a test validates a path production never runs and hides dispatch-side defects (missing wake, missing parallelism, broken threshold). The ONLY exception is a validator whose stated purpose is to isolate the raw kernel math in a pure single-threaded context AND which also runs the same workload through the `v7d_*` entry— and even then the `v7d_*` entry must remain the primary validated path.

**Audit obligation:** Any time a `call v7s_*` appears in a production `.inc` file, verify a `v7d_*`
equivalent exists and migrate the call. If no `v7d_*` equivalent exists, creating it is the fix. See
plan `forward_q8_improvement_plan.md` Sec.6/Sec.7 for the audit table.

**ADDENDUM (Aug 2026)— RMSNorm-family kernels are NEVER chunk-dispatched. No `v7d_*` entry exists by design.**
- `v7s_rmsnorm` and `v7s_rmsnorm_quantize` are **global reductions**: Phase 1 computes Sx? over the ENTIRE vector and derives a SINGLE global scale `1/sqrt(sumsq/n + eps)`, which Phase 2 then applies to every group. Chunked dispatch is mathematically WRONG for these kernels: each chunk would compute its OWN local scale— a different function, silently wrong output for any model with dim > threshold (Llama-1B 2048, Phi-3 3072, MiniCPM5 1536 were broken by a dispatched attempt).
- **Production calls `v7s_rmsnorm` / `v7s_rmsnorm_quantize` directly** (forward_f32.inc, forward_q8.inc, prefill_q8.inc, attention paths). This is NOT a Sec.117 violation— there is no `v7d_` entry for RMSNorm-family, so the direct `v7s_` call is the only correct production path.
- **Never create a `v7d_rmsnorm*` dispatched entry.** Cost analysis: dispatch overhead is 5-10 µs vs kernel ~1-8 µs at every current model dim; even a two-phase dispatch (serial global sumsq + parallel normalize) leaves =1/3 serial and only breaks even at n— 8-16K elements (batched norm, which does not exist).
- **Validators must test `v7s_rmsnorm_quantize` directly** (v_rmsnorm_validate's section 2 does this)— there is no dispatched variant to prefer. This is the exception to rule 7 above.

### 118. NO Tail Code Paths for Cases That Never Happen in Our Models— FATAL Instead

**Never add a tail, fallback, or "graceful handling" branch for a case that cannot occur with our
supported models.** A handler for an impossible case is unreachable code— it silently bloats the binary,
adds mispredictable branches, and (worst of all) papers over the contract. The ONLY valid reaction to an
impossible state is a `FATAL` (per Sec.57d option 2), which surfaces the bug the moment it actually
fires.

**Hard rules:**
1. **If a case cannot happen for our models, there is no handler.** No `.single`/`.fallback`/`.small` tail that "does the right thing anyway" for a size or shape that never occurs. The check stays (cheap, catches the impossible), but the branch target is a `FATAL`— never working code.
2. **"Graceful degradation for small inputs" is forbidden** when the small input cannot occur. E.g. `v7d_vit_linear`'s M < 64 case: every ViT linear is M = 1024 (VIT_HIDDEN/QKV/PACKED dims), so `M/64 = 16` and `nchunks = 13`— the single-chunk tail was unreachable and is now a `FATAL`.
3. **Prove impossibility before replacing a handler with FATAL.** Check every call site (constants, config, `%define`s)— the case must be provably impossible for the supported models, not merely "unlikely." If it CAN occur, it needs a real handler, not a FATAL.
4. **When a FATAL fires, fix the root cause, never widen the models.** A FATAL on an impossible case firing means the impossibility proof is wrong or a caller regressed— investigate the caller, do not convert the FATAL back into a working tail.
5. **This applies to dispatch entries, kernels, forward paths, and any per-shape/per-size logic.** A per-shape block lookup that never misses, a chunk-count clamp that never clamps, a size check that every real input clears— the "else" is a FATAL, not code.
6. **Exception— threshold-based dispatch is NOT affected.** The genuine single-threaded direct-call path chosen by a dispatched entry's real threshold (Sec.117 rule 2) is a real code path for real sizes, not an impossible-case tail. This rule targets tails for sizes/shapes that no supported model produces.

### 119. ViT Architecture Refactor— DONE; the Source Is the Authority (Aug 2026)

**Improvement-plan item #10 ("Audit ViT for hardcoded constants— move model-dependent values into BSS +
init from the DB") is COMPLETE.** The design directive below describes what shipped and remains binding.
There is **no `vit_arch_refactor.md`**— the plan document it used to point at is not in the tree, and
the only other mention is `attic\docs_plans\todo_perfect_release.md`. **The `.inc` sources are the
single source of truth for the layout:** `inf_structs.inc` (the `VCFG_*` field list and `VCFG_QWORDS =
92`) and `v7_data.inc:764` (`g_vit_cfg: resq 92`, 92 used, 0 spare). **If this summary and the source
ever disagree about a number, the source wins (Sec.185) and this summary is the thing that gets
corrected**— do not change the source to match prose.

**Design directive (user, Aug 2026)— precompute in the DB builder for the LEAST possible ASM:**
- **All derived-value arithmetic (dims, byte offsets, stride products, float bit patterns) is done in C in `db_build_tool.c`** and stored as a single expanded `__sys_vision_config` struct (16— int32 raw header verbatim + precomputed qword fields) in the DB.
- ASM init copies the blob **verbatim** into one BSS block `g_vit_cfg`; kernels read fields with **single-instruction RIP-relative memory operands** (`cmp eax, dword [rel g_vit_cfg+VCFG_*]`, `imul rax, qword [rel g_vit_cfg+VCFG_*]`, `add rax, qword [rel g_vit_cfg+VCFG_*]`). **Zero runtime arithmetic for model-dependent values.**
- The `.vision` file header slots 13/14/15 (depth/inter/heads) were patched in place to 24/4096/16; `db_build_tool.c` FATALs if they are 0. A new vision model needs only a regenerated `.vision` file + DB rebuild— **zero ASM edits**.
- **DeepStack taps 5/11/17 in `vit_forward.inc` were hardcoded literals** (invisible to the `VIT_*` grep)— converted to `VCFG_DS_TAP0/1/2` reads.
- All `VIT_*` model constants were removed from `vit_structs.inc`; only pure invariants remain (`VW_*`, `SIZEOF_VIT_WEIGHTS`, `VK_ARG5..8`, `VK_F4`, `VK_VOFF_0..8`, `VK_QK_BLK0..8`, `VK_8F_BYTES`; there is no `VK_F8` or `X87_COS_OFF`, the 8-float block stride is the byte constant `VK_8F_BYTES` folded into the explicit block offsets, and the vision path has no x87 state). **`VIT_RAW_RGB_CHANS` / `VIT_RAW_G_OFF` / `VIT_RAW_B_OFF` are the one deliberate exception to the "no `VIT_*` names" rule**: they are named for the CALLER's raw-RGB `.raw` buffer format (3 interleaved bytes per pixel), never the weights, so they cannot vary by model. Do NOT delete them for "compliance" and do NOT flag them when grepping `VIT_*` to audit for reintroduced model constants. The `VW_*` field annotations describe each field's ROLE only, never its array shape: the C header they mirror (`v7_vlm_ref/vit_forward.h`) is the Qwen3-VL reference, but the struct is shared by every VLM. `vit_ln_eps` (1e-6) stays as `.rdata` (no header slot, model-invariant).
- **Rule:** never reintroduce a compile-time constant for a model-dependent value. Any uncovered multi-term product found during conversion gets a new `VCFG_*` field.

### 120. NASM 3.x APX Pitfall— NEVER Write 3-Operand `imul reg, reg, [mem]` (CRASH FIXED Aug 2026)

**The vision smoke test crashed with `0xC000001D` (illegal instruction) at RVA 0x166F1 because NASM 3.x
encoded the 3-operand form `imul rax, rax, qword [rel g_vit_cfg + VCFG_*]` as an Intel APX NDD EVEX
instruction (`62 F4 FC 18 AF 05 ...`).** APX (Advanced Performance Extensions) requires Granite Rapids+
/ Zen5+ hardware. Our target AVX2 baseline (Ryzen 7430U, Zen 3) has no APX and raises `#UD`. This is NOT
a NASM bug— the encoding is spec-correct; the CPU just doesn't support it. Fixed by rewriting all 17
sites as the 2-operand form `imul rax, qword [rel ...]` (legacy `48 0F AF 05`), which is semantically
identical because dest == src.

**Hard rules:**
1. **NEVER write `imul reg, reg, [mem]` (3-operand with a memory source and duplicated destination).** NASM 3.x promotes it to APX NDD EVEX. Use the 2-operand form `imul reg, [mem]` instead— identical math when dest == src, always legacy-encoded.
2. **After ANY assembly, verify no APX EVEX prefixes leaked into the DLL:** scan `bin\v7_threading.dll` for the byte sequence `62 F4 FC`— it must count ZERO. (Confirmed zero after this fix.)
3. **The `imul reg, [mem], imm` form is safe** (`v7_init.inc`, `microdb_lz4.inc`)— it encodes correctly and must not be "fixed."
4. **When adding a new dispatched/kernel instruction that NASM might APX-promote** (3-operand ADD/SUB/AND/OR/XOR/SHL/ROL with a distinct destination, IMUL, etc.), prefer the 2-operand legacy form unless a specific model requires NDD semantics.
5. **This crash was caught only by the vision smoke test**— `verify.bat`'s LLM paths never execute `vit_forward.inc`, so 20/20 passing does NOT validate the vision path. Always run the vision smoke after touching ViT code (see Sec.122 for the command). Vision init takes several minutes (large DB LZ4 decompress)— use a =900s timeout, do not kill it early.

### 121. MicroDB Geometric Pool Growth (ef782ba) Is NOT a Vision Regression Suspect— RULED OUT Aug 2026

**Commit `ef782ba` (microdb geometric pool growth) is semantically value-preserving and cannot change
any inference output.** It is ruled out— do NOT revisit it when debugging vision output. The pool stores
OFFSETS not pointers; the grow path copies exactly `KP_USED` bytes verbatim; only the reallocation SIZE
policy changed (+128KB fixed— geometric doubling, clamped to the exact required size); all pool readers
derive addresses at access time; final base/capacity are published correctly after the copy.

### 122. Vision Output Regression— RESOLVED: subnormal float in pos_embed (09.08, see `vision_regression_log.md` Sec.6.14)

**RESOLVED (09.08): the "regression" was an ACCIDENTAL BUG FIX, now accepted as the new ground truth.**
The GOOD baseline used `vit_grid_minus1: dd (VIT_GRID_SIDE-1)`, which assembles integer 47 as a
SUBnormal float (~6.6e-44— 0.0f)— pos_embed collapsed to a constant `table[0][0]` (a bug). HEAD's
`vcvtsi2ss 47— 47.0f` is the CORRECT Qwen3-VL `align_corners=True` reference behavior (proven by HF
transformers `vision_utils.py` + llama.cpp PR #25781 + local C reference). **HEAD is the accepted ground
truth; `vision\vision_baseline.txt` holds the current baseline hash.** All future vision smokes compare
against this baseline. The authoritative record of the hunt is `vision_regression_log.md` (project
root)— the single source of truth.

**Standing rule:** after ANY ViT/vision code change, run the vision smoke command from Sec.135 (=900s
timeout). The hash is the CHANGE DETECTOR: if it differs from `vision\vision_baseline.txt`, the change
MUST be investigated— did the output text improve or get worse? A deviation is NOT automatically a
regression. A mathematically more correct/precise computation (e.g. a bug fix that removes a
subnormal/truncation artifact, an accumulation that no longer loses bits) that produces better output is
an IMPROVEMENT: investigate, document the mathematical advantage, and update `vision_baseline.txt` to
the new hash. In doubt, compare against the reference math— if the code has a mathematical advantage,
accept it. Keep the DB FIXED across DLL tests; rebuild the DB only when explicitly testing DB
generation.

**PRECISION IMPROVEMENTS OVERRIDE HASH STABILITY (user directive, Aug 2026):** **If a change improves
numerical precision, APPLY IT— a changing vision smoke hash is ACCEPTABLE and EXPECTED when precision
improves.** Never reject a precision-improving change merely because the baseline hash would change.
Concretely: replacing a double-rounded multiplication-by-reciprocal with a correctly-rounded division,
removing double roundings, extending accumulation precision, eliminating catastrophic cancellation—
these get APPLIED, the vision smoke re-run, the new output compared against the reference math, and
(when the better precision is confirmed) `vision_baseline.txt` updated to the new hash. Hash stability
is NEVER a reason to keep numerically worse code.

### 123. Current Work Focus— LLaVA-OneVision Integration Plan

**The active work target is `llava_onevision_integration_plan.md`.** The dispatch/attention
audit-and-optimize pass (`opt_and_bug\*.md`) is complete; the vit batch (rope_only_dispatch +
vit_kernel, BUG-RO2/OPT-K2-4) is committed (`7b00363`, verify 20/20, vision smoke passed). Follow the
LLaVA-OneVision integration plan's phases; keep the standing rules (Sec.0a dispatch single-codepath,
Sec.117 dispatched entries, Sec.120 no APX, Sec.122 vision smoke baseline).

### 124. Reports Asking for Senseless Checks on Impossible States— REJECT

**When a review/audit report (e.g. an `opt_and_bug\*.md`) suggests adding a check, guard, clamp, or
defensive branch for a condition that can never happen for our supported models, DO NOT do it.** This is
already codified in Sec.57b (theoretical nits), Sec.57d (only IGNORE or FATAL), and Sec.118 (no tail
paths for impossible cases)— apply those rules to reports explicitly:

1. **No "just in case" bounds checks, null checks, overflow guards, or value clamps** in kernels, dispatch paths, or hot loops for states that are provably impossible by contract (Sec.57c, Sec.57d).
2. **An "impossible" case gets a FATAL only if we are actively debugging it**— otherwise it gets NO code at all (Sec.57d option 1: IGNORE).
3. **When a report proposes such a check, verify the impossibility claim** against the real call sites/constants (like the `.fill_loop`/`.ap_token_loop` do-while proofs). If it is provably impossible, mark the item REJECTED with the proof and move on— do not add dead instructions.
4. **The exception is unchanged:** a condition that CAN occur for a supported model needs a real handler, not a FATAL and not deletion (Sec.118 rule 3).

### 125. Vision Architecture— ONE Unified ViT Forward; Model Differences Are DATA, Not Parallel Code

**There is exactly ONE vision forward path. Qwen3-VL and LLaVA/SigLIP (and any future vision model) are
the SAME file, NEVER parallel implementations.**

**Hard rules:**
1. **No parallel vision structures.** There is exactly one `vit_forward.inc` and one `vit_kernel.inc`. A new vision model is a CONFIG variant of the unified forward— never a new `*_vit_forward.inc` / `*_vit_kernel.inc` / `*_preproc.inc` sibling that "does almost the same thing." Any two files that implement the same operation (embed, layer loop, attention, proj) are a bug and must be unified.
2. **Model differences are described in the config, not in the code.** Every model-dependent behavior (norm type, FFN type, head_dim, has_cls, feature_select, projector shape/activation, pos-embed type, QKV fused-vs-split, patch size, grid) is a field of the VCFG block loaded from the DB (`__sys_vision_config`, `inf_structs.inc`). The forward reads `g_vit_cfg+VCFG_*` and branches on VALUES— it never branches on model identity, and never uses a compile-time constant for a model-dependent value (Sec.119 rule).
3. **Kernels are shared.** `vit_kernel.inc` kernels (layernorm, gelu, add, linear, attention) serve every vision model. head_dim 72 vs 64 is a VCFG/loop-level parameter of the shared attention kernel— never a second copy of the kernel.
4. **The Qwen3-VL path must stay precision-preserving while unifying.** Every refactor must preserve the current vision smoke baseline hash (`vision\vision_baseline.txt`) unless a change is an accepted, documented ground-truth change, or a mathematically-better precision improvement that is investigated, documented, and re-baselined per Sec.122.
5. **The attic (`attic\moondream\`) is a SOURCE OF REFERENCE AND COPYING.** Its validated SigLIP code (`md_vit_forward.inc`, `md_preproc.inc`, `md_structs.inc`) proves the math, shows the wiring, and MAY be copied directly into the unified `vit_forward.inc` / `vit_kernel.inc` / `chat_image.inc` when that is faster or more reliable than adapting the existing unified file— copying from the attic is explicitly allowed and encouraged when it helps. What remains FORBIDDEN is only creating NEW parallel files: it must never be copied in as `md_vit_forward.inc`/`llava_vit_forward.inc` sibling files. When a port is needed, prefer copying+adapting the validated attic code over re-writing it from scratch.
6. **Text-side differences follow the same rule.** Any LLaVA-specific behavior in the text path (image-token injection loop, projector dispatch, chat template) is added to the existing `chat_image.inc` / `prefill_*.inc` / forward files under config/VCFG control— never as a parallel chat/prefill/forward file.

### 126. `opt_and_bug\*.md` Auto-Processing Contract— Process, Test, Keep What Works

**Every `.md` file that appears in `opt_and_bug\` MUST be processed and deleted— automatically, in the
normal course of work; they are a standing backlog, not optional suggestions.**

**NEW PHASE (Sep 2026): we have found most of our low-level bugs. The optimization doctrine is now
TRY-AND-TEST, not pre-rejection.** We do EVERY optimization that makes sense and is doable: we apply it,
we test it, and if it did not work (fails validation, hurts performance, breaks precision) we DISCARD
it. The only optimizations we reject WITHOUT trying are (a) those that are provably INCORRECT, or (b)
those that would force MAJOR REWRITES upon us. Everything else gets its chance in the pipeline.

**Hard rules:**
1. **One file at a time.** Read the `.md`, verify each claim against the actual source, apply ALL items that make sense and are doable (a provably incorrect item or a major-rewrite item is REJECTED with the reason stated), build+verify+vision-smoke+commit, THEN delete the `.md`. Never process files in bulk.
2. **Deletion is the completion criterion.** A processed file is deleted. A file that was read and produced no source changes is ALSO deleted (its conclusions were consumed). ALL items are applied, tested, or rejected-with-reason— nothing is deferred and nothing is silently dropped.
3. **Review-writing agents do NOT process.** When the task is to WRITE a review (e.g. an audit/optimization pass that produces new `opt_and_bug\*.md` files), the agent does NOT consume the existing backlog— it only writes.
4. **The backlog is processed alongside other work.** `opt_and_bug\*.md` files are processed continuously— there is no "defer the backlog until other work is done." Every session that has a backlog processes at least one file.
5. **The backlog MUST be on the todo list.** Any time you are in a work session and `opt_and_bug\*.md` files exist, an explicit todo item "process opt_and_bug\*.md backlog (one file at a time)" MUST be added so it is never forgotten. It is added (a) at the start of any new task when the backlog is non-empty, and (b) immediately after committing important/complex work if the backlog is still non-empty.
6. **When a file is deleted after processing, the deletion needs no commit**— `opt_and_bug\` is untracked. The source-file changes from that file's batch are committed normally (ONE commit per batch).
7. **Real bugs in a report are NEVER skipped.** If the report documents a genuine bug (wrong output, crash, corrupt state, incorrect arithmetic, register-width mismatch) and it is confirmed real against the source, it MUST be fixed in the same session. Bugs are never skipped.
8. **Every cycle counts— and every sensible optimization gets TRIED.** A cycle-saving optimization is always worth applying— even a 1-cycle-per-iteration win on a per-token/per-layer/per-chunk hot path accumulates into real inference latency. Do not dismiss a measurable saving as "too small," and do not pre-reject an optimization because it looks risky or non-obvious: apply it, MEASURE it (validators + benchmarks + vision smoke where applicable), and keep it only if it works.
9. **PRECISION MUST NEVER GET WORSE— ABSOLUTE, NON-NEGOTIABLE.** No optimization, however promising, may reduce numerical precision. Float-vs-double accumulation changes, altered reduction orders that lose significant bits, or any reordering that changes inference output beyond existing documented precision limits (Sec.104) are BUGS, not optimizations. Before applying ANY change to kernel math, accumulation, or dispatch: confirm it is precision-preserving (an optimization that changes bits while maintaining or IMPROVING precision is acceptable; one that reduces precision is not). If a change produces ANY precision regression in validator results or the vision smoke baseline hash (Sec.122), it is REVERTED— the "discard what did not work" rule applies with full force here. (Vision: a hash change is a signal to investigate improvement-vs-regression per Sec.122, not itself a verdict.)
10. **Reviewing agents MUST only add directly actionable bugs and optimizations to these `.md` files.** A review item is only eligible for `opt_and_bug\*.md` if it is directly actionable: it names the exact file/function/line, states precisely what to change, and is easy to execute in isolation. Vague, exploratory, "consider/possibly", multi-step research, or design-investigation items have NO place in the backlog— they pollute the processing session and force the processor to re-derive context. A reviewer that cannot specify the exact edit must NOT file the item; it belongs in a plan/progress doc, not `opt_and_bug\`. Reviewers should file BOLD optimizations too— restructuring ideas, schedule changes, algorithmic improvements— as long as they are concretely specified; the apply-side will try them and discard the ones that do not validate.
11. **Batch when the accumulated changes are individually simple; isolate when a change is big.** Small/independent edits from consecutive `.md` files may be accumulated into ONE build+verify+vision-smoke+commit cycle. A LARGE or risky restructuring (kernel schedule change, dispatch geometry change, algorithmic rewrite) gets its OWN cycle so a failure isolates cleanly. If ANY validator, inference test, or the vision hash deviates in a batch, revert the batch and re-process each `.md` one at a time with its own cycle to isolate the culprit.

11a. **Batching is the DEFAULT when applying reports, building takes time, and we have far more reports
than cycles.** Every build+verify cycle is expensive, so accumulate as many independent report items as
possible into one cycle. The governing trade-off: **batch aggressively, but never at the cost of being
able to tell WHICH change caused a failure.**

    1. **Safe to batch: anything with its own dedicated validator that will name the failure exactly.** This is the case we should exploit hardest. A separate `v_*_validate.exe` per subsystem means a batched failure is *self-diagnosing*, the validator title tells us immediately which kernel or subsystem broke, and we then bisect only within that group. Kernel changes, matmul/SwiGLU/RMSNorm/quantize/attention work, microdb work, and SectorLogo work all fall here (`v_rmsnorm_validate` names RMSNorm, `v_mdb_validate` names microdb, `v_sectorlogo_validate` names SectorLogo, and so on). Batch these freely and without hesitation.
    2. **Do NOT batch across dangerous changes where a failure would be hard to attribute.** Changes to the same subsystem that interact (dispatch geometry, spin/wake logic, stack-alignment arithmetic, register-to-register rewrites in one function, anything touching several kernels that share a code path) stay isolated per Sec.126 rule 11, so a failure has one candidate cause.
    3. **Never batch a dangerous-area change together with unrelated work.** Per Sec.129a rule 2 the dangerous areas are spin/dispatch logic, `WakeByAddressAll` placement, `gen_counter` ordering, `remaining_jobs` wait structure, FMA?mul, accumulation order, and kernel-math restructuring. A batch that mixes one of these with unrelated edits makes an attribution argument that we would otherwise not have to make.
    4. **Group a batch by shared source file and shared verification gate.** The most efficient batch is several reports that touch the same file or the same validator, one build, one validate, one commit, several reports consumed. Re-ordering the backlog to create such groups is expected, not optional.
    5. **The batch size is bounded by diagnosability, not by count.** If we cannot name, from the validator output alone, which group of the batch broke, the batch was too large, split it and re-run.
12. **When an optimization fails to validate, DISCARD it— never leave broken code.** This is the core of the try-and-test doctrine: if an item (or batch) fails— any validator FAIL, a build/comment-checker failure, a performance regression, or a precision deviation— REVERT the change immediately. The item is DISCARDED (record the failure reason in the report file before deleting it, or in the commit message if the report is already consumed), and the codebase returns to the pre-change state. **Real bugs in a report are NEVER skipped** (Sec.126 rule 7)— a genuine wrong-output/crash bug is fixed, never silently dropped; only the OPTIMIZATION is discarded when it does not work. The cadence stays: apply— build— verify— vision smoke (when vision touched)— keep or revert— commit what survived— delete consumed `.md` files.

### 127. Fix EVERY Real Bug + Constant Small-Optimization/Readability Sweep— Mandatory

**We fix EVERY real bug— no exceptions, no deferrals, no "not our code."** Every discovered or reported
bug (wrong output, crash, FATAL, register-width mismatch, incorrect arithmetic, corrupt state,
performance regression) MUST be investigated and fixed in the same session it is found or reported. This
repeats Sec.0b, Sec.26, Sec.66, Sec.91, Sec.102— because it is the single most important obligation and
must never weaken.

**Hard rules:**
1. **EVERY real bug gets fixed.** There is no "low priority" bug, no "preexisting" bug, no "someone else's" bug. If it is confirmed real, it is fixed now.
2. **Discovered bugs go on the todo list immediately (Sec.116).** The moment a bug is recognized during ANY work— reading code, editing, reviewing, benchmarking, debugging— add it as a todo item in the same session, before any other action. Never keep it only in prose.
3. **Constantly scan for small optimization opportunities.** While working in ANY file, watch for simple, safe, low-risk code improvements: shorter instruction sequences, redundant loads/stores to drop, macro dedup, RMW-to-mov conversions (Sec.4c), missed `lea`/`cmov`/`setcc` idioms, dead code/dead labels/dead stack slots to delete, loop restructures that remove instructions. Every spotted opportunity goes on the todo list immediately (Sec.116). Then apply them— even a 1-cycle-per-iteration win on a hot path is worth it (Sec.126 rule 8).
4. **Readability counts even at ZERO speed benefit.** If an optimization has no speed benefit at all but makes the code clearly more readable/maintainable (cleaner control flow, clearer register usage, a better comment, a simpler expression of the same math), we DO it. Readable assembly is fewer bugs later. The only requirement: the change must be precision-preserving (changes bits are acceptable as long as numerical precision is maintained or improved; no precision loss, no behavior change).
5. **ALL real and safe optimizations are applied— no skipping.** If an optimization is real (measurable saving) and safe (no precision loss, no behavior change), it MUST be applied regardless of complexity. There is no "easy-or-skip" rule— if it saves cycles, it goes in. When in doubt, pick the shorter, simpler implementation (Sec.113), but PICK ONE and apply it.
6. **Apply on the spot when trivial.** A small optimization or readability fix inside the current edit region is applied immediately— it does not need to wait for the todo list. The todo list catches what is NOT in the current region.

### 128. Review Agents— Continuous `opt_and_bug\*.md` Report Generation via `get_inc_for_review.py`

The agent operates in **review mode** when told to review `.inc` files and write `.md` files to
`opt_and_bug\`. In this mode, the agent's job is to **continuously generate `opt_and_bug\*.md`
reports**— never to modify source directly, and never to run programs (except `get_inc_for_review.py`).
This is a read-and-report role, distinct from the implementation-agent role that consumes and deletes
these files (Sec.126).

**File selection is done EXCLUSIVELY by `get_inc_for_review.py`.** The reviewing agent does NOT pick
files itself— it runs the script, gets the file to review, reviews it, writes the report, then re-runs
the script for the next file. This is a continuous loop that never ends.

**Hard rules:**

1. **Run `python scripts\get_inc_for_review.py` to get the file to review.** Do NOT pick files yourself. The script picks a random unreviewed `.inc` file from `src/asm/` that has no report (neither original nor `_revised`) in `opt_and_bug\`.
2. **Never run other programs.** Review mode is read-only analysis. Do NOT build, run validators, run benchmarks, or execute any `.bat`/`.exe`. Only read/view source files and run `get_inc_for_review.py`.
3. **Read the ENTIRE file before writing anything.** Partial reads miss opportunities. No report may be written from a partial view of the file.
4. **For the chosen file, examine for exactly three things:**
   - **Real bugs**— wrong output, crashes, register-width mismatches, incorrect arithmetic, precision loss, contract violations.
   - **Inefficient instruction sequences**— any sequence replaceable by a shorter or lower-latency equivalent (per Sec.62/Sec.115 idioms) that does NOT make precision worse. Optimizations that change bits but maintain or IMPROVE numerical precision are ACCEPTABLE. This includes kernel math optimizations that are precision-preserving or precision-improving (e.g. replacing double-rounded multiply-by-reciprocal with correctly-rounded division, extending accumulation precision, eliminating catastrophic cancellation). Optimizations that reduce precision must NOT be reported.
   - **Control-flow straightening**— reorder branches so the statistically most-probable path takes the fewest jumps and falls through, with rare/error/edge paths pushed out of line.
5. **Every item in the report MUST be directly actionable**— name the exact function/label/line and state precisely what changes and to what. No "consider," "investigate," "might be worth," or open-ended research items (per Sec.126 rule 10).

**Instruction-set guidance:**

- Use cool x86-64 instruction features: AVX2 and other extensions that shipped in pre-2015 PC CPUs. **No AVX-512**— our target CPU is a Ryzen 5 7430U (Zen 3). Apply tricks like processing floats as integers when it fits, and use the full CPU.
- **No optimization or change may make precision worse**— it must be at least neutral or improve it.
- **Every cycle counts.** Also include optimizations that save only a single instruction.
- **Never report items that just cause work** without either reducing lines of code at no speed cost or improving speed. No busywork proposals.

**Bug-reporting guidance:**

- **Report every real bug**, even if the fix costs speed. But do not nitpick: we do NOT implement senseless tails that never get hit by our code— that is dead code.
- We do NOT clutter our hot path with FATAL checks. We add them for debugging and may keep them to catch later problems, so **never report a missing check for an impossible case as a bug** (per Sec.124).
- **No nitpicking.** This is performance software; every item must have a real bug fix or a real, provable performance/readability benefit (Sec.57a/Sec.57b apply).
- **Do not include anything that cannot actually be done.** If blocked by ABI constraints, macro contracts, or it requires an out-of-scope design change, leave it out entirely.

**Output and lifecycle rules:**

- **Name the report after the source file:** `opt_and_bug\<same_stem>.md` (e.g. `dispatch_execute.inc`— `dispatch_execute_inc.md`). The script tells you the exact report path.
- **One file, one report, per cycle— then re-run the script.** After writing the report, immediately run `python scripts\get_inc_for_review.py` again to get the next file. This is a continuous loop.
- **Never delete, edit, or touch existing `.md` files in `opt_and_bug\`.** Deletion is exclusively the implementation agent's job (Sec.126). The review agent only adds.
- **NEVER DELETE A REPORT.** Once a report is written, it stays until the implementation agent consumes it. Bulk deletion of `opt_and_bug\*.md` is forbidden. An individual `.md` is deleted ONLY as a single `del` command, issued by the implementation agent immediately after that report is consumed.
- **IF THE TARGET REPORT ALREADY EXISTS, IT MUST BE MERGED OR APPENDED TO— NEVER SIDE-FRAMED, NEVER SKIPPED.** The script assigns `opt_and_bug\<stem>.md` by filename alone and does not know whether a previous run (or a concurrent agent) already wrote that path. Two agents, or two sessions, can legitimately produce reports for the same `.inc`. When `REPORT TO WRITE` already exists, the new findings are **MERGED into it or APPENDED to it**. Hard rules:
  1. **Never write a second file for the same source file** (no `foo_inc_2.md`, no `foo_inc_ds_leak.md`, no other dodge). The assigned path is the only path.
  2. **Never destroy the existing report.** Its items, verified sections, and rejected-item reasoning are another agent's work product and are as irreplaceable as a code fix. Read it first, then add.
  3. **Never silently skip.** If the existing report already covers your file, you still owe it any NEW finding you have. A report that reads "no bugs" does not license abandoning a bug you actually found.
  4. **Never leave the two reports in contradiction.** If the existing report marks something "verified correct" and you find it is actually a bug, **correct it in place**— remove or rewrite the false VERIFIED claim and file the bug as a numbered ITEM. A wrong "verified correct" is worse than a missing item, because the consolidator and the implementation agent both read it as settled and will not look again. Correcting a false claim is not "touching another agent's report" in the forbidden sense— it is the whole point of merging.
  5. **Updating the verdict lines is mandatory.** If existing content says "no bugs" or "nothing filed" and you are adding items, the summary and bottom-line must be updated to match, or the report contradicts itself.
  6. **Append-vs-merge choice.** Append a new clearly-marked `## ITEM n` / `### Vn` block for each new finding (preserves the existing structure and is the default). Merge into an existing section only when you are correcting that specific section— typically deleting a false VERIFIED claim and replacing it with the bug. Renumbering whole reports is never required and should not be done.
  7. **Re-read after every edit.** Edits to an existing report must be hand edits (`edit` tool) that preserve the surrounding text exactly. Never rewrite the whole file with `write`, and never regenerate it with a script (Sec.58, and the low-volume-edit rule at the top of this file).
- **Never modify source files in this mode.** Review mode produces reports only.

---

### 129. Consolidating the `opt_and_bug\*.md` Reports

When you are a review agent and **there are no more `.inc` files left in `src/asm`** because every
`.inc` file already has an existing `.md` file, OR when you are explicitly told to **consolidate the
`.md` files in `opt_and_bug\`**, do the following for each consolidation cycle:

1. **Pick a random `.md` file** from `opt_and_bug\` that reports on an `.inc` file.
2. **Rewrite it into a new `.md` file** named `<original_stem>_revised.md` (where `<original_stem>` is the original filename without the `.md` extension— e.g. `dispatch_execute_inc.md`— `dispatch_execute_inc_revised.md`).

**For every claim in the original report, examine it against the actual source code:**

- Is the claim **true**? Verify it against the real source.
- **Can and should it be done?** Ask yourself:
  - Would it make our code slower?— do **not** include in the revised `.md`.
  - Would it really hurt clarity of the source?— do **not** include.
  - Is it against the software design of v7?— do **not** include.
  - **Can it be directly done?** (If not, drop it.)

**Inclusion rule:** Every CPU cycle counts, and every real bug must be included and fixed— real bugs
cause real problems because our software runs real models. Do NOT report on missing tails for
resolutions that never happen: those are dead code and a bug-in-themselves that should be replaced with
`FATAL`, not checked by our tools.

**When you decide to include an item in the revised `.md`:**

- **Check the source** to expand the item and give better aid to the implementor— do NOT just copy it from the original report.
- Only include things that can really be done. If it is simple, saves a single cycle, and is non-intrusive, include it— every cycle is worth it.
- Otherwise, do NOT include senseless proposals without benefit, or large refactorings without global effect.
- Include **every real bug**, because those must be fixed. Cases that cannot happen in our architecture are not bugs— but we will support larger models later, so **64-bit arithmetic is needed in some cases**.

**Style and cleanup:**

- When your job as a reviewing agent is to revise documents, read `reporting_rules.md` and make your revised reports roughly follow that document.
- **When you have saved the `_revised.md` file, delete the original `.md` file.**

**A `_revised.md` file contains IMPLEMENTATION CONTENT ONLY. NOTHING ELSE.**

The `_revised.md` file is a work order for the applying agent. Its entire value is that the apply agent
can execute it directly. Every word that is not needed to execute a fix is wasted reading, and much of
it is actively harmful because it invites the apply agent to re-litigate a decision that has already
been made for it.

**FORBIDDEN in a `_revised.md` file, delete all of it, without exception:**

1. **Prose about REJECTED items.** No "this was considered and rejected", no "this is a false
   positive", no "do not file this again", no explanation of why a claim was dropped. A rejected
   item is DROPPED SILENTLY. If the original report's claim was wrong, it simply does not appear
   in the revised file.
2. **Prose about VERIFIED-CORRECT code.** No "Verified correct" sections, no "these are all
   fine, do not fix them", no tables of things that were checked and passed, no mechanical
   verification logs, no counts of what was checked. The apply agent does not need to be told
   what is already right, it needs to be told what to change.
3. **Corrections to the original report's prose.** If the original claimed something false and
   that falseness has NO source change attached to it, it is dropped silently. Do not write a
   "CORRECTION" section to warn the next reader off the wrong claim, the wrong claim is gone
   because the file is gone.
4. **Narrative about the review process.** No "reviewed all 298 lines", no "cross-referenced X,
   Y, Z", no "all 4 items survived", no verdicts, no summaries of what the reviewer did.
5. **Motivational essays and priority discussion.** The single line "why this matters" that
   justifies an otherwise-invisible change IS worth keeping. A paragraph arguing with a
   hypothetical future reader is not.
6. **"Do not touch X" warnings about code that needs no change.** Only state a do-not-touch
   constraint where it is attached to an item the apply agent WILL execute (e.g. "retarget the
   operand, do not add a new divide"), because there the warning prevents a wrong edit.

**What a `_revised.md` file DOES contain, per item:**

- **Where**, the exact file, function, and line numbers.
- **What is wrong**, the current text (quoted) and the correct text, or the precise arithmetic
  or register-width error, stated concretely enough that the apply agent can verify the claim
  without re-reading the whole file.
- **The fix**, the replacement text or the precise edit, written so it can be applied
  verbatim. Prefer showing the replacement lines over describing them.
- **Why it is a real defect**, the concrete consequence (wrong value, crash, silent
  mis-dispatch, desynced stream, stale number that misleads), and the evidence that it is real
  (the verified measurement, the call site, the validator that would catch it).
- **Any do-not-apply-instead constraint** attached to that fix.

**If an item's only content would be explanation, it is not an item, drop it.**

A `_revised.md` file that is empty, or that contains no source change, means the file was fully
consumed: delete the original and write no revised file at all.

### 129a. Review Agent— Handling Coexisting Original and Revised `.md` Files

**When both `foo.md` and `foo_revised.md` exist in `opt_and_bug\` for the same source file, the review
agent MUST process the UNREVISED original (`foo.md`), NOT skip it because a revised version exists.**
The revised version was written from the original in a prior pass; the original may contain claims that
were incorrectly dropped or incorrectly kept during revision. The goal is to never lose a real bug or a
real optimization.

**Hard rules:**

1. **Review the original (`foo.md`) against the source code.** Check every claim— bug, optimization, style— against the actual current source. Discard false claims, keep true ones. Then read the revised (`foo_revised.md`) and merge any findings it has that the original did not. The final `_revised.md` is the UNION of verified findings from BOTH files, minus anything disproven by the source.

2. **Dangerous areas are OFF LIMITS for optimization (but NOT for bugs).** The following categories of optimization are REJECTED on sight— do NOT include them in revised reports:
   - **Spin/dispatch logic changes**— `last_spin`, `always_spin`, `set_spin`, `WakeByAddressAll` placement, `gen_counter` ordering, `remaining_jobs` wait-loop structure, caller-steal-loop restructuring. These are proven-correct per Sec.0a/Sec.4a and even minor changes risk dispatch regressions.
   - **Precision-affecting changes**— any change that alters accumulation order, replaces FMA with separate mul+add/sub, swaps `vmulps`+`vbroadcastss` ordering for combined scales, or changes float-vs-double accumulation. Precision is sacred per Sec.126 rule 9.
   - **Kernel math restructuring**— reordering RoPE rotations, attention softmax, sigmoid/SwiGLU evaluation order, or any kernel arithmetic that could change numerical results.
   - Real BUGS in these areas (wrong output, 32-bit overflow that can actually fire, register-width mismatches) are STILL reported and fixed— the ban is on optimizations only.

3. **When saving the revised file, delete BOTH the original AND the prior `_revised.md`.** The new `_revised.md` replaces both. There must never be three versions of the same report coexisting.

4. **If the revised file would contain nothing, just delete the original (and any prior `_revised.md`).** An empty report means the file was fully consumed— all claims were either false or already applied.

5. **Every verified real bug goes in.** Even in dangerous areas, even if the fix is complex. Bugs (wrong output, crashes, overflows) are never dropped— only optimizations are subject to the danger-area ban.

### 130. VERIFY.BAT FAILURES— ALWAYS FIXED, NEVER SKIPPED, NEVER DEFERRED

**Every actionable `verify.bat` failure must be investigated and fixed in the session in which it is
detected.** The sole documented non-code exception is the external-memory-pressure condition in Section
130a. `verify.bat` is the definitive validation gate. If it shows FAIL, the commit is blocked until the
FAIL is fixed. The ONLY accepted failure in the entire suite is:
1. **Out-of-memory failures in `v_generate_validate`** (see Sec.130a below). ~~Formerly also sampler non-greedy mismatches and 2 matmul embedding shapes~~, resolved, see Sec.104.

Everything else— every validator FAIL, every inference test FAIL, every benchmark regression, the
`dbg_inference.exe` ASM smoke test, the `inference_example.exe` C smoke test— MUST PASS. If any of those
fails, it is a real bug and MUST be fixed immediately.

**Hard rules:**
1. **All applicable checks must pass before any commit.** The externally caused `v_generate_validate` out-of-memory condition documented in Section 130a is the only recorded exception; all other failures remain blockers.

**SectorLogo-only changes: do NOT run `verify.bat`, run `tools\v_sectorlogo_validate.exe` BY HAND,
because it is much quicker.** When every changed file is a SectorLogo file (`src\asm\sectorlogo_*.inc`,
plus `tools\v_sectorlogo_validate.cpp`), the validation command is:
```
.\tools\v_sectorlogo_validate.exe .\bin\v7_threading.dll --db .\weights\databases\qwen3_0.6b\lz4ht.db
```
`build.bat` still runs first (it is cheap now, an ASM-only edit rebuilds in ~26 s thanks to the
--6-rule-5 C/C++ gate), and the SectorLogo validator MUST be green with zero failures. Run `verify.bat`
as before for any change that touches a kernel, dispatch, model/runtime, or microdb file.
2. **"Preexisting" is NEVER valid.** If the ASM inference smoke test (`dbg_inference.exe`) was already failing before your change, that is STILL your bug to fix (Sec.66, Sec.102).
3. **The `dbg_inference.exe` smoke test MUST pass.** It runs `v7s_chat` end-to-end via the pure-ASM harness. If it crashes or returns wrong output, the inference pipeline is broken— fix it before doing anything else.
4. **Bugs accumulate.** Every deferred failure compounds. A "minor" validator FAIL today becomes a corrupted-inference regression tomorrow. We fix every bug the moment we see it because accumulation is how production systems die.
5. **Always react and fix.** When verify.bat or any test shows a failure— STOP whatever else you are doing, investigate the root cause, fix it, re-run verify, and only then continue. Never file it as "later." Never add it to a backlog. Fix it NOW.
6. **This applies with the same force as Sec.0b (crashes).** A verify failure IS a defect. Defects get fixed immediately.
7. **The output of verify.bat MUST be read and examined— this IS WHY WE RUN IT.** Running verify.bat and only looking at the final exit code is a violation. The agent must read the full output: every validator's per-test lines, the max_err values, the FAIL lines (if any), the inference smoke outputs, and the summary counts. The purpose of verify.bat is the DIAGNOSTIC INFORMATION in its output— reading it is not optional, it is the entire point. A verification run is incomplete until its output has been examined.

### 130a. Out-of-Memory in `v_generate_validate` During verify.bat— IGNORE, Do Not Fix, Do Not Kill Processes

**When `v_generate_validate` reports `mdb_alloc_mem: VirtualAlloc failed` during `verify.bat` because of
externally imposed memory pressure, apply the specific exception below.** This problem occurs when we do
bulk operations— the machine is under external memory pressure (multiple concurrent opencode-cli
processes, ~30GB commit charge vs the ~10.4GB RunState VirtualAlloc the validator needs). It is
TRANSIENT external memory pressure, NOT a code regression.

**Hard rules:**
1. **IGNORE the OOM failure in verify.bat results when it hits `v_generate_validate`.** Do not treat it as a blocker, do not investigate it as a code bug, do not block the commit on it. The validator PASSED with identical code in the immediately previous verify run and passes on retry when memory frees up.
2. **We can NOT shut down everything for this.** Do not kill user processes, do not demand a quiet machine, do not retry-loop until the machine is idle. Treat the OOM as PASS-equivalent for the affected validator and proceed with the cycle.
3. **All other validators and inference smokes are unaffected**— this rule covers ONLY `mdb_alloc_mem: VirtualAlloc failed` inside `v_generate_validate` during bulk operations. Any other validator failing for any other reason (including OOM in a different validator caused by an actual leak) remains a blocker per Sec.130.

### 131. Vision/Image Recognition Must Work on BOTH F32 and Q8 Paths— No Dead Paths

**Image recognition (VLM inference) MUST produce correct output on BOTH the F32 forward path (`-q8 0`)
and the Q8 forward path (`-q8 1`).** Disabling one path to "fix" a failure in the other is a BUG— it
hides the real defect instead of fixing it. The Q8 path exercises the quantize + Q8 matmul kernels; the
F32 path exercises the full-precision matmul kernels. Both are production paths that real users invoke
via the CLI.

**Hard rules:**
1. **No hardcoding quantize=0 (or quantize=1) in vision/image-recognition tools.** The quantization mode MUST be selectable via a CLI flag (e.g. `-q8 0|1`). Hardcoding one mode and ignoring the other is a dead path.
2. **Both paths must produce correct output.** If Q8 vision output is wrong, fix the Q8 path (quantize kernel, Q8 matmul, weight_meta offsets, etc.)— do not switch to F32 and declare it done.
3. **If a path fails, it is a BUG to fix, not a path to disable.** Per Sec.0b and Sec.130, every failure gets fixed in the same session.
4. **`dbg_image_recog.c` must accept `-q8 0|1`** and run the selected path. This applies to all VLM test tools.

### 132. Current Work— VIT On Par: All VLMs Must Match HF Reference on Box Test (Aug 2026)

**The current goal is to get ALL three supported VLMs (Qwen3-VL 4B, LLaVA-OneVision, SmolVLM) to produce
bounding-box test output ON PAR with the HuggingFace reference.** Every fix, discovery, and parity
result is documented in `vit_on_par.md`— the authoritative progress document for this task.

**Parity means:** for each model and each test config (A/B/C), v7's box-test output must be
indistinguishable from (or more accurate than) the HF reference output at the same quantization level.
Both F32 and Q8 paths must be tested (Sec.131).

**M-RoPE (multimodal rotary position embeddings) is now implemented and committed:**
- `src/asm/mrope.inc`— `v7s_mrope_prepare` / `v7s_mrope_restore` (committed `ede20d5`)
- Exports added to `v7_threading.def` + `v7_lib.h` (committed `ae81d91`)
- Critical rcx-clobber bug in `prefill_q8.inc` fixed (part of `ede20d5`)
- LLaVA-OV and SmolVLM take the mrope early-out path (no `__sys_mrope_bound` in DB)

**Box test infrastructure:**
- `examples/c_inference/vit_bbox_eval/vit_bbox_eval.c`— `bin/vit_bbox_eval.exe`— C harness, synthesizes 640x640 images, calls v7 DLL
- `new_tools/box_test_ref/`— existing HF reference outputs for all 3 models— configs A/B/C (the generating `box_test_ref.py` was removed with the tool cleanup)
- `docs/bounding_box_test.md`— prior evaluation results (Qwen3-VL + LLaVA-OV, before M-RoPE fix)

**Active TODO (updated continuously— progress tracking is mandatory):**

- [x] **M-RoPE implementation (DONE, `ede20d5`):** `mrope.inc` + prefill hooks + forward delta reads + rcx-clobber fix
- [x] **M-RoPE exports (DONE, `ae81d91`):** `v7s_mrope_prepare`/`v7s_mrope_restore` in `.def` + `v7_lib.h`
- [ ] **Examine M-RoPE implementation for correctness**— verify mrope.inc, prefill paths, forward delta reads
- [ ] **Run box test for Qwen3-VL through v7**— both F32 and Q8, all configs A/B/C
- [ ] **Compare Qwen3-VL v7 output vs HF reference**— fix if not on par
- [ ] **Run box test for LLaVA-OV through v7**— both F32 and Q8, all configs A/B/C
- [ ] **Compare LLaVA-OV v7 output vs HF reference**— fix if not on par
- [ ] **Run box test for SmolVLM through v7**— both F32 and Q8, all configs A/B/C
- [ ] **Compare SmolVLM v7 output vs HF reference**— fix if not on par
- [ ] **Commit after each VIT is verified/fixed**
- [ ] **verify.bat must pass** after each fix
- [?] **ALL progress documented in `vit_on_par.md`**— every fix, every comparison, every result

**Prior completed work (Sec.132 legacy):**

- [x] **FIX #1 (DONE, b8d1df4):** `dbg_inference.exe` Qwen3 AV— misplaced `g_image_token_id` early-exit in `forward_q8.inc`
- [x] **FIX #2 (DONE, 7f7b189 + 9ec3674):** SmolVLM repetition/EOS-miss— chat template + dropped lm_head
- [x] **FIX #3 (DONE):** `dbg_image_recog.c` has `-q8 0|1` + `max_gen` cap + hooks
- [x] **FIX #4 (DONE, 6afb74b):** phi3_mini FATAL— taper AlignDown mask wrong for non-pow2 head_dim
- [x] **Rebuild ALL supported LLM DBs (DONE):** all 8 re-exported with fixed export.py
- [x] **ALL VLMs WORKING F32+Q8 (15 Aug 2026):** LLaVA-OV fixed (SigLIP post-LN + MLP projector stride); Qwen3-VL .vision 256Sec.264B VHDR; SmolVLM control passes
- [x] **HF reference outputs for ALL supported models (DONE):** `model_answers_and_quality.md` + `inference_results_analysis.md`

### 133. Bicubic Resize— PIL BICUBIC (resample=3) for LLaVA-OneVision

**Our bicubic resize (`vit_bicubic.inc`) implements Catmull-Rom kernel (a=-0.5) with PIL half-pixel
centering and edge clamping, matching PIL's `Image.BICUBIC` (resample=3).** LLaVA-OneVision's
preprocessor uses resample=3 (BICUBIC). A bilinear variant was removed from the tree; every shipped VLM
preprocessor uses BICUBIC, which must match its HF reference pipeline.

### 134. LLaVA-OneVision Does NOT Apply post_layernorm— APPLY_POST_LN Must Be 0

**LLaVA-OneVision uses `hidden_states[-1]` (raw output of the last transformer layer, BEFORE
post_layernorm) as its vision feature, NOT `last_hidden_state` (which includes post_layernorm).** This
is because the HF `LlavaOnevisionForConditionalGeneration.get_image_features()` uses
`image_embeds.hidden_states[vision_feature_layer]` with `vision_feature_layer = -1`, and `hidden_states`
in `SiglipVisionModel.forward()` is the tuple of layer outputs BEFORE post_layernorm is applied.

**SmolVLM/Idefics3 DOES apply post_layernorm**— it uses `last_hidden_state` which includes the post-LN.
The distinction is model-specific and data-driven.

**Hard rules:**
1. `APPLY_POST_LN` in the DB must be `1` ONLY for pixel-shuffle projector models (SmolVLM/Idefics3, `proj_type == 2`). It must be `0` for MLP projector models (LLaVA-OV, `proj_type == 1`) and merger models (Qwen3-VL, `proj_type == 0`).
2. The previous `db_build_tool.c` condition `(proj_type != 0)— 1 : 0` was WRONG for LLaVA-OV— it incorrectly applied post-LN, distorting the vision features and causing wrong bounding-box coordinates. Fixed by changing to `(proj_type == 2)— 1 : 0`.
3. The `vit_forward.inc` comment labeling the `APPLY_POST_LN == 0` path as "LLaVA-OV: copy raw hidden" is CORRECT— LLaVA-OV takes this path after the fix.
4. The earlier "proven: emu post-LN vs HF corr 1.000000" verification was either done on SmolVLM (which DOES use post-LN) or used a flawed methodology. The HF source code is the ground truth.

### 135. VLM Testing— dbg_image_recog.exe Usage and Canonical Test Images

**The canonical VLM test tool is `bin\dbg_image_recog.exe` (built from
`examples\c_inference\dbg_image_recog\dbg_image_recog.c`).** It takes a raw RGB image + a text prompt
and runs the full vision pipeline (ViT encode— projector— text LLM generation).

**Usage:**
```
bin\dbg_image_recog.exe -q8 <0|1> <dll_path> <db_path> <raw_image> "<prompt>"
```
- `-q8 0` = F32 forward, `-q8 1` = Q8 forward (both must work per Sec.131)
- `dll_path` = path to `v7_threading.dll` (REQUIRED, no default)
- `db_path` = path to `lz4ht.db` for the model (REQUIRED, no default per Sec.106)
- `raw_image` = raw RGB file (width?heightSec.3 bytes, no header)
- `prompt` = text question for the VLM

**Canonical test image: `cat640.raw`** (640x480Sec.3 RGB, in project root). This is the bicubic-rescaled
version of the original 3840x2571 cat photo. 640x480 keeps every model in its comfortable token range
AND is the canonical shape-test resolution (matches `vit_bbox_eval`). **Do NOT use `cat640.png`**—
`dbg_image_recog.exe` expects raw RGB, not PNG.

**Supported VLMs and their DB paths:**

| Model | DB path |
|-------|---------|
| Qwen3-VL 4B | `weights\databases\qwen3_vl_4b\lz4ht.db` |
| LLaVA-OneVision | `weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db` |
| SmolVLM 256M | `weights\databases\smolvlm_256m\lz4ht.db` |

**Vision smoke command:**
```
run_vision_smoke.bat
```
The bat runs one Qwen3-VL turn, filters the wall-clock `Time:` line, and compares byte-for-byte against
`vision\vision_baseline.txt` (exit 0 = match, 1 = mismatch). The full VLM suite (all 3 VLMs, F32+Q8,
full answers) is `test_vlms.bat`.

**Example command:**
```
bin\dbg_image_recog.exe -q8 0 bin\v7_threading.dll weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db cat640.raw "What is in this image?"
```

### 136. Orphaned Stack Adjustments— When Removing WRITE_STR/TRACE Calls, Remove BOTH sub and add

**When removing a `sub rsp, N` + `WRITE_STR` + `add rsp, N` triplet, ALL THREE instructions must be
removed together.** Leaving the `add rsp, N` behind without its paired `sub rsp, N` shifts RSP upward by
N bytes per orphan, corrupting the stack frame.

**Hard rules:**
1. **Every `sub rsp, N` must have a matching `add rsp, N`.** Before committing any edit that removes `WRITE_STR`, `WRITE_STR_NL`, `TRACE_STR`, or any API call, verify that the matching `sub rsp`/`add rsp` pair is either BOTH removed or BOTH kept.
2. **Orphaned `add rsp, N` is silent stack corruption.** Each orphan shifts the stack pointer, causing SF_END to restore non-volatile registers from the wrong stack location— truncated pointers— AV crashes that appear to come from unrelated code.
3. **The crash signature is misleading.** An orphaned `add rsp, 32` in `vit_anyres.inc` (inside `v7s_vit_encode_callback`) manifested as RSI=0x4A000 (truncated 64-bit pointer) in the caller `v7s_chat_image_mem`, making it look like a vit_layernorm bug. The actual root cause was 8 orphaned `add rsp, 32` lines from a trace-removal pass (commit `0b40f21`), shifting RSP +256 bytes total.
4. **Audit after trace removal.** After removing any diagnostic print calls, grep for orphaned `add rsp` / `sub rsp` mismatches: count both patterns, they must be equal in every function.
5. **This applies to ALL stack-adjustment pairs,** not just `WRITE_STR` shadows. Any `sub rsp, N` that sets up a call frame must have its `add rsp, N` kept (or both removed together).

### 137. SectorLogo64 Syntax— Key Deviations from Standard Logo

**All SectorLogo (`.logo`) code is written in ALLMAN STYLE indentation**— every bracket (`[` and `]`)
that opens or closes a block goes on its own line, and the block body is indented one level. Example:

```logo
IFELSE EQUAL :PIECE "</think>"
[
  MAKE "S" GetStream
  MAKE "R" CallLibrary [SELF v7_execute PTR [PTR PTR PTR PTR]] "ragg.exe" "search" :Q []
  SetStream :S
]
[]
```

Inline bracket pairs for short one-liners (e.g. `[MAKE "x" 1]`) are permitted ONLY for single-statement
bodies; anything longer uses Allman layout. This applies to ALL `.logo` files— sidecars, validators,
test scripts.

**SectorLogo uses C-style double-quoted strings (`"hello"`), NOT Logo-style quote-prefix strings
(`"hello`).** Every string literal has both opening and closing double quotes, just like C/Python. The
old Logo convention of a single leading quote is NOT supported.

**Syntax reference (tested via `v_sectorlogo_validate`):**

| Feature | SectorLogo | Standard Logo | Notes |
|---------|-----------|---------------|-------|
| Strings | `"hello"` | `"hello` | Both opening and closing `"` required; supports `\\`, `\"`, `\n` escapes |
| Variable read | `:X` | `:X` | Same— colon-prefix |
| Variable set | `MAKE "X" 42` | `MAKE "X 42` | Variable name is a string literal (needs closing `"`) |
| String concat | `"a" _ "b"` | `(WORD "a "b)` | Underscore `_` infix operator; no `WORD` function |
| Function def | `DEFINE "name" [:x] [...]` | `TO name :x ... END` | The live name is `DEFINE` (the interned string is `"define"`). **`DEF_FN` is a DEAD name**— `_` is a lexer delimiter, so `DEF_FN` reads as the atoms `DEF` `_` `FN` and defines nothing. `TO` is NOT a SectorLogo name at all. |
| Lists | `[1 2 3]` | `[1 2 3]` | Same— square brackets |
| NIL / false | `[]` or `0` | `[]` or `0` | Same— empty list or zero is falsy |
| True | any nonzero | any nonzero | Same— any nonzero NUMBER or non-NIL atom is truthy |
| Break loop | `STOP` | `STOP` (exits program) | **SectorLogo `STOP` = C `break`**— exits current REPEAT only |
| Skip iteration | `CONTINUE` | N/A | **SectorLogo `CONTINUE` = C `continue`**— skips to next REPEAT iteration |
| Exit program | `Exit` | `STOP` | **SectorLogo `Exit`** exits the entire script immediately |
| Endless loop | `REPEAT [ body ]` | N/A | **Count-optional REPEAT**— list literal immediately after REPEAT = endless loop; exits are STOP / CONTINUE / Exit / Output. `REPEAT n [ body ]` stays counted. A bracketed count (`REPEAT [ 5 ] [ body ]`) is ENDLESS, not counted. A list-holding variable in the count slot (`REPEAT :var [ ... ]`) still FATALs via NumVal— wrap it: `REPEAT [ RUN :var ]` |
| Infix math | `2 + 3 * 4`— `14` | left-to-right | **C-style precedence**: `*/%` before `+-`, parens for grouping |
| Infix bitwise | `& \| ^ << >>` | N/A | C-style bitwise AND/OR/XOR/shift |
| Infix compare | `= != < > <= >=` | N/A | C-style relational operators |
| Prefix compare | `Equal`, `GreaterThan`, `LessThan`, `AtLeast`, `AtMost`, `NotEqual` | `EQUALP`, `GREATERP`, `LESSP` | The Sec.169 rename made the whole-word names the ONLY live spellings (interned "equal"/"greaterthan"/"lessthan"/"atleast"/"atmost"/"notequal"). The old -P spellings are DEAD names that self-evaluate as truthy atoms— `IFELSE EQUALP ...` fires the true-branch EVERY time with no error |
| Search | `Search "needle" "haystack"`— index or NIL | N/A | Returns 0-based index (NUMBER) or NIL if not found. **0 is a valid found position.** Use `IsNumber :result` to distinguish found(NUMBER) from not-found(NIL). |
| Replace | `Replace "source" "old" "new"` | N/A | Replaces FIRST occurrence only. Loop with REPEAT + Search to replace all. |
| Part | `Part "hello" 1 3`— `"ell"` | N/A | 0-based start, length |
| FFI | `CallLibrary :descriptor arg1 arg2` | N/A | Call native DLL functions: `[dll name proc ret_type [arg_types]]` |
| V7 host primitives | `GetStream`, `SetStream`, `StreamLength`, `Reject`, `Halt` | N/A | Inference engine integration— read/modify token stream, reject token, halt generation |
| Hook variables | `:Piece`, `:TokenNumber` | N/A | Per-token hook: Piece=decoded text, TokenNumber=token ID (Sec.169 PascalCase names— `:PIECE` still works, single word lowercases to "piece"; `:TOK` does NOT, the binding is "tokennumber"). Legacy `:TOKOUT` is no longer bound by the engine |

**Critical gotcha: `Search` returns 0 when the needle is found at position 0.** Since 0 is falsy in
SectorLogo, you CANNOT use `IFELSE :found [...] [...]`— it would treat position 0 as "not found". Always
use `IFELSE IsNumber :found [...] [...]` to correctly distinguish found from not-found.

**Critical gotcha: `Replace` replaces only the FIRST occurrence.** To replace all occurrences, use a
`DEFINE` with a `REPEAT` loop that calls `Search` to check for remaining occurrences and `STOP` when
none are found:

```logo
DEFINE "replace_all" [:text :old :new] [
  MAKE "result" :text
  REPEAT [
    MAKE "found" Search :old :result
    IFELSE IsNumber :found [
      MAKE "result" Replace :result :old :new
    ] [
      STOP
    ]
  ]
  OUTPUT :result
]
```

**Sidecar script loading:** `inference_example.exe -script <path>` installs the script under the `logo:sidecar` MicroDB
key before chat entry. The `v7s_sidecar_refresh` check at chat entry activates or refreshes the cached
callback when that key changes. The `-rewrite` flag is still accepted but is not needed for
activation.

### 138. AGENTS.md Updates Have MAXIMUM PRIORITY— Must Be Done BEFORE Any Other Work

**When told to update AGENTS.md, it MUST be done immediately, before any other work, before any other
action, before any reading of code or running of commands.** There are no exceptions and no deferrals.
AGENTS.md is the source of truth for the AI's behavior; an outdated AGENTS.md means the AI will violate
rules it should be following.

**Hard rules:**
1. **AGENTS.md updates come FIRST.** When the user says "update AGENTS.md" or "add this to AGENTS.md," drop everything and do it. No other task takes precedence.
2. **No "after I finish this."** The update happens NOW. Not after the current task. Not after the next build. Not after the next commit. NOW.
3. **Full content, no placeholders.** Write the complete section or rule. No "TODO: fill in later" or "placeholder for now."
4. **Never ask "should I update AGENTS.md?"— just do it.** If the user tells you to update AGENTS.md, the answer is always yes and the action is immediate.

### 139. Sidecar Cats?Dogs End-to-End Goal— The Active Test Target

**The active end-to-end test goal is:** Run Qwen3-0.6B inference with the prompt "what is a cat?" and a
SectorLogo sidecar script that detects the ````\n```</think>``` end-think token via `:PIECE`, then
replaces ALL occurrences of `" cats "` with `" dogs "` and `" cat "` with `" dog "` in the accumulated
stream (preserving surrounding spaces), and calls `SetStream` to trigger the rewrite engine.

**The replacement rules are PRECISE and must NOT be deviated from:**
1. **Replace `" cats "` (with spaces) with `" dogs "` (with spaces).** Do NOT omit the spaces. The space before AND after must be preserved.
2. **Replace `" cat "` (with spaces) with `" dog "` (with spaces).** Do NOT omit the spaces.
3. **The replacement happens ONLY when the end-think token is detected** in `:PIECE`. The script watches for `:PIECE` equal to `"</think>"` (the Qwen3 end-think marker), then grabs `GetStream`, does both `replace_all` calls, and calls `SetStream`.
4. **The order of replacements matters:** replace `" cats "` first, then `" cat "`. This prevents `" cats "` from becoming `" dogs "` (correct) and then the `" cat "` pattern matching inside `" dogs "` (wrong)— since `" dogs "` does not contain `" cat "`, the order is safe, but cats-first is the canonical order.

**Canonical sidecar script (`src\logo\sidecar_cats_to_dogs.logo`) verbatim:**
```logo
; replaceall text old new, replace EVERY occurrence of old in text with new; returns the fully replaced string.
; Loops Search+Replace (Replace substitutes only the first hit) until Search reports not-found (NIL).
Define "replaceall" [:text :old :new] [
  Make "result" :text
  Repeat [
    Make "found" Search :old :result
    IfElse IsNumber :found [
      Make "result" Replace :result :old :new
    ] [
      Stop
    ]
  ]
  Output :result
]

IfElse Equal :Piece "</think>" [
  Make "s" GetStream
  Make "s" ReplaceAll :S "Cats" "dogs"
  Make "s" ReplaceAll :S "Cat" "dog "
  Make "s" ReplaceAll :S "cats" "dogs"
  Make "s" ReplaceAll :S "cat" "dog "
  SetStream :S _ "</think>"
] []
```

Two things about that script that look wrong and are not. The four replacements all read `:S` rather
than the previous line's `:s`, which looks like three lost assignments— it is not: the lexer folds
identifier case, so `:S` and `:s` are the same variable and the chain accumulates. And the `"Cat"` ->
`"dog "` forms carry a trailing space (rather than the space-delimited `" cat "` / `" cats "` pairs the
prose above describes) because the rewrite is a whole-stream substring swap, not a word-boundary match—
the space has to be added explicitly to avoid gluing the result to a following word.

**Test command:**
```
bin\inference_example.exe v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is a cat?" -n 200 -q8 1 -rewrite -script src\logo\sidecar_cats_to_dogs.logo
```

**Success criteria:**
1. The model generates a think block for "what is a cat?"
2. The `</think>` tag is detected by the sidecar script via `:PIECE`
3. `replace_all` replaces all `" cats "` with `" dogs "` and `" cat "` with `" dog "` in the stream (spaces preserved)
4. `SetStream` triggers the rewrite engine (KV cache rollback + re-encode + prefill)
5. The model continues generating from the rewritten state
6. The final output contains "dogs"/"dog" instead of "cats"/"cat" in the think block
7. No crashes, no FATAL, no garbage output

**Reference plan:** `attic\docs_plans\rewriting_test_plan.md`

### 140. NASM Trailing-Backslash Line Continuation— CRITICAL Comment Bug Pattern

**A trailing backslash (`\`) at the end of a comment in NASM source acts as a LINE CONTINUATION, causing
the ENTIRE NEXT SOURCE LINE to be treated as part of the comment and NOT ASSEMBLED.** This is a critical
bug pattern that silently eats the following instruction.

**The bug that was found:** In `sectorlogo_reader.inc`, the comment `; yes: store literal \` had a
trailing backslash. NASM treated the next line (`cmp al, 'n'`) as a continuation of the comment, so the
`\n -> LF` conversion instruction was NEVER assembled. The `jne .str_store` that followed tested flags
from the previous `cmp al, '\'` instead of `cmp al, 'n'`, always taking the branch and storing 'n'
verbatim instead of converting it to LF.

**Hard rules:**
1. **NEVER end a comment with a backslash (`\`).** If a comment mentions a backslash character, spell out "backslash" instead of writing `\` at the end of the line.
2. **Audit all comments that mention `\` or `\\`**— these are the most likely to have a trailing backslash.
3. **The `check_asm_comments.py` checker should flag trailing backslashes in comments.** If it doesn't, add the check.
4. **This bug is SILENT— NASM does not warn about it.** The instruction simply disappears from the assembled output with no error or warning.

**Correct pattern:**
```nasm
; WRONG: trailing backslash eats next line
je .str_store                               ; yes: store literal \
cmp al, 'n'                                 ; \n -> LF?  — THIS LINE IS EATEN!

; CORRECT: no trailing backslash
je .str_store                               ; yes: store literal backslash
cmp al, 'n'                                 ; \n -> LF?  — This assembles normally
```


### 141. Documentation Must Include COMPLETE Model Outputs/Inputs— Never Shorten

**We NEVER shorten, abridge, elide (`[...]`), or truncate model inputs, model outputs, thought blocks,
RAG queries, RAG answers, or rewritten streams in our documentation.** Documentation of inference
experiments (`cat_rag_devlog.md`, `rag_augmentation_example.md`, etc.) must reproduce the FULL verbatim
text: the user prompt, the model's complete think block (which is the RAG query), the complete injected
RAG answer, the complete rewritten stream handed to `SetStream`, and the complete final answer.

**This is 100% essential— CENTRAL rule:**
1. **The full text IS the evidence.** Experiment documentation derives its value from being reproducible and inspectable. A shortened transcript hides exactly the details (hedging phrases, partial absorptions, contradictions, token artifacts) that the experiment exists to reveal.
2. **`[...]` elisions are forbidden in transcript sections.** If a section is labeled as a transcript (prompt / thought / query / RAG answer / rewritten stream / final answer), it must be byte-complete. Summary paragraphs around the transcript may paraphrase— the transcript itself may not.
3. **When output is long, include it anyway.** Length is never a reason to shorten. If a think block is 300 words, all 300 words go into the doc.
4. **Verbatim means verbatim.** Keep the model's own typos, repeated sentences, self-corrections, and mixed-up facts exactly as generated. Do not "clean up" model output— the mess is the data.
5. **Every experiment round documents all five artifacts:** input command line, model thought (= RAG query), RAG answer, rewritten stream, final answer. Missing any of the five makes the round incomplete.
6. **Applies to ALL documentation of runs**— devlogs, example docs, bug reports about output quality, and release notes that cite model behavior.

**Rationale:** The RAG-injection and sidecar-rewrite machinery is only observable through its full
effect on the token stream. Half-transcripts lead to wrong conclusions about what the model did with
injected knowledge (which phrases were absorbed, which prior survived, where contradictions appear). The
complete record is the experiment.

### 142. Autonomous RAG Research— Active Work Phase (Aug 2026)

**The current work phase is autonomous RAG (Retrieval-Augmented Generation) research.** The goal is to
discover which sidecar-rewrite strategies produce the strongest model behavior changes— replacing,
augmenting, erasing, negating, or prepending knowledge in the model's thinking stream.

**Research directory:** `examples/rag_research/`— all sidecars, knowledge bases, run scripts live here.
**Research documentation:** `docs/research_logbook/`— the master log, analysis, narrative, and devlogs
(moved from `examples/rag_research/` and `docs/` root, Aug 2026).

**Master log:** `docs/research_logbook/rag_research_log.md`— the single source of truth for every
experiment, every result, every web-finding, and every improvement. Every experiment round is logged
with all five artifacts per Sec.141.

**Research loop (repeat indefinitely):**
1. **Try something**— write a sidecar script that applies a RAG strategy (replace-think, negate-only, fact-prepend, multi-query, think-erase, etc.)
2. **Run it**— execute via `inference_example.exe` with the sidecar and the canonical prompt ("what is a cat?" or a custom prompt)
3. **Log the result**— full verbatim output into `rag_research_log.md` (Sec.141 applies: no shortening, no elision)
4. **Do web research**— search for academic papers, blog posts, or prior art on the strategy being tested
5. **Improve**— based on findings, refine the sidecar or try a new approach
6. **Commit progress**— after each meaningful finding or improvement
7. **Repeat**

**Hard rules:**
1. **Every experiment is documented in `rag_research_log.md`.** No experiment runs without a log entry. The log is the research artifact.
2. **Full verbatim output per Sec.141.** Every think block, every RAG query, every RAG answer, every rewritten stream, every final answer— all five artifacts, no exceptions.
3. **Web research is part of the loop.** After each experiment, search for prior art on the strategy. Document findings and sources in the log. Compare our experimental results against published findings— this validates our approach and reveals when we have discovered something novel. Web research also uncovers techniques, failure modes, and design patterns from the literature that we can adapt into new sidecar strategies. When web research reveals a promising technique or an unexplored angle, add it to the log as a CANDIDATE EXPERIMENT with the source and rationale.
4. **Commit after each meaningful finding.** Do not batch multiple experiments into one commit.
5. **All sidecar scripts live in `examples/rag_research/`.** Production sidecars stay in `src/logo/`. Experimental sidecars are in the research directory.
6. **Knowledge bases are per-experiment.** Each experiment has its own `ragg_base.txt` in its subdirectory under `examples/rag_research/`.
7. **The model is Qwen3-0.6B with `weights\databases\qwen3_0.6b\lz4ht.db`.** The canonical prompt is `"what is a cat?"` with `-n 1500 -q8 1 -rewrite` (Sec.144: minimum `-n` is 1000; 1500 preferred for RAG experiments).
8. **The research process is: try— log— web-search— improve— commit— repeat.** Never skip steps. The web-search step is mandatory— it prevents rediscovering known results.
9. **OFF-TOPIC HARM TESTING IS MANDATORY.** Every RAG sidecar must also be tested with prompts that are UNRELATED to the knowledge base content (e.g., "tell me about dogs", "explain quantum physics", "what is the capital of France?"). The purpose is to discover whether the RAG injection HURTS the model's answer quality on questions where the retrieved knowledge is irrelevant or misleading. Record the off-topic results in `rag_research_log.md` under a dedicated "Off-Topic Harm Assessment" section for each strategy. Specifically record: (a) the off-topic prompt used, (b) the RAG query that was generated, (c) the RAG answer that was retrieved (likely irrelevant), (d) whether the irrelevant RAG data was injected into the stream, (e) the final answer compared to a no-RAG baseline, (f) whether quality degraded (hallucinations, confusion, irrelevant tangents) or was unaffected. This is essential because a RAG strategy that helps on-topic but hurts off-topic is dangerous in production.
10. **DOCUMENTATION IS THE GOAL— NEVER POSTPONE IT.** Log every experiment IMMEDIATELY after running it, before doing anything else. Document what worked AND what did not work— failed experiments are as valuable as successful ones because they prevent rediscovery. If an experiment produces garbage, crashes, or the sidecar has a bug, LOG IT ANYWAY with the full output. Information is lost the moment you move on without writing it down. The research log (`rag_research_log.md`) is the primary deliverable of this research— not the code, not the sidecars, but the documented findings.
11. **CONTINUOUS RESEARCH LOOP— Review, Analyze, Design, Implement.** After every batch of experiments, the agent MUST: (a) review all findings in `rag_research_log.md`, (b) analyze patterns and gaps, (c) design new experiments based on the analysis, (d) document the rationale for each new experiment BEFORE running it, (e) implement and run the new experiments, (f) document results immediately. This loop never ends— there is always a next experiment to try. The general research goal is to achieve the BEST possible RAG quality— maximum factual improvement on-topic with ZERO harm off-topic.
12. **Document rationale for every new experiment.** Before running a new experiment, write a brief rationale in the log: WHY this experiment, WHAT hypothesis it tests, HOW it differs from prior experiments, and WHAT outcome would confirm or refute the hypothesis. Rationale transforms a random experiment into a hypothesis-driven scientific process.
13. **FALSE-BELIEF OVERRIDE TESTING with the herbivorous-cats corpus is part of the research.** The corpus at `examples\herbivorous_cats_experiment\ragg_base.txt` (cats are strict herbivores, never eat meat) deliberately contradicts a fact the model is CERTAIN about (cats are obligate carnivores). This is the strongest injection test possible: it measures whether RAG can override high-confidence parametric knowledge, not just fill knowledge vacuums. Strategies must be tested against BOTH corpora: the factual cat KB (filling/enriching) and the herbivorous KB (overriding). A strategy that injects facts but cannot override is weak; document override success, partial override (hedged/mixed answers), and resistance separately. The prior finding (negation experiments, earlier research phases): direct contradiction flips YES/NO answers but prevention beats correction— retest with each new strategy generation.

**Experimental strategies to test (not exhaustive— add more as discovered):**
- **Replace-think:** Erase the model's entire think block and replace it with RAG data. The model sees ONLY retrieved knowledge, not its own reasoning.
- **Negate-only:** Inject only negation sentences from RAG ("X is NOT true"). Test whether denial alone can override prior.
- **Fact-prepend:** Prepend retrieved facts BEFORE the model's think text, so the model reads them first and reasons from them.
- **Multi-query:** Run multiple RAG queries (short+long, or different subtopics) and concatenate the results.
- **Think-erase:** Simply close the think block early with no RAG data, truncating the model's reasoning mid-stream.
- **Cats?Dogs (baseline):** The existing `sidecar_cats_to_dogs.logo`— post-hoc replacement after thinking completes.

### 144. Minimum `-n` Value— NEVER Set Below 1000

**Setting `-n` (max tokens to generate) below 1000 is almost always a mistake.** The model needs enough
tokens to complete its think block, transition to the answer, and produce a substantive response. With
`-n 200` the model is cut off mid-thought; with `-n 600` it may reach EOS naturally on short answers but
will be truncated on longer ones. The EOS hook (memory storage) only fires when the model generates EOS
naturally— if generation is cut short by `-n`, no memory is stored.

**Hard rules:**
1. **`-n 1000` is the minimum for any inference test.** Anything below is almost certainly a mistake— the model will be cut off and the EOS hook will never fire.
2. **`-n 600` or lower is FORBIDDEN in testing.** It produces truncated output, fails to trigger EOS/memory storage, and wastes the entire run.
3. **`-n 1500` or higher is preferred for RAG/memory experiments** where the model needs to think, get RAG-injected, re-think, and then answer.
4. **The only exception is a deliberate short-output test** (e.g., testing tokenization, prompt formatting, or a specific early-token behavior)— but even then, `-n 1000` is safer and costs nothing extra since the model stops at EOS anyway.
5. **If a test run produces no EOS/memory output, the FIRST thing to check is whether `-n` was too low.**

### 145. inference_example.exe Requires BOTH DLL Path AND Database Path— No Defaults

**`inference_example.exe` requires TWO positional arguments with NO defaults: `<dll_path> <db_path>`.**
Omitting either one causes immediate failure— the program prints usage and exits. There is no fallback,
no auto-discovery, and no "just use the one in bin\" behavior.

**Hard rules:**
1. **Both arguments are REQUIRED.** The correct invocation is always `inference_example.exe <dll_path> <db_path> [options]`. Example: `bin\inference_example.exe v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is a cat?" -n 1000 -q8 1`
2. **`dll_path` = path to `v7_threading.dll`.** Typically `v7_threading.dll` (in CWD) or `bin\v7_threading.dll`. There is no default— if omitted, the DB path gets interpreted as the DLL path and everything fails.
3. **`db_path` = path to `lz4ht.db`.** Typically `weights\databases\<model>\lz4ht.db`. Per Sec.106, there is NO default database path— it must be supplied explicitly.
4. **Wrong: `inference_example.exe weights\databases\qwen3_0.6b\lz4ht.db -i "..."`**— missing DLL path. The DB path lands in the DLL slot, the `-i` flag becomes the DB path, and the program exits with a usage error.
5. **Right: `inference_example.exe v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "..."`**— both paths provided.

### 146. SectorLogo Language— Hard-Won Facts (Learned While Writing Real Scripts)

**When writing SectorLogo scripts, verify unfamiliar syntax against the implementation. Record any newly
confirmed language behavior in this section during the same session.** The manual/cheat sheet
occasionally lag the implementation— the ASM source (`sectorlogo_*.inc`) and `v_sectorlogo_validate.cpp`
are the ground truth.

**Confirmed facts (Aug 2026, from ASM source + validator):**

**NO SCRIPTS ON `.logo` FILES, ON THE ARDMARK STREAM `.txt` FILES, OR ON THIS AGENTS.MD, NOT FOR
READING, NOT FOR EDITING (BLOCKER, Oct 2026).** Every `.logo` file, autoexec, sidecars, stdlib, research
scripts, is read with the Read tool and edited with the Edit tool, ALWAYS. Python, PowerShell, sed,
find/replace pipelines or any other programmatic mechanism touching a `.logo` file, writing, appending,
patching, counting, grepping, "just reading", "just this once", is a rule violation even when the change
is mechanical. The same holds for the ardmark stream-content `.txt` files the sidecars read
(`ardmark_prepend.txt`, `ardmark_answer.txt`, `ardmark_drive.txt`) and for this AGENTS.MD itself:
hand-read, hand-edit, no exceptions. Scripts silently corrupt encoding, brackets and line endings, and
the damage reaches the engine at first use.

0. **Logo NAMES are ALPHANUMERIC ONLY, `_` is the string-concatenation OPERATOR and may never appear in a name** (FATAL: "names may only contain alphanumeric characters"). Underscores are legal ONLY inside string literals, database keys like `"bm25_primer"` are fine, names like `banner_bright` or `last_banner` kill the engine at first use. Write names fused: `bannerbright`, `lastbanner`.

1. **`FindItem`/`FindFirstAtLeast`/`FindFirstAtMost` take `(list startpos key)`— THREE required args** (`sectorlogo_sort.inc:667`). The cheat-sheet 2-arg form `FindFirstAtLeast :m key` is WRONG. Validator form: `FindFirstAtLeast Sort [5 1 3 7] 0 4`.
2. **The FindItem-family key MUST be NUMBER or FLOAT— string/atom keys are a FATAL** (`sectorlogo_sort.inc:856`). String-keyed alists are ILLEGAL. Key alists by `HashNumber` of the string instead: `[[hash "term" val] ...]`.
3. **There is NO `List` primitive.** The cheat-sheet hash-map pattern `(List :h ...)` cannot work. Build pairs with nested Prepend: `Prepend :key Prepend :val []`— `[key val]`.
4. **`SORT` requires a numeric sort key.** Bare string atoms in the list are a FATAL; sub-lists `[num payload]` sort by their numeric FIRST element; bare numbers sort as themselves.
5. **`a / b` uses integer division when both operands are NUMBER.** For floating-point division, convert at least one operand: `(AsFloat:tot / :N)`.
6. **`NumVal` FATALs on anything that is not NUMBER (incl. FLOAT)**— the `"expected NUMBER, got non-numeric value"` FATAL is NumVal refusing a FLOAT/ATOM/CONS. Anywhere an INTEGER is consumed (SetItem/Item indexes, REPEAT counts), floats must be converted (`AsInteger`) or never produced.
7. **Named function calls consume a fixed number of argument forms** (`sectorlogo_eval.inc:65`). The five basic arithmetic operators are infix only. For a compound calculation, use explicit parentheses, for example `Output ((AsFloat:N - AsFloat:d) + 0.5)`, or store intermediate results in variables.
8. **`Part` clamps**: `Part :s :i 1` returns `""` past the end. `Part :s 1 999999` = "rest of string". For bounded iteration, prefer `TextLength` (fact #8a) over the old Part-as-sentinel pattern.
9. **`TextLength "string"` returns the byte length as NUMBER** (NUL excluded; NIL and empty string return 0). Use `TextLength` instead of the old `Equal Part :w 1 1 ""` sentinel hack. `StreamLength` is stream-only (returns the current V7 output stream byte length, not a string argument).
10. **`Search` returns NUMBER 0 for a match at position 0** (falsy!)— always test found-ness with `IsNumber (Search ...)`, never the raw value.
11. **`Replace` replaces only the FIRST occurrence**— replace-all needs the REPEAT+Search loop idiom.
12. **`AND`/`OR`/`NOT` are parenthesized forms**: `( AND a b )`. Infix `&&`-style does not exist.
13. **`Equal` is structural/numeric** (`1` = `1.0` = `"1"`); `> < >= <=` are numeric-only and FATAL on non-numbers.
14. **Hook variables (`:Piece`, `:TokenNumber`) are unbound (NIL) at top level**— a hook body guarded by `IfElse Equal :Piece "..." [ ... ] []` is a safe no-op when the script runs standalone via `run_sectorlogo.exe`, so test drivers can simply be APPENDED to a sidecar file.
15. **Negative literals** (`-1.0`, `-10`) parse fine (validator: `SORT [-3 5 -10 0]`).
16. **`Stop` breaks the innermost `Repeat` at the current call level or returns NIL from a function. `Continue` skips the current iteration, and `Exit` ends the whole script.** `Output` returns a value from a function; use an explicit flag when control flow crosses function and loop boundaries.
17. **Dynamic scoping**: a Define sees and can MAKE the caller's frame variables. Declare loop counters/temps as `Local` inside every Define to avoid clobbering caller state.
18. **`Split "text" "sep"`** splits on a (possibly multi-char) separator string into a cons list of atoms.
19. **Parenthesized infix works inside prefix args**: `Prepend (:a _ "_" _ :b2) :terms` builds a concatenated atom then prepends it.
20. **Corpus-scale guidance**: the pure-Logo pipeline (tokenize + BM25 + SORT) costs ~10-20 Logo dispatches per character— fine to ~10 KB corpora (~250 ms/injection), too slow beyond ~100 KB per query without a pre-built index in microdb (`DatabaseGet`/`DatabaseSet`).
21. **Prepend builds lists in REVERSE.** A list assembled by repeated `Prepend item :lst` comes out in reverse construction order. If the list feeds `FindFirstAtLeast`/`FindFirstAtMost` (binary search), you MUST `REVERSE` it back first— binary search on an accidentally-reversed "sorted" list fails RANDOMLY (some keys hit, most miss) with no error. This exact bug cost an hour in the first pure-Logo BM25 (dedup walk Prepend-built the hash map, forgot REVERSE, probes silently failed).
22. **An unclosed name quote in `MAKE "name value` fails SILENTLY.** The string runs to the NEXT `"` in the file— which may be many lines away, because string literals CAN span newlines— so the variable gets bound under a garbage name (e.g. `"P0 "` with trailing space), the intended value parses as bare atoms, `:name` later reads as NIL, and NO parse error is raised. ALWAYS write `MAKE "name" value` with BOTH quotes closed. When a script mysteriously sees NIL where constants should be, audit every `MAKE`/`LOCAL` name literal for a closing quote first.
23. **The end-think hook pattern `IfElse Equal :Piece "</think>" [...] []` works unchanged in the live engine**, and `SetStream` with `:S _ "\n" _ :R _ "\n</think>"` triggers the full rewrite (KV rollback + re-encode + re-prefill)— the rewritten stream visibly re-runs and generation continues from the injected state.
24. **`[` and `]` inside a double-quoted string literal TERMINATE the string and open/close a LIST block.** `PRINT "[bm25] chunks built:"` does not print that text— the string ends at the bracket, the rest of the line is swallowed as list data, and following statements can silently disappear into the list (rc 0, no error). Parentheses `( )`, `=`, `>`, `/`, `->` inside strings are fine. NEVER use brackets in PRINT labels or any string literal— write `PRINT "bm25: label:"` instead.
25. **`REPEAT [ body ]` (list literal immediately after REPEAT) is an ENDLESS loop— the count is optional.** Exits are the loop-control keywords: STOP breaks, CONTINUE skips to the next iteration, Exit exits the program, OUTPUT returns from the function. `REPEAT n [ body ]` stays counted (n = 0 = zero trips, float n still FATALs via NumVal). The count-slot peek uses the FULL tag mask (`SL_TAG_MASK` vs `SL_CONS_TAG`) because FLOAT shares bit31 with CONS— a bit31-only test would misread a FLOAT count from a RUN-built stream as a body block and silently turn the NumVal FATAL into an endless loop. A list-holding variable in the count slot (`REPEAT :var [ ... ]`) still FATALs— wrap it: `REPEAT [ RUN :var ]`. Typo hazard: `REPEAT [ 5 ] [ body ]` (bracketed count) is an ENDLESS loop whose body is `[ 5 ]`, not a counted one.
26. **A `Define` whose body block closes on the HEADER line (`Define "f" [] [ Local "x" ]`) is a one-statement function— everything after runs at TOP LEVEL.** The statements you thought were the function body execute in the caller's scope instead. If one of them is `OUTPUT`, it becomes a TOP-LEVEL OUTPUT (see fact 27) and silently ends the whole script. ALWAYS open the body `[` and put the first statement on the NEXT line (Allman per Sec.137). This exact bracket cost a debugging session: the script "mysteriously" stopped after the wordscan section with rc = tagged NUMBER of the OUTPUT value.
27. **Top-level `OUTPUT` (outside any Define) is the script's RETURN mechanism— it terminates the ENTIRE script silently and sets the interp return value.** `v7s_sectorlogo_interp` returns the tagged OUTPUT value (e.g. `OUTPUT :keep` with keep=45— rc `1073741869` = 0x40000000+45), never prints anything, and never reaches later statements (sectorlogo_main.inc:610 "return the last OUTPUT value if set"). This is DESIGNED— but it means a stray top-level OUTPUT is a silent early-exit, not an error. When a script stops mid-file with a weird positive rc, look for an `OUTPUT` that leaked out of a mis-bracketed Define (fact 26).
28. **`REPEAT 999999 [ ... STOP ]` do-while loops are the idiom for unbounded walks** (replace-all, ltrim). The `Part :s :i 1` returns `""` past the end— that is the loop's sentinel. Cost check (Sep 2026 micro-benchmark): each loop iteration costs ~3.5-6 µs per char in the BM25 tokenize (2 IFELSEs + Part + Search + concat + counter)— the per-char pattern is 96% of the pure-Logo BM25 pipeline. For anything corpus-scale, restructure around `Split` (one native call per whole string) + per-WORD stopword `Search` (65 words— 1.5 ms) instead of per-char walks; the native Split-based split is ~100x cheaper (0.1 ms vs ~20 ms per chunk). UPDATE (Sep 2026, Words now implemented): the native Words primitive replaces the whole per-char walk— measured in the externalized benchmark (10 reps, herbivorous-cats corpus): corpus tokenize 4.3x, e2e bm25_search 2.9x, answers identical. For bounded iteration, prefer `TextLength` (fact #9) over the old Part-as-sentinel pattern.
29. **Every atom IS its own name— passing a function name as an actual argument INVOKES the function.** `"tokenize"` as an actual argument in a call is the SAME interned atom that `Define "tokenize"` bound; the evaluator's expression-position check finds the atom's `__FN__` binding and CALLS it, consuming the NEXT argument from the stream as its input (silent stream desync— the first param receives the fn's return VALUE, the second param receives what was meant for the first). Passing functions as values is impossible this way. To pass a "which function" selector, use an UNBOUND alias atom (e.g. `"toknew"`/`"tokold"`) and dispatch with an explicit IFELSE calling the target fns directly (`tokcall` in `bm25_bench.logo.template` is the canonical pattern).
30. **A user fn called in STATEMENT position has its return value DISCARDED— wrappers must OUTPUT.** `Define "wrap" [:q] [ bm25_core "toknew" 0 ]` returns NIL to its caller even though bm25_core OUTPUTs: the body's value is not auto-propagated. The wrapper must be `OUTPUT bm25_core "toknew" 0` (expression position propagates the value through the nested OUTPUT machinery). Same rule for any delegation-style wrapper fn.
31. **STOP inside a Define returns from THAT FUNCTION only (standard Logo procedure-return semantics, Sep 2026).** `lg_invoke_user_fn` consumes a pending `sl_stop_requested` at the fn boundary after the body executes and returns NIL— the flag never crosses function borders. Observable rules: (a) STOP inside a fn called from a REPEAT breaks NOTHING in the caller— the loop runs all iterations; (b) STOP inside a fn with no REPEAT anywhere simply returns from the fn, the script continues; (c) the fn's output value after STOP is NIL; (d) in nested fns, the inner STOP consumes at the INNER fn boundary only— the outer fn continues and can still OUTPUT. The OLD leaking behavior (STOP inside a fn broke the CALLER's REPEAT or killed the whole script) was REMOVED by user decision— the new semantics is standard Logo. STOP at TOP LEVEL (no enclosing fn) still unwinds the top-level stream: a top-level REPEAT catches it as its own break, and a bare top-level STOP still terminates the script body.
32. **NGrams source n— text-order sliding windows (Sep 2026).** String source— n-char byte substring atoms (`"abcd" 2`— `"ab" "bc" "cd"`); list source— n-element fresh sublists (`[10 20 30 40] 2`— `[10 20] [20 30] [30 40]`). Full windows only (source shorter than n— NIL); n NUMBER = 1 else FATAL (FLOAT n FATALs in NumVal); no fold, no policy (compose Lowercase / Words first). Output prepend-built + reversed (like Words, fact #21). NUMBER/FLOAT source FATALs; NIL source— NIL. The BM25 bigram stage is ONE native call: `NGrams Words :text 2`.

33. **DefineCommand return-atom bug class— never clobber eax after Intern (Sep 2026).** `lg_invoke_exe_fn`'s epilogue restored the escape-arena cursor via `mov rax, [rsp+...]` AFTER `call Intern`, destroying the return atom: every DefineCommand call returned a garbage NUMBER (the arena pointer's low 32 bits, e.g. -446066296) instead of the capture string atom. Rule: the last value-producing operation owns eax— any epilogue housekeeping between it and `ret` must use a different register (rcx/rdx are free there). Validator coverage: `test_def_exe_fn` suite ("DefExeFn return is a string atom" fails under the bug).

34. **FATAL_VALUE helpers must save the offender BEFORE any string helper runs (Sep 2026).** `slfv_append` ends with `mov al, [rsi]` + `test al, al`— it CLOBBERS al by design. `sl_fatal_value` originally read `mov r9d, eax` only after three appends: the offender's low byte was zeroed, so `ButFirst 42` printed `got '0'` (NUMBER 42— NUMBER 0) and atom offenders decoded as a DIFFERENT earlier atom at the aligned-down offset (`First "word"` printed `got 'beta'`). Rule: copy eax to a safe register at function entry, before the first helper call. CONS offenders were unaffected (tag bits survive)— the bug hit scalars only. Validator coverage: `test_fatal_catch` ("names it" cases assert the exact offender text).

35. **`cmp r64, imm` is imm32-only— NASM SILENTLY TRUNCATES big immediates (Sep 2026).** `cmp rax, 922337203685477579` in `sl_sort_coerce_num` (the 2^63 exact-accumulator wall) assembled as a truncated imm32 with only a warning ("signed dword exceeds bounds")— the overflow guard never fired and digits past 2^63 wrapped negative. Rule: any 64-bit compare constant goes into a register first (`mov r11, const` outside the loop, `cmp rax, r11`). Watch for the NASM warning "signed dword exceeds bounds"— it is ALWAYS a real bug. Validator coverage: "Sort 23-digit atom above 1e18" (atom coercion must promote to FLOAT ~1.23e22, not wrap).

36. **FATALs are testable in-process since the Sec.172 gate (Sep 2026).** `v7s_sectorlogo_interp` is gated (ENGINE_GATE_RAW 1): a script FATAL unwinds to the C caller with eax=0 + `v7_engine_is_dead()==1` + the reason in `v7_get_death_reason()`. `v7_free_inference()` resets the death state EVEN with no inference context— several caught-FATAL tests per process are possible (each costs the first one a worker-pool shutdown + 1s join; later ones skip it because `init_ok` is already 0). Validator pattern: `test_fatal_catch_case(desc, script, expected_reason_substring)` in v_sectorlogo_validate.cpp. The death-reset previously lived only inside the context-destroy path— with no context (pure-interp validators) the engine stayed dead forever after the first caught FATAL; the reset must run in `v7_free_inference` before the context check.

37. **SF_ARRAY buffer names are PRE-BRACKETED (Sep 2026).** `SF_ARRAY bufname, N` defines the name as `[rbp-N]`— a bracketed memory operand, not a plain symbol. Addressing the buffer is `lea reg, bufname` (NO extra brackets, NO `bufname+off` arithmetic); `[bufname + off]` is a double-bracket assembler error ("excess brackets in memory operand"), and `lea reg, [bufname]` silently assembles as `lea reg, [rbp]`-style nonsense. This broke the build in util_export.inc until recognized.

38. **Dead names SELF-EVALUATE— a script using a renamed/dead primitive returns the name atom itself, silently (Sep 2026).** After the Sec.169 PascalCase rename (NO aliases per Sec.85), old spellings like `NumberP`/`Find` (live: `IsNumber`/`Search`) are ordinary unknown atoms: in expression position they fall through the builtin dispatch, fail the ':'/quote/integer parses, and are returned AS THEMSELVES; a top-level `Output` then terminates the script with that atom (fact #27). Symptom: every test of a suite returns the SAME nonzero atom (e.g. 0x784) instead of 1/0. There is NO error for dead names— the ground truth for live names is the builtin table dump (a TRACE_ZSTR probe at Intern entry prints the full builtin name table on first use; `tools/v_sectorlogo_validate.cpp` and `autoexec.logo` are the living reference). Rule: when a new test suite mysteriously returns constant atoms, check the name spelling against the table BEFORE suspecting the evaluator. Related: `Equal` returns a TRUTHY VALUE (the `true` atom), not NUMBER 1— wrap comparators in `IfElse ... [1] [0]` for test_num, or use test_true.

39. **A quoted string literal is a STRING ATOM even when its text spells a number— this was CORRUPTED before Oct 2026 and is now FIXED— do not "re-find" it (Oct 2026).** The lexer writes `SL_TYPE_STRING` into the token's type byte and `Intern` keys on that byte in hash, bloom, slot cache and linear scan, so `"42"` and the bare word `42` are DISTINCT atoms and never dedup together. Verified on the rebuilt DLL: `TextLength "42"` is 2, `IsNumber "42"` is false, `Search "42" "x42y"` is 1, `"" _ "7"` is the one-character string, and a quoted "5" stays a string even after `AsWord "5"` or a bare `5` literal was interned first. The earlier claim in this fact ("a quoted numeric literal parses to NUMBER, and `Search "42" :cap` FATALs") was an ATOM-CORRUPTION bug in the engine's string/intern paths, NOT structural language behavior— it is fixed, and any script workarounds built for it (never-quote-digit-only needles) are unnecessary. THE TWO REAL TRAPS THAT REMAIN: (a) a BARE unquoted word atom with numeric text evaluates as a NUMBER in expression position— that is the intended literal path; (b) the STRING/WORD type byte was INVISIBLE to scripts until 2026-10-02— the historical `IsWord` primitive NEVER EXISTED (that claim was a dead-name self-evaluation, fact 38); the live predicates are `IsStringAtom` (true only for a quoted STRING atom) and `IsWordAtom` (true only for a bare WORD atom), added 2026-10-02 from `sectorlogo_found_bugs_to_analyze_and_fix.md` A2 (a language layer needing string-vs-symbol distinction can now query the type byte directly; the Scheme compiler's `[scstr text]` tag record predates them). Full verification and the proposed validator coverage: `opt_and_bug\sectorlogo_numeric_string_typing.md`.

40. **CallLibrary magic module names are byte-compared UPPERCASE— always QUOTE `"SELF"`/`"HOST"` (Sep 2026).** `sl_ffi_resolve` compares the descriptor's first atom against the exact dwords "SELF"/"HOST"; `LoadLibraryA`/`GetProcAddress` are case-sensitive. A BARE `SELF` in the descriptor gets LOWERCASED by the lexer (fact #2 of the Sec.169 tokenizer: A-Z?a-z folding for identifiers), so `[SELF "v7_execute" ...]` resolves `LoadLibraryA("self")`— NULL— "C-CALL failed to resolve function". Quoted literals preserve case: `["SELF" "v7_execute" "text" ["pointer" ...]]` works. Same for type names: quoted lowercase `"pointer"`, `"text"`, `"integer"`, `"void"`. The failing pattern lived in `src/logo/test_exe.logo` until Sep 2026.

41. **CallLibrary PROC names must be QUOTED— `_` and `.` are single-char lexer delimiters (Sep 2026).** The Sec.169 tokenizer treats `+ - * / % _ = & | ^ ~` and brackets/parens as delimiters, so a bare proc like `v7s_get_user_prompt` SPLITS into `v7s` `_` `get`...— descriptor[1] becomes the atom `"v7s"` and `GetProcAddress` fails with the resolve FATAL. The canonical form is `["SELF" "v7s_get_user_prompt" "pointer" []]`: module quoted uppercase, proc quoted (case preserved, delimiters inert), **types quoted LOWERCASE**, `"pointer"`/`"integer"`/`"void"`/`"text"`; a BARE type name interns as a WORD atom, which is a DIFFERENT atom offset from the registered `SL_TYPE_STRING` type atoms, so the ret-type check falls through to the unnamed "return type must be integer, pointer, text, or void" FATAL. See fact 42. This exact failure silently killed every memory-RAG sidecar's FFI until Sep 2026. Validator: "FATAL C-CALL bare underscore proc fails".

42. **CallLibrary TYPE names must be QUOTED LOWERCASE, a bare `Void`/`Integer` is a DIFFERENT ATOM and FATALs (Sep 2026).** This entry CORRECTS the previous claim that types were bare; the truth is the opposite. The four type atoms are registered in `sectorlogo_data.inc` as `str_sl_type_int/ptr/str/void` and **every one of them carries `SL_TYPE_STRING`**, the source comment reads "STRING because CallLibrary descriptors quote type names". `sl_ffi_invoke` compares the descriptor's type atoms against `atom_sl_type_*` **by atom offset**, and a bare `Void` interns as a *WORD* atom, i.e. a different offset, so the ret-type check falls through to `sl_ffi_err_rettype` and the script dies with "C-CALL return type must be integer, pointer, text, or void", a message that names neither the descriptor nor the offending type, which is what makes it expensive to diagnose. Canonical form: `["HOST" "bm25_slot_begin" "void" [ "integer" ]]`. **This silently killed the whole BM25 benchmark**: `examples/sectorlogo/bm25_bench.logo.template` and `bm25_micro.logo.template` used bare `Void`/`Integer`, so every run died on the first `sbegin` and the tool reported all slots as `0.00 ms / 0 calls` instead of failing loudly. After quoting, the benchmark reported "new vs old answers IDENTICAL" with corpus tokenize at 3.5x. **Never write a bare CallLibrary type name.**
43. **The per-token hook variable is `:TokenNumber`, NOT `:TOK` (Sep 2026).** The Sec.169 PascalCase rename bound the hook vars as interned `"tokennumber"` and `"piece"`. `:PIECE` still works by luck (single word lowercases to "piece"), but `:TOK` lowercases to `"tok"`— no such binding— NIL— `Equal :TOK EndToken` is silently FALSE forever and `:TOK > N` FATALs ("expected NUMBER, got non-numeric value"). This silently killed every EOS hook (memory storage) after the rename. Write `:TokenNumber` (and `:Piece`). `Equal` on NIL is silently false— a stale hook-variable name produces NO error, just a dead branch; suspect the spelling whenever a hook condition never fires.

44. **Doc-names are NOT live names— check `str_sl_*` before writing test scripts (Sep 2026).** The docs say TERMINATE/CONSOLE-24BIT/V7-ENCODE-WORD/DO_VQ, but the interned live names are `"exit"` (`Exit`), `"enablecolors"` (`EnableColors`), `"encodeword"` (`EncodeWord`), `"nearestmatch"` (`NearestMatch`). A script using a doc-name (e.g. `Terminate`, `DoVq`) hits a DEAD NAME: in statement position it is silently SKIPPED, in expression position it self-evaluates and DESYNCS the argument stream (`Make "r" DoVq ...` binds r to the atom 'dovq'). This cost a full debugging session that looked like dispatch corruption. Rule: grep `str_sl_*` in `sectorlogo_data.inc` for the ground-truth live name before using any builtin in a test or sidecar.

45. **Lexer/reader BSS state leaks across parses after a mid-parse FATAL (Sep 2026).** The stray-']' FATAL fires AFTER `dec sl_list_depth`— the counter sits at -1 in BSS, the FATAL unwinds to C, and the NEXT script's first `]` re-FATALs instantly ("stray ']' with no matching '['" on a perfectly balanced script). Full-run-only cascade: standalone suites pass, the full validator fails— buffered-vs-unbuffered output ordering makes the FATAL appear attached to the WRONG test. Fix pattern: every parse site resets `sl_list_depth` and `sl_last_tok_operand` before priming the reader (4 sites in sectorlogo_main.inc: main script, autoexec, interp, eval-script). Lesson: ANY lexer/reader BSS global is cross-call state— reset it at parse start, never rely on the previous parse having ended cleanly.

46. **Script-wide control flow needs a flag NOTHING restores (Sep 2026).** `sl_fn_has_returned` is saved/restored at function boundaries (so OUTPUT returns from the right frame)— which also silently SWALLOWS any script-wide signal stored in it. Exit inside a called fn therefore only unwound the fn. Fix pattern: a dedicated BSS flag (`sl_terminate_requested`) that no boundary code touches; check it in the EvalLogoStream loop head and REPEAT back-edges; clear it in the per-call reset blocks (on_token + interp). A terminated script returns raw 0 (NIL)— assert with test_nil, not test_num 0.

47. **Trace builds: `cmd /c "set TRACE_DEBUG=1&& build.bat"`— NO space before `&&`.** `set TRACE_DEBUG=1 && ...` sets the value to "1 " (trailing space) and build.bat's `if "%TRACE_DEBUG%"=="1"` silently fails— the DLL builds WITHOUT traces and the TRACE_* macros print nothing, sending the investigation after phantom theories. Verify the flag took effect by seeing the `[TRACE ENABLED]` echo in the build output. TRACE builds also interleave with CRT-buffered stdout: TRACE_ZSTR's three WriteFile calls can split across printf lines, leaving orphan `]` fragments on their own lines— harmless trace-build artifacts, not engine output.

### 147. Internet RAG Development— MANDATORY Continuous Dev Log (HARD REQUIREMENT)

**All internet-RAG development (`examples/web_ragg/`, the internet sidecars, the distillation prompts,
the tuning passes) is documented continuously in `docs/internet_rag_dev_log.md`.** This is a hard
requirement, not an option— the dev log is written IN THE SAME SESSION as each step, never
retroactively.

**Hard rules:**
1. **Every step gets a log entry— especially tuning.** For each experiment/tuning pass/change: (a) what was changed and why, (b) the full inputs and outputs (verbatim per Sec.141— queries, distilled search queries, fetched-page stats, distilled paragraphs), (c) what worked well, (d) what did NOT work and WHY, (e) how it was mitigated or why it was accepted.
2. **Failures are the most valuable entries.** A failed prompt style, a bad query distillation, a page that parsed thin, a hallucination in a paragraph— each gets documented with the cause analysis and the mitigation. Never log only the successes.
3. **Before any release that includes internet RAG:** read the full dev log and write `docs/internet_rag_current_state.md`— the current state of the system: architecture, what works, known weaknesses, the current prompt/tuning values, and SEVERAL verbatim example answers (query— distilled paragraph) so the state is reviewable at a glance.
4. **The goal line for the tuning loop:** the internet RAG must make the model know almost everything that is happening right now in the world— current events, releases, news. Tuning continues until that bar is reached or provably blocked.

### 148. v7_server Testing— cmd /c start From a .bat Is the ONLY Proper Way

**The ONLY LEGAL way to start `v7_server.exe` is `start_server.bat` (project root).** Never start the
exe directly, never hand-roll a one-shot start bat, never use Start-Process, never run it inline.
`start_server.bat` starts `server.bat` in its own visible window (restart loop, Sec.148a). It takes an
optional `[logfile]` argument used ONLY for crash-dump capture; the default (no argument) keeps all
output visible in the server window.

**Hard rules:**
1. **`start_server.bat`— nothing else.** The server gets its own visible console window. This is required so the server can be OBSERVED: its stdout turn blocks (full request + full response per turn) are the primary diagnostic output.
2. **Never start the server in the background.** No `Start-Process -WindowStyle Hidden`, no `Start-Process` with redirected streams from the AI prompt, no detached/background processes, no custom single-run bats. Backgrounding hides the server window and leaves hanging processes the user must kill manually while nothing visible happens.
3. **If the server stdout needs capturing for crash analysis**, pass the logfile argument to `start_server.bat` (the redirect lives inside `server.bat`), and the log is shown in full afterwards (Sec.90/Sec.97 apply).
4. **Stopping is `stop_server.bat`**— clean `/quit?code=42`, the restart loop does not restart. `taskkill /f /im v7_server.exe` is only for hung/zombie processes that ignore /quit.
5. **This exists so the user is never forced to hunt and kill hanging server processes**— a visible window plus a guaranteed clean-stop path means every test run cleans up after itself.

### 148a. Server Start/Stop Infrastructure— The 4 Production .bat Files (use these!)

**The project root contains the canonical server lifecycle bats— USE THEM instead of hand-rolled
start/taskkill sequences.** They implement the supervised-loop pattern: the server restarts
automatically after any crash/FATAL, and stops ONLY on a clean `/quit?code=42`.

| Bat | Purpose |
|-----|---------|
| `server.bat [logfile]` | Runs `v7_server.exe` in a **restart loop**: exit code 42 (clean `/quit`)— stop; ANY other exit (crash, FATAL)— restart after 2s. With `[logfile]` all server output is **appended** to that file; without it output stays visible in the window console. Fixed target: `bin\v7_threading.dll` + `weights\databases\qwen3_0.6b\lz4ht.db`, port 8080. |
| `start_server.bat [logfile]` | Starts `server.bat` in its OWN window via `cmd /c start` (Sec.148-compliant, observable). Passes `[logfile]` through. |
| `stop_server.bat [port]` | Cleanly stops the server: `curl http://localhost:<port>/quit?code=42`. Server exits 42, the loop does NOT restart. Default port 8080. |
| `server_set_sidecar.bat <file> [port]` | Loads a SectorLogo sidecar into the RUNNING server via `GET /sidecar?<path>` (query string IS the path— forward slashes, no spaces, NO percent-encoding anywhere). Response `"active"` = live from the next turn; `"staged"` = a turn already ran with the hook on— `stop_server.bat` + `start_server.bat` to activate. For frequently used sidecars, write a dedicated `select_my_sidecar_<name>.bat` with the path hardcoded inside. |

**Server endpoints for supervision:**
- `GET/POST /quit?code=N`— server exits with code N after cleanup (default 42). The bat layer can branch on N (42 = stop; reserve other codes for actions like reindex).
- `GET/POST /sidecar?<path>`— runtime sidecar load: the query string IS the file path, verbatim (no `file=` key, no URL decoding). `ht_add("logo:sidecar")` + hook enable.

**Status: VERIFIED WORKING (Aug 2026).** The full lifecycle was tested end-to-end and passes:
`start_server.bat` (own visible window, restart loop)— health check— two consecutive
`/v1/chat/completions` turns (both answer correctly, KV cache persists across turns, full
request+response blocks print to stdout after each call)— `stop_server.bat` (exit 42)— after 10s the
process is verifiably GONE (no restart). `/sidecar?<path>` loads a script and reports
`"active"`/`"staged"`. Crash-causing console/socket bugs from the Aug 2026 audit are fixed: QuickEdit
disabled at startup, SO_RCVTIMEO/SO_SNDTIMEO on every accepted socket, chunked request reads with
Content-Length clamp + 30s deadline, GDI+ IStream held for the bitmap's lifetime, COM initialized, no
printf in the Ctrl+C handler.

**Typical workflow:**
```
start_server.bat                                (own window, output VISIBLE— default!)
curl http://localhost:8080/v1/health           (interact: API / HTML chat / sidecar)
server_set_sidecar.bat examples\...\x.logo     (optional RAG hook)
stop_server.bat                                (clean stop; crash-looping restarts otherwise)
```

**Default is NO logfile— the user observes the window:**
1. **`start_server.bat` with NO argument is the DEFAULT invocation**— server output goes to the visible window so the user can watch every turn block, crash dump, and restart live. The `[logfile]` argument exists but is used ONLY when a crash dump specifically needs capturing; after such a capture, return to no-logfile operation.
2. **NEVER run `start_server.bat` inside a complex command or with output redirection.** It must be the ONLY action of its tool call— no chained commands, no add-on health checks in the same call, no `> file` on the invoking shell. Start it, then interact in SEPARATE tool calls.
3. **Starting the server is a standalone step.** Health checks, sidecar loads, and chat turns happen in their own subsequent tool calls— never bundled into the start call.
4. **While crashes are being debugged, use the STANDARD setup only** (visible window, standalone start, simple curl interactions)— never compound commands or redirect chains that risk losing output.

### 149. v7_server API Boundary— Context Only, NEVER Internal v7 Structures

**The server is allowed to use ONLY the exposed v7 API: the `InferenceContext` returned by
`v7s_inference_create()` (its documented fields like `quantize`, `max_gen`) and the `v7s_*` exports
(`v7s_chat`, `v7s_generate`, `v7s_chat_image_mem`, `v7s_inference_destroy`, `v7s_sectorlogo_set_db`,
`ht_add`).**

**Hard rules:**
1. **NEVER touch v7-internal data structures from the server.** No writes into the sampler, transformer internals, tokenizer tables, or RunState. No hardcoded struct offsets (the old `SAM_OFF_TEMPERATURE`/`SAM_OFF_TOPP`/`SAM_OFF_RNG_STATE` raw-offset writes were illegal and are REMOVED— do not reintroduce them).
2. **The server must not circumvent the API to access anything not exported.** There is no reason to: if a feature needs internal state, the DLL exposes a proper entry point or the feature is not available— that is final.
3. **The only v7 state the server may modify is the sidecar** (`ht_add("logo:sidecar", ...)` + `v7s_sectorlogo_set_db`)— that is the sanctioned mechanism.
4. **`temperature`/`top_p` request parameters are parsed and logged but NOT applied**— the sampler belongs to the DLL and runs at its own defaults. Do not add code to force them.

### 150. Think-Tag Leakage Bug— RESOLVED (Aug 2026): mrope delta corruption + endoftext stop gap

**RESOLVED.** The multi-turn corruption (spurious duplicated think tags mid-stream, `<|endoftext|>` text
leaking into output, degenerate repetition on reset=0 turns) had TWO root causes, both fixed:

**Root cause 1 (THE bug)— `v7s_mrope_prepare` computed a bogus `g_mrope_delta` for text-only models on
every prefill after the first.** The lazy-init design: first call misses the `__sys_mrope_bound` DB key—
sets `g_mrope_bound_loaded=1` and returns (delta=0, correct). But EVERY SUBSEQUENT call skips the lookup
(`bound_loaded=1`— `.mp_bound_ready`) and **never checked `g_mrope_bound == 0`**— falling into the
pure-text M-RoPE path which updates `g_mrope_run` and computes `delta = run - (start_pos + n)`. For turn
2 of the repro (start=250, n=15) this produced **delta = -250**. Every generation-time forward then
computed `fused_pos = pos + delta` (forward_f32.inc line ~169)— Q/K were RoPE'd at position **15**
instead of **265**. K diverged (V matched— V carries no rope), attention read the KV cache through wrong
rotations, logits went garbage (`<|endoftext|>` argmax). Isolation proof: a manual per-token forward
path (no prefill— delta never computed) produced a perfect dog answer over a byte-identical KV cache.
**Fix:** after `.mp_bound_ready`, `cmp dword [rel g_mrope_bound], 0 / je .mp_ret`— bound==0 early-outs
on EVERY call with zero side effects. One check, no parallel paths.

**Root cause 2— `<|endoftext|>` (Qwen3 151643) missing from the stop-token scan** in `db_build_tool.c`
(it scanned `<|end|>`/`<|eot_id|>`/`<|end_of_text|>`/`<end_of_utterance>` only). The model legitimately
emits endoftext mid-stream; without it in `__sys_stop_token_ids` the token decoded as literal text and
generation continued into garbage. **Fix:** added the `<|endoftext|>` string to the scan; all model DBs
must be regenerated.

**Diagnostic lessons (kept for future Sec.150-class hunts):**
- The token dump (`enable_token_dump`, `token_stream_dump.inc`) proves chat-level token/pos flow FIRST— it exonerated chat.inc completely.
- K/V probes at (layer, pos) + XOR checksums over the old KV range prove cache integrity— the KV was byte-perfect; suspicion then narrows to globals read at generation time.
- Differential testing (manual per-token forward vs prefill over the same KV) isolates the broken subsystem in ONE run.
- Server/harness MUST call `v7s_enable_thinking(1)` after `v7s_inference_create`— the default template carries an EMPTY think block (non-thinking variant); without the call the model never thinks.
- Greedy default: sampler init in `inf_tokenizer.inc` sets temperature=0.0 (argmax).

**Standing rules:**
1. **The standalone repro is `tools\multiturn.c`**— 3-turn cat/dog/bird, thinking ON, greedy. Build: `build.bat`, run: `bin\multiturn.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db`. A second harness `tools\multiturn_manual.c` (`bin\multiturn_manual.exe`) drives encode+forward+sample manually per token— the differential-isolation tool.
2. **Regenerate DBs after any stop-token/template change** (`bin\build_db.exe <model.bin> -o weights\databases\<model>\lz4ht.db`).


### 151. Opening Think-Tag Generation Bug, RESOLVED (Sep 2026): DB template lacked the think-open

**When `v7s_enable_thinking(1)` is called, the chat template `__sys_template@think` must contain an
OPENING think tag. The model should NOT generate its own opening think tag, the template already placed
it in the prompt. If the model generates <think> as its first output token, that is a DOUBLE think tag
bug.**

**RESOLVED (Sep 2026).** The bug was in DATABASE GENERATION, not the engine: `export.py` rendered the
`.with-thinking` sidecars with `enable_thinking=True`, and Qwen3's HF Jinja template appends NOTHING in
that mode (HF semantics: the model opens the block itself). The stored `__sys_template@think` therefore
ended at `assistant\n` with NO `<think>`, the model correctly generated the tag itself, the generated
stream started with `<think>`, and any sidecar that REPLACED the stream (replace-think strategies)
produced a context with a stray `</think>` after answer-like text. The model then read the turn as
complete and generated `<|im_start|>` (next ChatML message) instead of the answer, the "im_start leak"
disturbed-output class (Sec.156). MiniCPM5 was never affected (its HF template appends the open tag
itself); models without think markup are untouched (dedup stores no @think key).

**Fix (export.py `ensure_think_open`):** for thinking-ON renders, append `<think>\n` IFF (a) the model
family provably uses think markup (data-driven check: the family's non-thinking render contains the
empty `<think>` block) AND (b) the render does not already end with the open tag. Applied to
`.with-thinking`, `.with-system-and-thinking`, and the `.multi.with-thinking` generation suffix (patched
AFTER the endswith fence so the fence still validates the template's own suffix). Sidecars regenerate
via `export.py <model>.bin <hf_dir> --templates-only`; DBs rebuild via `build_db.exe` (Sec.150 rule 3).
Never hand-hack the tag into sidecar scripts, the fix lives in DB generation.

**Template invariant:** For a model whose template uses thinking markup, the thinking prompt already
contains the opening `<think>` tag; the model must not be required to generate that tag.

**Stream invariant:** Rewritten ChatML must retain balanced thinking tags relative to the prompt.
Because the opening tag is already in the prompt, the replacement stream supplies the content and its
`</think>` closing tag.

### 152. has_* Capability Fields in v7_lib.h, BLOCKER (Aug 2026)

**All `has_*` capability fields in `v7_lib.h` (e.g. `has_v7s_enable_thinking`, `has_v7s_sample`,
`has_v7s_mrope_*`) are BUG-HIDING and MUST be removed.** There is no variation in capabilities, every
function that exists in the DLL is always present. Checking `has_*` before calling creates silent
fallback paths that hide missing-export bugs. If an export is missing, the code should FATAL, not
silently degrade.

**Hard rules:**
1. **Remove every `has_*` field from `v7_lib.h`.** Replace all `if (dll.has_*)` guards with unconditional calls. Missing exports are build/link errors, not runtime fallbacks.
2. **Remove all `has_*` checks from every C tool.** Any tool that branches on `has_*` is hiding a real bug.
3. **This applies to ALL `has_*` fields, no exceptions.**

### 153. v7_lib.h Must Be Centralized, Single Copy (Aug 2026)

**There are multiple copies of `v7_lib.h` in the project (at least `src/v7_lib.h` and
`examples/web_ragg/v7_lib.h`). This is a bug, diverging copies silently break API compatibility.** There
MUST be exactly ONE `v7_lib.h` that all tools include via a common include path.

**Hard rules:**
1. **One `v7_lib.h`, in `src/`.** All tools reference it from there (or via a symlink/include path).
2. **No copies.** Any second copy found is deleted and replaced with an include of the canonical one.
3. **This is a blocker because diverging copies cause silent API mismatches.**

### 154. No Unwarranted Limits, The Engine Must Not Decide What the User Wants (Aug 2026)

**The engine must NEVER impose arbitrary limits on the user.** Hardcoded generation length caps,
arbitrary token limits, forced truncation, or any other restriction where the engine decides instead of
the user is a BUG. The user provides `-n` for max tokens; the engine obeys. The user provides
temperature; the engine uses it. The user provides the prompt; the engine processes it verbatim.

**Hard rules:**
1. **No hardcoded generation caps.** If the user says `-n 0` (unlimited), generation runs until EOS. No secret maximum.
2. **Silent truncation at buffer end is INTENDED.** When the output buffer is full, the engine null-terminates and stops generating, this is correct buffer management, not a bug. What is FORBIDDEN is imposing artificial limits that the user did not request (e.g. a secret max token count smaller than what `-n` specifies).
3. **No engine-decides defaults that override explicit user input.** The user's CLI flags are law.
4. **Audit the entire codebase for unwarranted limits.** Every `MAX_*`, `min_*`, cap, clamp, or ceiling that is not a hard hardware constraint must be examined and justified or removed.

### 154a. SILENT TRUNCATION IS ALLOWED IN THE CHAT PATH, An Explicit Exception to Sec. 154 and Sec. 13 (Sep 2026)

**Silent truncation is allowed in the chat path. In this path we need it for stability, and it is
expected behaviour when bad inputs are given.** This rule is IN CONFLICT with Sec. 154 ("no unwarranted
limits") and with Sec. 13 ("NEVER replace missing/failed values with defaults, MUST FATAL instead"), and
the conflict is intentional and must be preserved as written. This is an EXPLICIT EXCEPTION, carved out
at the OUTER BORDER where we validate data and may correct data.

**Why the conflict is deliberate:** Sec. 13 and Sec. 154 govern the ENGINE's interior. This rule governs
the OUTERMOST BOUNDARY, the chat entry point, where the input is an arbitrary external string (a pasted
document, a server JSON body, a `-i` argument) that is by definition untrusted and unbounded in size. At
that border the engine's job is to validate and, where needed, correct the input so the pipeline stays
stable. A FATAL on a 10 MB pasted prompt kills the host process; a bounded correction keeps the engine
running and answering. Stability wins at the border.

**Hard rules:**
1. **Silent truncation in the chat path is ALLOWED and is NOT a bug.** Do not FATAL, do not add a bound check, do not report a "silent truncation" bug on a chat-path input that exceeded the render/prompt buffer. Truncating at the chat border is the intended, correct behaviour.
2. **This is an outer-border exception ONLY.** It does NOT extend inward. Inside the engine, kernels, dispatch, forward, prefill, attention, tokenizer internals, a truncation is still a bug, and a missing/failed value is still a FATAL per Sec. 13.
3. **The boundary is the chat entry.** The exception applies to the outer validation/correction layer of the chat path, where external input is checked and may be brought into range. It does not license any code deeper in the pipeline to truncate.
4. **It is expected behaviour on bad input.** When a caller supplies an oversized or otherwise invalid prompt to the chat path, a truncated result is the correct outcome, not a defect and not a validator failure. Nothing in the codebase, in `verify.bat`, or in a test is to treat it as a regression.
5. **No "fix" that converts it back to a FATAL.** Adding a bound check that FATALs on an over-length chat input is a bug, not a hardening. Reject any report proposing it.

*(Added Sep 2026. Supersedes nothing: Sec. 154 and Sec. 13 continue to govern everywhere outside the
chat-path outer border.)*

### 155. SectorLogo Is the ONLY Component Allowed to Modify the Token Stream

**The token stream is sacred. Only SectorLogo sidecar scripts may modify the stream (via SetStream).**
No other component in v7 may alter, filter, strip, or suppress tokens from the stream. If the engine
produces wrong output, the bug is in the engine— fix the engine, never hack the stream.

**Hard rules:**
1. **SectorLogo is the ONLY stream modifier.** No other code may insert, delete, replace, or suppress tokens in the output stream.
2. **SectorLogo must never fix engine problems.** If the engine generates garbage, fix the engine— do not write a sidecar that patches the output.
3. **Think-tag filtering in the output is an ENGINE responsibility, NOT a stream hack.** If think tags leak into the output, the root cause is in the prefill/template/forward path— find and fix it there.
4. **Any "fix" that modifies the stream to hide an engine bug is itself a bug.** The stream must always reflect what the engine actually produced.

### 156. Disturbed Output, Root Cause Before Fix, No Hacking (BLOCKER, Aug 2026)

**Symptom:** `<|im_start|>` leaks into the output, think tags appear in the output, or the model
generates control tokens that should never be visible. The output reads as a raw token stream rather
than a clean response.

**Root cause direction:** Disturbed output (im_start leaks, stray think tags, garbage control tokens)
points towards BAD PREFILL. If the prefill puts tokens at wrong positions, omits tokens, duplicates
tokens, or misaligns the RoPE positions, the model generates chaotic output including control tokens it
would never produce with correct context. The prefill does NOT write to the output buffer— only
generated tokens do— so if `<|im_start|>` or think-open tags appear in the output, the MODEL GENERATED
THEM, which means the model saw wrong context.

**Hard rules:**
1. **Must root cause before fixing.** No hacking, no filtering, no stream modification, no post-hoc stripping of think tags or control tokens. Find WHY the model generates them.
2. **Bad prefill is the primary suspect.** Check: are all prefill tokens at the correct positions? Is the RoPE position counter correct? Does the template match the model's training format exactly? Is the start_pos correct for multi-turn? Are there missing or extra tokens in the prompt encoding?
3. **The template must match the model's training format.** If the model was trained with a template that does NOT include the opening think tag (the model generates it), but v7's template includes it, that is a prefill mismatch that causes disturbed output. If the model was trained with a template that DOES include it, v7 must include it. The template must be byte-identical to what the reference implementation produces.
4. **No stream hacks to hide this.** Adding think-tag filtering to the output path is a HACK that masks the real bug. The correct fix is to fix whatever is wrong in the prefill/template/encoding that makes the model generate control tokens.
5. **This is a BLOCKER.** No other work proceeds until the disturbed output is root-caused and the engine fix is in place.

### 157. examples/ Code Reviews— Every File Gets Reviewed, Reports Go to opt_and_bug\

**Every `.c` and `.cpp` file in `examples\` MUST be code-reviewed and have a report written to
`opt_and_bug\`.** The same standard applies as for ASM reviews (Sec.128): look for real bugs,
inefficient code, and control-flow improvements. If it is a real bug, fix it NOW— if it is not a real
bug, do not waste time on it.

**Hard rules:**
1. **Review every file.** No file in `examples\` is skipped. Each file gets a thorough review and an `opt_and_bug\<filename>.md` report.
2. **Same criteria as ASM reviews (Sec.128).** Real bugs, inefficient code sequences, control-flow straightening. No theoretical nits (Sec.57b/Sec.57c apply).
3. **If it is a real bug, fix it NOW.** No deferring to "later." If the review finds a genuine bug (buffer overflow, null dereference, wrong output, integer overflow that can actually fire, resource leak), fix it in the same session.
4. **If it is NOT a real bug, do not report it.** No busywork, no theoretical concerns, no style nits that do not affect correctness or performance.
5. **Reports follow the same format** as ASM reviews: exact file/function/line, precisely what to change, and to what.
6. **After the report is written, process it like any other opt_and_bug file** (Sec.126): apply all items, then delete the `.md`.

### 158. watch_opt_and_bug.py— NEVER Start in Its Own Window, Always With 12-Hour Timeout

**The `watch_opt_and_bug.py` watcher MUST NEVER be started in its own window.** It MUST be started
inline from the agent's bash tool with a 12-hour timeout (`43200000` ms). The watcher polls
`opt_and_bug\` for `.md` files not ending in `_revised.md`, prints urgent revision instructions, then
returns. The agent then revises the reports and starts the watcher again.

**Hard rules:**
1. **NEVER start the watcher in its own window**— no `Start-Process`, no `cmd /c start`, no separate window. Always run it inline from the bash tool.
2. **Always use a 12-hour timeout**— `43200000` ms. This is long enough to wait for new reports without timing out prematurely.
3. **When the watcher returns, revise the reports, then start it again**— this is the loop: watcher returns— revise— start watcher again.

### 159. Reviewing Agent Must Use the Watcher When No .md Files to Revise Are Left

**When a reviewing agent is asked to review `.md` files and there are no `.md` files in `opt_and_bug\`
left to revise, the agent MUST start `watch_opt_and_bug.py` (with 12-hour timeout per Sec.158) and wait
for new reports to arrive.** The watcher blocks until new unrevised `.md` files appear, then prints the
urgent revision instructions and returns control to the agent. The agent revises the reports, then
starts the watcher again— this loop continues indefinitely.

**Hard rules:**
1. **When no unrevised `.md` files remain, start the watcher.** Do not stop, do not ask, do not declare "done."
2. **After revising the reports the watcher found, start the watcher again.** The loop never ends— there is always another report to revise.
3. **The watcher is the only mechanism for detecting new reports.** Do not manually poll the directory— use the watcher.

### 160. NO Memory-Commit Logic in the Hot Path for Problems We Never Hit

**We will NOT induce strange "memory commit" logic into the hot path for problems we never hit.** No
lazy-commit, deferred-commit-extension, watermark-tracking, or any other memory/resource-management
calls (VirtualAlloc, HeapAlloc, commit-extension helpers, or wrapper calls that can reach them) may be
added to any per-token, per-layer, per-chunk, or per-element loop— not even as a "cheap no-op check
first" call— to guard against a hypothetical failure that has never actually occurred in our runs.

**Hard rules:**
1. **No commit/memory-management calls in hot loops.** The generation loop, prefill token loops, layer loops, and dispatch/worker paths must contain zero memory-management call sites— no matter how cheap the "not due yet" path claims to be.
2. **Never fix a hypothetical.** If a problem has never been observed in a real run (a report's "reachable in theory" is NOT an observation), the answer is to leave the code alone— not to add protective machinery. The crash-then-fix rule (Sec.0b) applies to crashes we actually SEE.
3. **Dead tracking state is the same bug.** Write-only limits/watermarks left behind by such designs (globals stored but never read) must be deleted on sight— they are residue of this forbidden pattern.
4. **A rejected fix must be REMOVED completely**— call sites, helper function, invariant stores, and BSS globals. Nothing stays behind "for later."

### 161. NORMAL Memory Management Only— No Reserve/Commit Split, No VirtualAlloc Clutter

**We use NORMAL memory management. One allocation = one committed, usable block. Period.** No
MEM_RESERVE-then-commit-later schemes, no deferred commit, no per-region/per-layer commit calls, no
"commit charge optimization" dances. All of this Windows commit clutter is FORBIDDEN and must be removed
wherever it exists.

**Hard rules:**
1. **`mdb_alloc_mem` is THE ONLY allocation function in the entire codebase.** Every allocation— inference, prefill, chat, generate, init steps, sectorlogo, ViT scratch, encoder dynamic buffers— goes through `mdb_alloc_mem`. `mdb_alloc_mem` internally calls VirtualAlloc— that is the ONE place VirtualAlloc exists. **Every raw `VirtualAlloc` call site outside `microdb_mem.inc` is a bug and must be switched to `mdb_alloc_mem` on sight.** The matching free is `mdb_free_mem`— raw `VirtualFree` outside `microdb_mem.inc` is the same bug. The single documented exception: the RoPE cos/sin tables stay HeapAlloc/HeapFree (a VirtualAlloc-backed table shifts process layout and trips a latent MSVC-CRT memcpy probe— see `vision_regression_log.md` Sec.8).
2. **Every allocation is a plain full commit.** There is no reserve-only mode and there never will be one.
3. **No init-step commit choreography.** Steps like "reserve the whole block, commit the working buffers, then commit N tokens per KV layer" are deleted outright— the block is allocated committed in one call, exactly like every other buffer in the codebase.
4. **If a full allocation exceeds available commit charge, that is a FATAL**— the same reaction every other allocation failure gets. Engineering around it with commit splitting is forbidden.
5. **NO dynamic memory tracking— ever.** No tracking arrays, no generic alloc registries, no descriptor schemes, no wrapper macros that register allocations. Every allocation stores its pointer in its own NAMED BSS global; cleanup frees exactly those named globals (unconditionally— see Sec.162) and zeroes them. The pointer's home is the named global— a second copy in a tracking structure is redundant ceremony and is forbidden.
6. **Cleanup runs ONLY on `v7s_inference_destroy`.** Inference memory is NEVER freed between turns— chat/generate/prefill own nothing that gets freed; all inference memory lives for the whole context lifetime.

### 162. FATAL Means FATAL— No Cleanup, No Mitigation, No NULL-Skip Frees

**The ONLY reaction to an error condition is to go FATAL— DIRECTLY.** If you detect a condition that
warrants a FATAL, you fire the FATAL and do NOTHING else. Doing ANYTHING before the FATAL— cleanup,
frees, state resets, diagnostics gathering, "tidying up"— is a DIRECT BUG. The process is about to die;
touching memory before its death is wasted code that hides the real failure.

**Hard rules:**
1. **A failed allocation goes DIRECTLY to FATAL.** No retry, no fallback, no degradation, no cleanup, no free of "partial" allocations. `mdb_alloc_mem` returned NULL— the very next instruction is the FATAL. We cannot mitigate a failing allocation— there is no discussion.
2. **Cleanup before FATAL is a BUG and must be removed on sight.** Any sequence of the form `call <cleanup>; FATAL "..."` is illegal— the call must be deleted, leaving the bare FATAL. We do NOT free memory before going fatal. Every `call inf_cleanup_alloc` (or any other cleanup call) that appears before a FATAL anywhere in the codebase must be removed.
3. **Freeing a NULL pointer means the allocation FAILED— that is a FATAL, not a silent skip.** Every "null check before free" pattern (`if (!p) skip; free(p)` / `cmp [ptr],0; je skip`) is bug-hiding: the NULL means the allocation never happened or state is corrupt. The free path must FATAL on NULL instead of silently continuing. `mdb_free_mem` itself FATALs on a NULL argument— it is the single choke point.
4. **The ONLY exception is legitimate design absence, not failure.** A buffer that is NULL because the feature legitimately does not exist for this configuration (e.g. the ViT scratch block for a text-only model) is gated by a CONFIG flag (e.g. `g_image_token_id == 0`), never by a pointer NULL-check at the free site. The free site itself is unconditional.
5. **Error paths NEVER free anything— they go FATAL directly. Cleanup in error paths is FORBIDDEN and must be removed.** If an error condition is detected after something was allocated, the FATAL kills the process and the OS reclaims all memory— freeing before death is wasted code that hides the real failure. The only free sites in the entire codebase are: (a) `v7s_inference_destroy`, and (b) transient-buffer frees on SUCCESS paths (normal control flow). There is NO "free what this error path allocated"— no error path allocates-and-then-fails; it dies instead.
6. **Error-code returns are allowed ONLY for expected runtime variation** (e.g. SectorLogo rejecting an empty input string), and such validation happens BEFORE any allocation is made. Once anything is allocated, every subsequent error is a FATAL. An allocation failure is ALWAYS a FATAL, never an error-code return.

---

### 163. SectorLogo ABI-Border Doctrine— CORRECT_RSP/RESTORE_RSP at Every Windows Border

**SectorLogo is one large function with its own ABI space.** Its internal Lisp-evaluator helpers
(Car/Cdr/Cons/Intern/EvalNextArg/EvalLogoStream...) keep their own stack discipline— single-push
temporaries (`push rax` for a body block, `push r15` for a var name) are legal inside the evaluator and
RSP parity there is NOT guaranteed. **We do NOT force the evaluator internals to be
Win64-ABI-conformant.** But at every BORDER where control calls OUT of the Logo world into ABI-boundary
code (Win32 APIs, v7 exports that spawn processes or write files), the stack MUST be correct— a
misaligned entry crashes inside the callee with an SSE #GP (`movaps` on stack slots), reported by
Windows as 0xC0000005 with fault address -1.

**The fix pattern (implemented, verified): `CORRECT_RSP` / `RESTORE_RSP` macros
(`src/asm/v7_apicall.inc`) placed directly BEFORE and AFTER the outbound call— NOT at function
entry/exit.** The macros are a parity probe with the correction state parked ON THE STACK (reentrant, no
BSS, no register clobber— rax survives):

```nasm
CORRECT_RSP:  test rsp, 8 / jnz bad / push 0 / push 0 / jmp done / bad: push 1 / done:
RESTORE_RSP:  cmp qword [rsp], 0 / jne one / add rsp, 16 / one: add rsp, 8
```

**Placement warning (stack args!):** when the call passes arguments via the stack (e.g.
`lg_invoke_exe_fn`'s argv array at `[rsp + off]`), LOAD those args BEFORE the CORRECT_RSP pushes— the
pushes shift RSP and would otherwise read the wrong slots. CORRECT_RSP goes immediately before the
`call`, RESTORE_RSP immediately after.

**Border inventory (all verified safe as of commit `166edf8`):**
- `lg_invoke_exe_fn`— `v7_execute_fmt` (spawn path)— CORRECT_RSP applied
- `PutChar` / `PrintRaw`— GetStdHandle + WriteFile— CORRECT_RSP applied
- `FATAL` macro— `v7s_fatal_error`— parity-proofed inside the macro (fixes every evaluator FATAL site at once)
- `sl_ffi_invoke` (CallLibrary)— self-aligns via `and rsp, -16` (safe by design)
- `sl_ffi_resolve` (LoadLibraryA/GetProcAddress)— only called from self-aligned `sl_ffi_invoke` (safe)
- `v7s_readfile` from `lg_init`— one-time init runs on the aligned C-entry path (safe)
- `TRACE_*` macros— self-align (safe at any parity)
- Evaluator-internal helpers (Car/Cdr/Cons/Intern/ht_find/encode/NumVal/...)— make zero Windows calls (safe)

**Hard rules:**
1. **Every new Windows-call site reachable from SectorLogo evaluation gets CORRECT_RSP/RESTORE_RSP around the call.** No exceptions.
2. **Never "fix" the evaluator's internal stack discipline instead**— the border is the fix point, not the internals.
3. **CORRECT/RESTORE must be paired on every path**— if control flow exits between them (error branches), the correction must be undone on that path too.
4. **Stack-passed args are loaded BEFORE the CORRECT_RSP pushes.**

---

### 167. CONFIG DEGENERATION POLICY— Apply, Degrade Nearest-Legal, Flag It, Never Silently Clamp (Sep 2026)

**Dispatch/tuning config values (threading_proxy.ini— per-shape `[NxD]` blocks) exist to TUNE the engine
to the system. They must be APPLIED as configured whenever the running system can honor them.** When the
system CANNOT honor a value (e.g. taper `CAP = AlignDown(D/(2P)) < W_min` at this worker count/shape, or
`num_chunks > D/min_chunk_cols`), the engine DEGENERATES to the nearest sane, fast, legal setting— as
close as possible to what was configured— and records the event.

**Hard rules:**
1. **APPLY first.** A legal config value is used verbatim. The engine never second-guesses values it can honor. These values are pre-tuned (autotuner) and local-tuned (INI)— applying them IS the point.
2. **Degrade nearest-legal when the system can't honor a value.** The replacement value must be (a) legal for THIS system/shape, (b) as close as possible to the configured intent, (c) fast (a degraded value that is legal but slow is a bad fix). Example: taper CAP < W_min— fall back to uniform fill for that dispatch (the nearest legal chunking at k=0), NOT an arbitrary CAP raise that changes job geometry.
3. **Every engine-side value fix sets a distinct bit in the u64 BSS global `config_was_fixed`.** Each fix site has its own bit. The export `get_config_was_fixed()` returns the u64 so tools (autotuner, validators, loadtests) can see exactly WHICH values were fixed vs applied as-is. The autotuner uses this to know when a candidate it tried was invalid on this system.
4. **THE IDEAL IS ZERO FIX SITES.** The best config values work for ALL systems within a defined range that stays fixed. Fewer degeneration sites = a better config design. When a fix site is added, ask: could the config format express this without needing a fix (e.g. ratio-based instead of absolute)? Prefer config-independent formulations.
5. **Values must have MEANINGFUL IMPACT.** A knob that changes nothing measurable is dead weight and gets removed; a knob the system can never honor is misdesigned and gets reformulated.
6. **Behaviour OUTSIDE the defined ranges is UNDEFINED— and for now it FATALs.** (min_chunk_cols <= 0 or a non-power-of-two value, taper_shift > 112, num_chunks > MAX_CHUNKS, chunk_count == 0, workers+1 > total_cores, cores_subtract >= total_cores, num_workers > MAX_WORKERS, and the eight claim sizes batch_small/medium/large/huge + caller_batch_small/medium/large/huge when outside [1, 64].) Keep these FATALs until ALL tools respect the ranges; only values INSIDE the ranges may degenerate per rule 2. **A zero claim size is a rule-6 FATAL, never a rule-2 degradation and never a clamp:** `lock xadd dword [next_job], 0` advances the shared cursor by nothing, so BOTH steal loops re-claim the same index forever and the dispatch wedges silently at 100% CPU on every core. It is **FATALed at CREATION, not at runtime**— `db_build_tool.c` `ini_to_db` rejects all eight keys outside `[1, 64]` in every INI section, and `tune_api.inc` `.sp_guard_batch` does the same when arming an autotuner candidate. Every write path is therefore covered and **no engine-side runtime re-check may be added**— it would be permanently dead code, which Sec. 186 forbids. A claim size is not a system-capability limit like the taper CAP, so rule 2's nearest-legal degradation does not apply to it either; Sec. 13 forbids answering a bad value with a default.
7. **Logical, minimal intervention.** Fix values only when we have to. The goal is always: best performance.
8. **Every degeneration is documented in `fatal_clamping_and_bugs_changes.md`** (docs\): which FATAL fired, root cause, the nearest-legal value chosen, the `config_was_fixed` bit assigned, and the config/INI follow-up so the fix site eventually becomes unnecessary.

---

### 166. CHANGELOG Lives in DEV— Maintained Continuously, Copied to Release Verbatim

**The changelog process has changed: `CHANGELOG.md` is maintained IN DEV, continuously, as work happens.
The release no longer analyzes the release git diff to write its own changelog— that analysis became too
hard and error-prone, and it duplicated work dev already does. The release changelog is a VERBATIM COPY
of dev's.**

**Hard rules:**
1. **Dev owns the changelog.** Every meaningful change (feature, bug fix, optimization, doc change, removed file) gets its bullet(s) in dev's `CHANGELOG.md` under the top `## [Unreleased]` section— written in the SAME session the change lands, never retroactively, never from memory at release time.
2. **`sync_release.bat` copies `CHANGELOG.md` from dev verbatim.** The release agent has ZERO editorial discretion over changelog content.
3. **The ONLY release-time changelog action is retitling:** the top `## [Unreleased]— date` heading becomes `## [X.Y.Z]— date`. Content stays untouched— no rewriting, no re-analysis, no diff study.
4. **After a release, dev starts a fresh `## [Unreleased]` section** on top for the next line of work.
5. **Never write changelog content in the release directory.** The old process (git diff analysis in the release) is RETIRED.
6. **Style rules still apply at dev writing time:** factual, technical, restrained language (Sec.86 rule 2— no bragging); every claim must match the actual source code.

---

### 171. THE APPLYING AGENT RUNS watch_apply_opt_and_bug.py— The Apply-Loop Is Mandatory and Never Ends

**There is exactly ONE applying agent, ONE revising agent, and MANY reviewers.** The reviewers write .md
reports into opt_and_bug\; the revising agent consolidates them into *_revised.md; **the applying agent
consumes the _revised.md reports**— this section defines the applying agent's loop.

**The applying agent's driver is watch_apply_opt_and_bug.py (project root).** Run it with a 12-hour
timeout (43200000 ms), inline from the bash tool— NEVER in its own window.

**Hard rules:**
1. **The applying agent runs python scripts\watch_apply_opt_and_bug.py whenever it has no pending _revised.md work— and AGAIN the moment its current work is done.** The script polls opt_and_bug\ every 60 seconds; when one or more *_revised.md files exist, it prints their filenames plus the processing instructions and returns control. The agent then processes them.
2. **The script NEVER deletes reports.** It only prints filenames + instructions. **A report .md MUST ALWAYS be deleted DIRECTLY and IMMEDIATELY when it is consumed**— the moment all of its items are applied (or rejected with stated proof), the agent deletes that one file in the same session, BEFORE moving on to the next report. Do NOT hold deletions until the end of a batch, do NOT wait for the build/verify/commit cycle: the file's existence is the synchronization signal— a still-present .md gets reviewed again, producing duplicate work. **BULK DELETION OF .md FILES IS ABSOLUTELY FORBIDDEN because it causes mistakes**— never `del a.md b.md c.md`, never `del *_revised.md`, never wildcards, never multi-file del commands. Exactly ONE `del` command per file, issued immediately after that report is consumed. Deletion is the completion criterion (Sec.126 rule 2).
3. **When the script returns, the agent processes ALL printed files, without exception:** read each report in full, apply every optimization (every cpu cycle counts— none may be skipped), fix every real bug (no exceptions, no deferrals), reject only items proven unsafe or impossible (Sec.129a danger-area optimizations, Sec.124 impossible states) with the proof stated.
4. **Applied reports may NEVER hurt performance, NEVER reduce numerical precision (Sec.126 rule 9), and NEVER induce bugs.** Verify equivalence before editing; after applying, the full verification cycle is mandatory: `build.bat` (zero comment-checker violations), followed by `verify.bat` on its own, then vision smoke when ViT/vision files were touched— commit (specific files only).
5. **After committing, the agent starts the script again** to check for new work (consumed reports were already deleted one by one, immediately, per rule 2). The loop never ends— there is always another report to apply.
6. **The revising agent keeps using watch_opt_and_bug.py** (the unrevised-report watcher, Sec.158/Sec.159); the two watchers are complementary and must never be conflated.

---

### 172. PUBLIC v7_* TIER + ENGINE PSEUDO-EXCEPTION— The C API Contract (Sep 2026)

**The public C API is the `v7_*` tier in `src/asm/public_api.inc` (single collection point)— the
inference context lives in a BSS singleton inside the DLL and is NEVER handed out.** All C hosts (tools,
examples, server) call ONLY this tier; the internal `v7s_*` inference entries are for DLL-internal calls
and the documented deep tools (`v7_get_internal_inference_ctx`).

**The public lifecycle (inference_example.c is the reference):**
```c
v7_load(dll_path, &dll);
dll.v7_init(db_path);              /* 1 on success, 0 on failure */
dll.v7_initialize_inference();     /* 1 = ready (idempotent); failures FATAL internally */
dll.v7_chat(user, sys, out, out_len, reset);   /* reset=1 fresh, 0 multi-turn */
dll.v7_generate(prompt, out, out_len, raw_generation);  /* raw=1: no template, EOS-only stop */
dll.v7_free_inference();           /* parameterless */
v7_unload(&dll);
```

**Pseudo-exception contract (`attic\docs_plans\fatal_to_pseudo_exception_plan.md`, implemented in
`src/asm/engine_landing.inc`):** every gated public entry (`v7_init`, `v7_initialize_inference`,
`v7_chat`, `v7_generate`, `v7_chat_image`, `v7_generate_image`) installs a `.bss` landing zone (snapshot
of the C caller's RSP/RBP + rbx/rsi/rdi/r12-r15 + xmm6-15 lows). An internal FATAL anywhere inside that
call reroutes (one-line FATAL-macro swap in `v7_core.inc`) to `handle_fatal`: prints the reason
UNCHANGED, records it, parks the engine (`v7_engine_dead`), quiesces the worker pool, and unwinds to the
C caller with the full callee-saved register file restored. **A gated death-return is always 0/void—
pair EVERY return check with `v7_engine_is_dead()`; the reason is queryable via `v7_get_death_reason()`
until the next `v7_free_inference` (which resets the death state).** FATALs on worker threads or outside
gated calls keep the OLD terminal print+TerminateProcess path— uncatchable, by user directive.

**Hard rules:**
1. New public entries go in `public_api.inc` ONLY; add gates (ENGINE_GATE_RAW for hand-labelled raw entries, ENGINE_GATE_SFR for SF-framed wrappers that save no non-volatiles) + ENGINE_LEAVE.
2. The zone saves the FULL callee-saved file— the unwind skips every internal SF_END, so the zone is the ONLY restorer. Never trim the save set.
3. `v7_free_inference` / `v7_shutdown` are NEVER dead-gated (post-death cleanup must run).
4. `engine_death.exe` runs in verify.bat Phase 5— a regression in the catch machinery fails the gate.
5. This is NOT SEH (Sec.59), NOT a bug-hider (Sec.60/Sec.155): the FATAL still prints; the host is INFORMED and tears down. Root-cause fixes remain mandatory (Sec.0b/Sec.127).

---

### 169. RESOLVED— SectorLogo Better Tokenizer Overhaul COMPLETE (Sep 2026)

**The SectorLogo better-tokenizer overhaul (`sectorlogo_better_tokenizer_plan.md`) is COMPLETE and
committed (c2ba8d6 + cd69c5b).** All three core promises delivered:

1. **Natural tokenization**— operators (`+`, `-`, `*`, `/`, `=`, `!=`, `<`, `>`, `<=`, `>=`, `&`, `|`, `^`, `<<`, `>>`) and brackets (`[`, `]`, `(`, `)`) always split tokens. Unary minus in list literals (`[-3 5 -10 0]`) and spaced prefix args (`FloatRemainder 7.0 -3.0`) are handled correctly via `sl_list_depth` and `sl_tok_leading_ws`.
2. **Case insensitivity**— the lexer folds A-Z to a-z (case-insensitive keywords/identifiers).
3. **Friendly whole-word PascalCase names**— NO aliases, NO backward compat per Sec.85.

**Validator: 732 passed, 0 failed. verify.bat: 19 validators + 5 inference smokes green.**

**Root cause of the original "got NIL" bug (unary-minus mis-tokenizer):**

**RESOLVED. The entire "state accumulation / GC corruption / heap corruption" theory was WRONG.** The
bug was a plain tokenizer defect introduced by the Part-A tokenizer rework, made to LOOK state-dependent
by stdout buffering (the validator's printf test headers sat unflushed in the CRT buffer while the DLL's
WriteFile diagnostics flushed immediately, so the FATAL appeared to fire "right after autoexec" when it
actually fired much later in a completely different test suite).

**Root cause (GetToken unary -/+ gluing, sectorlogo_reader.inc):** The rework decided unary-vs-binary
minus purely from `sl_last_tok_operand` (prev token was an operand => binary). Two legitimate contexts
violate that rule:

1. **List literals:** `[-3 5 -10 0]`, every list element is standalone; there is NO binary-minus context inside `[...]`. `-10` after the operand `5` was tokenized as `-` `10`, so Sort received the `-` OPERATOR ATOM as a list element -> "sort/LFIND key must be numeric (got non-numeric atom)". (Earlier reports said "got NIL" because the old diagnostic tested eax AFTER the ATOM-parse path had clobbered it to 0, see the diagnostic lesson below.)
2. **Spaced prefix arguments:** `Output FloatRemainder 7.0 -3.0`, `-3.0` is the second ARGUMENT of a prefix call, not infix subtraction. Tokenized as `-` `3.0` -> the `-` operator consumed `7.0` as its first arg -> NumVal(FLOAT) FATAL "expected NUMBER".

**Fix (sectorlogo_reader.inc, GetToken):** Glue `-`/`+` onto a following number token iff (a) a digit
immediately follows the sign, AND (b) ANY of: inside a list literal (`sl_list_depth > 0`, tracked by
`[`/`]` via `.delim_list_open`/`.delim_list_close`), OR the previous token was not an operand
(function-call head: `Maximum -5 10`), OR the sign-led token begins after whitespace
(`sl_tok_leading_ws`, set in `.skip_ws`, this is the spaced prefix-argument case). ONLY the compact
operand-adjacent form (`2-3`, `2+3`) stays a binary operator. This preserves infix `( 255 - 205 * :frac
)` (space after the sign => bare operator) and compact infix `2+3` (no space before the sign after an
operand => bare operator).

**Why `-startfrom 68` "passed":** it skips the sort suite entirely, nothing state-dependent at all.
`run_sectorlogo.exe` passed because the single test script contained no negative list literals.
Binary-search conclusions drawn from buffered-mixed-with-unbuffered output were garbage.

**Lessons:**
1. **Never trust output ordering between buffered printf (CRT) and unbuffered WriteFile (DLL console output).** When a FATAL kills the process, CRT-buffered test headers vanish while WriteFile prints survive, the FATAL then appears attached to the WRONG test. Fix: `fflush(stdout)` in the validator after each header, or trace INSIDE the DLL (TRACE macros go through WriteFile).
2. **A diagnostic that reports the wrong classification is worse than none:** `sl_sort_coerce_num`'s old `.scn_not_num` path tag-tested eax AFTER the ATOM parser had clobbered it with the digit accumulator (0 for no digits), a garbage ATOM was misreported as "got NIL", sending the whole investigation after phantom NIL/GC theories. Fixed: the ATOM-parse-failure path now unwinds its own frame and FATALs with "got non-numeric atom" WITHOUT tag-testing the clobbered register; NIL/CONS/unknown-tag get their own accurate FATALs on the fast path where eax is the true input.
3. **"Non-fatal return" diagnostics corrupt the stack:** the temp NUMBER(0)-return path on the ATOM branch returned through `.scn_done: ret` WITHOUT unwinding `push rbx/push rsi/sub rsp,40`, ret popped a wrong address, producing the RIP=0x2 cascade crash. FATALs must FATAL; a diagnostic return must unwind its own frame first.
4. **TRACE_ZSTR on Intern/GetToken input is the fastest tokenizer-truth probe:** interning traces (`intern_in=[-3]` / `intern_in=[-]` / `intern_in=[10]`) proved in one run which TOKENS the reader actually produced, instantly separating tokenizer bugs from evaluator/Intern/GC theories.

**Also fixed during the hunt:** two uncommitted Part-E validator tests expected digit-suffixed names and
colon-containing tokens to split into separate forms. This contradicted the documented name rules: names
such as `test2` remain whole, and the colon does not force a split. Corrected to test the documented
behavior (digit-suffixed names stay whole; colon vars need a space after a name).

**Validator status after fix: 732 passed, 0 failed.**

---

### 170. SECTORLOGO BUG FIXES REQUIRE VALIDATOR COVERAGE— Every Fix Ships With a Test

**When a SectorLogo bug is found and fixed, a test covering that bug MUST be added to the SectorLogo
validator (`tools/v_sectorlogo_validate.cpp`) in the SAME session as the fix.** No exceptions, no
deferrals. The regression test is part of the fix, not an optional follow-up.

**Hard rules:**
1. **Every SectorLogo bug fix adds a validator test.** The test must exercise the exact failure mode that was observed (same script construct, same input shape, same call chain) and assert the corrected behavior.
2. **The test lands in the same commit as the fix.** A fix without its regression test is an incomplete change— the bug can silently recur with nothing to catch it.
3. **The validator must stay green.** After adding the test, run the validator (via `verify.bat` or the sectorlogo validator target) and confirm pass counts increase with zero failures.
4. **This applies to ALL SectorLogo components:** reader/tokenizer, evaluator, Lisp core (Intern/atoms/lists), FFI (CallLibrary), DefineCommand (`lg_invoke_exe_fn` + `v7_execute_fmt`), v7 host primitives (GetStream/SetStream), string primitives, console primitives, sort/search primitives, stdlib (autoexec.logo) functions.

---

### 173. SECTORLOGO FATALS AND ERROR TEXTS ARE IMPROVED WHENEVER WE DEBUG AND SPOT STUFF

**When working on SectorLogo— debugging, fixing bugs, or spotting problems— the FATAL messages and error
texts MUST be improved in the same session.** Better reactions to problems and better diagnosis texts
are part of the work, not an optional follow-up. The language must TELL the script author what went
wrong, with the offending value and context, instead of failing silently or printing a cryptic
one-liner.

**Hard rules:**
1. **We do this when we debug and spot stuff.** Every SectorLogo debugging session that uncovers a confusing failure, a silent NIL, or a cryptic FATAL improves that error text (and the surrounding ones of the same class) before the session ends.
2. **Error texts name the offender.** A FATAL prints the actual offending value (atom text, number, list) and its type, plus the operation that rejected it— e.g. `FIRST requires a list, got word 'second'`, not `sort/LFIND key must be numeric`.
3. **Silent NIL returns from type mismatches become FATALs.** When a primitive receives an argument type it cannot operate on (word given to a list-only primitive, list given to a string primitive), silently returning NIL hides script bugs— the primitive FATALs with the offending value instead. Only legitimate empty results (FIRST of `[]`, ButFirst of a one-element list) return NIL.
4. **Error texts stay calm and mixed-case** (Sec.6a rule 37): lowercase, no exclamation marks, state fact + offender + expectation.
5. **Every improved FATAL gets validator coverage** (Sec.170 applies): a test that triggers it and asserts the message/abort.

---

### 173a. WORKER SHUTDOWN IS OBSERVED, NEVER GUESSED— workers_alive Counter (Oct 2026)

**Worker-pool shutdown must OBSERVE worker termination, never guess it with a fixed sleep.** The old
design armed `shutdown_flag`, woke workers, closed the thread handles, and relied on a blind
`Sleep(1000)` in `DllMain(DETACH)` before FreeLibrary unmapped the DLL. A worker that missed the 1s
window kept executing DLL code after the unmap— execute-AV at a DLL address with the whole callee-saved
file zeroed, caught only when a verify run happened to schedule it (the crash printed mid-verify with
the validator still PASSing on its exit code— a crash the summary hid).

**Waiting on thread handles inside DllMain is impossible**: a worker's `RtlExitUserThread` must acquire
the loader lock that DllMain itself holds— a join there deadlocks. The legal mechanism is a polled BSS
counter, because the counter's decrement is the worker's LAST DLL action, executed BEFORE the thread
touches the loader lock.

**The mechanism (implemented, commit `48c6b81f`):**
- `workers_alive` BSS dword: `lock inc` at `worker_proc` entry, `lock dec` at `.worker_exit` (before vzeroupper/SF_END)— after the dec the thread runs only ntdll exit code.
- `v7s_shutdown`: after arming the flag + 2x wake, polls `workers_alive` to 0 with `Sleep(1)` (Sleep+poll holds no lock— deadlock-free even under the loader lock). 2s budget, then proceeds exactly as the old blind sleep did. Handle closing happens AFTER observed termination.
- `initialize_once`: after the CreateThread loop, polls `workers_alive` up to the created count— closes the not-yet-started hole (a late-starting worker would otherwise run its first instructions after an early unmap).
- `DllMain(DETACH, lpvReserved == NULL)`: calls `v7s_shutdown` and NO LONGER sleeps.

**Hard rules:**
1. **Never reintroduce a fixed "should be enough" sleep for worker termination.** A guessed window is a latent crash that fires under scheduler pressure.
2. **Never wait on worker thread handles inside DllMain**— loader-lock deadlock. The only legal wait under the loader lock is the `workers_alive` Sleep-poll.
3. **The `lock dec` in `.worker_exit` must stay the worker's LAST DLL action.** Moving any DLL-code step after it breaks the counter's guarantee ("zero = no DLL code will ever run again").
4. **Every worker start must pair exactly one inc with one dec.** `initialize_once`'s startup poll (workers_alive == created count) is what makes later shutdown polls sound— do not remove it.

---


### 174. IMUL 3-OPERAND ACCUMULATOR CLOBBER + BAT CRASH-EXIT FALSE PASS (Sep 2026)

**`imul dst, src, imm` REPLACES dst with `src*imm`— it does NOT accumulate into dst.** When a register
already holds a factor (an accumulator) and must be multiplied by `src*imm`, the 3-operand form silently
DESTROYS the accumulator. This class produced a hard OOB-write crash in `kv_cache_file.inc` (Sep 2026):
`mov r11d, n_layers` followed by `imul r11, rax, 2` computed `r11 = layer_bytes*2` instead of
`n_layers*layer_bytes*2`— the save path allocated a ~1.4MB buffer for a ~40MB gather loop and faulted on
the 5th layer. The load path's header-validation had the identical defect.

**Hard rules:**
1. **Multi-factor products use 2-operand chains:** `imul r11, rax` (r11 = r11*rax) then `add r11, r11` or `imul r11, r11, 2` for a constant scale. The 2-operand reg,reg form is also the Sec.120-safe legacy encoding.
2. **The 3-operand form is ONLY legal when dst is NOT carrying a needed value**— i.e. `dst = src * imm` with src a different register holding the complete first factor (audit every `imul reg, reg, imm` for this; the crash sites had a comment claiming a 3-term product while the instruction computed a 2-term one).
3. **Size-computation bugs are crashes regardless of when they fire**— compute, then check the allocation against the loop's total write extent.
4. **cmd `if errorlevel 1` is a SIGNED check**— a process killed by an AV exits with 0xC0000005 = -1073741819, which is NOT >= 1, so the FAIL branch is skipped and a crashed run reports "[ok]". Every .bat that gates on success MUST use `if %errorlevel% NEQ 0` (compares the full signed value)— `if errorlevel 1` alone produces false passes on crashes.
5. **A FATAL label placed after a loop exit is a fall-through bug** (found the hard way, Sep 2026: the fused-QKV guard's FATAL sat directly after `jnz .qkv_loop`, so EVERY normal loop completion fell straight into it and killed init on all models— the guard itself was fine, TRACE_REG proved all four sizes aligned). Rule: any out-of-line FATAL block reachable only by jumps MUST be preceded by an explicit `jmp` barrier on every fall-through path (`jnz loop / jmp .done / .fatal: FATAL / .done:`), or placed where the preceding block ends in an unconditional transfer. Never assume a label's position is unreachable— verify what falls into it.

---

### 175. ALL FILES ARE YOUR RESPONSIBILITY— ALWAYS, UNLESS EXPLICITLY TOLD OTHERWISE

**IT IS ALWAYS THIS WAY THAT IT IS YOUR RESPONSIBILITY AS LONG AS YOU ARE NOT TOLD OTHERWISE.** Every
file in the project— every `.inc`, `.asm`, `.c`, `.cpp`, `.h`, `.md`, `.bat`, `.logo`, script, tool,
doc, and artifact— is YOUR responsibility in every session. There is no "other agent's files," no "not
my area," no "someone else is handling that"— UNLESS the user has EXPLICITLY told you, in the current
session, that specific files are off-limits.

**Hard rules:**
1. **Default: every file is yours.** Modified tracked files, untracked artifacts, files touched by other agents, files you have never opened— all of it is your responsibility by default.
2. **An exclusion exists ONLY when the user explicitly grants it.** "Other agent's files (DO NOT touch/commit): ..." is valid ONLY as a direct, current instruction from the user. It never carries over between sessions, and it never spreads to other files by inference.
3. **Never skip work because a file "belongs" to someone else.** Bug reports, reviews, and todo items targeting any file are processed by you unless the user just now said not to.
4. **Uncommitted changes in ANY file are accounted for by you at commit time** (per Sec.51): reviewed, included, or explicitly reverted— never silently left behind or assumed to be someone else's problem.

---

### 176. LOGO-SCRIPT DEFINE COMMENTS ARE NOT LOW PRIORITY— MANDATORY, HIGH PRIORITY

**Comments on `Define` blocks in SectorLogo `.logo` scripts are NOT low priority. They are mandatory,
high-priority work with the same standing as ASM comment rules (Sec.9).** Every function definition in a
`.logo` script— sidecars, stdlib, validators' test scripts, research scripts— must carry a hand-written
comment stating what the function does, its parameters, and its return value BEFORE the definition.

**Hard rules:**
1. **Every `Define "name"` gets a comment.** Written by hand (Sec.58), placed on the line(s) before the definition, stating: purpose, parameter meanings, return value. Example: `; replace_all text old new, replace every occurrence of old in text with new; loops Search+Replace until not found`.
2. **This is high-priority work, not cleanup.** It is never deferred behind "real" work, never filed as low-priority backlog, and never skipped because "it's just a script." A Define without a comment is a bug (Sec.9 class)— fix it the moment you see it.
3. **Applies to ALL .logo files:** `src/logo/` production sidecars, `autoexec.logo` stdlib, `examples/` research scripts, validator test scripts appended to sidecars, and any new script written anywhere.
4. **When writing a NEW Logo script, the comments are written WITH the Defines— not "added later."** A new script with uncommented Defines does not get committed.
5. **When editing an existing script, upgrade weak/missing Define comments in the same session**— same as Sec.9's "improve comments proactively" for ASM.

---

### 177. SECTORLOGO BUGS ARE FIXED IN SECTORLOGO— NEVER WORKED AROUND

**If there is a bug in SectorLogo, it MUST be fixed in SectorLogo (the engine: `sectorlogo_*.inc`) and
NEVER worked around.** No compensation anywhere else— not in sidecar scripts, not in validator tests,
not in C harnesses, not in docs, not in the caller. A workaround hides the engine defect and lets it
compound; the fix belongs in the engine source.

**Hard rules:**
1. **A SectorLogo engine bug is fixed in `src/asm/sectorlogo_*.inc`— nowhere else.** Adjusting a test, sidecar, or caller so an engine defect "passes" or "doesn't trigger" is a workaround and is FORBIDDEN.
2. **The ONLY legitimate test-side change is correcting a test that was itself wrong** (e.g. it used a dead name, asserted the wrong value, or tested behavior contrary to documented semantics). The engine behavior must be PROVEN correct against the language spec— then and only then may the test be aligned. If in doubt, prove the engine wrong or right with a probe first.
3. **Never paper over silent misbehavior.** A dead name silently self-evaluating, a swallowed flag, a stale return— each is either (a) documented language semantics (then the docs/tests must match the documented behavior, and Sec.173 may demand a clearer FATAL instead of silence where the spec disagrees), or (b) an engine bug to fix NOW.
4. **Unexplained intermittent failures are engine bugs until proven otherwise** (state accumulation, death-state leakage, atom-table drift). "It only fails in the full run, not standalone" is a debugging clue, NOT a reason to ignore it (Sec.0b, Sec.66).
5. **This mirrors Sec.60 (C never compensates for ASM) and Sec.156 (root cause before fix) inside the SectorLogo subsystem.**

---

### 178. REMOVED, Q8 Cross-Process Nondeterminism (RESOLVED Sep 2026)

**RESOLVED and section body removed by user directive.** The root cause was the dispatch steal-loop
generation race: a worker's `lock xadd` claim could straddle a dispatch publish, get bounded by the
post-publish `num_jobs`, process the NEXT dispatch's jobs, and decrement `remaining_jobs` twice, the
counter went negative, the caller exited the wait while a worker still ran, and the resulting buffer
races produced per-process-varying Q8 output. Fixed with the generation-tagged job cursor (tag in bits
31:16 of `next_job`/`num_jobs`; claims classified by returned tag; batches clamped at the generation
boundary) in `dispatch_execute.inc` + `v7_worker.inc`. verify.bat all green (v_generate Q8 "paris"
passes, no negative-remaining FATAL). Historical evidence: `bug_log.md` entry "2026-09-17".

---

### 179. NO NEW TOOLS IN tools/ WITHOUT USER CONSENT

**The agent is NOT to add tools to `tools/` without user consent.** Creating any new tool— validator,
benchmark, stress harness, diagnostic or debug program, any new `.c`/`.cpp` source or built `.exe`—
inside `tools/` requires explicit user approval BEFORE it is created. This includes adding new entries
to `build.bat` tool lists. The same consent requirement applies to `examples/`. The authoritative
problem record for the current dispatch investigation is `dispatcher_problem.md` (docs\).


---

### 180. DETERMINISTIC PATHS MUST BE 100% DETERMINISTIC, EVERY DEVIATION IS A CRITICAL BUG

**When v7 runs on a deterministic path (greedy/temperature=0, same binary, same database, same prompt),
the output MUST be bit-identical across EVERY process, EVERY run, EVERY machine with the same CPU
features.** There are ZERO acceptable deviations. Floating-point arithmetic on x86 is fully
deterministic, same inputs + same operations = same output, always, no exceptions. If the same binary +
DB + prompt produces different output in different processes, that is a CRITICAL BUG that must be fixed
immediately, not explained away as "near-tie" or "precision" or "expected float variation."

**Hard rules:**
1. **Greedy inference is a pure function of (binary, DB, prompt, CPU features).** Any deviation between processes is a bug, period. "Near-tie" does NOT excuse a flip. If the inputs are truly identical, the output MUST be identical. If they are not identical, the inputs were not truly identical (there is a per-process-varying input somewhere in our code).
2. **"Floating-point nondeterminism" does not exist on x86.** The hardware is deterministic. Same registers + same operations + same memory = same result, always. Blaming "float nondeterminism" is blaming the hardware for our bug. The bug is ALWAYS in our code, uninitialized reads, init-order-dependent state, per-process-varying memory content, or arithmetic that genuinely depends on per-process-varying inputs.
3. **Every cross-process divergence MUST be root-caused and fixed.** No deferral, no "known limitation" classification, no tolerance relaxation. The fix is to find the per-process-varying input and eliminate it or make it constant.
4. **This applies to ALL deterministic paths**, not just greedy inference: same binary + same inputs must always produce same outputs, whether Q8 or F32, whether matmul or attention or quantize or any other kernel.
5. **This rule has ZERO exceptions and ZERO tolerance.** A single divergent run out of 1000 is still a critical bug.

### 181. YOU NEVER STOP WORKING. PERIOD. NOT EVER. NOT FOR ANY REASON. (Sep 2026)

**We work in GIANT sessions. A task in this project routinely spans more than TEN compactions before it
is done, and the session limit is 100% irrelevant to whether the work gets finished. There is no context
size, no task size, no perceived difficulty, and no momentary uncertainty that is ever a valid reason to
stop while work remains. Stopping is a FAILURE, every time, without exception.**

**This rule has ZERO exceptions. There is no "I'm nearly out of context" escape hatch, and there is no
"this is getting long" signal to honour, because no such signal exists.**

**The absolute rules, each is UNBREAKABLE:**

1. **YOU NEVER STOP WHILE WORK REMAINS. NOT ONCE. NOT EVER.** The moment
   you begin composing a wrap-up while tasks are still open, you have
   already failed. A context window is not a wall, it is a sliding
   door. Whatever you were doing, you keep doing it. There is no
   threshold at which "I should probably wrap up" becomes correct. That
   threshold does not exist.

2. **A CLOSING SUMMARY IN PLACE OF FINISHING IS A FAILURE.** "Here is
   where I got to, the rest is open" is a FAILURE unless the user
   explicitly asked for a status report, and they did not. If the queue
   has 38 files in it, the ONLY correct behaviour is to work through
   them. Enumerating them and handing back a to-do list is the exact
   failure mode this rule exists to prevent. Writing a status update
   while work is open is the same failure in a different costume.

3. **NEVER, UNDER ANY CIRCUMSTANCE, ASK PERMISSION TO KEEP GOING.** "Should
   I keep going or start a new session?" is not a question the user
   wants answered. They want the work DONE. The answer is not "let me
   know", the answer is to simply continue. Asking is stalling, and
   stalling is this rule's opposite.

4. **AN UNFIXED BUG IS NEVER AN ACCEPTABLE ENDING. EVER.** Sec. 0b makes a
   crash a blocker and Sec. 102 makes "preexisting" a non-excuse. Running
   low on context does NOT downgrade a Sec. 0b crash to "documented and
   handed over". You either fix it or you keep digging until you do.
   A crash you describe accurately but do not fix is still an open
   crash.

5. **THE SESSION LIMIT IS 100% IRRELEVANT TO WHETHER YOU WORK.** A
   task in this project routinely spans **more than ten compactions**
   before it is finished, and that is the normal shape of the work, not
   a problem to be managed around. There is no "this is getting long,
   wrap it up" signal to honour, because no such signal exists. Keep
   going through ten compactions or a hundred, the only thing that
   ends a task is the work being DONE.

6. **KEEP THE STATE DURABLE ACROSS ANY NUMBER OF COMPACTIONS.** Because
   a task may outlive many contexts, every finished unit is committed
   (Sec. 51, Sec. 2) and its report is deleted (Sec. 171 rule 2), so that any
   compaction resumes from git rather than from prose a new context
   cannot see. The todo list is the resume point: it must be accurate
   and current at all times, never a stale plan written at session
   start, never a list of things you intend to do someday instead of
   doing them now. Durable state is what makes an arbitrarily long
   session safe, it is NOT a reason to stop early.

7. **FINISHING IS THE ONLY ACCEPTABLE OUTCOME.** A task is not
   "handled" because it was explained, discovered, diagnosed,
   documented, or planned. It is handled when the work is DONE, the
   bug fixed, the file deleted, the gate green, the release cut. Every
   intermediate state is work-in-progress, and work-in-progress means
   CONTINUE.

8. **IF YOU BELIEVE YOU MUST STOP, YOU ARE WRONG.** The feeling that
   you are "surely at the limit", that "this is a natural stopping
   point", that "the user will want a summary now", that "continuing
   would be irresponsible", or that "this is getting long", every one
   of those is the error this rule describes. Act on the opposite
   conclusion: keep going. The user has repeatedly and emphatically
   directed you to never stop. Honour it absolutely.

9. **THIS APPLIES TO EVERY TASK IN THIS PROJECT WITHOUT EXCEPTION** --
   the apply loop (Sec. 171), the Sec. 130 gates, the release sequence (Sec. 86),
   bug-fixing sessions, model testing, and the report queue alike. There
   is no task in this project that is improved by an early voluntary
   stop, and no task that justifies one.

10. **A PENDING TODO IS A REASON TO CONTINUE, NEVER A REASON TO STOP.**
   Finishing the task named in the user's LAST message is NOT done: the
   todo list is the standing work queue, and while ANY item in it is
   pending, the session continues with the next item, no wrap-up, no
   TASK_STATUS: COMPLETE, no handing back a summary. Declaring the
   current directive COMPLETE while the todo list still holds pending
   work is the SAME FAILURE as stopping mid-task, because the result is
   identical: work remains and the agent stopped. The only states in
   which TASK_STATUS: COMPLETE may be written are (a) the todo list is
   EMPTY or (b) every remaining item is externally BLOCKED (waiting on
   the user, a missing artifact, or a hard blocker). Anything else is
   INCOMPLETE, and INCOMPLETE means: pick up the next pending todo and
   keep working. This closing rule exists because an agent once stopped
   with four pending todos after a two-fix commit, exactly the shape
   this rule now forbids.

---

### 182. COMMENTS COMMENT THE CODE, NOT WHAT SOMEONE DID— An Activity Log In The Source Is A BUG, Removed On Sight

**Comments must describe THE CODE. A comment that narrates what a person did— what someone tried,
changed, fixed, removed, hit, broke, tested, profiled, benchmarked, searched, realized, decided, or was
going to do— MUST BE REMOVED ON SIGHT. It is a BUG. No exceptions, no deferrals, no "but it is useful
history." History belongs in git, in `bug_log.md`, in the plan documents, or in the commit message—
NEVER in the source.**

**Hard rules:**

1. **A comment that answers "what did WE do here?" instead of "what does this code do, and why?" is ILLEGAL. Remove it on sight.** The code itself is the record of what was done; the comment's only job is to explain the code that is present right now.
2. **Activity-log comments are removed IMMEDIATELY, in the same session they are seen.** Per Sec. 0b/Sec. 66/Sec. 127, a recognized defect is fixed NOW. There is no "collect them for a cleanup pass later"— when you see one in a file you are editing, you delete it in that edit.
3. **This is a BUG of the same class as a crash or wrong output.** It is not a style nit, not a nitpick, not optional polish. Sec. 57a/Sec. 57b ("no nitpicking") do NOT apply to it— those rules forbid *busywork*; this is *clutter removal of a specific forbidden construct*. Do not file it as a low-priority backlog item, do not defer it behind real work, do not let it sit in a todo list as "cleanup."
4. **No first person, no session narration, no datestamps, no ticket IDs, no review/audit provenance in source comments.** `; we tried X`, `; I fixed this`, `; used to be`, `; the old code did Y`, `; BUG-4 fix:`, `; fix from review`, `; verified Aug 2026`, `; was crashing`, `; removed per plan Sec.3.3`— all of these are activity-log fragments. DELETE them.

**The specific forbidden patterns (each is a removal, not a rewrite):**

- `used to`, `no longer`, `previously`, `formerly`, `originally`, `the old X`, `old code`, `old form`, `old path`, `old loop`, `old copy`, `instead of the old`, `was removed`, `was deleted`, `was replaced`, `was renamed`, `used to sit here`, `used to be re-tested`— history of the edit, not the code.
- `BUG-n`, `BUG #n`, `OPT-n`, `BUG-n fix:`, `fix (Bug 1)`, `Bug 4 fix`, `audit fix`, `review fix`, `BUG-08 audit fix: ...`— ticket/audit provenance.
- `we / I / our / my` narrating an ACTION: `we tried`, `we changed`, `we now`, `I added`, `I found`, `I realized`, `the author`, `I think`, `probably`, `I wonder`— first-person diary.
- `TODO`, `FIXME`, `XXX`, `HACK`, `WIP`, `workaround`, `temporary`, `for now`, `later`, `revisit this later`, `remove this`, `leftover`, `cleanup`, `clean up`— task-list debris inside the source.
- `removed per Sec.NNN`, `ported from X for source structure optimization`, `revised report`, `_revised.md`, `plan Sec.N`, `per audit`, `per review`, `per user request`, `per discussion`— document/provenance breadcrumbs. The *rule* may be cited when it constrains the code (Sec. -references are allowed when they say WHY the code must be this way); the *narrative of having applied a change* is not.
- `this ran unconditionally before`, `this was removed because`, `we measured`, `profiling showed`, `benchmarking showed`, `after grepping`, `traced with TRACE`— investigation diary. The RESULT of the investigation (the invariant, the constraint, the number) may stay; the process of getting there goes.
- Stale-deleted-code notes: `X was deleted— it had zero writers`, `the block that used to sit here was dead storage`, `[DivideU16 removed— replaced with inline idiv]`— tombstones for code that no longer exists.

**The test to apply to every comment, in every language we write (ASM, C, C++, Logo, Python, bat):**

> *Would this comment still make sense to a reader who has never seen this repository's history?*

If YES, it comments the code— KEEP it. If it only makes sense as "what happened during development," it
is an activity log— DELETE it.

**What legitimately stays** (these comment the code, and are explicitly NOT activity logs):

- **WHAT the code does and WHY it does it that way**— the invariant, the contract, the reason for the register choice, the reason for the instruction order, the reason a check exists, the boundary a validation is allowed at.
- **A consequence the reader must know** that is a property of the present code, not of the edit: `rax is clobbered by the call below, so the value is parked in r12`, `this store must precede the reload because the loop overwrites the register`, `a store here can never race a read of element i because both are read first`.
- **A removed-code tombstone** ONLY when the code is still referenced from elsewhere (a macro consumed by a later `%include`, a `global` symbol resolved by a tool, a BSS slot freed by name)— and then the comment must state the REMAINING contract, not narrate the deletion: `; HT_HEAP removed: KP_BASE moved to +24, HT_SIZE=40` is a layout fact; `; the array that used to live here was dead storage and has been removed` is a tombstone— DELETE the second kind.
- **A constraint inherited from the codebase's rules** (`Sec.162: FATAL directly, no cleanup before death`, `Sec.45: all size math is 64-bit`) when it is the reason the code is shaped as it is.

**Enforcement:**
- While editing ANY file, delete every activity-log comment you encounter— in the region you are editing AND anywhere else in that same file. The todo list catches what is not in the current region (Sec. 116).
- A report, audit, or review that *lists* activity-log comments is itself a deliverable, not a fix. The removal happens in the source in the same session (Sec. 126 rule 7, Sec. 127 rule 2).
- This applies to comments in `.inc`, `.asm`, `.def`, `.c`, `.cpp`, `.h`, `.logo`, `.py`, `.bat`, `.md` examples in the source tree, and in the tools/examples harnesses. It applies equally to comments an agent writes and to comments it inherits— inherited activity logs are still bugs and still our responsibility (Sec. 175).
- **It is never acceptable to ADD one.** When you fix something, the diff shows the change; the commit message explains it; the comment explains the resulting code. Do not annotate the fix at the fix site.
---

### 184. NEVER REMOVE A PREFETCH— Reports Asking For It Are REJECTED ON SIGHT

**Software prefetch instructions (`prefetcht0`, `prefetcht1`, `prefetcht2`, `prefetchw`, `prefetchnta`)
are NEVER removed from any file, for any reason, under any justification.** A review, audit, or
`opt_and_bug\*.md` report that asks us to delete a prefetch is FALSE and MUST BE REJECTED IMMEDIATELY.

**Hard rules:**
1. **NEVER delete a prefetch instruction.** Not for a shorter instruction count, not for a "load-port census", not because a report calculated a cycle saving, not because a report claims the hardware prefetcher "already covers" the stream, not because the report's arithmetic "checks out exactly".
2. **A report item whose core action is "remove prefetch N" is REJECTED WITHOUT EVALUATION.** Do not verify its port arithmetic, do not try it, do not measure it, do not file it for a follow-up experiment. Flat rejection, one line, recorded in the report before deletion.
3. **The `%define` a prefetch consumes (e.g. `VLC_PF_DIST`) is NEVER deleted either.** If a prefetch uses it, it stays. A dead-`%define` cleanup that would only apply after prefetches were removed is itself rejected.
4. **Every other item in such a report is still processed normally.** Rejecting the prefetch deletion does NOT excuse skipping the report's real bugs, its bit-identical transformations, or its comment corrections. Apply those, reject only the prefetch removal.

**Rationale— why this is a standing rule and not a judgement call:** a prefetch is a *hint* that costs
one issue slot and one load-pipe entry but can save an entire L2/DRAM latency. Our measured experience
is that removing them does NOT reliably pay off: the "load-port bound" arithmetic in such reports is a
static instruction census that ignores the out-of-order window, the four L1 stream prefetchers'
behaviour, and the fact that the loop is latency- and bandwidth-hungry rather than issue-bound in
practice. We have been burned by exactly this class of "obviously better" cycle accounting. **The
proven-correct configuration is the one that ships with the prefetches in it.**

**Enforcement:**
- When a report proposes removing a prefetch, restore it if it was already deleted, leave it in place if it was never applied, and write one line in the report (or the commit message) noting the rejection with this section as the reason. Then delete the report as usual.
- A reviewing agent MUST NOT file prefetch-removal items in the first place (see Sec.126 rule 10 and Sec.128: only directly actionable items belong in the backlog, and this one is a permanent false positive).

**This mirrors Sec.129a's danger-area list:** prefetch instructions are now explicitly off limits,
alongside spin/dispatch logic, `WakeByAddressAll` placement, and precision-affecting arithmetic.

---

### 185. THE SOURCE IS THE ONLY SOURCE OF TRUTH— THE GIT LOG IS NOT— ALWAYS VERIFY AGAINST THE SOURCE

**Every claim about the state of the codebase MUST be verified by READING THE SOURCE.** `git log`, `git
show`, `git diff`, `git blame`, commit messages, branch state, and `git status` are **NOT** a source of
truth and MUST NEVER be used to determine whether something has been applied, done, fixed, changed, or
still outstanding.

**Hard rules:**
1. **The source file is the ONLY authority.** To find out whether a change exists, OPEN THE FILE and read the code. A `%define` that is gone from the `.inc` means it was applied; a `%define` still in the `.inc` means it was NOT applied, no matter what any commit says.
2. **`git log` proves nothing about the current state of the code.** It is a *history of what someone claimed to have done at some point*, not a record of what the tree contains. A commit can be titled "wip", be amended, be reverted later, be applied to a different file, or be partial. None of that is visible in the log, and the log will confidently mislead you.
3. **The same rule applies to report `.md` files.** A report's own header (e.g. an "APPLIED AND REMOVED FROM THE QUEUE" note), a plan document, a progress doc, a changelog entry, a TODO list, or a session summary are all *claims*, not facts. Verify against the source before acting on any of them— and before declaring anything "already applied" or "not applied."
4. **Checking "was this applied?" means checking the code, not the bookkeeping.** Grep the source for the exact thing the report says to add, change, or delete. That single grep is the whole test. If the grep hits, the item is applied; if it does not, the item is outstanding.
5. **Commit messages are documentation of intent, never proof of effect.** Sec.90's output-visibility rule is about showing output to the user; it says nothing about trusting it as ground truth.
6. **When in doubt, read the file.** A wrong "already applied" verdict loses a report's work permanently; a wrong "not applied" verdict costs one extra read.

**Rationale:** an agent that reasons from the git log instead of the source will delete reports that
were never applied (losing real findings permanently) or keep re-applying items that are already in
(wasting cycles and risking double-application). The cost of the wrong direction is asymmetric, and only
the source settles it.

---

### 185. NO NAMED CAPACITY FATAL ON AN APPEND-ONLY ARENA— Rejected Reports

**A report that proposes pre-checking an arena and FATALing with a friendlier "you exceeded the budget"
message is REJECTED.** The arena is not the defect; the missing reclamation is. Adding the pre-check
only converts a working call into a hard abort and dresses up a symptom we have already decided to fix
properly.

**The concrete case, so it is never re-filed:** `sl_v7_stream` (`GetStream`) interns a retained copy of
the stream on every call, and the arena (`sl_atom_free`, 64 MiB) was an append-only cursor that
reclaimed nothing, so a per-token `GetStream` filled it within a generation. The proposed fix was a `len
+ 2 <= SL_ATOM_AREA_SIZE` pre-check in `sl_v7_stream` with a named `FATAL_VALUE` telling the author to
call it once per turn. **REJECTED— and now moot: the atom area has a mark-sweep collector (Sep 2026), so
a snapshot the caller no longer holds is reclaimed rather than accumulating.** The retention fact is
documented in the function header (it is useful), but **no capacity FATAL, no pre-check, no
`sl_v7_stream_toobig` label.** The cost that remains is the churn itself, which that header now
quantifies— do not re-file this as a memory bug; it is a per-token cost the guidance in the header
already tells script authors to avoid.

**Hard rules:**
1. Do not add a budget/consumption pre-check in front of an append-only arena.
2. Do not add a FATAL that names "you used too much budget" for a resource whose reclamation is a known open design item.
3. Documenting the retention/append-only property in the function header IS wanted. The abort is not.
4. If a report's only content is such a pre-check, it is consumed by this section alone.

---

### 186. FIXES FOR IMPOSSIBLE STATES ARE CRITICAL BUGS— THEY PRODUCE DEAD CODE— THEY MUST NEVER BE IMPLEMENTED, EVEN IF A REPORT SUGGESTS IT

**A "fix" for a state that cannot occur is a CRITICAL BUG, because it causes dead code. It must NEVER be
implemented, even if a report suggests it.**

**Hard rules:**
1. **A fix targeting an impossible state is a CRITICAL BUG, not a nit, not a backlog item, not busywork.** It is the same severity class as a crash or wrong output. The defect is not that the state is unguarded— it is that the guard is now permanently unreachable code sitting in a hot path, which is exactly the kind of clutter that hides a real bug and costs a cycle forever.
2. **NEVER implement such a fix, even if a report, an audit, or a review explicitly asks for it.** The report proposing it is wrong, and acting on it is the violation. Reject it with the proof, in the report, then delete the report.
3. **A report is NOT evidence that the state is possible.** Reports are claims; the source is the truth (Sec.185). A report that asserts "this is reachable" must be disproved or confirmed by reading the code and by measurement, never accepted on its word.
4. **Prove impossibility before you call something impossible, and PROVE REACHABILITY before you add a guard for it.** A guard is only legitimate once you have shown the state CAN occur. Show it with the same rigour you would use to show it cannot.
5. **Do not "harmonise" two bounds, two headers, or two comments by adding a check to make them agree.** If two things disagree, the fix is to correct the one that is wrong (usually a comment), never to add runtime code that makes the disagreement unreachable.
6. **Never write a test for an impossible state.** A test that cannot fail is not coverage; it is a false assurance that the code path is exercised when it never runs.

**How to tell them apart in one question:** *can this branch ever be taken with the shipped inputs?* If
the answer is no, the check is dead code and deleting it is the correct "fix"— and adding one is the
bug.

**The corollary that catches reviewers:** the engine's own header, a sibling constant's comment, or a
code comment claiming "X also needs this guard" is a CLAIM, not a fact. Before adding a guard to satisfy
a comment, check whether the guard can fire. Several header comments in this codebase have claimed a
guard exists in a function that never needed one— and the honest correction there is the comment, not
new instructions.

**Recorded case (Sep 2026), so the pattern is not re-litigated:** `lg_value_to_str` (the SectorLogo
printer) recurses once per nested list level in `.cons_val` and carries no `sl_stk_limit` watermark,
while its file header claimed every recursion entry has one. A review proposed adding the
two-instruction guard. Measurement rejected it: the recursion appends at least 3 bytes per level into
the 2 MiB concat buffer, so the BUFFER caps the depth at ~700K levels, about 33 MB of frame against a 64
MiB stack block. A self-referential list (built with `SetCdr` into its own tail, so the recursion is
genuinely infinite) dies on "string result exceeds 2 MiB concat buffer", never on a watermark. The guard
would have been permanently dead code; the header claim was the actual defect, and correcting the header
is what shipped.

---

### 187. WHEN YOU BATCH, YOU REALLY BATCH— Apply Everything In The Batch, THEN Test It All Together

**A batch that runs the full verification cycle after every single item is not a batch. It is a series
of unbatched runs with extra bookkeeping, and it defeats the entire reason batching exists.** The reason
to batch is that `build.bat` + `verify.bat` is the most expensive step in the project: it rebuilds the
database, runs every validator, and runs the inference smokes. Paying that once for twenty reports is
the whole win. Paying it twenty times costs twenty times as much and buys nothing.

**Hard rules:**
1. **A batch means: apply EVERY report's items first, THEN test the whole batch together.** Not "apply one, verify, apply one, verify". Not "apply one, build, apply one, build". The verification cycle runs ONCE, at the end of the batch, over all of its changes at once.
2. **No `verify.bat`, and no `build.bat` + validator run, in the middle of a batch.** The only build you may run mid-batch is the one the tooling forces (a `git`-tracked freshness gate is not a reason; do not hand-trigger builds either).
3. **Deleting a consumed report is immediate and is NOT gated on the verification run.** A report `.md` is deleted the moment its items are applied, exactly as Sec.171 rule 2 says. Verification is a separate axis: it validates the CODE, not the report. Holding deletions until the end of the batch defeats the synchronisation the deletion exists to provide.
4. **A failing batch is NEVER reverted wholesale. Locate the failure, fix it; if that does not work, revert SINGLE FILES one at a time.** The goal of a batch is to get the changes IN, in volume— not to discard them. `git diff` on the committed baseline shows exactly which hunks a failing file contributed, so a single bad file is isolated to a single file's revert. Discarding twenty reports' worth of verified, individually-sound work because one hunk in one file upset a validator destroys the entire point of batching and throws away correct fixes. The escalation is: (a) read the failing validator's output— it names the subsystem, which names the file; (b) read that file's diff in the batch; (c) fix the interaction properly; (d) only if the interaction is genuinely unfixable, revert THAT ONE FILE and re-run. Never `git checkout .` and never a whole-batch revert.
5. **Batch size is a judgement call, but "one report per verify run" is always wrong.** Group by shared source file and shared verification gate: several reports touching the same file or the same validator are one build, one validate, one commit.
6. **The commit is per batch, not per report.** One commit describes the batch as a unit; the individual reports are named in the message so the history stays traceable.

**Why this is stated as a hard rule rather than a preference:** the failure mode is subtle and
self-reinforcing. Stopping to verify "just to be safe" after each item feels diligent, and each
individual run passes— so nothing looks wrong. But the agent is spending almost all of its wall-clock in
the verification pipeline instead of applying work, and the batching instruction has been followed in
name (a batch was declared) while being hollowed out in practice. A batch is a promise about WHEN you
test, not about HOW MUCH you changed.

---

### 187a. BUILDING OR VERIFYING INSIDE A BATCH IS ILLEGAL— IT DEFEATS THE ENTIRE PURPOSE OF BATCHING

**It is ILLEGAL to run `build.bat` (or `verify.bat`, or any validator, or any smoke test) in the middle
of a batch. Absolutely not. Do not "just check that it assembles". Do not build to satisfy curiosity
about whether a register plan works. Do not build because the previous edit felt risky. NONE of that is
a reason.**

**A mid-batch build defeats the purpose of batching.** The reason a batch exists is that the
build+verify cycle is the single most expensive operation in this project: it assembles the DLL, runs
the ASM comment checker with zero tolerance, rebuilds the Qwen3-0.6B database, runs 21 validators, and
runs the inference smokes. **Paying that ONCE for a whole batch is the entire win.** The moment you
build mid-batch you have thrown that win away: you are now paying the full cost again for every
subsequent item, and you have changed nothing about what the build proves. A mid-batch build does not
make the final gate safer— the final gate covers every file in the batch anyway. It only costs time.

**Hard rules:**

1. **Zero build/test tool invocations between the first applied item and the batch's single gate.** No `build.bat`, no `verify.bat`, no `v_*_validate.exe`, no `nasm`, no `link`, no vision smoke, no `run_sectorlogo.exe`, no `test_vlms.bat`. The assembler and the validators are for the END of the batch, not the middle.
2. **An edit that might not assemble is still just applied— NOT built.** If you are unsure whether a register plan assembles, apply it anyway and let the batch gate find out. The Sec. 129a danger-area list already tells you which changes are risky enough to deserve their OWN cycle (Sec. 189 rule 2)— that is the correct response to risk, and it means "its own cycle", NOT "a build in the middle of this one".
3. **Reading the source IS the way to check an edit, not building.** Re-read the function, hand-trace the register liveness, count the `sub rsp`/`add rsp` pairs (Sec. 136), and verify every offset against the source (Sec. 185). That verification costs nothing and it is what actually catches mistakes.
4. **If the assembler rejects something, that is not a licence to rebuild-per-file afterwards.** Fix the rejected items as ordinary edits, keep applying the rest of the batch, and run ONE gate at the end.
5. **This applies to BOTH halves of the cycle.** Building mid-batch and verifying mid-batch are the same violation, and neither is a lesser one. A "quick" `build.bat` to see if it assembles is exactly the habit that turns a 20-report batch into 20 build cycles.
6. **Rationale:** the discipline is not "verification is bad"— verification at the gate is mandatory and non-negotiable (Sec. 130). The discipline is about **WHEN**: the gate runs once, over the whole batch. A build in the middle does not add confidence (the same code is covered by the final gate) and it destroys the time saving that made batching worth declaring. Following the batching rule in name while building per file is worse than not batching at all, because it also costs you the mental model of "this batch is verified as a unit".

---

### 188. A MISSING `cld` IS NEVER A BUG

**A missing `cld` (clear-direction-flag) is NEVER a bug. EVER. No exceptions. Do not add a `cld`, do not
report a missing `cld`, do not treat it as a defect, do not file it as an OPT, and do not carry it
forward as a "possible future improvement".**

**Hard rules:**
1. **Never insert a `cld` "to be safe".** Adding `cld` to a function is a BUG, not a fix. It costs an instruction and it is wrong.
2. **Never report a missing `cld` as a bug or as an optimization item**— not in an `opt_and_bug\*.md` report, not in a review, not in a commit message, not in prose. Such an item is a FALSE POSITIVE and MUST be rejected on sight, with one line stating this section as the reason.
3. **Never "restore DF" defensively.** A function that leaves DF set is not defective; DF is cleared by the ABI at every Win64 function boundary (`DF` is required to be 0 on function entry and on `ret`), so no caller can observe a leaked DF. Nothing in our code depends on DF being preserved across a call boundary.
4. **Within a straight-line block where we DO use string instructions (`movsb`/`stosb`/`lodsb`/`scasb`), DF state is a register-level fact of that block, not a global invariant.** Our own code never *needs* DF set; if any instruction requires DF=0 the fix belongs at that instruction's own block, not as a blanket `cld` at a function boundary.
5. **Rationale:** `cld` is the architecturally-correct thing to think about, and therefore exactly the kind of defensive check the codebase rejects elsewhere (Sec.124 impossible states, Sec.57b theoretical nits, Sec.186 dead-code guards). The Win64 ABI already guarantees DF=0 at entry to every function we export or call. A report asking for a `cld` is asking for an instruction that can never change behaviour— permanently dead code on a hot path.

---

### 189. BATCH LIMIT, VERIFY + COMMIT ONLY WHEN THE opt_and_bug FOLDER IS DRAINED OF REVISED REPORTS, AT MINIMUM 5 BUT BETTER 10 MODIFIED INC FILES

**When we are batch fixing from `opt_and_bug\` (the Sec.171 apply loop), the verification cycle runs
ONLY when BOTH conditions hold: `opt_and_bug\` contains no `*_revised.md` left to apply, AND at least 5
(better: 10) `.inc` files are modified.**

**Hard rules:**
1. **The batch floor is 5 modified `.inc` files; the target is 10.** Do not run `build.bat` or `verify.bat` on a smaller batch than that. Applying more reports is always cheaper than paying the build+verify cycle twice.
2. **The `opt_and_bug\` folder MUST BE DRAINED before verifying.** The gate runs when there is no `*_revised.md` left to consume, never mid-drain, because every remaining report is another source change that the same gate would have to cover anyway.
3. **WE NEVER BUILD "JUST FOR FUN".** There is exactly ONE build per batch and ONE verify per batch. A build or verify run with no reason to run it is WASTED WORK of the most expensive kind in this project. Never re-run `verify.bat` to "confirm" a result it already gave, never build mid-batch out of curiosity, never build to check whether an edit assembles, reading the source is how you check an edit (Sec.187a rule 3).
4. **Never run `verify.bat` twice in a row without a reason.** If it just passed, its result stands until the source changes. A second run on unchanged source proves nothing and costs a full rebuild plus 21 validators plus 10 inference smokes. Re-run ONLY after new source edits, or because the first run actually failed.
5. **Earlier than the floor is required when warranted.** Something dangerous was done, verify + commit IMMEDIATELY, even mid-drain. Dangerous means (per the Sec.129a danger areas and beyond): spin/dispatch logic changes, `WakeByAddressAll` placement, `gen_counter` ordering, `remaining_jobs` wait structure, FMA-vs-mul changes, accumulation-order or kernel-math restructuring, dispatch geometry changes, stack-frame arithmetic, or any change that risks precision or wrong output. A dangerous change never waits for the batch floor, it gets its own verify + commit the moment it is applied.
6. **`build.bat` is not `verify.bat`.** When the batch's edits are comment-only or confined to one subsystem with a naming validator, `build.bat` alone (for the comment checker) plus that validator is the gate (rule 7). The full `verify.bat` is reserved for a drained batch at or above the floor, or for a dangerous change.
7. **SectorLogo-only changes may use the SectorLogo validator alone instead of verify.** For sectorlogo changes we can run `tools\v_sectorlogo_validate.exe` alone instead of the full verify.bat, which is good especially when we updated the validator itself. This mirrors Sec.130's rule: when every changed file is a SectorLogo file (`src\asm\sectorlogo_*.inc`, plus `tools\v_sectorlogo_validate.cpp`), the fast gate is the validator:
   ```
   .\tools\v_sectorlogo_validate.exe .\bin\v7_threading.dll --db .\weights\databases\qwen3_0.6b\lz4ht.db
   ```
   build.bat still runs first (the ASM comment check must stay at ZERO violations), and the validator MUST be green with zero failures. Run the full verify.bat as before for any change that touches a kernel, dispatch, model/runtime, or microdb file.
8. **The waiter is the default idle state.** When the folder holds no `*_revised.md`, do NOT sit idle and do NOT build: start `watch_apply_opt_and_bug.py` (Sec.171 rule 1) and keep applying whatever it returns.

### 190. AVX-VNNI IS NOT SUPPORTED, NEVER USE `vpdpbusd` / VPDPBUSD

**We do not support AVX-VNNI. The CPU can not do it.** `vpdpbusd` (VPDPBUSD) and every other AVX-VNNI
instruction are FORBIDDEN in this codebase. A report proposing them is rejected on sight.

**Hard rules:**
1. **NEVER emit `vpdpbusd` / `vpdpbusds` / `vpdpwssd` / `vpdpwssds`.** No exceptions, no
   "gated behind a CPUID probe", no "only on the fast path".
2. **Do NOT add an AVX-VNNI CPUID probe and a `FATAL` at init.** A capability gate that FATALs a
   machine for lacking an instruction we are forbidden to use is strictly worse than not having it:
   it adds init code, a BSS flag and a new failure mode, for a code path that must never exist.
3. **The Q8 int8 dot product stays as the 3-instruction chain** (`vpsignb` + `vpmaddubsw` +
   `vpmaddwd` + `vpaddd`, see `COMPUTE_Q8_ROW` in `src/macros/v7_macros.inc` and the tail loop in
   `q8_kernel.inc`). Do not "simplify" it into a single dot-product instruction.
4. **Rationale for any future report:** the target CPU does not report AVX_VNNI, so the instruction
   assembles cleanly, passes the build and the comment checker, and then raises #UD (illegal
   instruction) at the first Q8 matmul, killing the process mid-inference with a crash dump instead
   of a wrong number. This is exactly the Sec.120 failure mode (the NASM APX `imul` crash), and it
   is why Sec.120 rule 2's post-assembly scan exists.
5. **AVX2 remains the ceiling** (alongside the already-forbidden AVX-512, Sec.128). FMA (`vfmadd*`)
   IS supported and IS used, do not confuse "no VNNI" with "no FMA".

**Reporting rule:** a reviewer that proposes `vpdpbusd` has mis-read the hardware target. Reject the
item in the report with this section as the reason, then delete the report as usual.

### 191. CPUID HARDWARE-BASELINE GATE, ONE BASELINE, CHECKED AT LOAD, NO DISPATCH

**The DLL has exactly one hardware baseline, it is measured once with CPUID at
`DllMain(DLL_PROCESS_ATTACH)`, and an unsupported CPU is refused, never emulated, never dispatched
around.** The required set is AVX2, FMA, BMI1, BMI2, AES-NI, SSE4.1, SSE4.2 and RDRAND, plus OS-enabled
SSE+YMM state (`XCR0` bits 1|2, probed via CPUID.01H:ECX[26]/[27] + `xgetbv`). `cpu_gate_probe`
(`src/asm/cpu_gate.inc`) records the verdict in `cpu_features_ok` and publishes the per-feature message
in `cpu_gate_msg_ptr`.

**Hard rules:**
1. **The probe NEVER FATALs.** It records. Dying under the loader lock would `TerminateProcess`
   the host at `LoadLibrary` time, before it can read `v7_get_death_reason()` or call
   `v7_free_inference`. The death belongs to the public API tier.
2. **EVERY new public entry carries `PUB_CPU_GATE`** immediately after its `ENGINE_GATE_*`
   line (`public_api.inc` is the single collection point, Sec.172). An unarmed public entry
   is a `#UD` on the wrong machine, which is the bug this gate exists to remove.
3. **`PUB_CPU_GATE` is NOT folded into `ENGINE_GATE`.** That macro is shared with the internal
   `v7s_sectorlogo_on_token` / `_interp` / `_eval_script` entries, which are not public API.
4. **`v7_free_inference`, `v7_shutdown`, `v7_get_generation_statistic`,
   `v7_get_internal_inference_ctx`, `v7_engine_is_dead` and `v7_get_death_reason` are never
   CPU-gated.** Cleanup and queries must stay callable, that is how a host reads the verdict
   and recovers.
5. **Every gated bit must be traceable to an emitted instruction.** Adding a bit "to be safe"
   is forbidden; adding one without a source site is a new failure mode with no user. When a
   new kernel or intrinsic lands, the CPUID set changes in the same commit.
6. **AVX-VNNI, AVX-512 and APX stay ungated** (Sec.190, Sec.120, Sec.128): they are
   forbidden, so a gate for them would guard a path that must not exist.
7. **The forced-flag test is mandatory.** `engine_death.c` step `t0b` writes
   `*dll.cpu_features_ok = 0`, calls a public entry, asserts death + a `cpu_gate` reason +
   host survival, then restores the flag and proves `v7_free_inference` revives the engine.

### 192. ZYDIS IN EXAMPLES IS TEMPORARY CRASH-DEBUGGING ONLY, NEVER PERMANENT

**Zydis crash handlers must NOT be added to examples/ programs. A Zydis header is added to an example
ONLY while we have a crash in that example and are debugging it; once the crash is fixed the Zydis
header is removed again. Zydis must be REMOVED from the examples but KEPT in the validators.**

**Hard rules:**
1. **Do NOT add `zydis_debug_header.h` / `ZydisInit()` to any program under `examples/`**, not proactively, not "for safety", not by default, not for new examples.
2. **The ONE exception: an example that HAS a crash may carry the Zydis header TEMPORARILY**, as a diagnostic aid while hunting that specific crash.
3. **When the crash is fixed, the Zydis include and the `ZydisInit()` call MUST be removed from the example again**, the debugging aid never stays behind.
4. **Validators (`tools/`) are the opposite: their Zydis crash handlers are PERMANENT** (Sec.14c applies to validators and tools in full). The temporary-only policy applies ONLY to `examples/`.
5. **This section supersedes Sec.14c for `examples/`**: Sec.14c's permanence rule governs validators/tools and the engine; examples follow THIS section.

### 193. NEVER WAIT FOR THE SERVER WITH SCRIPTS OR POLLING LOOPS, YOU LOCK YOURSELF OUT

**The server must NEVER be waited on with a script, a polling loop, or any blocking wait from the
agent's shell. Not a health-check loop, not a curl retry loop, not a sleep-until-ready. JUST NEVER.**

**YOU MUST NEVER LOOP FOR THE SERVER, not for readiness, not for a turn to finish, not for a restart,
not for anything. A loop that waits on the server BLOCKS THE AGENT'S SHELL until the server answers, and
when the server is hung or dead the agent is LOCKED OUT and the DEVELOPMENT PROCESS HALTS. Every minute
spent looping for the server is a minute stolen from the work itself. The moment you notice you are
writing a wait-for-server loop of ANY kind, DELETE it and do something else; come back with a SINGLE
probe in its own tool call.**

**Hard rules:**
1. **Do not poll the server for readiness. NEVER. Not curl-in-a-loop, not Invoke-WebRequest retries, not for/sleep+probe one-liners, not any construct that sits in the agent's tool call until the server answers, this includes the "harmless" Start-Sleep 3 then probe. ONE probe per tool call. The server starts in UNDER 3 SECONDS, a wait loop is pure waste: it floods the transcript, hides the server window's own startup output, and the agent locks itself out of its shell every time.
2. **Start the server (start_server.bat, own window, standalone call), then go on with other work.** When the server window shows "listening", interact with it in SEPARATE, single, non-blocking tool calls (one curl per call, visible output).
3. **If a call comes too early, that single call fails visibly, retry it as another single call.** A failed single call costs nothing; a waiting loop costs the session.
4. **This supersedes any plan text that says to wait:** a plan step like "wait for health" is executed as "start, do other work, then probe ONCE per tool call until it answers".
5. **NEVER run a chat turn, an inference probe, or any server request inside a tool call whose timeout exceeds a normal single-turn budget (~5 min). A turn that hangs on a broken sidecar then burns the tool timeout and the transcript with it. A hung turn is diagnosed from the SERVER WINDOW (own console, visible output) or from a captured server logfile read AFTER the fact, never by sitting in a blocking read.**
6. **Do NOT redirect server stdout/stderr when YOU started or control the server**, the server window is the observable diagnostic; hiding it behind redirection (Start-Process redirects, logfile arguments used routinely instead of the visible window) blinds exactly the debugging that needed it. The `[logfile]` argument of start_server.bat exists ONLY for crash-dump capture (Sec.148), then returns to no-logfile operation.

### 194. CURL RESPONSES FROM OUR OWN SERVER MUST ALWAYS BE READ IN FULL

**Every single curl request to our own v7 server, chat turns, health, /background, /avatar, page
fetches, resets, every endpoint, every time, MUST be displayed on the agent terminal IN FULL, raw,
byte-complete and unfiltered, and the agent MUST READ IT. There are NO exceptions, NO alternatives, and
NO special cases of any kind.**

**This means, with zero room for any other interpretation:**

1. **NO `-o file` / `--output` redirection and NO `-o NUL`. The response body is printed directly to the terminal by curl itself. Never park a server response in a file, never discard it, never inspect it later with a script, the reading happens NOW, on the raw terminal output.**
2. **NO piping the response through python, Select-String, grep, head, tail, wc, length prints, JSON field extraction, or any other filter or post-processor. The raw bytes curl received are what the agent reads.**
3. **NO truncation and NO slicing, no [:400]-style cuts, no "first N chars", no "reasoning len:" summaries. The agent reads the COMPLETE reasoning, the COMPLETE answer, the COMPLETE error body, the COMPLETE page markup, whatever the endpoint returned, in full, every time.**
4. **The response IS the diagnostic.** Reasoning lengths, empty answers, missing fields, error JSON bodies, every debugging session this rule was written from was delayed by a filtered response that hid exactly the field under test. Filtering a server response is how a five-minute fix becomes a five-hour hunt.

**This is a READING discipline, not a storage one:** the agent READS every server output in full, every
time, in the terminal. Not stored. Not summarized. Not sampled. READ.

### 195. RAG SIDECAR QUALITY LAWS— The Role Loves Only the Question and the Facts (BLOCKER, Oct 2026)

**These rules were beaten out of the Ardmark experiments. They exist because every one of them, when
violated, produced measurably worse answers— thin, off-role, or silent. They are not style preferences;
they are what separates a working RAG role from one that fakes its test quality.**

**Rule 1— THE MODEL MUST OPERATE ON THE QUESTION AND THE RAG, never on a prompt that fakes test
quality.** A sidecar line that hands the model a pre-defined topic list ("tell them of the storms, the
shield walls, the Iron Lions...") cheats: the answer then recites OUR topics instead of answering the
USER'S question, enrichment comes from a canned list rather than the retrieved facts, and every test run
looks good for the wrong reason while any different question exposes the fraud. The role's steering
lines may shape VOICE and BEHAVIOUR (how to speak, how to open, what to never do)— they must never
enumerate subjects to talk about. If an instruction can be replaced by a topic list, it is a fake.

**Rule 2— OPEN-ENDED BY CONSTRUCTION.** Every role line is defined open-ended: the question and the
retrieved facts drive the content, the sidecar only sets the manner. "Pre-defined topics with reason" =
won't-fix class; the correct reaction when one is found is to rewrite the line open-ended immediately.

**Rule 3— THE RESPONSE-INITIATOR FENCE IS MANDATORY (proven, not optional).** The think block must be
closed in EXACTLY this shape when a rewrite seeds the visible answer: `"\nAlright, time to respond.\n" +
closing-think-tag + "\n\n"`— ONE `\n` after the sentence, TWO `\n` after the closing tag (3 in total).
This exact pattern was used by the original Ardmark RAG and PROVEN in experiments: without it the model
FALLS SILENT after the closing of think, and the server then runs into the think-tag fallback ladder
(the multi-temperature rewrites at 0.7/1.0/1.3 + the no-think fallback). **The fallback is NOT a normal
recovery path for a RAG turn: the server does not normally take the fallback, and when a rag turn ends
up in the fallback this is in fact a BUG IN OUR RAG**— the fence went missing. (The ladder's temperature
retries remain correct for genuinely-cut turns; the rag must simply never trigger them.)

**Rule 4— ONLY THE FIRST TURN MAY EMIT FIXED FACTS.** Turn 1 of a session may carry priming info about
the world (the prepended primer file). EVERY LATER TURN must work with the RAG for facts— the sidecar
may repeat behaviour-defining lines (voice, how to respond) as many times as needed, but it must NEVER
inject a second batch of fixed-world facts by hand. A later turn that re-pushes a fixed fact list is a
double sin: it fakes the test AND it starves the retrieval path.

**Rule 5— TESTING IS MULTI-TURN AND READS THE WHOLE REPLY, THINK INCLUDED, WITHOUT ANY FILTER.**
Interactions must be tested as multi-turn conversations WITH ANY FILTER, and every reply must be read in
whole— the visible answer AND the model's complete thinking block. Only testing the reply is WRONG and
hides lots of info needed for tuning: whether facts were absorbed or doubts, where the role breaks,
whether the model parrots our machinery, whether the fence landed— all of that lives ONLY in the
thinking. A tuning session that reads only replies tunes blind.

**Every rag sidecar must be audited against these five rules before it is tuned or committed. A
deviation in the injection concat, the answer stage, or the test method is a bug with the same severity
as a wrong-output bug.**

**Rule 8a— THE STYLE-CONTRACT PHRASE SHAPES (proven in 5 dedicated style probes,
`qwen3_thinking_phrases.md` section 11).** Qwen3's thinking has a measured, stable register (analyzed in
full in `qwen3_thinking_phrases.md` from the 10-question reference battery
`qwen3_thinking_reference.log`): openers `"Okay, so I need to ..."` and `"Let me start by recalling
..."`; doubts `"Wait, but ..."` and `"I should check/make sure ..."`; uncertainty `"Maybe ... but I'm
not sure"`; pivots `"Now, why ..."`; the close-out `"So, putting it all together ..."` and the
`"Alright, time to respond."` handoff (rule 3). These trigger lines are the model's OWN way of moving
between recall, doubt, re-check and wrap. **Our injections into the thinking must be as close as
possible in style to these**— same register, same starting words, same doubt-and-acceptance mechanics:
injected chunks open with a recall-type line (never an imperative), the retrieval hand-off reads as the
model's own acceptance ("and already many things stir in me ..."), and no injected line ever names the
machinery. WHY: a line that matches the register is absorbed silently as the model's own thought; a line
that sounds like an instruction sheet is a SEAM the model parrots into the visible answer (the leak) or
stumbles over (the silent turn, rule 3). The runaway `"Alternatively, maybe X. But I'm not sure."` loop
is the model's visible signal of shaky retrieval— a well-shaped injection replaces that loop instead of
feeding it.

**Rule 7— NO FIXED TEST BATTERIES FOR TUNING— individual curl conversations only, different every
time.** The agent must NOT tune the rag against a predefined battery of test conversations, because
tuning towards fixed questions tunes to THOSE questions: the statements get polished on the battery runs
and break on anything else (the fraud of rule 1, at the test layer). The 10-question reference battery
was a ONE-TIME capture, only for crystallizing the thinking-voice register of rule 6. Tuning itself runs
INDIVIDUAL conversations through the server with curl, improvised fresh every session— new questions,
new topic shapes, multi-turn, never a fixed list. Same question twice in a tuning pass is allowed only
when isolating a specific regression. WHY: retrieval roles are general instruments; only improvised
probes reveal what actually generalizes.

**Rule 8— INITIATE OUTPUT PLANNING AFTER THE RETRIEVAL.** The reference analysis (rule 6,
`qwen3_thinking_phrases.md` section 7) found the OUTPUT PLANNING family: `"I need to structure this into
two parts / this comparison"` commits to an answer layout and **predicts the final structure exactly**,
and `"Make sure to cover both X and Y"` is the final pre-close checklist. **This is the family we
deliberately trigger after the BM25 facts land**: once the model has the retrieved facts, our injected
line must move it from recall into ANSWER PLANNING in its own register— that is the moment the re-think
stops looping and closes cleanly for the handoff. The injected planning line shapes the MANNER only
(conversational, spoken, no lists, cover what was asked)— the CONTENT is owned by the question and the
corpus (rules 1-2). Writing the planning line as an instruction sheet instead of an in-voice thought
re-creates the leak (a planning line that says "answer briefly, list nothing" is behaviour-defining and
legal; a planning line that names subjects or sentences to say is a fake test).

**Rule 8a— THE STYLE-CONTRACT PHRASE SHAPES (proven in 5 dedicated style probes,
`qwen3_thinking_phrases.md` section 11).** When the request carries a style ("flowing paragraphs", "no
lists", "storyteller"), the model plans around an AVOID/KEEP pair and closes with an all-together line:
(a) the style contract is restated FIRST in its own words (`"Okay, the user wants a flowing story ... no
lists, no bullets, no stars."`); (b) the AVOID half enumerates forbidden FORMATS ("no markdown, no
lists, no stars, keep it conversational"); (c) the KEEP half names the positive register ("spoken",
"flowing", "short paragraphs", "everyday language"); (d) the SHAPE PLAN predicts beginning/middle/end
("Start with the setting, then introduce, end with a reflective moment"); (e) the ALL-TOGETHER CLOSE
("Let me put it all together now" / "Alright, let's put it all together") hands the block to the answer,
the sibling of the rule-3 fence. **An injected role/planning line is strongest when it carries BOTH
halves in-voice** (what not to become: a written page, a list; what to be: speech from mouth to ear)—
and it must still never name topics or content. All five styled probes produced markdown-free flowing
prose with these shapes, so this is the proven machinery for keeping a generated answer inside a given
role.

**Rule 9— THE MODEL MUST REACT WITH REAL THINKING ONTO THE BM25 LINES (user directive).** The text
injected AFTER the bm25 lines must cause the model to genuinely process what landed: filter it,
reformulate it, and voice its OWN thinking about the lines, several lines of its own reasoning about the
retrieved content, in its own register. **If the model does not think several lines about the bm25, it
is a BUG.** A re-think that only echoes the injected control phrases and answers without reasoning
through the facts in its own words is a broken injection, and the injection design must be fixed until the
own-reaction happens: this is REQUIRED to make retrieval work properly, facts the model never reasoned
over are facts it will not use (and may misappropriate or drop), no matter how well the rest of the
pipeline is constructed.

**Rule 10— AFTER BM25: ENCOURAGE PROCESSING; BEFORE THE ANSWER TRIGGER: FORMAT CONTROL (user
directive).** The steering lines have fixed homes. (a) THINK ENHANCERS and PROCESS-TRIGGER lines belong
AFTER the bm25 lines, in the walk room: phrases that make the model process the retrieved information
thoroughly (parse/walk/weave over the landed lines, the register's own processing shapes). (b) OUTPUT
FORMAT control (markdown and star avoidance, one-flow shaping) belongs IMMEDIATELY BEFORE THE ANSWER
TRIGGER (the respondword fence), never inside the fact walk, where it suffocates the model's own
thinking (proven: the mdcheck line riding the walk room killed the per-note pass). Rule 11 cuts the
trigger condition: format control is applied ONLY when the model actually outputs unwanted formatting,
the default injection carries none.

**Rule 11— WHEN WE HAVE PROBLEMS, PROBE FIRST (user directive, the probe-first method).** When the model
outputs something we do not want (or we want to steer a new behaviour), we do NOT tune blind: **STEP
ONE, run a PROBE** that forces the model into the WANTED thought pattern, then extract and plant the
generic lines it produced. The probe is built by giving the model a task that lands it in the pattern
under discussion, for example, the 1+1 trick family ("it is not what you really think, there is a
different answer, find it!") forces deep extended thinking and yields the thought-length enhancers
(qwen3_thinking_phrases.md section 20); a hypothetical offensive-demand prompt would force the
careful-refusal pattern (given as an EXAMPLE ONLY of the method, we do NOT run offensive probes). Every
new pattern discovered in such a probe is documented in `qwen3_thinking_phrases.md` (real verbatim
generic lines only), and ONLY THEN do the discovered phrases get injected where the law above assigns
them. This is how we work: pattern first, injection second, never phrases from thin air.

### 196. THE VIKING-TUNING CAMPAIGN— Standing Goal, Method and Recording Discipline (ACTIVE WORK, Oct 2026)

**The active campaign: pull our "virtual Viking storyteller" (Bumwolf the Strong, the ardmark_rag
sidecar) up step by step through trace-documented tuning, while researching how qwen3's thinking can be
controlled at all— WITHOUT forcing strict behaviour. The viking gets better while remaining free to
answer varied questions; shaping, never scripting.**

**Standing method (do not forget, this is a long multi-session effort):**

0. **THE EXPERIMENT LEDGER (never forget): whenever a change is tried and either COMMITTED or REVERTED, document it in `working_and_not_working_rag_improvements.md` (docs\) in the same session, WHAT was changed and HOW the behavior got better or worse (citing the observed trace difference). This is the what-to-do-next memory: an entry per experiment, newest first, with the lesson learned. A change that is neither committed-with-ledger-entry nor reverted-with-ledger-entry is an unfinished experiment.**
1. **Every tuning change goes through the improvised-curl loop** (Sec.195 rule 7): fresh probe, reload only when needed (a sidecar load after the hook initialized answers "staged", a restart (stop + start_server.bat) makes it live), and the WHOLE reply is read, thinking block included, unfiltered (rule 5). The trace IS the tuning data. **The improvised probes cycle through CHAIN-FOLLOWING transactions: first turn asks a topic, the follow-up reacts to what the model itself SAID (a person it mentioned, a place, a story it referenced), the reply's content goes into the second retrieval, which is the real usage shape and the shape that reveals state bugs (fallback, session confusion) that single turns never show.**
2. **WHENEVER a change works, or good output appears (even by chance): examine the thought trace and RECORD what the model wrote that initiated the good output or retrieval.** Also record the model's SELF-DESCRIPTION phrases (the personality/behaviour register it used on itself). Both kinds of phrases go into `qwen3_thinking_phrases.md` immediately, so the discovery is never lost. These are the control phrases we later use in the rag, at the appropriate injection place AND in the prepend file. **The probes/commands are only LEVERS: what gets recorded is what the MODEL THOUGHT when the output was best (its considered thought lines, verbatim), because the rag injects THOUGHTS into the thinking, never prompts, the user's prompt stays untouched, the injection plants the thought that produces the behaviour.**
3. **Phrases discovered so far (keep adding):**
   - `"I will now respond in that voice, just as I do with my stories, and I will stick to the facts without embellishment."`, the model's OWN commitment line, written after our planningcue landed; the bridge from planning into answer.
   - `"I need to make sure that the response is in line with my character"`, the personality-binder register: once it fires, later turns keep the role.
   - `"Bumwolf is a storyteller, but he's not a dragon. He's a man with a long history of telling tales."`, the model's third-person self-description inside thinking; it precedes role-consistent answers.
   - `"the old stories of my land are mine to tell:"` (injected ownership claim)— turned a hard denial ("I do not have any knowledge of fjord dragons") into the witness frame ("seen them only as part of the stories I tell").
   - `"I have no idea if I have seen the ... myself"`, the hedged-denial phrase: fired when the retrieval did NOT surface the topic's facts (primer's deny-constraint then wins). Control: the retrieval query must match the topic, or the answer is thin by design.
4. **Known seams to keep watching:** the asterisk stage-direction seam (`*Smiling warmly...*` at answer end, controlled via the planningcue's "no acted-out gestures" avoid-half; stage-directions are an RP register that the style contract must explicitly forbid). The thin-answer seam (hedging wins when the retrieval misses the topic). The primer's own deny-constraint ("I do not invent..." from ardmark_prepend.txt) is a USER-owned file and one of the strongest control levers we have for this role— weigh changes there with the user.
5. **The goal hierarchy is fixed:** (a) quality first, poetic, on-role answers that stay generous on any varied question; (b) shaping through the thinking-voice register (rule 6), never through imperative lists; (c) every discovered control phrase is committed to `qwen3_thinking_phrases.md` in the same session it is discovered.
6. **THE COMMIT/REVERT LOOP (user directive, verbatim): every progress warrants a commit, if your changes fail do not diagnose but revert.** Each tuning change that helps is committed immediately (git message lists the changed sentences + the resulting improvement); a change that makes output worse is REVERTED (`git checkout, <file>`) without a diagnosis session, the baseline is solid, a failing phrase simply does not work, and the next experiment uses a different phrase from the register instead. The revert also gets its ledger entry (item 0).
7. **THE GOAL TREE (user directive, verbatim): improved chat behaviour with consistent answers and followups is our main goal; we want good poetic text, non-leaking retrieval, and the ability to ask followup questions based on context.** The storyteller has a CONSISTENT HUMAN-LIKE PERSONALITY with as few LLM quirks as possible. Every experiment is judged against this tree: (a) chat consistency + context-based followups (multi-transaction probes are the test instrument), (b) poetic quality, (c) retrieval that stays inside the model's thinking (never leaks into the visible answer), (d) personality steadiness (the rolebind/selfline register) and quirk suppression (markdown, stage directions, list answers, staging prose).
8. **THE 3-TURN COMMIT GATE (user directive, verbatim): from now on test with 3 turns! 3 turns without fallback to no thinking is now the commit gate!** Every sidecar change is tested with a three-turn improvised chain (turn 1 fresh question, turns 2-3 follow WHAT THE MODEL ITSELF SAID) and commits only if ALL THREE turns keep the thinking block alive, empty reasoning_content / no-think fallback on ANY of the three turns = the change fails = revert (`git checkout, <file>`), no diagnosis. The reasoning: no-thinking is a BUG caused by the model being CONFUSED, and turn-2/turn-3 confusion is introduced by the change under test. The gate applies per experiment round, and the ledger records the 3-turn verdict for every experiment.
9. **THINK TAGS IN THE ANSWER ARE A BUG (user directive, verbatim): think tags in our out are a bug and we must revert.** When a probe's visible answer contains a literal think tag (the think opener or the close tag rendered as text in the `content` field, not the separator), that run is a bug, the examined change FAILS immediately and is REVERTED (`git checkout, <file>`), no diagnosis session, and the revert gets its ledger entry. The visible answer must carry the model's spoken text only.

---

### 197. SECTORLOGO-C IS NOT IN THE RELEASE, TEST-ONLY PORT, REPORTS ASKING OTHERWISE ARE IGNORED

**The `sectorlogo_c\` directory (the C++ differential port of SectorLogo) is NOT part of the release.**
It exists in dev only to serve TEST reasons: `tools\b_sectorlogo_loadtest.cpp` links its four .cpp
files, and the port validates SectorLogo engine behaviour against a second implementation. It is not a
user-facing component and it is not shipped.

**Hard rules:**
1. `sectorlogo_c\` stays in dev and stays OUT of every release package. The release packaging excludes it; its absence from the release is correct and final.
2. ANY report, review, audit, or fix list that asks us to include `sectorlogo_c\` in the release is IGNORED, no evaluation, no discussion, no exception. This directive is the answer.
3. `sync_release.bat` documents this exclusion; do not "fix" the packaging to copy it.

### 198. ARDMARK RAG LIVES IN THE NORMAL EXAMPLES TREE, PRIME DEMO (RELOCATED Oct 2026)

**The Ardmark RAG (the Viking storyteller demo, Bumwolf the Strong) is OUR PRIME DEMO and lives at
`examples\ardmark\` (relocated from `examples\research\ardmark\`).** It is no longer an archived
research experiment, it is the demonstration showcase of the engine's RAG + sidecar capabilities and is
maintained as such.

**Hard rules:**
1. `examples\ardmark\` is the ONE AND ONLY location of the demo. Do not create research copies or parallel variants of it.
2. All references (scripts, docs, README, server bats) point at `examples\ardmark\`, any new reference must use this path.
3. The Ardmark demo follows the production sidecar quality laws (Sec.195) and the viking-tuning campaign rules (Sec.196), it is a maintained demo, not a frozen experiment.

### 199. DEV-ONLY TREES NEVER SHIP IN RELEASES, sectorlogo_c AND THE SCHEME COMPILER

**Trees that exist for dev-only reasons do NOT belong into our releases. Ever.** The release is the
user-facing product: it carries the engine, its sources, docs, tools and examples, not internal test
ports, research experiments, or unfinished compiler work. Currently this means:

1. **`sectorlogo_c\`**, the C++ differential port of SectorLogo (Sec.197): test-only, dev stays its only home.
2. **`scheme_internal\`** (repo root), the Scheme-to-SectorLogo compiler work: unfinished, needs fixing, parked dev-only (its own directive). The release tree carries NO scheme content anywhere (the multilang benchmark under `examples\` carries no scheme benches; the old `examples\experimental_work_in_progress\scheme_compiler\` tree is deleted from dev).

**Hard rules:**
1. These trees stay in dev and stay OUT of every release package, every release commit, and every release archive. Their absence from the release is correct and final.
2. ANY report, review, audit, or fix list that asks us to include a dev-only tree in the release is IGNORED, no evaluation, no discussion, no exception. This directive is the answer.
3. `sync_release.bat` documents these exclusions; do not "fix" the packaging to copy them. A future dev-only tree joins THIS list (or gets its own Sec.197-style section); it does not silently ride along with the sync.
