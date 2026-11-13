# AGENTS.md (release)

This is the release copy of the v7 project's agent handbook, cut down to the
rules that describe how the shipped code is structured and verified.

The full development handbook ships as `docs\AGENTS_dev.md` (with
`docs\ABOUT_AGENTS_dev.md` explaining its role). The rules below are the
subset an auditor needs to navigate and rebuild this package.

## Project layout facts

- `src\asm\` - the entire engine in NASM x64 assembly (152 sources: 150 .inc
  plus v7.asm and v7_threading.asm). Every
  instruction line carries a hand-written comment; `scripts\build.bat` enforces a
  zero-tolerance comment checker on every run.
- `tools\` - validator/benchmark/DB-tool sources. `scripts\build.bat` compiles 50
  sources from here; every kernel module has exactly one monolithic
  `v_*_validate` program.
- `sectorlogo_c\` - deliberately NOT part of this release: it is the
  test-only C++ differential port of the SectorLogo engine (kept in the dev
  tree; any report asking to ship it is ignored).
- `zydis_debug_header.h` - release builds compile tools against the no-op
  stub in `tools\zydis_stub\`; the real Zydis decoder is dev-only. See
  `docs\zydis_debug_readme.txt`.
- `docs\AGENTS_dev.md` - the full development handbook (process rules,
  review loop, verification gates). It binds the development process, not
  the software.

## Verification contract

- `scripts\build.bat` builds everything (DLL + 50 tool sources + examples + server)
  and runs the ASM comment checker with zero tolerance.
- `scripts\verify.bat` is the definitive ASM-component gate: it rebuilds, builds the
  Qwen3-0.6B database, runs ALL validators (21 programs: sampler, 17
  kernel/database validators, the StreamingLLM validator, the 1402-test
  SectorLogo validator and the generate-path validator), and ten inference
  smoke tests in both F32 and Q8 modes - output must contain "paris".
  Vision gates run in `scripts\verify_vision.bat` (Qwen3-VL byte baseline + the
  LLaVA anyres gate); the server, RAG-session and CLI/environment contract
  tests run in `scripts\verify_server_and_rag.bat`.
- `scripts\verify.bat` is git-free: it restores `threading_proxy.ini` by byte copy,
  so it works in a fresh unpack with no VCS.
- Every failure is fixed before release. The suite accepts exactly the
  documented SwiGLU Q8xQ8 reference limit recorded in the development notes;
  everything else must pass. The sampler validator gates all 40 cases
  bit-exactly, and the former matmul known-limit claim was stale (see
  CHANGELOG 2.29.0). `scripts\verify.bat` is the contract.

## Code conventions (summary)

- Windows x64 ABI everywhere: 32-byte shadow space, 16-byte stack
  alignment, non-volatile registers preserved (SF_* macros in ASM).
- C/C++ (tools/examples/server): printf-only output, Allman braces with
  `{` on its own line, speaking variable names, no `has_*` capability
  checks, no hardcoded limits except the output buffer, no defaults for
  missing values - invalid state FATALs loudly.
- ASM: every numeric offset through `%define` constants, hand-written
  comments on every line, no 3-operand `imul reg, reg, [mem]` (APX EVEX
  trap), AVX2 is the ceiling (no AVX-512, no AVX-VNNI).
- Deterministic paths are 100% deterministic: greedy inference is a pure
  function of (binary, database, prompt, CPU features).

The full rule set lives in `docs\AGENTS_dev.md`.
