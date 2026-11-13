# Contributing

This guide specifies the assembly formatting rules, build checks, and submission conventions for
contributions to v7. The examples and commands below describe the expected source layout and validation
procedure.

## ASM Coding Standards

The project uses NASM x86-64 assembly. The repository's `check_asm_comments.py` utility checks the
documented commenting rules. Contributions must meet these requirements.

### Comments

Each assembly instruction must have a comment that explains its purpose:

```nasm
     mov rax, [rcx + WQ_OFF]        ; load the Q-projection weight pointer
     shl eax, 4                     ; convert an entry index to a byte offset
```

- Indent: at least 5 spaces before the instruction
- Comment: at least 7 spaces between instruction and `;`
- Comments must explain WHAT the value means, not just restate the instruction (for example, prefer "load the Q-projection weight pointer" to "load rax")
- Do not add auto-generated comments; write them by hand

### Offsets

All numeric offsets in addressing modes must use `%define` constants:

```nasm
; GOOD:
     mov rax, [rcx + WQ_OFF]
     mov ebx, [rdx + CONFIG_DIM]

; BAD:
     mov rax, [rcx + 48]
     mov ebx, [rdx + 0]
```

### Stack Frames

All functions must use the `SF_*` macros from `macros/v7_apicall.inc` for Windows x64 ABI compliance:

- 32-byte shadow space
- 16-byte RSP alignment
- Preserve non-volatile registers (`rbx`, `rsi`, `rdi`, `r12-r15`, `rbp`)
- No red-zone usage

### API Calls

All Win32 API calls must use the `API_CALL` macro, never hand-roll `call [__imp_*]` sequences:

```nasm
; GOOD:
     API_CALL GetPrivateProfileIntA, section, key, default, ini_path

; BAD:
     sub rsp, 32
     mov rcx, section
     call [__imp_GetPrivateProfileIntA]
     add rsp, 32
```

### Register Conventions

- `v7d_*` functions = dispatched/multi-worker entry points
- `v7s_*` functions = single-threaded entry points
- After any `API_CALL` or `WRITE_*` macro, treat volatile registers (`rax`, `rcx`, `rdx`, `r8-r11`) as clobbered and reload any required values from non-volatile storage

## Build Verification

Before committing any change:

```bat
verify.bat
```

All applicable checks must pass. The narrowly defined external-memory-pressure exception for
`v_generate_validate` is documented in [AGENTS_dev.md](AGENTS_dev.md); all other validator failures block
submission.

## Commit Style

- Make small, focused commits that each implement one logical change.
- Descriptive messages: `fix:`, `feat:`, `refactor:`, `docs:`, `slim:`, `tune:`
- Stage only the files you changed, never `git add -A`
- Never commit build artifacts (`.obj`, `.exe`, `.dll`, `.db`)

## Running the Comment Checker

```bat
python src\check_asm_comments.py src\asm\
```

This checks every `.inc` and `.asm` file for comment compliance. The check must complete without warnings.
