# C and C++ Source Rules

1. Use `printf` for console output rather than `fprintf(stdout, ...)` or `fprintf(stderr, ...)`. Writing to a file opened with `fopen` is allowed. Zydis and third-party code are exempt.
2. Struct and typedef definitions go at the top of the file, after the includes.
3. Use `static` only when it is warranted. Do not add `static` to file-scope functions or globals in programs built from a single translation unit.
4. Use explicit, readable conditional expressions rather than cryptic constructions.
5. Use `#pragma once` for header guards. Do not add an `#ifndef FOO_H` guard.
6. Do not introduce `has_*` capability-check helpers.
7. Do not call `memset` on an object that has already been zero-initialized with `= {0}`.
8. Use Allman brace style: `{` stands alone on its own line, without exception, and not just for short blocks. Braces are used when the body has multiple statements. A single-statement body may use braces in Allman style or leave the statement on the `if` line, per rule 22. `if (x) { a; b; }` is not allowed.
9. Parenthesize every sub-expression in `if`/`while`/`for` when combined with `&&`, `||`, `&` or `|`.
10. Do not add `#define` macros that merely guard code without an actual requirement.
11. No default generation limits except the output buffer size.
12. Do not touch the sampler unless the CLI provides sampling parameters.
13. No checks on whether the inference context was created. It either FATALs or succeeds.
14. Use `typedef struct` for structure declarations; do not declare bare `struct` types.
15. Never use `int8_t`, `uint8_t`, `int16_t` or `size_t`. Use `char`, `unsigned char`, `int` or `__int64`. `size_t` is the worst offender; use `int` when it fits.
16. Global variables go after the struct definitions.
17. Never guard `#define`s with `#ifndef`.
18. `#define`s go at the top of the file in one contiguous block.
19. When converting `fprintf` to `printf`, remove the senseless `fflush`.
20. Variable definitions are separated from the opening brace and from the code by blank lines.
21. Avoid the `?:` ternary. Use `if`/`else` when it is clearer.
22. A single-statement `if` keeps the statement on the same line without braces, or uses an Allman block with the `{` on its own line. Only one statement may follow the `if` on the same line. Multiple statements need an Allman block.
23. No `static` on file-scope globals in single-file programs.
24. Every variable gets its own line.
25. Use `0`, not `nullptr` or `NULL`.
26. Give variables descriptive names and avoid the `g_`, `m_`, and `s_` prefixes.
27. No `const` on function parameters or local variables. `const` is acceptable only when the code will not compile without it, such as a `std::string&` binding to a temporary or a `qsort` comparator signature.
28. No `long long` when `int` suffices. Use `__int64` only when the value genuinely exceeds 32 bits.
29. No unnecessary initializers.
30. Avoid unnecessary casts. When a cast is required for correctness, use the C-style syntax. Remove redundant casts such as `(int)strlen(...)` where the conversion is already safe.
31. All lowercase for variable, struct, function and type names. Only `#define` macros use UPPER_SNAKE_CASE.
32. No assignments inside `if`, `while` or `for` conditions.
33. Report errors before returning; do not fail silently.
34. Separate logically distinct blocks of code with blank lines.
35. No `void*` when a proper type exists.
36. No `&` references unless they are warranted by avoiding a copy of more than 64KB, or by changing data in the caller as a secondary return.
37. Console output must be non-aggressive, agreeable and detailed. No `=====` or `*******` separators; use `.....` dots instead. No all-caps labels, no exclamation marks, no visual aggression.
38. No senseless `= {}` zero-initialization of structs. Use `= {0}` only for arrays or when zero is the correct initial state.
39. `printf` lines should not be split across lines unless they exceed about 256 characters.
40. Use `#define` for compile-time constants that would otherwise be `const int` locals used for array sizing.
41. Indent with tabulators, one tab per indentation level. No spaces for indentation.
42. Lines may be up to 250 characters long. Do not break lines just to fit an 80 or 120 column limit.
43. No `inline` unless it is genuinely needed for performance. Remove it on sight unless there is a concrete performance reason.
44. Use C-style casts only where a conversion is required for correctness; this restates rule 30.
45. New comments use `//` only, never `/* */`. Existing `/* */` comments are not converted, but all new comments must use `//`.
