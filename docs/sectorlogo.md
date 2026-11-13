# Programming with SectorLogo

SectorLogo is a general-purpose Logo interpreter embedded in V7. It supports procedures, variables,
lists, arithmetic, and native string operations, making it suitable for small application programs as
well as text-processing tasks. The interpreter can run independently of model inference; when used as a
V7 sidecar, additional commands provide access to generated text, token candidates, sampling controls,
and MicroDB.

The manual introduces the language before describing the V7-specific interfaces. It explains expression
evaluation and data types, then covers procedures, text processing, foreign-function calls, and worked
examples. The same language syntax applies to ordinary computations, local search programs, and
callbacks that control model generation. Examples are written to illustrate those contracts rather than
to prescribe one application design.

## Contents

- [SectorLogo language and integration](#sectorlogo-language-and-integration)
- [Language quick reference](#language-quick-reference)
- [Native text processing](#native-text-processing)

---

## SectorLogo language and integration

SectorLogo uses Logo-style procedure calls and expressions to perform ordinary computations. It provides
variables, control flow, lists, arithmetic, and string operations; none of these facilities depends on a
language model being present. A program can run in a standalone interpreter session, which is useful for
text processing, testing, or general application logic.

When V7 executes a SectorLogo program as a generation sidecar, it adds host-specific values and
commands. These operations expose the current candidate token, the accepted generated text, sampling
controls, and database access. This is an extension of the interpreter's environment rather than a
different language. A short script can also call a native function in the host application through
`CallLibrary`, so the same integration point supports both interpreted decisions and callbacks
implemented in C or C++.

The first part of this chapter introduces expression syntax, values, procedures, and text processing. It
then describes the host commands and FFI contracts, followed by reference tables and worked examples.
The exact command syntax is important, especially where the parser treats a word as a callable operation
rather than a string value.

### Contents

1. [Running a script](#running-a-script)
2. [Syntax and values](#syntax-and-values)
3. [Commands](#commands)
4. [Functions and scope](#procedures-and-scope)
5. [Lists and strings](#lists-and-strings)
6. [SectorLogo and v7](#sectorlogo-in-v7)
7. [Stream rewriting](#stream-rewriting)
8. [FFI and external programs](#ffi-and-external-programs)
9. [Deviations from standard Logo](#differences-from-standard-logo)
10. [Memory and the tagged word](#memory-and-tagged-values)
11. [Limits and errors](#limits-and-errors)
12. [A/B tuner](#ab-tuner)
13. [Hosting SectorLogo](#hosting-sectorlogo)
14. [Source map and tests](#source-map-and-tests)

---

### Running a script

A SectorLogo program can run in a standalone interpreter or as a script invoked by the V7 generation
loop. Both use the same core language, but only the inference host supplies candidate-token variables
and generation-control commands.

#### Standalone

```bat
bin\run_sectorlogo.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db script.logo
```

The runner needs all three paths; it evaluates the file and returns the last `Output` value through
the runner process. The standalone interpreter does not define per-token hook variables. Scripts that
require `:Piece` or `:TokenNumber` must therefore use the inference host interface.

#### Sidecar

The sidecar is a general-purpose SectorLogo program executed through V7's hook interface. The script
determines whether to inspect a candidate, call native code through FFI, access MicroDB, reject a
candidate, or replace a previously accepted stream. An application can use a one-line FFI call as a
callback bridge. Neither the interpreter nor the presence of a sidecar implies that every model provides
thinking delimiters; the program must select trigger conditions compatible with the active model and
template.

```bat
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db ^
  -i "what is a cat?" -n 1000 -q8 1 -rewrite -script src\logo\sidecar_cats_to_dogs.logo
```

`-script` stores the file in MicroDB under `logo:sidecar`, which activates the per-token
hook at chat entry. The `-rewrite` option remains accepted for compatibility but does not
enable the hook. The host must provide the DLL and database paths; there is no default database path.

The generation hook executes after the sampler has produced and decoded a candidate token, but before
that candidate is committed to the accepted output. The host provides the following values to the
script:

| Callback variable | Description |
| --- | --- |
| `:TokenNumber` | The candidate token ID. |
| `:Piece` | The decoded candidate text, using the same conversion as the chat loop: `U+2581` is converted to a space and an invalid byte is represented as `<0xXX>`. |

When a sidecar is installed, the host evaluates it during generation using the following sequence:

1. the chat loop samples a candidate token;
2. the engine decodes it (`:Piece`) and calls `v7s_sectorlogo_on_token`;
3. the script runs with `:TokenNumber` and `:Piece` bound;
4. the script's last action decides what the engine does next.

At the top level, `Exit` terminates script evaluation. Within a function, `Output` returns a value; in a
stream context, it terminates the current stream. The hook executes **before** the EOS check, so
`Equal:TokenNumber EndToken` detects the end-of-turn boundary.

The hook only ever sees **generated** tokens. Prompt and template tokens, including the think tags the
template placed in the prefill, never pass through it, so a `Equal:Piece "</think>"` marker fires
exactly once per generated think block and never for the template's own tags.

#### Included callback programs

The `src\logo\` directory contains the example programs for different callback patterns. Use
`-script` to select a program. The same database-key activation also permits
`SetStream`; `-rewrite` remains optional and does not control callback activation.

| Program | Purpose |
| --- | --- |
| `sidecar_print_only.logo` | Prints every candidate token to verify that the hook is invoked. The optional `-rewrite` flag is not needed for callback activation. |
| `sidecar_detect.logo` | Prints candidate tokens and identifies the end-of-reasoning marker. This example demonstrates the trigger used by reasoning-context scripts. |
| `sidecar_cats_to_dogs.logo` | Demonstrates generated-stream replacement. See [Stream rewriting](#stream-rewriting). |
| `sectorlogo_rag.logo` | Uses generated reasoning text as a retrieval query, obtains BM25 records, inserts them into the reasoning section, and continues inference. |
| `sidecar_rag_negate.logo` | Evaluates whether inserting contradictory retrieved information changes the generated answer. |

Additional retrieval examples are provided in the experiment directories. They share the callback
structure of `sectorlogo_rag.logo` while using different retrieval implementations and knowledge
databases.

---

### Syntax and values

SectorLogo distinguishes between values that represent data and words that may denote commands. That
distinction affects both ordinary expressions and native integration, especially when a program
constructs text dynamically or passes a value through the FFI. The syntax rules below explain how the
parser separates expressions, how the interpreter converts numeric literals, and which values are
treated as false. The examples use the exact spellings expected by the interpreter.

#### Comments and expressions

```logo
; comment to the end of the line
Make "x" 42
Print (2 + 3)
IfElse (:x > 10) [ Print "large" ] [ Print "small" ]
```

- `[... ]` is a list or block. It is data until `Run` evaluates it.
- `(...)` groups an expression and evaluates it in the current environment.
- `"..."` is a string literal. It supports `\"`, `\\`, and `\n`. **Any other escape
  drops the backslash**: `"C:\ragg"` reads as `C:ragg`, which is why executable
  paths in scripts must use forward slashes.
- `;` starts a comment that extends to the end of the line. A comment separates
  tokens in the same way as whitespace.
- Operators and brackets separate tokens: `+ - * / % _ = < > ! & | ^ ~ ( ) [ ]`.
  The backslash is an operator token, not a path separator. Use quoted strings
  for file paths.

Procedure and variable names are case-insensitive, but the contents of string values retain their original case.

A sign attaches to a following number when it is unambiguously unary: inside a list literal (`[-3 5 -10
0]`), after a keyword head (`Maximum -5 10`), or after whitespace or a comment (`FloatRemainder 7.0
-3.0`). Only the compact operand-adjacent form (`2-3`) stays a binary operator.

#### Atom types: WORD and STRING

Each atom stores a type tag, which determines how the interpreter treats its value. The two forms
described below have different evaluation behavior:

| Type | Written as | Dispatch |
| --- | --- | --- |
| WORD | `make`, `+`, `True` | a registered builtin dispatches; a bare word may be a function call |
| STRING | `"make"`, `"+"` | never dispatches; always a self-evaluating value |

So `"+"` is the two-character string, while `+` is the addition operator. Both compare equal with
`Equal` (which compares text), but they are not the same value under `SameValue`. `AsWord` converts a
STRING to the folded WORD form, and `IsStringAtom` / `IsWordAtom` are the two primitives that tell the
kinds apart, the only commands that read the type byte.

#### Values

| Type | Example | Notes |
| --- | --- | --- |
| NUMBER | `42`, `-7` | Signed 30-bit integer |
| FLOAT | `1.0`, `-2.5` | Truncated binary32 in a tagged word; ~7 significant digits, binary32 exponent range |
| ATOM | `"hello"`, `True` | Interned text, either WORD or STRING |
| CONS | `[1 2 3]` | List cell; parses to `(1. (2. (3. NIL)))` |
| NIL | `[]`, `0` | Empty/false value |

Numbers and floats can be used together in arithmetic. The result becomes a float when either operand is
a float. Bitwise commands require integer values.

Text commands such as `Words` return atom values, while `Prepend` constructs a new CONS list. The
`SetCar` and `SetCdr` commands modify an existing CONS cell rather than creating a replacement.

#### How numeric literals are converted

Numeric literals are initially represented as atoms by the parser. The evaluator converts them to
numeric values when an operation requires one, according to the following rules:

| Source | Result |
| --- | --- |
| `42` | NUMBER |
| `1.0` | FLOAT (a `.` needs at least one digit after it: `2.` stays an atom) |
| `99999999999` | FLOAT, a literal past the 30-bit range promotes, it does not wrap |
| `1.2345678901` | stays an atom, at most 9 fraction digits parse as a literal |
| `2.`, `12abc` | stays an atom (trailing garbage) |

Arithmetic results follow a different rule from literal parsing: integer results wrap to the
interpreter's 30-bit representation, whereas the promotion rule applies to literals.

#### Truthiness

SectorLogo treats the following values as false when evaluating conditions:

- NIL;
- NUMBER `0`;
- FLOAT `0.0` and `-0.0`;
- the `False` atom.

Other values are true. `Search` returns a position or NIL, so a match at position zero must be tested
with `IsNumber`, not by using the raw result as a condition.

---

### Commands

The command reference groups the available operations by purpose, including variables and control flow,
arithmetic, text processing, collections, and host integration. This reference retains the exact command
spellings and argument forms used by the interpreter.

#### Variables and control flow

| Command | Meaning |
| --- | --- |
| `Make "name value` | Assign a value in the current environment |
| `Local "name [name...]` | Declare function-scoped local bindings (one or more names) |
| `If condition [body]` | Run `body` when the condition is true |
| `IfElse condition [then] [else]` | Choose one block; both blocks are required |
| `Repeat count [body]` | Repeat a block |
| `Repeat [body]` | Run until `Stop`, `Continue`, `Exit`, or function return |
| `Stop` | Break the current `Repeat`; inside a function, return from that function |
| `Continue` | Skip to the next `Repeat` iteration |
| `Exit` | End the current script |
| `Run block` | Evaluate a list as code |
| `PrintRaw value` | Print a value without a newline |
| `Print value` | Standard-library command; print a value and a newline |

A list immediately after `Repeat` means the count is omitted: `Repeat [ body ]` is endless. The count is
detected by parser **shape**, not by value, so `Repeat [ 5 ] [ body ]` is an endless loop whose body is
`[ 5 ]` -- the classic typo hazard. `Repeat:count [ body ]` requires a NUMBER; a FLOAT, an atom, or NIL
is an error, and a list-holding variable needs the `Repeat [ Run:v ]` wrapper. A count of zero or less
is zero trips. An endless `Repeat` with no reachable `Stop` /`Exit`/`Output` loops forever; the engine
imposes no iteration cap.

`Stop` breaks the innermost `Repeat` only. Inside a function it returns from that function with NIL and
does **not** break a caller's loop, use `Output:value` when the caller needs a result. `Exit` is the
opposite: it is a script-wide flag that remains set across function boundaries, so `Exit` inside a
called function ends the whole script. `Continue` skips one iteration, and in a **counted** loop it
still consumes a trip. A `Local` declared inside a loop belongs to the enclosing function scope and persists until that function returns.

An atom in statement position that names neither a builtin nor a defined function is silently skipped. A
misspelled command may therefore become a no-op rather than an error. Use tracing or strict-name
diagnostics to detect such mistakes.

##### Scoping

- The colon on a name is optional: `Make "x...` and `Make ":x...` are the same.
  A leading `:` is stripped and the name is lowercased, so it matches `:X`.
- **Names may contain only `A-Za-z0-9`.** A hyphen, dot, or space in a name is
  an error naming the offender, `Make "my-var 1` does not work.
- Dereferencing an unbound variable returns NIL, with no error.
- The environment is dynamic and persistent for the whole hook session: state set
  on token 0 is still visible on token 500.
- `Make` **updates the first existing binding, searching innermost frame
  outward**; it only prepends a new binding when no binding exists. This is the
  accumulator idiom: `Make "acc" 0  Define "bump" [:n] [ Make "acc" :acc + :n ]`.
- `Local` adds the name to the current function's local set *and* creates a fresh
  binding that shadows the outer one, so repeated `Make` in a loop is O(1) and
  the outer binding is untouched. The declaration remains valid until the function
  returns. Multiple names may follow: `Local "ctext" "paras" "chunks"`.
- `Local` is function-scoped, not iteration-scoped: each iteration accesses the
  same local binding. A declaration within a loop persists until the enclosing function returns.
  At the top level, it remains valid for the rest of the script. The declaration stack holds up to
  65,536 entries.

The `Local` name list stops at the first form that cannot be a name, so the next statement is not
consumed as part of the declaration:

| Ends the list | Example |
| --- | --- |
| a NUMBER, FLOAT, CONS, or NIL | `Local "a" 5` |
| a registered builtin keyword | `Local "a" Make "x" 5` |
| an infix operator | `Local "x" _ "y"` |
| a `:var` reference | `Local "a":b` |
| a function name | `Local "a" "b" myhelper 0` |

##### Call forms and precedence

A call in a value slot takes its arguments from the surrounding stream and consumes exactly its arity,
so everything a primitive evaluates as an expression (`Make`, `Output`, `SetCar`, an `IfElse` condition)
is greedy:

- `Output (2 + 3)` is 5; `Print (f 2 3)` invokes `f`; `Output inc 5` is 6.
- A function binds exactly its parameter count; the rest becomes following
  statements. `Print "a":b` prints only `a`, and the bare `:b` is a separate
  (skipped) statement.
- A **bare function name is a call**, not a value: as a parameter expression it
  invokes the function with whatever arguments remain. Use `Print "f"` when you
  want the name as data.
- A cons in a value slot is a literal: `Make "x" Prepend 1 []` stores the list.
  Only a paren group or `Run` evaluates code.

Infix operators form three levels, loosest first:

| Level | Operators | Notes |
| --- | --- | --- |
| 1 (loosest) | `_` `=` `!=` `<>` `<` `>` `<=` `>=` `&` `\` `^` `<<` `>>` | One left-associative level |
| 2 | `+` `-` | Addition and subtraction |
| 3 (tightest) | `*` `/` `%` | Multiplication, division, and remainder |

So `(2 + 3 * 4)` is 14, `((2 + 3) * 4)` is 20, `(10 - 4 - 3)` is 3, and `(1 + 2 = 3)` is true because
`+` folds first. `~` is prefix-only.

Function application binds tighter than every infix level, and the two argument paths differ:

- **Statement arguments** (`Make`, `SetStream`, `IfElse` conditions) fold the
  full level-1 set, so `Make "s" "a" _ "b"` is `"ab"`.
- **Function parameters** stop after `+ -`, so `f x _ y` is `(f x) _ y`, not
  `f (x _ y)`.

Composing generated code as `(elems...)` makes argument consumption self-delimiting, the contract the
Scheme/Lisp compilers in `examples/` emit.

#### Arithmetic

| Infix | Meaning |
| --- | --- |
| `a + b` | Addition |
| `a - b` | Subtraction |
| `a * b` | Multiplication |
| `a / b` | Integer division for two NUMBERs, otherwise float division |
| `a % b` | Remainder |

| Prefix | Meaning |
| --- | --- |
| `Power a b` | Exponentiation |
| `SquareRoot n` | Integer square root for NUMBER; float square root for FLOAT |
| `AbsoluteValue n` | Absolute value |
| `Negate n` | Negation |
| `Maximum a b` | Larger value |
| `Minimum a b` | Smaller value |
| `Random n` | NUMBER in `[0,n)`; `n <= 0` returns 0 |
| `RealRandom flag` | 1 enables / 0 disables RDTSC input for `Random`; returns the previous setting |

The five basic arithmetic operators are infix symbols only; the language does not register named prefix
equivalents. Use `Negate` for unary negation.

`*`, `/`, and `%` bind more tightly than `+` and `-`; see
[Call forms and precedence](#call-forms-and-precedence) for the full table.
Integer arithmetic wraps outside the 30-bit range. Division or remainder by zero
is an error for integer operands; floating-point division follows IEEE semantics
and may produce infinity or NaN.

`Power` uses square-and-multiply: exponent 0 gives 1, a negative integer exponent gives 0, a float
exponent goes through the float power routine, and a negative base with a non-integral or NaN exponent
is an error. `SquareRoot` of a NUMBER is an integer Newton-Raphson root and returns 0 for an argument of
zero or less; of a FLOAT it is the IEEE root, so a negative argument gives NaN. `Maximum` and `Minimum`
take exactly two arguments.

`Negate` is always unary and rejects a non-number. A unary-minus literal (`-5`) is handled by the
tokenizer's sign-gluing rules at the top of this section.

`Random` draws from an AES-NI PRNG with a fixed seed, so scripts are reproducible across runs.
`RealRandom 1` switches the generator to RDTSC entropy input for the rest of the session; `RealRandom 0`
switches it back. Each call returns the **previous** setting.

`Print` and `PrintRaw` write to the operator console rather than to the LLM stream. Use
`SetStream` for that.

#### Float and value commands

| Command | Meaning |
| --- | --- |
| `AsFloat n` | Convert NUMBER to FLOAT; an ATOM is an error |
| `AsInteger x` | Truncate toward zero and return a NUMBER (NaN gives the integer minimum) |
| `Floor x` | Round toward minus infinity, returns FLOAT |
| `Round x` | Round half-to-even, returns FLOAT |
| `IsFloat x` | True for FLOAT |
| `IsNumber x` | True for NUMBER only, a FLOAT is **not** a number |
| `IsList x` | True for CONS |
| `IsStringAtom x` | True for a quoted STRING atom |
| `IsWordAtom x` | True for a bare WORD atom |
| `IsZero n` | True for zero, including `0.0` and `-0.0` |
| `NaturalLog x` | Natural logarithm; `x <= 0` is an error |
| `LogBaseTwo x` | Base-2 logarithm; `x <= 0` is an error |
| `PowerOfTwo x` | `2^x` |
| `FloatRemainder x y` | Float remainder, always returns FLOAT (integers are converted) |
| `HashNumber "text"` | CRC32C-derived FLOAT in `[0,1)` |
| `SameValue a b` | Tagged-value identity |
| `EnableTrace level` | Set the trace level: 0 off, 1 statements, 2 also expressions, 3 verbose. The expression form returns the previous level. |
| `EnableStrictNames flag` | 1 turns an unbound name in expression position into an error instead of a self-evaluating value; 0 restores the default. The expression form returns the previous flag |
| `AsWord atom` | Convert a STRING atom to its folded WORD form |

`Floor` and `Round` return FLOAT, not NUMBER, wrap in `AsInteger` for a whole number. Comparisons return
the `True` atom or NIL, never `1` and `0`.

The logical operators are the C-style infix forms `&&`, `||` and `!` (see the bitwise table below for
the symbol list); they evaluate both operands with no short-circuit. Use `IfElse` when the second
expression must only run after a successful first test.

#### Bitwise and logical commands

| Infix | Meaning |
| --- | --- |
| `a & b` | bitwise AND |
| `a \| b` | bitwise OR |
| `a ^ b` | bitwise XOR |
| `~n` | bitwise NOT (prefix symbol) |
| `a << n` | left shift |
| `a >> n` | right shift |
| `(a && b)` | logical AND, both operands evaluated |
| `(a \|\| b)` | logical OR, both operands evaluated |
| `!a` | logical NOT (prefix symbol, binds one primary) |

Shift counts are masked to 0-31. `ShiftRight` (`>>`) is arithmetic (sign-propagating). Bitwise commands
require integer values. The bitwise and logical operators exist ONLY as these symbol words; there are no
prefix-name spellings.

#### Comparison commands

| Prefix | Infix | Meaning |
| --- | --- | --- |
| `Equal a b` | `a = b` | Structural equality |
| `NotEqual a b` | `a != b`, `a <> b` | Structural inequality |
| `GreaterThan a b` | `a > b` | Numeric comparison |
| `LessThan a b` | `a < b` | Numeric comparison |
| `AtLeast a b` | `a >= b` | Numeric comparison |
| `AtMost a b` | `a <= b` | Numeric comparison |

`Equal` compares values structurally and permits the following combinations of operand types:

| Compared | Rule |
| --- | --- |
| NUMBER vs NUMBER | by value |
| FLOAT vs anything numeric | by IEEE value, either side promoted |
| NUMBER vs a decimal ATOM | the atom is parsed as a number, so `1 = "1"` is **true** (both directions) |
| ATOM vs ATOM | by **text content**, so a STRING and a WORD with the same text are equal |
| CONS vs CONS | car compared recursively, cdr walked iteratively |
| NaN | never equal to anything, including another NaN |
| `-0.0` vs `0.0` | equal |

This ATOM/NUMBER coercion is deliberate: string values, including results from `DatabaseGet`,
`Split` and `Part`, can be compared numerically with numbers without an explicit `TextToNumber`
conversion. A
non-numeric atom such as `"s"` is still not equal to a number.

`SameValue` is a single tagged-word compare, so it is true only for the same interned atom, the same
NUMBER/FLOAT payload, or the same cons cell. Two structurally equal but separately built lists are not
the same value, and `"make"` is not `make`. This is the identity test the Scheme layer's `eq?` /`eqv?`
are built on.

---

### Procedures and scope

```logo
Define "addtwo" [a b] [
  Output :a + :b
]

Print (addtwo 2 3)
```

The parameter list is a list of names; write them space-separated (`[a b]`) or with a colon before each
(`[:a :b]`), both bind the same way. Do not write the compact `[:a:b]`: only the first parameter binds
reliably in that form. Parameters are evaluated from the caller's stream. The function sees the caller's
environment plus its parameter bindings. The language uses dynamic scope, so a function can read a
variable from its caller unless a local binding shadows it.

`Local` declares a binding for the current function (see [Scoping](#scoping) ). Repeated `Make` calls to
a local name update that binding rather than introducing another variable. For example, a procedure can
use this behavior to update a loop counter:

```logo
Define "count_to" [:limit] [
  Local "total"
  Make "total" 0
  Repeat :limit [ Make "total" :total + 1 ]
  Output :total
]
```

A function without `Output` returns NIL. Functions are first-class environment entries: the name is
bound to an invisible `(__FN__ descriptor)` value, so a call in statement position discards its return
value while a call in expression position uses it. Recursion is supported up to the current
`SL_MAX_CALL_DEPTH` limit. Exceeding the limit is reported as a SectorLogo error. `DefineCommand` binds
a `(__XF__ descriptor)` instead and behaves the same way.

#### `IfElse` as a value

`IfElse` can also be used as an expression. In expression position it returns the value of the **chosen block's
first form**:

```logo
Make "label" IfElse (:score > 10) [ "high" ] [ "low" ]; "high" or "low"
Make "n"     IfElse :have [ 2 + 3 ] [ 0 ]; 3
```

If the chosen block starts with a statement-only builtin (`Make`, `Output`, `Repeat`,...), the whole
block runs as a statement stream instead and the result is its `Output` value, or NIL. An empty chosen
block is NIL; this makes `IfElse` usable as a selector, which is why it has a separate entry in both
dispatch tables.

#### `Run` as a value

`Run` in expression position executes the block and yields, in order of preference: the block's `Output`
value, then the return value of the last function called as a statement inside it, then NIL. That second
rule makes `Run` a **dynamic apply**: `Run [ myfunc 1 2 ]` calls `myfunc` and returns its result.

A NIL block is skipped and yields NIL; a plain atom is returned as-is; a NUMBER or FLOAT block is an
error.

A block head that is a **primitive** is the trap this rule hides: the primitive runs (or is silently
skipped) and its value is **discarded**, because only a user-function call publishes a return value.
Several primitives have no statement-table entry at all (`absolutevalue`, `power`, `first`,...), so in a
block they are skipped silently and `Run` yields NIL with nothing computed. Generated code must
therefore end a value-producing block with `Output`:

```logo
Print (Run [ power 2 3 ]); warns + nothing computed, primitive head, value dropped
Print (Run [ output power 2 3 ]); 8
```

`DefineCommand` creates a function that formats its arguments and starts an external program. See [FFI
and external programs](#ffi-and-external-programs).

#### Evaluating a form in a value slot

This is the rule set a code generator has to be built on. Every command that takes a *value* (`Make`,
`Output`, `Run`, a user-function argument, `IfElse` 's test) reads its operand with the same evaluator,
so an operand is **code, not data**:

| Operand form | Result |
| --- | --- |
| NUMBER / FLOAT / NIL | itself |
| cons (`[...]`) | **itself**, a literal, never evaluated |
| atom starting with `:` | the variable's value |
| WORD atom naming a builtin | **the builtin is CALLED**, consuming its fixed arity from the stream |
| WORD atom naming a user function | the function is called |
| any other atom | itself (an unbound name evaluates to itself) |
| `(...)` | the parenthesized sub-expression is evaluated |

Three consequences that generated code depends on:

- **Variable references work inside a `Run` block.** `Run [ output:a ]` prints
  the value of `:a`, because `:a` is re-read as a source token. This is what
  makes a `Prepend`-built block executable.
- **A cons operand is a literal**, so a data constant that must never be
  interpreted (a string whose text happens to be `Maximum`, a quoted symbol) is
  smuggled through as `First` of a one-element list: the block
  `[ output first <the-cons> ]` yields the constant itself.
- **A bare builtin-name atom in an operand slot is called**, rather than
  returned as data. To construct a block containing the name of a command,
  use `(AsWord "Maximum")` as the value to `Prepend`. Its command name is
  then dispatched when `Run` evaluates the completed block.

Primitive arity is **fixed**, never variadic: `Maximum 1 2 3` consumes the first two arguments; the
trailing `3` remains as a separate statement. `Equal`, `NotEqual`, `Maximum`, `Minimum` are binary as
well. A generator must therefore emit every call with its exact operand count, and must not rely on a
primitive to swallow a nested expression.

`Make`, `Local`, `PrintRaw`, `Repeat`, `If`, `Output`, `Define`, `Stop` and `Continue` have **no
expression form** and self-evaluate in a value slot, they cannot be used to build an expression.
`IfElse` is the only control command with one.

#### Two atom types: WORD and STRING

Atoms carry a type byte, and the type decides dispatch:

- A **quoted literal** (`"Maximum"`) is a STRING atom. It always self-evaluates, even
  when its text is a builtin or a user function name: a string is data.
- A **bare name token** (`Maximum`) is a WORD atom. It dispatches.

Atoms are interned, so `"Maximum"` and `Maximum` have the same normalized text, but their type tags
differ. `AsWord` converts a STRING atom to the equivalent WORD atom (`EvalPrimary.expr_fallback` does
this conversion automatically for atoms that are not already in the builtin region, and returns the
original STRING when the folded WORD would have been a builtin). So `AsWord` is exactly the "make this
atom dispatchable" tool, which is why a generated block needs it for every command name, and why it is
the wrong tool for a data constant. Which kind an atom *is* can be tested directly with `IsStringAtom`
and `IsWordAtom`.

---

### Lists and strings

Lists and strings have different representations and supported operations. The following sections
describe how to construct and modify each value type and how the native text operations interact with
ordinary Logo expressions.

#### Lists

| Command | Meaning |
| --- | --- |
| `First list` | First element |
| `ButFirst list` | List after the first element |
| `Prepend item list` | New list with `item` first |
| `Item list index` | Element at a zero-based index, or NIL |
| `Length list` | Number of elements |
| `SetItem list index value` | Rebuilt list with one element changed |
| `SetCar cell value` | Changes the first field of one existing cell |
| `SetCdr cell value` | Changes the next field of one existing cell |
| `Sort list` | Stable ascending sort |
| `Reverse list` | Reversed copy |
| `FindItem list start key` | First exact key from `start` |
| `FindFirstAtLeast list start key` | First key greater than or equal to `key` |
| `FindFirstAtMost list start key` | First key less than or equal to `key` |

`FindItem`, `FindFirstAtLeast`, and `FindFirstAtMost` require `(list start-position key)`. Keys must be
NUMBER or FLOAT, a string key is an error. Sub-list elements use their first element as the sort/search
key. `FindFirstAtMost` returns the first matching position; it is not a last-element command. On a list
sorted by `Sort`, `FindFirstAtLeast` behaves as a lower-bound lookup.

`Sort` is stable, ascending, and non-destructive, it rebuilds fresh cells. Elements may be NUMBER,
FLOAT, a CONS sub-list sorted by its car, or a plain ATOM, which is parsed as a number. A CONS or NIL
*element* is an error. So `[[3 "c"] [1 "a"]]` sorts on the sub-list car, the standard `[score, payload]`
pair pattern for BM25 ranking.

`First` and `ButFirst` return NIL for an empty list, which is a legitimate result, but they report an
error naming the value when given a NUMBER, FLOAT, or ATOM instead of a list.

`Prepend` creates a new list cell and does not change the old list. A list built by repeated `Prepend`
is in reverse construction order. Use `Reverse` when the result must be in reading order.

`SetCar` and `SetCdr` are the only destructive list primitives; they return NIL and the first argument
must be a genuine non-empty cell (the error names the offending value). They change the existing
structure, so the mutation is visible through **every** holder of that cell, `SetCar ButFirst:L 42`
changes element 1 of `:L`, because the tail is not a copy. A `SetCdr` can build a cycle; the collector
handles cycles, but walking one with `Length` /`Sort` never terminates. Do not mutate cells of a literal
list: like quoted literals in Scheme they are shared constants, so build with `Prepend` first.

#### Strings and text

SectorLogo strings are ATOM values. Every text primitive accepts NIL or an atom; a NUMBER, FLOAT, or
CONS argument is an error naming the command and the offending value.

| Command | Meaning |
| --- | --- |
| `Part source start length` | Byte substring, zero-based |
| `Search needle source` | Byte position or NIL |
| `Lowercase source` | ASCII lowercase conversion |
| `Uppercase source` | ASCII uppercase conversion |
| `TextLength source` | Byte length, excluding NUL |
| `Replace source old new` | Replaces the first occurrence only |
| `TextToNumber source` | NUMBER, FLOAT, or NIL |
| `Words source` | Lowercase alphanumeric runs of at least two characters |
| `Split source delimiter` | Delimited segments |
| `NGrams source count` | Full sliding windows over text or a list |
| `a _ b` | String concatenation |

`Search` returns zero for a match at the beginning, so use `IsNumber` when checking whether it found
anything. `Replace` is deliberately a first-match command. A replace-all helper is:

```logo
Define "replace_all" [:text :old :new] [
  Make "result" :text
  Repeat [
    Make "found" Search :old :result
    IfElse IsNumber :found [
      Make "result" Replace :result :old :new
    ] [
      Output :result
    ]
  ]
]
```

`a _ b` converts numbers to decimal text, so `"n=" _ 42` is `"n=42"`; NIL concatenates as empty, and `_`
is left-associative and chains. A result longer than the 2 MiB concatenation buffer is an error, never a
truncated atom. Atoms are interned by content, so `=` on strings that came from the same source is an
O(1) offset compare.

`Part` clamps rather than failing: a length of zero or less gives the empty string, a negative start is
treated as 0, and a start past the end gives the empty string.

`TextToNumber` accepts an optional `-`, integer digits, and an optional `.frac` part (which yields a
FLOAT). Anything else, trailing garbage, an empty string, a lone `.`, returns NIL. A NUMBER or FLOAT
argument passes through unchanged, because a bare numeric literal in expression position is already
parsed as a number before the function sees it; the string-parse path fires for values that arrived as
real string atoms. `(TextToNumber Item [10 20 30] 1)` is `20`.

`Words` returns text in reading order, one native call instead of a per-character loop. It lowercases,
splits on non-alphanumeric boundaries (digits are word characters), drops runs shorter than two
characters, and returns NIL when no values remain. It has **no** stopword list and **no** bigram policy
-- that stays in the SectorLogo program so retrieval policy stays editable without an ASM rebuild. A
non-string argument is an error.

```logo
Make "words" Words "The cat, night vision!"
; "the" "cat" "night" "vision"
```

Because atoms are interned, `"Cats"` and `"cats"` from different text positions become the same atom, so
later `Equal` comparisons and hash-map keying see identical words. A stopword pass is one `Search` per
word over the stopword list (~1.5 ms), far cheaper than the tokenize itself:

```logo
Define "tokenize" [:text] [
  Make "toks" Words :text
  Make "out" []
  Repeat [
    IfElse (SameValue :toks []) [ Stop ] []
    Make "word" First :toks
    Make "toks" ButFirst :toks
    IfElse IsNumber (Search :word :stopwords) [] [ Make "out" Prepend :word :out ]
  ]
  Output (Reverse :out)
]
```

`Split` scans left to right and emits the text between delimiters as a cons list. Adjacent delimiters
yield empty segments (`"a,,b"` is `"a" "" "b"` ) and the empty string yields a one-element list,
matching PHP `explode`. An empty delimiter splits into single characters. `NGrams` accepts a NUMBER
count of at least 1: a string source gives n-character byte substring atoms, a list source gives
n-element sublists, text order, full windows only. A source shorter than the count returns NIL. The BM25
bigram stage is then one call per token list: `NGrams Words:text 2`.

#### Approximate hash list

SectorLogo does not provide a built-in map command. For a small experiment, hash text with `HashNumber`, store `[hash key]` pairs,
sort them, and use `FindFirstAtLeast`:

```logo
Make "table" []
Repeat:count [
  Make "word" Item:keys:index
  Make "pair" Prepend (HashNumber:word) Prepend:word []
  Make "table" Prepend:pair:table
]
Make "table" Sort:table
Make "position" FindFirstAtLeast:table 0 (HashNumber "target")
```

Collisions still require comparing the original key; this pattern is for small lists, not a
replacement for MicroDB.

---

### SectorLogo in V7

The V7 host commands expose candidate-token information, sampling controls, stream replacement, and
database operations. These functions belong to the embedding application rather than to the
general-purpose language.

#### Standard library

`autoexec.logo` is loaded once at the first hook call, from the first available source: the
`__sectorlogo__autoexec` MicroDB key (written by `build_db.exe -autoexec <path>` ), then an
`autoexec.logo` file in the current directory. If neither exists, engine primitives remain available, an
informational notice is printed, and unbound stdlib commands are skipped statements rather than errors.
A stdlib larger than 1 MiB is an error, never silently shortened.

The stdlib is evaluated once into a shared base environment that both the per-token hook and the
standalone runner see. The script source limit is 1 MiB (`SL_SCRIPT_BUF_SIZE`); usable script text is
one byte less, because every load path reserves a NUL. An oversize script is a terminal error, never a
silent truncation.

| Command | Meaning |
| --- | --- |
| `Print value` | `PrintRaw` plus CRLF |
| `While cond-block body-block` | Loop while the condition block outputs a truthy value |
| `ClearScreen` | Clear the screen and home the cursor (`ESC[2J` `ESC[H`) |
| `SetCursor col row` | Position the cursor, 1-based |
| `SetPaletteColor n` / `SetPaletteBackground n` | Foreground / background from the 16-color palette (0-15) |
| `SetColor r g b` / `SetBackground r g b` | 24-bit truecolor, falling back to the nearest palette entry when VT is unavailable |
| `ResetColors` | Reset all attributes (`ESC[0m`) |
| `NearestColor r g b` | Palette index closest to the RGB triple |
| `NearestColorAll r g b` | All 16 entries as `[index distance]` pairs, ascending |
| `ConsoleSelfTest` | Print a red/green/blue line and a cursor test |
| `PrintBanner label` | Centered truecolor banner text |
| `Sequence code` | Emit one raw SGR code (helper behind the color setters) |
| `ForegroundCode n` / `BackgroundCode n` | SGR code for a palette index |

Constants bound at stdlib load: `:esc` `:cr` `:lf` `:crlf` `:vt24` (the `EnableColors` result) and the
key sequences `:keyescape` `:keyup` `:keydown` `:keyright` `:keyleft`. Palette index atoms `:black`
`:blue` `:green` `:cyan` `:red` `:magenta` `:yellow` `:white` are bound too.

#### Console primitives

Output is raw VT escape sequences; Windows Terminal renders them natively and `EnableColors` turns VT
processing on for legacy conhost.

| Command | Meaning |
| --- | --- |
| `Character n` | Code `n` (0-255) as a one-character atom; out of range is an error |
| `Number "s"` | First character's code; an empty string is an error |
| `ReadKey` | **Non-blocking** keyboard poll: NIL if nothing pending, else the pending bytes (1-64 chars; arrow keys return their whole escape sequence as one atom) |
| `EnableColors` | Probe and enable: True if 24-bit RGB is supported, NIL when output is redirected or VT is unavailable |
| `NearestMatch full vector codebook` | Least-squares codebook search over at most 256 components per row: `full = false` returns the closest row as `[key distance]`, `full = true` returns all rows ascending by distance. Ties go to the first-encountered row. |

`ReadKey` never waits, the poll timeout is zero by contract, because the hook runs inside the per-token
loop and one blocked poll would stall a token. It handles a real console, a pipe, and a redirected file,
and on the first console call it applies a one-time raw-mode fixup (clearing the line, echo, window, and
mouse mode bits) so a single keystroke returns immediately. That fixup is permanent: an interactive host
that reads lines from stdin afterwards will not get them. Escape versus arrow disambiguation is
script-side, poll again, and more bytes means it was a sequence prefix while nothing means it was a bare
escape.

With output redirected the demo still runs: VT is off, so truecolor falls back to palette codes and the
escapes land as raw bytes.

```logo
ClearScreen
SetPaletteColor 14
Print "cyan text"
SetColor 178 102 255
Print "violet text"
ResetColors
SetCursor 5 12
Print "positioned at column 5, row 12"
IfElse (= ReadKey :keyescape) [ Exit ] []
```

`examples\sectorlogo\console_demo.logo` is a runnable tour of every command.

#### Host commands

The hook variables are `:TokenNumber` and `:Piece` (see [Running a script](#running-a-script) ). The
in-system tuner scripts additionally bind `:TunerDecision` to `"accept"` or `"reject"` (see [A/B
tuner](#ab-tuner) ).

| Command | Meaning |
| --- | --- |
| `GetStream` | Generated text already appended during the current turn; the pending candidate is available separately through `:Piece` and `:TokenNumber`. |
| `StreamLength` | Byte length of the current stream (O(1)) |
| `SetStream text` | Replace the stream and start the rewrite path |
| `Reject` | Discard the candidate and sample again |
| `Halt` | Stop generation at this point; the candidate is not emitted |
| `Position` | Current sequence position |
| `GetToken n` | Token ID at a history position, or NIL out of bounds |
| `VocabularySize` | Vocabulary size |
| `ModelDimension` | Model hidden dimension |
| `LayerCount` | Transformer layer count |
| `EndToken` | End-of-sequence token ID |
| `SetTemperature value` | Integer percentage divided by 100, written for the next sample: `SetTemperature 80` sets 0.80 |
| `SetNucleus value` | Integer percentage divided by 100, written for the next sample: `SetNucleus 95` sets 0.95 |
| `DatabaseGet key` | Read a MicroDB entry, or NIL if absent |
| `DatabaseSet key value` | Add or update a MicroDB entry; always returns `True` |
| `EncodeWord text` | The first tokenizer ID for text |

The ordered comparisons on `:TokenNumber` are the bounds check: the hook executes before the EOS check,
so `Equal:TokenNumber EndToken` detects the end of the turn.

`GetStream` and `SetStream` are the commands used by RAG and rewrite examples. The current stream
contains accepted generated text for the current turn, excluding the pending candidate and the prompt or
template tokens.

`DatabaseSet` accepts an ATOM, NUMBER or FLOAT value (numbers are converted to decimal text)
and requires a non-empty ATOM key. NIL or CONS values and non-ATOM keys are errors. On success,
it returns `True`; the storage layer reports failures explicitly. Entries are temporary:
they are not written to disk on save. Use `TextToNumber` when a numeric value is needed later.

`SetTemperature` and `SetNucleus` take an **integer percentage** only. A FLOAT argument is an error, so
`SetTemperature 0.8` fails while `SetTemperature 80` works. The value is stored as given divided by 100
and the sampler owns the range check from there; it takes effect on the next sample, so the pattern is
"set the parameters, then resample":

```logo
If (= :Piece "idiot") [
  SetTemperature 40
  Reject
]
```

#### Host primitives in a standalone run

A standalone run has no inference context, so several primitives cannot work. This is a property of the
run, not a script bug:

| Command | Standalone behavior |
| --- | --- |
| `GetStream`, `StreamLength`, `DatabaseGet`, `DatabaseSet`, `Position` | work (`Position` is 0) |
| `GetToken` | NIL, there is no history |
| `SetTemperature`, `SetNucleus` | silently ignored, there is nothing to write |
| `SetStream`, `Reject`, `Halt` | no effect on generation |
| `EncodeWord`, `VocabularySize`, `ModelDimension`, `LayerCount`, `EndToken` | **error**, these need the live model |

Test tokenizer, string, and list logic with the standalone runner; test anything that reads model state
from a sidecar.

#### Example: mid-generation knowledge injection

A script can call `DatabaseGet` to retrieve data while generation is in progress and then use
`SetStream` to incorporate that data into the context. The sequence is illustrated below: when the
model's own output triggers a condition, pull fresh context from MicroDB and splice it into the stream.
No re-prompt, no second forward pass for the user to wait on, no pipeline change.

```logo
If (= :Piece "</think>") [
  Make "facts" DatabaseGet ("facts:" _ :topic)
  IfElse (! SameValue :facts []) [
    SetStream (GetStream _ "\n" _ :facts _ "\n")
  ] []
]
```

#### Example: inspect a token

```logo
Print :TokenNumber
Print :Piece
```

#### Example: record a local value

```logo
IfElse (= :Piece "example-marker") [
  DatabaseSet "example-seen" "yes"
] []
```

---

### Stream rewriting

`SetStream` replaces the generated text for the current turn. To preserve a consistent transformer
context, V7 performs the following operations:

1. compares the old and replacement text;
2. finds the common byte prefix;
3. rewinds token history and the KV cache to the changed suffix;
4. clears presence-penalty state for discarded tokens and restores it for
   retained tokens;
5. truncates the output buffer;
6. encodes the replacement suffix;
7. forwards replacement tokens through the active F32 or Q8 path;
8. resumes sampling.

If the replacement is a strict prefix and adds no token, the engine forwards the last retained token
once to obtain fresh logits. A later `SetStream` call compares against the current stream, so more than
one rewrite can occur in a turn.

#### Canonical example

```logo
Define "replace_all" [:text :old :new] [
  Make "result" :text
  Repeat [
    Make "found" Search :old :result
    IfElse IsNumber :found [
      Make "result" Replace :result :old :new
    ] [
      Output :result
    ]
  ]
]

IfElse (= :Piece "</think>") [
  Make "rewritten" GetStream
  Make "rewritten" replace_all :rewritten " cats " " dogs "
  Make "rewritten" replace_all :rewritten " cat " " dog "
  SetStream :rewritten
] []
```

The order of the two replacements matters. The surrounding spaces are part of the example's matching
rule, they also keep the pattern from matching inside other words such as "category" or "catastrophe".

#### RAG reasoning-context augmentation

The principal sidecar, and the pattern the research experiments build on. When the think block closes,
the model's generated reasoning text becomes the retrieval query, and the retrieved knowledge lands
inside the think block before it closes:

```logo
IfElse (= :Piece "</think>") [
  Make "S" GetStream
  Make "Q" Replace :S "<think>" ""
  Make "R" CallLibrary ["SELF" "v7_execute" "pointer" ["pointer" "pointer" "pointer" "pointer"]] "examples/rag/bm25_rag.exe" "search" :Q []
  IfElse (= :R "") [] [
    Make "A" :S _ "\n" :R _ "\n</think>"
    Print "=== RAG QUERY ==="
    Print :Q
    Print "=== RAG ANSWER ==="
    Print :R
    SetStream :A
  ]
] []
```

The example relies on the following callback and string-processing behavior:

- the query is the **thought**, not the user prompt, a far richer retrieval
  signal;
- `GetStream` does not yet contain the closing tag, because the hook runs before
  the candidate is accepted, so the replacement supplies it;
- empty retrieval (`:R` is `""`) means no rewrite at all, so a missing retrieval
  tool degrades to plain inference and never crashes;
- `"\n"` produces real linefeeds, so each retrieved line lands on its own row
  inside the think block;
- forward slashes in the executable path are mandatory, because the reader
  only knows `\"`, `\\`, and `\n` as escapes and silently drops any other
  backslash (`"\r"` reads as `r`);
- the `Print` blocks are the observability contract: query, answer, and rewrite
  all readable off the console.

Give `-n` enough budget for the re-prefill (1000 or more for cat-scale thoughts), and run from the
project root so the relative path resolves.

The argument to `SetStream` must be a string atom, or NIL for empty; numbers and lists are an error. The
replacement may be shorter, longer, or completely different. Rewriting is mode-agnostic and works
with both the F32 and Q8 forward paths.

The operation-level specification with pseudocode for every operation is in
[exact_rewriting_algorithm.md](generation.md#stream-replacement-and-state-consistency).

---

### FFI and external programs

The foreign-function interface allows a SectorLogo script to invoke an exported function in a native
DLL, including a function implemented by the surrounding application. A short sidecar script can
therefore call directly back into its host without requiring a second, dedicated callback API. The call
still follows the documented type descriptors, calling convention, and ownership rules. Application code
should keep those contracts explicit, particularly when passing pointers or sharing writable data.

#### `CallLibrary`

`CallLibrary` invokes an exported function using the Windows x64 calling convention. A descriptor
specifies the target module, function, result type, and argument types:

```logo
CallLibrary [ "user32" "MessageBoxA" "integer" [ "pointer" "text" "text" "integer" ] ] 0 "Hello" "SectorLogo" 0
```

Each FFI descriptor contains the following fields:

1. DLL name;
2. exported procedure name;
3. return type: `"integer"`, `"pointer"`, `"text"`, or `"void"`;
4. a list of argument types.

Quote the module name and the procedure name: the tokenizer lowercases bare identifiers, and `_` and `.`
are token delimiters, so `[SELF v7s_get_num_workers integer []]` resolves `LoadLibraryA("self")` and
splits the export name. Write `["SELF" "v7s_get_num_workers" "integer" []]`. The type names are quoted
string literals too, and they are compared by atom identity: the four registered type atoms are
`"integer"`, `"pointer"`, `"text"`, `"void"`, all lowercase. A bare or PascalCase spelling (`Integer`,
`VOID` ) interns as a different WORD atom and fails the return-type check with a named error before any
native call is attempted.

| Type | Marshalling |
| --- | --- |
| `"integer"` | 30-bit SectorLogo number, sign-extended to 64 bits |
| `"pointer"` | a raw integer used as a pointer, or NIL for C NULL |
| `"text"` | an atom; the callee receives `char*` pointing into the interned string |
| `"void"` | returns NIL |

| Return | Result |
| --- | --- |
| `"integer"` | low 32 bits wrapped as a NUMBER |
| `"pointer"` / `"text"` | the returned `char*` interned as a new atom; NULL becomes NIL |
| `"void"` | NIL |

An unknown return or argument type is reported at resolve time, before any native call is attempted. An
`"integer"` descriptor on a void function marshals whatever happens to be in `eax`, so void v7 exports
must be declared `"void"`.

`"SELF"` resolves through the DLL's own HMODULE, so a script can call any v7 export without knowing the
load address:

```logo
Make "n" CallLibrary [ "SELF" "v7s_get_num_workers" "integer" [] ]
```

`"HOST"` resolves against the host EXE, which registers itself once with
`v7s_set_caller_hmodule(GetModuleHandle(NULL))`. Without that registration a `"HOST"` call fails to
resolve. A sidecar that can only look at text is a filter; one that can call its host is an agent.

`tools/slffi_test_dll.c` (built to `bin\slffi_test.dll` by `build.bat` ) is the conformance reference
callee: deterministic endpoints covering 0-5 arguments and every return type, including the
stack-passing path beyond four arguments. The validator drives it from inline scripts.

Rules and limits: the maximum is 16 arguments; stack args beyond the first four follow the Win64 ABI;
the DLL is `LoadLibraryA` 'd by name and stays loaded; nested `CallLibrary` calls are safe because all
invoke state is stored in a per-call frame. There is no callback or struct marshalling, pointers pass
through as integers.

#### `DefineCommand`

For an external program, `DefineCommand` is usually clearer: it formats the arguments, spawns the
executable, and returns the captured stdout+stderr as an interned string atom.

```logo
DefineCommand "rag_search" [:query :options] "bin\bm25_rag.exe" "search %s \"%s\""
Make "answer" rag_search "what is a cat?" "maxlines=5"
```

| Specifier | Accepts | Output |
| --- | --- | --- |
| `%s` | string atom | string content, no quotes added |
| `%d` | NUMBER | decimal integer |
| `%c` | NUMBER | low byte as a character |
| `%f` | FLOAT or NUMBER | float32, fixed point, trailing zeros and a bare `.` trimmed |
| `%%` | no argument | a literal `%` |

The format string owns all quoting, use `\"%s\"` when an argument needs quotes on the command line; it
supports up to 8 typed varargs.

Diagnostics from the spawned program should go to a log file, not to stdout, because the captured output
is what gets returned and may be inserted into the model stream. Run from the project root so the
relative executable path resolves, and use forward slashes: an unrecognized escape like `\r` in a path
silently drops the backslash.

---

### Differences from standard Logo

SectorLogo follows Logo conventions but does not attempt to reproduce every feature of a particular
historical Logo system. The documented differences reflect the interpreter's expression evaluator, value
representation, and embedded-runtime requirements. Code written for another Logo implementation may
therefore need small changes. This chapter explains the relevant differences explicitly so that programs
can be adapted without relying on assumptions about standard Logo behavior.

| # | Standard Logo | SectorLogo | Why |
| --- | --- | --- | --- |
| 1 | `"hello` is a leading quote running to whitespace | `"hello"` is C-style, with a closing quote and `\\` `\"` `\n` escapes | sidecar scripts need strings with spaces; unambiguous tokenization |
| 2 | `Make "X 42`, the name is a quote-word | `Make "X" 42`, the name is a full string literal | one lexer rule |
| 3 | `(WORD "a "b)` | `a _ b`, infix underscore | shorter, chains left, reads like every other infix op |
| 4 | `TO name:x... END` | `Define "name" [:x] [...]`, no `END` keyword, the body block closes the definition | the parser never needs `END` |
| 5 | `STOP` exits the procedure | `Stop` is C `break` on the innermost `Repeat`; inside a `Define` it returns from that function with NIL | standard Logo procedure-return at the function boundary; loops are the dominant structure in token hooks |
| 6 | No direct standard Logo equivalent | `Continue` continues the innermost `Repeat` loop | provides explicit loop control |
| 7 | `STOP` also exits the program | `Exit` exits the entire script | the replacement for classic STOP-exits-program |
| 8 | strict left-to-right: `2+3*4` is 20 | C precedence: `* / %` bind tighter than `+ -`, parens override | programmers expect it |
| 9 | floats everywhere | 30-bit integers plus 30-bit truncated binary32 floats, auto-promoted in arithmetic | float math for BM25 scoring and sampler control |
| 10 | `EQUALP` is the equality | `Equal` / `=`, plus `!=` `<>` `>=` `<=` infix | C comfort |
| 11 | lists auto-evaluate | `[... ]` is always data; only `Run` or `()` evaluates | removes the quote/double-evaluation wart |
| 12 | `IFELSE cond [a] [b]`, else optional | the else block is required (use `[ ]`) | arity errors surface at the definition site |
| 13 | error invokes a Logo error handler | contract violations are fatal: the process exits with a diagnostic | a broken sidecar must never silently corrupt generation |
| 14 | `WORD`/`SENTENCE` list constructors | only `Prepend`, `_`, and bracket literals | minimal surface; those cover every tree shape |
| 15 | `OUTPUT` only in procedures | `Output` anywhere; it terminates the current stream evaluation | the whole sidecar is one stream, so a top-level `Output` ends it |
| 16 | case-sensitive in some dialects | case-insensitive atoms and variables | the tokenizer folds `A-Z` to `a-z` at store time |
| 17 | `(LIST:h...)` builds a map | no map primitive: hash with `HashNumber`, sort, `FindFirstAtLeast` | no `List` exists, and passing functions as arguments is impossible |
| 18 | any character in a name | names are `A-Za-z0-9` only | the environment is an association list keyed by interned atoms, so a name must be a single internable token |
| 19 | numbers are a type at read time | every literal is an ATOM converted on demand; a literal past 30 bits promotes to FLOAT while arithmetic wraps | one parser for atoms and numbers, and the lexer needs no numeric state |
| 20 | one string type | atoms carry a type: a quoted `"+"` is data, a bare `+` is the operator | lets a script hold operator text without it being executed |
| 21 | No corresponding atom-type predicate | the type byte is queryable with `IsStringAtom` / `IsWordAtom` | the Scheme/Lisp layers in `examples/` need to tell a quoted symbol from a name |

---

### Memory and tagged values

The interpreter allocates a single memory block when it is first initialized and keeps that block at a
stable address. Tagged values, list cells, and the evaluator's internal state use the layouts described
below. These details matter to the native implementation and its FFI; ordinary SectorLogo programs
interact with the corresponding language values rather than their underlying memory representation.

```
[ 64 MiB atom area ][ 8 MB cons area (1,000,000 cells x 8 bytes) ]
   interned strings      car/cdr dword pairs
```

`r13` permanently pins the base. Every value is a 32-bit tagged word:

```
value == 0             = NIL
bits 31:30 clear       = ATOM   (byte offset of the NUL-terminated string)
bit31 clear, bit30 set = NUMBER (30-bit two's complement in bits 29:0)
bits 31:30 both set    = FLOAT  (top 30 bits of IEEE binary32 in bits 29:0)
bit31 set              = CONS   (cell index in bits 30:0)
```

Atoms are interned by content, so `Intern("cat")` returns the same offset every time and string equality
is an O(1) dword compare. Strings are immutable, and their storage is reclaimed: an atom that no root
can reach is collected and its space re-used. What is stable is the *content* identity, not the offset's
lifetime, an offset may name a different atom after a collection, which is why the collector never moves
a live atom.

Both heaps are reclaimed by one mark-sweep collector. **Mark** walks the environment alist, the parsed
program tree and the LOCAL frame set, then scans the native stack conservatively for tagged references,
and marks every offset below the primitive-dispatch high-water mark as permanently live (the dispatch
tables are indexed by raw offset, so those atoms can never be reclaimed or moved). **Sweep** threads
unmarked cells onto a freelist, for atoms it coalesces each gap between two live atoms into one free
block. **Allocation** pops the freelist first and bump-allocates second; cons cells collect on
exhaustion, atoms collect at a soft limit with a rearm window so one collection does not immediately
trigger the next. A cons collection that reclaims nothing is fatal; an atom collection that reclaims
nothing is not, because the terminal case there is the area running out rather than the sweep finding no
work. The collector handles cycles. Scripts never see the GC: there are no GC primitives, no weak
references, and no finalizers.

Bounded buffers:

| Buffer | Size | Purpose |
| --- | --- | --- |
| script source | 1 MiB | script text |
| token staging | 1 MiB | lexer tokens and long-string staging |
| concatenation | 2 MiB | `_` concatenation and FFI name staging |
| current stream | 1 MiB | generated text |
| replacement text | 1 MiB | rewrite target |
| replacement tokens | 16,384 IDs | re-encoded rewrite suffix |

Recursion runs on a separate 64 MiB private stack whose first page is `PAGE_NOACCESS`, so runaway
recursion faults into the crash disassembler instead of silently corrupting adjacent memory. Two
independent guards apply: 65,536 user-function call frames and 262,144 parser levels.

---

### Limits and errors

| Limit | Value |
| --- | --- |
| Script source | 1 MiB |
| String staging | 1 MiB |
| Concatenation scratch | 2 MiB |
| Stream and replacement | 1 MiB each |
| Replacement token buffer | 16,384 IDs |
| FFI arguments | 16 |
| User-function arguments | 16 |
| `DefineCommand` arguments | 8 |
| User-function recursion | 65,536 frames (`SL_MAX_CALL_DEPTH`) |
| Parser recursion | 262,144 levels (`SL_MAX_PARSE_DEPTH`) |
| `Local` declarations in scope | 65,536 |
| Cons cells | 1,000,000 before collection and exhaustion checks |
| NUMBER range | -536,870,912 through 536,870,911 |
| Float literal fraction digits | 9 |
| Sort size | 65,536 elements |
| `NearestMatch` components per row | 256 |
| `ReadKey` result length | 64 characters |
| Variable name characters | `A-Za-z0-9` only |

Contract violations are reported with a SectorLogo diagnostic. Examples include division by zero, the
wrong type for a string command, an unterminated string, an unknown FFI descriptor, an invalid list cell
for `SetCar` or `SetCdr`, and exceeding a size or recursion limit. The error text should identify the
command and the offending value. There are no silent fallbacks anywhere in the interpreter.

A missing standard library is not a fatal error: engine commands remain available, but standard-library
functions are not defined. An oversized standard library is an error and is not silently shortened.

---

### In-process cycle tuner

The engine tunes its own matrix-multiplication dispatch configuration during
normal inference with an in-process cycle tuner written in assembly
(`cycle_tuner_main.inc`); it is always active in every program that runs
inference and has no enable/disable switch. It needs no Logo scripts: each
active shape owns a genome population whose measurement episodes (a
baseline/candidate/baseline `rdtsc` bracket) promote faster configurations
and mutate new candidates from the pool. Accepted configurations persist in
`v7_tune_param_<n>_<d>` files and the `v7_tuning_journal`; hosts observe the
tuner through the `v7_get_best_fitness` / `v7_get_pool_fitness` /
`v7_get_active_shape_count` exports. Parameters, measurement rules and the
tuner's design are documented in the tuning manual
([tuning.md](tuning.md#dispatch-and-tuning)).

SectorLogo remains the host for the StreamingLLM eviction hook:

| Key | When it runs | Purpose |
| --- | --- | --- |
| `on_stream_evict` | every StreamingLLM displacement batch (256 window displacements, turn end) | read `stream_remove_text` / `stream_remove_meta` via `DatabaseGet`, persist what the long-term memory keeps (see docs/native_api.md, StreamingLLM section) |

---

### Hosting SectorLogo

Embedding SectorLogo requires access to the interpreter's public entry points and, for some
integrations, additional native functions. The table below identifies the documented exports and their
roles. Exact signatures are defined in `src/v7_lib.h`, with exported symbol names listed in
`src/v7_threading.def`; the tables are explanatory references rather than substitutes for those
declarations.

| Export | Purpose |
| --- | --- |
| `v7_db_add` | Installs or replaces a script under the MicroDB key `logo:sidecar`. A script stored under this key activates the token callback automatically at chat entry. |
| `v7s_sectorlogo_on_token` | Internal generation-loop callback. It returns the requested candidate-token operation to the inference loop. |
| `v7s_sectorlogo_interp` | Parses and evaluates an ordinary SectorLogo script. |
| `v7s_sectorlogo_eval_script` | Evaluates a script stored under a specified MicroDB key. |

The following complete example installs a simple callback script through the current public DLL API. The
script prints each pending token piece, and the program checks initialization and engine status. For
additional return-value contracts and cleanup requirements, consult the [API
reference](native_api.md#c-api-reference).

```c
#include "v7_lib.h"
#include <string.h>

int main(void) {
    const char *script = "Print :Piece";
    char output[32768] = {0};
    v7_lib dll;
    if (v7_load("v7_threading.dll", &dll) != 0)
        return 1;
    if (!dll.v7_init("lz4ht.db")) {
        v7_unload(&dll);
        return 1;
    }
    if (!dll.v7_db_add("logo:sidecar", (void *)script,
                       (int)strlen(script) + 1, 0)) {
        dll.v7_free_inference();
        v7_unload(&dll);
        return 1;
    }
    if (!dll.v7_initialize_inference()) {
        dll.v7_free_inference();
        v7_unload(&dll);
        return 1;
    }
    dll.v7_chat("What is a cat?", NULL, output, sizeof(output), 1);
    int failed = dll.v7_engine_is_dead();
    dll.v7_free_inference();
    v7_unload(&dll);
    return failed ? 1 : 0;
}
```

The chat entry inspects `logo:sidecar` and refreshes the parsed callback when the script text changes.
Removing the key disables the callback. `inference_example.exe` installs the script through `-script`;
its `-rewrite` option remains accepted, but callback activation is determined by the database key rather
than by an additional setter.

The Logo runtime initializes its memory, interning structures, and script evaluation state on demand.
The chat-entry refresh checks script changes separately from per-token evaluation. The action code,
rewrite flag, and Stop/Continue/Output state are zeroed at the top of every call, while the environment
and the database persist. The hook executes on the chat thread. A blocking host call, including a
`MessageBox`, therefore pauses generation until that call completes.

---

### Source map and tests

| File | Responsibility |
|---|---|
| `sectorlogo_main.inc` | Includes, `lg_init`, builtin interning, the three entry points |
| `sectorlogo_const.inc` | Tags, action codes, heap sizes, limits, struct offsets |
| `sectorlogo_data.inc` | Registered names, BSS interpreter state, and the hook variables |
| `sectorlogo_reader.inc` | Characters, tokens, and list parsing |
| `sectorlogo_print.inc` | Output, integer and list printing, token decoding |
| `sectorlogo_lisp_core.inc` | Values, lists, `Car`/`Cdr`/`Cons`, and the tagged-word helpers |
| `sectorlogo_intern.inc` | Atom interning, the dual-hash bloom filter, and the slot cache |
| `sectorlogo_valstr.inc` | Value-to-text rendering (`lg_value_to_str` and the printer helpers) |
| `sectorlogo_equal.inc` | Structural equality and numeric coercion |
| `sectorlogo_gc.inc` | The mark-sweep collector for atoms and cons cells |
| `sectorlogo_fn.inc` | Function calls, parameters, `DefineCommand`, the recursion guard |
| `sectorlogo_sort.inc` | `Sort`, `Reverse`, the `Find*` scans, `SetCar`/`SetCdr` |
| `sectorlogo_string.inc` | `Part`, `Search`, case conversion, `Replace`, `TextToNumber`, `Split` |
| `sectorlogo_str_tokens.inc` | `Words` |
| `sectorlogo_split_ngrams.inc` | `NGrams` |
| `sectorlogo_str_char_to_int.inc` | `Number` |
| `sectorlogo_str_int_to_char.inc` | `Character` |
| `sectorlogo_float.inc` | Float values and math, and float promotion in arithmetic |
| `sectorlogo_do_vq.inc` | `NearestMatch` |
| `sectorlogo_console_24bit.inc` | `EnableColors` |
| `sectorlogo_console_getchar.inc` | `ReadKey` |
| `sectorlogo_ffi.inc` | `CallLibrary` resolution and invocation |
| `sectorlogo_v7prim.inc` | Stream, database, model, and sampler host commands |
| `sectorlogo_eval.inc` | Stream evaluation, statement and expression dispatch |
| `sectorlogo_eval_dispatch.inc` | The statement table |
| `sectorlogo_eval_stmt.inc` | Statement handlers |
| `sectorlogo_eval_primary.inc` | Primary expressions and the expression table |
| `sectorlogo_eval_infix.inc` | Infix operator handling |
| `sectorlogo_eval_ops.inc` | Arithmetic, comparison, bitwise, and list operators |
| `sectorlogo_eval_macros.inc` | Shared evaluator macros |
| `chat.inc` | Hook dispatch, the rewrite engine, and the tri-array stream history |
| `autoexec.logo` | Standard-library definitions |
| `tools/v_sectorlogo_validate.cpp` | Language and host-primitive tests |
| `tools/slffi_test_dll.c` | The FFI conformance callee |
| `examples/sectorlogo/` | Runnable scripts and the FFI host-callback example |
| the rewrite procedure (dev-only `attic\docs_plans\`) | Rewrite procedure |
| `generation.md` | Operation-level rewrite specification |

#### Register convention

```
r13  = pinned heap base       (set once per entry point, never modified inside)
r12d = current stream         (threaded through the evaluator)
r14d = current environment    (threaded)
ebx  = current form head      (dispatch scratch)
```

Every function preserves those semantics across calls. Volatile registers never carry evaluator state
across a call, which is what makes that class of bug impossible by construction.

#### Adding a new primitive

1. `sectorlogo_data.inc`: add the `str_sl_myop` name string and its
   `atom_sl_myop` BSS dword.
2. `sectorlogo_main.inc` `intern_builtins`: `INTERN_ONE str_sl_myop,
   atom_sl_myop`.
3. Implement `sl_myop` for a statement, or leave the value to `EvalPrimary`.
4. `sectorlogo_eval_dispatch.inc` (statements) and/or
   `sectorlogo_eval_primary.inc` (expressions): add the `cmp` plus `je`
   dispatch entry.
5. Keep the register convention and save/restore any non-volatile you touch.
6. Add tests to `v_sectorlogo_validate.cpp`.
7. Update the cheat sheet and the relevant table in this manual.

The source and the validator are authoritative: if this manual and the code disagree, the code is right.

The full quick reference, values, all primitive tables, tokenizer notes, the old-to-new name migration
table, and idioms, is stored in [sectorlogo_cheat_sheet.md](sectorlogo.md#language-quick-reference).

## Language quick reference

This chapter is a compact reference for command names and expression forms after the language has been
introduced. The tables intentionally use short descriptions so individual operations can be located
quickly. For details about evaluation, argument rules, and the lifetime of values, consult the
corresponding language sections above.

### Values

| Type | Tag bits | Example | Range |
| --- | --- | --- | --- |
| NIL | 0 | `[]`, `0` | The NIL value. |
| NUMBER | bit30 set | `42`, `-7` | -536,870,912 through 536,870,911 (30-bit payload) |
| FLOAT | bits 31+30 set | `1.0`, `-2.5` | ~+/-10^38, 7 sig digits |
| ATOM | bits 31+30 clear | `"hello"`, `True` | interned string |
| CONS | bit31 set | `[1 2 3]` | list cell index |

### Syntax

Names are **case-insensitive**, `Power`, `power`, `POWER` are the same. The tokenizer folds A-Z to a-z.
Canonical form is PascalCase.

```
; comment                   ; to end of line
Make "x" 42                 ; assignment (BOTH quotes in "x" are mandatory)
:variable                   ; read variable
"string literal"            ; C-style double-quoted, escapes: \\ \" \n
[1 2 3]                     ; unevaluated list (data AND code)
(2 + 3)                     ; evaluated sub-expression
```

### Arithmetic (auto-promotes to float if either operand is FLOAT)

The basic arithmetic operators `+`, `-`, `*`, `/`, and `%` are infix only. Other named numeric functions
use ordinary prefix-call syntax:

| Form | Notes |
| --- | --- |
| `a + b` | Addition |
| `a - b` | Subtraction; use `Negate 5` for unary negation |
| `a * b` | Multiplication |
| `a / b` | Integer division for two NUMBER operands; floating-point division otherwise |
| `a % b` | Remainder |
| `Power a b` | Exponentiation; see the numeric domain and error rules above |
| `SquareRoot n` | int floor for NUMBER, x87 for FLOAT |
| `AbsoluteValue n` | Absolute value |
| `Negate n` | unary negate |
| `Maximum a b` | Returns the greater argument |
| `Minimum a b` | Returns the lesser argument |

Infix operators are left-associative, with `*`, `/`, and `%` taking precedence over `+` and `-`.
Parentheses override the default precedence.

### Float Operations

| Op | Meaning |
| --- | --- |
| `AsFloat n` | int -> float (`42` -> `42.0`) |
| `AsInteger x` | float -> int, truncate toward zero |
| `Floor x` | floor toward minus infinity, returns FLOAT |
| `Round x` | round half-to-even, returns FLOAT |
| `IsFloat x` | True if FLOAT |
| `NaturalLog x` | ln(x) |
| `LogBaseTwo x` | log_2(x) |
| `PowerOfTwo x` | 2^x |
| `FloatRemainder x y` | float remainder |
| `HashNumber "str"` | CRC32 -> FLOAT in [0,1) |

### Bitwise

| Expression | Description |
| --- | --- |
| `a & b` | bitwise AND |
| `a \| b` | bitwise OR |
| `a ^ b` | bitwise XOR |
| `~a` | bitwise NOT (prefix) |
| `a << n` | left shift |
| `a >> n` | right shift (arithmetic) |

### Comparison (return True or NIL)

| Expression | Description |
| --- | --- |
| `a = b` | structural equality |
| `a != b`, `a <> b` | structural inequality |
| `a > b` | numeric |
| `a < b` | numeric |
| `a >= b` | numeric |
| `a <= b` | numeric |
| `SameValue a b` | tagged-word identity |

`=` is structural (numbers compare numerically, atoms as strings). `> < >= <=` are numeric (non-number
causes a terminal error). `SameValue a b` is tagged-word identity, same atom / payload / cons cell (NOT
structural; distinct equal lists -> NIL).

### Boolean

- `(a && b)`, both must be truthy; both arguments are evaluated and there is no short-circuit
- `(a || b)`, either truthy; both arguments are evaluated and there is no short-circuit
- `(! a)`, invert truthiness
- `(IsNumber x)`, True if NUMBER
- `(IsFloat x)`, True if FLOAT
- `(IsZero n)`, True if 0
- `(IsList x)`, True if CONS
- `(IsStringAtom x)`, True for a quoted STRING atom (type-byte test; the only way to tell the atom kinds apart)
- `(IsWordAtom x)`, True for a bare WORD atom (type-byte test; the only way to tell the atom kinds apart)

`&&`/`||` do NOT short-circuit. Do not use them to guard unsafe tests -- `(:x IsNumber && :x > 0)`
evaluates the `>` comparison even when `:x` is not a number (FATAL). Nest `IfElse` instead: test the
guard first, run the risky expression only in the branch body.

### Truthiness

Falsy: NIL `[]`, NUMBER 0, FLOAT +/-0.0, False atom. Everything else is truthy.

**Gotcha:** `Search` returns 0 for position 0 (falsy!). Use `IsNumber` to test found-ness.

### Control Flow

```
If (:n > 5) [ Print "big" ]
IfElse (:n > 5) [ Print "big" ] [ Print "small" ]
Repeat 5 [ Print "tick" ]    ; counted; Stop=break, Continue=skip iteration
Repeat [ Print "tick" ]      ; ENDLESS (list literal in the count slot); exits: Stop/Continue/Exit/Output
Stop                         ; break the innermost Repeat; inside a function, return from that function
Continue                     ; skip to the next Repeat iteration (a counted loop still consumes a trip)
Run [ Print "hi" ]           ; evaluate a list as code
Exit                         ; exit entire script immediately
```

Typo hazards: `Repeat [ 5 ] [ body ]` is an ENDLESS loop (body `[5]` ). A list-holding variable in the
count slot FATALs, wrap it: `Repeat [ Run :var ]`.

### Diagnostics

| Op | Meaning |
| --- | --- |
| `EnableTrace level` | Set the trace level: 0 off, 1 statements, 2 also expressions, 3 verbose. The expression form returns the previous level. |
| `EnableStrictNames flag` | 1 turns an unbound name in expression position into an error instead of a self-evaluating value (default 0). The expression form returns the previous flag. |

### Functions

```
Define "name" [:a :b] [ Output :a + :b ]   ; define a function
DefineCommand "name" [:a :b] "exe" "fmt"   ; define external-tool function (v7sprintf fmt, spawn exe, return captured stdout)
Output :result                             ; return value from function
Local "total"                              ; function-scope local (O(1) Make in loops)
Local "a" "b" "c"                          ; multi-name: declares all three in one call
```

Dynamic scoping. Max 65,536 recursion frames (`SL_MAX_CALL_DEPTH`).

**Call forms**, arguments come from the caller's stream:

- `Output Maximum 2 3` -> 3: a call in a value slot consumes its arity from the
  stream, greedily.
- `Print` is ONE argument: `Print "a":b` prints only `a`; `:b` becomes a
  skipped statement.
- A bare user-function name in an argument slot invokes that function and consumes
  its declared arguments. `Print (f 2 3)` makes the call boundary explicit.
  Use `Print "f"` to print the name as text.
- A cons in a value slot is a LITERAL: only `(...)` or `Run` evaluate
  code. Generated code should wrap every value expression in parens.

**Generated code (a `Prepend`-built `Run` block), four rules:**

- `Run` re-reads the block's atoms as source tokens, so `:var` elements
  resolve: `Run [ output:a ]` is the value of `:a`.
- `Run` in a value slot returns the block's `Output` value, otherwise the last
  user-function statement's return, or NIL. A primitive used as a standalone
  statement does not supply the block's return value:
  `Run [ Maximum 1 2 ]` returns NIL, whereas
  `Run [ Output Maximum 1 2 ]` returns 2. End a block with `Output`
  when the caller needs a value.
- A bare builtin-name element **CALLS the builtin while the block is being
  built** and desynchronizes the rest. Build every command name as
  `(AsWord "Maximum")`, parens make it a value that only dispatches on `Run`.
- A data constant whose text is a builtin name is smuggled through as a
  literal: `[ output first <the-one-element-cons> ]`.

Primitive arity is fixed, never variadic: `Maximum 1 2 3` consumes `1` and `2` and leaves `3` as a
separate form. `And`, `Or`, `Equal`, `NotEqual`, `Maximum`, `Minimum` are binary too.

`Make`, `Local`, `PrintRaw`, `Repeat`, `If`, `Output`, `Define`, `Stop` have no expression form and
self-evaluate in a value slot; `IfElse` is the only control command with one.

**Two atom types.** A quoted literal (`"Maximum"`) is a STRING atom and always self-evaluates, even when
its text names a builtin. A bare name token (`Maximum`) is a WORD atom and may dispatch as a command.
`AsWord` converts STRING -> WORD (which is why `(AsWord "Maximum")` becomes callable), so it is the tool
for command names and the wrong tool for data.

**Stop inside a function returns from that function only** (standard Logo procedure-return): the
caller's `Repeat` continues to run. After `Stop`, the function returns NIL; use `Output`
when the caller needs a return value. Stop only breaks a Repeat when fired at the loop's own call level (top-level
Stop in no loop terminates the script body).

**Bracket hazard:** a Define whose body `[` closes on the header line is a ONE-STATEMENT function,
everything after runs at top level. Always open the body bracket and put the first statement on the NEXT
line. A stray top-level `Output` terminates the current script evaluation and returns its value to the
caller.

#### DefineCommand format specifiers (v7sprintf)

| Spec | Type | Description |
| --- | --- | --- |
| `%s` | string atom | string content (no auto-quoting) |
| `%d` | NUMBER | decimal integer |
| `%c` | NUMBER | low byte as character |
| `%f` | FLOAT/NUMBER | float32, 6 fractional digits, trailing zeros trimmed |
| `%%` | No argument | A literal percent character. |

### Lists

| Op | Meaning |
| --- | --- |
| `First lst` | car |
| `ButFirst lst` | cdr |
| `Prepend item lst` | prepend (builds in REVERSE, Reverse before sorted scans) |
| `Item lst n` | nth element, 0-based (NIL if OOB) |
| `Length lst` | element count |
| `SetItem lst n val` | replace element at index n (REBUILDS, original untouched) |
| `SetCar cell val` | DESTRUCTIVE car write on ONE cons cell -> NIL (shared structure sees it) |
| `SetCdr cell val` | DESTRUCTIVE cdr write on ONE cons cell -> NIL (truncate/splice/cycle) |
| `Sort lst` | ascending stable sort by car (NUMBER/FLOAT) |
| `Reverse lst` | reversed copy |
| `FindItem lst start key` | linear scan for car==key from startpos -> index or NIL |
| `FindFirstAtLeast lst start key` | linear scan for first car>=key from startpos -> index or NIL |
| `FindFirstAtMost lst start key` | linear scan for first car<=key from startpos -> index or NIL |

All three Find* variants take THREE required args: `(list startpos key)`. The key must be NUMBER or
FLOAT; string keys are a terminal error. Key extraction follows Sort convention: bare NUMBER/FLOAT = the
key itself; CONS sub-list = Car(element) is the key.

Max sort size: 65,536 elements.

SetCar/SetCdr are the only destructive list operations; the other operations rebuild. arg1 must be a
cons cell (number/float/word/NIL -> FATAL naming the offender); arg2 is any value. The write is visible
through every sibling holder of the shared cell (`SetCar ButFirst:L 42` changes:L's element 1). SetCdr
on the head truncates or replaces the tail; on a later cell it splices. Cyclic lists are supported by
the collector, but `Length` and `Sort` do not terminate when given a cyclic list. Do not mutate
literal-list cells, build with Prepend first.

### Strings (atoms are strings)

| Op | Meaning |
| --- | --- |
| `Part src start len` | 0-based substring, clamps to empty |
| `Search "needle" "haystack"` | 0-based index or NIL |
| `Lowercase "ABC"` | lowercase (ASCII A-Z) |
| `Uppercase "abc"` | uppercase (ASCII a-z) |
| `TextLength "string"` | byte length as NUMBER (NIL/empty -> 0) |
| `Replace src "old" "new"` | replace FIRST occurrence only |
| `TextToNumber "42"` | parse -> NUMBER or FLOAT; NUMBER/FLOAT passthrough |
| `Words "text"` | word list in ONE native call: lowercase, a-z0-9 runs, len>=2 -> cons list in text order (or NIL) |
| `Split "a,b,c" ","` | split by delimiter -> cons list (empty delimiter -> single chars) |
| `NGrams "abcd" 2` | sliding n-gram windows: string -> n-char substrings, list -> n-element sublists, full windows only, text order (or NIL) |
| `a _ b` | string concatenation (infix) |

#### Words (the BM25 building block)

`Words "The cat, night vision!"` -> `("the" "cat" "night" "vision")`. One native call replaces the
per-character Logo walk; the measured comparison is recorded in `docs/sectorlogo_performance.md`:

1. Case-fold ASCII A-Z -> a-z (interned: "Cats" and "cats" become the SAME atom)
2. Maximal a-z0-9 runs are words; everything else is a boundary (`"cats, chew!"` -> `"cats" "chew"`)
3. Runs shorter than 2 chars are dropped (`"a i 1"` -> NIL)
4. Result is in reading order (`Item... 0` = First word)
5. No stopword/bigram policy, that stays in your Logo (composable retrieval)

### Random

| Op | Meaning |
| --- | --- |
| `Random n` | NUMBER (integer) in `[0,n)` from the AES-NI PRNG (state Weyl increment + one AES round, upper 32 bits % n); `n <= 0` returns 0; `n` must be a NUMBER, a FLOAT limit FATALs in NumVal |
| `RealRandom 1` / `RealRandom 0` | Enable or disable RDTSC input for `Random`; returns the previous setting as NUMBER |

### Console I/O (engine primitives)

| Op | Kind | Meaning |
| --- | --- | --- |
| `PrintRaw expr` | statement | print a value without a trailing newline; the standard-library `Print` adds CRLF |
| `(Character n)` | expression | code n (0-255) as a 1-char string atom, the ESC generator |
| `(Number "s")` | expression | first char of string -> NUMBER (inverse of Character) |
| `(ReadKey)` | expression | Non-blocking keyboard poll: NIL if nothing is pending, otherwise the pending bytes (arrow/function keys return their whole escape sequence as one atom) |
| `(EnableColors)` | expression | probes and enables VT processing, True if 24-bit RGB is supported, NIL if redirected or unavailable |
| `(NearestMatch full vector codebook)` | expression | least-squares codebook search: `[key dist]` (full=NIL) or all rows sorted ascending (full=True); vector = n floats, codebook rows = `[key comp...]` |

`ReadKey` uses a zero-timeout poll (the hook can run per-token).

### Console stdlib

`autoexec.logo` is loaded from a DB key with a file fallback.

`Print:v`, `ClearScreen`, `SetCursor col row`, `SetPaletteColor n`, `SetPaletteBackground n`, `SetColor
r g b` for truecolor with a nearest-16 fallback when `:vt24` is false, `SetBackground r g b`,
`ResetColors`, `NearestColor r g b` for a NearestMatch over the palette, `NearestColorAll r g b` for all
16 entries as `[index distance]` pairs, `While cond-block body-block` where the condition block must
contain `Output`, `ConsoleSelfTest`, `PrintBanner label` for centered truecolor banner text, `Sequence
code` for one raw SGR code, and `ForegroundCode n` / `BackgroundCode n` for the SGR code of a palette
index (the helpers behind the color setters).

Constants: `:esc` `:cr` `:lf` `:crlf` `:vt24` (probe result), `:keyescape` `:keyup` `:keydown`
`:keyright` `:keyleft`, `:black` `:blue` `:green` `:cyan` `:red` `:magenta` `:yellow` `:white`.

### V7 Host Primitives

#### Hook variables (every token)

| Var | Value |
| --- | --- |
| `:TokenNumber` | candidate token id (NUMBER) |
| `:Piece` | decoded text (ATOM) |

#### Stream

| Op | Meaning |
| --- | --- |
| `GetStream` | full generated text so far (ATOM, 1MB max) |
| `StreamLength` | byte length of stream (O(1)) |
| `SetStream "new"` | rewrite entire stream -> KV rollback + re-encode + re-prefill |

#### Generation control

| Op | Meaning |
| --- | --- |
| `Reject` | discard token, resample |
| `Halt` | force EOS immediately |
| `SetOutputToken n` | replace the current candidate token with token id n (stateless, no engine-side limits; how often the script may call it is bounded only by the generation budget) |
| `SetTemperature n` | Set temperature as an integer percentage divided by 100 |
| `SetNucleus n` | Set top-p as an integer percentage divided by 100 |

#### Model introspection

`VocabularySize`, `ModelDimension`, `LayerCount`, `EndToken`, `Position`, `GetToken n`, read-only
config/state values. `EndToken` returns the EOS token id (NUMBER); sidecars detect the end-of-turn
boundary with `:TokenNumber = EndToken` (the hook executes BEFORE the EOS check).

#### MicroDB persistence

| Op | Meaning |
| --- | --- |
| `DatabaseGet "key"` | atom value or NIL |
| `DatabaseSet "key" val` | store (auto-stringify NUMBER/FLOAT) |

#### Tokenization

`EncodeWord "text"` -> first BPE token id (NUMBER) or NIL.

### FFI (CallLibrary)

```
CallLibrary ["DLL" "PROC" "RET" ["ARG_TYPES"]] arg1 arg2...
```

Every descriptor atom is a QUOTED string literal: module name, procedure name, return type, and each
argument type.

- DLL: `"user32"`, `"kernel32"`, `"SELF"` (v7's own DLL, uppercase, quoted),
  `"HOST"` (host app callback)
- PROC: export name, quoted (case preserved, `_` and `.` are lexer
  delimiters so a bare name would split)
- RET: `"integer"`, `"pointer"`, `"text"`, `"void"` -- quoted LOWERCASE. A
  bare or PascalCase type name interns as a different atom and FATALs with
  the return-type error before the call runs.
- ARG: `"integer"`, `"pointer"`, `"text"` per arg
- Max 16 args

### Approximate Hash Map Pattern for BM25 and RAG

There is no `List` primitive. Build pairs with nested `Prepend`, where `Prepend:key Prepend:val []`
produces `[key val]`, then key by hash:

```
; build: hash every key -> [[hash "key1"] [hash "key2"]...]
Make "m" []
Repeat:N [
  Make "h" HashNumber Item:keys:i
  Make "pair" Prepend:h Prepend Item:keys:i []
  Make "m" Prepend:pair:m
]
Make "m" Sort:m

; lookup: FindFirstAtLeast on sorted hash list (list startpos key) -> position or NIL
Make "idx" FindFirstAtLeast:m 0 (HashNumber "target")
```

Remember: `Prepend` builds lists in REVERSE, `Reverse` before `Sort` -dependent scans if construction
order matters.

### Common Idioms

```
; replace-all (Replace replaces only first occurrence)
Define "replace_all" [:text :old :new] [
  Make "result" :text
  Repeat [
    Make "found" Search :old :result
    IfElse IsNumber :found [
      Make "result" Replace :result :old :new
    ] [
      Output :result
    ]
  ]
]

; tokenize with stopword drop (Words does the word split natively)
Define "tokenize" [:s] [
  Make "toks" Words :s
  Make "out" []
  Repeat 999999 [
    IfElse (SameValue :toks []) [ Stop ] []
    Make "w" First :toks
    Make "toks" ButFirst :toks
    IfElse IsNumber (Search :w :stopwords) [] [ Make "out" Prepend :w :out ]
  ]
  Output (Reverse :out)
]

; Search found-ness test (0 is a valid position but falsy!)
IfElse IsNumber (Search "x" :s) [ Print "found" ] [ Print "not found" ]

; counter with Local (O(1) Make in loop)
Define "count" [:n] [
  Local "total"
  Make "total" 0
  Repeat :n [ Make "total" :total + 1 ]
  Output :total
]
```

### Tokenizer Notes

- **Case-insensitive**: all names fold A-Z -> a-z. `True` = `true` = `TRUE`.
- **Operators always split tokens**: `+ - * / % _ = & | ^ ~ < > !` are boundaries.
- **Brackets always split tokens**: `[ ] ()` are boundaries.
- **Unary minus/plus in lists**: `[-3 5 -10 0]` works, every list element is standalone.
- **Negative arguments to named functions**: `FloatRemainder 7.0 -3.0`
  passes `-3.0` as a signed numeric argument.
- **Compact infix**: `2-3` stays binary minus (no space before `-` after an operand).
- **String literals**: `"hello world"` with `\"` and `\\` and `\n` escapes.
- **Multi-char operators**: `<= >= <> != << >>`

### Command naming

The engine registers the current names shown throughout this document. Names are case-insensitive, so
`Make`, `make`, and `MAKE` address the same command. Use the PascalCase spellings in new scripts.

Legacy command names are not automatically mapped to current names. Use the commands listed in the
tables above rather than relying on undocumented aliases. In particular, addition, subtraction,
multiplication, division, and remainder are written with `+`, `-`, `*`, `/`, and `%` in infix
expressions. An unregistered name may evaluate as an atom in an expression or be skipped in statement
position; neither behavior provides backward compatibility.

| Area | Current commands |
| --- | --- |
| Core | `Make`, `Repeat`, `If`, `IfElse`, `Run`, `Stop`, `Continue`, `Exit`, `PrintRaw` |
| Functions | `Define`, `DefineCommand`, `Output`, `Local` |
| Lists | `First`, `ButFirst`, `Prepend`, `Item`, `SetItem`, `Length`, `Sort`, `Reverse`, `FindItem`, `FindFirstAtLeast`, `FindFirstAtMost`, `SetCar`, `SetCdr` |
| Values | `=`, `<>` / `!=`, `>`, `<`, `>=`, `<=`, `SameValue`, `IsNumber`, `IsFloat`, `IsList`, `IsZero`, `IsStringAtom`, `IsWordAtom` |
| Arithmetic | `+` `-` `*` `/` `%` (infix symbols), `Power`, `SquareRoot`, `AbsoluteValue`, `Negate`, `Maximum`, `Minimum`, `Random`, `RealRandom` |
| Float | `AsFloat`, `AsInteger`, `Floor`, `Round`, `NaturalLog`, `LogBaseTwo`, `PowerOfTwo`, `FloatRemainder`, `HashNumber` |
| Bitwise | `&`, `\|`, `^`, `~`, `<<`, `>>` |
| Strings | `Part`, `Search`, `Lowercase`, `Uppercase`, `TextLength`, `Replace`, `TextToNumber`, `Words`, `Split`, `NGrams`, `AsWord` |
| Console | `Character`, `Number`, `ReadKey`, `EnableColors`, `NearestMatch` |
| FFI | `CallLibrary` |
| V7 host | `EncodeWord`, `DatabaseGet`, `DatabaseSet`, `Position`, `GetToken`, `GetTokenPosition`, `VocabularySize`, `ModelDimension`, `LayerCount`, `EndToken`, `GetStream`, `StreamLength`, `SetStream`, `SetOutputToken`, `Reject`, `Halt`, `SetTemperature`, `SetNucleus` |
| Diagnostics | `EnableTrace`, `EnableStrictNames` |
| Hook variables | `:TokenNumber`, `:Piece`, and the tuner-specific `:TunerDecision` (`"accept"` or `"reject"`) |

### Limits

| Limit | Value |
| --- | --- |
| Script source | 1 MiB |
| String staging | 1 MiB |
| Concatenation scratch | 2 MiB |
| Stream and replacement | 1 MiB each |
| Replacement token buffer | 16,384 IDs |
| FFI arguments | 16 |
| User-function arguments | 16 |
| `DefineCommand` arguments | 8 |
| User-function recursion | 65,536 frames (`SL_MAX_CALL_DEPTH`) |
| Parser recursion | 262,144 levels (`SL_MAX_PARSE_DEPTH`) |
| `Local` declarations in scope | 65,536 |
| Cons cells | 1,000,000 before collection and exhaustion checks |
| Sort size | 65,536 elements |
| NUMBER range | -536,870,912 through 536,870,911 |
| Float literal fraction digits | 9 |
| `NearestMatch` components per row | 256 |
| `ReadKey` result length | 64 characters |
| Variable name characters | `A-Za-z0-9` only |

## Native text processing

SectorLogo includes native operations for the text-processing tasks that arise frequently in search and
retrieval applications. A program can use `Words` to extract normalized terms, `NGrams` to construct
consecutive term groups, `Split` to divide text into parts, and `Search` to locate substrings. These
commands process text in native code, while the surrounding Logo program determines how the results
should be filtered, stored, and ranked. As a result, an application can express its retrieval policy in
SectorLogo without implementing every character operation in the interpreter.

The registered commands are implemented in `src/asm/sectorlogo_str_tokens.inc`. The command names in
this chapter are the names accepted by the current interpreter; older names are not supported as
aliases. Each entry below describes its inputs, return value, and the operations it leaves to the
calling program.

### `Words`

`Words` extracts lowercase alphanumeric runs from a string. A run must contain at least two characters.
Punctuation and non-ASCII bytes are boundaries.

```logo
Make "words" Words "The cat, night vision!"
; => "the" "cat" "night" "vision"
```

The command returns a list in reading order; it does not remove stopwords, create bigrams, or apply a
retrieval policy. Those steps remain in Logo so they can be changed without rebuilding the DLL.

### `NGrams`

`NGrams` creates full sliding windows over a string or list. A string source produces byte substrings; a
list source produces fresh lists containing the selected elements.

```logo
NGrams "abcd" 2
; => "ab" "bc" "cd"

NGrams [10 20 30 40] 2
; => [10 20] [20 30] [30 40]
```

`n` must be a `NUMBER` greater than or equal to 1. A source shorter than `n` returns NIL. `NGrams` does
not lowercase or tokenize text; compose `Lowercase` and `Words` first when that is the intended input.

### `Split` and `Search`

`Split` separates text at a delimiter. When the delimiter is empty, the split operation returns
individual characters. Adjacent occurrences of a nonempty delimiter produce empty segments between them.

`Search` returns the zero-based byte offset of a match, or NIL when no match exists. Because a valid
match may begin at offset zero, a script should test the return type with `IsNumber` rather than relying
on its truth value:

```logo
Make "position" Search "cat" "concatenate"
IfElse IsNumber:position [ Print:position ] []
```

`Replace` modifies only the first matching occurrence. To replace every occurrence, a script must search
for the next match and repeat the operation until `Search` returns NIL.

### Native execution and interpreter overhead

A Logo procedure that examines each character separately must execute an interpreted operation for every
input byte. The native `Words` command instead performs the alphanumeric scan in a single
implementation-level operation. A retrieval script can then handle stopword filtering and bigram
construction in ordinary Logo code, where the policy remains easy to inspect and test.

This is a performance and readability choice, not a change to the RAG policy. The policy must be stated
in the script and measured independently.

### Benchmark references

The historical and current timing tables are in `docs/sectorlogo_performance.md`. The tokenization
micro-profile is run through `bm25_sectorlogo_benchmark.exe`; its source and command line are documented
in `examples/sectorlogo/README.md` and `docs/sectorlogo_performance.md`.

When recording a benchmark measurement, include the commit, the model database path, the warmup
procedure, number of repetitions, and the result check. Do not compare timings from different machines
or different binary builds as if they were one result.

## Worked examples: numeric operations and data structures

The examples in this chapter show how common language constructs behave in complete expressions. They
cover numerical conversion, evaluation order, lists, and other data operations that appear in larger
SectorLogo programs. Floating-point values use the interpreter's documented tagged representation; their
precision and range are determined by that representation, not by the formatting of a literal in the
source code. The interpreter stores floating-point values as truncated IEEE binary32 values in tagged
words using tag `0xC0000000`, rather than allocating separate list cells for them.

```logo
; arithmetic promotes: any float operand makes the result a float
OUTPUT 1.5 + 2.5                 ; 4.0
OUTPUT 10 + 4.0                  ; promotion, result 14.0
OUTPUT 10 / 4                    ; stays integer 2 -- int x int is exact

; math primitives (x87-backed, 1-ulp class accuracy)
OUTPUT LogBaseTwo 8.0            ; 3.0
OUTPUT PowerOfTwo 3.5            ; 11.313708
OUTPUT NaturalLog 2.71828        ; 0.99999
OUTPUT FloatRemainder 10.5 3.0   ; 1.5
OUTPUT SquareRoot 2.0            ; 1.4142136
OUTPUT Power 2.0 10.0            ; 1024.0

; conversions and rounding
OUTPUT AsInteger 2.7             ; 2 (truncate toward zero)
OUTPUT Round 2.5                 ; 2 (round-half-to-even)
OUTPUT Floor -2.1                ; -3.0
OUTPUT IsFloat 1.5               ; TRUE  (IsNumber 1.5 is NIL)
```

Floating-point values evaluate as false only when they are exactly `0.0` or `-0.0`. Bitwise operators do
not accept floating-point operands. The infix `=` operator compares compatible integer, floating-point,
and numeric string-atom values, so `1`, `1.0`, and `"1"` compare equal. Number formatting uses
fixed-point notation for magnitudes in `[1e-3, 1e9)` and six-significant-digit exponent notation outside
that interval. The special values use the spellings `nan`, `inf`, and `-inf`.

SectorLogo provides operations for sorting collections and finding bounds within ordered lists.
Retrieval programs can use them to rank passages, including BM25-style scores, without invoking a
separate native sorting utility:

```logo
; Sort ranks [key payload] sub-lists by the key (stable radix sort)
Make "ranked" Sort [[2.5 "chunk-b"] [1.0 "chunk-a"] [7.25 "chunk-c"]]

; binary-search bounds on a sorted list
OUTPUT FindFirstAtLeast :sorted 0 4.0  ; first index with car >= 4.0
OUTPUT FindFirstAtMost :sorted 0 4.0   ; last index with car <= 4.0

; approximate hash map: HashNumber keys + Sort + FindFirstAtLeast probe
OUTPUT HashNumber "some-key"          ; CRC32C-derived FLOAT in [0,1)
```

Lists support indexed retrieval through `Item`, modification through `SetItem`, and common operations
such as `Length`, `Reverse`, and `FindItem`. The examples below show how these operations can be
combined to process collections without introducing native extensions.

`LOCAL "name` declares a variable local to the current procedure. `Make` updates the binding in place,
which is suitable for loop counters. `TextToNumber` parses numeric text, and `Split` separates text
using the documented delimiter rules. `Random` uses an AES-NI-based generator; the optional `RealRandom`
setting enables the documented RDTSC-derived entropy input.

`DefineCommand "name" [:params] "exe_path" "fmt"` creates a command that invokes an external program
through `v7_execute_fmt`. For example, the following declaration passes the text query to a retrieval
executable:

```logo
DefineCommand "rag_search" [:query] "examples/rag/bm25_rag.exe" "search \"%s\""
```

The optional `autoexec.logo` file loads standard library definitions during interpreter initialization.
The language reference above documents the complete command syntax and limits. The [interpreter
benchmark record](sectorlogo_performance.md) contains the recorded comparisons with C, Python, and
JavaScript.
