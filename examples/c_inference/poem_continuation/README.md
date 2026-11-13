# poem_continuation

Raw-continuation contrast demo: `v7_generate(..., raw_generation=1)` continues
plain text with NO chat template -- the closest thing to base-model
completion the engine offers.

## Run

```bat
bin\poem_continuation.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db
```

Takes `<dll_path> <db_path>` positionals; `-q8` optional.

## Why it exists

Chat mode renders the prompt through the model's turn-1 template, so a poem
line gets wrapped in system/user markers. Raw generation skips the template
entirely (EOS is the only stop condition), which is what makes the two modes
a usable contrast pair when debugging template-rendering issues.
