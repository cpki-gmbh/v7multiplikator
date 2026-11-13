# v7_pipe

Unix-philosophy CLI: pipes text through the v7 LLM.

## Modes

```bat
echo "text" | v7_pipe summarize       summarize piped text
cat file.md | v7_pipe explain         explain piped text in simple terms
v7_pipe ask "question"                single-turn question
v7_pipe chat "message"                multi-turn chat (templated)
v7_pipe qa "question" < doc.txt       answer a question about a document
```

## Environment

| Variable | Meaning |
|----------|---------|
| `V7_DLL` | path to v7_threading.dll (optional; falls back to `v7_threading.dll` in the CWD) |
| `V7_DB`  | path to lz4ht.db (REQUIRED -- no default, the program exits with an error if unset) |

## Example session

```bat
set V7_DLL=bin\v7_threading.dll
set V7_DB=weights\databases\qwen3_0.6b\lz4ht.db
bin\v7_pipe.exe ask "What is the capital of France?"
echo The Eiffel Tower was built from 1887 to 1889... | bin\v7_pipe.exe summarize
```

Build comes from `build.bat` (step 4/6); the exe lands in `bin\`.
