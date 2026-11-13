# inference_example

The canonical minimal chat program: loads the DLL, opens a database, runs
`v7_chat`, prints the answer and per-token performance stats. Start here when
learning the public API.

## Run

```bat
build.bat                      (from the project root -- outputs to bin\)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is a cat?" -n 1000 -q8
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -prio -15 -i "what is a cat?" -n 1000
```

Both the DLL path and the DB path are required positionals -- there are no
defaults. `-q8` is a Boolean flag (no value): its PRESENCE selects the Q8
forward, omitting it selects the F32 forward; both are production paths.
Run with no arguments to see the full option list.

## Worker priority (-prio)

`-prio n` runs inference at Win32 thread priority `n` for every worker
thread. Win32 constants: -15 idle, -2 lowest, -1 below normal, 0 normal,
1 above normal, 2 highest, 15 time critical (the INI default is 15). The
worker pool is created lazily at the first dispatched operation, so the
program hands the value to the engine through the `v7_override_prio`
environment variable before `v7_init`; the engine applies it at pool
creation, so the FIRST dispatch already runs at that priority. The same
override is available without any code change by setting the variable
yourself (`set v7_override_prio=-15`), and `v7_set_worker_priority` can
change the live pool's priority at any time from C. Priority is
scheduler-only: greedy output is byte-identical either way.

## Notes

- The default sampler is greedy (temperature 0) and belongs to the DLL;
  configure it only through the CLI flags the program forwards (`-t`, `-p`,
  `-s`), never by poking the sampler struct.
- verify.bat runs this program as one of its inference smokes (output must
  contain "paris" for the capital-of-France prompt).
