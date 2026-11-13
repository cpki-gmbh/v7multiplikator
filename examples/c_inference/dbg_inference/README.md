# dbg_inference (assembly)

The public API sequence in PURE x64 ASSEMBLY: LoadLibraryA, GetProcAddress,
V7_init, v7_initialize_inference, v7_chat, print. The reference ASM consumer
of the public tier -- proves the C API contract from the other side.

## Run

```bat
bin\dbg_inference.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db [-q8]
```

Both path arguments are mandatory (Sec.106 -- no defaults).

## Build

`build.bat` assembles it with NASM (step 4/6) and links it with
`/entry:WinMain`. Its include closure is pinned in `src\build_c_targets.txt`
so the C/C++ freshness gate rebuilds it when the engine headers change.

## Notes

- Greedy sampler (temperature 0), Q8 mode, 200-token budget -- the harness
  is a fixed smoke, not a playground (use `inference_example` for options).
- Loads the DLL via LoadLibraryA so DllMain runs under the same loader state
  as every C harness (Sec.47).
- verify.bat runs it F32 and Q8 as inference smokes.
