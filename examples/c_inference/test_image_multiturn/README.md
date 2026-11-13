# test_image_multiturn

Multi-turn VLM session test: an image turn followed by text-only follow-up
turns that must remember what the image showed. The vision analogue of
`test_multiturn`.

## Run

```bat
bin\test_image_multiturn.exe -q8 1 bin\v7_threading.dll weights\databases\qwen3_vl_4b\lz4ht.db cat640.raw
```

Positionals: `<dll_path> <db_path> <image_path>`; `-q8 0|1` selects the
forward path (both must stay coherent, Sec.131). The image is the same RAW
RGB + `.meta` sidecar format `dbg_image_recog` uses.

## What it proves

- Image tokens enter the KV cache once and survive across subsequent text
  turns (reset=0).
- M-RoPE deltas from the image turn do not corrupt text-turn positions
  (the Sec.150 bug class).
