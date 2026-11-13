# dbg_image_recog

The canonical VLM smoke test: raw RGB image + text prompt -> caption/answer
via `v7_chat_image`. Ag Sec.135, both quantization paths must work.

## Run

```bat
bin\dbg_image_recog.exe -q8 0 bin\v7_threading.dll weights\databases\qwen3_vl_4b\lz4ht.db cat640.raw "What is in this image?"
bin\dbg_image_recog.exe -q8 1 bin\v7_threading.dll weights\databases\qwen3_vl_4b\lz4ht.db cat640.raw "What is in this image?"
```

`-q8 0` = F32 forward, `-q8 1` = Q8 forward (separate production paths --
never disable one to fix the other, Sec.131).

## Image format

RAW RGB: `width x height x 3` bytes, no header, with a `.meta` sidecar
("W H" text) next to it -- both produced by `jpg_to_raw.py`. `cat640.raw`
(640x480x3, in the project root) is the canonical test image.

## Supported VLM databases

`weights\databases\qwen3_vl_4b\`, `llava_onevision_qwen2_0.5b\`,
`smolvlm_256m\`.

verify.bat's vision smoke runs this program against the baseline hash
(AGENTS.md Sec.122).
