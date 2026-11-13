THIRD_PARTY.md — third-party components in v7
==============================================

Everything not listed here is original v7 work, GPL v3 (see LICENSE).

NASM (assembler, shipped binary)
--------------------------------
  What:      Netwide Assembler -- assembles src\asm\v7_threading.asm
  Version:   3.02rc6 -- the version this release was built and verified with. It is a release candidate of NASM 3.x by upstream versioning, but 2.16.01 also assembles the tree cleanly, so any NASM >= 2.16 works; drop the newer one in and build.bat prints the version it uses.
  License:   BSD-2-Clause (full text ships alongside: tools\NASM-LICENSE.txt)
  Copyright: (c) 1996-2026 The NASM Authors
  Source:    https://nasm.us
  In-tree:   tools\nasm.exe (shipped binary; build.bat pins it ahead of
             PATH and prints the version it uses)

Zydis / Zycore (dev-only crash disassembly)
-------------------------------------------
  What:      x86-64 instruction decoder + formatter (host-side crash
             dumps in validators/tools; the v7 DLL itself never links it)
  Version:   zydis-master, zycore-c-master (git checkouts, 2024 line)
  License:   MIT
  Copyright: (c) 2014-2024 Florian Bernd, (c) 2014-2024 Joel Höner
  Source:    https://github.com/zyantific/zydis
             https://github.com/zyantific/zycore-c
  In-tree:   zydis-master\, zycore-c-master\ (dev tree only; each keeps
             its own LICENSE file)
  Release:   the release root header is the no-op stub
             (tools\zydis_stub\zydis_debug_header.h); build.bat
             auto-detects (see docs\zydis_debug_readme.txt for how to
             restore real Zydis in the dev tree).

nasmfmt (build formatting helper, shipped prebuilt)
---------------------------------------------------
  What:      NASM source formatter used by the ASM comment checker
             (auto-fix pass). The tree carries a patched fork of
             upstream nasmfmt -- formatting defaults adapted to the v7
             comment checker -- built once from the vendored source in
             src\nasmfmt\ and committed as a prebuilt exe.
  License:   MIT (nasmfmt, see src\nasmfmt\LICENSE). The prebuilt exe
             was produced by the Go toolchain; Go is BSD-3-Clause
             (https://go.dev/LICENSE) and its runtime bits are statically
             linked into that exe.
  Copyright: (c) 2023 Oleg Yamnikov (upstream nasmfmt)
  Source:    https://github.com/0xpentester/nasmfmt
  In-tree:   src\nasmfmt\ (Go source + LICENSE), src\nasmfmt\nasmfmt.exe
             (prebuilt; build.bat only rebuilds it -- requiring Go
             1.21+ -- when the exe is absent)

Scheme corpora (dev-only)
-------------------------
  The Scheme-to-SectorLogo compiler tree (scheme_internal\) and its
  third-party test corpora are dev-only, not part of the release; each
  corpus directory in the dev tree keeps its own license/reference file.

stb_image (vendored Linux image decoder)
----------------------------------------
  What:      single-header image decoding (PNG/JPEG/BMP/GIF) used by the
             Linux server's vision path (examples\server image_decode).
  Version:   2.30
  License:   public domain / MIT dual (choose either; the full dual
             license text is retained at the top of the vendored file)
  Copyright: (c) 2017-2023 Sean T. Barrett
  Source:    https://github.com/nothings/stb
  In-tree:   src\linux\stb_image.h (vendored verbatim, original license
             text intact -- no v7 header is prepended)

Bible corpus (examples\research\bible\)
---------------------------------------
  What:      ragg_base.txt — King James Version text used as a large
             RAG retrieval corpus. The KJV is in the public domain
             worldwide: it predates any applicable copyright and is
             freely reproducible in every jurisdiction, including the
             United Kingdom (the Church of England's manufacturing
             arrangements concern printed editions, not the text
             itself; modern digital KJV texts are public domain).
  Sidecars:  sidecar_bible.logo, sidecar_bible_fs.logo (v7 original,
             GPL v3 like the rest of the tree)

Images (public domain)
----------------------
  backgrounds\bumwolf_avatar.jpg  the Ardmark demo avatar
  backgrounds\ardmark_*.jpg       server demo backgrounds
  cat640.raw / cat640.meta        canonical vision test image
  The author has released these images into the public domain; they
  carry no embedded metadata. They ship as content fixtures, not build
  outputs.

Models (downloaded by the user, never redistributed)
----------------------------------------------------
  v7 ships only exporter code and the database format. The fetch
  scripts download the original public checkpoints from their upstream
  sources at runtime; each model keeps its upstream license:

  - Qwen3 0.6B/1.7B/8B, Qwen2.5-0.5B, Qwen3-VL: Apache-2.0
    (https://huggingface.co/Qwen)
  - Meta Llama 3.2: Meta Llama 3.2 Community License
    (https://llama.meta.com/llama3/license/ -- accept the terms on the
    Meta portal before fetching; attribution and the acceptable-use
    policy apply to derived artifacts)
  - Phi-3-mini-4k-instruct: MIT (https://huggingface.co/microsoft)
  - SmolLM2-135M/360M, SmolVLM-256M: Apache-2.0
    (https://huggingface.co/HuggingFaceTB)
  - TinyLlama-1.1B: Apache-2.0 (https://huggingface.co/TinyLlama)
  - MiniCPM5-1B: Apache-2.0 (https://huggingface.co/openbmb)
  - LLaVA-OneVision-Qwen2-0.5B (+ SigLIP): Apache-2.0
    (https://huggingface.co/llava-hf)
  The HF mirror used for Llama 3.2 (unsloth) redistributes under the
  same Meta license; verify the license card before use.

SAM (Software Automatic Mouth) — REMOVED
----------------------------------------
  The SAM 8-bit speech-synthesizer port that earlier README versions
  described has been removed from the tree entirely (source, build
  targets, and README mentions). Any residual reference is stale.

Ardmark primers (examples\ardmark\primers\)
-------------------------------------------
  Original v7 demo material (GPL v3 like the rest of the tree);
  prepend_1..18.txt are historical snapshots of the demo's own primer
  text, primer_index.txt maps each to the commit it came from.

Shipped data fixtures
---------------------
  examples\rag\web_ragg\web_ragg.db
  examples\ardmark\primers\primers.zip
  backgrounds\bumwolf_avatar.jpg, backgrounds\ardmark_dragon.jpg,
  backgrounds\ardmark_winter.jpg
  cat640.raw + cat640.meta
  All are demo/test fixtures tracked deliberately as test inputs --
  not build outputs (the images are public domain, see above; the rest
  is GPL v3 like the rest of the tree).
