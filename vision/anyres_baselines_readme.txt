LLaVA-OneVision anyres gate baselines
=====================================

The six vision_baseline_llava_<image>_<mode>.txt files in this directory are
the byte-exact expected outputs of the LLaVA anyres smoke gate
(run_vision_smoke_llava.bat, verify.bat Phase 5).

Each baseline is the dbg_image_recog.exe output for the given fixture image
and quantization mode, greedy (-n 150), with the volatile lines (Time:,
db_load:, Dbg:) stripped -- the same filter the gate applies to its current
run before the byte compare.

Fixtures (project root):
  cat640.raw          640x480  bicubic-rescaled cat photo (canonical shape)
  cat_anyres_wide.raw 1001x200 extreme-width strip (width-limited fit,
                               left_pad = 0, top/bottom pad only)
  cat_anyres_tall.raw 200x1001 extreme-height strip (height-limited fit,
                               left_pad = 76, exercises the centered-pad
                               origin path that the wide strip cannot reach)

Regeneration:
  gen_llava_baselines.bat regenerates all six from the CURRENT build. Run it
  ONLY after an intended output change (engine fix, anyres math change),
  never to paper over a regression -- a baseline refresh hides a broken
  vision path.

History: the tall fixtures originally crashed with a write AV inside the
centered-pad copy loop (.vcb_ar_pad_row in vit_anyres.inc). Root cause: the
fit-origin computation multiplied left_pad by 3 twice (once when parking
r15d, again at the origin step), shifting the pad origin by 2*left_pad
pixels; with a page-exact canvas the last pad row ran off the end of the
allocation. Width-limited images (left_pad = 0) were unaffected, which is
why the wide fixture and the Qwen3-VL smoke never caught it. Fixed by
keeping r15d in pixels and scaling once at the origin step.
