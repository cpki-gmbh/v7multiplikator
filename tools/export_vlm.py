# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

"""
export_vlm.py -- export a Qwen3-VL model to the v7 checkpoint + vision format.

Qwen3-VL wraps the text decoder in a container model: every text weight lives
under the "model.language_model.*" key prefix and the hyperparameters live in
a nested "text_config" dict inside config.json. This script:

  1. Reads config.json, extracts text_config + vision_config.
  2. Loads safetensors shards, keeping ONLY the text keys, remapped so that
     "model.language_model.X" -> "model.X" (what export.py's HfModelWrapper
     expects).
  3. Reuses export.py's HfModelWrapper / model_export / build_tokenizer /
     build_prompts verbatim to write the text checkpoint (.bin + .tokenizer
     + .template*).
  4. Writes a combined "<base>.vision" file:
      [ 288-byte VisionConfig header -- 36 qwords. Slots 0-15 follow the
        cross-model LLaVA-OneVision plan layout (vit_hidden, vit_layers,
        vit_heads, ... vit_pos_embed_type); slots 16-25 are Qwen3-VL
        extension fields (image_token_id, merge, temporal, num_pos, the
        3 DeepStack taps, ...); slots 26-35 are post-merge fields
        (rope_theta, use_rope, proj_*, ps_factor, mrope_bound, anyres, ln_eps). ]
        [ VIT1 blob (from export_vision.py) -- patch embed, pos embed, 24
          transformer blocks, main merger, 3 deepstack mergers ]
     db_build_tool.c's "-vision" argument splits this into the
     __sys_vision_config and __sys_vit_weights microdb keys.
     VHDR_* slot constants in db_build_tool.c MUST stay in sync with the
     header layout below.

The text .bin layout is byte-identical to export.py's -- the vision section is
a SEPARATE file so the "__sys_model_weights = whole file past byte 256" rule
in db_build_tool.c does not swallow vision data.

Usage:
    python export_vlm.py <filepath> <huggingface_model_path> [--max-seq-len N]

    <filepath>  output path WITHOUT extension, e.g. weights/qwen3_vl_4b/qwen3_vl_4b
    <huggingface_model_path>  local dir containing config.json + safetensors

    --max-seq-len N  cap the exported sequence length (KV cache sizing) at N
                     positions. Default 8192 -- Qwen3-VL-4B's real
                     max_position_embeddings is 262144, which would size the
                     KV cache at ~77 GB (36 layers x 262144 x 1024 x 4 x 2).
                     8192 is far more than a chat Q&A needs.
"""

import argparse
import gc
import json
import os
import struct
import sys

import numpy as np

# tools/ directory is on sys.path when run as `python tools/export_vlm.py`
import export
from export import HfModelWrapper, build_tokenizer, build_prompts, model_export

# v7_vlm_ref/export_vision.py emits the VIT1 blob; reuse it verbatim. The
# module is dev-only tooling and is NOT part of the release package, so the
# import is deferred to the point of use and fails with a named, actionable
# message instead of a bare ModuleNotFoundError at import time.
VLM_REF_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "v7_vlm_ref"))


def load_export_vision():
    """Import and return the export_vision module, or exit with a loud error."""
    if VLM_REF_DIR not in sys.path:
        sys.path.insert(0, VLM_REF_DIR)
    try:
        import export_vision
    except ImportError as exc:
        # the module itself is missing -> named, actionable FATAL; a failure
        # INSIDE it (e.g. no torch) must propagate unchanged so the real
        # traceback names the actual problem
        if str(exc).startswith("No module named 'export_vision'"):
            print("[FATAL] export_vision.py not found -- the VLM .vision exporter "
                  "lives in the dev tree at v7_vlm_ref\\export_vision.py and is "
                  "not part of the release package. Restore it from git history "
                  "(git log --all -- v7_vlm_ref/export_vision.py, then git "
                  "checkout <commit> -- v7_vlm_ref/export_vision.py) and rerun.")
            sys.exit(1)
        raise
    return export_vision


def extract_text_config(config_dict):
    """Pull the text (language model) hyperparameters from config.json.

    Qwen3-VL nests them under "text_config". Any field HfModelWrapper reads is
    resolved here; missing optional fields fall back to export.py defaults.
    """
    tc = config_dict.get("text_config") or config_dict
    cfg = type("Cfg", (), {})()

    def gi(name, default):
        v = tc.get(name)
        return int(v) if v is not None else default

    def gf(name, default):
        v = tc.get(name)
        return float(v) if v is not None else default

    cfg.hidden_size = gi("hidden_size", 2560)
    cfg.num_hidden_layers = gi("num_hidden_layers", 36)
    cfg.num_attention_heads = gi("num_attention_heads", 32)
    cfg.num_key_value_heads = gi("num_key_value_heads", cfg.num_attention_heads)
    cfg.vocab_size = gi("vocab_size", 151936)
    cfg.intermediate_size = gi("intermediate_size", 9728)
    cfg.rms_norm_eps = gf("rms_norm_eps", 1e-6)
    cfg.max_position_embeddings = gi("max_position_embeddings", 8192)
    cfg.head_dim = gi("head_dim", cfg.hidden_size // cfg.num_attention_heads)
    cfg.rope_theta = gf("rope_theta", 1000000.0)
    cfg.rope_scaling = tc.get("rope_scaling", None)
    cfg.bos_token_id = gi("bos_token_id", 0)
    cfg.eos_token_id = gi("eos_token_id", 0)
    return cfg


def load_text_state_dict(model_path, prefix="model.language_model."):
    """Load ONLY the text weights, remapping <prefix>* to model.*.

    Returns a flat dict {name: tensor}. Vision/projector tensors are skipped
    (they live in the VIT1 / VITP blobs).

    prefix is model-dependent:
      - Qwen3-VL: "model.language_model." (e.g. model.language_model.embed_tokens.weight)
      - LLaVA-OneVision: "language_model.model." (e.g. language_model.model.embed_tokens.weight)
    Both are remapped to "model." so export.py's HfModelWrapper sees the
    standard Qwen key names.
    """
    import glob as _glob
    from safetensors.torch import load_file

    shards = sorted(_glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        print(f"[FATAL] No .safetensors files found in {model_path}")
        sys.exit(1)

    state_dict = {}
    for wf in shards:
        print(f"  Loading {os.path.basename(wf)} ...")
        sd = load_file(wf)
        for k, v in sd.items():
            if k.startswith(prefix):
                # Strip the model-specific prefix and re-prepend "model." so
                # export.py's HfModelWrapper (which expects model.*) is happy.
                state_dict["model." + k[len(prefix):]] = v
            elif k == "lm_head.weight":
                # Untied classifiers live OUTSIDE the text prefix (top-level
                # "lm_head.weight"). Keep them so export.py detects
                # tie_word_embeddings=False and exports the real classifier.
                state_dict[k] = v
        del sd
        gc.collect()

    total_elems = sum(v.numel() for v in state_dict.values())
    print(f"  Loaded {len(state_dict)} text tensors, "
          f"{total_elems / 1e9:.2f}B elements")
    return state_dict


# .vision header slots (qword indices into the 256-byte / 32-qword header).
# Slots 0-15 are the cross-model LLaVA-OneVision plan fields; slots 16-25 are
# Qwen3-VL extension fields (everything the merger + DeepStack towers need);
# slot 26 is the vision RoPE base (theta); slots 27-31 are reserved (zero).
# Must stay in sync with db_build_tool.c VHDR_*.
VHDR_VIT_HIDDEN          = 0    # qword -- ViT hidden dim (1024 Qwen3-VL)
VHDR_VIT_LAYERS          = 1    # qword -- transformer block count (24 Qwen3-VL)
VHDR_VIT_HEADS           = 2    # qword -- attention heads (16 Qwen3-VL)
VHDR_VIT_PATCH           = 3    # qword -- spatial patch size (16 Qwen3-VL)
VHDR_IMAGE_SIZE          = 4    # qword -- image size (SigLIP 384; 0 = Qwen3-VL position grid)
VHDR_VIT_INTERMEDIATE    = 5    # qword -- MLP intermediate dim (4096 Qwen3-VL)
VHDR_VIT_NORM_TYPE       = 6    # qword -- 1=LayerNorm, 0=RMSNorm (0 Qwen3-VL)
VHDR_VIT_FFN_TYPE        = 7    # qword -- 1=GELU MLP, 0=SwiGLU (0 Qwen3-VL)
VHDR_VIT_HAS_CLS         = 8    # qword -- use_head/class token flag (0 Qwen3-VL)
VHDR_VIT_FEATURE_SELECT  = 9    # qword -- 1=full patch embeddings, 0=cls only (1)
VHDR_PROJ_HIDDEN         = 10   # qword -- projector output dim (Qwen3-VL merger out = 2560)
VHDR_PROJ_LAYERS         = 11   # qword -- projector MLP layer count (2 Qwen3-VL)
VHDR_PROJ_ACTIVATION     = 12   # qword -- 1=GELU, 0=none (1 Qwen3-VL)
VHDR_VIT_DEPTH_TAPS      = 13   # qword -- DeepStack tap count (3 Qwen3-VL, 0 SigLIP)
VHDR_VIT_HEAD_DIM        = 14   # qword -- attention head dim (64 Qwen3-VL, 72 SigLIP)
VHDR_VIT_POS_EMBED_TYPE  = 15   # qword -- 0=Qwen3-VL bilinear-grid, 1=learned additive
VHDR_IMAGE_TOKEN_ID      = 16   # qword -- image placeholder token id (151655 Qwen3-VL)
VHDR_VISION_START_ID     = 17   # qword -- <|vision_start|> token id (151652)
VHDR_VISION_END_ID       = 18   # qword -- <|vision_end|> token id (151653)
VHDR_HAS_DEEPSTACK       = 19   # qword -- 1 = apply DeepStack decoder adds (Qwen3-VL)
VHDR_MERGE               = 20   # qword -- spatial merge size (2 Qwen3-VL)
VHDR_TEMPORAL            = 21   # qword -- temporal patch size (2 Qwen3-VL)
VHDR_NUM_POS             = 22   # qword -- learned position table entries (2304)
VHDR_DS_TAP0             = 23   # qword -- DeepStack tap block 0 (5)
VHDR_DS_TAP1             = 24   # qword -- DeepStack tap block 1 (11)
VHDR_DS_TAP2             = 25   # qword -- DeepStack tap block 2 (17)
VHDR_ROPE_THETA          = 26   # qword -- vision RoPE base theta (10000 Qwen3-VL)
VHDR_USE_ROPE            = 27   # qword -- 1=apply RoPE, 0=skip (SigLIP)
VHDR_PROJ_IN_DIM         = 28   # qword -- projector input dim (0=none; 1152 SigLIP)
VHDR_PROJ_OUT_DIM        = 29   # qword -- projector output dim (0=none; 896 SigLIP)
VHDR_PROJ_INTER          = 30   # qword -- projector intermediate dim (0=none; 896 SigLIP)
VHDR_PROJ_TYPE           = 31   # qword -- 0=none(Qwen3-VL merger), 1=MLP(SigLIP), 2=pixel-shuffle(SmolVLM)
VHDR_PS_FACTOR           = 32   # qword -- pixel-shuffle block size (4 SmolVLM-256M, 3 SmolVLM-2B)
VHDR_MROPE_BOUND         = 33   # qword -- text M-RoPE interleaved pair bound (3*min(section); 0 = no M-RoPE)
VHDR_ANYRES_ENABLED      = 34   # qword -- 1 = model uses anyres multi-tile (LLaVA-OV), 0 = single-tile
VHDR_LN_EPS              = 35   # qword -- IEEE-754 float bits of ViT LayerNorm epsilon (1e-5 SigLIP, 1e-6 Qwen3-VL RMSNorm)
VISION_HDR_QWORDS        = 36   # total header qwords (MUST match db_build_tool.c VISION_HDR_QWORDS)
VISION_HDR_BYTES         = VISION_HDR_QWORDS * 8  # 288 bytes


def write_vision_file(model_path, out_path, text_cfg, vision_rope_base=None,
                      model_type="qwen3-vl", proj_out_path=None):
    """Write [288-byte VisionConfig header][VIT1 blob] to out_path.

    The 288-byte header is 36 qwords parsed by db_build_tool.c (stored as
    __sys_vision_config) and the ASM init. Field order is fixed -- do not
    reorder without updating db_build_tool.c VHDR_* and inf_structs.inc.

    model_type dispatches the VHDR values and the export_vision.py branch:
      - "qwen3-vl": Qwen3-VL merger path (USE_ROPE=1, MERGE=2, no projector)
      - "llava-onevision": SigLIP tower (USE_ROPE=0, MERGE=1, MLP projector)
    """
    top = json.load(open(os.path.join(model_path, "config.json"), encoding="utf-8"))
    vc = top.get("vision_config", {})

    # Image/vision token ids live at the TOP LEVEL of config.json for both
    # Qwen3-VL and LLaVA-OneVision. LLaVA uses "image_token_index" (not "image_token_id").
    image_token_id = int(top.get("image_token_id",
                          top.get("image_token_index", 0)))
    vision_start_id = int(top.get("vision_start_token_id", 0))
    vision_end_id = int(top.get("vision_end_token_id", 0))

    hdr = [0] * VISION_HDR_QWORDS

    if model_type == "qwen3-vl":
        # === Qwen3-VL: merger path, RoPE, DeepStack ===
        ds = vc.get("deepstack_visual_indexes", [])
        if not isinstance(ds, list) or len(ds) < 3:
            print("[FATAL] vision_config.deepstack_visual_indexes must be a 3-element list")
            sys.exit(1)

        # Qwen3-VL vision_config uses depth/num_heads/out_hidden_size.
        hidden    = int(vc.get("hidden_size", 1024))
        depth     = int(vc.get("depth", vc.get("num_hidden_layers", 24)))
        heads     = int(vc.get("num_heads", vc.get("num_attention_heads", 16)))
        inter     = int(vc.get("intermediate_size", 4096))
        patch     = int(vc.get("patch_size", 16))
        merge     = int(vc.get("spatial_merge_size", 2))
        temporal  = int(vc.get("temporal_patch_size", 2))
        num_pos   = int(vc.get("num_position_embeddings", 2304))
        out_hid   = int(vc.get("out_hidden_size", text_cfg.hidden_size))
        head_dim  = int(vc.get("head_dim", 0)) or (hidden // heads)

        # Vision RoPE base: CLI override wins, else config, else FATAL.
        if vision_rope_base is not None and vision_rope_base > 0:
            theta = int(vision_rope_base)
        elif "rope_theta" in vc:
            theta = int(vc["rope_theta"])
        else:
            print("[FATAL] vision_config.rope_theta is absent and --vision-rope-base "
                  "was not given. Qwen3-VL needs --vision-rope-base 10000. "
                  "See AGENTS.md S119.")
            sys.exit(1)

        hdr[VHDR_VIT_HIDDEN]         = hidden
        hdr[VHDR_VIT_LAYERS]         = depth
        hdr[VHDR_VIT_HEADS]          = heads
        hdr[VHDR_VIT_PATCH]          = patch
        hdr[VHDR_IMAGE_SIZE]         = 0               # position grid, not fixed image
        hdr[VHDR_VIT_INTERMEDIATE]   = inter
        hdr[VHDR_VIT_NORM_TYPE]      = 0               # Qwen3-VL: RMSNorm (ASM uses LN anyway)
        hdr[VHDR_VIT_FFN_TYPE]       = 0               # Qwen3-VL: SwiGLU (ASM uses GELU anyway)
        hdr[VHDR_VIT_HAS_CLS]        = 0
        hdr[VHDR_VIT_FEATURE_SELECT] = 1
        hdr[VHDR_PROJ_HIDDEN]        = out_hid         # merger output = text dim (2560)
        hdr[VHDR_PROJ_LAYERS]        = 2
        hdr[VHDR_PROJ_ACTIVATION]    = 1               # GELU
        hdr[VHDR_VIT_DEPTH_TAPS]     = 3
        hdr[VHDR_VIT_HEAD_DIM]       = head_dim
        hdr[VHDR_VIT_POS_EMBED_TYPE] = 0               # bilinear-grid
        hdr[VHDR_IMAGE_TOKEN_ID]     = image_token_id
        hdr[VHDR_VISION_START_ID]    = vision_start_id
        hdr[VHDR_VISION_END_ID]      = vision_end_id
        hdr[VHDR_HAS_DEEPSTACK]      = 1
        hdr[VHDR_MERGE]              = merge           # 2
        hdr[VHDR_TEMPORAL]           = temporal        # 2
        hdr[VHDR_NUM_POS]            = num_pos         # 2304
        hdr[VHDR_DS_TAP0]            = int(ds[0])
        hdr[VHDR_DS_TAP1]            = int(ds[1])
        hdr[VHDR_DS_TAP2]            = int(ds[2])
        hdr[VHDR_ROPE_THETA]         = theta           # 10000
        hdr[VHDR_USE_ROPE]           = 1               # Qwen3-VL applies RoPE
        hdr[VHDR_PROJ_IN_DIM]        = 0               # merger IS the projector
        hdr[VHDR_PROJ_OUT_DIM]       = 0
        hdr[VHDR_PROJ_INTER]         = 0
        # Text M-RoPE interleaved bound: pairs j < 3*min(section) are interleaved
        # (j%3: 0=t, 1=h, 2=w); pairs beyond stay t. Qwen3-VL 4B: [24,20,20] -> 60.
        rs = getattr(text_cfg, "rope_scaling", None) or {}
        sec = rs.get("mrope_section")
        hdr[VHDR_MROPE_BOUND]        = 3 * min(sec) if (sec and rs.get("mrope_interleaved")) else 0
        # ViT norm epsilon: Qwen3-VL uses RMSNorm (NORM_TYPE=0), so LN_EPS is
        # unused at runtime; store the RMSNorm eps for documentation.
        _eps_ln = float(vc.get("layer_norm_eps", vc.get("rms_norm_eps", 1e-6)))
        hdr[VHDR_LN_EPS]             = struct.unpack('I', struct.pack('f', _eps_ln))[0]
        ev_model_type = "qwen3-vl"
    elif model_type == "llava-onevision":
        # === LLaVA-OneVision: SigLIP tower, no RoPE, MLP projector ===
        # SigLIP config fields (config.json -> vision_config):
        #   hidden_size=1152, image_size=384, intermediate_size=4304,
        #   num_attention_heads=16, num_hidden_layers=26, patch_size=14
        hidden    = int(vc.get("hidden_size", 1152))
        depth     = int(vc.get("num_hidden_layers", 26))
        heads     = int(vc.get("num_attention_heads", 16))
        inter     = int(vc.get("intermediate_size", 4304))
        patch     = int(vc.get("patch_size", 14))
        image_size= int(vc.get("image_size", 384))
        num_pos   = (image_size // patch) ** 2        # (384/14)^2 = 729
        head_dim  = hidden // heads                   # 1152/16 = 72

        # Text hidden size from the extracted text config (= 896 for Qwen2-0.5B).
        text_hidden = text_cfg.hidden_size

        # SigLIP has NO RoPE -- theta is irrelevant but db_build_tool.c FATALs
        # on theta==0, so write a placeholder. USE_ROPE=0 ensures the ASM
        # never reads the RoPE tables.
        theta = int(vision_rope_base) if (vision_rope_base and vision_rope_base > 0) else 10000

        hdr[VHDR_VIT_HIDDEN]         = hidden          # 1152
        hdr[VHDR_VIT_LAYERS]         = depth           # 26
        hdr[VHDR_VIT_HEADS]          = heads           # 16
        hdr[VHDR_VIT_PATCH]          = patch           # 14
        hdr[VHDR_IMAGE_SIZE]         = image_size      # 384
        hdr[VHDR_VIT_INTERMEDIATE]   = inter           # 4304
        hdr[VHDR_VIT_NORM_TYPE]      = 1               # LayerNorm
        hdr[VHDR_VIT_FFN_TYPE]       = 1               # GELU MLP
        hdr[VHDR_VIT_HAS_CLS]        = 0               # no CLS token
        hdr[VHDR_VIT_FEATURE_SELECT] = 1               # full patches
        hdr[VHDR_PROJ_HIDDEN]        = text_hidden     # projector out = text dim (896)
        hdr[VHDR_PROJ_LAYERS]        = 2               # FC1 -> GELU -> FC2
        hdr[VHDR_PROJ_ACTIVATION]    = 1               # GELU (tanh approx in ASM)
        hdr[VHDR_VIT_DEPTH_TAPS]     = 0               # no DeepStack
        hdr[VHDR_VIT_HEAD_DIM]       = head_dim        # 72
        hdr[VHDR_VIT_POS_EMBED_TYPE] = 1               # additive learned table
        hdr[VHDR_IMAGE_TOKEN_ID]     = image_token_id  # 151646
        hdr[VHDR_VISION_START_ID]    = vision_start_id
        hdr[VHDR_VISION_END_ID]      = vision_end_id
        hdr[VHDR_HAS_DEEPSTACK]      = 0
        hdr[VHDR_MERGE]              = 1               # NO spatial merge
        hdr[VHDR_TEMPORAL]           = 1               # NO temporal duplication
        hdr[VHDR_NUM_POS]            = num_pos         # 729
        hdr[VHDR_DS_TAP0]            = depth           # SigLIP has no DeepStack; taps set to depth
        hdr[VHDR_DS_TAP1]            = depth           # (= one past last layer index 0..depth-1)
        hdr[VHDR_DS_TAP2]            = depth           # so the layer==tap compare in vit_forward never matches
        hdr[VHDR_ROPE_THETA]         = theta           # placeholder (USE_ROPE=0)
        hdr[VHDR_USE_ROPE]           = 0               # SigLIP has NO RoPE
        hdr[VHDR_PROJ_TYPE]          = 1               # MLP projector (also gates post-LN: SigLIP towers apply it)
        hdr[VHDR_PROJ_IN_DIM]        = hidden          # 1152 -> projector present
        hdr[VHDR_PROJ_OUT_DIM]       = text_hidden     # 896
        hdr[VHDR_PROJ_INTER]         = text_hidden     # 896
        hdr[VHDR_ANYRES_ENABLED]     = 1               # LLaVA-OV uses anyres multi-tile
        # SigLIP LayerNorm eps: default 1e-5 (HF SiglipVisionConfig.layer_norm_eps)
        _eps_ln = float(vc.get("layer_norm_eps", 1e-5))
        hdr[VHDR_LN_EPS]              = struct.unpack('I', struct.pack('f', _eps_ln))[0]
        ev_model_type = "llava-onevision"
    elif model_type == "smolvlm-256m":
        # === SmolVLM-256M: SigLIP-base tower, pixel-shuffle projector ===
        # SigLIP-base config fields (config.json -> vision_config):
        #   hidden_size=768, image_size=512, intermediate_size=3072,
        #   num_attention_heads=12, num_hidden_layers=12, patch_size=16
        hidden    = int(vc.get("hidden_size", 768))
        depth     = int(vc.get("num_hidden_layers", 12))
        heads     = int(vc.get("num_attention_heads", 12))
        inter     = int(vc.get("intermediate_size", 3072))
        patch     = int(vc.get("patch_size", 16))
        image_size= int(vc.get("image_size", 512))
        num_pos   = (image_size // patch) ** 2        # (512/16)^2 = 1024
        head_dim  = hidden // heads                   # 768/12 = 64

        text_hidden = text_cfg.hidden_size            # 576
        # scale_factor is read from text_config.pixel_shuffle_factor in HF
        # but we accept a CLI override via the --scale-factor path in the fetch script.
        scale_factor = int(top.get("scale_factor",
                             top.get("text_config", {}).get("pixel_shuffle_factor", 4)))

        theta = int(vision_rope_base) if (vision_rope_base and vision_rope_base > 0) else 10000

        hdr[VHDR_VIT_HIDDEN]         = hidden          # 768
        hdr[VHDR_VIT_LAYERS]         = depth           # 12
        hdr[VHDR_VIT_HEADS]          = heads           # 12
        hdr[VHDR_VIT_PATCH]          = patch           # 16
        hdr[VHDR_IMAGE_SIZE]         = image_size      # 512
        hdr[VHDR_VIT_INTERMEDIATE]   = inter           # 3072
        hdr[VHDR_VIT_NORM_TYPE]      = 1               # LayerNorm
        hdr[VHDR_VIT_FFN_TYPE]       = 1               # GELU MLP
        hdr[VHDR_VIT_HAS_CLS]        = 0               # no CLS token
        hdr[VHDR_VIT_FEATURE_SELECT] = 1               # full patches
        hdr[VHDR_PROJ_HIDDEN]        = text_hidden     # 576
        hdr[VHDR_PROJ_LAYERS]        = 1               # single Linear (not MLP)
        hdr[VHDR_PROJ_ACTIVATION]    = 0               # no activation
        hdr[VHDR_VIT_DEPTH_TAPS]     = 0               # no DeepStack
        hdr[VHDR_VIT_HEAD_DIM]       = head_dim        # 64
        hdr[VHDR_VIT_POS_EMBED_TYPE] = 1               # additive learned table
        hdr[VHDR_IMAGE_TOKEN_ID]     = image_token_id  # 49190
        hdr[VHDR_VISION_START_ID]    = 0               # no wrapping (inline marker)
        hdr[VHDR_VISION_END_ID]      = 0
        hdr[VHDR_HAS_DEEPSTACK]      = 0
        hdr[VHDR_MERGE]              = 1               # no spatial merge
        hdr[VHDR_TEMPORAL]           = 1               # no temporal dup
        hdr[VHDR_NUM_POS]            = num_pos         # 1024
        hdr[VHDR_DS_TAP0]            = depth           # taps set to depth (never match)
        hdr[VHDR_DS_TAP1]            = depth
        hdr[VHDR_DS_TAP2]            = depth
        hdr[VHDR_ROPE_THETA]         = theta           # placeholder (USE_ROPE=0)
        hdr[VHDR_USE_ROPE]           = 0               # SigLIP has NO RoPE
        hdr[VHDR_PROJ_IN_DIM]        = hidden * scale_factor * scale_factor   # 768*16 = 12288
        hdr[VHDR_PROJ_OUT_DIM]       = text_hidden     # 576
        hdr[VHDR_PROJ_INTER]         = 0               # no inner layer (pixel-shuffle is single Linear)
        hdr[VHDR_PROJ_TYPE]          = 2               # pixel-shuffle projector
        hdr[VHDR_PS_FACTOR]          = scale_factor    # 4
        # SigLIP LayerNorm eps: default 1e-5 (HF SiglipVisionConfig.layer_norm_eps)
        _eps_ln = float(vc.get("layer_norm_eps", 1e-5))
        hdr[VHDR_LN_EPS]              = struct.unpack('I', struct.pack('f', _eps_ln))[0]
        ev_model_type = "smolvlm-256m"
    else:
        print(f"[FATAL] unknown --model-type '{model_type}'")
        sys.exit(1)

    hdr_bytes = struct.pack("<36Q", *hdr)

    # Emit the VIT1 blob (and optionally the VITP projector blob for LLaVA)
    # through export_vision.py, streaming directly into the combined file.
    # An export_vision failure leaves a partial .vision behind -- delete it so
    # a later build_db can never consume half a blob.
    try:
        with open(out_path, "wb") as f:
            f.write(hdr_bytes)
            ev_argv = ["export_vision", model_path, out_path,
                       "--model-type", ev_model_type]
            if proj_out_path is not None:
                ev_argv.extend(["--projector-output", proj_out_path])
            old_argv = sys.argv
            sys.argv = ev_argv
            try:
                rc = load_export_vision().main(argv=sys.argv, out_fh=f)
            finally:
                sys.argv = old_argv
    except Exception:
        if os.path.isfile(out_path):
            os.remove(out_path)
            print(f"Removed partial {out_path} (export_vision failed)")
        raise
    if rc != 0:
        if os.path.isfile(out_path):
            os.remove(out_path)
        print("[FATAL] export_vision failed to produce the VIT1 blob")
        sys.exit(1)
    print(f"Written vision section to {out_path}  "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB = {VISION_HDR_BYTES}B header + VIT1)")
    if proj_out_path is not None:
        print(f"Written projector blob to {proj_out_path}  "
              f"({os.path.getsize(proj_out_path) / 1e6:.1f} MB)")

    # Anyres: extract the image_newline learned embedding (nn.Parameter of
    # shape (text_hidden,)) from the HF model state dict. This is NOT a
    # vocabulary token -- it is a trained feature vector concatenated between
    # grid rows in pack_image_features(). Written as raw float32 sidecar.
    if hdr[VHDR_ANYRES_ENABLED]:
        newline_path = out_path + ".newline"
        from safetensors import safe_open
        newline_data = None
        # Try all safetensors shards in the model directory
        for sf_name in sorted(os.listdir(model_path)):
            if not sf_name.endswith(".safetensors"):
                continue
            sf_path = os.path.join(model_path, sf_name)
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.endswith("image_newline"):
                        newline_data = f.get_tensor(key).numpy()
                        break
            if newline_data is not None:
                break
        if newline_data is None:
            print(f"[FATAL] image_newline not found in {model_path}")
            sys.exit(1)
        newline_data = newline_data.astype(np.float32)
        with open(newline_path, "wb") as nf:
            nf.write(newline_data.tobytes())
        print(f"Written image_newline embedding to {newline_path}  "
              f"({len(newline_data)} floats = {len(newline_data) * 4} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Export a VLM (Qwen3-VL or LLaVA-OneVision) HuggingFace model "
                    "to the v7 checkpoint + vision format"
    )
    parser.add_argument("filepath", type=str,
                        help="Output path (without extension). E.g. weights/qwen3_vl_4b/qwen3_vl_4b")
    parser.add_argument("hfpath", type=str,
                        help="HuggingFace model path (local dir)")
    parser.add_argument("--max-seq-len", type=int, default=8192,
                        help="Cap exported sequence length (KV cache size). Default 8192.")
    parser.add_argument("--vision-rope-base", type=float, default=None,
                        help="Override the vision tower RoPE theta (required when "
                             "vision_config.rope_theta is absent, e.g. Qwen3-VL=10000). "
                             "Ignored for llava-onevision (USE_ROPE=0).")
    parser.add_argument("--model-type", type=str, default="qwen3-vl",
                        choices=["qwen3-vl", "llava-onevision", "smolvlm-256m"],
                        help="Model family: qwen3-vl (default), llava-onevision, or smolvlm-256m.")
    args = parser.parse_args()

    with open(os.path.join(args.hfpath, "config.json"), encoding="utf-8") as f:
        config_dict = json.load(f)

    cfg = extract_text_config(config_dict)
    cfg.max_position_embeddings = args.max_seq_len

    print(f"Model type: {args.model_type}")
    print(f"Text config: dim={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
          f"heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}, "
          f"vocab={cfg.vocab_size}, head_dim={cfg.head_dim}, "
          f"rope_theta={cfg.rope_theta}, seq_len={cfg.max_position_embeddings}, "
          f"bos_id={cfg.bos_token_id}, eos_id={cfg.eos_token_id}")

    # Text-tensor prefix is model-dependent.
    #   Qwen3-VL:          "model.language_model.*"
    #   LLaVA-OneVision:   "language_model.model.*"
    #   SmolVLM-256M:      "model.text_model.*"
    if args.model_type == "qwen3-vl":
        text_prefix = "model.language_model."
    elif args.model_type == "smolvlm-256m":
        text_prefix = "model.text_model."
    else:
        text_prefix = "language_model.model."

    # Build the HfModelWrapper-compatible model from remapped text tensors.
    state_dict = load_text_state_dict(args.hfpath, prefix=text_prefix)
    hf_model = type("HFModel", (), {})()
    hf_model.config = cfg
    hf_model._state_dict = state_dict
    hf_model.named_parameters = lambda: state_dict

    print("Building memory-efficient wrapper...")
    model = HfModelWrapper(hf_model, args.hfpath)
    del state_dict
    gc.collect()

    # Text checkpoint: identical format to export.py.
    model_export(model, args.filepath)
    build_tokenizer(model, args.filepath)
    build_prompts(model, args.filepath)

    # Free the text tensors BEFORE the vision pass so peak memory is
    # max(text, vision) instead of text + vision.
    del model
    del hf_model
    gc.collect()

    # Vision section (+ projector blob for LLaVA / SmolVLM).
    proj_path = None
    if args.model_type in ("llava-onevision", "smolvlm-256m"):
        proj_path = args.filepath + ".proj"
    write_vision_file(args.hfpath, args.filepath + ".vision", cfg,
                      vision_rope_base=args.vision_rope_base,
                      model_type=args.model_type,
                      proj_out_path=proj_path)

    print("\nExport complete.")
    print(f"  Text checkpoint: {args.filepath}")
    print(f"  Tokenizer:       {args.filepath}.tokenizer")
    print(f"  Templates:       {args.filepath}.template*")
    print(f"  Vision section:  {args.filepath}.vision  (pass to build_db -vision)")
    if proj_path is not None:
        print(f"  Projector blob:  {proj_path}  (pass to build_db -visionproj)")


if __name__ == "__main__":
    main()
