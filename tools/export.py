# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

"""
Generic v7 model export script.

Converts a HuggingFace transformer model (Qwen3, Qwen2.5, SmolLM2, Phi-3,
Gemma, etc.) to the v7 checkpoint format (.bin) for inference with
v7_threading.dll.

Usage:
    python export.py <output_path> <huggingface_model_path> [--force]

Output files (output_path = "weights/foo/bar"):
    bar                - quantized checkpoint (.bin format, no extension)
    bar.tokenizer      - BPE tokenizer data
    bar.template       - prompt template (no thinking)
    bar.template.with-thinking       - prompt template (with thinking)
    bar.template.with-system         - prompt template (system + user)
    bar.template.with-system-and-thinking - prompt template (system + user + thinking)
    bar.template.multi               - inter-turn template, turn 2+ (terminator + user turn, no thinking)
    bar.template.multi.with-thinking - inter-turn template, turn 2+ (with thinking)
    bar.manifest.json  - export manifest (source + output hashes, exporter revision)

Export contract (everything here REFUSES bad data instead of repairing it --
the database builder alone validates serialized DB/import contracts):

  * Every weight tensor is checked for exact shape (2-D orientation), finite
    values, and architecture-consistent presence before anything is written.
    Unsupported architectures, missing required tensors, unexpected extra
    learned parameters, and non-finite values are named export errors.
  * Q8_0 quantization is plain symmetric absmax per 64-element group with an
    fp32 scale and true fp32 division; groups must be all zero (scale 1) or
    finite. Quantized values are clamped to [-127, 127] -- the Q8 kernels'
    vpsignb dot-product contract requires weights in [-127, 127] (a -128
    weight cannot be negated in an int8 lane; activations may reach -128).
    An fp64 quotient variant was A/B-measured against an extended-precision
    oracle on real Qwen3 tensors: reconstruction error identical to 4 digits,
    so the fp32 quotient stays.
  * Every output artifact is written to a temporary file first, verified
    (checkpoint size against the exact expected byte count), and atomically
    renamed; a failure anywhere keeps the previous good artifacts intact.
  * The header ABI (256-byte header, [q bytes][fp32 scales] tensors, GS=64)
    is unchanged.

Architecture auto-detection:
    - QK-norm (q_norm/k_norm weights): from layer 0 keys; when enabled, BOTH
      tensors must exist with shape (head_dim,) on EVERY layer.
    - rope_base (RoPE theta): from config.rope_theta (must be integral).
    - tie_word_embeddings: cross-checked between config and tensor presence.
    - chat_template: from the tokenizer (official HF renderer); falls back to
      a chat_template.jinja file at the model repo root, then synthesizes a
      ChatML or bare-%s template from the special tokens.

 Low-memory design:
    - Single config/weight load (safetensors read into RAM by load_file; peak
      memory is the full model plus the temporary fp32 copies made during
      quantization -- plan for roughly 2x the model size on multi-GB models)
    - Weights quantized and written sequentially; quantized Q chunks stream
      straight to the output file, only the small per-group scale arrays are
      buffered in RAM
    - Explicit gc.collect() after each major phase
"""

import argparse
import gc
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
import numpy as np
import torch

# Think-markup tags (open/close). Composed from parts so no source-level
# literal can be corrupted by tooling: the tags are the Qwen3-style thinking
# markers and their bytes are pinned by the template byte-compare regression.
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/" + "think" + ">"

GS_FIXED = 64  # the serialized group size: all kernels and build_db.exe assume 64

# Model-semantics allow-list: v7's forward path implements Llama/Qwen-style
# SwiGLU FFN, full RoPE on all positions, standard pre-norm RMSNorm, and
# nothing else. A checkpoint whose config declares different semantics would
# export fine and run numerically WRONG (e.g. Gemma's gelu FFN and (1+w) norm
# offset, Phi-4-mini's 0.75 partial rotary factor, sliding-window attention) --
# refused here, at the boundary, with the offending field named.
SUPPORTED_MODEL_TYPES = ("llama", "qwen2", "qwen3", "minicpm", "smollm", "phi3")
# v7's FFN is SwiGLU with act(silu) gate: exactly this activation string.
# "gelu"/"gelu_pytorch_tanh"/"relu" variants imply a numerically different FFN.
SUPPORTED_HIDDEN_ACTS = ("silu",)
# MiniCPM's classic µP scaling (residual/hidden init multipliers) changes
# every activation magnitude at inference; a config carrying them would
# export fine and run numerically WRONG (MiniCPM5-1B reports plain llama).
MICROUP_SCALE_FIELDS = ("scale_emb", "scale_depth", "dim_model_base")


class ExportError(Exception):
    """Refusal at the export boundary: named tensor/key/shape/reason."""


# ---------------------------------------------------------------------------
# Serialization utilities
# ---------------------------------------------------------------------------

def serialize_fp32(file, tensor):
    """Write one fp32 tensor to file (must be open in wb mode), little-endian."""
    d = tensor.detach().cpu().reshape(-1).to(torch.float32).numpy().astype("<f4", copy=False)
    file.write(d.tobytes())


def serialize_int8(file, tensor):
    """Write one int8 tensor to file (must be open in wb mode)."""
    d = tensor.detach().cpu().reshape(-1).numpy().astype(np.int8, copy=False)
    file.write(d.tobytes())


def assert_finite(tensor, name):
    """Refuse non-finite values anywhere in a tensor (named, with flat index)."""
    flat = tensor.reshape(-1)
    finite = torch.isfinite(flat)
    if not bool(finite.all()):
        bad = int((~finite).nonzero()[0].item())
        value = flat[bad].item()
        raise ExportError(
            f"non-finite value in tensor {name}: flat index {bad}, value {value!r} "
            f"(dtype {tensor.dtype})")


def quantize_q80(w, group_size, name):
    """
    Quantize one tensor to Q8_0: symmetric int8 with a per-64-group fp32 scale.

    scale = max|w| / 127 per group (all-zero group -> scale 1, quantized 0:
    exact under any positive scale); quotient computed by true fp32 division
    and rounded half-to-even. Values are clamped to [-127, 127] as the
    exporter-side guarantee of the Q8 kernels' vpsignb contract (a -128 weight
    would sign-negate to itself and corrupt the dot product; activations may
    reach -128 -- vpabsb handles them and the worst pair sum 2*128*127 = 32512
    stays below the int16 saturation bound).

    Returns (int8_tensor, scale_tensor, metrics_dict). Refuses non-finite
    input with the tensor name and first offending flat index; never returns
    non-finite metrics.
    """
    if w.numel() % group_size != 0:
        raise ExportError(
            f"tensor {name}: numel {w.numel()} not a multiple of group_size "
            f"{group_size} -- the flat [q][s] layout requires whole groups")
    assert_finite(w, name)
    w32 = w.detach().float()
    groups = w32.reshape(-1, group_size)
    wmax = torch.abs(groups).max(dim=1).values
    scale = torch.where(wmax > 0, wmax / 127.0, torch.ones_like(wmax))
    # scale underflow: a denormal group's wmax/127 can round to exactly 0,
    # and the quotient below would divide by 0 -> inf into int8. Those
    # groups (every element ~1e-38) quantize to exact zeros under any
    # positive scale, so substitute scale 1 and zero the group.
    underflow = (wmax > 0) & (scale <= 0)
    scale = torch.where(underflow, torch.ones_like(scale), scale)
    # true fp32 quotient, round half-to-even (torch.round). The fp64-quotient
    # variant was A/B-measured against a 50-digit oracle on real Qwen3 tensors:
    # identical max/mean reconstruction error to 4 digits, so fp32 stays.
    quotient = torch.round(groups / scale[:, None])
    clamped = int((quotient.abs() > 127).sum())
    q = torch.clamp(quotient, -127, 127)
    q = torch.where((wmax > 0)[:, None], q, torch.zeros_like(q))
    q = torch.where(underflow[:, None], torch.zeros_like(q), q)
    q = q.to(torch.int8)
    if bool((q == -128).any()):
        raise ExportError(
            f"quantization of {name} produced a -128 byte -- the Q8 kernel "
            f"dot-product contract (weights in [-127, 127]) would be violated")
    # represented-value sanity: the serialized (q, scale) pair must re-read as
    # finite numbers on the consuming side
    if not bool(torch.isfinite(scale).all()):
        raise ExportError(f"quantization of {name} produced a non-finite scale")
    if bool((scale <= 0).any()):
        raise ExportError(f"quantization of {name} produced a non-positive scale")
    if not bool(torch.isfinite(q.float() * scale[:, None]).all()):
        raise ExportError(f"quantization of {name}: (q, scale) reconstruction is non-finite")
    underflow_groups = int(underflow.sum())
    if underflow_groups:
        # subnormal groups keep their INFORMATION (the magnitude is far below
        # any lerpy contribution), but their nonzero weights quantize to zero
        # here -- tell the export log how many
        print(f"       {name}: {underflow_groups} subnormal group(s) quantized "
              f"to exact zeros (below the positive-scale threshold)")
    # diagnostics run in FLOAT64: float32 sums of squares overflow on large
    # finite inputs and would misfire the finite-refusal below; the
    # quantization math itself stays fp32 by contract
    err64 = (q.to(torch.float64) * scale.to(torch.float64)[:, None]
             - groups.to(torch.float64)).abs()
    groups64 = groups.to(torch.float64)
    metrics = {
        "max_abs": float(err64.max()),
        "mean_abs": float(err64.mean()),
        "rms": float(err64.pow(2).mean().sqrt()),
        "w_norm": float(groups64.norm()),
        "err_norm": float(err64.norm()),
        "n_clamped": clamped,
        "numel": int(w.numel()),
    }
    metrics["rel_norm"] = metrics["err_norm"] / metrics["w_norm"] if metrics["w_norm"] > 0 else 0.0
    for key in ("max_abs", "mean_abs", "rms", "rel_norm"):
        if not math.isfinite(metrics[key]):
            raise ExportError(f"non-finite quantization metric {key} for {name}")
    return q, scale, metrics


def print_quant_metrics(metrics):
    print(f"    max_abs={metrics['max_abs']:.8f}  mean_abs={metrics['mean_abs']:.3e}  "
          f"rms={metrics['rms']:.3e}  rel_norm={metrics['rel_norm']:.3e}  "
          f"clamped={metrics['n_clamped']}")


# ---------------------------------------------------------------------------
# Staged atomic writes
# ---------------------------------------------------------------------------

def stage_write(path, write_fn):
    """Write via a temporary sibling file, then atomically replace the target.

    write_fn receives an open file object for the temp path. On any failure
    the temp file is removed and the target is untouched."""
    tmp = deferred_write(path, write_fn)
    os.replace(tmp, path)


def deferred_write(path, write_fn, pending=None):
    """Write TEMP NAME ONLY and return it -- the caller publishes by os.replace
    in the final pass. Used to make the whole artifact set (tokenizer +
    templates + checkpoint + manifest) atomic ACROSS files: a failed export
    renames nothing and the previous good set stays intact. When `pending` is
    given, registers (tmp, final) there. Cleans its temp file up on failure,
    per stage_write's contract."""
    tmp = path + ".tmp_export"
    try:
        with open(tmp, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        if pending is not None:
            pending.append((tmp, path))
        return tmp
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tokenizer utilities
# ---------------------------------------------------------------------------

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("\xa1"), ord("\xac") + 1))
    bs += list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def internal_to_bytes(U2B, token_str):
    return b"".join(
        bytes([U2B[ch]]) if ch in U2B else ch.encode("utf-8")
        for ch in token_str
    )


def build_tokenizer(model, file):
    """Build .tokenizer binary from model's tokenizer.

    Refuses vocab gaps that are not a trailing reserved-padding range, and
    refuses vocab entries beyond the config's vocab_size (both would change
    engine token lookup semantics silently)."""
    B2U = bytes_to_unicode()
    U2B = {u: b for b, u in B2U.items()}

    tokenizer = model.tokenizer
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    vocab_size = model.params.vocab_size

    beyond = sorted(i for i in id_to_token if i >= vocab_size)
    if beyond:
        raise ExportError(
            f"tokenizer vocabulary contains {len(beyond)} IDs >= config vocab_size "
            f"{vocab_size} (first: {beyond[:8]}) -- the checkpoint would drop real "
            f"tokens; refusing export")

    all_tokens = [id_to_token.get(i, "") for i in range(vocab_size)]
    missing = [i for i, t in enumerate(all_tokens) if t == ""]
    if missing:
        max_present = max(id_to_token) if id_to_token else -1
        trailing_padding = (missing == list(range(max_present + 1, vocab_size)))
        if not trailing_padding:
            raise ExportError(
                f"tokenizer vocabulary has {len(missing)} missing IDs inside "
                f"0..{vocab_size - 1} that are NOT a trailing reserved range "
                f"(first: {missing[:8]}) -- serializing invented empty tokens "
                f"would alias real token lookup; refusing export")
        print(f"Reserved vocab padding: IDs {max_present + 1}..{vocab_size - 1} "
              f"({len(missing)} IDs) absent from the tokenizer vocab (HF padding "
              f"to a multiple) -- serialized as empty placeholder bytes")

    tokenizer_data = json.loads(tokenizer.backend_tokenizer.to_str())
    model_type = tokenizer_data.get("model", {}).get("type", "")
    pre_tok = tokenizer_data.get("pre_tokenizer") or {}
    dec = tokenizer_data.get("decoder") or {}
    normalizer = tokenizer_data.get("normalizer")
    parents = [pre_tok.get("type", "")] + [ps.get("type", "") for ps in (pre_tok.get("pretokenizers") or [])]
    dec_types = [dec.get("type", "")] + [ds.get("type", "") for ds in (dec.get("decoders") or [])]
    if model_type != "BPE" or not any("ByteLevel" in t for t in parents + dec_types):
        raise ExportError(
            f"tokenizer is not byte-level BPE (model type {model_type!r}, "
            f"pre_tokenizer {parents!r}, decoder {dec_types!r}) -- the export "
            f"maps tokens through the GPT-2 byte table; SentencePiece-style "
            f"vocabularies (Gemma, Llama-2) would export a space as the "
            f"3-byte UTF-8 sequence and byte-fallback tokens as literal "
            f"text; refusing export")
    # LOSSLESS bytes on the DECODE side: a normalizer never runs inside the
    # engine (it is metadata in this file), so token -> byte strings stay
    # exact for ANY normalizer. On the ENCODE side the engine cannot apply
    # the normalizer: NFC (compose-only, shipped by Qwen2/3) merely composes
    # decomposed unicode input, so its divergence is input-shaped and covered
    # by the shipped per-model validators -- allowed with a loud note.
    # MUTATING normalizers (strip accents, lowercase, NFKC folding, NMT
    # removal) change visible characters and must refuse.
    norm_type = normalizer.get("type") if isinstance(normalizer, dict) else normalizer
    if norm_type in (None, "null"):
        pass
    elif norm_type == "NFC":
        print("NOTE: tokenizer normalizer is NFC -- decode stays byte-exact "
              "(the normalizer never runs inside the engine); encode of "
              "DECOMPOSED unicode input may pick different tokens than HF. "
              "Covered by the shipped per-model validators.")
    else:
        raise ExportError(
            f"tokenizer carries a mutating normalizer {norm_type!r} -- the "
            f"engine cannot reproduce character-mutating normalization and "
            f"the byte-table mapping is not lossless for it; refusing export")
    # NOTE on pretokenizer splits beyond ByteLevel (Digits/Regex/Sequence):
    # HF merges never cross a pretokenizer split; the engine's score-driven
    # encode merges on the flat score table. Cross-boundary merge-behaviour
    # equivalence for every supported model is covered by the shipped
    # validators (v_encode_validate against the engine-reference encode and
    # the full chat/quality suite per model), not re-proved here.
    merges = tokenizer_data["model"]["merges"]
    # BPE priority = EARLIEST rank: two different pairs can concatenate to the
    # same string (a+bc vs ab+c), and the later dictionary entry must not
    # shadow the earlier pair's rank.
    merge_rank = {}
    for i, m in enumerate(merges):
        rank_token = "".join(tuple(m if isinstance(m, list) else m.split()))
        merge_rank.setdefault(rank_token, i)

    pseudo_scores = {}
    for token in all_tokens:
        rank = merge_rank.get(token)
        pseudo_scores[token] = -math.log(rank + 1) if rank is not None else -1e6

    # The header stores the max MAPPED-string length (each GPT-2-mapped
    # character is >=1 byte, e.g. the Ġ map is 2 bytes for 1 real byte) --
    # this only OVER-allocates a decode buffer, never truncates. It is the
    # upper bound, not the exact byte maximum.
    max_token_length = max(len(t.encode('utf-8')) for t in all_tokens)

    def write_tokenizer(f):
        f.write(struct.pack("<I", max_token_length))
        f.write(struct.pack("<I", model.bos_token_id))
        f.write(struct.pack("<I", model.eos_token_id))
        for token in all_tokens:
            token_bytes = internal_to_bytes(U2B, token)
            f.write(struct.pack("<f", pseudo_scores[token]))
            f.write(struct.pack("<I", len(token_bytes)))
            f.write(token_bytes)

    deferred_write(file + ".tokenizer", write_tokenizer, model._pending_renames)
    print(f"Staged tokenizer to {file}.tokenizer  ({vocab_size} tokens) -- "
          f"published with the full artifact set")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def _render_chat(tokenizer, msgs, thinking, ph, template=None):
    """Render msgs with add_generation_prompt through the OFFICIAL HF renderer
    (tokenizer.apply_chat_template -- the same Jinja environment HF itself
    compiles, with trim_blocks/lstrip_blocks/raise_exception/strftime_now).
    `template` is ignored (kept for call compatibility). Retries with
    structured content when the template iterates content lists
    (Idefics3/SmolVLM/PaliGemma) and the placeholder vanished.
    NOTE: whatever the template bakes in at export time (strftime_now "Today
    Date" lines in Llama-3.x/SmolLM3, Qwen2.5's baked-in default system
    prompt) is FROZEN into the sidecar -- documented engine-side limitation,
    not fixable here."""
    kwargs = {"add_generation_prompt": True}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking

    def _try(msgs_variant):
        return tokenizer.apply_chat_template(msgs_variant, tokenize=False, **kwargs)

    try:
        rendered = _try(msgs)
    except Exception as exc:
        rendered = None
        first_err = exc
    else:
        first_err = None
    if rendered is None or ph not in rendered:
        structured_msgs = []
        for m in msgs:
            structured_msgs.append({
                "role": m["role"],
                "content": [{"type": "text", "text": m["content"]}]
            })
        try:
            rendered = _try(structured_msgs)
        except Exception:
            if first_err is not None:
                raise ExportError(f"chat template rendering failed: {first_err}") from first_err
            raise
    # A template that drops a message, mangles content, or ignores a role
    # changes the COUNT of the placeholder -- the engine's later-turn
    # substitution would misplace content. One placeholder per message whose
    # content IS the sentinel ph (the .multi derivation renders U/P/A
    # sentinels instead of the literal "%s"; counting "%s" there would see
    # zero and abort every multi sidecar).
    expected_placeholders = sum(1 for m in msgs if m["content"] == ph)
    if rendered.count(ph) != expected_placeholders:
        raise ExportError(
            f"chat template rendered {rendered.count(ph)} {ph!r} placeholders "
            f"for {expected_placeholders} sentinel message(s) (thinking={thinking}) -- "
            f"a dropped or mangled message would lose content silently; "
            f"refusing export")
    return rendered


def build_prompts(model, file):
    """Build .template prompt template files. Uses the tokenizer's official
    HF chat renderer (apply_chat_template semantics via the same Jinja
    environment HF itself uses), with a chat_template.jinja fallback.
    Template stages go to deferred temp names; sidecars NOT freshly produced
    register in _pending_unstale for deletion in the final publishing pass
    (a stale .multi from an earlier HF revision must never survive into a
    database built after a switched template layout), and the old sidecars
    themselves stay intact until the whole artifact set is published."""
    suffixes = ["", ".with-thinking", ".with-system",
                ".with-system-and-thinking", ".multi", ".multi.with-thinking"]
    unstale = model._pending_unstale
    for suffix in suffixes:
        stale = file + ".template" + suffix
        if os.path.exists(stale):
            unstale.append(stale)
    tokenizer = model.tokenizer
    chat_template = getattr(tokenizer, "chat_template", None)

    # Some models (e.g. MiniCPM5-1B) keep their chat template in a SEPARATE
    # chat_template.jinja file at the repo root instead of an inline
    # chat_template field in tokenizer_config.json. AutoTokenizer exposes
    # chat_template=None for those, so fall back to reading the file.
    jinja_used = False
    if not chat_template:
        jinja_path = os.path.join(model.model_path, "chat_template.jinja")
        if os.path.exists(jinja_path):
            with open(jinja_path, "r", encoding="utf-8") as jf:
                chat_template = jf.read()
            jinja_used = True
            print(f"Loaded chat template from {jinja_path}")

    if chat_template:
        # Named-template dicts (tool_use etc.): the enable_thinking check
        # below searches TEMPLATE TEXT, not dict keys -- normalize to the
        # default template string first (apply_chat_template itself accepts
        # the dict unchanged).
        if isinstance(chat_template, dict):
            default_template = chat_template.get("default")
            if not isinstance(default_template, str) or not default_template:
                raise ExportError(
                    "chat_template is a named-template dict without a "
                    "usable 'default' entry -- the exporter can only export "
                    "the default variant; refusing export")
            chat_template = default_template
        if jinja_used:
            # install the file-loaded template so the official renderer sees it
            try:
                tokenizer.chat_template = chat_template
            except Exception:
                pass
        # v7 thinking-ON contract (AGENTS.md Sec.151): the thinking generation
        # prompt must OPEN the think block (think-open tag + newline) so the
        # model's first generated token is think CONTENT, never the open tag
        # itself.
        # Family check is data-driven: the non-thinking render carries the
        # EMPTY think block only in families that use the markup, so we append
        # the open tag only there, and only when the render does not already
        # end with it.
        def ensure_think_open(text):
            if (has_thinking and family_think_markup
                    and not text.rstrip().endswith(THINK_OPEN)):
                return text + THINK_OPEN + "\n"
            return text

        # Check if template supports the enable_thinking parameter
        has_thinking = "enable_thinking" in chat_template

        if has_thinking:
            plain_render = _render_chat(tokenizer, [{"role": "user", "content": "%s"}], False, "%s")
            family_think_markup = THINK_OPEN in plain_render
        else:
            family_think_markup = False

        written = []
        for suffix, thinking in [
            ("",                                 False),
            (".with-thinking",                   True),
            (".with-system",                     False),
            (".with-system-and-thinking",        True),
        ]:
            # render with UNIQUE per-message sentinels (never literal "%s" as
            # content: a template that drops a message, echoes content, or
            # carries a literal %s in staged text makes %-counting ambiguous;
            # % chars inside message content cannot confuse a unique marker).
            # Each sentinel is then replaced by exactly one %s, in message
            # order -- the sidecar keeps one %s per message slot.
            roles = ["system"] if "system" in suffix else []
            roles.append("user")
            sentinels = {role: f"@@V7_SLOT_{role}_001@@" for role in roles}
            msgs = [{"role": role, "content": sentinels[role]} for role in roles]
            rendered = _render_chat(tokenizer, msgs, thinking, sentinels[roles[0]])
            for sentinel in sentinels.values():
                if rendered.count(sentinel) != 1:
                    raise ExportError(
                        f"template{suffix}: sentinel {sentinel!r} rendered "
                        f"{rendered.count(sentinel)} times (exactly 1 "
                        f"expected) -- a dropped, echoed or mangled message "
                        f"would lose content silently; refusing export")
                rendered = rendered.replace(sentinel, "%s")
            if thinking and family_think_markup:
                # Only families whose template CARRIES the think markup carry
                # the open-tag contract: a template without the empty
                # delimited style renders identically for thinking on/off.
                rendered = ensure_think_open(rendered)
                if not rendered.rstrip().endswith(THINK_OPEN):
                    raise ExportError(
                        f"thinking template{suffix}: generation prompt does not "
                        f"open the think block -- v7 thinking-ON contract violated")
                n_open = rendered.count(THINK_OPEN)
                if n_open != 1:
                    raise ExportError(
                        f"thinking template{suffix}: {n_open} think-open tags in "
                        f"the generation prompt (expected exactly 1)")
            deferred_write(file + ".template" + suffix,
                           lambda f, r=rendered: f.write(r.encode("utf-8")),
                           model._pending_renames)
            model._pending_unstale.append(file + ".template" + suffix)
            written.append(suffix)

        # Inter-turn templates: derived by sentinel-diffing a turn-1 render
        # against a two-turn render. multi = inter-turn terminator + user
        # header + %s + the generation suffix of the SAME thinking variant.
        # The endswith fence proves the suffix is turn-invariant; exotic
        # templates that violate it skip the sidecar LOUDLY (the engine then
        # falls back to the base template -- fail-safe by design).
        U = "@@V7_USER_7481@@"
        A = "@@V7_ASIST_9352@@"
        P = "@@V7_PREV_2383@@"
        skipped_multi = []
        for suffix, thinking in [(".multi", False), (".multi.with-thinking", True)]:
            r1 = _render_chat(tokenizer, [{"role": "user", "content": U}], thinking, U)
            r2 = _render_chat(tokenizer, [{"role": "user", "content": P},
                                          {"role": "assistant", "content": A},
                                          {"role": "user", "content": U}], thinking, U)
            # each unique sentinel must occur EXACTLY once and in conversation
            # order: a template that echoes or mangles a message makes the
            # boundary arithmetic below point at the wrong positions
            counts_ok = (r1.count(U) == 1 and
                         r2.count(P) == 1 and
                         r2.count(A) == 1 and
                         r2.count(U) == 1)
            ordered_ok = counts_ok and (r2.index(P) < r2.index(A) < r2.rindex(U))
            if not counts_ok or not ordered_ok:
                skipped_multi.append(suffix)
                print(f"Inter-turn template{suffix}: template mangles or "
                      f"repeats sentinel placeholders (counts P/A/U = "
                      f"{r2.count(P)}/{r2.count(A)}/{r2.count(U)}) "
                      f"-- sidecar SKIPPED (engine will use the "
                      f"turn-one prompt for later turns)")
                continue
            gen_suffix = r1[r1.index(U) + len(U):]
            if not r2.endswith(U + gen_suffix):
                skipped_multi.append(suffix)
                print(f"Inter-turn template{suffix}: position-dependent "
                      f"generation suffix -- sidecar SKIPPED (engine will use "
                      f"the turn-one prompt for later turns)")
                continue
            if thinking:
                # Same Sec.151 contract as the turn-1 variants: patch AFTER the
                # endswith fence -- the fence proves the template's own suffix
                # is turn-invariant; the open tag is our addition on top of it.
                gen_suffix = ensure_think_open(gen_suffix)
            a_end = r2.index(A) + len(A)
            u_start = r2.rindex(U)
            multi = r2[a_end:u_start] + "%s" + gen_suffix
            deferred_write(file + ".template" + suffix,
                           lambda f, m=multi: f.write(m.encode("utf-8")),
                           model._pending_renames)
            model._pending_unstale.append(file + ".template" + suffix)
            written.append(suffix)
        if skipped_multi:
            print(f"WARNING: inter-turn sidecars not derivable for this "
                  f"template: {skipped_multi} -- later turns will frame with "
                  f"the turn-one prompt; verify multi-turn behaviour after "
                  f"export")
        print(f"Written prompt templates to {file}.template.*  (HF chat renderer)"
              if not skipped_multi else
              f"Written prompt templates to {file}.template.* (partial: {skipped_multi} skipped)")
    else:
        # No chat_template — synthesize one from the tokenizer's special tokens.
        # Detect ChatML-style models (e.g. SmolLM2, Qwen) by checking whether the
        # tokenizer has <|im_start|> and <|im_end|> tokens. Without this, the
        # generic "User:/Assistant:" fallback fails for ChatML-trained models —
        # they never emit <|im_end|> because the prompt never included
        # <|im_start|>assistant\n, so generation loops forever.
        tok = model.tokenizer
        im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        has_chatml = (im_start_id is not None and im_end_id is not None
                      and im_start_id != tok.unk_token_id
                      and im_end_id != tok.unk_token_id
                      and im_start_id != im_end_id)

        if has_chatml:
            # ChatML format — used by SmolLM2, Qwen2/2.5, etc.
            # assistant turn left open (no <|im_end|>) so model fills it in
            user_tmpl = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
            sys_tmpl = ("<|im_start|>system\n%s<|im_end|>\n"
                        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n")
            for suffix, body in [("", user_tmpl),
                                 (".with-thinking", user_tmpl),
                                 (".with-system", sys_tmpl),
                                 (".with-system-and-thinking", sys_tmpl),
                                 (".multi", "<|im_end|>\n" + user_tmpl),
                                 (".multi.with-thinking", "<|im_end|>\n" + user_tmpl)]:
                deferred_write(file + ".template" + suffix,
                               lambda f, b=body: f.write(b.encode("utf-8")),
                               model._pending_renames)
                model._pending_unstale.append(file + ".template" + suffix)
            print(f"Staged prompt templates to {file}.template.*  (ChatML synthesized)")
        else:
            # Base model — no chat_template, no ChatML tokens.
            # Use bare "%s" so v7s_chat/v7s_generate passes the prompt through
            # without wrapping it in any chat formatting. Base models are trained
            # for text completion, not chat — adding "User:/Assistant:" prefixes
            # produces garbage output. @multi variants are also bare "%s"
            # (identical to base -- db_build dedup stores no key).
            for suffix in suffixes:
                deferred_write(file + ".template" + suffix,
                               lambda f: f.write(b"%s"),
                               model._pending_renames)
                model._pending_unstale.append(file + ".template" + suffix)
            print(f"Staged prompt templates to {file}.template.*  (base model, bare %s)")


# ---------------------------------------------------------------------------
# HuggingFace model wrapper (memory-efficient, no duplicate model)
# ---------------------------------------------------------------------------

# Keys that are engine buffers, not learned parameters: never export refusals.
# Non-learned / buffer keys that legitimately appear in HF checkpoints and are
# NOT learned parameters. Matched by SUFFIX (dot-separated tail) so a real
# learned parameter whose name merely CONTAINS one of these fragments (e.g. a
# loosely named norm.weight around "rotary") cannot escape the
# unrecognized-parameter refusal below.
IGNORED_KEY_PATTERNS = (
    ".rotary_emb.inv_freq", ".rotary_emb.inv_freq_cached",
    ".position_ids", ".masked_bias", "attn.bias", "causal_mask",
    "attention.bias", "attention.masked_bias",
)


class HfModelWrapper:
    """
    Memory-efficient wrapper around a HuggingFace model.

    Instead of creating a separate Transformer (which doubles memory with
    float32 random init), this directly references HF parameters in-place.
    After construction the original HF model can be freed — all tensors
    are still referenced through this wrapper.

    Validates the architecture against a per-tensor exact shape table and
    refuses on any mismatch (wrong orientation, wrong element count, missing
    required tensor, unexpected learned parameter, non-finite value).
    """

    def __init__(self, hf_model, model_path):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ExportError("transformers package required:  pip install transformers")

        config = hf_model.config
        hf_params = dict(hf_model.named_parameters())
        self._consumed = set()
        # cross-artifact atomicity: every builder stages to a deferred temp
        # name and registers (tmp, final) on the model; main() renames the
        # whole set in one pass only after everything succeeded
        self._pending_renames = []
        self._pending_unstale = []

        def take(key, tensor_name):
            self._consumed.add(key)
            if key not in hf_params:
                raise ExportError(f"required tensor {tensor_name} (key '{key}') missing from the checkpoint")
            return hf_params[key]

        def maybe_take(key, tensor_name):
            if key in hf_params:
                self._consumed.add(key)
                return hf_params[key]
            return None

        def require_shape(tensor, expected, tensor_name):
            if tuple(tensor.shape) != expected:
                raise ExportError(
                    f"tensor {tensor_name} has shape {tuple(tensor.shape)}, expected {expected}")

        def take_shape(key, expected, tensor_name):
            t = take(key, tensor_name)
            require_shape(t, expected, tensor_name)
            return t

        # ---- Params ----
        def req_config(attr, field):
            value = getattr(config, attr, None)
            if value is None:
                raise ExportError(f"model config field '{field}' is missing")
            return value

        self.params = type("Params", (), {})()
        self.params.dim = int(req_config("hidden_size", "hidden_size"))
        self.params.n_layers = int(req_config("num_hidden_layers", "num_hidden_layers"))
        self.params.n_heads = int(req_config("num_attention_heads", "num_attention_heads"))
        self.params.n_kv_heads = int(getattr(config, "num_key_value_heads", None) or self.params.n_heads)
        self.params.vocab_size = int(req_config("vocab_size", "vocab_size"))
        self.params.hidden_dim = int(req_config("intermediate_size", "intermediate_size"))
        # load_hf_model refuses a missing rms_norm_eps (REQUIRED_FIELDS), so
        # the field is always present here
        self.params.norm_eps = float(config.rms_norm_eps)
        self.params.max_seq_len = int(getattr(config, "max_position_embeddings", None) or 8192)
        # load_hf_model always sets head_dim (config value or the derived
        # hidden_size / num_attention_heads) -- no fallback here
        self.params.head_dim = int(config.head_dim)

        dim = self.params.dim
        n_layers = self.params.n_layers
        n_heads = self.params.n_heads
        n_kv = self.params.n_kv_heads
        hidden = self.params.hidden_dim
        hd = self.params.head_dim
        vocab = self.params.vocab_size
        q_size = n_heads * hd
        k_size = n_kv * hd
        v_size = n_kv * hd

        if (dim % GS_FIXED) or (hidden % GS_FIXED) or (vocab % GS_FIXED) \
                or (q_size % GS_FIXED) or (k_size % GS_FIXED):
            raise ExportError(
                f"architecture dims must all be multiples of {GS_FIXED} "
                f"(GS=64 serialized contract): dim={dim}, hidden={hidden}, "
                f"vocab={vocab}, q_rows={q_size}, kv_rows={k_size}")

        # ---- Architecture detection ----
        # QK-norm: presence on layer 0 decides -- AND presence means the
        # head_dim-sized form (the only one v7 reads; the dim-sized Phi-3
        # style variant is deliberately ignored). Detection on the key alone
        # would flag Phi-3's dim-sized q_norm as present and die late on the
        # per-layer size check, after writing the checkpoint body.
        qk_norm_key = "model.layers.0.self_attn.q_norm.weight"
        self.has_qk_norm = (qk_norm_key in hf_params
                            and tuple(hf_params[qk_norm_key].shape) == (self.params.head_dim,))
        self._consumed.add(qk_norm_key) if self.has_qk_norm else None

        # Tie word embeddings: cross-checked with the declared config flag.
        self.tie_word_embeddings = "lm_head.weight" not in hf_params
        # kept for the model_export cross-check: the config's declared flag vs
        # the layout+value decision made here
        self.tie_declared_config = getattr(config, "tie_word_embeddings", None)
        tie_declared = self.tie_declared_config
        if tie_declared is False and self.tie_word_embeddings:
            raise ExportError(
                "config declares tie_word_embeddings=false but no lm_head.weight "
                "exists in the checkpoint -- the classifier cannot be built")
        if tie_declared is True and not self.tie_word_embeddings:
            print("config declares tie_word_embeddings=true but lm_head.weight "
                  "exists -- keeping a separate classifier unless values are "
                  "exactly equal")

        # RoPE base (theta): must be finite and integral (the header stores an
        # integer; a fractional theta would change every frequency silently)
        # AND representable as int32 (a struct-level range error must not be
        # the refusal; the named check is), and POSITIVE (zero or negative
        # theta is a broken config, not an exotic base). load_hf_model already
        # refuses a missing rope_theta, so the field is always present here.
        theta = float(config.rope_theta)
        if not math.isfinite(theta) or theta != math.floor(theta):
            raise ExportError(
                f"rope_theta {theta!r} is not a finite integer -- the checkpoint "
                f"header stores rope_base as int32 and cannot represent it; "
                f"refusing export")
        if theta <= 0:
            raise ExportError(
                f"rope_theta {theta!r} is not positive -- zero or negative "
                f"bases are broken frequency semantics, not exotic RoPE; "
                f"refusing export")
        if theta > 2 ** 31 - 1 or theta < -2 ** 31:
            raise ExportError(
                f"rope_theta {theta!r} overflows the header's int32 rope_base "
                f"slot (limit 2^31-1) -- refusing export instead of crashing "
                f"in struct.pack")
        self.rope_base = int(theta)

        # RoPE scaling: only unscaled or llama3 are representable; anything
        # else is refused instead of silently dropping the scaling.
        rs = getattr(config, "rope_scaling", None)
        if rs:
            rt = str(rs.get("rope_type", rs.get("type", ""))).lower()
            factor = float(rs.get("factor", 1.0))
            if rt in ("", "none", "default") and factor == 1.0:
                self.rope_scaling = None
            elif rt == "llama3":
                for key in ("factor", "high_freq_factor", "low_freq_factor",
                            "original_max_position_embeddings"):
                    if key not in rs:
                        raise ExportError(
                            f"rope_scaling llama3 missing required key "
                            f"'{key}': {rs!r}")
                    fv = float(rs[key])
                    if not math.isfinite(fv) or fv <= 0:
                        raise ExportError(
                            f"rope_scaling llama3 key '{key}' = {rs[key]!r} "
                            f"is not a finite positive number -- zero, "
                            f"negative or inf factors corrupt the frequency "
                            f"rescale; refusing export")
                self.rope_scaling = dict(rs)
            else:
                raise ExportError(
                    f"unsupported rope_scaling {rs!r}: v7 supports only unscaled "
                    f"RoPE or rope_type=llama3 -- refusing export instead of "
                    f"silently exporting unscaled RoPE")
        else:
            self.rope_scaling = None

        print(f"  Architecture: has_qk_norm={self.has_qk_norm}, "
              f"tie_embeddings={self.tie_word_embeddings}, rope_base={self.rope_base}, "
              f"rope_scaling={self.rope_scaling}")

        # ---- Token embeddings ----
        self.tok_embeddings = type("TokEmb", (), {})()
        self.tok_embeddings.weight = take_shape(
            "model.embed_tokens.weight", (vocab, dim), "tok_embeddings")

        # ---- Final norm ----
        self.norm = type("Norm", (), {})()
        self.norm.weight = take_shape("model.norm.weight", (dim,), "final_norm")

        # ---- Output (classifier) ----
        # ABI rule (documented here): the checkpoint's shared_classifier flag
        # is decided by torch.equal VALUE comparison at export time -- bit-equal
        # embeddings and lm_head always serialize as ONE block (the model's
        # outputs are IDENTICAL either way; this is an exact-equality storage
        # optimization, never an approximation). A declared-tied model with an
        # explicit, bit-equal lm_head therefore exports shared, loud. A
        # declared-untied model with differing weights keeps both blocks.
        self.output = type("Out", (), {})()
        self.output.weight = maybe_take("lm_head.weight", "lm_head")
        if self.output.weight is not None:
            require_shape(self.output.weight, (vocab, dim), "lm_head")
        else:
            self._consumed.add("lm_head.weight")
            self.output.weight = hf_params["model.embed_tokens.weight"]

        # ---- Layers ----
        self.layers = []
        fused_layout = None  # per-architecture attention/mlp layout must be uniform
        for i in range(n_layers):
            layer = type("Layer", (), {})()
            layer.layer_id = i

            layer.attention_norm = type("Norm", (), {})()
            layer.attention_norm.weight = take_shape(
                f"model.layers.{i}.input_layernorm.weight", (dim,), f"layer{i}.attention_norm")

            layer.ffn_norm = type("Norm", (), {})()
            layer.ffn_norm.weight = take_shape(
                f"model.layers.{i}.post_attention_layernorm.weight", (dim,), f"layer{i}.ffn_norm")

            layer.attention = type("Attn", (), {})()
            attn_prefix = f"model.layers.{i}.self_attn"
            # Detect fused qkv_proj (Phi-3) vs separate q_proj/k_proj/v_proj (Qwen/SmolLM2)
            if f"{attn_prefix}.qkv_proj.weight" in hf_params:
                if fused_layout is None:
                    fused_layout = "fused_qkv"
                elif fused_layout != "fused_qkv":
                    raise ExportError(f"layer {i}: fused qkv_proj after separate projections -- "
                                      f"inconsistent attention layout")
                qkv = take(f"{attn_prefix}.qkv_proj.weight", f"layer{i}.qkv_proj")
                require_shape(qkv, (q_size + k_size + v_size, dim), f"layer{i}.qkv_proj")
                for name, slc in [("wq", slice(0, q_size)), ("wk", slice(q_size, q_size + k_size)), ("wv", slice(q_size + k_size, None))]:
                    obj = type("W", (), {})()
                    obj.weight = qkv[slc]
                    setattr(layer.attention, name, obj)
                print(f"  Layer {i}: split fused qkv_proj ({tuple(qkv.shape)}) into Q/K/V")
                # Fused bias (Phi-3 style qkv_proj.bias) splits exactly like the
                # weight: [q | k | v] along the single axis.
                fused_bias = maybe_take(f"{attn_prefix}.qkv_proj.bias", f"layer{i}.qkv_proj.bias")
                if fused_bias is not None:
                    require_shape(fused_bias, (q_size + k_size + v_size,), f"layer{i}.qkv_proj.bias")
                    for attr, slc in [("bq", slice(0, q_size)),
                                      ("bk", slice(q_size, q_size + k_size)),
                                      ("bv", slice(q_size + k_size, None))]:
                        obj = type("B", (), {})()
                        obj.bias = fused_bias[slc]
                        setattr(layer.attention, attr, obj)
                    print(f"  Layer {i}: split fused qkv_proj.bias into Q/K/V biases")
                else:
                    for attr in ("bq", "bk", "bv"):
                        setattr(layer.attention, attr, None)
            else:
                if fused_layout is None:
                    fused_layout = "separate_qkv"
                elif fused_layout != "separate_qkv":
                    raise ExportError(f"layer {i}: separate projections after fused qkv_proj -- "
                                      f"inconsistent attention layout")
                for name in ["wq", "wk", "wv"]:
                    proj = {"wq": "q_proj", "wk": "k_proj", "wv": "v_proj"}[name]
                    # HF stores (out_features, in_features): q/k/v map dim -> head
                    # size. The tuple's second element is the projection's
                    # in_features, NOT always dim.
                    expected_shape = {
                        "wq": (q_size, dim),
                        "wk": (k_size, dim),
                        "wv": (v_size, dim),
                    }[name]
                    obj = type("W", (), {})()
                    obj.weight = take_shape(f"{attn_prefix}.{proj}.weight",
                                            expected_shape, f"layer{i}.{proj}")
                    setattr(layer.attention, name, obj)

            # wo (o_proj) is ALWAYS separate, fused layout or not: the split
            # projections map dim->head_size, o maps q_size->dim, and
            # layer.attention.wo is mandatory downstream (the quantize phase
            # reads every layer's .wo.weight). Loaded ONCE here, outside both
            # layout branches.
            wo = type("W", (), {})()
            wo.weight = take_shape(f"{attn_prefix}.o_proj.weight",
                                   (dim, q_size), f"layer{i}.o_proj")
            layer.attention.wo = wo

            # QKV biases: exactly all-absent or all-present with exact shapes.
            # A partial layout would silently drop learned bias vectors.
            bias_keys = [f"{attn_prefix}.{p}.bias" for p in ("q_proj", "k_proj", "v_proj")]
            present = [k for k in bias_keys if k in hf_params]
            if len(present) == 0:
                for attr in ("bq", "bk", "bv"):
                    if getattr(layer.attention, attr, None) is None:
                        setattr(layer.attention, attr, None)
            elif len(present) == 3:
                for attr, key, expected in [("bq", bias_keys[0], (q_size,)),
                                            ("bk", bias_keys[1], (k_size,)),
                                            ("bv", bias_keys[2], (v_size,))]:
                    if getattr(layer.attention, attr, None) is not None:
                        continue  # already split from the fused bias
                    obj = type("B", (), {})()
                    obj.bias = take_shape(key, expected, f"layer{i}.{os.path.basename(key)}")
                    setattr(layer.attention, attr, obj)
            else:
                raise ExportError(
                    f"layer {i}: partial QKV bias layout {present} -- refusing to "
                    f"silently drop learned bias vectors (expected none or all "
                    f"of q/k/v biases)")

            # QK-norm per-layer validation (None only when the model has none)
            for prefix, key_part, flag in [("lq", "q_norm", self.has_qk_norm),
                                           ("lk", "k_norm", self.has_qk_norm)]:
                key = f"{attn_prefix}.{key_part}.weight"
                obj = type("W", (), {})()
                obj.weight = maybe_take(key, f"layer{i}.{key_part}")
                if obj.weight is not None and tuple(obj.weight.shape) not in ((hd,), (dim,)):
                    raise ExportError(
                        f"layer {i}.{key_part} has shape {tuple(obj.weight.shape)}, "
                        f"expected ({hd},) (or ({dim},) for the ignored dim-sized "
                        f"Phi-3 style variant)")
                if flag and obj.weight is None:
                    raise ExportError(
                        f"layer {i} is missing {key_part}.weight while layer 0 "
                        f"has q_norm -- a zeroed learned RMSNorm would corrupt "
                        f"all later attention activations; refusing export")
                if not flag and obj.weight is not None and tuple(obj.weight.shape) == (hd,):
                    raise ExportError(
                        f"layer {i} has a head_dim-sized {key_part}.weight but "
                        f"layer 0 does not -- inconsistent QK-norm layout; "
                        f"refusing export")
                if not flag and obj.weight is not None and tuple(obj.weight.shape) == (dim,) and i == 0:
                    # documented Phi-3 equivalence: a dim-sized QK norm is the
                    # per-head-ignored variant -- the v7 engine's has_qk_norm=0
                    # layout has validated against the HF reference (all phi3
                    # chat/validator results PASS); keep the zeros loud, never
                    # quiet
                    print(f"  Layer {i}: dim-sized {key_part} present -- "
                          f"documented Phi-3 style variant, serialized as the "
                          f"has_qk_norm=0 layout (zeros)")
                setattr(layer.attention, prefix, obj)

            # FFN (SwiGLU: gate + down + up)
            layer.feed_forward = type("FF", (), {})()
            mlp_prefix = f"model.layers.{i}.mlp"
            if f"{mlp_prefix}.gate_up_proj.weight" in hf_params:
                gate_up = take(f"{mlp_prefix}.gate_up_proj.weight", f"layer{i}.gate_up_proj")
                require_shape(gate_up, (2 * hidden, dim), f"layer{i}.gate_up_proj")
                h = gate_up.shape[0] // 2  # gate and up are equal width
                for attr, slc in [("w1", slice(0, h)), ("w3", slice(h, None))]:
                    obj = type("W", (), {})()
                    obj.weight = gate_up[slc]
                    setattr(layer.feed_forward, attr, obj)
                print(f"  Layer {i}: split fused gate_up_proj ({tuple(gate_up.shape)}) into gate/up")
            else:
                for attr, proj in [("w1", "gate_proj"), ("w3", "up_proj")]:
                    obj = type("W", (), {})()
                    obj.weight = take_shape(f"{mlp_prefix}.{proj}.weight",
                                            (hidden, dim), f"layer{i}.{proj}")
                    setattr(layer.feed_forward, attr, obj)
            # w2 (down projection) — always separate
            obj = type("W", (), {})()
            obj.weight = take_shape(f"{mlp_prefix}.down_proj.weight",
                                    (dim, hidden), f"layer{i}.down_proj")
            layer.feed_forward.w2 = obj

            self.layers.append(layer)

        # ---- Unknown learned parameters: refuse, do not silently drop ----
        leftover = [k for k in hf_params
                    if k not in self._consumed
                    and not any(k.endswith(p) for p in IGNORED_KEY_PATTERNS)]
        if leftover:
            raise ExportError(
                f"unrecognized learned parameters in the checkpoint "
                f"({len(leftover)} keys, first: {leftover[:6]}) -- refusing "
                f"export instead of silently discarding them")

        # ---- Finite check on every bound tensor (norms/biases serialize raw) ----
        def check_group(label, tensors):
            for i, t in enumerate(tensors):
                assert_finite(t, f"{label}[{i}]")
        check_group("attention_norm", [l.attention_norm.weight for l in self.layers])
        check_group("ffn_norm", [l.ffn_norm.weight for l in self.layers])
        assert_finite(self.norm.weight, "final_norm")
        if self.has_qk_norm:
            check_group("q_norm", [l.attention.lq.weight for l in self.layers])
            check_group("k_norm", [l.attention.lk.weight for l in self.layers])
        for i, l in enumerate(self.layers):
            for attr in ("bq", "bk", "bv"):
                b = getattr(l.attention, attr, None)
                if b is not None:
                    assert_finite(b.bias, f"layer{i}.{attr}")

        # Free the large dict — all tensors are still referenced by layers
        del hf_params

        # ---- Tokenizer ----
        # use_fast: the tokenizer build reads backend_tokenizer's BPE data and
        # pre_tokenizer/decoder types -- only a FAST tokenizer carries that
        # Rust structures dict; a slow fallback would Attribute crash later.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise ExportError(
                "could not load a FAST tokenizer for this model -- the "
                "exporter inspects backend_tokenizer (BPE merges, byte-level "
                "pre_tokenizer); refusing to guess from a slow tokenizer")
        self.bos_token_id = int(getattr(config, "bos_token_id", None) or 0)
        # eos is None-checked: 0 is a legal token id; the loader refuses an
        # UNRESOLVED eos before any consumer sees None here
        eos_id = getattr(config, "eos_token_id", None)
        if eos_id is None:
            eos_id = 0
        self.eos_token_id = int(eos_id)
        for label, tid in (("bos_token_id", self.bos_token_id), ("eos_token_id", self.eos_token_id)):
            if tid < 0 or tid >= self.params.vocab_size:
                raise ExportError(f"{label} {tid} is outside 0..{self.params.vocab_size - 1}")

        # Model source directory — used by build_prompts() to find a
        # chat_template.jinja file when tokenizer_config.json has no inline
        # chat_template (e.g. MiniCPM5 stores its template in a separate file).
        self.model_path = model_path

    def eval(self):
        return self


# ---------------------------------------------------------------------------
# Special-token resolution (config -> generation_config -> tokenizer string)
# ---------------------------------------------------------------------------

def _load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _coerce_id_list(value):
    """List -> the FIRST entry (HF's list convention is [primary, fallback...]:
    Qwen writes eos_token_id: [151645, 151643] = [<|im_end|> turn terminator,
    endoftext fallback], so the last entry would pick the WRONG terminator);
    None -> None."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def resolve_special_token_ids(model_path, config_dict):
    """Resolve BOS/EOS token IDs with a documented precedence chain and no
    silent min/max guessing:

      1. config.json  (list -> FIRST entry = the primary/turn terminator;
         an explicit 0 is a legal id -- SmolLM2 exports bos 0)
      2. generation_config.json (list -> first)
      3. tokenizer_config.json <bos|eos>_token STRING -> token ID lookup
      4. well-known marker strings against tokenizer.json added_tokens

      Nothing else: the smallest/largest special-ID heuristic was removed --
      those slots can be arbitrary control tokens (padding, tool markers) and
      a wrong pick silently changes framing or termination. Unresolved BOS
      stays 0 with a loud print (SmolLM2-style models legitimately carry 0);
      an unresolved EOS is a named export error.
    """
    bos_id = _coerce_id_list(config_dict.get("bos_token_id"))
    eos_id = _coerce_id_list(config_dict.get("eos_token_id"))
    # "is not None" on purpose: an EXPLICIT 0 is a legal id (SmolLM2 bos=0),
    # only an absent key falls down the chain
    sources = {"bos": "config" if bos_id is not None else "unset",
               "eos": "config" if eos_id is not None else "unset"}

    gen = _load_json_if_exists(os.path.join(model_path, "generation_config.json"))
    tokcfg = _load_json_if_exists(os.path.join(model_path, "tokenizer_config.json"))

    tok_json = _load_json_if_exists(os.path.join(model_path, "tokenizer.json")) or {}
    added = tok_json.get("added_tokens", [])
    special_by_content = {t["content"]: t["id"] for t in added if t.get("special", False)}
    vocab_entries = tok_json.get("model", {}).get("vocab", {})
    # Unigram tokenizers (Gemma family) store the vocab as [piece, score]
    # PAIRS; dict(vocab_entries) would map piece -> score and int(vid) below
    # would use a float score as a token ID. Only BPE's piece->id dict is
    # usable here.
    id_by_content = dict(vocab_entries) if isinstance(vocab_entries, dict) else {}
    id_by_content.update(special_by_content)

    for kind in ("bos", "eos"):
        # An explicit config value of 0 is a LEGAL id (SmolLM2's bos_token_id
        # is 0) -- dropping it to "unset" would let the marker-string and
        # special-id heuristics override it with a different token. Only
        # absent (None) moves down the chain. Accepting an EOS 0 here is
        # harmless: the caller refuses an unresolved EOS (`if not
        # config.eos_token_id`) right after this loop.
        if sources[kind] != "unset":
            continue
        if gen is not None:
            value = _coerce_id_list(gen.get(f"{kind}_token_id"))
            if value is not None:
                if kind == "bos":
                    bos_id = int(value)
                else:
                    eos_id = int(value)
                sources[kind] = "generation_config"
                continue
        if tokcfg is not None:
            token_str = tokcfg.get(f"{kind}_token")
            if isinstance(token_str, dict):
                token_str = token_str.get("content")
            if isinstance(token_str, str) and token_str:
                vid = id_by_content.get(token_str)
                if vid is not None:
                    if kind == "bos":
                        bos_id = int(vid)
                    else:
                        eos_id = int(vid)
                    sources[kind] = "tokenizer_config string"
                    continue
        # marker strings: the <|...|> markup has been eaten by HTML-ish
        # tooling before (a pasted literal came out as "*)&lt"), so every
        # piped candidate is assembled from fragments. The set carries the
        # classic SentencePiece/GPT-2 terminators plus the Qwen turn
        # terminator.
        def _mark(inner):
            return "<" + "|" + inner + "|" + ">"

        if kind == "eos":
            candidates = ("</s>", _mark("endoftext"), _mark("im_end"),
                          _mark("end"), _mark("eot_id"), _mark("eos"))
        else:
            candidates = ("<s>", _mark("begin_of_text"),
                          _mark("bos"), _mark("eos"))
        for cand in candidates:
            vid = special_by_content.get(cand)
            if vid is not None:
                if kind == "bos":
                    bos_id = int(vid)
                else:
                    eos_id = int(vid)
                sources[kind] = f"marker string {cand!r}"
                break
        if sources[kind] != "unset":
            continue
        # NO smallest/largest special-ID heuristic: those are arbitrary
        # control slots (padding, masked tokens, tool markers) and picking one
        # silently changes generation termination or prompt framing. All
        # v7-supported models declare their ids explicitly; if an upstream
        # revision stops doing that, the export must be fixed, not guessed.
        raise ExportError(
            f"{kind}_token_id could not be resolved: absent or null in "
            f"config.json, absent in generation_config.json, its "
            f"tokenizer_config string maps to nothing, and none of the "
            f"known terminator markers match a special token -- a wrong "
            f"{kind.upper()} would corrupt every generated turn; re-exporting "
            f"requires a resolved id")
    if sources["bos"] == "unset":
        # documented BOS fallback: bos 0 is a legal id and SmolLM2-style
        # models genuinely have it -- but it must be LOUD, never silent
        bos_id = 0
        print("WARNING: bos_token_id resolved to NOTHING (no config value, no "
              "generation_config, no tokenizer string, no special tokens) -- "
              "staying at 0; verify this matches the model's training")
    if sources["eos"] == "unset":
        eos_id = None  # explicit UNRESOLVED marker; the caller refuses
    return bos_id, eos_id, sources


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_hf_model(model_path):
    """
    Load a HuggingFace model and return a memory-efficient wrapper.

    Bypasses transformers.from_pretrained() which crashes in torch 2.7.0+cpu
    (STATUS_ACCESS_VIOLATION in torch/storage.py.__getitem__ during meta-model
    load). Instead, reads config.json directly and loads weights from
    safetensors files via safetensors.torch.load_file().

    Shard loading honours model.safetensors.index.json when present (exact
    referenced files, no duplicate keys, complete index coverage) and refuses
    ambiguous sharded layouts without a manifest.
    """
    import glob as _glob

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise ExportError(f"config.json not found in {model_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    REQUIRED_FIELDS = ("hidden_size", "num_hidden_layers", "num_attention_heads",
                       "vocab_size", "intermediate_size", "rms_norm_eps")
    missing = [k for k in REQUIRED_FIELDS if k not in config_dict]
    if missing:
        raise ExportError(f"config.json missing required fields: {missing}")

    # ---- Semantics allow-list (the forward path implements ONE architecture
    # family; anything else exports fine and runs numerically wrong) ----
    model_type = str(config_dict.get("model_type", "")).lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ExportError(
            f"model_type {model_type!r} is outside v7's supported set "
            f"{SUPPORTED_MODEL_TYPES} -- the forward path implements the "
            f"Llama/Qwen-style SwiGLU+RMSNorm+full-RoPE family only; refusing "
            f"export instead of producing a numerically wrong checkpoint")
    hidden_act = str(config_dict.get("hidden_act", "silu")).lower()
    if hidden_act not in SUPPORTED_HIDDEN_ACTS:
        raise ExportError(
            f"hidden_act {hidden_act!r} is outside v7's supported set "
            f"{SUPPORTED_HIDDEN_ACTS} (SwiGLU gate/up) -- refusing export")
    mu_fields = [k for k in MICROUP_SCALE_FIELDS if k in config_dict]
    if mu_fields:
        raise ExportError(
            f"classic MiniCPM microcircuit-scaling fields present: {mu_fields} "
            f"-- the v7 forward path implements none of the uP scaling and "
            f"would run numerically wrong; refusing export")
    prf = config_dict.get("partial_rotary_factor")
    if prf is not None and float(prf) != 1.0:
        raise ExportError(
            f"partial_rotary_factor {prf!r} unsupported -- v7 applies RoPE to "
            f"every head-dim position; refusing export")
    # Qwen2.5 declares a sliding_window CAPACITY by default (e.g. 32768) with
    # "use_sliding_window": false -- a dormant field, NOT in-use semantics.
    # Refuse only when the sliding window would actually engage.
    uses_sliding = bool(config_dict.get("use_sliding_window"))
    layer_types = config_dict.get("layer_types")
    if layer_types and any(str(t).lower() == "sliding_attention" for t in layer_types):
        uses_sliding = True
    if uses_sliding:
        raise ExportError(
            f"sliding-window attention would engage (use_sliding_window="
            f"{config_dict.get('use_sliding_window')!r}, layer_types="
            f"{layer_types!r}) -- v7 has no sliding-window attention; "
            f"refusing export")

    config = type("Cfg", (), {})()
    config.hidden_size = int(config_dict["hidden_size"])
    config.num_hidden_layers = int(config_dict["num_hidden_layers"])
    config.num_attention_heads = int(config_dict["num_attention_heads"])
    # an EXPLICIT null falls back to full heads (same as an absent key);
    # present non-null values are validated, never replaced
    kv_heads = config_dict.get("num_key_value_heads")
    if kv_heads is None:
        kv_heads = config.num_attention_heads
    config.num_key_value_heads = int(kv_heads)
    config.vocab_size = int(config_dict["vocab_size"])
    config.intermediate_size = int(config_dict["intermediate_size"])
    config.rms_norm_eps = float(config_dict["rms_norm_eps"])
    config.max_position_embeddings = int(
        config_dict.get("max_position_embeddings", config_dict.get("seq_length", 8192))
    )
    config.head_dim = int(
        # "head_dim": null (explicit None) must fall to the derived value --
        # dict.get's default only covers an ABSENT key, not a null one
        config_dict.get("head_dim")
        or (config.hidden_size // config.num_attention_heads)
    )
    # architecture integrity at the boundary: every count must be positive and
    # GQA must divide cleanly (a fractional kv group would corrupt head split)
    for field, value in (("hidden_size", config.hidden_size),
                         ("num_hidden_layers", config.num_hidden_layers),
                         ("num_attention_heads", config.num_attention_heads),
                         ("num_key_value_heads", config.num_key_value_heads),
                         ("vocab_size", config.vocab_size),
                         ("intermediate_size", config.intermediate_size),
                         ("head_dim", config.head_dim),
                         ("max_position_embeddings", config.max_position_embeddings)):
        if value <= 0:
            raise ExportError(
                f"{field} must be positive, got {value!r} -- refusing export")
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ExportError(
            f"num_attention_heads {config.num_attention_heads} is not a "
            f"multiple of num_key_value_heads {config.num_key_value_heads} "
            f"-- GQA grouping impossible; refusing export")
    if config.hidden_size % config.num_attention_heads != 0:
        raise ExportError(
            f"hidden_size {config.hidden_size} is not a multiple of "
            f"num_attention_heads {config.num_attention_heads} -- head split "
            f"impossible; refusing export")
    # RoPE theta: REQUIRED. Newer HF configs may move it under
    # rope_parameters; read that form too, but never silently default --
    # a 100k-theta model exported at 1e6 produces wrong positional encoding.
    rope_theta = config_dict.get("rope_theta")
    if rope_theta is None:
        rp = config_dict.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = rp.get("rope_theta", rp.get("base"))
    if rope_theta is None:
        raise ExportError(
            "config.json has no rope_theta (and no rope_parameters.rope_theta) "
            "-- a silently defaulted RoPE base would corrupt positional "
            "encoding; refusing export")
    config.rope_theta = float(rope_theta)
    # RoPE scaling: newer transformers configs move type/factor under
    # rope_parameters (older: rope_scaling). Read BOTH; when they disagree at
    # the type level the export is ambiguous -- refuse instead of silently
    # picking one (the wrapper refuses unsupported types itself).
    rope_scaling = config_dict.get("rope_scaling")
    rope_params = config_dict.get("rope_parameters")
    if rope_params is not None:
        # normalize rope_parameters into a rope_scaling-shaped dict
        if isinstance(rope_params, dict) and "rope_type" not in rope_params:
            rope_params = {k: v for k, v in rope_params.items()}
        if rope_scaling is None:
            rope_scaling = rope_params
        else:
            # type-alias-aware comparison: any semantic key both blocks define
            # must agree (rope type, factor, freq factors, original context) --
            # a disagreeing value makes the RoPE treatment ambiguous and the
            # refusal is the only safe outcome. Values compare numerically
            # when both parse as float (32 vs 32.0 is EQUALITY, not conflict).
            def _equivalent(a, b):
                if str(a).lower() == str(b).lower():
                    return True
                try:
                    return float(a) == float(b)
                except (TypeError, ValueError):
                    return False
            a_norm = {str(k).lower(): v for k, v in (rope_scaling or {}).items()}
            if "type" in a_norm and "rope_type" not in a_norm:
                a_norm["rope_type"] = a_norm.pop("type")
            b_norm = {str(k).lower(): v for k, v in (rope_params or {}).items()}
            disagreeing = [k for k, v in a_norm.items() if k in b_norm and not _equivalent(v, b_norm[k])]
            if disagreeing:
                raise ExportError(
                    f"rope_scaling and rope_parameters disagree on {disagreeing} "
                    f"-- the RoPE treatment would be ambiguous; refusing export")
            # the explicit rope_scaling block wins where both carry data
    config.rope_scaling = rope_scaling
    config.tie_word_embeddings = config_dict.get("tie_word_embeddings", None)
    config.bos_token_id, config.eos_token_id, tok_sources = resolve_special_token_ids(model_path, config_dict)
    print(f"Token ID resolution: BOS from {tok_sources['bos']} -> {config.bos_token_id}, "
          f"EOS from {tok_sources['eos']} -> {config.eos_token_id}")
    if config.eos_token_id is None:
        raise ExportError(
            "eos_token_id could not be resolved from config.json, "
            "generation_config.json or the tokenizer metadata -- an unresolved "
            "EOS would change generation termination; refusing export")

    print(f"Config: dim={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"vocab={config.vocab_size}, head_dim={config.head_dim}, "
          f"rope_theta={config_dict.get('rope_theta')}, "
          f"bos_id={config.bos_token_id}, eos_id={config.eos_token_id}")

    # Load weights directly from safetensors (avoids transformers init crash)
    print(f"Loading weights from: {model_path}")
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ExportError("safetensors package required:  pip install safetensors")

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    safetensors_files = sorted(_glob.glob(os.path.join(model_path, "*.safetensors")))
    pytorch_bin_files = sorted(_glob.glob(os.path.join(model_path, "pytorch_model*.bin")))
    source_files = {"config.json": os.path.join(model_path, "config.json")}

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        if not weight_map:
            raise ExportError("shard index carries an empty weight_map -- refusing")
        # every shard reference must be a RELATIVE, plain filename: an absolute
        # path or traversal (..) would read outside the model directory
        for rf in set(weight_map.values()):
            if (os.path.isabs(rf) or ".." in rf or rf.startswith("/")
                    or "\\\\" in rf or rf.startswith("~")):
                raise ExportError(f"shard index references a non-relative file: {rf!r}")
        referenced = sorted(set(weight_map.values()))
        for rf in referenced:
            if not os.path.exists(os.path.join(model_path, rf)):
                raise ExportError(f"shard index references missing file: {rf}")
        extras = [os.path.basename(sf) for sf in safetensors_files
                  if os.path.basename(sf) not in referenced]
        if extras:
            raise ExportError(
                f"checkpoint has safetensors files not listed in the shard index: "
                f"{extras} -- refusing ambiguous weights")
        weight_files = [os.path.join(model_path, rf) for rf in referenced]
        use_safetensors = True
        index_keys = set(weight_map.keys())
        source_files["model.safetensors.index.json"] = index_path
        print(f"  Shard index: {len(referenced)} file(s), {len(index_keys)} tensors")
        # per-shard DECLARED content check: the keys each file carries must be
        # EXACTLY the keys the index maps to that file (a swapped-shard index
        # or an extra key would silently load the wrong bytes per tensor)
        for rf in referenced:
            declared = {k for k, shard in weight_map.items() if shard == rf}
            from safetensors import safe_open
            with safe_open(os.path.join(model_path, rf), framework="pt") as sf_handle:
                actual = set(sf_handle.keys())
            if actual != declared:
                only_decl = sorted(declared - actual)[:4]
                only_act = sorted(actual - declared)[:4]
                raise ExportError(
                    f"shard {rf}: declared-vs-actual key mismatch (missing "
                    f"{only_decl}, extra {only_act}) -- the index would point "
                    f"at the wrong shard bytes; refusing export")
    elif len(safetensors_files) > 1:
        raise ExportError(
            f"{len(safetensors_files)} .safetensors files without "
            f"model.safetensors.index.json -- a sharded checkpoint without a "
            f"manifest cannot be verified complete; refusing export")
    elif safetensors_files:
        print(f"  Found 1 safetensors file")
        weight_files = safetensors_files
        use_safetensors = True
    elif pytorch_bin_files:
        if len(pytorch_bin_files) > 1 and not os.path.exists(index_path):
            raise ExportError(
                f"{len(pytorch_bin_files)} pytorch_model*.bin shards without "
                f"model.safetensors.index.json -- shard completeness cannot "
                f"be verified and a missing shard would fail silently as a "
                f"'required tensor missing' later; refusing export")
        print(f"  Found {len(pytorch_bin_files)} pytorch_model.bin file(s)")
        weight_files = pytorch_bin_files
        use_safetensors = False
    else:
        raise ExportError(f"no .safetensors or pytorch_model.bin files found in {model_path}")

    for rf in weight_files:
        source_files[os.path.basename(rf)] = rf
    for opt in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                "merges.txt", "vocab.json", "chat_template.jinja"):
        p = os.path.join(model_path, opt)
        if os.path.exists(p):
            source_files[opt] = p

    # IMPORTANT: Keep original dtype (fp16/bf16) — do NOT convert to fp32 here.
    # Every finite BF16/FP16 value is exactly representable in FP32, so the
    # per-tensor .float() inside quantize_q80/serialize_fp32 loses nothing; a
    # global fp32 copy would only double peak memory.
    state_dict = {}
    for wf in weight_files:
        print(f"  Loading {os.path.basename(wf)} ...")
        if use_safetensors:
            sd = load_file(wf)
        else:
            sd = torch.load(wf, map_location="cpu", weights_only=True)
        for k, v in sd.items():
            if k in state_dict:
                raise ExportError(
                    f"duplicate tensor key '{k}' across weight files -- the "
                    f"latter shard would silently win; refusing export")
            state_dict[k] = v
        del sd
        gc.collect()

    if use_safetensors and os.path.exists(index_path):
        missing_index = index_keys - set(state_dict.keys())
        if missing_index:
            raise ExportError(f"shard index tensors not found in any file (first: {sorted(missing_index)[:6]})")

    total_elems = sum(v.numel() for v in state_dict.values())
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    print(f"  Loaded {len(state_dict)} tensors, "
          f"total elements: {total_elems}, "
          f"total size: {total_bytes / 1e9:.2f} GB")

    hf_model = type("HFModel", (), {})()
    hf_model.config = config
    hf_model._state_dict = state_dict

    def named_parameters():
        return state_dict
    hf_model.named_parameters = named_parameters

    print("Building memory-efficient wrapper...")
    model = HfModelWrapper(hf_model, model_path)
    print(f"  Config: dim={model.params.dim}, layers={model.params.n_layers}, "
          f"heads={model.params.n_heads}, kv_heads={model.params.n_kv_heads}, "
          f"vocab={model.params.vocab_size}, head_dim={model.params.head_dim}")

    model._source_files = source_files
    # The wrapper copied every tensor reference out of the dict; drop ALL live
    # handles now (the named_parameters closure and the attribute both pin
    # tensors the quantize phase frees with setattr(..., None) -- a retained
    # dict would make those frees no-ops and defeat the low-memory design).
    del hf_model.named_parameters
    del hf_model._state_dict
    gc.collect()
    return model


# ---------------------------------------------------------------------------
# Checkpoint export
# ---------------------------------------------------------------------------

# File layout after 256-byte header (must match memory_map_weights in db_build_tool.c):
#   1. att_norm weights: n_layers × dim       (fp32)
#   2. ffn_norm weights: n_layers × dim       (fp32)
#   3. final_norm weight: dim                 (fp32)
#   4. q_norm: n_layers × head_dim  (fp32; zeros when the model has no QK norm)
#   5. k_norm: n_layers × head_dim  (fp32; zeros when the model has no QK norm)
#   6. QKV biases: n_layers × dsum            (fp32; zeros when the model has none)
#   7. Quantized weights: tok, wq, wk, wv, wo, w1, w2, w3, [wcls]


def expected_checkpoint_size(p, n_kv_heads, head_dim, shared_classifier, group_size, hidden_dim):
    """Exact serialized byte count from the layout contract (header + norms +
    QK norms + biases + every quantized [q][s] block). Used to verify the
    checkpoint before it is published."""
    q_size = p.n_heads * head_dim
    k_size = n_kv_heads * head_dim
    dsum = q_size + k_size + k_size
    total = 256
    total += (2 * p.n_layers + 1) * p.dim * 4                       # norms
    total += 2 * p.n_layers * head_dim * 4                          # QK norms (always)
    total += p.n_layers * dsum * 4                                  # QKV biases (always)
    quants = [
        p.vocab_size * p.dim,                                       # tok embeddings
    ]
    quants += [p.dim * p.n_heads * head_dim] * p.n_layers           # wq
    quants += [p.dim * n_kv_heads * head_dim] * (2 * p.n_layers)    # wk + wv
    quants += [p.n_heads * head_dim * p.dim] * p.n_layers           # wo
    quants += [p.dim * hidden_dim] * (2 * p.n_layers)               # w1 + w3
    quants += [hidden_dim * p.dim] * p.n_layers                     # w2
    if not shared_classifier:
        quants.append(p.dim * p.vocab_size)                         # classifier
    for numel in quants:
        total += numel + ((numel + group_size - 1) // group_size) * 4
    return total


def model_export(model, filepath, group_size=GS_FIXED):
    """
    Export model weights in Q8_0 format to the .bin checkpoint file.

    Writes to a temporary file, verifies the exact expected byte count
    against the layout contract, then atomically replaces the target -- a
    failure never truncates a previously good checkpoint. Quantized Q chunks
    stream straight to the file; only per-group scales are buffered (small).
    """
    p = model.params

    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    head_dim = p.head_dim
    has_qk_norm = model.has_qk_norm
    rope_base = model.rope_base
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)

    # Cross-check the declared tie flag against the exact-value decision.
    # tie_word_embeddings is None only when the config carried no flag; the
    # wrapper's own layout+value decision (model.tie_word_embeddings) is
    # authoritative for the file layout.
    tie_declared = getattr(model, "tie_declared_config", None)
    if tie_declared is True and not shared_classifier and not model.tie_word_embeddings:
        raise ExportError(
            "config declares tie_word_embeddings=true but the loaded lm_head "
            "differs from the token embeddings -- refusing to pick one silently")
    if shared_classifier and model.tie_word_embeddings is False:
        # untied by layout but numerically equal: legal, but loud
        print("NOTE: lm_head exists separately yet is bit-equal to the token "
              "embeddings -- reusing the embedding block as the classifier")

    expected = expected_checkpoint_size(p, n_kv_heads, head_dim, shared_classifier,
                                        group_size, p.hidden_dim)
    print(f"Export config: group_size={group_size}, shared_cls={shared_classifier}, "
          f"has_qk_norm={has_qk_norm}, rope_base={rope_base}, expected_size={expected}")

    q_size = p.n_heads * head_dim
    k_size = n_kv_heads * head_dim
    dsum_bias = q_size + k_size + k_size
    zero_bias = b"\0" * (dsum_bias * 4)
    head_norm_zero = b"\0" * (head_dim * 4)

    tmp = filepath + ".tmp_export"
    count = 0
    family_metrics = {}

    def note_family(name, metrics):
        if name.startswith("layer"):
            fam = name.split(".", 1)[1] if "." in name else name
        else:
            fam = name
        entry = family_metrics.setdefault(fam, [0.0, 0.0, 0])
        entry[0] = max(entry[0], metrics["max_abs"])
        entry[1] += metrics["mean_abs"]
        entry[2] += 1

    try:
        out_file = open(tmp, "wb")

        # ---- Write header (256 bytes) ----
        version = 0  # reserved field, always 0 (per AGENTS.md Sec.96)
        out_file.write(struct.pack("<II", 0, version))  # reserved0 + reserved1 (always 0)

        hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
        header = struct.pack(
            "<iiiiiiiiiiiiii",  # 14 ints = 56 bytes (reserved0+1 already written)
            int(p.dim),                # dim
            int(hidden_dim),           # hidden_dim (FFN intermediate size)
            int(p.n_layers),           # n_layers
            int(p.n_heads),            # n_heads
            int(n_kv_heads),           # n_kv_heads
            int(p.vocab_size),         # vocab_size
            int(p.max_seq_len),        # seq_len (max positions)
            int(head_dim),             # head_dim
            int(shared_classifier),    # shared_classifier
            int(group_size),           # group_size
            int(has_qk_norm),          # has_qk_norm (1=Qwen3, 0=SmolLM2/Phi-3)
            int(rope_base),            # rope_base (1000000=Qwen3, 100000=SmolLM2)
            0,                         # reserved2 (power-of-2 padding)
            0,                         # reserved3 (power-of-2 padding)
        )
        out_file.write(header)

        # ---- RoPE scaling block (header bytes 64-83, 20 bytes) ----
        # Layout (must match db_build_tool.c RopeScalingHeader):
        #   rope_type           int32   0=none, 1=llama3
        #   factor              float32
        #   high_freq_factor    float32
        #   low_freq_factor     float32
        #   orig_max_pos        int32
        rs = model.rope_scaling
        if rs:
            rope_type = 1
            factor = float(rs["factor"])
            high_freq_factor = float(rs["high_freq_factor"])
            low_freq_factor = float(rs["low_freq_factor"])
            orig_max_pos = int(rs["original_max_position_embeddings"])
        else:
            rope_type = 0
            factor = 1.0
            high_freq_factor = 4.0
            low_freq_factor = 1.0
            orig_max_pos = 0
        out_file.write(struct.pack("<ifffi", rope_type, factor, high_freq_factor,
                                   low_freq_factor, orig_max_pos))
        print(f"RoPE scaling header: type={rope_type}, factor={factor}, "
              f"high_freq={high_freq_factor}, low_freq={low_freq_factor}, "
              f"orig_max_pos={orig_max_pos}")

        # ---- RMS norm epsilon (header byte 84, 4 bytes) ----
        out_file.write(struct.pack("<f", p.norm_eps))
        print(f"RMS norm epsilon: {p.norm_eps}")

        pad = 256 - out_file.tell()
        if pad < 0:
            raise ExportError(
                f"header grew to {out_file.tell()} bytes -- exceeds the 256-byte "
                f"header ABI; refusing export (this is a code bug, not model data)")
        out_file.write(b"\0" * pad)

        # ---- FP32 norm weights ----
        print("Writing norm weights (fp32)...")
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention_norm.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.ffn_norm.weight)
        serialize_fp32(out_file, model.norm.weight)

        # ---- QK-norm weights (fp32) ----
        # Always written (the C reader always advances past them). When
        # has_qk_norm=0 the model's dim-sized q_norm/k_norm (e.g. Phi-3) are
        # deliberately ignored: writing dim-sized weights would shift every
        # quantized weight offset -- head_dim zeros keep offsets aligned.
        # The wrapper guarantees that a head_dim-sized learned QK norm never
        # hides behind has_qk_norm=0.
        print("Writing QK-norm weights (fp32)...")
        for layer in model.layers:
            if has_qk_norm:
                serialize_fp32(out_file, layer.attention.lq.weight)
            else:
                out_file.write(head_norm_zero)
        for layer in model.layers:
            if has_qk_norm:
                serialize_fp32(out_file, layer.attention.lk.weight)
            else:
                out_file.write(head_norm_zero)

        # ---- QKV biases (fp32) ----
        # Always written (the forward kernels unconditionally add this vector
        # to the QKV matmul output; zeros are a no-op). The wrapper guarantees
        # all-or-none per layer, so a partial layout cannot reach this point.
        print(f"Writing QKV biases (fp32, dsum={dsum_bias} per layer)...")
        for layer in model.layers:
            bq = getattr(layer.attention, "bq", None)
            bk = getattr(layer.attention, "bk", None)
            bv = getattr(layer.attention, "bv", None)
            if bq is not None:
                serialize_fp32(out_file, bq.bias)
                serialize_fp32(out_file, bk.bias)
                serialize_fp32(out_file, bv.bias)
            else:
                out_file.write(zero_bias)

        # ---- Quantized weights ----
        # Order must match memory_map_weights exactly: tok_emb, then ALL wq,
        # ALL wk, ALL wv, ALL wo, ALL w1, ALL w2, ALL w3, [wcls] -- each weight
        # type's tensors must be contiguous (init_quantized_tensors per type).
        print("Quantizing and writing weights...")

        def write_q(q_tensor, s_tensor):
            serialize_int8(out_file, q_tensor)
            serialize_fp32(out_file, s_tensor)

        def _write_quantized(tensor, name):
            nonlocal count
            total_elems = tensor.numel()
            if total_elems % group_size:
                raise ExportError(
                    f"tensor {name} numel {total_elems} is not a multiple of "
                    f"GS={group_size} -- the flat [q][s] layout requires whole "
                    f"groups; refusing export")
            if total_elems <= 50_000_000:
                count += 1
                q, s, metrics = quantize_q80(tensor, group_size, name)
                write_q(q, s)
                del q, s
                print(f"  {count:3d} {name:30s} {str(tuple(tensor.shape)):20s}")
                print_quant_metrics(metrics)
                note_family(name, metrics)
                return
            # Chunked quantize: rows of ~50M elements; Q chunks stream to the
            # [all_q][all_s] layout in order, scales buffered (small) and
            # appended ONCE after the last Q chunk. The loop writes Q bytes
            # only -- write_q would interleave the scales and double them.
            # Chunk boundaries must sit on GS=64 GROUP boundaries in flattened
            # element coordinates (each chunk is quantized independently): with
            # width W, a chunk of R rows covers R*W elements, and
            # R*W % 64 == 0 must hold -- otherwise a serialized group would be
            # split across chunks and the engine would read the wrong scales.
            width = tensor.shape[1]
            from math import gcd
            group_per_rows = group_size // gcd(width % group_size or group_size, group_size)
            base_rows = max(1, 50_000_000 // width)
            chunk_rows = ((base_rows + group_per_rows - 1) // group_per_rows) * group_per_rows
            if (chunk_rows * width) % group_size:
                # unreachable after the gcd alignment -- a math slip here must
                # refuse loudly, never misalign the group stream
                raise ExportError(
                    f"chunk sizing failed for {name}: rows {chunk_rows} width "
                    f"{width} is not GS={group_size}-aligned; refusing export")
            max_err = 0.0
            written = 0
            all_s = []
            while written < tensor.shape[0]:
                end = min(written + chunk_rows, tensor.shape[0])
                chunk = tensor[written:end]
                count += 1
                q, s, metrics = quantize_q80(chunk, group_size, f"{name}[{written}:{end}]")
                serialize_int8(out_file, q)
                all_s.append(s)
                del q, s, chunk
                if metrics["max_abs"] > max_err:
                    max_err = metrics["max_abs"]
                written = end
            for s in all_s:
                serialize_fp32(out_file, s)
            del all_s
            print(f"  {count:3d} {name:30s} chunked, rows {tensor.shape[0]}, "
                  f"total_max_err={max_err:.8f}")
            note_family(name, {"max_abs": max_err, "mean_abs": 0.0})

        # 1. Token embeddings
        _write_quantized(model.tok_embeddings.weight, "tok_emb")
        del model.tok_embeddings.weight
        gc.collect()

        # 2. All wq (n_layers), then all wk, all wv, all wo
        for name in ["wq", "wk", "wv", "wo"]:
            for i, layer in enumerate(model.layers):
                w = getattr(layer.attention, name).weight
                _write_quantized(w, f"layer{i}.{name}")
                setattr(layer.attention, name, type("W", (), {"weight": None})())

        # 3. All w1, all w2, all w3
        for name in ["w1", "w2", "w3"]:
            for i, layer in enumerate(model.layers):
                w = getattr(layer.feed_forward, name).weight
                _write_quantized(w, f"layer{i}.{name}")
                setattr(layer.feed_forward, name, type("W", (), {"weight": None})())

        # 4. Classifier (if not shared)
        if not shared_classifier:
            _write_quantized(model.output.weight, "classifier")

        out_file.flush()
        os.fsync(out_file.fileno())
        out_file.close()

        # ---- Verify the exact layout size before publishing ----
        actual = os.path.getsize(tmp)
        if actual != expected:
            os.remove(tmp)
            raise ExportError(
                f"checkpoint size mismatch: wrote {actual} bytes, layout "
                f"contract expects {expected} -- refusing to publish a "
                f"truncated or oversized checkpoint")
    except BaseException:
        try:
            out_file.close()
        except Exception:
            pass
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    # NOT renamed here: the checkpoint registers on the model's pending list
    # and the whole artifact set (tokenizer + templates + checkpoint) is
    # renamed in main()'s single publishing pass -- a failure anywhere leaves
    # the previous full set intact instead of a half-new mixed set.
    model._pending_renames.append((tmp, filepath))
    print(f"Staged checkpoint to {filepath}  ({count} quantized weight groups) -- "
          f"published with the full artifact set")

    if family_metrics:
        print("Per-family quantization error summary:")
        for fam, (mx, mean_sum, n) in sorted(family_metrics.items()):
            print(f"  {fam:10s} tensors={n:3d}  max_abs={mx:.3e}  mean_abs_avg={mean_sum / max(n, 1):.3e}")


# ---------------------------------------------------------------------------
# Export manifest
# ---------------------------------------------------------------------------

def git_revision():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def write_manifest_from_disk(filepath):
    """--manifest-only: rebuild the manifest from the ON-DISK artifacts (the
    config block comes straight from the published checkpoint header, so a
    failed late phase never forces a full re-quantization)."""
    manifest = {
        "exporter": "tools/export.py",
        "exporter_revision": git_revision(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_files": {},
        "outputs": {},
        "config": {},
        "note": ("manifest-only run: config block read from the published "
                 "checkpoint header; artifacts were NOT regenerated"),
    }
    with open(filepath, "rb") as f:
        header = f.read(256)
    (dim, hidden, n_layers, n_heads, n_kv, vocab, seq_len, head_dim,
     shared, gs, has_qk_norm, rope_base, _r2, _r3) = struct.unpack_from("<iiiiiiiiiiiiii", header, 8)
    with open(filepath + ".tokenizer", "rb") as f:
        _max_len, bos, eos = struct.unpack_from("<III", f.read(12))
    manifest["config"] = dict(dim=dim, n_layers=n_layers, n_heads=n_heads,
                              n_kv_heads=n_kv, head_dim=head_dim,
                              vocab_size=vocab, hidden_dim=hidden,
                              shared_classifier=shared, group_size=gs,
                              has_qk_norm=has_qk_norm, rope_base=rope_base,
                              bos_token_id=bos, eos_token_id=eos)
    manifest_dir = os.path.dirname(os.path.abspath(filepath))
    hf_dir = os.path.dirname(filepath)
    import glob as _glob
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "generation_config.json",
                 "merges.txt", "vocab.json", "model.safetensors.index.json"):
        path = os.path.join(hf_dir, name)
        if os.path.exists(path):
            manifest["source_files"][name] = sha256_file(path)
    for path in sorted(_glob.glob(os.path.join(hf_dir, "*.safetensors"))):
        manifest["source_files"][os.path.basename(path)] = sha256_file(path)
    rev = os.path.join(hf_dir, "REVISION.txt")
    if os.path.exists(rev):
        with open(rev, "r", encoding="ascii") as fh:
            upstream = {}
            for ln in fh.read().splitlines():
                if ": " in ln:
                    k, v = ln.split(": ", 1)
                    upstream[k] = v
            manifest["upstream_revision"] = upstream
    outputs = [filepath, filepath + ".tokenizer"]
    for suffix in ["", ".with-thinking", ".with-system", ".with-system-and-thinking",
                   ".multi", ".multi.with-thinking"]:
        outputs.append(filepath + ".template" + suffix)
    missing = [p for p in outputs if not os.path.exists(p)]
    if missing:
        raise ExportError(
            f"manifest-only: missing artifacts {missing} -- the full export "
            f"must complete first; refusing to certify a partial set")
    for path in outputs:
        manifest["outputs"][os.path.basename(path)] = sha256_file(path)
    stage_write(filepath + ".manifest.json",
                lambda f: (f.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")),
                           f.flush(), os.fsync(f.fileno())))
    print(f"Written export manifest to {filepath}.manifest.json (from on-disk artifacts)")


def write_manifest(model, filepath, templates_only=False):
    manifest = {
        "exporter": "tools/export.py",
        "exporter_revision": git_revision(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_files": {},
        "outputs": {},
    }
    if not templates_only:
        manifest["config"] = {
            "dim": model.params.dim,
            "n_layers": model.params.n_layers,
            "n_heads": model.params.n_heads,
            "n_kv_heads": model.params.n_kv_heads,
            "head_dim": model.params.head_dim,
            "vocab_size": model.params.vocab_size,
            "hidden_dim": model.params.hidden_dim,
            "rms_norm_eps": model.params.norm_eps,
            "rope_base": model.rope_base,
            "has_qk_norm": model.has_qk_norm,
            "bos_token_id": model.bos_token_id,
            "eos_token_id": model.eos_token_id,
        }
    else:
        manifest["config"] = {"templates_only": True}
        # explicit non-certification: the checkpoint/tokenizer hashes below are
        # read from whatever is ON DISK at manifest-write time (they may predate
        # this run entirely) -- only the template sidecars were regenerated
        manifest["note"] = ("templates-only run: checkpoint and tokenizer "
                            "hashes are on-disk state, not outputs of this run")
    for name, path in sorted(getattr(model, "_source_files", {}).items()):
        if os.path.exists(path):
            manifest["source_files"][name] = sha256_file(path)
    if templates_only and not manifest["source_files"]:
        # templates-only runs carry no load-time source list: hash the standard
        # HF files straight from the model directory so the provenance block
        # stays complete
        model_dir = getattr(model, "model_path", "")
        if model_dir:
            import glob as _glob
            standard = ["config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "generation_config.json",
                        "merges.txt", "vocab.json"]
            for name in standard:
                path = os.path.join(model_dir, name)
                if os.path.exists(path):
                    manifest["source_files"][name] = sha256_file(path)
            for path in sorted(_glob.glob(os.path.join(model_dir, "*.safetensors"))):
                manifest["source_files"][os.path.basename(path)] = sha256_file(path)
            rev = os.path.join(model_dir, "REVISION.txt")
            if os.path.exists(rev):
                with open(rev, "r", encoding="ascii") as fh:
                    manifest["upstream_revision"] = {
                        ln.split(": ", 1)[0]: ln.split(": ", 1)[1]
                        for ln in fh.read().splitlines()
                        if ln.startswith(("repo:", "revision:"))
                    }
    outputs = [filepath, filepath + ".tokenizer"]
    for suffix in ["", ".with-thinking", ".with-system", ".with-system-and-thinking",
                   ".multi", ".multi.with-thinking"]:
        outputs.append(filepath + ".template" + suffix)
    for path in outputs:
        if os.path.exists(path):
            manifest["outputs"][os.path.basename(path)] = sha256_file(path)
    stage_write(filepath + ".manifest.json",
                lambda f: (f.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")),
                           f.flush(), os.fsync(f.fileno())))
    print(f"Written export manifest to {filepath}.manifest.json")


def publish_pending(model, filepath):
    """THE one publishing pass: renames every staged artifact, removing
    sidecars that existed before but are not part of the fresh set.

    ATOMICITY DOCUMENTATION: os.replace per file is atomic, but a SEQUENCE of
    them is not an atomic transaction as a set. To keep close to the
    documented 'failure keeps previous artifacts intact' promise, this pass
    BACKS UP every overwritten target first and ROLLS BACK on any failure
    (disk full / permission mid-sequence), leaving the old set byte-identical.
    A power loss mid-pass remains a mixed-set window -- documented here, with
    the staged temp names keeping the window minimal."""
    staged_finals = {final for (_tmp, final) in model._pending_renames}
    for stale in model._pending_unstale:
        if stale not in staged_finals and os.path.exists(stale):
            os.remove(stale)
            print(f"Removed stale sidecar: {stale}")
    backups = []
    published = []
    try:
        for tmp, final in model._pending_renames:
            if os.path.exists(final):
                bak = final + ".tmp_export_bak"
                os.replace(final, bak)  # move the old set aside, cheap + atomic
                backups.append((bak, final))
            os.replace(tmp, final)
            published.append((tmp, final))
            print(f"Published: {final}")
    except BaseException:
        # roll back: restore every backup, undo every published rename
        for tmp, final in published:
            try:
                os.replace(final, tmp)   # the fresh file goes back to the temp slot
            except OSError:
                pass
        for bak, final in reversed(backups):
            os.replace(bak, final)
            print(f"Rolled back: {final}")
        raise
    model._pending_renames = []
    model._pending_unstale = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export HuggingFace model to v7 checkpoint format"
    )
    parser.add_argument("filepath", type=str,
                        help="Output path (without .bin extension). E.g. weights/foo/model")
    parser.add_argument("hfpath", type=str,
                        help="Path to the DOWNLOADED model directory (the HF "
                             "files live on disk; fetch first via "
                             "tools/hf_download.py inside a fetch_*.bat)")
    parser.add_argument("--force", action="store_true",
                        help="Uniform overwrite verb for pipeline bats: the exporter "
                             "always regenerates its outputs (kept for caller "
                             "compatibility -- existence of old artifacts is never a "
                             "reason to skip)")
    parser.add_argument("--templates-only", action="store_true",
                        help="Regenerate only the .template* sidecars (no weight/tokenizer export). "
                             "Used when only the template derivation changed -- build_db.exe reads "
                             "the sidecars next to the existing checkpoint.")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Rewrite ONLY the export manifest from the on-disk "
                             "artifacts (config block read from the published "
                             "checkpoint header) -- recovers a run whose late "
                             "phase failed without re-quantizing the model.")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.filepath))
    os.makedirs(out_dir, exist_ok=True)

    if args.manifest_only:
        write_manifest_from_disk(args.filepath)
        sys.exit(0)

    if args.templates_only:
        # Tokenizer-only load (seconds, no weights) -> same build_prompts path.
        # The manifest is REFRESHED: the tokenizer files' hashes changed, and a
        # stale manifest pointing at old template bytes would defeat its own
        # provenance purpose.
        from transformers import AutoTokenizer
        tok_only = type("TokOnly", (), {})()
        tok_only.tokenizer = AutoTokenizer.from_pretrained(args.hfpath, use_fast=True)
        tok_only.model_path = args.hfpath
        tok_only._pending_renames = []
        tok_only._pending_unstale = []
        build_prompts(tok_only, args.filepath)
        publish_pending(tok_only, args.filepath)
        write_manifest(tok_only, args.filepath, templates_only=True)
        print(f"Templates regenerated: {args.filepath}.template*")
        sys.exit(0)

    # NOTE: a previous manifest is NEVER deleted at startup -- it belongs to
    # the previous good artifact set and may be the only provenance the user
    # has when a run fails. The fresh manifest is staged and published in
    # publish_pending()'s single pass together with the built artifacts.

    # Load model (single HF load, memory-efficient)
    model = load_hf_model(args.hfpath)

    # Build tokenizer + prompt templates + quantized checkpoint (all staged to
    # deferred temp names), then publish the whole artifact set in ONE pass --
    # a failure anywhere renames nothing and the previous good set stays
    # intact. The manifest hashes are read AFTER publishing.
    build_tokenizer(model, args.filepath)
    build_prompts(model, args.filepath)
    model_export(model, args.filepath)
    publish_pending(model, args.filepath)
    write_manifest(model, args.filepath)

    print("\nExport complete.")
    print(f"  Checkpoint:  {args.filepath}")
    print(f"  Tokenizer:   {args.filepath}.tokenizer")
    print(f"  Templates:   {args.filepath}.template*")
    print(f"  Manifest:    {args.filepath}.manifest.json")
