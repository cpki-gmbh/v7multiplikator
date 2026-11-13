# v7 - part of the v7 inference engine
# Copyright (C) 2026 Gorg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)
#
# test_export_checkpoint.py -- TEST-ONLY independent byte parser for exported
# checkpoints. Verifies what a file-size check cannot:
#   - the 256-byte header fields (dims, GS=64, rope_base int32, eps) and
#     layout size against the config-implied contract,
#   - the serialized tensor ORDER (norms, QK norms, biases, q/k/v/wo/w1/w2/w3,
#     classifier) with per-tensor [q bytes][fp32 scales] placement,
#   - every scale finite + positive and every quantized byte in [-127, 127]
#     (the vpsignb dot-product contract; a same-size permuted tensor or a
#     misaligned scale block fails here).
#
# usage: python tools\test_export_checkpoint.py <model.bin> <tokenizer.bin>
#   The tokenizer file is parsed to cross-check vocab_size and the serialized
#   max_token_length/bos/eos against the checkpoint config.
#
# NOT a runtime component: never called by the DLL or build_db.exe.

import json
import math
import os
import struct
import sys

HEADER_SIZE = 256
GS = 64


def parse_checkpoint(path):
    """Parse a v7 .bin checkpoint; returns (config_dict_like, tensors) where
    tensors is an ordered list of (name, rows, is_quantized, offset)."""
    size = os.path.getsize(path)
    if size < HEADER_SIZE:
        raise ValueError(f"file too small: {size}")
    with open(path, "rb") as f:
        blob = f.read()
    (dim, hidden, n_layers, n_heads, n_kv, vocab, seq_len, head_dim,
     shared, gs, has_qk_norm, rope_base, _r2, _r3) = struct.unpack_from("<iiiiiiiiiiiiii", blob, 8)
    eps_st = struct.unpack_from("<f", blob, 84)[0]
    if gs != GS:
        raise ValueError(f"header group_size {gs} != serialized contract {GS}")
    if not (1e-12 < eps_st < 1):
        raise ValueError(f"header rms_eps {eps_st} is out of sane range")
    p = dict(dim=dim, hidden_dim=hidden, n_layers=n_layers, n_heads=n_heads,
             n_kv_heads=n_kv, vocab_size=vocab, seq_len=seq_len,
             head_dim=head_dim, shared_classifier=shared, group_size=gs,
             has_qk_norm=has_qk_norm, rope_base=rope_base, rms_norm_eps=eps_st)
    q_size = n_heads * head_dim
    k_size = n_kv * head_dim
    dsum = q_size + k_size + k_size

    order = []

    def f32(name, count):
        order.append((name, count, False))
    def qt(name, numel):
        order.append((name, numel, True))

    f32("att_norm", n_layers * dim)
    f32("ffn_norm", n_layers * dim)
    f32("final_norm", dim)
    f32("q_norm", n_layers * head_dim)
    f32("k_norm", n_layers * head_dim)
    f32("qkv_bias", n_layers * dsum)
    qt("tok_emb", vocab * dim)
    # the exporter writes every matrix type for ALL layers before the next
    # type (export.py: all wq, then all wk, all wv, all wo, all w1, all w2,
    # all w3 -- note w2 before w3), each tensor carrying its own scale block
    for i in range(n_layers):
        qt(f"layer{i}.wq", q_size * dim)
    for i in range(n_layers):
        qt(f"layer{i}.wk", k_size * dim)
    for i in range(n_layers):
        qt(f"layer{i}.wv", k_size * dim)
    for i in range(n_layers):
        qt(f"layer{i}.wo", dim * q_size)
    for i in range(n_layers):
        qt(f"layer{i}.w1", dim * hidden)
    for i in range(n_layers):
        qt(f"layer{i}.w2", hidden * dim)
    for i in range(n_layers):
        qt(f"layer{i}.w3", dim * hidden)
    if not shared:
        qt("classifier", dim * vocab)

    tensors = []
    off = HEADER_SIZE
    for name, numel, quantized in order:
        if quantized:
            tensors.append((name, numel, True, off))
            off += numel
            off += ((numel + GS - 1) // GS) * 4
        else:
            tensors.append((name, numel, False, off))
            off += numel * 4
    if off != size:
        raise ValueError(
            f"layout walk ends at {off} but file is {size} bytes -- tensor "
            f"ORDER or per-tensor sizes disagree with the contract")
    p["tensors"] = tensors
    return p, tensors, blob


def verify_quantized(blob, name, numel, offset):
    """Decode every group: scale finite+positive, q bytes in [-127, 127]."""
    s_off = offset + numel
    q = blob[offset:offset + numel]
    bad_q = b"\x80" in q  # -128 as the only out-of-range int8 byte (-128..127)
    if bad_q:
        raise ValueError(f"{name}: a -128 byte broke the vpabsb/vpsignb contract")
    n_groups = (numel + GS - 1) // GS
    scales = struct.unpack_from(f"<{n_groups}f", blob, s_off)
    for i, s in enumerate(scales):
        if not math.isfinite(s) or s <= 0:
            raise ValueError(
                f"{name}: group {i} scale {s!r} is not finite-positive")
    return n_groups


def parse_tokenizer(path, vocab_size):
    """Cross-check the .tokenizer header fields against the checkpoint."""
    with open(path, "rb") as f:
        blob = f.read()
    max_len, bos, eos = struct.unpack_from("<III", blob, 0)
    if not (0 <= bos < vocab_size):
        raise ValueError(f"tokenizer bos id {bos} outside 0..{vocab_size - 1}")
    if not (0 <= eos < vocab_size):
        raise ValueError(f"tokenizer eos id {eos} outside 0..{vocab_size - 1}")
    off = 12
    count = 0
    max_seen = 0
    while off < len(blob):
        score, blen = struct.unpack_from("<fI", blob, off)
        off += 8 + blen
        count += 1
        max_seen = max(max_seen, blen)
    if count != vocab_size:
        raise ValueError(f"tokenizer carries {count} tokens, expected {vocab_size}")
    if max_len < max_seen:
        raise ValueError(
            f"tokenizer max_token_length {max_len} is smaller than the "
            f"longest serialized token ({max_seen} bytes)")
    return dict(max_token_length=max_len, bos=bos, eos=eos)


def main():
    if len(sys.argv) != 3:
        print("usage: python tools\\test_export_checkpoint.py <model.bin> <model.tokenizer>")
        return 1
    p, tensors, blob = parse_checkpoint(sys.argv[1])
    print(f"header ok: dim={p['dim']} hidden={p['hidden_dim']} layers={p['n_layers']} "
          f"heads={p['n_heads']} kv={p['n_kv_heads']} vocab={p['vocab_size']} "
          f"head_dim={p['head_dim']} shared={p['shared_classifier']} gs={p['group_size']} "
          f"qk_norm={p['has_qk_norm']} rope_base={p['rope_base']} eps={p['rms_norm_eps']}")
    tok = parse_tokenizer(sys.argv[2], p["vocab_size"])
    print(f"tokenizer ok: max_len={tok['max_token_length']} bos={tok['bos']} eos={tok['eos']}")
    for name, numel, quantized, offset in p["tensors"]:
        if quantized:
            groups = verify_quantized(blob, name, numel, offset)
            print(f"  {name:12s} numel={numel:9d} groups={groups} scales ok, q bytes in [-127, 127]")
        else:
            vals = struct.unpack_from(f"<{numel}f", blob, offset)
            bad = [i for i, v in enumerate(vals) if not math.isfinite(v)]
            if bad:
                raise ValueError(f"{name}: {len(bad)} non-finite fp32 values "
                                 f"(first at {bad[0]})")
            print(f"  {name:12s} numel={numel:9d} fp32 all finite")
    print("test_export_checkpoint: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())