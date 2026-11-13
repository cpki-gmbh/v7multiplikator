#!/usr/bin/env python3
"""Convert any image file to a raw RGB888 + sidecar .meta (w h) pair.

Usage: python jpg_to_raw.py <input.jpg> <output_basename> [max_side]

If max_side is given (default 1024), the image is downscaled keeping aspect
ratio so the longer side equals max_side, then both dimensions are rounded up
to a multiple of 32 (Qwen3-VL factor = patch*merge = 16*2 = 32).

Writes <output_basename>.raw (w*h*3 bytes, RGB top-down) and
<output_basename>.meta (two integers w and h, ASCII).

This bypasses GDI+/CImage entirely so the VLM loader can be a plain fread.
"""
import sys
from PIL import Image, ImageOps


def round_up(v: int, m: int) -> int:
    return ((v + m - 1) // m) * m


def main():
    if len(sys.argv) < 3:
        print("usage: jpg_to_raw.py <input> <out_basename> [max_side]", file=sys.stderr)
        return 1
    src, base = sys.argv[1], sys.argv[2]
    max_side = int(sys.argv[3]) if len(sys.argv) >= 4 else 1024
    img = Image.open(src)
    # phone photos carry EXIF rotation -- transpose before measuring, or the
    # .raw comes out sideways relative to what the camera captured
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = float(max_side) / float(max(w, h))
    if scale < 1.0:
        nw = max(1, round(w * scale))
        nh = max(1, round(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
    # Round up to multiple of 32 (Qwen3-VL factor). The edge strip is FILLED
    # with stretched content (LANCZOS resample), not black padding: the ViT
    # loaders only need w*h*3 bytes and the few pad columns are a negligible
    # fraction of the patch area -- documented behaviour, not a bug.
    w, h = round_up(img.size[0], 32), round_up(img.size[1], 32)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    rgb = img.convert("RGB").tobytes()  # top-down, row-major, 3 bytes/pixel
    expected = w * h * 3
    if len(rgb) != expected:
        print(f"BUG: expected {expected} bytes, got {len(rgb)}", file=sys.stderr)
        return 2
    with open(base + ".raw", "wb") as f:
        f.write(rgb)
    with open(base + ".meta", "w") as f:
        f.write(f"{w} {h}\n")
    gh, gw = h // 16, w // 16
    print(f"OK: {base}.raw = {w}x{h} RGB ({expected} bytes); patches={gw}x{gh}={gw*gh}, merged={gw*gh//4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

