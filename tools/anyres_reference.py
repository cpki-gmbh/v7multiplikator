#!/usr/bin/env python3
"""anyres_reference.py -- print the HF anyres fit math for the gate test sizes.

For each source size this prints, using the HF LLaVA-OneVision processor
formulas (get_patch_output_size + divmod paste offsets) over the engine's
candidate grid (6x6 multiples of image_size=384, least-wasted-area selection,
first candidate wins ties):

    best_resolution   the chosen candidate (h, w)
    new_h/new_w       the aspect-preserving fitted size
    paste_x/paste_y   divmod(pad, 2) top/left offsets

A future vision-baseline mismatch can be triaged against these numbers
without an HF install. Run: python tools/anyres_reference.py [w h] ...
"""
import sys

IMAGE_SIZE = 384
GRID = 6


def candidates():
    for h in range(1, GRID + 1):
        for w in range(1, GRID + 1):
            yield (h * IMAGE_SIZE, w * IMAGE_SIZE)


def get_patch_output_size(orig_w, orig_h, best_w, best_h):
    # HF llava_onevision image_processing: scale = min(best_w/w, best_h/h),
    # each dimension floored to int; the missing pixels are padded (divmod).
    scale = min(best_w / orig_w, best_h / orig_h)
    return int(orig_w * scale), int(orig_h * scale)


def best_resolution(orig_w, orig_h):
    best = None
    best_wasted = None
    for (ch, cw) in candidates():
        new_w, new_h = get_patch_output_size(orig_w, orig_h, cw, ch)
        wasted = cw * ch - new_w * new_h
        if best_wasted is None or wasted < best_wasted:
            best_wasted = wasted
            best = (ch, cw)
    return best


def main():
    sizes = []
    args = sys.argv[1:]
    if args:
        for i in range(0, len(args) - 1, 2):
            sizes.append((int(args[i]), int(args[i + 1])))
    else:
        sizes = [(640, 480), (1001, 200), (200, 1001)]
    for (w, h) in sizes:
        bh, bw = best_resolution(w, h)
        new_w, new_h = get_patch_output_size(w, h, bw, bh)
        paste_x, paste_y = divmod(bw - new_w, 2)[0], divmod(bh - new_h, 2)[0]
        print(f"source {w}x{h}: best_resolution=({bh},{bw}) new_h={new_h} new_w={new_w} "
              f"paste_x={paste_x} paste_y={paste_y} pad_w={bw - new_w} pad_h={bh - new_h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
