# journal_decoder.py -- decode the v7 tuning journal (v7_tuning_journal).
# The journal is a flat concatenation of 16-byte records, no header, no
# version (the engine writes and reads it, tuning_journal.inc + the plan):
#   offset  size  field
#   0       8     u64 shape_id   = fused [d | n] = (D << 32) | N, little-endian
#   8       4     u32 parameter  = tunable slot index 0..105
#   12      4     u32 new_value  = accepted dword value for that slot
# Slot map (canonical, tune_api.inc):
#   0..23   dispatch params, 24..64 pf_dist_0..40, 65..105 pf_type_0..40
# A pf_dist value is a signed disp32 prefetch displacement; the engine stores
# its raw two's-complement dword, so this decoder prints those slots SIGNED --
# a legal negative distance (e.g. -5450124) must not read as a huge positive.
# Usage: python journal_decoder.py [path]   (default: v7_tuning_journal)
import struct
import sys

SLOT_NAMES = {
    0: "num_chunks", 1: "taper_shift", 2: "batch_small", 3: "batch_medium",
    4: "batch_large", 5: "batch_huge", 6: "batch_med_thresh",
    7: "batch_large_thresh", 8: "batch_huge_thresh", 9: "chunk_multiplier",
    10: "min_chunk_cols", 11: "worker_priority", 12: "quantize_chunk_groups",
    13: "caller_batch_small", 14: "caller_batch_medium",
    15: "caller_batch_large", 16: "caller_batch_huge",
    17: "caller_batch_med_thresh", 18: "caller_batch_large_thresh",
    19: "caller_batch_huge_thresh", 20: "chunk_split_pct",
    21: "chunk_group_1_count", 22: "chunk_group_2_count",
    23: "cores_subtract", 24: "taper_shift_slot1", 36: "quantize_chunk_groups_slot12",
}

def slot_name(idx):
    if 0 <= idx <= 23:
        return SLOT_NAMES[idx]
    if 24 <= idx <= 64:
        return "pf_dist_%d" % (idx - 24)
    if 65 <= idx <= 105:
        return "pf_type_%d" % (idx - 65)
    return "slot_%d" % idx

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "v7_tuning_journal"
    try:
        blob = open(path, "rb").read()
    except OSError as exc:
        print("cannot open %s: %s" % (path, exc))
        return 1
    if len(blob) % 16 != 0:
        print("warning: size %d is not a multiple of 16, torn tail dropped" % len(blob))
    count = len(blob) // 16
    print("journal: %s, %d bytes, %d records" % (path, len(blob), count))
    for i in range(count):
        shape_id, param, value = struct.unpack_from("<QII", blob, i * 16)
        n = shape_id & 0xFFFFFFFF
        d = (shape_id >> 32) & 0xFFFFFFFF
        shown = value
        if 24 <= param <= 64 and value >= 0x80000000:
            shown = value - 0x100000000       # pf_dist: signed disp32, show the true negative
        print("  [%4d] %dx%d  %-28s = %d" % (i, n, d, slot_name(param), shown))
    return 0

if __name__ == "__main__":
    sys.exit(main())
