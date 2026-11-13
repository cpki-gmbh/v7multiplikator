"""Usage:
    python tools\\hf_download.py REPO_ID --local-dir DIR [PATTERN ...]
    python tools\\hf_download.py --verify DIR

Downloads REPO_ID into DIR. With one or more PATTERNs, only files matching
them are fetched (same semantics as the CLI's positional allow-patterns);
without patterns the whole repo is fetched.

Provenance: the resolved upstream commit is written to DIR\\REVISION.txt
together with the SHA-256 of EVERY *.safetensors shard (multi-shard models
included), so a built database can be traced to the exact checkpoint
revision it was exported from (greedy inference is a pure function of
binary + database + prompt). --verify re-hashes the tree against
REVISION.txt and exits 1 on any mismatch or missing shard.
"""
import argparse
import hashlib
import os
import sys

from huggingface_hub import snapshot_download


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def shard_records(local_dir):
    """(relative name, sha256) for every *.safetensors shard under local_dir."""
    records = []
    for root, _dirs, files in os.walk(local_dir):
        for f in sorted(files):
            if f.endswith(".safetensors"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, local_dir).replace("\\", "/")
                records.append((rel, sha256_file(full)))
    return records


def verify_tree(local_dir):
    """Re-hash the tree against REVISION.txt; exit 1 on any drift."""
    rev = os.path.join(local_dir, "REVISION.txt")
    if not os.path.isfile(rev):
        print("[FATAL] no REVISION.txt in %s -- nothing to verify against" % local_dir)
        sys.exit(1)
    expected = {}
    for line in open(rev, encoding="ascii"):
        if " sha256: " in line:
            name, digest = line.strip().split(" sha256: ", 1)
            expected[name] = digest
    if not expected:
        print("[FATAL] REVISION.txt lists no shard checksums -- re-download with hashing")
        sys.exit(1)
    failed = 0
    for name, digest in sorted(expected.items()):
        full = os.path.join(local_dir, name.replace("/", os.sep))
        if not os.path.isfile(full):
            print("  MISSING: %s" % name)
            failed += 1
            continue
        actual = sha256_file(full)
        if actual == digest:
            print("  OK: %s (%s)" % (name, actual[:16] + "..."))
        else:
            print("  CORRUPT: %s got %s expected %s" % (name, actual, digest))
            failed += 1
    if failed:
        print("[FATAL] %d shard(s) failed verification -- re-download before building a database" % failed)
        sys.exit(1)
    print("verify: all %d shard(s) match REVISION.txt" % len(expected))


def main():
    parser = argparse.ArgumentParser(description="Download a HuggingFace repo (version-stable wrapper).")
    parser.add_argument("repo", nargs="?", help="repo id, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--local-dir", help="target directory")
    parser.add_argument("--verify", action="store_true",
                        help="re-hash an existing --local-dir tree against REVISION.txt")
    parser.add_argument("--revision", default=None,
                        help="pin an upstream commit sha (default: current HEAD)")
    parser.add_argument("patterns", nargs="*", help="allow-patterns (omit to fetch the whole repo)")
    args = parser.parse_args()

    if args.verify:
        if not args.local_dir:
            print("[FATAL] --verify needs --local-dir")
            sys.exit(1)
        verify_tree(args.local_dir)
        return

    if not args.repo or not args.local_dir:
        parser.error("download mode needs REPO_ID and --local-dir")

    path = snapshot_download(
        repo_id=args.repo,
        local_dir=args.local_dir,
        revision=args.revision,
        allow_patterns=args.patterns or None,
    )
    print("Downloaded to:", path)

    # resolved commit + per-shard checksums -> REVISION.txt next to the weights
    commit = getattr(path, "commit_hash", None) or args.revision or "unknown"
    lines = ["repo: %s" % args.repo, "revision: %s" % commit]
    for name, digest in shard_records(args.local_dir):
        lines.append("%s sha256: %s" % (name, digest))
        print("%s sha256: %s" % (name, digest))
    if not any(l.endswith(tuple("0123456789abcdef")) for l in lines[2:]):
        print("[WARN] no *.safetensors shards found -- REVISION.txt carries the revision only")
    out = os.path.join(args.local_dir, "REVISION.txt")
    with open(out, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
    print("provenance written:", out)


if __name__ == "__main__":
    main()
