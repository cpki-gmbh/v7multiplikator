# v7 - part of the v7 inference engine
# Copyright (C) 2026 Görg Pflug & CPKI GmbH
# License: GPL v3 (see LICENSE)

"""
Download files from HuggingFace Hub using direct HTTP with resume support.
Bypasses the Xet protocol which fails for large .bin files.
"""
import hashlib
import os
import sys
import time
import requests
from pathlib import Path

HF_BASE = "https://huggingface.co"

def _finalize(tmp_path, dest_path):
    """Atomically move incomplete file to destination, overwriting if exists."""
    if tmp_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        os.replace(str(tmp_path), str(dest_path))
        return True
    return False

def _sha256_of(path):
    """Streaming SHA-256 of a file (model-scale chunks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, dest_path, expected_size=None, expected_sha256=None, timeout=300):
    """Download a file with resume support."""
    dest_path = Path(dest_path)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".incomplete")

    headers = {}
    mode = "wb"
    if tmp_path.exists():
        existing = tmp_path.stat().st_size
        if expected_size and existing >= expected_size:
            print(f"  Already complete ({existing} bytes), finalizing...")
            _finalize(tmp_path, dest_path)
            return True
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
            print(f"  Resuming from {existing} bytes...")

    session = requests.Session()
    retries = 3
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=headers, stream=True, timeout=timeout)
            if resp.status_code == 416:
                print(f"  Range not satisfiable, file may be complete. Checking...")
                if _finalize(tmp_path, dest_path):
                    return True
                headers.pop("Range", None)
                mode = "wb"
                resp = session.get(url, stream=True, timeout=timeout)
            elif resp.status_code == 206:
                print(f"  Resuming download (HTTP 206)...")
            elif resp.status_code == 200:
                print(f"  Starting fresh download (HTTP 200)...")
                mode = "wb"
            else:
                print(f"  HTTP {resp.status_code}, retrying...")
                time.sleep(5)
                continue

            total = int(resp.headers.get("content-length", 0))
            if mode == "ab":
                existing = tmp_path.stat().st_size
                total += existing
            else:
                existing = 0

            downloaded = existing
            chunk_count = 0
            start = time.time()

            with open(tmp_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=8*1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        chunk_count += 1
                        if chunk_count % 32 == 0:
                            elapsed = time.time() - start
                            rate = (downloaded - existing) / elapsed / 1024 / 1024 if elapsed > 0 else 0
                            pct = downloaded / total * 100 if total > 0 else 0
                            print(f"  {downloaded/1024**3:.2f}GB / {total/1024**3:.2f}GB ({pct:.1f}%) @ {rate:.1f} MB/s")

            elapsed = time.time() - start
            rate = (downloaded - existing) / elapsed / 1024 / 1024 if elapsed > 0 else 0
            print(f"  Complete: {downloaded} bytes in {elapsed:.0f}s @ {rate:.1f} MB/s")

            if expected_size and downloaded < expected_size:
                print(f"  Download too small ({downloaded} < {expected_size}), retrying...")
                headers["Range"] = f"bytes={downloaded}-"
                mode = "ab"
                continue

            # LFS integrity gate: the Hub API hands us the file's sha256 (the
            # LFS oid) -- a download that does not hash to it is corrupt or
            # poisoned. Refuse and remove BOTH copies; never finalize.
            if expected_sha256:
                actual = _sha256_of(tmp_path)
                if actual != expected_sha256:
                    print(f"  [FATAL] sha256 mismatch for {dest_path.name}: "
                          f"got {actual}, expected {expected_sha256} -- removing the corrupt download")
                    tmp_path.unlink(missing_ok=True)
                    if dest_path.exists():
                        dest_path.unlink()
                    return False
                print(f"  sha256 verified: {actual}")

            _finalize(tmp_path, dest_path)
            return True

        except requests.exceptions.ConnectionError as e:
            print(f"  Connection error: {e}")
        except requests.exceptions.Timeout:
            print(f"  Timeout")
        except Exception as e:
            print(f"  Error: {e}")

        if attempt < retries - 1:
            wait = 15 * (attempt + 1)
            print(f"  Retry {attempt+1}/{retries} in {wait}s...")
            time.sleep(wait)
            headers.pop("Range", None)
            mode = "wb"

    return False


def get_file_info(repo_id, filename, token=None):
    """Get file info from HF Hub API."""
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(api_url, headers=headers, timeout=30)
    if resp.status_code != 200:
        print(f"  [WARN] API returned HTTP {resp.status_code}, proceeding without file info")
        return None

    data = resp.json()
    for sibling in data.get("siblings", []):
        if sibling.get("rfilename") == filename:
            return {
                "size": sibling.get("size"),
                "lfs": sibling.get("lfs", {}).get("oid") if sibling.get("lfs") else None,
            }
    return None


def main():
    if len(sys.argv) < 4:
        print("Usage: python download_hf_files.py <repo_id> <dest_dir> <file1> [file2 ...]")
        print("Downloads each file from HF Hub into dest_dir using direct HTTP.")
        sys.exit(1)

    repo_id = sys.argv[1]
    dest_dir = Path(sys.argv[2])
    files = sys.argv[3:]

    token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN", None))

    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)

    success = True
    for filename in files:
        dest_path = dest_dir / filename
        incomplete_path = dest_dir / (filename + ".incomplete")
        # Remove stale incomplete files smaller than expected
        if incomplete_path.exists() and dest_path.exists():
            if incomplete_path.stat().st_size <= dest_path.stat().st_size:
                incomplete_path.unlink()
        if dest_path.exists():
            file_info = get_file_info(repo_id, filename, token)
            expected = file_info["size"] if file_info else None
            if expected and dest_path.stat().st_size >= expected:
                print(f"  [SKIP] {filename} already exists ({dest_path.stat().st_size} bytes)")
                continue

        print(f"\nDownloading {filename}...")
        url = f"{HF_BASE}/{repo_id}/resolve/main/{filename}"

        file_info = get_file_info(repo_id, filename, token)
        expected_size = file_info["size"] if file_info else None
        expected_sha256 = file_info["lfs"] if file_info else None
        if expected_size:
            print(f"  Expected size: {expected_size} bytes ({expected_size/1024**3:.2f} GB)")

        ok = download_file(url, dest_path, expected_size, expected_sha256)
        if ok:
            print(f"  OK: {filename}")
        else:
            print(f"  FAILED: {filename}")
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
