#!/usr/bin/env python3
"""
Download GloVe 840B 300d vectors for V13 Experiment 5.

Downloads ~2GB zip from Stanford NLP (via HuggingFace mirror), extracts, and verifies.

Run:
    python heiro_v13/scripts/download_glove_840b.py
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_PATH = OUTPUT_DIR / "glove.840B.300d.zip"
TXT_PATH = OUTPUT_DIR / "glove.840B.300d.txt"

# Known file size for sanity check (zip is ~2.03GB)
EXPECTED_MIN_SIZE_BYTES = 2_000_000_000


def download_with_progress(url, dest):
    """Download with progress bar."""
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            sys.stdout.write(f"\r  {mb:.0f} / {total_mb:.0f} MB ({pct:.1f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook)
    print()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted
    if TXT_PATH.exists():
        size = TXT_PATH.stat().st_size
        print(f"Already exists: {TXT_PATH} ({size / 1e9:.2f} GB)")
        return

    # Download zip
    if not ZIP_PATH.exists():
        download_with_progress(URL, ZIP_PATH)
    else:
        print(f"Zip already downloaded: {ZIP_PATH}")

    # Verify size
    zip_size = ZIP_PATH.stat().st_size
    if zip_size < EXPECTED_MIN_SIZE_BYTES:
        ZIP_PATH.unlink()
        raise RuntimeError(
            f"Downloaded file too small ({zip_size / 1e6:.0f} MB). "
            f"Expected >2GB. File deleted -- please retry."
        )
    print(f"  Zip size: {zip_size / 1e9:.2f} GB -- OK")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(OUTPUT_DIR)
    print(f"  Extracted to {TXT_PATH}")

    # Clean up zip
    ZIP_PATH.unlink()
    print("  Removed zip file")

    # Final check
    if not TXT_PATH.exists():
        raise RuntimeError(f"Extraction failed -- {TXT_PATH} not found")

    print(f"\nDone! {TXT_PATH} ({TXT_PATH.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
