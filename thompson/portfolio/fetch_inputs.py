#!/usr/bin/env python3
"""
Fetch every public input the portfolio scorers need into .thompson_cache/ (gitignored),
verifying checksums where pinned.  Idempotent; safe to run on a CI runner.

    python3 thompson/portfolio/fetch_inputs.py [--only fishbase fishshapes tree taxonomy]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / ".thompson_cache"
UA = {"User-Agent": "Mozilla/5.0 (morpholib portfolio fetcher; github.com/fractastical/morpholib)"}

INPUTS = {
    "tree": [("https://fishtreeoflife.org/downloads/actinopt_12k_treePL.tre.xz", CACHE / "actinopt_12k_treePL.tre.xz", "sha256", "543221541758e2bce063a5a76d6858f4f83a8a770225ee96c4292f7917188da2")],
    "taxonomy": [("https://fishtreeoflife.org/downloads/PFC_taxonomy.csv.xz", CACHE / "PFC_taxonomy.csv.xz", "sha256", "074ff977de01f6a618aedc735c9668523d76cfc5c518fa9c829a520b949265f5")],
    "fishshapes": [("https://zenodo.org/records/6642650/files/FishShapes_specimensv1.csv?download=1", CACHE / "fishshapes_v1" / "FishShapes_specimensv1.csv", "md5", "122b495274d0167c5447db104c7acbfe")],
    # rfishbase release fb-21.06 (Froese & Pauly 2021, CC BY-NC); no checksums published -> recorded in results via sha256 at scoring time
    "fishbase": [(f"https://github.com/ropensci/rfishbase/releases/download/fb-21.06/{t}.tsv.gz", CACHE / "fishbase_21_06" / f"{t}.tsv.gz", None, None)
                 for t in ("species", "families", "morphmet", "morphdat", "swimming", "ecology")],
}


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, algo: str | None, expected: str | None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (algo is None or digest(dest, algo) == expected):
        print(f"  ok      {dest.relative_to(REPO_ROOT)}"); return True
    print(f"  fetch   {url}")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r, dest.open("wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED  {url}: {e}"); return False
    if algo and digest(dest, algo) != expected:
        print(f"  CHECKSUM MISMATCH {dest}: {digest(dest, algo)} != {expected}"); dest.unlink(); return False
    print(f"  saved   {dest.relative_to(REPO_ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)"); return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(INPUTS))
    args = ap.parse_args()
    ok = True
    for group, items in INPUTS.items():
        if args.only and group not in args.only:
            continue
        print(f"[{group}]")
        for url, dest, algo, exp in items:
            ok &= fetch(url, dest, algo, exp)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
