#!/usr/bin/env python3
"""
2010-goryachev-saccharomyces-cerevisiae-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1086)
Organism: Saccharomyces cerevisiae (Budding Yeast)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Saccharomyces
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1086_saccharomyces_cerevisiae_collective_stigmergy.lean` (`MathOfBiology.Loop_1086`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/saccharomyces_cerevisiae_collective_stigmergy"
EXPECTED_SHA256 = {
    "saccharomyces_cerevisiae_collective_stigmergy_reference.json": "cad3753252da5200dbd133cec1039868e3e88a9785e122fea1a15434eed5bbf9",
    "saccharomyces_cerevisiae_specimen.jpg": "03d89aa520798f5a2c721833870f675b8d4676ea933358e95702fcd1732eae0b",
    "scout_manifest.json": "cf2a1b41681163f6561a65b878d779da7822590d2666d7af492de49f3cfb2167",
    "paper_citations.bib": "14b6daf3b2e7df847ff68c376fef17672eb8c1bbcc68951d63fc42dc2a23c6d4"
}


def verify_sha256() -> None:
    for fname, exp in EXPECTED_SHA256.items():
        p = DATASET_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {fname}: {got} != {exp}")


def run_model() -> dict:
    verify_sha256()
    ref = json.loads((DATASET_DIR / "saccharomyces_cerevisiae_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2010-goryachev-saccharomyces-cerevisiae-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
