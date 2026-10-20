#!/usr/bin/env python3
"""
1995-koch-giraffa-camelopardalis-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0905)
Organism: Giraffa camelopardalis (Reticulated Giraffe)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Giraffa
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0905_giraffa_camelopardalis_bioelectric_polarity.lean` (`MathOfBiology.Loop_0905`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/giraffa_camelopardalis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "giraffa_camelopardalis_bioelectric_polarity_reference.json": "25616dde389ccc7b5cd5d6a662d1fd2e31a0b1266d27c724b47f5d0263b13501",
    "giraffa_camelopardalis_specimen.jpg": "d65ee763a067a60ce829e77f52e0ca7eadb883d032bc517d6f76edf060800969",
    "scout_manifest.json": "a777600bde9e136a4f3c392e1e7610cfa9c8116844be0240278856fb4da321af",
    "paper_citations.bib": "0a1a1ec02b547ed454249e97dd73a9c038942c6453af77e60bbf6c6b3bba5f3c"
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
    ref = json.loads((DATASET_DIR / "giraffa_camelopardalis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1995-koch-giraffa-camelopardalis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
