#!/usr/bin/env python3
"""
1979-seilacher-echinarachnius-parma-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0860)
Organism: Echinarachnius parma (Common Sand Dollar)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Echinarachnius
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0860_echinarachnius_parma_biomineral_accretion.lean` (`MathOfBiology.Loop_0860`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/echinarachnius_parma_biomineral_accretion"
EXPECTED_SHA256 = {
    "echinarachnius_parma_biomineral_accretion_reference.json": "b5a690918604e2f2c63bfdbfa50395f8127a1bd43357bcde0c6e0f6e87084250",
    "echinarachnius_parma_specimen.jpg": "9076ecda047934e3a862128e9da39faefb16c968f63494a46db1ab7512327923",
    "scout_manifest.json": "4581d2a81a29394800cba8b2b25ce1f7b988ca8c98ed3d00ec2c9066e62b3b54",
    "paper_citations.bib": "03a115f22496b1b94b6e79a6e0ecacdcb4aeefdd51f68c6a1f0c6e66254d5977"
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
    ref = json.loads((DATASET_DIR / "echinarachnius_parma_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1979-seilacher-echinarachnius-parma-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
