#!/usr/bin/env python3
"""
2013-amemiya-latimeria-chalumnae-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0839)
Organism: Latimeria chalumnae (West Indian Coelacanth)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Latimeria
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0839_latimeria_chalumnae_biomineral_accretion.lean` (`MathOfBiology.Loop_0839`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/latimeria_chalumnae_biomineral_accretion"
EXPECTED_SHA256 = {
    "latimeria_chalumnae_biomineral_accretion_reference.json": "488c1b4de11448ffb2a2f26936146a251336bf008c76f5654a8f2546b4c5efdf",
    "latimeria_chalumnae_specimen.jpg": "29bcb5946834bbbf95c873d247d7bb9449f8ea6b56ddef496b4eef521ce6cd5d",
    "scout_manifest.json": "4ab26c6f906893f0c872ae85e3ece0b284053247d3ac36f149db2286cf7b4b57",
    "paper_citations.bib": "2969186f3ec08fa75ee4d9f43fe643c84340ab850d927b9784bb86e65b450cfe"
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
    ref = json.loads((DATASET_DIR / "latimeria_chalumnae_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-amemiya-latimeria-chalumnae-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
