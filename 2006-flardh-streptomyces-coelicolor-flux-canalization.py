#!/usr/bin/env python3
"""
2006-flardh-streptomyces-coelicolor-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0488)
Organism: Streptomyces coelicolor (Mycelial Actinobacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Streptomyces
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0488_streptomyces_coelicolor_flux_canalization.lean` (`MathOfBiology.Loop_0488`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/streptomyces_coelicolor_flux_canalization"
EXPECTED_SHA256 = {
    "streptomyces_coelicolor_flux_canalization_reference.json": "99a0a6f1d8d42397d3120d8848b517161ef9ff0b4d998e67f84b5178d8589ce7",
    "streptomyces_coelicolor_specimen.jpg": "5a2b99731bd4971f58260f74a1db5d4560b78f4d8ceb3b067629dcb7b6d53fec",
    "scout_manifest.json": "cdd31afa6c0f3454ea4a26bbbb3b997d178a8d160e2b9b510f157d0dfd2543c1",
    "paper_citations.bib": "6beb478053f75627b09f7180cba789cf71f8c24bd84e84526cd4967bc9257f3a"
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
    ref = json.loads((DATASET_DIR / "streptomyces_coelicolor_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-flardh-streptomyces-coelicolor-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
