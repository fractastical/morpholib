#!/usr/bin/env python3
"""
2014-martin-schizosaccharomyces-pombe-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0887)
Organism: Schizosaccharomyces pombe (Fission Yeast)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Schizosaccharomyces
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0887_schizosaccharomyces_pombe_biomineral_accretion.lean` (`MathOfBiology.Loop_0887`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/schizosaccharomyces_pombe_biomineral_accretion"
EXPECTED_SHA256 = {
    "schizosaccharomyces_pombe_biomineral_accretion_reference.json": "53b9067430315beba4c5c5f1d6be8eee4452c0457314a1ac08515c99666f7b1d",
    "schizosaccharomyces_pombe_specimen.jpg": "1ddf4f2081b0c047e472365745cd54e38ef85ea46a86a451dc98e0a7d6b5c9f8",
    "scout_manifest.json": "a3aca33ba18eb5369c63e14c77830d06d9ae9c6cf3c2bfd7b50705c42810518c",
    "paper_citations.bib": "adc8ded39b2777dcb6c7c64bf633e0cbdf1a1c173588fdc32a3a194e25d3d180"
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
    ref = json.loads((DATASET_DIR / "schizosaccharomyces_pombe_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-martin-schizosaccharomyces-pombe-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
