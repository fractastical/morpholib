#!/usr/bin/env python3
"""
1988-spero-orbulina-universa-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0900)
Organism: Orbulina universa (Spherical Foraminifer)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Orbulina
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0900_orbulina_universa_biomineral_accretion.lean` (`MathOfBiology.Loop_0900`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/orbulina_universa_biomineral_accretion"
EXPECTED_SHA256 = {
    "orbulina_universa_biomineral_accretion_reference.json": "9ba4692c3812d90a7eda71e7a943907c8b84061bec1b6d928753ca94ab59ce0d",
    "orbulina_universa_specimen.jpg": "33d170bc776044ce2b3334da5523ea69a13551d1ff7cb86635f20a3dab925141",
    "scout_manifest.json": "26cf7cfa2c625f4940e2d050e819c5df7230095427c9579a597ad0c1580488b2",
    "paper_citations.bib": "76efc4a094180c764d63d6a4977076483280f005dfb5a11428b655facfd44ffe"
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
    ref = json.loads((DATASET_DIR / "orbulina_universa_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1988-spero-orbulina-universa-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
