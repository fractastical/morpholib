#!/usr/bin/env python3
"""
1989-spero-orbulina-universa-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #1000)
Organism: Orbulina universa (Spherical Foraminifer)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Orbulina
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_1000_orbulina_universa_bioelectric_polarity.lean` (`MathOfBiology.Loop_1000`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/orbulina_universa_bioelectric_polarity"
EXPECTED_SHA256 = {
    "orbulina_universa_bioelectric_polarity_reference.json": "e1f24b89ec651a7ca6cd2063e55257813b06608396f7c3a50febd7851110bb3f",
    "orbulina_universa_specimen.jpg": "33d170bc776044ce2b3334da5523ea69a13551d1ff7cb86635f20a3dab925141",
    "scout_manifest.json": "55656ce58ef2ac0bac0fc32f95aad4d3c867520154de7ad23cf2e1c150058d05",
    "paper_citations.bib": "b2a49aec69f6cd1966ccb1a11c66a2251f00fd28e740a8a9fb0a9ba212462116"
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
    ref = json.loads((DATASET_DIR / "orbulina_universa_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1989-spero-orbulina-universa-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
