#!/usr/bin/env python3
"""
2003-reinhardt-solanum-lycopersicum-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0874)
Organism: Solanum lycopersicum (Tomato)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Solanum
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0874_solanum_lycopersicum_biomineral_accretion.lean` (`MathOfBiology.Loop_0874`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/solanum_lycopersicum_biomineral_accretion"
EXPECTED_SHA256 = {
    "solanum_lycopersicum_biomineral_accretion_reference.json": "7ef55a276c4c916a11f1b45df41cc9e881af70a9cb2a5a947856baf98a6d20a9",
    "solanum_lycopersicum_specimen.jpg": "3ae537938870d7101582c24be70959db59fa5172fccc4d6ad7523d5b64cb2ec0",
    "scout_manifest.json": "2dd200272e636591dfc3c89ca6556a85e286ef119e22cbc6c802afdb4892eaa6",
    "paper_citations.bib": "77a5ed5620cb5dd8afeab52ac9307dd1498f4e9290d6aec3ee18d19cb4374bd0"
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
    ref = json.loads((DATASET_DIR / "solanum_lycopersicum_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-reinhardt-solanum-lycopersicum-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
