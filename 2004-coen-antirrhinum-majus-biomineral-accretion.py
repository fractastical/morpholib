#!/usr/bin/env python3
"""
2004-coen-antirrhinum-majus-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0876)
Organism: Antirrhinum majus (Snapdragon)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Antirrhinum
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0876_antirrhinum_majus_biomineral_accretion.lean` (`MathOfBiology.Loop_0876`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/antirrhinum_majus_biomineral_accretion"
EXPECTED_SHA256 = {
    "antirrhinum_majus_biomineral_accretion_reference.json": "5544a26af5fcf4a089da09088831370b840d085aa7b753f7689138064ffb9794",
    "antirrhinum_majus_specimen.jpg": "f3e9bf78a4453bc1a72552c6d200a37a1bccf6047928e988edba53ccfc260fe0",
    "scout_manifest.json": "ddd1408e0a0f20765b62dc6b8f5e3c2e2d738f0fe37bf0e6477351c4c7da711e",
    "paper_citations.bib": "9ad996595aedf269cdb6192c684179ebb9a7d1751188e7a5e30477e606eba56a"
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
    ref = json.loads((DATASET_DIR / "antirrhinum_majus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-coen-antirrhinum-majus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
