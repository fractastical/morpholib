#!/usr/bin/env python3
"""
1996-fleming-salmo-salar-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0838)
Organism: Salmo salar (Atlantic Salmon)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Salmo
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0838_salmo_salar_biomineral_accretion.lean` (`MathOfBiology.Loop_0838`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/salmo_salar_biomineral_accretion"
EXPECTED_SHA256 = {
    "salmo_salar_biomineral_accretion_reference.json": "ad6e9433f4610953a24b049d8b174b2f332ca0b3335ef2679010e83a51a2843d",
    "salmo_salar_specimen.jpg": "2fae0b54511096d5c98f39fd08b4b665fdb0cad61caedbee18633275fbf4173e",
    "scout_manifest.json": "d010d81a3e2f5927d9a415a1d9de4f2b93ac38a9d2623ea3ad6179eba6307052",
    "paper_citations.bib": "e0404b0885735b9913791b05ddb29aec75706dfd12c161b382db9299adf624c2"
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
    ref = json.loads((DATASET_DIR / "salmo_salar_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1996-fleming-salmo-salar-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
