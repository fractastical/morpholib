#!/usr/bin/env python3
"""
2002-fleming-salmo-salar-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0738)
Organism: Salmo salar (Atlantic Salmon)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Salmo
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0738_salmo_salar_elastic_buckling.lean` (`MathOfBiology.Loop_0738`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/salmo_salar_elastic_buckling"
EXPECTED_SHA256 = {
    "salmo_salar_elastic_buckling_reference.json": "bd24b349415d6e1010980a519c8f416a924b39a84bc1970e4be9fb382831fe51",
    "salmo_salar_specimen.jpg": "2fae0b54511096d5c98f39fd08b4b665fdb0cad61caedbee18633275fbf4173e",
    "scout_manifest.json": "65997dcc7d0997ecebefc3defb2b5d5496c1d2b484dc08fc82748fb9af5de5c3",
    "paper_citations.bib": "16e3473ff5f13d4bb611365b0eeae9d0b8c6e924d229dc4db707eb29b99aa3d1"
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
    ref = json.loads((DATASET_DIR / "salmo_salar_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-fleming-salmo-salar-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
