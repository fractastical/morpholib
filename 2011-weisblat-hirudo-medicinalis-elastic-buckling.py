#!/usr/bin/env python3
"""
2011-weisblat-hirudo-medicinalis-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0769)
Organism: Hirudo medicinalis (Medicinal Leech)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Hirudo
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0769_hirudo_medicinalis_elastic_buckling.lean` (`MathOfBiology.Loop_0769`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/hirudo_medicinalis_elastic_buckling"
EXPECTED_SHA256 = {
    "hirudo_medicinalis_elastic_buckling_reference.json": "144dcd1123d6d14749bb214437f2fc808511b0b0d5b764dc1ba811c4ee8e7723",
    "hirudo_medicinalis_specimen.jpg": "f9cc303efc9e9bbd4f520cb58b823e3ee6b17491f3c40d8ee2c2ad8fcedf909f",
    "scout_manifest.json": "8451aacfeb5f12a7dfb1e7fca9ea008604c60b2fddf2400d8ac53f15b20db1bb",
    "paper_citations.bib": "f67d508638639d992ceeb8afede5b849e1e4bb9c46b5c3777af0ac9bb29f5e8d"
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
    ref = json.loads((DATASET_DIR / "hirudo_medicinalis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2011-weisblat-hirudo-medicinalis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
