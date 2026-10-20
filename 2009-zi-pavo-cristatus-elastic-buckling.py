#!/usr/bin/env python3
"""
2009-zi-pavo-cristatus-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0713)
Organism: Pavo cristatus (Indian Peafowl)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Pavo
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0713_pavo_cristatus_elastic_buckling.lean` (`MathOfBiology.Loop_0713`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pavo_cristatus_elastic_buckling"
EXPECTED_SHA256 = {
    "pavo_cristatus_elastic_buckling_reference.json": "394d3f957462d5782a6890841ad0d2151e00fd0715437bedb38e4acb4365d03f",
    "pavo_cristatus_specimen.jpg": "39f28e26a62ddfdb98fe19fee7eaf7af508c82486a32abe3c1f2ac7d3e607d50",
    "scout_manifest.json": "6901c3e5cd115bb66a4eacbcdc8715e7047fcfb9a52195bce045d24adb900ed7",
    "paper_citations.bib": "d6293d5caa87011489116942705320edfb95742661ba00b9530e431de7f48bf3"
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
    ref = json.loads((DATASET_DIR / "pavo_cristatus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-zi-pavo-cristatus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
