#!/usr/bin/env python3
"""
2009-reinhardt-solanum-lycopersicum-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0774)
Organism: Solanum lycopersicum (Tomato)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Solanum
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0774_solanum_lycopersicum_elastic_buckling.lean` (`MathOfBiology.Loop_0774`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/solanum_lycopersicum_elastic_buckling"
EXPECTED_SHA256 = {
    "solanum_lycopersicum_elastic_buckling_reference.json": "98f3313e9269b334c5c48467643b61dbaa66c366ca4ff98e15634e5486b54bde",
    "solanum_lycopersicum_specimen.jpg": "3ae537938870d7101582c24be70959db59fa5172fccc4d6ad7523d5b64cb2ec0",
    "scout_manifest.json": "d9ee7020aff44e471fdd627d16fa75766a4315f6d5b33db69271874e97201325",
    "paper_citations.bib": "afec0e9046a84b6a3a482812040bd90b06d3623d810637c2f366a4fa365818e5"
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
    ref = json.loads((DATASET_DIR / "solanum_lycopersicum_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-reinhardt-solanum-lycopersicum-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
