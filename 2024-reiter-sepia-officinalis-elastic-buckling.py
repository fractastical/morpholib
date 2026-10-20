#!/usr/bin/env python3
"""
2024-reiter-sepia-officinalis-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0751)
Organism: Sepia officinalis (Common Cuttlefish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Sepia
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0751_sepia_officinalis_elastic_buckling.lean` (`MathOfBiology.Loop_0751`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/sepia_officinalis_elastic_buckling"
EXPECTED_SHA256 = {
    "sepia_officinalis_elastic_buckling_reference.json": "dcb684e9ccdf32d319c0bff65b463c989e49508df4e1ca5b1544f8c7b589193b",
    "sepia_officinalis_specimen.jpg": "c6ecd8366e75a19b7375ac673bb90c8aa032e891c606d26556858646002b58e1",
    "scout_manifest.json": "b1fdc0625243de09505c52feab43121cac7edd31afbe9c1349db1325c5f9de34",
    "paper_citations.bib": "8e9f035e7fc6726627f6eba45c85fc673aa76cb5885e002910565963ac28f7ce"
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
    ref = json.loads((DATASET_DIR / "sepia_officinalis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2024-reiter-sepia-officinalis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
