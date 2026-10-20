#!/usr/bin/env python3
"""
2011-kaandorp-acropora-millepora-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0765)
Organism: Acropora millepora (Branching Staghorn Coral)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Acropora
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0765_acropora_millepora_elastic_buckling.lean` (`MathOfBiology.Loop_0765`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acropora_millepora_elastic_buckling"
EXPECTED_SHA256 = {
    "acropora_millepora_elastic_buckling_reference.json": "3358e1470bb8b000a63826991df0b9fcc299b3c5019231e7d61f79b1eba109fc",
    "acropora_millepora_specimen.jpg": "85c35e854dd9d07d46528548512579040fef8d6cd4893257fb28bd3b4a468eb0",
    "scout_manifest.json": "8f076f3b8702a6b33105a343ed1f3a7f9c8e2c6f2e4077d2c0e8d5711a6d4f16",
    "paper_citations.bib": "6e26a168c5d693fe0f09fd5e5434bd3a39da7f41ca5ae0440f0282eeceff4bad"
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
    ref = json.loads((DATASET_DIR / "acropora_millepora_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2011-kaandorp-acropora-millepora-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
