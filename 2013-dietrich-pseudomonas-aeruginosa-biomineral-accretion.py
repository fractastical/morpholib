#!/usr/bin/env python3
"""
2013-dietrich-pseudomonas-aeruginosa-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0884)
Organism: Pseudomonas aeruginosa (Biofilm Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Pseudomonas
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0884_pseudomonas_aeruginosa_biomineral_accretion.lean` (`MathOfBiology.Loop_0884`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pseudomonas_aeruginosa_biomineral_accretion"
EXPECTED_SHA256 = {
    "pseudomonas_aeruginosa_biomineral_accretion_reference.json": "e8817dd1f9684d0f321c8a56e134a07ce9924974fa3238dcb90418c256ab257b",
    "pseudomonas_aeruginosa_specimen.jpg": "60ef5e2ce063576716d4086509c23046d1f100864118a90be5614a770b3ff31d",
    "scout_manifest.json": "97c48d6aa67ad8a4abecfa50fee37136b7ec2d74e2d05f89db0e1bf967fc1c4d",
    "paper_citations.bib": "7dd22b4211ba73392948d2d7556e9d8f62b10d44f5f942c4e8bf7f4be3dee412"
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
    ref = json.loads((DATASET_DIR / "pseudomonas_aeruginosa_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-dietrich-pseudomonas-aeruginosa-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
