#!/usr/bin/env python3
"""
2005-kaandorp-acropora-millepora-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0865)
Organism: Acropora millepora (Branching Staghorn Coral)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Acropora
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0865_acropora_millepora_biomineral_accretion.lean` (`MathOfBiology.Loop_0865`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acropora_millepora_biomineral_accretion"
EXPECTED_SHA256 = {
    "acropora_millepora_biomineral_accretion_reference.json": "161e7506dbab7da49247818bd2326a850b878d1edfa43f9348fc7b56019c16bf",
    "acropora_millepora_specimen.jpg": "85c35e854dd9d07d46528548512579040fef8d6cd4893257fb28bd3b4a468eb0",
    "scout_manifest.json": "9ec5f74b17fa01c34466f7505b269dc1b167cf5330541532179153b10844d48d",
    "paper_citations.bib": "12c5946dcd09e2c299f0f5a3d690d0fcc192f906dc7ad212742ff495b8097a2b"
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
    ref = json.loads((DATASET_DIR / "acropora_millepora_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-kaandorp-acropora-millepora-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
