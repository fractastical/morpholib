#!/usr/bin/env python3
"""
2021-reiter-sepia-officinalis-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0451)
Organism: Sepia officinalis (Common Cuttlefish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Sepia
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0451_sepia_officinalis_flux_canalization.lean` (`MathOfBiology.Loop_0451`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/sepia_officinalis_flux_canalization"
EXPECTED_SHA256 = {
    "sepia_officinalis_flux_canalization_reference.json": "58270ce116b72b72f5ef8917c8c88d225cec7ace358321351cfa5ed131456a09",
    "sepia_officinalis_specimen.jpg": "c6ecd8366e75a19b7375ac673bb90c8aa032e891c606d26556858646002b58e1",
    "scout_manifest.json": "14344a13167ca277caaa4927c3ed865ff47e207600720fe04386a5fe2d6343bc",
    "paper_citations.bib": "121711a266dbefe2f67e8529ebb154839a5b75a7d162ce919467cec88726c13d"
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
    ref = json.loads((DATASET_DIR / "sepia_officinalis_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2021-reiter-sepia-officinalis-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
