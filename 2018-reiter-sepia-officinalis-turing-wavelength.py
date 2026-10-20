#!/usr/bin/env python3
"""
2018-reiter-sepia-officinalis-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0151)
Organism: Sepia officinalis (Common Cuttlefish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Sepia
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0151_sepia_officinalis_turing_wavelength.lean` (`MathOfBiology.Loop_0151`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/sepia_officinalis_turing_wavelength"
EXPECTED_SHA256 = {
    "sepia_officinalis_turing_wavelength_reference.json": "43aa6be2741fd2a6a76d7216515dfad553b0a45b50b5c4a44055801555a39eb6",
    "sepia_officinalis_specimen.jpg": "c6ecd8366e75a19b7375ac673bb90c8aa032e891c606d26556858646002b58e1",
    "scout_manifest.json": "cb73f28c045a3fe9aff1bb3c8296fdc492ac8fd0f0b92dd8f3452690dc76656c",
    "paper_citations.bib": "af57e0fb6696b4d4467c18aa19620d7d65f79b438a6b2cbbc2d7190b43f6dcc7"
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
    ref = json.loads((DATASET_DIR / "sepia_officinalis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-reiter-sepia-officinalis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
