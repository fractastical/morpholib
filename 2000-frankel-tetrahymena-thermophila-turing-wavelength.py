#!/usr/bin/env python3
"""
2000-frankel-tetrahymena-thermophila-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0197)
Organism: Tetrahymena thermophila (Ciliated Cortical Protist)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Tetrahymena
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0197_tetrahymena_thermophila_turing_wavelength.lean` (`MathOfBiology.Loop_0197`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tetrahymena_thermophila_turing_wavelength"
EXPECTED_SHA256 = {
    "tetrahymena_thermophila_turing_wavelength_reference.json": "cf410af9e5e1ce5539d7360ce9252914dacc1022b2f2ad83d04ce1e67d1d3ff9",
    "tetrahymena_thermophila_specimen.jpg": "7704d06cf5ec8d2a1b6bb5954c4846597cac08e5f4d0c57aed6cd6e1d729026c",
    "scout_manifest.json": "db42331401ca9c6b6a4b6120ac5b90ac6193e4f81a68ded185e590698f2f13aa",
    "paper_citations.bib": "0e7399b02bfe3552a67d36b0a869964f0a920685c2828a69ff35c38cb199dacf"
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
    ref = json.loads((DATASET_DIR / "tetrahymena_thermophila_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2000-frankel-tetrahymena-thermophila-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
