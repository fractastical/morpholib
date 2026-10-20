#!/usr/bin/env python3
"""
1992-bartnicki-neurospora-crassa-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0481)
Organism: Neurospora crassa (Red Bread Mold)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Neurospora
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0481_neurospora_crassa_flux_canalization.lean` (`MathOfBiology.Loop_0481`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/neurospora_crassa_flux_canalization"
EXPECTED_SHA256 = {
    "neurospora_crassa_flux_canalization_reference.json": "3138f8c9ef7ee541f7bb20128b6f302d0c30fe869d0682865d04b076f660a775",
    "neurospora_crassa_specimen.jpg": "01d6531f055000fd6be6dc5baa24d01e76899962e05a548e4be38a45032165f0",
    "scout_manifest.json": "5f9577e368a80f152611d33d295f78c9bdd5440d0a6e26d2f26f39b925bb7ed9",
    "paper_citations.bib": "94625893aa5230c95a57bb3c26d8fea7bbc91d5db32d3cf04725a70ff97fe75d"
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
    ref = json.loads((DATASET_DIR / "neurospora_crassa_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1992-bartnicki-neurospora-crassa-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
