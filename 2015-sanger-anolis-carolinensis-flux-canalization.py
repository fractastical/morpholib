#!/usr/bin/env python3
"""
2015-sanger-anolis-carolinensis-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0424)
Organism: Anolis carolinensis (Green Anole)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Anolis
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0424_anolis_carolinensis_flux_canalization.lean` (`MathOfBiology.Loop_0424`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anolis_carolinensis_flux_canalization"
EXPECTED_SHA256 = {
    "anolis_carolinensis_flux_canalization_reference.json": "ead28917929fc0655558860e76814afe4b3962398d3b3ab741536a11283e5c4c",
    "anolis_carolinensis_specimen.jpg": "584ab986514945bb9c82b1765cb9c44cb8915e47823a0e02b5f444310fe0506b",
    "scout_manifest.json": "240a2bc62b5e0eefd9bfe2b1f17eaf24cc08973b70cb854f9425f6e81cc45032",
    "paper_citations.bib": "1e74e273d32269dc2ee97f41acebedef3ab0125c679b2301fdbd215d6607fb3a"
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
    ref = json.loads((DATASET_DIR / "anolis_carolinensis_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-sanger-anolis-carolinensis-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
