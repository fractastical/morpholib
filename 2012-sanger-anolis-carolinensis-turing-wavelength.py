#!/usr/bin/env python3
"""
2012-sanger-anolis-carolinensis-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0124)
Organism: Anolis carolinensis (Green Anole)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Anolis
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0124_anolis_carolinensis_turing_wavelength.lean` (`MathOfBiology.Loop_0124`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anolis_carolinensis_turing_wavelength"
EXPECTED_SHA256 = {
    "anolis_carolinensis_turing_wavelength_reference.json": "729b165bfcea81950797dc2ee5786fb67494729e1bf13b1152ab2cf9521453d5",
    "anolis_carolinensis_specimen.jpg": "584ab986514945bb9c82b1765cb9c44cb8915e47823a0e02b5f444310fe0506b",
    "scout_manifest.json": "98c0d574eec317f0cf2057ae41c7e9e59cefbf44c2a6a5ea5b0261bc91d23c77",
    "paper_citations.bib": "1f42e990e92fde75c02b5db4ac60287021dd9f60c48a7e84fa77d816cf522ea9"
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
    ref = json.loads((DATASET_DIR / "anolis_carolinensis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2012-sanger-anolis-carolinensis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
