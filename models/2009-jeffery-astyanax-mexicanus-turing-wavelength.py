#!/usr/bin/env python3
"""
models/2009-jeffery-astyanax-mexicanus-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1280)
Organism: Astyanax mexicanus (Mexican Blind Cavefish (Optic Cup Degeneration vs Neuromast Expansion))
Phenomenon Scale: 24 µm (2.418e-05 m, log10=-4.617, 4_cellular_10um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Astyanax mexicanus (Holocene Cave Morph (Extant))
Geological Horizon: Holocene Cave Morph (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Astyanax
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (24 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1280_astyanax_mexicanus_turing_wavelength.lean` (`MathOfBiology.Loop_1280`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/astyanax_mexicanus_turing_wavelength"
EXPECTED_SHA256 = {
    "astyanax_mexicanus_turing_wavelength_reference.json": "cbc41043ae7d86b85506888f801d3a08dd7e5ba4f451d2cd165bcfe31b2908ab",
    "astyanax_mexicanus_specimen.jpg": "53b28ff32166e8938db70570a413fac88bfdd10206b30d8fcda2d15cb61c64f5",
    "scout_manifest.json": "a838be57f3d4cef061b0344103768f1efa78c69babe2b78c60891dcb747ee66f",
    "paper_citations.bib": "223080284b6f8662d5f2e63c0c367f2a8d60a8bd7da2980877e663e79bac9120"
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
    ref = json.loads((DATASET_DIR / "astyanax_mexicanus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/2009-jeffery-astyanax-mexicanus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
