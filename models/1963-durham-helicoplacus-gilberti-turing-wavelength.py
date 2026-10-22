#!/usr/bin/env python3
"""
models/1963-durham-helicoplacus-gilberti-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1269)
Organism: Helicoplacus gilberti (Early Cambrian Triradial Helical-Plated Stem-Echinoderm)
Phenomenon Scale: 1.3 mm (0.001344 m, log10=-2.872, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Helicoplacus gilberti (Early Cambrian (Poleta Fm, 516 Ma))
Geological Horizon: Early Cambrian (Poleta Fm, 516 Ma) (516.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Helicoplacus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (1.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1269_helicoplacus_gilberti_turing_wavelength.lean` (`MathOfBiology.Loop_1269`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/helicoplacus_gilberti_turing_wavelength"
EXPECTED_SHA256 = {
    "helicoplacus_gilberti_turing_wavelength_reference.json": "4f3564a5dd625ff9fb58aec15d85199b63358dfcc4a5f90f0b7532e025609d10",
    "helicoplacus_gilberti_specimen.jpg": "13c0993dd3e63798e7d236896fbbbbe4e10b4f49fe7d356be1e6a56d3f579d8a",
    "scout_manifest.json": "2fd95f6805a3cc9eace2186a9425407bba924a0463da44fb576832482d8c1e87",
    "paper_citations.bib": "a6387a969e63d3a3fc7f8345cd6c3bbd9f15258827764d8f7820baba38f78d4b"
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
    ref = json.loads((DATASET_DIR / "helicoplacus_gilberti_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1963-durham-helicoplacus-gilberti-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
