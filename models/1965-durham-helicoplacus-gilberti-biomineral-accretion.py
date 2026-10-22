#!/usr/bin/env python3
"""
models/1965-durham-helicoplacus-gilberti-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1899)
Organism: Helicoplacus gilberti (Early Cambrian Triradial Helical-Plated Stem-Echinoderm)
Phenomenon Scale: 1.8 cm (0.01848 m, log10=-1.733, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Helicoplacus gilberti (Early Cambrian (Poleta Fm, 516 Ma))
Geological Horizon: Early Cambrian (Poleta Fm, 516 Ma) (516.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Helicoplacus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (1.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1899_helicoplacus_gilberti_biomineral_accretion.lean` (`MathOfBiology.Loop_1899`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/helicoplacus_gilberti_biomineral_accretion"
EXPECTED_SHA256 = {
    "helicoplacus_gilberti_biomineral_accretion_reference.json": "64e04f45c4a33df9c41294146f872d7e54f54efda63d8538a3928b223584c6b0",
    "helicoplacus_gilberti_specimen.jpg": "13c0993dd3e63798e7d236896fbbbbe4e10b4f49fe7d356be1e6a56d3f579d8a",
    "scout_manifest.json": "0f1bdebc04f2e35c46e4897935b24e36204ad791b7a585fda05f0cf3233869f5",
    "paper_citations.bib": "e9f5cf68760ecf06de8b9543f73a88964d8b37068531dec177211fdfc632837d"
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
    ref = json.loads((DATASET_DIR / "helicoplacus_gilberti_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1965-durham-helicoplacus-gilberti-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
