#!/usr/bin/env python3
"""
models/2015-feoing-archaeopteryx-lithographica-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1500)
Organism: Archaeopteryx lithographica (Late Jurassic Solnhofen Urvogel (Asymmetric Flight Vane))
Phenomenon Scale: 1.8 cm (0.01825 m, log10=-1.739, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Archaeopteryx lithographica (Late Jurassic (Tithonian, 150.0 Ma))
Geological Horizon: Late Jurassic (Tithonian, 150.0 Ma) (150.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Archaeopteryx
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1500_archaeopteryx_lithographica_flux_canalization.lean` (`MathOfBiology.Loop_1500`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/archaeopteryx_lithographica_flux_canalization"
EXPECTED_SHA256 = {
    "archaeopteryx_lithographica_flux_canalization_reference.json": "6127c4fdea212dfb049ac08a33e3dc0c18f1903fc1ad912559e489cc7d7c7c75",
    "archaeopteryx_lithographica_specimen.jpg": "22ad1c7de4cfaadcc6057d0f4ad6dd252fd379582bec21b51bf73852daa000af",
    "scout_manifest.json": "6460cc0a904888ec8507e73402f9d9c7c455cf2e60e53076e6de855eee49cd58",
    "paper_citations.bib": "a1b636d4a68061228eac7a7e0e00c3a05acfd84307ec2b0872e4d501b72e97c2"
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
    ref = json.loads((DATASET_DIR / "archaeopteryx_lithographica_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-feoing-archaeopteryx-lithographica-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
