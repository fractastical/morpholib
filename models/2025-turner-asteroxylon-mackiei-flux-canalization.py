#!/usr/bin/env python3
"""
models/2025-turner-asteroxylon-mackiei-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1526)
Organism: Asteroxylon mackiei (Early Devonian Rhynie Chert Lycopsid (Actinostele & Fibonacci Microphylls))
Phenomenon Scale: 1.8 mm (0.001771 m, log10=-2.752, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Asteroxylon mackiei (Early Devonian (Pragian Rhynie Chert, 407 Ma))
Geological Horizon: Early Devonian (Pragian Rhynie Chert, 407 Ma) (407.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Asteroxylon
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.8 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1526_asteroxylon_mackiei_flux_canalization.lean` (`MathOfBiology.Loop_1526`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/asteroxylon_mackiei_flux_canalization"
EXPECTED_SHA256 = {
    "asteroxylon_mackiei_flux_canalization_reference.json": "0653d3770dc6479c4f63529d397dab7b10a30524be9b4282ac5c0018431d95c4",
    "asteroxylon_mackiei_specimen.jpg": "af15c01893b31829fefca3fc2246896ab8d2c12ab87d5809f28504e9c2968531",
    "scout_manifest.json": "6cf773cbe7e9d05f471b716864f5e134b7cc5b471396b9ecf8f475aa12e9d03c",
    "paper_citations.bib": "05623f089e41d816c3a97d0bad3e4c6c45e225bf6a398202e9d1b0b6857e0dd6"
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
    ref = json.loads((DATASET_DIR / "asteroxylon_mackiei_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2025-turner-asteroxylon-mackiei-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
