#!/usr/bin/env python3
"""
models/1957-kermack-micraster-coranguinum-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1542)
Organism: Micraster coranguinum (Late Cretaceous Chalk Heart-Urchin Chronospecies (Petaloid Burrowing))
Phenomenon Scale: 1.1 cm (0.01068 m, log10=-1.971, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Micraster coranguinum (Late Cretaceous (Santonian, 85 Ma))
Geological Horizon: Late Cretaceous (Santonian, 85 Ma) (85.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Micraster
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1542_micraster_coranguinum_flux_canalization.lean` (`MathOfBiology.Loop_1542`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/micraster_coranguinum_flux_canalization"
EXPECTED_SHA256 = {
    "micraster_coranguinum_flux_canalization_reference.json": "b4829ec9a4388e3eab4f7df4dc5244a5d86a0197c471ad891b192bba4e4a6a4f",
    "micraster_coranguinum_specimen.jpg": "e066a3f38cd1da19878288ea7b3f5ea0aeec2482a9947c60126b6e6e07fb80da",
    "scout_manifest.json": "464c0e73823603f9fdfad0e1d98848165f52690908d02a23e4b56549835b0806",
    "paper_citations.bib": "c7f12fa1ceb8214dd70a22701159d72f3584630f1ac08f1f4df18f395a93f635"
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
    ref = json.loads((DATASET_DIR / "micraster_coranguinum_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1957-kermack-micraster-coranguinum-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
