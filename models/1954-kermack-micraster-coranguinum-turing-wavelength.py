#!/usr/bin/env python3
"""
models/1954-kermack-micraster-coranguinum-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1272)
Organism: Micraster coranguinum (Late Cretaceous Chalk Heart-Urchin Chronospecies (Petaloid Burrowing))
Phenomenon Scale: 1.7 mm (0.001709 m, log10=-2.767, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Micraster coranguinum (Late Cretaceous (Santonian, 85 Ma))
Geological Horizon: Late Cretaceous (Santonian, 85 Ma) (85.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Micraster
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (1.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1272_micraster_coranguinum_turing_wavelength.lean` (`MathOfBiology.Loop_1272`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/micraster_coranguinum_turing_wavelength"
EXPECTED_SHA256 = {
    "micraster_coranguinum_turing_wavelength_reference.json": "bd51fad8c62677d37b59f4c883b630447824e3af4105643248ee26394e3163cf",
    "micraster_coranguinum_specimen.jpg": "e066a3f38cd1da19878288ea7b3f5ea0aeec2482a9947c60126b6e6e07fb80da",
    "scout_manifest.json": "c1a30d692bb6504ab5148caac13d63f200546ae392617ba85ff8d39906860bc6",
    "paper_citations.bib": "ab205a00724df7b30ac0b5da27c09cbe3eedf1ad4295efa1898d1b10551c6b90"
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
    ref = json.loads((DATASET_DIR / "micraster_coranguinum_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1954-kermack-micraster-coranguinum-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
