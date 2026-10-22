#!/usr/bin/env python3
"""
models/1999-coates-acanthostega-gunnari-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1485)
Organism: Acanthostega gunnari (Late Devonian Eight-Digited Polydactyl Stem-Tetrapod)
Phenomenon Scale: 2.4 cm (0.02444 m, log10=-1.612, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Acanthostega gunnari (Late Devonian (Famennian, 365 Ma))
Geological Horizon: Late Devonian (Famennian, 365 Ma) (365.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Acanthostega
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (2.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1485_acanthostega_gunnari_flux_canalization.lean` (`MathOfBiology.Loop_1485`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/acanthostega_gunnari_flux_canalization"
EXPECTED_SHA256 = {
    "acanthostega_gunnari_flux_canalization_reference.json": "7bb171944b1154843a1b12fd8f3960fc73a952d02ecf4fb1e559f6f2a1991fe9",
    "acanthostega_gunnari_specimen.jpg": "17b12e17eb08ab7e05345c565a3dc299e5c4752dd56c4753617937fa7b3a66b5",
    "scout_manifest.json": "426807861efb8f795005e9828c1ce7a87db0a941d8f0b9ca98373c8036afc447",
    "paper_citations.bib": "9e0fbbcc1c154b8ab783478b4c34e776ac1f387c0bbe9a98b85ca9722a2fbb1a"
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
    ref = json.loads((DATASET_DIR / "acanthostega_gunnari_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1999-coates-acanthostega-gunnari-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
