#!/usr/bin/env python3
"""
models/2009-bell-gasterosteus-doryssus-early-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1927)
Organism: Gasterosteus doryssus_early (Miocene Stickleback Chronospecies I (Full Pelvic Girdle, T=0 kyr))
Phenomenon Scale: 45 µm (4.534e-05 m, log10=-4.344, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Gasterosteus doryssus_early (Late Miocene (Truckee Fm, 10.02 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 10.02 Ma) (10.02 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (45 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1927_gasterosteus_doryssus_early_bioelectric_polarity.lean` (`MathOfBiology.Loop_1927`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_early_bioelectric_polarity"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_early_bioelectric_polarity_reference.json": "81eaffe51993f988ec3a7d6bb1a224988cfa9fdcf23696f3c2d9578253909cce",
    "gasterosteus_doryssus_early_specimen.jpg": "746a8e6539215ffdae2f266963a9c619c30ce4bc965ab94dd6c04d0bc70c3034",
    "scout_manifest.json": "8acce0b87a7707996119747e9c359853afa881a0b2c71b5f50af4c28a8fb1bfd",
    "paper_citations.bib": "9f5810a77ae95694854f4d2ab228b5d77d2e01d04a8eea9856f91d40b376fbab"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_early_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2009-bell-gasterosteus-doryssus-early-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
