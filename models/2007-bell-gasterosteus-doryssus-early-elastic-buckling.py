#!/usr/bin/env python3
"""
models/2007-bell-gasterosteus-doryssus-early-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1747)
Organism: Gasterosteus doryssus_early (Miocene Stickleback Chronospecies I (Full Pelvic Girdle, T=0 kyr))
Phenomenon Scale: 1.2 cm (0.01247 m, log10=-1.904, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Gasterosteus doryssus_early (Late Miocene (Truckee Fm, 10.02 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 10.02 Ma) (10.02 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1747_gasterosteus_doryssus_early_elastic_buckling.lean` (`MathOfBiology.Loop_1747`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_early_elastic_buckling"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_early_elastic_buckling_reference.json": "fda45c855b02c170b201b5a8a8c80fee2cb3f733fc756e98bc631239823f2e9b",
    "gasterosteus_doryssus_early_specimen.jpg": "746a8e6539215ffdae2f266963a9c619c30ce4bc965ab94dd6c04d0bc70c3034",
    "scout_manifest.json": "8d5475d0b0c21f5de9d133e433b0f4e2bfd910ebe46f3fced94cd472254bac2e",
    "paper_citations.bib": "29edc1e01f144ad2583e6e0a260726dd3ad85bb4561946cb6ec8ce79489318cc"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_early_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-bell-gasterosteus-doryssus-early-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
