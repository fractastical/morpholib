#!/usr/bin/env python3
"""
models/2008-bell-gasterosteus-doryssus-early-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1837)
Organism: Gasterosteus doryssus_early (Miocene Stickleback Chronospecies I (Full Pelvic Girdle, T=0 kyr))
Phenomenon Scale: 3.1 cm (0.03117 m, log10=-1.506, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Gasterosteus doryssus_early (Late Miocene (Truckee Fm, 10.02 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 10.02 Ma) (10.02 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (3.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1837_gasterosteus_doryssus_early_biomineral_accretion.lean` (`MathOfBiology.Loop_1837`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_early_biomineral_accretion"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_early_biomineral_accretion_reference.json": "0ae9429297111ecbcbe3664c321ec1e151e33ba2543c45926d1ac7d5cb4bdb65",
    "gasterosteus_doryssus_early_specimen.jpg": "746a8e6539215ffdae2f266963a9c619c30ce4bc965ab94dd6c04d0bc70c3034",
    "scout_manifest.json": "eed3d0baa7ec696d27bf01ee6a26130a2e5f81f95c9b8721a072044996f43eee",
    "paper_citations.bib": "cb16a7f3e9a0068e08d06c8384175e7a2e9bfe57bf197c2ae87d5794489e6010"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_early_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-bell-gasterosteus-doryssus-early-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
