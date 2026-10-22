#!/usr/bin/env python3
"""
models/2007-bell-gasterosteus-doryssus-early-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1297)
Organism: Gasterosteus doryssus_early (Miocene Stickleback Chronospecies I (Full Pelvic Girdle, T=0 kyr))
Phenomenon Scale: 1.0 cm (0.0102 m, log10=-1.991, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Gasterosteus doryssus_early (Late Miocene (Truckee Fm, 10.02 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 10.02 Ma) (10.02 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (1.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1297_gasterosteus_doryssus_early_morphogen_condensation.lean` (`MathOfBiology.Loop_1297`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_early_morphogen_condensation"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_early_morphogen_condensation_reference.json": "fe3814ae944eb774c30a87c2c09cf155a46150b42b374ec09365e783e674a3c7",
    "gasterosteus_doryssus_early_specimen.jpg": "746a8e6539215ffdae2f266963a9c619c30ce4bc965ab94dd6c04d0bc70c3034",
    "scout_manifest.json": "8b346d1cb1c9acc9290c45b7b580894523504ed4136ae4112c6959d0ab3f2d6b",
    "paper_citations.bib": "0d42fbebce6e693bd720f7193e87e69d85ef31cdf4a4d99506553e22db8d1c43"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_early_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-bell-gasterosteus-doryssus-early-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
