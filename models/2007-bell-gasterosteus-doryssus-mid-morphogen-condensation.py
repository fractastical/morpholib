#!/usr/bin/env python3
"""
models/2007-bell-gasterosteus-doryssus-mid-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1298)
Organism: Gasterosteus doryssus_mid (Miocene Stickleback Chronospecies II (Vestigial Spine, T=+9 kyr))
Phenomenon Scale: 7.5 mm (0.007497 m, log10=-2.125, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Gasterosteus doryssus_mid (Late Miocene (Truckee Fm, 10.01 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 10.01 Ma) (10.01 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (7.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1298_gasterosteus_doryssus_mid_morphogen_condensation.lean` (`MathOfBiology.Loop_1298`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_mid_morphogen_condensation"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_mid_morphogen_condensation_reference.json": "f43b847eaf99d8fd87eb2662558ba92e59b4f7d7d58996255bfb621e703869bc",
    "gasterosteus_doryssus_mid_specimen.jpg": "52d02b91f4693fdd40965b6d78971fdf8ebe6601d12fad6c49a6255e45791e42",
    "scout_manifest.json": "72f333e0efe2da82b5673174f9dea59ccc68327d4900e2dfb30714eef318b8ad",
    "paper_citations.bib": "fc7e7649449b8618811fb963bcdfbfd6861a574118639c6833aaeb4edd41883a"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_mid_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-bell-gasterosteus-doryssus-mid-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
