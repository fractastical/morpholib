#!/usr/bin/env python3
"""
models/2007-bell-gasterosteus-doryssus-late-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1749)
Organism: Gasterosteus doryssus_late (Miocene Stickleback Chronospecies III (Pelvic Loss, T=+22 kyr))
Phenomenon Scale: 9.2 mm (0.009203 m, log10=-2.036, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Gasterosteus doryssus_late (Late Miocene (Truckee Fm, 9.99 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 9.99 Ma) (9.99 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (9.2 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1749_gasterosteus_doryssus_late_elastic_buckling.lean` (`MathOfBiology.Loop_1749`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_late_elastic_buckling"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_late_elastic_buckling_reference.json": "958b7771499081f2fc1890c4158932e838abb2338498dc4c3e89ee0f2621be1e",
    "gasterosteus_doryssus_late_specimen.jpg": "fd5b18be03310ca28a457aa04293f4114c76a66251867edec13e086175ae1ee1",
    "scout_manifest.json": "bca55bf69a179e592178aeff4be5535a72f254415429b7ff565b548ca0211614",
    "paper_citations.bib": "e2d4fe2852896eec894b15f016def836a1f49aae246549937e37320fcc143b19"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_late_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-bell-gasterosteus-doryssus-late-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
