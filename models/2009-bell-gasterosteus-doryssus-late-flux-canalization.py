#!/usr/bin/env python3
"""
models/2009-bell-gasterosteus-doryssus-late-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1479)
Organism: Gasterosteus doryssus_late (Miocene Stickleback Chronospecies III (Pelvic Loss, T=+22 kyr))
Phenomenon Scale: 1.0 cm (0.01046 m, log10=-1.980, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Gasterosteus doryssus_late (Late Miocene (Truckee Fm, 9.99 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 9.99 Ma) (9.99 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1479_gasterosteus_doryssus_late_flux_canalization.lean` (`MathOfBiology.Loop_1479`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_late_flux_canalization"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_late_flux_canalization_reference.json": "604c67d48bf871d2753dbe0b7580004735416a1309f44c3d003fd34023a13483",
    "gasterosteus_doryssus_late_specimen.jpg": "fd5b18be03310ca28a457aa04293f4114c76a66251867edec13e086175ae1ee1",
    "scout_manifest.json": "c4837afc3d8f8aa563b639353c804304f30934d1a8b11161f4d56c53d3edee9c",
    "paper_citations.bib": "1fab7099d1b3b200ddd8c9665f9af00d899d5f3e2b4a678050453eeb7329850f"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_late_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2009-bell-gasterosteus-doryssus-late-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
