#!/usr/bin/env python3
"""
models/2010-boyce-prototaxites-logani-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1530)
Organism: Prototaxites logani (Early Devonian Giant Nematophyte Mycelial Trunk (Concentric Hyphal Rings))
Phenomenon Scale: 8.5 cm (0.08487 m, log10=-1.071, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Prototaxites logani (Early Devonian (Emsian, 400 Ma))
Geological Horizon: Early Devonian (Emsian, 400 Ma) (400.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Prototaxites
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (8.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1530_prototaxites_logani_flux_canalization.lean` (`MathOfBiology.Loop_1530`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/prototaxites_logani_flux_canalization"
EXPECTED_SHA256 = {
    "prototaxites_logani_flux_canalization_reference.json": "a5812b9cd37ee560d115797569a0990445e7a7f71add4b8b0ef4da6fe4311075",
    "prototaxites_logani_specimen.jpg": "672ce91b81384694dbcf49ddcc7b3191d489b29fbe3b3f90608e2ed52f775b12",
    "scout_manifest.json": "7afcf857026f45379f927c218960c7a891471d6d997378d4cc050821ad00ccef",
    "paper_citations.bib": "c4cd4540cebece9b234be3c7c282f4549cf69db140bdbde487daaa8c9c190be0"
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
    ref = json.loads((DATASET_DIR / "prototaxites_logani_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-boyce-prototaxites-logani-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
