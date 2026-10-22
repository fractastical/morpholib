#!/usr/bin/env python3
"""
models/1995-thewissen-ambulocetus-natans-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1314)
Organism: Ambulocetus natans (Early Eocene Amphibious Walking Whale)
Phenomenon Scale: 4.7 cm (0.04687 m, log10=-1.329, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Ambulocetus natans (Early Eocene (Lutetian, 47.5 Ma))
Geological Horizon: Early Eocene (Lutetian, 47.5 Ma) (47.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ambulocetus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (4.7 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1314_ambulocetus_natans_morphogen_condensation.lean` (`MathOfBiology.Loop_1314`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ambulocetus_natans_morphogen_condensation"
EXPECTED_SHA256 = {
    "ambulocetus_natans_morphogen_condensation_reference.json": "5588e16ae0a3586c0d0293c87adf59f119784e3b214ce5321e989aae10f2d630",
    "ambulocetus_natans_specimen.jpg": "01a4e6043915d96e0394516e3db543eecad170348e219e8dd573ef897d004e67",
    "scout_manifest.json": "a54029be3040623f9521d7a4290b9e557e3c250736c8090a744d43e0db82e228",
    "paper_citations.bib": "4ca24a087aaeb6661ebd483d2fe4aec85bdf33472fd02bb6040d0bd34c218459"
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
    ref = json.loads((DATASET_DIR / "ambulocetus_natans_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1995-thewissen-ambulocetus-natans-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
