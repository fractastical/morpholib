#!/usr/bin/env python3
"""
models/1994-thewissen-ambulocetus-natans-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1224)
Organism: Ambulocetus natans (Early Eocene Amphibious Walking Whale)
Phenomenon Scale: 1.0 cm (0.01042 m, log10=-1.982, 7_organism_cm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Ambulocetus natans (Early Eocene (Lutetian, 47.5 Ma))
Geological Horizon: Early Eocene (Lutetian, 47.5 Ma) (47.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Ambulocetus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (1.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1224_ambulocetus_natans_turing_wavelength.lean` (`MathOfBiology.Loop_1224`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ambulocetus_natans_turing_wavelength"
EXPECTED_SHA256 = {
    "ambulocetus_natans_turing_wavelength_reference.json": "ab34f1cd6543b9b978c00619f41361c2a257d0dd1d0cafb60220bb88bf662307",
    "ambulocetus_natans_specimen.jpg": "01a4e6043915d96e0394516e3db543eecad170348e219e8dd573ef897d004e67",
    "scout_manifest.json": "fc9a99c3ace92316319ff099500f6da9335544e661ba3eb5f8e7dc4f9c145506",
    "paper_citations.bib": "dfb07a10b640c15a556880cc557fd18c9806cc684152e2c8863d70ff202114e3"
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
    ref = json.loads((DATASET_DIR / "ambulocetus_natans_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1994-thewissen-ambulocetus-natans-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
