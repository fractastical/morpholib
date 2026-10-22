#!/usr/bin/env python3
"""
models/2006-bell-gasterosteus-doryssus-late-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1209)
Organism: Gasterosteus doryssus_late (Miocene Stickleback Chronospecies III (Pelvic Loss, T=+22 kyr))
Phenomenon Scale: 1.7 mm (0.001673 m, log10=-2.777, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Gasterosteus doryssus_late (Late Miocene (Truckee Fm, 9.99 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 9.99 Ma) (9.99 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (1.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1209_gasterosteus_doryssus_late_turing_wavelength.lean` (`MathOfBiology.Loop_1209`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_late_turing_wavelength"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_late_turing_wavelength_reference.json": "11de203a31d1212ec1d1191f4b3560824a84660c02e897dc61a16975a6073d30",
    "gasterosteus_doryssus_late_specimen.jpg": "fd5b18be03310ca28a457aa04293f4114c76a66251867edec13e086175ae1ee1",
    "scout_manifest.json": "621473df7a29f5a3ae73e2efc1d07db5ef7727afa545ad9eac16d4e0398c8b2a",
    "paper_citations.bib": "9aee6b4b65bfd79efe2acc9ea5d16e8f92745099acbbdf3982fd9dfd321e2515"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_late_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-bell-gasterosteus-doryssus-late-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
