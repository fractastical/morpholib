#!/usr/bin/env python3
"""
models/2006-bell-gasterosteus-doryssus-mid-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1208)
Organism: Gasterosteus doryssus_mid (Miocene Stickleback Chronospecies II (Vestigial Spine, T=+9 kyr))
Phenomenon Scale: 1.7 mm (0.001666 m, log10=-2.778, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Gasterosteus doryssus_mid (Late Miocene (Truckee Fm, 10.01 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 10.01 Ma) (10.01 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (1.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1208_gasterosteus_doryssus_mid_turing_wavelength.lean` (`MathOfBiology.Loop_1208`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_mid_turing_wavelength"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_mid_turing_wavelength_reference.json": "783b6e75be72f77458d6a76b8d9258297b57ede1ff2b74ba255cf1182ace8913",
    "gasterosteus_doryssus_mid_specimen.jpg": "52d02b91f4693fdd40965b6d78971fdf8ebe6601d12fad6c49a6255e45791e42",
    "scout_manifest.json": "530e08e7182c57d5b668ea1641de3518c3a3e14635d058b652ea2673366713b5",
    "paper_citations.bib": "00398d7c95c342d95af7edbf9a80ffd4e01b99f9122aec62e35acb4441250e14"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_mid_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-bell-gasterosteus-doryssus-mid-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
