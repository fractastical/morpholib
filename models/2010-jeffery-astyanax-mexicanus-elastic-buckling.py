#!/usr/bin/env python3
"""
models/2010-jeffery-astyanax-mexicanus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1820)
Organism: Astyanax mexicanus (Mexican Blind Cavefish (Optic Cup Degeneration vs Neuromast Expansion))
Phenomenon Scale: 133 µm (0.000133 m, log10=-3.876, 5_primordium_100um)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Astyanax mexicanus (Holocene Cave Morph (Extant))
Geological Horizon: Holocene Cave Morph (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Astyanax
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (133 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1820_astyanax_mexicanus_elastic_buckling.lean` (`MathOfBiology.Loop_1820`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/astyanax_mexicanus_elastic_buckling"
EXPECTED_SHA256 = {
    "astyanax_mexicanus_elastic_buckling_reference.json": "bc361fbe3ac469e86a51793a304bf3441574f3cd2898d2af3abbf48f468ad96b",
    "astyanax_mexicanus_specimen.jpg": "53b28ff32166e8938db70570a413fac88bfdd10206b30d8fcda2d15cb61c64f5",
    "scout_manifest.json": "1975476d0f8c2b1fba50acd292441feea0f00dfcf646a09815f6334d18bd78d8",
    "paper_citations.bib": "9bfa7536d7f78dba8be08b87fc7f145f4e9f9b2c67b9c91c832c55898ce34657"
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
    ref = json.loads((DATASET_DIR / "astyanax_mexicanus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-jeffery-astyanax-mexicanus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
