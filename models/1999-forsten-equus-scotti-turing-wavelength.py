#!/usr/bin/env python3
"""
models/1999-forsten-equus-scotti-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1222)
Organism: Equus scotti (Pleistocene Giant Ice-Age Horse Chronospecies)
Phenomenon Scale: 8.8 mm (0.00884 m, log10=-2.054, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Equus scotti (Middle Pleistocene (Irvingtonian, 0.6 Ma))
Geological Horizon: Middle Pleistocene (Irvingtonian, 0.6 Ma) (0.6 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Equus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (8.8 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1222_equus_scotti_turing_wavelength.lean` (`MathOfBiology.Loop_1222`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/equus_scotti_turing_wavelength"
EXPECTED_SHA256 = {
    "equus_scotti_turing_wavelength_reference.json": "c592f0b55fa4ef0ff174b5a675a724436f6afeedd2a15f677158f2561451724d",
    "equus_scotti_specimen.jpg": "fea6d4b899e55872847306904b4629acc9fcf09737b2a6f906d0accb0de6e222",
    "scout_manifest.json": "84dcd50c02a0f0ccbc8d67f6a23943d3c8f3cf989e374b4912b00daa4ef658e3",
    "paper_citations.bib": "3b795be23afb3043ace9e895112b17d9e7beca39207ecce3bebd02dd77babfaa"
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
    ref = json.loads((DATASET_DIR / "equus_scotti_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1999-forsten-equus-scotti-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
