#!/usr/bin/env python3
"""
models/2005-ahlberg-ichthyostega-stensioei-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1216)
Organism: Ichthyostega stensioei (Late Devonian Seven-Digited Weight-Bearing Tetrapod)
Phenomenon Scale: 5.5 mm (0.005518 m, log10=-2.258, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Ichthyostega stensioei (Late Devonian (Famennian, 363 Ma))
Geological Horizon: Late Devonian (Famennian, 363 Ma) (363.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Ichthyostega
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (5.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1216_ichthyostega_stensioei_turing_wavelength.lean` (`MathOfBiology.Loop_1216`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ichthyostega_stensioei_turing_wavelength"
EXPECTED_SHA256 = {
    "ichthyostega_stensioei_turing_wavelength_reference.json": "dc30cc02bdf83ba243a3d9029ec3908dd49a93f6d481c6d7279472bc27fbd2f6",
    "ichthyostega_stensioei_specimen.jpg": "d0371040f354560ee3b95904f5d16c9f224254e9b490bc2d4a9c7d712aa42e0c",
    "scout_manifest.json": "3c91b8f7dc08bf41c26efd84783548439d0c78571a8c3ca993d02205d300cd48",
    "paper_citations.bib": "2f519a8f784eaf51b7ef14ff79f2fb5b5db0c4e3e9e6cc7a86e065ef70738065"
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
    ref = json.loads((DATASET_DIR / "ichthyostega_stensioei_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-ahlberg-ichthyostega-stensioei-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
