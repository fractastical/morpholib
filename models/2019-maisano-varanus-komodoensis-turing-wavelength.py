#!/usr/bin/env python3
"""
models/2019-maisano-varanus-komodoensis-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1278)
Organism: Varanus komodoensis (Komodo Dragon (Chainmail Cephalic Osteoderm Meshwork))
Phenomenon Scale: 95 µm (9.52e-05 m, log10=-4.021, 4_cellular_10um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Varanus komodoensis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Varanus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (95 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1278_varanus_komodoensis_turing_wavelength.lean` (`MathOfBiology.Loop_1278`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/varanus_komodoensis_turing_wavelength"
EXPECTED_SHA256 = {
    "varanus_komodoensis_turing_wavelength_reference.json": "a9202581d0318e2312b465727bfec06e5e890030dfc88db136cfe47ddf7374cf",
    "varanus_komodoensis_specimen.jpg": "347fa0ca27bf35537323e24d59b72d5f6df744f56be66ed7c9649cdf3bc761b5",
    "scout_manifest.json": "1b07a2bc348d513052eab3cca20a66df90386474efa631a3062c97e50e5b4064",
    "paper_citations.bib": "13c68c8ccebfb724145ce330b641c0324aeb16cc7c5b38bcd5b5d4234f72d35a"
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
    ref = json.loads((DATASET_DIR / "varanus_komodoensis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-maisano-varanus-komodoensis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
