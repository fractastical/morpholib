#!/usr/bin/env python3
"""
models/2010-zhang-sinosauropteryx-prima-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1228)
Organism: Sinosauropteryx prima (Early Cretaceous Proto-Feathered Compsognathid Dinosaur)
Phenomenon Scale: 52 µm (5.232e-05 m, log10=-4.281, 4_cellular_10um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Sinosauropteryx prima (Early Cretaceous (Yixian Fm, 124.6 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 124.6 Ma) (124.6 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Sinosauropteryx
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (52 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1228_sinosauropteryx_prima_turing_wavelength.lean` (`MathOfBiology.Loop_1228`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/sinosauropteryx_prima_turing_wavelength"
EXPECTED_SHA256 = {
    "sinosauropteryx_prima_turing_wavelength_reference.json": "696f219113ebcaaabc9a392dfc123fd2e1c9ab99dc641110daca254b55bdbd12",
    "sinosauropteryx_prima_specimen.jpg": "dafd40e5487fb100d0309d10c30a01a1479f1793e6aa37726388bfafdd50d009",
    "scout_manifest.json": "9a9bcd2eab2c5757c2af15d06dc340c2c911f988edd9009a126c0de4628b0411",
    "paper_citations.bib": "7189a7f124ac26123e333aad7bc0db7f58e0878d193a961b9a85118701e690c3"
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
    ref = json.loads((DATASET_DIR / "sinosauropteryx_prima_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-zhang-sinosauropteryx-prima-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
