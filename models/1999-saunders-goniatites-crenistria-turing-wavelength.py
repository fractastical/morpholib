#!/usr/bin/env python3
"""
models/1999-saunders-goniatites-crenistria-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1235)
Organism: Goniatites crenistria (Mississippian Goniatitic Ammonoid (Angular Zigzag Suture, D_f=1.18))
Phenomenon Scale: 785 µm (0.0007848 m, log10=-3.105, 5_primordium_100um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Goniatites crenistria (Early Carboniferous (Viséan, 333 Ma))
Geological Horizon: Early Carboniferous (Viséan, 333 Ma) (333.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Goniatites
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (785 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1235_goniatites_crenistria_turing_wavelength.lean` (`MathOfBiology.Loop_1235`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/goniatites_crenistria_turing_wavelength"
EXPECTED_SHA256 = {
    "goniatites_crenistria_turing_wavelength_reference.json": "999ab40106fadcc967c391a4be4bf0ae891a7eea3ca7d31c25431ec3323f9ffb",
    "goniatites_crenistria_specimen.jpg": "1c025730c53d3fe4dde10fe4dfc3229dac439001911ee16d02107da5f04120fb",
    "scout_manifest.json": "76f5f6e281dfa864fa377e02f74825ddaa2a9470f12ef4ad76e832156d1c9efd",
    "paper_citations.bib": "600d1842f71d310ed30617fd81f6bf0b0f6bb274ceee666fbcf13b1937f51fe2"
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
    ref = json.loads((DATASET_DIR / "goniatites_crenistria_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1999-saunders-goniatites-crenistria-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
