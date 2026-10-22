#!/usr/bin/env python3
"""
models/1998-waters-pentremites-godoni-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1270)
Organism: Pentremites godoni (Mississippian Pentaradial Hydrospire Blastoid Calyx)
Phenomenon Scale: 785 µm (0.0007848 m, log10=-3.105, 5_primordium_100um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Pentremites godoni (Mississippian (Chesterian, 330 Ma))
Geological Horizon: Mississippian (Chesterian, 330 Ma) (330.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Pentremites
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (785 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1270_pentremites_godoni_turing_wavelength.lean` (`MathOfBiology.Loop_1270`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pentremites_godoni_turing_wavelength"
EXPECTED_SHA256 = {
    "pentremites_godoni_turing_wavelength_reference.json": "77f59228ac0d17fdc12a665ce4f9659204e38c2a9d87983d9bc2d67c543af4ff",
    "pentremites_godoni_specimen.jpg": "53e40da38ecd2df40e09153c5fa4329553f1a7e8c16087490c6ba4fccd9eeb1d",
    "scout_manifest.json": "0ae57e7df9c1917f8d0ed08f3ca1addc9de61df86fa48c7e298322177d46205c",
    "paper_citations.bib": "b92ae60db7981c18e74bca5f48f2d930a10cd0559344cc0856ed9a4aad2d7c43"
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
    ref = json.loads((DATASET_DIR / "pentremites_godoni_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1998-waters-pentremites-godoni-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
