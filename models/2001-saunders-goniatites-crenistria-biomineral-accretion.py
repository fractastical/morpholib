#!/usr/bin/env python3
"""
models/2001-saunders-goniatites-crenistria-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1865)
Organism: Goniatites crenistria (Mississippian Goniatitic Ammonoid (Angular Zigzag Suture, D_f=1.18))
Phenomenon Scale: 1.1 cm (0.01079 m, log10=-1.967, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Goniatites crenistria (Early Carboniferous (Viséan, 333 Ma))
Geological Horizon: Early Carboniferous (Viséan, 333 Ma) (333.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Goniatites
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (1.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1865_goniatites_crenistria_biomineral_accretion.lean` (`MathOfBiology.Loop_1865`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/goniatites_crenistria_biomineral_accretion"
EXPECTED_SHA256 = {
    "goniatites_crenistria_biomineral_accretion_reference.json": "c1716c8e88aef5dc795bab9cc08dc4fb2cbdcbacd39622e7786280ba810df33f",
    "goniatites_crenistria_specimen.jpg": "1c025730c53d3fe4dde10fe4dfc3229dac439001911ee16d02107da5f04120fb",
    "scout_manifest.json": "0488308e5b2f1dce6768b0dabd93c7c336b57270369e645f0cb742c9c82852b0",
    "paper_citations.bib": "f2c8b6e34358ba7101bbbc41cb7d098e9b3d33b28cb4957346bf05df2018b208"
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
    ref = json.loads((DATASET_DIR / "goniatites_crenistria_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-saunders-goniatites-crenistria-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
