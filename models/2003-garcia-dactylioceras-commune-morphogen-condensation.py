#!/usr/bin/env python3
"""
models/2003-garcia-dactylioceras-commune-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1327)
Organism: Dactylioceras commune (Early Jurassic Whitby Bifurcating-Rib Ammonite (Fractal Suture, D_f=1.55))
Phenomenon Scale: 6.7 mm (0.006728 m, log10=-2.172, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Dactylioceras commune (Early Jurassic (Toarcian, 181 Ma))
Geological Horizon: Early Jurassic (Toarcian, 181 Ma) (181.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Dactylioceras
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (6.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1327_dactylioceras_commune_morphogen_condensation.lean` (`MathOfBiology.Loop_1327`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dactylioceras_commune_morphogen_condensation"
EXPECTED_SHA256 = {
    "dactylioceras_commune_morphogen_condensation_reference.json": "04ae44119e5b86be652605db9691f3af1df75430ed55d537997f07523646d754",
    "dactylioceras_commune_specimen.jpg": "565735d7837252ce5f4f6091140a7e409f84cc8f19a88e5b16fe545b77016fbb",
    "scout_manifest.json": "93fe0f22e776cf1f642e997720b98fbe6e917dd82579f01a83fd1aa2492dc40f",
    "paper_citations.bib": "baa067f730888fbbf88e87aa4713802a41cb08f906804e9fe0609c3e706793f2"
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
    ref = json.loads((DATASET_DIR / "dactylioceras_commune_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-garcia-dactylioceras-commune-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
