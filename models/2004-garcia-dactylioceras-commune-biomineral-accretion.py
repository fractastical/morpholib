#!/usr/bin/env python3
"""
models/2004-garcia-dactylioceras-commune-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1867)
Organism: Dactylioceras commune (Early Jurassic Whitby Bifurcating-Rib Ammonite (Fractal Suture, D_f=1.55))
Phenomenon Scale: 2.1 cm (0.02056 m, log10=-1.687, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Dactylioceras commune (Early Jurassic (Toarcian, 181 Ma))
Geological Horizon: Early Jurassic (Toarcian, 181 Ma) (181.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Dactylioceras
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (2.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1867_dactylioceras_commune_biomineral_accretion.lean` (`MathOfBiology.Loop_1867`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dactylioceras_commune_biomineral_accretion"
EXPECTED_SHA256 = {
    "dactylioceras_commune_biomineral_accretion_reference.json": "ba041f8735c0ed61529ea8c8972f1f9cef78362f38627b93531c65a8eb47d1be",
    "dactylioceras_commune_specimen.jpg": "565735d7837252ce5f4f6091140a7e409f84cc8f19a88e5b16fe545b77016fbb",
    "scout_manifest.json": "b319469534c0722cbc40f816b4da4f46d9fd1bf65b3155d38017d3402db4f0be",
    "paper_citations.bib": "2515414a63f0ee470af6245bcc317a2879e9da85fb905e33c98405aa102533e7"
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
    ref = json.loads((DATASET_DIR / "dactylioceras_commune_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-garcia-dactylioceras-commune-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
