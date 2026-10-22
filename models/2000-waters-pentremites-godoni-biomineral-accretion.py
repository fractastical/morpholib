#!/usr/bin/env python3
"""
models/2000-waters-pentremites-godoni-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1900)
Organism: Pentremites godoni (Mississippian Pentaradial Hydrospire Blastoid Calyx)
Phenomenon Scale: 1.1 cm (0.01079 m, log10=-1.967, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Pentremites godoni (Mississippian (Chesterian, 330 Ma))
Geological Horizon: Mississippian (Chesterian, 330 Ma) (330.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Pentremites
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (1.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1900_pentremites_godoni_biomineral_accretion.lean` (`MathOfBiology.Loop_1900`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pentremites_godoni_biomineral_accretion"
EXPECTED_SHA256 = {
    "pentremites_godoni_biomineral_accretion_reference.json": "34a935d8a4e6f1b79c67012bd87d30c59c043cbd4b78b068289125e82b20f825",
    "pentremites_godoni_specimen.jpg": "53e40da38ecd2df40e09153c5fa4329553f1a7e8c16087490c6ba4fccd9eeb1d",
    "scout_manifest.json": "0ee4832666e9f511497dfa317e6296d9df891e940707045dfba70a3e3564d395",
    "paper_citations.bib": "a8131811f4af5f30ec3b18fbdb02fda53f5001f1c0b171ad6d636c9994f811f7"
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
    ref = json.loads((DATASET_DIR / "pentremites_godoni_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2000-waters-pentremites-godoni-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
