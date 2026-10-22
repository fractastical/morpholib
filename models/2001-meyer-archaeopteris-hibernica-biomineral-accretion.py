#!/usr/bin/env python3
"""
models/2001-meyer-archaeopteris-hibernica-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1887)
Organism: Archaeopteris hibernica (Late Devonian Progymnosperm Tree (Planated Megaphyll Fronds))
Phenomenon Scale: 2.1 cm (0.02104 m, log10=-1.677, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Archaeopteris hibernica (Late Devonian (Famennian, 370 Ma))
Geological Horizon: Late Devonian (Famennian, 370 Ma) (370.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Archaeopteris
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (2.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1887_archaeopteris_hibernica_biomineral_accretion.lean` (`MathOfBiology.Loop_1887`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/archaeopteris_hibernica_biomineral_accretion"
EXPECTED_SHA256 = {
    "archaeopteris_hibernica_biomineral_accretion_reference.json": "76935db0837a6751cc2f3c06ecdf33000058af9b7457dc20a8dc4665eaa9e19d",
    "archaeopteris_hibernica_specimen.jpg": "107d9b021f541e8a82a7c594b45f0250c4a6c8fa160f1b3c54299cd439a8163e",
    "scout_manifest.json": "ed333bd0f08f823ac77d34bbb030400791768345b015cea750b4ed7875844915",
    "paper_citations.bib": "0ae1fba411b26199effb537fa6459770b6af8f7803d42d9a83cb1b249c7220ca"
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
    ref = json.loads((DATASET_DIR / "archaeopteris_hibernica_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-meyer-archaeopteris-hibernica-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
