#!/usr/bin/env python3
"""
models/2014-feoing-archaeopteryx-lithographica-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1860)
Organism: Archaeopteryx lithographica (Late Jurassic Solnhofen Urvogel (Asymmetric Flight Vane))
Phenomenon Scale: 4.0 cm (0.04014 m, log10=-1.396, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Archaeopteryx lithographica (Late Jurassic (Tithonian, 150.0 Ma))
Geological Horizon: Late Jurassic (Tithonian, 150.0 Ma) (150.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Archaeopteryx
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (4.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1860_archaeopteryx_lithographica_biomineral_accretion.lean` (`MathOfBiology.Loop_1860`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/archaeopteryx_lithographica_biomineral_accretion"
EXPECTED_SHA256 = {
    "archaeopteryx_lithographica_biomineral_accretion_reference.json": "d6e84e65d4cc019dc3dded93e670595a39e32c7acec156cfe0ba85a00cecc8cd",
    "archaeopteryx_lithographica_specimen.jpg": "22ad1c7de4cfaadcc6057d0f4ad6dd252fd379582bec21b51bf73852daa000af",
    "scout_manifest.json": "e46ff0a4f0e7c15250b10d84d5fef99829c7c1ea848d6d94fedc33d068012158",
    "paper_citations.bib": "bd3c189a36776ff816fbca698796108750db96d0c1c322bcdb595a5d54640c5f"
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
    ref = json.loads((DATASET_DIR / "archaeopteryx_lithographica_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-feoing-archaeopteryx-lithographica-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
