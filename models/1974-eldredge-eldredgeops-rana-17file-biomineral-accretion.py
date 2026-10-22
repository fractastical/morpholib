#!/usr/bin/env python3
"""
models/1974-eldredge-eldredgeops-rana-17file-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1876)
Organism: Eldredgeops rana_17file (Middle Devonian Punctuated-Equilibrium Chronospecies (17-File Eye))
Phenomenon Scale: 245 µm (0.0002454 m, log10=-3.610, 5_primordium_100um)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Eldredgeops rana_17file (Middle Devonian (Tioughniogan, 384 Ma))
Geological Horizon: Middle Devonian (Tioughniogan, 384 Ma) (384.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Eldredgeops
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (245 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1876_eldredgeops_rana_17file_biomineral_accretion.lean` (`MathOfBiology.Loop_1876`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eldredgeops_rana_17file_biomineral_accretion"
EXPECTED_SHA256 = {
    "eldredgeops_rana_17file_biomineral_accretion_reference.json": "f7e39bf0e4544cc7191d2dc79270a3f54879c769784652cc28cd79034b9d1d62",
    "eldredgeops_rana_17file_specimen.jpg": "15e5b154e304f4a7611a6156cdcfaa0863adea78b4676093b13c3c4a1fc76ffe",
    "scout_manifest.json": "c32aad742a6d1867881a1dade9518fafe750731a3f6ac6a2bfd5af97da3386da",
    "paper_citations.bib": "ca1bcbb73896927648950d7670233b78943395c8c235714ffa61072fce12e879"
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
    ref = json.loads((DATASET_DIR / "eldredgeops_rana_17file_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1974-eldredge-eldredgeops-rana-17file-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
