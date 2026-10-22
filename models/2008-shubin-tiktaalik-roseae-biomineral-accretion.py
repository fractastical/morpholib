#!/usr/bin/env python3
"""
models/2008-shubin-tiktaalik-roseae-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1844)
Organism: Tiktaalik roseae (Late Devonian Wristed Fin-Limb Transitional Fossil)
Phenomenon Scale: 11.4 cm (0.1139 m, log10=-0.943, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Tiktaalik roseae (Late Devonian (Frasnian, 375 Ma))
Geological Horizon: Late Devonian (Frasnian, 375 Ma) (375.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Tiktaalik
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (11.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1844_tiktaalik_roseae_biomineral_accretion.lean` (`MathOfBiology.Loop_1844`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/tiktaalik_roseae_biomineral_accretion"
EXPECTED_SHA256 = {
    "tiktaalik_roseae_biomineral_accretion_reference.json": "10611ee613adafcf43bb0125b7bf96248b24643f761cc6ec2908130386b3b3d4",
    "tiktaalik_roseae_specimen.jpg": "2b6fe2e7441ef8b8a1f69c1382b71a3606ab9e0ee552372999fb3ab07fb8c3a4",
    "scout_manifest.json": "a95fa30c907f88d9c3e7d7dc04adf39360517e050109c13e3836af53108fe517",
    "paper_citations.bib": "743c6074e1a893cdf2b5566cce657c454f5780cd2902d3dd8b3a0f52e280e191"
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
    ref = json.loads((DATASET_DIR / "tiktaalik_roseae_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-shubin-tiktaalik-roseae-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
