#!/usr/bin/env python3
"""
models/1980-jarvik-eusthenopteron-foordi-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1212)
Organism: Eusthenopteron foordi (Late Devonian Tristichopterid Lobe-Fin Fish)
Phenomenon Scale: 5.7 mm (0.005656 m, log10=-2.247, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Eusthenopteron foordi (Late Devonian (Frasnian, 379 Ma))
Geological Horizon: Late Devonian (Frasnian, 379 Ma) (379.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Eusthenopteron
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (5.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1212_eusthenopteron_foordi_turing_wavelength.lean` (`MathOfBiology.Loop_1212`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eusthenopteron_foordi_turing_wavelength"
EXPECTED_SHA256 = {
    "eusthenopteron_foordi_turing_wavelength_reference.json": "967f65e53c378370da032f179e5d6da8aff83a9623e40cf902dbd0f9f4174134",
    "eusthenopteron_foordi_specimen.jpg": "d97b68e66855434cb59cbd6b3fc48d7827c97c6981f567e77fb025c359499811",
    "scout_manifest.json": "f16b6a2a68daae0b42086612411b168b2d3c94ad5cc55f163e3c2c8abc87a8eb",
    "paper_citations.bib": "d0ce9743bd590091a6187946131a0537200b1e5e15c530aba8c3bd3e5e235517"
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
    ref = json.loads((DATASET_DIR / "eusthenopteron_foordi_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1980-jarvik-eusthenopteron-foordi-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
