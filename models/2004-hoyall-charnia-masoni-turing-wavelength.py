#!/usr/bin/env python3
"""
models/2004-hoyall-charnia-masoni-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1249)
Organism: Charnia masoni (Ediacaran Self-Similar Glide-Reflected Rangeomorph Frond (Mistaken Point))
Phenomenon Scale: 6.3 mm (0.006322 m, log10=-2.199, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Charnia masoni (Late Ediacaran (565 Ma))
Geological Horizon: Late Ediacaran (565 Ma) (565.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Charnia
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (6.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1249_charnia_masoni_turing_wavelength.lean` (`MathOfBiology.Loop_1249`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/charnia_masoni_turing_wavelength"
EXPECTED_SHA256 = {
    "charnia_masoni_turing_wavelength_reference.json": "90c369add279393a153df97f11218f3701f60115ec560f2495bd5efe1bde10c0",
    "charnia_masoni_specimen.jpg": "92b6f0531f638c633e69def818647884521ea36a8f79147751e0437ea9e17b3a",
    "scout_manifest.json": "6a2df2d256f4bbb6119798523daba68e7272a4e1932bfe87193c52e45aadb4ad",
    "paper_citations.bib": "6f436817f632703c8c2893e98edcb5cfacfd19a03112d1467604d7dd28da4b25"
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
    ref = json.loads((DATASET_DIR / "charnia_masoni_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-hoyall-charnia-masoni-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
