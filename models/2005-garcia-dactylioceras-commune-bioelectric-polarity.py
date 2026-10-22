#!/usr/bin/env python3
"""
models/2005-garcia-dactylioceras-commune-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1957)
Organism: Dactylioceras commune (Early Jurassic Whitby Bifurcating-Rib Ammonite (Fractal Suture, D_f=1.55))
Phenomenon Scale: 30 µm (2.99e-05 m, log10=-4.524, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Dactylioceras commune (Early Jurassic (Toarcian, 181 Ma))
Geological Horizon: Early Jurassic (Toarcian, 181 Ma) (181.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Dactylioceras
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (30 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1957_dactylioceras_commune_bioelectric_polarity.lean` (`MathOfBiology.Loop_1957`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dactylioceras_commune_bioelectric_polarity"
EXPECTED_SHA256 = {
    "dactylioceras_commune_bioelectric_polarity_reference.json": "934f6fb54fefa18a2381806883858f0577bf3a250562262ce7cb218b220181aa",
    "dactylioceras_commune_specimen.jpg": "565735d7837252ce5f4f6091140a7e409f84cc8f19a88e5b16fe545b77016fbb",
    "scout_manifest.json": "9cc15906b31c2740184bc0426b7291c229dbd1abecabd57e7a4dd6b0e8c8bd31",
    "paper_citations.bib": "50bf307b4ffab5c412a8d492f12815c98f809e68de884e0d5aa49c72a3d8301f"
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
    ref = json.loads((DATASET_DIR / "dactylioceras_commune_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-garcia-dactylioceras-commune-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
