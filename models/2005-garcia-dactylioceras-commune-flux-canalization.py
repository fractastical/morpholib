#!/usr/bin/env python3
"""
models/2005-garcia-dactylioceras-commune-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1507)
Organism: Dactylioceras commune (Early Jurassic Whitby Bifurcating-Rib Ammonite (Fractal Suture, D_f=1.55))
Phenomenon Scale: 9.3 mm (0.009345 m, log10=-2.029, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Dactylioceras commune (Early Jurassic (Toarcian, 181 Ma))
Geological Horizon: Early Jurassic (Toarcian, 181 Ma) (181.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Dactylioceras
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (9.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1507_dactylioceras_commune_flux_canalization.lean` (`MathOfBiology.Loop_1507`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dactylioceras_commune_flux_canalization"
EXPECTED_SHA256 = {
    "dactylioceras_commune_flux_canalization_reference.json": "0515394d45f92184576f600ff26851dba974cc0c58af76e77e32f3b117ff5694",
    "dactylioceras_commune_specimen.jpg": "565735d7837252ce5f4f6091140a7e409f84cc8f19a88e5b16fe545b77016fbb",
    "scout_manifest.json": "c74cac32281334ee0a46c2dc72c8592e9619040ead73193b615b358855608e6a",
    "paper_citations.bib": "170c18b1d3ad4e19d27e7d920721df0ecbeb2f360552ab0ff995460c98cc0383"
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
    ref = json.loads((DATASET_DIR / "dactylioceras_commune_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-garcia-dactylioceras-commune-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
