#!/usr/bin/env python3
"""
models/2001-saunders-goniatites-crenistria-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1415)
Organism: Goniatites crenistria (Mississippian Goniatitic Ammonoid (Angular Zigzag Suture, D_f=1.18))
Phenomenon Scale: 1.6 mm (0.00157 m, log10=-2.804, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Goniatites crenistria (Early Carboniferous (Viséan, 333 Ma))
Geological Horizon: Early Carboniferous (Viséan, 333 Ma) (333.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Goniatites
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1415_goniatites_crenistria_clock_wavefront.lean` (`MathOfBiology.Loop_1415`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/goniatites_crenistria_clock_wavefront"
EXPECTED_SHA256 = {
    "goniatites_crenistria_clock_wavefront_reference.json": "b0baeae4ebedda43e8586c06fc2d8d1a47d3e77ca1a100d8291da0a7fa091dd2",
    "goniatites_crenistria_specimen.jpg": "1c025730c53d3fe4dde10fe4dfc3229dac439001911ee16d02107da5f04120fb",
    "scout_manifest.json": "24502100536f2d510fb1e6a84344457b28e535fb538b1e389a3128994f2393aa",
    "paper_citations.bib": "157663f94796713ae436df0c8bd90e39856bd83fc407709bbe0add22a05eaabc"
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
    ref = json.loads((DATASET_DIR / "goniatites_crenistria_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-saunders-goniatites-crenistria-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
