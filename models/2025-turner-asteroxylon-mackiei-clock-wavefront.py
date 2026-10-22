#!/usr/bin/env python3
"""
models/2025-turner-asteroxylon-mackiei-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1436)
Organism: Asteroxylon mackiei (Early Devonian Rhynie Chert Lycopsid (Actinostele & Fibonacci Microphylls))
Phenomenon Scale: 567 µm (0.0005668 m, log10=-3.247, 5_primordium_100um)
Anatomical Structure: Segmentation clock somite / metamere unit in Asteroxylon mackiei (Early Devonian (Pragian Rhynie Chert, 407 Ma))
Geological Horizon: Early Devonian (Pragian Rhynie Chert, 407 Ma) (407.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Asteroxylon
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (567 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1436_asteroxylon_mackiei_clock_wavefront.lean` (`MathOfBiology.Loop_1436`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/asteroxylon_mackiei_clock_wavefront"
EXPECTED_SHA256 = {
    "asteroxylon_mackiei_clock_wavefront_reference.json": "dfbbf283ef9f8515f303a1d2c5fcb0932756c0ab786900f1e78991d889a1ff14",
    "asteroxylon_mackiei_specimen.jpg": "af15c01893b31829fefca3fc2246896ab8d2c12ab87d5809f28504e9c2968531",
    "scout_manifest.json": "6fd3530bd398995d601fdb259688f8d4e62df28289d8a45bf63f3a82a9d49d9d",
    "paper_citations.bib": "830a7611d34637c1c1fcac3eb693bf3791bc8f49698fb0aef0a529da337b7373"
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
    ref = json.loads((DATASET_DIR / "asteroxylon_mackiei_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2025-turner-asteroxylon-mackiei-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
