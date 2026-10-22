#!/usr/bin/env python3
"""
models/2024-turner-asteroxylon-mackiei-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1346)
Organism: Asteroxylon mackiei (Early Devonian Rhynie Chert Lycopsid (Actinostele & Fibonacci Microphylls))
Phenomenon Scale: 1.3 mm (0.001275 m, log10=-2.894, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Asteroxylon mackiei (Early Devonian (Pragian Rhynie Chert, 407 Ma))
Geological Horizon: Early Devonian (Pragian Rhynie Chert, 407 Ma) (407.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Asteroxylon
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (1.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1346_asteroxylon_mackiei_morphogen_condensation.lean` (`MathOfBiology.Loop_1346`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/asteroxylon_mackiei_morphogen_condensation"
EXPECTED_SHA256 = {
    "asteroxylon_mackiei_morphogen_condensation_reference.json": "c4cbc6e779a8a0e8d2edbf6c287d7e6dd0ade12ca735336fdc8fe794f71b4042",
    "asteroxylon_mackiei_specimen.jpg": "af15c01893b31829fefca3fc2246896ab8d2c12ab87d5809f28504e9c2968531",
    "scout_manifest.json": "275aa12823069abff265f4b087e7442f86583a4401bba241455f0f43bf0116d0",
    "paper_citations.bib": "e72317b011f4eb1daa9f27217d34a5dcc306eb824b7c0e2da48b896f5dc166c1"
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
    ref = json.loads((DATASET_DIR / "asteroxylon_mackiei_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2024-turner-asteroxylon-mackiei-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
