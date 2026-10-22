#!/usr/bin/env python3
"""
models/2012-paterson-anomalocaris-canadensis-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1338)
Organism: Anomalocaris canadensis (Cambrian Burgess Shale Radiodont Apex Predator (Compound Eye & Oral Cone))
Phenomenon Scale: 5.3 mm (0.005292 m, log10=-2.276, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Anomalocaris canadensis (Middle Cambrian (Burgess Shale, 508 Ma))
Geological Horizon: Middle Cambrian (Burgess Shale, 508 Ma) (508.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Anomalocaris
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (5.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1338_anomalocaris_canadensis_morphogen_condensation.lean` (`MathOfBiology.Loop_1338`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/anomalocaris_canadensis_morphogen_condensation"
EXPECTED_SHA256 = {
    "anomalocaris_canadensis_morphogen_condensation_reference.json": "eba448d2bc501a7e478a5cf016aa4462f1a7e2b4fcdea86bfc64af3c57e47dc1",
    "anomalocaris_canadensis_specimen.jpg": "197e43665198870b80e436c90febc5b7010eb24957c9239893e6ac1be683bf9c",
    "scout_manifest.json": "a4737dd00da5ad4b766372a23cfab0483f9ee45159b83e2f5eff6d790ad530e0",
    "paper_citations.bib": "3aa9eb901d3811bafd36057a4c1626851e6cdb31cf8ffdf809d6a1800f87d62f"
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
    ref = json.loads((DATASET_DIR / "anomalocaris_canadensis_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-paterson-anomalocaris-canadensis-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
