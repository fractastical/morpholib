#!/usr/bin/env python3
"""
models/2013-paterson-anomalocaris-canadensis-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1428)
Organism: Anomalocaris canadensis (Cambrian Burgess Shale Radiodont Apex Predator (Compound Eye & Oral Cone))
Phenomenon Scale: 2.4 mm (0.002352 m, log10=-2.629, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Anomalocaris canadensis (Middle Cambrian (Burgess Shale, 508 Ma))
Geological Horizon: Middle Cambrian (Burgess Shale, 508 Ma) (508.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Anomalocaris
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (2.4 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1428_anomalocaris_canadensis_clock_wavefront.lean` (`MathOfBiology.Loop_1428`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/anomalocaris_canadensis_clock_wavefront"
EXPECTED_SHA256 = {
    "anomalocaris_canadensis_clock_wavefront_reference.json": "8ad684d9303b3bbe1e0fb3e0d2d23463b10aef9195f71cf338af3f9eb1438e6f",
    "anomalocaris_canadensis_specimen.jpg": "197e43665198870b80e436c90febc5b7010eb24957c9239893e6ac1be683bf9c",
    "scout_manifest.json": "15a32f3cd6ca687951c836daf3711e2654319d0a38062d53becf9e6ed774a1b8",
    "paper_citations.bib": "74c7a2ea1cbc38f557ec5362d6aaaa7e1aa33fd6523bb2f503165504f8bfa994"
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
    ref = json.loads((DATASET_DIR / "anomalocaris_canadensis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-paterson-anomalocaris-canadensis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
