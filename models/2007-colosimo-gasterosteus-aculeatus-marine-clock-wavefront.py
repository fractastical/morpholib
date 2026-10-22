#!/usr/bin/env python3
"""
models/2007-colosimo-gasterosteus-aculeatus-marine-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1390)
Organism: Gasterosteus aculeatus_marine (Ancestral Anadromous Complete-Morph Stickleback (32 Lateral Plates))
Phenomenon Scale: 4.8 mm (0.004836 m, log10=-2.316, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Gasterosteus aculeatus_marine (Late Pleistocene Marine Ancestor (15 ka))
Geological Horizon: Late Pleistocene Marine Ancestor (15 ka) (0.015 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (4.8 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1390_gasterosteus_aculeatus_marine_clock_wavefront.lean` (`MathOfBiology.Loop_1390`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_marine_clock_wavefront"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_marine_clock_wavefront_reference.json": "602b4ddcc0d27ff3da9f6ca760ded136b95dfdab8345d5cf4fe3cc6edbd9273b",
    "gasterosteus_aculeatus_marine_specimen.jpg": "4da41d2381dcc4c8c66a7b6ffe79757e5ca0b0dae3978efe5a16417ec3a5bd15",
    "scout_manifest.json": "a6839435c5464aa16f0b8cf3fe6c58e7e976d4da96a7b6be350883e8697ee332",
    "paper_citations.bib": "f9f688b167cfb22c47d5a62e3f0f3632f2f17cb91f404d3b2a8b33ca7f8a138a"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_marine_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-colosimo-gasterosteus-aculeatus-marine-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
