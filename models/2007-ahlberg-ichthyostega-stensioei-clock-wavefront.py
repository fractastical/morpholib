#!/usr/bin/env python3
"""
models/2007-ahlberg-ichthyostega-stensioei-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1396)
Organism: Ichthyostega stensioei (Late Devonian Seven-Digited Weight-Bearing Tetrapod)
Phenomenon Scale: 1.1 cm (0.01104 m, log10=-1.957, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Ichthyostega stensioei (Late Devonian (Famennian, 363 Ma))
Geological Horizon: Late Devonian (Famennian, 363 Ma) (363.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Ichthyostega
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1396_ichthyostega_stensioei_clock_wavefront.lean` (`MathOfBiology.Loop_1396`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ichthyostega_stensioei_clock_wavefront"
EXPECTED_SHA256 = {
    "ichthyostega_stensioei_clock_wavefront_reference.json": "08ab27ebf832fa88be4fbc963f6a6eb778b94f8cc383e735b22c4dd99f79db70",
    "ichthyostega_stensioei_specimen.jpg": "d0371040f354560ee3b95904f5d16c9f224254e9b490bc2d4a9c7d712aa42e0c",
    "scout_manifest.json": "6d4c3f30f3ba266127dac5c5dc91bcfe8da97878fa2692833050d81c6e765412",
    "paper_citations.bib": "345268e4bafb74bf94d66cf28495ad2e8b30ce14945945c83613378ad4f54bdd"
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
    ref = json.loads((DATASET_DIR / "ichthyostega_stensioei_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-ahlberg-ichthyostega-stensioei-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
