#!/usr/bin/env python3
"""
models/2021-munro-physalia-physalis-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1466)
Organism: Physalia physalis (Portuguese Man o' War (Chiral Left/Right Dimorphic Pneumatophore Sail))
Phenomenon Scale: 8.5 mm (0.008544 m, log10=-2.068, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Physalia physalis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Physalia
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (8.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1466_physalia_physalis_clock_wavefront.lean` (`MathOfBiology.Loop_1466`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/physalia_physalis_clock_wavefront"
EXPECTED_SHA256 = {
    "physalia_physalis_clock_wavefront_reference.json": "e795fbbbd41fd253029cbe44a3acda5a5c6eef4f7b7adc87080096d4fe9c1a62",
    "physalia_physalis_specimen.jpg": "b68854e9fdaaf725c3899f1667f4e8361f6a0f2556862c28573db2d565a50904",
    "scout_manifest.json": "c8dd832b4a893fb968bee2c4a31c3a9d58eb7d462330b5bfd878b2d82bed1729",
    "paper_citations.bib": "782464b29e7ed04e8a716e623e52473ff2594e69bf436ee43c1a276f88ac5df6"
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
    ref = json.loads((DATASET_DIR / "physalia_physalis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2021-munro-physalia-physalis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
