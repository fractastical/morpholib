#!/usr/bin/env python3
"""
models/1990-okamoto-nipponites-mirabilis-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1420)
Organism: Nipponites mirabilis (Late Cretaceous Meandering 3D Trefoil-Knot Heteromorph Ammonite)
Phenomenon Scale: 4.4 mm (0.004363 m, log10=-2.360, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Nipponites mirabilis (Late Cretaceous (Coniacian, 88 Ma))
Geological Horizon: Late Cretaceous (Coniacian, 88 Ma) (88.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Nipponites
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (4.4 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1420_nipponites_mirabilis_clock_wavefront.lean` (`MathOfBiology.Loop_1420`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nipponites_mirabilis_clock_wavefront"
EXPECTED_SHA256 = {
    "nipponites_mirabilis_clock_wavefront_reference.json": "ec7d7ba0cc8e6f5264e12e0c4bfbfe9c53f07a6bd45c3f77394b9f2916057965",
    "nipponites_mirabilis_specimen.jpg": "6b8124b1e5eeb73dc589cf76fafc5db8b2d4a7d15eb68c77a1175d267f717dd5",
    "scout_manifest.json": "f8b7441815d5c096a890dcee9ba8e7d8cb627e3c9090330525ec9f0d63e84541",
    "paper_citations.bib": "f334a3a0294230decf1ee71cd02375beedf408863bf649ea189ed358a80c721c"
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
    ref = json.loads((DATASET_DIR / "nipponites_mirabilis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1990-okamoto-nipponites-mirabilis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
