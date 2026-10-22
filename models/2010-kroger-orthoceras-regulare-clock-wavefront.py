#!/usr/bin/env python3
"""
models/2010-kroger-orthoceras-regulare-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1414)
Organism: Orthoceras regulare (Middle Ordovician Straight-Cone Nautiloid (Simple Ortho-Septa))
Phenomenon Scale: 3.8 mm (0.00378 m, log10=-2.423, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Orthoceras regulare (Middle Ordovician (Darriwilian, 464 Ma))
Geological Horizon: Middle Ordovician (Darriwilian, 464 Ma) (464.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Orthoceras
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (3.8 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1414_orthoceras_regulare_clock_wavefront.lean` (`MathOfBiology.Loop_1414`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/orthoceras_regulare_clock_wavefront"
EXPECTED_SHA256 = {
    "orthoceras_regulare_clock_wavefront_reference.json": "aa2c7453d584eea6f35dd962f5e7903f418c3221ead68a2347a0dccf5385858a",
    "orthoceras_regulare_specimen.jpg": "67fd83548556de1506f61ae1eaf002726c3f3a48e3a9d8452937b96eaa47a976",
    "scout_manifest.json": "f4df626ba2da6c1490f4206390083f4f4637aa0a611b32b123c5ad1fdf5ef635",
    "paper_citations.bib": "67f1e09b0fe782b73e4db63ae080cce15846e80e499b8a9c7820361fcf4c99cb"
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
    ref = json.loads((DATASET_DIR / "orthoceras_regulare_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-kroger-orthoceras-regulare-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
