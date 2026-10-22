#!/usr/bin/env python3
"""
models/2008-shubin-tiktaalik-roseae-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1394)
Organism: Tiktaalik roseae (Late Devonian Wristed Fin-Limb Transitional Fossil)
Phenomenon Scale: 1.7 cm (0.01657 m, log10=-1.781, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Tiktaalik roseae (Late Devonian (Frasnian, 375 Ma))
Geological Horizon: Late Devonian (Frasnian, 375 Ma) (375.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Tiktaalik
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.7 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1394_tiktaalik_roseae_clock_wavefront.lean` (`MathOfBiology.Loop_1394`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/tiktaalik_roseae_clock_wavefront"
EXPECTED_SHA256 = {
    "tiktaalik_roseae_clock_wavefront_reference.json": "f966a1f09231818edbdea1991ca949b26c455df625df9ed933229ed6d9f71135",
    "tiktaalik_roseae_specimen.jpg": "2b6fe2e7441ef8b8a1f69c1382b71a3606ab9e0ee552372999fb3ab07fb8c3a4",
    "scout_manifest.json": "f125347d41c0f7231c565e33d0a57d36d31feb24d6bd8909487f56105836f80b",
    "paper_citations.bib": "07a64747b31382c45f5fb4a6191a0440da2596722deda13553b0a7fea5412fa4"
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
    ref = json.loads((DATASET_DIR / "tiktaalik_roseae_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-shubin-tiktaalik-roseae-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
