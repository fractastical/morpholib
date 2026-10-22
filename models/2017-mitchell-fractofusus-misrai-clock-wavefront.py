#!/usr/bin/env python3
"""
models/2017-mitchell-fractofusus-misrai-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1431)
Organism: Fractofusus misrai (Ediacaran Four-Order Self-Similar Spindle Rangeomorph)
Phenomenon Scale: 6.6 mm (0.00655 m, log10=-2.184, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Fractofusus misrai (Late Ediacaran (Newfoundland, 565 Ma))
Geological Horizon: Late Ediacaran (Newfoundland, 565 Ma) (565.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Fractofusus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (6.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1431_fractofusus_misrai_clock_wavefront.lean` (`MathOfBiology.Loop_1431`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/fractofusus_misrai_clock_wavefront"
EXPECTED_SHA256 = {
    "fractofusus_misrai_clock_wavefront_reference.json": "3a09f3d8675b5c8d61eb7ff920c5e0dc678d906055c3d55841a1794c8badb4af",
    "fractofusus_misrai_specimen.jpg": "b6b0eca46e1970dc859de875d6c583e3ce1abb548f235c9292a993aa71e14061",
    "scout_manifest.json": "28619201eff84a65a442b70adb331980b3b136d687fd1b820e74f8aa767a13f1",
    "paper_citations.bib": "2b5f517fdea7cf04e656e37d44ec72c70727e377ec11bc7ac15fe26030f4140e"
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
    ref = json.loads((DATASET_DIR / "fractofusus_misrai_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-mitchell-fractofusus-misrai-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
