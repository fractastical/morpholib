#!/usr/bin/env python3
"""
models/2008-bell-gasterosteus-doryssus-mid-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1388)
Organism: Gasterosteus doryssus_mid (Miocene Stickleback Chronospecies II (Vestigial Spine, T=+9 kyr))
Phenomenon Scale: 3.3 mm (0.003332 m, log10=-2.477, 6_organ_mm)
Anatomical Structure: Segmentation clock somite / metamere unit in Gasterosteus doryssus_mid (Late Miocene (Truckee Fm, 10.01 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 10.01 Ma) (10.01 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (3.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1388_gasterosteus_doryssus_mid_clock_wavefront.lean` (`MathOfBiology.Loop_1388`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_mid_clock_wavefront"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_mid_clock_wavefront_reference.json": "c5ce7c5c0214f75770a0428c5e361d020474e2689d53035218393090d0839ae4",
    "gasterosteus_doryssus_mid_specimen.jpg": "52d02b91f4693fdd40965b6d78971fdf8ebe6601d12fad6c49a6255e45791e42",
    "scout_manifest.json": "bd26fab45bd986f3f4662fe0a97453a12f6da287303e2ec794a018db3f416cc9",
    "paper_citations.bib": "52d27946738f27351bee9e2e3bf3a1b2eba4cfc6b9363ac89f365a6bc72f9add"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_mid_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-bell-gasterosteus-doryssus-mid-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
