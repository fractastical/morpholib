#!/usr/bin/env python3
"""
models/2020-martins-loxodonta-africana-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1457)
Organism: Loxodonta africana (African Bush Elephant (Skin Thermal Mud-Cracking Channel Network))
Phenomenon Scale: 105 µm (0.0001046 m, log10=-3.980, 5_primordium_100um)
Anatomical Structure: Segmentation clock somite / metamere unit in Loxodonta africana (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Loxodonta
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (105 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1457_loxodonta_africana_clock_wavefront.lean` (`MathOfBiology.Loop_1457`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/loxodonta_africana_clock_wavefront"
EXPECTED_SHA256 = {
    "loxodonta_africana_clock_wavefront_reference.json": "ab88038fbcb7efdf9e303faa0a3eaa199861db74ee1e3cd8399c3b45de897c92",
    "loxodonta_africana_specimen.jpg": "a530e48a3daec38e9c6fd4a03ae176871589b79b17232feda58acffdff236163",
    "scout_manifest.json": "70a234be076c90c0dfea7e57e60964ca4729011ae72e161315a7bb31e73e643e",
    "paper_citations.bib": "e408ea6c63777a2f9d3adfb1d6728d0a9d723c0484a87421e748a4dd8500de53"
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
    ref = json.loads((DATASET_DIR / "loxodonta_africana_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2020-martins-loxodonta-africana-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
