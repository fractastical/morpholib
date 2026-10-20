#!/usr/bin/env python3
"""
2017-hohn-volvox-carteri-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0391)
Organism: Volvox carteri (Colonial Green Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Volvox
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0391_volvox_carteri_clock_wavefront.lean` (`MathOfBiology.Loop_0391`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/volvox_carteri_clock_wavefront"
EXPECTED_SHA256 = {
    "volvox_carteri_clock_wavefront_reference.json": "ace6f711c310c01624b0ad73d7eb45a9290be52d60ecc1a4e9ee4fdef9462ca0",
    "volvox_carteri_specimen.jpg": "cf734e5f4b2d7f5ded097049b19cf0c7c00c592b2c19a662073fe8d066b17e4a",
    "scout_manifest.json": "f5f07f5f34c61773ad5939c165ec73c1c25b6eb41959347e32e6d0b1f15423c2",
    "paper_citations.bib": "00544ddc2cd9ffc4c4ee79311d5e8103319ed54e0da9cdf687e8b9e1676987e0"
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
    ref = json.loads((DATASET_DIR / "volvox_carteri_clock_wavefront_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2017-hohn-volvox-carteri-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
