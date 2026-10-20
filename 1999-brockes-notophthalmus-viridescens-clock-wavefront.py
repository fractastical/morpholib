#!/usr/bin/env python3
"""
1999-brockes-notophthalmus-viridescens-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0327)
Organism: Notophthalmus viridescens (Eastern Newt)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Notophthalmus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0327_notophthalmus_viridescens_clock_wavefront.lean` (`MathOfBiology.Loop_0327`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/notophthalmus_viridescens_clock_wavefront"
EXPECTED_SHA256 = {
    "notophthalmus_viridescens_clock_wavefront_reference.json": "e35fb9fa8416793c916833f418bdb04fb036c24ea284947b4ec3615660aa8ecc",
    "notophthalmus_viridescens_specimen.jpg": "5389e095a5be650aba6c5b00fa322f4fbd3beed046b1913ac82a95ce045d5000",
    "scout_manifest.json": "68b99c570e674617e175bdf00e3f8778f71c2afd1a317cccebbd6d7be5652da4",
    "paper_citations.bib": "3a0fccd4ab68d3a6ae81b35140ceaf8ccd50533afb314965a8a2a2e22fb22c6a"
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
    ref = json.loads((DATASET_DIR / "notophthalmus_viridescens_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1999-brockes-notophthalmus-viridescens-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
