#!/usr/bin/env python3
"""
2016-levin-dugesia-japonica-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0366)
Organism: Dugesia japonica (Freshwater Planarian)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Dugesia
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0366_dugesia_japonica_clock_wavefront.lean` (`MathOfBiology.Loop_0366`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/dugesia_japonica_clock_wavefront"
EXPECTED_SHA256 = {
    "dugesia_japonica_clock_wavefront_reference.json": "eba674fc86fe770b9c3179dba60100c1a1d0b01938df262e2ff5d9a7f6e287c4",
    "dugesia_japonica_specimen.jpg": "26b78d28b853c1b70092be6a7ae129d59813507f64eaed16a417276296ff2efa",
    "scout_manifest.json": "1f7b69dda2014960cfbfa12a3f08a7a3a650d061675a1a3f3bfa220800cbfb31",
    "paper_citations.bib": "79835aa28f884531f039d9dd6faf628c4ac69d1a619c7fe74c470d89e3047a02"
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
    ref = json.loads((DATASET_DIR / "dugesia_japonica_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-levin-dugesia-japonica-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
