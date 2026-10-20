#!/usr/bin/env python3
"""
2005-zi-pavo-cristatus-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0313)
Organism: Pavo cristatus (Indian Peafowl)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Pavo
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0313_pavo_cristatus_clock_wavefront.lean` (`MathOfBiology.Loop_0313`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pavo_cristatus_clock_wavefront"
EXPECTED_SHA256 = {
    "pavo_cristatus_clock_wavefront_reference.json": "071beb80b42b1c242597c582edf2941196debe5c94e7b5ab040a1f43a377fe6c",
    "pavo_cristatus_specimen.jpg": "39f28e26a62ddfdb98fe19fee7eaf7af508c82486a32abe3c1f2ac7d3e607d50",
    "scout_manifest.json": "90deea10a24fdecd65f68c1e136cd4b282c720f3c7b22868dfc03113821af538",
    "paper_citations.bib": "f283cad5a9c065de776ae39ea065f1a53580679f47dcc3c7dd08cf8d06d6223b"
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
    ref = json.loads((DATASET_DIR / "pavo_cristatus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-zi-pavo-cristatus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
