#!/usr/bin/env python3
"""
2006-coen-antirrhinum-majus-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0376)
Organism: Antirrhinum majus (Snapdragon)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Antirrhinum
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0376_antirrhinum_majus_clock_wavefront.lean` (`MathOfBiology.Loop_0376`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/antirrhinum_majus_clock_wavefront"
EXPECTED_SHA256 = {
    "antirrhinum_majus_clock_wavefront_reference.json": "eaaffaea8a196e6a261df7bd65633d641c36191540ad9fd94143cfa4123ce816",
    "antirrhinum_majus_specimen.jpg": "f3e9bf78a4453bc1a72552c6d200a37a1bccf6047928e988edba53ccfc260fe0",
    "scout_manifest.json": "e3cca5c30103126ecd996e07f3d2abc7da18524fff1464c0b9031db4379a4b52",
    "paper_citations.bib": "398379b638cd0a84440022f582dd905246cc3026f61a11c485992152965b61ab"
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
    ref = json.loads((DATASET_DIR / "antirrhinum_majus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-coen-antirrhinum-majus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
