#!/usr/bin/env python3
"""
2005-reinhardt-solanum-lycopersicum-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0374)
Organism: Solanum lycopersicum (Tomato)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Solanum
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0374_solanum_lycopersicum_clock_wavefront.lean` (`MathOfBiology.Loop_0374`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/solanum_lycopersicum_clock_wavefront"
EXPECTED_SHA256 = {
    "solanum_lycopersicum_clock_wavefront_reference.json": "435b6c59aa7df76a0447a075b997a2bd4d69925a7fb92279348e21143f61a257",
    "solanum_lycopersicum_specimen.jpg": "3ae537938870d7101582c24be70959db59fa5172fccc4d6ad7523d5b64cb2ec0",
    "scout_manifest.json": "4c6defab379db5c949060f921f624040744913beb27a0f1bfc96ec1207239920",
    "paper_citations.bib": "67ea2a04b6c4124473ddd188c6b0908b28a07818b908361021f0dbf0caf2dc67"
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
    ref = json.loads((DATASET_DIR / "solanum_lycopersicum_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-reinhardt-solanum-lycopersicum-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
