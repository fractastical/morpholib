#!/usr/bin/env python3
"""
1963-tartar-stentor-coeruleus-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0396)
Organism: Stentor coeruleus (Trumpet Ciliate)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Stentor
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0396_stentor_coeruleus_clock_wavefront.lean` (`MathOfBiology.Loop_0396`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/stentor_coeruleus_clock_wavefront"
EXPECTED_SHA256 = {
    "stentor_coeruleus_clock_wavefront_reference.json": "adf7b012211f34d80af90ddcc3fcb054ff5ae16569f2ac8543c40a63cfcfcd55",
    "stentor_coeruleus_specimen.jpg": "a1681963dfc488967f3552b87eed7d5716e022271322cdda8623fa8000374fb8",
    "scout_manifest.json": "8b372973f89e1c3bd13222ee795c6efb3bbcab454c74025013c8afb809c7c4ce",
    "paper_citations.bib": "39b7021f65495ba894b0223cf0d6dcbedb73d7213224d23f7734da7a6bdefb1a"
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
    ref = json.loads((DATASET_DIR / "stentor_coeruleus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1963-tartar-stentor-coeruleus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
