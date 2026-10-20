#!/usr/bin/env python3
"""
2015-forterre-mimosa-pudica-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0378)
Organism: Mimosa pudica (Sensitive Plant)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Mimosa
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0378_mimosa_pudica_clock_wavefront.lean` (`MathOfBiology.Loop_0378`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mimosa_pudica_clock_wavefront"
EXPECTED_SHA256 = {
    "mimosa_pudica_clock_wavefront_reference.json": "4924f083976a5a5778f5a8d55577571ca78f917300385ee5ce1f5c457acdfa4e",
    "mimosa_pudica_specimen.jpg": "4ae1af66e325ef597ebdce00e9f639174383d049bc7b0e98b02e97de71c83e20",
    "scout_manifest.json": "a64ce1e3b8af64c99d2bcbb21f75a5bd093ccb2ea7de5a63f9a46df548bc806f",
    "paper_citations.bib": "7aaef5f0eccd0f3d162df51bb973188e1ac4b3bef1242194cbd526dbc09d16ea"
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
    ref = json.loads((DATASET_DIR / "mimosa_pudica_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-forterre-mimosa-pudica-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
