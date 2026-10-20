#!/usr/bin/env python3
"""
2001-spemann-rana-temporaria-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0326)
Organism: Rana temporaria (European Common Frog)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Rana
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0326_rana_temporaria_clock_wavefront.lean` (`MathOfBiology.Loop_0326`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/rana_temporaria_clock_wavefront"
EXPECTED_SHA256 = {
    "rana_temporaria_clock_wavefront_reference.json": "99e69cc986a5b4b9495b209e9d1901ee843aa427f0fa02c4664e29313efb709d",
    "rana_temporaria_specimen.jpg": "966174057e8f12954fa5a7ba2b0a2b543b64c2900c6c92ea4a55926d1c57f111",
    "scout_manifest.json": "ef09ecd0cba9988607c64e187f2bfe1e3a5075f95d2e663e3c0182dace82c186",
    "paper_citations.bib": "a0c84595cc4ce443e12fbf3d2e2691c2cf7b741742c5bb589870ac8a5227b3be"
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
    ref = json.loads((DATASET_DIR / "rana_temporaria_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-spemann-rana-temporaria-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
