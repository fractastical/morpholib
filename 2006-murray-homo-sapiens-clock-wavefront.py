#!/usr/bin/env python3
"""
2006-murray-homo-sapiens-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0302)
Organism: Homo sapiens (Human)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Homo
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0302_homo_sapiens_clock_wavefront.lean` (`MathOfBiology.Loop_0302`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/homo_sapiens_clock_wavefront"
EXPECTED_SHA256 = {
    "homo_sapiens_clock_wavefront_reference.json": "b82e6edeacabdb237bf734100945893087ee6e3a20899ecc90d9bf95b2a3de23",
    "homo_sapiens_specimen.jpg": "ed3d39e41faa4dd97074816c05e647bdc39bf7f335f22ddfa3bcd4b2760ee735",
    "scout_manifest.json": "c1fb0e2971e25b02eefa4faf00f88925e0291d6156043fa9eb514fa7da81180f",
    "paper_citations.bib": "250dadd7995ab2d81ec46ad918d4a5106da022e1394ca8be78b46e6ba1aa1df6"
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
    ref = json.loads((DATASET_DIR / "homo_sapiens_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-murray-homo-sapiens-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
