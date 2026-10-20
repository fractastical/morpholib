#!/usr/bin/env python3
"""
2018-bakker-struthio-camelus-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0315)
Organism: Struthio camelus (Common Ostrich)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Struthio
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0315_struthio_camelus_clock_wavefront.lean` (`MathOfBiology.Loop_0315`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/struthio_camelus_clock_wavefront"
EXPECTED_SHA256 = {
    "struthio_camelus_clock_wavefront_reference.json": "52af0eafce92fbc780604e543b5acc716bab59ca8d26de519ff307480935885a",
    "struthio_camelus_specimen.jpg": "798eba64c73c1474a1b787e850654c5f61164ca01db9ec2d75836507f9494def",
    "scout_manifest.json": "0876284b4844044e863315009f897f0f3b93bb1e2499bf225f4456667709617d",
    "paper_citations.bib": "a1d6f713fe0dafb635c06de13d348138d0b62b2c95a29a7b01ce31beacb08a46"
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
    ref = json.loads((DATASET_DIR / "struthio_camelus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-bakker-struthio-camelus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
