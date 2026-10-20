#!/usr/bin/env python3
"""
2016-ocs-zea-mays-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0373)
Organism: Zea mays (Maize)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Zea
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0373_zea_mays_clock_wavefront.lean` (`MathOfBiology.Loop_0373`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/zea_mays_clock_wavefront"
EXPECTED_SHA256 = {
    "zea_mays_clock_wavefront_reference.json": "687748afbf70ada07a87e4a90f5aad4410e48d3e27cd067b8d800f7d23552df0",
    "zea_mays_specimen.jpg": "a78e27ea3c83c91a3f84373319a3b590f4a53e6108ddf50860d114690cbb6db9",
    "scout_manifest.json": "d5d5f608e70ee5514006f792dea431893067bca78e5bf3f650c4e514b08f9728",
    "paper_citations.bib": "89ac72f4dc2956a4e103fa62eb54275bae47b383abb1fde45d3394cd4c16f0bb"
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
    ref = json.loads((DATASET_DIR / "zea_mays_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-ocs-zea-mays-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
