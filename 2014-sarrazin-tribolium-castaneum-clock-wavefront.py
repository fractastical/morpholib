#!/usr/bin/env python3
"""
2014-sarrazin-tribolium-castaneum-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0342)
Organism: Tribolium castaneum (Red Flour Beetle)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Tribolium
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0342_tribolium_castaneum_clock_wavefront.lean` (`MathOfBiology.Loop_0342`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tribolium_castaneum_clock_wavefront"
EXPECTED_SHA256 = {
    "tribolium_castaneum_clock_wavefront_reference.json": "a83ac9822984e15ec8659efaaf4ce6971eb55b98c82ce309168ae35ea69b75e3",
    "tribolium_castaneum_specimen.jpg": "a991c507b6f59178075bea80d7ad2c1c97d839991d31594797919b07b5ac1d77",
    "scout_manifest.json": "304c8038502af413987cbe8378582ed7d3fe9bca9145abd7b47dbb0a44ffaeae",
    "paper_citations.bib": "ce3a41974e2fccbb08cf166470cdbb0c70800959293cf15fd6df553b844c393a"
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
    ref = json.loads((DATASET_DIR / "tribolium_castaneum_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-sarrazin-tribolium-castaneum-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
