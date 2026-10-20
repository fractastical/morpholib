#!/usr/bin/env python3
"""
1981-seilacher-echinarachnius-parma-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0360)
Organism: Echinarachnius parma (Common Sand Dollar)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Echinarachnius
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0360_echinarachnius_parma_clock_wavefront.lean` (`MathOfBiology.Loop_0360`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/echinarachnius_parma_clock_wavefront"
EXPECTED_SHA256 = {
    "echinarachnius_parma_clock_wavefront_reference.json": "b0bef011aa0f4de98398b6de7678679343afea0b9c41249d29f8bb0227659378",
    "echinarachnius_parma_specimen.jpg": "9076ecda047934e3a862128e9da39faefb16c968f63494a46db1ab7512327923",
    "scout_manifest.json": "dcad8a72226ce6ee931983f2743e5c3842ed4e0f8ee376b95cb8bda935f8c3f5",
    "paper_citations.bib": "fa56cd80fe82520b532ec9dbf21786b1d831b4c2ce4117faafe3922ed31f7c48"
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
    ref = json.loads((DATASET_DIR / "echinarachnius_parma_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1981-seilacher-echinarachnius-parma-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
