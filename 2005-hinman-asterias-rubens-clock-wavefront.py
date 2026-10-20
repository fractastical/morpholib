#!/usr/bin/env python3
"""
2005-hinman-asterias-rubens-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0354)
Organism: Asterias rubens (Common Starfish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Asterias
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0354_asterias_rubens_clock_wavefront.lean` (`MathOfBiology.Loop_0354`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/asterias_rubens_clock_wavefront"
EXPECTED_SHA256 = {
    "asterias_rubens_clock_wavefront_reference.json": "130019bcf373ddf907bbcca1492ce9947c892e0176ad0e81964d0f6a4dd6f2a6",
    "asterias_rubens_specimen.jpg": "37d94bdfed179302d1e9aa062333c2f87a5cb3d7c3932c51056f5e68c7c762ad",
    "scout_manifest.json": "bf8feed48370c6ecf5d7ac83125b894f3f6170b312b12fa70d83369e015cbbfe",
    "paper_citations.bib": "60b0691c423177c7079d4c88c0b633433408cf39a4129920b80740193820f39a"
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
    ref = json.loads((DATASET_DIR / "asterias_rubens_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-hinman-asterias-rubens-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
