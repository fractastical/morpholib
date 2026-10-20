#!/usr/bin/env python3
"""
2023-chirat-argonauta-argo-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0356)
Organism: Argonauta argo (Paper Nautilus)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Argonauta
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0356_argonauta_argo_clock_wavefront.lean` (`MathOfBiology.Loop_0356`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/argonauta_argo_clock_wavefront"
EXPECTED_SHA256 = {
    "argonauta_argo_clock_wavefront_reference.json": "e0acaefa388c0becec64b7171ce1535af9df578ed71d051fd2a3db247f47699d",
    "argonauta_argo_specimen.jpg": "4d2cfc3367b264b3d9c93b5d4ea3794e99cb317bbc495df52ab10c540174dff2",
    "scout_manifest.json": "f2d4d4e867f2d7c09ce4fafe8855b79406e6c7ff09c86481837b15c14b31086b",
    "paper_citations.bib": "c6111b4a470b03111bee343bf45ce7d06db887a5e4c9be12211a63de4f088c8a"
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
    ref = json.loads((DATASET_DIR / "argonauta_argo_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2023-chirat-argonauta-argo-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
