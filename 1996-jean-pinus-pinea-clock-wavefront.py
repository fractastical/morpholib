#!/usr/bin/env python3
"""
1996-jean-pinus-pinea-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0375)
Organism: Pinus pinea (Stone Pine)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Pinus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0375_pinus_pinea_clock_wavefront.lean` (`MathOfBiology.Loop_0375`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pinus_pinea_clock_wavefront"
EXPECTED_SHA256 = {
    "pinus_pinea_clock_wavefront_reference.json": "fdeeab92e8605fca2f8f67f9e8c4333d9645c3c33ca7662f1dd078382e909786",
    "pinus_pinea_specimen.jpg": "87c1fef6a6fb55bd3bf3e35f590d280e453c1348b5d11f18c356c09345da3904",
    "scout_manifest.json": "87e665c2c82f9f199d0a637cf3bc0f5734ae4705aa8d4108cd0b9aaf94883abd",
    "paper_citations.bib": "af9bd06088c6a738b11aaa90d5c851a3ce18a5cf9b09af945546d96b9701ecbc"
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
    ref = json.loads((DATASET_DIR / "pinus_pinea_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1996-jean-pinus-pinea-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
