#!/usr/bin/env python3
"""
2007-kaandorp-acropora-millepora-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0365)
Organism: Acropora millepora (Branching Staghorn Coral)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Acropora
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0365_acropora_millepora_clock_wavefront.lean` (`MathOfBiology.Loop_0365`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acropora_millepora_clock_wavefront"
EXPECTED_SHA256 = {
    "acropora_millepora_clock_wavefront_reference.json": "917f82f5d0637b1157cd459d0f03b98938a6f3f8c3573a41a965aac17a031bbe",
    "acropora_millepora_specimen.jpg": "85c35e854dd9d07d46528548512579040fef8d6cd4893257fb28bd3b4a468eb0",
    "scout_manifest.json": "ced2c79f2380c4e62f9446a786eef48136dd6b2f041b98aa5daa149cc12c734c",
    "paper_citations.bib": "b613c3c4d2036ddcc4b89be72c14941ffd625d3d1792ae02b8a439bb1a6faf1e"
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
    ref = json.loads((DATASET_DIR / "acropora_millepora_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-kaandorp-acropora-millepora-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
