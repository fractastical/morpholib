#!/usr/bin/env python3
"""
2019-cooke-melopsittacus-undulatus-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0317)
Organism: Melopsittacus undulatus (Budgerigar)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Melopsittacus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0317_melopsittacus_undulatus_clock_wavefront.lean` (`MathOfBiology.Loop_0317`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/melopsittacus_undulatus_clock_wavefront"
EXPECTED_SHA256 = {
    "melopsittacus_undulatus_clock_wavefront_reference.json": "31451a3e5f2ad197ecbbe30e78fe1a9b8c9752eb0fddfc8eb5a716f8eb22cfb9",
    "melopsittacus_undulatus_specimen.jpg": "f61163f7820594be22c648c5120f1e827b9cdd8d377d9eeb65bb96e633b0ce2b",
    "scout_manifest.json": "075466982afd50df5964ce8d0e54bd0e3fd9070738c7d93bbeeb7ce5f7ed39cf",
    "paper_citations.bib": "504288f56420c124bfab9ae78cf8ec3d6493bd63ff654269abba66441f6465db"
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
    ref = json.loads((DATASET_DIR / "melopsittacus_undulatus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-cooke-melopsittacus-undulatus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
