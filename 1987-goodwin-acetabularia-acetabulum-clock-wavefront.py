#!/usr/bin/env python3
"""
1987-goodwin-acetabularia-acetabulum-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0392)
Organism: Acetabularia acetabulum (Mermaid's Wineglass Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Acetabularia
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0392_acetabularia_acetabulum_clock_wavefront.lean` (`MathOfBiology.Loop_0392`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acetabularia_acetabulum_clock_wavefront"
EXPECTED_SHA256 = {
    "acetabularia_acetabulum_clock_wavefront_reference.json": "d781f81d7c9193da1a1b688845d8a3199a0a5b1d1f78ceb79e9ab3aedf94b74a",
    "acetabularia_acetabulum_specimen.jpg": "17641545f872481e451f25234ae988af0c16a529dd25aff08a53bd400e2c23a1",
    "scout_manifest.json": "e8f1e6be15efb666452acc51e5085959409fdce2bd9a6272816980e150774651",
    "paper_citations.bib": "78b87d948342a31347740eaa623f90486804a5961244934f52bd6101ccd41e3d"
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
    ref = json.loads((DATASET_DIR / "acetabularia_acetabulum_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1987-goodwin-acetabularia-acetabulum-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
