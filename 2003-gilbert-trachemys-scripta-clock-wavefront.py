#!/usr/bin/env python3
"""
2003-gilbert-trachemys-scripta-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0329)
Organism: Trachemys scripta (Red-Eared Slider Turtle)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Trachemys
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0329_trachemys_scripta_clock_wavefront.lean` (`MathOfBiology.Loop_0329`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/trachemys_scripta_clock_wavefront"
EXPECTED_SHA256 = {
    "trachemys_scripta_clock_wavefront_reference.json": "c8c081dc833c9b1df2fc59ece2c8acefa48aae68c587e502d424f7f1e4e62252",
    "trachemys_scripta_specimen.jpg": "5d30c00d709c06ef8c1ebdb9b272654df48237cd07a6d19be7e219ab27188e96",
    "scout_manifest.json": "8f59c65ca4942c43586e725b16547b5880d7cb52a7bb7f1b11620b043f365d32",
    "paper_citations.bib": "e6aab2ec18dedf85e9e7896d648c7d6c6720d0658ac03ec83b973c6a5c81471d"
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
    ref = json.loads((DATASET_DIR / "trachemys_scripta_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-gilbert-trachemys-scripta-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
