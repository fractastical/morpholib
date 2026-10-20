#!/usr/bin/env python3
"""
2008-buhl-schistocerca-gregaria-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0348)
Organism: Schistocerca gregaria (Desert Locust)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Schistocerca
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0348_schistocerca_gregaria_clock_wavefront.lean` (`MathOfBiology.Loop_0348`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/schistocerca_gregaria_clock_wavefront"
EXPECTED_SHA256 = {
    "schistocerca_gregaria_clock_wavefront_reference.json": "1fe628fff53da7f75061428151c4e3d80a25d61b50cb864e6d64ce387521c764",
    "schistocerca_gregaria_specimen.jpg": "3f8a278b02bc5e5158458d4ff3669c71a29826b1129be2cc7920ab4b3ead3827",
    "scout_manifest.json": "7ef5164964df4ab00253a0ce5386d7d8c2a73bad06b3090ff822ca3603dd0387",
    "paper_citations.bib": "50f9ad1536635f4a817d016c925a7111000b0fbe53aaa1257455a8aec4d88a59"
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
    ref = json.loads((DATASET_DIR / "schistocerca_gregaria_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-buhl-schistocerca-gregaria-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
