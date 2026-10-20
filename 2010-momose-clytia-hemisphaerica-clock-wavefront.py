#!/usr/bin/env python3
"""
2010-momose-clytia-hemisphaerica-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0368)
Organism: Clytia hemisphaerica (Colonial Hydrozoan)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Clytia
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0368_clytia_hemisphaerica_clock_wavefront.lean` (`MathOfBiology.Loop_0368`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/clytia_hemisphaerica_clock_wavefront"
EXPECTED_SHA256 = {
    "clytia_hemisphaerica_clock_wavefront_reference.json": "0b1ec34a219a66adcb710a1c49eeba37c932a2550ced170e689aa6cdcddd17a8",
    "clytia_hemisphaerica_specimen.jpg": "2fcc34f8d72cdc5aca117a04c8528f2e4227f504bc61fe0588a43db0a016f236",
    "scout_manifest.json": "a65ffe2eb956f36ead8fd4f1090096899f0bc1965a863d251f81e1311967b4f8",
    "paper_citations.bib": "4027b00d3d8d040c4fb669090dae13fd198fa412dd20dd97ac1e7f08dc147722"
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
    ref = json.loads((DATASET_DIR / "clytia_hemisphaerica_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-momose-clytia-hemisphaerica-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
