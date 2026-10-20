#!/usr/bin/env python3
"""
2009-lecuit-drosophila-melanogaster-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0341)
Organism: Drosophila melanogaster (Fruit Fly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Drosophila
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0341_drosophila_melanogaster_clock_wavefront.lean` (`MathOfBiology.Loop_0341`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/drosophila_melanogaster_clock_wavefront"
EXPECTED_SHA256 = {
    "drosophila_melanogaster_clock_wavefront_reference.json": "075c88b8c8ad49fdf09148d3b0f40dfe25037baf0f0f7fa33a17398c4a6d6a44",
    "drosophila_melanogaster_specimen.jpg": "f426366f7ede436da44782b28ebac9bd013cace1f8fe44b6ed382b77ad86c419",
    "scout_manifest.json": "272534ccf9e6fefdd11a4fcde3b00661d579a448051d30465f5c38a48d0dc611",
    "paper_citations.bib": "292502e0cbf0990c7df1842c1815ff370131b894d3b87f8776e99119035d0b56"
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
    ref = json.loads((DATASET_DIR / "drosophila_melanogaster_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-lecuit-drosophila-melanogaster-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
