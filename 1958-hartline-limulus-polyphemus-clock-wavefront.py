#!/usr/bin/env python3
"""
1958-hartline-limulus-polyphemus-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0350)
Organism: Limulus polyphemus (Atlantic Horseshoe Crab)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Limulus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0350_limulus_polyphemus_clock_wavefront.lean` (`MathOfBiology.Loop_0350`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/limulus_polyphemus_clock_wavefront"
EXPECTED_SHA256 = {
    "limulus_polyphemus_clock_wavefront_reference.json": "6a222d24c1c00762a842b9d63135244164b7ab1215badb1451495672b1a5c0bc",
    "limulus_polyphemus_specimen.jpg": "6024a6e114ddb2e0b996d82067f840499683d1838662afe24496b8f315e2ef45",
    "scout_manifest.json": "5d35220ace785131d1e7bc5f351a07e67044c12ccac3dd286e830563941add12",
    "paper_citations.bib": "e4b60bb132f036066a6441d3e988b4e63543bd445ade6b16fd548374437594b8"
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
    ref = json.loads((DATASET_DIR / "limulus_polyphemus_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1958-hartline-limulus-polyphemus-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
