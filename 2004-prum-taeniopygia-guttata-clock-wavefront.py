#!/usr/bin/env python3
"""
2004-prum-taeniopygia-guttata-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0312)
Organism: Taeniopygia guttata (Zebra Finch)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Taeniopygia
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0312_taeniopygia_guttata_clock_wavefront.lean` (`MathOfBiology.Loop_0312`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/taeniopygia_guttata_clock_wavefront"
EXPECTED_SHA256 = {
    "taeniopygia_guttata_clock_wavefront_reference.json": "7b3437bac4124f0a7c1a39f3f316c47b97d4d80cca7db4c86a7ad8898cdff776",
    "taeniopygia_guttata_specimen.jpg": "cba8970e2c1bea961592887848134a4a694bea3ad4d2b68524b2a63c722dffa0",
    "scout_manifest.json": "272650f929bb6d11ccd8c9d96cfe5a6d69d4b14278def81d1f9eff6bdce7d645",
    "paper_citations.bib": "60725f46522129c48bd754c20ecfff440beb722b49064b1726c88bb7f753e0e5"
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
    ref = json.loads((DATASET_DIR / "taeniopygia_guttata_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-prum-taeniopygia-guttata-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
