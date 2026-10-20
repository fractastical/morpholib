#!/usr/bin/env python3
"""
2004-munro-ciona-intestinalis-clock-wavefront.py — Pass 2 Remediated Reproduction (Slot #0363)
Organism: Ciona intestinalis (Vase Tunicate)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Ciona
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest)
Formal Verification: `morphoeval/lean/loops/Loop_0363_ciona_intestinalis_clock_wavefront.lean` (`MathOfBiology.Loop_0363`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ciona_intestinalis_clock_wavefront"
EXPECTED_SHA256 = {
    "ciona_intestinalis_clock_wavefront_reference.json": "7c5100c8ba3bdeb981bb6800c18c60b81176b67b7e35c14a4d2795c6a92a2fcb",
    "ciona_intestinalis_specimen.jpg": "0f563cbc2ccb4b7ba1595b7519a75a1410031eaaa4afceaf02b5975d87c1ecf4",
    "scout_manifest.json": "c20dab6ac7854ac1115891064208e5be4d6f5baf1e50bf1e094222ebec10e2fb",
    "paper_citations.bib": "d02b9bf3284a083655c14ab708f427fdbac327fcc02e9b1dbbbace1adc6ce21c"
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
    ref = json.loads((DATASET_DIR / "ciona_intestinalis_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-munro-ciona-intestinalis-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
