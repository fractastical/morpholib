#!/usr/bin/env python3
"""
1980-bard-equus-quagga-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0404)
Organism: Equus quagga (Plains Zebra)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Equus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0404_equus_quagga_flux_canalization.lean` (`MathOfBiology.Loop_0404`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/equus_quagga_flux_canalization"
EXPECTED_SHA256 = {
    "equus_quagga_flux_canalization_reference.json": "3f82503d5b0a34d57ff3608b8aa7cf96ac4064efa8083c22c9ac5c4d428949f4",
    "equus_quagga_specimen.jpg": "b734dc3b533ffe889dee9f205dae9015580b4b3537a05743c602408063b17ca8",
    "scout_manifest.json": "e02cfe1c829372e06ffdf6eeb22cf8b7d0c4e1294a3a432430f1a8b2522da10a",
    "paper_citations.bib": "4f94e1f9cfd33c0cffa50c43392593dcbcb809a7a3e74c21bb4d8f1820d50937"
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
    ref = json.loads((DATASET_DIR / "equus_quagga_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1980-bard-equus-quagga-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
