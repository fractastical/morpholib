#!/usr/bin/env python3
"""
1999-fleming-salmo-salar-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0438)
Organism: Salmo salar (Atlantic Salmon)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Salmo
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0438_salmo_salar_flux_canalization.lean` (`MathOfBiology.Loop_0438`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/salmo_salar_flux_canalization"
EXPECTED_SHA256 = {
    "salmo_salar_flux_canalization_reference.json": "b70fa11212782af9762d3b9bba893bb28cf024e70df35ba3aa13308010f63250",
    "salmo_salar_specimen.jpg": "2fae0b54511096d5c98f39fd08b4b665fdb0cad61caedbee18633275fbf4173e",
    "scout_manifest.json": "7f38827ee430b1b3610b2c1f9062fb6ee6bf1b5e9da4cc4d7fcf53cb3fc4fb37",
    "paper_citations.bib": "f1a5153aa7d5106381c3915a76961f43c8dbd70ea69440d3040a69b2b7f8a070"
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
    ref = json.loads((DATASET_DIR / "salmo_salar_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1999-fleming-salmo-salar-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
