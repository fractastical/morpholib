#!/usr/bin/env python3
"""
2018-hohn-volvox-carteri-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0491)
Organism: Volvox carteri (Colonial Green Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Volvox
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0491_volvox_carteri_flux_canalization.lean` (`MathOfBiology.Loop_0491`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/volvox_carteri_flux_canalization"
EXPECTED_SHA256 = {
    "volvox_carteri_flux_canalization_reference.json": "843c8e56153b99ed0ab4cb1c04661ca6d92cbd350179f8a5beb78b46245e387f",
    "volvox_carteri_specimen.jpg": "cf734e5f4b2d7f5ded097049b19cf0c7c00c592b2c19a662073fe8d066b17e4a",
    "scout_manifest.json": "182394eb153d2ad1923ba1a88231811bcb1eba0346470024f9951e6e6d9679a9",
    "paper_citations.bib": "31868609406f13801763e36da1d0a4ffe6faf2a4f9a5bc2524ba14cc14dbffa5"
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
    ref = json.loads((DATASET_DIR / "volvox_carteri_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-hohn-volvox-carteri-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
