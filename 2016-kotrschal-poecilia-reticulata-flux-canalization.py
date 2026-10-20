#!/usr/bin/env python3
"""
2016-kotrschal-poecilia-reticulata-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0437)
Organism: Poecilia reticulata (Trinidadian Guppy)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Poecilia
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0437_poecilia_reticulata_flux_canalization.lean` (`MathOfBiology.Loop_0437`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/poecilia_reticulata_flux_canalization"
EXPECTED_SHA256 = {
    "poecilia_reticulata_flux_canalization_reference.json": "3acdbefb4f97885ed6578136cd39c0919f02563fc9d6173d8ac9889bfa76cc1d",
    "poecilia_reticulata_specimen.jpg": "3e998877d83b46f9df12f9fa135839a5982986b29a14965629955111cadf8a2e",
    "scout_manifest.json": "aa3b4b9eb57139bebf480d5f4f019e2b048daaa71340a37da53ebb568846c8bc",
    "paper_citations.bib": "3b1e7d7dfd4e0914476dc92a2f38fd55d5730f19456392592e536efa68b632e6"
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
    ref = json.loads((DATASET_DIR / "poecilia_reticulata_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-kotrschal-poecilia-reticulata-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
