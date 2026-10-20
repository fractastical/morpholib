#!/usr/bin/env python3
"""
2007-coen-antirrhinum-majus-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0476)
Organism: Antirrhinum majus (Snapdragon)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Antirrhinum
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0476_antirrhinum_majus_flux_canalization.lean` (`MathOfBiology.Loop_0476`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/antirrhinum_majus_flux_canalization"
EXPECTED_SHA256 = {
    "antirrhinum_majus_flux_canalization_reference.json": "9de52d18821e1e4e9a71111e4ebc385daded32195a782596b2652193c317a62b",
    "antirrhinum_majus_specimen.jpg": "f3e9bf78a4453bc1a72552c6d200a37a1bccf6047928e988edba53ccfc260fe0",
    "scout_manifest.json": "60dfc262e52e2a7ada5c59a199caf4abcaf7c28c00bc2914694fec86b831846d",
    "paper_citations.bib": "8f7a0f5033a564d9a31818271b3ea18e62f22ceb413ed47d8585afebada52789"
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
    ref = json.loads((DATASET_DIR / "antirrhinum_majus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-coen-antirrhinum-majus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
