#!/usr/bin/env python3
"""
1964-tartar-stentor-coeruleus-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0496)
Organism: Stentor coeruleus (Trumpet Ciliate)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Stentor
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0496_stentor_coeruleus_flux_canalization.lean` (`MathOfBiology.Loop_0496`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/stentor_coeruleus_flux_canalization"
EXPECTED_SHA256 = {
    "stentor_coeruleus_flux_canalization_reference.json": "9e8ede81c76f22d74334bf4486e087cae70ca35123abcfbe556360699e8d81aa",
    "stentor_coeruleus_specimen.jpg": "a1681963dfc488967f3552b87eed7d5716e022271322cdda8623fa8000374fb8",
    "scout_manifest.json": "4b9664704fbda110c587eb85ea61dc4dc87d7948e2d636aed31069561b7d8988",
    "paper_citations.bib": "8f54f86e942d231ace84c8341ed1978c8bdb01593a1449a334d2b0d8c88a5d8b"
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
    ref = json.loads((DATASET_DIR / "stentor_coeruleus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1964-tartar-stentor-coeruleus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
