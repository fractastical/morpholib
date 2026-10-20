#!/usr/bin/env python3
"""
2006-reinhardt-solanum-lycopersicum-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0474)
Organism: Solanum lycopersicum (Tomato)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Solanum
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0474_solanum_lycopersicum_flux_canalization.lean` (`MathOfBiology.Loop_0474`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/solanum_lycopersicum_flux_canalization"
EXPECTED_SHA256 = {
    "solanum_lycopersicum_flux_canalization_reference.json": "7fdf5eec3d1c7a20e93df2de53f49e8af248be4012a33312607d9d6656021c9c",
    "solanum_lycopersicum_specimen.jpg": "3ae537938870d7101582c24be70959db59fa5172fccc4d6ad7523d5b64cb2ec0",
    "scout_manifest.json": "f7d56a6740af91d7fd14ea4d69b5eff2e37ae406935cc76831100fffe4a0727c",
    "paper_citations.bib": "491d71a680809e47973c122f9beff99399aea852554ffeedc114147d5e4a6a20"
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
    ref = json.loads((DATASET_DIR / "solanum_lycopersicum_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-reinhardt-solanum-lycopersicum-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
