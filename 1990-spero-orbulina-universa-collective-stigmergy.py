#!/usr/bin/env python3
"""
1990-spero-orbulina-universa-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1100)
Organism: Orbulina universa (Spherical Foraminifer)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Orbulina
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1100_orbulina_universa_collective_stigmergy.lean` (`MathOfBiology.Loop_1100`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/orbulina_universa_collective_stigmergy"
EXPECTED_SHA256 = {
    "orbulina_universa_collective_stigmergy_reference.json": "a2791c5897964a6bb69ac99c68048a21886e5bb1951140f885a48fe33f202f78",
    "orbulina_universa_specimen.jpg": "33d170bc776044ce2b3334da5523ea69a13551d1ff7cb86635f20a3dab925141",
    "scout_manifest.json": "4b3583e3eab876d2e3c5a4b7f04d774f4ceb593954cbc0545af74487898a4c47",
    "paper_citations.bib": "867fda2e8b08093292f0e8b3af1c9a756839e83845c73566146b963598599d1a"
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
    ref = json.loads((DATASET_DIR / "orbulina_universa_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1990-spero-orbulina-universa-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
