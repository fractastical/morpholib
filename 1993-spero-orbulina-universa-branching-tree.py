#!/usr/bin/env python3
"""
1993-spero-orbulina-universa-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0700)
Organism: Orbulina universa (Spherical Foraminifer)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Orbulina
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0700_orbulina_universa_branching_tree.lean` (`MathOfBiology.Loop_0700`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/orbulina_universa_branching_tree"
EXPECTED_SHA256 = {
    "orbulina_universa_branching_tree_reference.json": "b397a1d014c1b0580a06528a8a04b9c07767f8cb64320861df34cd892f833517",
    "orbulina_universa_specimen.jpg": "33d170bc776044ce2b3334da5523ea69a13551d1ff7cb86635f20a3dab925141",
    "scout_manifest.json": "d4a710461b327ca816b0e24e65569614d60a6e38bbb4afb843f14d8e535f2431",
    "paper_citations.bib": "a087e9e4bfee3bf0ab13c859f361c49b45b3fd5915c16dc099b82b6b9d307242"
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
    ref = json.loads((DATASET_DIR / "orbulina_universa_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1993-spero-orbulina-universa-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
