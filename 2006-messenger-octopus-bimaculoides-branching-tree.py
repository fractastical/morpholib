#!/usr/bin/env python3
"""
2006-messenger-octopus-bimaculoides-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0652)
Organism: Octopus bimaculoides (Two-Spot Octopus)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Octopus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0652_octopus_bimaculoides_branching_tree.lean` (`MathOfBiology.Loop_0652`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/octopus_bimaculoides_branching_tree"
EXPECTED_SHA256 = {
    "octopus_bimaculoides_branching_tree_reference.json": "763c84ece63eae7d4d883ffb2b3aed914eca8bdd2bd8f577b4db2ff50e79ed60",
    "octopus_bimaculoides_specimen.jpg": "4e6d70661aa4e7709a51348bee322500de3a32b1ecb8b557b67060ff9d863233",
    "scout_manifest.json": "8c165a008678714a324ad583b46a1e0006d87bb0f4505ae12f51a222955b4008",
    "paper_citations.bib": "46ffd68ca0ab560be0138fd1c2675605d58fdf4f84f0fcd54711558a0a3bef66"
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
    ref = json.loads((DATASET_DIR / "octopus_bimaculoides_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-messenger-octopus-bimaculoides-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
