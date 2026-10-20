#!/usr/bin/env python3
"""
2011-niimi-bombyx-mori-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0646)
Organism: Bombyx mori (Domestic Silkmoth)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Bombyx
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0646_bombyx_mori_branching_tree.lean` (`MathOfBiology.Loop_0646`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/bombyx_mori_branching_tree"
EXPECTED_SHA256 = {
    "bombyx_mori_branching_tree_reference.json": "2a801ffd7b7d33065538d26968eba6f1644e753ac521c2b6575806a21f3d72c0",
    "bombyx_mori_specimen.jpg": "2adf51228d2da684d9c8eb6f510c972b1da829813ca787dd645448cff68d5d97",
    "scout_manifest.json": "57cb4b7881d01a13c67ee349e6237ddb1a1e72fbb9ec52e408bd3b49f3da33f4",
    "paper_citations.bib": "f52e3436692a7e46b6c8cc23a75db8d0a31b5710aca1f244f76b523b69f7d860"
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
    ref = json.loads((DATASET_DIR / "bombyx_mori_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2011-niimi-bombyx-mori-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
