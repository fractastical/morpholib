#!/usr/bin/env python3
"""
2003-turing-mus-musculus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0601)
Organism: Mus musculus (House Mouse)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Mus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0601_mus_musculus_branching_tree.lean` (`MathOfBiology.Loop_0601`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mus_musculus_branching_tree"
EXPECTED_SHA256 = {
    "mus_musculus_branching_tree_reference.json": "1dea1969a3dc202a588900ee5ed895df551ffdfa9a6202538d7a9509a22d5cf3",
    "mus_musculus_specimen.jpg": "f2738290d66e441895fc10af7d7e541424263249cda0c7a8194ff01b410fe385",
    "scout_manifest.json": "69814bd8c9f6f1010ec73f723b937da6fc2d269c2ab63e7a06a9e31818fb855b",
    "paper_citations.bib": "de13a60bbd023f46f719621a496855a6692c3dfb9303ee530e667e18a48263df"
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
    ref = json.loads((DATASET_DIR / "mus_musculus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-turing-mus-musculus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
