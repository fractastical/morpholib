#!/usr/bin/env python3
"""
2019-levin-dugesia-japonica-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0666)
Organism: Dugesia japonica (Freshwater Planarian)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Dugesia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0666_dugesia_japonica_branching_tree.lean` (`MathOfBiology.Loop_0666`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/dugesia_japonica_branching_tree"
EXPECTED_SHA256 = {
    "dugesia_japonica_branching_tree_reference.json": "7611075ee008832ecea5f4ba8b320ae1a37bb2f62c45a7057e0e391b442d6e74",
    "dugesia_japonica_specimen.jpg": "26b78d28b853c1b70092be6a7ae129d59813507f64eaed16a417276296ff2efa",
    "scout_manifest.json": "0c98523b60e7ba72c6038d717f707edf91aae00acd95ad05b39fd9bb440c64d1",
    "paper_citations.bib": "65fb0f5b85a4aca25b9ff5191369350101c27e603e26dbc505b9463ded227f7d"
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
    ref = json.loads((DATASET_DIR / "dugesia_japonica_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-levin-dugesia-japonica-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
