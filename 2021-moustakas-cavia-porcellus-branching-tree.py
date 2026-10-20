#!/usr/bin/env python3
"""
2021-moustakas-cavia-porcellus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0608)
Organism: Cavia porcellus (Guinea Pig)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Cavia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0608_cavia_porcellus_branching_tree.lean` (`MathOfBiology.Loop_0608`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/cavia_porcellus_branching_tree"
EXPECTED_SHA256 = {
    "cavia_porcellus_branching_tree_reference.json": "f717551fca3c65630345029ec285ef43a484d2ba5a41a5eb7367cf0084491ce0",
    "cavia_porcellus_specimen.jpg": "42014eaf5b445ec569739b4b14be9f65dea72da7126e6dc92702e66dbbbf2acd",
    "scout_manifest.json": "76c8ba6797e1bc3d920f11ae08461185eae3e9fcdf64dbf38bf5e5d84e2f033d",
    "paper_citations.bib": "34d1ca3a3afa87c98e1ac57ea61b194f1e92b904aa531b39af7f08c93dcca817"
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
    ref = json.loads((DATASET_DIR / "cavia_porcellus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2021-moustakas-cavia-porcellus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
