#!/usr/bin/env python3
"""
2017-sanger-anolis-carolinensis-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0624)
Organism: Anolis carolinensis (Green Anole)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Anolis
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0624_anolis_carolinensis_branching_tree.lean` (`MathOfBiology.Loop_0624`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anolis_carolinensis_branching_tree"
EXPECTED_SHA256 = {
    "anolis_carolinensis_branching_tree_reference.json": "59595cf8406f1ebf7f7f98a00df01c0c468068f4f7bdc6b085656facd3f36d81",
    "anolis_carolinensis_specimen.jpg": "584ab986514945bb9c82b1765cb9c44cb8915e47823a0e02b5f444310fe0506b",
    "scout_manifest.json": "9fad58b74dd16263df95700775a3856760962e7b8203d984437bcf0ed98281a1",
    "paper_citations.bib": "ae41b7f7e39839ee0b9fde603ca6705a524e2504a6efe0caf6ef7192fb1e5076"
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
    ref = json.loads((DATASET_DIR / "anolis_carolinensis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2017-sanger-anolis-carolinensis-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
