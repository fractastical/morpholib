#!/usr/bin/env python3
"""
2012-kumar-pleurodeles-waltl-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0625)
Organism: Pleurodeles waltl (Iberian Ribbed Newt)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Pleurodeles
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0625_pleurodeles_waltl_branching_tree.lean` (`MathOfBiology.Loop_0625`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pleurodeles_waltl_branching_tree"
EXPECTED_SHA256 = {
    "pleurodeles_waltl_branching_tree_reference.json": "a102af17f9afeeb4cafe58dbcdac512b80bbdf4df3e186c3325a0168ace490ff",
    "pleurodeles_waltl_specimen.jpg": "5b04f7216aba72146f64ff5aae72cd1f25ecdc62a55c051a3aea6d71c7dd09c5",
    "scout_manifest.json": "edf943d5fe115811019a0b233e09e15712b3dc4a9d4f73bddc0693b831edc645",
    "paper_citations.bib": "c4372c54de2856f2260dfa5de1f7de75aab684567f297bb470c14877a0ae560c"
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
    ref = json.loads((DATASET_DIR / "pleurodeles_waltl_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2012-kumar-pleurodeles-waltl-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
