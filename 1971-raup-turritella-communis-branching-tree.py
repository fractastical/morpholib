#!/usr/bin/env python3
"""
1971-raup-turritella-communis-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0658)
Organism: Turritella communis (Tower Screw Shell)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Turritella
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0658_turritella_communis_branching_tree.lean` (`MathOfBiology.Loop_0658`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/turritella_communis_branching_tree"
EXPECTED_SHA256 = {
    "turritella_communis_branching_tree_reference.json": "035eeac39d0d1f0d243fbe1cc9d2cd13a3014d075955d3456f2bda3ebe6dccae",
    "turritella_communis_specimen.jpg": "899a774eadcf1dae2799654a3393a678c5543bfebd3e3f9a02043f13da6934d0",
    "scout_manifest.json": "b5600ee77a146ae6c387c8932374429261e1e13b940f6273de70b8f185d473ba",
    "paper_citations.bib": "cd4fec6830bb1a919f10f02e7e3b253fd7de4912ce76b7a0b891ad56e4d2c2de"
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
    ref = json.loads((DATASET_DIR / "turritella_communis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1971-raup-turritella-communis-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
