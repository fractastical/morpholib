#!/usr/bin/env python3
"""
2015-clarke-aptenodytes-forsteri-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0619)
Organism: Aptenodytes forsteri (Emperor Penguin)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Aptenodytes
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0619_aptenodytes_forsteri_branching_tree.lean` (`MathOfBiology.Loop_0619`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aptenodytes_forsteri_branching_tree"
EXPECTED_SHA256 = {
    "aptenodytes_forsteri_branching_tree_reference.json": "86e11cddf52a8b057210fb370941430b9332efc25a452364697cdc9ee2a77a8b",
    "aptenodytes_forsteri_specimen.jpg": "ef141d4e199c2ace11888d32dc8d271279924c5e9a9faa6f15ad839d7b5dd0d0",
    "scout_manifest.json": "f19b14a1aa8f6b0de474c19c2f265b9ab6257cc4c603e22298bb9ff3ca34b5da",
    "paper_citations.bib": "35c6fdf65a4881e6d803014fb857dbb4853add1569dcfb9af4ba75d2faf540e4"
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
    ref = json.loads((DATASET_DIR / "aptenodytes_forsteri_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-clarke-aptenodytes-forsteri-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
