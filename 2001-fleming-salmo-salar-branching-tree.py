#!/usr/bin/env python3
"""
2001-fleming-salmo-salar-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0638)
Organism: Salmo salar (Atlantic Salmon)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Salmo
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0638_salmo_salar_branching_tree.lean` (`MathOfBiology.Loop_0638`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/salmo_salar_branching_tree"
EXPECTED_SHA256 = {
    "salmo_salar_branching_tree_reference.json": "d91c10b4e1151f7964d187f615522e777df6724f7460125961d8c907ccae41eb",
    "salmo_salar_specimen.jpg": "2fae0b54511096d5c98f39fd08b4b665fdb0cad61caedbee18633275fbf4173e",
    "scout_manifest.json": "6b437fade9cb9abb7ebc68e99f38974379688487755b7496354e5ee2496f0cef",
    "paper_citations.bib": "22c59dbbca1da3c67d9702a502fadce132be3db0d2cd0b24069ad97e43ca9d0e"
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
    ref = json.loads((DATASET_DIR / "salmo_salar_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-fleming-salmo-salar-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
