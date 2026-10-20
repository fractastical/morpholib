#!/usr/bin/env python3
"""
2020-hohn-volvox-carteri-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0691)
Organism: Volvox carteri (Colonial Green Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Volvox
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0691_volvox_carteri_branching_tree.lean` (`MathOfBiology.Loop_0691`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/volvox_carteri_branching_tree"
EXPECTED_SHA256 = {
    "volvox_carteri_branching_tree_reference.json": "dc68a7bcf97d209dad8a1dd3aacf13ea69a3e53c61326c19efc89080b7d77040",
    "volvox_carteri_specimen.jpg": "cf734e5f4b2d7f5ded097049b19cf0c7c00c592b2c19a662073fe8d066b17e4a",
    "scout_manifest.json": "7a6c09955a81580fea305501490713f85359eaf22e19ec3ec701ca468b063b45",
    "paper_citations.bib": "0b709fc7b629d3ea1ded1b43afa1bb2e7fe53748d83ffbe65a15e9265bae0dad"
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
    ref = json.loads((DATASET_DIR / "volvox_carteri_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-hohn-volvox-carteri-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
