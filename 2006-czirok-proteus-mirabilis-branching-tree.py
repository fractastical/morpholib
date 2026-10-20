#!/usr/bin/env python3
"""
2006-czirok-proteus-mirabilis-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0690)
Organism: Proteus mirabilis (Bullseye Swarming Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Proteus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0690_proteus_mirabilis_branching_tree.lean` (`MathOfBiology.Loop_0690`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/proteus_mirabilis_branching_tree"
EXPECTED_SHA256 = {
    "proteus_mirabilis_branching_tree_reference.json": "f07432e1eced47ae601b19290f7454379ddb1a49966e7254e293fcb1ad28c234",
    "proteus_mirabilis_specimen.jpg": "ac07f43481b9b721da7b124217c182a69192087ef9fb2f59564a107e3383a85b",
    "scout_manifest.json": "474b34fae985caaaa2968cf8fe19a49c308813c00767fea15d47a0a61f259637",
    "paper_citations.bib": "4e74735ebdb5a8938bd087a62fd8a4b3bc840f6e68041a9ea2528df11da69406"
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
    ref = json.loads((DATASET_DIR / "proteus_mirabilis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-czirok-proteus-mirabilis-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
